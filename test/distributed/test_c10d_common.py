# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Common c10d test infrastructure shared by the backend-specific distributed test
# files: timeout tests, backend registration and entry point tests, default backend type
# selection tests, small nn.Module models used by DDP tests, a mixin of DDP helpers and
# checkpointing tests, abstract communication test mixins (sequence numbers, group membership,
# dtype checks, new_group sync), a dummy Python ProcessGroup/Work with extension tests, ReduceOp
# tests, local rank tests, record-comm tests and a lint that forbids non-trivial thread locals in
# c10d C++ sources. Import-time side effects: exits silently (sys.exit 0) if torch.distributed is
# unavailable or when running under dev/dbg ASAN; defines LOOPBACK ("lo0" on macOS else "lo") and
# device_type (current accelerator type or cpu); and applies instantiate_parametrized_tests to
# some classes. setUpModule/tearDownModule are defined for the unittest runner.

import copy
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from itertools import product
from pathlib import Path
from sys import platform

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.distributed.distributed_c10d as c10d
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._C._distributed_c10d import Backend as C10DBackend
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    load_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)
from torch.utils.checkpoint import checkpoint


if TEST_WITH_DEV_DBG_ASAN:
    print("Multiprocessing spawn is not compatible with dev/dbg asan", file=sys.stderr)
    sys.exit(0)

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if platform == "darwin":
    LOOPBACK = "lo0"
else:
    LOOPBACK = "lo"


_PRIOR_FP32_PRECISION: str | None = None


def setUpModule():
    # [FUNCTION DOC] HELPER: unittest module fixture. Saves the current CUDA matmul
    # fp32_precision setting into the module global _PRIOR_FP32_PRECISION and then disables TF32
    # matmul via allow_tf32, so numerics comparisons are exact. Mutates global backend state.
    global _PRIOR_FP32_PRECISION
    # Snapshot fp32_precision (not allow_tf32) so tearDownModule restores the
    # exact original; writing allow_tf32 back can't reproduce the "none" default.
    _PRIOR_FP32_PRECISION = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.allow_tf32 = False


def tearDownModule():
    # [FUNCTION DOC] HELPER: unittest module fixture. If a precision was saved by setUpModule,
    # restores the CUDA matmul fp32_precision setting and clears the saved global; otherwise does
    # nothing.
    global _PRIOR_FP32_PRECISION
    if _PRIOR_FP32_PRECISION is not None:
        torch.backends.cuda.matmul.fp32_precision = _PRIOR_FP32_PRECISION
        _PRIOR_FP32_PRECISION = None


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class MultiProcContinuousSkipTest(MultiProcContinuousTest):
    # [CLASS DOC] Two-process (world_size 2) continuous multi-process test over the gloo backend
    # on cpu, used to check that a skip raised inside a worker does not stop the workers from
    # running later tests. Class methods backend_str and device_type supply "gloo" and "cpu".
    world_size = 2

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] Classmethod returning the backend name "gloo" for the continuous test
        # harness.
        return "gloo"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] Classmethod returning the device type "cpu" for the continuous test
        # harness.
        return "cpu"

    def test_1_worker_skip(self) -> None:
        # [FUNCTION DOC] Calls skipTest from inside the worker; the test is expected to be
        # reported as skipped and not break later tests in the same worker processes.
        self.skipTest("skip from worker")

    def test_2_worker_continues_after_skip(self) -> None:
        # [FUNCTION DOC] Runs after the skipped test: each rank all-reduces (sum) a scalar tensor
        # holding rank + 1 and asserts the result is 3, which checks the workers and process
        # group are still usable after a skip.
        tensor = torch.tensor(self.rank + 1)
        dist.all_reduce(tensor)
        self.assertEqual(tensor, 3)


def gpus_for_rank(world_size):
    # [FUNCTION DOC] Splits the visible accelerator devices (by torch.accelerator.device_count)
    # evenly into world_size contiguous index lists, each of device_count // world_size entries,
    # and returns the list of lists indexed by rank. Leftover devices are not assigned to anyone.
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    device_count = torch.accelerator.device_count()
    visible_devices = list(range(device_count))
    gpus_per_process = device_count // world_size
    gpus_for_rank = []
    for rank in range(world_size):
        gpus_for_rank.append(
            visible_devices[rank * gpus_per_process : (rank + 1) * gpus_per_process]
        )
    return gpus_for_rank


class AbstractTimeoutTest:
    # [CLASS DOC] Mixin (not a test case itself) with helpers for checking that the default store
    # of a process group honours the init timeout, to be mixed into backend-specific test cases
    # that provide assertion methods.
    def _test_store_timeout(self, backend, init_method, c2p):
        # [FUNCTION DOC] HELPER: Intended to run in a thread. Initializes a world_size 1 process
        # group on the given backend and init_method with a 1 second timeout, then times a get of
        # a nonexistent key on the default store, which must raise a RuntimeError matching
        # "timeout", destroys the group and appends the elapsed seconds (float) to the list c2p.
        # If initialization (or the check) raises a RuntimeError, the exception object itself is
        # appended instead so the caller can report it, for example for address in use.
        try:
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                world_size=1,
                rank=0,
                timeout=timedelta(seconds=1),
            )
            default_store = c10d._get_default_store()
            tik = time.time()
            with self.assertRaisesRegex(RuntimeError, "(?i)timeout"):
                default_store.get("nonexistent key")
            tok = time.time()
            dist.destroy_process_group()
            c2p.append(float(tok - tik))
        except RuntimeError as e:
            # catch "Address already in use" error and report it to the main
            # thread
            c2p.append(e)

    def _init_methods(self):
        # [FUNCTION DOC] HELPER: Generator that creates a temporary file (not deleted) and yields
        # a file:// init method for it; on non-Windows platforms it also yields a tcp:// init
        # method on 127.0.0.1 with a free port.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.close()
            if sys.platform == "win32":
                yield "file:///{}".format(f.name.replace("\\", "/"))
            else:
                yield f"file://{f.name}"
                yield f"tcp://127.0.0.1:{common.find_free_port():d}"

    def _test_default_store_timeout(self, backend):
        # [FUNCTION DOC] HELPER: For each init method from _init_methods, runs
        # _test_store_timeout in a daemon thread (joined for up to 5 seconds) and checks exactly
        # one result was appended. A float result must be below 3 seconds (the wait should be
        # about 1 second); a RuntimeError result is re-raised so retry_on_connect_failures can
        # handle it; any other type raises RuntimeError.
        for init_method in self._init_methods():
            c2p = []
            t = threading.Thread(
                target=self._test_store_timeout, args=(backend, init_method, c2p)
            )
            t.daemon = True
            t.start()
            t.join(5)

            self.assertEqual(1, len(c2p))
            if isinstance(c2p[0], float):
                # waiting time should be 1s, use 3s to rule out false alarm
                self.assertGreater(3, c2p[0])
            elif isinstance(c2p[0], RuntimeError):
                # let @retry_on_connect_failures handle the error
                raise c2p[0]
            else:
                raise RuntimeError(f"Unexpected type {type(c2p[0])}")


class TimeoutTest(TestCase):
    # [CLASS DOC] Tests for timeouts in c10d store based rendezvous helpers.
    @retry_on_connect_failures
    def test_store_based_barrier(self):
        # [FUNCTION DOC] Spawns 4 threads per init type (file, tcp, hash store) that call the
        # private store based barrier with a 3 second timeout, with the last rank deliberately
        # not participating. Asserts that world_size - 1 threads fail with DistStoreError (a
        # DistError) whose message says it timed out initializing the process group in the store
        # based barrier. Retried on connect failures. Creates a temporary file and finds a free
        # port.
        f = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        f.close()
        port = common.find_free_port()

        def thread_work(timeout, init_type, world_size, rank, error_list):
            # [FUNCTION DOC] HELPER: Thread body. Builds a barrier store of the requested
            # init_type (FileStore, TCPStore on the shared port with rank 0 as master, or
            # HashStore); then, unless this is the last rank, calls the store based barrier with
            # logging every half timeout. A DistStoreError is checked to be a DistError and
            # appended to error_list.
            # we need to create a separate store just for the store barrier test
            if init_type == "file":
                barrier_store = dist.FileStore(f.name)
            elif init_type == "tcp":
                barrier_store = dist.TCPStore(
                    "localhost",
                    port,
                    world_size,
                    is_master=rank == 0,
                    wait_for_workers=False,
                )
            elif init_type == "hash":
                barrier_store = dist.HashStore()
            try:
                # 1 missing worker will cause it to timeout
                if rank != world_size - 1:
                    c10d._store_based_barrier(
                        rank=rank,
                        store=barrier_store,
                        group_name="_",
                        rendezvous_count=world_size,
                        timeout=timeout,
                        logging_interval=timeout / 2,
                    )
            except torch.distributed.DistStoreError as e:
                self.assertTrue(isinstance(e, torch.distributed.DistError))
                error_list.append(e)

        world_size = 4
        error_list = []
        threads = []
        for init_type in ["file", "tcp", "hash"]:
            for rank in range(world_size):
                t = threading.Thread(
                    target=thread_work,
                    args=(
                        timedelta(seconds=3),
                        init_type,
                        world_size,
                        rank,
                        error_list,
                    ),
                )
                threads.append(t)
                t.start()

            for thread in threads:
                thread.join()

            # we expect the world_size-1 threads to have failed
            self.assertEqual(len(error_list), world_size - 1)
            for error in error_list:
                self.assertTrue(
                    "Timed out initializing process group in store based barrier"
                    in error.args[0]
                )
            error_list = []
            threads = []


class BackendEntryPointTest(TestCase):
    # [CLASS DOC] Tests registration of distributed backends through plugin entry points and
    # built-in NCCL registration. setUp snapshots the class level registries of dist.Backend
    # (plugins, backend list, capabilities, type map, default device map, and whether custom
    # attributes exist) and tearDown restores them, so registrations made by tests do not leak.
    def setUp(self):
        # [FUNCTION DOC] Snapshots copies of the dist.Backend registries and records whether the
        # attributes ENTRYPOINT_TEST and NCCL-LEGACY already exist on dist.Backend, for
        # restoration in tearDown.
        super().setUp()
        self._plugins = dist.Backend._plugins.copy()
        self._backend_list = dist.Backend.backend_list.copy()
        self._backend_capability = copy.deepcopy(dist.Backend.backend_capability)
        self._backend_type_map = dist.Backend.backend_type_map.copy()
        self._default_device_backend_map = (
            dist.Backend.default_device_backend_map.copy()
        )
        self._custom_backend_attrs = {
            "ENTRYPOINT_TEST": hasattr(dist.Backend, "ENTRYPOINT_TEST"),
            "NCCL-LEGACY": hasattr(dist.Backend, "NCCL-LEGACY"),
        }

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved dist.Backend registries and deletes any of the
        # tracked custom backend attributes that were created during the test, then calls the
        # parent tearDown.
        dist.Backend._plugins = self._plugins
        dist.Backend.backend_list = self._backend_list
        dist.Backend.backend_capability = self._backend_capability
        dist.Backend.backend_type_map = self._backend_type_map
        dist.Backend.default_device_backend_map = self._default_device_backend_map
        for attr, existed in self._custom_backend_attrs.items():
            if not existed and hasattr(dist.Backend, attr):
                delattr(dist.Backend, attr)
        super().tearDown()

    def test_backend_entrypoint_loads_on_availability_check(self):
        # [FUNCTION DOC] Patches importlib.metadata.entry_points to return a fake entry point
        # whose load returns a function registering a "entrypoint_test" cpu backend. Asserts
        # is_backend_available triggers exactly one load, that the plugin is registered under the
        # upper-case name, that BackendConfig stringifies to "cpu:entrypoint_test" and that the
        # registered creator is the supplied factory (which must never be called).
        load_count = 0

        def create_backend(store, rank, world_size, timeout):
            # [FUNCTION DOC] HELPER: Backend factory that must not run; raises AssertionError if
            # called.
            raise AssertionError("backend factory should not run in this test")

        class EntryPoint:
            # [CLASS DOC] HELPER: Fake entry point object with a name and a load method that
            # counts loads in the enclosing test and returns a register function.
            name = "entrypoint_test"

            def load(self):
                # [FUNCTION DOC] HELPER: Increments the enclosing test's load counter and returns
                # a function that registers the "entrypoint_test" backend (creator
                # create_backend, devices cpu) with dist.Backend.
                nonlocal load_count
                load_count += 1

                def register():
                    # [FUNCTION DOC] HELPER: Registers the fake "entrypoint_test" backend with
                    # the cpu device on dist.Backend.
                    dist.Backend.register_backend(
                        "entrypoint_test",
                        create_backend,
                        devices=["cpu"],
                    )

                return register

        with unittest.mock.patch(
            "importlib.metadata.entry_points", return_value=[EntryPoint()]
        ):
            self.assertTrue(dist.is_backend_available("entrypoint_test"))

        self.assertEqual(load_count, 1)
        self.assertIn("ENTRYPOINT_TEST", dist.Backend._plugins)
        backend_config = dist.BackendConfig("entrypoint_test")
        self.assertEqual(str(backend_config), "cpu:entrypoint_test")
        self.assertIs(
            dist.Backend._plugins["ENTRYPOINT_TEST"].creator_fn, create_backend
        )

    def test_backend_entrypoint_loads_for_backend_config(self):
        # [FUNCTION DOC] Same fake entry point setup as the availability test, but constructs
        # dist.BackendConfig directly under the patch. Asserts the entry point was loaded exactly
        # once, the config string is "cpu:entrypoint_test" and the plugin creator is the supplied
        # factory.
        load_count = 0

        def create_backend(store, rank, world_size, timeout):
            # [FUNCTION DOC] HELPER: Backend factory that must not run; raises AssertionError if
            # called.
            raise AssertionError("backend factory should not run in this test")

        class EntryPoint:
            # [CLASS DOC] HELPER: Fake entry point object with a name and a load method that
            # counts loads in the enclosing test and returns a register function.
            name = "entrypoint_test"

            def load(self):
                # [FUNCTION DOC] HELPER: Increments the enclosing test's load counter and returns
                # a function that registers the "entrypoint_test" backend with dist.Backend.
                nonlocal load_count
                load_count += 1

                def register():
                    # [FUNCTION DOC] HELPER: Registers the fake "entrypoint_test" backend with
                    # the cpu device on dist.Backend.
                    dist.Backend.register_backend(
                        "entrypoint_test",
                        create_backend,
                        devices=["cpu"],
                    )

                return register

        with unittest.mock.patch(
            "importlib.metadata.entry_points", return_value=[EntryPoint()]
        ):
            backend_config = dist.BackendConfig("entrypoint_test")

        self.assertEqual(load_count, 1)
        self.assertEqual(str(backend_config), "cpu:entrypoint_test")
        self.assertIs(
            dist.Backend._plugins["ENTRYPOINT_TEST"].creator_fn, create_backend
        )

    def test_backend_config_device_form_registers_per_pair(self):
        # [FUNCTION DOC] Spies on dist.Backend._ensure_backend_registered (patched as a
        # classmethod that records each name and delegates to the original) while constructing
        # BackendConfig for "cpu:gloo,cuda:nccl". Asserts each backend name was looked up
        # individually and the whole comma-separated string was not, and that the config
        # stringifies back to the same string.
        # The "device:backend" form must look up each backend individually, not
        # the whole comma-separated string (which can never match a backend and
        # would waste an entry-point scan on every BackendConfig construction).
        looked_up = []
        original = dist.Backend._ensure_backend_registered.__func__

        def spy(cls, name):
            # [FUNCTION DOC] HELPER: Replacement classmethod body that appends the looked-up
            # backend name to the shared list and calls the original implementation.
            looked_up.append(name)
            return original(cls, name)

        with unittest.mock.patch.object(
            dist.Backend, "_ensure_backend_registered", classmethod(spy)
        ):
            backend_config = dist.BackendConfig("cpu:gloo,cuda:nccl")

        self.assertNotIn("cpu:gloo,cuda:nccl", looked_up)
        self.assertIn("gloo", looked_up)
        self.assertIn("nccl", looked_up)
        self.assertEqual(str(backend_config), "cpu:gloo,cuda:nccl")

    @parametrize("use_nccl2", [False, True])
    def test_nccl_backend_registration(self, use_nccl2):
        # [FUNCTION DOC] Parametrized by use_nccl2. Under a patched copy of os.environ, sets or
        # removes TORCH_DIST_USE_NCCL2, calls the private built-in NCCL backend registration, and
        # asserts the registered NCCL creator is the nccl2 process group factory when set and the
        # regular NCCL factory otherwise, and that the type map maps "nccl" to BackendType.NCCL.
        with unittest.mock.patch.dict(os.environ):
            if use_nccl2:
                os.environ["TORCH_DIST_USE_NCCL2"] = "1"
            else:
                os.environ.pop("TORCH_DIST_USE_NCCL2", None)
            c10d._register_builtin_nccl_backend()

        expected_creator = (
            c10d._create_nccl2_process_group
            if use_nccl2
            else c10d._create_nccl_process_group
        )
        self.assertIs(
            dist.Backend._plugins["NCCL"].creator_fn,
            expected_creator,
        )
        self.assertEqual(
            dist.Backend.backend_type_map["nccl"],
            dist.ProcessGroup.BackendType.NCCL,
        )

    def test_nccl_legacy_backend_registration(self):
        # [FUNCTION DOC] Calls the private legacy NCCL registration and asserts the "NCCL-LEGACY"
        # plugin uses the standard NCCL process group factory, that its capability is the cuda
        # device only, and that its backend type is CUSTOM.
        c10d._register_builtin_nccl_legacy_backend()

        self.assertIs(
            dist.Backend._plugins["NCCL-LEGACY"].creator_fn,
            c10d._create_nccl_process_group,
        )
        self.assertEqual(dist.Backend.backend_capability["nccl-legacy"], ["cuda"])
        self.assertEqual(
            dist.Backend.backend_type_map["nccl-legacy"],
            dist.ProcessGroup.BackendType.CUSTOM,
        )


instantiate_parametrized_tests(BackendEntryPointTest)


class DefaultBackendTypeTest(TestCase):
    # [CLASS DOC] Tests for the private default-backend-type selection for a BackendConfig. The
    # default type of a multi-backend group must be one that some device actually registers,
    # otherwise ProcessGroup default backend lookup fails on the first rank query. Parametrized
    # tests mock torch.accelerator.current_accelerator.
    """Cover ``_get_default_backend_type_for_backend_config``.

    A multi-backend group registers one backend per device, but ``ProcessGroup``
    holds a single default ``BackendType``. If that default names a type no
    device registered, ``ProcessGroup::getDefaultBackend()`` raises "Could not
    find the default backend type N" on the first ``pg.rank()``.
    """

    @staticmethod
    def _registrable_backend_types(backend_config):
        # [FUNCTION DOC] HELPER: Staticmethod. For a BackendConfig returns the set of BackendType
        # values that would be registered for its device backends, looking each up in the backend
        # type map and defaulting to CUSTOM.
        """The BackendTypes ``_new_process_group_helper`` would register."""
        return {
            dist.Backend.backend_type_map.get(
                str(backend), dist.ProcessGroup.BackendType.CUSTOM
            )
            for backend in backend_config.device_backend_map.values()
        }

    @parametrize(
        "backend_str",
        [
            "cpu:gloo",
            "cpu:gloo,cuda:nccl",
            "cpu:gloo,xpu:xccl",
            "cuda:nccl",
            "xpu:xccl",
            "cuda:ucc",
            "cpu:gloo,cuda:ucc",
        ],
    )
    @parametrize("accelerator", [None, "cuda", "xpu"])
    def test_default_backend_type_is_always_registrable(self, backend_str, accelerator):
        # [FUNCTION DOC] Parametrized over several backend strings and accelerators (none, cuda,
        # xpu). With current_accelerator mocked, builds the BackendConfig and asserts the
        # computed default backend type is one of the registrable types.
        """The default type must never name a backend that no device registers,
        regardless of which accelerator the host reports."""
        acc_device = torch.device(accelerator) if accelerator else None
        with unittest.mock.patch(
            "torch.accelerator.current_accelerator", return_value=acc_device
        ):
            backend_config = dist.BackendConfig(backend_str)
            self.assertIn(
                c10d._get_default_backend_type_for_backend_config(backend_config),
                self._registrable_backend_types(backend_config),
                f"default backend type is not registered for {backend_str!r} "
                f"with accelerator {accelerator!r}",
            )

    @parametrize(
        "backend_str, accelerator, expected_type",
        [
            # A mixed host+accelerator group defaults to the accelerator
            # backend so ProcessGroup::barrier() stays on the accelerator.
            ("cpu:gloo,xpu:xccl", "xpu", "XCCL"),
            ("cpu:gloo,cuda:nccl", "cuda", "NCCL"),
            # Accelerator-only groups must not fall back to a gloo backend that
            # was never registered.
            ("xpu:xccl", "xpu", "XCCL"),
            ("cuda:ucc", "cuda", "UCC"),
            # No device for the reported accelerator: any non-host device still
            # outranks cpu.
            ("cpu:gloo,xpu:xccl", "cuda", "XCCL"),
            ("cpu:gloo,xpu:xccl", None, "XCCL"),
            # Host-only groups are unambiguous.
            ("cpu:gloo", "cuda", "GLOO"),
        ],
    )
    def test_default_backend_type_prefers_accelerator_backend(
        self, backend_str, accelerator, expected_type
    ):
        # [FUNCTION DOC] Parametrized over (backend string, accelerator, expected type name).
        # With the accelerator mocked, asserts the default backend type is the expected
        # BackendType: the accelerator backend for mixed groups, any non-host device ahead of cpu
        # when the accelerator has no entry, and the host backend for cpu-only groups.
        acc_device = torch.device(accelerator) if accelerator else None
        with unittest.mock.patch(
            "torch.accelerator.current_accelerator", return_value=acc_device
        ):
            backend_config = dist.BackendConfig(backend_str)
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(backend_config),
                getattr(dist.ProcessGroup.BackendType, expected_type),
            )

    def test_default_backend_type_honors_bound_device_id(self):
        # [FUNCTION DOC] With the accelerator mocked to cuda and config
        # "cpu:gloo,xpu:xccl,cuda:nccl", asserts an explicit device argument wins: xpu:0 yields
        # XCCL and cpu yields GLOO.
        """An explicit ``device_id=`` outranks the reported accelerator."""
        with unittest.mock.patch(
            "torch.accelerator.current_accelerator",
            return_value=torch.device("cuda"),
        ):
            backend_config = dist.BackendConfig("cpu:gloo,xpu:xccl,cuda:nccl")
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(
                    backend_config, torch.device("xpu:0")
                ),
                dist.ProcessGroup.BackendType.XCCL,
            )
            self.assertEqual(
                c10d._get_default_backend_type_for_backend_config(
                    backend_config, torch.device("cpu")
                ),
                dist.ProcessGroup.BackendType.GLOO,
            )


instantiate_parametrized_tests(DefaultBackendTypeTest)


class Net(nn.Module):
    # [CLASS DOC] HELPER: Small bias-free MLP (2 -> 10 -> 50 -> 4 linear layers with ReLU
    # between) whose forward returns softmax over dim 1. Used as the basic DDP test model.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the three bias-free linear layers fc1, fc2, fc3 and a
        # ReLU.
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False)
        self.fc2 = nn.Linear(10, 50, bias=False)
        self.fc3 = nn.Linear(50, 4, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies fc1, ReLU, fc2, ReLU, fc3 and returns the softmax over
        # dim 1.
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


class DoubleGpuNet(nn.Module):
    # [CLASS DOC] HELPER: Same MLP as Net split across two devices: fc1 on the first device, fc2
    # and fc3 on the second, plus a non-trainable integer parameter no_grad_param on the first.
    # The output is returned on the first device.
    def __init__(self, gpus):
        # [FUNCTION DOC] HELPER: Places the layers and the requires_grad False long parameter on
        # the devices given in gpus (indexes 0 and 1).
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[1])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Moves the activations between the weight devices of fc1 and fc2
        # as needed and returns the softmax of the output on the device of fc1.
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.fc3(x)
        return F.softmax(x, dim=1).to(dev0)


class QuadraGpuNet(nn.Module):
    # [CLASS DOC] HELPER: Four-layer MLP model-parallel across four devices, one linear layer per
    # device (2 -> 10 -> 50 -> 4 -> 4), with a non-trainable integer parameter on the first
    # device. The output is returned on the first device.
    def __init__(self, gpus):
        # [FUNCTION DOC] HELPER: Places fc1..fc4 on gpus[0..3] and the requires_grad False long
        # parameter on gpus[0].
        super().__init__()
        self.fc1 = nn.Linear(2, 10, bias=False).to(gpus[0])
        self.fc2 = nn.Linear(10, 50, bias=False).to(gpus[1])
        self.fc3 = nn.Linear(50, 4, bias=False).to(gpus[2])
        self.fc4 = nn.Linear(4, 4, bias=False).to(gpus[3])
        self.relu = nn.ReLU()
        self.no_grad_param = nn.Parameter(
            torch.tensor([2, 2]).long(), requires_grad=False
        ).to(gpus[0])

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Moves activations to each layer's weight device in turn and
        # returns the softmax of the final output on the device of fc1.
        dev0 = self.fc1.weight.device
        dev1 = self.fc2.weight.device
        dev2 = self.fc3.weight.device
        dev3 = self.fc4.weight.device
        x = self.relu(self.fc1(x.to(dev0)))
        x = self.relu(self.fc2(x.to(dev1)))
        x = self.relu(self.fc3(x.to(dev2)))
        x = self.fc4(x.to(dev3))
        return F.softmax(x, dim=1).to(dev0)


class ConvNet(nn.Module):
    # [CLASS DOC] HELPER: Four convolution layers (8->16->32->16->8 channels, 2x2 kernels) each
    # with its own memory format and dtype, optionally on different devices. Records dtypes;
    # layer_gpus is set only if gpus was a list.
    def __init__(self, gpus, layouts, dtypes):
        # [FUNCTION DOC] HELPER: Stores dtypes and, when gpus is a list, layer_gpus; otherwise
        # the single device is repeated four times. Creates conv0..conv3 and moves each to its
        # device with its memory format and dtype from the layouts and dtypes sequences.
        super().__init__()
        self.dtypes = dtypes
        if isinstance(gpus, list):
            self.layer_gpus = gpus
        else:
            gpus = [gpus] * 4
        self.conv0 = torch.nn.Conv2d(8, 16, (2, 2)).to(
            device=gpus[0], memory_format=layouts[0], dtype=dtypes[0]
        )
        self.conv1 = torch.nn.Conv2d(16, 32, (2, 2)).to(
            device=gpus[1], memory_format=layouts[1], dtype=dtypes[1]
        )
        self.conv2 = torch.nn.Conv2d(32, 16, (2, 2)).to(
            device=gpus[2], memory_format=layouts[2], dtype=dtypes[2]
        )
        self.conv3 = torch.nn.Conv2d(16, 8, (2, 2)).to(
            device=gpus[3], memory_format=layouts[3], dtype=dtypes[3]
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Casts the input to dtypes[0], then runs each conv in turn,
        # moving the result to the next layer's device and dtype (devices come from layer_gpus
        # when present, else the input's device). Returns the last conv output.
        x = x.to(self.dtypes[0])
        # Could say
        # x = self.conv0(x).to(device=self.conv1.weight.device, dtype=self.dtypes[1])
        # etc.  But I don't want to appeal to the weights' devices directly, because part of this test's purpose
        # is to verify weights are where expected if the model gets replicated.
        gpus = self.layer_gpus if hasattr(self, "layer_gpus") else [x.device] * 4
        x = self.conv0(x).to(device=gpus[1], dtype=self.dtypes[1])
        x = self.conv1(x).to(device=gpus[2], dtype=self.dtypes[2])
        x = self.conv2(x).to(device=gpus[3], dtype=self.dtypes[3])
        return self.conv3(x)


# A model involving FFTs, used to test DDP with complex tensors
class FFTModel(nn.Module):
    # [CLASS DOC] HELPER: Model with a single complex (cfloat) parameter of shape features x
    # features x hin x (win // 2 + 1) used to test DDP with complex tensors; forward is a
    # spectral convolution (real FFT, per-frequency channel mixing, inverse FFT).
    def __init__(self, hin, win, n_features):
        # [FUNCTION DOC] HELPER: Stores hin and win and creates the all-ones complex weight
        # parameter.
        super().__init__()
        self.hin = hin
        self.win = win
        self.weight = nn.Parameter(
            torch.ones((n_features, n_features, hin, win // 2 + 1), dtype=torch.cfloat)
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Computes the orthonormal 2D real FFT of the input at size (hin,
        # win), contracts channels with the weight via an einsum, applies the inverse 2D real FFT
        # and returns the result.
        xc = torch.fft.rfft2(x, s=(self.hin, self.win), dim=(-2, -1), norm="ortho")
        xcw = torch.einsum("nchw,cohw->nohw", xc, self.weight)
        x = torch.fft.irfft2(xcw, dim=(-2, -1), norm="ortho")
        return x


class Task(nn.Module):
    # [CLASS DOC] HELPER: Tiny module holding one 2x2 all-ones parameter p; forward adds p to its
    # input.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the 2x2 ones parameter p.
        super().__init__()
        self.p = nn.Parameter(torch.ones(2, 2))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the parameter plus the input.
        return self.p + x


class ModuleForDdpCommHook(nn.Module):
    # [CLASS DOC] HELPER: Wrapper module around a Task submodule used for DDP communication hook
    # tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the Task submodule t0.
        super().__init__()
        self.t0 = Task()

    def forward(self, x, rank):
        # [FUNCTION DOC] HELPER: Returns the output of t0 applied to the input shifted by rank.
        return self.t0(x + rank)


class SparseGradientModule(nn.Module):
    # [CLASS DOC] HELPER: Module with a sparse EmbeddingBag (10 entries, dim 10, sparse
    # gradients) used to test DDP with sparse gradients; forward returns softmax of the embedding
    # output over dim 1.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the sparse EmbeddingBag.
        super().__init__()
        self.embedding = nn.EmbeddingBag(10, 10, sparse=True)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the softmax over dim 1 of the embedding bag output.
        return F.softmax(self.embedding(x), dim=1)


class CommonDistributedDataParallelTest:
    # [CLASS DOC] Mixin of DistributedDataParallel test helpers and tests for backend-specific
    # test cases. The concrete class must provide rank, file_name and _get_process_group (here it
    # raises NotImplementedError). Contains model preparation helpers, a checkpointing comparison
    # harness with several nested checkpoint-using modules, comm hook helpers, SyncBatchNorm
    # empty-input tests and dataclass output tests. world_size is fixed at 2.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Best-effort removal of the test's file_name (the FileStore file
        # the DDP tests leave behind). Ignores OSError and AttributeError. It does not call the
        # parent tearDown.
        # DistributedDataParallel test doesn't seem to call FileStore destructor
        # TODO: investigate this test and the test is known to have issues
        # Use this hack to remove files for that test
        try:
            os.remove(self.file_name)
        except (OSError, AttributeError):
            pass

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning 2, the number of processes these tests use.
        return 2

    def _prepare_single_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        # [FUNCTION DOC] HELPER: Builds a Net model and a DistributedDataParallel copy (deep-
        # copied, bucket cap 0.001 MB, given process group, device_ids and
        # gradient_as_bucket_view). The device is devices[0] if devices is non-empty, else the
        # accelerator device indexed by rank. Moves the model to that device and creates random
        # input (global_batch_size x 2) and target (global_batch_size x 4) on it. Returns (model,
        # ddp_model, input, target).
        model = Net()
        device = devices[0] if devices else torch.device(f"{device_type}:{self.rank:d}")
        ddp_model = DistributedDataParallel(
            copy.deepcopy(model).to(device),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        model.to(device)

        input = torch.randn(global_batch_size, 2).to(device)
        target = torch.randn(global_batch_size, 4).to(device)

        return model, ddp_model, input, target

    def _prepare_multi_device_module(
        self,
        process_group,
        devices,
        device_ids,
        global_batch_size,
        gradient_as_bucket_view=False,
    ):
        # [FUNCTION DOC] HELPER: Asserts devices has 2 or 4 entries, builds DoubleGpuNet or
        # QuadraGpuNet accordingly, wraps a deep copy in DistributedDataParallel (bucket cap
        # 0.001 MB, given process group, device_ids, gradient_as_bucket_view) and creates random
        # input on devices[0] and target on the cpu. Returns (model, ddp_model, input, target).
        self.assertTrue(
            len(devices) == 2 or len(devices) == 4,
            lambda msg: f"{msg}\nunexpected devices for ddp tests {devices}",
        )
        if len(devices) == 2:
            model = DoubleGpuNet(devices)
        elif len(devices) == 4:
            model = QuadraGpuNet(devices)

        ddp_model = DistributedDataParallel(
            copy.deepcopy(model),
            device_ids=device_ids,
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        input = torch.randn(global_batch_size, 2).to(devices[0])
        target = torch.randn(global_batch_size, 4)

        return model, ddp_model, input, target

    def _get_store(self):
        # [FUNCTION DOC] HELPER: Returns a FileStore over the test's file_name for world_size
        # participants.
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        # [FUNCTION DOC] HELPER: Abstract hook; raises NotImplementedError and must be overridden
        # by the concrete test class to return a process group.
        raise NotImplementedError("To be implemented by child class")

    def _train_model(
        self, model, input_var, target, loss, run_checkpoint=False, use_reentrant=True
    ):
        # [FUNCTION DOC] HELPER: Puts the model in train mode, runs it on input_var (through
        # activation checkpointing with the given use_reentrant flag when run_checkpoint is
        # true), computes the loss against target and runs backward, accumulating gradients.
        # Returns nothing.
        model.train()
        if run_checkpoint:
            output = checkpoint(model, input_var, use_reentrant=use_reentrant)
        else:
            output = model(input_var)
        l = loss(output, target)
        l.backward()

    def _test_ddp_checkpointing(
        self,
        input_model,
        process_group,
        use_bucket_view,
        find_unused_parameters=False,
        static_graph=False,
        run_checkpoint=False,
        use_reentrant=True,
        allow_none_grads=False,
    ):
        # [FUNCTION DOC] HELPER: Core harness comparing local training with DDP training for a
        # given module. Sets the device to the rank, seeds 31415, makes a local copy and a DDP
        # copy (bucket cap 1 MB, using the given bucket view, find_unused_parameters,
        # static_graph and the process group), and asserts the logged static_graph value matches.
        # Then for 5 iterations zeroes grads, trains both (the local one on the full batch, DDP
        # on this rank's slice from _prepare_dummy_data) and asserts parameter grads are equal
        # within tolerance (rtol 1.3e-6, atol 5e-5), and not None unless allow_none_grads is set.
        # to reproduce the same training results
        torch.accelerator.set_device_index(self.rank)
        torch.manual_seed(31415)
        model = copy.deepcopy(input_model).to(device_type)
        ddp_model = copy.deepcopy(input_model).to(device_type)
        ddp_model = nn.parallel.DistributedDataParallel(
            ddp_model,
            bucket_cap_mb=1,
            gradient_as_bucket_view=use_bucket_view,
            device_ids=[self.rank],
            process_group=process_group,
            find_unused_parameters=find_unused_parameters,
            static_graph=static_graph,
        )
        self.assertEqual(
            ddp_model._get_ddp_logging_data().get("static_graph", 0), static_graph
        )
        input, ddp_input, target, ddp_target = self._prepare_dummy_data()
        loss = nn.MSELoss()
        n_iters = 5
        for i in range(n_iters):
            model.zero_grad(set_to_none=False)
            ddp_model.zero_grad(set_to_none=False)
            self._train_model(
                model,
                input,
                target,
                loss,
                run_checkpoint=run_checkpoint,
                use_reentrant=use_reentrant,
            )
            self._train_model(
                ddp_model,
                ddp_input,
                ddp_target,
                loss,
                run_checkpoint=run_checkpoint,
                use_reentrant=use_reentrant,
            )
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                if not allow_none_grads:
                    self.assertTrue(i.grad is not None)
                    self.assertTrue(j.grad is not None)
                self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)

    # A list of tests for ddp with activation checkpointing
    # when gradient_as_bucket_view=True, False.
    # Most of the tests are referred to
    # https://github.com/facebookresearch/fairscale/blob/main/tests/nn/pipe/test_checkpoint_ddp.py
    class CheckpointOnceModule(nn.Module):
        # [CLASS DOC] HELPER: Module with two 20x20 linears where only the second layer is run
        # under activation checkpointing (with the configured use_reentrant flag).
        """
        Runs checkpoint for a single layer in the model.
        """

        def __init__(self, use_reentrant=True):
            # [FUNCTION DOC] HELPER: Creates linears l1 and l2 and stores use_reentrant.
            super().__init__()
            self.l1 = nn.Linear(20, 20)
            self.l2 = nn.Linear(20, 20)
            self.use_reentrant = use_reentrant

        def forward(self, inp):
            # [FUNCTION DOC] HELPER: Applies l1 normally, then l2 through checkpoint, and returns
            # the result.
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class CheckpointTwiceModule(CheckpointOnceModule):
        # [CLASS DOC] HELPER: Like CheckpointOnceModule but checkpoints the same layer l2 twice
        # in a row, simulating cases such as pipeline parallelism where one layer is checkpointed
        # more than once.
        """
        Runs checkpoint for the same layer twice in a model. This simulates use
        cases such as pipeline parallel where the same layer can be checkpointed
        more than one time.
        """

        def __init__(self, use_reentrant=True):
            # [FUNCTION DOC] HELPER: Forwards use_reentrant to the parent constructor (no other
            # state).
            super().__init__(use_reentrant=use_reentrant)

        def forward(self, inp):
            # [FUNCTION DOC] HELPER: Applies l1, then runs l2 through checkpoint twice and
            # returns the result.
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class CheckpointTwiceModuleWeightSharing(CheckpointTwiceModule):
        # [CLASS DOC] HELPER: Same as CheckpointTwiceModule but l1 and l2 share one weight
        # parameter.
        """
        Similar to CheckpointTwiceModule but the weights are shared.
        """

        def __init__(self, use_reentrant=True):
            # [FUNCTION DOC] HELPER: Calls the parent constructor and then makes l1.weight the
            # same parameter as l2.weight.
            super().__init__(use_reentrant=use_reentrant)
            # Share weights
            self.l1.weight = self.l2.weight

        def forward(self, inp):
            # [FUNCTION DOC] HELPER: Applies l1, then runs l2 through checkpoint twice and
            # returns the result (same as the parent forward).
            x = self.l1(inp)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            x = checkpoint(self.l2, x, use_reentrant=self.use_reentrant)
            return x

    class DynamicCheckpointTwiceModule(CheckpointTwiceModule):
        # [CLASS DOC] HELPER: Module that alternates which layer it checkpoints per call: it
        # keeps a call counter and checkpoints l2 on even calls and l1 on odd calls, so which
        # parameters are used varies across iterations.
        def __init__(self, use_reentrant=True):
            # [FUNCTION DOC] HELPER: Calls the parent constructor and initializes the call
            # counter count to 0.
            super().__init__(use_reentrant=use_reentrant)
            self.count = 0

        def forward(self, inp):
            # [FUNCTION DOC] HELPER: Checkpoints l1 when the call counter is odd, otherwise l2;
            # increments the counter (module state mutation) and returns the output.
            if self.count % 2:
                x = checkpoint(self.l1, inp, use_reentrant=self.use_reentrant)
            else:
                x = checkpoint(self.l2, inp, use_reentrant=self.use_reentrant)

            self.count += 1
            return x

    class DynamicCheckpointTwiceModuleWeightSharing(DynamicCheckpointTwiceModule):
        # [CLASS DOC] HELPER: DynamicCheckpointTwiceModule whose two linear layers share one
        # weight parameter.
        def __init__(self, use_reentrant=True):
            # [FUNCTION DOC] HELPER: Calls the parent constructor and then makes l1.weight the
            # same parameter as l2.weight.
            super().__init__(use_reentrant=use_reentrant)
            # Share weights
            self.l1.weight = self.l2.weight

    def _prepare_dummy_data(self):
        # [FUNCTION DOC] HELPER: Creates a random input (requires grad) and random target of 16 *
        # world_size rows by 20 columns on the accelerator device, and slices this rank's 16
        # rows. Returns (input, ddp_input, target, ddp_target) where the last of each pair is the
        # per-rank slice.
        ddp_bs = 16
        bs = ddp_bs * self.world_size
        input = torch.rand((bs, 20), device=device_type, requires_grad=True)
        target = torch.randn((bs, 20), device=device_type)
        offset = self.rank * ddp_bs
        ddp_input = input[offset : offset + ddp_bs]
        ddp_target = target[offset : offset + ddp_bs]
        return input, ddp_input, target, ddp_target

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_once(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by use_reentrant, needs at least 2 GPUs. For every
        # combination of bucket view and static graph, runs the checkpointing comparison with a
        # module that checkpoints one layer once; for static graph also runs with
        # find_unused_parameters set (the second run uses the default use_reentrant, not the
        # parametrized one).
        """
        DDP works as expected when layer is checkpointed only once.
        """
        process_group = self._get_process_group()
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
            )
            if static_graph:
                # find_unused_parameters does not make a difference, since it is
                # ignored for static graph.
                self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=static_graph,
                    find_unused_parameters=True,
                )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_unused_params(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by use_reentrant, needs at least 2 GPUs. With
        # find_unused_parameters and no static graph, expects a RuntimeError "Expected to mark a
        # variable ready only once." for the reentrant checkpoint implementation and success for
        # the non-reentrant one; with static graph enabled it must succeed in both cases. Loops
        # over bucket view on/off.
        """
        With reentrant autograd checkpointing impl, DDP will fail when there are
        unused params in the model and no static graph training. With
        non-reentrant checkpointing implementation, this works as expected.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            err_ctx = (
                nullcontext()
                if not use_reentrant
                else self.assertRaisesRegex(
                    RuntimeError, "Expected to mark a variable ready only once."
                )
            )
            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointOnceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    find_unused_parameters=True,
                )
            # test passes when static_graph is true
            self._test_ddp_checkpointing(
                self.CheckpointOnceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                find_unused_parameters=True,
                static_graph=True,
            )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_twice(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by use_reentrant, needs at least 2 GPUs. Checkpointing the
        # same layer twice without static graph, with and without find_unused_parameters, must
        # raise RuntimeError "Expected to mark a variable ready only once." for the reentrant
        # implementation and succeed for the non-reentrant one. Loops over bucket view on/off.
        """
        Checkpointing twice fails for non-static graph with reentrant checkpoint
        implementation, succeeds with non-reentrant checkpoint implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            err_ctx = (
                nullcontext()
                if not use_reentrant
                else self.assertRaisesRegex(
                    RuntimeError, "Expected to mark a variable ready only once."
                )
            )
            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=False,
                )

            with err_ctx:
                self._test_ddp_checkpointing(
                    self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                    process_group=process_group,
                    use_bucket_view=use_bucket_view,
                    static_graph=False,
                    find_unused_parameters=True,
                )

    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_twice_static_graph(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by use_reentrant, needs at least 2 GPUs. Checkpointing a
        # layer twice must work for both implementations when static_graph is enabled; loops over
        # bucket view on/off.
        """
        Regardless of reentrant or non-reentrant checkpointing impl,
        checkpointing twice works with static graph enabled.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            # Test passes when static_graph=True.
            self._test_ddp_checkpointing(
                self.CheckpointTwiceModule(use_reentrant=use_reentrant),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=True,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_dynamic_module(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Runs the DDP checkpointing comparison with the
        # dynamic checkpoint module (non-reentrant, no static graph, find_unused_parameters on),
        # allowing None grads because some parameters are unused in some iterations; loops over
        # bucket view on/off.
        """
        Dynamic module can be checkpointed, multiple times, with non-reentrant
        checkpointing implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.DynamicCheckpointTwiceModule(use_reentrant=False),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=False,
                find_unused_parameters=True,
                # Grads can be none sometimes due to dynamic module not using
                # all params.
                allow_none_grads=True,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_dynamic_weight_sharing(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Same as the dynamic module test but with the
        # weight-sharing dynamic module (non-reentrant, find_unused_parameters, None grads
        # allowed); loops over bucket view on/off.
        """
        Dynamic module can be checkpointed multiple times with weight sharing
        using non-reentrant checkpointing implementation.
        """
        process_group = self._get_process_group()
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.DynamicCheckpointTwiceModuleWeightSharing(use_reentrant=False),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=False,
                find_unused_parameters=True,
                # Grads can be none sometimes due to dynamic module not using
                # all params.
                allow_none_grads=True,
            )

    # DDP works as expected if there is weight sharing among layers
    @skip_if_lt_x_gpu(2)
    @parametrize("use_reentrant", [True, False])
    def test_ddp_checkpointing_weight_sharing(self, use_reentrant):
        # [FUNCTION DOC] Parametrized by use_reentrant, needs at least 2 GPUs. Builds a
        # sequential model of two linears sharing one weight (seeded identically each time) and
        # runs the checkpointing comparison with the whole model checkpointed, for all
        # combinations of bucket view and static graph.
        """
        Test that checkpointing with weight sharing works.
        """
        process_group = self._get_process_group()
        torch.accelerator.set_device_index(self.rank)
        for use_bucket_view, static_graph in product((False, True), (False, True)):
            torch.manual_seed(31415)
            l1 = nn.Linear(20, 20)
            l2 = nn.Linear(20, 20)
            l1.weight = l2.weight
            model = nn.Sequential(l1, l2)
            self._test_ddp_checkpointing(
                model,
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=static_graph,
                run_checkpoint=True,
                use_reentrant=use_reentrant,
            )

    @skip_if_lt_x_gpu(2)
    def test_ddp_checkpointing_twice_weight_sharing(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Runs the checkpointing comparison with the
        # weight-sharing twice-checkpointed module under static graph, looping over bucket view
        # on/off.
        """
        Checkpointing should work with static graph in the case of checkpointing
        same layer twice and having weights shared across layers.
        """
        process_group = self._get_process_group()
        torch.accelerator.set_device_index(self.rank)
        for use_bucket_view in (True, False):
            self._test_ddp_checkpointing(
                self.CheckpointTwiceModuleWeightSharing(),
                process_group=process_group,
                use_bucket_view=use_bucket_view,
                static_graph=True,
            )

    def test_invalid_powerSGD_state(self):
        # [FUNCTION DOC] Loops over start_powerSGD_iter in 0 and 1, use_error_feedback and
        # warm_start flags. Whenever error feedback or warm start is enabled (i.e. not both off),
        # constructing PowerSGDState must raise ValueError saying start_powerSGD_iter must be
        # above 1 because PowerSGD only applies after the first two iterations. Combinations with
        # both flags off are skipped.
        for start_powerSGD_iter, use_error_feedback, warm_start in product(
            [0, 1], [True, False], [True, False]
        ):
            if not use_error_feedback and not warm_start:
                continue
            with self.assertRaisesRegex(
                ValueError,
                "Expect `start_powerSGD_iter` > 1 if `use_error_feedback` or `warm_start` is enabled, "
                "because PowerSGD can only be applied after the first two iterations in DDP.",
            ):
                powerSGD.PowerSGDState(
                    process_group=None,
                    matrix_approximation_rank=1,
                    start_powerSGD_iter=start_powerSGD_iter,
                    use_error_feedback=use_error_feedback,
                    warm_start=warm_start,
                )

    def _test_ddp_with_process_group(
        self,
        process_group,
        devices,
        device_ids,
        multi_device=False,
        gradient_as_bucket_view=False,
    ):
        # [FUNCTION DOC] HELPER: Trains a plain local model and a DDP model over 2 iterations
        # with local batch size equal to the number of devices (1 if devices is None) and global
        # batch size world_size times that. Builds the pair via the multi-device or single-device
        # helper (checking the logged is_multi_device_module flag). Each iteration runs
        # forward/backward with MSE loss locally on the full batch and under DDP on this rank's
        # slice, applies a plain gradient step of the full gradient to each (clearing grads),
        # asserts parameters match within tolerance, then reshuffles the input with a seeded
        # permutation.
        """
        Note: we pass down `device_ids` all the way to DistributedDataParallel
        as part of the test. Below you find tests that either use a list of
        integers, a list of `torch.Device` instances, or an empty list.
        The `devices` argument is used to control placement of the model and
        must always be specified as list of `torch.Device` instances.
        """
        local_batch_size = 1 if devices is None else len(devices)
        global_batch_size = self.world_size * local_batch_size

        if multi_device:
            model, ddp_model, input, target = self._prepare_multi_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertTrue(ddp_logging_data.get("is_multi_device_module"))
        else:
            model, ddp_model, input, target = self._prepare_single_device_module(
                process_group,
                devices,
                device_ids,
                global_batch_size,
                gradient_as_bucket_view,
            )
            ddp_logging_data = ddp_model._get_ddp_logging_data()
            self.assertFalse(ddp_logging_data.get("is_multi_device_module"))

        def step_model(model, input, target):
            # [FUNCTION DOC] HELPER: Puts the model in train mode, runs forward, computes MSE
            # loss against the target (moved to the output device) and runs backward.
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        def update_parameters(model):
            # [FUNCTION DOC] HELPER: Subtracts each parameter's gradient from the parameter
            # without autograd tracking (learning rate 1) and clears the gradient to None.
            for param in model.parameters():
                with torch.no_grad():
                    param -= param.grad
                param.grad = None

        # check two model parameters over 2 iterations
        for iteration in range(2):
            # single cpu/gpu training
            step_model(model, input, target)

            # DDP training, DDP scatters subsets of input_cpu to nodes/GPUs
            step_model(
                ddp_model,
                input[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
                target[
                    self.rank * local_batch_size : (self.rank + 1) * local_batch_size
                ],
            )

            # Update weights and run a second iteration to shake out errors
            update_parameters(model)
            update_parameters(ddp_model)
            self.assertEqual(
                len(list(model.parameters())), len(list(ddp_model.parameters()))
            )
            for i, j in zip(model.parameters(), ddp_model.parameters()):
                self.assertEqual(i, j, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    def _gpu_model_with_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False, state=None
    ):
        # [FUNCTION DOC] HELPER: Builds a ModuleForDdpCommHook on this rank's first GPU (from
        # gpus_for_rank), wraps it in DistributedDataParallel with the process group and
        # gradient_as_bucket_view, and registers the given hook with the given state if hook is
        # not None. Returns the DDP model.
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

    def _gpu_model_with_builtin_ddp_comm_hook(
        self, process_group, hook=None, gradient_as_bucket_view=False
    ):
        # [FUNCTION DOC] HELPER: Same as the custom hook variant but registers a built-in DDP
        # comm hook (through the private register method) when hook is not None. Returns the DDP
        # model.
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Register a built-in DDP communication hook if defined
        if hook is not None:
            gpu_model._register_builtin_comm_hook(hook)

        return gpu_model

    def _run_and_verify_hook(self, model, input, expected_grad):
        # [FUNCTION DOC] HELPER: Runs forward with the rank as extra argument, backward on the
        # mean of the output, and asserts every parameter's gradient equals expected_grad.
        # Run forward
        output = model(input, self.rank)

        # Run backward
        output.mean().backward()

        [self.assertEqual(p.grad, expected_grad) for p in model.parameters()]

    def _simple_hook(
        self, state: object, bucket: dist.GradBucket
    ) -> torch.futures.Future[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Example DDP comm hook. Creates a future already completed with a
        # ones tensor shaped like the bucket buffer, and returns a continuation that adds one to
        # it, so the reduced gradient is a tensor of twos. The state argument is ignored.
        fut = torch.futures.Future()
        fut.set_result(torch.ones_like(bucket.buffer()))

        def fut_then(fut):
            # [FUNCTION DOC] HELPER: Continuation for the hook future; returns the future value
            # plus ones.
            # Add ones to fut's result.
            t = fut.value()
            return t + torch.ones_like(t)

        return fut.then(fut_then)

    def _test_not_nan(self, model, x):
        # [FUNCTION DOC] HELPER: Runs the model on x, asserts the output has no NaN,
        # backpropagates the sum and asserts no parameter gradient has NaN.
        y = model(x)
        self.assertFalse(y.isnan().any().item())
        y.sum().backward()
        for p in model.parameters():
            self.assertFalse(p.grad.isnan().any().item())

    @skip_if_lt_x_gpu(2)
    def test_sync_batch_norm_only_empty_input(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Wraps a BatchNorm2d model in DDP and converts it
        # to SyncBatchNorm on the process group, then checks outputs and grads are NaN-free when
        # only rank 0 gets an empty batch and when all ranks get empty batches, with the input
        # requiring grad and not requiring grad.
        pg = self._get_process_group()

        model = torch.nn.Sequential(
            nn.BatchNorm2d(2),
        ).to(device=self.rank)
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            process_group=pg,
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,
        )

        model.train()

        # only rank 0 receives empty inputs
        x = torch.zeros(
            (1 if self.rank != 0 else 0, 2, 11, 13),
            dtype=torch.float32,
            device=self.rank,
        )

        # input requires grad, this will trigger the collective communication
        # in the backward pass
        x.requires_grad = True
        self._test_not_nan(model, x)

        # input does not requires grad
        x.requires_grad = False
        self._test_not_nan(model, x)

        # all ranks receive empty inputs
        x = torch.zeros((0, 2, 11, 13), dtype=torch.float32, device=self.rank)

        # input requires grad, this will trigger the collective communication
        # in the backward pass
        x.requires_grad = True
        self._test_not_nan(model, x)

        # input does not requires grad
        x.requires_grad = False
        self._test_not_nan(model, x)

    @skip_if_lt_x_gpu(2)
    def test_sync_batch_norm_empty_input(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Same check with a conv, batchnorm and linear
        # stack: NaN-free output and grads when only rank 0 has an empty batch and when all ranks
        # have empty batches.
        pg = self._get_process_group()

        model = torch.nn.Sequential(
            nn.Conv2d(2, 2, 3),
            nn.BatchNorm2d(2),
            nn.Linear(28, 2),
        ).to(device=self.rank)
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            process_group=pg,
        )
        model = nn.SyncBatchNorm.convert_sync_batchnorm(
            model,
            process_group=pg,
        )

        model.train()
        # only rank 0 receives empty inputs
        x = torch.zeros(
            (3 if self.rank != 0 else 0, 2, 30, 30),
            dtype=torch.float32,
            device=self.rank,
        )

        self._test_not_nan(model, x)

        # all ranks receive empty inputs
        x = torch.zeros((0, 2, 30, 30), dtype=torch.float32, device=self.rank)

        self._test_not_nan(model, x)

    @skip_if_lt_x_gpu(2)
    def test_ddp_buffer_sync_multi_forward_with_batchnorm(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Runs two forward passes in one iteration on a DDP
        # model containing BatchNorm1d with buffer broadcasting enabled and backpropagates the
        # sum of both means; passes if nothing raises (no value assertions).
        pg = self._get_process_group()

        model = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
        ).to(device=self.rank)

        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            process_group=pg,
            broadcast_buffers=True,
        )

        x = torch.randn(4, 10, device=self.rank)

        model.zero_grad()
        out1 = model(x)
        out2 = model(x)
        (out1.mean() + out2.mean()).backward()

    @dataclass
    class CustomOutput:
        # [CLASS DOC] HELPER: Dataclass test output with an optional tensor o1 and a mapping o2
        # from names to tensors.
        o1: torch.Tensor | None
        o2: dict[str, torch.Tensor]

    class DataclassOutputModule(nn.Module):
        # [CLASS DOC] HELPER: Module with two 3-layer linear stacks seq1 and seq2 (10 features)
        # and a ReLU; its forward returns a CustomOutput. If skip_o1 is set seq1 is not used so
        # its parameters receive no gradient.
        def __init__(self, skip_o1):
            # [FUNCTION DOC] HELPER: Builds the two linear stacks and the ReLU and stores
            # skip_o1.
            super().__init__()
            self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.relu = nn.ReLU()
            self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(3)])
            self.skip_o1 = skip_o1

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Returns a CustomOutput whose o1 is None when skip_o1 is set,
            # else ReLU of seq1 applied to x, and whose o2 maps "a" to seq2(x) and "b" to ReLU of
            # seq2(x).
            o1 = None if self.skip_o1 else self.relu(self.seq1(x))
            o2 = {"a": self.seq2(x), "b": self.relu(self.seq2(x))}
            return CommonDistributedDataParallelTest.CustomOutput(o1=o1, o2=o2)

    def _test_dataclass_output(self, skip_o1):
        # [FUNCTION DOC] HELPER: Compares a local model on the concatenated batch with a DDP copy
        # (find_unused_parameters on, no static graph) on this rank's batch, where the module
        # returns a dataclass. Builds an MSE loss from the sum of the output fields,
        # backpropagates both, and asserts the parameter grads match (allclose for tensors,
        # equality otherwise).
        net_x = torch.cat([torch.ones(4, 10) * i for i in range(self.world_size)]).to(
            self.rank
        )
        ddp_x = torch.ones(4, 10, device=self.rank) * self.rank

        # use manual_seed to make sure local models start with the same values
        torch.manual_seed(0)
        net = self.DataclassOutputModule(skip_o1=skip_o1).to(self.rank)
        ddp = DistributedDataParallel(
            copy.deepcopy(net),
            device_ids=[self.rank],
            find_unused_parameters=True,
            static_graph=False,
            process_group=self._get_process_group(),
        )

        net_out = net(net_x)
        ddp_out = ddp(ddp_x)

        net_loss = F.mse_loss(
            (
                net_out.o1 + net_out.o2["a"] + net_out.o2["b"]
                if not skip_o1
                else net_out.o2["a"] + net_out.o2["b"]
            ),
            torch.ones_like(net_out.o2["a"], device=self.rank),
        )
        ddp_loss = F.mse_loss(
            (
                ddp_out.o1 + ddp_out.o2["a"] + ddp_out.o2["b"]
                if not skip_o1
                else ddp_out.o2["a"] + ddp_out.o2["b"]
            ),
            torch.ones_like(ddp_out.o2["a"], device=self.rank),
        )

        net_loss.backward()
        ddp_loss.backward()

        for p1, p2 in zip(net.parameters(), ddp.parameters()):
            if torch.is_tensor(p1.grad):
                self.assertTrue(p1.grad.allclose(p2.grad))
            else:
                self.assertEqual(p1.grad, p2.grad)

    @skip_if_lt_x_gpu(2)
    def test_dataclass_output(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Runs the dataclass output comparison with all
        # outputs used.
        self._test_dataclass_output(skip_o1=False)

    @skip_if_lt_x_gpu(2)
    def test_dataclass_output_unused_param(self):
        # [FUNCTION DOC] Needs at least 2 GPUs. Runs the dataclass output comparison with the
        # first output skipped so some parameters are unused.
        self._test_dataclass_output(skip_o1=True)


class ComputeBucketAssignmentTest(TestCase):
    # [CLASS DOC] Tests the private bucket assignment function that groups tensors into reducer
    # buckets by byte size limits and dtype.
    def test_single_limit_single_dtype(self):
        # [FUNCTION DOC] Four float tensors (400, 800, 400, 200 bytes) with a single 400 byte
        # limit: asserts each lands in its own bucket and all per-bucket limits are 400.
        tensors = [
            torch.empty([100], dtype=torch.float),
            torch.empty([200], dtype=torch.float),
            torch.empty([100], dtype=torch.float),
            torch.empty([50], dtype=torch.float),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        self.assertEqual([[0], [1], [2], [3]], result)

    def test_single_limit_multi_dtype(self):
        # [FUNCTION DOC] Interleaved float and double tensors with a single 400 byte limit:
        # asserts buckets never mix dtypes, grouping indices as [0,2], [1,3], [4], [5] with all
        # limits 400.
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [400]
        )
        self.assertTrue(all(size_lim == 400 for size_lim in per_bucket_size_limits))
        self.assertEqual([[0, 2], [1, 3], [4], [5]], result)

    def test_multi_limit_single_dtype(self):
        # [FUNCTION DOC] Four 40 byte float tensors with limits [40, 80]: asserts buckets [0],
        # [1,2], [3] and per-bucket limits [40, 80, 80] (the last limit repeats).
        tensors = [
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
            torch.empty([10], dtype=torch.float),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [40, 80]
        )
        self.assertEqual(per_bucket_size_limits, [40, 80, 80])
        self.assertEqual([[0], [1, 2], [3]], result)

    def test_multi_limit_multi_dtype(self):
        # [FUNCTION DOC] Interleaved float and double tensors with limits [200, 400]: asserts
        # buckets [0], [1], [2,4], [3,5] and limits [200, 200, 400, 400].
        tensors = [
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
            torch.empty([50], dtype=torch.float),
            torch.empty([25], dtype=torch.double),
        ]
        result, per_bucket_size_limits = dist._compute_bucket_assignment_by_size(
            tensors, [200, 400]
        )
        self.assertEqual([[0], [1], [2, 4], [3, 5]], result)
        self.assertEqual(per_bucket_size_limits, [200, 200, 400, 400])


class AbstractCommTest:
    # [CLASS DOC] Mixin of backend-agnostic communication tests that backend test cases run via
    # private _test_* helpers taking a backend name. Needs rank and file_name from the concrete
    # test case and overridden device. Covers sequence number tracking, warnings for ranks
    # outside a group, rank membership lookups, dtype mismatch checks, complex dtype all_gather
    # and bool tensor broadcast. world_size is 2.
    @property
    def op_timeout_sec(self):
        # [FUNCTION DOC] HELPER: Property returning 1, the per-operation timeout in seconds.
        return 1

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning 2, the test world size.
        return 2

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property that fails the test with a message; subclasses must
        # override it to return the device.
        self.fail("test subclass didn't override device")

    def _verify_sequence_number_across_pg(self, pg, verify_pg):
        # [FUNCTION DOC] HELPER: Reads the sequence number of process group pg, all-gathers it as
        # an object over the separate group verify_pg (so the verification does not alter pg's
        # counter), asserts all ranks report the same value and returns it.
        seq_num = pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
        # We use a separate pg to verify the sequence numbers, otherwise these
        # collectives will themselves increment the sequence number.
        dist.all_gather_object(obj_list, seq_num, group=verify_pg)
        self.assertEqual(len(set(obj_list)), 1)
        return obj_list[0]

    def _test_sequence_num_incremented(self, process_group, ranks):
        # [FUNCTION DOC] HELPER: Creates a gloo verification group over ranks and checks both
        # groups have equal world size (raises AssertionError otherwise). Records the initial
        # sequence number agreed across ranks of process_group (or -1 for a rank not in the
        # group), then runs 10 all_reduce calls on process_group, after each asserting the agreed
        # sequence number grew by exactly one. For groups larger than 2, ranks other than 0 and 2
        # issue one extra async all_reduce, and the test asserts the sequence numbers form two
        # distinct values: ranks 0 and 2 equal and one behind the others, which are all equal.
        # verify initial sequence numbers. Use a distinct process group for
        # verification to keep counts as expected with respect to process_group.
        verify_pg = dist.new_group(
            ranks=ranks,
            backend="gloo",
        )
        if dist.get_world_size(process_group) != dist.get_world_size(verify_pg):
            raise AssertionError(
                f"World sizes mismatch: {dist.get_world_size(process_group)} "
                f"vs {dist.get_world_size(verify_pg)}"
            )

        initial_num = (
            self._verify_sequence_number_across_pg(
                pg=process_group, verify_pg=verify_pg
            )
            if not c10d._rank_not_in_group(process_group)
            else -1
        )

        # Verify sequence numbers are appropriately incremented
        for i in range(10):
            t = torch.ones(1, device=device_type)
            dist.all_reduce(t, group=process_group)
            if not c10d._rank_not_in_group(process_group):
                seq_num = self._verify_sequence_number_across_pg(
                    pg=process_group,
                    verify_pg=verify_pg,
                )
                self.assertEqual(initial_num + i + 1, seq_num)

        if dist.get_world_size(process_group) > 2:
            # Test when certain ranks don't call collectives
            if dist.get_rank(process_group) not in [0, 2]:
                dist.all_reduce(t, group=process_group, async_op=True)
            # Now ranks 0 and 2 should be lagging by 1.
            if not c10d._rank_not_in_group(process_group):
                seq_num = process_group._get_sequence_number_for_group()
                rank = dist.get_rank(process_group)
                obj_list = [None for _ in range(dist.get_world_size(verify_pg))]
                dist.all_gather_object(obj_list, (rank, seq_num), group=verify_pg)
                rank_to_seq_num = dict(obj_list)
                self.assertEqual(len(set(rank_to_seq_num.values())), 2)
                self.assertEqual(rank_to_seq_num[0], rank_to_seq_num[2])
                expected_same = {
                    rank_to_seq_num[i] for i in rank_to_seq_num if i not in [0, 2]
                }
                self.assertEqual(len(expected_same), 1)
                self.assertEqual(rank_to_seq_num[0] + 1, rank_to_seq_num[1])

    def _test_sequence_num_incremented_default_group(self, backend_name):
        # [FUNCTION DOC] HELPER: Sets the accelerator device to this rank, initializes the
        # default process group on backend_name via a FileStore, and runs the sequence number
        # increment check over the whole world.
        torch.accelerator.set_device_index(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self._test_sequence_num_incremented(
            c10d._get_default_group(),
            ranks=list(range(dist.get_world_size())),
        )

    def _test_sequence_num_incremented_subgroup(self, backend_name):
        # [FUNCTION DOC] HELPER: Sets the accelerator device to this rank, initializes the
        # default group on backend_name via a FileStore, creates a subgroup of ranks 0, 1 and 2
        # and runs the sequence number increment check on it.
        # KNOWN ISSUE: the subgroup contains rank 2, which does not exist when world_size is 2
        # (the mixin default), so it only works with a larger world size.
        torch.accelerator.set_device_index(self.rank)
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        subgroup_ranks = [0, 1, 2]
        subgroup = dist.new_group(subgroup_ranks)
        self._test_sequence_num_incremented(subgroup, subgroup_ranks)

    def _test_sequence_num_set_default_pg(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group on the backend via a FileStore and
        # asserts the default group's sequence number is identical on all ranks (all-gathered as
        # objects).
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        default_pg = c10d._get_default_group()
        seq_num = default_pg._get_sequence_number_for_group()
        obj_list = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(obj_list, seq_num)
        self.assertEqual(len(set(obj_list)), 1)

    def _test_sequence_num_set_new_group(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, creates a subgroup of ranks 0 and
        # 1, and on member ranks asserts the subgroup's sequence number is identical across the
        # subgroup.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        subgroup = dist.new_group([0, 1])

        if not c10d._rank_not_in_group(subgroup):
            subgroup_seq = subgroup._get_sequence_number_for_group()
            obj_list = [None for _ in range(dist.get_world_size(subgroup))]
            dist.all_gather_object(obj_list, subgroup_seq, group=subgroup)
            self.assertEqual(len(set(obj_list)), 1)

    def _test_warn_not_in_group(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, creates a group of the even
        # ranks, and for ranks outside it asserts that all_gather, all_reduce, barrier and
        # broadcast on that group each emit a UserWarning (once) matching "does not belong to";
        # ranks inside the group simply run the four collectives.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        in_group_ranks = list(filter(lambda x: x % 2 == 0, range(self.world_size)))
        group = dist.new_group(in_group_ranks)

        x = torch.zeros(2, 2).to(self.rank)
        xs = [torch.zeros(2, 2).to(self.rank) for _ in range(len(in_group_ranks))]
        if self.rank not in in_group_ranks:
            msg = ".*{}.*does not belong to.*"
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_gather")):
                dist.all_gather(xs, x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("all_reduce")):
                dist.all_reduce(x, group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("barrier")):
                dist.barrier(group=group)
            with self.assertWarnsOnceRegex(UserWarning, msg.format("broadcast")):
                dist.broadcast(x, src=0, group=group)
        else:
            dist.all_gather(xs, x, group=group)
            dist.all_reduce(x, group=group)
            dist.barrier(group=group)
            dist.broadcast(x, src=0, group=group)

    def _test_rank_membership(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, creates a group containing only
        # rank 1 and asserts group rank and global rank translation (global 1 is group rank 0 and
        # back), the list of group ranks, and that looking up a rank outside the group raises
        # ValueError "not part of group" and passing an unregistered DummyProcessGroup raises
        # ValueError "not registered".
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        self.assertTrue(self.world_size > 1)

        group = dist.new_group(ranks=[1])
        self.assertEqual(dist.get_group_rank(group, 1), 0)
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_group_rank(group, 0)
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_group_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        self.assertEqual(dist.get_global_rank(group, 0), 1)
        with self.assertRaisesRegex(ValueError, "not part of group"):
            dist.get_global_rank(group, 1)
        with self.assertRaisesRegex(ValueError, "not registered"):
            dist.get_global_rank(DummyProcessGroup(self.rank, self.world_size), 0)

        self.assertEqual(dist.get_process_group_ranks(group), [1])

    def _test_tensor_dtype_mismatch(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, then checks that collectives
        # given a mix of float32 and half tensors (all_gather, coalesced variants,
        # reduce_scatter, all_to_all_single, all_to_all, scatter, gather) each raise ValueError
        # "tensors with different dtypes".
        # KNOWN ISSUE: in the all_gather_coalesced block the second call is never reached because
        # the first one raises inside the same assertRaises context.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        tensor = torch.ones(2, 2, device=self.device) * 7
        tensor_h = tensor.half()
        tensor_list = [
            torch.zeros(2, 2, device=self.device) for _ in range(self.world_size)
        ]
        tensor_list_h = list(tensor_list)
        tensor_list_h[1] = tensor_list_h[1].half()

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list_h, tensor)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather(tensor_list, tensor_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_gather_coalesced([tensor_list_h], tensor_list)
            dist.all_gather_coalesced([tensor_list], tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_reduce_coalesced(tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.reduce_scatter(tensor_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all_single(tensor_h, tensor)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.all_to_all(tensor_list, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.scatter(tensor, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor_h, tensor_list)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.gather(tensor, tensor_list_h)

        with self.assertRaisesRegex(ValueError, "tensors with different dtypes"):
            dist.scatter(tensor_h, tensor_list)

    def _test_tensor_dtype_complex(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group and calls all_gather with real and
        # complex-view (view_as_complex) tensors for input and output lists in all four
        # combinations; passes if nothing raises (no value assertions).
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        tensor = torch.rand(2, device=self.device)
        tensor_c = torch.view_as_complex(tensor)
        tensor_list = [
            torch.rand(2, device=self.device) for _ in range(self.world_size)
        ]
        tensor_list_c = list(tensor_list)
        tensor_list_c[1] = torch.view_as_complex(tensor_list_c[1])

        dist.all_gather(tensor_list, tensor)
        dist.all_gather(tensor_list, tensor_c)
        dist.all_gather(tensor_list_c, tensor)
        dist.all_gather(tensor_list_c, tensor_c)

    def _test_bool_tensors(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, picks the device from the backend
        # (cuda for nccl, xpu for xccl, else cpu), and broadcasts a bool tensor from rank 0
        # (other ranks start with zeros); asserts the result equals rank 0's tensor.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda" if backend == "nccl" else "xpu" if backend == "xccl" else "cpu"
        # test alltoall_base
        tensor = torch.tensor([1, 0, 0, 1], dtype=torch.bool, device=device)
        zeros = torch.tensor([0, 0, 0, 0], dtype=torch.bool, device=device)
        outensor = zeros if self.rank > 0 else tensor
        dist.broadcast(outensor, src=0)
        self.assertEqual(outensor, tensor)


# Variant of AbstractCommTest that expects world size of 4
class AbstractLargeCommTest:
    # [CLASS DOC] Variant of AbstractCommTest mixin with world_size 4, containing helper tests
    # for new_group with local synchronization and with explicit rank ordering. Concrete classes
    # must supply rank, file_name and device.
    @property
    def op_timeout_sec(self):
        # [FUNCTION DOC] HELPER: Property returning 1, the per-operation timeout in seconds.
        return 1

    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning 4, the test world size.
        return 4

    @property
    def device(self):
        # [FUNCTION DOC] HELPER: Property that raises RuntimeError; subclasses must override it.
        raise RuntimeError("Implement me")

    def _test_new_group_local_sync(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, then each rank creates a group of
        # itself and the rank two positions ahead (modulo world size) using local
        # synchronization. Asserts that creating a group for the other ranks returns the non-
        # member sentinel, that the new group is a ProcessGroup, and that group rank follows the
        # sorted order of global ranks and get_process_group_ranks returns the sorted list.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        rank = dist.get_rank()
        ranks_in = [rank, (rank + 2) % self.world_size]
        ranks_out = [i for i in range(self.world_size) if i not in ranks_in]
        self.assertIn(rank, ranks_in)
        self.assertNotIn(rank, ranks_out)

        self.assertIs(
            dist.new_group(ranks=ranks_out, use_local_synchronization=True),
            dist.GroupMember.NON_GROUP_MEMBER,
        )

        new_pg = dist.new_group(ranks=ranks_in, use_local_synchronization=True)
        self.assertIsInstance(new_pg, dist.ProcessGroup)

        # PTD sorts ranks before creating the PG, so [3, 1] actually gets assigned ranks [1, 0]
        ranks_in.sort()
        self.assertEqual(dist.get_group_rank(new_pg, rank), ranks_in.index(rank))
        self.assertEqual(
            ranks_in,
            dist.get_process_group_ranks(new_pg),
            lambda msg: f"{msg}\nexpecting {ranks_in} but got {dist.get_process_group_ranks(new_pg)}",
        )

    def _test_new_group_local_sync_sanity_check(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, splits 4 ranks into two pairs,
        # creates the pair group with local synchronization, all-gathers (pair index, rank)
        # tensors within it and asserts the gathered list equals the tensors of both pair
        # members.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # split the world in 2 PGs
        rank = dist.get_rank()
        pg_idx = rank // 2
        ranks_in = [pg_idx * 2, pg_idx * 2 + 1]
        new_pg = dist.new_group(ranks=ranks_in, use_local_synchronization=True)

        input_tensor = torch.tensor([pg_idx, rank], device=self.device)
        output_tensor_list = [
            torch.tensor(
                [-1, -1],
                device=self.device,
            )
            for _ in range(new_pg.size())
        ]
        dist.all_gather(output_tensor_list, input_tensor, group=new_pg)

        expected = [
            torch.tensor([pg_idx, ranks_in[0]], device=self.device),
            torch.tensor([pg_idx, ranks_in[1]], device=self.device),
        ]
        self.assertEqual(output_tensor_list, expected)

    def _test_new_group_local_sync_duplicate_pg(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, splits ranks into pairs, creates
        # two distinct groups with the same member set using local synchronization (no group name
        # conflict), and checks that an all_gather in each gives the expected (pair index, rank)
        # tensors.
        """
        We should support users create multiple PGs with the same set of
        members, and no conflict in group name
        """
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # split the world in 2 PGs
        rank = dist.get_rank()
        pg_idx = rank // 2
        ranks_in = [pg_idx * 2, pg_idx * 2 + 1]
        new_pgs = []
        for _ in range(2):
            new_pgs.append(
                dist.new_group(ranks=ranks_in, use_local_synchronization=True)
            )

        input_tensor = torch.tensor([pg_idx, rank], device=self.device)
        for new_pg in new_pgs:
            output_tensor_list = [
                torch.tensor(
                    [-1, -1],
                    device=self.device,
                )
                for _ in range(new_pg.size())
            ]
            dist.all_gather(output_tensor_list, input_tensor, group=new_pg)

            expected = [
                torch.tensor([pg_idx, ranks_in[0]], device=self.device),
                torch.tensor([pg_idx, ranks_in[1]], device=self.device),
            ]
            self.assertEqual(output_tensor_list, expected)

    def _test_new_group_ordered(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group and creates a group with ranks in
        # reverse order and sort_ranks False. Asserts this rank's group rank equals its index in
        # the reversed list, that get_process_group_ranks returns the reversed list, and that
        # all_gather output follows the reversed rank order.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        rank = dist.get_rank()

        # Reverse-order ranks: group rank 0 = highest global rank
        reversed_ranks = list(range(self.world_size - 1, -1, -1))
        new_pg = dist.new_group(ranks=reversed_ranks, sort_ranks=False)

        # Verify that the group rank assignment follows the user-provided order
        expected_group_rank = reversed_ranks.index(rank)
        self.assertEqual(dist.get_group_rank(new_pg, rank), expected_group_rank)
        self.assertEqual(
            dist.get_process_group_ranks(new_pg),
            reversed_ranks,
        )

        # Verify that all_gather results follow the custom rank order
        input_tensor = torch.tensor([rank], device=self.device)
        output_tensor_list = [
            torch.tensor([-1], device=self.device) for _ in range(self.world_size)
        ]
        dist.all_gather(output_tensor_list, input_tensor, group=new_pg)

        # Group rank i holds global rank reversed_ranks[i]
        expected = [
            torch.tensor([reversed_ranks[i]], device=self.device)
            for i in range(self.world_size)
        ]
        self.assertEqual(output_tensor_list, expected)


class CommTest(AbstractCommTest, MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case (AbstractCommTest on MultiProcessTestCase) that spawns
    # worker processes in setUp and removes the store file in tearDown. Holds the debug level
    # test.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and then removes the test's file_name,
        # ignoring OSError.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def test_debug_level(self):
        # [FUNCTION DOC] Deletes TORCH_DISTRIBUTED_DEBUG from the environment (if set) and checks
        # the default debug level is OFF. Then for each case-variant spelling of OFF, INFO and
        # DETAIL sets the environment variable and asserts the debug level read back matches, and
        # for invalid values ("foo", 0, 1, -1) asserts set_debug_level_from_env raises ValueError
        # "The value of TORCH_DISTRIBUTED_DEBUG must". Side effect: leaves the last invalid value
        # in the environment variable and does not restore it.
        try:
            del os.environ["TORCH_DISTRIBUTED_DEBUG"]
        except KeyError:
            pass

        dist.set_debug_level_from_env()
        # Default should be off
        default_debug_mode = dist.get_debug_level()
        self.assertEqual(default_debug_mode, dist.DebugLevel.OFF)
        mapping = {
            "OFF": dist.DebugLevel.OFF,
            "off": dist.DebugLevel.OFF,
            "oFf": dist.DebugLevel.OFF,
            "INFO": dist.DebugLevel.INFO,
            "info": dist.DebugLevel.INFO,
            "INfO": dist.DebugLevel.INFO,
            "DETAIL": dist.DebugLevel.DETAIL,
            "detail": dist.DebugLevel.DETAIL,
            "DeTaIl": dist.DebugLevel.DETAIL,
        }
        invalid_debug_modes = ["foo", 0, 1, -1]

        for mode in mapping:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            dist.set_debug_level_from_env()
            set_debug_mode = dist.get_debug_level()
            self.assertEqual(
                set_debug_mode,
                mapping[mode],
                lambda msg: f"{msg}\nExpected {mode} to map to {mapping[mode]} but got {set_debug_mode}",
            )

        for mode in invalid_debug_modes:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = str(mode)
            with self.assertRaisesRegex(
                ValueError, "The value of TORCH_DISTRIBUTED_DEBUG must"
            ):
                dist.set_debug_level_from_env()


class DummyWork(dist._Work):
    # [CLASS DOC] HELPER: Minimal Work subclass for DummyProcessGroup whose wait synchronizes the
    # current accelerator stream (when one is available) and returns True.
    def wait(self, timeout=5.0):
        # [FUNCTION DOC] HELPER: Synchronizes the current accelerator stream if an accelerator is
        # available and returns True; the timeout argument is ignored.
        if torch.accelerator.is_available():
            torch.accelerator.current_stream().synchronize()
        return True


class DummyProcessGroup(dist.ProcessGroup):
    # [CLASS DOC] HELPER: In-Python ProcessGroup used to test the Python extension path. Stores
    # rank and size from the first two positional constructor arguments, a bound device, abort
    # and shutdown flags, and collectives_called, a set of names of collectives dispatched to it.
    # Collectives apply fake arithmetic in place instead of communicating: allgather and gather
    # copy the input into every output; allreduce adds 2; broadcast adds 1; reduce adds 3;
    # recvAnysource adds 4; reduce_scatter and scatter copy the entry for this rank;
    # all_gather_single copies the input into each chunk; reduce_scatter_single copies this
    # rank's chunk; alltoall copies pairwise. send adds 1 and recv adds 2 (these two are not
    # recorded). All return DummyWork.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the parent constructor, then records global_rank and
        # group_size from the first two positional args, initializes bound device None, aborted
        # and shutdown flags False and an empty collectives_called set.
        super().__init__(*args, **kwargs)
        self._bound_device_id = None
        self.global_rank = args[0]
        self.group_size = args[1]
        self._aborted = False
        self._shutdown = False
        # Records the name of every collective dispatched into this PG, so a
        # test can assert a given collective routed through the trampoline.
        self.collectives_called = set()

    def rank(self):
        # [FUNCTION DOC] HELPER: Returns the rank given at construction.
        return self.global_rank

    def size(self):
        # [FUNCTION DOC] HELPER: Returns the group size given at construction.
        return self.group_size

    @property
    def supports_splitting(self):
        # [FUNCTION DOC] HELPER: Property always returning True: this group claims to support
        # splitting.
        return True

    @property
    def bound_device_id(self):
        # [FUNCTION DOC] HELPER: Property returning the stored bound device id (initially None).
        return self._bound_device_id

    @bound_device_id.setter
    def bound_device_id(self, device):
        # [FUNCTION DOC] HELPER: Property setter storing the bound device id.
        self._bound_device_id = device

    def eager_connect_single_device(self, device=None):
        # [FUNCTION DOC] HELPER: Records the given device as the bound device; no connection is
        # made.
        self._bound_device_id = device

    def _get_backend(self, device):
        # [FUNCTION DOC] HELPER: Returns self for any device, so this group serves as its own
        # backend.
        return self

    def comm_split_count(self):
        # [FUNCTION DOC] HELPER: Always returns 0 (no communicator splits are performed).
        return 0

    def perform_nocolor_split(self, device):
        # [FUNCTION DOC] HELPER: No-op split implementation.
        pass

    def getBackendName(self):
        # [FUNCTION DOC] HELPER: Returns the backend name "Dummy".
        return "Dummy"

    def allgather(self, output_tensor_lists, input_tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "allgather"; copies each input tensor into every tensor
        # of the corresponding output list (pairing lists by position) in place. Returns a
        # DummyWork.
        self.collectives_called.add("allgather")
        for output_tensor_list, input_tensor in zip(
            output_tensor_lists, input_tensor_list
        ):
            for output_tensor in output_tensor_list:
                output_tensor.copy_(input_tensor)

        return DummyWork()

    def allreduce(self, tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "allreduce"; adds 2 in place to each tensor and returns
        # a DummyWork.
        self.collectives_called.add("allreduce")
        for tensor in tensor_list:
            tensor.add_(2)

        return DummyWork()

    def barrier(self, opts=None):
        # [FUNCTION DOC] HELPER: Store-based barrier over the default store using a fixed key.
        # Rank 0 spins reading the counter (add of 0) until size - 1 other ranks have incremented
        # it, so it stays alive while others use the store; other ranks increment the counter by
        # 1. Returns a DummyWork. Not recorded in collectives_called.
        store = c10d._get_default_store()
        key = "TEST:DummyProcessGroup:barrier"
        if self.rank() == 0:
            worker_count = 0
            # By default, TCPServer lives on rank 0. So rank 0 needs to make
            # sure that it does not exit too early before other ranks finish
            # using the store.
            # Note that, _store_based_barrier does not solve this problem, as
            # all ranks need to run at least one store.add(key, 0) before
            # exiting, but there is no guarantee that rank 0 is still alive at
            # that point.
            while worker_count < self.size() - 1:
                worker_count = store.add(key, 0)
        else:
            store.add(key, 1)

        return DummyWork()

    def broadcast(self, tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "broadcast"; adds 1 in place to each tensor and returns
        # a DummyWork.
        self.collectives_called.add("broadcast")
        for tensor in tensor_list:
            tensor.add_(1)

        return DummyWork()

    def reduce_scatter(self, output_tensor_list, input_tensor_lists, opts=None):
        # [FUNCTION DOC] HELPER: Records "reduce_scatter"; for each output tensor copies the
        # entry at this rank's index from the matching input list, in place. Returns a DummyWork.
        self.collectives_called.add("reduce_scatter")
        for output_tensor, input_tensor_list in zip(
            output_tensor_list, input_tensor_lists
        ):
            output_tensor.copy_(input_tensor_list[self.rank()])

        return DummyWork()

    def all_gather_single(self, output_tensor, input_tensor, opts=None):
        # [FUNCTION DOC] HELPER: Records "all_gather_single"; copies the input tensor into each
        # of the size equal chunks of the output tensor. Returns a DummyWork.
        self.collectives_called.add("all_gather_single")
        for chunk in output_tensor.chunk(self.size()):
            chunk.copy_(input_tensor)

        return DummyWork()

    def reduce_scatter_single(self, output_tensor, input_tensor, opts=None):
        # [FUNCTION DOC] HELPER: Records "reduce_scatter_single"; copies this rank's chunk of the
        # input (split into size chunks) into the output tensor. Returns a DummyWork.
        self.collectives_called.add("reduce_scatter_single")
        output_tensor.copy_(input_tensor.chunk(self.size())[self.rank()])

        return DummyWork()

    def reduce(self, tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "reduce"; adds 3 in place to each tensor and returns a
        # DummyWork.
        self.collectives_called.add("reduce")
        for tensor in tensor_list:
            tensor.add_(3)

        return DummyWork()

    def gather(self, output_tensor_lists, input_tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "gather"; copies each input tensor into every tensor of
        # the corresponding output list in place. Returns a DummyWork.
        self.collectives_called.add("gather")
        for output_tensor_list, input_tensor in zip(
            output_tensor_lists, input_tensor_list
        ):
            for output_tensor in output_tensor_list:
                output_tensor.copy_(input_tensor)

        return DummyWork()

    def scatter(self, output_tensor_list, input_tensor_lists, opts=None):
        # [FUNCTION DOC] HELPER: Records "scatter"; for each output tensor copies the entry at
        # this rank's index from the matching input list. Returns a DummyWork.
        self.collectives_called.add("scatter")
        for output_tensor, input_tensor_list in zip(
            output_tensor_list, input_tensor_lists
        ):
            output_tensor.copy_(input_tensor_list[self.rank()])

        return DummyWork()

    def alltoall(self, output_tensor_list, input_tensor_list, opts=None):
        # [FUNCTION DOC] HELPER: Records "alltoall"; copies each input tensor into the matching
        # output tensor. Returns a DummyWork.
        self.collectives_called.add("alltoall")
        for output_tensor, input_tensor in zip(output_tensor_list, input_tensor_list):
            output_tensor.copy_(input_tensor)

        return DummyWork()

    def recvAnysource(self, tensor_list, tag):
        # [FUNCTION DOC] HELPER: Records "recvAnysource"; adds 4 in place to each tensor and
        # returns a DummyWork.
        self.collectives_called.add("recvAnysource")
        for tensor in tensor_list:
            tensor.add_(4)

        return DummyWork()

    def send(self, tensor_list, dst, tag=0):
        # [FUNCTION DOC] HELPER: Adds 1 in place to each tensor (fake send; dst and tag ignored,
        # nothing recorded) and returns a DummyWork.
        for tensor in tensor_list:
            tensor.add_(1)

        return DummyWork()

    def recv(self, tensor_list, src, tag=0):
        # [FUNCTION DOC] HELPER: Adds 2 in place to each tensor (fake recv; src and tag ignored,
        # nothing recorded) and returns a DummyWork.
        for tensor in tensor_list:
            tensor.add_(2)

        return DummyWork()

    def abort(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the aborted flag.
        self._aborted = True

    def shutdown(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the shutdown flag.
        self._shutdown = True


class BackendRegistrationTest(TestCase):
    # [CLASS DOC] Tests for registering custom backends with the dist.Backend registry.
    def test_register_backend_with_single_device_string(self):
        # [FUNCTION DOC] Registers a backend named "_test_fake_backend" for the single device
        # string "custom_device" and asserts the default device map points that device to the
        # backend, the capability is the one-element list holding the device (not a per-character
        # split of the string), and parsing the backend name against that device set yields the
        # device mapping; also asserts no single character of the device name was registered as a
        # device. Snapshots all dist.Backend registries and the upper-case attribute beforehand
        # and restores them in a finally block.
        name = "_test_fake_backend"
        device = "custom_device"
        backend_attr = name.upper()
        old_backend_attr = getattr(dist.Backend, backend_attr, None)
        had_backend_attr = hasattr(dist.Backend, backend_attr)
        old_backend_list = dist.Backend.backend_list.copy()
        old_default_device_backend_map = dist.Backend.default_device_backend_map.copy()
        old_backend_capability = dist.Backend.backend_capability.copy()
        old_backend_type_map = dist.Backend.backend_type_map.copy()
        old_plugins = dist.Backend._plugins.copy()

        try:

            def create_backend(*_args, **_kwargs):
                # [FUNCTION DOC] HELPER: Dummy backend factory returning None.
                return None

            dist.Backend.register_backend(name, create_backend, devices=device)

            self.assertEqual(dist.Backend.default_device_backend_map[device], name)
            self.assertEqual(dist.Backend.backend_capability[name], [device])
            self.assertEqual(
                c10d._parse_backend_string(name, available_devices={device}),
                {device: name},
            )
            for character in device:
                self.assertNotEqual(
                    dist.Backend.default_device_backend_map.get(character), name
                )
        finally:
            if had_backend_attr:
                setattr(dist.Backend, backend_attr, old_backend_attr)
            elif hasattr(dist.Backend, backend_attr):
                delattr(dist.Backend, backend_attr)
            dist.Backend.backend_list = old_backend_list
            dist.Backend.default_device_backend_map = old_default_device_backend_map
            dist.Backend.backend_capability = old_backend_capability
            dist.Backend.backend_type_map = old_backend_type_map
            dist.Backend._plugins = old_plugins


class PythonProcessGroupExtensionTest(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test (world size from the harness) for Python-defined process
    # group backends, using DummyProcessGroup registered as the "dummy" backend via the static
    # factories create_dummy and create_dummy_ext. Many tests set MASTER_ADDR and MASTER_PORT
    # (localhost, 6789) in os.environ and initialize and destroy the default group. Nested
    # Options is an inert stand-in for group options.
    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and removes the test's file_name, ignoring
        # OSError.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def test_get_backend_name(self):
        # [FUNCTION DOC] Asserts a DummyProcessGroup of rank 0 and size 1 reports its name as
        # "Dummy". (The rest of the body is commented-out code.)
        dpg = DummyProcessGroup(0, 1)
        self.assertEqual("Dummy", dpg.name())

        # dist.Backend.register_backend(
        #     "dummy", PythonProcessGroupExtensionTest.create_dummy
        # )

        # # os.environ["MASTER_ADDR"] = "localhost"
        # # os.environ["MASTER_PORT"] = "6789"
        # # dist.init_process_group(
        # #     "cpu:dummy", rank=0, world_size=1,
        # # )
        # dpg = DummyProcessGroup(0, 1)
        # from torch.distributed.distributed_c10d import _canonicalize_group_rank
        # self.assertEqual(123, _canonicalize_group_rank(dpg, group_rank=123, return_global=False))
        # with self.assertRaises(RuntimeError):
        # _canonicalize_group_rank(dpg, group_rank=123, return_global=True)

    def test_get_backend_impl(self):
        # [FUNCTION DOC] Registers the dummy backend and initializes the default group with it
        # (sets MASTER_ADDR/MASTER_PORT). Asserts get_backend_impl returns the DummyProcessGroup,
        # that it is the same object as the group's backend for the cpu device (with and without
        # a device argument), and that looking it up by group name gives a backend named "Dummy".
        # Destroys the group.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        backend = dist.get_backend_impl()
        self.assertIsInstance(backend, DummyProcessGroup)
        pg = c10d._get_default_group()
        self.assertIs(pg.get_backend(torch.device("cpu")), backend)
        self.assertIs(dist.get_backend_impl(device=torch.device("cpu")), backend)

        group_name = c10d._get_process_group_name(pg)
        self.assertEqual(
            dist.get_backend_impl(str(group_name)).getBackendName(), "Dummy"
        )

        dist.destroy_process_group()

    def test_canonicalize_helper(self):
        # [FUNCTION DOC] Initializes a default group with the dummy backend, then checks the
        # private rank canonicalization helper on a separate DummyProcessGroup of size 124: a
        # group rank 123 passes through unchanged, global rank 0 maps to 0, and converting group
        # rank 123 to a global rank raises ValueError.
        # KNOWN ISSUE: per the in-test TODO the ValueError actually comes from the group not
        # being registered, not from the rank being absent from the global mapping. Destroys the
        # group.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        dpg = DummyProcessGroup(0, 124)
        from torch.distributed.distributed_c10d import _canonicalize_group_rank

        # we ensure that a process group with more ranks than the 'default' group can still be used.
        # e.g. if the dpg had 124 ranks and the world had only 2 ranks.
        self.assertEqual(
            123, _canonicalize_group_rank(dpg, group_rank=123, return_global=False)
        )
        self.assertEqual(
            0, _canonicalize_group_rank(dpg, global_rank=0, return_global=True)
        )
        with self.assertRaises(ValueError):
            # TODO(whc) this is actually catching the wrong error:
            # ValueError: Group <__mp_main__.DummyProcessGroup object at 0x7faa0a844540> is not registered,
            # please create group with torch.distributed.new_group API
            # It should be catching a different error where the rank doesn't exist in the global mapping.
            # But it's still testing the same part of the _canonicalize_group_rank helper so maybe this is fine
            _canonicalize_group_rank(dpg, group_rank=123, return_global=True)

        dist.destroy_process_group()

    def test_backend_class_attr(self):
        # [FUNCTION DOC] Registers the dummy backend and asserts the class attribute
        # Backend.DUMMY equals "dummy" and the plugin entry's creator is create_dummy. Does not
        # restore the registration afterwards.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )
        self.assertEqual(dist.Backend.DUMMY, "dummy")
        self.assertEqual(
            dist.Backend._plugins["DUMMY"].creator_fn,
            PythonProcessGroupExtensionTest.create_dummy,
        )

    def test_is_backend_available(self):
        # [FUNCTION DOC] Asserts is_ucc_available matches is_backend_available for "ucc", that
        # "dummy" is unavailable before registration and available after registering it. Leaves
        # the registration in place.
        self.assertEqual(dist.is_ucc_available(), dist.is_backend_available("ucc"))
        self.assertFalse(dist.is_backend_available("dummy"))
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )
        self.assertTrue(dist.is_backend_available("dummy"))

    def test_backend_config(self):
        # [FUNCTION DOC] Registers the dummy backend and checks the string form of BackendConfig
        # for plain backend names (default device mapping, with the accelerator replaced on Intel
        # GPU runs), upper/lower-case dummy names and explicit device:backend lists, each in a
        # subTest. Also asserts that a trailing comma or a repeated device type raises
        # ValueError.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )
        dummy_backend_config = (
            f"cpu:dummy,{device_type}:dummy" if device_type != "cpu" else "cpu:dummy"
        )

        # Ensure backend config can be created with the following arguments
        backend_config_strings_and_expected_values = [
            (dist.Backend.GLOO, "cpu:gloo,cuda:gloo"),
            (dist.Backend.NCCL, "cuda:nccl"),
            (dist.Backend.MPI, "cpu:mpi,cuda:mpi"),
            (dist.Backend.UCC, "cpu:ucc,cuda:ucc"),
            (dist.Backend.DUMMY, dummy_backend_config),
            ("DUMMY", dummy_backend_config),
            ("dummy", dummy_backend_config),
            ("cpu:dummy,cuda:dummy", "cpu:dummy,cuda:dummy"),
            ("cpu:dummy,cuda:nccl", "cpu:dummy,cuda:nccl"),
            ("cpu:gloo,cuda:dummy", "cpu:gloo,cuda:dummy"),
            ("cpu:gloo,cuda:nccl", "cpu:gloo,cuda:nccl"),
        ]

        if TEST_XPU:
            # Override backend_config_strings_and_expected_values for Intel GPU.
            backend_config_strings_and_expected_values[4:10] = [
                (dist.Backend.DUMMY, dummy_backend_config),
                ("DUMMY", dummy_backend_config),
                ("dummy", dummy_backend_config),
                ("cpu:dummy,xpu:dummy", "cpu:dummy,xpu:dummy"),
                ("cpu:dummy,xpu:xccl", "cpu:dummy,xpu:xccl"),
                ("cpu:gloo,xpu:dummy", "cpu:gloo,xpu:dummy"),
                ("cpu:gloo,xpu:xccl", "cpu:gloo,xpu:xccl"),
            ]

        for config_str, expected_value in backend_config_strings_and_expected_values:
            with self.subTest(config_str):
                # ensures these configs strings are valid and no ValueError is raised
                config = dist.BackendConfig(config_str)
                self.assertEqual(str(config), expected_value)

        # Ensure backend config will raise ValueError with the following arguments
        invalid_backend_config_strings = [
            "cpu:gloo,cuda:nccl,",  # trailing comma
            "cpu:gloo,cuda:nccl,cpu:dummy",  # duplicate device
            "cpu:gloo,xpu:xccl,",  # trailing comma
            "cpu:gloo,xpu:xccl,cpu:dummy",  # duplicate device
        ]
        for config_str in invalid_backend_config_strings:
            with self.subTest(config_str):
                with self.assertRaises(ValueError):
                    dist.BackendConfig(config_str)

    def test_parse_backend_string(self):
        # [FUNCTION DOC] Exercises the private backend string parser with an available device set
        # of cpu, cuda, xpu and mps. Checks that a bare backend name maps to the devices where it
        # is the registered default (case-insensitive; gloo gives cpu and mps), that
        # device:backend lists are parsed case-insensitively and whitespace-tolerantly and accept
        # unknown device types, and that errors are raised as ValueError for an unknown backend,
        # a malformed pairing (extra colon or missing device) and a duplicate device type.
        from torch.distributed.distributed_c10d import _parse_backend_string

        all_devices = {"cpu", "cuda", "xpu", "mps"}

        # Simple form maps to device(s) where the backend is the registered default.
        self.assertEqual(
            _parse_backend_string("nccl", available_devices=all_devices),
            {"cuda": "nccl"},
        )
        self.assertEqual(
            _parse_backend_string("NCCL", available_devices=all_devices),
            {"cuda": "nccl"},
        )
        # gloo is the default for both cpu and mps in default_device_backend_map.
        self.assertEqual(
            _parse_backend_string("gloo", available_devices=all_devices),
            {"cpu": "gloo", "mps": "gloo"},
        )

        # Merged form returns exactly what was named, filtered by available_devices.
        self.assertEqual(
            _parse_backend_string("cpu:gloo,cuda:nccl", available_devices=all_devices),
            {"cpu": "gloo", "cuda": "nccl"},
        )
        self.assertEqual(
            _parse_backend_string(
                "CPU:GLOO , CUDA:NCCL", available_devices=all_devices
            ),
            {"cpu": "gloo", "cuda": "nccl"},
        )
        # Unknown device types in merged form are accepted (no validation here).
        self.assertEqual(
            _parse_backend_string("xpu:nccl", available_devices=all_devices),
            {"xpu": "nccl"},
        )

        # Errors.
        with self.assertRaisesRegex(ValueError, "Unknown backend"):
            _parse_backend_string(
                "definitely_not_a_backend", available_devices=all_devices
            )
        with self.assertRaisesRegex(ValueError, "Invalid device:backend pairing"):
            _parse_backend_string("cpu:gloo:extra", available_devices=all_devices)
        with self.assertRaisesRegex(ValueError, "Invalid device:backend pairing"):
            _parse_backend_string("cpu:gloo,bare", available_devices=all_devices)
        with self.assertRaisesRegex(ValueError, "Duplicate device type"):
            _parse_backend_string("cpu:gloo,cpu:dummy", available_devices=all_devices)

    def test_init_process_group_with_multiple_backends(self):
        # [FUNCTION DOC] Registers the dummy backend, initializes the default group with a
        # backend string naming dummy for cpu, cuda and xpu, runs an all_gather on cpu tensors
        # and a barrier, and destroys the group; passes if nothing raises.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "cpu:dummy,cuda:dummy,xpu:dummy", rank=self.rank, world_size=self.world_size
        )

        # test all_gather
        input_tensor = torch.ones(2, 2) * 7
        output_tensor_list = [torch.zeros(2, 2) for _ in range(self.world_size)]
        dist.all_gather(output_tensor_list, input_tensor)

        dist.barrier()
        dist.destroy_process_group()

    class Options:
        # [CLASS DOC] HELPER: Inert placeholder for process group options with class attributes
        # group_name, split_from, split_color and global_ranks_in_group all None; its constructor
        # and create do nothing.
        group_name = None
        split_from = None
        split_color = None
        global_ranks_in_group = None

        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: No-op constructor.
            pass

        def create(self):
            # [FUNCTION DOC] HELPER: No-op.
            pass

    @staticmethod
    def create_dummy(store, group_rank, group_size, timeout):
        # [FUNCTION DOC] HELPER: Staticmethod backend factory with the legacy signature (store,
        # rank, size, timeout); returns a DummyProcessGroup for that rank and size.
        return DummyProcessGroup(group_rank, group_size)

    @staticmethod
    def create_dummy_ext(dist_opts, pg_options=None):
        # [FUNCTION DOC] HELPER: Staticmethod backend factory for the extended API; returns a
        # DummyProcessGroup built from the group_rank and group_size of the options object
        # (pg_options is ignored).
        return DummyProcessGroup(dist_opts.group_rank, dist_opts.group_size)

    def test_collectives(self):
        # [FUNCTION DOC] Registers and initializes the dummy backend, then checks the fake
        # collective semantics end to end: all_gather copies the input into each output,
        # all_reduce adds 2, an async broadcast adds 1 after waiting, reduce_scatter copies this
        # rank's entry; then barrier and destroy.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        # test all_gather
        input_tensor = torch.ones(2, 2) * 7
        output_tensor_list = [torch.zeros(2, 2) for _ in range(self.world_size)]
        dist.all_gather(output_tensor_list, input_tensor)

        for tensor in output_tensor_list:
            self.assertEqual(tensor, input_tensor)

        # test all_reduce
        input_tensor = torch.ones(2, 2) * 7
        dist.all_reduce(input_tensor)
        self.assertEqual(input_tensor, torch.ones(2, 2) * 7 + 2)

        # test broadcast
        input_tensor = torch.zeros(2, 2)
        dist.broadcast(input_tensor, 0, async_op=True).wait()
        self.assertEqual(torch.ones(2, 2), input_tensor)

        # test reduce_scatter
        output_tensor = torch.zeros(2, 2)
        input_tensor_list = [torch.ones(2, 2) for _ in range(self.world_size)]
        dist.reduce_scatter(output_tensor, input_tensor_list)
        self.assertEqual(output_tensor, torch.zeros(2, 2) + 1)

        dist.barrier()
        dist.destroy_process_group()

    def test_send_recv(self):
        # [FUNCTION DOC] Initializes the dummy backend and checks send adds 1 to the tensor and
        # recv adds 2. Asserts ValueError for send to self rank, send with only group_dst equal
        # to self rank, with both dst and group_dst, or with neither, and for recv with both src
        # and group_src. Ends with a barrier and intentionally does not destroy the group.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        # test send
        input_tensor = torch.zeros(2, 2)
        dist.send(input_tensor, (self.rank + 1) % self.world_size)
        self.assertEqual(input_tensor, torch.zeros(2, 2) + 1)

        with self.assertRaises(ValueError):
            dist.send(input_tensor, dist.get_rank())
        with self.assertRaises(ValueError):
            dist.send(input_tensor, group_dst=dist.get_rank())

        with self.assertRaises(ValueError):
            dist.send(input_tensor, dist.get_rank(), group_dst=dist.get_rank())
        with self.assertRaises(ValueError):
            dist.send(input_tensor)

        # test recv
        input_tensor = torch.zeros(2, 2)
        dist.recv(input_tensor, (self.rank + 1) % self.world_size)
        self.assertEqual(input_tensor, torch.zeros(2, 2) + 2)
        with self.assertRaises(ValueError):
            dist.recv(input_tensor, src=0, group_src=0)

        dist.barrier()
        # intentionally not calling into `destroy_process_group` as not all
        # user applications would explicitly that.

    def test_shutdown(self) -> None:
        # [FUNCTION DOC] Initializes the dummy backend, destroys the default group and asserts
        # the dummy group's shutdown method was called.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        pg = c10d._get_default_group()

        dist.destroy_process_group()

        self.assertTrue(pg._shutdown)

    def test_abort(self) -> None:
        # [FUNCTION DOC] Initializes the dummy backend, calls the private abort-process-group
        # function and asserts the dummy group's abort method was called.
        dist.Backend.register_backend(
            "dummy", PythonProcessGroupExtensionTest.create_dummy
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group("dummy", rank=self.rank, world_size=self.world_size)

        pg = c10d._get_default_group()

        c10d._abort_process_group()

        self.assertTrue(pg._aborted)

    def test_new_group_delegates_to_pg(self):
        # [FUNCTION DOC] Defines a DummyProcessGroup subclass whose new_group method records the
        # call (ranks, backend) and returns a dummy sub-group for member ranks or None otherwise.
        # Registers it as a "delegating" extended-API backend, initializes the default group and
        # asserts dist.new_group delegates to it with the given ranks and the backend name
        # "delegating"; rank 0 gets a real group, other ranks get None or the non-member
        # sentinel. Destroys the group in a finally block.
        """dist.new_group delegates to default_pg.new_group if available."""

        new_group_called = False
        new_group_ranks = None
        new_group_backend = None

        class _DelegatingPG(DummyProcessGroup):
            # [CLASS DOC] HELPER: DummyProcessGroup subclass adding a new_group method that records
            # its arguments in the enclosing test and returns a dummy sub-group.
            def new_group(
                self,
                ranks,
                timeout=None,
                backend=None,
                pg_options=None,
                group_name=None,
                group_desc=None,
            ):
                # [FUNCTION DOC] HELPER: Records in the enclosing test that it was called with
                # the ranks and backend; returns None if this rank is not in ranks, else a
                # DummyProcessGroup with the rank's index and the number of ranks.
                nonlocal new_group_called, new_group_ranks, new_group_backend
                new_group_called = True
                new_group_ranks = list(ranks)
                new_group_backend = backend
                my_rank = self.rank()
                if my_rank not in ranks:
                    return None
                return DummyProcessGroup(ranks.index(my_rank), len(ranks))

        dist.Backend.register_backend(
            "delegating",
            lambda *args, **kwargs: _DelegatingPG(
                args[0].group_rank, args[0].group_size
            ),
            extended_api=True,
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "delegating", rank=self.rank, world_size=self.world_size
        )

        try:
            sub_pg = dist.new_group(ranks=[0])
            self.assertTrue(new_group_called)
            self.assertEqual(new_group_ranks, [0])
            self.assertEqual(new_group_backend, "delegating")

            if self.rank == 0:
                self.assertIsNotNone(sub_pg)
                self.assertNotEqual(
                    sub_pg, dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER
                )
            else:
                self.assertTrue(
                    sub_pg is None
                    or sub_pg == dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER
                )
        finally:
            dist.destroy_process_group()

    def test_new_group_delegates_to_pg_explicit_backend(self):
        # [FUNCTION DOC] Same delegating setup but registered for devices cpu and cuda:
        # initializes the default group, calls dist.new_group with the explicit backend string
        # "cpu:delegating,cuda:delegating" and asserts the delegated new_group received that
        # exact string. Destroys the group in a finally block.
        """dist.new_group forwards an explicit multi-backend string to the
        delegated default_pg.new_group."""

        new_group_called = False
        new_group_backend = None

        class _DelegatingPG(DummyProcessGroup):
            # [CLASS DOC] HELPER: DummyProcessGroup subclass adding a new_group method that records
            # its arguments in the enclosing test and returns a dummy sub-group.
            def new_group(
                self,
                ranks,
                timeout=None,
                backend=None,
                pg_options=None,
                group_name=None,
                group_desc=None,
            ):
                # [FUNCTION DOC] HELPER: Records in the enclosing test that it was called and the
                # backend it received; returns None if this rank is not in ranks, else a
                # DummyProcessGroup with the rank's index and the number of ranks.
                nonlocal new_group_called, new_group_backend
                new_group_called = True
                new_group_backend = backend
                my_rank = self.rank()
                if my_rank not in ranks:
                    return None
                return DummyProcessGroup(ranks.index(my_rank), len(ranks))

        dist.Backend.register_backend(
            "delegating",
            lambda *args, **kwargs: _DelegatingPG(
                args[0].group_rank, args[0].group_size
            ),
            extended_api=True,
            devices=["cpu", "cuda"],
        )

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "6789"
        dist.init_process_group(
            "delegating", rank=self.rank, world_size=self.world_size
        )

        try:
            backend = "cpu:delegating,cuda:delegating"
            dist.new_group(ranks=[0], backend=backend)
            self.assertTrue(new_group_called)
            self.assertEqual(new_group_backend, backend)
        finally:
            dist.destroy_process_group()


instantiate_parametrized_tests(CommonDistributedDataParallelTest)


class SplitGroupOptionsTest(TestCase):
    # [CLASS DOC] Tests how ProcessGroup.split_group passes options to the backends it splits.
    # Nested _SplittingBackend is a Python Backend whose split records the options it receives
    # and returns a child backend; _make_group builds a group of two such backends.
    class _SplittingBackend(C10DBackend):
        # [CLASS DOC] HELPER: Python Backend subclass with a name, a fixed options object (111
        # second timeout) and split_opts recording the options passed to its split method; it
        # reports that it supports splitting.
        def __init__(self, rank, size, name):
            # [FUNCTION DOC] HELPER: Calls the Backend constructor, stores the name, creates
            # options with that name and an 111 second timeout, and initializes split_opts to
            # None.
            super().__init__(rank, size)
            self._name = name
            self._options = C10DBackend.Options(name, timeout=timedelta(seconds=111))
            self.split_opts = None

        @property
        def supports_splitting(self):
            # [FUNCTION DOC] HELPER: Property always returning True.
            return True

        @property
        def options(self):
            # [FUNCTION DOC] HELPER: Property returning the backend's own options object.
            return self._options

        def getBackendName(self):
            # [FUNCTION DOC] HELPER: Returns the backend name given at construction.
            return self._name

        def split(self, store, ranks, opts):
            # [FUNCTION DOC] HELPER: Stores the options it was given in split_opts and returns a
            # new _SplittingBackend for this rank's index within ranks, sized to the rank list
            # and named with the "-child" suffix added to its own name.
            self.split_opts = opts
            return SplitGroupOptionsTest._SplittingBackend(
                ranks.index(self.rank()), len(ranks), f"{self._name}-child"
            )

    def _make_group(self):
        # [FUNCTION DOC] HELPER: Builds a one-rank ProcessGroup over a HashStore with two Python
        # backends: "cpu-backend" registered for the cpu device (GLOO type tag) and "default-
        # backend" for cuda (NCCL tag), makes NCCL the default backend type and sets the group
        # name. Returns (pg, cpu_backend, default_backend).
        # Shaped like a "cpu:gloo,cuda:nccl" group: two distinct backends, the
        # accelerator one being the group's default. The backend type tags are
        # just map keys here, the backends themselves are Python ones.
        cpu_backend = self._SplittingBackend(0, 1, "cpu-backend")
        default_backend = self._SplittingBackend(0, 1, "default-backend")
        pg = dist.ProcessGroup(dist.HashStore(), 0, 1)
        pg._register_backend(
            torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO, cpu_backend
        )
        pg._register_backend(
            torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL, default_backend
        )
        pg._set_default_backend(dist.ProcessGroup.BackendType.NCCL)
        pg._set_group_name("split-options-test")
        return pg, cpu_backend, default_backend

    def test_split_group_clones_parent_options(self):
        # [FUNCTION DOC] Splits the group with a new timeout and group name and asserts that for
        # both backends the options passed to split are a different object from the backend's own
        # options, that the parent's options keep an empty group name and the original 111 second
        # timeout, and that the split options carry the new name and 222 second timeout.
        # getBackendOptions() returns the backend's live options_, and split()
        # implementations write into what they are given, so splitGroup used to
        # rewrite the parent's group_name/timeout and hand the child an object
        # aliasing the parent's.
        pg, cpu_backend, default_backend = self._make_group()
        pg.split_group([0], timeout=timedelta(seconds=222), group_name="child")

        for backend in (cpu_backend, default_backend):
            self.assertIsNot(backend.split_opts, backend.options)
            self.assertEqual(backend.options.group_name, "")
            self.assertEqual(backend.options._timeout, timedelta(seconds=111))
            self.assertEqual(backend.split_opts.group_name, "child")
            self.assertEqual(backend.split_opts._timeout, timedelta(seconds=222))

    def test_split_group_opts_apply_to_default_backend_only(self):
        # [FUNCTION DOC] Splits the group passing an explicit options object plus a timeout and
        # asserts the explicit options are used only for the default backend (its split options
        # carry the caller's backend name) while the cpu backend gets its own options. Also
        # asserts the caller's object is a different object from what was passed and is not
        # mutated (group name stays empty, timeout stays 5 seconds).
        # An explicit `opts` describes the group's default backend, the same way
        # pg_options does for init_process_group. Handing it to every device's
        # backend made the others reject it (and silently fall back to their
        # defaults, losing the caller's timeout).
        pg, cpu_backend, default_backend = self._make_group()
        opts = C10DBackend.Options("caller-supplied", timeout=timedelta(seconds=5))
        pg.split_group([0], opts=opts, timeout=timedelta(seconds=222))

        self.assertEqual(default_backend.split_opts.backend, "caller-supplied")
        self.assertEqual(cpu_backend.split_opts.backend, "cpu-backend")
        # The caller's object is not mutated either.
        self.assertIsNot(default_backend.split_opts, opts)
        self.assertEqual(opts.group_name, "")
        self.assertEqual(opts._timeout, timedelta(seconds=5))


class ProcessGroupWithDispatchedCollectivesTests(MultiProcessTestCase):
    # [CLASS DOC] Single-process (world size 1) multi-process tests checking that process group
    # initialization and collectives dispatch for various backends. The _test_* helpers are
    # called from backend-specific subclasses with a backend name.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning 1.
        return 1

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and removes the test's file_name, ignoring
        # OSError.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def test_init_process_group_optional_backend(self):
        # [FUNCTION DOC] When both gloo and nccl are available, initializes the default group
        # without naming a backend (using a FileStore) and destroys it; does nothing otherwise.
        store = dist.FileStore(self.file_name, self.world_size)
        # creates both gloo and nccl backend
        if dist.is_gloo_available() and dist.is_nccl_available():
            dist.init_process_group(
                store=store,
                rank=self.rank,
                world_size=self.world_size,
            )
            dist.destroy_process_group()

    def test_init_process_group_for_all_backends(self):
        # [FUNCTION DOC] Loops over every name in the Backend registry list, skipping undefined
        # and backends unavailable on the system (mpi, nccl without cuda, gloo, ucc, xccl).
        # Initializes a group with each on a FileStore and asserts rank, size and that the group
        # name equals the backend name, except that unrecognized plugin names are expected to
        # report "custom" (the "threaded" backend reports its own name). Destroys the group after
        # each.
        for backend in dist.Backend.backend_list:
            excepted_backend = backend
            # skip if the backend is not available on the system
            if backend == dist.Backend.UNDEFINED:
                continue
            elif backend == dist.Backend.MPI:
                if not dist.is_mpi_available():
                    continue
            elif backend == dist.Backend.NCCL:
                if not dist.is_nccl_available() or not torch.cuda.is_available():
                    continue
            elif backend == dist.Backend.GLOO:
                if not dist.is_gloo_available():
                    continue
            elif backend == dist.Backend.UCC:
                if not dist.is_ucc_available():
                    continue
            elif backend == dist.Backend.XCCL:
                if not dist.is_xccl_available():
                    continue
            # Multi-threaded PG is defined as a pure python class.
            # Its pg.name() does not going through Pybind, so its backend name
            # is still "threaded" instead of "custom".
            elif backend != "threaded":
                excepted_backend = "custom"

            store = dist.FileStore(self.file_name, self.world_size)
            dist.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                store=store,
            )
            pg = c10d._get_default_group()
            self.assertEqual(pg.rank(), self.rank)
            self.assertEqual(pg.size(), self.world_size)
            self.assertEqual(pg.name(), str(excepted_backend))

            dist.destroy_process_group()

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    def test_default_process_group(self):
        # [FUNCTION DOC] Skipped under fbcode or sandcastle. Runs a child Python process, with
        # GPU visibility disabled through environment variables, that initializes the default
        # process group with a HashStore and no backend, expecting it to fall back to CPU. Fails
        # the test on a 20 second timeout or a non-zero exit, including the process output in the
        # failure message.
        script = """
# Hide all GPUs
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "!*:gpu"

import torch
from torch import distributed as dist

# This should initialize on CPU even though this is a CUDA-enabled build
dist.init_process_group(rank=0, world_size=1, store=dist.HashStore())
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
                # It is ok to have an extra long timeout here as a timeout means the test failed
                timeout=20,
            )
        except subprocess.TimeoutExpired:
            self.fail(
                msg="Example code timed out! See the code sample in the test for details."
            )
        except subprocess.CalledProcessError as e:
            self.fail(f"""Subprocess failed with {e.output.decode("utf-8")}""")

    def _call_collective_with_varying_tensors(self, backend, collective, *args):
        # [FUNCTION DOC] HELPER: Calls the given collective function with a zero 2x2 tensor on
        # the device matching the backend (cuda for nccl, xpu for xccl, else cpu), shaping
        # arguments per collective: barrier takes none, all_gather and gather take lists, scatter
        # takes a list input, reduce_scatter and all_to_all are only called when the backend is
        # not gloo, anything else takes the tensor plus the extra args.
        # call collective with varying tensors to ensure that the tensors are
        # correctly dispatched

        # TODO: this will be updated in the future to not be backend specific
        device = "cuda" if backend == "nccl" else "xpu" if backend == "xccl" else "cpu"
        # ensure supported devices (cpu, cuda) succeeds during dispatch call
        tensor = torch.zeros(2, 2, device=torch.device(device))
        # multi tensor collectives
        if collective is dist.barrier:
            collective()
        elif collective in (dist.all_gather, dist.gather):
            collective([tensor], tensor, *args)
        elif collective is dist.scatter:
            collective(tensor, [tensor], *args)
        elif collective in (dist.reduce_scatter, dist.all_to_all):
            # gloo does not support reduce_scatter or all_to_all
            if backend != "gloo":
                if collective is dist.reduce_scatter:
                    collective(tensor, [tensor], *args)
                else:
                    collective([tensor], [tensor], *args)
        else:
            collective(tensor, *args)

    # TODO: backend will be replaced with a non specified backend
    def _test_collectives(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group on the backend with a FileStore,
        # then in a subTest for each of reduce, broadcast, all_reduce, all_gather,
        # reduce_scatter, barrier, all_to_all and scatter calls the collective with varying
        # tensors, checking only that dispatch does not raise.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        collectives_and_args = [
            (dist.reduce, self.rank),
            (dist.broadcast, self.rank),
            (dist.all_reduce,),
            (dist.all_gather,),
            (dist.reduce_scatter,),
            (dist.barrier,),
            (dist.all_to_all,),
            (dist.scatter,),
        ]
        for collective, *args in collectives_and_args:
            with self.subTest(collective=collective, args=args):
                self._call_collective_with_varying_tensors(backend, collective, *args)

    def _test_allreduce_coalesced(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, runs all_reduce_coalesced with a
        # sum on a 10x10 ones tensor on the backend's device and asserts each tensor equals ones
        # times the world size.
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        # TODO: this will be updated in the future to not be backend specific
        device = "cuda" if backend == "nccl" else "xpu" if backend == "xccl" else "cpu"
        tensors = [torch.ones(10, 10, device=torch.device(device))]
        dist.all_reduce_coalesced(tensors, dist.ReduceOp.SUM)
        for tensor in tensors:
            self.assertEqual(tensor, torch.ones(10, 10) * self.world_size)

    def _test_all_to_all_single(self, backend):
        # [FUNCTION DOC] HELPER: Initializes the default group, runs all_to_all_single on 2x2
        # tensors on the backend's device, then asserts that a non-contiguous (transposed) input
        # raises ValueError "Tensors must be contiguous".
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend,
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda" if backend == "nccl" else "xpu" if backend == "xccl" else "cpu"
        # test alltoall_base
        input_tensor = torch.ones(2, 2, device=torch.device(device))
        output_tensor = torch.zeros(2, 2, device=torch.device(device))
        dist.all_to_all_single(output_tensor, input_tensor)

        input_tensor = input_tensor.t()
        with self.assertRaisesRegex(ValueError, "Tensors must be contiguous"):
            dist.all_to_all_single(output_tensor, input_tensor)


class ReduceOpTest(TestCase):
    # [CLASS DOC] Tests of ReduceOp objects: isinstance relations, copy, pickle and equality for
    # all built-in ops and premul-sum ops.
    # Ref: https://github.com/pytorch/pytorch/issues/87191
    def test_op_isinstance_of_reduceop(self):
        # [FUNCTION DOC] Asserts every built-in op is an instance of ReduceOp, and for tensor and
        # float scale factors that the private premul-sum constructor and PREMUL_SUM call return
        # ReduceOp instances whose factor equals the scale.
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            self.assertTrue(isinstance(reduce_op, c10d.ReduceOp))
        for scale in (torch.tensor(1.0), 2.0):
            self.assertIsInstance(dist._make_nccl_premul_sum(scale), c10d.ReduceOp)
            premul_sum = c10d.ReduceOp.PREMUL_SUM(scale)
            self.assertIsInstance(premul_sum, c10d.ReduceOp)
            self.assertEqual(premul_sum.factor, scale)

    # Ref: https://github.com/pytorch/pytorch/pull/87303#discussion_r1002879700
    def test_reduceop_copyable(self):
        # [FUNCTION DOC] Asserts shallow and deep copies (of the op and of a ReduceOp wrapped
        # around it) equal the original for every built-in op, and the same for premul-sum ops
        # with tensor and float scales.
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            self.assertEqual(copy.copy(reduce_op), reduce_op)
            self.assertEqual(copy.deepcopy(reduce_op), reduce_op)
            self.assertEqual(copy.copy(c10d.ReduceOp(reduce_op)), reduce_op)
            self.assertEqual(copy.deepcopy(c10d.ReduceOp(reduce_op)), reduce_op)

        for scale in (torch.tensor(1.0), 2.0):
            reduce_op = dist._make_nccl_premul_sum(scale)
            self.assertEqual(copy.copy(reduce_op), reduce_op)
            self.assertEqual(copy.deepcopy(reduce_op), reduce_op)

    def test_reduceop_pickle(self):
        # [FUNCTION DOC] Asserts every built-in op can be pickled and unpickled, and a ReduceOp
        # wrapping the op and premul-sum ops round trip to an equal object.
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            pickle.loads(pickle.dumps(reduce_op))
            orig = c10d.ReduceOp(reduce_op)
            self.assertEqual(pickle.loads(pickle.dumps(orig)), orig)
        for scale in (torch.tensor(1.0), 2.0):
            reduce_op = dist._make_nccl_premul_sum(scale)
            self.assertEqual(pickle.loads(pickle.dumps(reduce_op)), reduce_op)

    # Ref: https://github.com/pytorch/pytorch/issues/90072
    def test_reduceop_equal(self):
        # [FUNCTION DOC] For each built-in op wrapped in ReduceOp, asserts the wrapper equals
        # itself and the raw op, differs from a non-op string, and that the raw op is not equal
        # to the wrapper. Also asserts membership checks against None or the string are false.
        # KNOWN ISSUE: per the in-test TODO the comparison is not symmetric (raw op vs wrapper is
        # not equal while wrapper vs raw is), asserted as current behavior.
        not_reduceop = "abc"
        for reduce_op in (
            c10d.ReduceOp.SUM,
            c10d.ReduceOp.AVG,
            c10d.ReduceOp.PRODUCT,
            c10d.ReduceOp.MIN,
            c10d.ReduceOp.MAX,
            c10d.ReduceOp.BAND,
            c10d.ReduceOp.BOR,
            c10d.ReduceOp.BXOR,
        ):
            reduce_op_obj = c10d.ReduceOp(reduce_op)
            # this calls `ReduceOp.__eq__(self, other)`
            self.assertEqual(reduce_op_obj, reduce_op_obj)
            self.assertEqual(reduce_op_obj, reduce_op)
            self.assertNotEqual(reduce_op_obj, not_reduceop)
            self.assertNotEqual(reduce_op, not_reduceop)
            # TODO(crcrpar): This needs to be `assertEqual` for the associativity even though
            # the comparison of `RedOpType` and `ReduceOp` sounds less likely to happen compared
            # to that of `ReduceOp` and `RedOptype`.
            # this calls `RedOpType.__eq__(self, other)`
            self.assertNotEqual(reduce_op, reduce_op_obj)

            self.assertFalse(None in (reduce_op, reduce_op_obj))
            self.assertFalse(not_reduceop in (reduce_op, reduce_op_obj))


class LocalRankTest(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test (world size 4) for the node local rank lookup, which reads
    # the LOCAL_RANK environment variable.
    @property
    def world_size(self):
        # [FUNCTION DOC] HELPER: Property returning 4.
        return 4

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setUp and spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and removes the test's file_name, ignoring
        # OSError.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def testWithoutEnv(self):
        # [FUNCTION DOC] Asserts that get_node_local_rank with no fallback raises RuntimeError
        # mentioning LOCAL_RANK when that variable is not set.
        with self.assertRaisesRegex(RuntimeError, "LOCAL_RANK"):
            dist.get_node_local_rank()

    def testWithoutEnvWithFallback(self):
        # [FUNCTION DOC] Asserts that with no LOCAL_RANK set the fallback rank argument (2) is
        # returned.
        self.assertEqual(dist.get_node_local_rank(fallback_rank=2), 2)

    def testNodeLocalRankOverridesFallback(self):
        # [FUNCTION DOC] Sets LOCAL_RANK to this rank and asserts the value is returned even
        # though a fallback of 123 is given.
        os.environ["LOCAL_RANK"] = str(self.rank)
        self.assertEqual(dist.get_node_local_rank(fallback_rank=123), self.rank)

    def testNodeLocalRank(self):
        # [FUNCTION DOC] Sets LOCAL_RANK to this rank and asserts get_node_local_rank returns it.
        os.environ["LOCAL_RANK"] = str(self.rank)
        self.assertEqual(dist.get_node_local_rank(), self.rank)


class RecordCommTest(TestCase):
    # [CLASS DOC] Tests the record_comm context manager that sets a profiling name for
    # communications.
    def test_set_get(self):
        # [FUNCTION DOC] Asserts the profiling name is empty initially, equals the given name
        # inside the record_comm block and is empty again after it.
        self.assertEqual(torch._C._distributed_c10d._get_comm_profiling_name(), "")
        with dist.record_comm("test_name"):
            self.assertEqual(
                torch._C._distributed_c10d._get_comm_profiling_name(), "test_name"
            )
        self.assertEqual(torch._C._distributed_c10d._get_comm_profiling_name(), "")

    def test_nesting(self):
        # [FUNCTION DOC] Asserts nested record_comm blocks show the innermost name inside and
        # restore the outer name and then the empty name on exit.
        with dist.record_comm("outer"):
            self.assertEqual(
                torch._C._distributed_c10d._get_comm_profiling_name(), "outer"
            )
            with dist.record_comm("inner"):
                self.assertEqual(
                    torch._C._distributed_c10d._get_comm_profiling_name(), "inner"
                )
            self.assertEqual(
                torch._C._distributed_c10d._get_comm_profiling_name(), "outer"
            )
        self.assertEqual(torch._C._distributed_c10d._get_comm_profiling_name(), "")


# Trivially-destructible types safe for thread_local.
# Non-trivial destructors (std::string, std::vector, std::map, etc.)
# register via __cxa_thread_atexit_impl which holds a mutex — if fork()
# happens while that mutex is held, the child deadlocks.
_SAFE_TLS_TYPE_RE = re.compile(
    r"""
    (?:
        .*\*\s*$              # any pointer type (T*, const T*, etc.)
        | bool
        | u?int(?:8|16|32|64)_t
        | int
        | size_t
        | ssize_t
        | ptrdiff_t
        | float
        | double
        | char
        | (?:unsigned|signed)\s+(?:int|long(?:\s+long)?|short|char)
        | unsigned
        | signed
        | long(?:\s+long)?
        | short
    )
    """,
    re.VERBOSE,
)

# Regex to extract a thread_local declaration.
# Captures the type portion (group 1) and the variable name (group 2).
_TLS_DECL_RE = re.compile(
    r"""
    (?:^|(?<=\s))               # start of line or preceded by whitespace
    (?:static\s+)?              # optional leading static
    thread_local                # the keyword
    \s+(?:static\s+)?           # optional trailing static
    (.+?)                       # type (group 1) — non-greedy
    \s+((?:\w+(?:<[^>]*>)?::)*\w+)  # variable name, possibly qualified (group 2)
    \s*[=;(]                    # followed by = or ; or (
    """,
    re.VERBOSE,
)


class ThreadLocalSafetyLintTest(TestCase):
    # [CLASS DOC] Lint-style test that scans the c10d C++ sources for thread_local variables and
    # fails on any whose type is not trivially destructible, because such destructors register
    # with a libc mutex that can deadlock a forked child. Class attribute _KNOWN_VIOLATIONS lists
    # allowed (relative path, variable name) pairs.
    """Lint: every thread_local in c10d must use a trivially-destructible type.

    Non-trivially-destructible thread_local variables (std::string,
    std::vector, std::map, c10::intrusive_ptr, ...) register destructors
    via __cxa_thread_atexit_impl, which holds an internal glibc mutex.
    If fork() happens while that mutex is held the child process inherits
    a locked mutex and deadlocks on the next TLS access.

    Use raw pointers (T*) with lazy heap-allocation instead.
    """

    # Pre-existing violations outside this diff's scope.  Each entry is
    # (relative_path, variable_name).  Remove entries as they get fixed.
    _KNOWN_VIOLATIONS = {
        ("ProcessGroup.cpp", "pg"),
        ("cuda/CUDAEventCache.cpp", "cacheDeviceMap"),
    }

    @staticmethod
    def _c10d_src_dir() -> Path:
        # [FUNCTION DOC] HELPER: Staticmethod returning the path of torch/csrc/distributed/c10d,
        # computed as two directory levels above this file's directory (the repository root) plus
        # that relative path.
        # test file:  caffe2/test/distributed/test_c10d_common.py
        # source dir: caffe2/torch/csrc/distributed/c10d/
        return (
            Path(__file__).resolve().parents[2]
            / "torch"
            / "csrc"
            / "distributed"
            / "c10d"
        )

    @staticmethod
    def _is_safe_type(type_str: str) -> bool:
        # [FUNCTION DOC] HELPER: Staticmethod. Strips const, volatile, mutable and inline
        # qualifiers from a C++ type string; returns True if the result ends with a star
        # (pointer) or fully matches the whitelist regex of scalar types (bool, sized ints, int,
        # size_t, float, double, char, signed and unsigned variants, short, long), else False.
        """Return True if *type_str* is trivially destructible."""
        cleaned = type_str.strip()
        # Remove const / volatile / mutable qualifiers for matching
        cleaned = re.sub(r"\b(const|volatile|mutable|inline)\b", "", cleaned).strip()
        # Pointer types are always safe
        if cleaned.endswith("*"):
            return True
        # Check against known safe scalar types
        return bool(_SAFE_TLS_TYPE_RE.fullmatch(cleaned))

    def test_no_non_trivial_thread_locals(self):
        # [FUNCTION DOC] Asserts the c10d source directory exists, then reads every cpp, hpp, h
        # and cc file under it line by line. For lines mentioning thread_local (skipping comment-
        # starting lines and static_assert lines) it matches a declaration regex, also trying the
        # line joined with the next one for declarations split over two lines. Declarations with
        # unsafe types, not in the known-violations allowlist (matched by relative path and
        # variable name without class qualifiers), are collected as path:line messages, and the
        # test fails if any are found. Reads files only.
        """Scan c10d sources for thread_local with non-trivial destructors."""
        c10d_dir = self._c10d_src_dir()
        self.assertTrue(
            c10d_dir.is_dir(),
            lambda msg: f"{msg}\nc10d source directory not found: {c10d_dir}",
        )

        violations = []
        for ext in ("*.cpp", "*.hpp", "*.h", "*.cc"):
            for filepath in c10d_dir.rglob(ext):
                lines = filepath.read_text().splitlines()
                idx = 0
                while idx < len(lines):
                    stripped = lines[idx].strip()
                    idx += 1
                    # Skip pure comments and static_assert guards
                    if (
                        stripped.startswith(("//", "/*"))
                        or "static_assert" in stripped
                        or "thread_local" not in stripped
                    ):
                        continue
                    # Try single-line match first
                    m = _TLS_DECL_RE.search(stripped)
                    if not m and idx < len(lines):
                        # Multi-line declaration (type on one line,
                        # variable name on the next)
                        combined = stripped + " " + lines[idx].strip()
                        m = _TLS_DECL_RE.search(combined)
                    if not m:
                        continue
                    tls_type, var_name = m.group(1), m.group(2)
                    if self._is_safe_type(tls_type):
                        continue
                    rel = str(filepath.relative_to(c10d_dir))
                    # Strip class qualifiers from var_name for allowlist
                    # e.g. "Foo<T>::bar" -> "bar"
                    bare_name = var_name.rsplit("::", 1)[-1]
                    if (rel, bare_name) in self._KNOWN_VIOLATIONS:
                        continue
                    violations.append(
                        f"  {rel}:{idx}: thread_local {tls_type} {var_name}"
                    )

        self.assertEqual(
            violations,
            [],
            "Non-trivially-destructible thread_local variable(s) found in c10d.\n"
            "These cause fork-deadlocks via __cxa_thread_atexit.\n"
            "Use a raw pointer (T*) with lazy heap-allocation instead.\n\n"
            "Violations:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    if device_type != "cpu":
        if getattr(torch.get_device_module(device_type), "_initialized", False):
            raise AssertionError(
                f"test_distributed must not have initialized {device_type} context on main process"
            )

    run_tests()
