# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

# [MODULE DOC] Tests specific to the in-tree torchcomms NCCL backends (nccl2, nccl-lazy and, in
# one class, nccl-legacy). Most test classes derive from MultiProcContinuousTest, which spawns the
# worker processes once per class, builds the default group in a class method _init_pg, and runs
# each test on every rank. Covers watchdog safety, options and config, reductions of unusual
# dtypes, ephemeral timeouts, eager and nonblocking communicators, communicator shrink and split,
# abort and timeout handling, memory pool registration, deterministic mode NVLS handling, profiler
# and flight recorder metadata, and process group creation before CUDA is initialized (via a child
# process running an embedded script). Import-time side effects: none beyond imports; the main
# block runs the tests only when CUDA is available.
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ErrorType, ReconfigureOptions
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    run_tests,
    TEST_CUDA,
    TestCase,
)


class ProcessGroupNCCL2Test(MultiProcContinuousTest):
    # [CLASS DOC] Basic nccl2 backend tests on a two-or-more GPU default group built by the
    # harness with the nccl2 backend. Subclassed by the nccl-lazy tests, which override the
    # backend name.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        # [FUNCTION DOC] Issues an asynchronous all_gather whose input is an instance of a plain
        # Tensor subclass, then drops both the tensor and the work handle, synchronizes CUDA,
        # sleeps 2 seconds (to give the watchdog thread time to process the work) and runs a
        # barrier. Passing means the process does not crash when the watchdog releases the work
        # while the tensor is Python-backed.
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.cuda.synchronize()
        time.sleep(2)
        dist.barrier()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_shared_options_type(self) -> None:
        # [FUNCTION DOC] Asserts the nccl2 group class exposes the same Options type as the stock
        # NCCL group class, and that the config fields cga_cluster_size (set to 2) and max_ctas
        # (set to 4) can be assigned and read back on a fresh options object.
        self.assertIs(dist.ProcessGroupNCCL2.Options, dist.ProcessGroupNCCL.Options)
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        self.assertEqual(opts.config.cga_cluster_size, 2)
        self.assertEqual(opts.config.max_ctas, 4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduction_semantics(self) -> None:
        # [FUNCTION DOC] Checks all_reduce dtype and op rules on nccl2. Summing a bool tensor over
        # ranks leaves all ones (viewed as uint8), so bool behaves like a logical or. The AVG
        # reduce op raises TypeError mentioning ReduceOp.AVG. The two fnuz float8 types raise
        # RuntimeError mentioning Unsupported Float8, and the packed float4 type raises
        # RuntimeError mentioning Unsupported Float4.
        tensor = torch.ones(4, dtype=torch.bool, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

        with self.assertRaisesRegex(TypeError, "ReduceOp.AVG"):
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            tensor = torch.ones(4, device=self.device).to(dtype)
            with self.assertRaisesRegex(RuntimeError, "Unsupported Float8"):
                dist.all_reduce(tensor)

        tensor = torch.empty(4, dtype=torch.float4_e2m1fn_x2, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Unsupported Float4"):
            dist.all_reduce(tensor)

    @requires_nccl()
    @requires_nccl_version((2, 24), "Need NCCL 2.24+ for Float8")
    @skip_if_lt_x_gpu(2)
    def test_float8_reduction(self) -> None:
        # [FUNCTION DOC] Skipped unless the device is compute capability 9.0 or newer (the NCCL
        # 2.24 requirement is a decorator). For the float8 e4m3fn and e5m2 types, all_reduce of
        # ones must give a tensor filled with the world size.
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 reductions require sm90 or newer")
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = torch.ones(4, device=self.device).to(dtype)
            dist.all_reduce(tensor)
            self.assertEqual(tensor, torch.full_like(tensor, self.world_size))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_float4_transport(self) -> None:
        # [FUNCTION DOC] Broadcasts from rank 0 a packed float4 tensor (built as a uint8 tensor
        # filled with rank plus one and reinterpreted). Afterwards every rank, viewed as uint8,
        # must hold ones, showing the data is moved as raw bytes from the source.
        tensor = torch.full(
            (4,), self.rank + 1, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)
        dist.broadcast(tensor, src=0)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ephemeral_timeout(self) -> None:
        # [FUNCTION DOC] Sets the default group timeout to 3 seconds, starts an async all_reduce,
        # then adds a 10 second ephemeral timeout for all groups. The earlier work keeps its 3
        # second timeout while a new work reports 13 seconds. After both complete, polls for up to
        # 5 seconds issuing all_reduce calls until a work reports the 3 second timeout again;
        # fails with a message if the ephemeral timeout was never reset.
        dist.set_timeout(timedelta(seconds=3))

        existing_work = dist.all_reduce(
            torch.ones(4, device=self.device), async_op=True
        )
        dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=10))
        self.assertEqual(existing_work.timeout, timedelta(seconds=3))

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        self.assertEqual(work.timeout, timedelta(seconds=13))
        existing_work.wait()
        work.wait()
        torch.cuda.synchronize(self.device)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            work = dist.all_reduce(tensor, async_op=True)
            if work.timeout == timedelta(seconds=3):
                work.wait()
                return
            work.wait()
            time.sleep(0.1)
        self.fail("ephemeral timeout was not reset after collective completion")


class _ProcessGroupNCCL2OptionsTest(MultiProcContinuousTest):
    # [CLASS DOC] HELPER: Base for test classes whose default group is created with backend
    # specific options or environment settings (subclasses override opts or _init_pg). Provides
    # the nccl2 backend name, cuda device and a helper that verifies a world-wide all_reduce.
    """Base for groups initialized with backend specific options."""

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    def _check_all_reduce(self) -> None:
        # [FUNCTION DOC] HELPER: All-reduces a four-element tensor filled with this rank on the
        # default group and asserts the result equals the sum of all ranks.
        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(self.world_size)))
        self.assertEqual(t, torch.full((4,), expected, device=self.device))


class ProcessGroupNCCL2EagerNewGroupTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests new_group behavior when the default group was initialized eagerly
    # (device_id bound), so subgroup communicators are created eagerly.
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method overriding the harness group setup: raises an error
        # if the rendezvous file is missing, sets the LOCAL_RANK environment variable, builds a
        # FileStore, and initializes the default process group with the class backend, options and
        # timeout and with the device id bound to this rank's CUDA device. Saves the default group
        # on the class.
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        store = dist.FileStore(rdvz_file, world_size)
        dist.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
            device_id=torch.device("cuda", rank),
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_new_group_with_nonmembers(self) -> None:
        # [FUNCTION DOC] Creates a group containing only rank 0. Rank 0 all-reduces within it
        # (result must stay equal to its own value, 1) and destroys it; other ranks must receive
        # the non-member sentinel instead of a group. All ranks then use the default store to set
        # and wait on a per-rank key as a rendezvous and verify the default group still works.
        ranks = [0]
        group = dist.new_group(ranks=ranks)
        if self.rank in ranks:
            tensor = torch.tensor([self.rank + 1], device=self.device)
            dist.all_reduce(tensor, group=group)
            self.assertEqual(tensor, torch.ones_like(tensor))
            dist.destroy_process_group(group)
        else:
            self.assertEqual(group, dist.GroupMember.NON_GROUP_MEMBER)

        store = dist.distributed_c10d._get_default_store()
        store.set(f"eager_new_group/{self.rank}", b"1")
        store.wait([f"eager_new_group/{rank}" for rank in range(self.world_size)])
        self._check_all_reduce()


class ProcessGroupNCCL2ShrinkTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests communicator shrinking on a group created from the default options.
    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27+ for communicator shrink")
    @skip_if_lt_x_gpu(2)
    def test_shrink_group(self) -> None:
        # [FUNCTION DOC] Needs NCCL 2.27 or newer. Creates a new group bound to the device and
        # runs a barrier. All ranks except 0 destroy the group and return. Rank 0 shrinks the
        # group excluding the others, expects the shrunk group size to be 1, all-reduces ones
        # (result unchanged) and destroys the shrunk group.
        tensor = torch.ones(4, device=self.device)
        group = dist.new_group(device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))

        if self.rank in excluded:
            dist.destroy_process_group(group)
            return

        shrunk = dist.shrink_group(excluded, group=group)
        self.assertEqual(shrunk.size(), 1)
        dist.all_reduce(tensor, group=shrunk)
        self.assertEqual(tensor, torch.ones_like(tensor))
        dist.destroy_process_group(shrunk)


class ProcessGroupNCCL2ConfigTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests that NCCL config options set on the group options reach the backend.
    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] HELPER: Class method returning stock-NCCL style options with the high
        # priority stream flag always enabled, cga_cluster_size set to 2 and max_ctas set to 4.
        # KNOWN ISSUE: the high_priority_stream argument is accepted but ignored; the flag is
        # always set to enabled.
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_config(self) -> None:
        # [FUNCTION DOC] Reads the device's backend implementation and asserts the options carry
        # cga_cluster_size 2, max_ctas 4 and the high priority stream flag, then verifies an
        # all_reduce still produces the correct sum.
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.cga_cluster_size, 2)
        self.assertEqual(backend.options.config.max_ctas, 4)
        self.assertTrue(backend.options.is_high_priority_stream)
        self._check_all_reduce()


class ProcessGroupNCCL2NonblockingTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests nccl2 with a nonblocking communicator (config.blocking set to 0).
    # Subclassed to also run against the nccl-legacy and nccl-lazy backends.
    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] HELPER: Class method returning nccl2 options whose config.blocking is 0
        # (nonblocking).
        # KNOWN ISSUE: the high_priority_stream argument is accepted but unused.
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.blocking = 0
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collectives_with_nonblocking_communicator(self) -> None:
        # [FUNCTION DOC] Runs all_reduce, then a gather to rank 0 (rank 0 checks each gathered
        # tensor is filled with its source rank), then a scatter from rank 0 of tensors filled
        # with each destination rank (each rank must receive a tensor equal to its own input).
        self._check_all_reduce()

        tensor = torch.full((4,), float(self.rank), device=self.device)
        gathered = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.gather(tensor, gathered, dst=0)
        if self.rank == 0:
            for rank, output in enumerate(gathered):
                self.assertEqual(output, torch.full_like(output, rank))

        output = torch.empty_like(tensor)
        scatter_list = (
            [torch.full_like(tensor, float(rank)) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.scatter(output, scatter_list, src=0)
        self.assertEqual(output, tensor)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_with_nonblocking_communicator(self) -> None:
        # [FUNCTION DOC] Calls the backend split with the default store, a member list
        # containing only rank 0 and nonblocking options. Rank 0 then runs an all_reduce on
        # the child backend and asserts the single-rank result is unchanged; other ranks
        # only perform the split call.
        backend = dist.get_backend_impl(device=self.device)
        opts = self.opts()
        child = backend.split(dist.distributed_c10d._get_default_store(), [0], opts)
        if self.rank == 0:
            tensor = torch.ones(1, device=self.device)
            child.allreduce([tensor]).wait()
            self.assertEqual(tensor, torch.ones_like(tensor))

    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27+ for communicator shrink")
    @skip_if_lt_x_gpu(2)
    def test_shrink_with_nonblocking_communicator(self) -> None:
        # [FUNCTION DOC] Needs NCCL 2.27 or newer. Creates a group with nonblocking options and
        # runs a barrier. All ranks except 0 destroy the group and return; rank 0 shrinks the
        # group excluding them, all-reduces ones on the shrunk group (unchanged) and destroys it.
        group = dist.new_group(pg_options=self.opts(), device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))
        if self.rank in excluded:
            dist.destroy_process_group(group)
            return

        shrunk = dist.shrink_group(excluded, group=group)
        tensor = torch.ones(1, device=self.device)
        dist.all_reduce(tensor, group=shrunk)
        self.assertEqual(tensor, torch.ones_like(tensor))
        dist.destroy_process_group(shrunk)


class ProcessGroupNCCL2EnvironmentConfigTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests that TORCH_NCCL environment variables (high priority, nonblocking
    # communicator, timing, event cache) configure a group created with default options.
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method setting the environment variables for high priority
        # streams, nonblocking communicator, timing enabled and CUDA event cache disabled, then
        # calls the base group setup. Side effect: modifies this worker process environment.
        os.environ["TORCH_NCCL_HIGH_PRIORITY"] = "1"
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"
        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "0"
        super()._init_pg(rank, world_size, rdvz_file)

    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] HELPER: Class method returning default nccl2 options.
        # KNOWN ISSUE: the high_priority_stream argument is accepted but unused.
        return dist.ProcessGroupNCCL2.Options()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_environment_config(self) -> None:
        # [FUNCTION DOC] Asserts the backend options report config.blocking equal to 0 (from the
        # environment), then times an all_reduce of 1024 floats and asserts its duration is
        # positive after synchronization, proving timing was enabled.
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.blocking, 0)

        tensor = torch.ones(1024, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        torch.cuda.synchronize(self.device)
        self.assertGreater(work._get_duration(), 0)


class ProcessGroupNCCL2NonblockingOptionPrecedenceTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests that an explicit option takes precedence over the environment variable
    # that requests a nonblocking communicator.
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method setting the environment variable that requests a
        # nonblocking communicator and then calling the base group setup. Side effect: modifies
        # this worker process environment.
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        super()._init_pg(rank, world_size, rdvz_file)

    @classmethod
    def opts(cls, high_priority_stream=False):
        # [FUNCTION DOC] HELPER: Class method returning nccl2 options with config.blocking
        # explicitly set to 1.
        # KNOWN ISSUE: the high_priority_stream argument is accepted but unused.
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.blocking = 1
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_option_takes_precedence(self) -> None:
        # [FUNCTION DOC] Asserts the backend options still report blocking equal to 1 despite the
        # environment, and that an all_reduce gives the correct sum.
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.blocking, 1)
        self._check_all_reduce()


class ProcessGroupNCCLLegacyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    # [CLASS DOC] Reruns every nonblocking test of the parent class against the legacy stock NCCL
    # backend by overriding only the backend name.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl-legacy.
        return "nccl-legacy"


class ProcessGroupNCCL2ScalableInitTest(_ProcessGroupNCCL2OptionsTest):
    # [CLASS DOC] Tests scalable communicator initialization with the number of ranks per root
    # taken from the class attribute ranks_per_root (default 1), passed through an environment
    # variable before group creation.
    ranks_per_root = 1

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method exporting the ranks-per-root environment variable
        # from the class attribute and then calling the base group setup. Side effect: modifies
        # this worker process environment.
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = str(cls.ranks_per_root)
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_scalable_init(self) -> None:
        # [FUNCTION DOC] Verifies a world-wide all_reduce gives the correct sum with the scalable
        # init setting.
        self._check_all_reduce()


class ProcessGroupNCCL2UnevenScalableInitTest(ProcessGroupNCCL2ScalableInitTest):
    # [CLASS DOC] Scalable init test with three ranks and two ranks per root, so the last root
    # serves an unevenly sized set.
    world_size = 3
    ranks_per_root = 2

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_collective_with_scalable_init(self) -> None:
        # [FUNCTION DOC] Overrides the parent test only to require three GPUs; verifies a world-
        # wide all_reduce gives the correct sum.
        self._check_all_reduce()


class _ProcessGroupNCCL2SubgroupTest(MultiProcContinuousTest):
    # [CLASS DOC] HELPER: Base for tests that tear a group down. They use a throwaway subgroup so
    # the default group survives, and a final all_reduce on the default group proves the process
    # is still healthy.
    """Base for tests that tear a group down.

    They run on a throwaway subgroup so the class-wide default group survives;
    a collective on that default group afterwards is what proves the process is
    still alive and healthy.
    """

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    def _new_subgroup(self, timeout=timedelta(seconds=60)):
        # [FUNCTION DOC] HELPER: Creates and returns a new nccl2 group over all ranks using
        # local synchronization (only members need to call) and the given timeout, which
        # defaults to 60 seconds.
        return dist.new_group(
            backend="nccl2",
            use_local_synchronization=True,
            timeout=timeout,
        )

    def _check_all_reduce(self, group=None) -> None:
        # [FUNCTION DOC] HELPER: All-reduces a four-element tensor of ones on the given group (the
        # default group if none) and asserts every element equals the world size.
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t, group=group)
        self.assertEqual(t, torch.full_like(t, self.world_size))


class ProcessGroupNCCL2AbortTest(_ProcessGroupNCCL2SubgroupTest):
    # [CLASS DOC] Tests abort and destroy of a healthy subgroup.
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_abort_on_healthy_group(self) -> None:
        # [FUNCTION DOC] Runs a collective on a new subgroup, calls the backend abort (which must
        # return rather than terminate the process), asserts the backend now reports a
        # communication error, destroys the group, and checks the default group still works.
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        backend = pg._get_backend(self.device)
        # Must return instead of ::abort()ing the process, like stock NCCL.
        backend.abort()
        self.assertEqual(backend.get_error(), ErrorType.COMM_ERROR)

        dist.destroy_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_abort_process_group_on_healthy_group(self) -> None:
        # [FUNCTION DOC] Runs a collective on a new subgroup, aborts the group through the
        # distributed module helper, and checks the default group still works.
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        dist.distributed_c10d._abort_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_destroy_process_group_returns(self) -> None:
        # [FUNCTION DOC] Runs a collective on a new subgroup, destroys it, and checks the default
        # group still works (destroy must return and not hang).
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2WatchdogNoTearDownTest(_ProcessGroupNCCL2SubgroupTest):
    # [CLASS DOC] Tests that a collective timeout on a subgroup does not kill the process when the
    # async error handling mode does not tear down.
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_without_process_abort(self) -> None:
        # [FUNCTION DOC] Creates a subgroup (5 second timeout) while the async error handling
        # environment variable is set to 0 (no handling); the variable is restored afterwards.
        # Rank 0 alone issues an all_reduce that can never complete, polls up to 60 seconds until
        # the backend error is no longer success, asserts it is the timeout error, and asserts
        # that a further all_reduce on the group raises RuntimeError. Other ranks sleep 30
        # seconds. All ranks then destroy the group and check the default group works.
        # NoHandling is how a stock NCCL user opts out of the process being
        # taken down by the watchdog; nccl2 reads the same variable, when the
        # backend is constructed.
        env = {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}
        with mock.patch.dict(os.environ, env):
            pg = self._new_subgroup(timeout=timedelta(seconds=5))
        backend = pg._get_backend(self.device)
        self._check_all_reduce(pg)

        if self.rank == 0:
            # Nobody else joins, so this can never complete and the watchdog
            # trips. Without the tear-down the process must survive and the
            # timeout must become readable through get_error().
            dist.all_reduce(torch.ones(1024, device=self.device), group=pg)
            deadline = time.time() + 60
            while time.time() < deadline and backend.get_error() == ErrorType.SUCCESS:
                time.sleep(0.5)
            self.assertEqual(backend.get_error(), ErrorType.TIMEOUT)
            # The next collective on the timed-out group raises rather than
            # silently proceeding on a dead communicator.
            with self.assertRaises(RuntimeError):
                dist.all_reduce(torch.ones(4, device=self.device), group=pg)
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_with_communicator_cleanup(self) -> None:
        # [FUNCTION DOC] Same scenario as the previous test but with the async error handling
        # variable set to 2 (stock clean-up-only mode): the timeout must still become readable as
        # the timeout error, the next collective must raise RuntimeError, and the default group
        # must work after destroying the subgroup.
        env = {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "2"}
        with mock.patch.dict(os.environ, env):
            pg = self._new_subgroup(timeout=timedelta(seconds=5))
        backend = pg._get_backend(self.device)
        self._check_all_reduce(pg)

        if self.rank == 0:
            dist.all_reduce(torch.ones(1024, device=self.device), group=pg)
            deadline = time.time() + 60
            while time.time() < deadline and backend.get_error() == ErrorType.SUCCESS:
                time.sleep(0.5)
            self.assertEqual(backend.get_error(), ErrorType.TIMEOUT)
            with self.assertRaises(RuntimeError):
                dist.all_reduce(torch.ones(4, device=self.device), group=pg)
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2BlockingWaitTest(_ProcessGroupNCCL2SubgroupTest):
    # [CLASS DOC] Tests wait() timeouts when blocking wait is enabled through the environment.
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method enabling the blocking wait environment variable and
        # then calling the base group setup. Side effect: modifies this worker process
        # environment.
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_times_out_and_process_survives(self) -> None:
        # [FUNCTION DOC] Creates a subgroup with a 5 second timeout. Rank 0 starts an async
        # all_reduce that others never join and asserts wait raises RuntimeError mentioning timed
        # out; others sleep 30 seconds. Then all ranks destroy the subgroup and check the default
        # group still works.
        pg = self._new_subgroup(timeout=timedelta(seconds=5))
        self._check_all_reduce(pg)

        if self.rank == 0:
            work = dist.all_reduce(
                torch.ones(1024, device=self.device), group=pg, async_op=True
            )
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                work.wait()
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2ExpandableSegmentsTest(MultiProcContinuousTest):
    # [CLASS DOC] Tests nccl2 collectives when the CUDA caching allocator uses expandable
    # segments, set before the group is created.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method enabling the expandable segments allocator setting
        # in the worker process and then calling the base group setup.
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_large_in_place_all_gather(self) -> None:
        # [FUNCTION DOC] Allocates one bfloat16 output of world size times 16M elements, uses this
        # rank's slice as the input filled with the rank, and asserts the allocator reports an
        # expandable segment. Runs all_gather_single in place and asserts each chunk is filled
        # with its source rank. Skipped on ROCm 7.14 or newer.
        numel = 16 * 1024 * 1024
        output = torch.empty(
            self.world_size * numel, dtype=torch.bfloat16, device=self.device
        )
        input = output.narrow(0, self.rank * numel, numel)
        input.fill_(self.rank)
        self.assertTrue(
            any(segment["is_expandable"] for segment in torch.cuda.memory_snapshot())
        )

        dist.all_gather_single(output, input)

        for rank, chunk in enumerate(output.chunk(self.world_size)):
            self.assertEqual(chunk, torch.full_like(chunk, rank))


class ProcessGroupNCCL2MemPoolTest(MultiProcContinuousTest):
    # [CLASS DOC] Tests registering and deregistering a CUDA memory pool with the nccl2 backend,
    # the user-buffer path used by Megatron's allocator.
    """register_mem_pool / deregister_mem_pool: the NCCL user-buffer path that
    Megatron's nccl_allocator drives (megatron/core/nccl_allocator.py)."""

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method that sets the current CUDA device to this rank
        # before the base group setup. It is done here instead of setUp because only this hook
        # runs inside the worker processes, and memory pools bind to the current device.
        # The device is set here rather than in setUp because _init_pg is the
        # only hook that runs in the worker processes; setUp runs in the
        # dispatcher, on rank -1. MemPool and use_mem_pool bind to the current
        # device, so without this every rank would pool on device 0 while its
        # tensors live on cuda:rank -- only rank 0 would register anything and
        # the collective below would hang.
        torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    def _backend(self):
        # [FUNCTION DOC] HELPER: Runs a tiny all_reduce and synchronizes so the lazily created
        # communicator exists (pool registration needs it), then returns the device's backend
        # implementation.
        # nccl2 bootstraps its communicator on the first collective, and
        # register_mem_pool needs it up (as it does on the stock backend, whose
        # test eagerly inits via init_process_group(device_id=...) instead).
        dist.all_reduce(torch.zeros(1, device=self.device))
        torch.cuda.synchronize()
        return dist.get_backend_impl(device=self.device)

    @property
    def _all_reduced(self) -> float:
        # [FUNCTION DOC] HELPER: Property; the expected all_reduce result of tensors filled with
        # each rank id, i.e. the sum of all ranks as a float.
        return float(sum(range(self.world_size)))

    def _pool_tensor(self, pool, numel: int = 1024 * 1024) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Allocates, inside the given memory pool, a float tensor of numel
        # elements (default 1M) filled with this rank and returns it.
        with torch.cuda.use_mem_pool(pool):
            return torch.full((numel,), float(self.rank), device=self.device)

    def _check_all_reduce_over_pool(self, symm: bool) -> None:
        # [FUNCTION DOC] HELPER: Creates a memory pool from the backend's allocator, allocates a
        # tensor in it, registers the pool (symmetric when the flag is set), all-reduces, asserts
        # the sum, and always deregisters the pool in a final block.
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        tensor = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=symm)
        try:
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            self.assertEqual(tensor, torch.full_like(tensor, self._all_reduced))
        finally:
            backend.deregister_mem_pool(pool)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool(self) -> None:
        # [FUNCTION DOC] Runs the pool all_reduce check with ordinary (non-symmetric)
        # registration.
        self._check_all_reduce_over_pool(symm=False)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool_symmetric(self) -> None:
        # [FUNCTION DOC] Runs the pool all_reduce check with symmetric registration.
        self._check_all_reduce_over_pool(symm=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool_round_trip(self) -> None:
        # [FUNCTION DOC] Registers a pool symmetrically, deregisters it, allocates a second tensor
        # in the same pool, registers again, then all_reduces both tensors and asserts both sums
        # are correct. Verifies deregister and re-register work when the allocator hook already
        # registered some segments and the second registration must add segments the hook did not
        # see. Always deregisters at the end.
        # Megatron re-enters its allocator context once per bucket: deregister,
        # grow the pool, register again. Neither leg may trip over a segment
        # the caching-allocator hook already registered, and the second
        # register has to pick up segments the hook never saw (reused ones).
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        first = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=True)
        backend.deregister_mem_pool(pool)
        second = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=True)
        try:
            for tensor in (first, second):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            for tensor in (first, second):
                self.assertEqual(tensor, torch.full_like(tensor, self._all_reduced))
        finally:
            backend.deregister_mem_pool(pool)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deregister_unregistered_mem_pool_raises(self) -> None:
        # [FUNCTION DOC] Asserts that deregistering a pool that was never registered raises
        # RuntimeError mentioning not previously registered.
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        with self.assertRaisesRegex(RuntimeError, "not previously registered"):
            backend.deregister_mem_pool(pool)


class ProcessGroupNCCLLazyTest(ProcessGroupNCCL2Test):
    # [CLASS DOC] Runs all basic tests of the parent class against the nccl-lazy backend (primary
    # NCCL group plus lazily created point-to-point communicators) and adds lazy-specific tests.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl-lazy.
        return "nccl-lazy"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reconfigure_not_supported(self) -> None:
        # [FUNCTION DOC] Asserts the backend reports no reconfigure support, that getting the
        # reconfigure handle and calling reconfigure each raise RuntimeError with a does-not-
        # support message, and that constructing the lazy group directly with options requesting
        # reconfigure enabled raises RuntimeError saying nccl-lazy does not support
        # enable_reconfigure.
        backend = dist.get_backend_impl(device=self.device)
        self.assertFalse(backend.supports_reconfigure)
        with self.assertRaisesRegex(
            RuntimeError, "does not support get_reconfigure_handle"
        ):
            backend.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, "does not support reconfigure"):
            backend.reconfigure(ReconfigureOptions())

        opts = dist.ProcessGroupNCCL.Options()
        opts.enable_reconfigure = True
        with self.assertRaisesRegex(
            RuntimeError, "nccl-lazy does not support enable_reconfigure"
        ):
            dist.ProcessGroupNCCLLazy(
                dist.HashStore(), self.rank, self.world_size, opts
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_lazy_pair_channels(self) -> None:
        # [FUNCTION DOC] Records the number of active point-to-point channels, runs an all_reduce
        # and asserts the count is unchanged (collectives use the primary group). Then exchanges
        # data with the neighbor ranks using blocking send and receive (even ranks send first) and
        # asserts the received values; finally asserts at least 1 active channel when next and
        # previous peers are the same rank, otherwise at least 2.
        backend = dist.get_backend_impl(device=self.device)
        before_collective = backend._num_active_channels()
        t = torch.full((4,), 1.0, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(backend._num_active_channels(), before_collective)

        send_t = torch.full((4,), float(self.rank), device=self.device)
        recv_t = torch.empty((4,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        torch.cuda.synchronize()
        self.assertEqual(recv_t, torch.full((4,), float(prev), device=self.device))

        expected = 1 if nxt == prev else 2
        self.assertGreaterEqual(backend._num_active_channels(), expected)


class ProcessGroupNCCLLazyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    # [CLASS DOC] Reruns the nonblocking tests of the parent class against the nccl-lazy backend.
    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl-lazy.
        return "nccl-lazy"

    def test_shrink_with_nonblocking_communicator(self) -> None:
        # [FUNCTION DOC] Overrides the inherited shrink test to always skip, since nccl-lazy does
        # not support communicator shrink.
        self.skipTest("nccl-lazy does not support communicator shrink")


def _live_env(name: str) -> str | None:
    # [FUNCTION DOC] HELPER: Returns the current value of the named environment variable read
    # through the C library (or none if unset), because the Python environment mapping is a
    # startup snapshot that does not see values set later by native code.
    # os.environ is a snapshot taken at interpreter startup and does NOT observe
    # values set later by C++ via setenv (which is how the nccl2 backend forces
    # NCCL_ALGO under deterministic mode), so read the live environ via libc.
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    val = libc.getenv(name.encode())
    return val.decode() if val is not None else None


class _DeterministicNVLSTest(MultiProcContinuousTest):
    # [CLASS DOC] HELPER: Base for tests of the nccl2 behavior under deterministic mode:
    # when deterministic algorithms are on and the user has not set NCCL_ALGO, the backend
    # sets it to exclude NVLS during group initialization. Class attributes deterministic
    # and preset_nccl_algo select the scenario, each in its own subclass so each gets
    # freshly spawned processes.
    """Base for the nccl2 deterministic-mode NVLS-disable hook.

    Under torch deterministic mode, if the user has not set NCCL_ALGO the nccl2
    backend forces NCCL_ALGO=^NVLS (NVLS can give non-deterministic reductions),
    mirroring the stock ProcessGroupNCCL. The hook runs during PG init and the
    env var persists once set, so each scenario is its own subclass: every class
    gets a freshly spawned process pool (clean deterministic flag / NCCL_ALGO)
    and configures the env in _init_pg before the PG (and thus the hook) runs.
    """

    deterministic: bool = False
    preset_nccl_algo: str | None = None

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # [FUNCTION DOC] HELPER: Class method run before the group is created: sets the NCCL_ALGO
        # environment variable to the preset value if the class defines one, otherwise removes it;
        # enables deterministic algorithms when the class asks for it; then calls the base group
        # setup. Side effect: modifies environment and global determinism state.
        if cls.preset_nccl_algo is not None:
            os.environ["NCCL_ALGO"] = cls.preset_nccl_algo
        else:
            os.environ.pop("NCCL_ALGO", None)
        if cls.deterministic:
            torch.use_deterministic_algorithms(True)
        super()._init_pg(rank, world_size, rdvz_file)

    def _all_reduce_and_read_algo(self) -> str | None:
        # [FUNCTION DOC] HELPER: Runs an all_reduce of ones (forcing the communicator, and thus
        # the hook, to initialize), asserts the sum equals the world size, and returns the live
        # NCCL_ALGO value or none if unset.
        # Force the (possibly lazy) nccl2 comm to initialize so the hook runs,
        # and confirm the reduction is numerically correct under the chosen algo.
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(t, torch.full_like(t, self.world_size))
        return _live_env("NCCL_ALGO")


class ProcessGroupNCCL2DeterministicNVLSDisabledTest(_DeterministicNVLSTest):
    # [CLASS DOC] Deterministic mode with no preset NCCL_ALGO.
    deterministic = True

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_disables_nvls(self) -> None:
        # [FUNCTION DOC] Asserts the live NCCL_ALGO value after a collective is the string that
        # excludes NVLS.
        self.assertEqual(self._all_reduce_and_read_algo(), "^NVLS")


class ProcessGroupNCCL2DeterministicNVLSPresetTest(_DeterministicNVLSTest):
    # [CLASS DOC] Deterministic mode with NCCL_ALGO preset to Ring by the user.
    deterministic = True
    preset_nccl_algo = "Ring"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_respects_user_nccl_algo(self) -> None:
        # [FUNCTION DOC] Asserts the live NCCL_ALGO value stays Ring, i.e. the backend does not
        # override a user setting.
        self.assertEqual(self._all_reduce_and_read_algo(), "Ring")


class ProcessGroupNCCL2NonDeterministicNVLSTest(_DeterministicNVLSTest):
    # [CLASS DOC] Non-deterministic mode with no preset NCCL_ALGO.
    deterministic = False

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_non_deterministic_does_not_force_nccl_algo(self) -> None:
        # [FUNCTION DOC] Asserts NCCL_ALGO remains unset after a collective.
        self.assertIsNone(self._all_reduce_and_read_algo())


class ProcessGroupNCCL2ObservabilityTest(MultiProcContinuousTest):
    # [CLASS DOC] Tests what nccl2 reports to profilers and flight recorders.
    """What nccl2 reports about a collective to profilers and flight recorders."""

    @classmethod
    def backend_str(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the process group backend used by this class;
        # returns nccl2.
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method naming the device type for the harness;
        # returns cuda.
        return "cuda"

    @property
    def device(self) -> torch.device:
        # [FUNCTION DOC] HELPER: Property; the CUDA device indexed by this process rank.
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base setUp and then makes this rank's CUDA device the
        # current device.
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_profiler_sequence_number_is_per_group(self) -> None:
        # [FUNCTION DOC] Creates two nccl2 subgroups, bootstraps both with one all_reduce, then
        # under the profiler runs 3 collectives on the first and 1 on the second. Exports a Chrome
        # trace to a temporary file (deleted afterwards), collects the sequence numbers recorded
        # per process group name, and asserts each group's recorded numbers are consecutive
        # starting right after its own counter value before profiling, i.e. numbering is per group
        # rather than process-global. Destroys both groups at the end.
        # TracingGuard used to record a process-global counter of its own
        # instead of the issuing group's sequence_number_, so a rank's
        # numbering depended on how it interleaved its other groups'
        # collectives -- and profilers pair collectives across ranks by
        # (process group, sequence number).
        groups = [
            dist.new_group(
                backend=self.backend_str(),
                use_local_synchronization=True,
                timeout=timedelta(seconds=60),
            )
            for _ in range(2)
        ]
        t = torch.ones(4, device=self.device)
        try:
            # Bootstrap both communicators outside the profiled region.
            for group in groups:
                dist.all_reduce(t, group=group)
            torch.cuda.synchronize()
            # Three collectives on the first group, one on the second: a
            # process-global counter would number the second group's collective
            # after all of the first group's.
            counts = [3, 1]
            base = [group._get_sequence_number_for_group() for group in groups]

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                for group, count in zip(groups, counts):
                    for _ in range(count):
                        dist.all_reduce(t, group=group)
                torch.cuda.synchronize()

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                trace_path = f.name
            try:
                prof.export_chrome_trace(trace_path)
                with open(trace_path) as f:
                    trace = json.load(f)
            finally:
                os.unlink(trace_path)

            recorded: dict[str, set[int]] = {}
            for ev in trace.get("traceEvents", []):
                args = ev.get("args", {})
                if "Process Group Name" in args and "Seq" in args:
                    recorded.setdefault(args["Process Group Name"], set()).add(
                        args["Seq"]
                    )
            for group, count, first in zip(groups, counts, base):
                self.assertEqual(
                    sorted(recorded.get(group.group_name, set())),
                    [first + i + 1 for i in range(count)],
                )
        finally:
            for group in groups:
                dist.destroy_process_group(group)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_group_child_global_ranks(self) -> None:
        # [FUNCTION DOC] Brings the default communicator up, then splits the default group into
        # the lower and upper halves of the ranks. Asserts the child options list the parent-
        # global ranks of the half, that the config comm_name (if the attribute exists) is none,
        # that splitting the child again into a singleton maps through the parent's rank map
        # (grandchild global ranks equal this rank), and that an all_reduce on the child equals
        # the sum of the half's ranks.
        # nccl2's split() copied everything into the child's options except the
        # parent-global rank map, which FlightRecorderHook then falls back to
        # identity ranks for -- so a child's recorded membership claimed world
        # ranks instead of its own.
        pg = dist.distributed_c10d._get_default_group()
        # Bring the parent communicator up before splitting it.
        dist.all_reduce(torch.ones(4, device=self.device))
        half = self.world_size // 2
        halves = (
            list(range(half))
            if self.rank < half
            else list(range(half, self.world_size))
        )
        child = pg.split_group(halves, group_name=f"halves{halves}")
        child_options = child._get_backend(self.device).options
        self.assertEqual(child_options.global_ranks_in_group, halves)
        if hasattr(child_options.config, "comm_name"):
            # split() points config.commName at the group_name string for the
            # ncclCommSplit call; storing that pointer left the child holding
            # one that outlives its string.
            self.assertIsNone(child_options.config.comm_name)

        # Splitting the child again must map its ranks through the parent's map
        # rather than treating them as world ranks.
        mine = [halves.index(self.rank)]
        grandchild = child.split_group(mine, group_name=f"single{self.rank}")
        self.assertEqual(
            grandchild._get_backend(self.device).options.global_ranks_in_group,
            [self.rank],
        )

        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t, group=child)
        expected = float(sum(halves))
        self.assertEqual(t, torch.full_like(t, expected))


_UNINITIALIZED_CUDA_SCRIPT = """\
import torch
import torch.distributed as dist

# Nothing above this point may touch torch.cuda: the caching allocator is
# brought up lazily by the Python torch.cuda layer, and neither CUDAGuard nor
# stream/event creation does it.
dist.init_process_group(
    "cpu:gloo,cuda:nccl2",
    rank=0,
    world_size=1,
    store=dist.HashStore(),
    device_id=torch.device("cuda:0"),
)
assert not torch.cuda.is_initialized(), "creating a process group initialized CUDA"
{extra}
dist.destroy_process_group()
"""


class ProcessGroupNCCL2UninitializedCudaTest(TestCase):
    # [CLASS DOC] Checks that process group creation (including eager init with a bound device)
    # works in a fresh process that has not touched torch.cuda. Runs the embedded script in a
    # child Python process (the harness itself would hide the problem).
    """Constructing a process group must not need the CUDA caching allocator up.

    Eager init (`init_process_group(device_id=...)`) allocated the barrier
    buffer from the caching allocator, so a process that had made no torch.cuda
    call died inside init_process_group with "Allocator not initialized for
    device 0". Any external launcher or library that builds the PG before
    touching CUDA hits this. Runs in a subprocess because the harness (and
    every other test here) calls torch.cuda.set_device in setUp, which hides it.
    """

    def _run_child(self, extra: str = "") -> None:
        # [FUNCTION DOC] HELPER: Runs the embedded script in a subprocess with the optional extra
        # statement substituted into it, in the directory of this test file with a 300 second
        # timeout. Fails the test on timeout, or on non-zero exit with the captured output.
        try:
            subprocess.check_output(
                [sys.executable, "-c", _UNINITIALIZED_CUDA_SCRIPT.format(extra=extra)],
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.realpath(__file__)),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.fail("child process timed out")
        except subprocess.CalledProcessError as e:
            self.fail(f"child process failed with:\n{e.output.decode()}")

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_eager_init(self) -> None:
        # [FUNCTION DOC] Skipped under fbcode or sandcastle. Runs the child script with no extra
        # statement: it initializes a one-rank group with cpu gloo and cuda nccl2 and a bound cuda
        # device, asserts CUDA is still uninitialized, and destroys the group.
        self._run_child()

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_barrier_after_init(self) -> None:
        # [FUNCTION DOC] Skipped under fbcode or sandcastle. Runs the child script with a
        # barrier call added after init, covering the second allocator entry point that is
        # deferred past init.
        # The allocation is merely deferred, so barrier() is the second entrance
        # into the same allocator: without lazyInitDevice() it hits the identical
        # assert one call later.
        self._run_child("dist.barrier()")


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
