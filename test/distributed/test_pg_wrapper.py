# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Multi-process tests for the c10d process group wrapper, which checks that all
# ranks issue matching collectives (op type, tensor shape, dtype, device, sequence number) and
# reports a diff when they do not. Defines an abstract test mixin plus a Gloo-backed suite and
# an accelerator (NCCL/XCCL) suite. At import it exits the process silently if c10d is
# unavailable, and computes the module globals device_type (current accelerator or cpu) and
# backend (default backend for it). The main block refuses to run if CUDA/XPU was already
# initialized in the parent, then calls run_tests.

import os
import sys
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as c10d
from torch._C._distributed_c10d import _ProcessGroupWrapper


if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from test_c10d_common import LOOPBACK

from torch.testing._internal.common_distributed import (
    create_device,
    MultiProcessTestCase,
    requires_accelerator_dist_backend,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
backend = c10d.get_default_backend_for_device(device_type)


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    # [CLASS DOC] Abstract MultiProcessTestCase holding backend-independent checks for the
    # process group wrapper. Subclasses must provide _create_wrapper_pg(with_new_group,
    # timeout) returning a wrapped group. Shared helpers cover hang detection, op mismatch,
    # shape mismatch and coalesced-allgather op mismatch. Not collected on its own for real
    # work; setUp spawns the worker processes.
    def setUp(self):
        # [FUNCTION DOC] Run the base multi-process setUp, then spawn the worker processes.
        super().setUp()
        self._spawn_processes()

    def _validate_error(self, exception, op_type, rank, tensor, verify_diff=True):
        # [FUNCTION DOC] HELPER: Assert that an exception raised by the wrapper describes a
        # collective mismatch. Checks the message contains op_type; unless op_type is BARRIER,
        # also checks it contains the tensor shape, the device (only the device type when the
        # tensor lives on the accelerator), the C++ dtype name (Float for float dtypes, Long
        # for int dtypes, otherwise fails the test), a SequenceNumber entry, and, when
        # verify_diff is true, the text 'Collectives differ in the following'. The rank
        # argument is accepted but unused.
        # KNOWN ISSUE: the rank parameter is never used in the checks.
        err = str(exception)
        self.assertTrue(
            op_type in err,
            lambda msg: f"{msg}\nGot {err} but expected {op_type} to be in error.",
        )
        # User doesn't call barrier with tensor.
        if op_type != "BARRIER":
            self.assertTrue(
                f"{list(tensor.shape)}" in err,
                lambda msg: f"{msg}\nDid not find shapes {list(tensor.shape)} in error {err}",
            )
            # For CUDA, only assert on device type, not index
            if device_type in str(tensor.device):
                self.assertTrue(
                    device_type in err,
                    lambda msg: f"{msg}\nDid not find {device_type} device in error {err}",
                )
            else:
                self.assertTrue(
                    str(tensor.device) in err,
                    lambda msg: f"{msg}\nDid not find tensor device {str(tensor.device)} in error {err}",
                )
            # C++ and python type strings are not exactly the same.
            if "float" in str(tensor.dtype):
                self.assertTrue("Float" in err, "Expected Float type")
            elif "int" in str(tensor.dtype):
                self.assertTrue("Long" in err, "Expected Long type")
            else:
                self.fail(f"Unexpected dtype {str(tensor.dtype)} for error {err}")

            # Ensure sequence number is logged in error
            self.assertTrue("SequenceNumber" in err)
            # Ensure info about how collectives diff is in the error.
            if verify_diff:
                self.assertTrue(
                    "Collectives differ in the following" in err,
                    lambda msg: f"{msg}\nGot error {err}",
                )

    def _test_collective_hang(self, wrapper_pg, use_accel=False):
        # [FUNCTION DOC] HELPER: Hang detection check. Every rank except rank 1 issues an
        # allreduce that rank 1 never issues, and must get a RuntimeError. Rank 0 expects a
        # message naming rank 1 as failing the monitored barrier; other ranks expect a pointer
        # to rank 0 logs. The pattern also accepts Gloo connection closed/reset errors. When
        # use_accel is true the tensor is moved to the device whose index equals the rank.
        # All ranks besides 1 call allreduce and wrapper_pg should detect a hang
        # and report an issue with rank 1.
        faulty_rank = 1
        if self.rank != faulty_rank:
            tensor = torch.randn(20, 10)
            if use_accel:
                tensor = tensor.to(self.rank)

            if self.rank == 0:
                # Rank 0 reports faulty ranks
                err = f"Ranks {faulty_rank} failed to pass monitoredBarrier"
            else:
                err = "Please check rank 0 logs for faulty rank"

            # Gloo can sometimes throw the following error if a rank exits early
            # before rank 0 calls into the allreduce.
            err += "|Connection closed by peer|Connection reset by peer"
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    def _test_collectives_op_mismatch(self, wrapper_pg, use_accel=False):
        # [FUNCTION DOC] HELPER: Op mismatch check on wrapper_pg. First runs 500 allreduces
        # and waits for all of them. Then three mismatching rounds are run, each expected to
        # raise RuntimeError whose text is validated per rank: allreduce on rank 0 vs reduce
        # elsewhere; reduce on rank 0 vs barrier elsewhere; broadcast on rank 0 vs allgather
        # elsewhere. use_accel moves the tensor to the device indexed by the rank.
        tensor = torch.randn(20, 10)
        if use_accel:
            tensor = tensor.to(self.rank)
        works = []
        # Run a few successful collectives
        for _ in range(500):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)

        for w in works:
            w.wait()

        # Simulate mismatch: allreduce vs reduce.
        # Error including info about inconsistent collective, rank, tensor
        # shape, device, and dtype should be raised.
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE" if self.rank == 0 else "REDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()
        self._validate_error(
            exception=cm.exception,
            op_type="REDUCE" if self.rank == 0 else "BARRIER",
            rank=self.rank,
            tensor=tensor,
        )

        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                output_tensors = [
                    torch.zeros_like(tensor) for _ in range(self.world_size)
                ]
                wrapper_pg.allgather([output_tensors], [tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="BROADCAST" if self.rank == 0 else "ALLGATHER",
            rank=self.rank,
            tensor=tensor,
        )

    def _test_collective_shape_mismatch(self, wrapper_pg, use_accel=False):
        # [FUNCTION DOC] HELPER: Shape mismatch check. Starts with a barrier. Then expects
        # RuntimeError from allreduce when the second dimension differs across ranks, when the
        # number of dimensions differs (rank 0 has an extra one), and from scatter (root rank
        # 0) when the output/input sizes differ between root and other ranks. Each error is
        # validated with _validate_error for op type ALLREDUCE or SCATTER. use_accel places
        # tensors on the device indexed by the rank.
        wrapper_pg.barrier()
        dim = 2 if self.rank == 0 else 10
        tensor = torch.randn(20, dim)
        if use_accel:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        # Check errors are raised when dimensionality of shapes is different
        tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)
        if use_accel:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            wrapper_pg.allreduce([tensor])
        self._validate_error(
            exception=cm.exception,
            op_type="ALLREDUCE",
            rank=self.rank,
            tensor=tensor,
        )

        # Check shape errors with scatter
        input = [
            torch.tensor(
                [self.rank] if self.rank == 0 else [self.rank, self.rank],
                device=self.rank if use_accel else "cpu",
            )
            for _ in range(self.world_size)
        ]
        outputs = [
            torch.tensor(
                [-1] if self.rank == 0 else [-1, -1],
                device=self.rank if use_accel else "cpu",
            )
            for _ in range(self.world_size)
        ]
        root_rank = 0
        opts = c10d.ScatterOptions()
        opts.rootRank = root_rank
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == root_rank:
                wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
            else:
                wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()
        self._validate_error(
            exception=cm.exception,
            op_type="SCATTER",
            rank=self.rank,
            tensor=outputs[self.rank],
        )

    def _test_allgather_into_tensor_coalesced_op_mismatch(
        self, wrapper_pg, use_accel=False
    ):
        # [FUNCTION DOC] HELPER: Regression check that the coalesced allgather path runs the
        # wrapper collective checks. Rank 0 runs allgather_into_tensor_coalesced while other
        # ranks run reduce_scatter_tensor_coalesced, and every rank must get a RuntimeError
        # naming its own op type, the SequenceNumber and the 'Collectives differ in the
        # following' text. Shapes are not checked because these ops fingerprint with empty
        # tensors.
        # Verify runCollectiveChecks is actually invoked for the coalesced
        # allgather path by triggering an op type mismatch across ranks: rank 0
        # runs allgather_into_tensor_coalesced while the other ranks run
        # reduce_scatter_tensor_coalesced. Shape checking is intentionally
        # disabled for these ops (the fingerprint uses empty tensors), so only
        # the op type is verified in the raised error.
        dev = self.rank if use_accel else "cpu"
        with self.assertRaisesRegex(RuntimeError, ".*") as cm:
            if self.rank == 0:
                output = torch.zeros(4 * self.world_size, device=dev)
                input = torch.ones(4, device=dev)
                wrapper_pg.allgather_into_tensor_coalesced([output], [input]).wait()
            else:
                output = torch.zeros(4, device=dev)
                input = torch.ones(4 * self.world_size, device=dev)
                wrapper_pg.reduce_scatter_tensor_coalesced([output], [input]).wait()
        err = str(cm.exception)
        op_type = (
            "ALLGATHER_INTO_TENSOR_COALESCED"
            if self.rank == 0
            else "REDUCE_SCATTER_TENSOR_COALESCED"
        )
        self.assertTrue(
            op_type in err, f"Got {err} but expected {op_type} to be in error."
        )
        # These ops fingerprint with empty tensors, so no shape/dtype info is
        # present; only the sequence number and op-type diff are reported.
        self.assertTrue("SequenceNumber" in err)
        self.assertTrue(
            "Collectives differ in the following" in err, f"Got error {err}"
        )


# ASAN is not safe since we are spawning processes.
if not TEST_WITH_DEV_DBG_ASAN:

    @requires_gloo()
    @requires_accelerator_dist_backend(["nccl", "xccl"])
    class ProcessGroupNCCLWrapperTest(AbstractProcessGroupWrapperTest):
        # [CLASS DOC] Accelerator-backend (NCCL or XCCL) variant of the wrapper tests, run
        # with 2 processes. Requires gloo plus an accelerator backend. Builds either a wrapper
        # around a hand-built backend group or a new_group (which is wrapped automatically
        # under debug level DETAIL) and runs the shared mismatch checks plus NCCL-specific
        # base-op and forwarding tests. Only defined when ASAN dev-debug is not enabled.
        def setUp(self):
            # [FUNCTION DOC] Skip the abstract class's setUp (the call goes straight to the
            # grandparent), spawn the processes, then set the async error handling env var to
            # 1 in the parent environment.
            # KNOWN ISSUE: the env var is set after the worker processes are spawned, so the
            # children may not see it.
            super(AbstractProcessGroupWrapperTest, self).setUp()
            self._spawn_processes()
            # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
            # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
            os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

        @property
        def world_size(self) -> int:
            # [FUNCTION DOC] Property: the world size is always 2.
            return 2

        def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
            # [FUNCTION DOC] HELPER: Initialize the default process group with a file store
            # and the accelerator backend, using the timeout in seconds. With with_new_group a
            # new group is returned. Otherwise a ProcessGroupXCCL (xpu) or ProcessGroupNCCL is
            # built directly and wrapped with the process group wrapper factory. Returns the
            # group.
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                store=store,
                timeout=timedelta(seconds=timeout),
            )
            if with_new_group:
                pg = c10d.new_group(backend=backend, timeout=timedelta(seconds=timeout))
            else:
                if device_type == "xpu":
                    _pg = c10d.ProcessGroupXCCL(
                        store,
                        self.rank,
                        self.world_size,
                    )
                else:
                    _pg = c10d.ProcessGroupNCCL(
                        store,
                        self.rank,
                        self.world_size,
                        timeout=timedelta(seconds=timeout),
                    )
                pg = c10d._create_process_group_wrapper(
                    _pg,
                    "unused",
                    store,
                    self.rank,
                    self.world_size,
                    timeout=timeout,
                )
            return pg

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        def test_collective_hang(self):
            # [FUNCTION DOC] Test: with a 2 second timeout, the hang check must report a
            # failure on the faulty rank.
            pg = self._create_wrapper_pg(timeout=2.0)
            self._test_collective_hang(pg)

        # NOTE: these tests are separated by debug level instead of combined into
        # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
        # combined after that is resolved.
        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_collectives_op_mismatch_debug_mode(self):
            # [FUNCTION DOC] Test (debug level DETAIL, new group): runs the shared op mismatch
            # check on the accelerator, then the NCCL-only base op mismatch check.
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collectives_op_mismatch(pg, use_accel=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["OFF"])
        def test_collectives_op_mismatch(self):
            # [FUNCTION DOC] Test (debug level OFF, explicitly wrapped group): same op
            # mismatch checks as the DETAIL variant.
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collectives_op_mismatch(pg, use_accel=True)
            self._test_nccl_only_op_mismatch(pg)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_collective_shape_mismatch_debug_mode_detail(self):
            # [FUNCTION DOC] Test (debug level DETAIL, new group): runs the shared shape
            # mismatch check, then the NCCL-only shape mismatch check.
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_collective_shape_mismatch(pg, use_accel=True)
            self._test_nccl_only_shape_mismatch(pg)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["OFF"])
        def test_collective_shape_mismatch_debug_mode_off(self):
            # [FUNCTION DOC] Test (debug level OFF, explicitly wrapped group): same shape
            # mismatch checks as the DETAIL variant, despite the name saying debug mode off.
            pg = self._create_wrapper_pg(with_new_group=False)
            self._test_collective_shape_mismatch(pg, use_accel=True)
            self._test_nccl_only_shape_mismatch(pg)

        def _test_nccl_only_op_mismatch(self, wrapper_pg):
            # [FUNCTION DOC] HELPER: Expect RuntimeError when rank 0 calls the private
            # allgather base while other ranks call the private reduce-scatter base on the
            # accelerator; validates the error names ALLGATHER_BASE or REDUCE_SCATTER_BASE and
            # describes the input.
            device = f"{device_type}:{self.rank}"
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * self.world_size, device=device)
                if self.rank == 0:
                    wrapper_pg._allgather_base(output, input).wait()
                else:
                    wrapper_pg._reduce_scatter_base(output, input).wait()

            op_type = "ALLGATHER_BASE" if self.rank == 0 else "REDUCE_SCATTER_BASE"
            self._validate_error(
                exception=cm.exception,
                op_type=op_type,
                rank=self.rank,
                tensor=input,
            )

        def _test_nccl_only_shape_mismatch(self, wrapper_pg):
            # [FUNCTION DOC] HELPER: Expect RuntimeError from the private reduce-scatter base
            # when the input size is not world-size times the output size, first with a
            # fixed-size output and wrong input, then with per-rank differing input size.
            # Errors are validated for op REDUCE_SCATTER_BASE without requiring the collective
            # diff text.
            device = f"{device_type}:{self.rank}"
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4 + self.rank, device=device)
                input = torch.ones(4 * (self.world_size + 1), device=device)

                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(
                exception=cm.exception,
                op_type="REDUCE_SCATTER_BASE",
                rank=self.rank,
                tensor=input,
                verify_diff=False,
            )
            with self.assertRaisesRegex(RuntimeError, ".*") as cm:
                output = torch.zeros(4, device=device)
                input = torch.ones((4 + self.rank) * self.world_size, device=device)

                wrapper_pg._reduce_scatter_base(output, input).wait()
            self._validate_error(
                exception=cm.exception,
                op_type="REDUCE_SCATTER_BASE",
                rank=self.rank,
                tensor=input,
                verify_diff=False,
            )

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_coalescing_manager_debug_mode_detail(self):
            # [FUNCTION DOC] Test (DETAIL): starting and ending coalescing on the wrapped
            # group around one allreduce must not crash. Sets the accelerator device index to
            # the rank first.
            """
            Tests that coalescing manager w/TORCH_DISTRIBUTED_DEBUG
            does not crash: https://github.com/pytorch/pytorch/issues/109520
            """
            torch.accelerator.set_device_index(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            dev = torch.accelerator.current_device_index()
            pg._start_coalescing(torch.device(dev))
            pg.allreduce([torch.ones(1, device=dev)])
            pg._end_coalescing(torch.device(dev))

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_reduce_scatter_tensor_coalesced_debug_mode(self):
            # [FUNCTION DOC] Test (DETAIL): runs reduce_scatter_tensor_coalesced on a wrapped
            # group with two output shapes and checks each output equals the sum over ranks of
            # (rank + 1). Uses CUDA device calls directly.
            # KNOWN ISSUE: uses torch.cuda rather than the accelerator API, unlike the rest of
            # the class, so it would fail on XPU.
            torch.cuda.set_device(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            dev = torch.cuda.current_device()

            out_shapes = [(2, 2), (3, 3)]
            in_shapes = [(s[0] * self.world_size,) + s[1:] for s in out_shapes]

            outputs = [torch.zeros(s, device=dev) for s in out_shapes]
            inputs = [torch.ones(s, device=dev) * (self.rank + 1) for s in in_shapes]

            work = pg.reduce_scatter_tensor_coalesced(outputs, inputs)
            work.wait()

            for i, (output, _) in enumerate(zip(outputs, inputs)):
                expected = torch.ones(out_shapes[i], device=dev) * sum(
                    range(1, self.world_size + 1)
                )
                self.assertTrue(torch.allclose(output, expected))

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_allgather_into_tensor_coalesced_debug_mode(self):
            # [FUNCTION DOC] Test (DETAIL): runs allgather_into_tensor_coalesced on a wrapped
            # group with two input shapes and checks each output holds rank r's constant (r +
            # 1) in its r-th slice. Uses CUDA device calls directly.
            """
            Tests that allgather_into_tensor_coalesced correctly invokes
            runCollectiveChecks in ProcessGroupWrapper. Previously this was the
            only collective missing the check.
            Regression test for https://github.com/pytorch/pytorch/pull/185123
            """
            torch.cuda.set_device(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            dev = torch.cuda.current_device()

            in_shapes = [(2, 2), (3, 3)]
            out_shapes = [(s[0] * self.world_size,) + s[1:] for s in in_shapes]

            inputs = [torch.ones(s, device=dev) * (self.rank + 1) for s in in_shapes]
            outputs = [torch.zeros(s, device=dev) for s in out_shapes]

            work = pg.allgather_into_tensor_coalesced(outputs, inputs)
            work.wait()

            # Each output gathers inputs from all ranks: rank r contributes r + 1.
            for i, output in enumerate(outputs):
                expected = torch.zeros(out_shapes[i], device=dev)
                for r in range(self.world_size):
                    start = r * in_shapes[i][0]
                    end = (r + 1) * in_shapes[i][0]
                    expected[start:end] = torch.ones(in_shapes[i], device=dev) * (r + 1)
                self.assertTrue(torch.allclose(output, expected))

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_allgather_into_tensor_coalesced_op_mismatch_debug_mode(self):
            # [FUNCTION DOC] Test (DETAIL): sets the CUDA device and runs the coalesced
            # allgather op mismatch check on the accelerator.
            """
            Verify runCollectiveChecks detects a collective mismatch on the
            coalesced allgather path.
            Regression test for https://github.com/pytorch/pytorch/pull/185123
            """
            torch.cuda.set_device(self.rank)
            pg = self._create_wrapper_pg(with_new_group=True)
            self._test_allgather_into_tensor_coalesced_op_mismatch(pg, use_accel=True)

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        @patch(
            "torch.distributed.distributed_c10d.is_gloo_available",
            lambda: False,
        )
        def test_debug_level_detail_no_gloo(self):
            # [FUNCTION DOC] Test (DETAIL, gloo availability patched to false): creating the
            # wrapped group must raise an assertion error saying the wrapper is unsupported
            # without the Gloo backend.
            with self.assertRaisesRegex(
                AssertionError, "ProcessGroupWrapper unsupported without GLOO backend"
            ):
                self._create_wrapper_pg()

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_wrapper_forwards_nccl_methods(self):
            # [FUNCTION DOC] Test (DETAIL, NCCL only): checks that the wrapper forwards
            # backend-specific properties and methods to the wrapped group (splitting,
            # coalescing and time estimate support, tensor alloc support, error, options,
            # eager connect), and when tensor allocation is supported also the memory
            # allocator and allocate_tensor returning a tensor of the requested size.
            """
            Tests that ProcessGroupWrapper correctly forwards NCCL-specific
            utility methods to the wrapped backend. See issue #173538.
            """
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")
            wrapper = self._create_wrapper_pg(with_new_group=False)

            # Verify we're testing the wrapper
            self.assertIsInstance(wrapper, _ProcessGroupWrapper)
            unwrapped = wrapper.wrapped_pg

            # Verify wrapper forwards property/method calls to wrapped backend
            self.assertEqual(wrapper.supports_splitting, unwrapped.supports_splitting)
            self.assertEqual(wrapper.supports_coalescing, unwrapped.supports_coalescing)
            self.assertEqual(
                wrapper.supports_time_estimate, unwrapped.supports_time_estimate
            )
            self.assertEqual(
                wrapper.supports_tensor_alloc(device),
                unwrapped.supports_tensor_alloc(device),
            )
            self.assertEqual(wrapper.get_error(), unwrapped.get_error())
            self.assertIs(wrapper.options, unwrapped.options)

            # Test eager_connect_single_device forwarding (should not raise)
            wrapper.eager_connect_single_device(device)

            # mem_allocator and allocate_tensor require NCCL >= 2.19 and GPU
            # multicast support (e.g., not available on T4)
            if wrapper.supports_tensor_alloc(device):
                self.assertIs(wrapper.mem_allocator, unwrapped.mem_allocator)
                tensor = wrapper.allocate_tensor(
                    1024, dtype=torch.float32, device=device
                )
                self.assertEqual(tensor.shape, torch.Size([1024]))

        @requires_nccl()
        @skip_if_lt_x_gpu(2)
        @with_dist_debug_levels(levels=["DETAIL"])
        def test_wrapper_forwards_bound_device_id(self):
            # [FUNCTION DOC] Test (DETAIL, NCCL only): init_process_group with a device_id
            # must yield a default group whose backend for that device is a wrapper whose
            # wrapped group has bound_device_id equal to that device.
            """
            Tests that ProcessGroupWrapper propagates bound_device_id to the
            wrapped backend when init_process_group is called with device_id.
            See issue #178977.
            """
            torch.cuda.set_device(self.rank)
            device = torch.device(f"cuda:{self.rank}")
            store = c10d.FileStore(self.file_name, self.world_size)
            c10d.init_process_group(
                backend=backend,
                rank=self.rank,
                world_size=self.world_size,
                store=store,
                device_id=device,
            )
            pg = c10d.distributed_c10d._get_default_group()
            pg_backend = pg._get_backend(device)
            self.assertIsInstance(pg_backend, _ProcessGroupWrapper)
            self.assertEqual(pg_backend.wrapped_pg.bound_device_id, device)

        @requires_accelerator_dist_backend(["nccl", "xccl"])
        @skip_if_lt_x_gpu(2)
        @patch(
            "torch.distributed.distributed_c10d.is_gloo_available",
            lambda: False,
        )
        def test_new_group_no_gloo(self):
            # [FUNCTION DOC] Test: with gloo availability patched to false, creating a new
            # group must succeed without ever evaluating isinstance against the wrapper class;
            # the patched isinstance raises NameError if it does.
            def patched_isinstance(obj, clazz):
                # [FUNCTION DOC] HELPER: Replacement isinstance used by the test: raises
                # NameError when the class checked is the wrapper type, otherwise defers to
                # the real isinstance.
                if clazz is _ProcessGroupWrapper:
                    raise NameError
                else:
                    return isinstance(obj, clazz)

            with patch(
                "torch.distributed.distributed_c10d.isinstance",
                side_effect=patched_isinstance,
            ):
                self._create_wrapper_pg(with_new_group=True)
                # nothing to assert, isinstance(pg, _ProcessGroupWrapper)
                # should never be invoked since it is proceeded by
                # Gloo availability check, this test will fail on
                # an unexpected NameError if not.


@requires_gloo()
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):
    # [CLASS DOC] Gloo-backed variant of the wrapper tests (requires gloo). Builds either a
    # new gloo group or a manually wrapped ProcessGroupGloo over loopback, and runs the shared
    # hang, op mismatch, shape mismatch checks on CPU and (with 4 GPUs) on accelerator
    # tensors, plus coalesced collective checks.
    def opts(self, threads=2, timeout=10.0):
        # [FUNCTION DOC] HELPER: Build Gloo options with the given thread count and timeout
        # (seconds), using a single loopback device. Returns the options object.
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        # [FUNCTION DOC] HELPER: Initialize the default gloo group on a file store. With
        # with_new_group returns a new gloo group; otherwise builds a ProcessGroupGloo with
        # the options helper and wraps it using the wrapper factory. Returns the group.
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        if with_new_group:
            pg = c10d.new_group(backend="gloo")
        else:
            _pg = c10d.ProcessGroupGloo(
                store, self.rank, self.world_size, self.opts(timeout=timeout)
            )
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    def test_collective_hang(self):
        # [FUNCTION DOC] Test: with a 2 second timeout, the hang check must report the faulty
        # rank.
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        # [FUNCTION DOC] Test (DETAIL, new group): shared op mismatch check on CPU tensors.
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        # [FUNCTION DOC] Test (OFF, explicitly wrapped): shared op mismatch check on CPU
        # tensors.
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        # [FUNCTION DOC] Test (DETAIL, new group): shared shape mismatch check on CPU tensors.
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_debug_mode_off(self):
        # [FUNCTION DOC] Test (OFF, explicitly wrapped): shared shape mismatch check on CPU
        # tensors.
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        # [FUNCTION DOC] Test (needs 4 GPUs, skipped on XPU, DETAIL): op mismatch check with
        # accelerator tensors over gloo.
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_accel=True)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch_cuda(self):
        # [FUNCTION DOC] Test (needs 4 GPUs, skipped on XPU, OFF): op mismatch check with
        # accelerator tensors over gloo.
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_accel=True)

    @skip_if_lt_x_gpu(4)
    @unittest.skipIf(TEST_XPU, "XPU does not support gloo backend")
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        # [FUNCTION DOC] Test (needs 4 GPUs, skipped on XPU, DETAIL): shape mismatch check
        # with accelerator tensors over gloo.
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_accel=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_cuda(self):
        # [FUNCTION DOC] Test (needs 4 GPUs, OFF): shape mismatch check with accelerator
        # tensors over gloo.
        # KNOWN ISSUE: unlike the sibling cuda tests it lacks the XPU skip decorator.
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_accel=True)

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_reduce_scatter_tensor_coalesced_debug_mode(self):
        # [FUNCTION DOC] Test (DETAIL): reduce_scatter_tensor_coalesced on CPU tensors with
        # two shapes; each output must equal the sum over ranks of (rank + 1).
        pg = self._create_wrapper_pg(with_new_group=True)

        out_shapes = [(2, 2), (3, 3)]
        in_shapes = [(s[0] * self.world_size,) + s[1:] for s in out_shapes]
        outputs = [torch.zeros(s) for s in out_shapes]
        inputs = [torch.ones(s) * (self.rank + 1) for s in in_shapes]

        work = pg.reduce_scatter_tensor_coalesced(outputs, inputs)
        work.wait()
        for i, (output, _) in enumerate(zip(outputs, inputs)):
            expected = torch.ones(out_shapes[i]) * sum(range(1, self.world_size + 1))
            self.assertTrue(torch.allclose(output, expected))

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_allgather_into_tensor_coalesced_debug_mode(self):
        # [FUNCTION DOC] Test (DETAIL): allgather_into_tensor_coalesced on CPU tensors with
        # two shapes; each output must hold rank r's constant (r + 1) in its r-th slice.
        """
        Tests that allgather_into_tensor_coalesced correctly invokes
        runCollectiveChecks in ProcessGroupWrapper (Gloo backend).
        Regression test for https://github.com/pytorch/pytorch/pull/185123
        """
        pg = self._create_wrapper_pg(with_new_group=True)

        in_shapes = [(2, 2), (3, 3)]
        out_shapes = [(s[0] * self.world_size,) + s[1:] for s in in_shapes]
        inputs = [torch.ones(s) * (self.rank + 1) for s in in_shapes]
        outputs = [torch.zeros(s) for s in out_shapes]

        work = pg.allgather_into_tensor_coalesced(outputs, inputs)
        work.wait()

        # Each output gathers inputs from all ranks: rank r contributes r + 1.
        for i, output in enumerate(outputs):
            expected = torch.zeros(out_shapes[i])
            for r in range(self.world_size):
                start = r * in_shapes[i][0]
                end = (r + 1) * in_shapes[i][0]
                expected[start:end] = torch.ones(in_shapes[i]) * (r + 1)
            self.assertTrue(torch.allclose(output, expected))

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_allgather_into_tensor_coalesced_op_mismatch_debug_mode(self):
        # [FUNCTION DOC] Test (DETAIL): runs the coalesced allgather op mismatch check on CPU
        # tensors.
        """
        Verify runCollectiveChecks detects a collective mismatch on the
        coalesced allgather path (Gloo backend).
        Regression test for https://github.com/pytorch/pytorch/pull/185123
        """
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_allgather_into_tensor_coalesced_op_mismatch(pg)


if __name__ == "__main__":
    if torch.cuda.is_initialized() or torch.xpu.is_initialized():
        raise AssertionError(
            "test_pg_wrapper must not have initialized CUDA/XPU context on main process"
        )

    run_tests()
