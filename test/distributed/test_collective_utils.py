# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Tests for torch.distributed.collective_utils: broadcast and all_gather of a
# function result (with and without a process group, exception propagation), the RNG
# synchronization checker over gloo, and the rank-list summarizer (using a fake process group
# with 4096 ranks). Defines TestCollectiveUtils (multi-process) and TestUtils; parametrized
# tests are instantiated at import time.

from unittest import mock

import torch
import torch.distributed as c10d
from torch.distributed.collective_utils import (
    _check_rng_sync,
    _check_rng_sync_internal,
    _summarize_ranks,
    all_gather,
    broadcast,
)
from torch.distributed.device_mesh import init_device_mesh
from torch.testing import FileCheck
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.distributed.fake_pg import FakeStore


class TestCollectiveUtils(MultiProcessTestCase):
    # [CLASS DOC] Multi-process test case (processes spawned in setUp) exercising the
    # broadcast, all_gather and RNG-sync helpers against real gloo process groups.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup, then spawns the worker processes.
        super().setUp()
        self._spawn_processes()

    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Only runs the parent teardown.
        super().tearDown()

    def opts(self, threads=2):
        # [FUNCTION DOC] HELPER: Builds gloo process group options with a timeout of 50
        # seconds and the given number of threads (default 2); returns them.
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = 50.0
        opts._threads = threads
        return opts

    def test_broadcast_result(self) -> None:
        # [FUNCTION DOC] Creates a gloo group, then broadcasts the result of a mock function
        # returning each rank's own rank, first from source rank 0 then from rank 1. Asserts
        # all ranks receive the source's value and that only the source rank called the
        # function. Raises AssertionError on mismatch.
        """
        Basic unit test for broadcast using a process group of default world size.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = broadcast(data_or_fn=func, rank=0, pg=pg)
        if res != 0:
            raise AssertionError(f"Expect res to be 0 (got {res})")

        if pg.rank() == 0:
            func.assert_called_once()
        else:
            func.assert_not_called()

        func.reset_mock()

        res = broadcast(data_or_fn=func, rank=1, pg=pg)
        if res != 1:
            raise AssertionError(f"Expect res to be 1 (got {res})")

        if pg.rank() == 1:
            func.assert_called_once()
        else:
            func.assert_not_called()

    def test_broadcast_result_no_pg(self) -> None:
        # [FUNCTION DOC] Calls broadcast with no process group and checks the function is
        # invoked exactly once locally.
        """
        Ensure broadcast has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        broadcast(data_or_fn=func, rank=0)
        func.assert_called_once()

    def test_broadcast_result_raises_exceptions_from_func(
        self,
    ) -> None:
        # [FUNCTION DOC] With no process group, a function that raises an exception must
        # have that exception propagate out of broadcast (matched by message).
        """
        Ensure broadcast exception is propagated properly.
        """
        # no process group
        func = mock.MagicMock()
        exc = Exception("test exception")
        func.side_effect = exc
        expected_exception = "test exception"
        with self.assertRaisesRegex(Exception, expected_exception):
            broadcast(data_or_fn=func, rank=0)

    def test_all_gather_result(self) -> None:
        # [FUNCTION DOC] Creates a gloo group and gathers a mock function's per-rank result
        # (the rank). Asserts the function ran once per process and every rank gets the list
        # of ranks in order. Raises AssertionError on mismatch.
        """
        Basic unit test for all_gather using a process group of default world size.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        pg = c10d.new_group(pg_options=self.opts())

        func = mock.MagicMock()
        func.return_value = pg.rank()

        res = all_gather(data_or_fn=func, pg=pg)
        func.assert_called_once()
        if res != list(range(self.world_size)):
            raise AssertionError(
                f"Expect res to be list of 0 through {self.world_size} (got {res})"
            )

    def test_all_gather_result_no_pg(self) -> None:
        # [FUNCTION DOC] Calls all_gather with no process group and checks the function is
        # invoked exactly once locally.
        """
        Ensure all_gather has no dependency on torch.distributed when run in single process.
        """
        func = mock.MagicMock()
        all_gather(data_or_fn=func)
        func.assert_called_once()

    def test_all_gather_result_raises_exceptions_from_func(
        self,
    ) -> None:
        # [FUNCTION DOC] With no process group, a function that raises an exception must
        # have that exception propagate out of all_gather (matched by message).
        """
        Ensure all_gather exception is propagated properly.
        """
        # no process group
        func = mock.MagicMock()
        exc = Exception("test exception")
        func.side_effect = exc
        expected_exception = "test exception"
        with self.assertRaisesRegex(Exception, expected_exception):
            all_gather(data_or_fn=func)

    @parametrize("device", ["cpu", "cuda"])
    @skip_if_lt_x_gpu(4)
    def test_check_rng_sync(
        self,
        device,
    ) -> None:
        # [FUNCTION DOC] Parametrized over cpu and cuda (needs at least 4 GPUs, skips if cuda
        # is missing). On a 4-rank gloo group, seeds a generator identically and checks the
        # internal RNG sync check groups all ranks together; then advances the generator on
        # rank 1 (two groups: {0,2,3} and {1}); then reseeds on rank 0 (three groups). Finally
        # checks the textual report mentions a generator desync, ranks, and the summarized
        # range 2:4.
        if device == "cuda" and not torch.cuda.is_available():
            self.skipTest("Cuda is not available")
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", store=store, rank=self.rank, world_size=self.world_size
        )
        group = torch.distributed.distributed_c10d._get_default_group()
        generator = torch.Generator(device=device)
        generator.manual_seed(123)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 1, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0, 1, 2, 3}]):
            self.assertEqual(actual, expected, actual)

        if torch.distributed.get_rank() == 1:
            torch.randn((10,), device=device, generator=generator)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 2, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0, 2, 3}, {1}]):
            self.assertEqual(actual, expected, actual)

        if torch.distributed.get_rank() == 0:
            generator.manual_seed(456)
        value_ranks, _ = _check_rng_sync_internal(generator, group)
        self.assertEqual(len(value_ranks), 3, value_ranks)
        for actual, expected in zip(value_ranks.values(), [{0}, {1}, {2, 3}]):
            self.assertEqual(actual, expected, actual)

        log_str = _check_rng_sync(generator, group)
        FileCheck().check("Generator desync detected").check("Ranks").check("0").check(
            "1"
        ).check("2:4").run(log_str)


class TestUtils(TestCase):
    # [CLASS DOC] Single-process test case for the rank summarizer, running on a fake
    # process group of world size 4096 initialized in setUp when none exists.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setup and, if no process group is initialized,
        # sets rank 0 and world size 4096 and initializes the fake backend with a fake store.
        super().setUp()

        if not c10d.is_initialized():
            self.rank = 0
            self.world_size = 4096

            store = FakeStore()
            c10d.init_process_group(
                backend="fake",
                world_size=self.world_size,
                rank=self.rank,
                store=store,
            )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Destroys the default process group.
        c10d.destroy_process_group()

    def test_summarize_ranks(self):
        # [FUNCTION DOC] Builds a CPU device mesh of shape 8 by 64 by 8 (pp, dp, tp) and
        # checks the summarized rank strings of each named dimension's rank list against
        # strided range notation (pp, dp, tp), plus a mixed list giving comma-joined ranges
        # and a single rank giving just that number.
        mesh_dim_names = ("pp", "dp", "tp")
        mesh = init_device_mesh("cpu", (8, 64, 8), mesh_dim_names=mesh_dim_names)
        ranks_lists = {name: mesh[name].mesh.tolist() for name in mesh_dim_names}
        summaries = {
            name: _summarize_ranks(ranks_lists[name]) for name in mesh_dim_names
        }
        self.assertEqual(summaries["pp"], "0:4096:512")
        self.assertEqual(summaries["dp"], "0:512:8")
        self.assertEqual(summaries["tp"], "0:8")

        self.assertEqual(
            _summarize_ranks([1, 2, 3, 6, 7, 8, 10, 12, 14, 16]),
            "1:4,6:9,10:18:2",
        )
        self.assertEqual(
            _summarize_ranks([1]),
            "1",
        )


instantiate_parametrized_tests(TestCollectiveUtils)

if __name__ == "__main__":
    run_tests()
