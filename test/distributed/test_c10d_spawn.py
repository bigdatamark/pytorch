# Owner(s): ["oncall: distributed"]
# [MODULE DOC] Shared helpers for c10d tests that spawn processes. Provides
# AbstractProcessGroupShareTensorTest (mixin that spawns ranks with torch multiprocessing, shares
# tensors with them and compares per-rank results gathered through queues) and
# TestDistributedNNFunctions (a 2-process MultiProcessTestCase with backend-parametrised helpers
# for the torch.distributed.nn autograd collectives, reused by the gloo/nccl/ucc spawn test
# files). Import-time side effects: sets _torch_dist_nn_available to false if torch.distributed.nn
# cannot be imported, rebinds load_tests, and exits the process quietly with status 0 if c10d is
# unavailable. Running as a script calls run_tests.

import os
import sys
import tempfile

import torch
import torch.distributed as c10d
import torch.multiprocessing as mp
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import load_tests, run_tests


# Torch distributed.nn is not available in windows
# check #42095, it errors on import.
_torch_dist_nn_available = True
try:
    import torch.distributed.nn
except ImportError:
    _torch_dist_nn_available = False

# load_tests from common_utils is used to automatically filter tests for
# sharding on sandcastle. This line silences flake warnings
load_tests = load_tests  # noqa: PLW0127

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)


class AbstractProcessGroupShareTensorTest:
    # [CLASS DOC] Mixin (no TestCase base of its own) for tests that run a collective in
    # separately spawned processes with tensors shared across processes. world_size is 2. The
    # _test_multiprocess driver starts the ranks; the classmethod worker functions
    # _test_broadcast_process, _test_allreduce_process and _test_allgather_process run in the
    # children and report results through a queue.
    world_size = 2

    def _test_multiprocess(self, f, shared_tensors, init_pg, n_output):
        # [FUNCTION DOC] HELPER: Driver. Creates a temporary file (left undeleted here; the file
        # store is expected to remove it) and spawns world_size processes with the spawn context,
        # each running the given worker with (rank, file name, shared_tensors, world_size,
        # init_pg, c2p, p2c). Then reads world_size times n_output (rank, expected, result) tuples
        # from the child-to-parent queue and asserts expected equals result for each, with a
        # message naming the rank. Finally releases every child via the parent-to-child queue and
        # joins each process with a 2 second timeout (no exit-code check).
        ws = self.world_size
        # file store will delete the test file on destruction
        file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        ctx = mp.get_context("spawn")
        c2p = ctx.Queue(2)
        p2c = ctx.Queue(2)
        ps = []
        for i in range(ws):
            p = ctx.Process(
                target=f, args=(i, file.name, shared_tensors, ws, init_pg, c2p, p2c)
            )

            p.start()
            ps.append(p)

        for _ in range(ws * n_output):
            pid, expected, result = c2p.get()
            self.assertEqual(
                expected,
                result,
                msg=lambda msg: f"{msg}\nExpect rank {pid} to receive tensor {expected} but got {result}.",
            )

        for _ in range(ws):
            p2c.put(0)

        for p in ps:
            p.join(2)

    # Why classmethod? multiprocessing cannot pickle TestCase subclass when in
    # spawn mode. See https://bugs.python.org/issue33884.
    @classmethod
    def _test_broadcast_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        # [FUNCTION DOC] HELPER: Worker run in a spawned child. Creates the process group via
        # init_pg, broadcasts this rank's shared tensor (default root) and waits on the work, then
        # sends (rank, zero tensor expected, resulting tensor on cpu) to the parent and blocks
        # until the parent releases it. A classmethod because spawn mode cannot pickle TestCase
        # subclasses.
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.broadcast(xs).wait()
        c2p.put((rank, torch.zeros(2, 2), xs[0].to("cpu")))
        p2c.get()

    @classmethod
    def _test_allreduce_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        # [FUNCTION DOC] HELPER: Worker run in a spawned child. Creates the process group via
        # init_pg, all-reduces (sum) this rank's shared tensor in place and waits, then sends
        # (rank, 2x2 tensor of twos expected, resulting tensor on cpu) to the parent and blocks
        # until released.
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        pg.allreduce(xs, op=c10d.ReduceOp.SUM).wait()
        c2p.put((rank, torch.ones(2, 2) * 2, xs[0].to("cpu")))
        p2c.get()

    @classmethod
    def _test_allgather_process(
        cls, rank, filename, shared_tensors, world_size, init_pg, c2p, p2c
    ):
        # [FUNCTION DOC] HELPER: Worker run in a spawned child. Creates the process group via
        # init_pg, all-gathers this rank's shared tensor into world_size zero-initialised outputs
        # and waits, then sends one (rank, 2x2 tensor filled with i expected, gathered tensor i on
        # cpu) tuple per source rank i, and blocks until released.
        pg = init_pg(rank, filename, world_size)
        xs = [shared_tensors[rank]]
        ys = [[torch.zeros_like(xs[0]) for i in range(world_size)]]
        pg.allgather(ys, xs).wait()
        for i in range(world_size):
            c2p.put((rank, torch.ones(2, 2) * i, ys[0][i].to("cpu")))

        p2c.get()


class TestDistributedNNFunctions(MultiProcessTestCase):
    # [CLASS DOC] Two-process MultiProcessTestCase for the torch.distributed.nn autograd
    # collectives. setUp spawns the worker processes; the _test_* helpers take a backend name,
    # build a FileStore on the per-test file, init the default process group, and exercise one nn
    # collective on cuda tensors including the backward gradients. Backend-specific subclasses
    # call these helpers.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and then spawns the worker processes for this test.
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        # [FUNCTION DOC] Runs the base tearDown, then removes the test's file-store file, ignoring
        # OSError if it is already gone.
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        # [FUNCTION DOC] Property returning 1: the per-operation timeout in seconds used by the
        # test framework.
        return 1

    @property
    def world_size(self):
        # [FUNCTION DOC] Property returning 2, the number of ranks.
        return 2

    def _test_broadcast(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend on a FileStore,
        # then runs the nn broadcast with source rank 1 on a 5x5 tensor of ones plus rank
        # requiring grad. Checks the result equals the source rank's value, backpropagates the
        # sine sum, and checks the gradient is twice the cosine of x on rank 1 and zeros on rank
        # 0.
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.broadcast(x, 1)
        self.assertEqual(y, 1 + torch.ones(5, 5))
        z = y.sin().sum()
        z.backward()
        # We can't check the gradient of communications numerically so we have to do some calculations
        if self.rank == 1:
            self.assertEqual(x.grad, 2 * torch.cos(x))
        elif self.rank == 0:
            self.assertEqual(x.grad, torch.zeros(5, 5, device=device))

    def _test_reduce(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend, then runs the nn
        # reduce (sum) to destination rank 1 on ones plus rank. On rank 1 checks the result is
        # three everywhere. Backpropagates the sine sum and checks that every rank's gradient is
        # the cosine of three (the gradient is broadcast back).
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.reduce(x, 1, op=c10d.ReduceOp.SUM)

        if self.rank == 1:
            self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        # Gradients are broadcasted to both ranks
        x_g = (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def _test_allreduce(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend, runs the nn
        # all_reduce (sum) on ones plus rank, checks the result is three everywhere,
        # backpropagates the sine sum, and checks the gradient is twice the cosine of three.
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        y = torch.distributed.nn.all_reduce(x, op=c10d.ReduceOp.SUM)

        self.assertEqual(y, 3 * torch.ones(5, 5, device=device))

        z = y.sin().sum()
        z.backward()
        x_g = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_g)

    def _test_all_gather(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend, runs the nn
        # all_gather on ones plus rank, checks the i-th gathered tensor is ones plus i,
        # backpropagates the sine of the stacked sum, and checks the gradient is twice the cosine
        # of three.
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x = torch.ones(5, 5, device=device) + self.rank
        x.requires_grad = True
        tensors = torch.distributed.nn.all_gather(x)
        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.ones(5, 5, device=device) + i)
        y = torch.sum(torch.stack(tensors), axis=0)
        z = y.sin().sum()
        z.backward()

        x_s = 2 * (3 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x.grad, x_s)

    def _test_all_to_all(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend, runs the nn
        # all_to_all with two inputs per rank (ones plus twice the rank) into empty outputs,
        # checks the i-th received tensor is ones plus twice i, backpropagates the sine of the
        # summed outputs, and checks both input gradients are the cosine of four.
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        x0 = torch.ones(5, 5, device=device) + 2 * self.rank
        x1 = torch.ones(5, 5, device=device) + 2 * self.rank
        x0.requires_grad = True
        x1.requires_grad = True
        y0 = torch.empty_like(x0)
        y1 = torch.empty_like(x1)
        tensors = torch.distributed.nn.all_to_all([y0, y1], [x0, x1])
        for i, t in enumerate(tensors):
            self.assertEqual(t, torch.ones(5, 5, device=device) + 2 * i)
        y = torch.sum(torch.stack(tensors), axis=0)
        z = y.sin().sum()
        z.backward()
        x_s = (4 * torch.ones(5, 5, device=device)).cos()
        self.assertEqual(x0.grad, x_s)
        self.assertEqual(x1.grad, x_s)

    def _test_all_to_all_single(self, backend):
        # [FUNCTION DOC] HELPER: Inits the process group with the given backend, runs the nn
        # all_to_all_single using the same split sizes for input and output (the chunk for peer
        # i has i plus one times rank plus one rows) on a tensor filled with rank plus one. Checks
        # the output chunk i is filled with i plus one, backpropagates the sine sum, and checks
        # the gradient is the cosine of rank plus one.
        store = c10d.FileStore(self.file_name, self.world_size)
        # This is required because these functions calls directly to the .dist and needs
        # the world to be initialized
        c10d.init_process_group(
            store=store, rank=self.rank, world_size=self.world_size, backend=backend
        )
        device = torch.device(f"cuda:{self.rank}")
        row = self.world_size * (self.rank + 1) * (self.world_size + 1) / 2
        x = torch.ones(int(row), 5, device=device) * (self.rank + 1)
        x.requires_grad = True
        y = torch.empty_like(x)
        split_sizes = [(i + 1) * (self.rank + 1) for i in range(self.world_size)]
        y = torch.distributed.nn.all_to_all_single(
            y, x, output_split_sizes=split_sizes, input_split_sizes=split_sizes
        )
        expected = []
        for idx, tensor in enumerate(torch.split(x, split_sizes)):
            expected.append(torch.full_like(tensor, (idx + 1)))
        expected = torch.cat(expected)
        self.assertEqual(y, expected)
        z = y.sin().sum()
        z.backward()
        x_s = ((self.rank + 1) * torch.ones(int(row), 5, device=device)).cos()
        self.assertEqual(x.grad, x_s)


if __name__ == "__main__":
    run_tests()
