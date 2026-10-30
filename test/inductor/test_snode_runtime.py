# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor's scheduler-node runtime estimates and input-distance
# metrics. Compiles functions with a backend that runs Inductor lowering but executes the
# original graph eagerly, then inspects the metrics module for per-node runtimes and node
# depth ranges. Covers unsupported cases (zero estimate), compute-bound and memory-bound ops,
# fused-node input distances, and collective-communication estimates under a fake process
# group. Defines module-level aliases for aten and the c10d functional op namespaces, and the
# device constant set to the GPU type. Main block runs tests only if a GPU exists.

import contextlib
from unittest import skipIf

import torch
import torch.distributed as dist
from torch._inductor import config, metrics
from torch._inductor.comm_analysis import estimate_nccl_collective_runtime
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import is_collective
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


aten = torch.ops.aten
c10d = torch.ops.c10d_functional
_c10d = torch.ops._c10d_functional


def compile_but_use_eager(gm, example_inputs):
    # [FUNCTION DOC] HELPER: Backend for torch.compile taking a graph module and example
    # inputs. Runs the full Inductor compile with a custom inner-compile hook that compiles
    # (so scheduler metrics get populated) but hands back the original graph module, so
    # execution is eager. Returns whatever compile_fx returns.
    def inner_compile(gm, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner-compile hook passed to compile_fx; forwards all
        # arguments to the inner Inductor compile, ignores its output, and returns the input
        # graph module unchanged.
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def calculate_runtime(f, *args) -> float:
    # [FUNCTION DOC] HELPER: Resets metrics and enables inductor metrics logging, compiles and
    # runs the function on the given arguments with the eager-returning backend, prints the
    # recorded node runtimes, sums the runtime of each recorded node and returns the total as
    # a float. Restores default logging before returning. Returns 0.0 when no node runtime was
    # recorded.
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch._logging.set_logs(inductor_metrics=True)
    torch.compile(f, backend=compile_but_use_eager)(*args)
    print(metrics.node_runtimes)

    ret = 0.0
    for pair in metrics.node_runtimes:
        ret += pair[1]

    torch._logging.set_logs()
    return ret


DEVICE = GPU_TYPE


def T(*size, dtype=torch.float32, device=DEVICE, grad=False) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Creates a random normal tensor of the given size on the default
    # test device, with the given dtype (float32 by default) and requires-grad flag.
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


class TestCase(InductorTestCase):
    # [CLASS DOC] HELPER: Base class for the tests in this file, derived from the Inductor
    # test case. Sets the test device and provides exact-zero and non-zero assertions for
    # float runtime estimates. Its setUp enters a config patch disabling the remote FX graph
    # cache, since these tests read node runtimes that the cache does not save or restore.
    device = DEVICE

    """
    Helper methods to compare runtime estimate against 0. Since this estimate is hardware dependent,
    stronger comparisons may fail depending on the host's specs.

    atol/rtol must be provided explicitly with each call, since precision/rel_tol overrides are not always utilized
    """

    def setUp(self):
        # [FUNCTION DOC] Runs the parent setup, then creates an exit stack and enters a config
        # patch that turns off the FX graph remote cache for the duration of the test.
        super().setUp()
        # These tests check metrics.node_runtimes and we don't save / restore
        # those in the FX graph cache.
        self._test_snode_stack = contextlib.ExitStack()
        self._test_snode_stack.enter_context(
            config.patch({"fx_graph_remote_cache": False})
        )

    def tearDown(self):
        # [FUNCTION DOC] Closes the exit stack (undoing the config patch) and then runs the
        # parent teardown.
        self._test_snode_stack.close()
        super().tearDown()

    def assertZero(self, x: float):
        # [FUNCTION DOC] HELPER: Asserts the value is a float (raises AssertionError
        # otherwise) and equals 0.0 with zero absolute and relative tolerance.
        if not isinstance(x, float):
            raise AssertionError(f"Expected float, got {type(x)}")
        super().assertEqual(x, 0.0, atol=0, rtol=0)

    def assertNotZero(self, x):
        # [FUNCTION DOC] HELPER: Asserts the value is a float (raises AssertionError
        # otherwise) and differs from 0.0 with zero tolerances.
        if not isinstance(x, float):
            raise AssertionError(f"Expected float, got {type(x)}")
        super().assertNotEqual(x, 0.0, atol=0, rtol=0)


class UnsupportedTests(TestCase):
    # [CLASS DOC] Tests for cases where no runtime estimate is produced; the estimate must be
    # exactly zero.
    device = DEVICE

    def test_no_op(self):
        # [FUNCTION DOC] Checks that an identity function (no kernels) yields a zero runtime
        # estimate.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns its input unchanged.
            return a

        inp = (T(10, 10),)
        self.assertZero(calculate_runtime(f, *inp))

    def test_no_cuda(self):
        # [FUNCTION DOC] Checks that an identity function on a CPU tensor yields a zero
        # runtime estimate.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns its input unchanged.
            return a

        inp = (torch.randn((10, 10), device="cpu"),)
        self.assertZero(calculate_runtime(f, *inp))


class ComputeBoundedTests(TestCase):
    # [CLASS DOC] Tests that compute-bound operators (convolutions and matrix multiplies) get
    # a non-zero runtime estimate.
    device = DEVICE

    def test_conv1d(self):
        # [FUNCTION DOC] Checks a 1D convolution gets a non-zero runtime estimate.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies a 1D convolution of the first input with the
            # second as weight.
            return torch.nn.functional.conv1d(x, y)

        inp = (T(33, 16, 30), T(20, 16, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d(self):
        # [FUNCTION DOC] Checks a 2D convolution with padding 1 gets a non-zero runtime
        # estimate.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies a 2D convolution with padding 1.
            return torch.nn.functional.conv2d(x, y, padding=1)

        inp = (T(8, 4, 3, 3), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv2d_transpose(self):
        # [FUNCTION DOC] Checks a 2D transposed convolution with padding 1 gets a non-zero
        # estimate.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies a 2D transposed convolution with padding 1.
            return torch.nn.functional.conv_transpose2d(x, y, padding=1)

        inp = (T(8, 1, 1, 1), T(1, 4, 5, 5))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_conv3d(self):
        # [FUNCTION DOC] Checks a 3D convolution gets a non-zero runtime estimate.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Applies a 3D convolution of the first input with the
            # second as weight.
            return torch.nn.functional.conv3d(x, y)

        inp = (T(20, 16, 50, 10, 20), T(33, 16, 3, 3, 3))
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_mm(self):
        # [FUNCTION DOC] Checks a matrix multiply of two square matrices gets a non-zero
        # estimate.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the matrix product of its two inputs.
            return torch.mm(a, b)

        inp = (
            T(10, 10),
            T(10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_addmm(self):
        # [FUNCTION DOC] Checks an addmm (bias plus matrix product) of three square matrices
        # gets a non-zero estimate.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Returns the fused add-and-matrix-multiply of its three
            # inputs.
            return torch.addmm(a, b, c)

        inp = (
            T(10, 10),
            T(10, 10),
            T(10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_bmm(self):
        # [FUNCTION DOC] Checks a batched matrix multiply of two 3D tensors gets a non-zero
        # estimate.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the batched matrix product of its two inputs.
            return torch.bmm(a, b)

        inp = (
            T(10, 10, 10),
            T(10, 10, 10),
        )
        self.assertNotZero(calculate_runtime(f, *inp))


class MemoryBoundedTests(TestCase):
    # [CLASS DOC] Tests that memory-bound operators (pointwise and reductions) get a non-zero
    # runtime estimate.
    device = DEVICE

    def test_relu(self):
        # [FUNCTION DOC] Checks a relu on a square matrix gets a non-zero runtime estimate.
        def f(a):
            # [FUNCTION DOC] HELPER: Applies relu to its input.
            return torch.nn.functional.relu(a)

        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_horizontal_reduction_pointwise(self):
        # [FUNCTION DOC] Checks a function returning both a row-sum reduction and a cosine of
        # the same input gets a non-zero estimate.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns a pair of the sum over dimension 1 and the
            # elementwise cosine of the input.
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    def test_pointwise(self):
        # [FUNCTION DOC] Checks an elementwise cosine on a vector gets a non-zero estimate.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the elementwise cosine of the input.
            return x.cos()

        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_dynamic(self):
        # [FUNCTION DOC] Checks an elementwise cosine gets a non-zero estimate when the dynamo
        # config makes shapes dynamic (the config patch decorator disables assume-static-by-
        # default).
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the elementwise cosine of the input.
            return x.cos()

        inp = (T(10),)
        self.assertNotZero(calculate_runtime(f, *inp))


class InputDistanceTests(TestCase):
    # [CLASS DOC] Tests of the minimum and maximum input-distance attributes on scheduler
    # nodes, which measure depth from graph inputs through chains of non-fused kernels.
    device = DEVICE

    def _get_snodes(self, f, *args):
        # [FUNCTION DOC] HELPER: Resets metrics, compiles and runs the function on the
        # arguments using the eager-returning backend with inductor metrics logging enabled,
        # restores default logging, and returns the list of scheduler nodes recorded in the
        # nodes-num-elem metric.
        metrics.reset()
        torch._logging.set_logs(inductor_metrics=True)
        torch.compile(f, backend=compile_but_use_eager)(*args)
        torch._logging.set_logs()
        return [snode for snode, _ in metrics.nodes_num_elem]

    def test_chain_with_reduction(self):
        # [FUNCTION DOC] Compiles two chained last-dim sums and checks that across the
        # resulting scheduler nodes the smallest min input distance is 0 and the largest max
        # input distance is 1.
        """
        input -> sum (depth 0) -> sum (depth 1)
        Reductions prevent full fusion, giving us distinct depth levels.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Sums the last dimension twice.
            a = x.sum(dim=-1)
            return a.sum(dim=-1)

        snodes = self._get_snodes(f, T(10, 10, 10))
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertEqual(max(all_max), 1)

    def test_fused_node_depth_range(self):
        # [FUNCTION DOC] Compiles a reduction followed by a cosine, checks they fuse into
        # exactly one scheduler node, and that this node has min input distance 0 and max
        # input distance 1.
        """
        A reduction fused with its pointwise epilogue should have
        min_input_distance=0 and max_input_distance=1.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Sums the last dimension then applies cosine.
            a = x.sum(dim=-1)
            return a.cos()

        snodes = self._get_snodes(f, T(10, 10))
        # The reduction and pointwise get fused
        self.assertEqual(len(snodes), 1)
        self.assertEqual(snodes[0].min_input_distance, 0)
        self.assertEqual(snodes[0].max_input_distance, 1)

    def test_extern_kernel_chain(self):
        # [FUNCTION DOC] Compiles a matrix multiply (extern kernel) followed by cosine and a
        # last-dim sum, and checks the smallest min input distance across nodes is 0 and the
        # largest max input distance is 1.
        """
        mm (depth 0, extern) -> cos+sum fused (depth 1)
        """

        def f(a, b):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the inputs, applies cosine, then sums
            # the last dim.
            c = torch.mm(a, b)
            d = c.cos()
            return d.sum(dim=-1)

        snodes = self._get_snodes(f, T(10, 10), T(10, 10))
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertEqual(max(all_max), 1)

    def test_foreach_basic(self):
        # [FUNCTION DOC] Compiles a foreach add of two lists of graph inputs and checks every
        # resulting scheduler node has min and max input distance both 0.
        """
        foreach_add on graph inputs should have depth 0.
        """

        def f(xs, ys):
            # [FUNCTION DOC] HELPER: Returns the foreach elementwise add of two lists of
            # tensors.
            return torch._foreach_add(xs, ys)

        xs = [T(10), T(20)]
        ys = [T(10), T(20)]
        snodes = self._get_snodes(f, xs, ys)
        for s in snodes:
            self.assertEqual(s.min_input_distance, 0)
            self.assertEqual(s.max_input_distance, 0)

    def test_foreach_after_extern(self):
        # [FUNCTION DOC] Compiles a matrix multiply whose result feeds a foreach add, and
        # checks the smallest min input distance is 0 and the largest max input distance is at
        # least 1.
        """
        mm (extern, depth 0) -> foreach_add (depth 1)
        The extern kernel creates a fusion barrier so the foreach
        has a real dependency chain.
        """

        def f(a, b, ys):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the first two inputs and foreach-adds
            # the product twice, as a two-element list, to the third input list.
            c = torch.mm(a, b)
            return torch._foreach_add([c, c], ys)

        snodes = self._get_snodes(f, T(10, 10), T(10, 10), [T(10, 10), T(10, 10)])
        all_min = [s.min_input_distance for s in snodes]
        all_max = [s.max_input_distance for s in snodes]
        self.assertEqual(min(all_min), 0)
        self.assertGreaterEqual(max(all_max), 1)


@skipIf(not dist.is_available(), "requires distributed")
class TestCommAnalysis(TestCase):
    # [CLASS DOC] Tests of collective-communication runtime estimation, skipped when
    # torch.distributed is unavailable. Class attributes are a world size of 8 and the list of
    # ranks 0 through 7. Each test builds a function using a legacy or current functional
    # collective and verifies estimates through a shared helper.
    device = DEVICE

    WORLD_SIZE: int = 8
    RANKS = list(range(8))

    def _verify_runtime_estimation(self, fn, inps):
        # [FUNCTION DOC] HELPER: Initializes a fake-backend process group (rank 0, world size
        # 8, fake store), resets metrics, enables inductor metrics logging, compiles and runs
        # the function on the inputs. For every recorded scheduler node that is a collective,
        # calls the NCCL runtime estimator directly (so errors it would otherwise swallow
        # surface) and asserts both that estimate and the recorded runtime are non-zero.
        # Asserts at least one collective was found, restores logging, and always destroys the
        # process group.
        from torch.testing._internal.distributed.fake_pg import FakeStore

        store = FakeStore()
        dist.init_process_group(
            backend="fake", rank=0, world_size=self.WORLD_SIZE, store=store
        )
        try:
            metrics.reset()
            torch._logging.set_logs(inductor_metrics=True)
            torch.compile(fn)(*inps)
            found_collective = False
            for snode, runtime in metrics.node_runtimes:
                if not is_collective(snode.node):
                    continue
                found_collective = True
                # Inductor swallows errors from snode runtime estimations.
                # We call estimate_nccl_collective_runtime in a white-box
                # fashion here so potential issues can be surfaced in tests.
                est = estimate_nccl_collective_runtime(snode.node)
                self.assertNotZero(est)
                # Also make sure estimate_nccl_collective_runtime works
                # correctly in inductor.
                self.assertNotZero(runtime)
            # Make sure a collective kernel is found in graph
            self.assertTrue(found_collective)
            torch._logging.set_logs()
        finally:
            dist.destroy_process_group()

    def test_legacy_all_reduce(self):
        # [FUNCTION DOC] Verifies runtime estimation for the legacy all-reduce collective
        # followed by a wait.
        def fn(x):
            # [FUNCTION DOC] HELPER: Legacy all-reduce (sum) on the input over the class
            # ranks, then waits on the result.
            r = c10d.all_reduce(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return c10d.wait_tensor(r)

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_reduce_coalesced(self):
        # [FUNCTION DOC] Verifies runtime estimation for the legacy coalesced all-reduce on a
        # list of two tensors.
        def fn(x):
            # [FUNCTION DOC] HELPER: Legacy coalesced all-reduce (sum) of the input list,
            # returning the list of waited results.
            rs = c10d.all_reduce_coalesced(x, "sum", "", self.RANKS, self.WORLD_SIZE)
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_legacy_all_gather_into_tensor_coalesced(self):
        # [FUNCTION DOC] Verifies runtime estimation for the legacy coalesced all-gather into
        # tensor on a list of two tensors.
        def fn(x):
            # [FUNCTION DOC] HELPER: Legacy coalesced all-gather of the input list over the
            # class ranks, returning the list of waited results.
            rs = c10d.all_gather_into_tensor_coalesced(
                x,
                "",
                self.RANKS,
                self.WORLD_SIZE,
            )
            return [c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce(self):
        # [FUNCTION DOC] Verifies runtime estimation for the all-reduce collective followed by
        # a wait.
        def fn(x):
            # [FUNCTION DOC] HELPER: All-reduce (sum) on the input in group "0", then waits on
            # the result.
            r = _c10d.all_reduce(x, "sum", "0")
            return _c10d.wait_tensor(r)

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_reduce_coalesced(self):
        # [FUNCTION DOC] Verifies runtime estimation for the coalesced all-reduce on a list of
        # two tensors.
        def fn(x):
            # [FUNCTION DOC] HELPER: Coalesced all-reduce (sum) in group "0", returning the
            # list of waited results.
            rs = _c10d.all_reduce_coalesced(x, "sum", "0")
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor(self):
        # [FUNCTION DOC] Verifies runtime estimation for all-gather into tensor on a single
        # tensor.
        def fn(x):
            # [FUNCTION DOC] HELPER: All-gather into tensor with the class world size in group
            # "0", returning the list of waited results.
            rs = _c10d.all_gather_into_tensor(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(10, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_all_gather_into_tensor_coalesced(self):
        # [FUNCTION DOC] Verifies runtime estimation for the coalesced all-gather into tensor
        # on a list of two tensors.
        def fn(x):
            # [FUNCTION DOC] HELPER: Coalesced all-gather into tensor with the class world
            # size in group "0", returning the list of waited results.
            rs = _c10d.all_gather_into_tensor_coalesced(
                x,
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(10, 10), T(15, 15)]
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor(self):
        # [FUNCTION DOC] Verifies runtime estimation for reduce-scatter on a tensor whose
        # first dimension is the world size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reduce-scatter (sum) with the class world size in group
            # "0", returning the list of waited results.
            rs = _c10d.reduce_scatter_tensor(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = T(self.WORLD_SIZE, 10)
        self._verify_runtime_estimation(fn, (inp,))

    def test_reduce_scatter_tensor_coalesced(self):
        # [FUNCTION DOC] Verifies runtime estimation for coalesced reduce-scatter on a list of
        # two tensors whose first dimension is the world size.
        def fn(x):
            # [FUNCTION DOC] HELPER: Coalesced reduce-scatter (sum) with the class world size
            # in group "0", returning the list of waited results.
            rs = _c10d.reduce_scatter_tensor_coalesced(
                x,
                "sum",
                self.WORLD_SIZE,
                "0",
            )
            return [_c10d.wait_tensor(r) for r in rs]

        inp = [T(self.WORLD_SIZE, 10), T(self.WORLD_SIZE, 15)]
        self._verify_runtime_estimation(fn, (inp,))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU:
        run_tests(needs="filelock")
