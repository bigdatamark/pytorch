# Owner(s): ["module: cuda graphs"]
# [MODULE DOC] Dynamo tests for the "cudagraphs" torch.compile backend (CUDA only).
# Defines decorator helpers (composed, assert_aot_autograd_counter, patch_all) and the
# TestAotCudagraphs class, which compiles small models and checks AOT autograd counters,
# input/constant mutation, metadata mutation, cudagraph-tree warnings, and the error
# raised when compiling during CUDA graph capture. Importing has no side effects beyond
# imports; running as a script exits quietly when CUDA graph tests are unsupported.

import functools
import unittest

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._inductor.cudagraph_trees as cudagraph_trees
from torch._dynamo.testing import same
from torch.testing._internal.common_utils import TEST_CUDA_GRAPH


def composed(*decs):
    # [FUNCTION DOC] HELPER: Builds a decorator that applies all given decorators to a
    # function, with the first one given ending up outermost (applied last).
    def deco(f):
        # [FUNCTION DOC] HELPER: Applies the captured decorators in reverse order to f
        # and returns the final wrapped function.
        for dec in reversed(decs):
            f = dec(f)
        return f

    return deco


def assert_aot_autograd_counter(ok=True):
    # [FUNCTION DOC] HELPER: Returns a test-method decorator. After the wrapped test runs,
    # it asserts the aot_autograd counters: if ok is true, at least one success and no
    # failures; otherwise no successes and at least one failure.
    def deco(f):
        # [FUNCTION DOC] HELPER: Wraps test method f with the counter assertions.
        @functools.wraps(f)
        def wrap(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Clears the global dynamo counters, runs f, checks the
            # aot_autograd ok / not_ok counters as configured, and returns f's result.
            torch._dynamo.utils.counters.clear()
            r = f(self, *args, **kwargs)
            c_ok = torch._dynamo.utils.counters["aot_autograd"]["ok"]
            c_not_ok = torch._dynamo.utils.counters["aot_autograd"]["not_ok"]
            if ok:
                self.assertGreater(c_ok, 0)
                self.assertEqual(c_not_ok, 0)
            else:
                self.assertEqual(c_ok, 0)
                self.assertGreater(c_not_ok, 0)
            return r

        return wrap

    return deco


def patch_all(ok=True):
    # [FUNCTION DOC] HELPER: Returns a decorator combining a dynamo config patch
    # (verify_correctness and automatic_dynamic_shapes enabled) with the aot_autograd
    # counter assertion configured by ok.
    return composed(
        torch._dynamo.config.patch(
            verify_correctness=True, automatic_dynamic_shapes=True
        ),
        assert_aot_autograd_counter(ok),
    )


N_ITERS = 5


@unittest.skipIf(not torch.cuda.is_available(), "these tests require cuda")
class TestAotCudagraphs(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case (skipped without CUDA) exercising the cudagraphs backend of
    # torch.compile on small CUDA models. Each test compiles a loop function that runs a
    # model N_ITERS times; most are decorated to verify correctness and AOT autograd counters.
    @patch_all()
    def test_basic(self):
        # [FUNCTION DOC] Compiles a loop running forward and backward of a simple
        # elementwise model on CUDA tensors; passes if no error and aot_autograd succeeds.
        def model(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of x and y multiplied by y.
            return (x + y) * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Repeats N_ITERS times: runs the model, sums, backward.
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_dtoh(self):
        # [FUNCTION DOC] Same compiled forward/backward loop as the basic test, but the
        # model copies an intermediate to CPU (device-to-host) before scaling it.
        def model(x, y):
            # [FUNCTION DOC] HELPER: Adds x and y on CUDA, moves the sum to CPU, times 3.
            a = x + y
            b = a.cpu() * 3
            return b

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Repeats N_ITERS times: runs the model, sums, backward.
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_htod(self):
        # [FUNCTION DOC] Compiled forward/backward loop where y is a zero-dim CPU tensor
        # combined with a CUDA tensor (host-to-device scalar input).
        def model(x, y):
            # [FUNCTION DOC] HELPER: Returns (x + y) times 3.
            a = x + y
            return a * 3

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Repeats N_ITERS times: runs the model, sums, backward.
            for _ in range(N_ITERS):
                loss = model(x, y).sum()
                loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn((), device="cpu")
        fn(x, y)

    def test_mutate_input(self):
        # [FUNCTION DOC] Checks that an in-place mutation of an input inside the compiled
        # region is visible on the caller's tensor each iteration (y grows by 3). Not wrapped
        # by the aot_autograd counter decorator.
        def model(x, y):
            # [FUNCTION DOC] HELPER: Adds 3 to y in place and returns x times y.
            y.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Per iteration (as a subtest), runs the model and asserts
            # y equals its previous value plus 3, then runs backward.
            for i in range(N_ITERS):
                with self.subTest(i):
                    y_orig = y.clone()
                    loss = model(x, y).sum()
                    self.assertTrue(same(y, y_orig + 3))
                    loss.backward()

        x = torch.randn(3, device="cuda", requires_grad=True)
        y = torch.randn(3, device="cuda")
        fn(x, y)

    @patch_all()
    def test_mutate_constant(self):
        # [FUNCTION DOC] Checks that mutating a tensor constant created inside the model
        # does not leak across iterations: the loss must be 3.0 every time.
        def model(x, y):
            # [FUNCTION DOC] HELPER: Creates a CPU constant 1, adds 2 in place, and returns
            # it added to a zeroed product of x and y.
            c = torch.tensor(1)
            c.add_(2)
            return x * y * 0 + c

        @torch.compile(backend="cudagraphs")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Per iteration (as a subtest), asserts the summed loss
            # equals 3.0 on CUDA, then runs backward.
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(x, y).sum()
                    self.assertTrue(same(loss, torch.tensor(3.0, device="cuda")))
                    loss.backward()

        x = torch.randn(1, device="cuda", requires_grad=True)
        y = torch.randn(1, device="cuda")
        fn(x, y)

    @patch_all()
    def test_factory(self):
        # [FUNCTION DOC] Compiles a model that builds a fresh CUDA tensor via a factory
        # function and mutates it in place; runs forward and backward repeatedly.
        def model(y):
            # [FUNCTION DOC] HELPER: Makes zeros on cuda:0, adds 3 in place, multiplies by y.
            x = torch.zeros(3, device="cuda:0")
            x.add_(3)
            return x * y

        @torch.compile(backend="cudagraphs")
        def fn(y):
            # [FUNCTION DOC] HELPER: Per iteration (as a subtest), runs the model, sums,
            # and calls backward.
            for i in range(N_ITERS):
                with self.subTest(i):
                    loss = model(y).sum()
                    loss.backward()

        y = torch.randn(3, device="cuda:0", requires_grad=True)
        fn(y)

    @patch_all()
    def test_mutated_metadata(self):
        # [FUNCTION DOC] Checks that resizing a cloned tensor inside the compiled model
        # (metadata mutation) yields a size-20 tensor of 2.0 on every iteration, starting
        # from an empty input.
        # more tortured example at
        # https://github.com/pytorch/pytorch/issues/81385
        def model(x):
            # [FUNCTION DOC] HELPER: Clones x, resizes the clone to 20 elements, fills it
            # with 2, and returns it.
            x = x.clone()
            x.resize_(20)
            x.fill_(2)
            return x

        @torch.compile(backend="cudagraphs")
        def fn(x):
            # [FUNCTION DOC] HELPER: Per iteration (as a subtest), runs the model and
            # asserts the result is a 20-element tensor of 2.0 on cuda:0.
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))

        x = torch.empty(0, device="cuda:0")
        fn(x)

    @patch_all()
    def test_dead_fill(self):
        # [FUNCTION DOC] Checks that filling an empty view of a cloned tensor has no effect
        # on the parent: the full tensor stays 2.0 and the empty view stays empty.
        def model(x):
            # [FUNCTION DOC] HELPER: Clones x, takes an empty slice, fills the clone with 2
            # and the empty slice with 3, and returns both.
            x = x.clone()
            y = x[0:0]
            x.fill_(2)
            y.fill_(3)
            return x, y

        @torch.compile(backend="cudagraphs")
        def fn(x):
            # [FUNCTION DOC] HELPER: Per iteration (as a subtest), runs the model and
            # asserts the first output is 20 elements of 2.0 and the second is empty.
            for i in range(N_ITERS):
                with self.subTest(i):
                    rx, ry = model(x)
                    self.assertTrue(same(rx, torch.full((20,), 2.0, device="cuda:0")))
                    self.assertTrue(same(ry, torch.empty(0, device="cuda:0")))

        x = torch.empty(20, device="cuda:0")
        fn(x)

    @unittest.skipIf(not TEST_CUDA_GRAPH, "cuda graph test is skipped")
    def test_inference_mode_does_not_need_mark_step_begin(self):
        # [FUNCTION DOC] Resets cudagraph trees (and again at cleanup), compiles a function
        # that runs sin under inference mode with the cudagraphs backend, calls it 4 times
        # comparing against eager, then asserts the device's cudagraph manager recorded no
        # warned functions (no repeated-capture warning).
        # This tests that torch.inference_mode() removes the need for
        # the user to call torch.compiler.mark_step_begin()

        # reset to make sure that warned_functions is empty
        cudagraph_trees.reset_cudagraph_trees()
        self.addCleanup(cudagraph_trees.reset_cudagraph_trees)

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns sin of x computed inside inference mode.
            with torch.inference_mode():
                return x.sin()

        x = torch.randn(4, device="cuda", requires_grad=False)

        compiled_foo = torch.compile(foo, backend="cudagraphs")

        # First call runs eager, second captures and replays, rest
        # should just replay.
        for _ in range(4):
            out = compiled_foo(x)
            self.assertTrue(same(out, foo(x)))

        manager = cudagraph_trees.get_manager(x.device.index)
        self.assertTrue(
            len(manager.warned_functions) == 0,
            "Replaying inference workload should not warn about repeated graph captures",
        )

    def test_compile_during_capture_errors(self):
        # [FUNCTION DOC] Checks that first-time compilation of an eager-backend function
        # inside a CUDA graph capture raises TorchRuntimeError mentioning that JIT compile
        # is not allowed during capture, and that the function still works afterwards.
        # https://github.com/pytorch/pytorch/issues/185074
        from torch._dynamo.exc import TorchRuntimeError

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        comp_fn = torch.compile(backend="eager")(fn)

        x = torch.randn(10, device="cuda")

        # Should raise TorchRuntimeError when trying to compile during capture
        with self.assertRaisesRegex(
            TorchRuntimeError,
            r"cannot JIT compile during CUDA graph capture",
        ):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                comp_fn(x)

        self.assertEqual(comp_fn(x), fn(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            import sys

            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    run_tests()
