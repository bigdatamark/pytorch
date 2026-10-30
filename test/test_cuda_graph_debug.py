# Owner(s): ["module: cuda"]
# ruff: noqa: F821, F841
# [MODULE DOC] Tests for the input-liveness debugging option of CUDA graph capture. A graph
# captured with the liveness check enabled tracks external input tensors and, on replay, raises
# a runtime error if any of their storages died. Covers plain, view, nested, pinned, resized and
# model-weight inputs, shared pools, streams, error tracebacks, and custom ops. When CUDA is
# unavailable the module prints a notice to stderr and replaces the test base class with a
# no-op class so nothing runs.

import sys
import warnings

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import (
    NoTest,
    run_tests,
    skipIfRocm,
    TEST_CUDA,
    TestCase,
)


if not TEST_CUDA:
    print("CUDA not available, skipping tests", file=sys.stderr)
    TestCase = NoTest


def _warmup_op(op, n=3):
    # [FUNCTION DOC] HELPER: Run the given callable the given number of times (default three) on
    # a fresh side stream, to warm up kernels before capture. Returns nothing.
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(n):
            op()


def _capture_graph(op, **graph_kwargs):
    # [FUNCTION DOC] HELPER: Warm up the callable, then capture it into a new CUDA graph with
    # the input liveness check on by default (extra keyword arguments pass through to the graph
    # context manager and may override it). Returns the graph and the callable's output.
    g = torch.cuda.CUDAGraph()
    _warmup_op(op)
    graph_kwargs.setdefault("check_input_liveness", True)
    with torch.cuda.graph(g, **graph_kwargs):
        out = op()
    return g, out


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugInputs(TestCase):
    # [CLASS DOC] Tests that replaying a graph captured with the liveness check raises when an
    # external input tensor (or its storage) has died, and does not raise for intermediates or
    # inputs that are still alive. Skipped on ROCm.
    def test_nothing_dead_ok(self):
        # [FUNCTION DOC] When all captured inputs are still alive, replay succeeds and the
        # captured output equals the expected doubled input.
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        g.replay()
        self.assertEqual(y, x * 2)

    def test_dead_input_tensor(self):
        # [FUNCTION DOC] Deleting the only input tensor makes replay raise a runtime error that
        # mentions a dead tensor.
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        del x
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_dead_intermediate_tensors_ok(self):
        # [FUNCTION DOC] Tensors created inside the capture (an intermediate and the output) are
        # not tracked as inputs: replay works after deleting the intermediate and again after
        # deleting the output.
        x = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            tmp = x * 2
            y = tmp + 1
        del tmp
        g.replay()
        self.assertEqual(y, x * 2 + 1)
        del y
        g.replay()

    def test_model_weight_dead(self):
        # [FUNCTION DOC] Captures a small multilayer model on the GPU, replays it successfully and
        # checks the output matches eager, then deletes one layer's weight and expects replay to
        # raise a dead tensor error.
        class SimpleMLP(nn.Module):
            # [CLASS DOC] HELPER: Tiny feed-forward network holding a list of linear layers of
            # equal width; used as the captured model.
            def __init__(self, width, depth):
                # [FUNCTION DOC] HELPER: Build the given number of linear layers of the given
                # width and store them in a module list.
                super().__init__()
                self.layers = nn.ModuleList(
                    [nn.Linear(width, width) for _ in range(depth)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply each linear layer followed by a relu and return the
                # result.
                for layer in self.layers:
                    x = torch.relu(layer(x))
                return x

        model = SimpleMLP(width=64, depth=5).cuda()
        x = torch.randn(8, 64, device="cuda")
        g = torch.cuda.CUDAGraph()
        _warmup_op(lambda: model(x))
        with torch.no_grad(), torch.cuda.graph(g, check_input_liveness=True):
            y = model(x)
        g.replay()
        self.assertEqual(y, model(x))
        del model.layers[2].weight
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_model_training_weight_dead(self):
        # [FUNCTION DOC] Captures forward and backward of a linear layer into a graph with the
        # liveness check, replays it, then deletes the output and the weight parameter; replay is
        # expected to raise a dead tensor error. Backward warmup runs with user warnings ignored.
        model = nn.Linear(32, 16).cuda()
        x = torch.randn(8, 32, device="cuda")
        static_grad = torch.randn(8, 16, device="cuda")
        with warnings.catch_warnings():
            # First cuBLAS backward triggers "no current CUDA context" warning
            warnings.simplefilter("ignore", UserWarning)
            for _ in range(3):
                model(x).backward(static_grad)
        g = torch.cuda.CUDAGraph()
        model.zero_grad()
        with torch.cuda.graph(g, check_input_liveness=True):
            out = model(x)
            out.backward(static_grad)
        g.replay()
        # out's grad_fn retains weight as a saved tensor from forward
        del out
        del model.weight
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_view_tensor(self):
        # [FUNCTION DOC] Captured input is a slice view of a base tensor. Deleting the view alone
        # keeps replay working because the storage lives; deleting the base then makes replay
        # raise.
        base = torch.randn(100, device="cuda")
        view = base[10:50]
        g, y = _capture_graph(lambda: view * 2)
        del view
        g.replay()
        del base
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_nested_structure_inputs(self):
        # [FUNCTION DOC] Inputs are two lists of tensors passed to a foreach add; deleting one
        # tensor from the second list makes replay raise a dead tensor error.
        xs = [torch.randn(10, device="cuda") for _ in range(3)]
        ys = [torch.randn(10, device="cuda") for _ in range(3)]
        g, result = _capture_graph(lambda: torch._foreach_add(xs, ys))
        del ys[1]
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_empty_tensor_not_tracked(self):
        # [FUNCTION DOC] A zero-element tensor has a null data pointer, so it is not registered
        # among the graph tracker's external inputs, and deleting it does not break replay.
        x = torch.randn(100, device="cuda")
        empty = torch.empty(0, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2
            z = empty.view(0)
        self.assertEqual(empty.data_ptr(), 0)
        self.assertNotIn(empty.data_ptr(), g._tracker._external_inputs)
        del empty
        g.replay()

    def test_non_capturing_stream_inputs_not_tracked(self):
        # [FUNCTION DOC] Captures under one stream while also using a second stream inside the
        # capture in relaxed error mode. Only the input used on the capturing stream is tracked;
        # the input used on the other stream is not, so deleting it does not break replay.
        capture_stream = torch.cuda.Stream()
        other_stream = torch.cuda.Stream()
        x = torch.randn(100, device="cuda")
        y = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(capture_stream):
            with torch.cuda.graph(
                g, capture_error_mode="relaxed", check_input_liveness=True
            ):
                result = x * 2
                with torch.cuda.stream(other_stream):
                    other_result = y * 3
        self.assertIn(x.data_ptr(), g._tracker._external_inputs)
        self.assertNotIn(y.data_ptr(), g._tracker._external_inputs)
        del y
        g.replay()
        self.assertEqual(result, x * 2)

    def test_shared_pool_tensor_not_tracked(self):
        # [FUNCTION DOC] Two graphs share a memory pool. Tensors allocated by the first capture
        # and consumed by the second are not treated as external inputs, so deleting them (output
        # and the other pool-allocated tensor) does not break replays; the second graph result is
        # checked.
        pool = torch.cuda.graph_pool_handle()
        x = torch.randn(100, device="cuda")
        g1 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g1, pool=pool):
            y = x + 1
            w = torch.randn(100, device="cuda")
        external_input = torch.randn(100, device="cuda")
        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2, pool=pool, check_input_liveness=True):
            z = y * 2 + external_input
            p = w + 1
        del y
        g1.replay()
        g2.replay()
        self.assertEqual(z, (x + 1) * 2 + external_input)
        del w
        g1.replay()
        g2.replay()

    def test_alive_storage_dead_tensor_ok(self):
        # [FUNCTION DOC] Two tensors share one storage. Deleting the captured tensor keeps replay
        # working since the storage is alive through the other tensor; deleting the second tensor
        # then makes replay raise.
        a = torch.randn(100, device="cuda")
        b = torch.empty(100, device="cuda")
        b.set_(a.untyped_storage())
        g, y = _capture_graph(lambda: a * 2)
        del a
        g.replay()
        self.assertEqual(y, b * 2)
        del b
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_dead_storage_raises(self):
        # [FUNCTION DOC] Replacing the storage of the captured input tensor with a new one frees
        # the original storage, so replay raises a dead tensor error.
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        x.set_(torch.randn(100, device="cuda").untyped_storage())
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_resize_larger_detected(self):
        # [FUNCTION DOC] Resizing the captured input to a much larger size in place (which
        # reallocates storage) is detected, and replay raises a dead tensor error.
        x = torch.randn(100, device="cuda")
        g, y = _capture_graph(lambda: x * 2)
        x.resize_(100000)
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_pinned_memory_tensor_tracked(self):
        # [FUNCTION DOC] A pinned host tensor copied into a GPU buffer during capture is tracked
        # along with the GPU buffer as external inputs. Replay works while the pinned tensor lives
        # and raises once it is deleted.
        pinned = torch.randn(100).pin_memory()
        cuda_dest = torch.empty(100, device="cuda")
        g = torch.cuda.CUDAGraph()

        def op():
            # [FUNCTION DOC] HELPER: Unused capture body that copies the pinned tensor into the
            # GPU buffer asynchronously and returns the doubled buffer; the test instead repeats
            # these steps inline inside the capture context.
            cuda_dest.copy_(pinned, non_blocking=True)
            return cuda_dest * 2

        with torch.cuda.graph(g, check_input_liveness=True):
            cuda_dest.copy_(pinned, non_blocking=True)
            y = cuda_dest * 2
        self.assertIn(pinned.data_ptr(), g._tracker._external_inputs)
        self.assertIn(cuda_dest.data_ptr(), g._tracker._external_inputs)
        # This should be OK
        g.replay()
        del pinned
        # This should raise
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_recapture_without_clears_stale_state(self):
        # [FUNCTION DOC] After a failed replay from a dead input, resetting the graph and
        # capturing again with fresh input gives a graph whose replay succeeds, so stale tracking
        # state is cleared by the reset.
        g = torch.cuda.CUDAGraph()
        x = torch.randn(32, device="cuda")
        with torch.cuda.graph(g, check_input_liveness=True):
            y = x * 2
        del x
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()
        g.reset()
        x2 = torch.randn(32, device="cuda")
        with torch.cuda.graph(g, check_input_liveness=True):
            y2 = x2 * 3
        g.replay()

    def test_higher_order_op_during_capture(self):
        # [FUNCTION DOC] Captures a call through the higher-order wrap operator that doubles its
        # input, and checks replay output equals the expected doubled input.
        from torch._higher_order_ops.wrap import wrap

        def inner_fn(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by two; the body run through the
            # higher-order wrap operator.
            return x * 2

        x = torch.randn(32, device="cuda")
        g, y = _capture_graph(lambda: wrap(inner_fn, x))
        g.replay()
        self.assertEqual(y, x * 2)


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugBacktraces(TestCase):
    # [CLASS DOC] Tests the content of the error raised when a dead input is found at replay.
    # Skipped on ROCm.
    def test_error_message_contains_all_tracebacks(self):
        # [FUNCTION DOC] Deletes two captured inputs from a nested helper, replays, and checks the
        # lowercased error message reports two dead tensors and includes Python and C++ creation
        # tracebacks, a Python deletion traceback naming the helper, and a replay traceback.
        a = torch.randn(100, device="cuda")
        b = torch.randn(100, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, check_input_liveness=True):
            y = a + b

        def delete_tensors():
            # [FUNCTION DOC] HELPER: Delete the two enclosing-scope input tensors so the deletion
            # traceback will name this function.
            nonlocal a, b
            del a, b

        delete_tensors()
        with self.assertRaises(RuntimeError) as ctx:
            g.replay()
        error_msg = str(ctx.exception).lower()
        self.assertIn("2 dead tensor", error_msg)
        self.assertIn("creation traceback (python)", error_msg)
        self.assertIn("creation traceback (c++)", error_msg)
        self.assertIn("deletion traceback (python)", error_msg)
        self.assertIn("delete_tensors", error_msg)
        self.assertIn("replay traceback (python)", error_msg)


@skipIfRocm(msg="input liveness checking is only supported on NVIDIA CUDA")
class TestCUDAGraphDebugExternalOps(TestCase):
    # [CLASS DOC] Tests liveness tracking when inputs reach the graph through external operator
    # mechanisms (custom autograd function and a registered custom operator). Skipped on ROCm.
    def test_custom_autograd_function(self):
        # [FUNCTION DOC] A custom autograd function saves its scale tensor argument during capture
        # under no-grad. Deleting the scale tensor makes replay raise a dead tensor error.
        from torch.autograd import Function

        class ScaleOp(Function):
            # [CLASS DOC] HELPER: Autograd function multiplying an input by a scale tensor, saving
            # the scale for backward.
            @staticmethod
            def forward(ctx, x, scale):
                # [FUNCTION DOC] HELPER: Save the scale tensor on the context and return the input
                # times the scale.
                ctx.save_for_backward(scale)
                return x * scale

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Return the gradient times the saved scale for the input
                # and no gradient for the scale.
                (scale,) = ctx.saved_tensors
                return grad * scale, None

        x = torch.randn(100, device="cuda")
        scale = torch.tensor(2.0, device="cuda")
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g, check_input_liveness=True):
            y = ScaleOp.apply(x, scale)
        del scale
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()

    def test_torch_library_custom_op(self):
        # [FUNCTION DOC] Registers a custom operator that multiplies a tensor by a scale tensor,
        # captures a call to it, deletes the scale tensor, and expects replay to raise a dead
        # tensor error.
        @torch.library.custom_op("test_cuda_graph_debug::scale", mutates_args=())
        def scale_op(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom operator body returning the input times the scale
            # tensor.
            return x * scale

        @scale_op.register_fake
        def _(x, scale):
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor shaped like the
            # input, for tracing.
            return torch.empty_like(x)

        x = torch.randn(100, device="cuda")
        scale = torch.tensor(3.0, device="cuda")
        g, y = _capture_graph(lambda: scale_op(x, scale))
        del scale
        with self.assertRaisesRegex(RuntimeError, "dead.*tensor"):
            g.replay()


if __name__ == "__main__":
    run_tests()
