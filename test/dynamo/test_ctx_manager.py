# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for how torch.compile (Dynamo) traces and handles context managers: grad mode
# (no_grad, enable_grad, set_grad_enabled), autocast, profilers, saved-tensor hooks, SDPA kernel
# selection, functorch transforms, user-defined context managers, plus CUDA stream/event/mem-pool
# and device-generic stream tests. Defines small fixture context managers at import time, a module-
# level device_type derived from the current accelerator (else cpu), and global counters; imports
# test_functions (relative or absolute). At the bottom, parametrized tests are instantiated for two
# classes and the device-generic class is instantiated for non-CPU devices (XPU allowed); running
# the file calls Dynamo's run_tests.
# Many tests compare eager against compiled results, count frames/ops with CompileCounter, or check
# expected FX graph text.
import contextlib
import sys
import unittest
from collections import defaultdict
from contextlib import contextmanager

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import (
    check_dynamic_shape_capture,
    EagerAndRecordGraphs,
    normalize_gm,
    same,
)
from torch._dynamo.utils import counters
from torch.nn import functional as F
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCUDA,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.triton_utils import requires_cuda_and_triton


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"

try:
    from . import test_functions
except ImportError:
    import test_functions


_variable = 0
_variable1 = 0
z_glb = 0
k_glb = 0


@contextlib.contextmanager
def set_default_dtype(dtype):
    # [FUNCTION DOC] Temporarily sets the global default dtype for the duration of a with block
    # (contextmanager). Saves the old default, sets the new one, yields, and always restores the
    # old dtype in a finally clause. Side effect: mutates global torch default dtype.
    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(old_dtype)


class CustomizedCtxManager:
    # [CLASS DOC] Plain class-based context manager used as a fixture: on enter it sets the global
    # grad mode to the mode given at construction, on exit it restores the grad mode that was
    # active at construction time. Attributes: prev (grad mode at construction), mode.
    def __init__(self, mode):
        # [FUNCTION DOC] Records the grad mode currently enabled as prev and stores
        # the requested mode.
        self.prev = torch.is_grad_enabled()
        self.mode = mode

    def __enter__(self):
        # [FUNCTION DOC] Enter hook: sets the global grad-enabled flag to the stored mode via the
        # private C API; returns None.
        torch._C._set_grad_enabled(self.mode)

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] Exit hook: restores the grad-enabled flag saved at construction (prev);
        # exception arguments are ignored and nothing is suppressed.
        torch._C._set_grad_enabled(self.prev)


@contextlib.contextmanager
def customized_ctx_manager(mode):
    # [FUNCTION DOC] Generator-based (contextlib) equivalent of CustomizedCtxManager: saves the
    # current grad mode, sets the grad flag to mode and yields the result of the setter (None), and
    # restores the saved mode in a finally clause.
    prev = torch.is_grad_enabled()
    try:
        yield torch._C._set_grad_enabled(mode)
    finally:
        torch._C._set_grad_enabled(prev)


class CustomizedCtxManagerWithGraphBreak(CustomizedCtxManager):
    # [CLASS DOC] Subclass of CustomizedCtxManager whose enter hook first forces a Dynamo graph
    # break, then performs the normal grad-mode switch. Used to test graph breaks inside __enter__.
    def __enter__(self):
        # [FUNCTION DOC] Enter hook: triggers a graph break via torch._dynamo.graph_break, then
        # calls the parent enter to set the grad mode.
        torch._dynamo.graph_break()
        super().__enter__()


@contextlib.contextmanager
def customized_ctx_manager_with_graph_break(mode):
    # [FUNCTION DOC] Generator-based context manager like customized_ctx_manager, but it triggers a
    # Dynamo graph break before setting the grad mode and yielding. Restores the previous grad mode
    # in a finally clause.
    prev = torch.is_grad_enabled()
    try:
        torch._dynamo.graph_break()
        yield torch._C._set_grad_enabled(mode)
    finally:
        torch._C._set_grad_enabled(prev)


class CtxManagerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test suite for Dynamo's handling of context managers (grad mode, autocast,
    # profilers, saved tensor hooks, SDPA kernels, functorch levels and user-defined managers),
    # most running on CPU with eager or counting backends.
    def test_no_grad(self):
        # [FUNCTION DOC] Checks that Dynamo traces redundant grad-mode switches as no-ops: four
        # functions (no_grad, set_grad_enabled False, enable_grad, set_grad_enabled True with a
        # branch) are run under an ambient no_grad and an ambient enable_grad via the standard_test
        # helper, which compares results and asserts the expected op count (3 when the switch is
        # redundant and coalesced, 5 when a real mode toggle is recorded).
        def fn1(a, b):
            # [FUNCTION DOC] HELPER: Adds one and then adds b inside a redundant no_grad block (a
            # comment says redundant blocks are ignored), then adds two; returns the result.
            x = a + 1
            # redundant no_grad should get ignored
            with torch.no_grad():
                x = x + b
            x = x + 2
            return x

        def fn2(a, b):
            # [FUNCTION DOC] HELPER: Same arithmetic as fn1 but uses set_grad_enabled with False
            # for the middle addition.
            x = a + 1
            with torch.set_grad_enabled(False):
                x = x + b
            x = x + 2
            return x

        def fn3(a, b):
            # [FUNCTION DOC] HELPER: Same arithmetic as fn1 but wraps the middle
            # addition in enable_grad.
            x = a + 1
            with torch.enable_grad():
                x = x + b
            x = x + 2
            return x

        def fn4(a, b):
            # [FUNCTION DOC] HELPER: Same arithmetic, using set_grad_enabled with True and an inner
            # branch on is_grad_enabled before the middle addition.
            x = a + 1
            with torch.set_grad_enabled(True):
                if torch.is_grad_enabled():
                    x = x + b
            x = x + 2
            return x

        with torch.no_grad():
            torch._dynamo.testing.standard_test(
                self, fn=fn1, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn2, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(self, fn=fn3, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn4, nargs=2, expected_ops=5)
        with torch.enable_grad():
            torch._dynamo.testing.standard_test(self, fn=fn1, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(self, fn=fn2, nargs=2, expected_ops=5)
            torch._dynamo.testing.standard_test(
                self, fn=fn3, nargs=2, expected_ops=3
            )  # coalesced noop
            torch._dynamo.testing.standard_test(
                self, fn=fn4, nargs=2, expected_ops=3
            )  # coalesced noop

    def test_grad_mode_guard(self):
        # [FUNCTION DOC] Verifies that a function which flips grad mode globally with
        # set_grad_enabled(False), forces a graph break with tolist, then restores the previous
        # mode compiles into exactly two frames and reuses the cache across ten calls
        # (frame_count stays 2).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Saves the current grad mode, disables grad, adds one,
            # calls tolist to cause a graph break, adds b, restores the saved grad mode and
            # returns the sum.
            prev_grad = torch.is_grad_enabled()
            torch.set_grad_enabled(False)
            a = a + 1
            a.tolist()  # graph break
            ret = a + b
            torch.set_grad_enabled(prev_grad)
            return ret

        a = torch.randn([3, 4])
        b = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for _ in range(10):
            opt_fn(a, b)
        self.assertEqual(cnts.frame_count, 2)

    def test_nested_grad_mode_graph_break(self):
        # [FUNCTION DOC] Checks nested set_grad_enabled blocks (False then True) with graph breaks
        # inside; the asserts inside the function confirm grad mode is correct after each break and
        # restored on exit, and ten calls produce exactly two compiled frames.
        def fn(x):
            # [FUNCTION DOC] HELPER: Runs nested grad-mode blocks with graph breaks, asserting the
            # grad flag at each level and that the original mode is restored; returns sqrt(x*5).
            before = torch.is_grad_enabled()
            with torch.set_grad_enabled(False):
                torch._dynamo.graph_break()
                with torch.set_grad_enabled(True):
                    x = torch.mul(x, 5)
                    torch._dynamo.graph_break()
                    x = torch.sqrt(x)
                    assert torch.is_grad_enabled()  # noqa: S101
                assert not torch.is_grad_enabled()  # noqa: S101
            assert torch.is_grad_enabled() == before  # noqa: S101
            return x

        a = torch.randn([3, 4])
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        for _ in range(10):
            opt_fn(a)
        self.assertEqual(cnts.frame_count, 2)

    def test_torch_profiler(self):
        # [FUNCTION DOC] Checks torch.profiler.profile and record_function are treated as no-op
        # (null) context managers: compiled result matches eager despite a graph break (tolist)
        # inside, and exactly two frames are compiled.
        # wrap torch.profiler.* as NullContextVariable and do nothing
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares x, enters a profiler and a record_function block,
            # cubes, triggers a graph break with tolist, adds one, and returns.
            y = x**2
            with torch.profiler.profile():
                y = y + 2
                with torch.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler(self):
        # [FUNCTION DOC] Same as test_torch_profiler but with the torch.autograd.profiler.profile
        # and record_function context managers; asserts equal results and two compiled frames.
        # wrap torch.autograd.profiler.* as NullContextVariable and do nothing
        def fn(x):
            # [FUNCTION DOC] HELPER: Same body as the profiler test function: square, enter the
            # autograd profiler, cube, graph break, add one.
            y = x**2
            with torch.autograd.profiler.profile():
                y = y + 2
                with torch.autograd.profiler.record_function("my_function"):
                    z = y**3
                    z.tolist()  # graph break
                    z = z + 1
            return z

        x = torch.randn((2, 2), requires_grad=True)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))
        self.assertEqual(cnts.frame_count, 2)

    def test_autograd_profiler_enabled(self):
        # [FUNCTION DOC] Checks that torch.autograd._profiler_enabled is traced correctly: after
        # making sure the profiler is disabled, eager and compiled results agree; then inside an
        # active autograd profiler they agree again. Raises AssertionError if the profiler state is
        # not as expected. Side effect: may disable a running profiler.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one when the autograd profiler is enabled,
            # otherwise x minus one.
            if torch.autograd._profiler_enabled():
                return x + 1
            else:
                return x - 1

        x = torch.randn((2, 2), requires_grad=True)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        if torch.autograd._profiler_enabled():
            torch.autograd._disable_profiler()
        if torch.autograd._profiler_enabled():
            raise AssertionError("Expected profiler to be disabled")
        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

        with torch.autograd.profiler.profile():
            if not torch.autograd._profiler_enabled():
                raise AssertionError("Expected profiler to be enabled")
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))

    def test_is_autocast_cpu_enabled(self):
        # [FUNCTION DOC] Checks that is_autocast_cpu_enabled inside a CPU bfloat16 autocast
        # block is evaluated as True when compiled with the eager backend and fullgraph; results
        # must match eager.
        def fn(a_float32, b_float32):
            # [FUNCTION DOC] HELPER: Inside CPU bfloat16 autocast does a matmul of the inputs, adds
            # one if CPU autocast is enabled, returns the result.
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                c_float16 = torch.mm(a_float32, b_float32)
                if torch.is_autocast_cpu_enabled():
                    c_float16 = c_float16 + 1
            return c_float16

        a = torch.rand((8, 8))
        b = torch.rand((8, 8))
        ref = fn(a, b)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(a, b)
        self.assertTrue(same(ref, res))

    @unittest.skipIf(
        not PLATFORM_SUPPORTS_FLASH_ATTENTION,
        "Can't run fused SDPA on this platform",
    )
    def test_autocast_sdpa(self):
        # [FUNCTION DOC] Compiles with inductor a module doing scaled dot product attention under
        # nested CPU and accelerator autocast (float32) and compares device and dtype of the
        # compiled output to eager, expecting device index 0 and float32. Skipped when the platform
        # lacks flash attention support.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs scaled dot product attention under
            # nested CPU and accelerator autocast; has no state.
            def forward(self, query, key, value):
                # [FUNCTION DOC] HELPER: Nested autocast on cpu and on device_type with float32,
                # calls scaled_dot_product_attention with causal attention enabled and no mask;
                # returns the output.
                with torch.autocast("cpu"):
                    with torch.autocast(device_type, dtype=torch.float32):
                        out = F.scaled_dot_product_attention(
                            query, key, value, None, 0.0, True
                        )
                return out

        dtype = torch.float32
        seq_len_q = 1
        seq_len_k = 1
        head_dim = 8
        query = torch.ones(
            1,
            8,
            seq_len_q,
            head_dim,
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.ones(
            1,
            8,
            seq_len_k,
            head_dim,
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )
        value = torch.ones(
            1,
            8,
            seq_len_k,
            head_dim,
            device=device_type,
            dtype=dtype,
            requires_grad=True,
        )

        module = MyModule()
        real = module(query, key, value)
        real_device = real.device
        real_dtype = real.dtype

        opt_mod = torch.compile(module, backend="inductor")
        compiled = opt_mod(query, key, value)

        self.assertEqual(compiled.device, real_device)
        self.assertEqual(compiled.dtype, real_dtype)

        self.assertEqual(compiled.device.type, device_type)
        self.assertEqual(compiled.device.index, 0)
        self.assertEqual(compiled.dtype, torch.float32)

    def test_autocast_cpu(self):
        # [FUNCTION DOC] Exports a module that does two matmuls under CPU bfloat16 autocast with
        # torch._dynamo.export and checks that the exported graph's output device (cpu) and dtype
        # (bfloat16) match the eager module's.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward does two matmuls of random CPU tensors
            # under CPU bfloat16 autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates three random 8x8 CPU tensors and multiplies
                # them in two chained matmuls inside CPU bfloat16 autocast; ignores x and
                # returns the result.
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cpu")
        self.assertEqual(exported.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break(self):
        # [FUNCTION DOC] Checks that a CPU autocast block containing a graph break (between the two
        # matmuls) still yields bfloat16 output matching eager under the eager backend.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward makes random CPU tensors and does
            # matmuls under CPU bfloat16 autocast with graph breaks; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Makes random inputs, graph breaks, then runs two matmuls
                # in CPU bfloat16 autocast with another graph break between them.
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")
                torch._dynamo.graph_break()
                d_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    torch._dynamo.graph_break()
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        opt = torch.compile(module, backend="eager")
        res = opt(torch.tensor([0.5]))
        self.assertEqual(res.device, real_device)
        self.assertEqual(res.dtype, real_dtype)

        self.assertEqual(res.device.type, "cpu")
        self.assertEqual(res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_2(self):
        # [FUNCTION DOC] Regression test for issue 93890: a graph break inside a CPU bfloat16
        # autocast block must keep autocast active after the break; compiled result is close to
        # eager and both are bfloat16 while the input stays float32.
        # Regression for: https://github.com/pytorch/pytorch/issues/93890
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside CPU bfloat16 autocast, matmul x with itself, graph
            # break, then relu.
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                x = torch.mm(x, x)
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x

        x = torch.rand([4, 4])
        self.assertEqual(x.dtype, torch.float32)
        res = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        opt_res = opt_fn(x)
        self.assertTrue(torch.allclose(res, opt_res))
        self.assertEqual(res.dtype, torch.bfloat16)
        self.assertEqual(opt_res.dtype, torch.bfloat16)

    def test_autocast_cpu_graph_break_inner_fn(self):
        # [FUNCTION DOC] Checks nested autocast (enabled, disabled, enabled) with graph breaks and
        # a non-inlinable function that itself graph breaks: the outer autocast state is restored
        # correctly after the inner ones exit, so one result is bfloat16 (autocast) and the other
        # float32 (autocast disabled), matching eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module with a static method that graph breaks and a forward
            # that nests autocast contexts with graph breaks; has no state.
            @staticmethod
            def mm_breaks(x, y):
                # [FUNCTION DOC] HELPER: Static method that forces a graph break and then returns
                # the matmul of x and y.
                torch._dynamo.graph_break()
                return torch.mm(x, y)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Module forward that runs matmuls inside nested autocast
                # contexts with graph breaks; asserts the two autocast results have the same dtype
                # and returns the autocast result and the autocast-disabled float32 result.
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    torch._dynamo.graph_break()
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        torch._dynamo.graph_break()
                        g_float32 = torch.mm(a_float32, b_float32)
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            # Check that nested with non-inlineable function with graph break
                            torch._dynamo.graph_break()
                            f_float16_1 = self.mm_breaks(a_float32, b_float32)
                    # We remember to exit the inner autocast correctly to outer
                    # even after graph breaks
                    f_float16 = self.mm_breaks(a_float32, b_float32)
                    assert f_float16.dtype == f_float16_1.dtype  # noqa: S101
                return f_float16, g_float32

        module = MyModule()
        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch.compile(module, backend="eager")
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    def test_autocast_graph_break_method(self):
        # [FUNCTION DOC] Checks autocast with graph break in a module method: a method that
        # graph breaks and one that does not both give the same first element, and the
        # results keep the dtypes eager produces (bfloat16 for the autocast result, float32
        # for the disabled region).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module holding a bias tensor (attribute bias) with
            # methods that add the bias after a matmul, with or without a graph break; forward
            # nests autocast contexts.
            def __init__(self, bias):
                # [FUNCTION DOC] HELPER: Stores the bias tensor on the module.
                super().__init__()
                self.bias = bias

            def mm_not_break(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the matmul of x and y plus the module bias without
                # any graph break.
                return torch.mm(x, y) + self.bias

            def mm_breaks(self, x, y):
                # [FUNCTION DOC] HELPER: Triggers a graph break and then returns the matmul of x
                # and y plus the bias.
                torch._dynamo.graph_break()
                return torch.mm(x, y) + self.bias

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs a matmul with autocast disabled inside an autocast
                # region, then calls the graph-breaking method, and asserts its first element
                # equals that of the non-breaking method; returns both results.
                a_float32 = torch.rand((8, 8), device="cpu")
                b_float32 = torch.rand((8, 8), device="cpu")

                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    with torch.autocast(
                        device_type="cpu", dtype=torch.bfloat16, enabled=False
                    ):
                        g_float32 = torch.mm(a_float32, b_float32)
                    f_float16 = self.mm_breaks(a_float32, b_float32)

                    assert (  # noqa: S101
                        f_float16[0][0] == self.mm_not_break(a_float32, b_float32)[0][0]
                    )
                return f_float16, g_float32

        module = MyModule(bias=torch.rand((8, 8), device="cpu", dtype=torch.bfloat16))

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            # Autocast doesn't work on addition, so we need the bias to be `bfloat16`
            res = torch.rand((8, 8), device="cpu", dtype=torch.float32) + torch.rand(
                (8, 8), device="cpu", dtype=torch.bfloat16
            )
            self.assertEqual(res.dtype, torch.float32)

        real_16, real_32 = module(torch.tensor([0.5]))
        real_device_16 = real_16.device
        real_dtype_16 = real_16.dtype
        real_device_32 = real_32.device
        real_dtype_32 = real_32.dtype

        graph = torch.compile(module, backend="eager")
        out_16, out_32 = graph(torch.tensor([0.5]))
        self.assertEqual(out_16.device, real_device_16)
        self.assertEqual(out_16.dtype, real_dtype_16)
        self.assertEqual(out_32.device, real_device_32)
        self.assertEqual(out_32.dtype, real_dtype_32)

        self.assertEqual(out_16.device.type, "cpu")
        self.assertEqual(out_16.dtype, torch.bfloat16)
        self.assertEqual(out_32.device.type, "cpu")
        self.assertEqual(out_32.dtype, torch.float32)

    def test__enter__exit_autocast(self):
        # [FUNCTION DOC] Verifies that the low-level private autocast enter/exit
        # functions are traced under fullgraph: compiled output equals eager, autocast is
        # disabled afterwards, and the recorded FX graph text matches expected strings
        # (separate expectations for dynamic-shape capture and static shapes; the graph
        # spells out set_autocast_enabled/dtype/nesting/cache calls).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Enters CPU autocast via the private enter function, does a
            # matmul, exits autocast, returns the product.
            m = torch.amp.autocast_mode._enter_autocast("cpu")
            x = x @ y
            torch.amp.autocast_mode._exit_autocast(m)
            return x

        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(f, backend=eager, fullgraph=True)
        x = torch.randn(3, 3, dtype=torch.float32)
        y = torch.randn(3, 3, dtype=torch.float32)
        z = f(x, y)
        opt_z = opt_f(x, y)
        self.assertEqual(z, opt_z)
        self.assertEqual(z.dtype, opt_z.dtype)
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, s77]", s77: "Sym(s77)", L_y_: "f32[s77, s77]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _is_autocast_available = torch._C._is_autocast_available('cpu');  _is_autocast_available = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', True);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        autocast_increment_nesting = torch.autocast_increment_nesting();  autocast_increment_nesting = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None

        matmul: "bf16[s77, s77]" = l_x_ @ l_y_;  l_x_ = l_y_ = None

        autocast_decrement_nesting = torch.autocast_decrement_nesting();  autocast_decrement_nesting = None

        clear_autocast_cache = torch.clear_autocast_cache();  clear_autocast_cache = None

        set_autocast_enabled_1 = torch.set_autocast_enabled('cpu', False);  set_autocast_enabled_1 = None
        set_autocast_dtype_1 = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype_1 = None
        set_autocast_cache_enabled_1 = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled_1 = None
        return (matmul,)
""",
            )
        else:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _is_autocast_available = torch._C._is_autocast_available('cpu');  _is_autocast_available = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', True);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        autocast_increment_nesting = torch.autocast_increment_nesting();  autocast_increment_nesting = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None

        matmul: "bf16[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None

        autocast_decrement_nesting = torch.autocast_decrement_nesting();  autocast_decrement_nesting = None

        clear_autocast_cache = torch.clear_autocast_cache();  clear_autocast_cache = None

        set_autocast_enabled_1 = torch.set_autocast_enabled('cpu', False);  set_autocast_enabled_1 = None
        set_autocast_dtype_1 = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype_1 = None
        set_autocast_cache_enabled_1 = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled_1 = None
        return (matmul,)
""",
            )

    def test__enter__exit_autocast_graph_break(self):
        # [FUNCTION DOC] Checks that a graph break between _enter_autocast and _exit_autocast
        # works: the first graph contains the autocast enter calls, the second graph (resumed after
        # the break) contains the matmul in bfloat16 plus the autocast decrement/restore calls, but
        # not the enter functions. Compares outputs with eager and checks the expected graph text
        # for dynamic and static shapes.
        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Enters autocast, matmul with y, graph break, matmul with z,
            # exits autocast, returns x.
            m = torch.amp.autocast_mode._enter_autocast("cpu")
            x = x @ y
            torch._dynamo.graph_break()
            x = x @ z
            # At this point m is wrapped as an AutocastModeVariable, which will graph break on the __exit__ call
            torch.amp.autocast_mode._exit_autocast(m)
            return x

        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(f, backend=eager, fullgraph=False)
        x = torch.randn(3, 3, dtype=torch.float32)
        y = torch.randn(3, 3, dtype=torch.float32)
        z = torch.randn(3, 3, dtype=torch.float32)
        out = f(x, y, z)
        opt_out = opt_f(x, y, z)
        self.assertEqual(out, opt_out)
        self.assertEqual(out.dtype, opt_out.dtype)
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, s77]", s77: "Sym(s77)", L_y_: "f32[s77, s77]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _is_autocast_available = torch._C._is_autocast_available('cpu');  _is_autocast_available = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', True);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        autocast_increment_nesting = torch.autocast_increment_nesting();  autocast_increment_nesting = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None

        matmul: "bf16[s77, s77]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
        return (matmul,)
""",
            )
        else:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _is_autocast_available = torch._C._is_autocast_available('cpu');  _is_autocast_available = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', True);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        autocast_increment_nesting = torch.autocast_increment_nesting();  autocast_increment_nesting = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None

        matmul: "bf16[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None
        return (matmul,)
""",
            )

        # Doesn't include autocast functions, see comment above
        graph = eager.graphs[1]
        actual = normalize_gm(graph.print_readable(False))

        if check_dynamic_shape_capture():
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "bf16[s77, s77]", s77: "Sym(s77)", L_z_: "f32[s77, s77]"):
        l_x_ = L_x_
        l_z_ = L_z_

        matmul: "bf16[s77, s77]" = l_x_ @ l_z_;  l_x_ = l_z_ = None

        autocast_decrement_nesting = torch.autocast_decrement_nesting();  autocast_decrement_nesting = None

        clear_autocast_cache = torch.clear_autocast_cache();  clear_autocast_cache = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', False);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None
        return (matmul,)
""",
            )
        else:
            self.assertExpectedInline(
                actual,
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "bf16[3, 3]", L_z_: "f32[3, 3]"):
        l_x_ = L_x_
        l_z_ = L_z_

        matmul: "bf16[3, 3]" = l_x_ @ l_z_;  l_x_ = l_z_ = None

        autocast_decrement_nesting = torch.autocast_decrement_nesting();  autocast_decrement_nesting = None

        clear_autocast_cache = torch.clear_autocast_cache();  clear_autocast_cache = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', False);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None
        return (matmul,)
""",
            )

    def test__enter__exit_autocast_graph_break_explicit_dtype(self):
        # [FUNCTION DOC] Checks entering autocast with explicit arguments (cpu, float16, enabled,
        # cache enabled) across a graph break: the compiled run matches eager and afterwards the
        # global CPU autocast enabled flag, dtype (bfloat16) and cache flag are back to the values
        # set by the test. Restores previous autocast settings in a finally block.
        def f(x):
            # [FUNCTION DOC] HELPER: Enters autocast with explicit dtype float16, adds one, graph
            # breaks, adds two, exits autocast.
            m = torch.amp.autocast_mode._enter_autocast(
                "cpu", torch.float16, True, True
            )
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 2
            torch.amp.autocast_mode._exit_autocast(m)
            return x

        prev_enabled = torch.is_autocast_enabled("cpu")
        prev_dtype = torch.get_autocast_dtype("cpu")
        prev_cache = torch.is_autocast_cache_enabled()
        try:
            torch.set_autocast_enabled("cpu", False)
            torch.set_autocast_dtype("cpu", torch.bfloat16)
            torch.set_autocast_cache_enabled(True)

            opt_f = torch.compile(f, backend="eager", fullgraph=False)
            x = torch.randn(3, 3, dtype=torch.float32)
            out = f(x)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertEqual(torch.get_autocast_dtype("cpu"), torch.bfloat16)

            opt_out = opt_f(x)
            self.assertEqual(out, opt_out)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
            self.assertEqual(torch.get_autocast_dtype("cpu"), torch.bfloat16)
            self.assertTrue(torch.is_autocast_cache_enabled())
        finally:
            torch.set_autocast_enabled("cpu", prev_enabled)
            torch.set_autocast_dtype("cpu", prev_dtype)
            torch.set_autocast_cache_enabled(prev_cache)

    def test_autocast_low_level_api(self):
        # [FUNCTION DOC] Verifies that the low-level autocast setters (set_autocast_enabled,
        # set_autocast_dtype, set_autocast_cache_enabled, autocast_decrement_nesting,
        # clear_autocast_cache) trace under fullgraph and produce eager-identical output and
        # dtype, with autocast disabled afterwards. Saves and restores global autocast state in
        # a finally block.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Turns CPU autocast on with bfloat16 and cache, matmuls x and
            # y, decrements the nesting, clears the cache and turns CPU autocast off.
            torch.set_autocast_enabled("cpu", True)
            torch.set_autocast_dtype("cpu", torch.bfloat16)
            torch.set_autocast_cache_enabled(True)
            x = x @ y
            torch.autocast_decrement_nesting()
            torch.clear_autocast_cache()
            torch.set_autocast_enabled("cpu", False)
            return x

        prev_enabled = torch.is_autocast_enabled("cpu")
        prev_dtype = torch.get_autocast_dtype("cpu")
        prev_cache = torch.is_autocast_cache_enabled()

        try:
            opt_f = torch.compile(f, backend="eager", fullgraph=True)
            x = torch.randn(3, 3, dtype=torch.float32)
            y = torch.randn(3, 3, dtype=torch.float32)
            out = f(x, y)
            opt_out = opt_f(x, y)
            self.assertEqual(out, opt_out)
            self.assertEqual(out.dtype, opt_out.dtype)
            self.assertFalse(torch.is_autocast_enabled("cpu"))
        finally:
            torch.set_autocast_enabled("cpu", prev_enabled)
            torch.set_autocast_dtype("cpu", prev_dtype)
            torch.set_autocast_cache_enabled(prev_cache)

    def test__enter__exit_autocast_function_mode(self):
        # [FUNCTION DOC] Checks that a TorchFunctionMode counting calls sees exactly one
        # _enter_autocast and one _exit_autocast call in both eager and compiled runs, and that
        # results and dtypes match.
        class FunctionCount(torch.overrides.TorchFunctionMode):
            # [CLASS DOC] Torch function mode that counts how often each function is called.
            # Attribute: counts, a defaultdict mapping function to call count. Calls pass through
            # to the real function.
            def __init__(self):
                # [FUNCTION DOC] Initializes the counts mapping as a defaultdict with default zero.
                self.counts = defaultdict(int)

            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] Increments the counter for func and forwards the call with the
                # original arguments (empty keyword mapping if none); returns the real result.
                self.counts[func] += 1
                return func(*args, **(kwargs or {}))

        def f(x, y):
            # [FUNCTION DOC] HELPER: Enters CPU autocast, matmuls x and y, exits autocast,
            # returns the product.
            m = torch.amp.autocast_mode._enter_autocast("cpu")
            x = x @ y
            torch.amp.autocast_mode._exit_autocast(m)
            return x

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3, dtype=torch.float32)
        y = torch.randn(3, 3, dtype=torch.float32)
        with FunctionCount() as fc:
            z = f(x, y)
            self.assertEqual(fc.counts[torch.amp.autocast_mode._enter_autocast], 1)
            self.assertEqual(fc.counts[torch.amp.autocast_mode._exit_autocast], 1)
        with FunctionCount() as fc:
            opt_z = opt_f(x, y)
            self.assertEqual(fc.counts[torch.amp.autocast_mode._enter_autocast], 1)
            self.assertEqual(fc.counts[torch.amp.autocast_mode._exit_autocast], 1)
        self.assertEqual(z, opt_z)
        self.assertEqual(z.dtype, opt_z.dtype)
        self.assertFalse(torch.is_autocast_enabled("cpu"))

    def test__enter__exit_autocast_non_idempotent(self):
        # [FUNCTION DOC] Checks that tracing a normal autocast with block yields calls to the
        # private enter/exit functions in the captured graph (expected text asserted), and that re-
        # compiling that graph (by exec of its printed source) decomposes them into the low-level
        # autocast calls while still matching eager output. Returns early when dynamic shape
        # capture is on because the recompile trick does not work there. Side effect: executes the
        # printed graph code with exec.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def f(x, y):
            # [FUNCTION DOC] HELPER: Matmul of x and y inside a CPU autocast with block;
            # returns the product.
            with torch.amp.autocast("cpu"):
                x = x @ y
            return x

        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(f, backend=eager, fullgraph=False)
        x = torch.randn(3, 3, dtype=torch.float32)
        y = torch.randn(3, 3, dtype=torch.float32)
        out = f(x, y)
        opt_out = opt_f(x, y)
        self.assertEqual(out, opt_out)
        self.assertEqual(out.dtype, opt_out.dtype)
        self.assertFalse(torch.is_autocast_enabled("cpu"))
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3, 3]", L_y_: "f32[3, 3]"):
        l_x_ = L_x_
        l_y_ = L_y_

        _enter_autocast = torch.amp.autocast_mode._enter_autocast('cpu', None, True, None)

        matmul: "bf16[3, 3]" = l_x_ @ l_y_;  l_x_ = l_y_ = None

        _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
        return (matmul,)
""",
        )

        # Recompiling will decompose the _enter_autocast and _exit_autocast calls to lower level autocast functions
        eager = EagerAndRecordGraphs()
        d = {}
        exec(actual, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x, y)[0]
        self.assertEqual(out, retraced_out)
        self.assertEqual(out.dtype, retraced_out.dtype)
        self.assertFalse(torch.is_autocast_enabled("cpu"))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_L_x_: "f32[3, 3]", L_L_y_: "f32[3, 3]"):
        l_l_x_ = L_L_x_
        l_l_y_ = L_L_y_

        _is_autocast_available = torch._C._is_autocast_available('cpu');  _is_autocast_available = None

        set_autocast_enabled = torch.set_autocast_enabled('cpu', True);  set_autocast_enabled = None
        set_autocast_dtype = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype = None
        autocast_increment_nesting = torch.autocast_increment_nesting();  autocast_increment_nesting = None
        set_autocast_cache_enabled = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled = None
        matmul: "bf16[3, 3]" = l_l_x_ @ l_l_y_;  l_l_x_ = l_l_y_ = None
        autocast_decrement_nesting = torch.autocast_decrement_nesting();  autocast_decrement_nesting = None

        clear_autocast_cache = torch.clear_autocast_cache();  clear_autocast_cache = None

        set_autocast_enabled_1 = torch.set_autocast_enabled('cpu', False);  set_autocast_enabled_1 = None
        set_autocast_dtype_1 = torch.set_autocast_dtype('cpu', torch.bfloat16);  set_autocast_dtype_1 = None
        set_autocast_cache_enabled_1 = torch.set_autocast_cache_enabled(True);  set_autocast_cache_enabled_1 = None
        return (matmul,)
""",
        )

    def test_retrace_inference_mode(self):
        # [FUNCTION DOC] Checks that inference_mode is traced to
        # _enter_inference_mode/_exit_inference_mode calls in the graph, the global inference mode
        # flag is off afterward, and that re-compiling the printed graph (via exec) reproduces
        # eager output. Returns early under dynamic shape capture.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def f(x):
            # [FUNCTION DOC] HELPER: Adds one to x inside an inference_mode block and returns it.
            with torch.inference_mode():
                y = x + 1
            return y

        x = torch.randn(2, 2)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(f, backend=eager, fullgraph=True)
        out = f(x)
        opt_out = opt_f(x)
        self.assertEqual(out, opt_out)
        self.assertFalse(torch.is_inference_mode_enabled())

        first_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertIn(
            "torch.autograd.grad_mode._enter_inference_mode(True)", first_graph
        )
        self.assertIn("torch.autograd.grad_mode._exit_inference_mode", first_graph)

        d = {}
        exec(first_graph, globals(), d)
        eager = EagerAndRecordGraphs()
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(out, retraced_out)
        self.assertFalse(torch.is_inference_mode_enabled())

    @parametrize(
        "Ctx",
        [CustomizedCtxManagerWithGraphBreak, customized_ctx_manager_with_graph_break],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_ctx_manager_with_graph_break(self, Ctx):
        # [FUNCTION DOC] Parametrized over a class-based and a generator-based context manager
        # that graph break on entry (Ctx). Checks that the with body runs eagerly, produces the
        # no-grad result, matches eager, restores grad mode (enabled) after, and that the output
        # does not require grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Within the context manager built with False, if grad is
            # enabled adds 1000, otherwise computes sin(2x)+3; returns the result.
            with Ctx(False):
                # body runs on eager
                if torch.is_grad_enabled():
                    z = x + 1000
                else:
                    y = x * 2
                    z = y.sin() + 3
            return z

        self.assertTrue(torch.is_grad_enabled())
        x = torch.randn(2, 3, requires_grad=True)
        expected = fn(x)
        got = torch.compile(backend="eager", fullgraph=False)(fn)(x)
        self.assertEqual(expected, got)
        self.assertTrue(torch.is_grad_enabled())
        self.assertFalse(got.requires_grad)  # since it was run under torch.no_grad.

    def test_return_context_manager(self):
        # [FUNCTION DOC] Checks a context manager object created inside a fullgraph-compiled
        # function can be returned to the caller with its attributes intact (mode False).
        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Creates a CustomizedCtxManager(False), enters and exits it
            # with an empty body, and returns it.
            cm = CustomizedCtxManager(False)
            with cm:
                pass
            return cm

        x = torch.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    def test_return_context_manager_with_graph_break(self):
        # [FUNCTION DOC] Same as test_return_context_manager but with a graph break before the with
        # block (fullgraph disabled); the returned manager must still have mode False.
        @torch.compile(backend="eager", fullgraph=False)
        def f(x):
            # [FUNCTION DOC] HELPER: Creates the manager, graph breaks, enters and exits it with an
            # empty body, returns it.
            cm = CustomizedCtxManager(False)
            torch._dynamo.graph_break()
            with cm:
                pass
            return cm

        x = torch.randn(2, 3)
        cm = f(x)
        self.assertFalse(cm.mode)

    @torch._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager(self, Ctx):
        # [FUNCTION DOC] Parametrized over class-based and generator-based managers with
        # enable_trace_contextlib patched on. With fullgraph compile, checks results match eager
        # under no_grad and enable_grad ambient modes, with one frame/6 ops then a second frame/12
        # ops total (recompile for the changed grad mode).
        def fn(x):
            # [FUNCTION DOC] HELPER: Within the manager (grad forced on), adds one, multiplies by
            # two if grad is enabled, applies relu; returns the result minus one.
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=True)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 6)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 12)

    @torch._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager(self, Ctx):
        # [FUNCTION DOC] Same as test_generic_context_manager but with nested managers
        # (True then False); expects 1 frame/9 ops under no_grad and 2 frames/18 ops
        # cumulative under enable_grad.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested managers toggling grad on then off, with arithmetic
            # that depends on the grad flag at each level; returns the result.
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=True)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 1)
            self.assertEqual(cnts.op_count, 9)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            self.assertEqual(cnts.frame_count, 2)
            self.assertEqual(cnts.op_count, 18)

    @torch._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_generic_context_manager_with_graph_break(self, Ctx):
        # [FUNCTION DOC] Like test_generic_context_manager but with a graph break inside the with
        # block and fullgraph disabled. Frame and op counts are asserted only for the class-based
        # manager (2 frames/2 ops, then 4/4).
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside the manager, add one, optionally multiply by two when
            # grad enabled, graph break, relu; returns the result minus one.
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                torch._dynamo.graph_break()
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 2)
                self.assertEqual(cnts.op_count, 2)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    @torch._dynamo.config.patch(enable_trace_contextlib=True)
    @parametrize(
        "Ctx",
        [CustomizedCtxManager, customized_ctx_manager],
        name_fn=lambda x: x.__name__,
    )
    def test_nested_generic_context_manager_with_graph_break(self, Ctx):
        # [FUNCTION DOC] Nested managers with a graph break inside the inner one, fullgraph off.
        # Runs under no_grad, resets Dynamo, recompiles and runs under enable_grad; for the class-
        # based manager both runs expect 4 frames and 4 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Nested managers (True, then False) with a graph break inside
            # the inner block; returns the result minus one.
            with Ctx(True):
                x = x + 1
                if torch.is_grad_enabled():
                    x = x * 2
                with Ctx(False):
                    if torch.is_grad_enabled():
                        x = x - 3
                    torch._dynamo.graph_break()
                    x = x * 1.5
                x = torch.relu(x)
            return x - 1

        x = torch.rand(2, 3)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=cnts, fullgraph=False)(fn)

        with torch.no_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)

        with torch.enable_grad():
            ref = fn(x)
            res = opt_fn(x)
            self.assertTrue(same(ref, res))
            if Ctx is CustomizedCtxManager:
                self.assertEqual(cnts.frame_count, 4)
                self.assertEqual(cnts.op_count, 4)

    def test_graph_break_inlining_grad(self):
        # [FUNCTION DOC] Checks that a graph break inside an inlined function that uses no_grad
        # compiles into 2 frames and that backward through the compiled result runs without error.
        def gn(z):
            # [FUNCTION DOC] HELPER: Under no_grad, graph breaks and returns sin of z.
            with torch.no_grad():
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Computes the matmul of x and y, calls gn on z (discarding the
            # result) and returns the matmul.
            a = torch.mm(x, y)
            z = gn(z)
            return a

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        z = torch.randn(4)
        opt_fn(x, y, z).sum().backward()

        self.assertEqual(cnts.frame_count, 2)

    def _graph_break_inlining_autocast_test_helper(self, device):
        # [FUNCTION DOC] HELPER: Shared body for the autocast graph-break inlining test.
        # Compiles with the eager backend a function that calls an inlined function containing
        # bfloat16 autocast on the given device with a graph break; asserts the compiled
        # output equals eager.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Inside bfloat16 autocast on the device, matmul x and y, graph
            # break, return sin of the product.
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z = torch.mm(x, y)
                torch._dynamo.graph_break()
                return torch.sin(z)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Matmul of x and y plus the result of gn on the same inputs.
            z = torch.mm(x, y)
            z = z + gn(x, y)
            return z

        x = torch.rand(3, 3).to(device)
        y = torch.rand(3, 3).to(device)
        opt_fn = torch.compile(backend="eager")(fn)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_graph_break_inlining_autocast(self):
        # [FUNCTION DOC] Runs the autocast graph-break inlining helper for cuda, cpu and xpu
        # devices, skipping cuda or xpu when unavailable or lacking bfloat16 support.
        for device in ["cuda", "cpu", "xpu"]:
            if device == "cuda" and not (
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ):
                continue
            if device == "xpu" and not (
                torch.xpu.is_available() and torch.xpu.is_bf16_supported()
            ):
                continue
            self._graph_break_inlining_autocast_test_helper(device)

    def test_disable_saved_tensors_hooks(self):
        # [FUNCTION DOC] Checks that the torch.autograd.graph.disable_saved_tensors_hooks decorator
        # is traced under fullgraph into private disable/enable calls around the body; asserts the
        # recorded graph text.
        def fn(z):
            # [FUNCTION DOC] HELPER: Defines a decorated inner function that adds its two inputs
            # under disabled saved-tensor hooks, builds ones and zeros of size one and returns the
            # decorated call result.
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return x + y

            x, y = (
                torch.ones(
                    1,
                ),
                torch.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        ones: "f32[1]" = torch.ones(1)

        zeros: "f32[1]" = torch.zeros(1)

        add: "f32[1]" = ones + zeros;  ones = zeros = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (add,)
""",
        )

    def test_disable_saved_tensors_hooks_prev_disabled(self):
        # [FUNCTION DOC] Same as the disable_saved_tensors_hooks decorator test but with hooks
        # already disabled by an outer message: the traced graph re-disables with the previous
        # message at the end instead of enabling.
        def fn(z):
            # [FUNCTION DOC] HELPER: Builds ones and zeros and calls a decorated adder
            # under disabled hooks.
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                return x + y

            x, y = (
                torch.ones(
                    1,
                ),
                torch.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        ones: "f32[1]" = torch.ones(1)

        zeros: "f32[1]" = torch.zeros(1)

        add: "f32[1]" = ones + zeros;  ones = zeros = None

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_1 = None
        return (add,)
""",
        )

    def test_disable_saved_tensors_hooks_prev_disabled_nested(self):
        # [FUNCTION DOC] Nested decorated functions both disabling saved tensor hooks inside an
        # outer disabled context; asserts the traced graph restores the correct previous message
        # after each inner exit.
        def fn(z):
            # [FUNCTION DOC] HELPER: Builds ones and zeros and calls the decorated outer function.
            @torch.autograd.graph.disable_saved_tensors_hooks("This is not supported")
            def f(x, y):
                # [FUNCTION DOC] HELPER: Decorated outer function that calls a decorated inner
                # adder and then adds x.
                @torch.autograd.graph.disable_saved_tensors_hooks(
                    "This is not supported inner"
                )
                def inner_fn(x, y):
                    # [FUNCTION DOC] HELPER: Returns the sum of x and y.
                    return x + y

                return inner_fn(x, y) + x

            x, y = (
                torch.ones(
                    1,
                ),
                torch.zeros(
                    1,
                ),
            )
            return f(x, y)

        eager = EagerAndRecordGraphs()
        with torch.autograd.graph.disable_saved_tensors_hooks(
            "Previously disabled message"
        ):
            torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        ones: "f32[1]" = torch.ones(1)

        zeros: "f32[1]" = torch.zeros(1)

        _saved_tensors_hooks_disable_1 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported inner');  _saved_tensors_hooks_disable_1 = None

        add: "f32[1]" = ones + zeros;  zeros = None

        _saved_tensors_hooks_disable_2 = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable_2 = None

        add_1: "f32[1]" = add + ones;  add = ones = None

        _saved_tensors_hooks_disable_3 = torch._C._autograd._saved_tensors_hooks_disable('Previously disabled message');  _saved_tensors_hooks_disable_3 = None
        return (add_1,)
""",
        )

    def test_disable_saved_tensors_hooks_graph_break(self):
        # [FUNCTION DOC] Checks disable_saved_tensors_hooks as a with block across a graph
        # break: both resulting graphs are asserted to disable hooks at their start and enable
        # them at their end.
        def fn(x):
            # [FUNCTION DOC] HELPER: Within the disable block, adds one, graph breaks, and returns
            # the result times two.
            with torch.autograd.graph.disable_saved_tensors_hooks(
                "This is not supported"
            ):
                y = x + 1
                torch._dynamo.graph_break()
                return y * 2

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=False)(torch.randn(()))

        def check_graph(actual, expected):
            # [FUNCTION DOC] HELPER: Unused helper that asserts the actual text equals the expected
            # text with assertExpectedInline.
            self.assertExpectedInline(actual, expected)

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        add: "f32[]" = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (add,)
""",
        )

        graph = eager.graphs[1]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_y_: "f32[]"):
        l_y_ = L_y_

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        mul: "f32[]" = l_y_ * 2;  l_y_ = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (mul,)
""",
        )

    def test__saved_tensors_hooks_disable(self):
        # [FUNCTION DOC] Checks direct calls to torch._C._autograd._saved_tensors_hooks_disable and
        # enable are traced under fullgraph; asserts the expected graph text.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, disables saved tensor hooks with a message,
            # multiplies in place by two, re-enables hooks, returns the result.
            y = x + 1
            torch._C._autograd._saved_tensors_hooks_disable("This is not supported")
            y *= 2
            torch._C._autograd._saved_tensors_hooks_enable()
            return y

        eager = EagerAndRecordGraphs()
        torch.compile(fn, backend=eager, fullgraph=True)(torch.randn(()))
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))
        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[]"):
        l_x_ = L_x_

        add: "f32[]" = l_x_ + 1;  l_x_ = None

        _saved_tensors_hooks_disable = torch._C._autograd._saved_tensors_hooks_disable('This is not supported');  _saved_tensors_hooks_disable = None

        add *= 2;  imul: "f32[]" = add;  add = None

        _saved_tensors_hooks_enable = torch._C._autograd._saved_tensors_hooks_enable();  _saved_tensors_hooks_enable = None
        return (imul,)
""",
        )

    def test_context_wrapping_grad_mode_decorator(self):
        # [FUNCTION DOC] Exercises enable_grad and no_grad used as decorators (called with
        # parentheses or directly) on an inner function defined inside a compiled function; the
        # global grad mode must not change when the decorator is applied. Loops over both
        # decorator forms and both modes, resetting Dynamo each time, and compares outputs and
        # requires_grad with eager.
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]
        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, _ = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    # [FUNCTION DOC] HELPER: Wraps an inner sin function with the chosen grad
                    # decorator inside the inverse mode, asserts the global grad mode was not
                    # mutated, then calls the wrapped function under the inverse mode.
                    def inner_func(x):
                        # [FUNCTION DOC] HELPER: Returns the sine of x.
                        return x.sin()

                    with ctx_wrapper_inverse():
                        if call:
                            inner_func = ctx_wrapper()(inner_func)
                        else:
                            inner_func = ctx_wrapper(inner_func)

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse  # noqa: S101

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_grad_mode_nested_function_decorator(self):
        # [FUNCTION DOC] Same as the previous decorator test but the decorator is applied with
        # decorator syntax inside the function, either called or not. Compares outputs and
        # requires_grad with eager.
        ctx_wrappers = [(torch.enable_grad, True), (torch.no_grad, False)]

        for call in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                ctx_wrapper, _ = ctx_wrappers[i]
                ctx_wrapper_inverse, mode_inverse = ctx_wrappers[(i + 1) % 2]

                def fn(x):
                    # [FUNCTION DOC] HELPER: Defines inner_func with the grad decorator inside
                    # the inverse mode, asserts the grad mode is unchanged, then calls it under
                    # the inverse mode.
                    with ctx_wrapper_inverse():
                        if call:

                            @ctx_wrapper()
                            def inner_func(x):
                                # [FUNCTION DOC] HELPER: Returns the sine of x.
                                return x.sin()

                        else:

                            @ctx_wrapper
                            def inner_func(x):
                                # [FUNCTION DOC] HELPER: Returns the sine of x.
                                return x.sin()

                        # Calling no_grad or enabled_grad should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse  # noqa: S101

                    with ctx_wrapper_inverse():
                        return inner_func(x)

                x = torch.zeros(10, requires_grad=True)
                opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
                self.assertEqual(fn(x), opt_fn(x))
                self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_context_wrapping_set_grad_enabled_nested_function(self):
        # [FUNCTION DOC] Checks that applying set_grad_enabled as a decorator or by calling it on a
        # function does not mutate global grad mode.
        # KNOWN ISSUE: the compile-and-compare statements at the end are indented to the outer
        # decorator loop only, so just the last mode iteration is actually verified and fn refers
        # to the last defined version.
        modes = [True, False]
        for decorator in [True, False]:
            for i in range(2):
                torch._dynamo.reset()

                mode = modes[i]
                mode_inverse = modes[(i + 1) % 2]

                def fn(x):
                    # [FUNCTION DOC] HELPER: Inside set_grad_enabled with the inverse mode, builds
                    # inner_func decorated (or wrapped by calling) with set_grad_enabled(mode),
                    # asserts global grad mode unchanged, then calls it under the inverse mode.
                    with torch.set_grad_enabled(mode_inverse):
                        if decorator:

                            @torch.set_grad_enabled(mode)
                            def inner_func(x):
                                # [FUNCTION DOC] HELPER: Returns the sine of x.
                                return x.sin()

                        else:

                            def inner_func(x):
                                # [FUNCTION DOC] HELPER: Returns the sine of x.
                                return x.sin()

                            inner_func = torch.set_grad_enabled(mode)(inner_func)

                        # Consuming set_grad_enabled by calling it on a function
                        # should not mutate global state
                        assert torch.is_grad_enabled() == mode_inverse  # noqa: S101

                    with torch.set_grad_enabled(mode_inverse):
                        return inner_func(x)

            x = torch.zeros(10, requires_grad=True)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))
            self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_local(self):
        # [FUNCTION DOC] Checks that a set_grad_enabled object created but not yet entered, held in
        # a local across a graph break and entered afterwards, works: results and requires_grad
        # match eager and two frames compile.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, creates a set_grad_enabled(True) object, graph
            # breaks, then enters it and adds one.
            x = x + 1
            ctx = torch.set_grad_enabled(True)
            torch._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx(self):
        # [FUNCTION DOC] Same as the inactive-context local test but with contextlib.nullcontext (a
        # manager with no target values); two frames expected.
        import contextlib

        # test with context manager that results in None target_values
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, makes a nullcontext, graph breaks, then adds
            # one inside it.
            x = x + 1
            ctx = contextlib.nullcontext()
            torch._dynamo.graph_break()
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_local_nullctx2(self):
        # [FUNCTION DOC] Nullcontext created before a graph break that happens inside an inlined
        # function returning a list; the context is then used after the call. Expects two frames
        # and eager-equal results.
        import contextlib

        # test with nullcontext where graph break happens
        # in an inlined function that returns something
        def gn():
            # [FUNCTION DOC] HELPER: Graph breaks and returns the list [0, 1, 2].
            torch._dynamo.graph_break()
            return [0, 1, 2]

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, creates a nullcontext, calls gn, then adds the list
            # element at index 1 inside the context.
            x = x + 1
            ctx = contextlib.nullcontext()
            lst = gn()
            with ctx:
                x = x + lst[1]
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_inactive_context_graph_break_stack(self):
        # [FUNCTION DOC] Checks a not-yet-entered grad-mode context object that passes through a
        # graph-breaking function call (it lives on the stack) and is entered afterwards; asserts
        # equal results and requires_grad versus eager.
        def gn(ctx):
            # [FUNCTION DOC] HELPER: Graph breaks and returns its context argument unchanged.
            torch._dynamo.graph_break()
            return ctx

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, gets a context from gn, then adds one inside it.
            x = x + 1
            ctx = gn(torch.set_grad_enabled(True))
            # we expect a graph break on next line as well
            with ctx:
                x = x + 1
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)

    def test_inactive_context_graph_break_stack2(self):
        # [FUNCTION DOC] Checks that a context object passed as an argument across a graph break
        # in the argument list is entered correctly by the callee; two frames expected and
        # outputs equal eager.
        def gn(x, ctx, y, z, dummy):
            # [FUNCTION DOC] HELPER: Enters the given context and returns x*y*z.
            with ctx:
                return x * y * z

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one and calls gn with a set_grad_enabled(True) object,
            # constants 2 and 3, and a graph-break call as the dummy argument.
            x = x + 1
            x = gn(x, torch.set_grad_enabled(True), 2, 3, torch._dynamo.graph_break())
            return x

        x = torch.zeros(10, requires_grad=False)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(fn(x).requires_grad, opt_fn(x).requires_grad)
        self.assertEqual(cnts.frame_count, 2)

    def test_sdpa_kernel_ctx_manager1(self):
        # [FUNCTION DOC] Checks sdpa_kernel with the MATH backend under fullgraph: within
        # the block the current SDPA backend list equals the modified state (checked by
        # an allow_in_graph helper).
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            # [FUNCTION DOC] HELPER: Allowed-in-graph check asserting that the current SDPA kernel
            # backends equal the expected modified list.
            self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Enters sdpa_kernel with the MATH backend, runs
            # scaled_dot_product_attention on x converted to float32, then checks the backend
            # state, returns the output.
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [torch.nn.attention.SDPBackend.MATH]
            ):
                output = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

            return output

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        opt_f(torch.randn(2, 2, 2, 2).to(dtype=torch.float16))

    def test_sdpa_kernel_ctx_manager2(self):
        # [FUNCTION DOC] Checks sdpa_kernel across a graph break: the backend state is
        # original before and after the block and modified inside both the first and resumed
        # graphs; two frames expected.
        original_backend_state = set(torch.nn.attention._cur_sdpa_kernel_backends())
        modified_backend_state = [torch.nn.attention.SDPBackend.MATH]

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_original():
            # [FUNCTION DOC] HELPER: Allowed-in-graph check that the backend set equals the
            # original set captured before the test.
            self.assertEqual(
                set(torch.nn.attention._cur_sdpa_kernel_backends()),
                original_backend_state,
            )

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            # [FUNCTION DOC] HELPER: Allowed-in-graph check that the backends equal
            # the MATH-only state.
            self.assertEqual(
                torch.nn.attention._cur_sdpa_kernel_backends(), modified_backend_state
            )

        def g(x):
            # [FUNCTION DOC] HELPER: Graph breaks, runs scaled_dot_product_attention in float32 and
            # checks that the modified backend state is active.
            torch._dynamo.graph_break()
            output = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                torch.float32
            )
            check_backend_state_is_modified()
            return output

        def f(x):
            # [FUNCTION DOC] HELPER: Checks original backend state, enters sdpa_kernel(MATH), runs
            # attention, checks modified state, calls g (graph break), runs attention again, exits
            # and checks original state; returns the sum of three outputs.
            check_backend_state_is_original()
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [torch.nn.attention.SDPBackend.MATH]
            ):
                output1 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

                # graph break
                output2 = g(x)

                output3 = torch.nn.functional.scaled_dot_product_attention(x, x, x).to(
                    torch.float32
                )
                check_backend_state_is_modified()

            check_backend_state_is_original()

            return output1 + output2 + output3

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        opt_f(torch.randn(2, 2, 2, 2).to(dtype=torch.float16))
        self.assertEqual(cnts.frame_count, 2)

    # test sdpa_kernel graph break with 2 arguments
    def test_sdpa_kernel_ctx_manager3(self):
        # [FUNCTION DOC] Checks sdpa_kernel given two backends (MATH, FLASH_ATTENTION)
        # across a graph break (no real attention is run); the backend set seen after the
        # break must equal both.
        modified_backend_state = {
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        }

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            # [FUNCTION DOC] HELPER: Allowed-in-graph check that the backend set equals the
            # two expected backends.
            self.assertEqual(
                set(torch.nn.attention._cur_sdpa_kernel_backends()),
                modified_backend_state,
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Enters sdpa_kernel with two backends, adds one, graph breaks,
            # checks backend state, adds one more.
            with torch.nn.attention.sdpa_kernel(
                # pyre-fixme[16]: Module `torch.nn.attention` has no attribute `SDPBackend`.
                [
                    torch.nn.attention.SDPBackend.MATH,
                    torch.nn.attention.SDPBackend.FLASH_ATTENTION,
                ]
            ):
                # FLASH_ATTENTION may not be supported, but we're not actually
                # doing any sdpa
                x = x + 1
                torch._dynamo.graph_break()
                check_backend_state_is_modified()
                x = x + 1

            return x

        opt_f = torch.compile(f, backend="eager")
        opt_f(torch.randn(2, 2))

    # Regression test to make sure dynamo won't crash on these kwargs.
    def test_sdpa_kernel_ctx_manager_kwargs(self):
        # [FUNCTION DOC] Regression test that Dynamo does not crash when sdpa_kernel is called
        # with the backends and set_priority keyword arguments; the backend set is verified
        # inside the block.
        backends = [torch.nn.attention.SDPBackend.MATH]

        @torch._dynamo.allow_in_graph
        def check_backend_state_is_modified():
            # [FUNCTION DOC] HELPER: Allowed-in-graph check that the backend set equals
            # the requested backends.
            self.assertEqual(
                set(torch.nn.attention._cur_sdpa_kernel_backends()),
                set(backends),
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Enters sdpa_kernel with keyword arguments, adds one, checks
            # backends, adds one more.
            with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
                x = x + 1
                check_backend_state_is_modified()
                x = x + 1

            return x

        opt_f = torch.compile(f, backend="eager")
        opt_f(torch.randn(2, 2))

    # Regression test to make sure dynamo won't graph break on calling functions
    # decorated with special context manager.
    def test_sdpa_kernel_ctx_manager_as_decorator(self):
        # [FUNCTION DOC] Regression test that a function decorated with sdpa_kernel (priority
        # set) can be called inside a fullgraph compile without a graph break and gives the same
        # result as eager.
        SDPA_BACKEND_PRIORITY = [
            torch.nn.attention.SDPBackend.MATH,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
        ]

        @torch.nn.attention.sdpa_kernel(
            backends=SDPA_BACKEND_PRIORITY, set_priority=True
        )
        def scaled_dot_product_attention(q, k, v, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Forwards all arguments to scaled_dot_product_attention
            # (decorated by sdpa_kernel).
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, *args, **kwargs
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the decorated attention wrapper with x for q, k and v.
            return scaled_dot_product_attention(x, x, x)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.rand(16, 16, 64, 256, dtype=torch.float16)
        ref = f(x)
        res = opt_f(x)

        self.assertEqual(ref, res)

    # Regression test to make sure the value of set_priority is used correctly.
    def test_sdpa_kernel_ctx_manager_set_priority(self):
        # [FUNCTION DOC] Regression test that the set_priority flag is honoured: the SDP priority
        # order differs from the default inside a block with set_priority true and equals the
        # default with false.
        backends = [torch.nn.attention.SDPBackend.MATH]
        default_priority = torch._C._get_sdp_priority_order()

        @torch._dynamo.allow_in_graph
        def check_backend_priority(changed: bool):
            # [FUNCTION DOC] HELPER: Allowed-in-graph check that whether the SDP priority order
            # differs from the default equals the changed argument.
            self.assertEqual(
                changed,
                torch._C._get_sdp_priority_order() != default_priority,
            )

        def f(x):
            # [FUNCTION DOC] HELPER: Two sdpa_kernel blocks (set_priority True, then False),
            # checking the priority state after the first addition in each, returning x plus four.
            with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=True):
                x = x + 1
                check_backend_priority(changed=True)
                x = x + 1

            with torch.nn.attention.sdpa_kernel(backends=backends, set_priority=False):
                x = x + 1
                check_backend_priority(changed=False)
                x = x + 1

            return x

        opt_f = torch.compile(f, backend="eager")
        opt_f(torch.randn(2, 2))

    def test_torch_profiler_use_after_with_block(self):
        # [FUNCTION DOC] Checks that using a profiler object after its with block ends (reading
        # kineto results) causes exactly one graph break recorded in counters; results match eager.
        # Clears the graph-break counters first.
        counters.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: Enters a profiler, exits it, reads the event tree from the
            # profiler, returns x plus one.
            with torch.profiler.profile() as p:
                pass
            p.profiler.kineto_results.experimental_event_tree()
            return x + 1

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(1)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(len(counters["graph_break"]), 1)

    def test_311_resume_block_keyerror(self):
        # [FUNCTION DOC] Regression test for issue 162313 (Python 3.11 resume block KeyError):
        # graph breaks around with no_grad blocks in both branches of a conditional; flag is
        # flipped to False after the first call and the compiled function (recompiled for the other
        # branch) must still match eager. Mutates the enclosing flag variable.
        # https://github.com/pytorch/pytorch/issues/162313
        flag = True

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, graph break, adds two, then in either branch enters
            # no_grad with a graph break and adds 4 or 8; returns the sum plus 16. The branch is
            # chosen by the enclosing flag.
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 2
            if flag:
                with torch.no_grad():
                    torch._dynamo.graph_break()
                x = x + 4
            else:
                with torch.no_grad():
                    torch._dynamo.graph_break()
                x = x + 8
            return x + 16

        inp = torch.ones(3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp), opt_fn(inp))
        flag = False
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_311_resume_block_keyerror2(self):
        # [FUNCTION DOC] Regression test for issue 166176: nested no_grad blocks with graph breaks
        # before and inside them; compiled result must match eager.
        # https://github.com/pytorch/pytorch/issues/166176
        def fn(x):
            # [FUNCTION DOC] HELPER: Graph break, two nested no_grad blocks with a graph break
            # inside, then return x plus one.
            torch._dynamo.graph_break()
            with torch.no_grad():
                with torch.no_grad():
                    torch._dynamo.graph_break()
            return x + 1

        inp = torch.ones(3)
        opt_fn = torch.compile(fn, backend="eager")
        self.assertEqual(fn(inp), opt_fn(inp))

    def test_store_attr_on_function(self):
        # [FUNCTION DOC] Checks that setting an attribute on a function object within a no_grad
        # block is traced under fullgraph as a mutation replayed at runtime: after the compiled
        # call the function attribute equals input plus 2, and eager output matches.
        # setattr on a function object is traced as a generic attribute
        # mutation and replayed at runtime -- no graph break.
        def dummy():
            # [FUNCTION DOC] HELPER: Empty function used only as an object to store
            # an attribute on.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds 2, stores the result as an attribute on dummy within a
            # no_grad block, returns x plus 4.
            x = x + 2
            with torch.no_grad():
                dummy.attr1 = x
            return x + 4

        inp = torch.ones(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        # Check the compiled run applied the mutation before the eager run
        # below overwrites dummy.attr1.
        compiled_res = opt_fn(inp)
        self.assertEqual(dummy.attr1, inp + 2)
        self.assertEqual(fn(inp), compiled_res)

    @parametrize("gb", (True, False))
    def test_functorch_low_level(self, gb):
        # [FUNCTION DOC] Parametrized on whether to insert a graph break (gb). Checks the low-level
        # functorch calls for grad nesting and inplace-requires-grad flags are traced (fullgraph
        # when no break), the input gets requires_grad set, and global functorch level and inplace
        # flag return to their previous values.
        def f(x, gb):
            # [FUNCTION DOC] HELPER: Increments grad nesting, enables inplace-requires-grad, marks
            # x as requiring grad, optionally graph breaks, disables the flag, decrements nesting
            # and returns x plus the level.
            level = torch._C._functorch._grad_increment_nesting()
            torch._C._functorch.set_inplace_requires_grad_allowed(True)
            torch._functorch.eager_transforms._set_tensor_requires_grad(x)
            if gb:
                torch._dynamo.graph_break()
            torch._C._functorch.set_inplace_requires_grad_allowed(False)
            torch._C._functorch._grad_decrement_nesting()
            return x + level

        prev_inplace = torch._C._functorch.get_inplace_requires_grad_allowed()
        prev_level = torch._C._functorch.maybe_current_level()
        opt_f = torch.compile(f, fullgraph=not gb, backend="eager")
        x = torch.randn(3, 3, requires_grad=False)
        opt_y = opt_f(x, gb)
        self.assertTrue(x.requires_grad)
        y = f(x, gb)
        self.assertEqual(y, opt_y)
        self.assertEqual(torch._C._functorch.maybe_current_level(), prev_level)
        self.assertEqual(
            torch._C._functorch.get_inplace_requires_grad_allowed(), prev_inplace
        )

    @parametrize("gb", (True, False))
    def test_functorch_low_level_jvp(self, gb):
        # [FUNCTION DOC] Parametrized on gb. Checks the jvp nesting increment/decrement and dynamic
        # layer stack depth query are traced; results equal eager and the functorch level returns
        # to its previous value.
        def f(x, gb):
            # [FUNCTION DOC] HELPER: Increments jvp nesting, optionally graph breaks, reads the
            # dynamic layer stack depth, decrements nesting, returns x plus level plus depth.
            level = torch._functorch.predispatch._jvp_increment_nesting()
            if gb:
                torch._dynamo.graph_break()
            depth = torch._C._functorch.get_dynamic_layer_stack_depth()
            torch._functorch.predispatch._jvp_decrement_nesting()
            return x + level + depth

        prev_level = torch._C._functorch.maybe_current_level()
        opt_f = torch.compile(f, fullgraph=not gb, backend="eager")
        x = torch.randn(3, 3)
        opt_y = opt_f(x, gb)
        y = f(x, gb)
        self.assertEqual(y, opt_y)
        self.assertEqual(torch._C._functorch.maybe_current_level(), prev_level)

    def test_retrace_grad(self):
        # [FUNCTION DOC] Checks torch.func.grad is traced under fullgraph and that the captured
        # graph source, executed with exec and recompiled, reproduces the same output and the same
        # printed graph. Returns early under dynamic shapes.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin(x).
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Applies torch.func.grad of fn to x.
            return torch.func.grad(fn)(x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y = wrapper_fn(x)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_functorch_get_dynamic_layer_stack_depth(self):
        # [FUNCTION DOC] Checks get_dynamic_layer_stack_depth is traced with and without arguments
        # while the functorch nesting is increased on each iteration; the layer stack is popped
        # back to the original depth in a finally clause.
        def f(x=None):
            # [FUNCTION DOC] HELPER: Reads the dynamic layer stack depth and returns either x plus
            # depth (when x is given) or the depth.
            l = torch._C._functorch.get_dynamic_layer_stack_depth()
            if x is not None:
                return x + l
            else:
                return l

        prev_dynamic_layer_stack_depth = (
            torch._C._functorch.get_dynamic_layer_stack_depth()
        )
        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        N = 3
        x = torch.randn(3, 3)
        try:
            for _ in range(N):
                out = f()
                opt_out = opt_f()
                self.assertEqual(out, opt_out)

                out = f(x)
                opt_out = opt_f(x)
                self.assertEqual(out, opt_out)
                torch._C._functorch._grad_increment_nesting()
        finally:
            torch._C._functorch.pop_dynamic_layer_stack_and_undo_to_depth(
                prev_dynamic_layer_stack_depth
            )

    def test_retrace_hessian(self):
        # [FUNCTION DOC] Same retrace check as grad but for torch.func.hessian; returns early
        # under dynamic shapes.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of sin(x).
            return x.sin().sum()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Applies torch.func.hessian of fn to x.
            return torch.func.hessian(fn)(x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y = wrapper_fn(x)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_retrace_vmap(self):
        # [FUNCTION DOC] Same retrace check for torch.func.vmap.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Applies torch.func.vmap of fn to x.
            return torch.func.vmap(fn)(x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y = wrapper_fn(x)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_retrace_vjp(self):
        # [FUNCTION DOC] Same retrace check for torch.func.vjp; compares the primal outputs.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Calls torch.func.vjp on fn and x, returning the output and
            # the vjp function.
            return torch.func.vjp(fn, x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y, _ = wrapper_fn(x)
        opt_y, _ = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_retrace_jvp(self):
        # [FUNCTION DOC] Same retrace check for torch.func.jvp with a scalar input.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Calls torch.func.jvp with x as both primal and tangent.
            return torch.func.jvp(fn, (x,), (x,))

        x = torch.randn([])
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y, _ = wrapper_fn(x)
        opt_y, _ = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_retrace_jacrev(self):
        # [FUNCTION DOC] Same retrace check for torch.func.jacrev.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Applies jacrev of fn to x.
            return torch.func.jacrev(fn)(x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y = wrapper_fn(x)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_retrace_jacfwd(self):
        # [FUNCTION DOC] Same retrace check for torch.func.jacfwd.
        # Recompile trick doesn't work with dynamic shapes
        if check_dynamic_shape_capture():
            return

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x.
            return x.sin()

        def wrapper_fn(x):
            # [FUNCTION DOC] HELPER: Applies jacfwd of fn to x.
            return torch.func.jacfwd(fn)(x)

        x = torch.randn(3, 3)
        eager = EagerAndRecordGraphs()
        opt_f = torch.compile(wrapper_fn, backend=eager, fullgraph=True)
        y = wrapper_fn(x)
        opt_y = opt_f(x)
        self.assertEqual(y, opt_y)
        first_graph = normalize_gm(eager.graphs[0].print_readable(False))

        d = {}
        exec(first_graph, globals(), d)
        retraced = torch.compile(d["GraphModule"], backend=eager, fullgraph=True)
        retraced_out = retraced()(x)[0]
        self.assertEqual(y, retraced_out)
        retraced_graph = normalize_gm(eager.graphs[0].print_readable(False))
        self.assertEqual(first_graph, retraced_graph)

    def test_context_wrapping_variable_rejects_dict_target_values(self):
        # [FUNCTION DOC] Checks that constructing ContextWrappingVariable with a dict for
        # target_values raises TypeError with a message saying target_values must be None
        # or a Sequence.
        # target_values must be None or a Sequence; a dict silently iterates
        # only its keys, which is the bug pattern we want to catch early.
        from torch._dynamo.variables.ctx_manager import ContextWrappingVariable

        with self.assertRaisesRegex(
            TypeError,
            "ContextWrappingVariable.target_values must be None or a Sequence",
        ):
            ContextWrappingVariable(target_values={"key": "val"})


class CUDACtxManagerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for CUDA-specific context managers under Dynamo (mem pools, streams,
    # events, devices, autocast); most tests skip unless CUDA (and sometimes Triton or multiple
    # devices) is available.
    def test_cuda_use_mem_pool_fx_cse_preserves_distinct_contexts(self):
        # [FUNCTION DOC] Builds an FX graph with two identical add nodes that carry different
        # mempool metadata in their custom meta and checks that fx_graph_cse keeps the nodes
        # distinct instead of merging them (asserted on the next lines).
        from torch._functorch.compile_utils import fx_graph_cse

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        add_a = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        add_a.meta["custom"] = {"mempool": 0, "mempool_device": 0}
        add_b = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        add_b.meta["custom"] = {"mempool": 1, "mempool_device": 0}
        graph.output((add_a, add_b))

        cse_graph = fx_graph_cse(graph)
        add_nodes = [
            node
            for node in cse_graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.add.Tensor
        ]
        self.assertEqual(len(add_nodes), 2)

    def _check_cuda_use_mem_pool(self, backend=None):
        # [FUNCTION DOC] HELPER: Shared body for the mem pool tests. Empties the CUDA cache,
        # creates a MemPool, compiles a function that allocates a ones tensor plus one inside
        # use_mem_pool (using the given backend, or inductor when None, always fullgraph), and
        # checks the result is a vector of twos, the pool's memory snapshot is non-empty, and the
        # pool use count is 1 afterwards. Synchronizes CUDA.
        torch.cuda.empty_cache()

        def fn(pool):
            # [FUNCTION DOC] HELPER: Inside use_mem_pool(pool) creates a 16 element CUDA ones
            # tensor and returns it plus one.
            with torch.cuda.use_mem_pool(pool):
                return torch.ones(16, device="cuda") + 1

        pool = torch.cuda.MemPool()
        opt_fn = (
            torch.compile(fn, backend=backend, fullgraph=True)
            if backend is not None
            else torch.compile(fn, backend="inductor", fullgraph=True)
        )
        res = opt_fn(pool)
        torch.cuda.synchronize()
        self.assertEqual(res, torch.full((16,), 2.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_use_mem_pool_context_manager(self):
        # [FUNCTION DOC] Runs the mem pool check with the eager backend; skipped without CUDA.
        self._check_cuda_use_mem_pool(backend="eager")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_mem_pool_id(self):
        # [FUNCTION DOC] Checks that reading the id attribute of a MemPool inside a fullgraph
        # eager-compiled function returns the same id as eager.
        def fn(pool):
            # [FUNCTION DOC] HELPER: Returns the id of the given pool.
            return pool.id

        pool = torch.cuda.MemPool()
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(pool), pool.id)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_context_manager_inductor(self):
        # [FUNCTION DOC] Runs the mem pool check with inductor; requires CUDA and Triton.
        self._check_cuda_use_mem_pool()

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_cpp_wrapper_unsupported(self):
        # [FUNCTION DOC] Checks that inductor with the C++ wrapper enabled (plus Triton autotune
        # options) raises BackendCompilerFailed saying use_mem_pool is not supported with C++
        # wrapper codegen. Requires CUDA and Triton.
        def fn(pool):
            # [FUNCTION DOC] HELPER: Allocates a 16 element CUDA ones tensor inside
            # use_mem_pool(pool) and returns it.
            with torch.cuda.use_mem_pool(pool):
                return torch.ones(16, device="cuda")

        pool = torch.cuda.MemPool()
        with (
            torch._inductor.config.patch(
                {
                    "cpp_wrapper": True,
                    "triton.autotune_at_compile_time": True,
                    "triton.autotune_with_sample_inputs": True,
                }
            ),
            self.assertRaisesRegex(
                torch._dynamo.exc.BackendCompilerFailed,
                "torch.cuda.use_mem_pool is not supported with C\\+\\+ wrapper codegen",
            ),
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(pool)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_realize_at_boundary_inductor(self):
        # [FUNCTION DOC] Checks with inductor that a value computed inside use_mem_pool is realized
        # at the pool boundary: result is threes, the pool has allocations and its use count is 1.
        torch.cuda.empty_cache()

        def fn(pool, inp):
            # [FUNCTION DOC] HELPER: Adds one to inp inside use_mem_pool(pool), then adds one more
            # outside, returns it.
            with torch.cuda.use_mem_pool(pool):
                x = inp + 1
            return x + 1

        pool = torch.cuda.MemPool()
        inp = torch.ones(16, device="cuda")
        res = torch.compile(fn, backend="inductor", fullgraph=True)(pool, inp)
        torch.cuda.synchronize()
        self.assertEqual(res, torch.full((16,), 3.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_fallback_alloc_inductor(self):
        # [FUNCTION DOC] Checks with inductor that a fallback (non-generated) kernel, histc,
        # allocates inside the mem pool: result is the expected histogram of four per bin, pool
        # snapshot non-empty, use count 1.
        torch.cuda.empty_cache()

        def fn(pool, inp):
            # [FUNCTION DOC] HELPER: Returns histc of inp with 4 bins in range 0 to 4
            # computed inside use_mem_pool(pool).
            with torch.cuda.use_mem_pool(pool):
                return torch.histc(inp, bins=4, min=0, max=4)

        pool = torch.cuda.MemPool()
        inp = torch.arange(16, device="cuda", dtype=torch.float32) % 4
        res = torch.compile(fn, backend="inductor", fullgraph=True)(pool, inp)
        torch.cuda.synchronize()
        self.assertEqual(res, torch.full((4,), 4.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_foreach_alloc_inductor(self):
        # [FUNCTION DOC] Checks with inductor that a foreach add on a list of two tensors allocates
        # inside the mem pool and produces the expected values (twos and threes); pool snapshot
        # non-empty, use count 1.
        torch.cuda.empty_cache()

        def fn(pool, xs):
            # [FUNCTION DOC] HELPER: Returns the foreach add of one to the list
            # xs inside use_mem_pool(pool).
            with torch.cuda.use_mem_pool(pool):
                return torch._foreach_add(xs, 1.0)

        pool = torch.cuda.MemPool()
        xs = [
            torch.ones(64, device="cuda"),
            torch.full((64,), 2.0, device="cuda"),
        ]
        res = torch.compile(fn, backend="inductor", fullgraph=True)(pool, xs)
        torch.cuda.synchronize()
        self.assertEqual(res[0], torch.full((64,), 2.0, device="cuda"))
        self.assertEqual(res[1], torch.full((64,), 3.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_forced_extern_multi_template_inductor(self):
        # [FUNCTION DOC] Checks matmul inside a mem pool compiled with max-autotune without
        # cudagraphs while an inductor test config forces an extern kernel in the multi-
        # template choice and native matmul is off; result matches eager within tolerance and
        # the pool is used once.
        torch.cuda.empty_cache()

        def fn(pool, x, y):
            # [FUNCTION DOC] HELPER: Returns x matmul y computed inside use_mem_pool(pool).
            with torch.cuda.use_mem_pool(pool):
                return x @ y

        pool = torch.cuda.MemPool()
        x = torch.randn(32, 32, device="cuda")
        y = torch.randn(32, 32, device="cuda")
        with torch._inductor.config.patch(
            {
                "test_configs.force_extern_kernel_in_multi_template": True,
                "triton.native_matmul": False,
            }
        ):
            res = torch.compile(
                fn,
                backend="inductor",
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
            )(pool, x, y)
        torch.cuda.synchronize()
        self.assertEqual(res, x @ y, atol=1e-4, rtol=1e-4)
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_nested_inductor(self):
        # [FUNCTION DOC] Checks nested mem pools with inductor: the outer-scoped value is 5 and the
        # inner-scoped value is 3; both pools' snapshots are non-empty and each use count is 1.
        torch.cuda.empty_cache()

        def fn(outer_pool, inner_pool, inp):
            # [FUNCTION DOC] HELPER: Adds 1 inside the outer pool, 2 inside a nested inner pool,
            # then adds 3 to the outer value; returns both values.
            with torch.cuda.use_mem_pool(outer_pool):
                outer = inp + 1
                with torch.cuda.use_mem_pool(inner_pool):
                    inner = inp + 2
                outer = outer + 3
            return outer, inner

        outer_pool = torch.cuda.MemPool()
        inner_pool = torch.cuda.MemPool()
        inp = torch.ones(16, device="cuda")
        outer, inner = torch.compile(fn, backend="inductor", fullgraph=True)(
            outer_pool, inner_pool, inp
        )
        torch.cuda.synchronize()
        self.assertEqual(outer, torch.full((16,), 5.0, device="cuda"))
        self.assertEqual(inner, torch.full((16,), 3.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(outer_pool.id)), 0)
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(inner_pool.id)), 0)
        self.assertEqual(outer_pool.use_count(), 1)
        self.assertEqual(inner_pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_distinct_pools_not_cse_inductor(self):
        # [FUNCTION DOC] Checks that two identical additions made inside different mem pools are
        # not merged by common-subexpression elimination in inductor: both pools get allocations
        # and the result is fours.
        torch.cuda.empty_cache()

        def fn(pool_a, pool_b, inp):
            # [FUNCTION DOC] HELPER: Computes inp plus one inside pool_a and again inside pool_b
            # and returns the sum of the two.
            with torch.cuda.use_mem_pool(pool_a):
                a = inp + 1
            with torch.cuda.use_mem_pool(pool_b):
                b = inp + 1
            return a + b

        pool_a = torch.cuda.MemPool()
        pool_b = torch.cuda.MemPool()
        inp = torch.ones(16, device="cuda")
        res = torch.compile(fn, backend="inductor", fullgraph=True)(pool_a, pool_b, inp)
        torch.cuda.synchronize()
        self.assertEqual(res, torch.full((16,), 4.0, device="cuda"))
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool_a.id)), 0)
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool_b.id)), 0)
        self.assertEqual(pool_a.use_count(), 1)
        self.assertEqual(pool_b.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_reduce_overhead_cudagraph_asserts_inductor(self):
        # [FUNCTION DOC] Checks mem pool use with the reduce-overhead (CUDA graph) mode and the
        # slow-path cudagraph assertions config enabled: two successive calls give results equal to
        # eager matmul-relu, the pool is populated and its use count is 1.
        torch.cuda.empty_cache()

        def fn(pool, x, w):
            # [FUNCTION DOC] HELPER: Returns relu of x matmul w computed inside use_mem_pool(pool).
            with torch.cuda.use_mem_pool(pool):
                return (x @ w).relu()

        pool = torch.cuda.MemPool()
        x = torch.randn(64, 64, device="cuda")
        w = torch.randn(64, 64, device="cuda")
        with torch._inductor.config.patch({"triton.slow_path_cudagraph_asserts": True}):
            opt_fn = torch.compile(
                fn, backend="inductor", mode="reduce-overhead", fullgraph=True
            )
            res = opt_fn(pool, x, w)
            res_second = opt_fn(pool, x, w)
        torch.cuda.synchronize()
        self.assertEqual(res, (x @ w).relu(), atol=1e-4, rtol=1e-4)
        self.assertEqual(res_second, res)
        self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @requires_cuda_and_triton
    def test_cuda_use_mem_pool_stream_order_inductor(self):
        # [FUNCTION DOC] Checks that for both nesting orders of use_mem_pool and cuda.stream the
        # compiled results equal eager and that the generated inductor source always enters the mem
        # pool before the stream block. Uses run_and_get_code and subTest per function.
        from torch._inductor.utils import run_and_get_code

        torch.cuda.empty_cache()

        def pool_outer(pool, stream, inp):
            # [FUNCTION DOC] HELPER: Enters use_mem_pool then a stream block and returns
            # inp plus one.
            with torch.cuda.use_mem_pool(pool):
                with torch.cuda.stream(stream):
                    return inp + 1

        def stream_outer(pool, stream, inp):
            # [FUNCTION DOC] HELPER: Enters a stream block then use_mem_pool and returns
            # inp plus two.
            with torch.cuda.stream(stream):
                with torch.cuda.use_mem_pool(pool):
                    return inp + 2

        inp = torch.ones(16, device="cuda")
        stream = torch.cuda.Stream()
        for fn in (pool_outer, stream_outer):
            with self.subTest(fn=fn.__name__):
                pool = torch.cuda.MemPool()
                expected = fn(pool, stream, inp)
                actual, source_codes = run_and_get_code(
                    torch.compile(fn, backend="inductor", fullgraph=True),
                    pool,
                    stream,
                    inp,
                )
                torch.cuda.synchronize()
                self.assertEqual(actual, expected)
                self.assertGreater(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
                self.assertEqual(pool.use_count(), 1)

                source = "\n".join(source_codes)
                # The compiled wrapper deliberately normalizes to pool-outer,
                # stream-inner ordering so allocators with set_context/pop_context
                # side effects see a consistent pool lifetime.
                mempool_enter = source.index("with torch.cuda.use_mem_pool(")
                stream_enter = source.index("with stream")
                self.assertLess(mempool_enter, stream_enter)

    @requires_cuda_and_triton
    @unittest.skipIf(torch.cuda.device_count() < 2, "requires multiple cuda devices")
    def test_cuda_use_mem_pool_device_none_resolves_at_entry_inductor(self):
        # [FUNCTION DOC] Checks that a pool created while another device (index 1) is current
        # resolves the device at entry: the compiled function, allocating on cuda:0, returns ones
        # on device 0 and the pool snapshot is empty; skipped unless two CUDA devices exist and
        # Triton is available.
        torch.cuda.empty_cache()

        def fn(pool):
            # [FUNCTION DOC] HELPER: Allocates a CUDA ones tensor on device 0 inside
            # use_mem_pool(pool) and returns it.
            with torch.cuda.use_mem_pool(pool):
                return torch.ones(16, device="cuda:0")

        with torch.cuda.device(1):
            pool = torch.cuda.MemPool()
            res = torch.compile(fn, backend="inductor", fullgraph=True)(pool)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        self.assertEqual(res, torch.ones(16, device="cuda:0"))
        self.assertEqual(len(torch.cuda.memory.memory_snapshot(pool.id)), 0)
        self.assertEqual(pool.use_count(), 1)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_use_mem_pool_invalid_signatures(self):
        # [FUNCTION DOC] Checks that malformed calls to use_mem_pool raise Dynamo Unsupported
        # errors under eager fullgraph compile, with messages for a missing pool, too many
        # positional args, an unexpected keyword, and duplicate pool or device values. Each case
        # runs in a subTest.
        pool = torch.cuda.MemPool()

        def missing(_pool):
            # [FUNCTION DOC] HELPER: Calls use_mem_pool with no arguments and returns a
            # CUDA ones tensor.
            with torch.cuda.use_mem_pool():
                return torch.ones(1, device="cuda")

        def too_many(pool):
            # [FUNCTION DOC] HELPER: Calls use_mem_pool with three positional arguments.
            with torch.cuda.use_mem_pool(pool, None, None):
                return torch.ones(1, device="cuda")

        def unexpected_kwarg(pool):
            # [FUNCTION DOC] HELPER: Calls use_mem_pool with an unexpected foo keyword.
            with torch.cuda.use_mem_pool(pool, foo=None):
                return torch.ones(1, device="cuda")

        def duplicate_pool(pool):
            # [FUNCTION DOC] HELPER: Passes pool both positionally and as a keyword.
            with torch.cuda.use_mem_pool(pool, pool=pool):
                return torch.ones(1, device="cuda")

        def duplicate_device(pool):
            # [FUNCTION DOC] HELPER: Passes device both positionally and as a keyword.
            with torch.cuda.use_mem_pool(pool, None, device=None):
                return torch.ones(1, device="cuda")

        cases = [
            (missing, "missing 1 required positional argument: 'pool'"),
            (
                too_many,
                "takes from 1 to 2 positional arguments but 3 were given",
            ),
            (unexpected_kwarg, "unexpected keyword argument 'foo'"),
            (duplicate_pool, "multiple values for argument 'pool'"),
            (duplicate_device, "multiple values for argument 'device'"),
        ]
        for fn, msg in cases:
            with (
                self.subTest(msg=msg),
                self.assertRaisesRegex(torch._dynamo.exc.Unsupported, msg),
            ):
                torch.compile(fn, backend="eager", fullgraph=True)(pool)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager1(self):
        # [FUNCTION DOC] Checks a CUDA stream context manager under fullgraph: a new stream waits
        # on the current stream, runs relu in the stream context, then the current stream waits
        # back. Results equal eager, one frame, 9 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a stream, multiplies and adds, syncs streams, runs
            # relu in the stream, syncs back, adds one and returns the cosine.
            s = torch.cuda.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)
            with torch.cuda.stream(s):
                x = torch.relu(x)
            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """9""")

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_across_graph_break(self):
        # [FUNCTION DOC] Expected-failure test (issue 118204): a stream context created before a
        # print graph break and entered afterwards should compile into two frames with 9 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a stream and does arithmetic, prints (graph break),
            # creates the stream context, waits, runs relu in the context, syncs, adds one and
            # takes the cosine.
            s = torch.cuda.Stream()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            tcs = torch.cuda.stream(s)
            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)

            with tcs:
                x = torch.relu(x)

            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_context_manager2(self):
        # [FUNCTION DOC] Expected-failure test (issue 118204): multiple streams including a context
        # on the current stream and a second new stream in one fullgraph compile; expects one frame
        # and 18 ops.
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Runs mul, add, relu in a stream context for the given stream,
            # relu on the current stream context, relu on a newly created second stream, then add
            # one and cosine.
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            current_stream = torch.cuda.current_stream()
            s.wait_stream(current_stream)

            with torch.cuda.stream(s):
                x = torch.relu(x)

            current_stream.wait_stream(s)
            with torch.cuda.stream(current_stream):
                x = torch.relu(x)

            s2 = torch.cuda.Stream()
            s2.wait_stream(current_stream)
            with torch.cuda.stream(s2):
                x = torch.relu(x)

            current_stream.wait_stream(s2)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        s = torch.cuda.Stream()
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_method(self):
        # [FUNCTION DOC] Checks stream methods (wait_stream, query, synchronize) work in a
        # fullgraph compile: results equal eager, one frame, 15 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Mixes arithmetic with a new stream context, wait_stream, query
            # and synchronize calls on the current and new stream, then relu and cosine.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            new_stream = torch.cuda.Stream()
            cur_stream = torch.cuda.current_stream()
            new_stream.wait_stream(cur_stream)

            with torch.cuda.stream(new_stream):
                x = torch.sin(x)
                x = torch.add(x, 3)

            cur_stream.wait_stream(new_stream)

            x = torch.add(x, 4)
            cur_stream.query()
            cur_stream.synchronize()

            with torch.cuda.stream(new_stream):
                x = torch.add(x, 5)
            new_stream.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """15""")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_constant(self):
        # [FUNCTION DOC] Checks a CUDA stream object can be compared to None-ness and to a string
        # constant under fullgraph: both compiled functions give the same result as eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds, gets the current stream, and returns x plus one if it is
            # not None, otherwise x minus one.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            if cur_stream is not None:
                return x + 1
            return x - 1

        def fn2(x):
            # [FUNCTION DOC] HELPER: Same as fn but compares the current stream against a constant
            # string with inequality.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            if cur_stream != "const_str":
                return x + 1
            return x - 1

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        res2 = opt_fn2(x)
        self.assertEqual(ref, res)
        self.assertEqual(ref, res2)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_stream_compared_with_stream(self):
        # [FUNCTION DOC] Checks guards on stream equality: with two distinct streams the compiled
        # function recompiles when the inputs go from different streams to the same stream (2
        # frames), and in the reverse order after a Dynamo reset also needs 2 frames. Results equal
        # eager each time.
        def fn(x, s0, s1):
            # [FUNCTION DOC] HELPER: Returns x plus one when the two streams are equal, else
            # x minus one.
            if s0 == s1:
                return x + 1
            else:
                return x - 1

        s0 = torch.cuda.Stream()
        s1 = torch.cuda.Stream()
        x = torch.randn(2, 2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref0, res0)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref1, res1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref1, res1)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref0, res0)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_cuda_event_reconstruct(self):
        # [FUNCTION DOC] Skipped test (external events unsupported, issue 167257): a CUDA event
        # created inside the compiled function is returned; expects one frame and 3 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an event, multiplies and adds, returns the tensor
            # and the event.
            e = torch.cuda.Event()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            return x, e

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_cuda_event_across_graph_break(self):
        # [FUNCTION DOC] Skipped test (issue 167257): event recorded before a graph break and
        # waited on afterwards; expects two frames and 10 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates and records an event, does arithmetic, prints, waits
            # on the event with the current stream, adds one, cosine; returns tensor and event.
            e = torch.cuda.Event()
            e.record()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            torch.cuda.current_stream().wait_event(e)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x, e

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 10)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_cuda_event_created_outside_of_graph(self):
        # [FUNCTION DOC] Skipped test (issue 167257): an event created outside the compiled
        # function and recorded on a user stream is waited on inside the compiled function across
        # 10 iterations; expects one frame and 4 ops.
        user_stream = torch.cuda.Stream()
        event = torch.cuda.Event()
        foo = torch.empty((2, 2), device="cuda")

        def func(foo):
            # [FUNCTION DOC] HELPER: Waits on the outer event and returns the input plus one
            # and the event.
            event.wait()
            return foo + 1, event

        x = torch.randn((1024, 1024), device="cuda")
        cnts = torch._dynamo.testing.CompileCounter()

        def run_iters(fn, compile=False):
            # [FUNCTION DOC] HELPER: Runs fn ten times (compiled with the counting backend when
            # compile is true), each time doing a matmul into foo on user_stream and recording the
            # event, then synchronizing the current stream; returns the last output.
            if compile:
                fn = torch.compile(fn, backend=cnts)
            for _ in range(10):
                with torch.cuda.stream(user_stream):
                    torch.mm(x, x, out=foo)
                    event.record()
                out = fn(foo)
                torch.cuda.current_stream().synchronize()
            return out

        ref = run_iters(func, compile=False)
        res = run_iters(func, compile=True)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_cuda_event_method_create_stream_outside_of_compile(self):
        # [FUNCTION DOC] Skipped test (issue 167257): event methods (record_event, query,
        # wait_event, Event record/wait/synchronize) with streams created outside the compile;
        # expects one frame and 16 ops.
        def fn(x, cur_stream, new_stream):
            # [FUNCTION DOC] HELPER: Arithmetic with record_event/query on the current stream, a
            # wait_event on the new stream, a stream context add, a new Event recorded and waited,
            # synchronize, then relu and cosine.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with torch.cuda.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.cuda.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        cur_stream = torch.cuda.current_stream()
        new_stream = torch.cuda.Stream()
        ref = fn(x, cur_stream, new_stream)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, cur_stream, new_stream)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """16""")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_event_method(self):
        # [FUNCTION DOC] Checks event methods with the generic torch.Event inside fullgraph
        # compile, with streams created inside the function: results equal eager, one
        # frame, 17 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Same event/stream sequence as the previous test but creates
            # the new stream inside the function and uses torch.Event.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.cuda.current_stream()
            new_stream = torch.cuda.Stream()

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with torch.cuda.stream(new_stream):
                x = torch.add(x, 4)

            new_event = torch.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """17""")

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_device(self):
        # [FUNCTION DOC] Checks the cuda.device context manager (switching to device index minus
        # one) compiles under eager fullgraph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Enters a cuda.device context for the previous device index and
            # returns sin of x plus one.
            with torch.cuda.device(x.device.index - 1):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda__exchange_device_args(self):
        # [FUNCTION DOC] Checks that torch.cuda._exchange_device called with no arguments, too many
        # positional arguments, or an unexpected keyword raises Dynamo Unsupported, and that the
        # current CUDA device is unchanged afterward.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(args, kwargs):
            # [FUNCTION DOC] HELPER: Calls the private _exchange_device with the given positional
            # and keyword arguments.
            torch.cuda._exchange_device(*args, **kwargs)

        initial_dev = torch.cuda.current_device()
        for args, kwargs in (((), ()), ((0, 0), ()), ((), ("kwarg",))):
            self.assertRaises(torch._dynamo.exc.Unsupported, fn, args, kwargs)
            self.assertEqual(torch.cuda.current_device(), initial_dev)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast(self):
        # [FUNCTION DOC] Exports a module doing two matmuls under CUDA bfloat16 autocast and checks
        # the exported output device (cuda index 0) and dtype (bfloat16) match eager. Skips when
        # bfloat16 is unsupported.
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random CUDA
            # tensors under CUDA bfloat16 autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates three random 8x8 CUDA tensors and runs two chained
                # matmuls in bfloat16 autocast; ignores x.
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.bfloat16)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_amp_autocast(self):
        # [FUNCTION DOC] Exports a module that does a matmul under CUDA autocast with float64 and
        # checks the result is float64 on cuda index 0 matching eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs a matmul of random CUDA tensors
            # under CUDA float64 autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Random CUDA inputs and a matmul inside float64 autocast;
                # returns the result.
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    c_float64 = torch.mm(a_float32, b_float32)
                return c_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, "cuda")
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_float64(self):
        # [FUNCTION DOC] Exports a module with two chained matmuls under CUDA float64 autocast;
        # checks device index and float64 dtype match eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random CUDA
            # tensors under CUDA float64 autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Three random CUDA inputs and two chained matmuls
                # under float64 autocast.
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast(device_type="cuda", dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_device(self):
        # [FUNCTION DOC] Exports a module that uses autocast with just the device string (default
        # CUDA dtype); expects float16 output on cuda index 0 matching eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random CUDA
            # tensors under default CUDA autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Three random CUDA tensors and two chained matmuls under
                # default CUDA autocast.
                a_float32 = torch.rand((8, 8), device="cuda")
                b_float32 = torch.rand((8, 8), device="cuda")
                d_float32 = torch.rand((8, 8), device="cuda")

                with torch.autocast("cuda"):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float16)

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_arguments_binding(self):
        # [FUNCTION DOC] Checks that autocast arguments are bound correctly when only device_type
        # and enabled=False are passed, for CUDA and CPU; compiled results equal eager. The test
        # requires CUDA but the inputs are CPU tensors.
        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one and takes sin inside a disabled CUDA autocast.
            with torch.autocast(device_type="cuda", enabled=False):
                x = torch.sin(x + 1)
            return x

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds one and takes cos inside a disabled CPU autocast.
            with torch.autocast(device_type="cpu", enabled=False):
                x = torch.cos(x + 1)
            return x

        x = torch.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)
        opt_f1 = torch.compile(backend="eager")(f1)
        opt_f2 = torch.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_autocast_decorator(self):
        # [FUNCTION DOC] Checks autocast used as a decorator (via torch.amp.autocast,
        # torch.autocast for cuda, and for cpu, all float16) on wrapped matmul functions
        # under fullgraph eager compile; results equal eager and the two CUDA-decorated
        # outputs are float16.
        def autocast_func(orig_func):
            # [FUNCTION DOC] HELPER: Returns a function wrapping orig_func in torch.amp.autocast
            # for CUDA float16.
            @torch.amp.autocast(device_type="cuda", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cuda(orig_func):
            # [FUNCTION DOC] HELPER: Returns orig_func wrapped with torch.autocast
            # for CUDA float16.
            @torch.autocast(device_type="cuda", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cpu(orig_func):
            # [FUNCTION DOC] HELPER: Returns orig_func wrapped with torch.autocast for CPU float16.
            @torch.autocast(device_type="cpu", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Returns the matmul of a and b.
            return torch.mm(a, b)

        mm_float16 = autocast_func(mm)
        mm_float16_cuda = autocast_func_cuda(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the three autocast-wrapped matmuls on a and b and
            # returns the three results as a tuple.
            return mm_float16(a, b), mm_float16_cuda(a, b), mm_float16_cpu(a, b)

        a_float32 = torch.rand((8, 8), device="cuda")
        b_float32 = torch.rand((8, 8), device="cuda")

        ref = fn(a_float32, b_float32)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(a_float32, b_float32)
        self.assertTrue(same(ref, res))
        self.assertTrue(res[0].dtype == torch.float16)
        self.assertTrue(res[1].dtype == torch.float16)


class CtxManagerTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic stream/event context manager tests, instantiated per accelerator
    # device type by instantiate_device_type_tests; every test method receives a device argument.
    def test_stream_context_manager1(self, device):
        # [FUNCTION DOC] Checks the generic torch.Stream context manager under fullgraph: a stream
        # created on the device waits on the current accelerator stream, runs relu inside the
        # stream, and syncs back; results equal eager, one frame, 9 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a stream on the device, mul/add, waits, relu inside
            # the stream, waits back, adds one, cosine.
            s = torch.Stream(device=device)
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            current_stream = torch.accelerator.current_stream()
            s.wait_stream(current_stream)
            with s:
                x = torch.relu(x)
            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """9""")

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    def test_stream_across_graph_break(self, device):
        # [FUNCTION DOC] Expected-failure test (issue 118204): generic torch.Stream used across a
        # print graph break (stream created before, entered after); expects 2 frames and 9 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a stream on the device, arithmetic, print, waits on
            # the current accelerator stream, relu inside the stream, waits back, add one, cosine.
            s = torch.Stream(device=device)
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            current_stream = torch.accelerator.current_stream()
            s.wait_stream(current_stream)

            with s:
                x = torch.relu(x)

            current_stream.wait_stream(s)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 9)

    @unittest.expectedFailure  # https://github.com/pytorch/pytorch/issues/118204
    def test_stream_context_manager2(self, device):
        # [FUNCTION DOC] Expected-failure test (issue 118204): multiple generic streams (argument
        # stream, the current stream and a second new stream) in one fullgraph compile; expects 1
        # frame and 18 ops.
        def fn(x, s):
            # [FUNCTION DOC] HELPER: relu applied inside the context of stream s, of the
            # current stream, and of a new stream s2, with wait_stream calls in between, then
            # add one and cosine.
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            current_stream = torch.accelerator.current_stream()
            s.wait_stream(current_stream)

            with s:
                x = torch.relu(x)

            current_stream.wait_stream(s)
            with current_stream:
                x = torch.relu(x)

            s2 = torch.Stream(device=device)
            s2.wait_stream(current_stream)
            with s2:
                x = torch.relu(x)

            current_stream.wait_stream(s2)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        s = torch.Stream(device=device)
        ref = fn(x, s)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, s)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 18)

    def test_stream_method(self, device):
        # [FUNCTION DOC] Checks generic stream methods (wait_stream, query, synchronize) in
        # fullgraph compile; results equal eager, 1 frame, 15 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Arithmetic with a new device stream used as a context,
            # wait_stream, query and synchronize on current and new streams, relu and cosine.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            new_stream = torch.Stream(device=device)
            cur_stream = torch.accelerator.current_stream()
            new_stream.wait_stream(cur_stream)

            with new_stream:
                x = torch.sin(x)
                x = torch.add(x, 3)

            cur_stream.wait_stream(new_stream)

            x = torch.add(x, 4)
            cur_stream.query()
            cur_stream.synchronize()

            with new_stream:
                x = torch.add(x, 5)
            new_stream.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """15""")

    def test_stream_compared_with_constant(self, device):
        # [FUNCTION DOC] Checks the current accelerator stream compared to None and to a string
        # constant is traced under fullgraph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if the current accelerator stream is not
            # None, else x minus one.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.accelerator.current_stream()
            if cur_stream is not None:
                return x + 1
            return x - 1

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if the current stream differs from a
            # constant string, else x minus one.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.accelerator.current_stream()
            if cur_stream != "const_str":
                return x + 1
            return x - 1

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        opt_fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        res2 = opt_fn2(x)
        self.assertEqual(ref, res)
        self.assertEqual(ref, res2)

    def test_stream_compared_with_stream(self, device):
        # [FUNCTION DOC] Device-generic version of the stream equality guard test: compiling with
        # different streams then equal streams (and reversed after a Dynamo reset) needs two frames
        # each way; results equal eager.
        def fn(x, s0, s1):
            # [FUNCTION DOC] HELPER: Returns x plus one when the two streams are equal, else
            # x minus one.
            if s0 == s1:
                return x + 1
            else:
                return x - 1

        s0 = torch.Stream(device=device)
        s1 = torch.Stream(device=device)
        x = torch.randn(2, 2)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref0, res0)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref1, res1)

        torch._dynamo.reset()
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)

        ref1 = fn(x, s1, s1)
        res1 = opt_fn(x, s1, s1)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(ref1, res1)

        ref0 = fn(x, s0, s1)
        res0 = opt_fn(x, s0, s1)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(ref0, res0)

    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_event_reconstruct(self, device):
        # [FUNCTION DOC] Skipped test (issue 167257): an Event created in the compiled function is
        # returned; expects 1 frame and 3 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an Event on the device, multiplies and adds, returns
            # tensor and event.
            e = torch.Event(device=device)
            x = torch.mul(x, 5)
            x = torch.add(x, 2)
            return x, e

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_event_across_graph_break(self, device):
        # [FUNCTION DOC] Skipped test (issue 167257): event recorded, graph break, then waited on
        # by the current accelerator stream; expects 2 frames and 10 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates and records an Event, arithmetic, prints, current
            # stream waits on the event, add one, cosine; returns tensor and event.
            e = torch.Event(device=device)
            e.record()
            x = torch.mul(x, 5)
            x = torch.add(x, 2)

            print("foo")

            torch.accelerator.current_stream().wait_event(e)
            x = torch.add(x, 1)
            x = torch.cos(x)
            return x, e

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 10)

    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_event_created_outside_of_graph(self, device):
        # [FUNCTION DOC] Skipped test (issue 167257): event and stream created outside the compile,
        # used for ten iterations; expects 1 frame and 4 ops.
        user_stream = torch.Stream(device=device)
        event = torch.Event(device=device)
        foo = torch.empty((2, 2), device=device)

        def func(foo):
            # [FUNCTION DOC] HELPER: Waits on the outer event and returns the input plus one
            # and the event.
            event.wait()
            return foo + 1, event

        x = torch.randn((1024, 1024), device=device)
        cnts = torch._dynamo.testing.CompileCounter()

        def run_iters(fn, compile=False):
            # [FUNCTION DOC] HELPER: Runs fn ten times (compiled when compile is true) with a
            # matmul into foo on user_stream, event record and current stream synchronize each
            # time; returns the last output.
            if compile:
                fn = torch.compile(fn, backend=cnts)
            for _ in range(10):
                with user_stream:
                    torch.mm(x, x, out=foo)
                    event.record()
                out = fn(foo)
                torch.accelerator.current_stream().synchronize()
            return out

        ref = run_iters(func, compile=False)
        res = run_iters(func, compile=True)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 4)

    @unittest.skip(
        "Will not support external events for now: https://github.com/pytorch/pytorch/issues/167257"
    )
    def test_event_method_create_stream_outside_of_compile(self, device):
        # [FUNCTION DOC] Skipped test (issue 167257): event methods with streams created outside
        # the compile; expects 1 frame and 16 ops.
        def fn(x, cur_stream, new_stream):
            # [FUNCTION DOC] HELPER: Arithmetic with record_event/query on cur_stream, wait_event
            # on new_stream, a new Event recorded and waited, synchronize, relu and cosine.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with new_stream:
                x = torch.add(x, 4)

            new_event = torch.Event(device=device)
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        cur_stream = torch.accelerator.current_stream()
        new_stream = torch.Stream(device=device)
        ref = fn(x, cur_stream, new_stream)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x, cur_stream, new_stream)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """16""")

    def test_event_method(self, device):
        # [FUNCTION DOC] Checks generic Event methods in fullgraph compile with streams created
        # inside: results equal eager, 1 frame, 17 ops.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a new stream, records an event on the current stream,
            # makes the new stream wait on it, adds inside the new stream, records, waits and
            # synchronizes a second Event, then relu and cosine.
            x = torch.mul(x, 1)
            x = torch.add(x, 2)

            cur_stream = torch.accelerator.current_stream()
            new_stream = torch.Stream(device=device)

            x = torch.add(x, 3)

            event = cur_stream.record_event()
            event.query()

            new_stream.wait_event(event)
            with new_stream:
                x = torch.add(x, 4)

            new_event = torch.Event()
            new_event.record(new_stream)

            new_event.wait(cur_stream)
            x = torch.add(x, 5)

            new_event.synchronize()

            x = torch.relu(x)
            x = torch.cos(x)
            return x

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)
        self.assertExpectedInline(str(cnts.op_count), """17""")

    def test_device_context(self, device):
        # [FUNCTION DOC] Checks the device module's device context manager (index minus
        # one) compiles under eager fullgraph and matches eager. The device module is
        # found by torch.get_device_module.
        device_mod = torch.get_device_module(device)

        def fn(x):
            # [FUNCTION DOC] HELPER: Enters the device context for the previous device index and
            # returns sin of x plus one.
            with device_mod.device(x.device.index - 1):
                x = torch.sin(x + 1)
            return x

        x = torch.randn((2, 2), device=device)
        ref = fn(x)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @onlyCUDA
    def test_autocast_bf16(self, device):
        # [FUNCTION DOC] CUDA-only: exports a module with two chained matmuls under bfloat16
        # autocast on the given device and checks device type, index 0 and dtype bfloat16 match
        # eager. Skips if bfloat16 is unsupported.
        if not torch.cuda.is_bf16_supported():
            raise unittest.SkipTest("requires bf16")

        device_type = torch.device(device).type

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random device
            # tensors under bfloat16 autocast for the closed-over device type; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Random 8x8 tensors on the device and two chained matmuls
                # inside bfloat16 autocast.
                a_float32 = torch.rand((8, 8), device=device)
                b_float32 = torch.rand((8, 8), device=device)
                d_float32 = torch.rand((8, 8), device=device)

                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    e_float16 = torch.mm(a_float32, b_float32)
                    f_float16 = torch.mm(d_float32, e_float16)
                return f_float16

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, device_type)
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.bfloat16)

    # autocast with float64 not support on XPU
    @onlyCUDA
    def test_amp_autocast(self, device):
        # [FUNCTION DOC] CUDA-only: exports a module with a matmul under float64 autocast; checks
        # device type, index 0 and float64 dtype match eager (float64 autocast is unsupported on
        # XPU, per the comment).
        device_type = torch.device(device).type

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs a matmul of random device tensors
            # under float64 autocast for the closed-over device type; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Random inputs on the device and a matmul
                # inside float64 autocast.
                a_float32 = torch.rand((8, 8), device=device)
                b_float32 = torch.rand((8, 8), device=device)

                with torch.autocast(device_type=device_type, dtype=torch.float64):
                    c_float64 = torch.mm(a_float32, b_float32)
                return c_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.type, device_type)
        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    @onlyCUDA
    def test_autocast_float64(self, device):
        # [FUNCTION DOC] CUDA-only: exports a module with two chained matmuls under float64
        # autocast; checks device index and float64 dtype match eager.
        device_type = torch.device(device).type

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random device
            # tensors under float64 autocast; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Random inputs and two chained matmuls
                # inside float64 autocast.
                a_float32 = torch.rand((8, 8), device=device)
                b_float32 = torch.rand((8, 8), device=device)
                d_float32 = torch.rand((8, 8), device=device)

                with torch.autocast(device_type=device_type, dtype=torch.float64):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)
        self.assertEqual(exported.dtype, torch.float64)

    def test_autocast_device_context(self, device):
        # [FUNCTION DOC] Exports a module using default-dtype autocast for the device type and
        # checks the exported output device matches eager and has index 0 (dtype is only
        # compared to eager).
        device_type = torch.device(device).type

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Test module whose forward runs chained matmuls of random device
            # tensors under default autocast for the device type; has no state.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Random inputs on the device and two chained matmuls
                # under default autocast.
                a_float32 = torch.rand((8, 8), device=device)
                b_float32 = torch.rand((8, 8), device=device)
                d_float32 = torch.rand((8, 8), device=device)

                with torch.autocast(device_type):
                    e_float64 = torch.mm(a_float32, b_float32)
                    f_float64 = torch.mm(d_float32, e_float64)
                return f_float64

        module = MyModule()
        real = module(torch.tensor([0.5]))
        real_device = real.device
        real_dtype = real.dtype

        graph, _ = torch._dynamo.export(module)(torch.tensor([[0.0, 0], [0, 0]]))
        exported = graph(torch.tensor([0.5]))
        self.assertEqual(exported.device, real_device)
        self.assertEqual(exported.dtype, real_dtype)

        self.assertEqual(exported.device.index, 0)

    def test_autocast_arguments_binding(self, device):
        # [FUNCTION DOC] Checks autocast with only device_type and enabled=False arguments binds
        # correctly for the device type and CPU; compiled results equal eager.
        device_type = torch.device(device).type

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one and takes sin inside a disabled autocast for
            # the device type.
            with torch.autocast(device_type=device_type, enabled=False):
                x = torch.sin(x + 1)
            return x

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds one and takes cos inside a disabled CPU autocast.
            with torch.autocast(device_type="cpu", enabled=False):
                x = torch.cos(x + 1)
            return x

        x = torch.rand([2, 3])
        ref1 = f1(x)
        ref2 = f2(x)
        opt_f1 = torch.compile(backend="eager")(f1)
        opt_f2 = torch.compile(backend="eager")(f2)
        res1 = opt_f1(x)
        res2 = opt_f2(x)
        self.assertTrue(same(ref1, res1))
        self.assertTrue(same(ref2, res2))

    def test_autocast_decorator(self, device):
        # [FUNCTION DOC] Checks autocast used as a decorator for the device type (two APIs) and
        # for CPU, float16, under fullgraph eager; results equal eager and the first two
        # outputs are float16.
        device_type = torch.device(device).type

        def autocast_func(orig_func):
            # [FUNCTION DOC] HELPER: Returns orig_func wrapped with torch.amp.autocast for the
            # device type, float16.
            @torch.amp.autocast(device_type=device_type, dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_device(orig_func):
            # [FUNCTION DOC] HELPER: Returns orig_func wrapped with torch.autocast for the
            # device type, float16.
            @torch.autocast(device_type=device_type, dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def autocast_func_cpu(orig_func):
            # [FUNCTION DOC] HELPER: Returns orig_func wrapped with torch.autocast
            # for CPU, float16.
            @torch.autocast(device_type="cpu", dtype=torch.float16)
            def new_fwd(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the original function.
                return orig_func(*args, **kwargs)

            return new_fwd

        def mm(a, b):
            # [FUNCTION DOC] HELPER: Returns the matmul of a and b.
            return torch.mm(a, b)

        mm_float16 = autocast_func(mm)
        mm_float16_device = autocast_func_device(mm)
        mm_float16_cpu = autocast_func_cpu(mm)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the three autocast-wrapped matmuls and returns their
            # results as a tuple.
            return mm_float16(a, b), mm_float16_device(a, b), mm_float16_cpu(a, b)

        a_float32 = torch.rand((8, 8), device=device)
        b_float32 = torch.rand((8, 8), device=device)

        ref = fn(a_float32, b_float32)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(a_float32, b_float32)
        self.assertTrue(same(ref, res))
        self.assertTrue(res[0].dtype == torch.float16)
        self.assertTrue(res[1].dtype == torch.float16)


class ContextlibContextManagerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for Dynamo tracing of contextlib.contextmanager generators
    # (nonlocal and global propagation, graph breaks, exceptions, nesting). setUp turns on
    # the enable_trace_contextlib and enable_trace_unittest Dynamo config flags and
    # tearDown restores them.
    def setUp(self):
        # [FUNCTION DOC] Saves the current values of enable_trace_contextlib and
        # enable_trace_unittest in instance attributes after calling the parent setUp, then sets
        # both config flags to True. Side effect: mutates global Dynamo config.
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_contextlib
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_contextlib = True
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] Calls the parent tearDown and restores the two Dynamo config flags to the
        # values saved in setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_contextlib = self._prev
        torch._dynamo.config.enable_trace_unittest = self._u_prev

    def test_ctx_basic0(self):
        # [FUNCTION DOC] Checks a generator context manager that sets the default dtype to
        # float64 is traced under fullgraph: a complex tensor created inside has dtype
        # complex128, and the graph text (set_default_dtype to float64 before and float32 after)
        # matches the expectation.
        @contextlib.contextmanager
        def set_default_dtype(dtype):
            # [FUNCTION DOC] HELPER: Generator context manager (local copy): saves the default
            # dtype, sets the new one, yields, and restores in a finally clause.
            old_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(dtype)
                yield
            finally:
                torch.set_default_dtype(old_dtype)

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Builds a tensor of complex values inside the
            # set_default_dtype(float64) context and returns it.
            with set_default_dtype(torch.float64):
                x = torch.tensor([3.0, 3.0 + 5.0j])
            return x

        y = fn()
        self.assertEqual(y.dtype, torch.complex128)
        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self):
        set_default_dtype = torch.set_default_dtype(torch.float64);  set_default_dtype = None

        tensor: "c128[2]" = torch.tensor([3.0, (3+5j)])

        set_default_dtype_1 = torch.set_default_dtype(torch.float32);  set_default_dtype_1 = None
        return (tensor,)
""",
        )

    def test_ctx_basic1(self):
        # [FUNCTION DOC] Checks a simple generator context manager that yields sin(x) is traced
        # under fullgraph; the compiled result is cos(sin(x)).
        @contextlib.contextmanager
        def compute_sin(x):
            # [FUNCTION DOC] HELPER: Generator context manager that yields the sine of x with an
            # empty finally clause.
            try:
                yield x.sin()
            finally:
                pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Enters compute_sin on x and returns the cosine of
            # the yielded value.
            with compute_sin(x) as y:
                return y.cos()

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_change_parent_nonlocal_0(self):
        # [FUNCTION DOC] Checks that an assignment to an enclosing-function variable (k) made
        # inside a context manager body, and one made by the caller reading it (z), propagate as
        # nonlocal side effects after compilation: both end up as 100.
        # test if a nonlocal actually gets propagated
        z = 0
        k = 0

        def create_ctx():
            # [FUNCTION DOC] HELPER: Returns a generator context manager that sets the nonlocal k
            # to 100 and yields sin of its argument.
            @contextmanager
            def ctx(x):
                # [FUNCTION DOC] HELPER: Generator context manager that sets nonlocal k to 100
                # before yielding sin(x).
                nonlocal z
                nonlocal k
                try:
                    k = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            # [FUNCTION DOC] HELPER: Enters ctx on x, copies k into the nonlocal z inside the body,
            # and returns cos of the yielded value.
            nonlocal z
            with ctx(x) as y:
                z = k
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a context manager with create_ctx and runs
            # it with run_ctx.
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    def test_change_parent_nonlocal_1(self):
        # [FUNCTION DOC] Checks that the finally clause of a generator context manager runs and
        # reads the updated nonlocal: run_ctx sets z to 100, the finally assigns it to k; both are
        # 100 after the compiled call.
        # test if finally is executed and it is reading the correct variable
        z = 1
        k = 2

        def create_ctx():
            # [FUNCTION DOC] HELPER: Returns a context manager whose finally clause assigns
            # nonlocal z into nonlocal k.
            @contextmanager
            def ctx(x):
                # [FUNCTION DOC] HELPER: Generator context manager yielding sin(x); in finally
                # copies z into k.
                nonlocal z
                nonlocal k
                try:
                    yield x.sin()
                finally:
                    k = z

            return ctx

        def run_ctx(ctx, x):
            # [FUNCTION DOC] HELPER: Sets nonlocal z to 100, enters ctx and returns cos of
            # the yielded value.
            nonlocal z
            z = 100
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the context manager and runs it.
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z, 100)
        self.assertEqual(k, 100)

    def test_globals_change_in_other_file(self):
        # [FUNCTION DOC] Checks that global variables mutated inside a traced context manager (one
        # defined here, one from the test_functions module) are updated both in this module and in
        # the other module, and that later reads in the compiled code see the new values; result is
        # three times ones.
        @contextmanager
        def update_global_ctx():
            # [FUNCTION DOC] HELPER: Generator context manager that increments the module globals
            # _variable and _variable1 and yields nothing.
            global _variable, _variable1
            try:
                _variable += 1
                _variable1 += 1
                yield
            finally:
                pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Uses the local global-updating context, then
            # test_functions.update_global_ctx, calls a constant helper, and returns x times the
            # yielded value times the sum of the globals.
            with update_global_ctx():
                pass

            with test_functions.update_global_ctx(x) as a:
                # Ensure that the updated global values are read
                test_functions.constant3(2, 3)
                return x * a * (_variable + _variable1 + test_functions._variable)

        res = fn(torch.ones(10))
        self.assertEqual(_variable, 1)
        self.assertEqual(_variable1, 1)
        # Ensure that the reconstructed bytecode updates the global value in the
        # other file.
        self.assertEqual(test_functions._variable, 1)
        self.assertEqual(res, 3 * torch.ones(10))

    def test_change_parent_global_0(self):
        # [FUNCTION DOC] Same as the nonlocal test but with module-level globals: the context
        # manager sets global k_glb, the caller copies it to z_glb; both are 100 after the compiled
        # call. Resets the globals to zero first.
        # test if a global actually gets propagated
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            # [FUNCTION DOC] HELPER: Returns a generator context manager that sets global k_glb to
            # 100 and yields sin(x).
            @contextmanager
            def ctx(x):
                # [FUNCTION DOC] HELPER: Generator context manager that sets global k_glb
                # before yielding sin(x).
                global k_glb
                try:
                    k_glb = 100
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            # [FUNCTION DOC] HELPER: Enters ctx, copies k_glb into global z_glb and returns cos of
            # the yielded value.
            global z_glb
            with ctx(x) as y:
                z_glb = k_glb
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the context manager and runs it.
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    def test_change_parent_global_1(self):
        # [FUNCTION DOC] Checks the finally clause of a generator context manager reads the up-to-
        # date global: run_ctx sets z_glb to 100 and the finally copies it to k_glb; both end at
        # 100. Resets the globals first.
        # test if finally is executed and it is reading the correct variable
        global z_glb, k_glb
        z_glb, k_glb = 0, 0

        def create_ctx():
            # [FUNCTION DOC] HELPER: Returns a generator context manager whose finally clause
            # assigns global z_glb into k_glb.
            @contextmanager
            def ctx(x):
                # [FUNCTION DOC] HELPER: Generator context manager that yields sin(x) and copies
                # z_glb to k_glb in finally.
                global z_glb, k_glb
                try:
                    yield x.sin()
                finally:
                    k_glb = z_glb

            return ctx

        def run_ctx(ctx, x):
            # [FUNCTION DOC] HELPER: Sets global z_glb to 100, enters ctx and returns cos of
            # the yielded value.
            global z_glb
            z_glb = 100
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the context manager and runs it.
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())
        self.assertEqual(z_glb, 100)
        self.assertEqual(k_glb, 100)

    def test_change_parent_0(self):
        # [FUNCTION DOC] Checks a generator context manager created in a helper function and used
        # in another helper is traced under fullgraph; the result is cos of sin of the input.
        def create_ctx():
            # [FUNCTION DOC] HELPER: Returns a generator context manager that yields the sine
            # of its argument.
            @contextlib.contextmanager
            def ctx(x):
                # [FUNCTION DOC] HELPER: Generator context manager that yields sin(x).
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx, x):
            # [FUNCTION DOC] HELPER: Enters ctx on x and returns cos of the yielded value.
            with ctx(x) as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the context manager and runs it.
            ctx = create_ctx()
            return run_ctx(ctx, x)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_change_parent_1(self):
        # [FUNCTION DOC] Same as test_change_parent_0 but the context manager closes over x instead
        # of taking it as a parameter.
        def create_ctx(x):
            # [FUNCTION DOC] HELPER: Returns a zero-argument generator context manager closing over
            # x that yields sin(x).
            @contextlib.contextmanager
            def ctx():
                # [FUNCTION DOC] HELPER: Generator context manager yielding sin of the captured x.
                try:
                    yield x.sin()
                finally:
                    pass

            return ctx

        def run_ctx(ctx):
            # [FUNCTION DOC] HELPER: Enters ctx and returns cos of the yielded value.
            with ctx() as y:
                return y.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the context manager from x and runs it.
            ctx = create_ctx(x)
            return run_ctx(ctx)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x.sin().cos())

    def test_graph_break_inside_ctx(self):
        # [FUNCTION DOC] Checks a generator context manager containing a graph break before its
        # yield: compiled output equals eager and no FX graph is recorded at all, because every
        # frame is skipped after the break.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that computes tan of x, graph
            # breaks, then yields it.
            y = x.tan()
            try:
                torch._dynamo.graph_break()
                yield y
            finally:
                pass

        def f(x):
            # [FUNCTION DOC] HELPER: Adds sin(x), then in the context subtracts the yielded value
            # in place, then adds cos(x); returns the result.
            y = x.sin()
            with whoo(x) as z:
                y += z.neg()
            y += x.cos()
            return y

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        # no graph will be generated as we will skip all frames due to the graph break
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside_ctx_with_side_effects(self):
        # [FUNCTION DOC] Same as the graph-break-in-context test but the generator appends to a
        # list before the break and in its finally clause; checks the numeric result, that the
        # list records sin then cos in order (side effects run exactly once each), and that no
        # graph is recorded.
        L = []

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends sin(x) to the list L,
            # graph breaks, yields tan(x), and appends cos(x) in a finally clause.
            y = x.tan()
            try:
                L.append(x.sin())
                torch._dynamo.graph_break()
                yield y
            finally:
                L.append(x.cos())

        def f(x):
            # [FUNCTION DOC] HELPER: Same arithmetic as in the previous test: sin(x) plus the
            # negated yielded value plus cos(x).
            y = x.sin()
            with whoo(x) as z:
                y += z.neg()
            y += x.cos()
            return y

        x = torch.randn(2)
        eager = EagerAndRecordGraphs()
        y = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(y, x.sin() + x.tan().neg() + x.cos())
        self.assertEqual(L, [x.sin(), x.cos()])
        # no graph will be generated as we will skip all frames due to the graph break
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside_ctx_1(self):
        # [FUNCTION DOC] Checks a graph break inside a context manager used by a nested helper:
        # results equal eager and two graphs are recorded, with expected graph text for the first
        # (sin only) and the second (resumed part, with stack inputs added to cos).
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that computes tan(x), graph breaks,
            # then yields it.
            y = x.tan()
            try:
                torch._dynamo.graph_break()
                yield y
            finally:
                pass

        def bar(x):
            # [FUNCTION DOC] HELPER: Enters whoo on x and returns the negation of
            # the yielded value.
            with whoo(x) as z:
                return z.neg()

        def f(x):
            # [FUNCTION DOC] HELPER: Returns sin(x) plus bar(x) plus cos(x).
            return x.sin() + bar(x) + x.cos()

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2]"):
        l_x_ = L_x_

        sin: "f32[2]" = l_x_.sin();  l_x_ = None
        return (sin,)
""",
        )
        self.assertExpectedInline(
            normalize_gm(eager.graphs[1].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "f32[2]", L_stack1_: "f32[2]", L_x_: "f32[2]"):
        l_stack0_ = L_stack0_
        l_stack1_ = L_stack1_
        l_x_ = L_x_

        add: "f32[2]" = l_stack0_ + l_stack1_;  l_stack0_ = l_stack1_ = None
        cos: "f32[2]" = l_x_.cos();  l_x_ = None
        add_1: "f32[2]" = add + cos;  add = cos = None
        return (add_1,)
""",
        )

    def test_graph_break_inside_ctx_2(self):
        # [FUNCTION DOC] Checks a graph break in a context manager before the yield while a callee
        # g is used in the body; eager equals compiled and exactly one graph is recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that graph breaks
            # then yields cos(x).
            try:
                torch._dynamo.graph_break()
                yield x.cos()
            finally:
                pass

        def g(x):
            # [FUNCTION DOC] HELPER: Returns the negation of x plus the arccosine of x.
            return x.neg() + x.acos()

        def f(x):
            # [FUNCTION DOC] HELPER: Adds g of the yielded value to sin(x) inside the context, then
            # adds tan of the result to itself.
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 1)

    def test_graph_break_before___enter__(self):
        # [FUNCTION DOC] Checks a graph break between creating a contextmanager object and calling
        # its __enter__ and __exit__ directly; compiled result equals eager.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context object, graph breaks, calls __enter__ then
            # __exit__ manually and returns the entered value.
            ctx = whoo(x)
            torch._dynamo.graph_break()
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_graph_break_in_finally(self):
        # [FUNCTION DOC] Checks a graph break inside the finally clause of a generator context
        # manager called manually through __enter__/__exit__: the result is sin(x), the list
        # records x then cos(x), and no graph is recorded.
        z = []

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends x to list z, yields
            # sin(x), and in finally graph breaks then appends cos(x).
            nonlocal z
            try:
                z.append(x)
                yield x.sin()
            finally:
                torch._dynamo.graph_break()
                z.append(x.cos())

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context, calls __enter__ then __exit__ manually
            # and returns the entered value.
            ctx = whoo(x)
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(out, x.sin())
        self.assertEqual(z, [x, x.cos()])
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_inside___enter__(self):
        # [FUNCTION DOC] Checks a graph break inside the generator before it yields, with manual
        # __enter__ and __exit__ calls; equal to eager and no graphs recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that graph breaks and yields
            # x plus one.
            try:
                torch._dynamo.graph_break()
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls __enter__ and __exit__ manually and returns the value.
            ctx = whoo(x)
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_after___enter__(self):
        # [FUNCTION DOC] Checks a graph break after __enter__ (inside a try with __exit__ in
        # finally); result equals eager and no graphs are recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Enters manually, graph breaks, and exits in a finally clause;
            # returns the entered value.
            ctx = whoo(x)
            try:
                y = ctx.__enter__()
                torch._dynamo.graph_break()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_before_and_after___enter__(self):
        # [FUNCTION DOC] Checks graph breaks both before and after __enter__ with __exit__ in a
        # finally clause; equal to eager, no graphs recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Graph breaks, enters manually, graph breaks again, exits in
            # finally; returns the entered value.
            ctx = whoo(x)
            try:
                torch._dynamo.graph_break()
                y = ctx.__enter__()
                torch._dynamo.graph_break()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_before___enter___and_disable___exit__(self):
        # [FUNCTION DOC] Checks a graph break before __enter__ where __exit__ is called from a
        # function decorated with torch._dynamo.disable; equal to eager, no graphs recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Graph breaks, enters manually, then calls a Dynamo-disabled
            # helper that calls __exit__ in a finally clause; returns the entered value.
            ctx = whoo(x)
            try:
                torch._dynamo.graph_break()
                y = ctx.__enter__()
            finally:

                @torch._dynamo.disable
                def g():
                    # [FUNCTION DOC] HELPER: Dynamo-disabled function that calls __exit__ on the
                    # enclosing context object with no exception.
                    ctx.__exit__(None, None, None)

                g()
            return y

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_disable___enter__(self):
        # [FUNCTION DOC] Checks a context manager whose __enter__ is called from a Dynamo-disabled
        # function: compiled result equals eager.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding h(x) plus one.
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context, calls __enter__ through a disabled
            # helper, then calls __exit__ and returns the entered value.
            ctx = whoo(x)

            @torch._dynamo.disable
            def g():
                # [FUNCTION DOC] HELPER: Dynamo-disabled function that returns the result of
                # __enter__ on the enclosing context object.
                return ctx.__enter__()

            y = g()
            ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_disable___exit__(self):
        # [FUNCTION DOC] Checks a context manager whose __exit__ is called from a Dynamo-disabled
        # function; compiled result equals eager.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding h(x) plus one.
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Enters manually, then calls __exit__ through a disabled
            # helper; returns the entered value.
            ctx = whoo(x)
            y = ctx.__enter__()

            @torch._dynamo.disable
            def g():
                # [FUNCTION DOC] HELPER: Dynamo-disabled function that calls __exit__ on the
                # enclosing context object.
                ctx.__exit__(None, None, None)

            g()

            return y

        x = torch.tensor([1.0])
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected, result)

    def test_contextmanager_as_argument(self):
        # [FUNCTION DOC] Checks a generator context manager passed as an argument to the
        # compiled function and used through __enter__/__exit__: results equal eager and two
        # graphs are recorded.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding h(x) plus one.
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Calls __enter__ and __exit__ on the given context and returns
            # x plus the entered value.
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return x + y

        x = torch.tensor([1.0])
        expected = fn(x, whoo(x))

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x, whoo(x))
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)

    def test_return_new_contextmanager(self):
        # [FUNCTION DOC] Checks that a newly created generator context manager can be returned from
        # a compiled function: the first output equals eager and the returned object has the same
        # type name as in eager.
        L = []

        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends sin(x) to L, yields
            # h(x) plus one, and appends cos(x) in finally.
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context object and returns x plus one with it.
            ctx = whoo(x)
            return x + 1, ctx

        x = torch.tensor([1.0])
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected[0], result[0])
        self.assertEqual(type(expected[1]).__name__, type(result[1]).__name__)

    def test_return_advanced_contextmanager(self):
        # [FUNCTION DOC] Checks that an already entered generator context manager can be
        # returned from a compiled function (output value equals eager and the returned object
        # type name matches).
        L = []

        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends sin(x) to L, yields
            # h(x) plus one, and appends cos(x) in finally.
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context, calls __enter__ and returns x plus the
            # entered value with the context.
            ctx = whoo(x)
            y = ctx.__enter__()
            return x + y, ctx

        x = torch.tensor([1.0])
        expected = fn(x)
        result = torch.compile(fn, backend="eager", fullgraph=False)(x)
        self.assertEqual(expected[0], result[0])
        self.assertEqual(type(expected[1]).__name__, type(result[1]).__name__)

    def test_contextmanager_as_argument_only___enter__(self):
        # [FUNCTION DOC] Checks a context manager passed as an argument and only entered inside the
        # compiled function: result is x + cos(x) + 1, the list L has just the sin entry until
        # __exit__ is called outside, and two graphs are recorded.
        L = []

        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends sin(x) to L, yields
            # h(x) plus one, and appends cos(x) in finally.
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Calls __enter__ on the given context and returns x plus
            # the entered value.
            y = ctx.__enter__()
            return x + y

        x = torch.tensor([1.0])
        ctx = whoo(x)
        eager = EagerAndRecordGraphs()
        y = torch.compile(backend=eager, fullgraph=False)(fn)(x, ctx)
        self.assertEqual(y, x + x.cos() + 1)
        self.assertEqual(L, [x.sin()])  # we should only have one item in L

        ctx.__exit__(None, None, None)
        self.assertEqual(L, [x.sin(), x.cos()])  # Two items now

        self.assertEqual(len(eager.graphs), 2)

    def test_contextmanager_as_argument_only___exit__(self):
        # [FUNCTION DOC] Checks a context manager that was entered outside the compiled function
        # and exited inside it: the output is sin(x), L gets the sin entry before and the cos entry
        # after the call, and one graph is recorded.
        L = []

        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager that appends sin(x) to L, yields
            # h(x) plus one, and appends cos(x) in finally.
            try:
                L.append(x.sin())
                yield h(x) + 1
            finally:
                L.append(x.cos())

        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Calls __exit__ on the given context and returns sin(x).
            ctx.__exit__(None, None, None)
            return x.sin()

        x = torch.tensor([1.0])
        ctx = whoo(x)
        ctx.__enter__()
        self.assertEqual(L, [x.sin()])

        eager = EagerAndRecordGraphs()
        y = torch.compile(backend=eager, fullgraph=False)(fn)(x, ctx)
        self.assertEqual(y, x.sin())
        self.assertEqual(L, [x.sin(), x.cos()])
        self.assertEqual(len(eager.graphs), 1)

    def test_advanced_contextmanager_as_argument(self):
        # [FUNCTION DOC] Checks a context manager entered outside the compiled function and exited
        # inside; the result is x plus one.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding h(x) plus one.
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Calls __exit__ on the context and returns x plus one.
            ctx.__exit__(None, None, None)
            return x + 1

        x = torch.tensor([1.0])
        ctx = whoo(x)
        y = ctx.__enter__()
        self.assertEqual(y, x.cos() + 1)
        z = torch.compile(backend="eager", fullgraph=False)(fn)(x, ctx)
        self.assertEqual(z, x + 1)

    def test_advanced_contextmanager_as_argument_error(self):
        # [FUNCTION DOC] Checks that passing an already entered generator context manager to a
        # function that calls both __enter__ and __exit__ raises AttributeError mentioning
        # args when compiled.
        def h(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x.
            return x.cos()

        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding h(x) plus one.
            try:
                yield h(x) + 1
            finally:
                pass

        def fn(x, ctx):
            # [FUNCTION DOC] HELPER: Calls __enter__ and __exit__ on the context and returns
            # the entered value.
            y = ctx.__enter__()
            ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        ctx = whoo(x)
        y = ctx.__enter__()
        self.assertEqual(y, x.cos() + 1)

        with self.assertRaisesRegex(AttributeError, "args"):
            torch.compile(backend="eager", fullgraph=False)(fn)(x, ctx)

    def test_disable_ctx_manager(self):
        # [FUNCTION DOC] Checks a generator context manager used inside a function decorated with
        # torch._dynamo.disable: result equals eager and no graph is recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        @torch._dynamo.disable
        def g(x):
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that enters the context and returns
            # the yielded value.
            with whoo(x) as y:
                return y

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns g of x.
            return g(x)

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_and_disable___enter__(self):
        # [FUNCTION DOC] Checks a graph break followed by a Dynamo-disabled call of __enter__ with
        # __exit__ in finally; equal to eager and no graphs recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates the context, graph breaks, enters via a disabled
            # helper, exits in finally, returns the entered value.
            ctx = whoo(x)
            try:
                torch._dynamo.graph_break()

                @torch._dynamo.disable
                def g():
                    # [FUNCTION DOC] HELPER: Dynamo-disabled function returning the result of
                    # __enter__ on the enclosing context.
                    return ctx.__enter__()

                y = g()
            finally:
                ctx.__exit__(None, None, None)
            return y

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    def test_dynamo_disable_ctx(self):
        # [FUNCTION DOC] Same shape as test_disable_ctx_manager: a Dynamo-disabled function using a
        # generator context manager gives eager-equal output with no graphs recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding x plus one.
            try:
                yield x + 1
            finally:
                pass

        @torch._dynamo.disable
        def g(x):
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that enters the context and returns
            # the yielded value.
            with whoo(x) as y:
                return y

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns g of x.
            return g(x)

        x = torch.tensor([1.0])
        expected = fn(x)

        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False)(fn)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 0)

    @torch._dynamo.config.patch(enable_trace_contextlib=False)
    def test_disable_trace_contextmanager(self):
        # [FUNCTION DOC] With enable_trace_contextlib patched off, checks a generator context
        # manager in a body with a callee falls back to graph breaks: output equals eager (dynamic
        # off) and two graphs are recorded.
        @contextlib.contextmanager
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator context manager yielding cos(x).
            try:
                yield x.cos()
            finally:
                pass

        def g(x):
            # [FUNCTION DOC] HELPER: Returns the negation of x plus the arccosine of x.
            return x.neg() + x.acos()

        def f(x):
            # [FUNCTION DOC] HELPER: Adds g of the yielded value to sin(x) inside the context, then
            # adds tan of the result to itself.
            y = x.sin()
            with whoo(x) as z:
                y += g(z)
            y += y.tan()
            return y

        x = torch.randn(2)
        expected = f(x)
        eager = EagerAndRecordGraphs()
        out = torch.compile(backend=eager, fullgraph=False, dynamic=False)(f)(x)
        self.assertEqual(expected, out)
        self.assertEqual(len(eager.graphs), 2)

    @parametrize("name", ("stdout", "stderr"))
    def test_contextlib_suppress(self, name):
        # [FUNCTION DOC] Parametrized on stdout or stderr: checks redirect_stdout/redirect_stderr
        # are unsupported context managers, causing one graph break whose recorded message says the
        # contextlib class is not supported; output equals eager. Clears the graph-break counters.
        counters.clear()
        eager = EagerAndRecordGraphs()

        def fn(t):
            # [FUNCTION DOC] HELPER: Computes sin(t), enters the redirect context chosen by name
            # inside which cos(t) is added, and returns tan of the sum.
            y = t.sin()
            # ensure we graph break on the suppress call below
            if name == "stdout":
                ctx = contextlib.redirect_stdout(sys.stderr)
            else:
                ctx = contextlib.redirect_stderr(sys.stdout)

            with ctx:
                y += t.cos()
            return y.tan()

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend=eager, fullgraph=False)(fn)(t)
        self.assertEqual(expected, got)
        self.assertEqual(len(counters["graph_break"]), 1)
        name = f"redirect_{name}" if name in ("stdout", "stderr") else name
        self.assertRegex(
            next(iter(counters["graph_break"])),
            f"<class 'contextlib.{name}'> not supported",
        )

    def test_contextlib_nullcontext(self):
        # [FUNCTION DOC] Checks contextlib.nullcontext is handled by Dynamo under fullgraph with no
        # graph break recorded and the right result.
        counters.clear()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Enters a nullcontext and returns sin(t).
            with contextlib.nullcontext():
                return t.sin()

        t = torch.randn(2)
        y = fn(t)
        # nullcontext is correctly handled in dynamo
        self.assertEqual(len(counters["graph_break"]), 0)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 11), "Python 3.11+")
    def test_WITH_EXCEPT_START(self):
        # [FUNCTION DOC] Python 3.11+ only: checks an exception raised inside a generator context
        # manager body and caught outside is handled under fullgraph; the result is sin(t).
        @contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: Generator context manager with empty try/finally around
            # a bare yield.
            try:
                yield
            finally:
                pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Raises a ValueError inside the context and catches
            # it, returning sin(t).
            try:
                with ctx():
                    raise ValueError
            except ValueError:
                return t.sin()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())


instantiate_parametrized_tests(CtxManagerTests)
instantiate_parametrized_tests(ContextlibContextManagerTests)
instantiate_device_type_tests(
    CtxManagerTestsDevice, globals(), except_for=("cpu",), allow_xpu=True
)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
