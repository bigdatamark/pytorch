# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for how dynamo traces dispatch key set queries, thread-local dispatch key
# include and exclude sets, functorch interpreter lookups and Python/TLS snapshot dispatch keys.
# Defines device_type at import time as the current accelerator type or cpu when none exists.
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter, EagerAndRecordGraphs, normalize_gm
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import TEST_XPU


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class PythonDispatcherTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case covering dispatch key set handling: results of compiled
    # functions are compared with eager dispatch key sets, graph contents are checked for a
    # constant-folded branch, and recompilation is counted via guards on key sets.
    def test_dispatch_key1(self):
        # [FUNCTION DOC] Compiles a function (aot_eager, full graph) that adds one to a tensor
        # and returns its dispatch key set; checks the raw representation equals the eager key
        # set of the incremented tensor.
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: increments the input and returns its
            # dispatch key set.
            x = x + 1
            return torch._C._dispatch_keys(x)

        x = torch.randn(2, 3)
        self.assertTrue(fn(x).raw_repr() == torch._C._dispatch_keys(x + 1).raw_repr())

    def test_dispatch_key2(self):
        # [FUNCTION DOC] Same check for a TwoTensor wrapper subclass: the compiled key set after
        # applying sine must equal the eager one for the same operation.
        from torch.testing._internal.two_tensor import TwoTensor

        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: applies sine to the input and returns
            # its dispatch key set.
            x = x.sin()
            return torch._C._dispatch_keys(x)

        x = torch.randn(3)
        y = torch.randn(3)
        z = TwoTensor(x, y)
        self.assertTrue(fn(z).raw_repr() == torch._C._dispatch_keys(z.sin()).raw_repr())

    def test_dispatch_key3(self):
        # [FUNCTION DOC] Compiles a function returning both a sine result and the thread-local
        # include dispatch key set; checks the value and that the returned key set equals the
        # eager thread-local include set.
        @torch.compile(backend="aot_eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: reads the thread-local include set and
            # returns sine of input plus one together with that set.
            key_set = torch._C._dispatch_tls_local_include_set()
            return torch.sin(x + 1), key_set

        x = torch.randn(2, 3)
        self.assertEqual(fn(x)[0], torch.sin(x + 1))
        self.assertTrue(
            fn(x)[1].raw_repr() == torch._C._dispatch_tls_local_include_set().raw_repr()
        )

    def test_dispatch_key4(self):
        # [FUNCTION DOC] Compiles a function that combines the thread-local include set with the
        # input keys, subtracts the exclude set, and branches on whether the top-priority key is
        # the Python dispatcher. Checks the result matches the else branch and that the recorded
        # graph contains only the subtract and sine operations (expected text).
        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: builds a key set from include set,
            # input keys and exclude set, then returns sine of input plus one if the highest
            # priority key is the Python dispatcher, else sine of input minus one.
            key_set = torch._C._dispatch_tls_local_include_set()
            key_set = key_set | torch._C._dispatch_keys(x)
            key_set = key_set - torch._C._dispatch_tls_local_exclude_set()
            if key_set.highestPriorityTypeId() == torch.DispatchKey.PythonDispatcher:
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x = torch.randn(2, 3)
        self.assertEqual(fn(x), torch.sin(x - 1))

        graph = eager.graphs[0]
        actual = normalize_gm(graph.print_readable(False))

        self.assertExpectedInline(
            actual,
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 3]"):
        l_x_ = L_x_

        sub: "f32[2, 3]" = l_x_ - 1;  l_x_ = None
        sin: "f32[2, 3]" = torch.sin(sub);  sub = None
        return (sin,)
""",
        )

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "requires cuda or xpu")
    def test_dispatch_key_set_guard(self):
        # [FUNCTION DOC] Skipped unless CUDA or XPU is available. Compiles a function branching
        # on whether a passed dispatch key set contains the CPU key, with a compile counter. Two
        # CPU tensors with equal key sets cause one compile and no recompile; a tensor on the
        # accelerator device has a different key set and triggers a second compile.
        counter = CompileCounter()

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, dks):
            # [FUNCTION DOC] HELPER: Function under test: returns sine of input plus one if the
            # key set has CPU, else sine of input minus one.
            if dks.has("CPU"):
                return torch.sin(x + 1)
            else:
                return torch.sin(x - 1)

        x1 = torch.randn(2, 3)
        dks1 = torch._C._dispatch_keys(x1)
        self.assertEqual(fn(x1, dks1), torch.sin(x1 + 1))
        self.assertEqual(counter.frame_count, 1)

        x2 = torch.randn(2, 3)
        dks2 = torch._C._dispatch_keys(x2)
        self.assertEqual(fn(x2, dks2), torch.sin(x2 + 1))
        # No recompile since the dispatch key set is the same though the tensor is different.
        self.assertEqual(counter.frame_count, 1)

        x3 = torch.randn(2, 3, device=device_type)
        dks3 = torch._C._dispatch_keys(x3)
        self.assertEqual(fn(x3, dks3), torch.sin(x3 - 1))
        # Re-compile since the dispatch key set is different.
        self.assertEqual(counter.frame_count, 2)

    def test_functorch_interpreter(self):
        # [FUNCTION DOC] Compiles a function that applies vmap to a helper reading the current
        # functorch interpreter, checks results for two input pairs and that only one frame is
        # compiled (no recompile for different values).
        counter = CompileCounter()

        def square_and_add(x, y):
            # [FUNCTION DOC] HELPER: Reads the current functorch interpreter; under vmap returns
            # the sum of squares and second input scaled by the interpreter level, otherwise
            # returns the square scaled by the level.
            interpreter = (
                torch._functorch.pyfunctorch.retrieve_current_functorch_interpreter()
            )
            level = interpreter.level()
            if interpreter.key() == torch._C._functorch.TransformType.Vmap:
                return (x**2 + y) * level
            else:
                return x**2 * level

        @torch.compile(backend=counter, fullgraph=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled entry point applying vmap of the helper to both
            # inputs.
            return torch.vmap(square_and_add)(x, y)

        x = torch.tensor([1, 2, 3, 4])
        y = torch.tensor([10, 20, 30, 40])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 56]))
        self.assertEqual(counter.frame_count, 1)

        x = torch.tensor([1, 2, 3, 1])
        y = torch.tensor([10, 20, 30, 10])
        self.assertEqual(fn(x, y), torch.tensor([11, 24, 39, 11]))
        # No recompile
        self.assertEqual(counter.frame_count, 1)

    def test_graph_break_recovers_missing_python_tls_snapshot(self):
        # [FUNCTION DOC] Checks that a graph break recovers the Python TLS snapshot key: inside a
        # compiled function the Python dispatch key is included, the PythonTLSSnapshot key is
        # excluded, then a graph break occurs. The output must equal eager sine. The original
        # thread-local inclusion state of both keys is saved and restored in a finally block.
        @torch.compile(backend="eager_noexcept")
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test: sets thread-local key inclusion flags
            # for the Python and PythonTLSSnapshot keys, forces a graph break, then returns sine
            # of input.
            torch._C._dispatch_tls_set_dispatch_key_included(
                torch._C.DispatchKey.Python, True
            )
            torch._C._dispatch_tls_set_dispatch_key_included(
                torch._C.DispatchKey.PythonTLSSnapshot, False
            )
            torch._dynamo.graph_break()
            return torch.sin(x)

        saved_python = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.Python
        )
        saved_python_tls_snapshot = torch._C._dispatch_tls_is_dispatch_key_included(
            torch._C.DispatchKey.PythonTLSSnapshot
        )
        try:
            x = torch.randn(3)
            self.assertEqual(fn(x), x.sin())
        finally:
            torch._C._dispatch_tls_set_dispatch_key_included(
                torch._C.DispatchKey.Python, saved_python
            )
            torch._C._dispatch_tls_set_dispatch_key_included(
                torch._C.DispatchKey.PythonTLSSnapshot, saved_python_tls_snapshot
            )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
