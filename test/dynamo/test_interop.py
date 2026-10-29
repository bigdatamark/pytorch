# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that dynamo compiles (eager backend, full graph) functions that call into
# other tracing systems: torch.fx symbolic traces, TorchScript scripted and traced functions,
# and vmap. Picks the current accelerator device, else cpu, at import time.
from functools import wraps

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import allow_in_graph


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


def fn(a, b):
    # [FUNCTION DOC] HELPER: Module-level sample function; returns a plus b scaled by 0.67.
    return a + b * 0.67


class InteropTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Checks that functions wrapping fx, TorchScript or vmap callables compile
    # under dynamo and match eager results.
    def _common(self, fn):
        # [FUNCTION DOC] HELPER: Calls fn eagerly on two random length-10 tensors on the test
        # device, compiles it with the eager backend requiring a full graph, runs the compiled
        # version on the same inputs, and asserts the results are equal.
        inputs = [
            torch.randn(10, device=device_type),
            torch.randn(10, device=device_type),
        ]
        ref = fn(*inputs)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(*inputs)
        self.assertEqual(ref, res)

    def test_fx_fn(self):
        # [FUNCTION DOC] Symbolically traces fn with fx and checks a lambda wrapping it
        # compiles and matches eager.
        fx_fn = torch.fx.symbolic_trace(fn)
        self._common(lambda a, b: fx_fn(a, b) + 1)

    def test_script_fn(self):
        # [FUNCTION DOC] Scripts fn with TorchScript and checks a lambda wrapping it compiles
        # and matches eager.
        script_fn = torch.jit.script(fn)
        self._common(lambda a, b: script_fn(a, b) + 1)

    def test_trace_fn(self):
        # [FUNCTION DOC] Traces fn with TorchScript tracing using zero example inputs and checks
        # a lambda wrapping it compiles and matches eager.
        trace_fn = torch.jit.trace(
            fn,
            [torch.zeros(10, device=device_type), torch.zeros(10, device=device_type)],
        )
        self._common(lambda a, b: trace_fn(a, b) + 1)

    def test_staticmethod_script_fn(self):
        # [FUNCTION DOC] Checks a method calling a scripted static method (squares its input)
        # compiles with a full graph and matches eager.
        class Foo:
            # [CLASS DOC] HELPER: Holds a scripted static method and an instance method using it.
            @staticmethod
            @torch.jit.script
            def _g(a):
                # [FUNCTION DOC] HELPER: Scripted static method returning the square of a.
                return a**2

            def g(self, a, b):
                # [FUNCTION DOC] HELPER: Returns the scripted square of a plus b.
                return self._g(a) + b

        foo = Foo()
        self._common(lambda a, b: foo.g(a, b) + 1)

    def test_vmap_in_graph(self):
        # [FUNCTION DOC] Compiles a function that vmaps Tensor.t over a batch, both plain and
        # wrapped by a decorator that marks it allow_in_graph. Asserts results match eager and
        # the compile counter shows one frame after the first compile and two after the second.
        def traceable(f):
            # [FUNCTION DOC] HELPER: Registers f as allowed in the graph and returns a
            # functools-wrapped forwarding wrapper around it.
            f = allow_in_graph(f)

            @wraps(f)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to f and returns its result.
                return f(*args, **kwargs)

            return wrapper

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(3, 5, 3, device=device_type)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies vmap of Tensor.t over the leading dimension of x.
            return torch.vmap(torch.Tensor.t)(x)

        fn_opt = torch.compile(fn, backend=cnts, fullgraph=True)
        fn_opt_traceable = torch.compile(traceable(fn), backend=cnts, fullgraph=True)

        self.assertEqual(fn(x), fn_opt(x))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(fn_opt(x), fn_opt_traceable(x))
        self.assertEqual(cnts.frame_count, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
