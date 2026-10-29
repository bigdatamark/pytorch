# Owner(s): ["module: functorch"]

# [MODULE DOC] Tests for the functorch graph minifier (minifier from functorch.compile) and
# for the state dumping helper dump_state. Defines one TestMinifier case class and runs
# through the common test runner when executed as a script. No import-time side effects.
import contextlib
import io

import torch
from functorch import make_fx
from functorch.compile import minifier
from torch._functorch.compile_utils import get_outputs, get_placeholders
from torch._functorch.fx_minifier import dump_state
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMinifier(TestCase):
    # [CLASS DOC] Tests for the functorch graph minifier. Each test traces or symbolically
    # traces a small function into an fx graph, defines a predicate that stands for a
    # failure, runs the minifier, and checks the node count and input count of the reduced
    # graph. One extra test checks the state dumper keeps the device index. No
    # state is kept.
    def test_has_mul_minifier(self):
        # [FUNCTION DOC] Minifies a traced graph whose failure predicate is the
        # presence of a multiply node. Asserts the reduced graph has exactly four
        # nodes and still takes two inputs.
        def failing_f(x, y):
            # [FUNCTION DOC] HELPER: Traced function: divides y by three, adds
            # three to x, multiplies the two and adds y again. Returns one tensor.
            y = y / 3
            x = x + 3
            x = x * y
            return x + y

        inps = [torch.randn(3), torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_mul(fx_g, inps):
            # [FUNCTION DOC] HELPER: Failure predicate: true when any graph node
            # targets the aten multiply tensor operator. The inputs are ignored.
            return torch.ops.aten.mul.Tensor in (i.target for i in fx_g.graph.nodes)

        min_f, inps = minifier(failing_f, inps, has_mul)
        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_has_add_mul(self):
        # [FUNCTION DOC] Minifies a graph that produces NaN values. Asserts the
        # reduced graph has three nodes and a single input.
        def failing_f(x):
            # [FUNCTION DOC] HELPER: Traced function: scales by three, adds five,
            # takes cosine, forms zero by subtracting the value from itself,
            # divides zero by zero and adds three, then returns a one-element
            # tuple holding the doubled result.
            x = x * 3
            x = x + 5
            x = x.cos()
            zero = x - x
            result = zero / zero
            result = result + 3
            return (result * 2,)

        inps = [torch.randn(3)]
        failing_f = make_fx(failing_f)(*inps)

        def has_nans(fx_g, inps):
            # [FUNCTION DOC] HELPER: Failure predicate: returns False if any
            # supplied input already contains a NaN so the failure must come from
            # the graph; otherwise runs the graph and returns a tensor that is
            # true if the first output contains any NaN.
            # Basically, make sure none of the nodes are computing nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, has_nans)
        self.assertEqual(len(min_f.graph.nodes), 3)
        self.assertEqual(len(inps), 1)

    def test_input_returned(self):
        # [FUNCTION DOC] Minifies with a predicate that fires when some graph input is
        # also a graph output. Asserts the reduced graph has two nodes and one
        # remaining input.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Traced function: sine of a, cosine of c,
            # their product d; returns the tuple of a, b, c and d, so the raw
            # input b is returned unchanged.
            a = a.sin()
            c = c.cos()
            d = a * c
            return (a, b, c, d)

        inps = [torch.randn(3) for _ in range(3)]

        def inputs_returned(fx_g, inps):
            # [FUNCTION DOC] HELPER: Failure predicate: builds sets of
            # placeholders and output nodes of the graph using the compile utility
            # helpers and returns True when the sets overlap. The input values
            # passed in are shadowed and unused.
            inps = set(get_placeholders(fx_g.graph))
            outs = set(get_outputs(fx_g.graph))
            return len(inps & outs) > 0

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, inputs_returned)
        self.assertEqual(len(min_f.graph.nodes), 2)
        self.assertEqual(len(inps), 1)

    def test_tup_use(self):
        # [FUNCTION DOC] Minifies a graph that uses a tuple result of a standard-
        # deviation-and-mean operation, with a predicate that fires on an add node.
        # Asserts the reduced graph has four nodes and two inputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Traced function: takes the std_mean pair of
            # a, and returns a one-element tuple of the std plus b times the mean.
            tup = torch.std_mean(a)
            return (tup[0] + b * tup[1],)

        inps = [torch.randn(3), torch.randn(3)]

        def has_add(fx_g, inps):
            # [FUNCTION DOC] HELPER: Failure predicate: true when any graph node
            # targets the aten add tensor operator. The inputs are ignored.
            return torch.ops.aten.add.Tensor in (i.target for i in fx_g.graph.nodes)

        failing_f = make_fx(f)(*inps)
        min_f, inps = minifier(failing_f, inps, has_add)

        self.assertEqual(len(min_f.graph.nodes), 4)
        self.assertEqual(len(inps), 2)

    def test_module(self):
        # [FUNCTION DOC] Minifies a graph obtained by symbolic tracing of a module,
        # using a NaN-producing predicate. Raises AssertionError explicitly if the
        # reduced graph does not have three nodes or if one input does not remain.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module holding a ReLU layer. Its forward
            # applies the layer, builds zero by subtracting the value from itself,
            # divides zero by zero and adds three, yielding NaN values.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor: calls the parent
                # constructor and stores a ReLU layer as the relu attribute.
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the relu layer, divides the
                # zero difference by itself (NaN) and adds three.
                # Returns the tensor.
                y = self.relu(x)
                zero = y - y
                result = zero / zero
                result = result + 3
                return result

        mod = MockModule()
        failing_f = torch.fx.symbolic_trace(mod)

        inps = [torch.randn(3)]

        def pass_checker(fx_g, inps):
            # [FUNCTION DOC] HELPER: Failure predicate: returns False if any input
            # has a NaN; otherwise runs the graph on the inputs and returns
            # whether the first output contains any NaN.
            # Basically, make sure none of the inputs are nans
            for i in inps:
                if torch.isnan(i).any():
                    return False
            return torch.isnan(fx_g(*inps)[0]).any()

        min_f, inps = minifier(failing_f, inps, pass_checker)
        if len(min_f.graph.nodes) != 3:
            raise AssertionError(
                f"Expected 3 graph nodes, got {len(min_f.graph.nodes)}"
            )
        if len(inps) != 1:
            raise AssertionError(f"Expected 1 input, got {len(inps)}")

    def test_dump_state_preserves_device_index(self):
        # [FUNCTION DOC] Checks that dump_state writes the full device string with
        # index for a fake tensor. Builds an identity graph module, a fake tensor
        # whose device is the seventh CUDA device, calls dump_state with standard
        # output redirected to a buffer, and asserts the captured repro text contains
        # the quoted cuda device string with index seven. Needs no actual GPU.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        graph.output(x)
        gm = torch.fx.GraphModule({}, graph)

        fake_mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fake_tensor = (
            torch._subclasses.fake_tensor.FakeTensorConverter().from_meta_and_device(
                fake_mode,
                torch.empty_strided((2, 3), (3, 1), device="meta"),
                torch.device("cuda:7"),
            )
        )

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            dump_state(gm, [fake_tensor])

        repro = buf.getvalue()
        self.assertIn("'cuda:7'", repro)


if __name__ == "__main__":
    run_tests()
