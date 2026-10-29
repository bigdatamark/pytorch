# Owner(s): ["module: higher order operators"]
# flake8: noqa: E731
# [MODULE DOC] Tests for the invoke_subgraph higher-order op and the nested_compile_region
# decorator. Covers eager/AOT tracing, torch.compile with inductor and aot_eager backends,
# graph-structure expectations, caching and dedup of repeated regions, mutation handling,
# autograd, functionalization, export capture, and inductor pattern-matcher integration.
# Import-time effects: defines the nested_compile_region alias and imports triton if a GPU
# is available.

import contextlib
import enum
import re
import unittest
import unittest.mock as mock
import warnings
from dataclasses import dataclass

from parameterized import parameterized_class

import torch
import torch._dynamo
import torch._functorch
import torch._inductor
import torch._inductor.decomposition
import torch.utils._pytree as pytree
from functorch.compile import aot_function, nop
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.functional_export import dynamo_graph_capture_for_export
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    EagerAndRecordGraphs,
    empty_line_normalizer,
    InductorAndRecordGraphs,
    normalize_gm,
)
from torch._higher_order_ops.auto_functionalize import FunctionalCallableWithEpilogue
from torch._higher_order_ops.schema import find_hop_schema
from torch._higher_order_ops.utils import FunctionalizeCtxWrapper
from torch._inductor import config as inductor_config
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._subclasses.functional_tensor import (
    FunctionalTensorMode,
    PythonFunctionalizeAPI,
)
from torch.fx.graph import _BoxedCodeGen
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    TEST_WITH_CROSSREF,
    TestCase,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton, requires_gpu


nested_compile_region = torch.compiler.nested_compile_region

if HAS_GPU:
    import triton


def _aot_eager_with_runtime_epilogue():
    # [FUNCTION DOC] HELPER: Build an aot_autograd backend that does not keep input mutations in
    # the graph (so a runtime epilogue applies them), plus a list that the forward compiler
    # fills with each compiled forward graph module. Returns the pair (backend, graph list).
    """Create a recording AOT eager backend with a runtime mutation epilogue."""
    fw_graphs = []

    def fw_compiler(gm, _):
        # [FUNCTION DOC] HELPER: Forward compiler hook that records the graph module in the
        # enclosing list and returns its forward callable unchanged.
        fw_graphs.append(gm)
        return gm.forward

    return aot_autograd(
        fw_compiler=fw_compiler, keep_inference_input_mutations=False
    ), fw_graphs


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraph(TestCase):
    # [CLASS DOC] Eager and AOT-function tests of nested_compile_region without torch.compile.
    # Skipped when the suite itself runs under dynamo.
    def test_simple(self):
        # [FUNCTION DOC] A nested_compile_region-wrapped multiply called directly gives the same
        # output and input gradients as the plain function.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Elementwise multiply of the two inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region-wrapped version of the multiply.
            return nested_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_aot_function(self):
        # [FUNCTION DOC] Same comparison as the eager test, but the wrapper function is
        # traced with aot_function and a no-op compiler; checks outputs and gradients match
        # the reference.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Elementwise multiply of the two inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region-wrapped version of the multiply.
            return nested_compile_region(gn)(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = gn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_multiple(self):
        # [FUNCTION DOC] Two different regions (cos and sin) are composed, with cos used twice,
        # and traced through aot_function; the output must equal the eager result.
        @nested_compile_region
        def cos(x):
            # [FUNCTION DOC] HELPER: Region computing cosine.
            return torch.cos(x)

        @nested_compile_region
        def sin(x):
            # [FUNCTION DOC] HELPER: Region computing sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies cos, then sin, then cos again.
            a = cos(x)
            b = sin(a)
            return cos(b)

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)
        aot_fn = aot_function(fn, nop)
        res = aot_fn(x)

        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_duplicate_backward_variants_warn_and_trace(self):
        # [FUNCTION DOC] With inlining of single-use subgraphs disabled, a region called twice
        # in sequence produces two backward variants. Checks the compile emits a warning
        # mentioning multiple backward graphs, yields one forward and one backward graph, and
        # that a mocked structured-trace call logs exactly one duplicate-backward artifact whose
        # payload has the expected variant suffix, variant count, and tangent shape and stride.
        @nested_compile_region
        def block(x):
            # [FUNCTION DOC] HELPER: Region returning twice the sine of the input.
            return torch.sin(x) * 2.0

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that applies the region twice, concatenates with a tail
            # tensor along dim 1, and returns the sum of squares.
            def forward(self, x, tail):
                # [FUNCTION DOC] HELPER: Runs the region on x and then on its result,
                # concatenates the result with tail, and returns the scalar sum of squares.
                y0 = block(x)
                y1 = block(y0)
                z = torch.cat([y1, tail], dim=1)
                return z.square().sum()

        torch._dynamo.reset()
        backend = AotEagerAndRecordGraphs()
        with mock.patch("torch._logging.trace_structured") as trace_structured_mock:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                compiled = torch.compile(
                    Model(),
                    backend=backend,
                    fullgraph=True,
                )
                x = torch.randn(4, 3, 5, requires_grad=True)
                tail = torch.randn(4, 7, 5, requires_grad=True)
                compiled(x, tail).backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertTrue(
            any(
                "invoke_subgraph traced multiple backward graphs"
                in str(warning.message)
                for warning in caught
            )
        )

        duplicate_payloads = []
        for call in trace_structured_mock.call_args_list:
            args, kwargs = call
            if not args or args[0] != "artifact":
                continue
            metadata_fn = kwargs.get("metadata_fn")
            if metadata_fn is None and len(args) > 1:
                metadata_fn = args[1]
            if metadata_fn is None:
                continue
            metadata = metadata_fn()
            if metadata.get("name") != "invoke_subgraph_backward_duplicate":
                continue
            payload_fn = kwargs["payload_fn"]
            duplicate_payloads.append(payload_fn())

        self.assertEqual(len(duplicate_payloads), 1)
        payload = duplicate_payloads[0]
        self.assertEqual(payload["new_variant_suffix"], 1)
        self.assertEqual(payload["num_variants_for_identifier"], 2)
        self.assertEqual(
            payload["new_variant"]["backward_identifier"],
            payload["backward_identifier"],
        )
        tangent = payload["new_variant"]["tangents"][0]
        self.assertIsInstance(tangent["backward_input_name"], str)
        self.assertEqual(tangent["shape"], [4, 3, 5])
        self.assertEqual(tangent["stride"], [15, 5, 1])

    def test_make_fx_without_shape_env(self):
        # [FUNCTION DOC] Traces the invoke_subgraph op with make_fx under a fake tensor
        # mode that has no shape environment, and checks tracing succeeds and the generated
        # code mentions invoke_subgraph (no deferred runtime assert insertion should need a
        # shape env).
        """Test that make_fx with invoke_subgraph works without a ShapeEnv.

        When using FakeTensorMode without a ShapeEnv (shape_env=None),
        the invoke_subgraph HOP should not require a shape_env for inserting
        deferred runtime asserts.
        """
        from torch._higher_order_ops.invoke_subgraph import invoke_subgraph
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx

        def subgraph_fn(x, y):
            # [FUNCTION DOC] HELPER: Subgraph body returning a one-element tuple with x doubled
            # plus y.
            return (x * 2 + y,)

        def outer_fn(x, y):
            # [FUNCTION DOC] HELPER: Invokes the subgraph body through the invoke_subgraph op
            # with a fixed identifier string.
            return invoke_subgraph(subgraph_fn, "test_subgraph", x, y)

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        # Trace with make_fx using FakeTensorMode without a ShapeEnv
        fake_mode = FakeTensorMode()
        self.assertIsNone(fake_mode.shape_env)
        with fake_mode:
            fake_x = fake_mode.from_tensor(x)
            fake_y = fake_mode.from_tensor(y)
            traced = make_fx(outer_fn)(fake_x, fake_y)

        self.assertIn("invoke_subgraph", traced.code)


@skipIfTorchDynamo("Not a torch._dynamo test")
@torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
class TestInvokeSubgraphCompile(TestCase):
    # [CLASS DOC] Tests of nested_compile_region under torch.compile. The class decorator
    # patches dynamo config so output graph node order is canonicalized, and the class is
    # skipped under dynamo.
    def count_unique_get_attr_nodes(self, gm, args, expected):
        # [FUNCTION DOC] HELPER: Asserts that the graph module has exactly the expected number
        # of distinct get_attr target names. The args parameter is unused.
        subgraph_attr_names = set()
        for node in gm.graph.nodes:
            if node.op == "get_attr":
                subgraph_attr_names.add(node.target)
        self.assertEqual(len(subgraph_attr_names), expected)

    def test_simple(self):
        # [FUNCTION DOC] Compiles a function calling a region with the inductor backend (full
        # graph) and checks outputs and gradients match eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying its two inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_module_forward(self):
        # [FUNCTION DOC] Region applied to a module forward method, called twice in one compiled
        # function; outputs and gradients must match eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer constant attribute and a region-
            # decorated forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the integer constant 5 as an attribute.
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product of inputs plus
                # the constant.
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_stack_trace(self):
        # [FUNCTION DOC] Compiles with the aot_eager recording backend and, outside crossref
        # mode, checks that each invoke_subgraph node's recorded stack trace (after
        # stripping caret underline lines) ends at the user line calling the module, not at
        # library internals.
        # Last frame in the stack trace on invoke_subgraph should point to
        # model code, not files in torch._higher_order_ops directory.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a region-decorated forward and a
            # constant attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the integer constant 5 as an attribute.
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product of inputs plus
                # the constant.
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module; the stack trace test looks
            # for this line.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8)
        y = torch.randn(8)
        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x, y)

        graph = backend.graphs[0]
        invoke_subgraph_nodes = graph.graph.find_nodes(
            op="call_function", target=torch._higher_order_ops.invoke_subgraph
        )
        for node in invoke_subgraph_nodes:
            stack_trace = node.meta["stack_trace"]
            if not TEST_WITH_CROSSREF:
                # Strip caret lines (e.g. "  ~~~^^^^") that 3.13+ appends
                clean = re.sub(r"\n[ ~^]*[~^][ ~^]*(?=\n|\Z)", "", stack_trace)
                self.assertTrue(clean.endswith("return mod(x, y) + mod(x, y)\n"))

    def test_gen_schema(self):
        # [FUNCTION DOC] Compiles a twice-called region with aot_eager and runs backward. Checks
        # that one forward and one backward graph are produced and compares the printed schemas
        # of the invoke_subgraph nodes in both graphs against inline expected strings.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a region-decorated forward and a
            # constant attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the integer constant 5 as an attribute.
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product of inputs plus
                # the constant.
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)
        res.sum().backward()

        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        fw_schema = find_hop_schema(
            backend.fw_graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        bw_schema = find_hop_schema(
            backend.bw_graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        self.assertExpectedInline(
            str(fw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1) -> (Tensor, Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(fw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1) -> (Tensor, Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(bw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1, Tensor arg2) -> (Tensor, Tensor)""",
        )
        self.assertExpectedInline(
            str(bw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor arg0, Tensor arg1, Tensor arg2) -> (Tensor, Tensor)""",
        )

    def test_gen_schema_with_buffer_mutation(self):
        # [FUNCTION DOC] Region that mutates a registered buffer in place, compiled under no-
        # grad with the eager recording backend. Checks the graph passes the buffer as a
        # subgraph input, the schema marks the first argument as mutated, the printed graph
        # (skipped under crossref) matches the expected text, and output and final buffer value
        # equal the eager reference.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a constant and a buffer; its region forward adds
            # one to the buffer in place and returns a value depending on it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a constant and registers a ones buffer of
                # size 8.
                super().__init__()
                self.c = 5
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Mutates the buffer in place, then returns the sine of
                # the product plus the constant plus the buffer.
                self.buf.add_(1)
                return torch.mul(x, y).sin() + self.c + self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = EagerAndRecordGraphs()
        with (
            torch.no_grad(),
        ):
            res = torch.compile(fn, backend=backend, fullgraph=True)(
                mod, x_clone, y_clone
            )

        self.assertEqual(len(backend.graphs), 1)
        fw_schema = find_hop_schema(
            backend.graphs[0], torch.ops.higher_order.invoke_subgraph
        )
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_mod_buffers_buf_: "f32[8]", L_x_: "f32[8]", L_y_: "f32[8]"):
        l_mod_buffers_buf_ = L_mod_buffers_buf_
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_mod_buffers_buf_, l_x_, l_y_);  subgraph_0 = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_mod_buffers_buf_, l_x_, l_y_);  subgraph_1 = l_mod_buffers_buf_ = l_x_ = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_mod_buffers_buf_: "f32[8]", l_x_: "f32[8]", l_y_: "f32[8]"):
            add_: "f32[8]" = l_mod_buffers_buf_.add_(1);  add_ = None

            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            sin: "f32[8]" = mul.sin();  mul = None
            add: "f32[8]" = sin + 5;  sin = None
            add_1: "f32[8]" = add + l_mod_buffers_buf_;  add = l_mod_buffers_buf_ = None
            return (add_1,)
""",
            )
        self.assertExpectedInline(
            str(fw_schema[0]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor(a2!) arg0, Tensor arg1, Tensor arg2) -> ((Tensor))""",
        )
        self.assertExpectedInline(
            str(fw_schema[1]),
            """invoke_subgraph(Any subgraph, str identifier, Tensor(a2!) arg0, Tensor arg1, Tensor arg2) -> ((Tensor))""",
        )
        self.assertEqual(res, ref)
        self.assertEqual(mod.buf, mod_ref.buf)

    def test_auto_functionalize(self):
        # [FUNCTION DOC] Region reading a buffer, compiled with aot_eager and backward run.
        # Checks forward and backward graph counts, that results match eager, and that the
        # printed partitioned forward and backward graphs match inline expected text (empty
        # lines ignored).
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a constant, a buffer, and a region-decorated
            # forward that multiplies by both.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores a constant and registers a ones buffer of
                # size 8.
                super().__init__()
                self.c = 5
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product times the constant
                # times the buffer.
                return torch.mul(x, y).sin() * self.c * self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(mod, x_clone, y_clone)
        res.sum().backward()
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.assertEqual(ref, res)
        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]", primals_2: "f32[8]", primals_3: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_2, primals_3, primals_1);  partitioned_fw_subgraph_0_0 = None
        getitem_12: "f32[8]" = invoke_subgraph_4[3]
        getitem_11: "f32[8]" = invoke_subgraph_4[2]
        getitem_10: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_2, primals_3, primals_1);  partitioned_fw_subgraph_0_1 = primals_2 = primals_3 = primals_1 = None
        getitem_15: "f32[8]" = invoke_subgraph_6[3]
        getitem_14: "f32[8]" = invoke_subgraph_6[2]
        getitem_13: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_12, getitem_11, getitem_10, getitem_15, getitem_14, getitem_13)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]", primals_2: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            sin: "f32[8]" = torch.ops.aten.sin.default(mul);  mul = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(sin, 5);  sin = None
            mul_2: "f32[8]" = torch.ops.aten.mul.Tensor(mul_1, primals_2);  mul_1 = None
            return (mul_2, primals_0, primals_1, primals_2)""",
            ignore_empty_lines=True,
        )
        self.assertExpectedInline(
            normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_12: "f32[8]", getitem_11: "f32[8]", getitem_10: "f32[8]", getitem_15: "f32[8]", getitem_14: "f32[8]", getitem_13: "f32[8]", tangents_1: "f32[8]"):
        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_0', getitem_13, getitem_14, getitem_15, tangents_1);  partitioned_bw_subgraph_0_1 = getitem_13 = getitem_14 = getitem_15 = None
        getitem_2: "f32[8]" = invoke_subgraph_7[0]
        getitem_3: "f32[8]" = invoke_subgraph_7[1];  invoke_subgraph_7 = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_10, getitem_11, getitem_12, tangents_1);  partitioned_bw_subgraph_0_0 = getitem_10 = getitem_11 = getitem_12 = tangents_1 = None
        getitem_6: "f32[8]" = invoke_subgraph_5[0]
        getitem_7: "f32[8]" = invoke_subgraph_5[1];  invoke_subgraph_5 = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(getitem_2, getitem_6);  getitem_2 = getitem_6 = None
        add_2: "f32[8]" = torch.ops.aten.add.Tensor(getitem_3, getitem_7);  getitem_3 = getitem_7 = None
        return (None, add_1, add_2)
    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]", primals_2: "f32[8]", tangents_0: "f32[8]"):
            mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(tangents_0, primals_2);  tangents_0 = primals_2 = None
            mul_4: "f32[8]" = torch.ops.aten.mul.Tensor(mul_3, 5);  mul_3 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            cos: "f32[8]" = torch.ops.aten.cos.default(mul);  mul = None
            mul_5: "f32[8]" = torch.ops.aten.mul.Tensor(mul_4, cos);  mul_4 = cos = None
            mul_6: "f32[8]" = torch.ops.aten.mul.Tensor(mul_5, primals_0);  primals_0 = None
            mul_7: "f32[8]" = torch.ops.aten.mul.Tensor(mul_5, primals_1);  mul_5 = primals_1 = None
            return (mul_7, mul_6, None)""",
            ignore_empty_lines=True,
        )

    def test_buffer_mutation_works_under_no_grad(self):
        # [FUNCTION DOC] A region that mutates a buffer must compile and match eager under no-
        # grad, under inference mode, and when inputs do not require grad. Each case uses a
        # fresh module and compares both outputs and the final buffer value to a reference.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer that its region forward increments
            # in place.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers a ones buffer of size 8.
                super().__init__()
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Increments the buffer in place and returns sine of the
                # product times the buffer.
                self.buf.add_(1)
                return torch.mul(x, y).sin() * self.buf

        mod_ref = Mod()
        mod = Mod()

        def fn(mod, x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(mod_ref, x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        with torch.no_grad():
            res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

        mod = Mod()
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        with torch.inference_mode():
            res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

        mod = Mod()
        x_clone = x.detach().clone().requires_grad_(False)
        y_clone = y.detach().clone().requires_grad_(False)
        res = torch.compile(fn, fullgraph=True)(mod, x_clone, y_clone)
        self.assertEqual(ref, res)
        self.assertEqual(mod_ref.buf, mod.buf)

    def test_buffer_mutation_errors_under_training(self):
        # [FUNCTION DOC] With inputs requiring grad, compiling a region that mutates a buffer
        # must raise RuntimeError saying training with in-place input or buffer mutations is
        # not supported.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a buffer that its region forward increments
            # in place.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers a ones buffer of size 8.
                super().__init__()
                self.register_buffer("buf", torch.ones(8, requires_grad=False))

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Increments the buffer in place and returns sine of the
                # product times the buffer.
                self.buf.add_(1)
                return torch.mul(x, y).sin() * self.buf

        mod = Mod()

        def fn(mod, x, y):
            # [FUNCTION DOC] HELPER: Sums two calls of the module.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        with self.assertRaisesRegex(
            RuntimeError,
            "does not currently support training with in-place input or buffer mutations",
        ):
            torch.compile(fn, backend="inductor", fullgraph=True)(mod, x, y)

    def test_list(self):
        # [FUNCTION DOC] Region returning a list of two tensors; the caller appends a third and
        # sums them. Checks outputs and gradients against eager under inductor.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning a list with product and sum of inputs.
            return [torch.mul(x, y), torch.add(x, y)]

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Appends the sine of x to the region result and sums all
            # three entries.
            lst = gn(x, y)
            lst.append(torch.sin(x))
            return lst[0] + lst[1] + lst[2]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_tuple_of_tuple(self):
        # [FUNCTION DOC] Region returning nested tuples; the caller adds the inner and outer
        # items. Checks outputs and gradients against eager under inductor.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning a tuple holding a one-tuple of the product
            # and the sum.
            return ((torch.mul(x, y),), torch.add(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the nested product and the sum from the region result.
            tup = gn(x, y)
            return tup[0][0] + tup[1]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    @unittest.skip("FunctionCtx ops is not cacheable right now")
    def test_differing_strides_for_grad_outs(self):
        # [FUNCTION DOC] Skipped test. A custom autograd function inside a region has a backward
        # that views the gradient, so a permuted output would break if gradients were not made
        # contiguous; compiled with aot_eager and compared with eager.
        class CustomOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with sine forward and a backward that
            # reshapes the incoming gradient and returns its cosine.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Views the gradient as a 12 by 5 matrix, reshapes it to
                # 3 by 4 by 5, and returns its cosine.
                a = grad_out.view(12, 5)
                return torch.cos(torch.reshape(a, (3, 4, 5)))

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying the custom autograd function.
            return CustomOp.apply(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region and permutes the output dims so
            # strides change.
            a = gn(x)
            # Force stride changes so that backward view causes a failure if
            # contiguous not called.
            b = torch.permute(a, (0, 2, 1))
            return b

        x = torch.randn(3, 4, 5, requires_grad=True)
        ref = torch.permute(gn(x), (0, 2, 1))

        x_clone = x.clone().detach().requires_grad_(True)
        opt_fn = torch.compile(fn, backend="aot_eager")
        res = opt_fn(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    @requires_cuda_and_triton
    @unittest.skipIf(not SM80OrLater, "Requires sm80 or later.")
    def test_sdpa(self):
        # [FUNCTION DOC] Compiles scaled dot product attention (flash backend, causal) in a
        # region with inductor on CUDA in bfloat16; requires CUDA, Triton and SM80 or later.
        # Compares to eager, runs backward, then calls the compiled function again to
        # exercise reuse.
        @nested_compile_region
        def gn(q, k, v):
            # [FUNCTION DOC] HELPER: Region computing causal scaled dot product attention
            # without a mask.
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True
            )

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: Restricts SDPA to the flash attention backend and calls
            # the region.
            with torch.nn.attention.sdpa_kernel(
                [torch.nn.attention.SDPBackend.FLASH_ATTENTION]
            ):
                return gn(q, k, v)

        q = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            1, 1, 32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )

        ref = fn(q, k, v)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(q, k, v)
        res.sum().backward()
        self.assertEqual(ref, res)

        res = opt_fn(q, k, v)
        res.sum().backward()

    def test_symint_from_fwd_to_bwd(self):
        # [FUNCTION DOC] Region whose reshape depends on input shapes. The same compiled
        # function is run with three different shape pairs (dynamic shapes), checking results
        # against eager and running backward on the later two so symbolic sizes flow from
        # forward to backward.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Sums x over dim 1, reshapes to the transposed shape of y,
            # and matrix-multiplies with y.
            a = torch.sum(x, (1,), keepdim=True).view(y.shape[1], y.shape[0])
            return torch.matmul(a, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x = torch.randn(64, 1, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

        x = torch.randn(256, 1, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

        x = torch.randn(16, 1, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)
        res.sum().backward()

    @inductor_config.patch("fx_graph_cache", False)
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_dropout_checks_joint_graph(self):
        # [FUNCTION DOC] Region with dropout, compiled via inductor with the FX graph cache
        # disabled and single-use inlining off. Checks (outside crossref) the printed
        # inductor graph shows random seed ops in the dropout partition, i.e. joint-graph
        # passes ran on the HOP graphs. Numeric results are not compared since RNG differs
        # between eager and Triton.
        # `dropout` tests that joint graph passes (not just partitioner) is ran
        # on the hop graphs. Inductor rng functionalization happens in the joint
        # graph passes. Without running joint graph passes, we would get an
        # error like AssertionError: should have been handled in
        # replace_random.py
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying dropout with probability 0.5 to the sine of
            # the input.
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        @nested_compile_region
        def hn(x):
            # [FUNCTION DOC] HELPER: Region computing sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the outputs of both regions.
            return gn(x) + hn(x)

        x = torch.randn(8, requires_grad=True)
        # Difficult to check the results here because we random does not match
        # between eager and Triton.
        backend = InductorAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x)
        res.sum().backward()

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1);  partitioned_fw_subgraph_0_0 = None
        getitem_7: "b8[8]" = invoke_subgraph_4[2]
        getitem_6: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_1_0 = self.partitioned_fw_subgraph_1_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_1_0, 'partitioned_fw_subgraph_1_0', primals_1);  partitioned_fw_subgraph_1_0 = primals_1 = None
        getitem_8: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None

        add: "f32[8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_7, getitem_6, getitem_8)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)

            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))

            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1, primals_0, gt)

    class partitioned_fw_subgraph_1_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(primals_0)
            return (sin, primals_0)
""",
                ignore_empty_lines=True,
            )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @inductor_config.patch("fx_graph_cache", False)
    def test_dropout_checks_joint_graph_inference(self):
        # [FUNCTION DOC] Same dropout region compiled with inputs not requiring grad. Checks the
        # inductor graph has the random seed ops inside the single repeated subgraph and matches
        # the inline expected text (outside crossref).
        # Checks that joint graph results in inductor seeds for just the inference graph
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying dropout with probability 0.5 to the sine of
            # the input.
            return torch.nn.functional.dropout(torch.sin(x), p=0.5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        backend = InductorAndRecordGraphs()
        x = torch.randn(8, requires_grad=False)
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            inductor_seeds_default: "i64[1]" = torch.ops.prims.inductor_seeds.default(1, device(type='cpu'))

            inductor_lookup_seed_default: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0);  inductor_seeds_default = None
            inductor_random_default: "f32[8]" = torch.ops.prims.inductor_random.default([8], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
            gt: "b8[8]" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.5);  inductor_random_default = None
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(gt, sin);  gt = sin = None
            mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(mul, 2.0);  mul = None
            return (mul_1,)
""",
                ignore_empty_lines=True,
            )

    def test_dedupe(self):
        # [FUNCTION DOC] Calls the same region twice. Checks results and gradients, that the
        # dynamo, forward and backward graphs each hold a single subgraph module (via the
        # get_attr counter), and that printed dynamo and forward graphs match expected text.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying its inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the region to the inputs and then to the result
            # with y.
            a = gn(x, y)
            return gn(a, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 1)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', getitem, l_y_);  subgraph_1 = getitem = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            return (mul,)
""",
            )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8]", primals_2: "f32[8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = primals_1 = None
        getitem_9: "f32[8]" = invoke_subgraph_4[2]
        getitem_8: "f32[8]" = invoke_subgraph_4[1]
        getitem: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None

        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', getitem, primals_2);  partitioned_fw_subgraph_0_1 = getitem = primals_2 = None
        getitem_11: "f32[8]" = invoke_subgraph_6[2]
        getitem_10: "f32[8]" = invoke_subgraph_6[1]
        getitem_1: "f32[8]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        return (getitem_1, getitem_9, getitem_8, getitem_11, getitem_10)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8]", primals_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(primals_0, primals_1)
            return (mul, primals_0, primals_1)
""",
            ignore_empty_lines=True,
        )

    def test_no_raise_on_reparameterize_module(self):
        # [FUNCTION DOC] A region calling functional_call on a linear module with explicit
        # parameter and buffer dicts, with the dynamo allow_side_effects option on the HOP
        # variable patched off. Must compile without error, match eager, and produce expected
        # printed graphs where parameters are lifted as graph inputs.
        mod = torch.nn.Linear(10, 10)
        params = dict(mod.named_parameters())
        buffers = dict(mod.named_buffers())
        params_and_buffers = {
            **dict(params),
            **dict(buffers),
        }

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region calling the module functionally with the
            # captured dicts.
            return torch.func.functional_call(
                mod, parameter_and_buffer_dicts=params_and_buffers, args=(x,)
            )

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two calls of the region.
            return gn(x) + gn(x)

        x = torch.randn(10, 10)
        ref = fn(x)
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with mock.patch(
            "torch._dynamo.variables.higher_order_ops.InvokeSubgraphHigherOrderVariable.allow_side_effects",
            False,
        ):
            res = opt_fn(x)

        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_: "f32[10]", L_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_: "f32[10, 10]", L_x_: "f32[10, 10]"):
        l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_ = L_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_
        l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_ = L_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_);  subgraph_0 = None
        getitem: "f32[10, 10]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_);  subgraph_1 = l_x_ = l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_ = l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_ = None
        getitem_1: "f32[10, 10]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[10, 10]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[10, 10]", l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_: "f32[10, 10]", l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_: "f32[10]"):
            linear: "f32[10, 10]" = torch._C._nn.linear(l_x_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_, l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_);  l_x_ = l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_weight_ = l_gn_closure_0_cell_contents_closure_0_cell_contents_parameters_bias_ = None
            return (linear,)
""",
            )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[10]", primals_2: "f32[10, 10]", primals_3: "f32[10, 10]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_3, primals_2, primals_1);  partitioned_fw_subgraph_0_0 = None
        getitem_10: "f32[10, 10]" = invoke_subgraph_4[1]
        getitem: "f32[10, 10]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', primals_3, primals_2, primals_1);  partitioned_fw_subgraph_0_1 = primals_3 = primals_2 = primals_1 = None
        getitem_11: "f32[10, 10]" = invoke_subgraph_6[1]
        getitem_1: "f32[10, 10]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        add: "f32[10, 10]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add, getitem_10, getitem_11)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[10, 10]", primals_1: "f32[10, 10]", primals_2: "f32[10]"):
            t: "f32[10, 10]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
            addmm: "f32[10, 10]" = torch.ops.aten.addmm.default(primals_2, primals_0, t);  primals_2 = t = None
            return (addmm, primals_0)""",
            ignore_empty_lines=True,
        )

    def test_raise_with_reparameterize_but_module_mutation(self):
        # [FUNCTION DOC] Like the reparameterize test, but the module forward assigns a nonlocal
        # variable (a side effect). With allow_side_effects patched off, compilation must raise
        # RuntimeError mentioning the invoke_subgraph higher order operator.
        foo = 0

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear-layer module whose forward sets a captured outer
            # variable to one before applying the layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 10 by 10 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets the nonlocal flag to one (side effect) and applies
                # the linear layer.
                nonlocal foo
                foo = 1
                return self.linear(x)

        mod = Mod()
        params = dict(mod.named_parameters())
        buffers = dict(mod.named_buffers())
        params_and_buffers = {
            **dict(params),
            **dict(buffers),
        }

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region calling the module functionally with the
            # captured dicts.
            return torch.func.functional_call(
                mod, parameter_and_buffer_dicts=params_and_buffers, args=(x,)
            )

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two calls of the region.
            return gn(x) + gn(x)

        x = torch.randn(10, 10)
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        with mock.patch(
            "torch._dynamo.variables.higher_order_ops.InvokeSubgraphHigherOrderVariable.allow_side_effects",
            False,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"Higher Order Operator: torch\.ops\.higher_order\.invoke_subgraph",
            ):
                opt_fn(x)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_dce(self):
        # [FUNCTION DOC] Dead code elimination: an unused cosine inside the region must not
        # appear in the AOT forward subgraph. Compiled without grad and matched against
        # expected text.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning the sine; also computes an unused cosine.
            x = torch.sin(x)
            # should be dce'd
            y = torch.cos(x)  # noqa: F841
            return x

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(
            torch.randn(4, requires_grad=False)
        )

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[4]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4]"):
            sin: "f32[4]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None
            return (sin,)
""",
            )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_dce_recursive(self):
        # [FUNCTION DOC] Region wrapping a non-reentrant checkpoint of a function with an unused
        # cosine, result detached; compiled with the eager recording backend with the config
        # option skipping forward side effects in backward. Outside crossref, compares the
        # dynamo graph text with expected output (one invoke_subgraph whose body holds the
        # checkpoint HOP and its nested body module).
        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns the sine of x; computes and discards the cosine.
            a = torch.sin(x)
            _ = torch.cos(x)  # unused intermediate
            return a

        @nested_compile_region
        def fn1_checkpoint(x):
            # [FUNCTION DOC] HELPER: Region running the previous function under non-
            # reentrant checkpoint.
            return torch.utils.checkpoint.checkpoint(fn1, x, use_reentrant=False)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the checkpoint region and detaches the result.
            return fn1_checkpoint(x).detach()

        x = torch.randn(8, requires_grad=True)

        with torch._dynamo.config.patch(
            skip_fwd_side_effects_in_bwd_under_checkpoint=True
        ):
            backend = EagerAndRecordGraphs()
            torch.compile(fn, backend=backend, fullgraph=True)(x)

            if not TEST_WITH_CROSSREF:
                # Verify that DCE applied recursively:
                # - invoke_subgraph subgraph should be DCE'd
                # - nested tag_activation_checkpoint subgraph should also be DCE'd (requires recursion)
                self.assertExpectedInline(
                    normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        detach: "f32[8]" = getitem.detach();  getitem = None
        return (detach,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]"):
            wrap_body_0 = self.wrap_body_0
            tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
            getitem: "f32[8]" = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
            return (getitem,)

        class wrap_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[8]"):
                a: "f32[8]" = torch.sin(l_x_)

                _: "f32[8]" = torch.cos(l_x_);  l_x_ = _ = None
                return (a,)
""",
                )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_nonlocal_update(self):
        # [FUNCTION DOC] A region reads a nonlocal counter whose value differs between two
        # calls (2 then 3). Single-use inlining is off. Checks inductor results and gradients
        # match eager, then with aot_eager that the dynamo graph has two separate subgraphs,
        # one per constant.
        counter = 2

        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning a one-tuple of product scaled by the
            # nonlocal counter.
            nonlocal counter
            return (torch.mul(x, y) * counter,)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sets the counter to 2, calls the region, sets it to 3,
            # calls again on the result, and returns the second output.
            nonlocal counter
            counter = 2
            a = gn(x, y)[0]
            counter = 3
            return gn(a, y)[0]

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone, y_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        torch._dynamo.reset()
        backend = AotEagerAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x_clone, y_clone)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', getitem, l_y_);  subgraph_1 = getitem = l_y_ = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(l_x_, l_y_);  l_x_ = l_y_ = None
            mul_1: "f32[8]" = mul * 2;  mul = None
            return (mul_1,)

    class subgraph_1(torch.nn.Module):
        def forward(self, a: "f32[8]", l_y_: "f32[8]"):
            mul: "f32[8]" = torch.mul(a, l_y_);  a = l_y_ = None
            mul_1: "f32[8]" = mul * 3;  mul = None
            return (mul_1,)
""",
            )

    def test_nonlocal_list_mutation_hidden(self):
        # [FUNCTION DOC] A region appends a tensor to a list passed in from outside. Compiling
        # with aot_eager must handle that hidden list mutation and extra outputs, giving the
        # same outputs as eager for both the tensor result and the list item.
        """Test that nonlocal list mutation inside nested_compile_region is handled correctly."""

        @nested_compile_region
        def gn(x, z):
            # [FUNCTION DOC] HELPER: Region that appends sin of x to the list and returns
            # a tuple.
            o = torch.matmul(x, x) @ x
            out = x.sin()
            z.append(out)
            return torch.cos(torch.sin(o)), torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an empty list, calls the region, and returns the
            # sum of first and last outputs plus the first list item.
            z = []

            outs = gn(x, z)
            out1 = outs[0]
            # Check that the extra output pytree handling is done properly
            out2 = outs[-1]

            return out1 + out2, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @inductor_config.patch("fx_graph_cache", False)
    def test_view_to_reshape(self):
        # [FUNCTION DOC] Checks that a view call inside the region becomes a reshape in the
        # inductor graph (checked against expected printed text outside crossref); FX graph
        # cache off.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region: sine, view to 1 by 8, sine.
            x = torch.sin(x)
            x = x.view(1, 8)
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        torch._dynamo.reset()
        backend = InductorAndRecordGraphs()
        torch.compile(fn, backend=backend, fullgraph=True)(x)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(
                    backend.inductor_graphs[0].print_readable(print_output=False)
                ),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', arg0_1);  repeated_subgraph0 = arg0_1 = None
        getitem: "f32[1, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        return (getitem,)

    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]"):
            sin: "f32[8]" = torch.ops.aten.sin.default(arg0_1);  arg0_1 = None

            view: "f32[1, 8]" = torch.ops.aten.reshape.default(sin, [1, 8]);  sin = None

            sin_1: "f32[1, 8]" = torch.ops.aten.sin.default(view);  view = None
            return (sin_1,)
""",
            )

    def test_normalize_gm(self):
        # [FUNCTION DOC] Region with an internal loop is called five times in a loop. Checks the
        # normalized printed graph reuses a single subgraph name for all calls.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying x by y five times in a loop.
            # Different graph give different names to intermediate nodes
            for _ in range(5):
                x = x * y
            return x

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the region five times, feeding output back
            # as input.
            for _ in range(5):
                x = gn(x, y)
            return x

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        opt_fn(x, y)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", L_y_: "f32[8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', getitem, l_y_);  subgraph_1 = getitem = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        subgraph_2 = self.subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_2, 'subgraph_0', getitem_1, l_y_);  subgraph_2 = getitem_1 = None
        getitem_2: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        subgraph_3 = self.subgraph_0
        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(subgraph_3, 'subgraph_0', getitem_2, l_y_);  subgraph_3 = getitem_2 = None
        getitem_3: "f32[8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        subgraph_4 = self.subgraph_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(subgraph_4, 'subgraph_0', getitem_3, l_y_);  subgraph_4 = getitem_3 = l_y_ = None
        getitem_4: "f32[8]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        return (getitem_4,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", l_y_: "f32[8]"):
            x: "f32[8]" = l_x_ * l_y_;  l_x_ = None
            x_1: "f32[8]" = x * l_y_;  x = None
            x_2: "f32[8]" = x_1 * l_y_;  x_1 = None
            x_3: "f32[8]" = x_2 * l_y_;  x_2 = None
            x_4: "f32[8]" = x_3 * l_y_;  x_3 = l_y_ = None
            return (x_4,)
""",
            )

    def test_input_mutation(self):
        # [FUNCTION DOC] Region mutating its input in place (no grad). The compiled function
        # result must equal the eager result on a cloned input.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region adding one to x in place and returning the product.
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        x_clone = x.clone()
        self.assertEqual(opt_fn(x, y), fn(x_clone, y))

    def test_input_mutation_mutiple_times(self):
        # [FUNCTION DOC] Region mutating its input is called 17 times in a loop under no-grad.
        # The output and the mutated input must match an eager run.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region adding one to x in place and returning the product.
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region once then adds 16 more calls' results
            # in place.
            z = gn(x, y)
            for _ in range(16):
                z += gn(x, y)
            return z

        x = torch.randn(8, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with (
            torch.no_grad(),
        ):
            out = opt_fn(x, y)
        exp_out = fn(x_clone, y)
        self.assertEqual(exp_out, out)
        self.assertEqual(x_clone, x)

    def test_input_mutation_mutiple_times_fake_tensor_cache_hit(self):
        # [FUNCTION DOC] Same as the repeated-mutation test but with fake propagation of the HOP
        # replaced by a counting mock. Asserts fake propagation runs exactly twice (once in AOT
        # metadata analysis, once in graph creation, whose cache keys differ), later calls hit
        # the cache, and results and mutated input match eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region adding one to x in place and returning the product.
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region once then adds 16 more calls' results
            # in place.
            z = gn(x, y)
            for _ in range(16):
                z += gn(x, y)
            return z

        x = torch.randn(8, requires_grad=False)
        x_clone = x.clone()
        y = torch.randn(8, requires_grad=False)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        fake_prop_count = 0

        def _mock_invoke_subgraph(mode, subgraph, identifier, *operands):
            # [FUNCTION DOC] HELPER: Replacement fake implementation of the invoke_subgraph op.
            # Increments the enclosing call counter and returns a one-tuple holding a clone of
            # the first operand.
            nonlocal fake_prop_count
            fake_prop_count += 1
            return (operands[0].clone(),)

        with (
            mock.patch.dict(
                "torch._higher_order_ops.utils.registered_hop_fake_fns",
                {torch.ops.higher_order.invoke_subgraph: _mock_invoke_subgraph},
            ),
            torch.no_grad(),
        ):
            out = opt_fn(x, y)

        # Fake propagation occurs only twice, with subsequent calls using cached results.
        #
        # First fake propagation (in collect_metadata_analysis of AOT):
        #   - Uses the original Dynamo graph
        #   - Flow: functionalization -> fake tensor
        #
        # Second fake propagation (in _create_graph of AOT):
        #   - Uses a materialized graph that includes epilogue operations
        #   - Flow: functionalization -> proxy -> fake tensor
        #
        # The key difference: the second time we materialize the graph with epilogue
        # operations included in the proxy key. Since the dynamo graph module is not
        # in the functional + epilogue format, the cache key should be different,
        # preventing cache reuse between these two phases.
        self.assertEqual(fake_prop_count, 2)
        exp_out = fn(x_clone, y)
        self.assertEqual(exp_out, out)
        self.assertEqual(x_clone, x)

    @unittest.expectedFailure
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_input_mutation_inference_mode(self):
        # [FUNCTION DOC] Expected-failure test: a region that mutates its input inside inference
        # mode, whose input was created outside it, should raise RuntimeError about in-place
        # update to an inference tensor outside inference mode when the compiled function runs.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region adding one to x in place and returning the product.
            x.add_(1)
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Computes cos, then calls the region on a second cos inside
            # inference mode.
            z = torch.cos(x)
            with torch.inference_mode():
                return gn(torch.cos(z), y)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        with self.assertRaisesRegex(
            RuntimeError,
            "Inplace update to inference tensor outside InferenceMode is not allowed",
        ):
            opt_fn(x, y)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_input_mutation_alias_annotated_custom_op(self):
        # [FUNCTION DOC] Registers a temporary custom op in a scoped library whose second
        # argument is annotated as mutated, and calls it inside a region that also mutates its
        # input. Checks results and the mutated input against eager and that the AOT forward
        # graph shows auto functionalization inside the subgraph (expected text compared).
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("scale_into(Tensor x, Tensor! out, float scale) -> ()")

            def scale_into(x, out, scale):
                # [FUNCTION DOC] HELPER: Custom op implementation: writes x scaled by scale
                # into out.
                out.copy_(x * scale)

            lib.impl("scale_into", scale_into, "CompositeExplicitAutograd")

            @nested_compile_region
            def gn(x, y):
                # [FUNCTION DOC] HELPER: Region: allocates an output buffer like y, fills it via
                # the custom op, adds it to x in place, and returns the product of x and y.
                out = torch.empty_like(y)
                torch.ops.mylib.scale_into(y, out, 2.0)
                x.add_(out)
                return torch.mul(x, y)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Calls the region.
                return gn(x, y)

            # x is mutated in place, so it cannot require grad (autograd leaf error).
            x = torch.randn(8, requires_grad=False)
            x_clone = x.clone()
            y = torch.randn(8, requires_grad=False)

            backend = AotEagerAndRecordGraphs()
            opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

            self.assertEqual(opt_fn(x, y), fn(x_clone, y))
            self.assertEqual(x_clone, x)

            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
        auto_functionalized_subgraph_0 = self.auto_functionalized_subgraph_0
        _tree_spec_constant0 = self._tree_spec_constant0
        auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.higher_order.invoke_subgraph, subgraph = auto_functionalized_subgraph_0, identifier = 'auto_functionalized_subgraph_0', arg0 = arg1_1, _arg1_base_index = 0, _all_bases = [arg0_1], _op_schema = _tree_spec_constant0);  auto_functionalized_subgraph_0 = arg1_1 = _tree_spec_constant0 = None
        getitem: "f32[8]" = auto_functionalized_v2[0]
        getitem_1: "f32[8]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
        copy_: "f32[8]" = torch.ops.aten.copy_.default(arg0_1, getitem_1);  arg0_1 = getitem_1 = copy_ = None
        return (getitem,)
    class auto_functionalized_subgraph_0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
            empty_like: "f32[8]" = torch.ops.aten.empty_like.default(arg0_1, pin_memory = False)
            auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.scale_into.default, x = arg0_1, scale = 2.0, _out_base_index = 0, _all_bases = [empty_like]);  empty_like = None
            getitem_1: "f32[8]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
            add: "f32[8]" = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  getitem_1 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(add, arg0_1);  arg0_1 = None
            copy_: "f32[8]" = torch.ops.aten.copy_.default(arg1_1, add);  arg1_1 = add = copy_ = None
            return (mul,)""",
                ignore_empty_lines=True,
            )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_input_mutation_alias_annotated_custom_op_runtime_epilogue(self):
        # [FUNCTION DOC] Like the previous test, but the region also mutates a third input and
        # the backend keeps mutations out of the graph so a runtime epilogue applies them.
        # Checks all three inputs match eager afterwards and the forward graph text matches.
        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define("scale_into(Tensor x, Tensor! out, float scale) -> ()")

            def scale_into(x, out, scale):
                # [FUNCTION DOC] HELPER: Custom op implementation: writes x scaled by scale
                # into out.
                out.copy_(x * scale)

            lib.impl("scale_into", scale_into, "CompositeExplicitAutograd")

            @nested_compile_region
            def gn(x, y, z):
                # [FUNCTION DOC] HELPER: Region: uses the custom op to fill a buffer, adds it to
                # x in place, subtracts y from z in place, and returns the product of x and z.
                out = torch.empty_like(y)
                torch.ops.mylib.scale_into(y, out, 2.0)
                x.add_(out)
                z.sub_(y)
                return torch.mul(x, z)

            def fn(x, y, z):
                # [FUNCTION DOC] HELPER: Calls the region.
                return gn(x, y, z)

            backend, fw_graphs = _aot_eager_with_runtime_epilogue()
            opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
            x = torch.randn(8, requires_grad=False)
            x_clone = x.clone()
            y = torch.randn(8, requires_grad=False)
            y_clone = y.clone()
            z = torch.randn(8, requires_grad=False)
            z_clone = z.clone()

            self.assertEqual(opt_fn(x, y, z), fn(x_clone, y_clone, z_clone))
            self.assertEqual(x_clone, x)
            self.assertEqual(y_clone, y)
            self.assertEqual(z_clone, z)
            self.assertExpectedInline(
                normalize_gm(fw_graphs[0].print_readable(print_output=False)),
                """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]", arg2_1: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'functionalized_subgraph_0', arg1_1, arg0_1, arg2_1);  repeated_subgraph0 = arg1_1 = arg0_1 = arg2_1 = None
        getitem: "f32[8]" = invoke_subgraph[0]
        getitem_1: "f32[8]" = invoke_subgraph[1]
        getitem_2: "f32[8]" = invoke_subgraph[2];  invoke_subgraph = None
        return (getitem_1, getitem_2, getitem)
    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]", arg2_1: "f32[8]"):
            empty_like: "f32[8]" = torch.ops.aten.empty_like.default(arg0_1, pin_memory = False)
            auto_functionalized_v2 = torch.ops.higher_order.auto_functionalized_v2(torch.ops.mylib.scale_into.default, x = arg0_1, scale = 2.0, _out_base_index = 0, _all_bases = [empty_like]);  empty_like = None
            getitem_1: "f32[8]" = auto_functionalized_v2[1];  auto_functionalized_v2 = None
            add: "f32[8]" = torch.ops.aten.add.Tensor(arg1_1, getitem_1);  arg1_1 = getitem_1 = None
            sub: "f32[8]" = torch.ops.aten.sub.Tensor(arg2_1, arg0_1);  arg2_1 = arg0_1 = None
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(add, sub)
            return (mul, add, sub)""",
                ignore_empty_lines=True,
            )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    @torch._functorch.config.patch(unlift_effect_tokens=True)
    def test_input_mutation_with_effect_runtime_epilogue(self):
        # [FUNCTION DOC] Region with a side-effecting print op plus an in-place input mutation,
        # with effect tokens unlifted and the runtime-epilogue backend. Checks output and input
        # match eager, and the forward graph text uses the functionalized subgraph, a
        # with_effects call, no copy op, and no auto-functionalized invoke_subgraph wrapper.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region: print effect op, add y into x in place, return
            # the product.
            torch.ops.aten._print.default("effect")
            x.add_(y)
            return x * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        backend, fw_graphs = _aot_eager_with_runtime_epilogue()
        x = torch.randn(8)
        x_clone = x.clone()
        y = torch.randn(8)
        actual = torch.compile(fn, backend=backend, fullgraph=True)(x, y)
        expected = fn(x_clone, y)

        self.assertEqual(actual, expected)
        self.assertEqual(x, x_clone)
        graph = normalize_gm(fw_graphs[0].print_readable(print_output=False))
        self.assertIn("'functionalized_subgraph_0'", graph)
        self.assertIn("torch.ops.higher_order.with_effects", graph)
        self.assertNotIn("torch.ops.aten.copy_.default", graph)
        self.assertNotIn(
            "auto_functionalized_v2(torch.ops.higher_order.invoke_subgraph", graph
        )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_repeated_input_mutation_view_runtime_epilogue(self):
        # [FUNCTION DOC] Region mutating a slice view of its input is called twice on the same
        # slice. Checks result and base tensor match eager, both calls share one
        # functionalized subgraph, a slice scatter appears, and there is no copy op or auto-
        # functionalized wrapper.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region adding two to x in place and returning its sine.
            x.add_(2)
            return x.sin()

        def fn(base):
            # [FUNCTION DOC] HELPER: Sums region calls on two identical slices of the base.
            return gn(base[1:5]) + gn(base[1:5])

        backend, fw_graphs = _aot_eager_with_runtime_epilogue()
        base = torch.randn(8)
        base_clone = base.clone()
        actual = torch.compile(fn, backend=backend, fullgraph=True)(base)
        expected = fn(base_clone)

        self.assertEqual(actual, expected)
        self.assertEqual(base, base_clone)
        graph = normalize_gm(fw_graphs[0].print_readable(print_output=False))
        invoke_subgraph_nodes = fw_graphs[0].graph.find_nodes(
            op="call_function", target=torch._higher_order_ops.invoke_subgraph
        )
        self.assertEqual(len(invoke_subgraph_nodes), 2)
        self.assertEqual(
            invoke_subgraph_nodes[0].args[0].target,
            invoke_subgraph_nodes[1].args[0].target,
        )
        self.assertEqual(graph.count("'functionalized_subgraph_0'"), 2)
        self.assertIn("torch.ops.aten.slice_scatter.default", graph)
        self.assertNotIn("torch.ops.aten.copy_.default", graph)
        self.assertNotIn(
            "auto_functionalized_v2(torch.ops.higher_order.invoke_subgraph", graph
        )

    def test_input_mutation_boxed_subgraph(self):
        # [FUNCTION DOC] Builds a symbolically traced module that mutates its first input,
        # switches its graph to boxed calling convention, and checks that
        # FunctionalCallableWithEpilogue runs it with a list of inputs, matching eager
        # including the mutated input. Also checks FunctionalizeCtxWrapper under a
        # functionalize API returns the output plus updated mutated input while leaving the
        # original input untouched.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds y to x in place and returns a one-tuple of
            # the product.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Mutates x in place and returns the product in a tuple.
                x.add_(y)
                return (torch.mul(x, y),)

        gm = torch.fx.symbolic_trace(Mod())
        gm.graph.set_codegen(_BoxedCodeGen())
        gm.recompile()
        self.assertTrue(gm._boxed_call)

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        x_clone = x.clone()
        ref = Mod()(x_clone, y)

        self.assertEqual(FunctionalCallableWithEpilogue(gm)([x, y]), ref)
        self.assertEqual(x_clone, x)

        x = torch.randn(8, requires_grad=False)
        x_clone = x.clone()
        ctx = PythonFunctionalizeAPI(FunctionalTensorMode(_keep_input_mutations=False))
        functionalized = FunctionalizeCtxWrapper(ctx, gm, mutated_input_indices=(0,))
        out, updated_x = functionalized([x, y])
        self.assertEqual(x, x_clone)
        self.assertEqual(updated_x, x_clone + y)
        self.assertEqual(out, updated_x * y)

    def test_simple_module(self):
        # [FUNCTION DOC] Region calling a linear module and a cosine, compiled by inductor with
        # inputs not requiring grad (so joint graph outputs are None). Compares output; the
        # backward calls and gradient comparison then run on non-grad tensors.
        mod = torch.nn.Linear(8, 8)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning a cosine and the module output.
            return torch.cos(x), mod(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the two region outputs.
            out = gn(x)
            return out[0] + out[1]

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # requires_grad is False deliberately to force None the joint_graph
        # outputs
        x = torch.randn(8, 8, requires_grad=False)
        x_clone = x.detach().clone().requires_grad_(False)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

    def test_fail_with_direct_invoke_subgraph(self):
        # [FUNCTION DOC] Calling the invoke_subgraph function directly inside compiled
        # code must raise the dynamo Unsupported error with a message about directly
        # using invoke_subgraph.
        from torch._higher_order_ops import invoke_subgraph

        def gn(x):
            # [FUNCTION DOC] HELPER: Plain function returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls invoke_subgraph directly with the function, no
            # identifier and a tuple of operands.
            return invoke_subgraph(gn, None, (x,))

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "Directly using invoke_subgraph is not"
        ):
            opt_fn(x)

    def test_input_output_aliasing(self):
        # [FUNCTION DOC] A region that returns its input unchanged must raise the dynamo
        # UncapturedHigherOrderOpError mentioning aliasing during higher order op tracing.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning its input x and the product.
            return (x, torch.mul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the two region outputs.
            outs = gn(x, y)
            return outs[0] * outs[1]

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x, y)

    def test_input_input_aliasing(self):
        # [FUNCTION DOC] Passing a tensor and a view of it as two region inputs must raise
        # UncapturedHigherOrderOpError about aliasing during higher order op tracing.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying its two inputs.
            return torch.mul(x, y)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region with x and a 1 by 8 view of x.
            return gn(x, x.view(1, 8))

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_output_output_aliasing(self):
        # [FUNCTION DOC] A region returning a tensor and a view of it must raise
        # UncapturedHigherOrderOpError about aliasing during higher order op tracing.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning cos(x) and a 1 by 8 view of the
            # same tensor.
            z = torch.cos(x)
            return z, z.view(1, 8)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.UncapturedHigherOrderOpError,
            "Encountered aliasing during higher order op tracing",
        ):
            opt_fn(x)

    def test_side_effect_with_aliased_intermediate(self):
        # [FUNCTION DOC] A region stores a view of an intermediate in an outer list, and the
        # caller uses it afterwards. Compilation must raise InternalTorchDynamoError whose
        # message says the value aliases an input or output and tells the user to clone it.
        captured_views = []

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region: sine, view as 1 by 8, append the view to the outer
            # list, return the sine of the view.
            original = torch.sin(x)
            view = original.view(1, 8)  # Aliases with original
            captured_views.append(view)
            return torch.sin(view)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region and adds the captured view if the list
            # is non-empty.
            result = gn(x)
            if captured_views:
                return result + captured_views[0]
            return result

        x = torch.randn(8, requires_grad=False)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        # When a filtered aliased intermediate is captured by side effects,
        # the tainted proxy raises a clear error telling the user to clone.
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "aliases an input or output.*clone",
        ):
            opt_fn(x)

    def test_mod_attr_aliasing(self):
        # [FUNCTION DOC] A region calls a module that mutates an attribute tensor in place.
        # Compiles with inductor, then resets the attribute and checks the compiled result
        # equals eager.
        class MutateParam(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a ones tensor attribute that its forward adds one
            # to in place before multiplying the input with it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the size 8 ones attribute.
                super().__init__()
                self.a = torch.ones(8)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Increments the attribute in place and returns input
                # times it.
                self.a.add_(1)
                return torch.mul(x, self.a)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region calling the module (defined later in the test body).
            return mod(x)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Multiplies the region output with y.
            return gn(x) * y

        mod = MutateParam()
        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)

        compiled_out = opt_fn(x, y)
        # reset constant attr
        mod.a = torch.ones(8)
        self.assertEqual(compiled_out, fn(x, y))

    def test_redundant_compile_region(self):
        # [FUNCTION DOC] Applying the region decorator twice must behave like once: result
        # matches eager and the printed dynamo graph holds a single shared subgraph used by
        # two calls.
        @nested_compile_region
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Doubly decorated region returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = l_x_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)
""",
            )

    def test_kwargs_only(self):
        # [FUNCTION DOC] Region with a keyword-only argument, called by keyword; compiled result
        # with inductor must equal eager.
        @nested_compile_region
        def gn(x, *, y):
            # [FUNCTION DOC] HELPER: Region multiplying x with the keyword-only y.
            return x * y

        x = torch.randn(8, requires_grad=False)
        y = torch.randn(8, requires_grad=False)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region passing y by keyword.
            return gn(x, y=y)

        ref = fn(x, y)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_module_method(self):
        # [FUNCTION DOC] Region applied to a module helper method (bound method) that calls a
        # linear layer, used twice in forward. Checks results and that the printed dynamo graph
        # lifts weight and bias as inputs and shares one subgraph.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a linear layer with a region-decorated
            # helper method.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 8 by 8 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)

            @nested_compile_region
            def helper(self, x):
                # [FUNCTION DOC] HELPER: Region applying the linear layer.
                return self.linear(x)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Combines input with the product of two helper calls
                # plus input.
                return x + self.helper(x) * self.helper(x) + x

        mod = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_mod = torch.compile(mod, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_self_modules_linear_parameters_bias_: "f32[8]", L_self_modules_linear_parameters_weight_: "f32[8, 8]", L_x_: "f32[8, 8]"):
        l_self_modules_linear_parameters_bias_ = L_self_modules_linear_parameters_bias_
        l_self_modules_linear_parameters_weight_ = L_self_modules_linear_parameters_weight_
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  subgraph_1 = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]", l_self_modules_linear_parameters_weight_: "f32[8, 8]", l_self_modules_linear_parameters_bias_: "f32[8]"):
            linear: "f32[8, 8]" = torch._C._nn.linear(l_x_, l_self_modules_linear_parameters_weight_, l_self_modules_linear_parameters_bias_);  l_x_ = l_self_modules_linear_parameters_weight_ = l_self_modules_linear_parameters_bias_ = None
            return (linear,)
""",
            )

    def test_module(self):
        # [FUNCTION DOC] A module wrapped as a region and stored as a submodule, called twice.
        # Checks results match eager and the printed dynamo graph shares one subgraph.
        class SubMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Stateless module computing sine.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Only runs base class initialization.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return torch.sin(x)

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a region-wrapped submodule and calling
            # it twice.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the region-wrapped sine submodule.
                super().__init__()
                self.submod = nested_compile_region(SubMod())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x plus the product of two submodule calls
                # plus x.
                return x + self.submod(x) * self.submod(x) + x

        mod = Mod()
        backend = AotEagerAndRecordGraphs()
        opt_mod = torch.compile(mod, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)

        ref = mod(x)
        res = opt_mod(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        mul: "f32[8, 8]" = getitem * getitem_1;  getitem = getitem_1 = None
        add: "f32[8, 8]" = l_x_ + mul;  mul = None
        add_1: "f32[8, 8]" = add + l_x_;  add = l_x_ = None
        return (add_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)
""",
            )

    @requires_cuda_and_triton
    def test_return_none(self):
        # [FUNCTION DOC] CUDA and Triton only. A region applying dropout with train set to false
        # to a product with a grad-requiring weight is run via an assert-optimize inductor
        # wrapper. Backward is run and the weight gradient cloned; passes if nothing raises.
        from torch.nn import functional as F

        weight = torch.ones(
            1000, device="cuda:0", dtype=torch.float32, requires_grad=True
        )
        ones = torch.ones(1000, device="cuda:0", dtype=torch.float32)

        @nested_compile_region
        def fn(x, train):
            # [FUNCTION DOC] HELPER: Region: product with the outer weight, then dropout at 0.33
            # with the given train flag.
            return F.dropout(x * weight, 0.33, train)

        @torch._dynamo.optimize_assert("inductor")
        def run(x, train=True):
            # [FUNCTION DOC] HELPER: Compiled entry calling the region; train defaults to true.
            return fn(x, train)

        r1 = run(ones, train=False)
        r1.sum().backward()
        weight.grad.clone()

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_return_none_from_fwd(self):
        # [FUNCTION DOC] Region returning a None among its outputs. Checks inductor results and
        # gradients match eager, then with aot_eager that one subgraph exists in the dynamo,
        # forward and backward graphs and the printed graphs match expected text.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning x doubled, None, and x tripled.
            return x * 2, None, x * 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the first and third region outputs.
            ys = gn(x)
            return ys[0] + ys[2]

        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        res = opt_fn(x_clone)
        res.sum().backward()

        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 1)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 1)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph[1];  invoke_subgraph = None

        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            mul: "f32[8, 8]" = l_x_ * 2
            mul_1: "f32[8, 8]" = l_x_ * 3;  l_x_ = None
            return (mul, mul_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1);  partitioned_fw_subgraph_0_0 = primals_1 = None

        getitem: "f32[8, 8]" = invoke_subgraph_2[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph_2[1];  invoke_subgraph_2 = None

        add: "f32[8, 8]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        return (add,)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]"):
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(primals_0, 2)
            mul_1: "f32[8, 8]" = torch.ops.aten.mul.Tensor(primals_0, 3);  primals_0 = None
            return (mul, mul_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, tangents_1: "f32[8, 8]"):
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', tangents_1, tangents_1);  partitioned_bw_subgraph_0_0 = tangents_1 = None

        getitem_2: "f32[8, 8]" = invoke_subgraph_3[0];  invoke_subgraph_3 = None
        return (getitem_2,)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, tangents_0: "f32[8, 8]", tangents_1: "f32[8, 8]"):
            mul_2: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, 3)
            mul_3: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None
            add: "f32[8, 8]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
            return (add,)
""",
            )

    def test_dynamic(self):
        # [FUNCTION DOC] Region used twice with the batch dimension marked dynamic; inductor
        # result must equal eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_grad_accumulation(self):
        # [FUNCTION DOC] Region containing two linear layers whose difference is taken, plus a
        # third layer outside. Checks the backward graph has exactly one add node at top level
        # (the gradient accumulation for the outside layer) and exactly one inside the subgraph.
        mod1 = torch.nn.Linear(8, 8)
        mod2 = torch.nn.Linear(8, 8)
        mod3 = torch.nn.Linear(8, 8)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning the difference of two linear
            # layer outputs.
            return mod1(x) - mod2(x)

        def fn(c):
            # [FUNCTION DOC] HELPER: Subtracts a third linear layer from the region output
            # and doubles.
            d = gn(c) - mod3(c)
            return d * 2

        c = torch.randn((8, 8), requires_grad=True)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(c)
        res.sum().backward()

        # The gradient addition node for mod3 is not in the subgraph.
        bw_add_nodes = backend.bw_graphs[0].graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )
        self.assertEqual(len(bw_add_nodes), 1)
        subgraph_node = backend.bw_graphs[0].graph.find_nodes(op="get_attr")[0]
        subgraph_name = subgraph_node.target
        # The gradient addition node between mod1 and mode2 will be in the subgraph
        bw_add_nodes = getattr(backend.bw_graphs[0], subgraph_name).graph.find_nodes(
            op="call_function", target=torch.ops.aten.add.Tensor
        )
        self.assertEqual(len(bw_add_nodes), 1)

    def test_complex(self):
        # [FUNCTION DOC] Region on a complex64 tensor (case seen in Wan2.1); inductor result
        # must equal eager.
        # Observed in Wan2.1
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        x = torch.randn(2, 2, dtype=torch.complex64)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_pending_unbacked(self):
        # [FUNCTION DOC] With scalar output capture enabled and a dynamic dim, a region calls
        # item on an element, producing an unbacked symbol. Compiled with the eager backend
        # (inductor fails to compile this); result must equal eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region multiplying x by the scalar value of its
            # first element.
            u = x[0].item()
            return x * u

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(8)
        torch._dynamo.mark_dynamic(x, 0)
        ref = fn(x)
        opt_fn = torch.compile(
            fn, backend="eager", fullgraph=True
        )  # Inductor fails with cpp compilation error
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked1(self):
        # [FUNCTION DOC] Region slices a tensor using a size obtained from item (unbacked
        # symbol), compiled with the eager backend; result must equal eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region: reads the scalar from x and returns a clone of that
            # many leading elements of y.
            b = x.item()
            return y[:b].clone()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        x = torch.tensor(4)
        y = torch.randn(8)
        ref = fn(x, y)
        opt_fn = torch.compile(
            fn, backend="eager", fullgraph=True
        )  # Inductor fails with assertion error when lowering aten.sym_constrain_range_for_size.default
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked2(self):
        # [FUNCTION DOC] Like the previous test, but the region first asserts the unbacked size
        # is within range using runtime checks; eager-backend result must equal eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region: reads the scalar, checks it is nonnegative and
            # below the length of y, and returns a clone of that many leading elements of y.
            b = x.item()
            torch._check(b >= 0)
            torch._check(b < y.shape[0])
            return y[:b].clone()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        x = torch.tensor(4)
        y = torch.randn(8)
        ref = fn(x, y)
        opt_fn = torch.compile(
            fn, backend="eager", fullgraph=True
        )  # Inductor fails with assertion error when lowering aten.sym_constrain_range_for_size.default
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_bwd_partitioning(self):
        # [FUNCTION DOC] Region with a matmul and sine inside, then a sine outside; compiled
        # with aot_eager. Checks values and gradients against eager and that the printed forward
        # and backward graphs match expected text, showing the partitioner saves matmul output
        # and transposed inputs and recomputes the cosine inside the backward subgraph.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region: matmul then sine.
            z = torch.matmul(x, y)
            return torch.sin(z)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sine of the region output.
            return torch.sin(gn(x, y))

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 8]", primals_2: "f32[8, 8]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0

        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_1, primals_2);  partitioned_fw_subgraph_0_0 = primals_1 = primals_2 = None
        getitem_6: "f32[8, 8]" = invoke_subgraph_2[3]
        getitem_5: "f32[8, 8]" = invoke_subgraph_2[2]
        getitem_4: "f32[8, 8]" = invoke_subgraph_2[1]

        getitem: "f32[8, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        sin: "f32[8, 8]" = torch.ops.aten.sin.default(getitem)

        cos: "f32[8, 8]" = torch.ops.aten.cos.default(getitem);  getitem = None
        return (sin, getitem_6, getitem_5, getitem_4, cos)

    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]", primals_1: "f32[8, 8]"):
            mm: "f32[8, 8]" = torch.ops.aten.mm.default(primals_0, primals_1)

            sin: "f32[8, 8]" = torch.ops.aten.sin.default(mm)

            t: "f32[8, 8]" = torch.ops.aten.t.default(primals_0);  primals_0 = None
            t_1: "f32[8, 8]" = torch.ops.aten.t.default(primals_1);  primals_1 = None
            return (sin, mm, t, t_1)
""",
            )

            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_6: "f32[8, 8]", getitem_5: "f32[8, 8]", getitem_4: "f32[8, 8]", cos: "f32[8, 8]", tangents_1: "f32[8, 8]"):
        mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, cos);  tangents_1 = cos = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0

        invoke_subgraph_3 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_4, getitem_5, getitem_6, mul);  partitioned_bw_subgraph_0_0 = getitem_4 = getitem_5 = getitem_6 = mul = None

        getitem_1: "f32[8, 8]" = invoke_subgraph_3[0]
        getitem_2: "f32[8, 8]" = invoke_subgraph_3[1];  invoke_subgraph_3 = None
        return (getitem_1, getitem_2)

    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, mm: "f32[8, 8]", t: "f32[8, 8]", t_1: "f32[8, 8]", tangents_0: "f32[8, 8]"):
            cos: "f32[8, 8]" = torch.ops.aten.cos.default(mm);  mm = None
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_0, cos);  tangents_0 = cos = None

            mm_1: "f32[8, 8]" = torch.ops.aten.mm.default(t, mul);  t = None
            mm_2: "f32[8, 8]" = torch.ops.aten.mm.default(mul, t_1);  mul = t_1 = None
            return (mm_2, mm_1)
""",
            )

    def test_const_tensor(self):
        # [FUNCTION DOC] Region that creates a constant tensor inside; compiled with aot_eager
        # and compared with eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region multiplying x by a newly created scalar float
            # tensor 64.
            return torch.tensor(64, dtype=torch.float32) * x

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        x = torch.randn(64, requires_grad=True)

        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_ac(self):
        # [FUNCTION DOC] Activation checkpointing inside regions. fn1 region is used twice and
        # fn2 region once. Checks values and gradients, that dynamo, forward and backward
        # graphs have two distinct subgraphs (the repeated one is deduplicated), and inductor
        # matches too.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns cosine.
            return torch.cos(x)

        @nested_compile_region
        def fn1_checkpoint(x):
            # [FUNCTION DOC] HELPER: Region running the cosine function under non-
            # reentrant checkpoint.
            return torch.utils.checkpoint.checkpoint(fn1, x, use_reentrant=False)

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns sine.
            return torch.sin(x)

        @nested_compile_region
        def fn2_checkpoint(x):
            # [FUNCTION DOC] HELPER: Region running the sine function under non-
            # reentrant checkpoint.
            return torch.utils.checkpoint.checkpoint(fn2, x, use_reentrant=False)

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds fn1 region twice and fn2 region once.
            return (
                fn1_checkpoint(x)
                # repeat the same fn1_checkpoint to see that we dedupe
                + fn1_checkpoint(x)
                # Check that a new fn2_checkpoint goes through a different HOP
                + fn2_checkpoint(x)
            )

        x = torch.randn(8, requires_grad=True)
        ref = fn(x)

        x_clone = x.clone().detach().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone)

        # Run backward
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)
        self.count_unique_get_attr_nodes(backend.graphs[0], [], 2)
        self.count_unique_get_attr_nodes(backend.fw_graphs[0], [], 2)
        self.count_unique_get_attr_nodes(backend.bw_graphs[0], [], 2)

        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone)
        self.assertEqual(ref, res)

    @torch._inductor.config.patch(fallback_random=True)
    def test_ac_rng(self):
        # [FUNCTION DOC] Activation checkpointing of a dropout function inside a region
        # used twice, with random fallback in inductor. With a fixed seed, compiled values
        # and gradients must equal eager (aot_eager), and inductor output also matches the
        # eager reference.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Cosine of dropout (probability 0.5) of x.
            return torch.cos(torch.nn.functional.dropout(x, p=0.5))

        @nested_compile_region
        def fn1_checkpoint(x):
            # [FUNCTION DOC] HELPER: Region running the dropout function under non-
            # reentrant checkpoint.
            return torch.utils.checkpoint.checkpoint(fn1, x, use_reentrant=False)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return fn1_checkpoint(x) + fn1_checkpoint(x)

        x = torch.randn(8, requires_grad=True)
        torch.manual_seed(0)
        ref = fn(x)
        ref.sum().backward()

        x_clone = x.clone().detach().requires_grad_(True)
        backend = AotEagerAndRecordGraphs()

        torch.manual_seed(0)
        res = torch.compile(fn, backend=backend, fullgraph=True)(x_clone)
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        # Check that the Dynamo and AOT graphs have just one subgraph module
        self.assertEqual(len(backend.graphs), 1)
        self.assertEqual(len(backend.fw_graphs), 1)
        self.assertEqual(len(backend.bw_graphs), 1)

        torch.manual_seed(0)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x_clone)
        self.assertEqual(ref, res)
        res.sum().backward()

    @requires_gpu
    def test_ac_rng_cudagraphs(self):
        # [FUNCTION DOC] GPU test: checkpointed attention with dropout inside a region, compiled
        # with inductor in reduce-overhead (cuda graphs) mode in bfloat16; runs backward and
        # passes if nothing raises.
        def fn1(q, k, v):
            # [FUNCTION DOC] HELPER: Scaled dot product attention with dropout 0.5, causal.
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.5, is_causal=True
            )

        @nested_compile_region
        def fn1_checkpoint(q, k, v):
            # [FUNCTION DOC] HELPER: Region running the attention function under non-
            # reentrant checkpoint.
            return torch.utils.checkpoint.checkpoint(fn1, q, k, v, use_reentrant=False)

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: Sums the region on the inputs and on cos(q) with the same k
            # and v.
            return fn1_checkpoint(q, k, v) + fn1_checkpoint(q.cos(), k, v)

        q = torch.randn(
            1, 1, 32, 32, device=GPU_TYPE, dtype=torch.bfloat16, requires_grad=True
        )
        k = torch.randn(
            1, 1, 32, 32, device=GPU_TYPE, dtype=torch.bfloat16, requires_grad=True
        )
        v = torch.randn(
            1, 1, 32, 32, device=GPU_TYPE, dtype=torch.bfloat16, requires_grad=True
        )

        res = torch.compile(
            fn, backend="inductor", fullgraph=True, mode="reduce-overhead"
        )(q, k, v)
        res.sum().backward()

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_fake_tensor_checking(self):
        # [FUNCTION DOC] The same region called on two differently shaped inputs must produce
        # two separate subgraphs in the dynamo graph (checked against expected text).
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return torch.sin(x)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns region outputs for x and y as a tuple.
            # x and y are different shapes, so we should use different graph
            return gn(x), gn(y)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)

        ref = fn(x, y)
        res = opt_fn(x, y)

        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_y_: "f32[16, 16]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', l_y_);  subgraph_1 = l_y_ = None
        getitem_1: "f32[16, 16]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem, getitem_1)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)

    class subgraph_1(torch.nn.Module):
        def forward(self, l_y_: "f32[16, 16]"):
            sin: "f32[16, 16]" = torch.sin(l_y_);  l_y_ = None
            return (sin,)
""",
            )

    def test_return_size(self):
        # [FUNCTION DOC] Region returning a tensor and the input shape, used twice. For both
        # dynamic and static compilation, checks outputs match eager and gradients flow.
        def run(dynamic):
            # [FUNCTION DOC] HELPER: Resets the compiler, then compiles and compares the
            # scenario with the given dynamic flag.
            torch.compiler.reset()

            @nested_compile_region
            def gn(x):
                # [FUNCTION DOC] HELPER: Region returning x plus one and the shape of x.
                y = x + 1
                z = x.shape
                return y, z

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the region twice, returns the sum of first
                # outputs and the shape from the first call.
                z0 = gn(x)
                z1 = gn(x)
                return z0[0] + z1[0], z0[1]

            x = torch.randn(8, 8, requires_grad=True)
            x_clone = x.detach().clone().requires_grad_(True)
            ref = fn(x)
            opt_fn = torch.compile(
                fn, backend="inductor", fullgraph=True, dynamic=dynamic
            )
            res = opt_fn(x_clone)
            self.assertEqual(ref, res)

            ref[0].sum().backward()
            res[0].sum().backward()
            self.assertEqual(x.grad, x_clone.grad)

        run(dynamic=True)
        run(dynamic=False)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_different_symint(self):
        # [FUNCTION DOC] The same region applied to a tensor and to a narrowed half of it
        # (different symbolic sizes). First compiled with the default backend, then with the
        # aot_eager recording backend, checking results match eager.
        """
        Tests check that the same subgraph called with different symints use different graphs
        """

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the region, narrows dim 0 to half the size,
            # applies again.
            a = gn(x)
            # Get first half of the tensor
            b = torch.narrow(a, 0, 0, a.size()[0] // 2)
            return gn(b)

        opt_fn = torch.compile(fn, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)

        ref = fn(x)
        res = opt_fn(x)
        torch._dynamo.reset()

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)

        ref = fn(x)
        res = opt_fn(x)

        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, 8]", s77: "Sym(s77)"):
        l_x_ = L_x_

        floordiv: "Sym((s77//2))" = s77 // 2

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', s77, l_x_);  subgraph_0 = l_x_ = None
        getitem: "f32[s77, 8]" = invoke_subgraph[0];  invoke_subgraph = None

        narrow: "f32[(s77//2), 8]" = torch.narrow(getitem, 0, 0, floordiv);  getitem = floordiv = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', s77, narrow);  subgraph_1 = s77 = narrow = None
        getitem_1: "f32[(s77//2), 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", l_x_: "f32[s77, 8]"):
            sin: "f32[s77, 8]" = torch.sin(l_x_);  l_x_ = None
            return (sin,)

    class subgraph_1(torch.nn.Module):
        def forward(self, s77: "Sym(s77)", b: "f32[(s77//2), 8]"):
            sin: "f32[(s77//2), 8]" = torch.sin(b);  b = None
            return (sin,)
""",
            )

    def test_autograd_function(self):
        # [FUNCTION DOC] Region wrapping a custom autograd function, called twice. Checks
        # results and gradients against eager under aot_eager, and (outside crossref) that the
        # dynamo graph shares one subgraph that contains the autograd function apply op with
        # forward and backward body modules.
        class CustomOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function saving its input; forward is sine, and
            # backward returns the saved input times the cosine of the incoming gradient (not
            # the true derivative, but consistent between eager and compiled runs).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x for backward and returns its sine.
                ctx.save_for_backward(x)
                return torch.sin(x)

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Returns the saved input multiplied by the cosine
                # of grad_out.
                (x,) = ctx.saved_tensors
                return x * torch.cos(grad_out)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying the custom autograd function.
            return CustomOp.apply(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]"):
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        getitem: "f32[8, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_x_);  subgraph_1 = l_x_ = None
        getitem_1: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        add: "f32[8, 8]" = getitem + getitem_1;  getitem = getitem_1 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]"):
            fwd_body_0 = self.fwd_body_0
            bwd_body_0 = self.bwd_body_0
            autograd_function_apply = torch.ops.higher_order.autograd_function_apply(fwd_body_0, bwd_body_0, l_x_, non_differentiable_idx = [], saved_for_backward_idx = [0]);  fwd_body_0 = bwd_body_0 = l_x_ = None
            getitem: "f32[8, 8]" = autograd_function_apply[0];  autograd_function_apply = None
            return (getitem,)

        class fwd_body_0(torch.nn.Module):
            def forward(self, l_x_: "f32[8, 8]"):
                _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

                outs: "f32[8, 8]" = torch.sin(l_x_)

                _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
                return ((outs,), (l_x_,))

        class bwd_body_0(torch.nn.Module):
            def forward(self, grad_out: "f32[8, 8]", l_x_: "f32[8, 8]"):
                _set_grad_enabled = torch._C._set_grad_enabled(False);  _set_grad_enabled = None

                cos: "f32[8, 8]" = torch.cos(grad_out);  grad_out = None
                mul: "f32[8, 8]" = l_x_ * cos;  l_x_ = cos = None

                _set_grad_enabled_1 = torch._C._set_grad_enabled(True);  _set_grad_enabled_1 = None
                return (mul,)
""",
            )

    @requires_gpu
    def test_triton_kernel_native(self):
        # [FUNCTION DOC] GPU test: a region launches the shared add Triton kernel into a zeroed
        # buffer and returns its sine; the region is used twice. Checks the result against eager
        # and the dynamo graph text (the note in the code says the two calls do not truly share
        # one subgraph invocation because the constant args index changes).
        from torch.testing._internal.triton_utils import add_kernel

        def call_triton_add(
            x: torch.Tensor,
            y: torch.Tensor,
            output: torch.Tensor,
            grid_type: int,
            num=1,
            positional=False,
        ):
            # [FUNCTION DOC] HELPER: Launches the add kernel on x and y writing into output,
            # then returns output. The grid is a fixed tuple, a lambda over element count, or
            # the nested grid function depending on grid_type (0, 1, other); block size is
            # passed positionally or as a keyword depending on the positional flag.
            n_elements = output.numel()

            def grid_fn(meta):
                # [FUNCTION DOC] HELPER: Grid function computing a one-element tuple, the
                # ceiling division of the enclosing num argument by the block size taken from
                # the metadata mapping.
                return (triton.cdiv(num, meta["BLOCK_SIZE"]),)

            if grid_type == 0:
                grid = (x.numel(),)
            elif grid_type == 1:
                grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
            else:
                grid = grid_fn

            if positional:
                add_kernel[grid](x, y, output, n_elements, 16)
            else:
                add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)

            return output

        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region: zeroed buffer like x, filled by the Triton add
            # helper with a fixed grid, returns its sine.
            o = torch.zeros_like(x)
            call_triton_add(x, y, o, 0)
            return o.sin()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sine of both inputs, then the region applied twice
            # chaining outputs.
            x = x.sin()
            y = y.sin()
            z = gn(x, y)
            return gn(z, y)

        t1 = torch.rand(5, device=GPU_TYPE)
        t2 = torch.rand(5, device=GPU_TYPE)

        ref = fn(t1, t2)
        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        self.assertEqual(opt_fn(t1, t2), ref)

        # NOTE THAT THIS TEST DOES NOT REALLY WORK
        # We wanted one invoke_subgraph called twice, but because of
        # constant_args_idx changing in the graph, the graph equivalence fails

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[5]", L_y_: "f32[5]"):
        l_x_ = L_x_
        l_y_ = L_y_

        sin: "f32[5]" = l_x_.sin();  l_x_ = None

        sin_1: "f32[5]" = l_y_.sin();  l_y_ = None

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', sin, sin_1);  subgraph_0 = sin = None
        getitem: "f32[5]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', getitem, sin_1);  subgraph_1 = getitem = sin_1 = None
        getitem_1: "f32[5]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)

    class subgraph_0(torch.nn.Module):
        def forward(self, x: "f32[5]", y: "f32[5]"):
            o: "f32[5]" = torch.zeros_like(x)

            triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 0, grid = [(5, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'in_ptr0': x, 'in_ptr1': y, 'out_ptr': o}, launch_kwargs = ('BLOCK_SIZE',));  x = y = triton_kernel_wrapper_mutation = None

            sin: "f32[5]" = o.sin();  o = None
            return (sin,)
""",
            )

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_unbacked_symbol(self):
        # [FUNCTION DOC] Region whose output size depends on data (nonzero) with dynamic output
        # shape capture enabled; compiled with aot_eager (inductor fails to lower this) and
        # compared with eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning the sine of the nonzero indices of x.
            return torch.sin(torch.nonzero(x))

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x) + gn(x)

        x = torch.randn(64, 1, requires_grad=True)

        # Inductor fails with a lowering error
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_different_strides_in_backward(self):
        # [FUNCTION DOC] Chain of region calls with a dynamic batch dimension and a sine between
        # them. Checks compile (default backend) matches eager, then with aot_eager recording
        # that values and gradients match and the forward and backward graphs match expected
        # text, showing region calls reusing the same partitioned subgraph names.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning cosine.
            return torch.cos(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the region twice, sine, region twice more, and
            # returns the sum of the sums of the last two region outputs.
            a = gn(x)
            a2 = gn(a)
            b = torch.sin(a2)
            c = gn(b)
            c2 = gn(c)
            return c.sum() + c2.sum()

        opt_fn = torch.compile(fn, fullgraph=True)

        x = torch.randn(8, 16, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        x_clone = x.detach().clone().requires_grad_(True)
        torch._dynamo.mark_dynamic(x_clone, 0)

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)

        torch.compiler.reset()
        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 16, requires_grad=True)
        torch._dynamo.mark_dynamic(x, 0)
        x_clone = x.detach().clone().requires_grad_(True)
        torch._dynamo.mark_dynamic(x_clone, 0)
        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[s77, 16]", primals_2: "Sym(s77)"):
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_1
        invoke_subgraph_8 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_1', primals_2, primals_1);  partitioned_fw_subgraph_0_1 = primals_1 = None
        getitem_17: "Sym(s77)" = invoke_subgraph_8[2]
        getitem_16: "f32[s77, 16]" = invoke_subgraph_8[1]
        getitem: "f32[s77, 16]" = invoke_subgraph_8[0];  invoke_subgraph_8 = None
        partitioned_fw_subgraph_0_2 = self.partitioned_fw_subgraph_0_1
        invoke_subgraph_10 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_2, 'partitioned_fw_subgraph_0_1', primals_2, getitem);  partitioned_fw_subgraph_0_2 = getitem = None
        getitem_19: "Sym(s77)" = invoke_subgraph_10[2]
        getitem_18: "f32[s77, 16]" = invoke_subgraph_10[1]
        getitem_1: "f32[s77, 16]" = invoke_subgraph_10[0];  invoke_subgraph_10 = None
        sin: "f32[s77, 16]" = torch.ops.aten.sin.default(getitem_1)
        partitioned_fw_subgraph_0_3 = self.partitioned_fw_subgraph_0_1
        invoke_subgraph_12 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_3, 'partitioned_fw_subgraph_0_1', primals_2, sin);  partitioned_fw_subgraph_0_3 = sin = None
        getitem_21: "Sym(s77)" = invoke_subgraph_12[2]
        getitem_20: "f32[s77, 16]" = invoke_subgraph_12[1]
        getitem_2: "f32[s77, 16]" = invoke_subgraph_12[0];  invoke_subgraph_12 = None
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_14 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', primals_2, getitem_2);  partitioned_fw_subgraph_0_0 = None
        getitem_23: "Sym(s77)" = invoke_subgraph_14[2]
        getitem_22: "f32[s77, 16]" = invoke_subgraph_14[1]
        getitem_3: "f32[s77, 16]" = invoke_subgraph_14[0];  invoke_subgraph_14 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(getitem_2);  getitem_2 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(getitem_3);  getitem_3 = None
        add_15: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        cos: "f32[s77, 16]" = torch.ops.aten.cos.default(getitem_1);  getitem_1 = None
        return (add_15, getitem_16, getitem_18, getitem_20, getitem_22, cos, primals_2, getitem_17, getitem_19, getitem_21, getitem_23)
    class partitioned_fw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]"):
            cos: "f32[s77, 16]" = torch.ops.aten.cos.default(primals_1)
            return (cos, primals_1, primals_0)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]"):
            cos: "f32[s77, 16]" = torch.ops.aten.cos.default(primals_1)
            return (cos, primals_1, primals_0)""",
                ignore_empty_lines=True,
            )
            self.assertExpectedInline(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "Sym(s77)", getitem_17: "Sym(s77)", getitem_19: "Sym(s77)", getitem_21: "Sym(s77)", getitem_23: "Sym(s77)", getitem_16: "f32[s77, 16]", getitem_18: "f32[s77, 16]", getitem_20: "f32[s77, 16]", getitem_22: "f32[s77, 16]", cos: "f32[s77, 16]", tangents_1: "f32[]"):
        expand: "f32[s77, 16]" = torch.ops.aten.expand.default(tangents_1, [primals_2, 16]);  tangents_1 = primals_2 = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_15 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_23, getitem_22, expand);  partitioned_bw_subgraph_0_0 = getitem_23 = getitem_22 = None
        getitem_5: "f32[s77, 16]" = invoke_subgraph_15[1];  invoke_subgraph_15 = None
        add_16: "f32[s77, 16]" = torch.ops.aten.add.Tensor(expand, getitem_5);  expand = getitem_5 = None
        partitioned_bw_subgraph_0_3 = self.partitioned_bw_subgraph_0_1
        invoke_subgraph_13 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_3, 'partitioned_bw_subgraph_0_1', getitem_21, getitem_20, add_16);  partitioned_bw_subgraph_0_3 = getitem_21 = getitem_20 = add_16 = None
        getitem_8: "f32[s77, 16]" = invoke_subgraph_13[1];  invoke_subgraph_13 = None
        mul_10: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(getitem_8, cos);  getitem_8 = cos = None
        partitioned_bw_subgraph_0_2 = self.partitioned_bw_subgraph_0_1
        invoke_subgraph_11 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_2, 'partitioned_bw_subgraph_0_1', getitem_19, getitem_18, mul_10);  partitioned_bw_subgraph_0_2 = getitem_19 = getitem_18 = mul_10 = None
        getitem_11: "f32[s77, 16]" = invoke_subgraph_11[1];  invoke_subgraph_11 = None
        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_1
        invoke_subgraph_9 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_1', getitem_17, getitem_16, getitem_11);  partitioned_bw_subgraph_0_1 = getitem_17 = getitem_16 = getitem_11 = None
        getitem_14: "f32[s77, 16]" = invoke_subgraph_9[1];  invoke_subgraph_9 = None
        return (getitem_14, None)
    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]", tangents_0: "f32[s77, 16]"):
            sin: "f32[s77, 16]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None
            neg: "f32[s77, 16]" = torch.ops.aten.neg.default(sin);  sin = None
            mul_9: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(tangents_0, neg);  tangents_0 = neg = None
            return (None, mul_9)
    class partitioned_bw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "Sym(s77)", primals_1: "f32[s77, 16]", tangents_0: "f32[s77, 16]"):
            sin: "f32[s77, 16]" = torch.ops.aten.sin.default(primals_1);  primals_1 = None
            neg: "f32[s77, 16]" = torch.ops.aten.neg.default(sin);  sin = None
            mul_10: "f32[s77, 16]" = torch.ops.aten.mul.Tensor(tangents_0, neg);  tangents_0 = neg = None
            return (None, mul_10)""",
                ignore_empty_lines=True,
            )

    def test_div(self):
        # [FUNCTION DOC] Region computing an integer truncating division of constants and using
        # it as a tensor shape; compiled result must equal eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Builds a ones tensor of shape 64 by the quotient and
            # multiplies by the quotient and x.
            div = torch.div(1024, 256, rounding_mode="trunc")
            return div * torch.ones(64, div) * x

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(64, 1, requires_grad=True)

        opt_fn = torch.compile(fn, fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @requires_gpu
    def test_preserves_strides(self):
        # [FUNCTION DOC] GPU test that inductor preserves strides for region inputs. Installs a
        # post-grad pass that rewrites permutes to force a channels-last layout through a
        # temporary custom op, plus an add custom op whose meta returns a contiguous tensor.
        # Checks compiled output equals eager and that the pass fired.
        class _CustomPass(PatternMatcherPass):
            # [CLASS DOC] HELPER: Pattern matcher pass that applies its registered patterns to a
            # graph when called.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the parent initializer.
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Applies the pass patterns to the graph g;
                # returns nothing.
                self.apply(g)

        g = _CustomPass()
        called = False

        x = torch.randn(4, 4, 2, 2, device=GPU_TYPE)
        other = torch.randn(4, 4, 2, 2, device=GPU_TYPE)

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler registered for permute calls. Flattens the
            # match args, then replaces the match using an example function that applies permute
            # followed by the force_channels_last custom op, and sets the enclosing called flag.
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                # [FUNCTION DOC] HELPER: Unflattens the arguments, permutes, and passes the
                # result through the force_channels_last custom op.
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                # [FUNCTION DOC] HELPER: Implementation returning a channels-last clone of x.
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                # [FUNCTION DOC] HELPER: Implementation for the add custom op: clones y
                # (contiguous strides) and adds the transpose of x on the last two dims.
                out = y.clone()  # contiguous with strides (16, 4, 2, 1)
                out.add_(x.transpose(-1, -2))
                return out

            def meta(x, y):
                # [FUNCTION DOC] HELPER: Meta implementation returning an empty contiguous
                # tensor like y.
                return torch.empty_like(y, memory_format=torch.contiguous_format)

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            @nested_compile_region
            def gn(y, z):
                # [FUNCTION DOC] HELPER: Region calling the add custom op.
                return torch.ops.mylib.add_op.default(y, z)

            def f(x, other):
                # [FUNCTION DOC] HELPER: Builds a transposed-contiguous-transposed y and a
                # transposed sine z, and calls the region.
                y = x.transpose(2, 3).contiguous().transpose(2, 3)
                z = y.sin().transpose(2, 3)
                return gn(y, z)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                f_compile = torch.compile(f, fullgraph=True)
                self.assertEqual(f(x, other), f_compile(x, other))
                self.assertTrue(called)

    @requires_gpu
    def test_preserves_output_strides(self):
        # [FUNCTION DOC] GPU test that the output strides of the region are preserved when a
        # graph pass changes layouts. The add custom op implementation raises AssertionError if
        # its input strides are not as expected. Compiled result must equal eager and the pass
        # must fire.
        # Have a graph pass that changes strides for the output op of the
        # invoke_subgraph, and check if the output strides are preserved
        x = torch.randn(4, 4, 2, 2, device=GPU_TYPE)
        other = torch.randn(4, 4, 2, 2, device=GPU_TYPE)

        class _CustomPass(PatternMatcherPass):
            # [CLASS DOC] HELPER: Pattern matcher pass that applies its registered patterns to a
            # graph when called.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only runs the parent initializer.
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Applies the pass patterns to the graph g;
                # returns nothing.
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler registered for permute calls. Flattens the
            # match args, then replaces the match using an example function that applies permute
            # followed by the force_channels_last custom op, and sets the enclosing called flag.
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                # [FUNCTION DOC] HELPER: Unflattens the arguments, permutes, and passes the
                # result through the force_channels_last custom op.
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                # [FUNCTION DOC] HELPER: Implementation returning a channels-last clone of x.
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                # [FUNCTION DOC] HELPER: Implementation for the add custom op. Raises
                # AssertionError if the stride of x is not (16, 4, 1, 2) or of y not (16, 4, 2,
                # 1); otherwise clones y and adds the transpose of x on the last two dims.
                # Check that the input strides are preserved. This helps in
                # testing that the HOP preserves the output strides.
                if x.stride() != (16, 4, 1, 2):
                    raise AssertionError(
                        f"Expected x.stride() == (16, 4, 1, 2), got {x.stride()}"
                    )
                if y.stride() != (16, 4, 2, 1):
                    raise AssertionError(
                        f"Expected y.stride() == (16, 4, 2, 1), got {y.stride()}"
                    )
                out = y.clone()  # contiguous with strides (16, 4, 2, 1)
                out.add_(x.transpose(-1, -2))
                return out

            def meta(x, y):
                # [FUNCTION DOC] HELPER: Meta implementation returning an empty contiguous
                # tensor like y.
                return torch.empty_like(y, memory_format=torch.contiguous_format)

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            @nested_compile_region
            def gn(x, other):
                # [FUNCTION DOC] HELPER: Region building y (transposed twice around a
                # contiguous copy) and z (transposed sine of y), returning both. The other
                # argument is unused.
                y = x.transpose(2, 3).contiguous().transpose(2, 3)
                z = y.sin().transpose(2, 3)
                return y, z

            def f(x, other):
                # [FUNCTION DOC] HELPER: Runs the region then applies the add custom op to its
                # two outputs.
                y, z = gn(x, other)
                return torch.ops.mylib.add_op.default(y, z)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                f_compile = torch.compile(f, fullgraph=True)
                self.assertEqual(f(x, other), f_compile(x, other))
                self.assertTrue(called)

    def test_udf_output(self):
        # [FUNCTION DOC] Region returning an instance of a user-defined class; outputs are
        # flattened. Checks values and gradients under aot_eager and the dynamo graph text with
        # one shared subgraph returning two tensors.
        class Foo:
            # [CLASS DOC] HELPER: Plain holder class with two attributes a and b.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores the two arguments.
                self.a = a
                self.b = b

        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning the holder of sine of x and cosine of y.
            a = torch.sin(x)
            b = torch.cos(y)
            return Foo(a, b)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region twice (second on the first's a field) and
            # adds the first b field and the second a field.
            foo1 = gn(x, y)
            foo2 = gn(foo1.a, y)
            return foo1.b + foo2.a  # + foo2.b

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        ref = fn(x, y)
        res = opt_fn(x_clone, y_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8, 8]", L_y_: "f32[8, 8]"):
        l_x_ = L_x_
        l_y_ = L_y_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, l_y_);  subgraph_0 = l_x_ = None
        getitem: "f32[8, 8]" = invoke_subgraph[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph[1];  invoke_subgraph = None

        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', getitem, l_y_);  subgraph_1 = getitem = l_y_ = None
        getitem_2: "f32[8, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        add: "f32[8, 8]" = getitem_1 + getitem_2;  getitem_1 = getitem_2 = None
        return (add,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8, 8]", l_y_: "f32[8, 8]"):
            a: "f32[8, 8]" = torch.sin(l_x_);  l_x_ = None

            b: "f32[8, 8]" = torch.cos(l_y_);  l_y_ = None
            return (a, b)
""",
            )

    def test_grad_accuracy_check(self):
        # [FUNCTION DOC] Checks gradients when the region is called on a result of itself and
        # the backward graph must be traced twice because tangent aliasing differs between
        # calls. Compares values and gradients with eager and compares forward and backward
        # graph text (blank lines normalized) to expected output.
        class Foo:
            # [CLASS DOC] HELPER: Unused holder class with two attributes a and b.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Stores the two arguments.
                self.a = a
                self.b = b

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning the sine and cosine of x as a tuple.
            a = torch.sin(x)
            b = torch.cos(x)
            return (a, b)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region on x, then on the first output, and sums
            # the second of the first and both of the second call.
            foo1 = gn(x)
            foo2 = gn(foo1[0])
            return foo1[1] + foo2[0] + foo2[1]

        backend = AotEagerAndRecordGraphs()

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)

        x = torch.randn(8, 8, requires_grad=True)
        x_clone = x.detach().clone().requires_grad_(True)
        x.grad = None
        x_clone.grad = None

        ref = fn(x)
        res = opt_fn(x_clone)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)

        # We should see a single forward subgraph
        self.assertExpectedInline(
            empty_line_normalizer(
                normalize_gm(backend.fw_graphs[0].print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[8, 8]"):
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_1
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_1', primals_1);  partitioned_fw_subgraph_0_1 = primals_1 = None
        getitem_10: "f32[8, 8]" = invoke_subgraph_4[2]
        getitem: "f32[8, 8]" = invoke_subgraph_4[0]
        getitem_1: "f32[8, 8]" = invoke_subgraph_4[1];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', getitem);  partitioned_fw_subgraph_0_0 = getitem = None
        getitem_11: "f32[8, 8]" = invoke_subgraph_6[2]
        getitem_2: "f32[8, 8]" = invoke_subgraph_6[0]
        getitem_3: "f32[8, 8]" = invoke_subgraph_6[1];  invoke_subgraph_6 = None
        add: "f32[8, 8]" = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
        add_1: "f32[8, 8]" = torch.ops.aten.add.Tensor(add, getitem_3);  add = getitem_3 = None
        return (add_1, getitem_10, getitem_11)
    class partitioned_fw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.ops.aten.sin.default(primals_0)
            cos: "f32[8, 8]" = torch.ops.aten.cos.default(primals_0)
            return (sin, cos, primals_0)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]"):
            sin: "f32[8, 8]" = torch.ops.aten.sin.default(primals_0)
            cos: "f32[8, 8]" = torch.ops.aten.cos.default(primals_0)
            return (sin, cos, primals_0)
""",
        )

        # The backward graph should be traced twice because tangents aliasing changed
        # Note that in the first partitioned_bw_subgraph_0_0, mul_1 is using tangents_1, which is
        # only correct when tangents_1 and tangents_0 are aliases.
        self.assertExpectedInline(
            empty_line_normalizer(
                normalize_gm(backend.bw_graphs[0].print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_10: "f32[8, 8]", getitem_11: "f32[8, 8]", tangents_1: "f32[8, 8]"):
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_11, tangents_1, tangents_1);  partitioned_bw_subgraph_0_0 = getitem_11 = None
        getitem_4: "f32[8, 8]" = invoke_subgraph_7[0];  invoke_subgraph_7 = None
        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_1
        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_1', getitem_10, getitem_4, tangents_1);  partitioned_bw_subgraph_0_1 = getitem_10 = getitem_4 = tangents_1 = None
        getitem_7: "f32[8, 8]" = invoke_subgraph_5[0];  invoke_subgraph_5 = None
        return (getitem_7,)
    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]", tangents_0: "f32[8, 8]", tangents_1: "f32[8, 8]"):
            sin_1: "f32[8, 8]" = torch.ops.aten.sin.default(primals_0)
            neg: "f32[8, 8]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, neg);  neg = None
            cos_1: "f32[8, 8]" = torch.ops.aten.cos.default(primals_0);  primals_0 = None
            mul_1: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, cos_1);  tangents_1 = cos_1 = None
            add: "f32[8, 8]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
            return (add,)
    class partitioned_bw_subgraph_0_1(torch.nn.Module):
        def forward(self, primals_0: "f32[8, 8]", tangents_0: "f32[8, 8]", tangents_1: "f32[8, 8]"):
            sin_1: "f32[8, 8]" = torch.ops.aten.sin.default(primals_0)
            neg: "f32[8, 8]" = torch.ops.aten.neg.default(sin_1);  sin_1 = None
            mul: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_1, neg);  tangents_1 = neg = None
            cos_1: "f32[8, 8]" = torch.ops.aten.cos.default(primals_0);  primals_0 = None
            mul_1: "f32[8, 8]" = torch.ops.aten.mul.Tensor(tangents_0, cos_1);  tangents_0 = cos_1 = None
            add: "f32[8, 8]" = torch.ops.aten.add.Tensor(mul, mul_1);  mul = mul_1 = None
            return (add,)
""",
        )

    def test_do_not_remove_used_output(self):
        # [FUNCTION DOC] Nested regions: the outer region calls an inner region and returns all
        # of its outputs, and the caller uses only some of the inner outputs directly. Exports
        # via dynamo graph capture and checks the exported module matches eager (no output
        # pruning mistakes).
        # Test that the ggn's outputs are not pruned.

        @nested_compile_region
        def ggn(x):
            # [FUNCTION DOC] HELPER: Inner region returning max and argmax along dim 0.
            return torch.max(x, 0)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Outer region calling the inner one and returning its
            # two results.
            a, b = ggn(x)
            return a, b

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the inner region (keeping index) and the outer
            # region; returns the index and both outer outputs.
            _, b = ggn(x)
            c, d = gn(x)
            return b, c, d

        x = torch.randn(64, 1)

        ref = fn(x)
        gm = dynamo_graph_capture_for_export(fn)(x)

        res = gm(x)
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_remove_unused_output(self):
        # [FUNCTION DOC] Same nested-region structure, but only the index outputs are used with
        # single-use inlining off. After capture for export, the outer subgraph should return
        # only the used output; matches eager and expected graph text outside crossref.
        # Test that the ggn's graph's output is pruned.

        @nested_compile_region
        def ggn(x):
            # [FUNCTION DOC] HELPER: Inner region returning max and argmax along dim 0.
            return torch.max(x, 0)

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Outer region calling the inner one and returning its
            # two results.
            a, b = ggn(x)
            return a, b

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the inner region and outer region and returns the
            # two indices.
            _, b = ggn(x)
            _, d = gn(x)
            return b, d

        x = torch.randn(64, 1)

        ref = fn(x)
        gm = dynamo_graph_capture_for_export(fn)(x)

        res = gm(x)
        self.assertEqual(ref, res)

        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(gm.print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        _fn_args = (x, )
        L_x_, = self._dynamo_bytecode_flatten(*_fn_args)
        l_x_ = L_x_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = None
        b: "i64[1]" = invoke_subgraph[1];  invoke_subgraph = None

        subgraph_1 = self.subgraph_1
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_1', l_x_);  subgraph_1 = l_x_ = None
        getitem_4: "i64[1]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return self._dynamo_bytecode_unflatten((b, getitem_4,), _fn_args)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[64, 1]"):
            max_1 = torch.max(l_x_, 0);  l_x_ = None
            getitem: "f32[1]" = max_1[0]
            getitem_1: "i64[1]" = max_1[1];  max_1 = None
            return (getitem, getitem_1)

    class subgraph_1(torch.nn.Module):
        def forward(self, l_x_: "f32[64, 1]"):
            subgraph_0 = self.subgraph_0
            invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_);  subgraph_0 = l_x_ = None
            a: "f32[1]" = invoke_subgraph[0];  a = None
            b: "i64[1]" = invoke_subgraph[1];  invoke_subgraph = None
            return (b,)

        class subgraph_0(torch.nn.Module):
            def forward(self, l_x_: "f32[64, 1]"):
                max_1 = torch.max(l_x_, 0);  l_x_ = None
                getitem: "f32[1]" = max_1[0]
                getitem_1: "i64[1]" = max_1[1];  max_1 = None
                return (getitem, getitem_1)
""",
            )

    @torch._dynamo.config.patch(trace_autograd_ops=True)
    def test_trace_autograd_ops_partitions_bw_invoke_subgraph(self):
        # [FUNCTION DOC] With autograd op tracing enabled, a function that computes a gradient
        # with the autograd API through a region called twice must give the same result as eager
        # under aot_eager, and the forward graph text must show backward regions partitioned
        # (expected text compared ignoring comments and blank lines).
        """
        Checks that the backward invoke_subgraphs have been partitioned.
        """

        @nested_compile_region
        def gn(x, w):
            # [FUNCTION DOC] HELPER: Region computing a matrix product.
            return torch.matmul(x, w)

        def fn(x, w):
            # [FUNCTION DOC] HELPER: Calls the region twice, sums the outputs, computes
            # the gradient of the sum with respect to w, and returns the detached loss and
            # the gradient.
            # Call gn twice so it doesn't get inlined as single-use
            out1 = gn(x, w)
            out2 = gn(x, w)
            loss = (out1 + out2).sum()
            (grad_w,) = torch.autograd.grad(loss, w)
            return loss.detach(), grad_w

        x = torch.randn(4, 4)
        w = torch.randn(4, 4, requires_grad=True)

        ref = fn(x, w)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x, w)
        self.assertEqual(ref, res)

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4, 4]"):
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', arg1_1, arg0_1);  partitioned_fw_subgraph_0_0 = None
        getitem_8: "f32[4, 4]" = invoke_subgraph_4[1]
        getitem: "f32[4, 4]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        partitioned_fw_subgraph_0_1 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_6 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_1, 'partitioned_fw_subgraph_0_0', arg1_1, arg0_1);  partitioned_fw_subgraph_0_1 = arg1_1 = arg0_1 = None
        getitem_9: "f32[4, 4]" = invoke_subgraph_6[1]
        getitem_1: "f32[4, 4]" = invoke_subgraph_6[0];  invoke_subgraph_6 = None
        add: "f32[4, 4]" = torch.ops.aten.add.Tensor(getitem, getitem_1);  getitem = getitem_1 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(add);  add = None
        detach: "f32[]" = torch.ops.aten.detach.default(sum_1)
        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
        expand: "f32[4, 4]" = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
        partitioned_bw_subgraph_0_1 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_7 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_1, 'partitioned_bw_subgraph_0_0', getitem_9, expand);  partitioned_bw_subgraph_0_1 = getitem_9 = None
        getitem_3: "f32[4, 4]" = invoke_subgraph_7[1];  invoke_subgraph_7 = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        invoke_subgraph_5 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_8, expand);  partitioned_bw_subgraph_0_0 = getitem_8 = expand = None
        getitem_6: "f32[4, 4]" = invoke_subgraph_5[1];  invoke_subgraph_5 = None
        add_1: "f32[4, 4]" = torch.ops.aten.add.Tensor(getitem_3, getitem_6);  getitem_3 = getitem_6 = None
        return (detach, add_1)
    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4, 4]"):
            mm: "f32[4, 4]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (mm,)
    class repeated_subgraph1(torch.nn.Module):
        def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4, 4]", arg2_1: "f32[4, 4]"):
            mm: "f32[4, 4]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg1_1 = None
            t: "f32[4, 4]" = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
            mm_1: "f32[4, 4]" = torch.ops.aten.mm.default(t, arg2_1);  t = arg2_1 = None
            return (None, mm_1, mm)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[4, 4]", primals_1: "f32[4, 4]"):
            mm: "f32[4, 4]" = torch.ops.aten.mm.default(primals_0, primals_1);  primals_1 = None
            t: "f32[4, 4]" = torch.ops.aten.t.default(primals_0);  primals_0 = None
            return (mm, t)
    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, t: "f32[4, 4]", tangents_0: "f32[4, 4]"):
            mm_1: "f32[4, 4]" = torch.ops.aten.mm.default(t, tangents_0);  t = tangents_0 = None
            return (None, mm_1)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    @torch._dynamo.config.patch(
        trace_autograd_ops=True, inline_single_use_invoke_subgraph=False
    )
    def test_trace_autograd_ops_with_activation_checkpointing(self):
        # [FUNCTION DOC] With autograd op tracing and no single-use inlining, activation
        # checkpoint bodies call a region inside; the function computes a gradient via the
        # autograd API. Compiled with aot_eager; results must match eager.
        from torch.utils.checkpoint import checkpoint

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region computing x matrix-multiplied with itself.
            return torch.matmul(x, x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Checkpoints the body twice, computes the gradient of the
            # summed output with respect to x, and returns the detached loss and the gradient.
            def body(x):
                # [FUNCTION DOC] HELPER: Checkpoint body: sine, region call, relu.
                y = torch.sin(x)
                z = gn(y)
                return torch.relu(z)

            out1 = checkpoint(body, x, use_reentrant=False)
            out2 = checkpoint(body, x, use_reentrant=False)
            loss = (out1 + out2).sum()
            (grad,) = torch.autograd.grad(loss, x)
            return loss.detach(), grad

        x = torch.randn(4, 4, requires_grad=True)

        ref = fn(x)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)


_reuse_test_global = None


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraphReuse(TestCase):
    # [CLASS DOC] Tests that dynamo reuses an already traced subgraph for repeated calls of the
    # same region instead of tracing again, by counting calls to the speculation function.
    # Skipped under dynamo.
    @contextlib.contextmanager
    def _count_speculate_calls(self):
        # [FUNCTION DOC] HELPER: Context manager patching the dynamo subgraph speculation
        # function with a counting wrapper that forwards to the original. Yields a callable
        # returning the current call count.
        count = 0
        orig = torch._dynamo.variables.higher_order_ops.speculate_subgraph_with_auto_output_flattening

        def _counting(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Counting wrapper: increments the enclosing count and
            # forwards all arguments to the original function, returning its result.
            nonlocal count
            count += 1
            return orig(*args, **kwargs)

        with mock.patch.object(
            torch._dynamo.variables.higher_order_ops,
            "speculate_subgraph_with_auto_output_flattening",
            _counting,
        ):
            yield lambda: count

    def test_subgraph_reuse_skips_tracing(self):
        # [FUNCTION DOC] Calling one region three times with the same inputs compiles with
        # exactly one speculation (trace) call.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying its inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums three region calls.
            a = gn(x, y)
            b = gn(x, y)
            c = gn(x, y)
            return a + b + c

        x = torch.randn(8)
        y = torch.randn(8)

        with self._count_speculate_calls() as count:
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y)

        self.assertEqual(count(), 1)

    def test_subgraph_reuse_different_shapes(self):
        # [FUNCTION DOC] Calling one region on inputs with different shapes needs two
        # speculation calls; result must equal eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return x.sin()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums of the region applied to x and to y.
            a = gn(x)
            b = gn(y)
            return a.sum() + b.sum()

        x = torch.randn(4)
        y = torch.randn(8)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y)

        # Different shapes → two separate traces
        self.assertEqual(count(), 2)
        self.assertEqual(res, fn(x, y))

    def test_subgraph_reuse_module(self):
        # [FUNCTION DOC] Module forward region called twice with the same module: one
        # speculation call; values and gradients must match eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a constant and a region-decorated forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the integer constant 5.
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product plus the constant.
                return torch.mul(x, y).sin() + self.c

        mod = Mod()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two module calls.
            return mod(x, y) + mod(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(
                x_clone, y_clone
            )

        # Second call reuses the first trace
        self.assertEqual(count(), 1)
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_subgraph_reuse_module_different_instances(self):
        # [FUNCTION DOC] Two module instances of the same class with equal constants share a
        # region trace: speculation runs once (source replacement), and values and gradients
        # match eager under aot_eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a constructor-supplied constant used by its
            # region-decorated forward.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c as an attribute.
                super().__init__()
                self.c = c

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sine of the product plus the constant.
                return torch.mul(x, y).sin() + self.c

        mod1 = Mod(5)
        mod2 = Mod(5)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums the outputs of the two module instances.
            return mod1(x, y) + mod2(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(
                x_clone, y_clone
            )

        # mod1 and mod2 have the same structure and c value; source replacement
        # means only one trace is needed.
        self.assertEqual(count(), 1)
        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_subgraph_reuse_tuple_output(self):
        # [FUNCTION DOC] A region returning a tuple of two tensors, called twice with the same
        # inputs: one speculation call; values and gradients (backward through the sums of all
        # outputs) match eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning sine of x and cosine of y.
            return torch.sin(x), torch.cos(y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region twice and adds the matching outputs.
            a1, a2 = gn(x, y)
            b1, b2 = gn(x, y)
            return a1 + b1, a2 + b2

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x_clone = x.detach().clone().requires_grad_(True)
        y_clone = y.detach().clone().requires_grad_(True)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(
                x_clone, y_clone
            )

        # Second call reuses the first trace
        self.assertEqual(count(), 1)
        sum(r.sum() for r in ref).backward()
        sum(r.sum() for r in res).backward()

        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])
        self.assertEqual(x.grad, x_clone.grad)
        self.assertEqual(y.grad, y_clone.grad)

    def test_subgraph_reuse_mutated_attribute(self):
        # [FUNCTION DOC] The module attribute read by the region is changed between two calls (5
        # then 10). Reuse must be skipped: result equals fifteen times x as in eager, and
        # speculation runs twice.
        """Reuse must be skipped when a captured attribute is mutated between calls."""

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute used by its region-
            # decorated forward.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the attribute to 5.
                super().__init__()
                self.c = 5

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by the attribute.
                return x * self.c

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the module, sets the attribute to 10, calls again,
            # and sums.
            a = mod(x)
            mod.c = 10
            b = mod(x)
            return a + b

        x = torch.randn(8)
        # Eager: first call uses c=5, then c is set to 10, second call uses c=10.
        # Result = x*5 + x*10 = x*15.
        mod.c = 5
        ref = fn(x)
        self.assertEqual(ref, x * 15)

        # Compiled should produce the same result. If reuse incorrectly
        # fires, both calls would use c=5, giving x*10 instead of x*15.
        mod.c = 5
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, res)
        # c=5 and c=10 are distinct constants → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_unrelated_attr_mutation(self):
        # [FUNCTION DOC] An attribute not read by the region is mutated between calls; reuse
        # must still happen (one speculation call) and the result equals ten times x.
        """Reuse should still fire when a different attribute is mutated."""

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with attributes c and d; its region forward reads
            # only c.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets c to 5 and d to 100.
                super().__init__()
                self.c = 5
                self.d = 100

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by attribute c.
                # Only reads self.c, never self.d
                return x * self.c

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the module, sets the unrelated attribute d, calls
            # again, sums.
            a = mod(x)
            mod.d = 999  # unrelated attribute
            b = mod(x)
            return a + b

        x = torch.randn(8)
        ref = fn(x)
        # Both calls use c=5, so result = x*5 + x*5 = x*10.
        self.assertEqual(ref, x * 10)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        # Mutating mod.d should not prevent reuse of the subgraph that only reads mod.c.
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_same_class_attr_mutated(self):
        # [FUNCTION DOC] Two instances of one class start with equal constants; the second
        # instance's constant is mutated between the calls. Reuse via source replacement
        # must be rejected: eager gives fifteen times x and compiled matches, with two
        # speculation calls.
        """Reuse must be skipped when a captured attr changes between calls.

        submod1 and submod2 are instances of the same class with the same
        initial value for .c.  The first call traces submod1; the second call
        to submod2 could reuse the cache entry via source replacement.  But
        submod2.c is mutated between the two calls, so reuse must be skipped.
        """

        class Block(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a constructor-supplied constant used by its
            # region forward.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by the constant.
                return x * self.c

        submod1 = Block(5)
        submod2 = Block(5)  # same initial .c as submod1

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the first block, mutates the second block's constant
            # to 10, calls the second block, and sums.
            a = submod1(x)  # traces with c=5
            submod2.c = 10  # mutate submod2.c
            b = submod2(x)  # must NOT reuse the c=5 subgraph
            return a + b

        x = torch.randn(8)
        ref = fn(x)
        # a = x*5, b = x*10 → x*15
        self.assertEqual(ref, x * 15)

        submod2.c = 5  # reset for compiled run
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, res)
        # submod1 traces with c=5; submod2 has c mutated to 10 → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_pre_existing_attr_guard(self):
        # [FUNCTION DOC] A condition on a module attribute installs a guard before the region
        # trace snapshots guards. Reuse must still be rejected when a different module with a
        # different constant is used: result equals 5x plus 10x and speculation runs twice.
        """Guards installed before the subgraph trace must still block incorrect reuse.

        If ``block.c`` is read in a conditional before the nested compile region,
        its guard is installed before ``guards_before`` is snapshotted and won't
        appear in the delta.  Reuse must still be rejected when a different module
        with a different ``c`` is passed.
        """

        class Block(torch.nn.Module):
            # [CLASS DOC] HELPER: Plain module holding a constructor-supplied constant c.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

        @nested_compile_region
        def apply_block(mod, x):
            # [FUNCTION DOC] HELPER: Region taking a module and a tensor and returning x scaled
            # by mod.c.
            return x * mod.c

        block1 = Block(5)
        block2 = Block(10)

        def fn(x):
            # [FUNCTION DOC] HELPER: For each of two blocks, applies the region if the block's
            # constant has the expected value (5 and 10), otherwise passes x through; returns
            # the sum.
            # The conditional installs EQUALS_MATCH on block1.c *before*
            # the subgraph trace snapshots guards_before.
            if block1.c == 5:
                a = apply_block(block1, x)
            else:
                a = x
            if block2.c == 10:
                b = apply_block(block2, x)
            else:
                b = x
            return a + b

        x = torch.randn(8)
        ref = fn(x)
        self.assertEqual(ref, x * 5 + x * 10)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, res)
        # block1.c=5 and block2.c=10 differ → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_mutated_captured_variable(self):
        # [FUNCTION DOC] A region reads a closure-captured object whose field is mutated between
        # two calls. Reuse must be skipped: eager gives fifteen times x, compiled matches, and
        # speculation runs twice.
        """Reuse must be skipped when a captured (non-input) variable is mutated."""

        class Config:
            # [CLASS DOC] HELPER: Simple object holding a mutable field c.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores c.
                self.c = c

        cfg = Config(5)

        @nested_compile_region
        def apply(x):
            # [FUNCTION DOC] HELPER: Region returning x scaled by the captured config field c.
            # cfg is captured from closure, not an explicit input
            return x * cfg.c

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region, sets c to 10, calls again, and sums.
            a = apply(x)
            cfg.c = 10
            b = apply(x)
            return a + b

        x = torch.randn(8)
        cfg.c = 5
        ref = fn(x)
        self.assertEqual(ref, x * 15)

        cfg.c = 5
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, res)
        # cfg.c=5 and cfg.c=10 are distinct → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_synthetic_source(self):
        # [FUNCTION DOC] Reuse with a hoisted opaque value (synthetic local source): a
        # region applying a string-parameterized custom op twice must reuse the first trace
        # (one speculation call) and match eager. Imports helpers from the opaque object v2
        # test module.
        """Reuse must handle CustomClassObjectVariable with SyntheticLocalSource.

        Hoisted opaque value types get a SyntheticLocalSource that can't be
        resolved via VariableBuilder. On cache hit, stamp_out_subgraph must
        call synthetic_graph_input to create a fresh graph input.
        """
        from test_opaque_obj_v2 import HoistedString, op_with_string

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying the string op to x with a hoisted
            # string "double".
            return op_with_string(x, HoistedString("double"))

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            a = gn(x)
            b = gn(x)
            return a + b

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_synthetic_source_different_args(self):
        # [FUNCTION DOC] Three layers whose regions build hoisted string objects from a name
        # attribute (double, square, double). Compiles with the eager recording backend; expects
        # matching results, a single speculation call, one dynamo graph, and expected printed
        # graph text with one shared subgraph taking separate synthetic inputs.
        """Reuse when hoisted opaque ctor args differ across submodules."""
        from test_opaque_obj_v2 import HoistedString, op_with_string

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a name attribute; its region forward applies the
            # string op with a hoisted string built from the name.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores the name.
                super().__init__()
                self.name = name

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the string custom op with a hoisted string made
                # from the name.
                return op_with_string(x, HoistedString(self.name))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model chaining three Layer instances named double,
            # square, double.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the three layers.
                super().__init__()
                self.layer0 = Layer("double")
                self.layer1 = Layer("square")
                self.layer2 = Layer("double")

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the three layers in sequence.
                x = self.layer0(x)
                x = self.layer1(x)
                x = self.layer2(x)
                return x

        model = Model()
        x = torch.randn(8)
        ref = model(x)

        backend = EagerAndRecordGraphs()
        with self._count_speculate_calls() as count:
            res = torch.compile(model, backend=backend, fullgraph=True)(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

        self.assertEqual(len(backend.graphs), 1)
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[8]", SYNTHETIC_LOCAL_tmp_0_ : test_opaque_obj_v2_HoistedString, SYNTHETIC_LOCAL_tmp_2_ : test_opaque_obj_v2_HoistedString, SYNTHETIC_LOCAL_tmp_4_ : test_opaque_obj_v2_HoistedString):
        l_x_ = L_x_
        synthetic_local_tmp_0_ = SYNTHETIC_LOCAL_tmp_0_
        synthetic_local_tmp_2_ = SYNTHETIC_LOCAL_tmp_2_
        synthetic_local_tmp_4_ = SYNTHETIC_LOCAL_tmp_4_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_x_, synthetic_local_tmp_0_);  subgraph_0 = l_x_ = synthetic_local_tmp_0_ = None
        x: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None

        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', x, synthetic_local_tmp_2_);  subgraph_1 = x = synthetic_local_tmp_2_ = None
        x_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None

        subgraph_2 = self.subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_2, 'subgraph_0', x_1, synthetic_local_tmp_4_);  subgraph_2 = x_1 = synthetic_local_tmp_4_ = None
        x_2: "f32[8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        return (x_2,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_x_: "f32[8]", synthetic_local_tmp_0_ : test_opaque_obj_v2_HoistedString):
            op_with_string_default: "f32[8]" = torch.ops.mylib.op_with_string.default(l_x_, synthetic_local_tmp_0_);  l_x_ = synthetic_local_tmp_0_ = None
            return (op_with_string_default,)
""",
            )

    def test_subgraph_reuse_symint_arg(self):
        # [FUNCTION DOC] With dynamic shapes, a symbolic size passed as an argument across two
        # region calls must be matched by expression not identity: one speculation call. A
        # second run with a different batch size must give the eager result.
        """A symint arg must match on its expression, not on object identity.

        Each tensor holds its own SymInt objects, so the same symbol threaded
        through successive regions arrives as a different object every call.
        """

        @nested_compile_region
        def gn(x, n):
            # [FUNCTION DOC] HELPER: Region multiplying x by the integer n.
            return x * n

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region with the first dim size, then with the
            # result's size.
            a = gn(x, x.shape[0])
            b = gn(a, a.shape[0])
            return b

        x = torch.randn(8, 4)
        ref = fn(x)

        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True, dynamic=True)
        with self._count_speculate_calls() as count:
            res = compiled(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

        x2 = torch.randn(5, 4)
        self.assertEqual(fn(x2), compiled(x2))

    def test_subgraph_reuse_transposed_buffer_dynamic(self):
        # [FUNCTION DOC] Layers whose region reads a transposed view of a buffer, compiled with
        # dynamic shapes. Checks one speculation call, matching results, no recompile for a new
        # batch size (one dynamo graph), and expected printed graph text.
        """Reuse must work when a region reads a transposed view of a buffer.

        `w.T` is desugared into an aten op traced into the subgraph, so the
        region only lifts `w` itself. With dynamic shapes the region also lifts
        the symint for the (specialized) feature dim.
        """

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a square random buffer w whose region forward
            # multiplies the input by w transposed.
            def __init__(self, d):
                # [FUNCTION DOC] HELPER: Registers a random buffer of shape d by d.
                super().__init__()
                self.register_buffer("w", torch.randn(d, d))

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Matrix-multiplies x with the transposed buffer.
                return torch.matmul(x, self.w.T)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Container holding a module list of n Layer instances.
            def __init__(self, d, n):
                # [FUNCTION DOC] HELPER: Builds n layers of width d.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(d) for _ in range(n)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each layer in order.
                for layer in self.layers:
                    x = layer(x)
                return x

        model = Model(8, 3)
        x = torch.randn(4, 8)
        ref = model(x)

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(model, backend=backend, fullgraph=True, dynamic=True)
        with self._count_speculate_calls() as count:
            res = compiled(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

        # A different batch size must not recompile.
        x2 = torch.randn(9, 8)
        self.assertEqual(model(x2), compiled(x2))

        self.assertEqual(len(backend.graphs), 1)
        if not TEST_WITH_CROSSREF:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, s77: "Sym(s77)", s27: "Sym(8)", L_x_: "f32[s77, 8]", L_self_modules_layers_modules_0_buffers_w_: "f32[8, 8]", L_self_modules_layers_modules_1_buffers_w_: "f32[8, 8]", L_self_modules_layers_modules_2_buffers_w_: "f32[8, 8]"):
        l_x_ = L_x_
        l_self_modules_layers_modules_0_buffers_w_ = L_self_modules_layers_modules_0_buffers_w_
        l_self_modules_layers_modules_1_buffers_w_ = L_self_modules_layers_modules_1_buffers_w_
        l_self_modules_layers_modules_2_buffers_w_ = L_self_modules_layers_modules_2_buffers_w_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_self_modules_layers_modules_0_buffers_w_, s77, s27, l_x_);  subgraph_0 = l_self_modules_layers_modules_0_buffers_w_ = l_x_ = None
        x: "f32[s77, 8]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', l_self_modules_layers_modules_1_buffers_w_, s77, s27, x);  subgraph_1 = l_self_modules_layers_modules_1_buffers_w_ = x = None
        x_1: "f32[s77, 8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        subgraph_2 = self.subgraph_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(subgraph_2, 'subgraph_0', l_self_modules_layers_modules_2_buffers_w_, s77, s27, x_1);  subgraph_2 = l_self_modules_layers_modules_2_buffers_w_ = s77 = s27 = x_1 = None
        x_2: "f32[s77, 8]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        return (x_2,)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_self_modules_layers_modules_0_buffers_w_: "f32[8, 8]", s77: "Sym(s77)", s27: "Sym(8)", l_x_: "f32[s77, 8]"):
            numpy_t: "f32[8, 8]" = torch.ops.aten.numpy_T(l_self_modules_layers_modules_0_buffers_w_);  l_self_modules_layers_modules_0_buffers_w_ = None
            matmul: "f32[s77, 8]" = torch.matmul(l_x_, numpy_t);  l_x_ = numpy_t = None
            return (matmul,)
""",
            )

    def test_subgraph_reuse_different_list_lengths(self):
        # [FUNCTION DOC] Two region calls with list arguments of lengths two and three: the
        # pytree structures differ so reuse is rejected and two speculation calls occur; outputs
        # match eager.
        """Reuse must be skipped when list args have different lengths.

        The first call passes lists of length 2; the second passes lists of
        length 3.  The pytree treespec will differ, so the cache lookup must
        fall through and trigger a second trace.
        """

        @nested_compile_region
        def gn(xs, ys):
            # [FUNCTION DOC] HELPER: Region adding corresponding list items and returning
            # a list.
            return [a + b for a, b in zip(xs, ys)]

        def fn(xs1, ys1, xs2, ys2):
            # [FUNCTION DOC] HELPER: Calls the region on two argument list pairs and returns
            # both results.
            a = gn(xs1, ys1)
            b = gn(xs2, ys2)
            return a, b

        xs1 = [torch.randn(4), torch.randn(4)]
        ys1 = [torch.randn(4), torch.randn(4)]
        xs2 = [torch.randn(4), torch.randn(4), torch.randn(4)]
        ys2 = [torch.randn(4), torch.randn(4), torch.randn(4)]

        ref = fn(xs1, ys1, xs2, ys2)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(
                xs1, ys1, xs2, ys2
            )

        # Different list lengths → treespec mismatch → two separate traces
        self.assertEqual(count(), 2)
        for r, e in zip(res, ref):
            for ri, ei in zip(r, e):
                self.assertEqual(ri, ei)

    def _cleanup_pytree_registration(self, cls):
        # [FUNCTION DOC] HELPER: Removes the given class from the pytree supported node,
        # serialized type and constant node registries directly (the public deregister
        # call fails for classes registered without a serialized type name). Missing
        # entries are ignored.
        # _deregister_pytree_node raises KeyError for classes registered
        # without serialized_type_name (they share a sentinel key in
        # SERIALIZED_TYPE_TO_PYTHON_TYPE), so pop the registries directly
        # instead, as in test_activation_checkpointing.py.
        pytree.SUPPORTED_NODES.pop(cls, None)
        pytree.SUPPORTED_SERIALIZED_TYPES.pop(cls, None)
        pytree.CONSTANT_NODES.discard(cls)

    def test_subgraph_reuse_dataclass_pytree_arg(self):
        # [FUNCTION DOC] A registered pytree dataclass wrapping a tensor is the region argument,
        # called four times in a loop: one speculation call and a result equal to eager. This
        # guards fingerprint correctness. The dataclass registration is removed at cleanup.
        """Reuse must work for args that are registered pytree dataclasses.

        Fingerprinting such args goes through the pytree-flatten path of
        build_input_fingerprint (rather than the flat-leaf fast path), which
        must classify the tensor field inside the dataclass as a reusable
        leaf so that repeated calls with structurally-identical dataclass
        instances hit the cache instead of retracing every time.

        Note: this also passes on the pre-fix implementation (reuse already
        worked, just slowly by re-tracing pytree.tree_flatten's dispatch on
        every call) -- this guards the fingerprinting correctness, not the
        compile-time regression fixed alongside it (see
        test_subgraph_reuse_pytree_avoids_retracing_tree_flatten).
        """
        import dataclasses

        @dataclasses.dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        pytree.register_pytree_node(
            RegionInput,
            lambda value: ([value.hidden], None),
            lambda children, _: RegionInput(*children),
        )
        self.addCleanup(self._cleanup_pytree_registration, RegionInput)

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the sine of the wrapped tensor.
            return inp.hidden.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value in the dataclass and applies the
            # region four times.
            out = x
            for _ in range(4):
                out = gn(RegionInput(out))
            return out

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        # Same treespec/leaf types on every call -> single trace, rest reused.
        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_pytree_avoids_retracing_tree_flatten(self):
        # [FUNCTION DOC] Regression test for a compile-time blowup: the input fingerprinting
        # must not inline the generic tree flatten function on each lookup. Patches the inlining
        # helper of the invoke_subgraph variable module with a recorder; checks one speculation
        # call, that the dataclass's own flatten function was inlined, and that the generic tree
        # flatten was not.
        """Regression test for the #191809 compile-time blowup.

        build_input_fingerprint used to call
        _make_inlined(tx, pytree.tree_flatten) on every single reuse-lookup
        (i.e. once per region invocation, not once per compile), making
        Dynamo bytecode-trace tree_flatten's own recursive dispatch from
        scratch every time. Only the leaf-level flatten_fn of each container
        node should be inlined/traced now, so _make_inlined must never be
        called with pytree.tree_flatten itself for a plain dataclass arg.
        """
        import dataclasses

        from torch._dynamo.variables import invoke_subgraph as invoke_subgraph_mod

        def flatten_fn(value):
            # [FUNCTION DOC] HELPER: Pytree flatten function for the dataclass: returns the one-
            # item child list and no context.
            return ([value.hidden], None)

        @dataclasses.dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        pytree.register_pytree_node(
            RegionInput, flatten_fn, lambda children, _: RegionInput(*children)
        )
        self.addCleanup(self._cleanup_pytree_registration, RegionInput)

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the sine of the wrapped tensor.
            return inp.hidden.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value and applies the region eight times.
            out = x
            for _ in range(8):
                out = gn(RegionInput(out))
            return out

        x = torch.randn(8)

        orig_make_inlined = invoke_subgraph_mod._make_inlined
        inlined_fns = []

        def _tracking_make_inlined(tx, f):
            # [FUNCTION DOC] HELPER: Recording wrapper for the inlining helper: appends the
            # function to the enclosing list and forwards to the original helper.
            inlined_fns.append(f)
            return orig_make_inlined(tx, f)

        with (
            mock.patch.object(
                invoke_subgraph_mod, "_make_inlined", _tracking_make_inlined
            ),
            self._count_speculate_calls() as count,
        ):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        # Reuse must still work (same treespec/tags on every call).
        self.assertEqual(count(), 1)
        # The dataclass's own flatten_fn must have been inlined at least
        # once, to confirm this test actually reaches
        # build_fingerprint_with_pytree rather than passing vacuously (e.g.
        # if a future change routed this case through the flat-leaf fast
        # path instead).
        self.assertIn(flatten_fn, inlined_fns)
        # But the *generic recursive dispatch* must not be re-traced on
        # every call -- that's the bug being fixed.
        self.assertNotIn(pytree.tree_flatten, inlined_fns)

    def test_subgraph_reuse_pytree_partial_flatten_fn(self):
        # [FUNCTION DOC] The flatten function registered for the dataclass is a partial object
        # rather than a plain function; compilation with full graph must not graph-break and the
        # result must equal eager.
        """flatten_fn need not be a plain function -- e.g. functools.partial.

        _make_inlined only supports plain Python functions (it always wraps
        its argument in a UserFunctionVariable, which graph-breaks on
        anything else). build_input_fingerprint must fall back rather than
        call _make_inlined directly on a non-function flatten_fn.
        """
        import dataclasses
        import functools

        @dataclasses.dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        def _flatten_impl(config_flag, value):
            # [FUNCTION DOC] HELPER: Flatten implementation taking a flag first: returns the
            # one-item child list and the flag as context.
            return ([value.hidden], config_flag)

        pytree.register_pytree_node(
            RegionInput,
            functools.partial(_flatten_impl, "cfg"),
            lambda children, _: RegionInput(*children),
        )
        self.addCleanup(self._cleanup_pytree_registration, RegionInput)

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the cosine of the wrapped tensor.
            return inp.hidden.cos()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value and applies the region four times.
            out = x
            for _ in range(4):
                out = gn(RegionInput(out))
            return out

        x = torch.randn(8)
        ref = fn(x)
        # fullgraph=True raises on any graph break instead of silently
        # falling back, so this proves the partial flatten_fn doesn't
        # graph-break.
        res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_namedtuple_arg(self):
        # [FUNCTION DOC] A plain namedtuple argument must be fingerprinted as a pytree node
        # (shared registry key normalization): one speculation call and the result equals eager.
        """Reuse must work for args that are plain namedtuples.

        All namedtuple types are implicitly registered as pytree nodes under
        a single shared registry key (collections.namedtuple), unlike
        explicitly-registered dataclasses. The fingerprinting path must
        normalize the concrete namedtuple subtype to that shared key the same
        way pytree._get_node_type does, or it will be misclassified as an
        unsupported leaf and never get reused.
        """
        import collections

        RegionInput = collections.namedtuple("RegionInput", ["hidden"])

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the sine of the namedtuple field.
            return inp.hidden.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value in the namedtuple and applies the
            # region four times.
            out = x
            for _ in range(4):
                out = gn(RegionInput(out))
            return out

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_two_distinct_namedtuple_types(self):
        # [FUNCTION DOC] Two distinct namedtuple types with the same field name must not be
        # treated as the same structure. Region called with the first type, the second type,
        # then the first type again: exactly two speculation calls, results equal eager.
        """Two different namedtuple types must not be conflated by reuse.

        Both types normalize to the same registry key (collections.
        namedtuple), so correctness depends on the concrete class surviving
        in the TreeSpec context (set by pytree._namedtuple_flatten, which
        returns type(d)) rather than being lost during normalization.
        """
        import collections

        Left = collections.namedtuple("Left", ["hidden"])
        Right = collections.namedtuple("Right", ["hidden"])

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the sine of the namedtuple field.
            return inp.hidden.sin()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region with each type and returns the
            # three results.
            a = gn(Left(x))
            b = gn(Right(y))
            # A second Left(...) call should hit the cache from the first,
            # proving reuse isn't simply broken outright (which would also
            # produce two traces, same as the false Left/Right conflation
            # this test is meant to catch).
            c = gn(Left(y))
            return a, b, c

        x = torch.randn(8)
        y = torch.randn(8)
        ref = fn(x, y)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y)

        # Left and Right each need their own trace (2), but the second Left
        # call reuses the first rather than adding a third trace.
        self.assertEqual(count(), 2)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_dataclass_pytree_kwarg(self):
        # [FUNCTION DOC] A registered pytree dataclass passed as a keyword argument is
        # fingerprinted through the pytree path: one speculation call and the result
        # equals eager.
        """Reuse must work for pytree dataclasses passed as kwargs.

        build_input_fingerprint's fast path only applies when there are no
        kwargs (build_input_fingerprint:217), so the pytree-flatten path
        (exercised here with a kwarg) is the one primarily exercised by
        real callers of nested_compile_region.
        """
        import dataclasses

        @dataclasses.dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        pytree.register_pytree_node(
            RegionInput,
            lambda value: ([value.hidden], None),
            lambda children, _: RegionInput(*children),
        )
        self.addCleanup(self._cleanup_pytree_registration, RegionInput)

        @nested_compile_region
        def gn(*, inp):
            # [FUNCTION DOC] HELPER: Region with a keyword-only argument returning the sine of
            # the field.
            return inp.hidden.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value and applies the region by keyword
            # four times.
            out = x
            for _ in range(4):
                out = gn(inp=RegionInput(out))
            return out

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_nested_pytree_arg(self):
        # [FUNCTION DOC] Nested pytree argument (a mapping holding a tensor and a tuple of a
        # tensor and a registered dataclass) across four calls: one speculation call and the
        # result equals eager.
        """Reuse must work for multi-level nested pytree structures.

        Exercises the recursive `flatten(child)` call for a dict containing
        a tuple of a tensor and a registered dataclass, rather than only a
        depth-1 structure.
        """
        import dataclasses

        @dataclasses.dataclass
        class Wrapper:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        pytree.register_pytree_node(
            Wrapper,
            lambda value: ([value.hidden], None),
            lambda children, _: Wrapper(*children),
        )
        self.addCleanup(self._cleanup_pytree_registration, Wrapper)

        @nested_compile_region
        def gn(d):
            # [FUNCTION DOC] HELPER: Region unpacking the mapping and returning sine of a plus
            # cosine of the tuple's tensor plus relu of the wrapped tensor.
            b_tensor, wrapper = d["b"]
            return d["a"].sin() + b_tensor.cos() + wrapper.hidden.relu()

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Builds the nested argument from the running value and the
            # two other tensors and applies the region four times.
            out = x
            for _ in range(4):
                out = gn({"a": out, "b": (y, Wrapper(z))})
            return out

        x, y, z = torch.randn(8), torch.randn(8), torch.randn(8)
        ref = fn(x, y, z)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, y, z)

        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_pytree_unsupported_leaf(self):
        # [FUNCTION DOC] A region argument holding an object created during tracing (no source,
        # no guards) cannot be fingerprinted, so reuse is disabled: speculation runs four times
        # and the result still equals eager.
        """An arg containing an opaque/unregistered object must not crash.

        build_input_fingerprint's has_unknown flag is set instead, which
        gates reuse off for that call (the region is simply retraced) rather
        than raising or misclassifying the object. The object is built during
        tracing so it has no source and no guards; a sourceful one is reusable,
        see test_subgraph_reuse_pytree_sourceful_object_leaf.
        """

        class Opaque:
            # [CLASS DOC] HELPER: Opaque plain object holding a value v.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores v.
                self.v = v

        @nested_compile_region
        def gn(pair):
            # [FUNCTION DOC] HELPER: Region unpacking the pair and returning the sine of
            # the tensor.
            t, _opaque = pair
            return t.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Pairs the running value with a new opaque object each of
            # four iterations.
            out = x
            for _ in range(4):
                out = gn((out, Opaque(1)))
            return out

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        # has_unknown must actually be reached: every call retraces rather
        # than misclassifying Opaque as something reusable.
        self.assertEqual(count(), 4)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_pytree_sourceful_object_leaf(self):
        # [FUNCTION DOC] An opaque object that comes from an input (has a source) is a reusable
        # leaf: one speculation call and the result equals eager.
        """An opaque object with a source is a reusable leaf inside a pytree.

        classify_vt tags it InputTag.OBJECT and reuse safety comes from
        re-evaluating the guards on its source, so unlike the sourceless case
        above the region is traced once.
        """

        class Opaque:
            # [CLASS DOC] HELPER: Opaque plain object holding a value v.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores v.
                self.v = v

        @nested_compile_region
        def gn(pair):
            # [FUNCTION DOC] HELPER: Region unpacking the pair and returning the sine of
            # the tensor.
            t, _opaque = pair
            return t.sin()

        def fn(pair):
            # [FUNCTION DOC] HELPER: Applies the region four times pairing the running value
            # with the object from the input pair.
            out = pair[0]
            for _ in range(4):
                out = gn((out, pair[1]))
            return out

        opaque = Opaque(1)
        x = torch.randn(8)
        ref = fn((x, opaque))

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)((x, opaque))

        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_subgraph_reuse_pytree_registry_change_stays_correct(self):
        # [FUNCTION DOC] After compiling with a registered dataclass, the pytree registration is
        # removed and the already compiled function is called with a new input; results must
        # still equal eager.
        """Correctness must hold even if the pytree registry changes
        between calls to an already-compiled function.

        build_fingerprint_with_pytree's fast path no longer installs a
        guard on the pytree registry itself for plain-function flatten_fns
        (see its docstring), on the theory that the registry can't change
        mid-trace, and a stale/changed registry can only affect whether a
        subgraph gets reused at compile time -- never the values the
        compiled graph actually computes. This pins that: after
        deregistering the pytree node the compiled artifact was built
        against, calling that artifact again (whether or not it triggers a
        recompile) must still produce correct results.
        """
        import dataclasses

        @dataclasses.dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        pytree.register_pytree_node(
            RegionInput,
            lambda value: ([value.hidden], None),
            lambda children, _: RegionInput(*children),
        )
        self.addCleanup(self._cleanup_pytree_registration, RegionInput)

        @nested_compile_region
        def gn(inp):
            # [FUNCTION DOC] HELPER: Region returning the sine of the wrapped tensor.
            return inp.hidden.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the running value and applies the region four times.
            out = x
            for _ in range(4):
                out = gn(RegionInput(out))
            return out

        x1 = torch.randn(8)
        ref1 = fn(x1)
        compiled = torch.compile(fn, backend="aot_eager", fullgraph=True)
        self.assertEqual(ref1, compiled(x1))

        # Deregister the pytree node the compiled artifact was built
        # against, then call it again -- via the same compiled wrapper, not
        # a fresh torch.compile call, so this exercises whatever guard
        # check (or lack thereof) protects the existing compiled artifact.
        self._cleanup_pytree_registration(RegionInput)

        x2 = torch.randn(8)
        ref2 = fn(x2)
        self.assertEqual(ref2, compiled(x2))

    def test_subgraph_reuse_different_constants_retrace(self):
        # [FUNCTION DOC] Calling a region with three different scalar constants requires three
        # separate traces (three speculation calls) and the outputs equal eager.
        """Constant args with different values each require a fresh trace.

        Three calls with three distinct scalar constants → call_count == 3.
        """

        @nested_compile_region
        def gn(x, scale):
            # [FUNCTION DOC] HELPER: Region multiplying x by the scale.
            return x * scale

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region with scale 1, 2, and 3 and returns the
            # three results.
            a = gn(x, 1)
            b = gn(x, 2)
            c = gn(x, 3)
            return a, b, c

        x = torch.randn(4)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        for r, e in zip(res, ref):
            self.assertEqual(r, e)
        # Three distinct constants → three separate traces
        self.assertEqual(count(), 3)

    def test_subgraph_reuse_tuple_destructure_with_intermediates(self):
        # [FUNCTION DOC] Model of three layers whose region forward returns a hidden state and a
        # new residual (tuple destructured by the caller, with intermediates). All layers see
        # tensors with identical metadata after the embedding, so layers two and three reuse the
        # first trace: one speculation call, and both outputs equal eager.
        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Bias-free linear layer with a region-decorated forward
            # returning the hidden state and half of it as a new residual.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 8 by 8 bias-free linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(8, 8, bias=False)

            @nested_compile_region
            def forward(self, x, residual):
                # [FUNCTION DOC] HELPER: Applies the linear layer, adds the residual, and
                # returns the result together with half of it.
                h = self.linear(x)
                h = h + residual
                new_residual = h * 0.5
                return h, new_residual

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with an embedding linear layer followed by a list
            # of layers.
            def __init__(self, num_layers):
                # [FUNCTION DOC] HELPER: Creates the embedding layer and the requested number
                # of layers.
                super().__init__()
                self.embed = torch.nn.Linear(8, 8, bias=False)
                self.layers = torch.nn.ModuleList([Layer() for _ in range(num_layers)])

            def forward(self, x, residual):
                # [FUNCTION DOC] HELPER: Embeds both inputs, then threads hidden state and
                # residual through each layer; returns the final pair.
                # embed gives x/residual requires_grad=True (same as layer outputs),
                # so all layers see identical tensor metadata → single trace suffices.
                x = self.embed(x)
                residual = self.embed(residual)
                for layer in self.layers:
                    # Must support extra outputs
                    hidden_states, residual = layer(x, residual)
                    x = hidden_states
                return x, residual

        model = Model(3)
        x = torch.randn(4, 8)
        residual = torch.randn(4, 8)
        ref = model(x, residual)

        torch._dynamo.reset()

        with self._count_speculate_calls() as count:
            res = torch.compile(model, backend="aot_eager", fullgraph=True)(
                x.clone(), residual.clone()
            )
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])
        # All layers see identical tensor metadata (requires_grad=True throughout)
        # so layers[1] and layers[2] reuse layers[0]'s trace.
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_different_dynamic_symnodes(self):
        # [FUNCTION DOC] With dynamic shapes, the same region receives two different symbolic
        # sizes (dim 0 then dim 1 of the input) as arguments. They must be treated as distinct,
        # so two speculation calls occur; outputs equal eager.
        @nested_compile_region
        def gn(x, n):
            # [FUNCTION DOC] HELPER: Region multiplying x by n.
            return x * n

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region with each of the two dimension sizes.
            a = gn(x, x.shape[0])
            b = gn(x, x.shape[1])
            return a, b

        x = torch.ones(4, 6)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", dynamic=True)(x)

        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])
        # s0 (dim 0) and s1 (dim 1) are distinct symbols → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_cache_multiple_entries(self):
        # [FUNCTION DOC] Region called with shapes four, eight, four: the third call reuses the
        # first cache entry, so two speculation calls; outputs equal eager.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return x.sin()

        def fn(x4, x8, x4_again):
            # [FUNCTION DOC] HELPER: Applies the region to each of the three inputs.
            a = gn(x4)
            b = gn(x8)
            c = gn(x4_again)
            return a, b, c

        x4 = torch.randn(4)
        x8 = torch.randn(8)
        x4_again = torch.randn(4)
        ref = fn(x4, x8, x4_again)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(
                x4, x8, x4_again
            )

        for r, e in zip(res, ref):
            self.assertEqual(r, e)
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_kwargs(self):
        # [FUNCTION DOC] Region with a keyword-only scale called twice with the same keyword
        # value: one speculation call and the result equals eager.
        @nested_compile_region
        def gn(x, *, scale=1.0):
            # [FUNCTION DOC] HELPER: Region scaling x by the keyword-only scale (default 1.0).
            return x * scale

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two calls with scale 2.0.
            a = gn(x, scale=2.0)
            b = gn(x, scale=2.0)
            return a + b

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_max_entries_raises(self):
        # [FUNCTION DOC] A region created with a maximum of two reuse entries is called with
        # three distinct constants; compilation must raise RuntimeError saying the maximum reuse
        # entries were exceeded.
        """Exceeding max_reuse_entries raises RuntimeError."""

        @nested_compile_region(max_reuse_entries=2)
        def gn(x, c):
            # [FUNCTION DOC] HELPER: Region multiplying x by the constant c.
            return x * c

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the region called with constants 1, 2, and 3.
            # Three distinct constants exceed the limit of 2
            return gn(x, 1) + gn(x, 2) + gn(x, 3)

        x = torch.randn(4)
        with self.assertRaisesRegex(RuntimeError, "exceeded maximum reuse entries"):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

    def test_subgraph_reuse_enum_arg(self):
        # [FUNCTION DOC] Enum arguments reuse like constants. For each of the two members
        # the region is compiled afresh with one speculation call and eager-equal results.
        # A function using both members in one trace needs two speculation calls (members
        # not shared).
        """Enum members are UserDefinedObjectVariable but reuse like constants."""

        class Mode(enum.IntEnum):
            # [CLASS DOC] HELPER: Integer enum with members ADD (1) and SUB (2).
            ADD = 1
            SUB = 2

        @nested_compile_region
        def gn(x, mode):
            # [FUNCTION DOC] HELPER: Region returning the sine of x when mode is ADD, else
            # the cosine.
            if mode == Mode.ADD:
                return x.sin()
            return x.cos()

        def fn(x, mode):
            # [FUNCTION DOC] HELPER: Sums two region calls with the same mode.
            return gn(x, mode) + gn(x, mode)

        x = torch.randn(8)
        for mode in (Mode.ADD, Mode.SUB):
            torch._dynamo.reset()
            with self._count_speculate_calls() as count:
                res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, mode)
            self.assertEqual(res, fn(x, mode))
            self.assertEqual(count(), 1)

        # Distinct enum members must not share a traced subgraph.
        def fn2(x):
            # [FUNCTION DOC] HELPER: Sums the region called with ADD and with SUB.
            return gn(x, Mode.ADD) + gn(x, Mode.SUB)

        torch._dynamo.reset()
        with self._count_speculate_calls() as count:
            res = torch.compile(fn2, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(res, fn2(x))
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_constant_equal_but_different_type(self):
        # [FUNCTION DOC] Constants that compare equal but differ in type (an integer enum member
        # vs a plain int; a bool vs an int) must not share a trace because type checks inside
        # the region differ. For two functions, each compiled afresh, two speculation calls
        # occur and results equal eager.
        """Constants that compare equal but differ in type must not be reused.

        Mode.ADD == 1 and True == 1, yet an isinstance() check inside the
        region traces differently for each.
        """

        class Mode(enum.IntEnum):
            # [CLASS DOC] HELPER: Integer enum with one member ADD (1).
            ADD = 1

        @nested_compile_region
        def gn(x, c):
            # [FUNCTION DOC] HELPER: Region: sine if the constant is an instance of the enum,
            # else cosine.
            return x.sin() if isinstance(c, Mode) else x.cos()

        @nested_compile_region
        def hn(x, c):
            # [FUNCTION DOC] HELPER: Region: sine if the constant is a bool, else cosine.
            return x.sin() if isinstance(c, bool) else x.cos()

        x = torch.randn(8)
        fns = (
            lambda x: gn(x, Mode.ADD) + gn(x, 1),
            lambda x: hn(x, True) + hn(x, 1),
        )
        for fn in fns:
            torch._dynamo.reset()
            with self._count_speculate_calls() as count:
                res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
            self.assertEqual(res, fn(x))
            self.assertEqual(count(), 2)

    def test_subgraph_reuse_user_defined_object_arg(self):
        # [FUNCTION DOC] A sourceful user-defined object argument is reusable through guards on
        # its source. One object called twice: one speculation call. Two objects with matching
        # attribute metadata: one call. A second object whose tensor attribute has a different
        # shape: two calls. Results equal eager in each case.
        """A sourceful user-defined object arg reuses via guards on its source."""

        class Batch:
            # [CLASS DOC] HELPER: Plain holder with a delta attribute.
            def __init__(self, delta):
                # [FUNCTION DOC] HELPER: Stores delta.
                self.delta = delta

        @nested_compile_region
        def gn(x, batch):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the object's delta.
            return x.sin() + batch.delta

        def fn(x, batch):
            # [FUNCTION DOC] HELPER: Sums two region calls on the same object.
            return gn(x, batch) + gn(x, batch)

        x = torch.randn(8)
        batch = Batch(torch.randn(8))
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, batch)
        self.assertEqual(res, fn(x, batch))
        self.assertEqual(count(), 1)

        # Two distinct objects with matching attribute metadata: the guards on
        # b1's source are re-evaluated against b2's, so this is reusable.
        def fn3(x, b1, b2):
            # [FUNCTION DOC] HELPER: Sums region calls on two different objects.
            return gn(x, b1) + gn(x, b2)

        torch._dynamo.reset()
        batch2 = Batch(torch.randn(8))
        with self._count_speculate_calls() as count:
            res = torch.compile(fn3, backend="aot_eager", fullgraph=True)(
                x, batch, batch2
            )
        self.assertEqual(res, fn3(x, batch, batch2))
        self.assertEqual(count(), 1)

        # An attribute whose tensor metadata differs must not be reused.
        def fn2(x, b1, b2):
            # [FUNCTION DOC] HELPER: Sums region calls on two different objects.
            return gn(x, b1) + gn(x, b2)

        torch._dynamo.reset()
        b2 = Batch(torch.randn(1))
        with self._count_speculate_calls() as count:
            res = torch.compile(fn2, backend="aot_eager", fullgraph=True)(x, batch, b2)
        self.assertEqual(res, fn2(x, batch, b2))
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_sourceless_object_not_eligible(self):
        # [FUNCTION DOC] Objects built inside the compiled function have no guards, so reuse is
        # not allowed: two speculation calls, and the result equals eager.
        """An object created during tracing has no guards, so it is not reused."""

        class Batch:
            # [CLASS DOC] HELPER: Plain holder with a delta attribute.
            def __init__(self, delta):
                # [FUNCTION DOC] HELPER: Stores delta.
                self.delta = delta

        @nested_compile_region
        def gn(x, batch):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the object's delta.
            return x.sin() + batch.delta

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two region calls on freshly built holders of cosine.
            return gn(x, Batch(x.cos())) + gn(x, Batch(x.cos()))

        x = torch.randn(8)
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(res, fn(x))
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_rebound_global_read_via_side_effects(self):
        # [FUNCTION DOC] A region reads a module global that the compiled function rebinds each
        # loop iteration. Both calls must retrace (two speculation calls) and, outside crossref,
        # the printed graph shows one shared subgraph with the tensor lifted as an input. The
        # global is reset to None in a finally block.
        """A region reading a global that is rebound between calls must retrace."""
        global _reuse_test_global

        @dataclass
        class RegionInput:
            # [CLASS DOC] HELPER: Dataclass holding one tensor field named hidden.
            hidden: torch.Tensor

        @nested_compile_region
        def gn():
            # [FUNCTION DOC] HELPER: Region returning the global object's tensor plus one.
            return _reuse_test_global.hidden + 1

        def fn(hidden):
            # [FUNCTION DOC] HELPER: Twice: rebinds the global to a new holder of hidden and
            # calls the region, replacing hidden with the result; returns the final value.
            global _reuse_test_global
            for _ in range(2):
                _reuse_test_global = RegionInput(hidden)
                hidden = gn()
            return hidden

        try:
            x = torch.tensor(0)
            ref = fn(x)
            backend = EagerAndRecordGraphs()
            with self._count_speculate_calls() as count:
                res = torch.compile(fn, backend=backend, fullgraph=True)(x)
            self.assertEqual(res, ref)
            # Both calls retrace: the global is rebound every iteration, so
            # GlobalSource lands in both traced_sources and mutated_sources and
            # no reuse entry is ever saved.
            self.assertEqual(count(), 2)
            # The captured tensor is lifted as a subgraph input, and the second
            # trace dedups onto the first graph via are_same_graph_modules, so
            # the retrace costs Dynamo time but yields one shared subgraph.
            # Getting to a single trace needs the capture re-read at the new
            # call site: it is not a user arg, so reuse would re-resolve its
            # recorded source, and on the second call the right value is an
            # intermediate that no source names.
            if not TEST_WITH_CROSSREF:
                self.assertExpectedInline(
                    normalize_gm(backend.graphs[0].print_readable(False)),
                    """\
class GraphModule(torch.nn.Module):
    def forward(self, L_hidden_: "i64[]"):
        l_hidden_ = L_hidden_

        subgraph_0 = self.subgraph_0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(subgraph_0, 'subgraph_0', l_hidden_);  subgraph_0 = l_hidden_ = None
        hidden: "i64[]" = invoke_subgraph[0];  invoke_subgraph = None
        subgraph_1 = self.subgraph_0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(subgraph_1, 'subgraph_0', hidden);  subgraph_1 = None
        hidden_1: "i64[]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (hidden, hidden_1)

    class subgraph_0(torch.nn.Module):
        def forward(self, l_hidden_: "i64[]"):
            add: "i64[]" = l_hidden_ + 1;  l_hidden_ = None
            return (add,)
""",
                )
        finally:
            _reuse_test_global = None

    def test_subgraph_reuse_rebound_global_read_via_builder(self):
        # [FUNCTION DOC] Variant where the global is first read through the variable builder
        # before being rebound. The second call's cache lookup rejects the saved entry, so two
        # speculation calls; the second call receives the new tensor correctly. Result equals
        # eager; the global is reset in a finally block.
        """A rebound global whose first read predates the rebinding.

        Unlike test_subgraph_reuse_rebound_global_read_via_side_effects, the
        region's first read goes through VariableBuilder rather than being
        served out of side effects.
        """
        global _reuse_test_global

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region adding the module global to x.
            return x + _reuse_test_global

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region, rebinds the global to 100, calls again
            # and sums.
            global _reuse_test_global
            a = gn(x)
            _reuse_test_global = torch.tensor(100)
            return a + gn(x)

        try:
            x = torch.tensor(0)
            _reuse_test_global = torch.tensor(10)
            ref = fn(x)
            _reuse_test_global = torch.tensor(10)
            with self._count_speculate_calls() as count:
                res = torch.compile(fn, backend="eager", fullgraph=True)(x)
            self.assertEqual(res, ref)
            # The first call saves a reuse entry, but the second call's lookup
            # rejects it: GlobalSource is in traced_sources and the rebinding
            # put it in mutated_sources. The global does get lifted as a
            # subgraph input, so both call sites share one graph module and the
            # second correctly receives the new tensor -- only the Dynamo trace
            # is duplicated. Reaching 1 needs the read re-resolved against
            # side-effect state at the second call site; replaying the saved
            # source yields the pre-rebinding value, which miscompiles.
            self.assertEqual(count(), 2)
        finally:
            _reuse_test_global = None

    def test_subgraph_reuse_global_written_once_before_loop(self):
        # [FUNCTION DOC] A global assigned once at the start of the compiled function (before
        # any region) still blocks reuse conservatively, so each of four calls retraces (four
        # speculation calls); results are correct. The global is reset in a finally block.
        """Conservative: a global written once, before any region runs, blocks reuse.

        has_mutated_vars asks whether a source was ever mutated, not whether it
        changed since the entry was saved, so every call retraces even though
        the global is stable by the time the first region runs. Results stay
        correct; only reuse is lost. Versioning mutations per source would let
        this collapse back to a single trace -- update the count here if that
        lands.
        """
        global _reuse_test_global

        class Ctx:
            # [CLASS DOC] HELPER: Holder with a scale attribute.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Stores the scale.
                self.scale = scale

        @nested_compile_region
        def gn(w, x):
            # [FUNCTION DOC] HELPER: Region returning x times w plus the global object's scale.
            return x * w + _reuse_test_global.scale

        def fn(x, ws):
            # [FUNCTION DOC] HELPER: Sets the global holder, then applies the region over
            # the weights.
            global _reuse_test_global
            _reuse_test_global = Ctx(2.0)
            for w in ws:
                x = gn(w, x)
            return x

        try:
            ws = [torch.full((4,), float(i + 1)) for i in range(4)]
            x = torch.ones(4)
            ref = fn(x, ws)
            with self._count_speculate_calls() as count:
                res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x, ws)
            self.assertEqual(res, ref)
            self.assertEqual(count(), 4)
        finally:
            _reuse_test_global = None

    def test_subgraph_reuse_sourceless_module_not_eligible(self):
        # [FUNCTION DOC] Modules built inside the compiled function carry no guards, so two
        # modules with different constants require two traces; result equals eager.
        """Sourceless modules carry no guards, so differing attrs must retrace."""

        class Scale(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        @nested_compile_region
        def gn(mod, x):
            # [FUNCTION DOC] HELPER: Region calling the given module on x.
            return mod(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the region on two newly built modules with scales 3
            # and 5.
            return gn(Scale(3.0), x) + gn(Scale(5.0), x)

        x = torch.randn(8)
        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)
        self.assertEqual(res, fn(x))
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_module_different_instances_retrace(self):
        # [FUNCTION DOC] Two instances with different constants (5 and 10) need separate traces:
        # two speculation calls, and the result equals eager.
        """Different module instances with different weights require separate traces."""

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant via a
            # region forward.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        mod1 = Mod(5)
        mod2 = Mod(10)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the two module calls.
            return mod1(x) + mod2(x)

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        # mod1.c=5 and mod2.c=10 differ → two separate traces
        self.assertEqual(count(), 2)

    def test_subgraph_reuse_monkeypatch_forward(self):
        # [FUNCTION DOC] Two modules whose forward was monkeypatched with region-wrapped copies
        # of the same code object (distinct function objects) must share a trace because the
        # cache key is based on the code object: asserts the inner functions differ but share
        # code, one speculation call, and eager-equal output.
        """Monkeypatching forward with nested_compile_region should reuse cache.

        When a user wraps a module's forward with nested_compile_region via
        monkeypatching, each module instance gets a distinct bound method and
        thus a distinct function object passed to nested_compile_region. The
        cache key should be based on fn.__code__ rather than id(fn) so that
        reuse still works across instances whose forward shares the same code.
        """

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        def apply_nested_compile_region(mod):
            # [FUNCTION DOC] HELPER: Replaces the module instance forward with a region-wrapped
            # bound version of a fresh copy of the class forward function (same code, new
            # function object). Returns the module.
            # Grab the unbound function and create a fresh copy so that each
            # module instance gets a function with a *different* id() but the
            # *same* __code__ object, mimicking a monkeypatch/code-gen scenario.
            import types

            orig = type(mod).forward
            fresh_fn = types.FunctionType(
                orig.__code__,
                orig.__globals__,
                orig.__name__,
                orig.__defaults__,
                orig.__closure__,
            )
            mod.forward = nested_compile_region(fresh_fn).__get__(mod, type(mod))
            return mod

        mod1 = apply_nested_compile_region(Mod(5))
        mod2 = apply_nested_compile_region(Mod(5))

        # The inner functions passed to invoke_subgraph_placeholder are
        # different objects but share the same __code__.
        fn1 = mod1.forward.__func__.__marked_compile_region_fn__
        fn2 = mod2.forward.__func__.__marked_compile_region_fn__
        self.assertIsNot(fn1, fn2)
        self.assertIs(fn1.__code__, fn2.__code__)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the two module calls.
            return mod1(x) + mod2(x)

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        # Both modules have the same code and same c=5, so the second call
        # should reuse the first trace despite having different function ids.
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_module_apply(self):
        # [FUNCTION DOC] Applying a wrapper to every layer via the module apply traversal
        # wraps the class forward in a region once. Three equal layers need one speculation
        # call and the output equals the unwrapped reference model. Note this modifies the
        # class forward permanently.
        """Using module.apply to wrap transformer layers with nested_compile_region."""

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        model = torch.nn.Sequential(Layer(5), Layer(5), Layer(5))
        ref_model = torch.nn.Sequential(Layer(5), Layer(5), Layer(5))

        def wrap_layer(mod):
            # [FUNCTION DOC] HELPER: Callback for module apply: if the module is a Layer whose
            # class forward is not already marked as a compile region, replaces the class
            # forward with the region-wrapped version (mutates the class). Returns nothing.
            if isinstance(mod, Layer):
                fwd = type(mod).forward
                if not hasattr(fwd, "__marked_compile_region_fn__"):
                    type(mod).forward = nested_compile_region(fwd)

        model.apply(wrap_layer)

        x = torch.randn(8)
        ref = ref_model(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(model, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        # All three layers share the same class and c=5, only one trace needed.
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_class_level_wrap(self):
        # [FUNCTION DOC] Class-level wrap of the layer forward with the region decorator: two
        # instances with equal constants share a trace (one speculation call), and the output
        # equals eager. Note this modifies the class forward permanently.
        """Wrapping cls.forward with nested_compile_region for named_modules pattern."""

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        # Wrap at the class level — the pattern a user would use after
        # iterating named_modules to wrap all layers of a given type.
        Layer.forward = nested_compile_region(Layer.forward)

        mod1 = Layer(5)
        mod2 = Layer(5)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the two module calls.
            return mod1(x) + mod2(x)

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        self.assertEqual(count(), 1)

    def test_subgraph_reuse_module_instance_as_callable(self):
        # [FUNCTION DOC] Module instances passed directly to the region wrapper and called: with
        # equal constants they share one trace (one speculation call) and match eager.
        """Passing nn.Module instances directly to nested_compile_region."""

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module scaling its input by a constructor constant.
            def __init__(self, c):
                # [FUNCTION DOC] HELPER: Stores the constant c.
                super().__init__()
                self.c = c

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x scaled by c.
                return x * self.c

        mod1 = Layer(5)
        mod2 = Layer(5)

        wrapped1 = nested_compile_region(mod1)
        wrapped2 = nested_compile_region(mod2)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums the two wrapped module calls.
            return wrapped1(x) + wrapped2(x)

        x = torch.randn(8)
        ref = fn(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(ref, res)
        # Both modules have the same forward code and same c=5, so the
        # second call should reuse the first trace.
        self.assertEqual(count(), 1)


@skipIfTorchDynamo("Not a torch._dynamo test")
@parameterized_class(
    [
        {"strict": False},
        {"strict": True},
    ],
    class_name_func=lambda cls,
    _,
    params: f"{cls.__name__}{'Strict' if params['strict'] else 'Nonstrict'}",
)
class TestInvokeSubgraphExport(TestCase):
    # [CLASS DOC] Export tests for regions, run twice via a parameterized class generator: once
    # with the strict flag false and once true (class names get a Strict or Nonstrict suffix).
    # Each test reads the strict flag from the class attribute. Skipped under dynamo.
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_simple_func(self):
        # [FUNCTION DOC] Exports a module calling a region twice with inlining of single-use
        # disabled. Checks the exported module output equals eager, that the graph module has
        # two named modules (itself and one repeated subgraph), and the printed graph matches
        # expected text.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying the inputs.
            return torch.mul(x, y)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the region twice in sequence.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the region to (x, y) and then to the result
                # with y.
                x = gn(x, y)
                x = gn(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

        self.assertExpectedInline(
            empty_line_normalizer(
                normalize_gm(ep.graph_module.print_readable(print_output=False))
            ),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x: "f32[8]", y: "f32[8]"):
        repeated_subgraph0 = self.repeated_subgraph0
        invoke_subgraph = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0, 'subgraph_0', x, y);  repeated_subgraph0 = x = None
        getitem: "f32[8]" = invoke_subgraph[0];  invoke_subgraph = None
        repeated_subgraph0_1 = self.repeated_subgraph0
        invoke_subgraph_1 = torch.ops.higher_order.invoke_subgraph(repeated_subgraph0_1, 'subgraph_0', getitem, y);  repeated_subgraph0_1 = getitem = y = None
        getitem_1: "f32[8]" = invoke_subgraph_1[0];  invoke_subgraph_1 = None
        return (getitem_1,)
    class repeated_subgraph0(torch.nn.Module):
        def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
            mul: "f32[8]" = torch.ops.aten.mul.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
            return (mul,)
""",
        )

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_unbacked(self):
        # [FUNCTION DOC] Exports a module calling a data-dependent region (item with a runtime
        # check) ten times, runs decompositions, and checks the output matches eager and the
        # graph module still has only two named modules.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region reading the scalar from x, asserting it is below the
            # length of y, and returning a clone of that many leading elements of y.
            b = x.item()
            torch._check(b < y.shape[0])
            return y[:b].clone()

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module collecting ten region results and concatenating them.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the region ten times and concatenates
                # the outputs.
                res = []
                for _ in range(10):
                    res.append(gn(x, y))
                return torch.cat(res)

        x = torch.tensor(4)
        y = torch.randn(8)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        ep = ep.run_decompositions()

        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_pending_unbacked(self):
        # [FUNCTION DOC] Exports a module whose region method reads an item from a dynamic-size
        # input, called four times. Done once with a plain input and once with an input
        # requiring grad; after decompositions each export must still have two named modules.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a region-decorated method called four times
            # in forward.
            @nested_compile_region
            def gn(self, x):
                # [FUNCTION DOC] HELPER: Region method multiplying x by its first element's
                # scalar value.
                u = x[0].item()
                return x * u

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region method four times.
                for _ in range(4):
                    x = self.gn(x)
                return x

        ep = torch.export.export(
            M(),
            (torch.randn(8),),
            strict=self.strict,
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        )
        ep = ep.run_decompositions()

        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

        ep = torch.export.export(
            M(),
            (torch.randn(8, requires_grad=True),),
            strict=self.strict,
            dynamic_shapes={"x": {0: torch.export.Dim.DYNAMIC}},
        )
        ep = ep.run_decompositions()

        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_simple_method(self):
        # [FUNCTION DOC] Exports a module whose region is a method called twice; output must
        # match eager and the graph module must contain two named modules.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a region-decorated method called twice in forward.
            @nested_compile_region
            def gn(self, x, y):
                # [FUNCTION DOC] HELPER: Region method multiplying the inputs.
                return torch.mul(x, y)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the region method twice.
                x = self.gn(x, y)
                x = self.gn(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_multiple_module(self):
        # [FUNCTION DOC] Exports a model with ten instances of a module sharing one buffer
        # tensor and a region forward. Output must match eager and the graph module must have
        # two named modules (the instances share one subgraph).
        b = torch.randn(8)

        class N(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering the shared outer tensor as a buffer; its
            # region forward returns x times y plus the buffer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Registers the shared tensor as buffer buf.
                super().__init__()
                self.register_buffer("buf", b)

            @nested_compile_region
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x times y plus the buffer.
                return x * y + self.buf

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Container with a list of ten modules applied in sequence.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the ten-module list.
                super().__init__()
                self.mod_list = torch.nn.ModuleList(N() for _ in range(10))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Threads x through each module with y.
                for m in self.mod_list:
                    x = m(x, y)
                return x

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertTrue(torch.allclose(ep.module()(x, y), M()(x, y)))
        self.assertEqual(len(list(ep.graph_module.named_modules())), 2)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_nested_lifted_tensor_constants(self):
        # [FUNCTION DOC] A region creating an inline tensor constant is replayed twice. After
        # export and decomposition the constant appears as a persistent buffer scoped to the
        # repeated subgraph. Checks the flat-input helper works before and after decomposition
        # (last item is the input), that the buffer is in the signature but not in state_dict or
        # constants, that named_buffers resolves it with the right shape and dtype, and that
        # signature verification passes.
        # KNOWN ISSUE: the second argument of the assertTrue over the buffer names is a lambda,
        # not a message string, so a failure would show the function object as the message.
        # When the region captured by nested_compile_region closes over an
        # inline tensor constant and is replayed multiple times, export
        # promotes the subgraph to ``repeated_subgraph0``. After decomposing
        # the ExportedProgram, the inner tensor constant surfaces in the
        # top-level graph signature as a persistent buffer named like
        # ``repeated_subgraph0._tensor_constant0``, even though the tensor
        # actually lives inside the subgraph submodule rather than in the
        # top-level ``state_dict`` or ``constants``. Verify that export,
        # decomposition, verification, ``named_buffers``, and direct
        # invocation (``ep(x)`` via ``_graph_module_flat_inputs``) all handle
        # that shape.
        @nested_compile_region
        def block(x):
            # [FUNCTION DOC] HELPER: Region creating a four-element constant tensor, returning x
            # times it plus one.
            w = torch.tensor([1.0, 2.0, 3.0, 4.0])
            return x * w + 1.0

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying the region twice.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region twice in sequence.
                x = block(x)
                x = block(x)
                return x

        x = torch.randn(4)

        ep = torch.export.export(M(), (x,), strict=self.strict)

        # _graph_module_flat_inputs resolves all buffer inputs including
        # subgraph-owned lifted constants; call it directly to verify no
        # KeyError is raised and the flat inputs are well-formed.
        flat_inputs = ep._graph_module_flat_inputs((x,), {})
        self.assertIsInstance(flat_inputs, tuple)
        # x must be the last element; preceding entries are the constants/buffers
        self.assertIs(flat_inputs[-1], x)

        ep = ep.run_decompositions({})

        # After decomposition the lifted constant moves from the CONSTANT_TENSOR
        # path to a persistent buffer in the graph signature backed by the
        # subgraph submodule's state. Call _graph_module_flat_inputs again
        # to exercise the new gm_state_dict fallback branch in that method.
        flat_inputs_post = ep._graph_module_flat_inputs((x,), {})
        self.assertIsInstance(flat_inputs_post, tuple)
        self.assertIs(flat_inputs_post[-1], x)

        # The inner tensor constant should be exposed as a subgraph-scoped
        # buffer in the top-level graph signature.
        buffer_names = list(ep.graph_signature.buffers)
        self.assertTrue(
            any(
                name.startswith("repeated_subgraph") and "._tensor_constant" in name
                for name in buffer_names
            ),
            lambda msg: f"{msg}\nExpected repeated_subgraph*._tensor_constant* buffer; got {buffer_names}",
        )
        # Use the first buffer that actually matches the expected naming pattern
        # so the subsequent assertions are meaningful rather than checking an
        # arbitrary first element.
        nested_buf = next(
            name
            for name in buffer_names
            if name.startswith("repeated_subgraph") and "._tensor_constant" in name
        )
        self.assertNotIn(nested_buf, ep.state_dict)
        self.assertNotIn(nested_buf, ep.constants)

        # named_buffers must resolve the constant through the subgraph's
        # submodule state rather than raising KeyError. After decomposition
        # the buffer is a FakeTensor, so just check the metadata.
        resolved = dict(ep.named_buffers())
        self.assertIn(nested_buf, resolved)
        self.assertEqual(resolved[nested_buf].shape, torch.Size([4]))
        self.assertEqual(resolved[nested_buf].dtype, torch.float32)

        # Verification must pass despite the buffer not being in state_dict.
        from torch._export.verifier import _verify_exported_program_signature

        _verify_exported_program_signature(ep)

    def test_invoke_subgraph_placeholder_forwards_kwargs(self):
        # [FUNCTION DOC] Regression test: under non-strict export the region placeholder must
        # forward keyword arguments. Exports a module calling the region with a keyword scale
        # and checks the exported module output equals eager.
        # Under non-strict export (``torch.compiler.is_compiling()`` True but
        # not Dynamo) the placeholder goes through a small wrapper that must
        # forward both *args and **kwargs. Regression test for a path that
        # silently dropped kwargs.
        @nested_compile_region
        def gn(x, y, *, scale=1.0):
            # [FUNCTION DOC] HELPER: Region returning x times y times the keyword-only scale.
            return x * y * scale

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the region with scale 2.0.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Calls the region with a keyword scale.
                return gn(x, y, scale=2.0)

        x = torch.randn(4)
        y = torch.randn(4)

        ep = torch.export.export(M(), (x, y), strict=self.strict)
        self.assertEqual(ep.module()(x, y), M()(x, y))


class NegativeTesting(TestCase):
    # [CLASS DOC] Tests of expected failures for regions (not skipped under dynamo).
    def test_graph_break(self):
        # [FUNCTION DOC] A graph break inside a region with the eager backend (no full graph)
        # must raise RuntimeError mentioning the invoke_subgraph higher order operator.
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region that forces a graph break and then returns cosine.
            torch._dynamo.graph_break()
            return torch.cos(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(8, 8, requires_grad=True)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Higher Order Operator: torch\.ops\.higher_order\.invoke_subgraph",
        ):
            torch.compile(fn, backend="eager")(x)


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInlineInvokeSubgraph(TestCase):
    # [CLASS DOC] Tests with the config flag that inlines all invoke_subgraph calls at trace
    # time. Skipped under dynamo.
    def _assert_no_invoke_subgraph(self, fn, args):
        # [FUNCTION DOC] HELPER: Compiles fn with the eager recording backend (full graph) on
        # the given argument tuple, asserts at least one graph was captured and no graph
        # contains an invoke_subgraph call node, and returns the result.
        """Compile fn and verify the backend receives no invoke_subgraph HOPs."""
        backend = EagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(*args)
        self.assertTrue(len(backend.graphs) > 0)
        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(
                    node.op == "call_function"
                    and node.target is torch.ops.higher_order.invoke_subgraph,
                )
        return res

    @torch._dynamo.config.patch(inline_invoke_subgraph=True)
    def test_simple(self):
        # [FUNCTION DOC] With inlining enabled, a region called twice leaves no invoke_subgraph
        # nodes and the result equals eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x, y) + gn(x, y)

        x = torch.randn(8)
        y = torch.randn(8)
        ref = fn(x, y)
        res = self._assert_no_invoke_subgraph(fn, (x, y))
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_invoke_subgraph=True)
    def test_module(self):
        # [FUNCTION DOC] With inlining enabled, a module region forward called twice leaves no
        # invoke_subgraph nodes and matches eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose region forward returns sine plus cosine.
            @nested_compile_region
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sine of x plus cosine of x.
                return x.sin() + x.cos()

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two module calls.
            return mod(x) + mod(x)

        x = torch.randn(8)
        ref = fn(x)
        res = self._assert_no_invoke_subgraph(fn, (x,))
        self.assertEqual(ref, res)

    @torch._dynamo.config.patch(inline_invoke_subgraph=True)
    def test_backward(self):
        # [FUNCTION DOC] With inlining enabled, compiles with inductor and checks values and
        # gradients match eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region returning the sine of the product.
            return torch.mul(x, y).sin()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x, y)

        x = torch.randn(8, requires_grad=True)
        y = torch.randn(8, requires_grad=True)
        ref = fn(x, y)

        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)
        res = torch.compile(fn, backend="inductor", fullgraph=True)(x2, y2)

        ref.sum().backward()
        res.sum().backward()

        self.assertEqual(ref, res)
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInlineSingleUseInvokeSubgraph(TestCase):
    # [CLASS DOC] Tests that regions used only once are inlined while regions used multiple
    # times remain as invoke_subgraph calls. Skipped under dynamo.
    def _assert_no_invoke_subgraph(self, fn, args):
        # [FUNCTION DOC] HELPER: Compiles fn with the eager recording backend (full graph),
        # asserts at least one graph exists and none contains an invoke_subgraph call node, and
        # returns the result.
        backend = EagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(*args)
        self.assertTrue(len(backend.graphs) > 0)
        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(
                    node.op == "call_function"
                    and node.target is torch.ops.higher_order.invoke_subgraph,
                )
        return res

    def _count_invoke_subgraph(self, fn, args):
        # [FUNCTION DOC] HELPER: Compiles fn with the eager recording backend and returns the
        # result together with the total count of invoke_subgraph call nodes over all
        # captured graphs.
        backend = EagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(*args)
        count = 0
        for gm in backend.graphs:
            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.higher_order.invoke_subgraph
                ):
                    count += 1
        return res, count

    def test_single_use_inlined(self):
        # [FUNCTION DOC] A region used once is inlined (no invoke_subgraph nodes) and the result
        # equals eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Calls the region once.
            return gn(x, y)

        x = torch.randn(8)
        y = torch.randn(8)
        ref = fn(x, y)
        res = self._assert_no_invoke_subgraph(fn, (x, y))
        self.assertEqual(ref, res)

    def test_multi_use_preserved(self):
        # [FUNCTION DOC] A region used twice is preserved: two invoke_subgraph nodes and the
        # result equals eager.
        @nested_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region multiplying inputs.
            return torch.mul(x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Sums two region calls.
            return gn(x, y) + gn(x, y)

        x = torch.randn(8)
        y = torch.randn(8)
        ref = fn(x, y)
        res, count = self._count_invoke_subgraph(fn, (x, y))
        self.assertEqual(ref, res)
        self.assertEqual(count, 2)

    def test_mixed_single_and_multi_use(self):
        # [FUNCTION DOC] One region used twice and another used once: the compiled graph has
        # two invoke_subgraph nodes (the shared one twice, the unique one inlined); result
        # equals eager.
        @nested_compile_region
        def shared(x):
            # [FUNCTION DOC] HELPER: Region returning sine (used twice).
            return x.sin()

        @nested_compile_region
        def unique(x):
            # [FUNCTION DOC] HELPER: Region returning cosine (used once).
            return x.cos()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the shared region twice, sums, then applies the
            # unique region.
            a = shared(x) + shared(x)
            b = unique(a)
            return b

        x = torch.randn(8)
        ref = fn(x)
        backend = EagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x)
        self.assertEqual(ref, res)
        # shared is called twice -> kept as invoke_subgraph
        # unique is called once -> inlined
        invoke_count = 0
        for gm in backend.graphs:
            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.higher_order.invoke_subgraph
                ):
                    invoke_count += 1
        self.assertEqual(invoke_count, 2)

    def test_ac_shared_subgraph_not_inlined(self):
        # [FUNCTION DOC] Two activation-checkpoint calls over the same region must keep
        # invoke_subgraph: reference eager forward and backward, then compile with the eager
        # recording backend and check the result equals eager and that exactly two
        # invoke_subgraph call nodes exist across all graph modules of the captured graph (use
        # counting is global).
        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Sums two checkpointed calls of the region.
            a = torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=False)
            b = torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=False)
            return a + b

        x1 = torch.randn(8, requires_grad=True)
        x2 = x1.clone().detach().requires_grad_(True)
        ref = fn(x1)
        ref.sum().backward()

        # Two checkpoint bodies share the same gn subgraph module.
        # Global counting should see 2 uses and preserve invoke_subgraph.
        backend = EagerAndRecordGraphs()
        res = torch.compile(fn, backend=backend, fullgraph=True)(x2)
        self.assertEqual(ref, res)

        invoke_count = sum(
            1
            for mod in backend.graphs[0].modules()
            if isinstance(mod, torch.fx.GraphModule)
            for node in mod.graph.nodes
            if node.op == "call_function"
            and node.target is torch.ops.higher_order.invoke_subgraph
        )
        self.assertEqual(invoke_count, 2)

    def test_inline_single_use_with_autograd_function(self):
        # [FUNCTION DOC] A region wrapping a custom autograd function is used once (so inlined).
        # Compiled with aot_eager: value, and gradient after backward, must match eager.
        class MyOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function with sine forward and a backward multiplying
            # the incoming gradient by the cosine of the saved input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Saves x for backward and returns its sine.
                ctx.save_for_backward(x)
                return x.sin()

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: Returns grad_out times the cosine of the saved input.
                (x,) = ctx.saved_tensors
                return grad_out * x.cos()

        @nested_compile_region
        def gn(x):
            # [FUNCTION DOC] HELPER: Region applying the custom autograd function.
            return MyOp.apply(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        x_clone = x.clone().detach().requires_grad_(True)
        opt_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        res = opt_fn(x_clone)

        self.assertEqual(ref, res)

        ref.sum().backward()
        res.sum().backward()
        self.assertEqual(x.grad, x_clone.grad)


@skipIfTorchDynamo("Not a torch._dynamo test")
class TestInvokeSubgraphReuseHashFn(TestCase):
    # [CLASS DOC] Tests of the reuse hash function option of the region decorator, which lets
    # the user supply a function whose return value decides whether two calls can share a trace.
    # Skipped under dynamo.
    @contextlib.contextmanager
    def _count_speculate_calls(self):
        # [FUNCTION DOC] HELPER: Context manager patching the dynamo subgraph speculation
        # function with a counting wrapper that forwards to the original. Yields a callable
        # returning the current call count.
        count = 0
        orig = torch._dynamo.variables.higher_order_ops.speculate_subgraph_with_auto_output_flattening

        def _counting(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Counting wrapper: increments the enclosing count and
            # forwards all arguments to the original function, returning its result.
            nonlocal count
            count += 1
            return orig(*args, **kwargs)

        with mock.patch.object(
            torch._dynamo.variables.higher_order_ops,
            "speculate_subgraph_with_auto_output_flattening",
            _counting,
        ):
            yield lambda: count

    def test_reuse_hash_fn_module_distinct_hashes(self):
        # [FUNCTION DOC] Hash function returns the layer id of a module argument. Four layers
        # have distinct ids so four traces occur (four speculation calls); output equals eager.
        """nn.Module arg with hash fn returning different values per layer."""

        def hash_fn(mod, x):
            # [FUNCTION DOC] HELPER: Hash function returning the module's layer_id attribute.
            return mod.layer_id

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def layer_fn(mod, x):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the module's weight.
            return x.sin() + mod.weight

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a layer_id attribute and a random parameter weight
            # of size 8.
            def __init__(self, layer_id):
                # [FUNCTION DOC] HELPER: Stores layer_id and creates the weight parameter.
                super().__init__()
                self.layer_id = layer_id
                self.weight = torch.nn.Parameter(torch.randn(8))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a list of four Layer modules applied through
            # the region.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates four layers with ids 0 to 3.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(i) for i in range(4)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region to each layer in turn.
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        mod = Model()
        x = torch.randn(8)
        ref = mod(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(mod, backend="aot_eager", fullgraph=True)(x)

        # Each layer has a distinct layer_id → 4 separate traces
        self.assertEqual(count(), 4)
        self.assertEqual(ref, res)

    def test_reuse_hash_fn_module_same_hash(self):
        # [FUNCTION DOC] Hash function returns a constant for all layers, so four layers share a
        # single trace (one speculation call); output equals eager.
        """nn.Module arg with hash fn returning same value → single trace."""

        def hash_fn(mod, x):
            # [FUNCTION DOC] HELPER: Hash function returning the constant zero.
            return 0

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def layer_fn(mod, x):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the module's weight.
            return x.sin() + mod.weight

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random parameter weight of size 8.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the weight parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a list of four Layer modules applied through
            # the region.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates four layers.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(4)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region to each layer in turn.
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        mod = Model()
        x = torch.randn(8)
        ref = mod(x)

        with self._count_speculate_calls() as count:
            res = torch.compile(mod, backend="aot_eager", fullgraph=True)(x)

        # All layers hash to 0 → single trace + 3 stamp-outs
        self.assertEqual(count(), 1)
        self.assertEqual(ref, res)

    def test_reuse_hash_fn_tensor_shape(self):
        # [FUNCTION DOC] Hash function returns the first dimension size. Inputs of size 4, 8, 8
        # yield two traces (the second 8 reuses); output equals eager.
        """Hash fn that uses tensor shape to differentiate inputs."""

        def hash_fn(x):
            # [FUNCTION DOC] HELPER: Hash function returning the size of dim 0.
            return x.shape[0]

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning sine.
            return x.sin()

        def fn(x4, x8a, x8b):
            # [FUNCTION DOC] HELPER: Applies the region to each input and sums all the results.
            # x8a and x8b have the same shape → same hash → reuse
            a = gn(x4)
            b = gn(x8a)
            c = gn(x8b)
            return a.sum() + b.sum() + c.sum()

        x4 = torch.randn(4)
        x8a = torch.randn(8)
        x8b = torch.randn(8)
        ref = fn(x4, x8a, x8b)

        with self._count_speculate_calls() as count:
            res = torch.compile(fn, backend="aot_eager", fullgraph=True)(x4, x8a, x8b)

        # shape[0]=4 and shape[0]=8 → 2 traces, third call reuses shape=8
        self.assertEqual(count(), 2)
        self.assertEqual(ref, res)

    def test_reuse_hash_fn_graph_break_raises(self):
        # [FUNCTION DOC] A hash function that triggers a graph break must make compilation raise
        # RuntimeError saying the hash function must be fully traceable.
        """reuse_hash_fn with a graph break raises a clear error."""

        def bad_hash_fn(mod, x):
            # [FUNCTION DOC] HELPER: Hash function that forces a graph break then returns zero.
            torch._dynamo.graph_break()
            return 0

        @nested_compile_region(reuse_hash_fn=bad_hash_fn)
        def layer_fn(mod, x):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the module's weight.
            return x.sin() + mod.weight

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random parameter weight of size 8.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the weight parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(8))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a list of two Layer modules applied through
            # the region.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two layers.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(), Layer()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region to each layer in turn.
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        mod = Model()
        x = torch.randn(8)
        with self.assertRaisesRegex(
            RuntimeError, "reuse_hash_fn must be fully traceable"
        ):
            torch.compile(mod, backend="aot_eager", fullgraph=True)(x)

    def test_reuse_hash_fn_if_cond_no_guard(self):
        # [FUNCTION DOC] A condition on a module attribute inside the hash function must not
        # leak guards. After a first compile, flipping the attribute on both layers and re-
        # running must not trigger recompilation (frame count unchanged) and results still
        # equal eager.
        """if-condition in hash fn should not install guards on the module."""

        def hash_fn(mod, x):
            # [FUNCTION DOC] HELPER: Hash function returning one if the module's use_gelu flag
            # is set, else zero.
            if mod.use_gelu:
                return 1
            return 0

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def layer_fn(mod, x):
            # [FUNCTION DOC] HELPER: Region returning sine of x plus the module's weight.
            return x.sin() + mod.weight

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a use_gelu flag and a random parameter weight of
            # size 8.
            def __init__(self, use_gelu):
                # [FUNCTION DOC] HELPER: Stores the flag and creates the weight parameter.
                super().__init__()
                self.use_gelu = use_gelu
                self.weight = torch.nn.Parameter(torch.randn(8))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two layers (flag true and false) applied through
            # the region.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two layers.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer(True), Layer(False)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region to each layer in turn.
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        mod = Model()
        x = torch.randn(8)
        ref = mod(x)

        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(mod, backend=cnt)
        res = compiled(x)
        self.assertEqual(ref, res)

        # Flip use_gelu on both layers — if hash fn guards leaked, this
        # would cause a recompilation. With proper guard stripping it
        # should not recompile (the outer frame guard count stays the same).
        frame_count_before = cnt.frame_count
        mod.layers[0].use_gelu = False
        mod.layers[1].use_gelu = True
        ref2 = mod(x)
        res2 = compiled(x)
        self.assertEqual(ref2, res2)
        self.assertEqual(cnt.frame_count, frame_count_before)

    def test_reuse_hash_fn_side_effect_allowed(self):
        # [FUNCTION DOC] A region that increments a module counter (a side effect) normally
        # blocks reuse, but with a hash function the three layers still share a trace: one
        # speculation call.
        """Side effects (attribute mutation) should not block reuse with reuse_hash_fn."""

        def hash_fn(mod, x):
            # [FUNCTION DOC] HELPER: Hash function returning the constant zero.
            return 0

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def layer_fn(mod, x):
            # [FUNCTION DOC] HELPER: Region incrementing the module's call_count then returning
            # sine of x plus the weight.
            mod.call_count += 1
            return x.sin() + mod.weight

        class Layer(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a call counter and a random parameter weight of
            # size 8.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the counter to zero and creates the
                # weight parameter.
                super().__init__()
                self.call_count = 0
                self.weight = torch.nn.Parameter(torch.randn(8))

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a list of three Layer modules applied through
            # the region.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates three layers.
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the region to each layer in turn.
                for layer in self.layers:
                    x = layer_fn(layer, x)
                return x

        mod = Model()
        x = torch.randn(8)

        # Without reuse_hash_fn, the side effect (mod.call_count += 1)
        # would prevent reuse. With reuse_hash_fn, it should still reuse.
        with self._count_speculate_calls() as count:
            torch.compile(mod, backend="aot_eager", fullgraph=True)(x)

        self.assertEqual(count(), 1)

    def test_reuse_hash_fn_unsupported_output_raises(self):
        # [FUNCTION DOC] A region returning nested tuples with a hash function is not eligible
        # for reuse; compilation must raise RuntimeError saying the hash function was provided
        # but the subgraph is not eligible.
        """Nested output (tuple of tuple of tensors) should raise with reuse_hash_fn."""

        def hash_fn(x):
            # [FUNCTION DOC] HELPER: Hash function returning the constant zero.
            return 0

        @nested_compile_region(reuse_hash_fn=hash_fn)
        def gn(x):
            # [FUNCTION DOC] HELPER: Region returning a one-tuple holding a tuple of sine
            # and cosine.
            return ((x.sin(), x.cos()),)

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the region.
            return gn(x)

        x = torch.randn(8)
        with self.assertRaisesRegex(
            RuntimeError, "reuse_hash_fn was provided but the subgraph is not eligible"
        ):
            torch.compile(fn, backend="aot_eager", fullgraph=True)(x)


@skipIfTorchDynamo("Not a torch._dynamo test")
@unittest.skipIf(TEST_WITH_CROSSREF, "crossref does not support trace_autograd_ops")
class TestInvokeSubgraphTrainStepCapture(TestCase):
    # [CLASS DOC] Train-step capture tests with autograd op tracing enabled (backward called
    # inside the compiled function). Skipped under dynamo and under crossref mode.
    @torch._dynamo.config.patch(
        trace_autograd_ops=True,
        inline_single_use_invoke_subgraph=False,
    )
    def test_mark_non_differentiable_propagation(self):
        # [FUNCTION DOC] A region wrapping an autograd function that marks one output as non-
        # differentiable must propagate that status to the outer op so the auxiliary output does
        # not require grad. Runs a train step (loss backward and a chained backward) compiled
        # with aot_eager and asserts the returned auxiliary tensor does not require grad.
        """invoke_subgraph wrapping an autograd.Function that calls
        mark_non_differentiable should propagate the non-differentiable
        status to the outer InvokeSubgraphAutogradOp.  Without this,
        the outer autograd engine sets requires_grad=True on all outputs,
        causing AOT autograd to take the joint tracing path and hit
        'backward through graph a second time'.
        """

        class MyOp(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function returning sine and the max of its input; the
            # max is marked non-differentiable; backward returns grad_out times the cosine of
            # the saved input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes sine and max, marks the max non-
                # differentiable, saves x, and returns both.
                out = x.sin()
                aux = x.max()
                ctx.mark_non_differentiable(aux)
                ctx.save_for_backward(x)
                return out, aux

            @staticmethod
            def backward(ctx, grad_out, grad_aux):
                # [FUNCTION DOC] HELPER: Returns grad_out times cosine of the saved input
                # (grad_aux ignored).
                (x,) = ctx.saved_tensors
                return grad_out * x.cos()

        @nested_compile_region()
        def nested_fn(x):
            # [FUNCTION DOC] HELPER: Region applying the custom autograd function.
            return MyOp.apply(x)

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two bias-free linear layers; forward runs a full
            # train step inside the call.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 4 by 4 and a 4 by 1 bias-free linear layers.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)
                self.head = torch.nn.Linear(4, 1, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer and the region, detaches the
                # region output, computes a loss through the head, runs backward on the loss,
                # then backward on the region output with the detached gradient. Returns the
                # detached loss and the auxiliary output.
                y = self.linear(x)
                z, aux = nested_fn(y)
                z_det = z.detach().requires_grad_()
                loss = self.head(z_det).sum()
                loss.backward()
                z.backward(z_det.grad)
                return loss.detach(), aux

        model = Model()
        x = torch.randn(4, 4)
        compiled = torch.compile(model, backend="aot_eager", fullgraph=True)
        loss, aux = compiled(x)
        self.assertFalse(aux.requires_grad)

    @torch._dynamo.config.patch(
        trace_autograd_ops=True,
        enable_invoke_subgraph_regional_compile=True,
        inline_single_use_invoke_subgraph=False,
    )
    def test_nested_region_config_propagated_to_backward(self):
        # [FUNCTION DOC] With regional compile options set on the region and backward traced in
        # the forward function, a recording forward compiler captures a single graph that must
        # have two invoke_subgraph nodes (forward and backward), each with a nested region
        # configuration entry in its custom metadata.
        """When joint tracing creates backward invoke_subgraph nodes,
        the nested_region_config should be propagated so the regional
        compiler can compile them.
        """
        from torch._dynamo.backends.common import aot_autograd
        from torch._higher_order_ops.invoke_subgraph import (
            get_invoke_subgraph_compile_options,
            invoke_subgraph,
        )

        config = get_invoke_subgraph_compile_options(
            decompositions=torch._decomp.core_aten_decompositions()
        )

        @nested_compile_region(options=config)
        def nested_fn(x):
            # [FUNCTION DOC] HELPER: Region returning the sine.
            return x.sin()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two bias-free linear layers; forward runs a full
            # train step inside the call.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 4 by 4 and a 4 by 1 bias-free linear layers.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)
                self.head = torch.nn.Linear(4, 1, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer and the region, detaches the
                # region output, computes a loss through the head, runs backward on the loss,
                # then backward on the region output with the detached gradient. Returns the
                # detached loss.
                y = self.linear(x)
                z = nested_fn(y)
                z_det = z.detach().requires_grad_()
                loss = self.head(z_det).sum()
                loss.backward()
                z.backward(z_det.grad)
                return loss.detach()

        fw_graphs = []

        def recording_compiler(gm, example_inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Forward compiler that appends the graph module to the
            # enclosing list and returns it unchanged.
            fw_graphs.append(gm)
            return gm

        backend = aot_autograd(fw_compiler=recording_compiler)
        model = Model()
        x = torch.randn(4, 4)
        opt_model = torch.compile(model, backend=backend, fullgraph=True)
        opt_model(x)

        # The inference graph should have both fw and bw invoke_subgraph nodes,
        # each with nested_region_config so regional inductor can compile them.
        self.assertEqual(len(fw_graphs), 1)
        gm = fw_graphs[0]
        invoke_nodes = [
            n
            for n in gm.graph.nodes
            if n.op == "call_function" and n.target is invoke_subgraph
        ]
        # 2 invoke_subgraph nodes: fw and bw
        self.assertEqual(len(invoke_nodes), 2)
        for node in invoke_nodes:
            self.assertIn("custom", node.meta)
            self.assertIn("nested_region_config", node.meta["custom"])

    @torch._dynamo.config.patch(
        trace_autograd_ops=True,
        enable_invoke_subgraph_regional_compile=True,
        inline_single_use_invoke_subgraph=False,
    )
    def test_ac_invoke_subgraph_has_recomputed_blocks(self):
        # [FUNCTION DOC] With regional compile options and an activation checkpoint around
        # the region in a train step, the forward graph (compared to expected text, ignoring
        # comments and empty lines) must contain a recomputed copy of the partitioned
        # forward invoke_subgraph.
        """When activation checkpointing wraps invoke_subgraph in a
        train-step capture, the partitioned fw invoke_subgraph should be
        recomputed by the remat pass, producing _recomputed nodes.
        """
        from torch._higher_order_ops.invoke_subgraph import (
            get_invoke_subgraph_compile_options,
        )

        config = get_invoke_subgraph_compile_options(
            decompositions=torch._decomp.core_aten_decompositions()
        )

        @nested_compile_region(options=config)
        def nested_fn(x):
            # [FUNCTION DOC] HELPER: Region returning the sine.
            return x.sin()

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two bias-free linear layers; forward runs a full
            # train step with a checkpointed region inside the call.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a 4 by 4 and a 4 by 1 bias-free linear layers.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, bias=False)
                self.head = torch.nn.Linear(4, 1, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first layer and the checkpointed region,
                # detaches the output, computes a loss through the head, runs backward on the
                # loss, then backward on the region output with the detached gradient. Returns
                # the detached loss.
                y = self.linear(x)
                z = torch.utils.checkpoint.checkpoint(nested_fn, y, use_reentrant=False)
                z_det = z.detach().requires_grad_()
                loss = self.head(z_det).sum()
                loss.backward()
                z.backward(z_det.grad)
                return loss.detach()

        backend = AotEagerAndRecordGraphs()
        model = Model()
        x = torch.randn(4, 4)
        torch.compile(model, backend=backend, fullgraph=True)(x)

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 4]", arg1_1: "f32[4, 4]", arg2_1: "f32[1, 4]"):
        t: "f32[4, 4]" = torch.ops.aten.t.default(arg0_1)
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(arg1_1, t);  t = None
        partitioned_fw_subgraph_0_0 = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_2 = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0, 'partitioned_fw_subgraph_0_0', mm);  partitioned_fw_subgraph_0_0 = None
        getitem: "f32[4, 4]" = invoke_subgraph_2[0];  invoke_subgraph_2 = None
        detach_1: "f32[4, 4]" = torch.ops.aten.detach.default(getitem);  getitem = None
        t_1: "f32[4, 1]" = torch.ops.aten.t.default(arg2_1)
        mm_1: "f32[4, 1]" = torch.ops.aten.mm.default(detach_1, t_1)
        sum_1: "f32[]" = torch.ops.aten.sum.default(mm_1);  mm_1 = None
        ones_like: "f32[]" = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
        expand: "f32[4, 1]" = torch.ops.aten.expand.default(ones_like, [4, 1]);  ones_like = None
        t_2: "f32[1, 4]" = torch.ops.aten.t.default(expand)
        mm_2: "f32[1, 4]" = torch.ops.aten.mm.default(t_2, detach_1);  t_2 = None
        t_3: "f32[4, 1]" = torch.ops.aten.t.default(mm_2);  mm_2 = None
        t_4: "f32[1, 4]" = torch.ops.aten.t.default(t_1);  t_1 = None
        mm_3: "f32[4, 4]" = torch.ops.aten.mm.default(expand, t_4);  expand = t_4 = None
        t_5: "f32[1, 4]" = torch.ops.aten.t.default(t_3);  t_3 = None
        empty_like: "f32[4, 4]" = torch.ops.aten.empty_like.default(detach_1, pin_memory = False);  detach_1 = None
        copy: "f32[4, 4]" = torch.ops.aten.copy.default(empty_like, mm_3);  empty_like = mm_3 = None
        empty_like_1: "f32[1, 4]" = torch.ops.aten.empty_like.default(arg2_1, pin_memory = False);  arg2_1 = None
        copy_1: "f32[1, 4]" = torch.ops.aten.copy.default(empty_like_1, t_5);  empty_like_1 = t_5 = None
        partitioned_bw_subgraph_0_0 = self.partitioned_bw_subgraph_0_0
        partitioned_fw_subgraph_0_0_recomputed = self.partitioned_fw_subgraph_0_0
        invoke_subgraph_2_recomputed = torch.ops.higher_order.invoke_subgraph(partitioned_fw_subgraph_0_0_recomputed, 'partitioned_fw_subgraph_0_0', mm);  partitioned_fw_subgraph_0_0_recomputed = mm = None
        getitem_3_recomputed: "f32[4, 4]" = invoke_subgraph_2_recomputed[1];  invoke_subgraph_2_recomputed = None
        invoke_subgraph_4 = torch.ops.higher_order.invoke_subgraph(partitioned_bw_subgraph_0_0, 'partitioned_bw_subgraph_0_0', getitem_3_recomputed, copy);  partitioned_bw_subgraph_0_0 = getitem_3_recomputed = copy = None
        getitem_1: "f32[4, 4]" = invoke_subgraph_4[0];  invoke_subgraph_4 = None
        t_6: "f32[4, 4]" = torch.ops.aten.t.default(getitem_1);  getitem_1 = None
        mm_4: "f32[4, 4]" = torch.ops.aten.mm.default(t_6, arg1_1);  t_6 = arg1_1 = None
        t_7: "f32[4, 4]" = torch.ops.aten.t.default(mm_4);  mm_4 = None
        t_8: "f32[4, 4]" = torch.ops.aten.t.default(t_7);  t_7 = None
        empty_like_2: "f32[4, 4]" = torch.ops.aten.empty_like.default(arg0_1, pin_memory = False);  arg0_1 = None
        copy_2: "f32[4, 4]" = torch.ops.aten.copy.default(empty_like_2, t_8);  empty_like_2 = t_8 = None
        detach_2: "f32[]" = torch.ops.aten.detach.default(sum_1);  sum_1 = None
        return (detach_2, copy_2, copy_1)
    class partitioned_fw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[4, 4]"):
            sin: "f32[4, 4]" = torch.ops.aten.sin.default(primals_0)
            return (sin, primals_0)
    class partitioned_bw_subgraph_0_0(torch.nn.Module):
        def forward(self, primals_0: "f32[4, 4]", tangents_0: "f32[4, 4]"):
            cos: "f32[4, 4]" = torch.ops.aten.cos.default(primals_0);  primals_0 = None
            mul: "f32[4, 4]" = torch.ops.aten.mul.Tensor(tangents_0, cos);  tangents_0 = cos = None
            return (mul,)
""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )


if __name__ == "__main__":
    run_tests()
