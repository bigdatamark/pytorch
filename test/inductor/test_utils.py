# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for assorted inductor utilities: subprocess environment construction,
# symbolic substitution and printing helpers, flop counting on fx nodes, device tflops lookup,
# template loading with UTF-8 handling, runtime estimation units, FP4 dtype support, the
# accumulator type mapping, FakeTensorUpdater incremental metadata updates (including nodes marked
# as inductor lowering functions), and Triton availability detection. Import-time side effect: the
# first test class is expanded per available device type by the device-type test instantiation
# helper (with XPU allowed) and written into this module's namespace.

import builtins
import importlib.util
import os
import sys
import tempfile
import unittest
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest import mock

from sympy import I, Max, Min, Symbol, sympify

import torch
from torch._dynamo.device_interface import DeviceInterface
from torch._dynamo.exc import TritonUnavailableError
from torch._dynamo.testing import AotEagerAndRecordGraphs
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config as inductor_config
from torch._inductor.compile_fx import _get_subgraph_names
from torch._inductor.fx_utils import (
    _is_fake_tensor_same,
    count_flops_fx,
    countable_fx,
    FakeTensorUpdater,
    get_fake,
)
from torch._inductor.utils import (
    get_device_tflops,
    load_template,
    python_subprocess_env,
    sympy_str,
    sympy_subs,
)
from torch._inductor.virtualized import V
from torch.fx.experimental.proxy_tensor import make_fx
from torch.ops import aten
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TestCase,
    xfailIfNoAcceleratorTriton,
)
from torch.utils import _triton as triton_utils
from torch.utils._sympy.functions import Identity


class TestUtils(TestCase):
    # [CLASS DOC] Device-generic tests of small utilities. Most tests are device independent; the
    # tflops test needs CUDA and a Triton-capable accelerator.
    def test_python_subprocess_env_prioritizes_loaded_torch(self):
        # [FUNCTION DOC] Puts a temporary directory at the front of the module search path and
        # builds the subprocess environment. Expects the first entry of its path variable to be
        # the root directory containing the loaded torch package rather than the shadowing entry.
        torch_package_root = os.path.dirname(
            os.path.dirname(os.path.abspath(torch.__file__))
        )
        with tempfile.TemporaryDirectory() as shadow_path:
            with mock.patch.object(sys, "path", [shadow_path, *sys.path]):
                env = python_subprocess_env()
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep)[0], torch_package_root)

    def test_python_subprocess_env_respects_override(self):
        # [FUNCTION DOC] With the custom Python path environment variable set to a fixed string,
        # builds the subprocess environment and expects its path variable to equal that string
        # exactly.
        with mock.patch.dict(
            os.environ, {"TORCH_CUSTOM_PYTHONPATH": "custom_python_path"}
        ):
            env = python_subprocess_env()
        self.assertEqual(env["PYTHONPATH"], "custom_python_path")

    def test_zip_schema(self):
        # [FUNCTION DOC] Registers a no-op custom operator in the test library namespace that
        # mutates its tensor argument, then walks the zipped schema arguments and values once with
        # the tensor given by keyword and once positionally. Expects the argument named x to be
        # seen both times. Side effect: registers a custom operator.
        def foo(x: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: No-op function taking a tensor, used as the body of the
            # custom operator.
            pass

        result = torch.library.custom_op("mylib::foo", foo, mutates_args={"x"})
        schema = result._opoverload._schema
        g = torch.tensor([11, 2])
        found = False
        for arg, val in torch._library.utils.zip_schema(schema, [], {"x": g}):
            if arg.name == "x":
                found = True

        self.assertTrue(found)

        found = False
        for arg, val in torch._library.utils.zip_schema(schema, [g], {}):
            if arg.name == "x":
                found = True
        self.assertTrue(found)

    def testSympySubs(self):
        # [FUNCTION DOC] Checks the symbol substitution helper: integer and non-negative
        # assumptions are preserved after substituting a symbol name; a replacement whose
        # assumptions do not match the target symbol leaves the expression unchanged, while
        # matching assumptions replace it; a string used as the key raises an assertion error; and
        # a compound expression can be replaced by a symbol, with the unknown assumptions staying
        # unknown.
        # integer and nonnegetaive attributes are preserved.
        expr = Symbol("x")
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

        expr = Symbol("x", integer=True, nonnegative=False)
        result = sympy_subs(expr, {expr: "y"})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, True)
        self.assertEqual(result.is_nonnegative, False)

        # invalid replacement.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x"): Symbol("y")})
        self.assertEqual(result.name, "x")

        # valid replacement since properties match.
        expr = Symbol("x", integer=True)
        result = sympy_subs(expr, {Symbol("x", integer=True): Symbol("y")})
        self.assertEqual(result.name, "y")

        # invalid replacement.
        expr = Symbol("x", integer=None)
        result = sympy_subs(expr, {Symbol("x", integer=False): Symbol("y")})
        self.assertEqual(result.name, "x")

        # replaced can't be string
        self.assertRaises(AssertionError, sympy_subs, expr, {"x": "y"})

        # replaced can be an expression
        expr = Symbol("x")
        expr = abs(expr)
        self.assertEqual(expr.is_integer, None)
        self.assertEqual(expr.is_nonnegative, None)
        # replace abs(x) with y
        # propagate abs(x) sympy properties.
        result = sympy_subs(expr, {expr: Symbol("y")})
        self.assertEqual(result.name, "y")
        self.assertEqual(result.is_integer, None)
        self.assertEqual(result.is_nonnegative, None)

    def testSympySubsIdentityNonComparable(self):
        # [FUNCTION DOC] Substitutes the imaginary unit for a symbol inside a minimum and maximum
        # wrapping an identity expression, and expects the result to contain the imaginary unit.
        q0 = Symbol("q0", integer=True, nonnegative=True)
        expr = Min(2, Max(0, Identity(q0)))
        result = sympy_subs(expr, {q0: I})
        self.assertTrue(result.has(I))

    def testIdentityComparisonNoRecursion(self):
        # [FUNCTION DOC] Compares identity expressions of numeric constants with zero in both
        # directions; expects the ordering relations to evaluate without infinite recursion and
        # give true or false as appropriate.
        self.assertTrue(Identity(sympify("0")) >= 0)
        self.assertFalse(Identity(sympify("-6")) >= 0)
        self.assertTrue(0 >= Identity(sympify("-6")))

    def testIdentityComparableNumbersInMinMax(self):
        # [FUNCTION DOC] Builds an identity expression of a negative integer and expects it to be
        # reported as a number and as comparable, and a maximum with zero to simplify to zero.
        expr = Identity(sympify("-6"))
        self.assertTrue(expr.is_number)
        self.assertTrue(expr.is_comparable)
        self.assertEqual(Max(0, expr), 0)

    def testIdentityRationalComparisonNoRecursion(self):
        # [FUNCTION DOC] Builds an identity expression of a positive rational number and expects
        # it to compare at least zero and a maximum with zero to retain the expression.
        expr = Identity(sympify("1/7"))
        self.assertTrue(expr >= 0)
        self.assertTrue(Max(0, expr).has(expr))

    def test_sympy_str(self):
        # [FUNCTION DOC] Checks the printing helper on parsed expressions: sums, products with
        # parentheses, negation, subtraction and addition of a negation all produce the expected
        # strings, with products sorted after plain terms.
        self.assertEqual(sympy_str(sympify("a+b+c")), "a + b + c")
        self.assertEqual(sympy_str(sympify("a*b+c")), "c + a * b")
        self.assertEqual(sympy_str(sympify("a+b*(c+d)")), "a + b * (c + d)")
        self.assertEqual(sympy_str(sympify("(a+b)*(c+d)")), "(a + b) * (c + d)")
        self.assertEqual(sympy_str(sympify("-a")), "-a")
        self.assertEqual(sympy_str(sympify("a-b")), "a - b")
        self.assertEqual(sympy_str(sympify("a+-b")), "a - b")

    def test_flops_fx(self):
        # [FUNCTION DOC] Under a fake tensor mode, creates fx nodes (one targeting an operator
        # packet and one targeting a specific overload) for several operations and checks flop
        # countability. Matrix multiplication variants and convolutions must be countable with a
        # non-null flop count; pointwise add and mul must not be countable. Failure messages are
        # callables.
        def create_fx_node(
            aten, op_overload: torch._ops.OpOverload, args, kwargs
        ) -> tuple[torch.fx.Node, torch.fx.Node]:
            # [FUNCTION DOC] HELPER: Builds two call-function nodes, each in its own fresh graph,
            # with the same arguments and keyword arguments; one targets the given packet and the
            # other the given overload. Returns both nodes.
            node1 = torch.fx.Node(
                graph=torch.fx.Graph(),
                name="",
                op="call_function",
                target=aten,
                args=args,
                kwargs=kwargs,
            )
            # name: str = aten.overloads()[0]
            # if aten == torch.ops.aten.addmm:
            #     name = "default"
            # print(aten)
            # print(aten.overloads())
            # print(name)
            # op_overload: torch._ops.OpOverload = getattr(aten, name)
            node2 = torch.fx.Node(
                graph=torch.fx.Graph(),
                name="",
                op="call_function",
                target=op_overload,
                args=args,
                kwargs=kwargs,
            )
            return node1, node2

        with V.set_fake_mode(
            torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        ):
            trues = [
                (
                    torch.ops.aten.addmm,
                    torch.ops.aten.addmm.default,
                    (torch.Tensor(4, 4), torch.Tensor(4, 5), torch.Tensor(5, 4)),
                    {},
                ),
                (
                    torch.ops.aten.bmm,
                    torch.ops.aten.bmm.default,
                    (torch.Tensor(10, 4, 5), torch.Tensor(10, 5, 4)),
                    {},
                ),
                (
                    torch.ops.aten.mm,
                    torch.ops.aten.mm.default,
                    (torch.Tensor(2, 3), torch.Tensor(3, 2)),
                    {},
                ),
                (
                    torch.ops.aten.convolution,
                    torch.ops.aten.convolution.default,
                    (
                        torch.Tensor(2, 2, 3),
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                    ),
                    {},
                ),
                (
                    torch.ops.aten._convolution,
                    torch.ops.aten._convolution.deprecated,
                    (
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2, 2, 2),
                        torch.Tensor(2),
                        (1,),
                        (0,),
                        (1,),
                        True,
                        (0,),
                        1,
                        False,
                        True,
                        False,
                    ),
                    {},
                ),
            ]
            # we don't support pointwise ops
            falses = [
                (
                    torch.ops.aten.add,
                    torch.ops.aten.add.Tensor,
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
                (
                    torch.ops.aten.mul,
                    torch.ops.aten.mul.Tensor,
                    (torch.Tensor(1, 2, 3), torch.Tensor(1, 2, 3)),
                    {},
                ),
            ]
            for t, t2, args, kwargs in trues:
                fx_node_1, fx_node_2 = create_fx_node(t, t2, args, kwargs)
                self.assertTrue(
                    countable_fx(fx_node_1),
                    lambda msg: f"{msg}\nExpected true {t}: {fx_node_1}",
                )
                self.assertTrue(
                    countable_fx(fx_node_2),
                    lambda msg: f"{msg}\nExpected true {t}: {fx_node_2}",
                )
                self.assertNotEqual(count_flops_fx(fx_node_1), None)
                self.assertNotEqual(count_flops_fx(fx_node_2), None)
            for f, f2, args, kwargs in falses:
                fx_node_1, fx_node_2 = create_fx_node(f, f2, args, kwargs)
                self.assertFalse(
                    countable_fx(fx_node_1),
                    lambda msg: f"{msg}\nExpected false {f}: {fx_node_1}",
                )
                self.assertFalse(
                    countable_fx(fx_node_2),
                    lambda msg: f"{msg}\nExpected false {f}: {fx_node_2}",
                )

    def test_flops_fx_higher_order_op(self):
        # [FUNCTION DOC] Builds a graph with three placeholders of meta-device bfloat16 tensors
        # and one call to the flex attention higher-order operator whose result metadata is a
        # tuple of an output and two float32 tensors. Expects the node to be countable and the
        # counted flops to equal the registered formula for that operator, i.e. no actual
        # invocation of the operator.
        """count_flops_fx must use the registered formula for HOP targets
        rather than invoking the HOP. flex_attention.__call__ requires a
        Dynamo/proxy tracing context (TransformGetItemToIndex) and raises
        TypeError when invoked on bare (fake) tensors.
        """
        from torch.utils.flop_counter import flop_registry

        flex_attention = torch.ops.higher_order.flex_attention
        self.assertIn(flex_attention, flop_registry)

        q_shape = (2, 16, 1024, 64)
        k_shape = (2, 4, 1024, 64)
        v_shape = (2, 4, 1024, 64)

        with V.set_fake_mode(
            torch._subclasses.FakeTensorMode(allow_non_fake_inputs=True)
        ):
            graph = torch.fx.Graph()
            q = graph.placeholder("q")
            k = graph.placeholder("k")
            v = graph.placeholder("v")
            q.meta["val"] = torch.randn(*q_shape, device="meta", dtype=torch.bfloat16)
            k.meta["val"] = torch.randn(*k_shape, device="meta", dtype=torch.bfloat16)
            v.meta["val"] = torch.randn(*v_shape, device="meta", dtype=torch.bfloat16)
            node = graph.call_function(flex_attention, args=(q, k, v))
            node.meta["val"] = (
                torch.randn(*q_shape, device="meta", dtype=torch.bfloat16),
                torch.randn(
                    q_shape[0],
                    q_shape[1],
                    q_shape[2],
                    device="meta",
                    dtype=torch.float32,
                ),
                torch.randn(
                    q_shape[0],
                    q_shape[1],
                    q_shape[2],
                    device="meta",
                    dtype=torch.float32,
                ),
            )

            self.assertTrue(countable_fx(node))
            flops = count_flops_fx(node)
            expected = flop_registry[flex_attention](
                q.meta["val"], k.meta["val"], v.meta["val"], out_val=node.meta["val"]
            )
            self.assertEqual(flops, expected)

    @xfailIfNoAcceleratorTriton
    @unittest.skipIf(not torch.cuda.is_available(), "skip if no device")
    @dtypes(torch.float16, torch.bfloat16, torch.float32)
    def test_get_device_tflops(self, dtype):
        # [FUNCTION DOC] Calls the device tflops query for the parametrized dtype and expects a
        # float. Expected to fail without an accelerator with Triton and skipped without CUDA.
        ret = get_device_tflops(dtype)
        self.assertTrue(type(ret) is float)


instantiate_device_type_tests(TestUtils, globals(), allow_xpu=True)


class TestLoadTemplate(TestCase):
    # [CLASS DOC] Tests for template loading.
    def test_load_template_uses_utf8(self):
        # [FUNCTION DOC] Replaces the built-in file opener with a wrapper that forces ASCII
        # decoding when no encoding is given, writes a template file containing a non-ASCII dash,
        # loads it with the template loader, and expects the dash in the result, proving UTF-8 is
        # used explicitly.
        # load_template must decode templates as UTF-8 regardless of the ambient
        # locale. On a host whose default encoding is ascii, reading a template
        # that contains a non-ascii byte otherwise raises UnicodeDecodeError,
        # producing a host-dependent (flaky) compile failure.
        real_open = builtins.open

        def ascii_default_open(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Opens a file like the real opener, but when no encoding was
            # passed by keyword or as the fourth positional argument, supplies ASCII to emulate an
            # ASCII locale.
            # Emulate an ascii-locale host: open() with no explicit encoding
            # decodes as ascii (open's 4th positional arg is encoding).
            if kwargs.get("encoding") is None and (len(args) < 4 or args[3] is None):
                kwargs["encoding"] = "ascii"
            return real_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "t.py.jinja").write_text("# unicode \u2014\n", encoding="utf-8")
            with mock.patch("builtins.open", ascii_default_open):
                content = load_template("t", Path(d))
        self.assertIn("\u2014", content)

    def test_load_template_invalid_utf8_names_the_file(self):
        # [FUNCTION DOC] Writes a template containing bytes that are invalid UTF-8 and expects the
        # loader to raise a value error whose message contains the template file name.
        # A template that is genuinely not valid UTF-8 (e.g. saved in a non-UTF-8
        # codepage) must raise an error that names the offending file, not an
        # opaque UnicodeDecodeError that hides which template is bad.
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "bad.py.jinja").write_bytes(b"# not utf-8: \x97\n")
            with self.assertRaises(ValueError) as cm:
                load_template("bad", Path(d))
        self.assertIn("bad.py.jinja", str(cm.exception))


class TestRuntimeEstimation(TestCase):
    # [CLASS DOC] Tests for the runtime estimation utilities.
    def test_get_compute_time_units(self):
        # [FUNCTION DOC] With the device tflops query patched to return a fixed value, computes
        # the compute time of a 64 by 64 float32 matrix multiplication and compares it with the
        # value derived by hand: multiply-accumulate count divided by 0.75 of the peak in
        # operations per second with tera scaling, times a billion. Guards the unit conversion
        # factor.
        """TFLOPS-to-FLOPS/s conversion must use 1e12, not 1e15."""
        from unittest.mock import patch

        from torch.utils._runtime_estimation import get_compute_time

        M, K, N = 64, 64, 64
        known_tflops = 1000.0
        a = torch.randn(M, K)
        b = torch.randn(K, N)
        out = torch.mm(a, b)

        with patch(
            "torch.utils._runtime_estimation.get_device_tflops",
            return_value=known_tflops,
        ):
            result_ns = get_compute_time(
                torch.ops.aten.mm, (a, b), {}, out, {torch.float32}
            )

        # mm flops = 2*M*K*N, divided by 2 for MACs, then time = macs / (0.75 * peak) * 1e9
        expected_macs = 2 * M * K * N / 2
        expected_ns = (expected_macs / (0.75 * known_tflops * 1e12)) * 1e9
        self.assertAlmostEqual(result_ns, expected_ns)


class TestFP4Support(TestCase):
    # [CLASS DOC] Tests for FP4 (packed 4-bit float) dtype support in inductor and test helpers.
    """Tests for FP4 (float4_e2m1fn_x2) infrastructure support."""

    @unittest.skipIf(
        not torch.cuda.is_available()
        or importlib.util.find_spec("cutlass_api") is None,
        "requires CUDA and cutlass_api",
    )
    def test_ensure_fp4_dtype_registered(self):
        # [FUNCTION DOC] Skipped without CUDA or the cutlass_api package. Calls the registration
        # helper and expects the cutlass type conversion to map the FP4 dtype to the cutlass FP4
        # type and float32 to the cutlass float32 type.
        """_ensure_fp4_dtype_registered should patch cutlass_api for FP4."""
        from torch._inductor.utils import _ensure_fp4_dtype_registered

        _ensure_fp4_dtype_registered()
        import cutlass
        import cutlass_api.utils

        result = cutlass_api.utils.cutlass_type_from_torch_type(torch.float4_e2m1fn_x2)
        self.assertEqual(result, cutlass.Float4E2M1FN)

        result_fp32 = cutlass_api.utils.cutlass_type_from_torch_type(torch.float32)
        self.assertEqual(result_fp32, cutlass.Float32)

    def test_rand_strided_fp4(self):
        # [FUNCTION DOC] Creates a strided CPU tensor of the packed FP4 dtype with shape 4 by 8
        # and strides 8 and 1, and expects the dtype, shape and strides to be as requested.
        """rand_strided should produce valid FP4 tensors."""
        from torch._dynamo.testing import rand_strided

        t = rand_strided((4, 8), (8, 1), dtype=torch.float4_e2m1fn_x2, device="cpu")
        self.assertEqual(t.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(t.shape, (4, 8))
        self.assertEqual(t.stride(), (8, 1))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_rand_strided_fp4_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Creates a strided tensor of the packed FP4 dtype on
        # the CUDA device with shape 16 by 32 and expects the dtype, shape and CUDA placement.
        from torch._dynamo.testing import rand_strided

        t = rand_strided((16, 32), (32, 1), dtype=torch.float4_e2m1fn_x2, device="cuda")
        self.assertEqual(t.dtype, torch.float4_e2m1fn_x2)
        self.assertEqual(t.shape, (16, 32))
        self.assertTrue(t.is_cuda)


class TestTritonTypeMapping(TestCase):
    # [CLASS DOC] Tests for the accumulator type mapping used by Triton matmul templates.
    """Tests for acc_type() dtype conversions."""

    def test_acc_type(self):
        # [FUNCTION DOC] Iterates over named cases (half, bfloat16, float32 and four float8
        # variants) and expects the accumulator type function to return the string for 32-bit
        # float in all of them, reporting each case as a sub-test.
        from torch._inductor.kernel.mm_common import acc_type

        cases = {
            "half promotes to float32": (torch.float16, "tl.float32"),
            "bfloat16 promotes to float32": (torch.bfloat16, "tl.float32"),
            "float32 passthrough": (torch.float32, "tl.float32"),
            "fp8 e4m3fn promotes to float32": (torch.float8_e4m3fn, "tl.float32"),
            "fp8 e5m2 promotes to float32": (torch.float8_e5m2, "tl.float32"),
            "fp8 e4m3fnuz promotes to float32": (torch.float8_e4m3fnuz, "tl.float32"),
            "fp8 e5m2fnuz promotes to float32": (torch.float8_e5m2fnuz, "tl.float32"),
        }
        for desc, (dtype, expected) in cases.items():
            with self.subTest(desc=desc, dtype=dtype):
                self.assertEqual(acc_type(dtype), expected)


class TestFakeTensorUpdater(TestCase):
    # [CLASS DOC] Tests for the incremental fake tensor metadata updater on fx graphs. Static and
    # class-level helpers build graphs (compiled, or hand built with inductor-lowering-marked
    # nodes) and shared routines apply graph edits.
    @staticmethod
    def _get_faketensormode(
        graph: torch.fx.GraphModule,
    ) -> torch._subclasses.FakeTensorMode:
        # [FUNCTION DOC] HELPER: Static. Returns the fake mode found from the fake value of the
        # graph's first node, or a new fake tensor mode when none is detected.
        return (
            detect_fake_mode(get_fake(next(iter(graph.graph.nodes)), graph))
            or torch._subclasses.FakeTensorMode()
        )

    @staticmethod
    def _get_graph(
        fn: Callable[..., torch.Tensor], *args: torch.Tensor
    ) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Static. Compiles the function with a recording AOT-eager backend
        # in full-graph mode, runs it on the given tensors and returns the first recorded forward
        # graph module.
        backend = AotEagerAndRecordGraphs()
        torch.compile(backend=backend, fullgraph=True)(fn)(*args)
        return backend.fw_graphs[0]

    @staticmethod
    def _get_call_function_nodes(
        graph: torch.fx.GraphModule,
    ) -> Iterator[tuple[torch.fx.GraphModule, torch.fx.Node]]:
        # [FUNCTION DOC] HELPER: Static generator. Yields pairs of (graph module, node) for every
        # call-function node, first recursing into subgraphs of the module and then the module's
        # own nodes. Such nodes are suitable edit targets since the updater focuses on callables.
        """Recursively yields all call_function nodes in a GraphModule.  These nodes are
        ideal to apply transformations to, since callables are the focus of
        FakeTensorUpdater."""
        for sn in _get_subgraph_names(graph):
            yield from TestFakeTensorUpdater._get_call_function_nodes(
                getattr(graph, sn)
            )

        yield from ((graph, n) for n in graph.graph.nodes if n.op == "call_function")

    @staticmethod
    def _make_inductor_lowering_function(
        *,
        output_metadata_ignores_input_storage: bool = False,
        output_metadata_is_input: int | str | None = None,
        output_metadata_fn: Callable[..., object] | None = None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        # [FUNCTION DOC] HELPER: Static. Returns a stand-in lowering function that raises an
        # assertion error if ever called. It is tagged with marker attributes for being an
        # inductor lowering function and carries the three keyword-only options (ignore input
        # storage, name or index of the input whose metadata is passed through, and a metadata
        # function) as further attributes.
        def lowering_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Always raises an assertion error; the updater must never run
            # it.
            raise AssertionError("lowering_fn should not run under FakeTensorUpdater")

        lowering_fn._inductor_lowering_function = True  # type: ignore[attr-defined]
        lowering_fn._inductor_lowering_output_metadata_ignores_input_storage = (  # type: ignore[attr-defined]
            output_metadata_ignores_input_storage
        )
        lowering_fn._inductor_lowering_output_metadata_is_input = (  # type: ignore[attr-defined]
            output_metadata_is_input
        )
        lowering_fn._inductor_lowering_output_metadata_fn = output_metadata_fn  # type: ignore[attr-defined]
        return lowering_fn

    @classmethod
    def _build_graph_with_inductor_lowering_node(
        cls,
    ) -> tuple[
        torch.fx.GraphModule,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
        torch.fx.Node,
    ]:
        # [FUNCTION DOC] HELPER: Class-level. Builds a graph with two placeholders, a negation of
        # the first, and an unmarked-option lowering node applied to the negation, used as the
        # output. Returns the graph module followed by the placeholders, the negation node and the
        # lowering node.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(cls._make_inductor_lowering_function(), (neg,))
        graph.output(lowered)
        return torch.fx.GraphModule({}, graph), x, y, neg, lowered

    def _add_delete_nodes_test(self, graph: torch.fx.GraphModule) -> None:
        # [FUNCTION DOC] HELPER: For each call-function node in the graph (subgraphs included),
        # inserts a clone after it, with channels-last format when the node's value is
        # four-dimensional (or all tuple elements are), reroutes all its other users to the clone
        # and runs an incremental update in the graph's fake mode. Expects at least as many
        # updates as users changed plus one. Then removes the clone, restores the users and
        # updates again, expecting one fewer update than before.
        updater = FakeTensorUpdater(graph)
        fake_mode = self._get_faketensormode(graph)

        for gm, fn in self._get_call_function_nodes(graph):
            fake_outputs = get_fake(fn, gm)
            self.assertIsNot(fake_outputs, fn, msg="No fake outputs for node!")

            # Since we're testing changes in subgraphs, we've explicitly disallowed
            # changes other than striding to anything input to a subgraph.  With
            # cascading changes, cloning is the most straightforward approach to ensure
            # that constraint is met.
            clone_function = (
                torch._foreach_clone if isinstance(fake_outputs, tuple) else torch.clone
            )
            with gm.graph.inserting_after(fn):
                # When tests use input tensors with dim == 4, shuffle striding order to
                # test that updating subgraphs handles striding changes.
                should_shuffle_strides = "val" in fn.meta and (
                    (
                        isinstance(fn.meta["val"], tuple)
                        and all(len(v.size()) == 4 for v in fn.meta["val"])
                    )
                    or len(fn.meta["val"].size()) == 4
                )
                if should_shuffle_strides:
                    cloned_node = gm.graph.call_function(
                        clone_function, (fn,), {"memory_format": torch.channels_last}
                    )
                else:
                    cloned_node = gm.graph.call_function(clone_function, (fn,))
            nodes_modified = fn.replace_all_uses_with(
                cloned_node, lambda n: n != cloned_node
            )

            with V.set_fake_mode(fake_mode):
                clone_num_updated = updater.incremental_update()

            # At a minimum, we have to update the newly inserted node and all the nodes
            # which had an input replaced.  There may be more nodes modified in
            # subgraphs, so we can't do a strict equality assertion here.
            self.assertGreaterEqual(clone_num_updated, len(nodes_modified) + 1)

            cloned_node.replace_all_uses_with(fn)
            gm.graph.erase_node(cloned_node)
            with V.set_fake_mode(fake_mode):
                erase_num_updated = updater.incremental_update()

            # Deleting the node should update the same number of nodes as previously,
            # excluding the reshaped node itself.
            self.assertEqual(clone_num_updated - 1, erase_num_updated)

    def test_hop_implicit_subgraph_inputs(self):
        # [FUNCTION DOC] Compiles a function using a conditional on the sum of a four-dimensional
        # tensor with sine and cosine branches, so the graph has subgraphs with implicit inputs,
        # and runs the add-and-delete node routine.
        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns sine of x if the sum of x is negative, else cosine,
            # via the conditional higher-order operator.
            return torch.cond(torch.sum(x) < 0, torch.sin, torch.cos, (x,))

        # Use 4-D tensor so that we can test re-striding with channels_last.
        a = torch.randn((8, 4, 2, 1))
        graph = self._get_graph(fn, a)
        self._add_delete_nodes_test(graph)

    def test_hop_subgraph_inputs(self):
        # [FUNCTION DOC] Compiles a function that calls nested compile regions with two integer
        # four-dimensional inputs and runs the add-and-delete node routine on the resulting graph.
        @torch.compiler.nested_compile_region
        def nested_section_inner(a: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Nested compile region returning sine of the input.
            return torch.sin(a)

        @torch.compiler.nested_compile_region
        def nested_section_outer(
            a: torch.Tensor, b: torch.Tensor
        ) -> tuple[torch.Tensor, ...]:
            # [FUNCTION DOC] HELPER: Nested compile region applying the inner region twice to the
            # first argument and once to the second, returning the pair.
            return nested_section_inner(nested_section_inner(a)), nested_section_inner(
                b
            )

        def fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Calls the outer region and returns the sum of its two
            # results.
            x, y = nested_section_outer(a, b)
            return x + y

        # Use 4-D tensor so that we can test re-striding with channels_last.
        a = torch.randint(0, (1 << 16), (8, 4, 2, 1), dtype=torch.int32)
        b = torch.randint(0, (1 << 16), (8, 4, 2, 1), dtype=torch.int32)
        graph = self._get_graph(fn, a, b)
        self._add_delete_nodes_test(graph)

    def test_flex_attention_backward_updates_subgraph_inputs(self):
        # [FUNCTION DOC] Under fake tensor mode, builds fake tensors and three fake-traced
        # subgraphs (score, joint and mask functions), then traces a call to the flex attention
        # backward higher-order operator. Swaps the score-buffer and mask-buffer tuple arguments
        # (positions 12 and 13) for permuted, differently strided versions and runs an incremental
        # update. Expects the last placeholder of each subgraph to carry the very same metadata
        # object as the new buffer placeholder.
        with torch._subclasses.FakeTensorMode() as mode:
            query = mode.from_tensor(torch.randn(2, 2, 4, 8))
            logsumexp = mode.from_tensor(torch.randn(2, 2, 4))
            score = query.new_zeros(())
            index = query.new_zeros((), dtype=torch.int)
            grad = query.new_zeros(())
            score_buffer = mode.from_tensor(torch.randn(2, 3, 4, 5).contiguous())
            restrided_score_buffer = mode.from_tensor(
                torch.randn(2, 3, 5, 4).permute(0, 1, 3, 2)
            )
            mask_buffer = mode.from_tensor(torch.randn(2, 3, 4, 5))
            restrided_mask_buffer = mode.from_tensor(
                torch.randn(2, 3, 5, 4).permute(0, 1, 3, 2)
            )

            def fw(score, b, h, q_idx, kv_idx, score_buffer):
                # [FUNCTION DOC] HELPER: Score modification function returning the score plus a
                # buffer element.
                return score + score_buffer[0, 0, 0, 0]

            def joint(score, b, h, q_idx, kv_idx, grad, score_buffer):
                # [FUNCTION DOC] HELPER: Joint function returning the gradient, four Nones and the
                # buffer.
                return grad, None, None, None, None, score_buffer

            def mask(b, h, q_idx, kv_idx, mask_buffer):
                # [FUNCTION DOC] HELPER: Mask function returning whether a buffer element is
                # positive.
                return mask_buffer[0, 0, 0, 0] > 0

            fw_subgraph = make_fx(fw, tracing_mode="fake")(
                score, index, index, index, index, score_buffer
            )
            joint_subgraph = make_fx(joint, tracing_mode="fake")(
                score, index, index, index, index, grad, score_buffer
            )
            mask_subgraph = make_fx(mask, tracing_mode="fake")(
                index, index, index, index, mask_buffer
            )

            def flex_backward(
                query,
                logsumexp,
                score_buffer,
                restrided_score_buffer,
                mask_buffer,
                restrided_mask_buffer,
            ):
                # [FUNCTION DOC] HELPER: Calls the flex attention backward higher-order operator
                # with the query in several positions, the log-sum-exp, the three subgraphs, scale
                # one, empty options and the two buffer tuples.
                return torch.ops.higher_order.flex_attention_backward(
                    query,
                    query,
                    query,
                    query,
                    logsumexp,
                    query,
                    None,
                    fw_subgraph,
                    joint_subgraph,
                    (mask_subgraph,),
                    1.0,
                    {},
                    (score_buffer,),
                    (mask_buffer,),
                )

            gm = make_fx(flex_backward, tracing_mode="fake")(
                query,
                logsumexp,
                score_buffer,
                restrided_score_buffer,
                mask_buffer,
                restrided_mask_buffer,
            )
            flex_backward_node = next(
                node
                for node in gm.graph.nodes
                if node.target is torch.ops.higher_order.flex_attention_backward
            )
            placeholders = {
                node.target: node for node in gm.graph.find_nodes(op="placeholder")
            }
            score_buffer_node = placeholders["score_buffer_1"]
            restrided_score_buffer_node = placeholders["restrided_score_buffer_1"]
            mask_buffer_node = placeholders["mask_buffer_1"]
            restrided_mask_buffer_node = placeholders["restrided_mask_buffer_1"]
            fw_subgraph = get_fake(flex_backward_node.args[7], gm)
            joint_subgraph = get_fake(flex_backward_node.args[8], gm)
            mask_subgraph = get_fake(flex_backward_node.args[9][-1], gm)
            fw_placeholders = list(fw_subgraph.graph.find_nodes(op="placeholder"))
            joint_placeholders = list(joint_subgraph.graph.find_nodes(op="placeholder"))
            mask_placeholders = list(mask_subgraph.graph.find_nodes(op="placeholder"))

            updater = FakeTensorUpdater(gm)
            flex_backward_args = list(flex_backward_node.args)
            self.assertEqual(flex_backward_args[12], (score_buffer_node,))
            self.assertEqual(flex_backward_args[13], (mask_buffer_node,))
            flex_backward_args[12] = (restrided_score_buffer_node,)
            flex_backward_args[13] = (restrided_mask_buffer_node,)
            flex_backward_node.args = tuple(flex_backward_args)

            with V.set_fake_mode(mode):
                updater.incremental_update()

        self.assertIs(
            fw_placeholders[-1].meta["val"], restrided_score_buffer_node.meta["val"]
        )
        self.assertIs(
            joint_placeholders[-1].meta["val"], restrided_score_buffer_node.meta["val"]
        )
        self.assertIs(
            mask_placeholders[-1].meta["val"], restrided_mask_buffer_node.meta["val"]
        )

    def test_reorder_nodes(self):
        # [FUNCTION DOC] Compiles a function multiplying a running product over four inputs of
        # rank one to four, reverses the placeholders feeding each multiplication node, updates,
        # and expects four updated nodes and every multiplication result to become
        # four-dimensional.
        def fn(*args: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Multiplies all arguments together, starting from ones shaped
            # like the first, and returns the product.
            ret = torch.ones_like(args[0])
            for a in args:
                ret = a * ret
            return ret

        a = torch.rand((8,))
        b = torch.rand((8, 8))
        c = torch.rand((8, 8, 8))
        d = torch.rand((8, 8, 8, 8))
        graph = self._get_graph(fn, a, b, c, d)
        updater = FakeTensorUpdater(graph)

        reversed_placeholders: list[torch.fx.Node] = list(
            reversed(graph.graph.find_nodes(op="placeholder"))
        )
        mul_nodes: list[torch.fx.Node] = graph.graph.find_nodes(
            op="call_function", target=aten.mul.Tensor
        )
        for p, m in zip(reversed_placeholders, mul_nodes, strict=True):
            # The argument tensor is always at index zero.
            m.replace_input_with(m.all_input_nodes[0], p)

        with V.set_fake_mode(self._get_faketensormode(graph)):
            num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 4)
        # With reversed multiplication order, all the mul_nodes should output 4-D
        # tensors.
        for m in mul_nodes:
            self.assertEqual(len(m.meta["val"].size()), 4)

    def test_fake_tensor_same_recursion(self):
        # [FUNCTION DOC] Builds a list that contains itself and a second list containing it.
        # Expects the fake-tensor equality helper to return true when comparing the
        # self-referential list to itself and false against the other list, without hitting
        # recursion limits.
        l = [1, 2, 3]
        l.append(l)
        m = [4, 5, 6, l]
        # If recursion is broken, we'll get a recursion error here.
        self.assertTrue(_is_fake_tensor_same(l, l, {}))
        self.assertFalse(_is_fake_tensor_same(l, m, {}))

    def test_unchanged_inductor_lowering_node_is_ignored(self):
        # [FUNCTION DOC] Builds a graph with a lowering-marked node and sets matching fake
        # metadata on all nodes. With no edits, an incremental update must report zero updated
        # nodes and keep the lowering node's shape.
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 0)
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_changed_node_back_to_previous_hash_updates_metadata(self):
        # [FUNCTION DOC] Builds a negation graph, redirects the negation to another input and
        # updates, expecting one update and the new shape; then redirects it back and updates,
        # expecting again one update and the original shape, even though the node's earlier state
        # was seen before.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()
            self.assertEqual(num_updated, 1)
            self.assertEqual(tuple(neg.meta["val"].shape), (4, 5))

            neg.args = (x,)
            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))

    def test_new_inductor_lowering_node_with_metadata_is_ignored(self):
        # [FUNCTION DOC] Adds a lowering-marked node that already carries fake metadata after the
        # updater was created. An update must report zero updates and leave the node's shape.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            lowered.meta["val"] = neg.meta["val"]
            output.args = (lowered,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 0)
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_marked_inductor_lowering_node_ignores_storage_only_dependency_change(
        self,
    ):
        # [FUNCTION DOC] Uses a lowering node marked to ignore input storage, takes the negation
        # as a positional argument, then redirects the negation to a different tensor of the same
        # shape. Expects one update (the negation) and unchanged shapes everywhere.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (neg,),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_marked_inductor_lowering_node_ignores_storage_only_kwarg_change(self):
        # [FUNCTION DOC] Same as the previous storage-only case but the negation is passed to the
        # marked lowering node as a keyword argument.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (),
            {"other": neg},
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(neg.meta["val"].shape), (2, 3))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_pass_through_inductor_lowering_node_updates_from_input_metadata(self):
        # [FUNCTION DOC] Uses a lowering node marked to ignore input storage and to pass through
        # the metadata of a keyword input; redirects the negation to a differently shaped tensor.
        # Expects two updates and both the negation and lowering node to take the new shape.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True,
                output_metadata_is_input="input_",
            ),
            (),
            {"input_": neg},
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 2)
        self.assertEqual(tuple(neg.meta["val"].shape), (4, 5))
        self.assertEqual(tuple(lowered.meta["val"].shape), (4, 5))

    def test_inductor_lowering_node_metadata_fn_updates_direct_arg_change(self):
        # [FUNCTION DOC] Uses a lowering node with a metadata function that reshapes its input,
        # built with positional arguments; changes them to another tensor and a new target shape.
        # Expects one update and the lowering node's shape to follow the new target.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_fn=lambda input_, shape: aten.reshape.default(
                    input_, shape
                )
            ),
            (x, (6,)),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            lowered.meta["val"] = aten.reshape.default(x.meta["val"], (6,))

            updater = FakeTensorUpdater(gm)
            lowered.args = (y, (20,))

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(lowered.meta["val"].shape), (20,))

    def test_inductor_lowering_node_metadata_fn_preserves_view_aliasing(self):
        # [FUNCTION DOC] Like the previous reshape case but with storage not ignored and an equal
        # target shape. Expects one update, the shape kept, and the node's storage to be the same
        # as the new input's storage, so view aliasing is preserved.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=False,
                output_metadata_fn=lambda input_, shape: aten.reshape.default(
                    input_, shape
                ),
            ),
            (x, (6,)),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            lowered.meta["val"] = aten.reshape.default(x.meta["val"], (6,))

            updater = FakeTensorUpdater(gm)
            lowered.args = (y, (6,))

            with V.set_fake_mode(mode):
                num_updated = updater.incremental_update()

        self.assertEqual(num_updated, 1)
        self.assertEqual(tuple(lowered.meta["val"].shape), (6,))
        self.assertEqual(
            lowered.meta["val"].untyped_storage()._cdata,
            y.meta["val"].untyped_storage()._cdata,
        )

    def test_pass_through_inductor_lowering_node_rejects_missing_input_metadata(
        self,
    ):
        # [FUNCTION DOC] Uses a pass-through lowering node, then deletes the fake metadata of the
        # negation's input and points the node at it. Expects the update to raise a runtime error
        # saying metadata is unavailable and the lowering node's shape to stay unchanged.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True,
                output_metadata_is_input="input_",
            ),
            (),
            {"input_": neg},
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            del x.meta["val"]
            lowered.kwargs = {"input_": x}

            with self.assertRaisesRegex(RuntimeError, "metadata is unavailable"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_unmarked_inductor_lowering_node_rejects_storage_only_dependency_change(
        self,
    ):
        # [FUNCTION DOC] With a lowering node not marked to ignore storage, redirects the negation
        # to another tensor of the same shape. Expects the update to raise a runtime error about a
        # changed dependency.
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_marked_inductor_lowering_node_rejects_dtype_dependency_change(self):
        # [FUNCTION DOC] With a lowering node marked to ignore storage, redirects the negation to
        # a tensor of a different dtype (float64 instead of float32). Expects a runtime error
        # about a changed dependency, so only storage differences are tolerated.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        lowered = graph.call_function(
            self._make_inductor_lowering_function(
                output_metadata_ignores_input_storage=True
            ),
            (neg,),
        )
        graph.output(lowered)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3, dtype=torch.float32))
            y.meta["val"] = mode.from_tensor(torch.randn(2, 3, dtype=torch.float64))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_new_inductor_lowering_node_with_changed_dependency_raises(self):
        # [FUNCTION DOC] Adds an unmarked lowering node carrying metadata after the updater is
        # built, then changes the negation's input to another shape. Expects a runtime error about
        # a changed dependency; afterwards the negation has the new shape but the lowering node
        # keeps its old one.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.placeholder("y")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            lowered.meta["val"] = neg.meta["val"]
            output.args = (lowered,)
            neg.args = (y,)

            with self.assertRaisesRegex(RuntimeError, "changed dependency"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

        self.assertEqual(tuple(neg.meta["val"].shape), (4, 5))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))

    def test_new_inductor_lowering_node_without_metadata_raises(self):
        # [FUNCTION DOC] Adds a lowering node without fake metadata after the updater is built.
        # Expects the update to raise a runtime error saying such nodes must already carry fake
        # metadata.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        neg = graph.call_function(aten.neg.default, (x,))
        output = graph.output(neg)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            neg.meta["val"] = aten.neg.default(x.meta["val"])

            updater = FakeTensorUpdater(gm)
            with graph.inserting_before(output):
                lowered = graph.call_function(
                    self._make_inductor_lowering_function(), (neg,)
                )
            output.args = (lowered,)

            with self.assertRaisesRegex(RuntimeError, "already carry fake metadata"):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

    def test_changed_inductor_lowering_node_raises_before_stale_metadata(self):
        # [FUNCTION DOC] Inserts a new negation of another tensor before the lowering node and
        # feeds it in. Expects a runtime error mentioning lowering-function nodes, raised before
        # stale metadata is used; the new negation has been updated to the new shape while the
        # lowering node keeps its old shape.
        gm, x, y, neg, lowered = self._build_graph_with_inductor_lowering_node()

        with torch._subclasses.FakeTensorMode() as mode, torch.no_grad():
            x.meta["val"] = mode.from_tensor(torch.randn(2, 3))
            y.meta["val"] = mode.from_tensor(torch.randn(4, 5))
            neg.meta["val"] = aten.neg.default(x.meta["val"])
            lowered.meta["val"] = neg.meta["val"]

            updater = FakeTensorUpdater(gm)
            with gm.graph.inserting_before(lowered):
                neg_replacement = gm.graph.call_function(aten.neg.default, (y,))
            lowered.args = (neg_replacement,)

            with self.assertRaisesRegex(
                RuntimeError,
                "_inductor_lowering_function nodes",
            ):
                with V.set_fake_mode(mode):
                    updater.incremental_update()

        self.assertEqual(tuple(neg_replacement.meta["val"].shape), (4, 5))
        self.assertEqual(tuple(lowered.meta["val"].shape), (2, 3))


# Stand-in for any exception that is not TritonUnavailableError and must not
# escape has_triton() for a sub-capable device. (The real GPUTooOldForTriton
# is a RuntimeError subclass; this one deliberately is not, so an escape
# cannot be mistaken for anything has_triton() legitimately catches.)
class _GPUTooOldForTriton(Exception):
    # [CLASS DOC] HELPER: Exception deliberately not a runtime error, used to detect that the
    # capability check happens before the backend probe. Adds no behavior.
    pass


def _make_triton_interface(*, available=True, capable=True, raise_exc=None):
    # [FUNCTION DOC] HELPER: Builds and returns a fake device interface class whose availability
    # and Triton capability follow the given flags and whose backend check raises the given
    # exception when one is supplied.
    class _FakeInterface(DeviceInterface):
        # [CLASS DOC] HELPER: Fake device interface subclass created per call; closes over the
        # options.
        @staticmethod
        def is_available() -> bool:
            # [FUNCTION DOC] HELPER: Static; returns the configured availability.
            return available

        @staticmethod
        def is_triton_capable(device=None) -> bool:
            # [FUNCTION DOC] HELPER: Static; returns the configured Triton capability.
            return capable

        @classmethod
        def raise_if_triton_unavailable(cls, device=None) -> None:
            # [FUNCTION DOC] HELPER: Class method; raises the configured exception if there is
            # one, else returns nothing.
            if raise_exc is not None:
                raise raise_exc

    return _FakeInterface


class TestHasTriton(TestCase):
    # [CLASS DOC] Tests for Triton availability detection, using fake device interfaces registered
    # through patching.
    def tearDown(self):
        # [FUNCTION DOC] HELPER: Clears the memoized Triton detection result and calls the base
        # teardown.
        triton_utils.has_triton.cache_clear()
        super().tearDown()

    def _run(self, registered, *, has_package=True, detection_disabled=False):
        # [FUNCTION DOC] HELPER: Runs Triton detection with the package-presence check, the
        # inductor option disabling device detection, and the registered device list all patched.
        # Clears the memoization before and returns the detection result.
        with (
            mock.patch.object(
                triton_utils, "has_triton_package", return_value=has_package
            ),
            inductor_config.patch(triton_disable_device_detection=detection_disabled),
            mock.patch(
                "torch._dynamo.device_interface.get_registered_device_interfaces",
                return_value=registered,
            ),
        ):
            triton_utils.has_triton.cache_clear()
            return triton_utils.has_triton()

    def test_no_triton_package(self):
        # [FUNCTION DOC] With the Triton package reported missing, expects detection to return
        # false.
        result = self._run([("fake", _make_triton_interface())], has_package=False)
        self.assertFalse(result)

    def test_detection_disabled(self):
        # [FUNCTION DOC] With the inductor device-detection-disable option on, expects false.
        result = self._run(
            [("fake", _make_triton_interface())], detection_disabled=True
        )
        self.assertFalse(result)

    def test_capable_available_backend_built(self):
        # [FUNCTION DOC] With one available, capable fake device whose backend check passes,
        # expects true.
        self.assertTrue(self._run([("fake", _make_triton_interface())]))

    def test_device_not_available(self):
        # [FUNCTION DOC] With a fake device that is not available, expects false.
        self.assertFalse(self._run([("fake", _make_triton_interface(available=False))]))

    def test_device_not_triton_capable(self):
        # [FUNCTION DOC] With a fake device that is not Triton capable, expects false.
        self.assertFalse(self._run([("fake", _make_triton_interface(capable=False))]))

    def test_backend_missing_is_swallowed(self):
        # [FUNCTION DOC] With a device whose backend check raises the Triton-unavailable error,
        # expects the error to be swallowed and false returned.
        iface = _make_triton_interface(
            raise_exc=TritonUnavailableError("backend not built")
        )
        self.assertFalse(self._run([("fake", iface)]))

    def test_unexpected_runtime_error_propagates(self):
        # [FUNCTION DOC] With a device whose backend check raises a generic runtime error, expects
        # it to propagate out of detection instead of being treated as no Triton.
        # A generic RuntimeError is NOT the "no triton backend" signal, so it
        # must surface instead of being silently treated as "no triton".
        iface = _make_triton_interface(raise_exc=RuntimeError("something else broke"))
        with self.assertRaisesRegex(RuntimeError, "something else broke"):
            self._run([("fake", iface)])

    def test_indexed_device_name_skipped(self):
        # [FUNCTION DOC] With a device registered under a name containing an index, expects it to
        # be skipped (returns false) even though it is available and capable.
        # "fake:0" is available+capable but must be skipped as an indexed alias.
        self.assertFalse(self._run([("fake:0", _make_triton_interface())]))

    def test_first_working_device_wins(self):
        # [FUNCTION DOC] With a first device not capable and a second one fully working, expects
        # true.
        registered = [
            ("bad", _make_triton_interface(capable=False)),
            ("good", _make_triton_interface()),
        ]
        self.assertTrue(self._run(registered))

    def test_capability_gate_precedes_backend_probe(self):
        # [FUNCTION DOC] With a non-capable device whose backend probe would raise a non-runtime
        # exception, expects false and no exception, showing the capability check runs first.
        # A sub-capable device whose backend probe would throw a NON-RuntimeError.
        # The capability gate must short-circuit before the probe is ever called;
        # if the ordering regresses, _GPUTooOldForTriton escapes instead of False.
        iface = _make_triton_interface(capable=False, raise_exc=_GPUTooOldForTriton())
        self.assertFalse(self._run([("fake", iface)]))


if __name__ == "__main__":
    run_tests()
