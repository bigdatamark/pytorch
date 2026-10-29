# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._dynamo.debug_utils and the debug override configs that
# route individual compiled graphs to different backends / inductor / dynamo settings.
# Covers storage-size serialization, repro-script import generation, fp64 casting, env-var
# string generation, nvcc probing errors, AOT graph input parsing, NNModule-to-source
# conversion, and per-graph-id override routing. Defines dtype aliases f32/i64/i32; device
# generic classes are instantiated at import time into the module globals via
# instantiate_device_type_tests.

import ast
import math
import os
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.config
from torch._dynamo import debug_utils
from torch._dynamo.debug_utils import (
    _serialize_storage_nbytes,
    aot_graph_input_parser,
    generate_env_vars_string,
    NNModuleToString,
)
from torch._dynamo.test_case import TestCase
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.inductor_utils import GPU_TYPE


f32 = torch.float32
i64 = torch.int64
i32 = torch.int32


class TestDebugUtils(TestCase):
    # [CLASS DOC] Device-independent tests of debug_utils helpers: storage size
    # serialization, repro string imports, fp64 casting, env var strings, nvcc errors.
    def test_serialize_symbolic_storage_nbytes(self):
        # [FUNCTION DOC] Builds a symbolic integer storage size from a shape-env symbol and
        # checks that its serialized source has no Max call and, evaluated with a max
        # builtin for several symbol values, equals the sympy-substituted result. Also checks
        # that a floor expression is serialized using the math floor function and evaluates
        # correctly.
        from sympy import floor

        from torch._dynamo.source import ConstantSource
        from torch.fx.experimental.sym_node import SymNode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv
        from torch.utils._sympy.functions import Max

        shape_env = ShapeEnv()
        symbol = shape_env.create_symbol(4, ConstantSource("storage_size"))
        expr = 4 * symbol + 18428 * Max(1, symbol)
        nbytes = torch.SymInt(SymNode(expr, shape_env, int, hint=73728))
        source = _serialize_storage_nbytes(nbytes)

        self.assertNotIn("Max", source)
        for value in (0, 1, 4):
            self.assertEqual(
                eval(source, {"max": max}, {str(symbol): value}),
                int(expr.subs(symbol, value)),
            )

        floor_expr = floor(symbol / 2)
        floor_nbytes = torch.SymInt(SymNode(floor_expr, shape_env, int, hint=2))
        floor_source = _serialize_storage_nbytes(floor_nbytes)
        self.assertIn("math.floor", floor_source)
        self.assertEqual(
            eval(floor_source, {"math": math}, {str(symbol): 4}),
            2,
        )

    def test_repro_templates_import_symexpr_dependencies(self):
        # [FUNCTION DOC] Generates the after-AOT and after-dynamo repro scripts for a traced
        # add-one graph, parses each, executes only its import statements in a fresh
        # namespace, and asserts the math module is available there (floor of 3/2 is 1).
        from torch._dynamo.repro.after_aot import generate_compiler_repro_string
        from torch._dynamo.repro.after_dynamo import generate_dynamo_fx_repro_string

        gm = torch.fx.symbolic_trace(lambda x: x + 1)
        args = [torch.ones(1)]

        repros = {
            "after_aot": generate_compiler_repro_string(gm, args),
            "after_dynamo": generate_dynamo_fx_repro_string(
                gm, args, compiler_name="eager"
            ),
        }
        for name, source in repros.items():
            with self.subTest(name=name):
                tree = ast.parse(source)
                imports = ast.Module(
                    body=[
                        node
                        for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                    ],
                    type_ignores=[],
                )
                namespace = {}
                exec(compile(imports, f"<{name}>", "exec"), namespace)
                self.assertEqual(eval("math.floor(3 / 2)", namespace), 1)

    def test_cast_model_to_fp64_dtype_args(self):
        # [FUNCTION DOC] Traces a function with make_fx and core ATen decompositions,
        # asserts the generated code uses float16/float32 dtypes, then applies the debug
        # cast-to-fp64 helper in place and asserts the example inputs become float64 and every
        # dtype argument in the graph code is rewritten to float64.
        # Test that dtype arguments are converted to fp64

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of a prim dtype conversion to half, a
            # to-half cast, a float32 full tensor and a new empty tensor, all shaped like x.
            return (
                torch.ops.prims.convert_element_type(x, torch.float16),
                x.to(torch.float16),
                torch.full(x.shape, 2, dtype=torch.float32, device=x.device),
                x.new_empty(x.shape),
            )

        x = torch.randn(32, device="cpu")
        decomps = torch._decomp.core_aten_decompositions()
        fx = make_fx(fn, decomposition_table=decomps)(x)

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float16)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float16);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float32, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,
        )

        _, fp64_examples = debug_utils.cast_to_fp64(fx, (x,))
        self.assertEqual(fp64_examples, (x.to(torch.float64),))

        self.assertExpectedInline(
            fx.code.lstrip(),
            """\
def forward(self, x_1):
    convert_element_type = torch.ops.prims.convert_element_type.default(x_1, torch.float64)
    _to_copy = torch.ops.aten._to_copy.default(x_1, dtype = torch.float64);  x_1 = None
    full = torch.ops.aten.full.default([32], 2, dtype = torch.float64, device = device(type='cpu'), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([32], dtype = torch.float64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
    return (convert_element_type, _to_copy, full, empty)
    """,
        )

    @patch.dict(
        os.environ,
        {
            "TORCHINDUCTOR_MAX_AUTOTUNE": "1",
            "TEST_ENV": "1",
            "TORCHINDUCTOR_ENV_SINGLE_QUOTES": "inductor_'env'",
            "TORCHINDUCTOR_ENV_DOUBLE_QUOTES": 'inductor_"env"',
            "TORCHDYNAMO_REPRO_AFTER": "aot",
            "TORCHDYNAMO_REPRO_LEVEL": "4",
        },
    )
    def test_generate_env_vars_string(self):
        # [FUNCTION DOC] With a patched environment, checks the generated env var string
        # contains an import of os, assignments for inductor variables (single quotes in a
        # value are expected to come out as double quotes), omits unrelated variables, omits
        # assignments for the repro-after and repro-level variables and instead emits pop
        # statements for them.
        env_strings = generate_env_vars_string()
        self.assertIn(
            """os.environ['TORCHINDUCTOR_MAX_AUTOTUNE'] = '1'
""",
            env_strings,
        )
        self.assertIn(
            """os.environ['TORCHINDUCTOR_ENV_SINGLE_QUOTES'] = 'inductor_"env"'
""",
            env_strings,
        )
        self.assertIn(
            """os.environ['TORCHINDUCTOR_ENV_DOUBLE_QUOTES'] = 'inductor_"env"'
""",
            env_strings,
        )
        self.assertIn(
            """import os
""",
            env_strings,
        )
        self.assertNotIn(
            """TEST_ENV
""",
            env_strings,
        )
        self.assertNotIn("os.environ['TORCHDYNAMO_REPRO_AFTER']", env_strings)
        self.assertNotIn("os.environ['TORCHDYNAMO_REPRO_LEVEL']", env_strings)
        self.assertIn("os.environ.pop('TORCHDYNAMO_REPRO_AFTER', None)", env_strings)
        self.assertIn("os.environ.pop('TORCHDYNAMO_REPRO_LEVEL', None)", env_strings)

    def test_cuda_system_info_comment_nvcc_os_errors(self):
        # [FUNCTION DOC] For a permission error and a generic OS error, clears the cache
        # of the cuda system info comment helper, mocks CUDA as available (non-HIP) with
        # the nvcc subprocess raising that error and zero devices, and asserts the result
        # reports nvcc not found and an empty GPU hardware line. The cache is cleared again
        # at cleanup.
        self.addCleanup(debug_utils._cuda_system_info_comment.cache_clear)
        errors = (
            PermissionError(13, "Permission denied", "nvcc"),
            OSError(5, "Input/output error", "nvcc"),
        )

        for error in errors:
            with self.subTest(error=type(error).__name__):
                debug_utils._cuda_system_info_comment.cache_clear()
                with (
                    patch.object(torch.cuda, "is_available", return_value=True),
                    patch.object(torch.version, "hip", None),
                    patch.object(
                        debug_utils.subprocess,
                        "check_output",
                        side_effect=error,
                    ),
                    patch.object(torch.cuda, "device_count", return_value=0),
                ):
                    result = debug_utils._cuda_system_info_comment()

                self.assertIn("# nvcc not found\n", result)
                self.assertIn("# GPU Hardware Info: \n", result)


class TestDebugUtilsDevice(TestCase):
    # [CLASS DOC] Device-generic tests of the AOT graph input parser; instantiated per
    # device (excluding mps, allowing xpu) by instantiate_device_type_tests.
    def test_aot_graph_parser(self, device):
        # [FUNCTION DOC] Defines a large AOT-style forward function whose parameters carry
        # shape/dtype string annotations, asks the input parser to build matching tensors on
        # the device, and asserts the function runs with those generated inputs.
        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[1001]",
            primals_3: "f32[1001, 64]",
            primals_4: "f32[4190]",
            primals_5: "f32[4190]",
            primals_6: "f32[1739, 4190]",
            primals_48: "f32[6144, 4191]",
        ):
            # [FUNCTION DOC] HELPER: Sample traced graph (embedding bag over indexed
            # inputs, using the enclosing device); returns the embedding bag output tuple.
            _tensor_constant0: "i64[4190]" = self._tensor_constant0
            lift_fresh_copy: "i64[4190]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant0
            )
            _tensor_constant0 = None
            index: "f32[6144, 4190]" = torch.ops.aten.index.Tensor(  # noqa: F841
                primals_48, [None, lift_fresh_copy]
            )
            lift_fresh_copy = None

            _tensor_constant1: "i64[6]" = self._tensor_constant1
            lift_fresh_copy_1: "i64[6]" = torch.ops.aten.lift_fresh_copy.default(
                _tensor_constant1
            )
            _tensor_constant1 = None
            index_1: "f32[6144, 6]" = torch.ops.aten.index.Tensor(
                primals_48, [None, lift_fresh_copy_1]
            )
            primals_48 = lift_fresh_copy_1 = None
            permute: "f32[6, 1001]" = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm: "f32[6144, 1001]" = torch.ops.aten.addmm.default(
                primals_2, index_1, permute
            )
            primals_2 = permute = None
            amax: "f32[6144, 1]" = torch.ops.aten.amax.default(addmm, [-1], True)
            sub: "f32[6144, 1001]" = torch.ops.aten.sub.Tensor(addmm, amax)
            exp: "f32[6144, 1001]" = torch.ops.aten.exp.default(sub)
            sub = None
            sum_1: "f32[6144, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
            div: "f32[6144, 1001]" = torch.ops.aten.div.Tensor(exp, sum_1)
            exp = None

            full_default: "i32[6144, 1001]" = torch.ops.aten.full.default(
                [6144, 1001],
                1,
                dtype=torch.int32,
                layout=torch.strided,
                device=device,
                pin_memory=False,
            )

            iota: "i32[1001]" = torch.ops.prims.iota.default(
                1001,
                start=0,
                step=1,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )

            mul: "i32[6144, 1001]" = torch.ops.aten.mul.Tensor(full_default, iota)
            full_default = iota = None

            iota_1: "i32[6144]" = torch.ops.prims.iota.default(
                6144,
                start=0,
                step=1001,
                dtype=torch.int32,
                device=device,
                requires_grad=False,
            )
            view: "i32[6150144]" = torch.ops.aten.reshape.default(mul, [-1])
            mul = None
            view_1: "f32[6150144]" = torch.ops.aten.reshape.default(div, [-1])
            div = None
            _embedding_bag = torch.ops.aten._embedding_bag.default(
                primals_3, view, iota_1, False, 0, False, view_1
            )

            return _embedding_bag

        kwargs = aot_graph_input_parser(forward, device=device)
        # runs successfully
        forward(**kwargs)

    def test_sym_aot_graph_parser(self, device):
        # [FUNCTION DOC] Parses a forward signature that has symbolic sizes. With s0 mapped
        # to 10 and a default of 5, asserts tensor inputs sized by s0 get length 10, the
        # symbolic int for s0 equals 10, and those for s1 use the default 5.
        def forward(
            self,
            primals_1: "f32[1001, 6]",
            primals_2: "f32[s0]",  # noqa: F821
            primals_3: "Sym(s0)",  # noqa: F821,
            primals_4: "f32[s1]",  # noqa: F821,
            primals_5: "Sym(s1)",  # noqa: F821,
        ):
            # [FUNCTION DOC] HELPER: Signature-only stub; its body reads one constant
            # attribute and is never called, only inspected by the parser.
            _tensor_constant0: "i64[4190]" = self._tensor_constant0

        kwargs = aot_graph_input_parser(
            forward, device=device, sym_shapes={"s0": 10}, default_sym_shape=5
        )

        self.assertEqual(list(kwargs["primals_2"].shape), [10])
        self.assertEqual(kwargs["primals_3"], 10)

        self.assertEqual(list(kwargs["primals_4"].shape), [5])
        self.assertEqual(kwargs["primals_5"], 5)


class TestNNModuleToStringBufferDevice(TestCase):
    # [CLASS DOC] Device-generic tests that converting a graph module to source preserves
    # buffer and parameter device placement; instantiated per device (xpu allowed).
    def test_nn_module_to_string_buffer_device(self, device):
        # [FUNCTION DOC] Registers a buffer on the device in a traced identity module and
        # converts it to source. On cpu asserts no device move appears; otherwise asserts the
        # source moves the buffer to the concrete device string and not to the generic GPU type.
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        gm.register_buffer("test_buf", torch.empty(5, device=device))

        result = NNModuleToString.convert(gm)

        if torch.device(device).type == "cpu":
            self.assertNotIn(".to(", result)
        else:
            expected_device = str(torch.empty(1, device=device).device)
            self.assertIn(f'.to("{expected_device}")', result)
            self.assertNotIn(f".{GPU_TYPE}()", result)

    def test_nn_module_to_string_param_device(self, device):
        # [FUNCTION DOC] Same as the buffer test but with a registered parameter: on cpu the
        # source has no device argument; otherwise it names the concrete device and does not
        # name the generic GPU type.
        gm = torch.fx.symbolic_trace(torch.nn.Identity())
        gm.register_parameter(
            "test_param", torch.nn.Parameter(torch.empty(5, device=device))
        )

        result = NNModuleToString.convert(gm)

        if torch.device(device).type == "cpu":
            self.assertNotIn("device=", result)
        else:
            expected_device = str(torch.empty(1, device=device).device)
            self.assertIn(f'device="{expected_device}"', result)
            self.assertNotIn(f', device="{GPU_TYPE}")', result)


instantiate_device_type_tests(
    TestNNModuleToStringBufferDevice, globals(), allow_xpu=True
)

instantiate_device_type_tests(TestDebugUtils, globals())

instantiate_device_type_tests(
    TestDebugUtilsDevice, globals(), except_for="mps", allow_xpu=True
)


class TestBackendOverrideIntegration(TestCase):
    # [CLASS DOC] Device-generic tests of the debug backend override config, which maps
    # rules on graph ids to backend names. Keeps a list of override backend names that
    # were actually applied during a run; resets dynamo before and after each test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, resets dynamo state, and starts an
        # empty list of backends called.
        super().setUp()
        torch._dynamo.reset()
        self._backends_called = []

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and calls the base tearDown.
        torch._dynamo.reset()
        super().tearDown()

    def _fn_with_4_graphs(self, x):
        # [FUNCTION DOC] HELPER: Function with three explicit graph breaks (four graphs
        # when compiled): add one, multiply by two, subtract one, divide by two.
        x = x + 1
        torch._dynamo.graph_break()
        x = x * 2
        torch._dynamo.graph_break()
        x = x - 1
        torch._dynamo.graph_break()
        x = x / 2
        return x

    def _run_with_override(self, device, override_config, default_backend="eager"):
        # [FUNCTION DOC] HELPER: Resets dynamo and the router cache, clears the recorded
        # list, and pre-parses the semicolon separated rules (filter before the first
        # colon, backend name after) into a graph id to backend name map for the first 100
        # ids (first matching rule wins). Then patches the debug backend override config and
        # the output graph's lookup to record the backend name for each graph where an
        # override was found, compiles the four-graph function with the default backend on a
        # random input of size 10, and returns a copy of the recorded names. Conflicting
        # rules surface as errors from the compile.
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            get_backend_override_for_compile_id,
        )

        torch._dynamo.reset()
        # Clear the router cache to ensure fresh routers for each test
        _create_backend_router.cache_clear()
        self._backends_called.clear()
        original_get_override = get_backend_override_for_compile_id

        # Pre-parse the config to build a mapping of graph_id -> backend_str
        # by using the same parsing logic but extracting the original strings
        backend_str_map: dict[int, str] = {}
        if override_config:
            for rule_str in override_config.split(";"):
                rule_str = rule_str.strip()
                if not rule_str or ":" not in rule_str:
                    continue
                colon_idx = rule_str.find(":")
                filter_str = rule_str[:colon_idx].strip()
                backend_str = rule_str[colon_idx + 1 :].strip()
                # Parse the filter to extract graph IDs
                from torch._dynamo.graph_id_filter import GraphIdFilter

                gf = GraphIdFilter(filter_str)
                # Store the backend_str for any graph that matches this filter
                for graph_id in range(100):  # Check first 100 graphs
                    if graph_id in gf and graph_id not in backend_str_map:
                        backend_str_map[graph_id] = backend_str

        def tracking_get_override(compile_id, config_str):
            # [FUNCTION DOC] HELPER: Wrapper around the real override lookup; when it finds
            # an override and the frame id has a known backend name, records that name.
            # Returns the real result unchanged.
            result = original_get_override(compile_id, config_str)
            if result is not None:
                graph_id = compile_id.frame_id
                if graph_id in backend_str_map:
                    self._backends_called.append(backend_str_map[graph_id])
            return result

        with (
            patch.object(
                torch._dynamo.config, "debug_backend_override", override_config
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_override,
            ),
        ):
            compiled_fn = torch.compile(self._fn_with_4_graphs, backend=default_backend)
            compiled_fn(torch.randn(10, device=device))

        return self._backends_called.copy()

    def test_no_override(self, device):
        # [FUNCTION DOC] An empty override config applies no overrides.
        result = self._run_with_override(device, "")
        self.assertEqual(result, [])

    def test_override_all_graphs(self, device):
        # [FUNCTION DOC] A greater-or-equal-zero rule overrides all four graphs.
        result = self._run_with_override(device, ">=0:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager", "aot_eager", "aot_eager"])

    def test_override_greater_than(self, device):
        # [FUNCTION DOC] A greater-than-zero rule overrides graphs 1 to 3 (three graphs).
        result = self._run_with_override(device, ">0:eager")
        self.assertEqual(result, ["eager", "eager", "eager"])

    def test_override_less_than(self, device):
        # [FUNCTION DOC] A less-than-two rule overrides graphs 0 and 1.
        result = self._run_with_override(device, "<2:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_less_or_equal(self, device):
        # [FUNCTION DOC] A less-or-equal-one rule overrides graphs 0 and 1.
        result = self._run_with_override(device, "<=1:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_single_id(self, device):
        # [FUNCTION DOC] A rule naming graph id 1 overrides exactly one graph.
        result = self._run_with_override(device, "1:aot_eager")
        self.assertEqual(result, ["aot_eager"])

    def test_override_multiple_ids(self, device):
        # [FUNCTION DOC] A comma separated id list (0 and 2) overrides those two graphs.
        result = self._run_with_override(device, "0,2:aot_eager")
        self.assertEqual(result, ["aot_eager", "aot_eager"])

    def test_override_range(self, device):
        # [FUNCTION DOC] An inclusive id range (1 to 2) overrides two graphs.
        result = self._run_with_override(device, "1-2:eager")
        self.assertEqual(result, ["eager", "eager"])

    def test_multiple_rules(self, device):
        # [FUNCTION DOC] Three rules for ids 0, 1 and 3 apply different backends; graph 2
        # is not overridden, so three names are recorded in order.
        result = self._run_with_override(device, "0:aot_eager;1:inductor;3:eager")
        self.assertEqual(result, ["aot_eager", "inductor", "eager"])

    def test_conflicting_rules_raise(self, device):
        # [FUNCTION DOC] Two overlapping rules naming different backends make compilation
        # raise an internal dynamo error mentioning a conflicting backend override.
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            "Conflicting backend override",
        ):
            self._run_with_override(device, ">=0:aot_eager;>=1:inductor")

    def test_complex_config(self, device):
        # [FUNCTION DOC] Combines an exact-id rule and a greater-or-equal rule that do not
        # overlap: graph 0 gets one backend, graphs 2 and 3 another.
        result = self._run_with_override(device, "0:aot_eager;>=2:inductor")
        self.assertEqual(result, ["aot_eager", "inductor", "inductor"])

    def test_override_with_backward(self, device):
        # [FUNCTION DOC] Compiles a sum-producing function with an override for all graphs
        # and runs forward and backward on an input requiring grad. Asserts the override
        # lookup returned a result only for frame 0 (once) and that the gradient exists.
        """Verify that backend override works when backward compilation occurs."""
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            get_backend_override_for_compile_id,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        overrides_applied = []
        original_get_override = get_backend_override_for_compile_id

        def tracking_get_override(compile_id, config_str):
            # [FUNCTION DOC] HELPER: Calls the real override lookup and, when it finds an
            # override, records the frame id. Returns the real result.
            result = original_get_override(compile_id, config_str)
            if result is not None:
                overrides_applied.append(compile_id.frame_id)
            return result

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x times two plus one.
            return (x * 2 + 1).sum()

        with (
            patch.object(
                torch._dynamo.config, "debug_backend_override", ">=0:aot_eager"
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_override,
            ),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            x = torch.randn(10, device=device, requires_grad=True)
            result = compiled_fn(x)
            result.backward()

        self.assertEqual(overrides_applied, [0])
        self.assertIsNotNone(x.grad)


instantiate_device_type_tests(
    TestBackendOverrideIntegration, globals(), except_for="mps"
)


class TestInductorConfigOverrideIntegration(TestCase):
    # [CLASS DOC] Device-generic tests of the per-graph inductor config override: the
    # config router parser (ids, ranges, comparisons, value typing, merging, conflicts),
    # the backend router conflict rules, and end-to-end application of overrides. Resets
    # dynamo around each test; instantiated for cpu, cuda and xpu only.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and resets dynamo state.
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and calls the base tearDown.
        torch._dynamo.reset()
        super().tearDown()

    def test_config_router_single_graph(self, device):
        # [FUNCTION DOC] A rule for graph 0 yields a one-entry mapping for graph 0 and
        # nothing (None) for graph 1.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:triton.cudagraph_skip_dynamic_graphs=False")
        self.assertEqual(
            router.get_value_for_graph(0),
            {"triton.cudagraph_skip_dynamic_graphs": False},
        )
        self.assertIsNone(router.get_value_for_graph(1))

    def test_config_router_multiple_options(self, device):
        # [FUNCTION DOC] Comma separated options in one rule are all returned for graph 0.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(
            "0:triton.cudagraphs=False,triton.cudagraph_trees=False"
        )
        self.assertEqual(
            router.get_value_for_graph(0),
            {"triton.cudagraphs": False, "triton.cudagraph_trees": False},
        )

    def test_config_router_comparison(self, device):
        # [FUNCTION DOC] A greater-than-1 rule matches graph 2 only among graphs 0 to 2.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(">1:triton.cudagraphs=True")
        self.assertIsNone(router.get_value_for_graph(0))
        self.assertIsNone(router.get_value_for_graph(1))
        self.assertEqual(router.get_value_for_graph(2), {"triton.cudagraphs": True})

    def test_config_router_range(self, device):
        # [FUNCTION DOC] An inclusive range 1 to 3 matches graphs 1, 2, 3 and not 0 or 4.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("1-3:triton.cudagraphs=False")
        self.assertIsNone(router.get_value_for_graph(0))
        self.assertEqual(router.get_value_for_graph(1), {"triton.cudagraphs": False})
        self.assertEqual(router.get_value_for_graph(2), {"triton.cudagraphs": False})
        self.assertEqual(router.get_value_for_graph(3), {"triton.cudagraphs": False})
        self.assertIsNone(router.get_value_for_graph(4))

    def test_config_router_value_types(self, device):
        # [FUNCTION DOC] Checks option values parse to bool, int, float, plain string and
        # None according to their text.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter(
            "0:bool_opt=True,int_opt=42,float_opt=3.14,str_opt=hello,none_opt=None"
        )
        config = router.get_value_for_graph(0)
        self.assertEqual(config["bool_opt"], True)
        self.assertEqual(config["int_opt"], 42)
        self.assertAlmostEqual(config["float_opt"], 3.14)
        self.assertEqual(config["str_opt"], "hello")
        self.assertIsNone(config["none_opt"])

    def test_config_router_aggregation(self, device):
        # [FUNCTION DOC] Options from all rules matching a graph are merged into one mapping.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;>=0:b=2")
        # Graph 0 matches both rules, configs are merged
        self.assertEqual(router.get_value_for_graph(0), {"a": 1, "b": 2})
        # Graph 1 matches only the second rule
        self.assertEqual(router.get_value_for_graph(1), {"b": 2})

    def test_config_router_conflict_raises(self, device):
        # [FUNCTION DOC] Constructing a router whose overlapping rules set the same option
        # to different values raises ValueError about a conflicting config override.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        with self.assertRaisesRegex(ValueError, "Conflicting config override"):
            GraphConfigRouter("0:a=1;>=0:a=2")

    def test_config_router_same_value_no_conflict(self, device):
        # [FUNCTION DOC] Overlapping rules that set an option to the same value are not a
        # conflict; graphs 0 and 1 both get that value.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;>=0:a=1")
        self.assertEqual(router.get_value_for_graph(0), {"a": 1})
        self.assertEqual(router.get_value_for_graph(1), {"a": 1})

    def test_config_router_aggregation_multiple_rules(self, device):
        # [FUNCTION DOC] Three rules (two id-specific, one catch-all) merge per graph: graph
        # 0 gets two options, graph 1 two different ones, graph 2 only the catch-all.
        from torch._dynamo.graph_id_filter import GraphConfigRouter

        router = GraphConfigRouter("0:a=1;1:b=2;>=0:c=3")
        self.assertEqual(router.get_value_for_graph(0), {"a": 1, "c": 3})
        self.assertEqual(router.get_value_for_graph(1), {"b": 2, "c": 3})
        self.assertEqual(router.get_value_for_graph(2), {"c": 3})

    def test_backend_router_conflict_raises(self, device):
        # [FUNCTION DOC] Overlapping ranges naming different backends make the backend
        # router constructor raise ValueError about a conflicting backend override.
        from torch._dynamo.graph_id_filter import GraphBackendRouter

        with self.assertRaisesRegex(ValueError, "Conflicting backend override"):
            GraphBackendRouter("0-5:eager;3-10:inductor")

    def test_backend_router_same_backend_no_conflict(self, device):
        # [FUNCTION DOC] Overlapping rules naming the same backend construct fine and give
        # a non-None result for graph 0.
        from torch._dynamo.graph_id_filter import GraphBackendRouter

        router = GraphBackendRouter("0:eager;>=0:eager")
        self.assertIsNotNone(router.get_value_for_graph(0))

    def test_get_inductor_config_override_empty(self, device):
        # [FUNCTION DOC] With a None compile id and an empty config string, the inductor
        # config lookup returns None.
        from torch._dynamo.graph_id_filter import (
            get_inductor_config_override_for_compile_id,
        )

        result = get_inductor_config_override_for_compile_id(None, "")
        self.assertIsNone(result)

    def test_combined_backend_and_config_override(self, device):
        # [FUNCTION DOC] Resets dynamo and router caches, patches both the backend and
        # inductor config override settings, wraps the lookup and the config-wrapping hook
        # to record calls, and compiles a three-graph function with the eager backend.
        # Asserts all three graphs were routed to inductor and that the recorded config
        # patches are merged per graph (graph 1 gets both of its matching rules).
        """
        Test combining backend override with config override.

        Scenario: Default backend is eager, but override all graphs to use
        inductor with cudagraphs enabled, and additionally override graph 1
        to use cudagraph_skip_dynamic_graphs=False.
        """
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            _create_inductor_config_router,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        _create_inductor_config_router.cache_clear()

        backends_used: list[str] = []
        configs_applied: list[dict] = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Three-graph function (two explicit graph breaks):
            # add one, multiply by two, subtract one.
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            x = x - 1
            return x

        from torch._dynamo import output_graph

        original_wrap = output_graph._wrap_with_inductor_config

        def tracking_wrap(compiler_fn, config_patches):
            # [FUNCTION DOC] HELPER: Records the config patches and delegates to the
            # original wrapper, returning its result.
            configs_applied.append(config_patches)
            return original_wrap(compiler_fn, config_patches)

        backend_override = ">=0:inductor"
        from torch._dynamo.graph_id_filter import get_backend_override_for_compile_id

        original_get_backend = get_backend_override_for_compile_id

        def tracking_get_backend(compile_id, config_str):
            # [FUNCTION DOC] HELPER: Calls the real backend override lookup and records
            # the name inductor whenever an override is found; returns the real result.
            result = original_get_backend(compile_id, config_str)
            if result is not None:
                backends_used.append("inductor")
            return result

        # Use both overrides:
        # - Backend: all graphs use inductor
        # - Config: all graphs enable cudagraphs, graph 1 also disables
        #   cudagraph_skip_dynamic_graphs
        with (
            patch.object(
                torch._dynamo.config,
                "debug_backend_override",
                backend_override,
            ),
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                "1:triton.cudagraph_skip_dynamic_graphs=False;>=0:triton.cudagraphs=True",
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_backend,
            ),
            patch.object(output_graph, "_wrap_with_inductor_config", tracking_wrap),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            compiled_fn(torch.randn(10, device=device))

        self.assertEqual(len(backends_used), 3)
        self.assertEqual(backends_used, ["inductor", "inductor", "inductor"])

        # All matching rules are aggregated. Graph 1 matches both rules.
        self.assertEqual(len(configs_applied), 3)
        self.assertEqual(configs_applied[0], {"triton.cudagraphs": True})
        self.assertEqual(
            configs_applied[1],
            {
                "triton.cudagraph_skip_dynamic_graphs": False,
                "triton.cudagraphs": True,
            },
        )
        self.assertEqual(configs_applied[2], {"triton.cudagraphs": True})

    def test_multiple_config_overrides_with_backend(self, device):
        # [FUNCTION DOC] Routes graphs 0 and 2 of a four-graph function to inductor and
        # gives each a different inductor config override. Asserts two backend overrides
        # and two config patches were recorded, each containing one of the expected patches.
        """
        Test multiple config overrides applied to different graphs with backend override.

        Scenario: Default backend is eager, override graphs 0,2 to use inductor,
        and apply different config overrides to each.
        """
        from torch._dynamo.graph_id_filter import (
            _create_backend_router,
            _create_inductor_config_router,
        )

        torch._dynamo.reset()
        _create_backend_router.cache_clear()
        _create_inductor_config_router.cache_clear()

        backends_used: list[str] = []
        configs_applied: list[dict] = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Four-graph function (three explicit graph breaks):
            # add one, multiply by two, subtract one, divide by two.
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            x = x - 1
            torch._dynamo.graph_break()
            x = x / 2
            return x

        from torch._dynamo import output_graph

        original_wrap = output_graph._wrap_with_inductor_config

        def tracking_wrap(compiler_fn, config_patches):
            # [FUNCTION DOC] HELPER: Records the config patches and delegates to the
            # original wrapper, returning its result.
            configs_applied.append(config_patches)
            return original_wrap(compiler_fn, config_patches)

        # Build backend tracking map
        backend_override = "0,2:inductor"
        backend_str_map: dict[int, str] = {}
        from torch._dynamo.graph_id_filter import GraphIdFilter

        for rule_str in backend_override.split(";"):
            if ":" not in rule_str:
                continue
            colon_idx = rule_str.find(":")
            filter_str = rule_str[:colon_idx].strip()
            backend_str = rule_str[colon_idx + 1 :].strip()
            gf = GraphIdFilter(filter_str)
            for gid in range(10):
                if gid in gf and gid not in backend_str_map:
                    backend_str_map[gid] = backend_str

        from torch._dynamo.graph_id_filter import get_backend_override_for_compile_id

        original_get_backend = get_backend_override_for_compile_id

        def tracking_get_backend(compile_id, config_str):
            # [FUNCTION DOC] HELPER: Calls the real backend override lookup; if an override
            # is found for a known frame id, records its backend name. Returns the result.
            result = original_get_backend(compile_id, config_str)
            if result is not None and compile_id.frame_id in backend_str_map:
                backends_used.append(backend_str_map[compile_id.frame_id])
            return result

        # Use both overrides:
        # - Backend: graphs 0,2 use inductor (graphs 1,3 stay with eager)
        # - Config: graph 0 disables cudagraphs, graph 2 disables skip_dynamic_graphs
        with (
            patch.object(
                torch._dynamo.config,
                "debug_backend_override",
                backend_override,
            ),
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                "0:triton.cudagraphs=False;2:triton.cudagraph_skip_dynamic_graphs=False",
            ),
            patch(
                "torch._dynamo.output_graph.get_backend_override_for_compile_id",
                tracking_get_backend,
            ),
            patch.object(output_graph, "_wrap_with_inductor_config", tracking_wrap),
        ):
            compiled_fn = torch.compile(fn, backend="eager")
            compiled_fn(torch.randn(10, device=device))

        self.assertEqual(len(backends_used), 2)
        self.assertEqual(backends_used, ["inductor", "inductor"])

        self.assertEqual(len(configs_applied), 2)
        self.assertIn({"triton.cudagraphs": False}, configs_applied)
        self.assertIn({"triton.cudagraph_skip_dynamic_graphs": False}, configs_applied)

    def test_config_override_backward_propagation(self, device):
        # [FUNCTION DOC] Compiles a three-graph function with overlapping inductor config
        # override rules, with autograd cache disabled and compile_fx wrapped so the inner
        # compile snapshots three tracked inductor options per (frame id, is-backward).
        # Asserts every graph has forward and backward snapshots equal to the baseline plus
        # that graph's expected overrides (no leakage across graphs), and that grad exists.
        """
        Verify that inductor config overrides are active at inductor compile
        time for both forward and backward, across multiple graph breaks.
        """
        import torch._functorch.config
        from torch._dynamo.graph_id_filter import _create_inductor_config_router
        from torch._inductor import (
            compile_fx as compile_fx_mod,
            config as inductor_config,
        )

        torch._dynamo.reset()
        _create_inductor_config_router.cache_clear()

        TRACKED_CONFIGS = [
            "triton.cudagraphs",
            "triton.dense_indexing",
            "triton.cudagraph_skip_dynamic_graphs",
        ]

        def _read_config(key):
            # [FUNCTION DOC] HELPER: Returns the inductor config value at a dotted path,
            # following attributes one segment at a time.
            obj = inductor_config
            for part in key.split("."):
                obj = getattr(obj, part)
            return obj

        baseline = {k: _read_config(k) for k in TRACKED_CONFIGS}
        configs_at_compile: dict[tuple[int, bool], dict] = {}

        original_compile_fx = compile_fx_mod.compile_fx
        original_inner_compile = compile_fx_mod.compile_fx_inner

        def tracking_inner_compile(gm, example_inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Snapshots the tracked inductor options under the current
            # compile id and backward flag (taken from the is_backward keyword, default
            # false), then delegates to the real inner compile and returns its result.
            compile_id = torch._guards.CompileContext.current_compile_id()
            is_backward = kwargs.get("is_backward", False)
            snapshot = {k: _read_config(k) for k in TRACKED_CONFIGS}
            configs_at_compile[(compile_id.frame_id, is_backward)] = snapshot
            return original_inner_compile(gm, example_inputs, **kwargs)

        def tracking_compile_fx(model_, example_inputs_, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls the real compile_fx, supplying the tracking inner
            # compile when the caller did not pass one; returns its result.
            # Inject tracking inner_compile so compile_fx's config.patch
            # wrapping covers it for both forward and backward.
            if "inner_compile" not in kwargs:
                kwargs["inner_compile"] = tracking_inner_compile
            return original_compile_fx(model_, example_inputs_, *args, **kwargs)

        def fn(x):
            # [FUNCTION DOC] HELPER: Three-graph function: affine step, sine, then
            # exponential summed to a scalar, with graph breaks between them.
            y = x * 2 + 1
            torch._dynamo.graph_break()
            z = y.sin()
            torch._dynamo.graph_break()
            return z.exp().sum()

        # Overlapping rules (all three configs default to False):
        config_override = (
            ">=1:triton.cudagraphs=True;"
            "0-1:triton.dense_indexing=True;"
            "1:triton.cudagraph_skip_dynamic_graphs=True"
        )
        expected_overrides = {
            0: {"triton.dense_indexing": True},
            1: {
                "triton.cudagraphs": True,
                "triton.dense_indexing": True,
                "triton.cudagraph_skip_dynamic_graphs": True,
            },
            2: {
                "triton.cudagraphs": True,
            },
        }

        with (
            patch.object(
                torch._dynamo.config,
                "debug_inductor_config_override",
                config_override,
            ),
            patch.object(compile_fx_mod, "compile_fx", tracking_compile_fx),
            patch.object(torch._functorch.config, "enable_autograd_cache", False),
        ):
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            x = torch.randn(10, device=device, requires_grad=True)
            result = compiled_fn(x)
            result.backward()

        # Verify each graph has fwd+bwd, correct overrides, no cross-graph
        # leak, and identical configs for forward and backward.
        for gid in range(3):
            self.assertIn(
                (gid, False),
                configs_at_compile,
                lambda msg: f"{msg}\ngraph {gid} fwd missing",
            )
            self.assertIn(
                (gid, True),
                configs_at_compile,
                lambda msg: f"{msg}\ngraph {gid} bwd missing",
            )
            expected = {**baseline, **expected_overrides[gid]}
            for is_bw in [False, True]:
                phase = "backward" if is_bw else "forward"
                self.assertEqual(
                    configs_at_compile[(gid, is_bw)],
                    expected,
                    lambda msg: f"{msg}\ngraph {gid} {phase}: config mismatch",
                )

        self.assertIsNotNone(x.grad)


instantiate_device_type_tests(
    TestInductorConfigOverrideIntegration,
    globals(),
    only_for=["cpu", "cuda", "xpu"],
    allow_xpu=True,
)


class TestConfigOverrideValidation(TestCase):
    # [CLASS DOC] Tests that invalid names in the debug override configs (backend, dynamo
    # option, inductor option) raise ValueError when compiling. Clears the validation
    # caches and resets dynamo around each test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, clears the cached validators for
        # backend names, dynamo config keys and inductor config keys, and resets dynamo.
        super().setUp()
        from torch._dynamo.graph_id_filter import (
            _validate_backend_names,
            _validate_dynamo_config_keys,
            _validate_inductor_config_keys,
        )

        _validate_backend_names.cache_clear()
        _validate_dynamo_config_keys.cache_clear()
        _validate_inductor_config_keys.cache_clear()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and calls the base tearDown.
        torch._dynamo.reset()
        super().tearDown()

    @torch._dynamo.config.patch(
        debug_backend_override="0:not_a_real_backend",
    )
    def test_invalid_backend_raises_on_compile(self):
        # [FUNCTION DOC] With an override naming an unknown backend, compiling and calling
        # a small function raises ValueError mentioning the bad name.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        with self.assertRaisesRegex(ValueError, "not_a_real_backend"):
            torch.compile(fn, backend="eager")(torch.randn(4))

    @torch._dynamo.config.patch(
        debug_dynamo_config_override="0:nonexistent_dynamo_option=True",
    )
    def test_invalid_dynamo_config_raises_on_compile(self):
        # [FUNCTION DOC] With a dynamo config override naming a nonexistent option,
        # compiling and calling a small function raises ValueError mentioning that option.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        with self.assertRaisesRegex(ValueError, "nonexistent_dynamo_option"):
            torch.compile(fn, backend="eager")(torch.randn(4))

    @torch._dynamo.config.patch(
        debug_inductor_config_override="0:nonexistent_inductor_option=True",
    )
    def test_invalid_inductor_config_raises_on_compile(self):
        # [FUNCTION DOC] With an inductor config override naming a nonexistent option,
        # compiling and calling a small function raises ValueError mentioning that option.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        with self.assertRaisesRegex(ValueError, "nonexistent_inductor_option"):
            torch.compile(fn, backend="eager")(torch.randn(4))


class TestDynamoConfigOverrideIntegration(TestCase):
    # [CLASS DOC] Tests per-graph dynamo config overrides (applied during each graph's
    # compile) and the warning emitted when such a router is created. Resets dynamo
    # around each test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp and resets dynamo state.
        super().setUp()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and calls the base tearDown.
        torch._dynamo.reset()
        super().tearDown()

    @torch._dynamo.config.patch(
        specialize_float=False,
        verbose=False,
        debug_dynamo_config_override=(
            "0:specialize_float=True;1:verbose=True,recompile_limit=10"
        ),
    )
    def test_dynamo_config_override_per_graph(self):
        # [FUNCTION DOC] Compiles a three-graph function with a capturing backend that records
        # three dynamo options per frame id while compiling. Asserts the overrides apply only
        # to their targeted graphs (graph 0: float specialization on; graph 1: verbose on and
        # recompile limit 10; graph 2: base values).
        """Per-graph dynamo config overrides target the right graphs.

        Graph 0: specialize_float overridden True (base False)
        Graph 1: verbose+recompile_limit overridden (multiple keys)
        Graph 2: no override, keeps base values
        """
        from torch._dynamo.graph_id_filter import _create_dynamo_config_router

        _create_dynamo_config_router.cache_clear()

        observed: dict[int, dict] = {}

        def capturing_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that stores the current values of three dynamo
            # options under the current frame id and returns the graph module unchanged.
            fid = torch._guards.CompileContext.current_compile_id().frame_id
            observed[fid] = {
                "specialize_float": torch._dynamo.config.specialize_float,
                "verbose": torch._dynamo.config.verbose,
                "recompile_limit": torch._dynamo.config.recompile_limit,
            }
            return gm

        def fn(x):
            # [FUNCTION DOC] HELPER: Three-graph function: add one, multiply by two, subtract
            # one, with graph breaks between.
            x = x + 1
            torch._dynamo.graph_break()
            x = x * 2
            torch._dynamo.graph_break()
            return x - 1

        torch.compile(fn, backend=capturing_backend)(torch.randn(4))

        self.assertTrue(observed[0]["specialize_float"])
        self.assertFalse(observed[0]["verbose"])

        self.assertFalse(observed[1]["specialize_float"])
        self.assertTrue(observed[1]["verbose"])
        self.assertEqual(observed[1]["recompile_limit"], 10)

        self.assertFalse(observed[2]["specialize_float"])
        self.assertFalse(observed[2]["verbose"])

    def test_dynamo_config_override_warning(self):
        # [FUNCTION DOC] Clears the dynamo config router cache and asserts that creating a
        # router from a rule string emits a UserWarning mentioning the override environment
        # variable.
        from torch._dynamo.graph_id_filter import _create_dynamo_config_router

        _create_dynamo_config_router.cache_clear()
        with self.assertWarnsRegex(
            UserWarning, "TORCH_COMPILE_OVERRIDE_DYNAMO_CONFIGS"
        ):
            _create_dynamo_config_router("0:specialize_float=True")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
