# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for make_fx with the dynamic_shapes argument built on the ShapesSpec /
# ParamsSpec spec API from torch.fx.experimental.dynamic_spec. Covers tensor dims declared
# dynamic via ShapeVar (traced as unbacked SymInts), static dims, integer inputs via IntVar,
# shared and derived variables, assumptions, bounds, optimization hints, container specs and the
# dynamic_spec decorator. Defines small helper modules and graph-inspection helpers plus one
# TestCase; has no import-time side effects beyond imports.

import itertools

import torch
from torch.fx.experimental.dynamic_spec import (
    DictSpec as DICT,
    IntVar,
    ObjectSpec as OBJ,
    ParamsSpec as PARAMS,
    SeqSpec as L,
    ShapesSpec,
    ShapeVar as VAR,
    STATIC,
    TensorSpec as T,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def _reset_uid_counter():
    # [FUNCTION DOC] HELPER: Resets the class-level uid counter of IntVar to a fresh counter
    # starting at zero so variable uids are deterministic per test. Mutates global state;
    # returns nothing.
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _user_input_placeholder_vals(gm):
    # [FUNCTION DOC] HELPER: Walks the nodes of the given GraphModule graph, skips every node
    # that is not a placeholder, and reads its meta value (falling back to the example_value
    # entry). Keeps only values that are tensors or SymInts and returns them as a list in
    # placeholder order.
    """Return the per-placeholder ``meta['val']`` (tensor or SymInt) for
    user inputs of ``gm``."""
    out = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(val, (torch.Tensor, torch.SymInt)):
            out.append(val)
    return out


def _first_tensor_placeholder_shape(gm):
    # [FUNCTION DOC] HELPER: Returns the shape of the first tensor-valued user-input placeholder
    # of the given GraphModule, using the helper that lists placeholder values. Raises
    # AssertionError if no placeholder holds a tensor.
    for val in _user_input_placeholder_vals(gm):
        if isinstance(val, torch.Tensor):
            return val.shape
    raise AssertionError("no tensor placeholder found")


class _ModX(torch.nn.Module):
    # [CLASS DOC] HELPER: Minimal one-input module used as a test fixture; reduces its input
    # along dimension zero.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the sum of x over dimension zero.
        return x.sum(0)


class _ModXYIndep(torch.nn.Module):
    # [CLASS DOC] HELPER: Two-input fixture module whose ops never relate the two inputs'
    # shapes, so distinct symbolic dims stay distinct in the traced graph.
    """Two-input model whose ops don't unify the input shapes (no broadcast)."""

    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Returns a tuple with the dimension-zero sum of x and the
        # dimension-zero sum of y.
        return x.sum(0), y.sum(0)


class _ModXN(torch.nn.Module):
    # [CLASS DOC] HELPER: Fixture module taking a tensor and a scalar (possibly symbolic int).
    def forward(self, x, n):
        # [FUNCTION DOC] HELPER: Returns x plus n, broadcasting the scalar.
        return x + n


class TestMakeFxDynamicSpec(TestCase):
    # [CLASS DOC] Test case for make_fx dynamic_shapes support. No instance state; each test is
    # independent and relies on setUp to reset the IntVar uid counter, so graph text with symbol
    # names (u0, u1, ...) is stable.
    """make_fx(dynamic_shapes=...) support for the ShapesSpec / ParamsSpec API."""

    def setUp(self):
        # [FUNCTION DOC] Runs the base setUp and then resets the global IntVar uid counter to
        # zero.
        super().setUp()
        _reset_uid_counter()

    def test_function_tensorspec_shape_var_produces_symint(self):
        # [FUNCTION DOC] Traces a function summing over dim zero in fake mode with a TensorSpec
        # whose first dim is a named ShapeVar and second is static. Checks the placeholder's
        # first dim is a SymInt with exactly one free unbacked symbol, the static dim stays 3,
        # and the printed graph matches the expected text (including the non-negativity runtime
        # assertion).
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x summed over dimension zero.
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)
        # Static dim stays an int.
        self.assertEqual(int(shape[1]), 3)

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x_1, [0]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_module_paramsspec_shape_var_produces_symint(self):
        # [FUNCTION DOC] Same as the function-based case but traces an nn.Module instance (a
        # dim-zero-sum fixture) with a dict spec. Checks the first placeholder dim is a SymInt
        # with unbacked free symbols.
        gm = make_fx(
            _ModX(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertTrue(free_unbacked_symbols(shape[0]))

    def test_all_static_spec_keeps_dims_static(self):
        # [FUNCTION DOC] Traces with a spec where every dim is static; checks both placeholder
        # dims remain concrete ints 4 and 5 (no symbols introduced).
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x plus one.
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC, STATIC])},
        )(torch.randn(4, 5))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertEqual(int(shape[0]), 4)
        self.assertEqual(int(shape[1]), 5)

    def test_static_dim_mismatch_raises(self):
        # [FUNCTION DOC] Declares dim 1 as static value 3 but supplies an input whose dim 1 is
        # 5; checks make_fx raises ValueError with the message about the static dim mismatch.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x plus one (never actually traced
            # since the spec check fails first).
            return x + 1

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([VAR("batch"), 3])},
            )(torch.randn(4, 5))

    def test_no_params_spec_short_circuits(self):
        # [FUNCTION DOC] Passes an empty ShapesSpec (no ShapeVars declared); checks tracing
        # succeeds and the input dims stay static at 4 and 5.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x plus one.
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes=ShapesSpec(),
        )(torch.randn(4, 5))

        # No ShapeVars declared -> dims remain static.
        shape = _first_tensor_placeholder_shape(gm)
        self.assertEqual(int(shape[0]), 4)
        self.assertEqual(int(shape[1]), 5)

    def test_two_tensors_independent_shape_vars(self):
        # [FUNCTION DOC] Uses two different ShapeVars on two inputs with a model that does not
        # relate them; checks the printed graph shows two distinct symbols u0 and u1, each with
        # its own non-negativity assertion.
        """Distinct ShapeVars across two inputs each produce their own SymInt
        (use a model whose ops don't unify the two shapes, so the symbols
        remain distinct in the printed graph)."""
        gm = make_fx(
            _ModXYIndep(),
            tracing_mode="fake",
            dynamic_shapes={
                "x": T([VAR("a"), STATIC]),
                "y": T([VAR("b"), STATIC]),
            },
        )(torch.randn(4, 3), torch.randn(7, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u1, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0]);  arg0_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0]);  arg1_1 = None
        return (sum_1, sum_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_same_shape_var_object_links_dims(self):
        # [FUNCTION DOC] Uses one shared ShapeVar object for dim zero of both inputs; checks the
        # printed graph uses the single symbol u0 for both and adds an equality runtime
        # assertion between the two sizes.
        """Passing the SAME ``ShapeVar`` instance for both inputs makes
        them share one symbol in the traced graph (``u0`` for both),
        even with the independent-ops model."""
        a = VAR("a")
        gm = make_fx(
            _ModXYIndep(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([a, STATIC]), "y": T([a, STATIC])},
        )(torch.randn(5, 3), torch.randn(5, 3))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u0)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(u0 >= 0)" = sym_size_int_1 >= 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        eq: "Sym(True)" = sym_size_int_1 == sym_size_int;  sym_size_int_1 = sym_size_int = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(eq, "Runtime assertion failed for expression Eq(u1, u0) on node 'eq'");  eq = _assert_scalar_default_2 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0]);  arg0_1 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_1, [0]);  arg1_1 = None
        return (sum_1, sum_2)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_int_input_with_intvar_spec(self):
        # [FUNCTION DOC] Declares an integer input via IntVar next to a static tensor spec;
        # checks exactly one SymInt placeholder exists, it has free unbacked symbols and its
        # expression is u0 (unbacked, not a backed symbol).
        """An int input declared via ``IntVar`` becomes an *unbacked* SymInt
        (``u0``), not a backed one (``s0``) — the ShapesSpec path is
        unbacked-only for soundness."""
        gm = make_fx(
            _ModXN(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC]), "n": IntVar("n")},
        )(torch.randn(4), 7)

        sym_placeholders = [
            val
            for val in _user_input_placeholder_vals(gm)
            if isinstance(val, torch.SymInt)
        ]
        self.assertEqual(len(sym_placeholders), 1)
        sym = sym_placeholders[0]
        self.assertTrue(free_unbacked_symbols(sym))
        self.assertEqual(str(sym.node.expr), "u0")

    def test_dict_shorthand_accepted(self):
        # [FUNCTION DOC] Checks that a plain dict is accepted as shorthand for a spec: the first
        # dim becomes an unbacked SymInt and the static dim stays 3.
        # KNOWN ISSUE: the test passes a plain dict exactly like most other tests here, so it
        # does not check anything distinct from them.
        """A bare ``dict`` is accepted as shorthand for
        ``ShapesSpec(PARAMS(dict))``."""

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x summed over dimension zero.
            return x.sum(0)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(8, 3))

        shape = _first_tensor_placeholder_shape(gm)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertTrue(free_unbacked_symbols(shape[0]))
        self.assertEqual(int(shape[1]), 3)

    def test_requires_tracing_mode_fake(self):
        # [FUNCTION DOC] For tracing modes real and symbolic (any mode other than fake), checks
        # make_fx raises ValueError saying dynamic_shapes requires tracing_mode fake.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x plus one.
            return x + 1

        for bad_mode in ("real", "symbolic"):
            with self.assertRaisesRegex(
                ValueError,
                r"make_fx\(dynamic_shapes=\.\.\.\) requires tracing_mode='fake'",
            ):
                make_fx(
                    f,
                    tracing_mode=bad_mode,
                    dynamic_shapes={"x": T([VAR("batch"), STATIC])},
                )(torch.randn(4, 5))

    def test_invalid_spec_type_raises(self):
        # [FUNCTION DOC] Passes an integer as dynamic_shapes; checks make_fx raises TypeError
        # saying a dict, ShapesSpec or ParamsSpec is expected.
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x unchanged.
            return x

        with self.assertRaisesRegex(
            TypeError,
            r"dynamic spec expects a dict, ShapesSpec, or ParamsSpec",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes=42,  # neither dict nor Spec object
            )(torch.randn(2))

    def test_symbolic_dim_flows_through_ops(self):
        # [FUNCTION DOC] Traces a dim-one sum with a dynamic first dim and checks the printed
        # graph shows the symbol u0 propagating into the output shape. Then runs the resulting
        # GraphModule on inputs with first dims 8 and 13 to confirm it accepts varied sizes.
        """The ShapeVar symbol propagates into op outputs in the traced graph,
        and the resulting GraphModule runs with varied sizes."""

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x summed over dimension one.
            return x.sum(1)

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch"), STATIC])},
        )(torch.randn(6, 4))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class f(torch.nn.Module):
    def forward(self, x_1: "f32[u0, 4]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[u0]" = torch.ops.aten.sum.dim_IntList(x_1, [1]);  x_1 = None
        return sum_1""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        gm(torch.randn(8, 4))
        gm(torch.randn(13, 4))

    def test_shape_var_min_bound_constrains_range(self):
        # [FUNCTION DOC] Declares a ShapeVar with min 4 and max 128 and checks the SymInt's own
        # shape env records a value range with lower bound 4 and upper bound 128 (the graph
        # module has no shape_env in fake mode, so the SymInt's node is used).
        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x plus one.
            return x + 1

        gm = make_fx(
            f,
            tracing_mode="fake",
            dynamic_shapes={"x": T([VAR("batch", min=4, max=128), STATIC])},
        )(torch.randn(10, 3))

        shape = _first_tensor_placeholder_shape(gm)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        self.assertTrue(free_unbacked_symbols(sym))
        # ``gm.shape_env`` is only set in ``tracing_mode='symbolic'``,
        # so reach into the SymInt's own shape env instead.
        shape_env = sym.node.shape_env
        vr = shape_env.var_to_range[sym.node.expr]
        self.assertEqual(int(vr.lower), 4)
        self.assertEqual(int(vr.upper), 128)

    def test_branching_on_shape_var_dim_raises_dde(self):
        # [FUNCTION DOC] Traces a function that branches on whether the first dim exceeds five,
        # with an unbounded ShapeVar; checks tracing raises GuardOnDataDependentSymNode (could
        # not guard on data-dependent expression).
        """Branching on a ShapeVar dim with no bounds raises a
        data-dependent error (mirrors ``test_unbacked_raises_dde_on_branching``
        in the strict-export suite)."""

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; branches on the size of dim zero being
            # greater than five and returns x plus one or x minus one.
            if x.size(0) > 5:
                return x + 1
            return x - 1

        with self.assertRaisesRegex(
            GuardOnDataDependentSymNode,
            r"Could not guard on data-dependent expression",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([VAR("batch"), STATIC])},
            )(torch.randn(8, 3))

    def test_derived_dim_in_traced_graph(self):
        # [FUNCTION DOC] Declares x with ShapeVar B and y with B times two, traces a module that
        # branches on the relation between their dim-zero sizes, and checks the graph keeps the
        # derived dim (2*u0) and takes the add branch with an equality assertion.
        """Derived expressions (``B * 2``) materialize as derived dim
        expressions in the traced graph. The ``add`` (not ``sub``) proves the
        relation ``2*u0 == y.size(0)`` held at trace time."""
        B = VAR("batch")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Nested fixture module whose forward branches on whether twice
            # x's dim zero equals y's dim zero.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x plus the sum of y when twice the
                # size of x's dim zero equals the size of y's dim zero, otherwise their
                # difference. Branching resolves statically only if the derived relation is
                # known.
                # Branching on the derived relation proves it is known at
                # trace time (otherwise this would be data-dependent).
                if x.size(0) * 2 == y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes={
                "x": T([B, STATIC]),
                "y": T([B * 2, STATIC]),
            },
        )(torch.randn(4, 3), torch.randn(8, 5))

        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[u0, 3]", arg1_1: "f32[2*u0, 5]"):
        sym_size_int_2: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_1, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int_2 >= 0
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_3: "Sym(2*u0)" = torch.ops.aten.sym_size.int(arg1_1, 0)
        ge_1: "Sym(2*u0 >= 0)" = sym_size_int_3 >= 0
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        mul_1: "Sym(2*u0)" = 2 * sym_size_int_2;  sym_size_int_2 = None
        eq_1: "Sym(True)" = sym_size_int_3 == mul_1;  sym_size_int_3 = mul_1 = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(eq_1, "Runtime assertion failed for expression Eq(u1, 2*u0) on node 'eq_1'");  eq_1 = _assert_scalar_default_2 = None
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
        sum_2: "f32[]" = torch.ops.aten.sum.default(arg1_1);  arg1_1 = None
        add: "f32[]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        return add""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_unbound_derived_dim_raises(self):
        # [FUNCTION DOC] Uses a derived dim (A times two) where the underlying variable A never
        # appears as a plain dim; checks make_fx raises ValueError about pending checks
        # referencing unbound IntVar a, showing the spec wiring finalization runs even without
        # assumptions.
        """A derived dim (``A * 2``) whose ``IntVar`` never appears as a bare
        slot leaves a pending check that can never drain. ``_finalize_spec_wiring``
        must run for any spec (not just ones with assumptions) so this surfaces
        as an error instead of silently dropping the relation."""
        A = VAR("a")

        def f(x):
            # [FUNCTION DOC] HELPER: Traced function; returns x summed over dimension zero.
            return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"pending check\(s\) reference unbound IntVar\(s\) \['a'\]",
        ):
            make_fx(
                f,
                tracing_mode="fake",
                dynamic_shapes={"x": T([A * 2, STATIC])},
            )(torch.randn(8, 3))

    def test_assumption(self):
        # [FUNCTION DOC] Declares an assumption that variable A is greater than B alongside two
        # tensor specs. Checks the traced graph took the add branch (not subtract), contains a
        # scalar assertion node, runs for a=5,b=3, and that running with a=2,b=3 raises
        # RuntimeError with Runtime assertion failed.
        """A relational ``assumptions=[A > B]`` is wired into the shape env:
        the assumed-true branch is taken at trace time and the relation
        is materialized as a runtime assertion in the graph."""
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Nested fixture module whose forward branches on x's dim zero
            # being greater than y's.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of x plus the sum of y if the dim-zero
                # size of x exceeds that of y, otherwise their difference.
                # Branching on the assumed relation proves it is known at
                # trace time.
                if x.size(0) > y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes=ShapesSpec(
                params=PARAMS({"x": T([A, STATIC]), "y": T([B, STATIC])}),
                assumptions=[A > B],
            ),
        )(torch.randn(5, 2), torch.randn(3, 2))

        # The assumption resolves the branch at trace time: ``A > B`` is
        # assumed true, so the ``+`` branch is taken.
        self.assertTrue(
            any(
                node.op == "call_function" and "aten.add" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # Sanity: the other branch must NOT have been taken.
        self.assertFalse(
            any(
                node.op == "call_function" and "aten.sub" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # The assumption is materialized as a runtime assertion node in the
        # graph and enforced when the graph is executed.
        self.assertTrue(
            any(
                node.op == "call_function" and "_assert_scalar" in str(node.target)
                for node in gm.graph.nodes
            )
        )
        # Correct: a=5 > b=3.
        gm(torch.randn(5, 2), torch.randn(3, 2))
        # Violation: a=2 not > b=3.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            gm(torch.randn(2, 2), torch.randn(3, 2))

    def test_tensor_dim_optimization_hint(self):
        # [FUNCTION DOC] Gives a ShapeVar an optimization hint of 32 and checks the SymInt's
        # shape env stores that value in its hint-override mapping for the symbol.
        """A ShapeVar's ``optimization_hint`` lands in the shape env's
        ``var_to_hint_override`` for the tensor dim's symbol."""
        b = VAR("batch", optimization_hint=32)
        gm = make_fx(
            _ModX(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([b, STATIC])},
        )(torch.randn(8, 3))

        sym = _first_tensor_placeholder_shape(gm)[0]
        self.assertIsInstance(sym, torch.SymInt)
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(sym.node.expr), 32)

    def test_scalar_input_optimization_hint(self):
        # [FUNCTION DOC] Gives an IntVar an optimization hint of 512 for a scalar input and
        # checks the SymInt placeholder's shape env stores that value in its hint-override
        # mapping.
        """An ``IntVar``'s ``optimization_hint`` lands in the shape env's
        ``var_to_hint_override`` for a scalar input's symbol."""
        gm = make_fx(
            _ModXN(),
            tracing_mode="fake",
            dynamic_shapes={"x": T([STATIC]), "n": IntVar("n", optimization_hint=512)},
        )(torch.randn(4), 7)

        sym = next(
            v for v in _user_input_placeholder_vals(gm) if isinstance(v, torch.SymInt)
        )
        self.assertEqual(
            sym.node.shape_env.var_to_hint_override.get(sym.node.expr), 512
        )

    def test_container_specs_dict_seq_object(self):
        # [FUNCTION DOC] Traces a module taking a dict, a list and a namedtuple with DictSpec,
        # SeqSpec and ObjectSpec each marking one nested tensor's first dim dynamic and omitting
        # or marking others static. Checks the printed graph shows unbacked dims u0, u1, u2 only
        # for those and static shapes for the rest.
        """DictSpec / SeqSpec / ObjectSpec each mark a nested tensor dim
        dynamic (``u0``/``u1``/``u2``); unspecified entries stay static."""
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Nested fixture module consuming a dict, a sequence and a
            # namedtuple.
            def forward(self, d, xs, p):
                # [FUNCTION DOC] HELPER: Returns the sum of dim-zero sums of the dict entry b,
                # sequence element one and the namedtuple field second.
                return d["b"].sum(0) + xs[1].sum(0) + p.second.sum(0)

        gm = make_fx(
            M(),
            tracing_mode="fake",
            dynamic_shapes={
                "d": DICT({"b": T([VAR("B"), STATIC])}),  # "a" omitted -> static
                "xs": L([STATIC, T([VAR("X"), STATIC])]),  # pos 0 static
                "p": OBJ({"second": T([VAR("P"), STATIC])}),  # "first" -> static
            },
        )(
            {"a": torch.randn(7, 3), "b": torch.randn(8, 3)},
            [torch.randn(4, 3), torch.randn(5, 3)],
            Pair(torch.randn(6, 3), torch.randn(9, 3)),
        )

        # d["b"], xs[1], p.second -> unbacked (u0/u1/u2); the rest stay static.
        self.assertExpectedInline(
            gm.print_readable(print_output=False).strip(),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0, arg1, arg2):
        arg0_1: "f32[7, 3]"; arg0_2: "f32[u0, 3]"; arg1_1: "f32[4, 3]"; arg1_2: "f32[u1, 3]"; arg2_1: "f32[6, 3]"; arg2_2: "f32[u2, 3]";
        arg0_1, arg0_2, arg1_1, arg1_2, arg2_1, arg2_2, = fx_pytree.tree_flatten_spec([arg0, arg1, arg2], self._in_spec)
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(arg0_2, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(arg1_2, 0)
        ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
        _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
        sym_size_int_2: "Sym(u2)" = torch.ops.aten.sym_size.int(arg2_2, 0)
        ge_2: "Sym(u2 >= 0)" = sym_size_int_2 >= 0;  sym_size_int_2 = None
        _assert_scalar_default_2 = torch.ops.aten._assert_scalar.default(ge_2, "Runtime assertion failed for expression u2 >= 0 on node 'ge_2'");  ge_2 = _assert_scalar_default_2 = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg0_2, [0]);  arg0_2 = None
        sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg1_2, [0]);  arg1_2 = None
        add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
        sum_3: "f32[3]" = torch.ops.aten.sum.dim_IntList(arg2_2, [0]);  arg2_2 = None
        add_1: "f32[3]" = torch.ops.aten.add.Tensor(add, sum_3);  add = sum_3 = None
        return pytree.tree_unflatten([add_1], self._out_spec)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    # ---- @dynamic_spec(...) decorator (auto-attach) ----

    def test_dynamic_spec_decorator_dict_form(self):
        # [FUNCTION DOC] Attaches the dynamic_spec decorator with a dict spec to a function and
        # traces it without a dynamic_shapes argument; checks the single placeholder tensor has
        # one free unbacked symbol in its first dim.
        """``@dynamic_spec({"x": ...})`` on a function is auto-applied by make_fx."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": T([VAR("B"), STATIC])})
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated traced function; returns x summed over dimension
            # zero.
            return x.sum(0)

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(8, 3))
        (val,) = _user_input_placeholder_vals(gm)
        self.assertIsInstance(val, torch.Tensor)
        self.assertEqual(len(free_unbacked_symbols(val.shape[0])), 1)

    def test_dynamic_spec_decorator_with_assumptions(self):
        # [FUNCTION DOC] Attaches the decorator with a ShapesSpec holding an assumption that
        # variable B is even. Checks tracing the function that branches on evenness succeeds and
        # the printed graph contains the modulo expression and a scalar assertion.
        """``@dynamic_spec(ShapesSpec(params=..., assumptions=...))`` is auto-applied
        and the assumption is wired into the shape env: branching on the
        assumed relation resolves statically (no DDE) and the assertion
        appears in the traced graph."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        B = VAR("batch")

        @dynamic_spec(
            ShapesSpec(
                PARAMS({"x": T([B, STATIC])}),
                assumptions=[B % 2 == 0],
            )
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated traced function; returns the dim-zero sum of x,
            # negated when dim zero is odd (the branch resolves statically via the assumption).
            # Branching on the assumption: without it, this would raise a DDE.
            if x.shape[0] % 2 == 0:
                return x.sum(0)
            return x.sum(0) * -1

        gm = make_fx(fn, tracing_mode="fake")(torch.randn(8, 3))
        # The assumption (``Eq(Mod(u0, 2), 0)``) is also materialized as a
        # runtime assertion in the traced graph.
        graph_repr = gm.print_readable(print_output=False)
        self.assertIn("Mod(u0, 2)", graph_repr)
        self.assertIn("_assert_scalar", graph_repr)

    def test_dynamic_spec_decorator_and_explicit_kwarg_raises(self):
        # [FUNCTION DOC] Attaches the decorator and also passes dynamic_shapes to make_fx;
        # checks make_fx raises ValueError because both sources of specs are ambiguous.
        """Mixing the decorator with ``make_fx(..., dynamic_shapes=...)``
        is ambiguous and must raise."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": T([VAR("B"), STATIC])})
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated traced function; returns x summed over dimension
            # zero.
            return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"`@dynamic_spec\(\.\.\.\)` is attached.*AND a `dynamic_shapes=`",
        ):
            make_fx(
                fn,
                tracing_mode="fake",
                dynamic_shapes=PARAMS({"x": T([VAR("Z"), STATIC])}),
            )(torch.randn(8, 3))


if __name__ == "__main__":
    run_tests()
