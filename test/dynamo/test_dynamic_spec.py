# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the dynamic shape specification API in torch.fx.experimental.dynamic_spec
# (ShapeVar, IntVar, TensorSpec, ParamsSpec, ShapesSpec, ObjectSpec, DictSpec, SeqSpec, STATIC)
# and its use through the dynamic_shapes argument of torch.compile and the dynamic spec decorator.
# Covers construction and repr/json forms, symbolic arithmetic on IntVar, derived dimensions,
# assumptions, deduplication of shared variables, container specs and error cases. Compile tests
# mostly use the eager backend and inspect recorded graphs. No import-time side effects.

import itertools

import torch
import torch._dynamo
import torch._dynamo.testing
import torch.fx.experimental._config as _fx_experimental_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.dynamic_spec import (
    DictSpec,
    IntVar,
    ObjectSpec,
    ParamsSpec,
    SeqSpec,
    ShapesSpec,
    ShapeVar,
    STATIC,
    TensorSpec,
)
from torch.fx.experimental.symbolic_shapes import (
    free_unbacked_symbols,
    GuardOnDataDependentSymNode,
)


def _reset_uid_counter():
    # [FUNCTION DOC] HELPER: Resets the class-level uid counter on IntVar to a fresh counter
    # starting at zero, so variable reprs such as name#0 are deterministic within each test.
    # Mutates global state.
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _tensor_placeholder_shape(gm):
    # [FUNCTION DOC] HELPER: Scans the graph nodes of the given graph module and returns the shape
    # of the example value of the first placeholder whose example value is a tensor. Raises
    # AssertionError if no placeholder holds a tensor.
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


def _tensor_placeholders(gm):
    # [FUNCTION DOC] HELPER: Returns a list of all placeholder nodes of the graph module whose
    # recorded example value is a tensor, in graph order.
    out = []
    for n in gm.graph.nodes:
        if n.op == "placeholder" and isinstance(
            n.meta.get("example_value"), torch.Tensor
        ):
            out.append(n)
    return out


class TestShapeVarConstruction(TestCase):
    # [CLASS DOC] Tests for constructing ShapeVar objects: names, bounds, hints, reprs and unique
    # ids. setUp resets the global uid counter.
    """Construction of ShapeVar."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_basic(self):
        # [FUNCTION DOC] A named ShapeVar keeps its name, has min 0 and no max or optimization
        # hint.
        s = ShapeVar("batch")
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.min, 0)
        self.assertIsNone(s.max)
        self.assertIsNone(s.optimization_hint)

    def test_with_max(self):
        # [FUNCTION DOC] The max argument is stored on the ShapeVar.
        s = ShapeVar("batch", max=64)
        self.assertEqual(s.max, 64)

    def test_with_optimization_hint(self):
        # [FUNCTION DOC] Max and optimization hint are both stored on the ShapeVar.
        s = ShapeVar("seq", max=2048, optimization_hint=512)
        self.assertEqual(s.max, 2048)
        self.assertEqual(s.optimization_hint, 512)

    def test_anonymous_name(self):
        # [FUNCTION DOC] Without a name the ShapeVar name is reported as anon.
        s = ShapeVar()
        self.assertEqual(s.name, "anon")

    def test_anonymous_repr_uses_id(self):
        # [FUNCTION DOC] An anonymous ShapeVar repr includes the anon name, uid 0 and the default
        # min 0.
        s = ShapeVar()
        self.assertEqual(repr(s), "ShapeVar(anon#0, min=0)")

    def test_anonymous_specs_have_distinct_reprs(self):
        # [FUNCTION DOC] Two anonymous ShapeVars get different uids and therefore different reprs.
        a = ShapeVar()
        b = ShapeVar()
        # Both have name=None but distinct ids -> distinct reprs.
        self.assertNotEqual(repr(a), repr(b))

    def test_same_named_specs_have_distinct_reprs(self):
        # [FUNCTION DOC] Two ShapeVars with the same name still get different uids and different
        # reprs.
        a = ShapeVar("batch")
        b = ShapeVar("batch")
        # Same name, distinct ids -> distinct reprs.
        self.assertNotEqual(repr(a), repr(b))

    def test_is_instance_of_intvar(self):
        # [FUNCTION DOC] ShapeVar is a subclass of IntVar.
        s = ShapeVar("batch")
        self.assertIsInstance(s, IntVar)

    def test_repr(self):
        # [FUNCTION DOC] Repr of a fully specified ShapeVar lists name#uid, min, max and
        # optimization_hint.
        s = ShapeVar("batch", max=64, optimization_hint=32)
        self.assertEqual(
            repr(s),
            "ShapeVar(batch#0, min=0, max=64, optimization_hint=32)",
        )

    def test_repr_minimal(self):
        # [FUNCTION DOC] Repr of a ShapeVar with only a name lists name#uid and min 0 only.
        s = ShapeVar("x")
        self.assertEqual(repr(s), "ShapeVar(x#0, min=0)")

    def test_implicit_min_is_zero(self):
        # [FUNCTION DOC] The default lower bound of a ShapeVar is zero.
        s = ShapeVar("x")
        self.assertEqual(s.min, 0)

    def test_custom_min(self):
        # [FUNCTION DOC] A custom min value is stored on the ShapeVar.
        s = ShapeVar("x", min=1)
        self.assertEqual(s.min, 1)


class TestIntVarConstruction(TestCase):
    # [CLASS DOC] Tests for constructing IntVar objects (scalar integer variables): names, ranges,
    # hints and reprs. setUp resets the global uid counter.
    """Construction of IntVar."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_basic(self):
        # [FUNCTION DOC] A named IntVar keeps its name; min, max and optimization hint default to
        # None.
        s = IntVar("offset")
        self.assertEqual(s.name, "offset")
        self.assertIsNone(s.min)
        self.assertIsNone(s.max)
        self.assertIsNone(s.optimization_hint)

    def test_with_range(self):
        # [FUNCTION DOC] Explicit min and max are stored on the IntVar.
        s = IntVar("offset", min=-100, max=100)
        self.assertEqual(s.min, -100)
        self.assertEqual(s.max, 100)

    def test_with_optimization_hint(self):
        # [FUNCTION DOC] The optimization hint is stored on the IntVar.
        s = IntVar("size", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s.optimization_hint, 512)

    def test_anonymous_name(self):
        # [FUNCTION DOC] Without a name the IntVar name is reported as anon.
        s = IntVar()
        self.assertEqual(s.name, "anon")

    def test_anonymous_repr_uses_id(self):
        # [FUNCTION DOC] An anonymous IntVar repr includes the anon name and uid 0.
        s = IntVar()
        self.assertEqual(repr(s), "IntVar(anon#0)")

    def test_anonymous_specs_have_distinct_reprs(self):
        # [FUNCTION DOC] Two anonymous IntVars have distinct reprs because of distinct uids.
        a = IntVar()
        b = IntVar()
        self.assertNotEqual(repr(a), repr(b))

    def test_allows_negative_range(self):
        # [FUNCTION DOC] An IntVar may be given an entirely negative range; min and max are kept
        # as given.
        s = IntVar("offset", min=-50, max=-10)
        self.assertEqual(s.min, -50)
        self.assertEqual(s.max, -10)

    def test_repr(self):
        # [FUNCTION DOC] Repr of a fully specified IntVar lists name#uid, min, max and
        # optimization_hint.
        s = IntVar("offset", min=-100, max=100, optimization_hint=0)
        self.assertEqual(
            repr(s),
            "IntVar(offset#0, min=-100, max=100, optimization_hint=0)",
        )

    def test_repr_minimal(self):
        # [FUNCTION DOC] Repr of an IntVar with only a name shows just name#uid.
        s = IntVar("x")
        self.assertEqual(repr(s), "IntVar(x#0)")


class TestIntVarArithmetic(TestCase):
    # [CLASS DOC] Checks that arithmetic, comparison and bitwise operators on IntVar and ShapeVar
    # produce symbolic ints or bools without calling ShapeEnv APIs that the spec environment
    # blocks, and that the operators that need such APIs fail loudly. setUp resets the uid
    # counter.
    """Arithmetic, comparison, and bitwise ops on IntVar/ShapeVar must
    compose without touching the spec ShapeEnv beyond its allowlist.

    The spec env blocks most ShapeEnv public APIs, so any op whose
    sympy lowering reaches blocked APIs (e.g. ``bound_sympy``,
    ``evaluate_sym_node``) must fail loudly here rather than silently
    returning garbage.
    """

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_supported_ops_compose(self):
        # [FUNCTION DOC] Builds one large expression from three IntVars using add, subtract,
        # multiply, power, floor divide, modulo, absolute value, unary plus and minus, and checks
        # it is a SymInt that mentions variable names with uids. Checks true division yields an
        # expression containing an integer true-division marker, bitwise and shift operators give
        # SymInts, sym_max and sym_min compose and show up in the text, all six comparisons give
        # SymBool, and reflected ops with Python ints give SymInts.
        """Build a large expression mixing every supported op and
        verify it produces a well-formed SymInt/SymBool without
        crashing or hitting the spec-env allowlist.
        """
        A = IntVar("a")
        B = IntVar("b")
        C = IntVar("c")

        # Arithmetic: +, -, *, **, //, /, %, abs, unary -/+
        arith = (
            abs(-A + B)
            + (A * 2 + 3 * B - 1)
            + (A**2)
            + (A // 2)
            + (A // B)
            + (A % 8)
            + (A % B)  # pyright: ignore[reportOperatorIssue]
            + (+A)
        )
        self.assertIsInstance(arith, torch.SymInt)
        self.assertIn("a#0", str(arith))
        self.assertIn("b#1", str(arith))

        # True division returns a SymFloat (IntTrueDiv lowering).
        truediv = A / 2
        self.assertIn("IntTrueDiv", str(truediv))

        # Bitwise: &, |, ^, <<, >> (NOTE: ~A is not supported on IntVar)
        bitwise = ((A & B) | (A ^ C)) + (A << 2) + (A >> 2)  # pyright: ignore[reportOperatorIssue]
        self.assertIsInstance(bitwise, torch.SymInt)

        # sym_min / sym_max compose with any IntVar.
        mm = torch.sym_max(A, B) + torch.sym_min(C, 5) + torch.sym_max(5, A)
        self.assertIsInstance(mm, torch.SymInt)
        self.assertIn("Max", str(mm))
        self.assertIn("Min", str(mm))

        # Comparisons (each returns a SymBool)
        for cmp in (A == B, A != B, A < B, A <= B, A > B, A >= B, A == 5, A < 5):
            self.assertIsInstance(cmp, torch.SymBool)

        # rsub / radd / rmul against Python ints
        self.assertIsInstance(1 + A, torch.SymInt)
        self.assertIsInstance(1 - A, torch.SymInt)
        self.assertIsInstance(2 * A, torch.SymInt)
        self.assertIsInstance(2 // A, torch.SymInt)

    def test_mod_uses_registered_range(self):
        # [FUNCTION DOC] Checks that modulo on an IntVar with min 0 prints as plain Mod, while an
        # unbounded IntVar (lower bound negative) prints as PythonMod, showing the registered
        # range is consulted.
        """``%`` calls ``ShapeEnv.bound_sympy`` to choose between ``Mod``
        (when both sides are non-negative) and ``PythonMod`` (otherwise).
        IntVar registers its min/max into ``var_to_range`` so this
        selection works without exploding on missing range info.
        """
        # Non-negative IntVar uses Mod.
        Apos = IntVar("apos", min=0)
        self.assertIn("Mod", str(Apos % 8))
        # Unbounded IntVar (min=None) defaults to (-int_oo, int_oo)
        # so the lower bound is negative → PythonMod.
        Aany = IntVar("aany")
        self.assertIn("PythonMod", str(Aany % 8))

    def test_rpow_fails_loudly(self):
        # [FUNCTION DOC] Raising an int to the power of an IntVar must raise TypeError mentioning
        # evaluate_sym_node, because that ShapeEnv API is blocked in the spec environment.
        """``int ** IntVar`` lowers through ``evaluate_sym_node``,
        which is blocked. Must raise rather than silently mis-evaluate.
        """
        A = IntVar("a")
        with self.assertRaisesRegex(TypeError, "evaluate_sym_node"):
            2**A  # pyright: ignore[reportUnusedExpression]


class TestTensorSpecConstruction(TestCase):
    # [CLASS DOC] Tests for TensorSpec construction from lists and tuples and its sequence-like
    # interface (index, length, iteration, repr). setUp resets the uid counter.
    """Construction and list-like interface."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_list_construction(self):
        # [FUNCTION DOC] Building from a list keeps ShapeVar, None and int entries retrievable by
        # index.
        ts = TensorSpec([ShapeVar("batch"), None, 10])
        self.assertIsInstance(ts[0], ShapeVar)
        self.assertIsNone(ts[1])
        self.assertEqual(ts[2], 10)

    def test_tuple_construction(self):
        # [FUNCTION DOC] Building from a tuple works; length is two and entries keep their types.
        ts = TensorSpec((ShapeVar("batch"), None))
        self.assertEqual(len(ts), 2)
        self.assertIsInstance(ts[0], ShapeVar)
        self.assertIsNone(ts[1])

    def test_len(self):
        # [FUNCTION DOC] Length of a TensorSpec equals the number of dimension entries, STATIC
        # counting.
        ts = TensorSpec([None, None, STATIC])
        self.assertEqual(len(ts), 3)

    def test_iter(self):
        # [FUNCTION DOC] Iterating yields entries in order, preserving the ShapeVar object
        # identity and showing STATIC as None.
        sv = ShapeVar("batch")
        ts = TensorSpec([sv, STATIC])
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertIs(items[0], sv)
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        # [FUNCTION DOC] Indexing past the end raises IndexError.
        ts = TensorSpec([None, STATIC])
        with self.assertRaises(IndexError):
            ts[5]

    def test_repr(self):
        # [FUNCTION DOC] Checks the multi-line repr of a TensorSpec: a Tensor header then one
        # numbered line per dimension (ShapeVar repr, None, int).
        sv = ShapeVar("batch")
        ts = TensorSpec([sv, None, 10])
        self.assertEqual(
            repr(ts),
            """\
Tensor:
  0: ShapeVar(batch#0, min=0)
  1: None
  2: 10""",
        )


class TestParamsSpecConstruction(TestCase):
    # [CLASS DOC] Tests for ParamsSpec and ShapesSpec construction, repr, JSON-able conversion and
    # validation of the special varargs and varkw keys. setUp resets the uid counter.
    """Construction of ParamsSpec and ShapesSpec."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_shapes_spec_repr(self):
        # [FUNCTION DOC] Checks the nested text repr of a ShapesSpec wrapping a ParamsSpec with
        # one tensor parameter.
        sv = ShapeVar("batch")
        ss = ShapesSpec(params=ParamsSpec({"x": TensorSpec([sv, STATIC])}))
        self.assertEqual(
            repr(ss),
            """\
shapes_spec:
  params:
    x:
      Tensor:
        0: ShapeVar(batch#0, min=0)
        1: None""",
        )

    def test_shapes_spec_repr_with_assumptions(self):
        # [FUNCTION DOC] Checks the repr of a ShapesSpec with two tensor params and two
        # assumptions: the text has a params section and an assumptions section printing each
        # assumption with uid-tagged variable names.
        A = ShapeVar("a")
        B = ShapeVar("b")
        ss = ShapesSpec(
            params={"x": TensorSpec([A, STATIC]), "y": TensorSpec([B, STATIC])},
            assumptions=[A > B, A + B > 10],
        )
        self.assertEqual(
            repr(ss),
            """\
shapes_spec:
  params:
    x:
      Tensor:
        0: ShapeVar(a#0, min=0)
        1: None
    y:
      Tensor:
        0: ShapeVar(b#1, min=0)
        1: None
  assumptions:
    a#0 > b#1
    a#0 + b#1 > 10""",
        )

    def test_shapes_spec_to_jsonable_with_assumptions(self):
        # [FUNCTION DOC] Checks that converting a ShapesSpec with assumptions to a JSON-able
        # structure gives nested dictionaries typed ShapesSpec, ParamsSpec, TensorSpec and
        # ShapeVar, plus assumptions as a list of strings.
        A = ShapeVar("a")
        B = ShapeVar("b")
        ss = ShapesSpec(
            params={"x": TensorSpec([A, STATIC])},
            assumptions=[A > B, A + B > 10],
        )
        self.assertEqual(
            ss.to_jsonable(),
            {
                "type": "ShapesSpec",
                "params": {
                    "type": "ParamsSpec",
                    "params": {
                        "x": {
                            "type": "TensorSpec",
                            "dims": [
                                {
                                    "type": "ShapeVar",
                                    "name": "a",
                                    "min": 0,
                                    "max": None,
                                    "optimization_hint": None,
                                },
                                None,
                            ],
                        },
                    },
                },
                "assumptions": ["a#0 > b#1", "a#0 + b#1 > 10"],
            },
        )

    def test_params_spec_repr_with_varargs_and_varkw(self):
        # [FUNCTION DOC] Checks the repr of a ParamsSpec that has a normal parameter, a varargs
        # entry (list) and a varkw entry (dict): sections are printed in that order with indexed
        # list entries and keyed dict entries.
        sv = ShapeVar("batch")
        ps = ParamsSpec(
            {
                "x": TensorSpec([sv]),
                "*args": [TensorSpec([sv]), STATIC],
                "**kwargs": {"foo": IntVar("k")},
            }
        )
        self.assertEqual(
            repr(ps),
            """\
x:
  Tensor:
    0: ShapeVar(batch#0, min=0)
*args:
  0:
    Tensor:
      0: ShapeVar(batch#0, min=0)
  1: None
**kwargs:
  foo: IntVar(k#1)""",
        )

    def test_params_spec_to_jsonable_with_varargs_and_varkw(self):
        # [FUNCTION DOC] Checks the JSON-able conversion of a ParamsSpec with varargs and varkw:
        # the varargs value becomes a list (None for static entries) and the varkw value becomes a
        # mapping of typed IntVar dictionaries.
        ps = ParamsSpec(
            {
                "x": TensorSpec([ShapeVar("batch")]),
                "*args": [TensorSpec([ShapeVar("a")]), STATIC],
                "**kwargs": {"foo": IntVar("k")},
            }
        )
        self.assertEqual(
            ps.to_jsonable(),
            {
                "type": "ParamsSpec",
                "params": {
                    "x": {
                        "type": "TensorSpec",
                        "dims": [
                            {
                                "type": "ShapeVar",
                                "name": "batch",
                                "min": 0,
                                "max": None,
                                "optimization_hint": None,
                            },
                        ],
                    },
                    "*args": [
                        {
                            "type": "TensorSpec",
                            "dims": [
                                {
                                    "type": "ShapeVar",
                                    "name": "a",
                                    "min": 0,
                                    "max": None,
                                    "optimization_hint": None,
                                }
                            ],
                        },
                        None,
                    ],
                    "**kwargs": {
                        "foo": {
                            "type": "IntVar",
                            "name": "k",
                            "min": None,
                            "max": None,
                            "optimization_hint": None,
                        }
                    },
                },
            },
        )

    def test_params_spec_rejects_unknown_sentinel(self):
        # [FUNCTION DOC] An unrecognized key starting with a star makes ParamsSpec raise
        # ValueError saying unknown sentinel key.
        with self.assertRaisesRegex(ValueError, r"Unknown sentinel key"):
            ParamsSpec({"*unknown": TensorSpec([ShapeVar("a")])})

    def test_params_spec_rejects_bad_varargs_value(self):
        # [FUNCTION DOC] A varargs key whose value is not a list makes ParamsSpec raise
        # ValueError.
        with self.assertRaisesRegex(ValueError, r"\*args.*must be a list"):
            ParamsSpec({"*args": TensorSpec([ShapeVar("a")])})  # not a list

    def test_params_spec_rejects_bad_varkw_value(self):
        # [FUNCTION DOC] A varkw key whose value is not a dict makes ParamsSpec raise ValueError.
        with self.assertRaisesRegex(ValueError, r"\*\*kwargs.*must be a dict"):
            ParamsSpec({"**kwargs": [TensorSpec([ShapeVar("a")])]})  # not a dict


class TestShapeVarCompile(TestCase):
    # [CLASS DOC] Integration tests of ShapeVar and IntVar with torch.compile via the
    # dynamic_shapes argument, using the eager backends; checks graph counts, symbol kinds,
    # errors, and shorthand normalization.
    """ShapeVar + torch.compile integration."""

    def test_static_int_spec_mismatch_raises(self):
        # [FUNCTION DOC] Declaring a scalar int parameter as the static value 10 and calling with
        # 42 must raise InternalTorchDynamoError reporting the declared and actual values.
        """If shapes_spec declares a scalar int as static=10, passing 42 should error."""

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Adds the scalar n to x.
            return x + n

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={"n": 10},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            r"shapes_spec declared L\['n'\] as static with value 10, but got 42 at trace time",
        ):
            compiled(torch.randn(4), 42)

    def test_static_tensor_dim_mismatch_raises(self):
        # [FUNCTION DOC] Declaring tensor dim 1 as static 3 and calling with a tensor whose dim 1
        # is 5 must raise InternalTorchDynamoError saying the declared static value differs from
        # the actual one.
        """If shapes_spec declares dim 1 as static=3, passing a tensor with dim 1=5 should error."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one to x.
            return x + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={"x": TensorSpec([ShapeVar("batch"), 3])},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.InternalTorchDynamoError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            compiled(torch.randn(4, 5))

    def test_unbacked_graph_has_unbacked_symbol(self):
        # [FUNCTION DOC] Compiles a sum over dim zero with a ShapeVar leading dim and calls it
        # with four different sizes. Expects a single recorded graph whose placeholder leading dim
        # is a SymInt with exactly one free unbacked symbol.
        """ShapeVar dim appears as an unbacked SymInt; single compile covers all shapes."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={"x": TensorSpec([ShapeVar("batch"), STATIC])},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        # [FUNCTION DOC] With data-dependent graph breaks disabled, a function branching on the
        # size of dim zero compiled with a ShapeVar there must raise GuardOnDataDependentSymNode.
        # The error condition must have exactly one free symbol, and its name must start with u
        # (unbacked).
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked with ShapeVar (unbacked)."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if the leading size exceeds five,
            # otherwise x minus one.
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"x": TensorSpec([ShapeVar(), STATIC])},
        )
        with self.assertRaises(GuardOnDataDependentSymNode) as cm:
            compiled(torch.randn(10, 3))
        free_syms = cm.exception.cond.free_symbols
        self.assertEqual(len(free_syms), 1)
        (sym,) = free_syms
        self.assertTrue(
            str(sym).startswith("u"),
            msg=lambda msg: f"{msg}\nexpected unbacked symbol (u-prefix), got {sym!r}",
        )

    def test_non_strict_raw_unbacked_symint_input_raises_dde_on_branching(self):
        # [FUNCTION DOC] Under the non-strict tracing context, passing a raw unbacked SymInt
        # created from a fresh ShapeEnv to a fullgraph function that branches on it must raise a
        # UserError about being unable to guard on a data-dependent expression.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def fn(n):
            # [FUNCTION DOC] HELPER: Returns a ones scalar if n is greater than one, else a zeros
            # scalar.
            if n > 1:
                return torch.ones(())
            return torch.zeros(())

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        n = ShapeEnv().create_unbacked_symint()
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            "Could not guard on data-dependent expression",
        ):
            with torch.compiler._non_strict_tracing_context():
                compiled(n)

    def test_raw_unbacked_symint_input_graph_breaks_outside_non_strict(self):
        # [FUNCTION DOC] Outside the non-strict context, passing a raw unbacked SymInt to a
        # fullgraph function must raise Unsupported saying an unbacked SymInt was attempted to be
        # wrapped.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def fn(n):
            # [FUNCTION DOC] HELPER: Returns a ones tensor whose shape is the single entry n.
            return torch.ones((n,))

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        n = ShapeEnv().create_unbacked_symint()
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported,
            "Attempted to wrap unbacked SymInt",
        ):
            compiled(n)

    def test_none_entry_is_static(self):
        # [FUNCTION DOC] A None entry in a TensorSpec is treated as static: with a ShapeVar
        # leading dim and None second dim, three different second-dim sizes yield three graphs.
        # The first graph has a SymInt leading dim and a plain int second dim.
        """A ``None`` entry is implicit static.
        Each distinct shape at dim 1 triggers a recompile."""
        backend = EagerAndRecordGraphs()
        ts = TensorSpec([ShapeVar("batch"), None])
        fn = torch.compile(
            lambda x: x + 1,
            fullgraph=True,
            backend=backend,
            dynamic_shapes={"x": ts},
        )

        fn(torch.randn(4, 3))
        fn(torch.randn(4, 5))
        fn(torch.randn(4, 7))
        self.assertEqual(len(backend.graphs), 3)
        # dim 0 is ShapeVar -> SymInt; dim 1 is None -> static int
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertIsInstance(shape[1], int)

    def test_none_scalar_int_is_static(self):
        # [FUNCTION DOC] A scalar int argument not mentioned in the spec is specialized: five
        # calls with three distinct values give three graphs, n is not a placeholder, and only the
        # tensor placeholder with a SymInt leading dim remains.
        """A scalar int arg not mentioned in shapes_spec stays static (specialized).
        Each distinct value triggers a recompile."""
        backend = EagerAndRecordGraphs()

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Adds the scalar n to x.
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={"x": TensorSpec([ShapeVar("batch")])},
        )

        compiled(torch.randn(2), 10)
        compiled(torch.randn(3), 10)
        compiled(torch.randn(4), 20)
        compiled(torch.randn(5), 20)
        compiled(torch.randn(6), 30)
        self.assertEqual(len(backend.graphs), 3)
        # n is specialized -> not a placeholder; x dim 0 is SymInt (ShapeVar)
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 1)
        self.assertIsInstance(phs[0].meta["example_value"].shape[0], torch.SymInt)

    def test_unspecified_tensor_is_all_static(self):
        # [FUNCTION DOC] When a spec is given, a tensor argument absent from it has all dims
        # static: with four different y shapes four graphs are recorded. In the first graph x
        # leading dim is a SymInt and every other dim of x and y is a plain int.
        """A tensor not mentioned in shapes_spec has all dims static when
        shapes_spec is provided."""
        backend = EagerAndRecordGraphs()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x scaled by 19 and y scaled by 10 as a tuple.
            return x * 19, y * 10

        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={"x": TensorSpec([ShapeVar("batch"), STATIC])},
        )
        x = torch.randn(12, 3)
        compiled(x, torch.randn(4, 3))
        # y not in spec -> all static; changing y's shape recompiles each time
        compiled(x, torch.randn(8, 3))
        compiled(x, torch.randn(12, 3))
        compiled(x, torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 4)

        # x dim 0 is SymInt (ShapeVar), x dim 1 is static; y is fully static
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        x_shape = phs[0].meta["example_value"].shape
        y_shape = phs[1].meta["example_value"].shape
        self.assertIsInstance(x_shape[0], torch.SymInt)
        self.assertIsInstance(x_shape[1], int)
        self.assertIsInstance(y_shape[0], int)
        self.assertIsInstance(y_shape[1], int)

    def test_params_spec_shorthand(self):
        # [FUNCTION DOC] Passing a ParamsSpec directly as dynamic_shapes is auto-wrapped; three
        # sizes of the ShapeVar dim produce one graph.
        """dynamic_shapes=ParamsSpec(...) is auto-wrapped into ShapesSpec."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes=ParamsSpec({"x": TensorSpec([ShapeVar("batch"), STATIC])}),
        )
        for n in [4, 8, 16]:
            fn(torch.randn(n, 3))
        # Unbacked dim 0 -> single compile
        self.assertEqual(len(backend.graphs), 1)

    def test_dict_shorthand(self):
        # [FUNCTION DOC] Passing a plain dict as dynamic_shapes is auto-wrapped into a ShapesSpec;
        # three sizes of the ShapeVar dim produce one graph.
        """dynamic_shapes={...} (bare dict) is auto-wrapped into
        ShapesSpec(params=ParamsSpec(dict))."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={"x": TensorSpec([ShapeVar("batch"), STATIC])},
        )
        for n in [4, 8, 16]:
            fn(torch.randn(n, 3))
        self.assertEqual(len(backend.graphs), 1)

    def test_normalize_rejects_bad_type(self):
        # [FUNCTION DOC] Passing a string as dynamic_shapes makes torch.compile raise TypeError
        # saying a dict, ShapesSpec or ParamsSpec is expected.
        """Passing something that's not dict/ParamsSpec/ShapesSpec/None
        should raise TypeError at compile entry."""
        with self.assertRaisesRegex(
            TypeError, "dynamic spec expects a dict, ShapesSpec, or ParamsSpec"
        ):
            torch.compile(lambda x: x, dynamic_shapes="not a spec")  # noqa: UNSPECIFIED_BACKEND

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_min_max_bypasses_dde_on_branching(self):
        # [FUNCTION DOC] With data-dependent graph breaks disabled, a ShapeVar with min 10 and max
        # 100 lets a branch on size greater than five resolve statically, so compiling and running
        # does not raise.
        """Mirror of test_unbacked_raises_dde_on_branching: setting min/max on
        ShapeVar constrains the symbol so the same branch is statically
        evaluated and does NOT raise a data-dependent error."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one if the leading size exceeds five,
            # otherwise x minus one.
            if x.size(0) > 5:
                return x + 1
            return x - 1

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([ShapeVar("batch", min=10, max=100), STATIC])
            },
        )
        # min=10 > 5 → branch resolves statically, no DDE.
        compiled(torch.randn(20, 3))

    def test_shapes_spec_with_dynamic_raises(self):
        # [FUNCTION DOC] Supplying both dynamic (True or False) and dynamic_shapes to
        # torch.compile raises ValueError saying they cannot both be set.
        """Setting both shapes_spec and dynamic should raise ValueError."""
        ts = TensorSpec([ShapeVar("batch"), STATIC])
        for dynamic in (True, False):
            with self.assertRaisesRegex(
                ValueError,
                r"`dynamic` and `dynamic_shapes` cannot both be set",
            ):
                torch.compile(
                    lambda x: x + 1,
                    backend="eager",
                    dynamic=dynamic,
                    dynamic_shapes={"x": ts},
                )

    def test_tensor_dim_optimization_hint_in_shape_env(self):
        # [FUNCTION DOC] Compiles with a ShapeVar optimization hint of 32 on the leading dim, runs
        # once, and checks the symbol's ShapeEnv records a hint override of 32 for that
        # expression.
        """ShapeVar's optimization_hint propagates to shape_env.var_to_hint_override."""
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={
                "x": TensorSpec([ShapeVar("batch", optimization_hint=32), STATIC])
            },
        )
        fn(torch.randn(8, 3))
        shape = _tensor_placeholder_shape(backend.graphs[0])
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 32)

    def test_scalar_int_optimization_hint_in_shape_env(self):
        # [FUNCTION DOC] Compiles with an IntVar optimization hint of 128 for scalar n, runs once,
        # finds the SymInt placeholder in the first graph and checks its ShapeEnv records a hint
        # override of 128.
        """IntVar's optimization_hint propagates to shape_env.var_to_hint_override."""
        backend = EagerAndRecordGraphs()

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Adds the scalar n to x.
            return x + n

        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={"n": IntVar("size", optimization_hint=128)},
        )
        compiled(torch.randn(4), 100)
        sym = None
        for node in backend.graphs[0].graph.nodes:
            if node.op == "placeholder":
                ev = node.meta.get("example_value")
                if isinstance(ev, torch.SymInt):
                    sym = ev
                    break
        self.assertIsNotNone(sym)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 128)


class TestShapeVarDedup(TestCase):
    # [CLASS DOC] Tests that reusing one ShapeVar or IntVar in several spec positions links the
    # resulting unbacked SymInts through a runtime equality check, so comparisons between those
    # dimensions do not raise data-dependent errors; also checks bound propagation and that
    # distinct variables are not linked. setUp resets the uid counter.
    """Sharing the same ShapeVar / IntVar across spec positions emits a
    runtime equality check between the resulting unbacked SymInts, so
    comparisons inside the compiled fn don't raise DDE.
    """

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_tensor_dim_vs_tensor_dim(self):
        # [FUNCTION DOC] Two tensors share a ShapeVar for dim zero; a fullgraph function that
        # branches on equality of the two sizes compiles and runs on equal sizes, returning a
        # tensor.
        """Two tensor dims sharing one ShapeVar: x.shape[0] == y.shape[0]
        must not DDE inside the compiled fn."""
        B = ShapeVar("batch")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of both tensors' sums if dim-zero sizes
            # match, else the difference.
            if x.shape[0] == y.shape[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, STATIC]),
                "y": TensorSpec([B, STATIC]),
            },
        )
        out = compiled(torch.randn(8, 3), torch.randn(8, 4))
        self.assertTrue(torch.is_tensor(out))

    def test_tensor_dim_vs_int(self):
        # [FUNCTION DOC] A tensor dim and a scalar int share one ShapeVar; branching on their
        # equality inside a fullgraph compile does not raise and the call returns a tensor.
        """A tensor dim and a scalar int sharing one ShapeVar: comparing
        them inside the compiled fn must not DDE."""
        B = ShapeVar("batch")

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Compares dim-zero size with n and returns the tensor sum plus
            # or minus n.
            if x.shape[0] == n:
                return x.sum() + n
            return x.sum() - n

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, STATIC]),
                "n": B,
            },
        )
        out = compiled(torch.randn(8, 3), 8)
        self.assertTrue(torch.is_tensor(out))

    def test_int_vs_int(self):
        # [FUNCTION DOC] Two scalar ints share one IntVar; branching on their equality compiles
        # without a data-dependent error and calling with equal values 4 and 4 returns 8.
        """Two scalar int args sharing one IntVar: a == b must not DDE."""
        S = IntVar("size")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns a plus b if they are equal, else a minus b.
            if a == b:
                return a + b
            return a - b

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"a": S, "b": S},
        )
        self.assertEqual(compiled(4, 4), 8)

    def test_min_max_propagate_to_repeat_occurrence(self):
        # [FUNCTION DOC] One ShapeVar with min 4 and max 16 is used for two tensors. A custom
        # backend captures the ShapeEnv from the first tensor placeholder with a symbolic leading
        # dim. Asserts the env was captured and that at least two symbols have the refined range 4
        # to 16, showing bounds set on the first occurrence propagate to the repeat occurrence via
        # the equality assert.
        """ShapeVar bounds are applied only on the FIRST occurrence (via
        torch._check in ``_wire_spec_slot``); subsequent occurrences rely
        on the dedup ``Eq(new, existing)`` runtime assert + ShapeEnv's
        ``_set_replacement`` / ``_refine_ranges`` to propagate the
        refined range to every other occurrence's symbol. This test
        guards that propagation by inspecting ``var_to_range`` for both
        of two tensor positions sharing one ShapeVar.
        """
        B = ShapeVar("b", min=4, max=16)
        from typing import cast

        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        captured_env = None

        def grab_env(gm, _example_inputs):
            # [FUNCTION DOC] HELPER: Backend callback; stores the ShapeEnv of the first
            # placeholder whose tensor has a symbolic leading dim into the enclosing variable
            # (nonlocal), then returns the graph's forward.
            nonlocal captured_env
            for node in gm.graph.nodes:
                if node.op != "placeholder":
                    continue
                ev = node.meta.get("example_value")
                if (
                    isinstance(ev, torch.Tensor)
                    and len(ev.shape) > 0
                    and hasattr(ev.shape[0], "node")
                ):
                    captured_env = ev.shape[0].node.shape_env  # type: ignore[attr-defined]
                    break
            return gm.forward

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the sums of x and y.
            return x.sum() + y.sum()

        compiled = torch.compile(
            fn,
            backend=grab_env,
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([B, None]),
            },
        )
        compiled(torch.randn(8, 3), torch.randn(8, 4))
        env = cast(ShapeEnv, captured_env)
        self.assertIsNotNone(env, "expected to capture the compile-time ShapeEnv")
        # Every symbol corresponding to ShapeVar B should have the
        # refined range [4, 16] — the second occurrence inherits it
        # via _set_replacement/_refine_ranges across the Eq runtime
        # assert in _wire_spec_slot.
        refined = [
            (sym, vr)
            for sym, vr in env.var_to_range.items()
            if vr.lower == 4 and vr.upper == 16
        ]
        self.assertGreaterEqual(
            len(refined),
            2,
            lambda msg: f"{msg}\nmin/max [4,16] did not propagate to repeat occurrence; "
            f"var_to_range={dict(env.var_to_range)}",
        )

    def test_distinct_shape_vars_still_dde(self):
        # [FUNCTION DOC] Two different ShapeVars on two tensors are not linked, so a branch
        # comparing their dim-zero sizes must raise a UserError about guarding on a data-dependent
        # expression.
        """Sanity check: distinct ShapeVars do not share a symbol, so
        comparing them still DDEs (no dedup contamination across vars)."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of both tensors' sums if dim-zero sizes
            # match, else the difference.
            if x.shape[0] == y.shape[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([A, STATIC]),
                "y": TensorSpec([B, STATIC]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled(torch.randn(8, 3), torch.randn(8, 4))


class TestDerivedDimSpec(TestCase):
    # [CLASS DOC] Tests derived dimension specs: ShapeVar expressions (such as multiples or sums)
    # used as a dim or scalar slot. Checks that a matching input runs, a mismatching one fails a
    # guard, that the same conditional raises a data-dependent error without the derived spec, and
    # error handling for unbound or foreign variables and misuse. setUp resets the uid counter.
    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_derived_dim(self):
        # [FUNCTION DOC] Spec says y dim zero is twice the ShapeVar of x. Matching shapes run;
        # after a Dynamo reset a mismatched y raises AssertionError mentioning a failed guard.
        # After another reset, giving y its own independent ShapeVar makes the same branch raise a
        # data-dependent UserError.
        """y's dim 0 = 2 * x's dim 0: correct shape runs; mismatched shape
        raises with the failed guard expression. The derived spec also lets
        ``y.size()[0] == 2 * x.size()[0]`` inside the compiled fn resolve
        without DDE."""
        B = ShapeVar("batch")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Branches on whether y's leading size is twice x's; returns
            # the sum or difference of sums.
            if y.size()[0] == 2 * x.size()[0]:
                return x.sum() + y.sum()
            return x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, STATIC]),
                "y": TensorSpec([B * 2, STATIC]),
            },
        )
        # Correct: y.shape[0] = 8 = 2 * x.shape[0]
        out = compiled(torch.randn(4, 3), torch.randn(8, 5))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 7 != 2 * 4 → guard fails
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(4, 3), torch.randn(7, 5))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Y = ShapeVar("y_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, None]),
                "y": TensorSpec([Y, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(torch.randn(4, 3), torch.randn(8, 5))

    def test_multi_var_derived(self):
        # [FUNCTION DOC] z dim zero is declared as the product of two ShapeVars plus one. Shapes
        # (3, 4, 13) run; shape 99 for z raises AssertionError with a guard failure after a reset;
        # with an independent ShapeVar for z the branch raises a data-dependent UserError.
        """Composite expression over multiple IntVars: z.shape[0] = A * B + 1.
        Correct shape runs; mismatched shape raises. The derived spec also
        lets ``z.size()[0] == x.size()[0] * y.size()[0] + 1`` resolve
        without DDE."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Branches on whether z's leading size equals x's times y's
            # plus one.
            if z.size()[0] == x.size()[0] * y.size()[0] + 1:
                return x.sum() + y.sum() + z.sum()
            return x.sum() - y.sum() - z.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([A, STATIC]),
                "y": TensorSpec([B, STATIC]),
                "z": TensorSpec([A * B + 1, STATIC]),
            },
        )
        # Correct: z.shape[0] = 3 * 4 + 1 = 13
        out = compiled(torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 99 != 3 * 4 + 1
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(3, 2), torch.randn(4, 2), torch.randn(99, 2))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Z = ShapeVar("z_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
                "z": TensorSpec([Z, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(
                torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2)
            )

    def test_orphan_intvar_raises(self):
        # [FUNCTION DOC] A derived dim built from two ShapeVars that never appear as bare slots
        # must fail at compile time with InternalTorchDynamoError whose message lists one pending
        # check referencing the unbound variables a and b and the rule that each must appear as a
        # bare slot.
        """B is used in derived expression but never as a bare slot → finalize raises."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        # A * B in slot but neither A nor B has a bare-IntVar binding via inputs
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"x": TensorSpec([A * B, STATIC])},
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as cm:
            compiled(torch.randn(4, 3))
        self.assertIn(
            "ValueError: shapes_spec: 1 pending check(s) reference unbound "
            "IntVar(s) ['a', 'b']. Every IntVar used in a derived "
            "expression or assumption must also appear as a bare-IntVar slot "
            "somewhere in the spec.",
            str(cm.exception),
        )

    def test_derived_scalar_arg(self):
        # [FUNCTION DOC] A scalar argument is declared as twice the batch ShapeVar of x. Matching
        # values run; a mismatch after a reset raises AssertionError mentioning a failed guard; a
        # separate IntVar for n makes the branch raise a data-dependent UserError.
        """Scalar arg slot can be a derived expression: n must equal 2 * x.shape[0].
        Correct value runs; mismatched value raises. The derived spec also
        lets ``n == 2 * x.size()[0]`` resolve without DDE."""
        B = ShapeVar("batch")

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Branches on whether n equals twice the leading size of x.
            if n == 2 * x.size()[0]:
                return x.sum() + n
            return x.sum() - n

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"x": TensorSpec([B, STATIC]), "n": B * 2},
        )
        # Correct: n = 8 = 2 * x.size()[0]
        out = compiled(torch.randn(4, 3), 8)
        self.assertTrue(torch.is_tensor(out))

        # Violation: n = 7 != 2 * 4
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(4, 3), 7)

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        N = IntVar("n_var")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"x": TensorSpec([B, None]), "n": N},
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(torch.randn(4, 3), 8)

    def test_foreign_symint_rejected_at_construction(self):
        # [FUNCTION DOC] A SymInt created by an unrelated ShapeEnv used as a TensorSpec entry
        # raises TypeError saying spec SymInt values must come from spec IntVar or ShapeVar.
        """A SymInt backed by a different ShapeEnv must be rejected at
        TensorSpec construction."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        real_env = ShapeEnv()
        foreign_symint = real_env.create_unbacked_symint()
        with self.assertRaisesRegex(
            TypeError,
            r"TensorSpec dim 0: SymInt spec values must originate from spec "
            r"IntVar / ShapeVar; got u0 backed by a different ShapeEnv\.",
        ):
            TensorSpec([foreign_symint, STATIC])

    def test_misuse_in_python_conditional_raises(self):
        # [FUNCTION DOC] Using a comparison of a ShapeVar as a Python condition raises TypeError
        # from the spec ShapeEnv, which forbids evaluation at spec-definition time.
        """Using a spec IntVar in a Python bool context raises (don't allow
        accidental guards on spec-time values)."""
        A = ShapeVar("a")
        with self.assertRaisesRegex(
            TypeError, r"_SpecShapeEnv:.*not allowed at spec-definition time"
        ):
            if A > 1:
                pass

    def test_misuse_torch_check_outside_assumptions_raises(self):
        # [FUNCTION DOC] Calling torch._check on a spec comparison outside the assumptions context
        # raises TypeError saying it is not allowed at spec-definition time.
        """torch._check on a spec IntVar outside the assumptions context raises."""
        A = ShapeVar("a")
        with self.assertRaisesRegex(
            TypeError, r"_SpecShapeEnv:.*not allowed at spec-definition time"
        ):
            torch._check(A > 1)

    def test_order_independence(self):
        # [FUNCTION DOC] The derived tensor z (product of two ShapeVars) is listed before the
        # tensors that bind those variables; the deferred check is emitted once both are bound.
        # Matching sizes run, mismatch raises an AssertionError for guard failure, and the version
        # without derivation raises a data-dependent UserError.
        """Composite slot (z = A * B) wired BEFORE bare-IntVar slots: the
        derived check is deferred, then emitted once both A and B are bound.
        Correct shape runs; mismatched shape raises with the failed guard."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(z, x, y):
            # [FUNCTION DOC] HELPER: Branches on whether z's leading size equals x's times y's;
            # returns sum or difference.
            # z processed first (dim references unbound A and B);
            # x binds A, y binds B; pending check then fires.
            if z.size()[0] == x.size()[0] * y.size()[0]:
                return z.sum() + x.sum() + y.sum()
            return z.sum() - x.sum() - y.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "z": TensorSpec([A * B, STATIC]),
                "x": TensorSpec([A, STATIC]),
                "y": TensorSpec([B, STATIC]),
            },
        )
        # Correct: z.shape[0] = 3 * 4 = 12
        out = compiled(torch.randn(12, 2), torch.randn(3, 2), torch.randn(4, 2))
        self.assertTrue(torch.is_tensor(out))

        # Violation: 99 != 3 * 4
        torch._dynamo.reset()
        with self.assertRaisesRegex(AssertionError, "Guard fail"):
            compiled(torch.randn(99, 2), torch.randn(3, 2), torch.randn(4, 2))

        # Sanity: WITHOUT the derived spec the same conditional DDEs.
        torch._dynamo.reset()
        Z = ShapeVar("z_dim")
        compiled_no_derived = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "z": TensorSpec([Z, None]),
                "x": TensorSpec([A, None]),
                "y": TensorSpec([B, None]),
            },
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_derived(
                torch.randn(12, 2), torch.randn(3, 2), torch.randn(4, 2)
            )

    def test_same_derived_expr_in_two_slots(self):
        # [FUNCTION DOC] Two tensors use the same derived expression (twice the batch var) for dim
        # zero. Compiling and running with consistent sizes returns a tensor.
        """Two tensor dims both spec'd as the same derived expression (B * 2)
        must both equal each other at runtime.

        Note: the equality is enforced at runtime (each slot deferred-asserts
        ``u_i == 2 * u_x``) but ShapeEnv doesn't transitively conclude
        ``u_y == u_z`` at compile time, so ``if y.size()[0] == z.size()[0]``
        would DDE."""
        B = ShapeVar("batch")

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the sum of the sums of x, y and z.
            return x.sum() + y.sum() + z.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={
                "x": TensorSpec([B, STATIC]),
                "y": TensorSpec([B * 2, STATIC]),
                "z": TensorSpec([B * 2, STATIC]),
            },
        )
        # y.shape[0] and z.shape[0] both = 8 (= 2 * x.shape[0])
        out = compiled(torch.randn(4, 3), torch.randn(8, 5), torch.randn(8, 5))
        self.assertTrue(torch.is_tensor(out))


class TestAssumptionsSpec(TestCase):
    # [CLASS DOC] Tests ShapesSpec assumptions: boolean expressions over spec variables that are
    # asserted at runtime and let compiled code branch without data-dependent errors; also error
    # cases for unbound variables and invalid assumption values. setUp resets the uid counter.
    """``ShapesSpec.assumptions`` — list of SymBool expressions over spec
    IntVars asserted at runtime via deferred runtime asserts."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_simple_assumption(self):
        # [FUNCTION DOC] With the assumption that A is greater than B, a branch comparing the two
        # sizes resolves without error. A valid call takes the if path; after a reset a violating
        # call raises RuntimeError for a failed runtime assertion; without the assumption the same
        # function raises a data-dependent UserError.
        """Assumption ``A > B`` enables ``if x.shape[0] > y.shape[0]`` to
        resolve without DDE."""
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x's sum plus 1000 if x is longer than y along dim
            # zero, else y's sum minus 1000.
            if x.shape[0] > y.shape[0]:
                return x.sum() + 1000.0
            return y.sum() - 1000.0

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes=ShapesSpec(
                params={
                    "x": TensorSpec([A, STATIC]),
                    "y": TensorSpec([B, STATIC]),
                },
                assumptions=[A > B],
            ),
        )
        # Correct: A=5 > B=3, takes if branch.
        out = compiled(torch.ones(5, 2), torch.ones(3, 2))
        self.assertTrue(out.item() > 500.0)

        # Violation: A=2 not > B=3.
        torch._dynamo.reset()
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            compiled(torch.ones(2, 2), torch.ones(3, 2))

        # Sanity check: same fn WITHOUT the assumption DDEs on the
        # conditional (proves the assumption is what makes the above work).
        torch._dynamo.reset()
        compiled_no_assumption = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes=ShapesSpec(
                params={
                    "x": TensorSpec([A, STATIC]),
                    "y": TensorSpec([B, STATIC]),
                },
            ),
        )
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            r"Could not guard on data-dependent expression",
        ):
            compiled_no_assumption(torch.ones(5, 2), torch.ones(3, 2))

    def test_orphan_intvar_in_assumption_raises(self):
        # [FUNCTION DOC] An assumption mentioning a ShapeVar B that is never bound by any slot
        # makes the first call raise InternalTorchDynamoError whose message names the unbound
        # variable b and lists the offending check.
        A = ShapeVar("a")
        B = ShapeVar("b")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes=ShapesSpec(
                params={"x": TensorSpec([A, STATIC])},
                assumptions=[A + B > 0],  # B never bound
            ),
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as cm:
            compiled(torch.randn(4, 3))
        self.assertIn(
            "ValueError: shapes_spec: 1 pending check(s) reference unbound "
            "IntVar(s) ['b']. Every IntVar used in a derived expression or "
            "assumption must also appear as a bare-IntVar slot somewhere in "
            "the spec. Offending checks:\n"
            "  - a + b > 0  (unbound: ['b'])",
            str(cm.exception),
        )

    def test_multiple_orphan_assumptions(self):
        # [FUNCTION DOC] Two assumptions over variables a and b, with only c bound, make the call
        # raise InternalTorchDynamoError whose message reports two pending checks and lists each
        # with the unbound names.
        """Multiple assumptions with multiple unbound IntVars: error lists
        all offending checks with user-friendly expressions."""
        A = ShapeVar("a")
        B = ShapeVar("b")
        C = ShapeVar("c")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of x.
            return x.sum()

        # C is bound by x; A and B are not.
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes=ShapesSpec(
                params={"x": TensorSpec([C, STATIC])},
                assumptions=[A + B > 0, A * 2 == B],
            ),
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as cm:
            compiled(torch.randn(4, 3))
        self.assertIn(
            "ValueError: shapes_spec: 2 pending check(s) reference unbound "
            "IntVar(s) ['a', 'b']. Every IntVar used in a derived expression "
            "or assumption must also appear as a bare-IntVar slot somewhere "
            "in the spec. Offending checks:\n"
            "  - a + b > 0  (unbound: ['a', 'b'])\n"
            "  - Eq(b, 2*a)  (unbound: ['a', 'b'])",
            str(cm.exception),
        )

    def test_foreign_symbool_rejected_at_construction(self):
        # [FUNCTION DOC] A SymBool built from an unbacked SymInt of a different ShapeEnv passed as
        # an assumption makes ShapesSpec raise TypeError naming the assumption index and the
        # foreign ShapeEnv.
        """A SymBool backed by a different ShapeEnv is rejected at
        ShapesSpec construction."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        real_env = ShapeEnv()
        u = real_env.create_unbacked_symint()
        foreign_symbool = u > 0  # SymBool from real_env
        with self.assertRaises(TypeError) as cm:
            ShapesSpec(assumptions=[foreign_symbool])
        self.assertIn(
            "ShapesSpec.assumptions[0]: SymBool spec values must originate "
            "from spec IntVar / ShapeVar; got u0 > 0 backed by a different "
            "ShapeEnv.",
            str(cm.exception),
        )

    def test_non_symbool_rejected(self):
        # [FUNCTION DOC] Plain True or the int 1 as an assumption makes ShapesSpec raise TypeError
        # saying a SymBool was expected.
        """Plain bool / non-SymBool elements are rejected."""
        with self.assertRaisesRegex(TypeError, r"expected SymBool"):
            ShapesSpec(assumptions=[True])
        with self.assertRaisesRegex(TypeError, r"expected SymBool"):
            ShapesSpec(assumptions=[1])


class TestObjectSpec(TestCase):
    # [CLASS DOC] Tests the ObjectSpec container (mapping of attribute names to specs):
    # construction, ordering, nesting, repr and JSON-able conversion. setUp resets the uid
    # counter.
    """``ObjectSpec`` data class — construction, access, repr."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_dict_construction(self):
        # [FUNCTION DOC] Fields passed to the constructor are kept by identity; the object has two
        # fields, a membership test finds a name, and a None value is stored as None.
        """Fields supplied via the constructor preserve identity and order."""
        spec = TensorSpec([ShapeVar("h"), STATIC])
        os = ObjectSpec({"weight": spec, "bias": None})
        self.assertEqual(len(os), 2)
        self.assertIn("weight", os)
        self.assertIs(os._fields["weight"], spec)
        self.assertIsNone(os._fields["bias"])

    def test_iter_and_items(self):
        # [FUNCTION DOC] Iterating yields the field names in insertion order and items yields name
        # and spec pairs in the same order.
        """Iteration walks fields in insertion order; items() yields pairs."""
        sv_w = ShapeVar("h")
        sv_b = ShapeVar("h")
        ts_w = TensorSpec([sv_w, STATIC])
        ts_b = TensorSpec([sv_b])
        os = ObjectSpec({"weight": ts_w, "bias": ts_b})
        self.assertEqual(list(os), ["weight", "bias"])
        self.assertEqual(list(os.items()), [("weight", ts_w), ("bias", ts_b)])

    def test_recursive_nesting(self):
        # [FUNCTION DOC] An ObjectSpec may hold another ObjectSpec as a field; identities are
        # preserved at each level.
        """A field's value may itself be an ``ObjectSpec``."""
        inner_spec = TensorSpec([ShapeVar("h"), STATIC])
        inner = ObjectSpec({"weight": inner_spec})
        outer = ObjectSpec({"inner": inner})
        self.assertIs(outer._fields["inner"], inner)
        self.assertIs(inner._fields["weight"], inner_spec)

    def test_repr_with_none_leaf(self):
        # [FUNCTION DOC] Repr of a one-field ObjectSpec with a None value prints an object_spec
        # header and a dotted field name with None inline.
        """Single-field repr with a ``None`` leaf renders inline."""
        os = ObjectSpec({"weight": None})
        self.assertEqual(
            repr(os),
            """\
object_spec:
  .weight: None""",
        )

    def test_repr_with_tensorspec(self):
        # [FUNCTION DOC] Repr places a multi-line TensorSpec repr indented under its dotted field
        # name.
        """Multi-line leaf repr is indented under its field name."""
        os = ObjectSpec({"weight": TensorSpec([ShapeVar("batch"), STATIC])})
        self.assertEqual(
            repr(os),
            """\
object_spec:
  .weight:
    Tensor:
      0: ShapeVar(batch#0, min=0)
      1: None""",
        )

    def test_repr_nested(self):
        # [FUNCTION DOC] Repr of nested ObjectSpecs indents each level under its dotted field
        # name.
        """Nested ``ObjectSpec`` repr indents recursively."""
        inner = ObjectSpec({"weight": None})
        outer = ObjectSpec({"inner": inner})
        self.assertEqual(
            repr(outer),
            """\
object_spec:
  .inner:
    object_spec:
      .weight: None""",
        )

    def test_to_jsonable(self):
        # [FUNCTION DOC] JSON-able conversion has type ObjectSpec, converts a TensorSpec field to
        # a typed dictionary, and leaves a None value as None.
        """``to_jsonable`` recurses into spec children; raw leaves pass through."""
        os = ObjectSpec(
            {
                "weight": TensorSpec([ShapeVar("h"), STATIC]),
                "bias": None,
            }
        )
        out = os.to_jsonable()
        self.assertEqual(out["type"], "ObjectSpec")
        self.assertIsInstance(out["fields"]["weight"], dict)
        self.assertEqual(out["fields"]["weight"]["type"], "TensorSpec")
        self.assertIsNone(out["fields"]["bias"])


class TestObjectSpecCompile(TestCase):
    # [CLASS DOC] End-to-end tests that an ObjectSpec applies to tensors reached through
    # attributes of an argument object during compilation. setUp resets the uid counter.
    """End-to-end: ``ObjectSpec`` routes through to a tensor reached
    via attribute access on a function arg (``obj.w``)."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_attr_tensor_dim_dynamic(self):
        # [FUNCTION DOC] Compiles a function reading an attribute tensor of its argument with an
        # ObjectSpec giving that tensor a ShapeVar leading dim. Calls with leading sizes 4 then 8
        # record only one graph.
        """A tensor reached via attribute access (``obj.w``) honors the
        ``ShapeVar`` in its ``ObjectSpec`` field; varying that dim does
        not recompile."""

        class Container:
            # [CLASS DOC] HELPER: Plain holder object with a single attribute w, the tensor.
            def __init__(self, w):
                # [FUNCTION DOC] HELPER: Stores the given tensor as attribute w.
                self.w = w

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the attribute tensor of obj plus one.
            return obj.w + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={
                "obj": ObjectSpec({"w": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )

        compiled(Container(torch.randn(4, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Different dim 0 size — dynamic absorbs it, no recompile.
        compiled(Container(torch.randn(8, 3)))
        self.assertEqual(len(backend.graphs), 1)

        # Captured weight placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_nn_module_parameter_dim_dynamic(self):
        # [FUNCTION DOC] Compiles a module whose forward adds one to its weight Parameter, with an
        # ObjectSpec on self giving the weight a ShapeVar leading dim. Calling, then replacing the
        # weight with one of a different leading size and calling again, records one graph total;
        # the last graph's weight placeholder has a SymInt in dim zero. Shows spec-driven dynamic
        # shapes override the static treatment of parameters.
        """An ``nn.Parameter`` reached via ``self.weight`` honors the
        spec. Parameters are normally force-marked static and routed
        as graph attributes, bypassing ``_automatic_dynamic``; the
        spec-aware specialization fixes in ``wrap_module`` /
        ``wrap_tensor`` skip those fast paths when a spec applies, so
        the Parameter flows through the dynamic-shape path."""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single 4x3 random Parameter named weight; forward
            # returns it plus one.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent constructor and creates the weight
                # Parameter.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 3))

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the weight plus one.
                return self.weight + 1

        m = M()
        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            m,
            backend=backend,
            dynamic_shapes={
                "self": ObjectSpec({"weight": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )

        compiled()
        self.assertEqual(len(backend.graphs), 1)
        m.weight = torch.nn.Parameter(torch.randn(8, 3))
        compiled()
        self.assertEqual(len(backend.graphs), 1)

        # Captured weight placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)


class TestSeqSpec(TestCase):
    # [CLASS DOC] Tests for SeqSpec (a per-position list of specs): construction, repr, JSON-able
    # conversion, and use through torch.compile on list arguments. setUp resets the uid counter.
    """Construction and integration of SeqSpec."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_accepts_tuple(self):
        # [FUNCTION DOC] A SeqSpec can be built from a tuple of two TensorSpecs and has length
        # two.
        B = ShapeVar("b")
        sp = SeqSpec((TensorSpec([B, 4]), TensorSpec([B, 8])))
        self.assertEqual(len(sp), 2)

    def test_empty(self):
        # [FUNCTION DOC] An empty SeqSpec has length zero and an empty internal entries list.
        sp = SeqSpec([])
        self.assertEqual(len(sp), 0)
        self.assertEqual(sp._entries, [])

    def test_repr(self):
        # [FUNCTION DOC] Repr prints a seq_spec header and each entry under an index label with
        # the nested TensorSpec repr.
        sp = SeqSpec([TensorSpec([ShapeVar("a"), 4])])
        self.assertEqual(
            repr(sp),
            """\
seq_spec:
  [0]:
    Tensor:
      0: ShapeVar(a#0, min=0)
      1: 4""",
        )

    def test_to_jsonable(self):
        # [FUNCTION DOC] JSON-able conversion has type SeqSpec with one entries item of type
        # TensorSpec.
        sp = SeqSpec([TensorSpec([ShapeVar("a"), 4])])
        j = sp.to_jsonable()
        self.assertEqual(j["type"], "SeqSpec")
        self.assertEqual(len(j["entries"]), 1)
        self.assertEqual(j["entries"][0]["type"], "TensorSpec")

    def test_compile_list_arg_marks_dynamic(self):
        # [FUNCTION DOC] A list argument of two tensors, both with the same ShapeVar leading dim
        # in a SeqSpec, is compiled fullgraph with a compile counter. Calls with leading sizes 4
        # and 7 give one frame compile.
        """List arg of two tensors with dim 0 dynamic via SeqSpec; no
        recompile on different dim-0 sizes."""
        B = ShapeVar("b")
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(xs: list[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the sum of the first two tensors' sums in the list.
            return xs[0].sum() + xs[1].sum()

        compiled = torch.compile(
            fn,
            backend=cnts,
            fullgraph=True,
            dynamic_shapes={"xs": SeqSpec([TensorSpec([B, 3]), TensorSpec([B, 5])])},
        )
        compiled([torch.ones(4, 3), torch.ones(4, 5)])
        compiled([torch.ones(7, 3), torch.ones(7, 5)])  # same dim 0 dyn
        self.assertEqual(cnts.frame_count, 1)

    def test_compile_out_of_range_element_treated_static(self):
        # [FUNCTION DOC] The SeqSpec covers only the first list element. After the first call (one
        # compile), changing the first tensor's leading dim does not recompile, but changing the
        # second tensor's leading dim (beyond the spec length, so static) causes a second compile.
        """Indices beyond the SeqSpec length are unspecified == static."""
        B = ShapeVar("b")
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(xs: list[torch.Tensor]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the sum of the first two tensors' sums in the list.
            return xs[0].sum() + xs[1].sum()

        compiled = torch.compile(
            fn,
            backend=cnts,
            fullgraph=True,
            dynamic_shapes={"xs": SeqSpec([TensorSpec([B, 3])])},
        )
        compiled([torch.ones(4, 3), torch.ones(4, 3)])
        self.assertEqual(cnts.frame_count, 1)

        # xs[0] dim 0 changes — covered by ShapeVar B → no recompile.
        compiled([torch.ones(7, 3), torch.ones(4, 3)])
        self.assertEqual(cnts.frame_count, 1)

        # xs[1] dim 0 changes — out-of-range entry treated as static → recompile.
        compiled([torch.ones(4, 3), torch.ones(8, 3)])
        self.assertEqual(cnts.frame_count, 2)


class TestDictSpecCompile(TestCase):
    # [CLASS DOC] Compile tests for DictSpec on dictionary arguments: string and int keys, keys
    # missing from the spec, nesting with SeqSpec, and the error for a container spec on a non-
    # container. setUp resets the uid counter.
    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_dict_spec_value_marked_dynamic(self):
        # [FUNCTION DOC] A dict entry given a ShapeVar leading dim through a DictSpec stays
        # dynamic: calls with leading sizes 4 and 8 record one graph whose placeholder has a
        # SymInt in dim zero.
        """A tensor reached via ``cfg["x"]`` honors the ``ShapeVar`` in
        its ``DictSpec`` entry; varying that dim does not recompile."""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the tensor under key x plus one.
            return cfg["x"] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={
                "cfg": DictSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )

        compiled({"x": torch.randn(4, 3)})
        compiled({"x": torch.randn(8, 3)})

        self.assertEqual(len(backend.graphs), 1)

        # Captured placeholder has a SymInt at dim 0.
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_dict_spec_missing_key_stays_static(self):
        # [FUNCTION DOC] Only key x is in the DictSpec. Three calls with different y shapes record
        # three graphs; in the first graph x has a SymInt leading dim and an int second dim, and y
        # is fully int.
        """A dict entry not present in ``DictSpec`` is all-static; each
        new shape recompiles."""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the sum of the sums of the tensors under keys x and
            # y.
            return cfg["x"].sum() + cfg["y"].sum()

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={
                "cfg": DictSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )

        x = torch.randn(4, 3)
        compiled({"x": x, "y": torch.randn(4, 3)})
        # y not in spec -> all static; changing y's shape recompiles each time
        compiled({"x": x, "y": torch.randn(4, 5)})
        compiled({"x": x, "y": torch.randn(4, 7)})
        self.assertEqual(len(backend.graphs), 3)

        # x dim 0 is SymInt (ShapeVar via DictSpec); y is fully static.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        x_shape = phs[0].meta["example_value"].shape
        y_shape = phs[1].meta["example_value"].shape
        self.assertIsInstance(x_shape[0], torch.SymInt)
        self.assertIsInstance(x_shape[1], int)
        self.assertIsInstance(y_shape[0], int)
        self.assertIsInstance(y_shape[1], int)

    def test_dict_spec_int_key_marked_dynamic(self):
        # [FUNCTION DOC] A DictSpec keyed by the int 0 applies to the dict entry under that int
        # key: two calls with leading sizes 4 and 8 give one graph with a SymInt in dim zero.
        """``DictSpec`` accepts int keys; ``cfg[0]`` honors the entry."""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the tensor under int key 0 plus one.
            return cfg[0] + 1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            fn,
            backend=backend,
            dynamic_shapes={"cfg": DictSpec({0: TensorSpec([ShapeVar("h"), STATIC])})},
        )

        compiled({0: torch.randn(4, 3)})
        compiled({0: torch.randn(8, 3)})
        self.assertEqual(len(backend.graphs), 1)

        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)

    def test_nested_dict_of_list_of_tensors(self):
        # [FUNCTION DOC] A DictSpec holding a SeqSpec of two TensorSpecs sharing one ShapeVar is
        # applied to a dict of a list of tensors; two calls with leading sizes 4 and 7 give a
        # single compile.
        """Nested: ``cfg["xs"][i]`` honors a DictSpec → SeqSpec → TensorSpec
        chain."""
        B = ShapeVar("b")
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the sum of the first two tensors in the list under
            # key xs.
            return cfg["xs"][0] + cfg["xs"][1]

        compiled = torch.compile(
            fn,
            backend=cnts,
            fullgraph=True,
            dynamic_shapes={
                "cfg": DictSpec(
                    {"xs": SeqSpec([TensorSpec([B, 3]), TensorSpec([B, 3])])}
                )
            },
        )
        compiled({"xs": [torch.ones(4, 3), torch.ones(4, 3)]})
        compiled({"xs": [torch.ones(7, 3), torch.ones(7, 3)]})
        self.assertEqual(cnts.frame_count, 1)

    def test_walk_terminal_container_raises(self):
        # [FUNCTION DOC] A DictSpec declared for an argument that is actually a plain tensor makes
        # the call raise InternalTorchDynamoError; the message before user-code context must be a
        # RuntimeError about the walk finding a DictSpec where the source has no further access.
        """The spec declares ``x`` as a ``DictSpec`` (a container), but the
        compiled function is called with a plain tensor for ``x``."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={"x": DictSpec({"k": TensorSpec([ShapeVar("h"), STATIC])})},
        )

        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as ctx:
            compiled(torch.randn(4, 3))
        self.assertEqual(
            str(ctx.exception).split("\n\nfrom user code:")[0],
            "RuntimeError: shapes_spec walk: while processing source "
            "'x', at 'x' the spec is DictSpec, but the source has no "
            "further access past it",
        )


class TestVarargsCompile(TestCase):
    # [CLASS DOC] Compile tests for specs on the star-args and star-kwargs parameters of a
    # function, including positions beyond the spec length. setUp resets the uid counter.
    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_named_and_varargs_and_varkw(self):
        # [FUNCTION DOC] Specs a named tensor, a varargs tensor and a keyword tensor each with a
        # ShapeVar leading dim. Varying leading sizes over three calls records one graph, and all
        # three tensor placeholders have SymInt dim zero and int dim one.
        """Combined: named tensor + ``*args`` tensor + ``**kwargs`` tensor.
        ``ShapeVar`` dims become SymInts
        """

        def f(x, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Adds x, the first extra positional tensor and the keyword
            # tensor foo.
            return x + args[0] + kwargs["foo"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            dynamic_shapes={
                "x": TensorSpec([ShapeVar("a"), STATIC]),
                "*args": [TensorSpec([ShapeVar("b"), STATIC])],
                "**kwargs": {"foo": TensorSpec([ShapeVar("c"), STATIC])},
            },
        )

        compiled(torch.randn(4, 3), torch.randn(4, 3), foo=torch.randn(4, 3))
        self.assertEqual(len(backend.graphs), 1)

        # Vary only the ShapeVar-marked dims -> no recompile.
        compiled(torch.randn(8, 3), torch.randn(8, 3), foo=torch.randn(8, 3))
        compiled(torch.randn(16, 3), torch.randn(16, 3), foo=torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        # All three tensor placeholders have SymInt at dim 0 and static int at dim 1.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 3)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)

    def test_pure_varargs(self):
        # [FUNCTION DOC] A spec with only a varargs list marks dim zero of both positional tensors
        # dynamic: three calls with growing sizes record one graph with two placeholders with
        # SymInt dim zero and int dim one.
        """A ``*args``-only spec marks each positional tensor's dim 0 dynamic."""

        def f(*args):
            # [FUNCTION DOC] HELPER: Adds the first two positional tensors.
            return args[0] + args[1]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            dynamic_shapes={
                "*args": [
                    TensorSpec([ShapeVar("a"), STATIC]),
                    TensorSpec([ShapeVar("b"), STATIC]),
                ],
            },
        )

        compiled(torch.randn(4, 3), torch.randn(4, 3))
        compiled(torch.randn(8, 3), torch.randn(8, 3))
        compiled(torch.randn(16, 3), torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)

    def test_varargs_extra_positions_are_static(self):
        # [FUNCTION DOC] The varargs list covers only two of three positional tensors. Changing
        # the third tensor's shape each call yields three graphs. In the first graph the first two
        # placeholders have SymInt dim zero while the third has int dims only.
        """Positional args past the end of the ``*args`` spec list are
        treated as fully static — changing their shape triggers a recompile."""

        def f(*args):
            # [FUNCTION DOC] HELPER: Returns the sum of the sums of the first three positional
            # tensors.
            return args[0].sum() + args[1].sum() + args[2].sum()

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            dynamic_shapes={
                # Spec covers only the first 2 *args entries.
                "*args": [
                    TensorSpec([ShapeVar("a"), STATIC]),
                    TensorSpec([ShapeVar("b"), STATIC]),
                ],
            },
        )

        # Vary the third (unspecified) arg's shape — each new shape recompiles
        # because it is treated as static.
        compiled(torch.randn(4, 3), torch.randn(4, 3), torch.randn(4, 3))
        compiled(torch.randn(8, 3), torch.randn(8, 3), torch.randn(4, 5))
        compiled(torch.randn(16, 3), torch.randn(16, 3), torch.randn(4, 7))
        self.assertEqual(len(backend.graphs), 3)

        # First two *args have SymInt dim 0 (ShapeVar); third is fully static.
        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 3)
        self.assertIsInstance(phs[0].meta["example_value"].shape[0], torch.SymInt)
        self.assertIsInstance(phs[1].meta["example_value"].shape[0], torch.SymInt)
        self.assertIsInstance(phs[2].meta["example_value"].shape[0], int)
        self.assertIsInstance(phs[2].meta["example_value"].shape[1], int)

    def test_pure_varkw(self):
        # [FUNCTION DOC] A spec with only a keyword mapping marks dim zero of keyword tensors a
        # and b dynamic: three calls with growing sizes record one graph with two SymInt-dim-zero
        # placeholders.
        """A ``**kwargs``-only spec marks each keyword tensor's dim 0 dynamic."""

        def f(**kwargs):
            # [FUNCTION DOC] HELPER: Adds the keyword tensors a and b.
            return kwargs["a"] + kwargs["b"]

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(
            f,
            backend=backend,
            dynamic_shapes={
                "**kwargs": {
                    "a": TensorSpec([ShapeVar("a"), STATIC]),
                    "b": TensorSpec([ShapeVar("b"), STATIC]),
                },
            },
        )

        compiled(a=torch.randn(4, 3), b=torch.randn(4, 3))
        compiled(a=torch.randn(8, 3), b=torch.randn(8, 3))
        compiled(a=torch.randn(16, 3), b=torch.randn(16, 3))
        self.assertEqual(len(backend.graphs), 1)

        phs = _tensor_placeholders(backend.graphs[0])
        self.assertEqual(len(phs), 2)
        for ph in phs:
            shape = ph.meta["example_value"].shape
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)


class TestWalkSpecRaises(TestCase):
    # [CLASS DOC] Tests that compiling raises InternalTorchDynamoError with a specific message
    # when the spec type does not match the structure of the actual argument. setUp resets the uid
    # counter.
    """Tests for spec/source type-mismatch errors raised during compilation."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter after the base setUp.
        super().setUp()
        _reset_uid_counter()

    def test_object_spec_with_dict_input_raises(self):
        # [FUNCTION DOC] An ObjectSpec on an argument that is passed a dict raises
        # InternalTorchDynamoError; the first message part names the source key access and says
        # the spec at the argument is ObjectSpec.
        """Spec is ``ObjectSpec``, but the user passes a dict."""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the tensor under key x plus one.
            return cfg["x"] + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={
                "cfg": ObjectSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as ctx:
            compiled({"x": torch.randn(4, 3)})
        self.assertEqual(
            str(ctx.exception).split("\n\nfrom user code:")[0],
            "RuntimeError: shapes_spec walk: while processing source "
            "\"cfg['x']\", at 'cfg' the spec is ObjectSpec",
        )

    def test_dict_spec_with_object_input_raises(self):
        # [FUNCTION DOC] A DictSpec on an argument that is passed a plain object with attribute x
        # raises InternalTorchDynamoError whose message says the spec at that argument is
        # DictSpec.
        """Spec is ``DictSpec``, but the user passes an object."""

        class Container:
            # [CLASS DOC] HELPER: Plain holder object with a single attribute x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the given value as attribute x.
                self.x = x

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the attribute x of obj plus one.
            return obj.x + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={
                "obj": DictSpec({"x": TensorSpec([ShapeVar("h"), STATIC])})
            },
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as ctx:
            compiled(Container(torch.randn(4, 3)))
        self.assertEqual(
            str(ctx.exception).split("\n\nfrom user code:")[0],
            "RuntimeError: shapes_spec walk: while processing source "
            "'obj.x', at 'obj' the spec is DictSpec",
        )

    def test_seq_spec_with_dict_input_raises(self):
        # [FUNCTION DOC] A SeqSpec on an argument that is passed a dict raises
        # InternalTorchDynamoError whose message says the spec at that argument is SeqSpec.
        """Spec is ``SeqSpec``, but the user passes a dict"""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the tensor under key x plus one.
            return cfg["x"] + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={"cfg": SeqSpec([TensorSpec([ShapeVar("h"), STATIC])])},
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as ctx:
            compiled({"x": torch.randn(4, 3)})
        self.assertEqual(
            str(ctx.exception).split("\n\nfrom user code:")[0],
            "RuntimeError: shapes_spec walk: while processing source "
            "\"cfg['x']\", at 'cfg' the spec is SeqSpec",
        )

    def test_leaf_spec_with_container_input_raises(self):
        # [FUNCTION DOC] A TensorSpec leaf on an argument that is passed a dict raises
        # InternalTorchDynamoError whose message says the spec is TensorSpec but the source has
        # further access past it.
        """Spec is a ``TensorSpec`` leaf, but the user passes a dict."""

        def fn(cfg):
            # [FUNCTION DOC] HELPER: Returns the tensor under key x plus one.
            return cfg["x"] + 1

        compiled = torch.compile(
            fn,
            backend="eager",
            dynamic_shapes={"cfg": TensorSpec([ShapeVar("h"), STATIC])},
        )
        with self.assertRaises(torch._dynamo.exc.InternalTorchDynamoError) as ctx:
            compiled({"x": torch.randn(4, 3)})
        self.assertEqual(
            str(ctx.exception).split("\n\nfrom user code:")[0],
            "RuntimeError: shapes_spec walk: while processing source "
            "\"cfg['x']\", at 'cfg' the spec is TensorSpec, but "
            "the source has further access past it",
        )


class TestDynamicSpecDecoratorCompile(TestCase):
    # [CLASS DOC] Tests for the dynamic_spec decorator, which attaches a spec to a function or
    # forward method and is picked up automatically by torch.compile. setUp resets the uid counter
    # and Dynamo.
    """``@dynamic_spec`` auto-applied by ``torch.compile``."""

    def setUp(self):
        # [FUNCTION DOC] Resets the global IntVar uid counter and resets Dynamo state after the
        # base setUp.
        super().setUp()
        _reset_uid_counter()
        torch._dynamo.reset()

    def test_dynamic_spec_decorator_dict_form(self):
        # [FUNCTION DOC] Decorates a function with a dict-form spec giving x a ShapeVar leading
        # dim, compiles it without dynamic_shapes, and runs with sizes 8 and 20. Expects one graph
        # whose single tensor placeholder has exactly one free unbacked symbol in dim zero.
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated function; sums x along dim zero.
            return x.sum(0)

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        compiled(torch.randn(8, 3))
        # Re-invoking at a different batch size must hit the same compiled
        # graph (dim 0 is unbacked) -- proving the decorator was picked up.
        compiled(torch.randn(20, 3))
        self.assertEqual(len(backend.graphs), 1)
        # Find the tensor placeholder.
        tensor_phs = [
            n
            for n in backend.graphs[0].graph.nodes
            if n.op == "placeholder"
            and isinstance(n.meta.get("example_value", n.meta.get("val")), torch.Tensor)
        ]
        self.assertEqual(len(tensor_phs), 1)
        val = tensor_phs[0].meta.get("example_value", tensor_phs[0].meta.get("val"))
        self.assertEqual(len(free_unbacked_symbols(val.shape[0])), 1)

    def test_dynamic_spec_decorator_on_nn_module_forward(self):
        # [FUNCTION DOC] The decorator on a module's forward is honored when compiling the module
        # instance: two calls with different leading sizes record one graph.
        """``@dynamic_spec`` on ``forward`` is picked up when
        ``torch.compile(module)`` is invoked (module path -- the resolver
        reads from ``module.forward``, not the bound method)."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose decorated forward sums its input along dim zero.
            @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Decorated with a spec; sums x along dim zero.
                return x.sum(0)

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(M(), backend=backend, fullgraph=True)
        compiled(torch.randn(8, 3))
        compiled(torch.randn(20, 3))
        self.assertEqual(len(backend.graphs), 1)

    def test_dynamic_spec_decorator_with_assumptions(self):
        # [FUNCTION DOC] A decorator spec includes an assumption that the batch ShapeVar is even.
        # Compiling fullgraph with sizes 8 and 20 records one graph, with the parity branch
        # resolved by the assumption.
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        B = ShapeVar("batch")

        @dynamic_spec(
            ShapesSpec(
                ParamsSpec({"x": TensorSpec([B, STATIC])}),
                assumptions=[B % 2 == 0],
            )
        )
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated; returns the dim-zero sum, negated when the leading
            # size is odd.
            # Branching on the assumed relation: without the assumption being
            # wired into the shape env, this would raise a DDE on the unbacked
            # symbol. With it, the branch resolves statically at trace time.
            if x.shape[0] % 2 == 0:
                return x.sum(0)
            return x.sum(0) * -1

        backend = EagerAndRecordGraphs()
        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        compiled(torch.randn(8, 3))
        compiled(torch.randn(20, 3))
        self.assertEqual(len(backend.graphs), 1)

    def test_dynamic_spec_decorator_and_explicit_kwarg_raises(self):
        # [FUNCTION DOC] Combining the decorator with an explicit dynamic_shapes argument to
        # torch.compile must raise ValueError saying both a decorator and a dynamic_shapes
        # argument were given.
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated function; sums x along dim zero.
            return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"`@dynamic_spec\(\.\.\.\)` is attached.*AND a `dynamic_shapes=`",
        ):
            torch.compile(  # noqa: UNSPECIFIED_BACKEND
                fn,
                dynamic_shapes=ParamsSpec({"x": TensorSpec([ShapeVar("Z"), STATIC])}),
            )(torch.randn(8, 3))

    def test_dynamic_spec_stacked_decorators_raise(self):
        # [FUNCTION DOC] Applying the decorator twice to one function raises ValueError saying a
        # spec is already attached, instead of silently overwriting.
        """Stacking ``@dynamic_spec`` on the same function should raise --
        silently overwriting the previously attached spec would mask user
        error."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        with self.assertRaisesRegex(
            ValueError, r"@dynamic_spec\(\.\.\.\) is already attached"
        ):

            @dynamic_spec({"x": TensorSpec([ShapeVar("A"), STATIC])})
            @dynamic_spec({"x": TensorSpec([ShapeVar("B"), STATIC])})
            def fn(x):
                # [FUNCTION DOC] HELPER: Doubly decorated function; sums x along dim zero.
                return x.sum(0)


if __name__ == "__main__":
    run_tests()
