# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for exporting with the experimental dynamic-spec API (ShapesSpec,
# ParamsSpec, tensor/int/container specs) in both strict and non-strict export modes.
# Contains small graph-inspection helpers, tiny reference modules, and two test families: one for
# tensor and int specs (names, derived dims, assumptions, decorator form, internal export entry
# points) and one for container specs (list, dict, object) including error paths.
# Each family is written once as a base test case with a strict flag supplied by two concrete
# subclasses; the bases are deleted from the module namespace so only subclasses are collected.
# Import time effect: none beyond imports; tests register pytree node types globally when
# they run.

import itertools
import re
from unittest import mock

import torch
import torch._dynamo
import torch.fx.experimental._config as _fx_experimental_config
import torch.utils._pytree as pytree
from torch.export import export
from torch.export._trace import (
    _export_to_aten_ir,
    _export_to_aten_ir_make_fx,
    _export_to_torch_ir,
    _non_strict_export,
    _strict_export,
)
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
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.testing._internal.common_utils import run_tests, TestCase


def _reset_uid_counter():
    # [FUNCTION DOC] HELPER: Restart the process-wide unique-id counter used for IntVar objects
    # by replacing the class attribute with a fresh counter that starts at zero. Mutates global
    # class state, returns nothing; called from setUp so symbol naming is reproducible per test.
    """Reset the global IntVar uid counter so uids start at 0 per test."""
    IntVar._uid_counter = itertools.count()


def _user_input_placeholders(gm):
    # [FUNCTION DOC] HELPER: Scan the graph nodes of a graph module and collect user-input
    # placeholders. For each placeholder node it reads the val metadata, falling back to
    # example_value, and keeps the pair of node and value only if the value is a Tensor or
    # SymInt. Returns the list in graph order; does not mutate the graph.
    """Return all user-input placeholders that have a tensor/symint val."""
    out = []
    for node in gm.graph.nodes:
        if node.op != "placeholder":
            continue
        val = node.meta.get("val", node.meta.get("example_value"))
        if isinstance(val, (torch.Tensor, torch.SymInt)):
            out.append((node, val))
    return out


def _first_tensor_placeholder_shape(gm):
    # [FUNCTION DOC] HELPER: Return the shape of the first placeholder whose value is a Tensor,
    # taken from the user-input placeholder pairs. Raises AssertionError if no tensor
    # placeholder exists.
    for _node, val in _user_input_placeholders(gm):
        if isinstance(val, torch.Tensor):
            return val.shape
    raise AssertionError("no tensor placeholder found")


def _placeholder_dim0_sym_names(gm):
    # [FUNCTION DOC] HELPER: For each tensor placeholder in graph order, take its dimension 0
    # size and, if it is a symbolic integer, append its string name to the result; static dim 0
    # sizes are skipped. Returns a list of names. Used so expected-text snapshots do not depend
    # on the process-wide fresh symbol counter.
    """Return the stringified dim-0 SymInt name for each tensor placeholder.

    Used to keep ``assertExpectedInline`` snapshots stable against
    ShapeEnv's process-wide fresh-symbol counter (which ``setUp`` does
    not reset).
    """
    names = []
    for _node, val in _user_input_placeholders(gm):
        if isinstance(val, torch.Tensor):
            sym = val.shape[0]
            if isinstance(sym, torch.SymInt):
                names.append(str(sym))
    return names


def _has_assert_scalar(gm):
    # [FUNCTION DOC] HELPER: Return True if any call_function node in the graph has a target
    # whose string form contains the _assert_scalar marker, i.e. the graph carries a runtime
    # assertion; otherwise False.
    """True if the graph contains an aten._assert_scalar runtime-assert node."""
    return any(
        node.op == "call_function" and "_assert_scalar" in str(node.target)
        for node in gm.graph.nodes
    )


# Modules with explicit signatures so ParamsSpec name lookup works.
class _ModX(torch.nn.Module):
    # [CLASS DOC] HELPER: Reference module with an explicitly named forward argument so
    # ParamsSpec can look parameters up by name.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Return the sum of the input over dimension 0.
        return x.sum(0)


class _ModXPlus(torch.nn.Module):
    # [CLASS DOC] HELPER: Reference module with a single named forward argument; adds one to
    # the input.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Return the input plus one.
        return x + 1


class _ModXN(torch.nn.Module):
    # [CLASS DOC] HELPER: Reference module with two named forward arguments, a tensor and a
    # number, that are added.
    def forward(self, x, n):
        # [FUNCTION DOC] HELPER: Return the tensor input plus the second argument (a scalar
        # or int).
        return x + n


class _ModBranch(torch.nn.Module):
    # [CLASS DOC] HELPER: Reference module whose forward branches on the size of dimension 0 of
    # its input; used to trigger data-dependent branching on dynamic dims.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Return the input plus one when its dim 0 size is greater than
        # five, otherwise the input minus one. The comparison forces a guard on the symbolic dim
        # 0 size.
        if x.size(0) > 5:
            return x + 1
        return x - 1


class _TestExportDynamicSpecBase(TestCase):
    # [CLASS DOC] Shared test body for tensor/int spec export tests. It defines no strict flag
    # itself; the concrete subclasses supply it as a class attribute and every test passes it to
    # the export call. The base is removed from the module namespace after the subclasses are
    # defined so it is not collected on its own.
    """torch.export.export support for the new ShapesSpec/ParamsSpec API."""

    def setUp(self):
        # [FUNCTION DOC] Call the parent setUp and reset the IntVar uid counter so symbol names
        # are deterministic.
        super().setUp()
        _reset_uid_counter()

    def test_unbacked_graph_has_unbacked_symbol(self):
        # [FUNCTION DOC] Export a module with a ParamsSpec naming a ShapeVar for dim 0 of the
        # input and check that the resulting dim 0 is an unbacked symbol: exactly one free
        # unbacked symbol, an exact textual snapshot of the program (with runtime assertion that
        # the symbol is non-negative and range zero to infinity), and that the exported module
        # runs with other batch sizes.
        B = VAR("batch")
        x_spec = T([B, STATIC])
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": x_spec}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)
        self.assertExpectedInline(
            str(ep).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        # Sanity-check: the exported program runs with different shapes.
        ep.module()(torch.randn(16, 3))
        ep.module()(torch.randn(32, 3))

    def test_static_int_spec_mismatch_raises(self):
        # [FUNCTION DOC] Passing a ParamsSpec that declares the int argument as static value 10
        # while the example input is 42 must raise ValueError saying the static declaration does
        # not match the trace-time value. The expected message differs by mode: strict mode names
        # the flattened argument source, non-strict uses the parameter name.
        # Source-name format differs: strict uses dynamo's flat-args path,
        # non-strict uses pytree keypath sources.
        if self.strict:
            regex = (
                r"shapes_spec declared L\['flat_args'\]\[1\] as static with "
                r"value 10, but got 42 at trace time"
            )
        else:
            regex = (
                r"shapes_spec declared L\['n'\] as static with value 10, "
                r"but got 42 at trace time"
            )
        with self.assertRaisesRegex(ValueError, regex):
            export(
                _ModXN(),
                (torch.randn(4), 42),
                dynamic_shapes=PARAMS({"n": 10}),
                strict=self.strict,
            )

    def test_static_tensor_dim_mismatch_raises(self):
        # [FUNCTION DOC] A tensor spec that declares dim 1 as static size 3 while the example
        # input has size 5 there must raise ValueError reporting the static mismatch.
        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec declares dim 1 as static with value 3, but got 5",
        ):
            export(
                _ModXPlus(),
                (torch.randn(4, 5),),
                dynamic_shapes=PARAMS({"x": T([VAR("batch"), 3])}),
                strict=self.strict,
            )

    def test_params_spec_shorthand(self):
        # [FUNCTION DOC] Check that a ParamsSpec given with a tensor spec containing a ShapeVar
        # and a static marker exports successfully and makes dim 0 of the placeholder a
        # symbolic integer.
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_min_max_bypasses_dde_on_branching(self):
        # [FUNCTION DOC] Under the config that turns data-dependent graph breaks into errors,
        # branch on a dim whose ShapeVar has min 10 and max 100; because the range lies above the
        # compared constant the branch resolves without a data-dependent error, so export just
        # needs to succeed.
        # min=10 > 5 → branch resolves statically, no DDE.
        export(
            _ModBranch(),
            (torch.randn(20, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("batch", min=10, max=100), STATIC])}),
            strict=self.strict,
        )

    @_fx_experimental_config.patch(no_data_dependent_graph_break=True)
    def test_unbacked_raises_dde_on_branching(self):
        # [FUNCTION DOC] Under the same no-graph-break config, a ShapeVar without bounds branched
        # on by the module must raise a data-dependent guard error. Strict mode surfaces it
        # wrapped as a dynamo UserError, non-strict raises the raw GuardOnDataDependentSymNode;
        # both match the message about not being able to guard on a data-dependent expression.
        """Without min/max, branching on a ShapeVar dim raises a DDE.
        Strict wraps it as UserError; non-strict raises the raw DDE.
        """
        exc = (
            torch._dynamo.exc.UserError
            if self.strict
            else torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode
        )
        with self.assertRaisesRegex(
            exc, r"Could not guard on data-dependent expression"
        ):
            export(
                _ModBranch(),
                (torch.randn(10, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR(), STATIC])}),
                strict=self.strict,
            )

    def test_tensor_dim_optimization_hint_in_shape_env(self):
        # [FUNCTION DOC] Create a ShapeVar with an optimization hint of 32, export with example
        # size 8, and verify that the shape environment of the dim 0 symbol records a hint
        # override of 32 for that expression.
        b = VAR("batch", optimization_hint=32)
        ep = export(
            _ModX(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([b, STATIC])}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        sym = shape[0]
        self.assertIsInstance(sym, torch.SymInt)
        expr = sym.node.expr
        self.assertEqual(sym.node.shape_env.var_to_hint_override.get(expr), 32)

    def test_params_spec_matched_by_name_across_positional_kwargs_and_call_order(
        self,
    ):
        # [FUNCTION DOC] Verify that spec entries bind by forward parameter name for positional
        # and keyword inputs and independent of keyword order. x is positional, z, n and y are
        # keywords in a different order; only x and y have specs. Checks an exact program
        # snapshot: placeholders in call order, x and y with unbacked dim 0 symbols, z static,
        # scalar n baked in as constant 2, and that the exported module runs with new sizes.
        """Spec entries bind by forward-param *name* — for positional and
        kwarg inputs alike, and regardless of kwarg order (placeholders
        follow call order). Inputs not in the spec stay static."""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module summing x, y and z over dim 0 and combining them
            # with scalar n.
            def forward(self, x, y, z, n):
                # [FUNCTION DOC] HELPER: Return sum of x plus sum of y plus the sum of z
                # multiplied by n, all reduced over dim 0.
                return x.sum(0) + y.sum(0) + z.sum(0) * n

        ep = export(
            M(),
            args=(torch.randn(8, 3),),  # x positional
            # kwargs in non-signature order (z, n before y):
            kwargs={"z": torch.randn(7, 3), "n": 2, "y": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "y": T([VAR("Y"), STATIC]),
                }
            ),
            strict=self.strict,
        )
        # Placeholders follow call order (x, z, n, y), not signature order:
        # x and y are spec'd by name → unbacked (u0, u1); z (kwarg, no spec)
        # keeps its literal shape; n (scalar, no spec) stays a plain input
        # with its value 2 baked into the math (mul by 2).
        self.assertExpectedInline(
            str(ep).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]", z: "f32[7, 3]", n, y: "f32[u1, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(y, 0)
            ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
            _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, sum_2);  sum_1 = sum_2 = None
            sum_3: "f32[3]" = torch.ops.aten.sum.dim_IntList(z, [0]);  z = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_3, 2);  sum_3 = None
            add_1: "f32[3]" = torch.ops.aten.add.Tensor(add, mul);  add = mul = None
            return (add_1,)
Graph signature:
    x: USER_INPUT
    z: USER_INPUT
    n: USER_INPUT
    y: USER_INPUT
    add_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo], u1: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        ep.module()(torch.randn(20, 3), z=torch.randn(7, 3), n=2, y=torch.randn(99, 3))

    def test_scalar_int_input_via_int_var(self):
        # [FUNCTION DOC] Export with an IntVar spec for the scalar int argument n plus a tensor
        # spec for x. Verifies the printed program shows n as an unbacked symbolic int and that
        # the exported module accepts other values of n and other batch sizes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module scaling the dim 0 sum of x by an integer argument.
            def forward(self, x, n):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x multiplied by n.
                return x.sum(0) * n

        ep = export(
            M(),
            (torch.randn(8, 3), 5),
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("B"), STATIC]),
                    "n": IntVar("n_size"),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'n: "Sym\(u\d+\)"')
        ep.module()(torch.randn(20, 3), 42)
        ep.module()(torch.randn(50, 3), 99)

    def test_multi_leaf_arg_with_leaf_spec_raises(self):
        # [FUNCTION DOC] A bare tensor spec given for an argument that is actually a list must
        # raise ValueError saying the spec is a TensorSpec but the argument is a list, with the
        # path to the parameter in the message.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the first two elements of a list argument.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Return the sum of the first two elements of the
                # input list.
                return xs[0] + xs[1]

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec\['xs'\]: spec is TensorSpec but the actual arg is list, not a Tensor\.",
        ):
            export(
                M(),
                args=([torch.randn(8, 3), torch.randn(8, 3)],),
                dynamic_shapes=PARAMS({"xs": T([VAR("B"), STATIC])}),
                strict=self.strict,
            )

    def test_multi_leaf_arg_no_spec_stays_static(self):
        # [FUNCTION DOC] With an empty ShapesSpec (no params) and a list-of-tensors input,
        # every flattened list element must stay static: the printed program contains both
        # elements with their literal shapes and the exported module returns a result of the
        # original shape.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the first two elements of a list argument.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Return the sum of the first two elements of the
                # input list.
                return xs[0] + xs[1]

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(8, 3)],),
            dynamic_shapes=ShapesSpec(),  # no params → all static
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both list elements come through as static placeholders.
        self.assertIn('xs_0: "f32[8, 3]"', ep_str)
        self.assertIn('xs_1: "f32[8, 3]"', ep_str)
        out = ep.module()([torch.randn(8, 3), torch.randn(8, 3)])
        self.assertEqual(out.shape, torch.Size([8, 3]))

    def test_spec_entry_for_omitted_default_raises(self):
        # [FUNCTION DOC] A ParamsSpec entry for a parameter that was left at its default (not
        # passed to export) must raise ValueError reporting that the spec has entries not
        # matching any passed argument, naming y.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose second argument defaults to None and is only
            # used when given.
            def forward(self, x, y=None):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x, plus the dim 0 sum of y if y
                # is not None.
                if y is None:
                    return x.sum(0)
                return x.sum(0) + y.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"ParamsSpec has entries \['y'\] that do not match",
        ):
            export(
                M(),
                args=(torch.randn(8, 3),),  # y not passed
                dynamic_shapes=PARAMS(
                    {
                        "x": T([VAR("B"), STATIC]),
                        "y": T([VAR("Y"), STATIC]),  # no such arg
                    }
                ),
                strict=self.strict,
            )

    def test_explicit_none_spec_for_passed_arg_does_not_raise(self):
        # [FUNCTION DOC] Regression check: an explicit None entry in a ParamsSpec for an argument
        # that is passed means static and must not be reported as an unmatched key. The exported
        # placeholder keeps its literal shape eight by three.
        """Explicit ``None`` in ParamsSpec for a passed arg means "static"
        and must not be flagged as an unmatched spec key (regression: a
        prior `.get()` check conflated key-absent with value-is-None).
        """
        ep = export(
            _ModXPlus(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": None}),
            strict=self.strict,
        )
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertEqual(tuple(shape), (8, 3))

    def test_named_dims_vs_shapes_spec(self):
        # [FUNCTION DOC] Compare the legacy Dim based dynamic shapes with the new ParamsSpec for
        # the same module. Snapshot both programs (legacy gives a backed symbol with a
        # zero-to-infinity range, new gives an unbacked symbol with a non-negativity assertion),
        # then check dim 0 is symbolic and dim 1 is the int 3 in both, and both programs run with
        # other batch sizes.
        from torch.export import Dim

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the dim 0 sum of x.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x.
                return x.sum(0)

        ep_legacy = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes={"x": (Dim("B"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            (torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("B"), STATIC])}),
            strict=self.strict,
        )

        # Snapshot legacy graph (backed s* symbol, range constraint).
        # Symbol name is whatever ShapeEnv allocated (counter is process-wide).
        (legacy_sym,) = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{legacy_sym}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {{{legacy_sym}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # Snapshot new-API graph (unbacked u* symbol, range constraint).
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        for ep in (ep_legacy, ep_new):
            shape = _first_tensor_placeholder_shape(ep.graph_module)
            self.assertIsInstance(shape[0], torch.SymInt)
            self.assertIsInstance(shape[1], int)
            self.assertEqual(int(shape[1]), 3)
            ep.module()(torch.randn(20, 3))
            ep.module()(torch.randn(64, 3))

    def test_named_dims_vs_shapes_spec2(self):
        # [FUNCTION DOC] Parity of legacy dynamic shapes and ParamsSpec when x is positional, y
        # is passed as keyword and z is omitted. Snapshots both programs, showing x then y
        # placeholders in the same order with z absent, and runs both exported programs with
        # other sizes.
        """Parity: forward(x, y, z=None) with x positional, y as kwarg,
        z omitted. Both APIs mark x and y dim 0 dynamic; z stays out
        of the graph."""
        from torch.export import Dim

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with a defaulted third argument z that scales the
            # second term when given.
            def forward(self, x, y, z=None):
                # [FUNCTION DOC] HELPER: Return sum of x plus sum of y times z if z is given,
                # else times one.
                return x.sum(0) + y.sum(0) * (z if z is not None else 1)

        ep_legacy = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3)},
            dynamic_shapes={"x": (Dim("X"), None), "y": (Dim("Y"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            kwargs={"y": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "y": T([VAR("Y"), STATIC]),
                }
            ),
            strict=self.strict,
        )

        # Snapshot legacy: placeholders are (x, y) in signature order.
        # Symbol names are whatever ShapeEnv allocated (counter is process-wide).
        sx, sy = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{sx}, 3]", y: "f32[{sy}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, 1);  sum_2 = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, mul);  sum_1 = mul = None
            return (add,)
Graph signature:
    x: USER_INPUT
    y: USER_INPUT
    add: USER_OUTPUT
Range constraints: {{{sx}: VR[0, int_oo], {sy}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # Snapshot new-API: same placeholder names in same order — proves the
        # translator preserved (x, y) ordering through the wrapper.
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]", y: "f32[u1, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sym_size_int_1: "Sym(u1)" = torch.ops.aten.sym_size.int(y, 0)
            ge_1: "Sym(u1 >= 0)" = sym_size_int_1 >= 0;  sym_size_int_1 = None
            _assert_scalar_default_1 = torch.ops.aten._assert_scalar.default(ge_1, "Runtime assertion failed for expression u1 >= 0 on node 'ge_1'");  ge_1 = _assert_scalar_default_1 = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            sum_2: "f32[3]" = torch.ops.aten.sum.dim_IntList(y, [0]);  y = None
            mul: "f32[3]" = torch.ops.aten.mul.Tensor(sum_2, 1);  sum_2 = None
            add: "f32[3]" = torch.ops.aten.add.Tensor(sum_1, mul);  sum_1 = mul = None
            return (add,)
Graph signature:
    x: USER_INPUT
    y: USER_INPUT
    add: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo], u1: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

        # Graphs are pinned above; just verify both run at varying shapes.
        for ep in (ep_legacy, ep_new):
            ep.module()(torch.randn(20, 3), y=torch.randn(99, 3))

    def test_legacy_parity_default_kwarg_omitted(self):
        # [FUNCTION DOC] Parity between legacy dynamic shapes and ParamsSpec when an optional
        # keyword argument is omitted: both programs have only the x input in the signature and
        # no y placeholder. Snapshots both (backed symbol for legacy, unbacked for the new API).
        from torch.export import Dim

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose second argument is optional and defaults
            # to None.
            def forward(self, x, y=None):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x, plus that of y when y is
                # not None.
                if y is None:
                    return x.sum(0)
                return x.sum(0) + y.sum(0)

        ep_legacy = export(
            M(),
            args=(torch.randn(8, 3),),
            dynamic_shapes={"x": (Dim("X"), None)},
            strict=self.strict,
        )
        ep_new = export(
            M(),
            args=(torch.randn(8, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("X"), STATIC])}),
            strict=self.strict,
        )

        # Legacy: only x in graph signature, no y placeholder.
        # Symbol name is whatever ShapeEnv allocated (counter is process-wide).
        (legacy_sym,) = _placeholder_dim0_sym_names(ep_legacy.graph_module)
        self.assertExpectedInline(
            str(ep_legacy).strip(),
            f"""\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[{legacy_sym}, 3]"):
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {{{legacy_sym}: VR[0, int_oo]}}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )
        # New API: same — y dropped from graph signature.
        self.assertExpectedInline(
            str(ep_new).strip(),
            """\
ExportedProgram:
    class GraphModule(torch.nn.Module):
        def forward(self, x: "f32[u0, 3]"):
            sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
            ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
            _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
            sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
            return (sum_1,)
Graph signature:
    x: USER_INPUT
    sum_1: USER_OUTPUT
Range constraints: {u0: VR[0, int_oo]}""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_user_varargs_in_forward_marked_dynamic_via_varargs_spec(self):
        # [FUNCTION DOC] A ParamsSpec entry keyed by the star-args name supplies a list of tensor
        # specs for the extra positional arguments. Verifies both flattened placeholders get
        # unbacked dim 0 symbols and the program runs with other sizes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module taking variable positional arguments and summing
            # the first two.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Return the sum of all elements of the first two
                # positional arguments.
                return args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=PARAMS(
                {
                    "*args": [
                        T([VAR("A"), STATIC]),
                        T([VAR("B"), STATIC]),
                    ]
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both args dim 0 are unbacked (different symbols).
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_with_named_arg_before(self):
        # [FUNCTION DOC] Combine a named parameter spec with a star-args list spec: x, args_0 and
        # args_1 all get unbacked dim 0 symbols, and the program runs with new sizes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module with one named argument and variable
            # positional arguments.
            def forward(self, x, *args):
                # [FUNCTION DOC] HELPER: Return the sum of all elements of x plus the first two
                # extra positional arguments.
                return x.sum() + args[0].sum() + args[1].sum()

        ep = export(
            M(),
            args=(torch.randn(4, 3), torch.randn(8, 3), torch.randn(5, 3)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([VAR("X"), STATIC]),
                    "*args": [
                        T([VAR("A"), STATIC]),
                        T([VAR("B"), STATIC]),
                    ],
                },
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'x: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'args_1: "f32\[u\d+, 3\]"')
        ep.module()(torch.randn(7, 3), torch.randn(20, 3), torch.randn(99, 3))

    def test_user_varargs_partial_spec_leaves_remainder_static(self):
        # [FUNCTION DOC] A star-args spec shorter than the number of extra arguments leaves the
        # remainder static: only the first positional is symbolic, the other two keep literal
        # shapes in the printed program.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module summing three variable positional arguments.
            def forward(self, *args):
                # [FUNCTION DOC] HELPER: Return the total of elements of the first three
                # positional arguments.
                return args[0].sum() + args[1].sum() + args[2].sum()

        ep = export(
            M(),
            args=(torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)),
            dynamic_shapes=PARAMS(
                {
                    "*args": [
                        T([VAR("A"), STATIC]),
                    ]
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # First arg dynamic, others literally sized.
        self.assertRegex(ep_str, r'args_0: "f32\[u\d+, 3\]"')
        self.assertIn('args_1: "f32[5, 3]"', ep_str)
        self.assertIn('args_2: "f32[6, 3]"', ep_str)

    def test_user_varkw_in_forward_marked_dynamic_via_varkw_spec(self):
        # [FUNCTION DOC] A ParamsSpec entry keyed by the double-star kwargs name supplies a
        # mapping of keyword names to tensor specs. Verifies foo and bar both get unbacked dim 0
        # symbols and the exported module can be called with keywords of other sizes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module taking variable keyword arguments and using the
            # foo and bar entries.
            def forward(self, **kwargs):
                # [FUNCTION DOC] HELPER: Return the sum of all elements of foo plus those of bar
                # from the keyword arguments.
                return kwargs["foo"].sum() + kwargs["bar"].sum()

        ep = export(
            M(),
            args=(),
            kwargs={"foo": torch.randn(8, 3), "bar": torch.randn(5, 3)},
            dynamic_shapes=PARAMS(
                {
                    "**kwargs": {
                        "foo": T([VAR("F"), STATIC]),
                        "bar": T([VAR("B"), STATIC]),
                    }
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'foo: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'bar: "f32\[u\d+, 3\]"')
        ep.module()(foo=torch.randn(20, 3), bar=torch.randn(99, 3))

    def test_prefer_deferred_runtime_asserts_raises_with_shapes_spec(self):
        # [FUNCTION DOC] Combining the prefer-deferred-runtime-asserts-over-guards export option
        # with the unbacked-only ShapesSpec API must raise ValueError mentioning that the option
        # cannot be combined with ShapesSpec.
        """`prefer_deferred_runtime_asserts_over_guards` is meaningful only
        for backed shapes; combining it with the unbacked-only ShapesSpec API
        raise."""
        with self.assertRaisesRegex(
            ValueError,
            r"`prefer_deferred_runtime_asserts_over_guards=True` cannot be "
            r"combined with `dynamic_shapes=ShapesSpec",
        ):
            export(
                _ModX(),
                (torch.randn(8, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
                strict=self.strict,
                prefer_deferred_runtime_asserts_over_guards=True,
            )

    def test_derived_dim_runtime_enforced(self):
        # [FUNCTION DOC] Derived dim test: ShapeVar B drives x dim 0 and an expression of B times
        # two drives y dim 0. Checks the printed program shows y as a twice-unbacked-symbol dim,
        # a runtime assert node exists, a consistent input runs, and an input with a mismatching
        # y dim 0 raises RuntimeError with a message about a failed runtime assertion.
        """Derived dim ``y dim0 = B * 2`` is enforced at runtime: the graph
        placeholder shows ``2*u0``, a correct input runs, and a violating
        input raises a runtime assertion (mirrors dynamo
        ``TestDerivedDimSpec.test_derived_dim`` but with export semantics)."""
        B = VAR("batch")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward branches on the derived relation
            # between the two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Branch on whether twice the dim 0 size of x equals the
                # dim 0 size of y, returning the sum of the sums or their difference. The branch
                # proves the relation is known at trace time, so no data-dependent error occurs.
                # Branching on the derived relation proves it is known at
                # trace time (otherwise this would be a data-dependent error).
                if x.size(0) * 2 == y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        ep = export(
            M(),
            (torch.randn(4, 3), torch.randn(8, 5)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([B, STATIC]),
                    "y": T([B * 2, STATIC]),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # y dim 0 is the derived expression 2*u0.
        self.assertRegex(ep_str, r'y: "f32\[2\*u\d+, 5\]"')
        # The derived constraint is materialized as a runtime assert.
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct input: y.shape[0] == 2 * x.shape[0].
        ep.module()(torch.randn(4, 3), torch.randn(8, 5))
        # Violation: 7 != 2 * 4 -> runtime assertion fires.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(4, 3), torch.randn(7, 5))

    def test_multi_var_derived_runtime_enforced(self):
        # [FUNCTION DOC] Multi-variable derived dim: z dim 0 is declared as A times B plus one.
        # Checks a runtime assert exists, a consistent call (3, 4, 13) runs, and an inconsistent
        # z dim 0 raises RuntimeError reporting a failed runtime assertion.
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module summing three inputs completely.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Return the sum of all elements of x, y and z.
                return x.sum() + y.sum() + z.sum()

        ep = export(
            M(),
            (torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2)),
            dynamic_shapes=PARAMS(
                {
                    "x": T([A, STATIC]),
                    "y": T([B, STATIC]),
                    "z": T([A * B + 1, STATIC]),
                }
            ),
            strict=self.strict,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: z.shape[0] == 3 * 4 + 1 == 13.
        ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(13, 2))
        # Violation: 99 != 3 * 4 + 1.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(3, 2), torch.randn(4, 2), torch.randn(99, 2))

    def test_assumption_runtime_enforced(self):
        # [FUNCTION DOC] Assumption test: a ShapesSpec with params and an assumption that A is
        # greater than B. Checks a runtime assert exists, a call satisfying it runs, and a call
        # violating it raises RuntimeError about a failed runtime assertion.
        A = VAR("a")
        B = VAR("b")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward branches on the relation between the
            # two dim 0 sizes.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Branch on whether the dim 0 size of x is larger than
                # that of y, returning the sum of the sums or their difference; the branch is
                # resolved by the declared assumption.
                # Branching on the assumed relation proves it is known at
                # trace time (otherwise this would be a data-dependent error).
                if x.size(0) > y.size(0):
                    return x.sum() + y.sum()
                return x.sum() - y.sum()

        ep = export(
            M(),
            (torch.randn(5, 2), torch.randn(3, 2)),
            dynamic_shapes=ShapesSpec(
                params=PARAMS(
                    {
                        "x": T([A, STATIC]),
                        "y": T([B, STATIC]),
                    }
                ),
                assumptions=[A > B],
            ),
            strict=self.strict,
        )
        self.assertTrue(_has_assert_scalar(ep.graph_module))
        # Correct: a=5 > b=3.
        ep.module()(torch.randn(5, 2), torch.randn(3, 2))
        # Violation: a=2 not > b=3.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(2, 2), torch.randn(3, 2))

    def test_min_max_in_range_constraints(self):
        # [FUNCTION DOC] Export with a ShapeVar bounded to 10 through 100 and verify the program
        # has exactly one range constraint whose lower bound is 10 and upper bound is 100.
        ep = export(
            _ModX(),
            (torch.randn(20, 3),),
            dynamic_shapes=PARAMS({"x": T([VAR("b", min=10, max=100), STATIC])}),
            strict=self.strict,
        )
        rcs = ep.range_constraints
        self.assertEqual(len(rcs), 1)
        (vr,) = rcs.values()
        self.assertEqual(int(vr.lower), 10)
        self.assertEqual(int(vr.upper), 100)

    # ---- @dynamic_spec(...) decorator (auto-attach) ----

    def test_spec_decorator_per_param_form(self):
        # [FUNCTION DOC] A dynamic_spec decorator with a per-parameter mapping on forward is
        # applied automatically by export when no dynamic shapes argument is given; dim 0 becomes
        # an unbacked symbol (exactly one free unbacked symbol).
        """``@dynamic_spec({"x": ...})`` attached to ``forward`` is auto-applied."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward carries the dynamic_spec decorator
            # with a tensor spec for x.
            @dynamic_spec({"x": T([VAR("B"), STATIC])})
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x.
                return x.sum(0)

        ep = export(M(), (torch.randn(8, 3),), strict=self.strict)
        shape = _first_tensor_placeholder_shape(ep.graph_module)
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 1)

    def test_spec_decorator_full_form_with_assumptions(self):
        # [FUNCTION DOC] A dynamic_spec decorator given a full ShapesSpec with an assumption
        # that B is even is applied automatically. Even dim 0 inputs run; an odd one raises
        # RuntimeError about a failed runtime assertion, which shows the assumption reached
        # the graph.
        """``@dynamic_spec(ShapesSpec(params=..., assumptions=...))`` form is auto-applied,
        and the assumption is enforced as a runtime assertion."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        B = VAR("batch")

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward is decorated with a ShapesSpec
            # including an evenness assumption.
            @dynamic_spec(
                ShapesSpec(
                    PARAMS({"x": T([B, STATIC])}),
                    assumptions=[B % 2 == 0],
                )
            )
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x.
                return x.sum(0)

        ep = export(M(), (torch.randn(8, 3),), strict=self.strict)
        # Valid input (dim 0 is even) -- runs cleanly.
        ep.module()(torch.randn(8, 3))
        ep.module()(torch.randn(20, 3))
        # Violation (dim 0 is odd) -- runtime assertion fires, proving the
        # assumption from the decorator was actually wired into the graph.
        with self.assertRaisesRegex(RuntimeError, "Runtime assertion failed"):
            ep.module()(torch.randn(7, 3))

    def test_spec_decorator_and_explicit_dynamic_shapes_raises(self):
        # [FUNCTION DOC] Attaching the dynamic_spec decorator to forward and also passing dynamic
        # shapes explicitly to export is ambiguous and must raise ValueError mentioning both.
        """Mixing ``@dynamic_spec(...)`` with an explicit ``dynamic_shapes=`` kwarg
        is ambiguous and must raise."""
        from torch.fx.experimental.dynamic_spec import dynamic_spec

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward has a dynamic_spec decorator.
            @dynamic_spec({"x": T([VAR("B"), STATIC])})
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of x.
                return x.sum(0)

        with self.assertRaisesRegex(
            ValueError,
            r"`@dynamic_spec\(\.\.\.\)` is attached.*AND a `dynamic_shapes=`",
        ):
            export(
                M(),
                (torch.randn(8, 3),),
                dynamic_shapes=PARAMS({"x": T([VAR("B"), STATIC])}),
                strict=self.strict,
            )

    def test_export_to_torch_ir_shapes_spec_direct(self):
        # [FUNCTION DOC] Strict-only (skipped otherwise): call the internal torch-IR export
        # helper directly with a ParamsSpec and check it returns a GraphModule whose snapshot
        # shows the input annotated with an unbacked dim 0 symbol.
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        gm = _export_to_torch_ir(
            _ModX(),
            (torch.randn(8, 3),),
            {},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
        )
        self.assertIsInstance(gm, torch.fx.GraphModule)
        # The user placeholder carries an unbacked dim (u0).
        self.assertExpectedInline(
            gm.print_readable(print_output=False),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, x):
        arg_0: "f32[u0, 3]";
        arg_0, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
        l_flat_args_0_ = arg_0
        res: "f32[3]" = l_flat_args_0_.sum(0);  l_flat_args_0_ = None
        return pytree.tree_unflatten((res,), self._out_spec)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_export_to_torch_ir_legacy_v1_shapes_spec_raises(self):
        # [FUNCTION DOC] Strict-only (skipped otherwise): with the new-tracer experimental config
        # patched to False, calling the internal torch-IR export helper with a ShapesSpec must
        # raise NotImplementedError saying it is unsupported on the legacy path. The config is
        # restored by the patch context manager.
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        with mock.patch.object(
            torch._export.config, "use_new_tracer_experimental", False
        ):
            with self.assertRaisesRegex(
                NotImplementedError,
                r"ShapesSpec is not supported on the legacy v1",
            ):
                _export_to_torch_ir(
                    _ModX(),
                    (torch.randn(8, 3),),
                    {},
                    dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
                )

    def test_strict_export_shapes_spec_direct(self):
        # [FUNCTION DOC] Strict-only (skipped otherwise): call the internal strict export
        # function directly with a ParamsSpec, no module call signature preservation, no deferred
        # asserts and the make_fx aten lowering; check the resulting aten graph snapshot has an
        # unbacked dim 0 symbol with its non-negative runtime assert.
        # Strict-only internal-API test; skip in non-strict mode.
        if not self.strict:
            self.skipTest("strict-only internal-API test")
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        artifact = _strict_export(
            mod=_ModX(),
            args=args,
            kwargs={},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            preserve_module_call_signature=(),
            orig_in_spec=in_spec,
            prefer_deferred_runtime_asserts_over_guards=False,
            _to_aten_func=_export_to_aten_ir_make_fx,
        )
        self.assertExpectedInline(
            artifact.aten.gm.print_readable(print_output=False),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, x: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
        return (sum_1,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )

    def test_non_strict_export_shapes_spec_direct(self):
        # [FUNCTION DOC] Non-strict-only (skipped when strict): call the internal non-strict
        # export function directly with a ParamsSpec and the aten lowering function; check that
        # the aten graph snapshot has an unbacked dim 0 symbol with its non-negative runtime
        # assert, showing ShapesSpec is accepted there.
        # Non-strict-only internal-API test; skip in strict mode.
        # Verifies ``_non_strict_export`` with ``_to_aten_func=_export_to_aten_ir``
        # accepts ShapesSpec (replaces the old NotImplementedError gate test).
        if self.strict:
            self.skipTest("non-strict-only internal-API test")
        args = (torch.randn(8, 3),)
        _, in_spec = pytree.tree_flatten((args, {}))
        artifact = _non_strict_export(
            mod=_ModX(),
            args=args,
            kwargs={},
            dynamic_shapes=PARAMS({"x": T([VAR("batch"), STATIC])}),
            preserve_module_call_signature=(),
            orig_in_spec=in_spec,
            prefer_deferred_runtime_asserts_over_guards=False,
            _to_aten_func=_export_to_aten_ir,
        )
        self.assertExpectedInline(
            artifact.aten.gm.print_readable(print_output=False),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, x: "f32[u0, 3]"):
        sym_size_int: "Sym(u0)" = torch.ops.aten.sym_size.int(x, 0)
        ge: "Sym(u0 >= 0)" = sym_size_int >= 0;  sym_size_int = None
        _assert_scalar_default = torch.ops.aten._assert_scalar.default(ge, "Runtime assertion failed for expression u0 >= 0 on node 'ge'");  ge = _assert_scalar_default = None
        sum_1: "f32[3]" = torch.ops.aten.sum.dim_IntList(x, [0]);  x = None
        return (sum_1,)""",
            ignore_comments=True,
            ignore_empty_lines=True,
        )


class TestExportDynamicSpecStrict(_TestExportDynamicSpecBase):
    # [CLASS DOC] Concrete strict-mode instantiation of the tensor/int spec tests; sets the
    # strict flag to True.
    strict = True


class TestExportDynamicSpecNonStrict(_TestExportDynamicSpecBase):
    # [CLASS DOC] Concrete non-strict-mode instantiation of the tensor/int spec tests; sets the
    # strict flag to False.
    strict = False


del _TestExportDynamicSpecBase


class _TestContainerSpecBase(TestCase):
    # [CLASS DOC] Shared test body for container spec export tests (sequence, dict and object
    # specs and their error cases). The concrete subclasses supply the strict flag; the base is
    # deleted from the module namespace so it is not collected itself.
    def setUp(self):
        # [FUNCTION DOC] Call the parent setUp and reset the IntVar uid counter so symbol naming
        # is deterministic.
        super().setUp()
        _reset_uid_counter()

    # ---- Positive cases ----

    def test_seq_spec_on_list_of_tensors(self):
        # [FUNCTION DOC] A sequence spec on a list argument marks each list position dynamic
        # independently: both flattened placeholders get unbacked dim 0 symbols and the program
        # runs with other sizes.
        """SeqSpec marks each list position dynamic independently."""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the dim 0 sums of the first two
            # list elements.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Return the sum of the dim 0 reductions of the first
                # two elements.
                return xs[0].sum(0) + xs[1].sum(0)

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(5, 3)],),
            dynamic_shapes=PARAMS(
                {
                    "xs": L(
                        [
                            T([VAR("A"), STATIC]),
                            T([VAR("B"), STATIC]),
                        ]
                    )
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Both list elements have unbacked dim 0.
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'xs_1: "f32\[u\d+, 3\]"')
        ep.module()([torch.randn(20, 3), torch.randn(99, 3)])

    def test_dict_spec_on_dict_of_tensors_partial(self):
        # [FUNCTION DOC] A dict spec naming only key b leaves key a static: a keeps its literal
        # shape, b gets an unbacked dim 0 symbol.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the dim 0 sums of the a and b entries of
            # a dict.
            def forward(self, d):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of entry a plus that of entry b.
                return d["a"].sum(0) + d["b"].sum(0)

        ep = export(
            M(),
            args=({"a": torch.randn(7, 3), "b": torch.randn(8, 3)},),
            dynamic_shapes=PARAMS({"d": DICT({"b": T([VAR("B"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('d_a: "f32[7, 3]"', ep_str)
        self.assertRegex(ep_str, r'd_b: "f32\[u\d+, 3\]"')

    def test_object_spec_on_namedtuple(self):
        # [FUNCTION DOC] An object spec on a namedtuple names only the second field: first stays
        # static with its literal shape, second gets an unbacked dim 0 symbol.
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the dim 0 sums of the first and second
            # fields of a namedtuple.
            def forward(self, p):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of the first field plus that of
                # the second.
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"p": OBJ({"second": T([VAR("S"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # first is omitted from the spec → static; second is dynamic.
        self.assertIn('p_first: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'p_second: "f32\[u\d+, 3\]"')

    def test_seq_spec_on_namedtuple(self):
        # [FUNCTION DOC] A sequence spec on a namedtuple argument, with None at position zero and
        # a tensor spec at position one, treats the namedtuple positionally: first stays static,
        # second is unbacked.
        from collections import namedtuple

        Pair = namedtuple("Pair", ["first", "second"])

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module adding the dim 0 sums of the first and second
            # fields of a namedtuple.
            def forward(self, p):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of the first field plus that of
                # the second.
                return p.first.sum(0) + p.second.sum(0)

        ep = export(
            M(),
            args=(Pair(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"p": L([None, T([VAR("B"), STATIC])])}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # Position 0 → static (None entry); position 1 → dynamic.
        self.assertIn('p_first: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'p_second: "f32\[u\d+, 3\]"')

    def test_object_spec_on_custom_pytree_node(self):
        # [FUNCTION DOC] An object spec works on a class registered as a pytree node with a
        # caller-supplied keyed flatten function yielding attribute keys. Registers the type
        # globally in pytree, exports with a spec for field b only, and checks a stays static
        # while b is unbacked.
        """ObjectSpec on a class registered via
        ``pytree.register_pytree_node`` with a caller-supplied
        ``flatten_with_keys_fn`` that yields ``GetAttrKey`` entries."""

        class MyContainer:
            # [CLASS DOC] HELPER: Plain user class holding two tensors a and b, later registered
            # as a custom pytree node.
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: Store the two given values as attributes a and b.
                self.a = a
                self.b = b

        # Caller supplies flatten / unflatten / flatten_with_keys_fn —
        # the KeyPath shape is whatever the caller chooses to return.
        # We use GetAttrKey so ObjectSpec can address fields by name.
        def _flatten(c):
            # [FUNCTION DOC] HELPER: Pytree flatten function: return the two attributes as the
            # child list and no context.
            return [c.a, c.b], None

        def _unflatten(values, _context):
            # [FUNCTION DOC] HELPER: Pytree unflatten function: rebuild the container from the
            # pair of values, ignoring the context.
            a, b = values
            return MyContainer(a, b)

        def _flatten_with_keys(c):
            # [FUNCTION DOC] HELPER: Pytree keyed flatten function: return the children paired
            # with attribute-name keys for a and b, plus no context, so an object spec can
            # address fields by name.
            children = [
                (pytree.GetAttrKey("a"), c.a),
                (pytree.GetAttrKey("b"), c.b),
            ]
            return children, None

        pytree.register_pytree_node(
            MyContainer,
            _flatten,
            _unflatten,
            serialized_type_name="test_dynamic_spec_export.MyContainer",
            flatten_with_keys_fn=_flatten_with_keys,
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module consuming the custom container.
            def forward(self, c):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of attribute a plus that of
                # attribute b.
                return c.a.sum(0) + c.b.sum(0)

        ep = export(
            M(),
            args=(MyContainer(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"c": OBJ({"b": T([VAR("B"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # a is omitted from the spec → static; b is dynamic.
        self.assertIn('c_a: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'c_b: "f32\[u\d+, 3\]"')

    def test_object_spec_via_torch_export_register_dataclass(self):
        # [FUNCTION DOC] Register a dataclass with the export API registration function and check
        # an object spec naming only field y makes y unbacked while x stays static.
        import dataclasses

        import torch.export as te

        @dataclasses.dataclass
        class ExportedBox:
            # [CLASS DOC] HELPER: Dataclass with two tensor fields, registered through the
            # export-level dataclass registration.
            x: torch.Tensor
            y: torch.Tensor

        te.register_dataclass(ExportedBox)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module consuming the dataclass.
            def forward(self, box):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of field x plus that of field y.
                return box.x.sum(0) + box.y.sum(0)

        ep = export(
            M(),
            args=(ExportedBox(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"box": OBJ({"y": T([VAR("Y"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # x is omitted from the spec → static; y is dynamic.
        self.assertIn('box_x: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'box_y: "f32\[u\d+, 3\]"')

    def test_object_spec_on_dataclass(self):
        # [FUNCTION DOC] Register a dataclass with the pytree dataclass registration and check an
        # object spec naming only field y makes y unbacked while x stays static.
        import dataclasses

        @dataclasses.dataclass
        class Box:
            # [CLASS DOC] HELPER: Dataclass with two tensor fields, registered as a pytree node.
            x: torch.Tensor
            y: torch.Tensor

        pytree.register_dataclass(Box)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module consuming the dataclass.
            def forward(self, box):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of field x plus that of field y.
                return box.x.sum(0) + box.y.sum(0)

        ep = export(
            M(),
            args=(Box(torch.randn(5, 3), torch.randn(8, 3)),),
            dynamic_shapes=PARAMS({"box": OBJ({"y": T([VAR("Y"), STATIC])})}),
            strict=self.strict,
        )
        ep_str = str(ep)
        # x is omitted from the spec → static; y is dynamic.
        self.assertIn('box_x: "f32[5, 3]"', ep_str)
        self.assertRegex(ep_str, r'box_y: "f32\[u\d+, 3\]"')

    def test_nested_dict_of_seq_spec(self):
        # [FUNCTION DOC] Nested dict spec containing a sequence spec: the first list element gets
        # a ShapeVar and the second a fully static tensor spec. Checks the first is unbacked and
        # the second keeps its literal shape.
        """Nested DictSpec({"foo": SeqSpec([TensorSpec, None])}) — first
        list elem dynamic, second left to default (static)."""

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module consuming a dict holding a list.
            def forward(self, d):
                # [FUNCTION DOC] HELPER: Return the dim 0 sums of the first two elements under
                # key foo added together.
                return d["foo"][0].sum(0) + d["foo"][1].sum(0)

        ep = export(
            M(),
            args=({"foo": [torch.randn(8, 3), torch.randn(5, 3)]},),
            dynamic_shapes=PARAMS(
                {
                    "d": DICT(
                        {
                            "foo": L(
                                [
                                    T([VAR("A"), STATIC]),
                                    T([5, 3]),
                                ]
                            )
                        }
                    )
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'd_foo_0: "f32\[u\d+, 3\]"')
        self.assertIn('d_foo_1: "f32[5, 3]"', ep_str)

    def test_mixed_leaf_and_container_under_params_spec(self):
        # [FUNCTION DOC] A ParamsSpec mixing a plain tensor spec for one parameter and a sequence
        # spec for another: all three flattened placeholders get unbacked dim 0 symbols.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module taking a tensor and a two element list.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Return the dim 0 sum of a plus those of both elements
                # of b.
                return a.sum(0) + b[0].sum(0) + b[1].sum(0)

        ep = export(
            M(),
            args=(
                torch.randn(8, 3),
                [torch.randn(5, 3), torch.randn(6, 3)],
            ),
            dynamic_shapes=PARAMS(
                {
                    "a": T([VAR("A"), STATIC]),
                    "b": L(
                        [
                            T([VAR("B0"), STATIC]),
                            T([VAR("B1"), STATIC]),
                        ]
                    ),
                }
            ),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'a: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_0: "f32\[u\d+, 3\]"')
        self.assertRegex(ep_str, r'b_1: "f32\[u\d+, 3\]"')

    # ---- Partial-spec cases ----

    def test_seq_spec_shorter_than_runtime_list_tail_static(self):
        # [FUNCTION DOC] A sequence spec shorter than the runtime list leaves the tail static:
        # the first element is unbacked, the other two keep literal shapes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module summing three list elements.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Return the sum of the first three elements of the
                # input list.
                return xs[0].sum() + xs[1].sum() + xs[2].sum()

        ep = export(
            M(),
            args=([torch.randn(8, 3), torch.randn(5, 3), torch.randn(6, 3)],),
            dynamic_shapes=PARAMS({"xs": L([T([VAR("A"), STATIC])])}),
            strict=self.strict,
        )
        ep_str = str(ep)
        self.assertRegex(ep_str, r'xs_0: "f32\[u\d+, 3\]"')
        self.assertIn('xs_1: "f32[5, 3]"', ep_str)
        self.assertIn('xs_2: "f32[6, 3]"', ep_str)

    # ---- Negative / structural cases ----

    def test_seq_spec_on_dict_arg_raises(self):
        # [FUNCTION DOC] A sequence spec applied to a dict argument must raise ValueError saying
        # a sequence was expected but a dict was found.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the entry a of a dict.
            def forward(self, d):
                # [FUNCTION DOC] HELPER: Return the value stored under key a.
                return d["a"]

        with self.assertRaisesRegex(
            ValueError,
            r"SeqSpec expected list/tuple, got dict",
        ):
            export(
                M(),
                args=({"a": torch.randn(4)},),
                dynamic_shapes=PARAMS({"d": L([T([VAR("A")])])}),
                strict=self.strict,
            )

    def test_seq_spec_longer_than_runtime_list_raises(self):
        # [FUNCTION DOC] A sequence spec with two entries applied to a list of length one must
        # raise ValueError reporting two entries beyond the runtime sequence length.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the first list element.
            def forward(self, xs):
                # [FUNCTION DOC] HELPER: Return the first element of the input list.
                return xs[0]

        with self.assertRaisesRegex(
            ValueError,
            r"SeqSpec has 2 entries beyond runtime sequence length 1",
        ):
            export(
                M(),
                args=([torch.randn(4)],),
                dynamic_shapes=PARAMS(
                    {
                        "xs": L(
                            [
                                T([VAR("A")]),
                                T([VAR("B")]),
                            ]
                        )
                    }
                ),
                strict=self.strict,
            )

    def test_dict_spec_key_not_in_runtime_dict_raises(self):
        # [FUNCTION DOC] A dict spec naming a key absent from the runtime dict must raise
        # ValueError listing the missing key and the runtime keys, prefixed with the path to
        # the parameter.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the entry a of a dict.
            def forward(self, d):
                # [FUNCTION DOC] HELPER: Return the value stored under key a.
                return d["a"]

        with self.assertRaisesRegex(
            ValueError,
            r"shapes_spec\['d'\]: DictSpec has entries \['missing'\] that do not match any key in the runtime dict\. Runtime keys: \['a'\]",
        ):
            export(
                M(),
                args=({"a": torch.randn(4)},),
                dynamic_shapes=PARAMS({"d": DICT({"missing": T([VAR("A")])})}),
                strict=self.strict,
            )

    def test_object_spec_attr_not_on_runtime_object_raises(self):
        # [FUNCTION DOC] An object spec naming an attribute that the runtime object lacks must
        # raise ValueError saying the entries do not match any attribute.
        import dataclasses

        @dataclasses.dataclass
        class Box2:
            # [CLASS DOC] HELPER: Dataclass with one tensor field, registered as a pytree node.
            x: torch.Tensor

        pytree.register_dataclass(Box2)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the field x of the dataclass.
            def forward(self, box):
                # [FUNCTION DOC] HELPER: Return the field x.
                return box.x

        with self.assertRaisesRegex(
            ValueError,
            r"ObjectSpec has entries .*'nope'.* that do not match any attribute",
        ):
            export(
                M(),
                args=(Box2(torch.randn(4)),),
                dynamic_shapes=PARAMS({"box": OBJ({"nope": T([VAR("A")])})}),
                strict=self.strict,
            )

    def test_where_path_accumulates_in_nested_error(self):
        # [FUNCTION DOC] Error path text test: a dict spec inside an object spec inside a dict
        # spec, with a nested sequence spec whose second item names a missing key. Expects the
        # ValueError message to carry the full accumulated path (key access, attribute access and
        # index access forms) followed by the missing-key explanation and runtime keys.
        """Errors from nested specs should report the full path to the
        offending spot, exercising all three ``where``-formatting
        variants: ``[str]`` (DictSpec), ``.attr`` (ObjectSpec), and
        ``[int]`` (SeqSpec)."""
        import dataclasses

        @dataclasses.dataclass
        class Box:
            # [CLASS DOC] HELPER: Dataclass with a single list field, registered as a
            # pytree node.
            items: list

        pytree.register_dataclass(Box)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module that digs into the dataclass list for a
            # nested value.
            def forward(self, batch):
                # [FUNCTION DOC] HELPER: Return the data entry of the second list item inside
                # entry b of the input.
                return batch["b"].items[1]["data"]

        with self.assertRaisesRegex(
            ValueError,
            re.escape(
                "shapes_spec['batch']['b'].items[1]: "
                "DictSpec has entries ['missing'] that do not match "
                "any key in the runtime dict. Runtime keys: ['data']"
            ),
        ):
            export(
                M(),
                args=({"b": Box(items=[torch.randn(3), {"data": torch.randn(3)}])},),
                dynamic_shapes=PARAMS(
                    {
                        "batch": DICT(
                            {
                                "b": OBJ(
                                    {
                                        "items": L(
                                            [
                                                None,
                                                DICT({"missing": T([VAR("A")])}),
                                            ]
                                        )
                                    }
                                )
                            }
                        )
                    }
                ),
                strict=self.strict,
            )

    def test_object_spec_on_pytree_node_without_keys_fn_raises(self):
        # [FUNCTION DOC] An object spec on a pytree node registered without a keyed flatten
        # function must raise ValueError saying export requires that function for the type.
        # Registers the type globally in pytree.
        class KeyslessContainer:
            # [CLASS DOC] HELPER: Plain class registered as a pytree node without a keyed
            # flatten function.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store the given value as attribute x.
                self.x = x

        pytree.register_pytree_node(
            KeyslessContainer,
            lambda c: ([c.x], None),
            lambda values, _context: KeyslessContainer(next(iter(values))),
            serialized_type_name="test_dynamic_spec_export.KeyslessContainer",
        )

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module returning the field x of its input.
            def forward(self, c):
                # [FUNCTION DOC] HELPER: Return attribute x.
                return c.x

        with self.assertRaisesRegex(
            ValueError,
            r"export requires `flatten_with_keys_fn` to be registered for "
            r"type KeyslessContainer",
        ):
            export(
                M(),
                args=(KeyslessContainer(torch.randn(4)),),
                dynamic_shapes=PARAMS({"c": OBJ({"x": T([VAR("A")])})}),
                strict=self.strict,
            )

    # ---- Alignment invariant: walker order == pytree.tree_flatten order ----

    def test_walker_alignment_with_pytree_flatten(self):
        # [FUNCTION DOC] Alignment invariant: for a set of nested inputs (bare tensor, list,
        # tuple, dicts in different insertion order, dataclasses, nesting), build a complete spec
        # that targets every tensor with a unique ShapeVar name encoding its object id, run the
        # spec walker and check the number of returned leaf specs equals the pytree leaf count
        # and that the spec at each position names the tensor at the same position in pytree leaf
        # order. Registers a dataclass globally as a pytree node.
        """For each structured input value, ``_walk_spec`` must
        return leaf specs in exactly the same order as
        ``pytree.tree_flatten(value)``.

        Proof technique: build a "complete" container spec where every
        leaf is a unique TensorSpec whose ShapeVar name encodes the
        ``id()`` of the tensor it targets. After flattening the user
        spec, assert that the i-th returned spec carries the name
        corresponding to the i-th tensor from
        ``pytree.tree_leaves(value)``. Any drift in iteration order
        would cause a name mismatch.
        """
        import dataclasses

        from torch.fx.experimental._spec_binding import _walk_spec

        @dataclasses.dataclass
        class _Box:
            # [CLASS DOC] HELPER: Local dataclass with two tensor fields u and v, registered as a
            # pytree node.
            u: torch.Tensor
            v: torch.Tensor

        pytree.register_dataclass(_Box)

        def _build_complete_spec(v):
            # [FUNCTION DOC] HELPER: Recursively build a spec mirroring a value: a tensor yields
            # a tensor spec with a uniquely named ShapeVar based on its object id, list or tuple
            # yields a sequence spec, dict a dict spec, dataclass instance an object spec over
            # its fields, anything else None (static).
            """Build a container spec that targets every tensor in v
            with a uniquely-named ShapeVar."""
            if isinstance(v, torch.Tensor):
                return T([VAR(f"t{id(v)}")])
            if isinstance(v, (list, tuple)):
                return L([_build_complete_spec(e) for e in v])
            if isinstance(v, dict):
                return DICT({k: _build_complete_spec(val) for k, val in v.items()})
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return OBJ(
                    {
                        f.name: _build_complete_spec(getattr(v, f.name))
                        for f in dataclasses.fields(v)
                    }
                )
            # Leaf type with no spec needed (None / int / etc.)
            return None

        cases = [
            # Bare tensor — trivial leaf case.
            torch.randn(3),
            # Plain containers.
            [torch.randn(3), torch.randn(4)],
            (torch.randn(3), torch.randn(4)),
            {"a": torch.randn(3), "b": torch.randn(4)},
            # Different dict insertion order — must still align.
            {"z": torch.randn(3), "a": torch.randn(4), "m": torch.randn(5)},
            # Dataclass.
            _Box(torch.randn(3), torch.randn(4)),
            # Nested dict-of-list-of-tensors.
            {"x": [torch.randn(3), torch.randn(4)], "y": torch.randn(5)},
            # Nested list-of-dataclass-and-dict.
            [
                _Box(torch.randn(3), torch.randn(4)),
                {"k": torch.randn(5), "j": torch.randn(6)},
            ],
            # Deeply nested.
            {
                "a": [
                    _Box(torch.randn(2), torch.randn(3)),
                    {"inner": [torch.randn(4), torch.randn(5)]},
                ],
                "b": torch.randn(6),
            },
        ]

        for arg_value in cases:
            spec = _build_complete_spec(arg_value)
            leaves = pytree.tree_leaves(arg_value)
            out = _walk_spec(spec, arg_value, where="<root>")
            self.assertEqual(
                len(out),
                len(leaves),
                lambda msg: f"{msg}\nleaf-count drift for case {arg_value!r}",
            )
            # Per-slot check: each slot's spec name must match the
            # tensor at that flat position from pytree.tree_flatten.
            for i, (slot_spec, leaf) in enumerate(zip(out, leaves, strict=True)):
                self.assertIsInstance(
                    slot_spec,
                    T,
                    msg=lambda msg: f"{msg}\nslot {i} is {slot_spec!r} (expected TensorSpec) "
                    f"for case {arg_value!r}",
                )
                expected = f"t{id(leaf)}"
                actual = slot_spec._specs[0].name
                self.assertEqual(
                    actual,
                    expected,
                    msg=(
                        lambda msg: f"{msg}\nalignment drift at slot {i}: spec name "
                        f"{actual!r} does not match pytree-flatten leaf "
                        f"name {expected!r} for case {arg_value!r}"
                    ),
                )

    def test_walker_alignment_no_spec_advances_correctly(self):
        # [FUNCTION DOC] With no user spec, the spec walker must return exactly as many leaf
        # entries as the pytree leaf count for several nested inputs, i.e. unspecified subtrees
        # stay in lockstep with flattening.
        """When user_spec=None, the walker must return exactly
        len(pytree.tree_leaves(value)) — i.e. no-spec subtrees stay in
        lockstep with pytree's flatten."""
        from torch.fx.experimental._spec_binding import _walk_spec

        cases = [
            torch.randn(3),
            [torch.randn(3), torch.randn(4), torch.randn(5)],
            {"a": [torch.randn(3), torch.randn(4)], "b": torch.randn(5)},
            (torch.randn(3), {"k": torch.randn(4)}, [torch.randn(5)]),
        ]
        for arg_value in cases:
            expected = len(pytree.tree_leaves(arg_value))
            out = _walk_spec(None, arg_value, where="<root>")
            self.assertEqual(
                len(out),
                expected,
                lambda msg: f"{msg}\nno-spec leaf-count drift for {arg_value!r}",
            )


class TestContainerSpecStrict(_TestContainerSpecBase):
    # [CLASS DOC] Concrete strict-mode instantiation of the container spec tests; sets the strict
    # flag to True.
    strict = True


class TestContainerSpecNonStrict(_TestContainerSpecBase):
    # [CLASS DOC] Concrete non-strict-mode instantiation of the container spec tests; sets the
    # strict flag to False.
    strict = False


del _TestContainerSpecBase
if __name__ == "__main__":
    run_tests()
