# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the AOTI minifier support utilities: the export wrapper that returns
# None for graphs export cannot handle (or raises a wrapped error), and the module-to-string
# conversion used to emit repro scripts, including nested conditional branch submodules. No
# import-time side effects.
from unittest.mock import patch

import torch
from torch._dynamo.debug_utils import NNModuleToString
from torch._dynamo.exc import UserError, UserErrorType
from torch._dynamo.repro.aoti import (
    AOTIMinifierError,
    export_for_aoti_minifier,
    get_module_string,
)
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class MinifierUtilsTests(TestCase):
    # [CLASS DOC] Test case covering export_for_aoti_minifier error classification and
    # NNModuleToString/get_module_string serialization.
    def test_invalid_output_user_error_classification(self):
        # [FUNCTION DOC] With the export call mocked to raise a user error of invalid-output
        # type, checks that the export wrapper in strict mode with skip_export_error False
        # swallows it and returns None.
        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Trivial module whose forward adds one to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

        with patch(
            "torch.export.export",
            side_effect=UserError(
                UserErrorType.INVALID_OUTPUT,
                "mocked invalid output user error",
            ),
        ):
            gm = export_for_aoti_minifier(
                SimpleModel(),
                (torch.randn(2, 2),),
                strict=True,
                skip_export_error=False,
            )
            self.assertTrue(gm is None)

    def test_invalid_output(self):
        # [FUNCTION DOC] Builds a graph by symbolic tracing a module whose forward returns a
        # submodule instead of a tensor, then checks the export wrapper returns None for both
        # strict and non-strict export (an illegal output is treated as a non-failing graph).
        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one 2x2 linear layer whose forward returns that layer
            # rather than a tensor, giving an invalid graph output.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear submodule.
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear submodule itself (not a tensor),
                # producing an invalid output.
                # return a graph module
                return self.linear

        model = SimpleModel()
        # Here we obtained a graph with invalid output by symbolic_trace for simplicity,
        # it can also obtained from running functorch.compile.minifier on an exported graph.
        traced = torch.fx.symbolic_trace(model)
        for strict in [True, False]:
            gm = export_for_aoti_minifier(traced, (torch.randn(2, 2),), strict=strict)
            self.assertTrue(gm is None)

    def test_non_exportable(self):
        # [FUNCTION DOC] Feeds a sum module two tensors (wrong arity) so export fails, then for
        # strict and non-strict export checks that skip_export_error True gives None (and prints
        # the result), while skip_export_error False raises AOTIMinifierError.
        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns the sum of the input; used with
            # mismatched inputs to force an export failure.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of all elements of the input.
                return x.sum()

        model = SimpleModel()
        # Force export failure by providing an input with in-compatible shapes
        inputs = (torch.randn(2), torch.randn(2))
        for strict in [True, False]:
            gm = export_for_aoti_minifier(
                model, inputs, strict=strict, skip_export_error=True
            )
            print(gm)
            self.assertTrue(gm is None)

            with self.assertRaises(AOTIMinifierError):
                export_for_aoti_minifier(
                    model, inputs, strict=strict, skip_export_error=False
                )

    def test_convert_module_to_string(self):
        # [FUNCTION DOC] Exports a module containing a conditional on a scalar read from a
        # tensor (non-strict), converts the unchecked module to repro source text, and compares
        # it with an expected string showing both branch classes as commented-out source plus
        # the main Repro class.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward reads a scalar from the flag tensor and runs
            # a conditional where both branches clone the input.
            def forward(self, x, flag):
                # [FUNCTION DOC] HELPER: Extracts the flag as a Python number and returns
                # the result of a conditional that clones x in either branch.
                flag = flag.item()

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: Branch body returning a clone of x.
                    return x.clone()

                return torch.cond(flag > 0, true_fn, true_fn, [x])

        inputs = (
            torch.rand(28, 28),
            torch.tensor(1),
        )

        model = M()
        gm = torch.export.export(model, inputs, strict=False).module(check_guards=False)

        model_string = get_module_string(gm)
        self.assertExpectedInline(
            model_string.strip(),
            """\
# from torch.nn import *
# class true_graph_0_0(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()



#     def forward(self, x):
#         clone = torch.ops.aten.clone.default(x);  x = None
#         return (clone,)


# class false_graph_0_1(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()



#     def forward(self, x):
#         clone = torch.ops.aten.clone.default(x);  x = None
#         return (clone,)


# class Repro(torch.nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.true_graph_0 = true_graph_0_0()
#         self.false_graph_0 = false_graph_0_1()



#     def forward(self, x, flag):
#         x, flag, = fx_pytree.tree_flatten_spec(([x, flag], {}), self._in_spec)
#         item = torch.ops.aten.item.default(flag);  flag = None
#         gt = item > 0;  item = None
#         true_graph_0 = self.true_graph_0
#         false_graph_0 = self.false_graph_0
#         cond = torch.ops.higher_order.cond(gt, true_graph_0, false_graph_0, (x,));  gt = true_graph_0 = false_graph_0 = x = None
#         getitem = cond[0];  cond = None
#         return pytree.tree_unflatten((getitem,), self._out_spec)""",
        )

    def test_cond_subgraph_roundtrips(self):
        # [FUNCTION DOC] Traces a conditional with sin and cos branches using make_fx,
        # serializes it with NNModuleToString, checks branch bodies appear and no empty
        # lambda placeholder remains, then executes the generated source and verifies the
        # rebuilt Repro module matches the traced graph for both predicate values.
        # torch.cond attaches its branches as nested GraphModule submodules.
        # NNModuleToString.convert must serialize them as real nested classes
        # that reconstruct -- not the invalid, body-less `<lambda>()` placeholder.
        def fn(pred, x):
            # [FUNCTION DOC] HELPER: Function applying a conditional that returns the sine of x if
            # pred is true, otherwise the cosine.
            return torch.cond(pred, lambda x: x.sin(), lambda x: x.cos(), (x,))

        pred = torch.tensor(False)
        x = torch.randn(4)
        gm = make_fx(fn, tracing_mode="real")(pred, x)

        result = NNModuleToString.convert(gm)

        # The branch bodies must be present, with no invalid placeholder.
        self.assertIn("sin", result)
        self.assertIn("cos", result)
        self.assertNotIn("<lambda>", result)

        # The generated source must reconstruct a module whose forward matches
        # the original graph for both branches.
        ns = {"torch": torch}
        exec(result, ns)
        repro = ns["Repro"]()
        self.assertEqual(repro(torch.tensor(False), x), gm(torch.tensor(False), x))
        self.assertEqual(repro(torch.tensor(True), x), gm(torch.tensor(True), x))


if __name__ == "__main__":
    run_tests()
