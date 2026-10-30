# Owner(s): ["oncall: jit"]

# [MODULE DOC] JIT tests for the torch.ops custom-operator namespace machinery: lazy namespace
# and op creation, argument validation errors, default arguments, and use of ops from eager,
# scripted and traced code. Importing adds the test directory to sys.path; running the file
# directly is refused in favor of test/test_jit.py.
import os
import sys
import unittest

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


def canonical(graph):
    # [FUNCTION DOC] HELPER: Returns the canonicalized text form of a JIT graph (without source
    # ranges), used so expected-graph strings are stable.
    return torch._C._jit_pass_canonicalize(graph).str(False)


class TestCustomOperators(JitTestCase):
    # [CLASS DOC] Test case for calling operators through torch.ops, covering dynamic namespace
    # registration, error messages for bad arguments, and scripted/traced invocation.
    def test_dynamic_op_registry(self):
        # [FUNCTION DOC] Checks torch.ops creates namespaces and ops lazily: the _test namespace
        # is removed from the ops dict if present, absent before first access and present
        # afterwards with type _OpNamespace; likewise leaky_relu is cached in the namespace dict
        # only after first access and repeated access returns an equal op. Mutates torch.ops
        # global state.
        from torch._ops import _OpNamespace

        self.assertTrue(hasattr(torch, "ops"))

        if "_test" in torch.ops.__dict__:
            torch.ops.__dict__.pop("_test")

        # Don't use `hasattr()` because it will call `__getattr__`.
        self.assertNotIn("_test", torch.ops.__dict__)
        torch.ops._test
        self.assertIn("_test", torch.ops.__dict__)
        self.assertEqual(type(torch.ops._test), _OpNamespace)

        self.assertNotIn("leaky_relu", torch.ops._test.__dict__)
        op = torch.ops._test.leaky_relu
        self.assertTrue(callable(op))
        self.assertIn("leaky_relu", torch.ops._test.__dict__)
        op2 = torch.ops._test.leaky_relu
        self.assertEqual(op, op2)

    def test_getting_invalid_attr(self):
        # [FUNCTION DOC] Verifies that accessing the dunder attributes __origin__ and __self__
        # on an op namespace raises AttributeError with an "Invalid attribute" message naming
        # the _OpNamespace and namespace _test.
        for attr in ["__origin__", "__self__"]:
            with self.assertRaisesRegexWithHighlight(
                AttributeError,
                f"Invalid attribute '{attr}' for '_OpNamespace' '_test'",
                "",
            ):
                getattr(torch.ops._test, attr)

    def test_simply_calling_an_operator(self):
        # [FUNCTION DOC] Calls the aten relu op via torch.ops on a random tensor and checks the
        # result equals the tensor relu method.
        input = torch.randn(100)
        output = torch.ops.aten.relu(input)
        self.assertEqual(output, input.relu())

    def test_default_arguments_are_used(self):
        # [FUNCTION DOC] Calls the _test leaky_relu op without a slope argument and checks the
        # default slope is applied (negative input scaled by 0.01, positive unchanged).
        output = torch.ops._test.leaky_relu(torch.tensor([-1.0, 1.0]))
        self.assertEqual(output, torch.tensor([-0.01, 1]))

    def test_passing_too_many_args(self):
        # [FUNCTION DOC] Passing two positional args to the one-argument relu op must raise
        # RuntimeError saying at most 1 argument was expected but 2 were received.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::relu\(\) expected at most 1 argument\(s\) but received 2 argument\(s\)",
            "",
        ):
            torch.ops.aten.relu(1, 2)

    def test_passing_too_few_args(self):
        # [FUNCTION DOC] Calling relu with no arguments must raise RuntimeError reporting the
        # missing value for argument 'self'.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"aten::relu\(\) is missing value for argument 'self'.", ""
        ):
            torch.ops.aten.relu()

    def test_passing_one_positional_but_not_the_second(self):
        # [FUNCTION DOC] Calling type_as with only the first tensor must raise RuntimeError
        # reporting the missing value for argument 'other'.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"aten::type_as\(\) is missing value for argument 'other'.",
            "",
        ):
            torch.ops.aten.type_as(torch.ones(5, 5))

    def test_passing_unknown_kwargs(self):
        # [FUNCTION DOC] Passing an unknown keyword argument to the _test leaky_relu op must
        # raise RuntimeError naming the unknown keyword 'foo' and the operator.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unknown keyword argument 'foo' for operator '_test::leaky_relu'",
            "",
        ):
            torch.ops._test.leaky_relu(torch.ones(5), foo=torch.ones(5))

    def test_passing_and_returning_lists(self):
        # [FUNCTION DOC] Passes a list of two random tensors to the _test cat op and checks the
        # returned tensor equals the eager concatenation of the same list.
        # Replace with actual test once we support lists.
        a, b = torch.rand(5), torch.rand(5)
        output = torch.ops._test.cat([a, b])
        output_ref = torch.cat([a, b])
        self.assertEqual(output, output_ref)

    def test_calling_scripted_custom_op(self):
        # [FUNCTION DOC] Scripts a function that calls the aten relu op via torch.ops and checks
        # the scripted result matches the relu method on a ones tensor.
        @torch.jit.script
            # [FUNCTION DOC] HELPER: Scripted function applying the aten relu op to its input.
        def func(x):
            return torch.ops.aten.relu(x)

        input = torch.ones(5, 5)
        self.assertEqual(func(input), input.relu())

    def test_calling_traced_custom_op(self):
        # [FUNCTION DOC] Traces the aten relu op on a ones tensor example and checks the traced
        # callable's output matches the relu method.
        input = torch.ones(5, 5)
        func = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertEqual(func(input), input.relu())

    @unittest.skip(
        "Need to figure out default dtype differences between fbcode and oss"
    )
    def test_script_graph_for_custom_ops_matches_traced_graph(self):
        # [FUNCTION DOC] Skipped test (default dtype differs between fbcode and OSS): would
        # compare the canonical traced graph of the aten relu op against an expected graph with
        # one relu node.
        input = torch.ones(5, 5)
        trace = torch.jit.trace(torch.ops.aten.relu, [input])
        self.assertExpectedInline(
            canonical(trace.graph),
            """\
graph(%0 : Float(5, 5)):
  %1 : Float(5, 5) = aten::relu(%0)
  return (%1)
""",
        )

    def test_script_graph_contains_custom_op(self):
        # [FUNCTION DOC] Scripts a function calling the aten relu op and checks the canonical
        # graph contains a single aten::relu node applied to the input.
        @torch.jit.script
            # [FUNCTION DOC] HELPER: Scripted function applying the aten relu op to its input.
        def func(x):
            return torch.ops.aten.relu(x)

        self.assertExpectedInline(
            canonical(func.graph),
            """\
graph(%x.1 : Tensor):
  %1 : Tensor = aten::relu(%x.1)
  return (%1)
""",
        )

    def test_generic_list(self):
        # [FUNCTION DOC] Checks the _test get_first op accepts a nested list of strings and
        # returns the first string, "hello".
        self.assertEqual(torch.ops._test.get_first([["hello"]]), "hello")

    # https://github.com/pytorch/pytorch/issues/80508
    def test_where_no_scalar(self):
        # [FUNCTION DOC] Regression test for issue 80508: calling the aten where op with a
        # tensor condition and two Python float scalars must not raise. Result is not checked.
        x = torch.rand(1, 3, 224, 224)
        torch.ops.aten.where(x > 0.5, -1.5, 1.5)  # does not raise


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
