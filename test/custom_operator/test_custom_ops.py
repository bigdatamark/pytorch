# Owner(s): ["module: cpp-extensions"]

# [MODULE DOC] Tests for custom operators loaded from a prebuilt shared library: python stub
# module suggestions, fake implementations, calls with strings, tensors and autograd, and use
# inside a saved script module. Import-time side effect: imports the pointwise module through
# torch.ops. Defines TestCustomOperators.

import sys
import tempfile
import unittest

from model import get_custom_op_library_path, Model

import torch
import torch._library.utils as utils
from torch import ops
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


torch.ops.import_module("pointwise")


class TestCustomOperators(TestCase):
    # [CLASS DOC] Test cases for ops of the custom library; setUp loads the library from the
    # build path each time.
    def setUp(self):
        # [FUNCTION DOC] HELPER: stores the library path and loads the library.
        super().setUp()
        self.library_path = get_custom_op_library_path()
        ops.load_library(self.library_path)

    def test_custom_library_is_loaded(self):
        # [FUNCTION DOC] Asserts the library path appears in the list of loaded libraries.
        self.assertIn(self.library_path, ops.loaded_libraries)

    def test_op_with_no_abstract_impl_pystub(self):
        # [FUNCTION DOC] Calls custom tan on a meta tensor. If the build requires a python module
        # to be set, expects a RuntimeError mentioning pointwise; otherwise just checks it runs.
        x = torch.randn(3, device="meta")
        if utils.requires_set_python_module():
            with self.assertRaisesRegex(RuntimeError, "pointwise"):
                torch.ops.custom.tan(x)
        else:
            # Smoketest
            torch.ops.custom.tan(x)

    def test_op_with_incorrect_abstract_impl_pystub(self):
        # [FUNCTION DOC] Calling custom cos on a meta tensor must raise RuntimeError mentioning
        # pointwise, because its declared stub module differs from where the fake impl lives.
        x = torch.randn(3, device="meta")
        with self.assertRaisesRegex(RuntimeError, "pointwise"):
            torch.ops.custom.cos(x)

    @unittest.skipIf(IS_WINDOWS, "torch.compile not supported on windows")
    def test_dynamo_pystub_suggestion(self):
        # [FUNCTION DOC] Compiles a function calling custom asin (full graph, eager backend) and
        # expects a RuntimeError saying fake tensors are unsupported and suggesting importing the
        # nonexistent stub module. Skipped on Windows.
        x = torch.randn(3)

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: returns custom asin applied to x.
            return torch.ops.custom.asin(x)

        with self.assertRaisesRegex(
            RuntimeError,
            r"(?s)Operator does not support running with fake tensors.*you may need to `import nonexistent`",
        ):
            f(x)

    def test_abstract_impl_pystub_faketensor(self):
        # [FUNCTION DOC] Symbolic tracing of custom nonzero fails with UnsupportedOperatorException
        # before importing my_custom_ops; after importing it, tracing works and the graph code
        # matches the expected single call to the op.
        from functorch import make_fx

        x = torch.randn(3, device="cpu")
        self.assertNotIn("my_custom_ops", sys.modules.keys())

        with self.assertRaises(
            torch._subclasses.fake_tensor.UnsupportedOperatorException
        ):
            gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)

        torch.ops.import_module("my_custom_ops")
        gm = make_fx(torch.ops.custom.nonzero.default, tracing_mode="symbolic")(x)
        self.assertExpectedInline(
            """\
def forward(self, arg0_1):
    nonzero = torch.ops.custom.nonzero.default(arg0_1);  arg0_1 = None
    return nonzero
""".strip(),
            gm.code.strip(),
        )

    def test_abstract_impl_pystub_meta(self):
        # [FUNCTION DOC] Calling custom sin on a meta tensor raises NotImplementedError naming
        # my_custom_ops2 until that module is imported; afterwards the call succeeds.
        x = torch.randn(3, device="meta")
        self.assertNotIn("my_custom_ops2", sys.modules.keys())
        with self.assertRaisesRegex(NotImplementedError, r"'my_custom_ops2'"):
            torch.ops.custom.sin.default(x)
        torch.ops.import_module("my_custom_ops2")
        torch.ops.custom.sin.default(x)

    def test_calling_custom_op_string(self):
        # [FUNCTION DOC] The string op returns a negative value when the first string sorts lower
        # and zero for equal strings.
        output = ops.custom.op2("abc", "def")
        self.assertLess(output, 0)
        output = ops.custom.op2("abc", "abc")
        self.assertEqual(output, 0)

    def test_calling_custom_op(self):
        # [FUNCTION DOC] The tensor op with a tensor, float and int returns a list of three tensors
        # equal to twice the input; the defaults variant returns a one-element list equal to the
        # input.
        output = ops.custom.op(torch.ones(5), 2.0, 3)
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 3)
        for tensor in output:
            self.assertTrue(tensor.allclose(torch.ones(5) * 2))

        output = ops.custom.op_with_defaults(torch.ones(5))
        self.assertEqual(type(output), list)
        self.assertEqual(len(output), 1)
        self.assertTrue(output[0].allclose(torch.ones(5)))

    def test_calling_custom_op_with_autograd(self):
        # [FUNCTION DOC] Checks the autograd op value and gradients, without and with the optional
        # third tensor argument, via backward with a given gradient output.
        x = torch.randn((5, 5), requires_grad=True)
        y = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y)
        self.assertTrue(output.allclose(x + 2 * y + x * y))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        grad = torch.ones(5, 5)

        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)

        # Test with optional arg.
        x.grad.zero_()
        y.grad.zero_()
        z = torch.randn((5, 5), requires_grad=True)
        output = ops.custom.op_with_autograd(x, 2, y, z)
        self.assertTrue(output.allclose(x + 2 * y + x * y + z))

        go = torch.ones((), requires_grad=True)
        output.sum().backward(go, False, True)
        self.assertEqual(x.grad, y + grad)
        self.assertEqual(y.grad, x + grad * 2)
        self.assertEqual(z.grad, grad)

    def test_calling_custom_op_with_autograd_in_nograd_mode(self):
        # [FUNCTION DOC] The autograd op still gives the correct value inside a no-grad region.
        with torch.no_grad():
            x = torch.randn((5, 5), requires_grad=True)
            y = torch.randn((5, 5), requires_grad=True)
            output = ops.custom.op_with_autograd(x, 2, y)
            self.assertTrue(output.allclose(x + 2 * y + x * y))

    def test_calling_custom_op_inside_script_module(self):
        # [FUNCTION DOC] A script Model forward on ones returns ones plus one.
        model = Model()
        output = model.forward(torch.ones(5))
        self.assertTrue(output.allclose(torch.ones(5) + 1))

    def test_saving_and_loading_script_module_with_custom_op(self):
        # [FUNCTION DOC] Saves a Model to a closed temporary file name, loads it with the jit
        # loader and checks forward on ones gives ones plus one.
        model = Model()
        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually.
        with tempfile.NamedTemporaryFile() as file:
            file.close()
            model.save(file.name)
            loaded = torch.jit.load(file.name)

            output = loaded.forward(torch.ones(5))
            self.assertTrue(output.allclose(torch.ones(5) + 1))


if __name__ == "__main__":
    run_tests()
