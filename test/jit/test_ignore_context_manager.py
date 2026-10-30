# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for torch.jit._IgnoreContextManager in scripted modules: the with-body is
# outlined, with named inputs and outputs declared as annotated strings, and results must match
# eager execution. Not runnable directly: running the file as a script raises an error that
# points to the umbrella test runner.

import os
import sys

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestIgnoreContextManager(JitTestCase):
    # [CLASS DOC] JitTestCase with tests of the ignore context manager under different
    # input/output declarations.
    def test_with_ignore_context_manager_with_inp_out(self):
        # [FUNCTION DOC] Scripts three modules whose with-body uses declared inputs and outputs
        # and checks each scripted result equals eager execution and the expected constants (20,
        # 11 and 6).
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward declares two integer inputs
            # and two integer outputs for the ignored block.
            def forward(self):
                # [FUNCTION DOC] HELPER: Sets four integers, runs a block that computes
                # from a list comprehension and sets both outputs, then returns their
                # sum (20 in practice).
                a: int = 4
                b: int = 5
                c: int = 0
                d: int = 6
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int", d="out:int"
                ):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                    d = 9
                return c + d

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), model())
        self.assertEqual(s(), 20)

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward declares two inputs and one
            # output for the ignored block.
            def forward(self):
                # [FUNCTION DOC] HELPER: Runs a block with two integer inputs and one output
                # computed from a list comprehension; returns the output (11 in practice).
                a: int = 4
                b: int = 5
                c: int = 0
                with torch.jit._IgnoreContextManager(
                    a="inp:int", b="inp:int", c="out:int"
                ):
                    l = [2 for i in range(a) if i > 2]
                    c = l[0] + a + b
                return c

        model = B()
        s = torch.jit.script(model)
        self.assertEqual(s(), 11)
        self.assertEqual(s(), model())

        class C(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward declares one input and one
            # output (b is both overwritten and returned).
            def forward(self):
                # [FUNCTION DOC] HELPER: Runs a block with input a and output b computed from a
                # list comprehension; returns b (6 in practice).
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="out:int"):
                    l = [2 for i in range(a) if i > 2]
                    b = l[0] + a
                return b

        model = C()
        s = torch.jit.script(model)
        self.assertEqual(s(), 6)
        self.assertEqual(s(), model())

    def test_with_ignore_context_manager_with_just_inp(self):
        # [FUNCTION DOC] Scripts a module whose ignored block takes only inputs and produces no
        # outputs, and checks scripted result equals eager and equals 4.
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward declares only inputs for
            # the ignored block.
            def forward(self):
                # [FUNCTION DOC] HELPER: Runs a block with two inputs that builds an unused
                # list; returns the first integer (4).
                a: int = 4
                b: int = 5
                with torch.jit._IgnoreContextManager(a="inp:int", b="inp:int"):
                    l = [2 + b for i in range(a) if i > 2]  # noqa: F841
                return a

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 4)
        self.assertEqual(s(), model())

    def test_with_ignore_context_manager_with_just_out(self):
        # [FUNCTION DOC] Scripts a module whose ignored block only declares an output (a list of
        # ints) and checks scripted result equals eager and equals 5.
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no state whose forward declares only an output
            # list for the ignored block.
            def forward(self):
                # [FUNCTION DOC] HELPER: The ignored block builds a list of four twos;
                # forward then sets the first element to 3 and returns the sum of the
                # first two elements (5).
                with torch.jit._IgnoreContextManager(c="out:List[int]"):
                    c = [2 for i in range(7) if i > 2]
                c[0] = 3
                return c[0] + c[1]

        model = A()
        s = torch.jit.script(model)
        self.assertEqual(s(), 5)
        self.assertEqual(s(), model())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
