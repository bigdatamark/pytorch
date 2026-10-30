# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for the builtin attribute getter with a default value: on a
# module with a missing attribute, on a named tuple, and on a plain tuple (which must fail to
# compile). Raises an error if run directly; run via the main jit test file.

from typing import NamedTuple, Tuple

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestGetDefaultAttr(JitTestCase):
    # [CLASS DOC] JIT test case for the default-value form of the attribute getter in scripted
    # code.
    def test_getattr_with_default(self):
        # [FUNCTION DOC] Scripts a module whose forward reads an existing attribute and a missing
        # one with a list default, appending the former to the latter. Checks the eager result has
        # two entries, then checks the graph with text pattern checks: it contains a read of the
        # existing attribute, no mention of the missing name, and a list construction of floats
        # for the default.
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a float attribute set to one; its forward uses the
            # attribute getter with and without a default.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor and sets the float attribute
                # to one.
                super().__init__()
                self.init_attr_val = 1.0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads the existing attribute and a missing attribute with
                # a one-element float list default, appends the former to the latter and returns
                # that list.
                y = getattr(self, "init_attr_val")  # noqa: B009
                w: list[float] = [1.0]
                z = getattr(self, "missing", w)
                z.append(y)
                return z

        result = A().forward(0.0)
        self.assertEqual(2, len(result))
        graph = torch.jit.script(A()).graph

        # The "init_attr_val" attribute exists
        FileCheck().check('prim::GetAttr[name="init_attr_val"]').run(graph)
        # The "missing" attribute does not exist, so there should be no corresponding GetAttr in AST
        FileCheck().check_not("missing").run(graph)
        # instead the getattr call will emit the default value, which is a list with one float element
        FileCheck().check("float[] = prim::ListConstruct").run(graph)

    def test_getattr_named_tuple(self):
        # [FUNCTION DOC] Declares a module-level global name and a named tuple type with a string
        # and a tensor field, then scripts a function reading both fields and a missing field with
        # defaults. Checks the scripted result equals the eager one for one instance.
        global MyTuple

        class MyTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with a string field x and a tensor field y.
            x: str
            y: torch.Tensor

        def fn(x: MyTuple) -> Tuple[str, torch.Tensor, int]:
            # [FUNCTION DOC] HELPER: Returns a tuple of the x field, the y field and a missing
            # field z, each read with a default (a string, a 3 by 3 ones tensor and the int
            # seven).
            return (
                getattr(x, "x", "fdsa"),
                getattr(x, "y", torch.ones((3, 3))),
                getattr(x, "z", 7),
            )

        inp = MyTuple(x="test", y=torch.ones(3, 3) * 2)
        ref = fn(inp)
        fn_s = torch.jit.script(fn)
        res = fn_s(inp)
        self.assertEqual(res, ref)

    def test_getattr_tuple(self):
        # [FUNCTION DOC] Expects scripting a function that reads a named attribute with a default
        # from a plain tuple argument to raise a runtime error mentioning a normal tuple.
        def fn(x: Tuple[str, int]) -> int:
            # [FUNCTION DOC] HELPER: Reads attribute x of a plain tuple with default two;
            # scripting this is expected to be rejected.
            return getattr(x, "x", 2)

        with self.assertRaisesRegex(RuntimeError, "but got a normal Tuple"):
            torch.jit.script(fn)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
