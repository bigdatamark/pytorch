# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the Python-side API of JIT IR graphs: parameter strides on traced graphs,
# input permutation, insertion-point guards, node schema matching and graph inlining. Importing
# has no side effects; running the file directly raises an error pointing at test_jit.

import unittest

import numpy as np

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_MACOS, raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestPythonIr(JitTestCase):
    # [CLASS DOC] JitTestCase suite exercising Python bindings of Graph, Node and Value objects.
    # Each test builds a traced or scripted function and inspects or mutates its graph.
    def test_param_strides(self):
        # [FUNCTION DOC] Traces an identity function on a 4-d tensor and checks that the type of
        # the graph parameter reports the same strides as the example tensor.
        def trace_me(arg):
            # [FUNCTION DOC] HELPER: Identity function returning its argument.
            # It is the traced callable.
            return arg

        t = torch.zeros(1, 3, 16, 16)
        traced = torch.jit.trace(trace_me, t)
        value = list(traced.graph.param_node().outputs())[0]
        real_strides = list(t.stride())
        type_strides = value.type().strides()
        self.assertEqual(real_strides, type_strides)

    def test_permute_inputs_binding(self):
        # [FUNCTION DOC] Names each input of a scripted graph by position, permutes the inputs
        # with a random permutation, and checks the debug names now follow the permuted order.
        @torch.jit.script
        def foo(i, j, k):
            # [FUNCTION DOC] HELPER: Scripted three-argument function with an empty body; only its
            # graph inputs matter.
            pass

        g = foo.graph

        idxs = []
        for i, inp in enumerate(g.inputs()):
            inp.setDebugName(f"inp{i}")
            idxs.append(i)

        permuted_idxs = list(np.random.permutation(idxs))
        g.permuteInputs(permuted_idxs)
        for i, inp in enumerate(g.inputs()):
            self.assertEqual(f"inp{permuted_idxs[i]}", inp.debugName())

    @unittest.skipIf(IS_MACOS, "Failing on MacOS only")
    def test_python_ir_utils(self):
        # [FUNCTION DOC] Uses insert point guards (nested and sequential) to insert string
        # constants at chosen nodes, then FileCheck verifies the resulting order of the constants.
        # Also checks that a node matches its own schema but not the schema of a different node.
        # Skipped on macOS.
        @torch.jit.script
        def foo(inp):
            # [FUNCTION DOC] HELPER: Scripted function: add one, divide by two, square the result.
            x = inp + 1
            y = x / 2
            z = y * y
            return z

        add_node = foo.graph.findNode("aten::add")
        div_node = foo.graph.findNode("aten::div")

        with foo.graph.insert_point_guard(add_node):
            with foo.graph.insert_point_guard(div_node):
                foo.graph.insertConstant("goodbye")
            foo.graph.insertConstant("hello")
        with foo.graph.insert_point_guard(foo.graph.findNode("aten::mul")):
            foo.graph.insertConstant("hello")
        FileCheck().check("hello").check("goodbye").check("hello").run(foo.graph)

        self.assertTrue(add_node.matches(add_node.schema()))
        self.assertFalse(add_node.matches(div_node.schema()))

    def test_python_ir_utils_graph(self):
        # [FUNCTION DOC] Replaces constant-integer scalar multiplications in a scripted graph by
        # an inlined copy of a loop-based unrolled multiply graph, then checks no mul remains, an
        # add is present, and the function still returns the input times four. Raises if the
        # inlined output count differs from the replaced node output count.
        @torch.jit.script
        def unrolled_mul(x: torch.Tensor, y: int):
            # [FUNCTION DOC] HELPER: Scripted function adding the input to itself a given number
            # of times minus one.
            out = x
            for _ in range(y - 1):
                out = out + x
            return out

        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function multiplying its input by the constant four.
            return x * 4

        g = foo.graph
        muls = g.findAllNodes("aten::mul")
        scalar_muls = filter(
            lambda x: x.matches("aten::mul(Tensor self, Scalar other) -> Tensor"), muls
        )
        mul_constant_int = filter(
            lambda x: isinstance(list(x.inputs())[1].toIValue(), int), scalar_muls
        )
        for mul in mul_constant_int:
            with g.insert_point_guard(mul):
                outputs = g.insertGraph(unrolled_mul.graph, list(mul.inputs()))
                if len(outputs) != len(list(mul.outputs())):
                    raise AssertionError(
                        f"Expected {len(list(mul.outputs()))} outputs, got {len(outputs)}"
                    )
                for new_out, old_out in zip(outputs, g.outputs()):
                    old_out.replaceAllUsesWith(new_out)
                mul.destroy()

        FileCheck().check_not("aten::mul").check("aten::add").run(foo.graph)
        self.assertEqual(foo(torch.ones([2, 2])), torch.ones([2, 2]) * 4)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
