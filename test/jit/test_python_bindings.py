# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for Python bindings of the TorchScript compilation unit, graph, node, value
# and alias database classes: function lookup and creation, invalidation of destroyed nodes,
# iterator lifetime, graph creation errors and canonicalization. No import-time side effects;
# running the file directly raises an error pointing at test_jit.

import torch
from torch.testing import FileCheck
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class TestPythonBindings(JitTestCase):
    # [CLASS DOC] JitTestCase suite for the torch._C JIT bindings. Tests are independent; each
    # scripts or builds its own graph or compilation unit.
    def test_cu_get_functions(self):
        # [FUNCTION DOC] Scripts a function and checks that the default Python compilation unit
        # lists it by name among its functions.
        @torch.jit.script
        def test_get_python_cu_fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function returning its tensor input doubled; looked
            # up by name in the shared compilation unit.
            return 2 * x

        cu = torch.jit._state._python_cu
        self.assertTrue(
            "test_get_python_cu_fn" in (str(fn.name) for fn in cu.get_functions())
        )

    def test_cu_create_function(self):
        # [FUNCTION DOC] Copies the graph of a scripted function into a fresh compilation unit
        # under a new name, then checks the function can be found and called both by lookup and as
        # an attribute, that a missing name yields None from lookup, and that calling a missing
        # attribute raises AttributeError.
        @torch.jit.script
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function returning its tensor input doubled; its
            # graph is copied into the new unit.
            return 2 * x

        cu = torch._C.CompilationUnit()
        cu.create_function("test_fn", fn.graph)

        inp = torch.randn(5)

        self.assertEqual(inp * 2, cu.find_function("test_fn")(inp))
        self.assertEqual(cu.find_function("doesnt_exist"), None)
        self.assertEqual(inp * 2, cu.test_fn(inp))
        with self.assertRaises(AttributeError):
            cu.doesnt_exist(inp)

    def test_invalidation(self):
        # [FUNCTION DOC] Inserts a profile node in a copied graph, checks node and value can be
        # printed, runs dead code elimination, then checks that printing the removed node or its
        # output raises RuntimeError mentioning invalidation.
        @torch.jit.script
        def test_invalidation_fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function returning its tensor input doubled; its
            # graph is copied and mutated.
            return 2 * x

        gr = test_invalidation_fn.graph.copy()
        n = gr.insertNode(gr.create("prim::profile"))
        v = n.output()
        # check that they work
        str((n, v))
        torch._C._jit_pass_dce(gr)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(n)
        with self.assertRaisesRegex(RuntimeError, "invalidated"):
            str(v)

    def test_graph_iterator_keepalive(self):
        # [FUNCTION DOC] Regression check for issue 50454: iterating the nodes, inputs and outputs
        # of a temporary inlined graph must keep the graph alive so the iteration does not crash.
        @torch.jit.script
        def test_iterator_keepalive_fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function returning its tensor input doubled;
            # supplies the inlined graph.
            return 2 * x

        # the list would segfault before because inlined_graph
        # is temporary and had been deleted (see issue #50454)
        n = test_iterator_keepalive_fn.inlined_graph.nodes()
        list(n)
        i = test_iterator_keepalive_fn.inlined_graph.inputs()
        list(i)
        o = test_iterator_keepalive_fn.inlined_graph.outputs()
        list(o)

    def test_aliasdb(self):
        # [FUNCTION DOC] Builds an alias database from a copied graph and checks that its text
        # form mentions the wildcard set and its graphviz output is a digraph called alias_db.
        @torch.jit.script
        def test_aliasdb_fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Scripted function returning its tensor input doubled;
            # supplies the graph.
            return 2 * x

        gr = test_aliasdb_fn.graph.copy()
        alias_db = gr.alias_db()
        self.assertTrue("WILDCARD" in str(alias_db))
        self.assertTrue("digraph alias_db" in alias_db.to_graphviz_str())

    def test_graph_create(self):
        # [FUNCTION DOC] Checks that creating a constant node in an empty graph with a None entry
        # in the inputs list raises ValueError.
        gr = torch._C.Graph()
        with self.assertRaises(ValueError):
            gr.create("prim::Constant", [None])

    def test_add_input(self):
        # [FUNCTION DOC] Adds a named input to an empty graph and checks the returned value is
        # among the graph inputs; raises AssertionError otherwise.
        gr = torch._C.Graph()
        foo_value = gr.addInput("foo")
        if foo_value not in gr.inputs():
            raise AssertionError("foo_value not found in graph inputs")

    def test_canonicalize(self):
        # [FUNCTION DOC] Parses an IR string of placeholder ops and runs the canonicalize pass
        # three ways: with the second flag true, omitted, and false. Checks the first two print
        # identically, keep the input name p207 and renumber intermediates so value 14 vanishes,
        # and that with the flag false the input name p207 is gone.
        ir = """
graph(%p207 : Tensor,
      %1 : Tensor,
      %p407 : int):
  %11 : Tensor = aten::view_expand_placeholder(%1)
  %12 : Tensor = aten::pointwise_placeholder(%11, %p207, %p407)
  %13 : Tensor = aten::view_expand_placeholder(%12)
  %14 : Tensor = aten::pointwise_placeholder(%13)
  return (%14)
        """

        graph1 = torch._C.parse_ir(ir)
        graph1 = torch._C._jit_pass_canonicalize(graph1, True)

        graph2 = torch._C.parse_ir(ir)
        graph2 = torch._C._jit_pass_canonicalize(graph2)

        self.assertEqual(str(graph1), str(graph2))
        FileCheck().check("%p207").check_not("%14").run(graph1)

        graph3 = torch._C.parse_ir(ir)
        graph3 = torch._C._jit_pass_canonicalize(graph3, False)
        FileCheck().check_not("%p207").run(graph3)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
