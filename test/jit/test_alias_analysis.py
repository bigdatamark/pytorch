# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript alias analysis tests: wildcard handling for container outputs, aliasing
# through list construction, writer detection across function and method calls when descending
# into callees, and a regression test for many same-named modules loaded from separate compilation
# units. Raises an error if run directly; run via the main jit test file.

import torch
from torch._C import parse_ir
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    TemporaryFileName,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestAliasAnalysis(JitTestCase):
    # [CLASS DOC] JIT test case grouping the alias database checks.
    def test_becomes_wildcard_annotations(self):
        # [FUNCTION DOC] Parses a textual IR graph with an add feeding a split, builds its alias
        # database and checks that the split input may alias the split output (list contains the
        # wildcard set) and also may alias a graph input, since the input joined the wildcard set.
        graph_str = """
        graph(%a.1 : Tensor, %b.1 : Tensor):
            %11 : NoneType = prim::Constant()
            %8 : int = prim::Constant[value=0]()
            %7 : int = prim::Constant[value=1]()
            %x.1 : Tensor = aten::add(%a.1, %b.1, %7)
            %y.1 : Tensor[] = aten::split(%x.1, %7, %8)
            return ()
        """
        graph = parse_ir(graph_str)
        alias_db = graph.alias_db()
        split_node = graph.findNode("aten::split")
        # split input enters wildcard set, list initialized as containing wildcard set
        self.assertTrue(
            alias_db.may_contain_alias(next(split_node.inputs()), split_node.output())
        )
        # because %x.1 enters wildcard set, it now aliases other members of wildcard set (graph inputs)
        self.assertTrue(
            alias_db.may_contain_alias(next(split_node.inputs()), next(graph.inputs()))
        )

    def test_nested_list_construct_not_wildcard(self):
        # [FUNCTION DOC] Scripts a function that returns a list holding a freshly created random
        # tensor. Checks that the tensor may alias the graph output but the graph input may not
        # alias the tensor, so list construction does not make the contents a wildcard.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function creating a 2 by 2 random tensor and
            # returning it in a one-element list; the argument is unused.
            y = torch.rand([2, 2])
            return [y]

        graph = foo.graph
        alias_db = graph.alias_db()
        ten_construct = graph.findNode("aten::rand").output()
        output = next(graph.outputs())
        self.assertTrue(alias_db.may_contain_alias(ten_construct, output))
        self.assertFalse(
            alias_db.may_contain_alias(next(graph.inputs()), ten_construct)
        )

    def test_recursive_calls(self):
        # [FUNCTION DOC] Builds an alias database that descends into callees, first for a scripted
        # caller with two calls to a function that mutates its first argument in place, then for a
        # scripted module with two calls to a method doing the same. For each call node it expects
        # the first argument to have writers and the second not, and exactly two call nodes in
        # each graph.
        @torch.jit.script
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Scripted function that mutates its first argument by adding
            # one in place and returns the sum of the two arguments.
            x.add_(1)
            return x + y

        @torch.jit.script
        def caller():
            # [FUNCTION DOC] HELPER: Scripted function calling the mutating function twice on
            # separate tensors and returning both results.
            a = torch.rand([2, 2])
            b = torch.ones([2, 2])
            out1 = foo(a, b)
            c = torch.rand([1])
            d = torch.ones([2])
            out2 = foo(d, c)
            return out1, out2

        isFrozen = False
        descend_function_calls = True
        alias_db = caller.graph.alias_db(isFrozen, descend_function_calls)
        func_calls = caller.graph.findAllNodes("prim::CallFunction")
        self.assertEqual(len(func_calls), 2)
        for node in func_calls:
            inps = list(node.inputs())
            self.assertTrue(alias_db.has_writers(inps[1]))
            self.assertFalse(alias_db.has_writers(inps[2]))

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward method calls a mutating helper method
            # twice.
            def forward(self):
                # [FUNCTION DOC] HELPER: Creates tensors, calls the mutating helper method twice
                # with different tensor pairs and returns both results.
                a = torch.rand([2, 2])
                b = torch.ones([2, 2])
                out1 = self.foo2(a, b)
                c = torch.rand([1])
                d = torch.ones([2])
                out2 = self.foo2(d, c)
                return out1, out2

            def foo2(self, x, y):
                # [FUNCTION DOC] HELPER: Mutates the first argument in place by adding one and
                # returns the sum of the two arguments.
                x.add_(1)
                return x + y

        mod = torch.jit.script(Mod())
        alias_db = mod.graph.alias_db(isFrozen, descend_function_calls)
        func_calls = mod.graph.findAllNodes("prim::CallMethod")
        self.assertEqual(len(func_calls), 2)
        for node in func_calls:
            inps = list(node.inputs())
            self.assertTrue(alias_db.has_writers(inps[1]))
            self.assertFalse(alias_db.has_writers(inps[2]))

    def test_multiple_compilation_units(self):
        # [FUNCTION DOC] Regression test for a hashing problem in alias analysis. Creates forty
        # temporary files, scripts, saves and reloads a same-named module from each (each load has
        # its own compilation unit) into a module list, then scripts a wrapper over the list and
        # runs it on a 2 by 2 tensor. Declares a module-level global name for the class.
        # Previously this took exponential time or ran out of memory due to hash collisions on the
        # module name.
        # This is a repro of an internal issue we saw.
        # Here, we have a large number (40) of modules each with the same name (MyModuleCUTest).
        # AliasDB uses some hash tables that hash on types; each of these 40 modules are not
        # identical because they have different compilation units, but they have the same name.
        # Therefore, if we hash only on the module name (which we previously did), we will have
        # hash collisions for all of these module types.
        #
        # flat_hash_map has very bad performance (exponential) for this hash collision behavior.
        # This OOMs prior to the fix.
        N = 40

        class MultiTmpFile:
            # [CLASS DOC] HELPER: Context manager that opens a given number of temporary file name
            # contexts together. Entering yields the list of file names; exiting closes all of
            # them.
            def __init__(self, N):
                # [FUNCTION DOC] HELPER: Stores the count and creates one temporary file name
                # context per count with write mode and a Python file suffix.
                self.N = N
                self.ctxs = [
                    TemporaryFileName(mode="w", suffix=".py") for _ in range(N)
                ]

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Enters every underlying context and returns the list of
                # file names.
                return [x.__enter__() for x in self.ctxs]

            def __exit__(self, exc_type, exc_value, traceback):
                # [FUNCTION DOC] HELPER: Exits every underlying context with the given exception
                # information and returns the list of their results.
                return [x.__exit__(exc_type, exc_value, traceback) for x in self.ctxs]

        class ModuleWrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that applies each module in a stored list to the input in
            # order.
            def __init__(self, module_list):
                # [FUNCTION DOC] HELPER: Stores the given list of modules as an attribute.
                super().__init__()
                self.module_list = module_list

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Passes the input through each stored module in sequence
                # and returns the final value.
                for mod in self.module_list:
                    x = mod(x)
                return x

        with MultiTmpFile(N) as fnames:
            module_list = torch.nn.ModuleList()
            global MyModuleCUTest

            class MyModuleCUTest(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with an input-adding-two forward; declared global so
                # every scripted copy shares the same qualified name.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the input plus two.
                    return x + 2

            for fname in fnames:
                mod = torch.jit.script(MyModuleCUTest())
                torch.jit.save(mod, fname)
                loaded_mod = torch.jit.load(fname)
                module_list.append(loaded_mod)

            mod = ModuleWrapper(module_list)
            mod = torch.jit.script(mod)
            mod(torch.zeros((2, 2)))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
