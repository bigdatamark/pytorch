# Owner(s): ["module: fx"]
# [MODULE DOC] Tests for FX constant folding (the const_fold experimental module). Each test
# builds a small module, splits it with the library's constant-subgraph splitter, checks the
# folded module's structure (a constant subgraph exists, folded attributes are created,
# split submodules are inlined) and checks that results equal the unfolded module's. Also
# covers tensor metadata, node meta retention, skip callbacks, impure nodes and exported
# subgraphs. Running the file directly raises an error pointing to the main FX test file.

import operator

import torch
import torch.fx
from torch.fx.experimental import const_fold
from torch.fx.passes.shape_prop import _extract_tensor_metadata, ShapeProp
from torch.testing._internal.common_utils import raise_on_run_directly, TestCase


class TestConstFold(TestCase):
    # [CLASS DOC] Test case for constant folding. Holds two helpers (attribute lookup and a
    # structural check of a folded module) and tests that fold small modules and compare
    # outputs with the originals. No instance state.
    def _get_attr(self, node):
        # [FUNCTION DOC] HELPER: Resolves a get_attr node to the real object. Walks the
        # dotted target path from the graph's owning module attribute by attribute and
        # returns the final object. Raises RuntimeError naming the missing prefix and the
        # full target if a step is absent.
        mod = node.graph.owning_module
        target = str(node.target)
        target_atoms = target.split(".")
        curr_obj = mod
        for i, atom in enumerate(target_atoms):
            if not hasattr(curr_obj, atom):
                raise RuntimeError(
                    f"Node referenced nonexistent target '{'.'.join(target_atoms[:i])}'; "
                    f" original whole target: '{target}'"
                )
            curr_obj = getattr(curr_obj, atom)
        return curr_obj

    def _verify_const_fold_mod(self, mod_folded: const_fold.FoldedGraphModule):
        # [FUNCTION DOC] HELPER: Asserts structural properties of a folded module: the
        # constant subgraph module exists, at least one get_attr node targets a folded-
        # constants attribute, and no module-call node targets the temporary split submodule
        # names (they must be inlined).
        self.assertTrue(mod_folded.const_subgraph_module is not None)

        # Check that we don't have the const or non-const fold graphs in the gm, and
        # that we do have the const folded get_attr.
        found_folded_attrs = False
        for n in mod_folded.graph.nodes:
            if n.op == "get_attr" and n.target.startswith("_FX_CONST_FOLDED_ATTRS"):
                found_folded_attrs = True
            elif n.op == "call_module":
                self.assertTrue(n.target not in {"submod_0", "submod_1"})
        self.assertTrue(found_folded_attrs)

    def test_const_fold_basic_one_attr_no_name_collision(self):
        # [FUNCTION DOC] Folds a module with two parameters where one addition of the first
        # parameter with itself is constant and the second parameter feeds the non-constant
        # part; the constant addition becomes one folded attribute and no names collide.
        # Verifies structure and that the folded and original modules give equal outputs on
        # fixed inputs.
        r"""
        Perform constant folding conversion, from original mod to split constant folding
        module with two split subgraphs, where there's a single attr to fold and
        a single output attr result to replace.

           attr1                 attr1
            | |                   | |
        x   add                   add
         \ /                       |
         sub   y                 output     (becomes attr add_1)
            \ /         ==> -------+------- (const/base subgraph split)
            mul  attr2       x   /          (input from previous subgraph
              \ /             \ /            is attr)
              add             sub   y
               |                 \ /
             output              mul  attr2
                                   \ /
                                   add
                                    |
                                  output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters; forward subtracts a constant
            # sum of the first from the input, multiplies by a second input and adds the
            # second parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two one-element parameters.
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([[-0.9]]))
                self.attr_2 = torch.nn.Parameter(torch.tensor([[17.1]]))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Doubles the first parameter, subtracts that from x,
                # multiplies by y and adds the second parameter.
                a = self.attr_1 + self.attr_1
                x = x - a
                return x * y + self.attr_2

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x, in_y = torch.tensor([[-0.45]]), torch.tensor([0.9])
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_basic_one_attr_name_collision(self):
        # [FUNCTION DOC] Same folding as the previous test, but the parameters are named so
        # that their names collide with the names the folding pass generates for folded
        # attributes. Checks the structure and equal outputs.
        r"""
        Perform constant folding conversion, from original mod to split constant folding
        module with two split subgraphs, where there's a single attr to fold and
        a single output attr result to replace. Name the attrs such that they will
        collide by name with folded attrs.

           add_1                 add_1
            | |                   | |
        x   add                   add
         \ /                       |
         sub   y                 output     (becomes attr add_1)
            \ /         ==> -------+------- (const/base subgraph split)
            mul  add_2       x   /          (input from previous subgraph
              \ /             \ /            is attr)
              add             sub   y
               |                 \ /
             output              mul  add_2
                                   \ /
                                   add
                                    |
                                  output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module like the previous one whose parameter names are
            # chosen to collide with generated folded attribute names.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two one-element parameters with colliding
                # names.
                super().__init__()
                # Note: Named as such to result in name collision.
                self.add_1__CF = torch.nn.Parameter(torch.tensor([[1.0]]))
                self.add_2__CF = torch.nn.Parameter(torch.tensor([[17.1]]))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Doubles the first parameter, subtracts it from x,
                # multiplies by y and adds the second parameter.
                a = self.add_1__CF + self.add_1__CF
                x = x - a
                return x * y + self.add_2__CF

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x, in_y = torch.tensor([[5.0]]), torch.tensor([4.0])
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_basic_placeholder_reordered(self):
        # [FUNCTION DOC] Covers the code path where a placeholder appears after a normal
        # operation in the graph. Symbolically traces a module, then moves the first
        # function-call node ahead of the second placeholder. Folding must find no constants
        # (constant subgraph module is None) and outputs of the folded and original modules
        # must be equal.
        """
        Test code path where placeholder comes after normal op node in FX
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing x times 2 plus y, with no constants.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns x times 2 plus y.
                return x * 2 + y

        mod = ConstFoldTestModule()
        mod = torch.fx.symbolic_trace(mod)
        yy = None
        for n in mod.graph.nodes:
            if n.op == "placeholder" and n.target == "y":
                yy = n
            elif yy is not None and n.op == "call_function":
                yy.prepend(n)
                break

        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)

        self.assertTrue(mod_folded.const_subgraph_module is None)
        # Now run both folded and non-folded to check results equal.
        in_x = torch.tensor([[-0.45]])
        in_y = torch.tensor([[0.45]])
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_noop(self):
        # [FUNCTION DOC] A graph with nothing foldable (input minus a parameter) is handled:
        # the constant subgraph module is None and outputs equal the original.
        r"""
        Check that a graph with no constant folding is handled correctly.

        x  attr1
         \ /
         sub
          |
        output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter subtracted from the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one one-element parameter.
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.tensor([[-0.9]]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus the parameter.
                return x - self.attr1

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)

        # Check that the folded graph module is None, since there was no folding to do.
        self.assertTrue(mod_folded.const_subgraph_module is None)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.tensor([[-0.45]])
        base_result = mod(in_x)
        fold_result = mod_folded(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_basic_two_attr_three_input(self):
        # [FUNCTION DOC] Folding with three placeholder inputs where the constant part is an
        # addition of parameter values: checks structure and equal outputs.
        # KNOWN ISSUE: the module assigns the same attribute name twice, so only one
        #     parameter exists (the second value replaces the first) although the docstring
        #     describes two attributes.
        r"""
        Perform constant folding conversion, from original mod to split constant
        folding module with two split subgraphs, where there are two attrs to
        fold into a single output, and there are three placeholder inputs.

        attr1   attr2         attr1   attr2
            \   /                 \   /
         x   add                   add
          \ /                       |
          sub     y               output     (becomes attr add_1)
             \   /     ==>   -------+------- (const/base subgraph split)
              mul  z           x   /         (input from previous subgraph
                \ /             \ /           is attr)
                div              sub  y
                 |                 \ /
               output              mul  z
                                     \ /
                                     div
                                      |
                                    output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one effective parameter; forward adds the
            # parameter to itself, subtracts from x, multiplies by y and divides by z.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Assigns the same parameter attribute twice; the
                # second value wins.
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.tensor([[-0.9]]))
                self.attr1 = torch.nn.Parameter(torch.tensor([[1.32]]))

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Doubles the parameter, subtracts from x, multiplies
                # by y, divides by z.
                a = self.attr1 + self.attr1
                sub = x - a
                mul = sub * y
                return mul / z

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x, in_y, in_z = (
            torch.tensor([[-0.45]]),
            torch.tensor([0.9]),
            torch.tensor([1.1]),
        )
        base_result = mod(in_x, in_y, in_z)
        fold_result = mod_folded(in_x, in_y, in_z)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_basic_two_attr(self):
        # [FUNCTION DOC] Two parameters added together are folded into a single attribute;
        # the remaining graph adds the input. Checks structure and equal outputs on a random
        # input.
        r"""
        Perform constant folding conversion, from original mod to split constant
        folding module with two split subgraphs, where there are two attrs to
        fold into a single output.

        attr1  attr2                attr1  attr2
            \ /                         \ /
        x   add                         add       (becomes attr add_1)
         \ /            ==>       -------+------- (const/base subgraph split)
         sub                         x   |        (input from previous subgraph is attr)
          |                           \ /
        output                        sub
                                       |
                                     output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two random matrices; forward adds the input to
            # the sum of the two parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two random 2 by 3 parameters.
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.randn(2, 3))
                self.attr2 = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the parameters and adds the input.
                y = self.attr1 + self.attr2
                return x + y

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = mod_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_multi_const_folded_attrs(self):
        # [FUNCTION DOC] Two independent constant subexpressions (a transposed-plus-original
        # parameter and a row sum of another) fold into two separate attributes, while a
        # linear layer and sigmoid remain in the non-constant part. Checks structure and
        # equal outputs.
        r"""
        Perform constant folding conversion, from original mod to split constant
        folding module with two split subgraphs, where there are two attrs to
        fold into two new attrs.

           attr1        attr2          attr1     attr2
           /    \         |           /     \      |
        permute  |       sum       permute   |    sum
            \   /        /                \ /      |
         x   add    y   /                 add      |
          \ /        \ /                   |       |
          sub        add                 output  output     (become attrs add_1 and mul_1)
             \       /        ==>   --------+-------+------ (const/base subgraph split)
              \     /                   x   |   y   |       (inputs from previous subgraph
                add                      \ /     \ /         are attrs)
                 |                       sub     add
               linear                       \   /
                 |                           add
               sigmoid                        |
                 |                          linear
               output                         |
                                            sigmoid
                                              |
                                            output
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two 4 by 4 parameters and a linear layer;
            # forward combines constant expressions with the inputs, applies the linear
            # layer and a sigmoid.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two random 4 by 4 parameters and a linear
                # layer of size 4 to 4.
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.randn(4, 4))
                self.attr2 = torch.nn.Parameter(torch.randn(4, 4))
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Adds the first parameter to its transpose,
                # subtracts from x, adds the row sum of the second parameter to y, and
                # returns the sigmoid of the linear layer on their sum.
                a = self.attr1 + self.attr1.permute(1, 0)
                x = x - a
                amax = torch.sum(self.attr2, dim=1)
                y = y + amax
                return torch.sigmoid(self.lin(x + y))

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x, in_y = torch.randn(4, 4), torch.randn(4)
        fold_result = mod_folded(in_x, in_y)
        base_result = mod(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_submod_hierarchy(self):
        # [FUNCTION DOC] Folding where one constant comes from a submodule deeper in the
        # module hierarchy that is called during tracing. Checks structure and equal
        # outputs.
        r"""
        Perform constant folding conversion, from original mod to split constant folding
        module where one of the folded attrs comes from a submod deeper in the hierarchy
        of the base module.
        """

        class TracedThroughModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module with one parameter; forward takes no inputs
            # and returns the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.internal_attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the stored parameter.
                return self.internal_attr

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module with a parameter and an inner module whose
            # call result is constant.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the inner module and one random 2 by 3
                # parameter.
                super().__init__()
                self.my_mod = TracedThroughModule()
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter plus the inner module output
                # plus x.
                return self.attr + self.my_mod() + x

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = mod_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_retain_node_meta(self):
        # [FUNCTION DOC] Verifies that node meta is kept through folding. Traces a module,
        # stores an index in the meta of every node except the output, folds, then checks
        # surviving nodes still carry their original indices: the placeholder zero, the
        # folded get_attr two (replacing the add) and the subtraction three. Also compares
        # outputs with the original.
        r"""
        Perform constant folding conversion, and validate that node meta is retained.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one random parameter, a constant doubled sum
            # and a subtraction from x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Subtracts the doubled parameter from x.
                a = self.attr + self.attr
                return x - a

        mod = ConstFoldTestModule()
        gm = torch.fx.symbolic_trace(mod)

        # Add a count for each node to check after we const fold.
        for idx, node in enumerate(gm.graph.nodes):
            if node.op != "output":
                node.meta["meta_idx"] = idx

        # Pre-folding:
        # idx 0: placeholder
        # idx 1: get_attr (will no longer be used, hence removed)
        # idx 2: add (will be folded into a get_attr)
        # idx 3: sub

        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        self._verify_const_fold_mod(gm_folded)

        # Post-folding:
        # idx 0: placeholder
        # idx 2: get_attr (replaced original add; original get_attr was removed)
        # idx 3: sub

        # Check the expected indices are still here.
        for node in gm_folded.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.meta["meta_idx"], 0)
            elif node.op == "get_attr":
                self.assertEqual(node.meta["meta_idx"], 2)
            elif node.op == "call_function" and node.target == operator.sub:
                self.assertEqual(node.meta["meta_idx"], 3)
            else:
                self.assertEqual(node.op, "output")

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_has_inlined_call_module_node(self):
        # [FUNCTION DOC] Folds a module containing a call to a nested submodule's relu (an
        # identity module with a relu child) and checks outputs equal the original.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and an identity module that has a
            # relu child attached.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a random parameter, an identity
                # submodule, and attaches a relu to it.
                super().__init__()
                self.attr = torch.nn.Parameter(torch.randn(2, 3))
                self.mod = torch.nn.Identity()
                self.mod.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the relu child to x minus the doubled
                # parameter.
                a = self.attr + self.attr
                return self.mod.relu(x - a)

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_module_attr(self):
        # [FUNCTION DOC] Folds an expression that reads an attribute of a nested module
        # together with a top-level parameter, where that nested attribute is also used non-
        # constantly. Outputs must equal the original.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and an identity submodule that
            # holds another parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a parameter, an identity submodule, and a
                # parameter on that submodule.
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))
                self.mod = torch.nn.Identity()
                self.mod.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the two parameters (constant), adds that to x,
                # then adds the nested parameter.
                a = self.const + self.mod.attr
                x = x + a
                return x + self.mod.attr

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_const_fold_unused_placeholder(self):
        # [FUNCTION DOC] Folding a module whose forward ignores two of its three inputs; the
        # folded module must still accept all three arguments and give the same output.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter whose forward ignores x and z.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns y plus the doubled parameter.
                a = self.const + self.const
                return y + a

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x, in_x, in_x)
        base_result = mod(in_x, in_x, in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_dict_output(self):
        # [FUNCTION DOC] Folding a module that returns a dictionary; compares the entry for
        # the result key with the original.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a dictionary with one tensor entry.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a mapping with one key holding x plus the
                # doubled parameter.
                a = self.const + self.const
                return {"result": x + a}

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result["result"], base_result["result"]))

    def test_two_outputs(self):
        # [FUNCTION DOC] Folding a module with a tuple output of two tensors where the first
        # output is just the input; compares both entries with the original.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a pair: the input and the input plus a
            # constant.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x and x plus the doubled parameter.
                a = self.const + self.const
                return x, x + a

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))

    def test_three_outputs(self):
        # [FUNCTION DOC] Folding a module with a three element tuple output where two
        # elements are the same value; compares all three with the original.
        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a triple: the input and two copies of the
            # input plus a constant.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.const = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x, and x plus the doubled parameter twice.
                a = self.const + self.const
                return x, x + a, x + a

        mod = ConstFoldTestModule()
        gm_folded = const_fold.split_const_subgraphs(mod)

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))
        self.assertTrue(torch.equal(fold_result[2], base_result[2]))

    def test_check_inline_non_const(self):
        # [FUNCTION DOC] After folding a symbolically traced module, no module-call nodes
        # may remain (the non-constant submodule has been inlined or extracted). Checks
        # structure and equal outputs.
        r"""
        Perform constant folding conversion and check that the non-const module is inlined
        correctly.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a constant doubled parameter and non-constant
            # arithmetic on x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus the constant times x, divided by 2.
                a = self.attr + self.attr
                return (x - a * x) / 2

        mod = ConstFoldTestModule()
        gm = torch.fx.symbolic_trace(mod)

        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        self._verify_const_fold_mod(gm_folded)

        # Check there are no call modules, because they've been inlined or extracted for
        # const folding.
        for node in gm_folded.graph.nodes:
            self.assertNotEqual(node.op, "call_module")

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_check_inline_non_const_mult_return(self):
        # [FUNCTION DOC] Like the previous inlining check but the module returns two values.
        r"""
        Perform constant folding conversion and check that the non-const module is inlined
        correctly.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a pair computed from x and a constant.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers one random 2 by 3 parameter.
                super().__init__()
                self.attr = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x minus the doubled parameter and x divided
                # by 2.
                a = self.attr + self.attr
                return x - a, x / 2

        mod = ConstFoldTestModule()
        gm = torch.fx.symbolic_trace(mod)

        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(gm)
        self._verify_const_fold_mod(gm_folded)

        # Check there are no call modules, because they've been inlined or extracted for
        # const folding.
        for node in gm_folded.graph.nodes:
            self.assertNotEqual(node.op, "call_module")

        # Now run both folded and non-folded to check results equal.
        in_x = torch.randn(2, 3)
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result[0], base_result[0]))
        self.assertTrue(torch.equal(fold_result[1], base_result[1]))

    def test_check_skip_folding_quant_dequant_pattern(self):
        # [FUNCTION DOC] Demonstrates the node skip callback: a function that returns True
        # for a quantize node whose user is a dequantize keeps the quantize-dequantize
        # weight pattern unfolded, so no constant subgraph module is produced. Outputs must
        # equal the original.
        r"""
        Set up skip_folding_quant_dequant function to skip quant/dequant pattern.
        This example shows how to use skip_folding_node_fn.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that quantizes its weight, dequantizes it, applies
            # a linear function with a bias and then a relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a random 4 by 4 weight, a random bias and
                # a relu module.
                super().__init__()
                self.weight = torch.nn.Parameter(torch.randn(4, 4))
                self.bias = torch.nn.Parameter(torch.randn(4))
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Quantizes the weight per tensor, dequantizes it,
                # applies linear with the bias to x and returns the relu of the result.
                quant_weight = torch.quantize_per_tensor(
                    self.weight, 0.5, 3, torch.quint8
                )
                dequant_weight = torch.dequantize(quant_weight)
                output = torch.nn.functional.linear(x, dequant_weight, self.bias)
                return self.relu(output)

        mod = ConstFoldTestModule()
        in_x = torch.randn(2, 4)
        gm = torch.fx.symbolic_trace(mod)

        def skip_folding_quant_dequant(node: torch.fx.Node):
            # [FUNCTION DOC] HELPER: Skip callback: returns True only for a quantize-per-
            # tensor node that has a dequantize user; False for anything else.
            if node.target != torch.quantize_per_tensor:
                return False
            # If quantize_per_node -> dequantize, then skip folding.
            for user in node.users:
                if user.target == torch.dequantize:
                    return True
            return False

        gm_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            gm, skip_folding_node_fn=skip_folding_quant_dequant
        )

        # Check that the folded graph module is None, since there was no folding to do.
        self.assertTrue(gm_folded.const_subgraph_module is None)

        # Now run both folded and non-folded to check results equal.
        fold_result = gm_folded(in_x)
        base_result = mod(in_x)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_fold_module(self):
        # [FUNCTION DOC] Folding where the constant part is a linear module call applied to
        # a parameter. Checks structure and equal outputs on a random input.
        r"""
        Perform constant folding with a call_module node.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds x to a linear layer applied to a
            # parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers a random 4 by 4 parameter and a linear
                # layer of size 4 to 4.
                super().__init__()
                self.lin_input = torch.nn.Parameter(torch.randn(4, 4))
                self.lin = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the linear layer applied to the parameter
                # plus x.
                return self.lin(self.lin_input) + x

        mod = ConstFoldTestModule()
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(mod)
        self._verify_const_fold_mod(mod_folded)

        # Now run both folded and non-folded to check results equal.
        inp = torch.randn(4, 4)
        self.assertTrue(torch.equal(mod_folded(inp), mod(inp)))

    def test_const_fold_tensor_meta(self):
        # [FUNCTION DOC] Runs the tensor metadata check twice, once with parameters
        # requiring gradients and once without.
        self._test_const_fold_tensor_meta(True)
        self._test_const_fold_tensor_meta(False)

    def _test_const_fold_tensor_meta(self, requires_grad):
        # [FUNCTION DOC] HELPER: Traces a module with two parameters whose requires_grad
        # flag is given, propagates shapes with sample inputs so nodes carry tensor
        # metadata, folds with the folded attributes on cpu, runs the folding, and asserts
        # each get_attr node's stored tensor_meta equals metadata extracted from the actual
        # attribute. Finally compares outputs with the original.
        """
        Verify tensor_meta is handled correctly.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters whose gradient flag is taken
            # from the enclosing test argument.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two one-element parameters with the given
                # requires_grad flag.
                super().__init__()
                self.attr_1 = torch.nn.Parameter(torch.tensor([[-0.9]]), requires_grad)
                self.attr_2 = torch.nn.Parameter(torch.tensor([[17.1]]), requires_grad)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Subtracts the doubled first parameter from x,
                # multiplies by y and adds the second parameter.
                a = self.attr_1 + self.attr_1
                x = x - a
                return x * y + self.attr_2

        mod = ConstFoldTestModule()
        gm = torch.fx.symbolic_trace(mod)
        in_x, in_y = torch.tensor([[-0.45]]), torch.tensor([0.9])
        ShapeProp(gm).propagate(in_x, in_y)
        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            gm, device_for_folded_attrs="cpu"
        )
        self._verify_const_fold_mod(mod_folded)

        mod_folded.run_folding()

        for n in mod_folded.graph.nodes:
            if n.op == "get_attr":
                attr = self._get_attr(n)
                self.assertEqual(_extract_tensor_metadata(attr), n.meta["tensor_meta"])

        # Now run both folded and non-folded to check results equal.
        base_result = mod(in_x, in_y)
        fold_result = mod_folded(in_x, in_y)
        self.assertTrue(torch.equal(fold_result, base_result))

    def test_fold_pure_subgraph(self):
        # [FUNCTION DOC] Exports a module that builds a constant tensor, embeds the exported
        # module as a call_module subgraph in a parent graph (fetching all outputs via a
        # slice), and folds. The pure subgraph must be folded: structure is verified.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs returning a constant filled tensor
            # plus 1.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a 5 by 10 tensor filled with 2.0 plus 1.
                return torch.full((5, 10), 2.0) + 1

        # Create a parent graph with this module as a subgraph and output
        ep = torch.export.export(SubModule(), ())
        parent_graph = torch.fx.Graph()
        call_mod = parent_graph.call_module("sub", args=())
        get_item = parent_graph.call_function(
            operator.getitem, args=(call_mod, slice(None))
        )
        parent_graph.output((get_item,))
        parent = torch.fx.GraphModule({"sub": ep.module()}, parent_graph)

        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            parent, device_for_folded_attrs="cpu"
        )
        self._verify_const_fold_mod(mod_folded)

    def test_do_not_fold_impure_subgraph(self):
        # [FUNCTION DOC] Same construction as the pure subgraph test but the submodule uses
        # a random operation (impure); folding must skip it, leaving no constant subgraph
        # module.
        """
        Skip folding any subgraph containing impure ops.
        """

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs returning a random tensor plus 1.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a 5 by 10 normal random tensor plus 1.
                return torch.randn(5, 10) + 1

        # Create a parent graph with this module as a subgraph and output
        ep = torch.export.export(SubModule(), ())
        parent_graph = torch.fx.Graph()
        call_mod = parent_graph.call_module("sub", args=())
        get_item = parent_graph.call_function(
            operator.getitem, args=(call_mod, slice(None))
        )
        parent_graph.output((get_item,))
        parent = torch.fx.GraphModule({"sub": ep.module()}, parent_graph)

        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            parent, device_for_folded_attrs="cpu"
        )
        self.assertIsNone(mod_folded.const_subgraph_module)

    def test_skip_folding_node_fn_in_subgraph(self):
        # [FUNCTION DOC] A call_module node is folded atomically, so a skip callback
        # matching a node inside its subgraph must skip the whole call. Checks that matching
        # the full op leaves nothing folded (constant subgraph module None), and as a
        # control that a callback matching nothing still lets the pure subgraph be folded.
        """
        A call_module node can only be folded as a whole, so if its subgraph
        contains a node that skip_folding_node_fn says to skip, the whole
        call_module node should be skipped.
        """

        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with no inputs returning a constant filled tensor
            # plus 1.
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a 5 by 10 tensor filled with 2.0 plus 1.
                return torch.full((5, 10), 2.0) + 1

        def _make_parent() -> torch.fx.GraphModule:
            # [FUNCTION DOC] HELPER: Exports the submodule and returns a new parent graph
            # module that calls it as a submodule and returns the sliced output.
            ep = torch.export.export(SubModule(), ())
            parent_graph = torch.fx.Graph()
            call_mod = parent_graph.call_module("sub", args=())
            get_item = parent_graph.call_function(
                operator.getitem, args=(call_mod, slice(None))
            )
            parent_graph.output((get_item,))
            return torch.fx.GraphModule({"sub": ep.module()}, parent_graph)

        # skip_folding_node_fn matches a node *inside* the submodule, so the
        # whole call_module node must be skipped and nothing gets folded.
        def skip_full(node: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: Skip callback returning True for the aten full node.
            return node.target == torch.ops.aten.full.default

        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            _make_parent(),
            skip_folding_node_fn=skip_full,
            device_for_folded_attrs="cpu",
        )
        self.assertIsNone(mod_folded.const_subgraph_module)

        # Control: when skip_folding_node_fn matches nothing in the submodule,
        # the same pure subgraph is still folded as usual.
        def skip_nothing(node: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: Skip callback that never skips; returns False.
            return False

        mod_folded = const_fold.split_const_subgraphs(
            _make_parent(),
            skip_folding_node_fn=skip_nothing,
            device_for_folded_attrs="cpu",
        )
        self._verify_const_fold_mod(mod_folded)

    def test_skip_folding_dynamic_node_in_subgraph(self):
        # [FUNCTION DOC] Regression test for a dynamic (symbolic size) node inside a
        # call_module subgraph. Without a skip callback, folding then running the folding
        # fails with an exception, since the symbol is unbound at compile time. With a
        # callback skipping nodes with symbolic integer arguments, no constant subgraph is
        # created and running folding is a no-op.
        """
        A dynamic (symbolic-shape) node inside a call_module subgraph. A call_module
        folds atomically, so folding it makes const folding *execute* the subgraph
        at compile time, which crashes with "NameError: name 's..' is not defined"
        because the shape symbol is only bound at runtime.
        """
        import torch.utils._pytree as pytree
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def _make_parent() -> torch.fx.GraphModule:
            # [FUNCTION DOC] HELPER: Builds a parent graph module that calls a graph
            # submodule whose single node creates a full tensor with an unbacked symbolic
            # size, and returns its first output.
            # Submodule whose forward materializes a tensor of symbolic size
            sub_graph = torch.fx.Graph()
            sym = ShapeEnv().create_unbacked_symint()
            full = sub_graph.call_function(
                torch.ops.aten.full.default,
                args=([sym, 1], 0),
                kwargs={"dtype": torch.bfloat16},
            )
            sub_graph.output((full,))
            sub = torch.fx.GraphModule(torch.nn.Module(), sub_graph)

            parent_graph = torch.fx.Graph()
            call_mod = parent_graph.call_module("sub", args=())
            get_item = parent_graph.call_function(operator.getitem, args=(call_mod, 0))
            parent_graph.output((get_item,))
            return torch.fx.GraphModule({"sub": sub}, parent_graph)

        # Skips nodes carrying a symbolic value. they can't be evaluated at
        # compile time, so folding them would fail.
        def skip_dynamic(node: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: Skip callback returning True when any leaf in the
            # node's args and kwargs is a symbolic integer.
            return any(
                isinstance(a, torch.SymInt)
                for a in pytree.tree_leaves((node.args, node.kwargs))
            )

        # Folding this subgraph without skipping fails, because the symbolic size
        # is unbound at compile time
        with self.assertRaises(Exception):
            unguarded = const_fold.split_const_subgraphs(
                _make_parent(), device_for_folded_attrs="cpu"
            )
            unguarded.run_folding()

        # With skip_dynamic, the dynamic node inside the submodule is skipped, so
        # the call_module is not folded and running folding is a safe no-op.
        mod_folded = const_fold.split_const_subgraphs(
            _make_parent(),
            skip_folding_node_fn=skip_dynamic,
            device_for_folded_attrs="cpu",
        )
        self.assertIsNone(mod_folded.const_subgraph_module)
        mod_folded.run_folding()

    def test_const_fold_partial_graph(self):
        # [FUNCTION DOC] Exports a module with an empty-permuted tensor and a Bernoulli
        # mask, then folds the exported module on cpu. The partially folded graph's non-
        # constant subgraph must be inlined back; verified via the shared structure check.
        """
        If a model graph is partially const folded,
        the non-const subgraph should be inlined back and erased.
        """

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a dropout-like probability; forward builds
            # a random mask and rescales x.
            def __init__(self, p):
                # [FUNCTION DOC] HELPER: Stores the probability p as an attribute.
                super().__init__()
                self.p = p

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Builds an empty permuted tensor shaped like x,
                # draws a Bernoulli mask with keep probability one minus p, and returns x
                # times mask divided by one minus p.
                probs = torch.empty_permuted(x.shape, [0, 1])
                mask = torch.bernoulli(probs, 1 - self.p)
                return x * mask / (1 - self.p)

        ep = torch.export.export(TestModule(0.4), (torch.randn(5, 10),))

        mod_folded: const_fold.FoldedGraphModule = const_fold.split_const_subgraphs(
            ep.module(), device_for_folded_attrs="cpu"
        )
        self._verify_const_fold_mod(mod_folded)

    def test_const_fold_is_impure_node_preserves_dead_node(self):
        # [FUNCTION DOC] Checks that the is_impure_node callback is passed to dead-code
        # elimination during folding. By default a user-less relu is eliminated; with a
        # callback marking it impure, it survives (exactly one user-less relu remains).
        r"""
        Const folding runs dead-code elimination, which by default drops a node
        with no users when ``Node.is_impure()`` is False. A caller can pass
        ``is_impure_node`` to preserve such a node -- e.g. a destination-passing
        op whose only effect is writing a buffer through an ``out=`` kwarg its
        schema does not declare as mutated, so the default check treats it as
        pure dead code.
        """

        class ConstFoldTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two parameters, a foldable sum, and an unused
            # relu of the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Registers two random 2 by 3 parameters.
                super().__init__()
                self.attr1 = torch.nn.Parameter(torch.randn(2, 3))
                self.attr2 = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes the parameter sum, an unused relu of x,
                # and returns x plus the sum.
                folded = self.attr1 + self.attr2  # const-foldable subgraph
                _unused = torch.relu(x)  # no users -> dead-code candidate
                return x + folded

        def _num_userless_relus(gm: torch.fx.GraphModule) -> int:
            # [FUNCTION DOC] HELPER: Counts call_function nodes whose target name mentions
            # relu and which have no users.
            return sum(
                1
                for n in gm.graph.nodes
                if n.op == "call_function"
                and "relu" in str(n.target)
                and len(n.users) == 0
            )

        def _is_userless_relu(n: torch.fx.Node) -> bool:
            # [FUNCTION DOC] HELPER: Impurity callback: True for a call_function node whose
            # target name mentions relu and which has no users.
            return (
                n.op == "call_function"
                and "relu" in str(n.target)
                and len(n.users) == 0
            )

        # Default: the user-less relu is dead-code-eliminated.
        mod_default = const_fold.split_const_subgraphs(ConstFoldTestModule())
        self.assertEqual(_num_userless_relus(mod_default), 0)

        # With is_impure_node marking the user-less relu impure, it survives.
        mod_kept = const_fold.split_const_subgraphs(
            ConstFoldTestModule(), is_impure_node=_is_userless_relu
        )
        self.assertEqual(_num_userless_relus(mod_kept), 1)


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
