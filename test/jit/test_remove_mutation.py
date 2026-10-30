# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests for the remove-mutation graph pass. Each test scripts a small
# function or module, runs the pass (through run_pass or the raw C++ binding), and uses
# FileCheck to confirm that in-place tensor ops and list mutations (append, insert, item
# assignment) are either rewritten to functional form or left alone when unsafe, and
# that results still match eager. Importing appends the test directory to sys.path;
# direct execution redirects to test/test_jit.py.

import os
import sys
from typing import List

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import freeze_rng_state, JitTestCase


class TestRemoveMutation(JitTestCase):
    # [CLASS DOC] Test case for the remove-mutation pass, covering tensor in-place ops, ops with
    # special functional mappings, conditional outputs, and list mutations.
    def test_aten_inplace(self):
        # [FUNCTION DOC] Checks which tensor in-place ops the pass can convert. Five nested
        # scripted functions are examined: a mutated view (not a new alias, so the in-place add
        # stays), item assignment (stays as copy), a mutation that cannot be moved before an
        # earlier use (stays), a chain of adds (all removed), and an intermediary use (exactly one
        # in-place add remains). Each compares scripted output with eager.
        def test_not_new_alias(x):
            # [FUNCTION DOC] HELPER: Takes the first row of the input, adds two in place on that
            # view, and returns it.
            y = x[0]
            y.add_(2)
            return y

        fn = torch.jit.script(test_not_new_alias)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(torch.ones([2, 2])), test_not_new_alias(torch.ones([2, 2])))

        def test_no_lowering():
            # [FUNCTION DOC] HELPER: Creates a tensor, assigns an element through indexing, and
            # returns it. No functional equivalent exists for element assignment.
            x = torch.tensor([2, 2])
            x[0] = 3
            return x

        # there is no functional equivalent of x[0] = ...
        fn = torch.jit.script(test_no_lowering)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::copy_").run(graph)
        self.assertEqual(fn(), test_no_lowering())

        def test_move_before_not_valid():
            # [FUNCTION DOC] HELPER: Computes z from y, then mutates y in place, returning both.
            # The mutation cannot be hoisted above the read of y.
            y = torch.tensor([2, 2])
            z = y + 2
            y.add_(2)
            return y, z

        fn = torch.jit.script(test_move_before_not_valid)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check("aten::add_").run(graph)
        self.assertEqual(fn(), test_move_before_not_valid())

        def test_successful():
            # [FUNCTION DOC] HELPER: Applies two in-place adds to a fresh tensor and then reads
            # it; every in-place add is removable.
            x = torch.tensor([2, 2])
            x.add_(1)
            x.add_(3)
            y = x + 4
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::add_").run(graph)
        self.assertEqual(test_successful(), fn())

        def test_intermediary_use():
            # [FUNCTION DOC] HELPER: Mutates x in place, reads x into y, then mutates x again; the
            # second mutation cannot be removed because of the read in between.
            x = torch.tensor([2, 2])
            x.add_(1)
            y = x + 4
            x.add_(3)
            return x, y

        fn = torch.jit.script(test_intermediary_use)
        graph = fn.graph
        FileCheck().check_count("aten::add_", 2).run(graph)
        self.run_pass("remove_mutation", graph)
        # Unable to remove the second add_ because of the y = x + 4 use
        # In the future we could duplicating the value of x as a temporary and replacing
        # its intermediary use (so long as aliasing is safe)
        FileCheck().check_count("aten::add_", 1).run(graph)
        self.assertEqual(test_intermediary_use(), fn())

    def test_if_output(self):
        # [FUNCTION DOC] Scripts a function whose tensor is produced by either branch of a
        # conditional and then mutated in place, checks the in-place add disappears after the
        # pass, and compares to eager.
        def foo(x, cond: bool):
            # [FUNCTION DOC] HELPER: Builds y from x plus a constant on either branch of the
            # condition, adds four in place, and returns y.
            if cond:
                y = x + 5
            else:
                y = x + 2
            y.add_(4)
            return y

        out_eager = foo(torch.tensor(5), True)
        foo_script = torch.jit.script(foo)
        FileCheck().check("aten::add_").run(foo_script.graph)
        self.run_pass("remove_mutation", foo_script.graph)
        FileCheck().check_not("aten::add_").run(foo_script.graph)

        self.assertEqual(out_eager, foo_script(torch.tensor(5), True))

    def test_if_output_fail(self):
        # [FUNCTION DOC] Two scripted functions where the mutated value may alias something else
        # (a tensor appended to a list in one branch; an input in the other branch). After
        # inlining and the pass, asserts the in-place add is still present in each graph.
        @torch.jit.script
        def foo(cond: bool):
            # [FUNCTION DOC] HELPER: Scripted. Creates a tensor in each branch (appending it to a
            # list in one), mutates the result in place, and returns it with the list.
            li = []
            if cond:
                x = torch.tensor(1)
                li.append(x)
            else:
                x = torch.tensor(2)
            y = x.add_(2)
            return y, li

        self.run_pass("inline", foo.graph)
        self.run_pass("remove_mutation", foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

        @torch.jit.script
        def foo(cond: bool, y):
            # [FUNCTION DOC] HELPER: Scripted. Selects the input tensor or a new tensor by
            # condition, mutates the selection in place, and returns it. The mutation may alias
            # the input so it is kept.
            if cond:
                x = y
            else:
                x = torch.tensor(2)
            z = x.add_(2)
            return z

        self.run_pass("inline", foo.graph)
        self.run_pass("remove_mutation", foo.graph)
        FileCheck().check("aten::add_").run(foo.graph)

    def test_special_mapped_op(self):
        # [FUNCTION DOC] Checks ops mapped specially by the pass: zero_ and fill_ with a scalar
        # are removed; fill_ with a tensor value is also removed (full_like is not available for
        # tensor fills, so only absence of fill_ is asserted); and normal_ on a random tensor is
        # removed with identical results under a frozen RNG state.
        def test_successful():
            # [FUNCTION DOC] HELPER: Zeroes one tensor and fills another with a scalar, in place,
            # and returns both.
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.zero_()
            y.fill_(3)
            return x, y

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::zero_").check_not("aten::fill_").run(graph)
        self.assertEqual(test_successful(), fn())

        # full_like is not implemented for a tensor fill value

        def test_successful():
            # [FUNCTION DOC] HELPER: Redefinition of the helper with the same name. Fills a tensor
            # with another tensor value in place and returns the sum of the tensor with itself.
            x = torch.tensor([2, 2])
            y = torch.tensor([2, 4])
            x.fill_(y)
            return x + x

        fn = torch.jit.script(test_successful)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("aten::fill_").run(graph)

        def normal():
            # [FUNCTION DOC] HELPER: Returns a random tensor of float dtype, explicitly typed,
            # filled in place by normal_.
            # NOTE: For some unknown reason, the
            # `torch._C._jit_pass_remove_mutation` call within `self.run_pass`
            # replaces `torch.randn(..., dtype=None).normal_()` with an
            # `aten::normal` call with dtype double, even if the default dtype
            # is float. So we must explicitly set the dtype here
            return torch.rand(2, 1, 3, 4, dtype=torch.float).normal_()

        fn = torch.jit.script(normal)
        graph = fn.graph
        self.run_pass("remove_mutation", graph)
        FileCheck().check_not("normal_").run(graph)
        with freeze_rng_state():
            out_eager = normal()
        with freeze_rng_state():
            out_script = fn()
        self.assertEqual(out_eager, out_script)

    def test_lists_append(self):
        # [FUNCTION DOC] Checks list append removal: a list comprehension over a range is
        # unrolled, mutation-removed and constant-propagated down to a single constant and return;
        # a list used by length before an append keeps append out of the graph after the pass (the
        # comment notes that is possible but not yet implemented).
        def successful_remove():
            # [FUNCTION DOC] HELPER: Builds a list of the
            # integers zero to four with a comprehension.
            return [i for i in range(5)]  # noqa: C416

        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        self.run_pass("loop_unrolling", graph)
        self.run_pass("remove_mutation", graph)
        self.run_pass("constant_propagation", graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        self.assertEqual(successful_remove(), successful_remove())

        def intermediary_use():
            # [FUNCTION DOC] HELPER: Makes a two-element list, reads its length into an unused
            # variable, appends an item, and returns the list.
            a = [1, 2]
            b = len(a)  # noqa: F841
            a.append(3)
            return a

        fn = torch.jit.script(intermediary_use)
        graph = fn.graph
        FileCheck().check("append").run(graph)
        self.run_pass("remove_mutation", graph)
        # it is possible to remove the append here but don't currently have the logic for it
        FileCheck().check_not("append").run(graph)
        self.assertEqual(intermediary_use(), fn())

    def test_lists_insert(self):
        # [FUNCTION DOC] Scripts a function that inserts items at several positions (including
        # negative and out-of-range indices) into an empty list, runs remove-mutation and constant
        # propagation through the C++ bindings, and checks the graph collapses to a constant and
        # return and matches eager.
        def successful_remove():
            # [FUNCTION DOC] HELPER: Starts from an empty list of ints and performs five inserts
            # at various indices, returning the list.
            a: List[int] = []
            a.insert(0, 1)
            a.insert(0, 2)
            a.insert(-10, 3)
            a.insert(-9, 4)
            a.insert(10, 5)
            return a

        fn = torch.jit.script(successful_remove)
        graph = fn.graph
        torch._C._jit_pass_remove_mutation(graph)
        torch._C._jit_pass_constant_propagation(graph)
        FileCheck().check("graph").check_next("Constant").check_next("return").run(
            graph
        )
        self.assertEqual(successful_remove(), fn())

    def test_list_indexing_removal(self):
        # [FUNCTION DOC] Checks removal of list item assignment: an out-of-bounds constant index
        # and an index known only at run time keep set_item in the graph; in-bounds constant
        # (positive or negative) indices are removed. Note the unknown-index check re-runs the
        # pass on the first function's graph rather than the second.
        @torch.jit.script
        def out_of_bounds():
            # [FUNCTION DOC] HELPER: Scripted. Assigns to index four of a two-element list and
            # returns the list.
            x = [1, 2]
            x[4] = 3
            return x

        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check("set_item").run(out_of_bounds.graph)

        @torch.jit.script
        def unknown(y: int):
            # [FUNCTION DOC] HELPER: Scripted. Assigns to a list element at an index given as a
            # function argument.
            x = [1, 2]
            x[y] = 3
            return x

        torch._C._jit_pass_remove_mutation(out_of_bounds.graph)
        FileCheck().check("set_item").run(out_of_bounds.graph)

        def successful():
            # [FUNCTION DOC] HELPER: Assigns to the first and last index of a three-element list
            # and returns it.
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x

        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            # [FUNCTION DOC] HELPER: Duplicate of the previous helper with the same name and body.
            x = [1, 2, 3]
            x[0] = 4
            x[-1] = 0
            return x

        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

        def successful():
            # [FUNCTION DOC] HELPER: Assigns to the last index of
            # a one-element list and returns it.
            x = [1]
            x[-1] = 3
            return x

        scripted_fn = torch.jit.script(successful)
        torch._C._jit_pass_remove_mutation(scripted_fn.graph)
        FileCheck().check_not("set_item").run(scripted_fn.graph)
        self.checkScript(successful, ())

    def test_common_pytorch_list_ops(self):
        # [FUNCTION DOC] For each of several list-taking ops (cat, stack, vstack, hstack, dstack),
        # scripts a module that mutates a tensor in place, builds a list holding it twice, and
        # applies the op. Asserts the in-place add is removed and results match eager, and that
        # the op output does not alias its inputs (changing the inputs later leaves earlier result
        # sums unchanged). Also scripts a function where the tensor has multiple uses and checks
        # the in-place add remains.
        # KNOWN ISSUE: the last run_pass call uses the module forward graph from the loop rather
        # than the multi-use function graph.
        for op in ["cat", "stack", "vstack", "hstack", "dstack"]:

            class OpMod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module that stores a torch list op and applies it to a
                # list of one mutated tensor repeated twice, adding a constant to the result.
                def __init__(self, op):
                    # [FUNCTION DOC] HELPER: Stores the op on the module.
                    # KNOWN ISSUE: the argument is ignored and the enclosing loop variable torch_op
                    # is stored instead (same object at call time).
                    super().__init__()
                    self.op = torch_op

                def forward(self):
                    # [FUNCTION DOC] HELPER: Makes a tensor, adds three in place, builds a list
                    # with the tensor twice, applies the stored op, and adds three.
                    x = torch.tensor([1, 2, 3, 4])
                    x.add_(3)
                    y = [x, x]
                    return self.op(y) + 3

            torch_op = getattr(torch, op)
            mod = OpMod(torch_op)
            mod_script = torch.jit.script(mod)
            self.run_pass("remove_mutation", mod_script.forward.graph)
            FileCheck().check_not("aten::add_").run(mod_script.forward.graph)
            self.assertEqual(mod(), mod_script())

            # test that the output doesn't alias the input
            for inputs in [torch.rand(2, 2)], [torch.rand(2, 2) for _ in range(2)]:
                result = torch_op(inputs)
                sums = [ten.sum() for ten in result]

                for inp in inputs:
                    inp.fill_(10)

                self.assertEqual(sums, [ten.sum() for ten in result])

        @torch.jit.script
        def test_multiple_uses():
            # [FUNCTION DOC] HELPER: Scripted. Mutates a fresh tensor in place, puts it twice in a
            # list, and returns the concatenation together with the list.
            x = torch.tensor([1, 2, 3, 4])
            x.add_(3)
            y = [x, x]
            return torch.cat(y), y

        self.run_pass("remove_mutation", mod_script.forward.graph)
        FileCheck().check("aten::add_").run(test_multiple_uses.graph)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
