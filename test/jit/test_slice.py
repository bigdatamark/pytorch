# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for the Python slice class used as an index. Each test scripts
# a small function that indexes a list, string, tensor, tuple or ModuleList with a slice object
# or with ordinary slice syntax, and compares scripted and eager results. Importing the module
# appends the test directory to the system path.

import os
import sys
from typing import List

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests that Python slice class is supported in TorchScript
class TestSlice(JitTestCase):
    # [CLASS DOC] JitTestCase suite covering slice support in TorchScript: slice construction
    # with one to three arguments (None allowed), keyword rejection, use on lists, strings,
    # tensors, tuples and ModuleList containers.
    def test_slice_kwarg(self):
        # [FUNCTION DOC] Checks that scripting a function that builds a slice with a keyword
        # argument raises a RuntimeError whose message says Slice does not accept any keyword
        # arguments.
        def slice_kwarg(x: List[int]):
            # [FUNCTION DOC] HELPER: Function under test; indexes an int list with a slice built
            # from a start positional and a stop keyword argument. Never run successfully;
            # compilation must fail.
            return x[slice(1, stop=2)]

        with self.assertRaisesRegex(
            RuntimeError, "Slice does not accept any keyword arguments"
        ):
            torch.jit.script(slice_kwarg)

    def test_slice_three_nones(self):
        # [FUNCTION DOC] Checks that a slice with all three arguments None selects the whole
        # list.
        def three_nones(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the whole input list via a slice of three None
            # values.
            return x[slice(None, None, None)]

        self.checkScript(three_nones, (range(10),))

    def test_slice_two_nones(self):
        # [FUNCTION DOC] Checks that a slice with two None arguments selects the whole list.
        def two_nones(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the whole input list via a slice of two None
            # values.
            return x[slice(None, None)]

        self.checkScript(two_nones, (range(10),))

    def test_slice_one_none(self):
        # [FUNCTION DOC] Checks that a slice with a single None argument selects the whole list.
        def one_none(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the whole input list via a slice of one None value.
            return x[slice(None)]

        self.checkScript(one_none, (range(10),))

    def test_slice_stop_only(self):
        # [FUNCTION DOC] Checks that a one-argument slice is treated as a stop index (first five
        # items).
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the first five items of the input list using a
            # stop-only slice.
            return x[slice(5)]

        self.checkScript(fn, (range(10),))

    def test_slice_stop_only_with_nones(self):
        # [FUNCTION DOC] Checks a three-argument slice with None start and step and a stop of
        # five.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the first five items using a slice with None start
            # and step.
            return x[slice(None, 5, None)]

        self.checkScript(fn, (range(10),))

    def test_slice_start_stop(self):
        # [FUNCTION DOC] Checks a two-argument slice giving start and stop on an int list.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the items from index one up to index five of the
            # input list.
            return x[slice(1, 5)]

        self.checkScript(fn, (range(10),))

    def test_slice_start_stop_with_none(self):
        # [FUNCTION DOC] Checks a three-argument slice with explicit start and stop and a None
        # step.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns items from index one up to index five, step given
            # as None.
            return x[slice(1, 5, None)]

        self.checkScript(fn, (range(10),))

    def test_slice_start_stop_step(self):
        # [FUNCTION DOC] Checks a slice with explicit start, stop and step on an int list.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns every second item from index zero up to index six.
            return x[slice(0, 6, 2)]

        self.checkScript(fn, (range(10),))

    def test_slice_string(self):
        # [FUNCTION DOC] Checks that a slice object can index a string (first three characters).
        def fn(x: str):
            # [FUNCTION DOC] HELPER: Returns the first three characters of the input string via
            # a slice.
            return x[slice(None, 3, 1)]

        self.checkScript(fn, ("foo_bar",))

    def test_slice_tensor(self):
        # [FUNCTION DOC] Checks that a slice object can index a one-dimensional tensor.
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Returns the first three elements of the input tensor via a
            # slice.
            return x[slice(None, 3, 1)]

        self.checkScript(fn, (torch.ones(10),))

    def test_slice_tensor_multidim(self):
        # [FUNCTION DOC] Checks a slice object combined with an integer index on a two-
        # dimensional tensor.
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Slices the first dimension to three rows and selects column
            # zero.
            return x[slice(None, 3, 1), 0]

        self.checkScript(fn, (torch.ones((10, 10)),))

    def test_slice_tensor_multidim_with_dots(self):
        # [FUNCTION DOC] Checks a slice object combined with an ellipsis on a two-dimensional
        # tensor.
        def fn(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Slices the first dimension to three rows and keeps the
            # remaining dims.
            return x[slice(None, 3, 1), ...]

        self.checkScript(fn, (torch.ones((10, 10)),))

    def test_slice_as_variable(self):
        # [FUNCTION DOC] Checks that a slice object stored in a local variable can later be used
        # as index.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Stores a stop-only slice in a variable and indexes the
            # input list with it.
            a = slice(1)
            return x[a]

        self.checkScript(fn, (range(10),))

    def test_slice_stop_clipped(self):
        # [FUNCTION DOC] Checks that a stop beyond the list length is clipped, returning the
        # whole list.
        def fn(x: List[int]):
            # [FUNCTION DOC] HELPER: Indexes a ten item list with a stop-only slice of 1000.
            return x[slice(1000)]

        self.checkScript(fn, (range(10),))

    def test_slice_dynamic_index(self):
        # [FUNCTION DOC] Checks that slice bounds computed at run time from local integer
        # variables work the same way as constant bounds on a tensor.
        def t(x):
            # [FUNCTION DOC] HELPER: Adds two slices of the first dimension, one with constant
            # bounds and one with bounds held in variables; both select the first row.
            slice1 = x[0:1]
            zero = 0
            one = zero + 1
            slice2 = x[zero:one]
            return slice1 + slice2

        self.checkScript(t, (torch.zeros(3, 2, 3),))

    def test_tuple_slicing(self):
        # [FUNCTION DOC] Checks slicing of tuples, including negative bounds, on a tuple chosen
        # by a branch. Verifies the scripted result is a tuple holding 2 and 3, that the graph
        # has exactly one tuple construction of length two, and that lowering all tuples leaves
        # no tuple in the graph.
        def tuple_slice(a):
            # [FUNCTION DOC] HELPER: Picks one of two four-item tuples by a condition, then
            # slices it twice and returns the two-item middle tuple.
            if bool(a):
                b = (1, 2, 3, 4)
            else:
                b = (4, 3, 2, 1)
            c = b[-4:4]
            e = c[1:-1]
            return e

        self.checkScript(tuple_slice, (torch.tensor([1]),), optimize=True)
        scripted_fn = torch.jit.script(tuple_slice)
        self.assertEqual(scripted_fn(torch.tensor(1)), (2, 3))
        tuple_graph = scripted_fn.graph
        slices = tuple_graph.findAllNodes("prim::TupleConstruct")
        num_outputs = {len(x.output().type().elements()) for x in slices}
        # there should be only one tupleSlice with length of 2
        self.assertTrue(num_outputs == {2})
        self.run_pass("lower_all_tuples", tuple_graph)
        self.assertTrue("Tuple" not in str(tuple_graph))

    def test_module_list_slicing(self):
        # [FUNCTION DOC] Checks that a ModuleList can be sliced with a negative step and with a
        # start and stop inside a scripted module forward, and that the returned submodules keep
        # their identifier attributes in the expected order.
        class Bar(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module that stores a string identifier; forward
            # returns zero.
            def __init__(self, identifier: str):
                # [FUNCTION DOC] HELPER: Stores the identifier string as an attribute.
                super().__init__()
                self.identifier = identifier

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the constant zero.
                return 0

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleList of five Bar modules labelled A to
            # E.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a ModuleList of five Bar modules named A to E.
                super().__init__()
                module_list = [Bar("A"), Bar("B"), Bar("C"), Bar("D"), Bar("E")]
                self.test = torch.nn.ModuleList(module_list)

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns two ModuleList slices. The first is reversed
                # with step minus two (modules E, C, A); the second is items one to three
                # (modules B, C, D).
                return self.test[::-2], self.test[1:4:]

        scripted_foo = torch.jit.script(Foo())
        result1, result2 = scripted_foo()

        self.assertEqual(len(result1), 3)
        self.assertEqual(result1[0].identifier, "E")
        self.assertEqual(result1[1].identifier, "C")
        self.assertEqual(result1[2].identifier, "A")

        self.assertEqual(len(result2), 3)
        self.assertEqual(result2[0].identifier, "B")
        self.assertEqual(result2[1].identifier, "C")
        self.assertEqual(result2[2].identifier, "D")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
