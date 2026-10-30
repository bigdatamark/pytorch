# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for typing features: dict membership, list and dict
# annotation mismatches, tuple, list and optional handling, iteration over strings, dicts and
# tuples with unpacking, dict comprehensions, tuple assignment forms, namedtuple resolution
# and errors, and annotation inheritance on modules. Most tests compile small functions
# through checkScript or torch.jit.script. Importing only extends the module search path;
# running directly is rejected.
# ruff: noqa: F841

import os
import sys
from collections import namedtuple
from typing import Dict, List, NamedTuple, Tuple

import torch
from torch.testing._internal.common_utils import IS_WINDOWS, raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestTyping(JitTestCase):
    # [CLASS DOC] Test case with the TorchScript typing tests of this file; stateless apart
    # from a few tests that define module-level globals for namedtuple name resolution.
    # get_sum_list_fn is a shared factory used by several sum tests.
    def test_dict_in_not_in(self):
        # [FUNCTION DOC] Checks dict membership operators in scripted code: membership and
        # non-membership with string keys, tensor keys, and evaluation order (left operand
        # evaluated before right, verified via captured stdout unless on Windows).
        # [FUNCTION DOC] HELPER: Returns whether the key hi is in the dict argument. Block
        # placed above the definition so the type comment stays the first body line.
        def test_in_dict(x):
            # type: (Dict[str, int]) -> bool
            return "hi" in x

        self.checkScript(test_in_dict, ({"hi": 2, "bye": 3},))
        self.checkScript(test_in_dict, ({"bye": 3},))

        # Check evaluation order
        @torch.jit.script
        def a():
            # [FUNCTION DOC] HELPER: Scripted function printing a and returning three; used to
            # check evaluation order.
            print("a")
            return 3

        @torch.jit.script
        def b():
            # [FUNCTION DOC] HELPER: Scripted function printing b and returning a two-entry
            # int dict.
            print("b")
            return {3: 2, 4: 1}

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function returning whether the result of a is a
            # key of the result of b.
            return a() in b()

        with self.capture_stdout() as captured:
            self.assertTrue(fn())
        if not IS_WINDOWS:
            # no stdout capturing on windows
            self.assertEqual(captured[0], "a\nb\n")

        # [FUNCTION DOC] HELPER: Returns False if hello is not a key of the dict argument,
        # else True. Block placed above the definition so the type comment stays the first
        # body line.
        def test_not_in_dict(a):
            # type: (Dict[str, int]) -> bool
            if "hello" not in a:
                return False
            else:
                return True

        self.checkScript(test_not_in_dict, ({"hello": 1, "world": 2},))
        self.checkScript(test_not_in_dict, ({"world": 2},))

        # [FUNCTION DOC] HELPER: Returns whether tensor t is a key in a dict with tensor keys.
        # Block placed above the definition so the type comment stays the first body line.
        def test_dict_tensor_key(a, t):
            # type: (Dict[Tensor, int], Tensor) -> bool
            if t in a:
                return True
            else:
                return False

        inp1 = torch.tensor(3)
        inp2 = torch.tensor(5)
        dict_a = {inp1: 1, inp2: 3}
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(4)))
        self.checkScript(test_dict_tensor_key, (dict_a, torch.tensor(3)))
        self.checkScript(test_dict_tensor_key, (dict_a, inp1))
        self.checkScript(test_dict_tensor_key, (dict_a, inp2))

    def test_list_type_refinement_annotation_element_mismatch(self):
        # [FUNCTION DOC] Checks that annotating a list of int but providing a list with a
        # string element fails scripting with RuntimeError about the list type annotation not
        # matching the element types.
        def fn():
            # [FUNCTION DOC] HELPER: Annotates a list of int with a mixed int and string
            # literal list and returns it; expected to fail compiling.
            l: List[int] = [1, 2, "foo", 3]
            return l

        with self.assertRaisesRegex(
            RuntimeError,
            "List type annotation"
            r" `List\[int\]` did not match the "
            "types of the given list elements",
        ):
            torch.jit.script(fn)

    def test_dict_type_refinement_annotation_key_mismatch(self):
        # [FUNCTION DOC] Checks that building a dict annotated with int keys from keys that
        # unify to a union of int and str fails scripting with RuntimeError saying dict keys
        # must be homogeneous.
        def fn():
            # [FUNCTION DOC] HELPER: Zips a mixed key list with string values into an
            # annotated dict; expected to fail compiling.
            l1 = [1, 2, "foo", 3]
            l2 = ["foo", "bar", "baz", "qux"]
            d: Dict[int, str] = dict(zip(l1, l2))
            return d

        with self.assertRaisesRegex(
            RuntimeError,
            "Dicts may only "
            "contain homogeneous keys, but the "
            "type of the first generated key "
            r"was Union\[int, str\]",
        ):
            torch.jit.script(fn)

    def test_dict_type_refinement_annotation_value_mismatch(self):
        # [FUNCTION DOC] Checks that an annotated dict of str to int built from values that
        # unify to a union of int and str fails scripting with RuntimeError about the dict
        # type annotation not matching the value type.
        def fn():
            # [FUNCTION DOC] HELPER: Zips string keys with mixed values into an annotated
            # dict; expected to fail compiling.
            l1 = ["foo", "bar", "baz", "qux"]
            l2 = [1, 2, "foo", 3]
            d: Dict[str, int] = dict(zip(l1, l2))
            return d

        with self.assertRaisesRegex(
            RuntimeError,
            "Dict type annotation"
            r" `Dict\[str, int\]` did not match"
            " the type of an actual value type"
            r" `Union\[int, str\]`",
        ):
            torch.jit.script(fn)

    def test_dict_invalid_annotations(self):
        # [FUNCTION DOC] Checks that functions whose parameters are annotated with dicts
        # having a ScriptModule key type, value type or both fail scripting with ValueError
        # containing unknown type annotation.
        # Check for invalid value type annotation
        def wrong_value_type(dictionary: Dict[str, torch.jit.ScriptModule]):
            # [FUNCTION DOC] HELPER: Function with an invalid dict value type annotation; does
            # nothing.
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_value_type)

        # Check for invalid key type annotation
        def wrong_key_type(dictionary: Dict[torch.jit.ScriptModule, str]):
            # [FUNCTION DOC] HELPER: Function with an invalid dict key type annotation; does
            # nothing.
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_key_type)

        # Check for invalid key and value type annotation
        def wrong_key_value_type(
            dictionary: Dict[torch.jit.ScriptModule, torch.jit.ScriptModule],
        ):
            # [FUNCTION DOC] HELPER: Function with invalid dict key and value type
            # annotations; does nothing.
            return

        with self.assertRaisesRegex(ValueError, "Unknown type annotation"):
            torch.jit.script(wrong_key_value_type)

    def test_tuple_specialization(self):
        # [FUNCTION DOC] Checks that calling a scripted function with nested tuple input
        # specializes the graph input types: the first element and the nested second element
        # are tensor types in the graph for the given input.
        # [FUNCTION DOC] HELPER: Scripted function unpacking a nested tuple and returning the
        # sum of the first tensor and nested tensor. Block placed above the definition so the
        # type comment stays the first body line.
        @torch.jit.script
        def f(t, s):
            # type: (Tuple[Tensor, Tuple[int, Tensor]], str) -> Tensor
            x, t2 = t
            _, y = t2
            return x + y

        t = (
            torch.randn(2, 2),
            (1, torch.randn(2, 2)),
        )
        f(t, "hi")
        graph = f.graph_for(t, "hi")
        input_types = list(next(graph.inputs()).type().elements())
        w = input_types[0]
        self.assertEqual(input_types[0].kind(), "TensorType")
        self.assertEqual(input_types[1].elements()[1].kind(), "TensorType")

    def test_tuple_io(self):
        # [FUNCTION DOC] Checks passing and returning a tuple of two tensors through a
        # scripted function via checkScript.
        # [FUNCTION DOC] HELPER: Unpacks a tuple of two tensors and returns them swapped.
        # Block placed above the definition so the type comment stays the first body line.
        def stuff(x):
            # type: (Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]
            a, b = x
            return b, a

        a = (torch.rand(3), torch.rand(3))
        self.checkScript(stuff, (a,))

    def test_tuple_keyword(self):
        # [FUNCTION DOC] Checks the tuple builtin: converting a literal tuple works, while
        # calling it with two arguments fails with an error mentioning one argument, and
        # converting a list of unknown length fails with an error about statically inferring
        # the size.
        def bar():
            # [FUNCTION DOC] HELPER: Converts a literal pair via the tuple builtin and returns
            # it.
            f = tuple((1, 2))  # noqa: C409
            return f

        self.checkScript(bar, ())

        # [FUNCTION DOC] HELPER: Calls the tuple builtin with two arguments; expected to fail
        # compiling.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def foo():
            return tuple(1, 2)

        self.checkScriptRaisesRegex(foo, (), Exception, "1 argument")

        # [FUNCTION DOC] HELPER: Calls the tuple builtin on a three-element list; expected to
        # fail compiling.
        # KNOWN ISSUE: block placed above the definition (not inside the body) because tests
        # assert on the compiled source text or error highlights.
        def cant_infer_size():
            return tuple([1, 2, 3])  # noqa: C409

        with self.assertRaisesRegex(Exception, "cannot statically infer the expected"):
            torch.jit.script(cant_infer_size)

    def test_tuple_create_return(self):
        # [FUNCTION DOC] Checks creating and returning a tuple of two tensors built from an
        # int size via checkScript.
        # [FUNCTION DOC] HELPER: Returns a tuple of ones and zeros tensors of the given size.
        # Block placed above the definition so the type comment stays the first body line.
        def stuff2(x):
            # type: (int) -> Tuple[Tensor, Tensor]
            a = (torch.ones(x), torch.zeros(x))
            return a

        self.checkScript(stuff2, (3,))

    def test_list_io(self):
        # [FUNCTION DOC] Checks a function taking a list of int and returning a tuple of a
        # ones tensor and the list via checkScript.
        # [FUNCTION DOC] HELPER: Returns a ones tensor shaped by the list, and the list. Block
        # placed above the definition so the type comment stays the first body line.
        def stuff3(x):
            # type: (List[int]) -> Tuple[Tensor, List[int]]
            return torch.ones(x), x

        self.checkScript(stuff3, ([3, 2],))

    def test_bool_list_io(self):
        # [FUNCTION DOC] Checks that a scripted function returning bool lists produces Python
        # bool elements in each returned list.
        # [FUNCTION DOC] HELPER: Returns the input bool list, a literal bool list and a nested
        # one-element bool list. Block placed above the definition so the type comment stays
        # the first body line.
        @torch.jit.script
        def stuff4(x):
            # type: (List[bool]) -> Tuple[List[bool], List[bool], List[List[bool]]]
            return x, [True, False], [[True]]

        li_1, li_2, li_3 = stuff4([True])
        li_3 = li_3[0]
        for li in [li_1, li_2, li_3]:
            self.assertTrue(type(li[0]) is bool)

    def test_nested_list(self):
        # [FUNCTION DOC] Checks indexing into a nested list obtained by unpacking a tuple, via
        # checkScript.
        # [FUNCTION DOC] HELPER: Unpacks an int and a list of lists and returns the second
        # element of the first inner list. Block placed above the definition so the type
        # comment stays the first body line.
        def foo(z):
            # type: (Tuple[int, List[List[int]]]) -> int
            x, y = z
            return y[0][1]

        self.checkScript(foo, ((1, [[1, 2], [3, 4]]),))

    def test_list_sum(self):
        # [FUNCTION DOC] Checks the sum builtin over lists of int, float and bool (including
        # int-valued lists given to a float-annotated and a bool-annotated function) via
        # checkScript.
        def fn(x: List[int]) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of a list of int.
            return sum(x)

        def fn1(x: List[float]):
            # [FUNCTION DOC] HELPER: Returns the sum of a list of float.
            return sum(x)

        def fn2(x: List[bool]):
            # [FUNCTION DOC] HELPER: Returns the sum of a list of bool.
            return sum(x)

        self.checkScript(fn, ([1, 2, 3],))
        self.checkScript(fn1, ([1.0, 2.0, 3.0],))
        self.checkScript(fn1, ([1, 2.8, 3],))
        self.checkScript(fn2, ([True, False, False],))
        self.checkScript(fn2, ([False, False, False],))
        self.checkScript(fn2, ([0, 1, 1, 0],))

    def test_list_unification(self):
        # [FUNCTION DOC] Checks that list literals mixing int and None, and tensors and None,
        # unify to optional element types, via checkScript.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a list holding two ints and a None.
            return [1, None, 2]

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns a list of a ones tensor, None and the argument.
            return [torch.ones(2, 2), None, x]

        self.checkScript(fn, [])
        self.checkScript(fn2, (torch.ones(2, 2),))

    # to avoid defining sum_list in multiple tests
    def get_sum_list_fn(self):
        # [FUNCTION DOC] HELPER: Factory returning a function that sums a list of int with a
        # loop; shared by several sum tests to avoid redefining it.
        # [FUNCTION DOC] HELPER: Sums the elements of an int list with a loop and returns the
        # total. Block placed above the definition so the type comment stays the first body
        # line.
        def sum_list(a):
            # type: (List[int]) -> int
            sum = 0
            for i in a:
                sum += i

            return sum

        return sum_list

    def test_sum_list_diff_elms(self):
        # [FUNCTION DOC] Runs the shared list-sum function on a five-element list via
        # checkScript.
        self.checkScript(self.get_sum_list_fn(), ([1, 2, 3, 4, 5],))

    def test_sum_list_empty(self):
        # [FUNCTION DOC] Runs the shared list-sum function on an empty list via checkScript.
        self.checkScript(self.get_sum_list_fn(), ([],))

    def test_sum_list_one(self):
        # [FUNCTION DOC] Runs the shared list-sum function on a one-element list via
        # checkScript.
        self.checkScript(self.get_sum_list_fn(), ([1],))

    def test_sum_list_literal(self):
        # [FUNCTION DOC] Checks summing over a list literal inside a scripted function via
        # checkScript.
        # [FUNCTION DOC] HELPER: Loops over a literal list of five ints and returns their sum.
        # Block placed above the definition so the type comment stays the first body line.
        def sum_list():
            # type: () -> int
            sum = 0
            for i in [1, 2, 3, 4, 5]:
                sum += i

            return sum

        self.checkScript(sum_list, ())

    def test_sum_list_wrong_type(self):
        # [FUNCTION DOC] Checks that scripting a function that iterates over an int parameter
        # and then calling it fails with RuntimeError saying an int object is not iterable.
        with self.assertRaisesRegex(RuntimeError, "'int' object is not iterable"):

            # [FUNCTION DOC] HELPER: Scripted function iterating over an int-annotated
            # argument (invalid) and summing. Block placed above the definition so the type
            # comment stays the first body line.
            @torch.jit.script
            def sum_list(a):
                # type: (int) -> int
                sum = 0
                for i in a:  # noqa: T484
                    sum += i

                return sum

            sum_list(1)

    def test_list_iterables(self):
        # [FUNCTION DOC] Checks that defining a function in a compilation unit that iterates
        # over a literal pair of lists with tuple unpacking fails with RuntimeError saying
        # list of iterables is not supported. The function source lives in a string and is not
        # documented here.
        with self.assertRaisesRegex(
            RuntimeError, "List of iterables is not supported currently"
        ):
            cu = torch.jit.CompilationUnit(
                """
            def list_iterables(x):
                for i, j in [2, 3, 4], [5, 6, 7]:
                    x += i
                    x += j
                return x
            """
            )

    def test_for_in_string(self):
        # [FUNCTION DOC] Checks iterating over the characters of a string and over a list of
        # strings in scripted functions, via checkScript.
        # [FUNCTION DOC] HELPER: Builds the reverse of a string by iterating over its
        # characters. Block placed above the definition so the type comment stays the first
        # body line.
        def test_strings(x):
            # type: (str) -> str
            reverse = ""
            for c in x:
                reverse = c + reverse
            return reverse

        self.checkScript(test_strings, ("hello",))
        self.checkScript(test_strings, ("",))

        # [FUNCTION DOC] HELPER: Concatenates the strings of a list. Block placed above the
        # definition so the type comment stays the first body line.
        def test_list_strings(x):
            # type: (List[str]) -> str
            result = ""
            for sub_str in x:
                result += sub_str
            return result

        self.checkScript(test_list_strings, (["hello", "world"],))
        self.checkScript(test_list_strings, (["hello", " ", "world", ""],))

    def test_for_in_dict(self):
        # [FUNCTION DOC] Checks iterating over a dict in a scripted function. The second
        # nested function is defined but never scripted, and the final check re-runs the first
        # function.
        # [FUNCTION DOC] HELPER: Sums the values of a str to int dict by iterating over its
        # keys. Block placed above the definition so the type comment stays the first body
        # line.
        def test_dicts(x):
            # type: (Dict[str, int]) -> int
            sum = 0
            for key in x:
                sum += x[key]
            return sum

        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

        # [FUNCTION DOC] HELPER: Concatenates dict keys and sums dict values; defined but
        # never exercised. Block placed above the definition so the type comment stays the
        # first body line.
        def test_dict_keys_values(x):
            # type: (Dict[str, int]) -> Tuple[str, int]
            key_str = ""
            sum = 0
            for key in x:
                key_str += key
            for val in x.values():
                sum += val
            return key_str, sum

        self.checkScript(test_dicts, ({"a": 1, "b": 2, "c": 3},))

    def test_for_tuple_unpack(self):
        # [FUNCTION DOC] Checks tuple unpacking in for loops over a list of lists and nested
        # unpacking over a zip of a list, an enumerate and a list, via checkScript.
        def for_tuple_unpack(x, y):
            # [FUNCTION DOC] HELPER: Loops over a literal list of pairs, adds the unpacked
            # items to the two arguments and returns both.
            for i, j in [[3, 4], [5, 6], [7, 8]]:
                x += i
                y += j
            return x, y

        self.checkScript(for_tuple_unpack, (torch.tensor(3), torch.tensor(5)))

        # [FUNCTION DOC] HELPER: Sums the zipped elements using nested unpacking of enumerate
        # results. Block placed above the definition so the type comment stays the first body
        # line.
        def nested_tuple_unpack(x, y):
            # type: (List[int], List[int]) -> int
            sum = 0
            for i, (j, k), v in zip(x, enumerate(x), y):
                sum += i + j + k + v
            return sum

        self.checkScript(nested_tuple_unpack, ([1, 3, 5], [2, 4, 6]))

    def test_dict_comprehension(self):
        # [FUNCTION DOC] Checks a dict comprehension mapping ints to letters via checkScript.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a dict comprehension from the first four ints to
            # characters.
            return {i: chr(i + 65) for i in range(4)}

        self.checkScript(fn, ())

    def test_dict_comprehension_with_type_annotation(self):
        # [FUNCTION DOC] Checks a dict comprehension with a dict annotation passes
        # checkScript, and that annotating the target as a tuple makes scripting raise an
        # error (the inner assertion about expected dict annotation is checked inside an outer
        # RuntimeError expectation).
        def fn():
            # [FUNCTION DOC] HELPER: Returns an annotated dict built by a comprehension.
            d: Dict[int, str] = {i: chr(i + 65) for i in range(4)}
            return d

        self.checkScript(fn, ())

        with self.assertRaisesRegex(RuntimeError, ""):
            with self.assertRaisesRegex(
                AssertionError,
                "Expected Dict "
                "type annotation for dict "
                "comprehension, found "
                "Tuple[int, str]",
            ):

                # [FUNCTION DOC] HELPER: Scripted function annotating a dict comprehension
                # result as a tuple; expected to fail compiling.
                # KNOWN ISSUE: block placed above the definition (not inside the body) because
                # tests assert on the compiled source text or error highlights.
                @torch.jit.script
                def fn():
                    d: Tuple[int, str] = {i: chr(i + 65) for i in range(4)}
                    return d

    def test_dict_comprehension_scope(self):
        # [FUNCTION DOC] Checks that a comprehension can read variables of the enclosing
        # scope, and that its loop variable is not visible afterwards (undefined value error).
        def comprehension_can_access_outer_scope_variables():
            # [FUNCTION DOC] HELPER: Returns a dict from words to lengths built from a local
            # list.
            lst = ["foo", "bar", "baz"]
            return {l: len(l) for l in lst}

        self.checkScript(comprehension_can_access_outer_scope_variables, ())

        with self.assertRaisesRegex(RuntimeError, "undefined value i"):

            # [FUNCTION DOC] HELPER: Scripted function reading the comprehension variable
            # after the comprehension; expected to fail compiling.
            # KNOWN ISSUE: block placed above the definition (not inside the body) because
            # tests assert on the compiled source text or error highlights.
            @torch.jit.script
            def outer_scope_cannot_access_comprehension_variables():
                d = {i: chr(i + 65) for i in range(4)}
                i = i + 1  # noqa: F821

    def test_for_tuple_assign(self):
        # [FUNCTION DOC] Checks iterating over tuples in scripted functions. The nested
        # method-like function at the end is never invoked.
        # [FUNCTION DOC] HELPER: Sums a heterogeneous int and float tuple by iterating over
        # it. Block placed above the definition so the type comment stays the first body line.
        def test_simple_assign(x):
            # type: (Tuple[int, float]) -> float
            sum = 0.0
            for a in x:
                sum += float(a)
            return sum

        self.checkScript(test_simple_assign, ((1, 2.5),))

        # [FUNCTION DOC] HELPER: Sums the entries of a tuple of two int pairs by iterating.
        # Block placed above the definition so the type comment stays the first body line.
        def test_tuple_assign(x):
            # type: (Tuple[Tuple[int, int], Tuple[int, int]]) -> int
            sum = 0
            for a in x:
                sum += a[0]
                sum += a[1]
            return sum

        self.checkScript(test_tuple_assign, (((1, 2), (4, 7)),))

        def test_single_starred_lhs(self):
            # [FUNCTION DOC] HELPER: Dead nested function (never called, takes an unused
            # self); it would check that a lone starred left-hand side raises RuntimeError.
            # Its source string is not compiled.
            with self.assertRaisesRegex(
                RuntimeError,
                "A Starred expression may only appear on the lhs within the presence"
                " of another non-starred expression",
            ):
                cu = torch.jit.CompilationUnit(
                    """
                def single_starred_lhs(x):
                    a = (x, x, x)
                    *b, = a
                    return b
                """
                )

    def test_singleton_tuple_unpack(self):
        # [FUNCTION DOC] Checks unpacking a one-element tuple, via checkScript.
        def foo(a):
            # [FUNCTION DOC] HELPER: Unpacks a one-element tuple of the argument and returns
            # it plus one.
            (b,) = (a,)
            return b + 1

        self.checkScript(foo, (torch.rand(3),))

    def test_tuple_assignments(self):
        # [FUNCTION DOC] Checks tuple assignment forms via checkScript: nested unpacking,
        # subscript targets, starred targets; and that augmented assignment on a tuple element
        # fails scripting with an error about unsupported augmented assign.
        # [FUNCTION DOC] HELPER: Unpacks a pair of tensors and a third, returning their sum.
        # Block placed above the definition so the type comment stays the first body line.
        def var_tuple_assign(x, y):
            # type: (Tuple[Tensor, Tensor], Tensor) -> Tensor
            (a, b), c = x, y
            return a + b + c

        tuple_inputs = (torch.randn(1, 4), torch.randn(3, 4))
        self.checkScript(var_tuple_assign, (tuple_inputs, torch.randn(3, 4)))

        # [FUNCTION DOC] HELPER: Unpacks nested tuples into six ints and returns their sum.
        # Block placed above the definition so the type comment stays the first body line.
        def nested_tuple_assign(x, y, z):
            # type: (int, Tuple[int, Tuple[int, int]], Tuple[int, int]) -> int
            a, (b, (c, d)), (e, f) = x, y, z
            return a + b + c + d + e + f

        self.checkScript(nested_tuple_assign, ((1, (2, (3, 4)), (5, 6))))

        # [FUNCTION DOC] HELPER: Assigns to list and tensor subscripts via tuple assignment
        # and returns derived values. Block placed above the definition so the type comment
        # stays the first body line.
        def subscript_tuple_assign(a, x, i):
            # type: (List[int], Tensor, int) -> Tuple[int, Tensor, int]
            a[i], (x[i], b) = 1, (2, 3)
            return a[i] + 1, x + 5, b

        self.checkScript(
            subscript_tuple_assign, ([12, 7, 9, 11], torch.tensor((3, 13, 17)), 0)
        )

        # [FUNCTION DOC] HELPER: Uses starred targets in a nested tuple assignment and returns
        # the pieces. Block placed above the definition so the type comment stays the first
        # body line.
        def star_tuple_assign():
            # type: () -> Tuple[int, int, Tuple[int, int], Tuple[int, int]]
            a, (b, *c), *d = 1, (2, 3, 4), 5, 6
            return a, b, c, d

        self.checkScript(star_tuple_assign, ())

        # [FUNCTION DOC] HELPER: Attempts augmented assignment to a tuple element; expected to
        # fail compiling. Block placed above the definition so the type comment stays the
        # first body line.
        def subscript_tuple_augmented_assign(a):
            # type: (Tuple[int, int]) -> Tuple[int, int]
            a[0] += 1
            return a

        with self.assertRaisesRegex(RuntimeError, "does not support augmented assign"):
            scripted_aug_assign = torch.jit.script(subscript_tuple_augmented_assign)

    def test_multiple_assign(self):
        # [FUNCTION DOC] Checks chained assignments, side effects of in-place ops through
        # chained targets, and evaluation order of swaps with tuple assignment, via
        # checkScript.
        def test():
            # [FUNCTION DOC] HELPER: Exercises chained tuple assignment, in-place tensor add
            # shared between two names, and a simultaneous swap, returning all values.
            a = b, c = d, f = (1, 1)

            # side effect
            ten = torch.tensor(1)
            ten1 = ten2 = ten.add_(1)

            # ordering
            x = 1
            y = 3
            x, y = y, x + y

            return a, b, c, d, f, ten, ten1, ten2, x, y

        self.checkScript(test, ())

    def test_opt_opt_refinement(self):
        # [FUNCTION DOC] Checks that a scripted function unifying optional values assigned in
        # nested branches compiles. No assertions on the result.
        # [FUNCTION DOC] HELPER: Assigns None or one depending on optional arguments and
        # returns the optional result. Block placed above the definition so the type comment
        # stays the first body line.
        @torch.jit.script
        def test_unify(weight, bias):
            # type: (Optional[int], Optional[int]) -> Optional[int]
            if weight is not None:
                opt = None
            else:
                if bias is not None:
                    opt = 1
                else:
                    opt = None

            return opt

    def test_optional_refinement(self):
        # [FUNCTION DOC] Checks that an optional int refined by an is-None assignment can be
        # used as an int, and the scripted function returns two for input one.
        # [FUNCTION DOC] HELPER: Replaces a None argument with one and returns the value plus
        # one. Block placed above the definition so the type comment stays the first body
        # line.
        @torch.jit.script
        def test_if_none_assignment(x):
            # type: (Optional[int]) -> int
            if x is None:
                x = 1
            return x + 1

        self.assertEqual(test_if_none_assignment(1), 2)

    def test_optional_conversion(self):
        # [FUNCTION DOC] Checks conversions involving optional types in scripted functions:
        # unwrapping an optional, unifying None and an int, and passing a tuple to functions
        # taking optional list parameters; asserts the expected returned values.
        # [FUNCTION DOC] HELPER: Unwraps an optional int using the internal unwrap helper.
        # Block placed above the definition so the type comment stays the first body line.
        @torch.jit.script
        def other_fn(x=None):
            # type: (Optional[int]) -> int
            return torch.jit._unwrap_optional(x)

        # [FUNCTION DOC] HELPER: Calls the optional-taking function with an int. Block placed
        # above the definition so the type comment stays the first body line.
        @torch.jit.script
        def fn(x):
            # type: (int) -> int
            return other_fn(x)

        self.assertEqual(fn(2), 2)

        # [FUNCTION DOC] HELPER: Returns None or two depending on the bool argument, unifying
        # to an optional int. Block placed above the definition so the type comment stays the
        # first body line.
        @torch.jit.script
        def unify_to_optional(x):
            # type: (bool) -> Optional[int]
            if x:
                a = None
            else:
                a = 2
            return a

        self.assertEqual(unify_to_optional(True), None)
        self.assertEqual(unify_to_optional(False), 2)

        # [FUNCTION DOC] HELPER: Accepts an optional list of float and returns two. Block
        # placed above the definition so the type comment stays the first body line.
        @torch.jit.script
        def opt_list(x):
            # type: (Optional[List[float]]) -> int
            return 2

        # [FUNCTION DOC] HELPER: Accepts an optional broadcasting list of two floats and
        # returns two. Block placed above the definition so the type comment stays the first
        # body line.
        @torch.jit.script
        def broadcast_opt_list(x):
            # type: (Optional[BroadcastingList2[float]]) -> int
            return 2

        # [FUNCTION DOC] HELPER: Calls both optional-list functions with a pair of floats and
        # returns the sum of results. Block placed above the definition so the type comment
        # stays the first body line.
        @torch.jit.script
        def opt_list_tuple_caller(x):
            # type: (Tuple[float, float]) -> int
            return opt_list(x) + broadcast_opt_list(x)

        self.assertEqual(opt_list_tuple_caller((2.0, 3.0)), 4)

    def test_optional_tuple(self):
        # [FUNCTION DOC] Checks a function with an optional tuple default through checkScript,
        # called with and without the argument.
        # [FUNCTION DOC] HELPER: Returns the argument, or the tuple one and two if it is None.
        # Block placed above the definition so the type comment stays the first body line.
        def fn(x=None):
            # type: (Optional[Tuple[int, int]]) -> Tuple[int, int]
            if x is None:
                new_x = (1, 2)
            else:
                new_x = x
            return new_x

        self.checkScript(fn, ((3, 4),))
        self.checkScript(fn, ())

    def test_namedtuple_redefine(self):
        # [FUNCTION DOC] Checks that using two different namedtuple types with the same name
        # in one scripted signature fails with RuntimeError mentioning redefinition. Sets two
        # module-level globals as a side effect.
        global _1, _2
        _1 = namedtuple("GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"])
        _2 = namedtuple("GoogLeNetOutputs", ["different"])

        with self.assertRaisesRegex(RuntimeError, r"redefine"):

            # [FUNCTION DOC] HELPER: Scripted function annotated with the two clashing
            # namedtuple types; expected to fail compiling. Block placed above the definition
            # so the type comment stays the first body line.
            @torch.jit.script
            def foo(x, y):
                # type: (_1, _2) -> _1
                return x

    def test_namedtuple_py2(self):
        # [FUNCTION DOC] Checks passing a namedtuple through a scripted function that names
        # its type via a module-level global (set here), and reading back its fields.
        global _GoogLeNetOutputs  # see [local resolution in python]
        _GoogLeNetOutputs = namedtuple(
            "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
        )

        # [FUNCTION DOC] HELPER: Identity function over the namedtuple type. Block placed
        # above the definition so the type comment stays the first body line.
        @torch.jit.script
        def foo(x):
            # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
            return x

        vals = torch.rand(3), torch.rand(4), torch.rand(5)
        out = foo(
            _GoogLeNetOutputs(logits=vals[0], aux_logits2=vals[1], aux_logits1=vals[2])
        )
        self.assertEqual(out.logits, vals[0])
        self.assertEqual(out.aux_logits2, vals[1])
        self.assertEqual(out.aux_logits1, vals[2])

    def test_namedtuple_good_error(self):
        # [FUNCTION DOC] Checks that calling a scripted function annotated with a namedtuple
        # type, using an instance whose fields are strings, raises RuntimeError whose message
        # includes the aka NamedTuple description with the field names. Sets a module-level
        # global as a side effect.
        global _GoogLeNetOutputs  # see [local resolution in python]
        _GoogLeNetOutputs = namedtuple(
            "GoogLeNetOutputs", ["logits", "aux_logits2", "aux_logits1"]
        )

        # [FUNCTION DOC] HELPER: Identity function over the namedtuple type. Block placed
        # above the definition so the type comment stays the first body line.
        @torch.jit.script
        def foo(x):
            # type: (_GoogLeNetOutputs) -> _GoogLeNetOutputs
            return x

        with self.assertRaisesRegex(
            RuntimeError, r"aka NamedTuple\(logits, aux_logits2, aux_logits1\)"
        ):
            out = foo(_GoogLeNetOutputs(logits="3", aux_logits2="4", aux_logits1="5"))

    def test_namedtuple_error_source_attribution(self):
        # [FUNCTION DOC] Checks that scripting a function whose NamedTuple argument has an
        # unresolvable forward-reference member annotation raises ValueError whose message
        # includes a file location.
        class _NamedTupleBadMemberType(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a tensor field and a field annotated with an
            # undefined forward reference; made global for name resolution.
            f1: torch.Tensor
            f2: "ABadForwardRefType"  # noqa: F821

        make_global(_NamedTupleBadMemberType)  # see [local resolution in python]

        def fn(x: _NamedTupleBadMemberType) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the relu of the first field of the argument.
            return x.f1.relu()

        # assert that this has a location associated with the error.
        # note the " +" is regex (i.e. "at least one space")
        with self.assertRaisesRegex(ValueError, "at +File"):
            torch.jit.script(fn)

    def test_inherited_annotations_python_310(self):
        # [FUNCTION DOC] Regression test for inherited annotations with newer Python versions:
        # scripts two instances of a module subclass chain whose base class declares a list
        # attribute annotation. Passes if scripting does not raise.
        # See #104484
        # In python >=3.10, inspect.get_annotations doesn't always return the same values.
        # Sometimes it will show all annotations; other times it will show only annotations
        # that show in that class, not classes it inherits fro.
        class BaseModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module base declaring an annotated list attribute; forward
            # does nothing.
            state: List[int]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forward that does nothing.
                pass

        def do_something_with_list(x: List[int]):
            # [FUNCTION DOC] HELPER: Returns the last element of the list if non-empty, else
            # five.
            if x:
                return x[-1]
            return 5

        class Submodule(BaseModule):
            # [CLASS DOC] HELPER: Subclass storing an int and an empty list state; forward
            # adds input, the int and a helper result on the state.
            def __init__(self, self_x_value):
                # [FUNCTION DOC] HELPER: Stores the given value in attribute x and sets state
                # to an empty list.
                super().__init__()
                self.x = self_x_value
                self.state = []

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns self.x plus input plus the helper applied to
                # state.
                return self.x + x + do_something_with_list(self.state)

        class LowestModule(Submodule):
            # [CLASS DOC] HELPER: Further subclass with a fixed constructor value one hundred
            # twenty-three.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Constructor calling the parent with 123.
                super().__init__(123)

        mod = LowestModule()
        mod2 = LowestModule()
        mod_s = torch.jit.script(mod)
        mod2_s = torch.jit.script(mod2)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
