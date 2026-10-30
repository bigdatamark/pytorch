# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Tests for TorchScript list, dict, NamedTuple and ScriptDict/ScriptList
# behavior: list and dict literals, keyword constructors, comprehensions, mutation
# methods, tolist, type-annotation errors and reference semantics of scripted containers.
# Appends the test directory to sys.path at import time; running the file directly is
# rejected via raise_on_run_directly.

import inspect
import os
import sys
import types
import unittest
from collections import defaultdict, OrderedDict
from textwrap import dedent
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import (
    raise_on_run_directly,
    skipIfTorchDynamo,
    TEST_CUDA,
)
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestList(JitTestCase):
    # [CLASS DOC] JitTestCase suite for TorchScript list semantics. Most tests define small
    # functions and compare eager and scripted results with checkScript, or compile functions
    # that must fail and check the error message.
    def test_list_bool_conversion(self):
        # [FUNCTION DOC] Checks that a list used as a condition (if, while, ternary) behaves as in
        # Python: true when non-empty, false when empty. Each predicate is run on a non-empty
        # and an empty list.
        def if_predicate(l: List[int]):
            # [FUNCTION DOC] HELPER: Returns the sum of the elements when the list is truthy,
            # otherwise -1.
            if l:
                s = 0
                for n in l:
                    s += n

                return s
            else:
                return -1

        self.checkScript(if_predicate, ([1, 2, 3],))
        self.checkScript(if_predicate, ([],))

        def while_predicate(l: List[int]):
            # [FUNCTION DOC] HELPER: Loops while the list is truthy, popping the last element each
            # time; returns nothing.
            s = 0

            while l:
                s += l.pop()

        self.checkScript(while_predicate, ([1, 2, 3],))
        self.checkScript(while_predicate, ([],))

        def ternary_predicate(l: List[int]):
            # [FUNCTION DOC] HELPER: Returns the string non-empty for a truthy list and
            # empty otherwise.
            return "non-empty" if l else "empty"

        self.checkScript(ternary_predicate, ([1, 2, 3],))
        self.checkScript(ternary_predicate, ([],))

    def test_bare_container_annotation(self):
        # [FUNCTION DOC] Checks that compiling a function with a bare list annotation (no element
        # type) raises a RuntimeError, while a bare list used only as the second argument of a
        # runtime type check is accepted and returns the list length.
        err = r"Attempted to use list without a contained type"

        with self.assertRaisesRegex(RuntimeError, err):

            @torch.jit.script
            def bare_list_empty():
                # [FUNCTION DOC] HELPER: Scripted function that annotates a local variable with a
                # bare list type; compilation is expected to fail with the contained-type error.
                x: list = []
                return x

        # `isinstance` against a bare container is still valid (it is a
        # type-erased runtime check, not a value's element type).
        @torch.jit.script
        def uses_isinstance(x: List[int]):
            # [FUNCTION DOC] HELPER: Scripted function taking a list of ints; returns its length
            # when it is a list, else 0.
            if isinstance(x, list):
                return len(x)
            return 0

        self.assertEqual(uses_isinstance([1, 2, 3]), 3)

    def test_in_check(self):
        # [FUNCTION DOC] Checks the in operator on int, float and str lists, with present, absent
        # and empty inputs.
        def int_in(x: List[int]) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the int 2 is in the list.
            return 2 in x

        self.checkScript(int_in, ([1, 2, 3],))
        self.checkScript(int_in, ([1, 3, 3],))

        def float_in(x: List[float]) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the float 2.0 is in the list.
            return 2.0 in x

        self.checkScript(float_in, ([1.0, 2.0, 3.0],))
        self.checkScript(float_in, ([1.0, 3.0, 3.0],))

        def str_in(x: List[str]) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether the string hi is in the list.
            return "hi" in x

        self.checkScript(str_in, (["not", "here"],))
        self.checkScript(str_in, (["hi", "bye"],))
        self.checkScript(str_in, ([],))

    def test_list_literal(self):
        # [FUNCTION DOC] Checks list literal type rules on reassignment inside an if branch:
        # same-type reassignment and arity change compile, an empty literal later assigned ints
        # fails because it was inferred as a list of tensors, and assigning a list of a
        # different element type fails with a previously-had-type error. All use checkScript
        # without optimization.
        def reassign():
            # [FUNCTION DOC] HELPER: Assigns a one-element int list and conditionally reassigns a
            # two-element int list; returns nothing.
            x = [1]
            if 1 == 1:
                x = [2, 3]
            return

        self.checkScript(reassign, (), optimize=False)

        def reassign_arity_change():
            # [FUNCTION DOC] HELPER: Like reassign but the replacement list has three elements
            # (length change is allowed).
            x = [1]
            if 1 == 1:
                x = [1, 2, 3]
            return

        self.checkScript(reassign_arity_change, (), optimize=False)

        # [FUNCTION DOC] HELPER: Starts from an empty list literal (inferred as a list of tensors)
        # and reassigns a list of ints; compilation is expected to fail.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def reassign_from_empty_literal():
            x = []
            if 1 == 1:
                x = [1, 2, 3]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"previously had type List\[Tensor\]", "x"
        ):
            self.checkScript(reassign_from_empty_literal, (), optimize=False)

        def reassign_from_empty_builtin():
            # [FUNCTION DOC] HELPER: Reassigns annotated empty int and float lists with matching
            # literals and an unannotated empty list with a list of tensors; all must compile.
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
            y = torch.jit.annotate(List[float], [])
            if 1 == 1:
                y = [1.0, 2.0, 3.0]
            z = []
            if 1 == 1:
                z = [torch.randn([1])]
            return

        self.checkScript(reassign_from_empty_builtin, (), optimize=False)

        # [FUNCTION DOC] HELPER: Reassigns an int list variable to a float list literal;
        # compilation is expected to fail.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def reassign_bad_type():
            x = [1]
            if 1 == 1:
                x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_bad_type, (), optimize=False)

        # [FUNCTION DOC] HELPER: Annotated int list, reassigned to ints and then, in a nested if,
        # to a float list; compilation is expected to fail.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def reassign_nested():
            x = torch.jit.annotate(List[int], [])
            if 1 == 1:
                x = [1, 2, 3]
                if 1 == 1:
                    x = [1.0]
            return

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "previously had type", "x"
        ):
            self.checkScript(reassign_nested, (), optimize=False)

    def test_list_variance(self):
        # [FUNCTION DOC] Checks list invariance: list literals and comprehensions take their type
        # from the lhs annotation (a list of optional ints), but assigning an existing list of
        # ints (or nested lists) to a variable annotated with a list of optional ints fails with
        # a type-mismatch message.
        """
        `List[T1]` is not a subtype of `List[T2]`, even if `T1` is a
        subtype of `T2`. However, if we have a temporary list object
        (that is, a list comprehension or a list literal) on the rhs of
        an assignment statement, we want to ignore the inferred type of
        the rhs if we can prove that: 1) both the lhs and the rhs are
        lists, and 2) the inner type of the lhs list is a subtype of the
        inner type of the rhs list.

        # This should pass
        x: List[Optional[int]] = [None, None, None]

        # This should fail
        y: List[None] = [None, None, None]
        x: List[Optional[int]] = y
        """

        def test_listliteral_is_typed_from_annotation():
            # [FUNCTION DOC] HELPER: Returns a list literal of None values annotated as a list of
            # optional ints.
            x: List[Optional[int]] = [None, None, None]
            return x

        self.checkScript(test_listliteral_is_typed_from_annotation, ())

        def test_listcomprehension_is_typed_from_annotation():
            # [FUNCTION DOC] HELPER: Returns a comprehension producing None values annotated as a
            # list of optional ints.
            x: List[Optional[int]] = [None for _ in range(3)]
            return x

        self.checkScript(test_listcomprehension_is_typed_from_annotation, ())

        def test_lists_with_different_internal_types_are_invariant(self):
            # [FUNCTION DOC] HELPER: Takes an unused self argument; assigns a list of ints to a
            # variable annotated as list of optional ints. Scripting it must raise the mismatch
            # error for variable y.
            x: List[int] = [1, 2, 3]
            y: List[Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[Optional\[int\]\] but is "
            "being assigned to a value of type "
            r"List\[int\]",
        ):
            torch.jit.script(test_lists_with_different_internal_types_are_invariant)

        def test_lists_with_different_internal_types_are_invariant_recursive(self):
            # [FUNCTION DOC] HELPER: Same as the previous one with nested lists: a list of lists
            # of ints assigned to a list of lists of optional ints; scripting must raise the
            # mismatch error.
            x: List[List[int]] = [[1, 2], [3]]
            y: List[List[Optional[int]]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"List\[List\[Optional\[int\]\]\] "
            "but is being assigned to a value "
            r"of type List\[List\[int\]\]",
        ):
            torch.jit.script(
                test_lists_with_different_internal_types_are_invariant_recursive
            )

    def test_del(self):
        # [FUNCTION DOC] Checks deleting a list element by index: eager and scripted (via
        # CompilationUnit and torch.jit.script) results match; deleting index 100 of an empty
        # list raises an out-of-range error, and deleting a slice fails to compile (only
        # single-index deletion is supported).
        def inputs():
            # [FUNCTION DOC] HELPER: Returns a fresh four-element int list, since the tests mutate
            # their input.
            return [1, 2, 3, 4]

        def fn(x: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Deletes index 1 of the list and returns it.
            del x[1]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)

        # [FUNCTION DOC] HELPER: Scripted; deletes index 100 of the given list and returns it
        # (fails at runtime on a short list).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def fn2(x: List[int]) -> List[int]:
            del x[100]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "out of range", "x[100]"
        ):
            fn2([])

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "deletion at a single index", "x[1:3]"
        ):

            # [FUNCTION DOC] HELPER: Scripted; tries to delete a slice of the list, which must
            # fail to compile.
            # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the
            # body because this scripted function is used by a test that asserts on highlighted
            # source text of the error, and a body comment could appear in that quoted source.
            @torch.jit.script
            def fn(x: List[int]) -> List[int]:
                del x[1:3]
                return x

    def test_list_keyword(self):
        # [FUNCTION DOC] Checks the list constructor on lists, tuples, ranges and strings, with no
        # arguments (typed empty list), and on a list of a list, where exactly two list
        # operations must appear in the graph.
        def foo():
            # [FUNCTION DOC] HELPER: Returns a tuple of lists built from a list, a tuple, a range
            # and a string.
            return (
                list([1, 2, 3]),  # noqa: C410
                list(("a", "b")),  # noqa: C410
                list(range(5)),
                list("abcdefg"),
            )

        self.checkScript(foo, ())

        def foo2():
            # [FUNCTION DOC] HELPER: Builds an empty annotated int list with the constructor,
            # appends 1, returns it in a tuple.
            x: List[int] = list()
            x.append(1)
            return (x,)

        self.checkScript(foo2, ())

        def foo3():
            # [FUNCTION DOC] HELPER: Returns the list constructor applied to the list constructor
            # of a string.
            return list(list("abc"))  # noqa: C414

        self.checkScript(foo3, ())
        FileCheck().check_count("aten::list", 2, exactly=True).run(
            torch.jit.script(foo3).graph
        )

    def test_dict_keyword_with_kwargs(self):
        # [FUNCTION DOC] Checks the dict constructor with keyword arguments only matches
        # eager results.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from three keyword arguments with
            # int values.
            return dict(foo=1, bar=2, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_kwargs_using_container_values(self):
        # [FUNCTION DOC] Checks the dict constructor with keyword arguments whose values are
        # int lists.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from three keyword arguments with
            # int-list values.
            return dict(foo=[1, 2, 3], bar=[4, 5, 6], baz=[7, 8, 9])

        self.checkScript(fn, ())

    def test_dict_keyword_with_iterable(self):
        # [FUNCTION DOC] Checks the dict constructor given an iterable of key/value pairs.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from a list of three string/int pairs.
            return dict([("foo", 1), ("bar", 2), ("baz", 3)])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_iterable(self):
        # [FUNCTION DOC] Checks the dict constructor given an empty list as the iterable.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from an empty list.
            return dict([])  # noqa: C406

        self.checkScript(fn, ())

    def test_dict_keyword_with_internal_aggregate_function(self):
        # [FUNCTION DOC] Checks the dict constructor given a zip of two lists.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from zipping three string keys with
            # three int values.
            return dict(zip(["foo", "baz", "bar"], [1, 2, 3]))

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping(self):
        # [FUNCTION DOC] Checks a dict literal (named for the mapping form) compiles and
        # matches eager.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a dict literal with three string keys.
            return {"foo": 1, "bar": 2, "baz": 3}

        self.checkScript(fn, ())

    def test_dict_keyword_with_mapping_and_kwargs(self):
        # [FUNCTION DOC] Checks the dict constructor given a mapping plus an extra
        # keyword argument.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from a two-key mapping plus one
            # keyword argument.
            return dict({"foo": 1, "bar": 2}, baz=3)

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension(self):
        # [FUNCTION DOC] Checks a dict comprehension mapping ints to characters.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a dict comprehension mapping each of four ints to a
            # letter starting at A.
            return {i: chr(i + 65) for i in range(4)}

        self.checkScript(fn, ())

    def test_dict_keyword_with_dict_comprehension_and_kwargs(self):
        # [FUNCTION DOC] Checks the dict constructor given a dict comprehension plus a
        # keyword argument.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict built from a letter-to-int comprehension
            # plus one keyword argument.
            return dict({chr(65 + i): i for i in range(4)}, foo=2)

        self.checkScript(fn, ())

    def test_dict_keyword_with_empty_dict_comprehension(self):
        # [FUNCTION DOC] Checks that an empty dict literal compiles and matches eager.
        def fn():
            # [FUNCTION DOC] HELPER: Returns an empty dict literal.
            return {}

        self.checkScript(fn, ())

    def test_dict_keyword_is_correctly_typed(self):
        # [FUNCTION DOC] Checks that a dict created with the no-argument constructor takes its
        # type from the annotation and accepts a string-keyed insert.
        def fn():
            # [FUNCTION DOC] HELPER: Creates an annotated empty string-to-int dict, sets one key
            # and returns it.
            x: Dict[str, int] = dict()
            x["foo"] = 1
            return x

        self.checkScript(fn, ())

    def test_dict_keyword_with_mismatched_annotations(self):
        # [FUNCTION DOC] Checks that an annotation whose key type (int) conflicts with the actual
        # key type (str) of the constructor argument fails to compile with a descriptive error.
        err_msg = (
            r"Dict type annotation `Dict\[int, str\]` did not "
            "match the type of an actual key type `str`"
        )
        with self.assertRaisesRegex(RuntimeError, err_msg):

            @torch.jit.script
            def fn():
                # [FUNCTION DOC] HELPER: Scripted; annotates a dict as int-to-string but builds it
                # from string/int pairs; compilation is expected to fail.
                x: Dict[int, str] = dict(  # noqa: C406
                    [("foo", 1), ("bar", 2), ("baz", 3)]
                )
                return x

    def test_dict_keyword_with_nested_call(self):
        # [FUNCTION DOC] Checks the dict constructor applied to another constructor call.
        def fn():
            # [FUNCTION DOC] HELPER: Returns the dict constructor applied to a keyword-built dict.
            return dict(dict(foo=1, bar=2, baz=3))

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable(self):
        # [FUNCTION DOC] Checks the dict constructor given a previously declared dict variable.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a local two-key dict and returns a copy made by
            # the constructor.
            d = {"foo": 1, "bar": 2}
            return dict(d)

        self.checkScript(fn, ())

    def test_dict_keyword_with_previously_declared_variable_and_kwargs(self):
        # [FUNCTION DOC] Checks the dict constructor given a dict variable plus a
        # keyword argument.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a local two-key dict and returns the constructor
            # result with one extra keyword key.
            d = {"foo": 1, "bar": 2}
            return dict(d, baz=3)

        self.checkScript(fn, ())

    def test_min_bool_list(self):
        # [FUNCTION DOC] Checks min over two bool lists compares them lexicographically,
        # matching eager.
        def jit_min_list(a: List[bool], b: List[bool]) -> List[bool]:
            # [FUNCTION DOC] HELPER: Returns the smaller of two bool lists.
            return min(a, b)

        self.checkScript(jit_min_list, ([True, False], [False, True]))

    def test_min_max_list(self):
        # [FUNCTION DOC] Checks min and max of two lists (int, float and bool), over several pairs
        # including empty lists and prefix pairs, through checkScript.
        def jit_min_list(a: List[int], b: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Returns the smaller of two int lists.
            return min(a, b)

        def jit_min_list_float(a: List[float], b: List[float]) -> List[float]:
            # [FUNCTION DOC] HELPER: Returns the smaller of two float lists.
            return min(a, b)

        def jit_min_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            # [FUNCTION DOC] HELPER: Returns the smaller of two bool lists.
            return min(a, b)

        def run_tests(func, a, b):
            # [FUNCTION DOC] HELPER: Runs checkScript on the given function once for each pair
            # formed by zipping the two argument lists.
            for t in zip(a, b):
                self.checkScript(func, t)

        args_left_int = [[1, 8, 8], [2, 1, 1], [], [2], [1], [1, 2, 3]]
        args_right_int = [[2, 1, 1], [1, 8, 8], [], [1], [], [1, 2]]
        run_tests(jit_min_list, args_left_int, args_right_int)

        args_left_float = [
            [1.0, 8.0, 8.0],
            [2.0, 1.0, 1.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        args_right_float = [[2.0, 1.0, 1.0], [1.0, 8.0, 8.0], [], [1.0], [], [1.0, 2.0]]
        run_tests(jit_min_list_float, args_left_float, args_right_float)

        args_left_bool = [
            [],
            [],
            [],
            [False],
            [True],
            [False, True],
            [True, True],
            [False, False, False],
            [False, False, True],
        ]
        args_right_bool = [
            [],
            [False],
            [True],
            [True],
            [False],
            [True, True],
            [False, True],
            [False, False, True],
            [False, False, False],
        ]
        run_tests(jit_min_list_bool, args_left_bool, args_right_bool)

        def jit_max_list(a: List[int], b: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Returns the larger of two int lists.
            return max(a, b)

        def jit_max_list_float(a: List[float], b: List[float]) -> List[float]:
            # [FUNCTION DOC] HELPER: Returns the larger of two float lists.
            return max(a, b)

        def jit_max_list_bool(a: List[bool], b: List[bool]) -> List[bool]:
            # [FUNCTION DOC] HELPER: Returns the larger of two bool lists.
            return max(a, b)

        args_left_int = [[1, 8, 8], [8, 1, 1], [], [1], [], [1, 2]]
        args_right_int = [[8, 1, 1], [1, 8, 8], [], [2], [1], [1, 2, 3]]
        run_tests(jit_max_list, args_left_int, args_right_int)

        args_left_float = [[1.0, 8.0, 8.0], [8.0, 1.0, 1.0], [], [1.0], [], [1.0, 2.0]]
        args_right_float = [
            [8.0, 1.0, 1.0],
            [1.0, 8.0, 8.0],
            [],
            [2.0],
            [1.0],
            [1.0, 2.0, 3.0],
        ]
        run_tests(jit_max_list_float, args_left_float, args_right_float)

        run_tests(jit_max_list_bool, args_left_bool, args_right_bool)

    def test_list_gather(self):
        # [FUNCTION DOC] Checks list indexing: positive and negative indices work, and
        # out-of-range positive or negative indices raise an error mentioning list index out of
        # range, in eager and scripted mode.
        def index():
            # [FUNCTION DOC] HELPER: Returns element 1 of a three-element list.
            a = [1, 2, 3]
            return a[1]

        self.checkScript(index, ())

        def negative_index():
            # [FUNCTION DOC] HELPER: Returns the last element using a negative index.
            a = [1, 2, 3]
            return a[-1]

        self.checkScript(negative_index, ())

        def bad_index():
            # [FUNCTION DOC] HELPER: Indexes position 4 of a three-element list (expected
            # to raise).
            a = [1, 2, 3]
            return a[4]

        self.checkScriptRaisesRegex(bad_index, (), Exception, "list index out of range")

        def bad_negative_index():
            # [FUNCTION DOC] HELPER: Indexes position -5 of a three-element list (expected
            # to raise).
            a = [1, 2, 3]
            return a[-5]

        self.checkScriptRaisesRegex(
            bad_negative_index, (), Exception, "list index out of range"
        )

    def test_list_len(self):
        # [FUNCTION DOC] Checks the length of a literal list and of an empty list.
        def func():
            # [FUNCTION DOC] HELPER: Returns whether a three-element list has length three.
            a = [1, 2, 3]
            return len(a) == 3

        self.checkScript(func, ())

        def func2():
            # [FUNCTION DOC] HELPER: Returns whether an empty list has length zero.
            a = []
            return len(a) == 0

        self.checkScript(func2, ())

    @skipIfTorchDynamo(
        "TorchDynamo fails to raise on this checkScriptRaisesRegex, because we trace it properly now"
    )
    def test_list_ops(self):
        # [FUNCTION DOC] Checks list equality, inequality, concatenation and tensor-list
        # comparison through checkScript; the last case expects an error when comparing lists of
        # multi-element tensors. Skipped under TorchDynamo.
        # KNOWN ISSUE: the nested test_non_inequality helper is never run; the line after it
        # re-checks test_non_equality instead.
        def test_equality():
            # [FUNCTION DOC] HELPER: Returns whether two equal int lists compare equal.
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a == b

        self.checkScript(test_equality, (), optimize=True)

        def test_equality_str():
            # [FUNCTION DOC] HELPER: Returns whether two equal string lists compare equal.
            a = ["foo", "bar"]
            b = ["foo", "bar"]
            return a == b

        self.checkScript(test_equality_str, (), optimize=True)

        def test_inequality():
            # [FUNCTION DOC] HELPER: Returns whether two equal int lists compare unequal.
            a = [1, 2, 3]
            b = [1, 2, 3]
            return a != b

        self.checkScript(test_inequality, (), optimize=True)

        def test_inequality_str():
            # [FUNCTION DOC] HELPER: Returns whether a two-string list and a three-string list
            # compare unequal.
            a = ["foo", "bar"]
            b = ["foo", "bar", "food"]
            return a != b

        self.checkScript(test_inequality_str, (), optimize=True)

        def test_non_equality():
            # [FUNCTION DOC] HELPER: Returns whether a three-element list equals a
            # one-element list.
            a = [1, 2, 3]
            b = [3]
            return a == b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_non_inequality():
            # [FUNCTION DOC] HELPER: Returns whether a three-element list differs from a
            # one-element list.
            # KNOWN ISSUE: never executed; the check that follows its definition passes the
            # equality function instead.
            a = [1, 2, 3]
            b = [3]
            return a != b

        self.checkScript(test_non_equality, (), optimize=True)

        def test_list_equality_as_cond():
            # [FUNCTION DOC] HELPER: Uses list equality as an if condition and returns 1 or 2.
            a = [1, 2, 3]
            b = [3]
            if a == b:
                c = 1
            else:
                c = 2
            return c

        self.checkScript(test_list_equality_as_cond, (), optimize=True)

        def test_list_add():
            # [FUNCTION DOC] HELPER: Concatenates two int lists and returns whether the result
            # equals the expected list.
            a = [1, 2, 3]
            b = [2]
            c = a + b
            return c == [1, 2, 3, 2]

        self.checkScript(test_list_add, (), optimize=True)

        def test_list_add_empty():
            # [FUNCTION DOC] HELPER: Concatenates a list with an annotated empty list and compares
            # the result.
            a = [1, 2, 3]
            b = torch.jit.annotate(List[int], [])
            c = a + b
            return c == [1, 2, 3]

        self.checkScript(test_list_add_empty, (), optimize=True)

        def test_tensor_list_equality():
            # [FUNCTION DOC] HELPER: Compares two lists holding the same one-element tensors in
            # swapped order and returns the result.
            t1 = torch.ones([1, 1])
            t2 = torch.ones([1, 1])
            x = [t1, t2]
            y = [t2, t1]
            return x == y

        self.checkScript(test_tensor_list_equality, (), optimize=True)

        def test_invalid_list_equality():
            # [FUNCTION DOC] HELPER: Compares two lists of 2x2 tensors; comparing multi-element
            # tensors as booleans is expected to raise.
            t1 = torch.ones([2, 2])
            t2 = torch.ones([2, 2])
            x = [t1, t2]
            y = [t2, t1]
            # will throw since the tensors have more than one element
            return x == y

        self.checkScriptRaisesRegex(
            test_invalid_list_equality, (), RuntimeError, "Boolean value of Tensor"
        )

    def test_list_sort(self):
        # [FUNCTION DOC] For several list literals (empty, int, bool, float, tensors), executes a
        # generated function in eager Python and in a CompilationUnit; it sorts in place
        # ascending, sorts in place descending and uses the sorted builtin, and results must
        # match. Also checks sorting multi-element tensors raises, that the in-place sort
        # produces the sort op in the graph, and that sorted returns an independent copy.
        template = dedent(
            """
        def func():
            li_1 = {list_create}
            li_2 = {list_create}
            li_3 = {list_create}
            li_1.sort()
            li_2.sort(reverse=True)
            li_4 = sorted(li_3)
            return li_1, li_2, li_3, li_4
        """
        )

        lists = [
            "[]",
            "[1, 3, 2]",
            "[True, False, True]",
            "[1.2, .2, 3.2]",
            "[torch.tensor(1.0), torch.tensor(0.2), torch.tensor(0.5)]",
            "[torch.tensor(5), torch.tensor(-2), torch.tensor(4)]",
        ]
        for li in lists:
            code = template.format(list_create=li)
            scope = {}
            exec(code, globals(), scope)
            cu = torch.jit.CompilationUnit(code)
            t1 = cu.func()
            t2 = scope["func"]()
            self.assertEqual(t1, t2)

        def test_fail(x: List[Tensor]) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Sorts a list of tensors in place and returns it (expected to
            # raise for multi-element tensors).
            x.sort()
            return x

        self.checkScriptRaisesRegex(
            test_fail,
            (([torch.zeros([2]), torch.zeros([2])],)),
            Exception,
            "Boolean value of Tensor with more than one value",
        )

        @torch.jit.script
        def test_mutation():
            # [FUNCTION DOC] HELPER: Scripted; sorts a three-int list in place and returns it.
            a = [1, 2, 3]
            a.sort()
            return a

        test_mutation()
        FileCheck().check("aten::sort").run(test_mutation.graph_for())

        def test_sorted_copy():
            # [FUNCTION DOC] HELPER: Builds a sorted copy of a tensor list, then overwrites the
            # first element of the original; returns both lists.
            a = [torch.tensor(2), torch.tensor(0), torch.tensor(1)]
            b = sorted(a)
            a[0] = torch.tensor(10)
            return a, b

        self.checkScript(test_sorted_copy, ())

    def test_list_slice(self):
        # [FUNCTION DOC] Checks list slicing forms through checkScript.
        # KNOWN ISSUE: the final check passes test_backward_slice again, so test_over_slice is
        # never run.
        def test_regular_slice():
            # [FUNCTION DOC] HELPER: Returns whether a slice from index 2 to 3 equals a
            # one-element list.
            a = [0, 1, 2, 3, 4]
            return a[2:3] == [2]

        self.checkScript(test_regular_slice, ())

        def test_open_ended_slice():
            # [FUNCTION DOC] HELPER: Returns whether a slice from index 2 to the end equals the
            # expected list.
            a = [0, 1, 2, 3, 4]
            return a[2:] == [2, 3, 4]

        self.checkScript(test_open_ended_slice, ())

        def test_open_ended_slice2():
            # [FUNCTION DOC] HELPER: Returns whether a slice up to index 2 equals the first
            # two elements.
            a = [0, 1, 2, 3, 4]
            return a[:2] == [0, 1]

        self.checkScript(test_open_ended_slice2, ())

        def test_negative_slice():
            # [FUNCTION DOC] HELPER: Returns whether a slice dropping the last element equals the
            # expected list.
            a = [0, 1, 2, 3, 4]
            return a[:-1] == [0, 1, 2, 3]

        self.checkScript(test_negative_slice, ())

        def test_negative_slice2():
            # [FUNCTION DOC] HELPER: Returns whether a slice using two negative bounds equals the
            # expected list.
            a = [0, 1, 2, 3, 4]
            return a[-3:-1] == [2, 3]

        self.checkScript(test_negative_slice2, ())

        def test_backward_slice():
            # [FUNCTION DOC] HELPER: Returns whether a slice with start after stop equals an
            # annotated empty list.
            a = [0, 1, 2, 3, 4]
            return a[3:2] == torch.jit.annotate(List[int], [])

        self.checkScript(test_backward_slice, ())

        def test_over_slice():
            # [FUNCTION DOC] HELPER: Returns whether a slice with an over-long stop equals the
            # last two elements. Never executed.
            a = [0, 1, 2, 3, 4]
            return a[3:10] == [3, 4]

        self.checkScript(test_backward_slice, ())

    def test_slice_index(self):
        # [FUNCTION DOC] Checks tensor indexing with list indices and slices on a 3x2x2
        # tensor, plus an out-of-bounds list index that must raise an error highlighting the
        # indexing expression.
        a = torch.tensor(
            [
                [[1, 11], [2, 22]],
                [[3, 33], [4, 44]],
                [[5, 55], [6, 66]],
            ]
        )

        def test_index_slice1(x):
            # [FUNCTION DOC] HELPER: Indexes the last dimension with a list of two positions.
            x = x[:, :, [0, 1]]
            return x

        self.checkScript(test_index_slice1, (a,))

        def test_index_slice2(x):
            # [FUNCTION DOC] HELPER: Indexes the first dimension with a reordering list.
            x = x[[2, 1, 0], :, :]
            return x

        self.checkScript(test_index_slice2, (a,))

        def test_index_slice3(x):
            # [FUNCTION DOC] HELPER: Indexes the first and last dimensions with lists.
            x = x[[0, 1], :, [1]]
            return x

        self.checkScript(test_index_slice3, (a,))

        def test_index_slice_empty_list(x):
            # [FUNCTION DOC] HELPER: Indexes the first dimension with an empty annotated int list.
            empty_list: List[int] = []
            x = x[empty_list, :, :]
            return x

        self.checkScript(test_index_slice_empty_list, (a,))

        # [FUNCTION DOC] HELPER: Indexes the first dimension with the out-of-range position 4 on a
        # size-three dimension.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def test_index_slice_out_of_bounds_index(x):
            x = x[[4], :, :]
            return x

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "index 4 is out of bounds for dimension 0 with size 3",
            "x[[4], :, :]",
        ):
            self.checkScript(test_index_slice_out_of_bounds_index, (a,))

    def test_mutable_list_append(self):
        # [FUNCTION DOC] Checks appending twice to a list literal.
        def test_append():
            # [FUNCTION DOC] HELPER: Appends 2 and 3 to a two-element list and checks the result.
            a = [0, 1]
            a.append(2)
            a.append(3)
            return a == [0, 1, 2, 3]

        self.checkScript(test_append, ())

    def test_comprehensions_basic(self):
        # [FUNCTION DOC] Checks a list comprehension that triples every int, run eagerly
        # and scripted.
        def comp(l: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Returns a comprehension tripling each element.
            n = [x * 3 for x in l]
            return n

        comp([1, 2, 3])
        self.checkScript(comp, ([1, 2, 3],))

    def test_comprehensions_basic_float(self):
        # [FUNCTION DOC] Checks the same comprehension on float lists.
        def comp(l: List[float]) -> List[float]:
            # [FUNCTION DOC] HELPER: Returns a comprehension tripling each float.
            n = [x * 3 for x in l]
            return n

        self.checkScript(comp, ([1.0, 2.0, 3.0],))

    def test_comprehensions_two_comps(self):
        # [FUNCTION DOC] Checks a scripted function with two comprehensions whose results
        # are concatenated.
        @torch.jit.script
        def comp(l1: List[int], l2: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Scripted; triples elements of the first list, adds two to
            # elements of the second, returns the concatenation.
            n = [x * 3 for x in l1]
            n2 = [x + 2 for x in l2]
            return n + n2

        self.assertEqual(comp([1, 2, 3], [4, 5]), [3, 6, 9, 6, 7])

    def test_comprehension_out_type_not_in_type(self):
        # [FUNCTION DOC] Checks a comprehension whose output type (int) differs from the iterated
        # element type (tensor).
        def list_cast() -> int:
            # [FUNCTION DOC] HELPER: Converts three tensors to ints with a comprehension and
            # returns their sum.
            li = [int(i) for i in [torch.tensor(0), torch.tensor(1), torch.tensor(2)]]
            return li[0] + li[1] + li[2]

        self.checkScript(list_cast, ())

    def test_comprehension_iterable(self):
        # [FUNCTION DOC] Checks comprehensions over zip, range, enumerate and an empty zip,
        # comparing eager with scripted results.
        def test_func(fn, inputs):
            # [FUNCTION DOC] HELPER: Asserts the eager result of the function equals the result of
            # the scripted version for the given inputs.
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def foo(names: List[int], results: List[int]) -> List[Tuple[int, int]]:
            # [FUNCTION DOC] HELPER: Returns pairs formed by adding 5 to one list and subtracting
            # 2 from the other, over their zip.
            return [(k + 5, v - 2) for k, v in zip(names, results)]

        test_func(foo, ([1, 2, 4], [4, 7, 9]))
        test_func(foo, ([5], [4, 7, 9]))

        def fn(x: int) -> List[int]:
            # [FUNCTION DOC] HELPER: Returns the range comprehension for the given int.
            return [i for i in range(x)]  # noqa: C416

        test_func(fn, (9,))
        test_func(fn, (0,))
        test_func(fn, (-1,))

        def changes_type():
            # [FUNCTION DOC] HELPER: Returns three comprehensions converting ints to floats, one
            # with enumerate pairs.
            a = [float(i) for i in range(5)]
            b = [float(i) for i in [1, 2, 3, 4]]
            c = [(float(i), j) for i, j in enumerate([1, 2, 3, 8])]
            return a, b, c

        test_func(changes_type, ())

        def test_zero_iter():
            # [FUNCTION DOC] HELPER: Returns a comprehension over a zip of two empty strings (an
            # empty list of strings).
            return [str(i) for i, j in zip("", "")]

        test_func(test_zero_iter, ())

    def test_mutable_list_append_2(self):
        # [FUNCTION DOC] Checks appending after the name is rebound to a fresh list.
        def test_append_2():
            # [FUNCTION DOC] HELPER: Appends to a list, rebinds the name to a new list, appends
            # again and checks the second list.
            a = [0, 1]
            a.append(2)
            a = [1]
            a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_2, ())

    def test_mutable_list_append_if(self):
        # [FUNCTION DOC] Checks appending inside the taken branch of an if.
        def test_append_if():
            # [FUNCTION DOC] HELPER: Conditionally appends inside an if and checks the result.
            a = [1]
            if 1 == 1:
                a.append(4)
            return a == [1, 4]

        self.checkScript(test_append_if, ())

    def test_mutable_list_append_if_else(self):
        # [FUNCTION DOC] Checks appending in the else branch of an if.
        def test_append_if_else():
            # [FUNCTION DOC] HELPER: Appends in the else branch of an if and checks the result.
            a = [1]
            if 1 == 2:
                a.append(4)
            else:
                a.append(10)
            return a == [1, 10]

        self.checkScript(test_append_if_else, ())

    def test_mutable_list_append_loop(self):
        # [FUNCTION DOC] Checks appending inside a loop.
        def test_append_loop():
            # [FUNCTION DOC] HELPER: Appends loop indices into an annotated empty list and checks
            # the result.
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                a.append(i)

            return a == [0, 1, 2, 3, 4]

        self.checkScript(test_append_loop, ())

    def test_mutable_list_append_loop_if(self):
        # [FUNCTION DOC] Checks appending inside a loop with a conditional.
        def test_append_loop_if():
            # [FUNCTION DOC] HELPER: Appends the index or zero depending on a condition inside a
            # loop and checks the result.
            a = torch.jit.annotate(List[int], [])
            for i in range(5):
                if i > 3:
                    a.append(i)
                else:
                    a.append(0)

            return a == [0, 0, 0, 0, 4]

        self.checkScript(test_append_loop_if, ())

    def test_mutable_list_nested_loop(self):
        # [FUNCTION DOC] Checks appending inside nested loops.
        def test_nested_loop():
            # [FUNCTION DOC] HELPER: Appends sums of indices of two nested loops and checks
            # the result.
            a = torch.jit.annotate(List[int], [])
            for i in range(2):
                for j in range(2):
                    a.append(i + j)

            return a == [0, 1, 1, 2]

        self.checkScript(test_nested_loop, ())

    def test_mutable_list_function_inline(self):
        # [FUNCTION DOC] Checks that a scripted function mutating a list argument is visible to
        # the scripted caller.
        @torch.jit.script
        def bar(y: List[int]) -> None:
            # [FUNCTION DOC] HELPER: Scripted; appends 4 to the given list and returns nothing.
            y.append(4)

        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Scripted; builds a three-element list, passes it to the
            # mutating function and returns it.
            x = [1, 2, 3]
            bar(x)
            return x

        self.assertEqual(foo(), [1, 2, 3, 4])

    def test_mutable_list_reverse_empty(self):
        # [FUNCTION DOC] Checks reverse on an empty list.
        def test_reverse_empty():
            # [FUNCTION DOC] HELPER: Reverses an empty list and checks it is still empty.
            a = []
            a.reverse()

            return a == []

        self.checkScript(test_reverse_empty, ())

    def test_mutable_list_reverse(self):
        # [FUNCTION DOC] Checks in-place reverse of an int list.
        def test_reverse():
            # [FUNCTION DOC] HELPER: Reverses a four-element list and checks the order.
            a = [1, 2, 3, 4]
            a.reverse()

            return a == [4, 3, 2, 1]

        self.checkScript(test_reverse, ())

    def test_mutable_tensor_list_reverse(self):
        # [FUNCTION DOC] Checks in-place reverse of a tensor list.
        def test_tensor_reverse():
            # [FUNCTION DOC] HELPER: Reverses a two-tensor list and compares with the
            # swapped list.
            a = [torch.tensor(1), torch.tensor(2)]
            a.reverse()

            return a == [torch.tensor(2), torch.tensor(1)]

        self.checkScript(test_tensor_reverse, ())

    def test_mutable_list_pop_empty(self):
        # [FUNCTION DOC] Checks popping an empty list raises an error highlighting the pop call.
        # [FUNCTION DOC] HELPER: Scripted; pops from an annotated empty int list (raises
        # at runtime).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def test_pop_empty():
            a = torch.jit.annotate(List[int], [])
            return a.pop()

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "pop from empty list", "a.pop"
        ):
            test_pop_empty()

    def test_mutable_list_pop(self):
        # [FUNCTION DOC] Checks pop with no argument returns the last element.
        def test_pop():
            # [FUNCTION DOC] HELPER: Pops the last element of a four-element list and checks it
            # is 4.
            a = [1, 2, 3, 4]
            b = a.pop()

            return b == 4

        self.checkScript(test_pop, ())

    def test_mutable_list_pop2(self):
        # [FUNCTION DOC] Checks pop with no argument shrinks the list.
        def test_pop2():
            # [FUNCTION DOC] HELPER: Pops the last element and checks the remaining length is 3.
            a = [1, 2, 3, 4]
            b = a.pop()

            return len(a) == 3

        self.checkScript(test_pop2, ())

    def test_mutable_list_pop_at(self):
        # [FUNCTION DOC] Checks pop with a positive index returns that element.
        def test_pop_at():
            # [FUNCTION DOC] HELPER: Pops index 1 and checks the popped value is 2.
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return b == 2

        self.checkScript(test_pop_at, ())

    def test_mutable_list_pop_at2(self):
        # [FUNCTION DOC] Checks pop with a positive index shrinks the list.
        def test_pop_at2():
            # [FUNCTION DOC] HELPER: Pops index 1 and checks the remaining length is 3.
            a = [1, 2, 3, 4]
            b = a.pop(1)

            return len(a) == 3

        self.checkScript(test_pop_at2, ())

    def test_mutable_list_pop_at_negative(self):
        # [FUNCTION DOC] Checks pop with a negative index returns the element counted from
        # the end.
        def test_pop_at_negative():
            # [FUNCTION DOC] HELPER: Pops index -2 and checks the popped value is 3.
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return b == 3

        self.checkScript(test_pop_at_negative, ())

    def test_mutable_list_pop_at_negative2(self):
        # [FUNCTION DOC] Checks pop with a negative index shrinks the list.
        def test_pop_at_negative2():
            # [FUNCTION DOC] HELPER: Pops index -2 and checks the remaining length is 3.
            a = [1, 2, 3, 4]
            b = a.pop(-2)

            return len(a) == 3

        self.checkScript(test_pop_at_negative2, ())

    def test_mutable_list_pop_slice(self):
        # [FUNCTION DOC] Checks that popping the last element equals slicing it off.
        def test_pop_slice():
            # [FUNCTION DOC] HELPER: Pops the last element of one list, slices the last element
            # off a copy and returns whether they are equal.
            a = [1, 2, 3, 4]
            b = [1, 2, 3, 4]

            a.pop()
            b = b[:-1]

            return a == b

        self.checkScript(test_pop_slice, ())

    def test_mutable_list_clear_empty(self):
        # [FUNCTION DOC] Checks clear on an empty annotated list.
        def test_clear_empty():
            # [FUNCTION DOC] HELPER: Clears an empty list and returns whether its length is zero.
            a = torch.jit.annotate(List[int], [])
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear_empty, ())

    def test_mutable_list_clear(self):
        # [FUNCTION DOC] Checks clear on a non-empty list.
        def test_clear():
            # [FUNCTION DOC] HELPER: Clears a four-element list and returns whether its length
            # is zero.
            a = [1, 2, 3, 4]
            a.clear()

            return len(a) == 0

        self.checkScript(test_clear, ())

    def test_mutable_list_insert(self):
        # [FUNCTION DOC] Checks insert at an interior index.
        def test_list_insert():
            # [FUNCTION DOC] HELPER: Inserts 5 at index 2 and checks the resulting list.
            a = [1, 2, 3, 4]
            a.insert(2, 5)

            return a == [1, 2, 5, 3, 4]

        self.checkScript(test_list_insert, ())

    def test_mutable_list_insert_negative(self):
        # [FUNCTION DOC] Checks insert at a negative index.
        def test_list_insert_negative():
            # [FUNCTION DOC] HELPER: Inserts 5 at index -1 (before the last element) and checks
            # the result.
            a = [1, 2, 3, 4]
            a.insert(-1, 5)

            return a == [1, 2, 3, 5, 4]

        self.checkScript(test_list_insert_negative, ())

    def test_mutable_list_insert_neg_out_of_bounds(self):
        # [FUNCTION DOC] Checks insert at a negative index beyond the start.
        def test_list_insert_neg_out_of_bounds():
            # [FUNCTION DOC] HELPER: Inserts 5 at index -10 (clamped to the front) and checks
            # the result.
            a = [1, 2, 3, 4]
            a.insert(-10, 5)

            return a == [5, 1, 2, 3, 4]

        self.checkScript(test_list_insert_neg_out_of_bounds, ())

    def test_mutable_list_insert_out_of_bounds(self):
        # [FUNCTION DOC] Checks insert at an index beyond the end.
        def test_list_insert_out_of_bounds():
            # [FUNCTION DOC] HELPER: Inserts 5 at index 10 (clamped to the end) and checks
            # the result.
            a = [1, 2, 3, 4]
            a.insert(10, 5)

            return a == [1, 2, 3, 4, 5]

        self.checkScript(test_list_insert_out_of_bounds, ())

    def test_mutable_list_remove_not_existing(self):
        # [FUNCTION DOC] Checks removing a missing element from a list raises an error
        # highlighting the remove call.
        # [FUNCTION DOC] HELPER: Scripted; removes the absent value 5 from a four-element list
        # (raises at runtime).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def test_list_remove_not_existing():
            a = [1, 2, 3, 4]
            a.remove(5)

            return a

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "x not in list", "a.remove"
        ):
            test_list_remove_not_existing()

    def test_mutable_list_remove(self):
        # [FUNCTION DOC] Checks list remove for an int list and a string list.
        def test_list_remove():
            # [FUNCTION DOC] HELPER: Removes the value 3 from a four-int list and checks the
            # remaining list.
            a = [1, 2, 3, 4]
            a.remove(3)

            return a == [1, 2, 4]

        self.checkScript(test_list_remove, ())

        def test_str_list_remove():
            # [FUNCTION DOC] HELPER: Removes foo from a two-string list and checks the
            # remaining list.
            a = ["foo", "bar"]
            a.remove("foo")

            return a == ["bar"]

        self.checkScript(test_str_list_remove, ())

    def test_list_index_not_existing(self):
        # [FUNCTION DOC] Checks that the index method on a missing value raises an error
        # highlighting the index call.
        # [FUNCTION DOC] HELPER: Scripted; looks up the index of the absent value 5 (raises
        # at runtime).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def list_index_not_existing():
            a = [4, 1, 3, 2]
            i = a.index(5)

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "'5' is not in list", "a.index"
        ):
            list_index_not_existing()

    def test_list_index(self):
        # [FUNCTION DOC] Checks the list index method for int and string lists.
        def list_index():
            # [FUNCTION DOC] HELPER: Returns whether the index of 3 in an int list is 2.
            a = [4, 1, 3, 2]
            i = a.index(3)

            return i == 2

        self.checkScript(list_index, ())

        def list_str_index():
            # [FUNCTION DOC] HELPER: Returns whether the index of bar in a two-string list is 1.
            a = ["foo", "bar"]
            i = a.index("bar")

            return i == 1

        self.checkScript(list_str_index, ())

    def test_tensor_list_index(self):
        # [FUNCTION DOC] Checks the index method on a tensor list.
        def tensor_list_index():
            # [FUNCTION DOC] HELPER: Returns whether the index of a tensor equal to 3 in a tensor
            # list is 2.
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(3))

            return i == 2

        self.checkScript(tensor_list_index, ())

    def test_tensor_list_index_not_existing(self):
        # [FUNCTION DOC] Checks the index method on a tensor list with a missing tensor raises an
        # error highlighting the call.
        # [FUNCTION DOC] HELPER: Scripted; looks up a tensor equal to 5 in a tensor list that
        # lacks it (raises at runtime).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def tensor_list_index_not_existing():
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(3), torch.tensor(2)]
            i = a.index(torch.tensor(5))

            return i

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is not in list", "a.index"
        ):
            tensor_list_index_not_existing()

    def test_list_count(self):
        # [FUNCTION DOC] Checks the list count method for ints and strings.
        def list_count():
            # [FUNCTION DOC] HELPER: Returns whether 4 occurs three times in an int list.
            a = [4, 1, 4, 2, 4]
            i = a.count(4)

            return i == 3

        self.checkScript(list_count, ())

        def list_str_count():
            # [FUNCTION DOC] HELPER: Returns whether foo occurs twice in a string list.
            a = ["foo", "bar", "foo"]
            i = a.count("foo")

            return i == 2

        self.checkScript(list_str_count, ())

    def test_list_count_not_existing(self):
        # [FUNCTION DOC] Checks count of an absent int is zero.
        def list_count_not_existing():
            # [FUNCTION DOC] HELPER: Returns whether the count of 5 in an int list is zero.
            a = [4, 1, 4, 2, 4]
            i = a.count(5)

            return i == 0

        self.checkScript(list_count_not_existing, ())

    def test_tensor_list_count(self):
        # [FUNCTION DOC] Checks count on a tensor list.
        def tensor_list_count():
            # [FUNCTION DOC] HELPER: Returns whether a tensor equal to 4 is counted three times in
            # a tensor list.
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(4))

            return i == 3

        self.checkScript(tensor_list_count, ())

    def test_tensor_list_count_not_existing(self):
        # [FUNCTION DOC] Checks count of an absent tensor is zero.
        def tensor_list_count_not_existing():
            # [FUNCTION DOC] HELPER: Returns whether the count of a tensor equal to 5 is zero.
            a = [torch.tensor(4), torch.tensor(1), torch.tensor(4), torch.tensor(4)]
            i = a.count(torch.tensor(5))

            return i == 0

        self.checkScript(tensor_list_count_not_existing, ())

    def test_mutable_list_remove_tensor(self):
        # [FUNCTION DOC] Checks removing a tensor from a tensor list.
        def test_list_remove_tensor():
            # [FUNCTION DOC] HELPER: Removes the tensor of zeros from a three-tensor list and
            # returns whether two remain.
            a = [torch.ones(1), torch.zeros(1), torch.ones(2)]
            a.remove(torch.zeros(1))

            return len(a) == 2

        self.checkScript(test_list_remove_tensor, ())

    def test_mutable_list_remove2(self):
        # [FUNCTION DOC] Checks removing the only element of a one-element list.
        def test_list_remove2():
            # [FUNCTION DOC] HELPER: Removes 1 from a one-element list and returns whether it
            # is empty.
            a = [1]
            a.remove(1)

            return len(a) == 0

        self.checkScript(test_list_remove2, ())

    def test_extend_list_mutable(self):
        # [FUNCTION DOC] Checks extend on tensor lists for all combinations of empty, one-element
        # and three-element lists; the scripted result must equal list concatenation.
        @torch.jit.script
        def extend_list(a: List[Tensor], b: List[Tensor]) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Scripted; extends the first tensor list with the second and
            # returns the first.
            a.extend(b)
            return a

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            for r in [
                [],
                [torch.rand(2)],
                [torch.rand(2), torch.rand(2), torch.rand(2)],
            ]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_extend_list_immutable(self):
        # [FUNCTION DOC] Same as the tensor version of the extend test but for int lists.
        @torch.jit.script
        def extend_list(a: List[int], b: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Scripted; extends the first int list with the second and
            # returns the first.
            a.extend(b)
            return a

        for l in [[], [1], [1, 2, 3]]:
            for r in [[], [1], [1, 2, 3]]:
                self.assertEqual(extend_list(l, r), l + r)

    def test_copy_list_mutable(self):
        # [FUNCTION DOC] Checks the copy method on tensor lists returns a list equal to the input.
        @torch.jit.script
        def copy_list(a: List[Tensor]) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Scripted; returns a copy of the tensor list.
            return a.copy()

        for l in [[], [torch.rand(2)], [torch.rand(2), torch.rand(2), torch.rand(2)]]:
            self.assertEqual(copy_list(l), l)

    def test_copy_list_immutable(self):
        # [FUNCTION DOC] Checks the copy method on int lists.
        @torch.jit.script
        def copy_list(a: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Scripted; returns a copy of the int list.
            return a.copy()

        for l in [[], [1], [1, 2, 3]]:
            self.assertEqual(copy_list(l), l)

    def test_min_max_single_list(self):
        # [FUNCTION DOC] Checks min and max of a single list for int, bool and float lists; an
        # empty list must raise an error mentioning empty, other lists must match eager.
        def min_intlist(li: List[int]) -> int:
            # [FUNCTION DOC] HELPER: Returns the smallest int of the list.
            return min(li)

        def max_intlist(li: List[int]) -> int:
            # [FUNCTION DOC] HELPER: Returns the largest int of the list.
            return max(li)

        def min_boollist(li: List[bool]) -> bool:
            # [FUNCTION DOC] HELPER: Returns the smallest bool of the list.
            return min(li)

        def max_boollist(li: List[bool]) -> bool:
            # [FUNCTION DOC] HELPER: Returns the largest bool of the list.
            return max(li)

        def min_floatlist(li: List[float]) -> float:
            # [FUNCTION DOC] HELPER: Returns the smallest float of the list.
            return min(li)

        def max_floatlist(li: List[float]) -> float:
            # [FUNCTION DOC] HELPER: Returns the largest float of the list.
            return max(li)

        int_lists = [1], [2, 1, 2], [-3, 4, 2], [-2, -7, 1, 4], [2, 1, 0, 4], []

        def check_list(fn, li):
            # [FUNCTION DOC] HELPER: Checks the function against an error if the list is empty,
            # otherwise compares eager and scripted results.
            if len(li) == 0:
                self.checkScriptRaisesRegex(fn, (li,), Exception, "empty")
            else:
                self.checkScript(fn, (li,))

        for int_list in int_lists:
            check_list(min_intlist, int_list)
            check_list(max_intlist, int_list)

            bool_li = [bool(x) for x in int_list]
            check_list(min_boollist, bool_li)
            check_list(max_boollist, bool_li)

            float_li = [float(x) for x in int_list]
            check_list(min_floatlist, float_li)
            check_list(max_floatlist, float_li)

    def test_to_list(self):
        # [FUNCTION DOC] Unit tests for tolist inside scripted functions, organized by bool, int,
        # float and complex dtypes over 0D to 3D tensors (including non-contiguous and wider
        # float types, which must be converted to the Python scalar types), followed by failure
        # cases: missing result annotation, result annotation of the wrong type, an unsupported
        # element type, a wrong dimension count and an element type that does not match the
        # tensor dtype.
        """Unit tests for Tensor.tolist() function."""

        """
        Boolean dtype unit tests.
        """

        def to_list_bool_0D(x: torch.Tensor) -> bool:
            # [FUNCTION DOC] HELPER: Converts a bool tensor to Python values with the tolist
            # method, annotated to give a Python bool, and returns it.
            li = torch.jit.annotate(bool, x.tolist())
            return li

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            # [FUNCTION DOC] HELPER: Converts a bool tensor to Python values with the tolist
            # method, annotated to give a list of bool, and returns it.
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_bool_2D(x: torch.Tensor) -> List[List[bool]]:
            # [FUNCTION DOC] HELPER: Converts a bool tensor to Python values with the tolist
            # method, annotated to give a nested list of bool, and returns it.
            li = torch.jit.annotate(List[List[bool]], x.tolist())
            return li

        def to_list_bool_3D(x: torch.Tensor) -> List[List[List[bool]]]:
            # [FUNCTION DOC] HELPER: Converts a bool tensor to Python values with the tolist
            # method, annotated to give a three-level nested list of bool, and returns it.
            li = torch.jit.annotate(List[List[List[bool]]], x.tolist())
            return li

        self.checkScript(to_list_bool_0D, (torch.tensor(False, dtype=torch.bool),))
        bool_input_1D = torch.tensor([True, False, True, False], dtype=torch.bool)
        self.checkScript(to_list_bool_1D, (bool_input_1D,))
        bool_input_2D = torch.tensor(
            [[True, True, False], [False, True, False]], dtype=torch.bool
        )
        self.checkScript(to_list_bool_2D, (bool_input_2D,))
        bool_input_3D = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        )
        self.checkScript(to_list_bool_3D, (bool_input_3D,))
        bool_input_noncontiguous = torch.tensor(
            [[[True, False], [False, True]], [[True, False], [False, False]]],
            dtype=torch.bool,
        ).transpose(0, 1)
        self.checkScript(to_list_bool_3D, (bool_input_noncontiguous,))

        """
        Int dtype unit tests.
        """

        def to_list_int_0D(x: torch.Tensor) -> int:
            # [FUNCTION DOC] HELPER: Converts a int tensor to Python values with the tolist
            # method, annotated to give a Python int, and returns it.
            li = torch.jit.annotate(int, x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            # [FUNCTION DOC] HELPER: Converts a int tensor to Python values with the tolist
            # method, annotated to give a list of int, and returns it.
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_int_2D(x: torch.Tensor) -> List[List[int]]:
            # [FUNCTION DOC] HELPER: Converts a int tensor to Python values with the tolist
            # method, annotated to give a nested list of int, and returns it.
            li = torch.jit.annotate(List[List[int]], x.tolist())
            return li

        def to_list_int_3D(x: torch.Tensor) -> List[List[List[int]]]:
            # [FUNCTION DOC] HELPER: Converts a int tensor to Python values with the tolist
            # method, annotated to give a three-level nested list of int, and returns it.
            li = torch.jit.annotate(List[List[List[int]]], x.tolist())
            return li

        self.checkScript(to_list_int_0D, (torch.tensor(1, dtype=torch.long),))
        int_input_1D = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        self.checkScript(to_list_int_1D, (int_input_1D,))
        int_input_2D = torch.tensor([[1, 2, 3], [3, 4, 5]], dtype=torch.long)
        self.checkScript(to_list_int_2D, (int_input_2D,))
        int_input_3D = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        )
        self.checkScript(to_list_int_3D, (int_input_3D,))
        int_input_noncontiguous = torch.tensor(
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=torch.long
        ).transpose(0, 1)
        self.checkScript(to_list_int_3D, (int_input_noncontiguous,))

        """
        Float dtype unit tests.
        """

        def to_list_float_0D(x: torch.Tensor) -> float:
            # [FUNCTION DOC] HELPER: Converts a float tensor to Python values with the tolist
            # method, annotated to give a Python float, and returns it.
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            # [FUNCTION DOC] HELPER: Converts a float tensor to Python values with the tolist
            # method, annotated to give a list of float, and returns it.
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        def to_list_float_2D(x: torch.Tensor) -> List[List[float]]:
            # [FUNCTION DOC] HELPER: Converts a float tensor to Python values with the tolist
            # method, annotated to give a nested list of float, and returns it.
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_float_3D(x: torch.Tensor) -> List[List[List[float]]]:
            # [FUNCTION DOC] HELPER: Converts a float tensor to Python values with the tolist
            # method, annotated to give a three-level nested list of float, and returns it.
            li = torch.jit.annotate(List[List[List[float]]], x.tolist())
            return li

        # Test with torch.float dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.float)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.float),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.float),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float),))
        self.checkScript(
            to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.float).transpose(0, 1),)
        )

        self.checkScript(to_list_float_0D, (torch.randn(5, dtype=torch.double)[0],))
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double),))
        self.checkScript(to_list_float_2D, (torch.randn(5, 6, dtype=torch.double),))
        self.checkScript(to_list_float_3D, (torch.randn(5, 6, 7, dtype=torch.double),))
        self.checkScript(
            to_list_float_3D,
            (torch.randn(5, 6, 7, dtype=torch.double).transpose(0, 1),),
        )

        """
        Complex dtype unit tests.
        """

        def to_list_complex_0D(x: torch.Tensor) -> complex:
            # [FUNCTION DOC] HELPER: Converts a complex tensor to Python values with the tolist
            # method, annotated to give a Python complex, and returns it.
            li = torch.jit.annotate(complex, x.tolist())
            return li

        def to_list_complex_1D(x: torch.Tensor) -> List[complex]:
            # [FUNCTION DOC] HELPER: Converts a complex tensor to Python values with the tolist
            # method, annotated to give a list of complex, and returns it.
            li = torch.jit.annotate(List[complex], x.tolist())
            return li

        def to_list_complex_2D(x: torch.Tensor) -> List[List[complex]]:
            # [FUNCTION DOC] HELPER: Converts a complex tensor to Python values with the tolist
            # method, annotated to give a nested list of complex, and returns it.
            li = torch.jit.annotate(List[List[complex]], x.tolist())
            return li

        def to_list_complex_3D(x: torch.Tensor) -> List[List[List[complex]]]:
            # [FUNCTION DOC] HELPER: Converts a complex tensor to Python values with the tolist
            # method, annotated to give a three-level nested list of complex, and returns it.
            li = torch.jit.annotate(List[List[List[complex]]], x.tolist())
            return li

        # Test with torch.complex dtype Tensors to check that they are converted to double automatically.
        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cfloat)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cfloat),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cfloat),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cfloat),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cfloat).transpose(0, 1),),
        )

        self.checkScript(to_list_complex_0D, (torch.randn(5, dtype=torch.cdouble)[0],))
        self.checkScript(to_list_complex_1D, (torch.randn(5, dtype=torch.cdouble),))
        self.checkScript(to_list_complex_2D, (torch.randn(5, 6, dtype=torch.cdouble),))
        self.checkScript(
            to_list_complex_3D, (torch.randn(5, 6, 7, dtype=torch.cdouble),)
        )
        self.checkScript(
            to_list_complex_3D,
            (torch.randn(5, 6, 7, dtype=torch.cdouble).transpose(0, 1),),
        )

        """
        Non-happy path tests:
            - missing type annotation
            - mismatch between type annotation and input
            - type annotation with unsupported type
            - type annotation with the wrong dimension
            - type annotation with scalar type that doesn't match the input scalar type
        """

        # [FUNCTION DOC] HELPER: Calls the tolist method with no type annotation on the result;
        # compilation is expected to fail.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def to_list_missing_type_annotation(x: torch.Tensor) -> List[float]:
            li = x.tolist()
            return li

        # [FUNCTION DOC] HELPER: Annotates the tolist result as a float although the function
        # declares a list of float as its return type; compilation is expected to fail on the
        # return statement.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def to_list_incorrect_type_annotation(x: torch.Tensor) -> List[float]:
            li = torch.jit.annotate(float, x.tolist())
            return li

        def to_list_unsupported_type_annotation(x: torch.Tensor) -> List[float]:
            # [FUNCTION DOC] HELPER: Annotates the tolist result as a list of strings, an
            # unsupported element type; compilation is expected to fail.
            li = torch.jit.annotate(List[str], x.tolist())
            return li

        def to_list_type_annotation_wrong_dim(x: torch.Tensor) -> List[List[float]]:
            # [FUNCTION DOC] HELPER: Annotates the tolist result as a nested float list while the
            # input is 1D; the dimension mismatch is expected to be reported at runtime.
            li = torch.jit.annotate(List[List[float]], x.tolist())
            return li

        def to_list_type_annotation_incorrect_scalar_type(
            x: torch.Tensor,
        ) -> List[float]:
            # [FUNCTION DOC] HELPER: Annotates the tolist result as a list of float; running it on
            # an integer tensor is expected to fail on the dtype mismatch.
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Expected type hint for result of tolist()", "x.tolist("
        ):
            self.checkScript(to_list_missing_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"Return value was annotated as having type List\[float\] but is actually of type float",
            "return li",
        ):
            self.checkScript(to_list_incorrect_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError, r"str is not one of the supported element types for tolist"
        ):
            self.checkScript(to_list_unsupported_type_annotation, (torch.randn(5),))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation list dimension and runtime tensor dimension must match",
        ):
            self.checkScript(
                to_list_type_annotation_wrong_dim, (torch.randn(5, dtype=torch.double),)
            )

        with self.assertRaisesRegex(
            RuntimeError,
            r"Output annotation element type and runtime tensor element type must match",
        ):
            self.checkScript(
                to_list_type_annotation_incorrect_scalar_type,
                (torch.ones(5, dtype=torch.long),),
            )

    @unittest.skipIf(not TEST_CUDA, "CUDA not available")
    def test_to_list_gpu(self):
        # [FUNCTION DOC] GPU variant of the tolist checks: bool, int and double CUDA tensors
        # converted to 1D lists. Skipped without CUDA.
        """GPU tests for Tensor.tolist() function."""

        def to_list_bool_1D(x: torch.Tensor) -> List[bool]:
            # [FUNCTION DOC] HELPER: Converts a bool tensor to a list of bool.
            li = torch.jit.annotate(List[bool], x.tolist())
            return li

        def to_list_int_1D(x: torch.Tensor) -> List[int]:
            # [FUNCTION DOC] HELPER: Converts an int tensor to a list of int.
            li = torch.jit.annotate(List[int], x.tolist())
            return li

        def to_list_float_1D(x: torch.Tensor) -> List[float]:
            # [FUNCTION DOC] HELPER: Converts a float tensor to a list of float.
            li = torch.jit.annotate(List[float], x.tolist())
            return li

        self.checkScript(
            to_list_bool_1D,
            (torch.tensor([True, False, True, False], dtype=torch.bool).cuda(),),
        )
        self.checkScript(
            to_list_int_1D, (torch.tensor([1, 2, 3, 4], dtype=torch.long).cuda(),)
        )
        self.checkScript(to_list_float_1D, (torch.randn(5, dtype=torch.double).cuda(),))

    def test_no_element_type_annotation(self):
        # [FUNCTION DOC] Checks that using an annotation for a bare list type fails with a
        # contained-type error when defined from source through a CompilationUnit and when
        # scripted directly.
        def fn_with_comment(x: torch.Tensor) -> List:
            # [FUNCTION DOC] HELPER: Annotates a local variable with a bare list type and returns
            # it; used as a compile-failure input (the name suggests a comment form but it is
            # identical to the next function).
            a: List = x.tolist()
            return a

        def annotated_fn(x: torch.Tensor) -> List:
            # [FUNCTION DOC] HELPER: Same signature and body as the previous function; used as a
            # compile-failure input.
            a: List = x.tolist()
            return a

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use List without a contained type"
        ):
            torch.jit.script(annotated_fn)

    def test_list_none(self):
        # [FUNCTION DOC] Checks that constructing a list type object with None as element type
        # raises a RuntimeError.
        with self.assertRaisesRegex(
            RuntimeError, "Can not create ListType with None type"
        ):
            x = torch._C.ListType(None)

    def test_list_unification_hint(self):
        # [FUNCTION DOC] Checks that annotating an int variable with a list literal fails to
        # compile with a message expecting a list annotation.
        with self.assertRaisesRegex(
            RuntimeError, "Expected an annotation of type List"
        ):

            @torch.jit.script
            def x():
                # [FUNCTION DOC] HELPER: Scripted; annotates a variable as int but assigns a
                # two-element list; compilation is expected to fail.
                b: int = [2, 3]
                return b


class TestDict(JitTestCase):
    # [CLASS DOC] JitTestCase suite for TorchScript dict semantics (Dict annotations, keys,
    # values, mutation methods, error cases, OrderedDict). Provides fixture methods that return
    # sample dicts.
    def dict(self):
        # [FUNCTION DOC] HELPER: Returns a dict with string keys a, b and c mapped to one-element
        # tensors holding 1, 2 and 3.
        return {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}

    def dict2(self):
        # [FUNCTION DOC] HELPER: Returns a dict with keys x, y and z mapped to one-element tensors
        # holding 101, 102 and 103.
        return {
            "x": torch.ones(1) + 100,
            "y": torch.ones(1) + 101,
            "z": torch.ones(1) + 102,
        }

    def dict_bool(self):
        # [FUNCTION DOC] HELPER: Returns a dict with a single True key mapped to 1.
        return {True: 1}

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_bool_conversion(self):
        # [FUNCTION DOC] Checks that a dict used as a condition (if, while, ternary) is true
        # exactly when non-empty. Skipped under TorchDynamo.
        def if_predicate(d: Dict[int, int]):
            # [FUNCTION DOC] HELPER: Sums keys and values over items and returns the pair when the
            # dict is non-empty, else the pair of minus ones.
            if d:
                s, t = 0, 0
                for k, v in d.items():
                    s += k
                    t += v

                return s, t
            else:
                return -1, -1

        self.checkScript(if_predicate, ({1: 2, 3: 5},))
        self.checkScript(if_predicate, ({},))

        def while_predicate(d: Dict[int, int]):
            # [FUNCTION DOC] HELPER: Clears the dict while it is truthy; returns nothing.
            while d:
                d.clear()

        self.checkScript(while_predicate, ({1: 2, 3: 5},))
        self.checkScript(while_predicate, ({},))

        def ternary_predicate(d: Dict[int, int]):
            # [FUNCTION DOC] HELPER: Returns non-empty or empty depending on the dict truthiness.
            return "non-empty" if d else "empty"

        self.checkScript(ternary_predicate, ({1: 2, 3: 5},))
        self.checkScript(ternary_predicate, ({},))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_del(self):
        # [FUNCTION DOC] Checks deleting a dict key works the same eagerly and scripted (via
        # CompilationUnit and torch.jit.script), and that deleting a missing key raises an error
        # highlighting the deletion. Skipped under TorchDynamo.
        def inputs():
            # [FUNCTION DOC] HELPER: Returns a fresh dict with keys hi and bye, because the test
            # mutates its input.
            return {"hi": 2, "bye": 3}

        # [FUNCTION DOC] HELPER: Deletes the key hi from the dict and returns the dict.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def fn(x: Dict[str, int]) -> Dict[str, int]:
            del x["hi"]
            return x

        python_out = fn(inputs())
        # checkScript reuses the same object, but here it's being mutated so do
        # it manually
        cu = torch.jit.CompilationUnit()
        cu.define(dedent(inspect.getsource(fn)))
        self.assertEqual(cu.fn(inputs()), python_out)
        self.assertEqual(torch.jit.script(fn)(inputs()), python_out)
        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", 'x["hi"]'):
            self.checkScript(fn, [{}])

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_variance(self):
        # [FUNCTION DOC] Checks dict value-type invariance: dict literals and comprehensions take
        # their type from the lhs annotation (optional int values), while assigning an existing
        # dict of ints, or nested dicts, to a variable annotated with optional values fails with
        # a type-mismatch message. Skipped under TorchDynamo.
        """
        `Dict[T1, _]` is not a subtype of `Dict[T2, _]`, even if `T1` is
        a subtype of `T2`; similarly `Dict[_, T1]` would not be a
        subtype of `Dict[_, T2]`.

        However, if we have a temporary dict object (that is, a dict
        comprehension or a dict literal) on the rhs of an assignment
        statement, we want to ignore the inferred type of the rhs if we
        can prove that: 1) both the lhs and the rhs are dicts with the
        same key types (TorchScript has a restricted set of allowed key
        types, so we don't need to worry about subtyping relationships
        here), and 2) the value type of the dict is a subtype of the
        value type of the rhs dict.
        """

        def test_dictliteral_is_typed_from_annotation():
            # [FUNCTION DOC] HELPER: Returns a dict literal of None values annotated with optional
            # int values.
            x: Dict[str, Optional[int]] = {"foo": None, "bar": None, "baz": None}
            return x

        self.checkScript(test_dictliteral_is_typed_from_annotation, ())

        def test_dictcomprehension_is_typed_from_annotation():
            # [FUNCTION DOC] HELPER: Returns a dict comprehension of None values over three words,
            # annotated with optional int values.
            metasyntactics = ["foo", "bar", "baz"]
            x: Dict[str, Optional[int]] = {  # noqa: C420, RUF025
                word: None for word in metasyntactics
            }
            return x

        self.checkScript(test_dictcomprehension_is_typed_from_annotation, ())

        def test_dicts_with_different_value_types_are_invariant(self):
            # [FUNCTION DOC] HELPER: Takes an unused self argument; assigns a dict of ints to a
            # variable annotated with optional values; scripting must raise the mismatch error.
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            y: Dict[str, Optional[int]] = x
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'y' is "
            "annotated with type "
            r"Dict\[str, Optional\[int\]\] but "
            "is being assigned to a value of "
            r"type Dict\[str, int\]",
        ):
            torch.jit.script(test_dicts_with_different_value_types_are_invariant)

        def test_dicts_with_different_value_types_are_invariant_recursive(self):
            # [FUNCTION DOC] HELPER: Takes an unused self argument; assigns a dict of dicts of
            # ints to a variable annotated with optional values; scripting must raise the
            # mismatch error for variable z.
            x: Dict[str, int] = {"foo": 1, "bar": 2, "baz": 3}
            y: Dict[str, Dict[str, int]] = {"foo": x, "bar": x, "baz": x}
            z: Dict[str, Dict[str, Optional[int]]] = y
            return x

        with self.assertRaisesRegex(
            RuntimeError,
            "Variable 'z' is "
            "annotated with type "
            r"Dict\[str, Dict\[str, Optional"
            r"\[int\]\]\] but is being assigned"
            r" to a value of type Dict\[str, "
            r"Dict\[str, int\]\]",
        ):
            torch.jit.script(
                test_dicts_with_different_value_types_are_invariant_recursive
            )

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_keys(self):
        # [FUNCTION DOC] Checks the keys method: the scripted key list matches the eager key
        # set, and a key view from a dict literal can be appended to like a list. Skipped
        # under TorchDynamo.
        @torch.jit.script
        def keys(x: Dict[str, Tensor]) -> List[str]:
            # [FUNCTION DOC] HELPER: Scripted; returns the keys of the dict as a list.
            return list(x.keys())

        self.assertEqual(set(keys(self.dict())), set(self.dict().keys()))

        @torch.jit.script
        def specialized_list():
            # [FUNCTION DOC] HELPER: Scripted; builds a key list from a two-key dict literal,
            # appends 3 and returns it.
            li = {1: 1, 2: 2}.keys()
            li.append(3)
            return li

        self.assertTrue(set(specialized_list()) == {1, 2, 3})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_values(self):
        # [FUNCTION DOC] Checks the values method of a scripted function against eager values.
        # Skipped under TorchDynamo.
        @torch.jit.script
        def values(x: Dict[str, Tensor]) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Scripted; returns the values of the dict as a list.
            return list(x.values())

        the_dict = self.dict()
        self.assertEqual(set(values(the_dict)), set(the_dict.values()))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_len(self):
        # [FUNCTION DOC] Checks the length of a dict. Skipped under TorchDynamo.
        def length(x: Dict[str, Tensor]) -> int:
            # [FUNCTION DOC] HELPER: Returns the number of entries.
            return len(x)

        self.checkScript(length, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_copy(self):
        # [FUNCTION DOC] Checks the copy method of a dict. Skipped under TorchDynamo.
        def func(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # [FUNCTION DOC] HELPER: Returns a copy of the dict.
            return x.copy()

        self.checkScript(func, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_items(self):
        # [FUNCTION DOC] Checks the items method by scripting directly and comparing the item
        # lists regardless of order, since dict ordering from eager Python is arbitrary. Skipped
        # under TorchDynamo.
        def func(x: Dict[str, Tensor]) -> List[Tuple[str, Tensor]]:
            # [FUNCTION DOC] HELPER: Returns the items of the dict.
            return x.items()

        # The value returned by Python is in arbitrary order, so we can't use
        # checkScript
        scripted_func = torch.jit.script(func)

        eager_out = func(self.dict())
        script_out = scripted_func(self.dict())

        self.assertEqual(len(eager_out), len(script_out))
        for item in eager_out:
            self.assertTrue(item in script_out)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_pop(self):
        # [FUNCTION DOC] Checks dict pop with and without a default, and that popping a
        # missing key without a default raises an error highlighting the pop call. Skipped
        # under TorchDynamo.
        # [FUNCTION DOC] HELPER: Pops the given key from the dict and returns the popped value
        # together with the dict.
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def pop(x: Dict[str, Tensor], key: str) -> Tuple[Tensor, Dict[str, Tensor]]:
            return x.pop(key), x

        # checkScript doesn't copy the inputs, so we can't use it since this mutates
        # the dict
        def tester(fn, *args):
            # [FUNCTION DOC] HELPER: Runs the function eagerly and scripted on fresh dicts
            # (checkScript is avoided because the function mutates its input) and asserts
            # equal results.
            eager_out = fn(self.dict(), *args)
            script_out = torch.jit.script(fn)(self.dict(), *args)
            self.assertEqual(eager_out, script_out)

        tester(pop, "a")

        with self.assertRaisesRegexWithHighlight(RuntimeError, "KeyError", "x.pop"):
            torch.jit.script(pop)(self.dict(), "x")

        def default_pop(
            x: Dict[str, Tensor], key: str, default: Tensor
        ) -> Tuple[Tensor, Dict[str, Tensor]]:
            # [FUNCTION DOC] HELPER: Pops the key with a default value and returns the popped
            # value together with the dict.
            return x.pop(key, default), x

        tester(default_pop, "a", torch.randn(2, 2))
        tester(default_pop, "x", torch.randn(2, 2))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_setdefault(self):
        # [FUNCTION DOC] Checks setdefault for an existing key and a new key. Skipped
        # under TorchDynamo.
        def setdefault(
            x: Dict[str, Tensor], key: str, default: Tensor
        ) -> Dict[str, Tensor]:
            # [FUNCTION DOC] HELPER: Calls setdefault with the key and default and returns
            # the dict.
            x.setdefault(key, default)
            return x

        self.checkScript(setdefault, (self.dict(), "a", torch.randn(2, 2)))
        self.checkScript(setdefault, (self.dict(), "nonexistent", torch.randn(2, 2)))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update(self):
        # [FUNCTION DOC] Checks updating a dict from another dict with the same and with different
        # keys. Skipped under TorchDynamo.
        def update(
            a: Dict[str, Tensor], b: Dict[str, Tensor]
        ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
            # [FUNCTION DOC] HELPER: Updates the first dict with the second and returns both.
            a.update(b)
            return a, b

        self.checkScript(update, (self.dict(), self.dict()))
        self.checkScript(update, (self.dict(), self.dict2()))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_update_existing_key(self):
        # [FUNCTION DOC] Checks repeated updates overwriting the same key. Skipped
        # under TorchDynamo.
        def foo() -> Dict[str, int]:
            # [FUNCTION DOC] HELPER: Updates an empty annotated dict with the key a and the loop
            # index three times and returns it.
            a: Dict[str, int] = {}
            for i in range(3):
                a.update({"a": i})
            return a

        self.checkScript(foo, ())

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_aug_assign(self):
        # [FUNCTION DOC] Checks augmented assignment on dict items for tensor and float values.
        # Skipped under TorchDynamo.
        def aug_assign_dict_tensor(a: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # [FUNCTION DOC] HELPER: Applies in-place add, subtract, multiply, divide and modulo
            # to tensor entries and returns the dict.
            a["a"] += 1
            a["b"] -= 12
            a["c"] *= 122
            a["c"] /= 2
            a["c"] %= 2
            return a

        def aug_assign_dict_prim(a: Dict[str, float]) -> Dict[str, float]:
            # [FUNCTION DOC] HELPER: Applies add, subtract, multiply, divide and modulo with float
            # constants to entries and returns the dict.
            a["a"] += 3.4
            a["b"] -= 2.4
            a["c"] *= 3.0
            a["c"] /= 2.0
            a["c"] %= 2.0
            return a

        self.checkScript(aug_assign_dict_tensor, (self.dict(),))
        self.checkScript(aug_assign_dict_prim, ({"a": 3.0, "b": 2.0, "c": 4.0},))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_popitem(self):
        # [FUNCTION DOC] Checks popitem of a scripted function: one item is removed and the result
        # has the right types (a string key and tensor value); order is not compared. Skipped
        # under TorchDynamo.
        @torch.jit.script
        def popitem(
            x: Dict[str, Tensor],
        ) -> Tuple[Tuple[str, Tensor], Dict[str, Tensor]]:
            # [FUNCTION DOC] HELPER: Scripted; pops an item from the dict and returns it with
            # the dict.
            item = x.popitem()
            return item, x

        # The value returned by Python is arbitrary, so we can't use checkScript
        eager_in = self.dict()
        eager_out = (eager_in.popitem(), eager_in)

        script_out = popitem(self.dict())

        # Check that an item was removed
        self.assertEqual(len(eager_out[1]), len(script_out[1]))

        # Check that the item is the correct types
        self.assertTrue(isinstance(script_out[0][0], str))
        self.assertTrue(isinstance(script_out[0][1], torch.Tensor))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_clear(self):
        # [FUNCTION DOC] Checks the clear method. Skipped under TorchDynamo.
        def clear(x: Dict[str, Tensor]) -> Dict[str, Tensor]:
            # [FUNCTION DOC] HELPER: Clears the dict and returns it.
            x.clear()
            return x

        self.checkScript(clear, (self.dict(),))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_get(self):
        # [FUNCTION DOC] Checks get with a present and an absent key. Skipped under TorchDynamo.
        # KNOWN ISSUE: the get-with-default helper is defined but never run; the checks after it
        # repeat the first helper.
        def get(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            # [FUNCTION DOC] HELPER: Returns the value for the key or None.
            return x.get(key)

        self.checkScript(get, (self.dict(), "a"))
        self.checkScript(get, (self.dict(), "doesn't exist"))

        def get_default(x: Dict[str, Tensor], key: str) -> Optional[Tensor]:
            # [FUNCTION DOC] HELPER: Returns the value for the key or a random tensor default.
            # Never executed.
            return x.get(key, torch.randn(2, 2))

        self.checkScript(get, (self.dict(), "a"))
        self.checkScript(get, (self.dict(), "doesn't exist"))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_get_boolkey(self):
        # [FUNCTION DOC] Checks get with a bool key, with and without a default. Skipped
        # under TorchDynamo.
        def get(x: Dict[bool, int], key: bool) -> Optional[int]:
            # [FUNCTION DOC] HELPER: Returns the value for the bool key or None.
            return x.get(key)

        self.checkScript(get, (self.dict_bool(), True))
        self.checkScript(get, (self.dict_bool(), False))

        def get_default(x: Dict[bool, int], key: bool) -> int:
            # [FUNCTION DOC] HELPER: Returns the value for the bool key or 42.
            return x.get(key, 42)

        self.checkScript(get_default, (self.dict_bool(), True))
        self.checkScript(get_default, (self.dict_bool(), False))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_basic(self):
        # [FUNCTION DOC] Checks basic dict behavior: identity, indexing, an empty typed dict, a
        # missing key raising an error highlighting the index, annotated literals via
        # CompilationUnit and a list of dicts. Skipped under TorchDynamo.
        def simple(x: Dict[str, int]) -> Dict[str, int]:
            # [FUNCTION DOC] HELPER: Returns the dict unchanged.
            return x

        self.checkScript(simple, ({"item": 20, "other_item": 120},))

        def index(x: Dict[str, int]) -> int:
            # [FUNCTION DOC] HELPER: Returns the entry with key item.
            return x["item"]

        self.checkScript(index, ({"item": 20, "other_item": 120},))

        def type_default() -> Dict[str, Tensor]:
            # [FUNCTION DOC] HELPER: Returns an empty dict typed as string to tensor.
            return {}

        self.checkScript(type_default, ())

        # [FUNCTION DOC] HELPER: Scripted; indexes a key that is not present (raises at runtime).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        @torch.jit.script
        def missing_index(x: Dict[str, int]) -> int:
            return x["dne"]  # codespell:ignore

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "KeyError",
            'x["dne"',  # codespell:ignore
        ):
            missing_index({"item": 20, "other_item": 120})

        code = dedent(
            """
            def literal1():
                return torch.jit.annotate(Dict[int, float], {})
            def literal2():
                return torch.jit.annotate(Dict[int, float], {10: 1.2})
        """
        )
        cu = torch.jit.CompilationUnit(code)
        self.assertEqual({}, cu.literal1())
        self.assertEqual({10: 1.2}, cu.literal2())

        cu = torch.jit.CompilationUnit(
            dedent(
                """
            def literal3():
                return torch.jit.annotate(Dict[int, float], {10: 1.2, 11: 1.3})
        """
            )
        )
        self.assertEqual({10: 1.2, 11: 1.3}, cu.literal3())

        def list_of_dicts() -> List[Dict[str, Tensor]]:
            # [FUNCTION DOC] HELPER: Returns a list of two dicts of string keys to tensors.
            return [{"word": torch.ones(2) + 3}, {"other word": torch.ones(1) + 2}]

        self.checkScript(list_of_dicts, ())

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_mutability(self):
        # [FUNCTION DOC] Checks that a scripted function can mutate a dict it creates. Skipped
        # under TorchDynamo.
        @torch.jit.script
        def fn() -> Dict[str, int]:
            # [FUNCTION DOC] HELPER: Scripted; creates an annotated empty dict, sets the key ok to
            # 10 and returns it.
            a = torch.jit.annotate(Dict[str, int], {})
            a["ok"] = 10
            return a

        self.assertEqual(fn(), {"ok": 10})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_key_type(self):
        # [FUNCTION DOC] Checks that indexing a string-keyed dict with None fails to compile with
        # an error highlighting the index. Skipped under TorchDynamo.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "but instead found type", "a[None]"
        ):

            # [FUNCTION DOC] HELPER: Scripted; indexes a dict argument with None (expected
            # compile failure).
            # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the
            # body because this scripted function is used by a test that asserts on highlighted
            # source text of the error, and a body comment could appear in that quoted source.
            @torch.jit.script
            def fn(a: Dict[str, int]) -> int:
                return a[None]

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_loop(self):
        # [FUNCTION DOC] Checks that assigning to a dict key in a scripted loop keeps the last
        # value. Skipped under TorchDynamo.
        @torch.jit.script
        def fn(x: int) -> Dict[str, int]:
            # [FUNCTION DOC] HELPER: Scripted; sets the key ok to each loop index and returns
            # the dict.
            a = torch.jit.annotate(Dict[str, int], {})
            for i in range(x):
                a["ok"] = i
            return a

        self.assertEqual(fn(10), {"ok": 9})

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_view(self):
        # [FUNCTION DOC] Checks that a value fetched from a dict aliases the stored tensor, so
        # in-place changes are visible. Skipped under TorchDynamo.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Stores x in a dict, fetches it as a view, compares x + x
            # before and after adding y in place and returns the equality.
            l = {"a": x}
            x_view = l["a"]
            a = x + x
            x_view.add_(y)
            b = x + x
            return a == b

        self.checkScript(fn, (torch.rand(2, 3), torch.rand(2, 3)))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_membership(self):
        # [FUNCTION DOC] Checks get on int dicts (present and absent keys), and that returning an
        # optional where an int is declared fails to compile with an error highlighting the
        # return. Skipped under TorchDynamo.
        # KNOWN ISSUE: the optional helper is never run; the later checks repeat the first helper.
        def fn(x: Dict[int, int], y: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the value for the key or 3.
            return x.get(y, 3)

        d = {1: 2, 3: 4}
        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))

        def optional(x: Dict[int, int], y: int) -> bool:
            # [FUNCTION DOC] HELPER: Returns whether get for the key is None. Never executed.
            res = x.get(y)
            return res is None

        self.checkScript(fn, (d, 3))
        self.checkScript(fn, (d, 2))

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "is actually of type Optional", "return x.get(y"
        ):

            # [FUNCTION DOC] HELPER: Scripted; returns the optional result of get where an int is
            # declared (expected compile failure).
            # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the
            # body because this scripted function is used by a test that asserts on highlighted
            # source text of the error, and a body comment could appear in that quoted source.
            @torch.jit.script
            def bad_types(x: Dict[int, int], y: int) -> int:
                return x.get(y)  # noqa: T484

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_to_python(self):
        # [FUNCTION DOC] Checks a scripted function calling an ignored Python function that does a
        # dict lookup over a key list. Skipped under TorchDynamo.
        @torch.jit.ignore
        def python_lookup(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            # [FUNCTION DOC] HELPER: Ignored by TorchScript; looks up each key in the dict and
            # returns the values in a list.
            return [my_dict[k] for k in keys]

        def fn(my_dict: Dict[str, int], keys: List[str]) -> List[int]:
            # [FUNCTION DOC] HELPER: Calls the Python lookup function with the dict and keys and
            # returns the result.
            return python_lookup(my_dict, keys)

        a_dict = {"a": torch.ones(1), "b": torch.ones(1) + 1, "c": torch.ones(1) + 2}
        self.checkScript(fn, (a_dict, ("a", "c")))

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_ordered_dict(self):
        # [FUNCTION DOC] Checks OrderedDict construction and the dict constructor with repeated
        # keys, insertion order of iteration, and that assigning an int key to an untyped
        # empty dict fails to compile with an error highlighting the assignment. Skipped
        # under TorchDynamo.
        def test_func(fn, inputs):
            # [FUNCTION DOC] HELPER: Asserts the eager result of the function equals the
            # scripted result.
            self.assertEqual(fn(*inputs), torch.jit.script(fn)(*inputs))

        def repeated_key():
            # [FUNCTION DOC] HELPER: Returns an ordered dict built from pairs that repeat the
            # key 1.
            return OrderedDict([(1, 2), (2, 3), (1, 4)])

        test_func(repeated_key, ())

        def no_args():
            # [FUNCTION DOC] HELPER: Builds an ordered dict with two string keys and
            # returns nothing.
            a = OrderedDict()
            a["one"] = torch.tensor(1)
            a["two"] = torch.tensor(2)

        test_func(no_args, ())

        def test_dict_constructor():
            # [FUNCTION DOC] HELPER: Creates a dict with the constructor, sets one key and returns
            # it with a second dict built from repeated-key pairs.
            a = dict()
            a["one"] = torch.tensor(1)
            return a, dict([(1, 2), (2, 3), (1, 4)])  # noqa: C406

        test_func(test_dict_constructor, ())

        def test_dict_initializer_list():
            # [FUNCTION DOC] HELPER: Builds a two-entry dict and returns its values in
            # iteration order.
            a = {"1": torch.tensor(1), "2": torch.tensor(2)}
            output_order = []
            for key in a:
                output_order.append(a[key])
            return output_order

        test_func(test_dict_initializer_list, ())

        # [FUNCTION DOC] HELPER: Creates an untyped dict, assigns an int key and value, returns it
        # (expected compile failure).
        # KNOWN ISSUE: doc block placed above the def (and decorators) instead of inside the body
        # because this scripted function is used by a test that asserts on highlighted source
        # text of the error, and a body comment could appear in that quoted source.
        def test_dict_error():
            a = dict()
            a[1] = 2
            return a

        with self.assertRaisesRegexWithHighlight(
            Exception, "Arguments for call are not", "a[1] = 2"
        ):
            torch.jit.script(test_dict_error)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_type_annotation_missing_contained_type(self):
        # [FUNCTION DOC] Checks a bare dict annotation fails with a missing-contained-types error
        # whether defined from source or scripted. Skipped under TorchDynamo.
        """
        Test that the use of a Dict type annotation without contained
        key and value types produces an error.
        """

        # This function uses a type comment.
        def fn_with_comment(input: Dict) -> Any:
            # [FUNCTION DOC] HELPER: Declares a bare dict argument and returns it;
            # compile-failure input.
            # KNOWN ISSUE: the leading comment says it uses a type comment, but it uses the same
            # annotations as the next function.
            return input

        # This function uses Python3 style type annotations.
        def annotated_fn(input: Dict) -> Any:
            # [FUNCTION DOC] HELPER: Declares a bare dict argument and returns it;
            # compile-failure input.
            return input

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(fn_with_comment)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            cu = torch.jit.CompilationUnit()
            cu.define(dedent(inspect.getsource(annotated_fn)))

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(fn_with_comment)

        with self.assertRaisesRegex(
            RuntimeError, r"Attempted to use Dict without contained types"
        ):
            m = torch.jit.script(annotated_fn)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_dict_preserves_order(self):
        # [FUNCTION DOC] Checks a scripted dict preserves insertion order for one thousand
        # entries. Skipped under TorchDynamo.
        def dict_ordering():
            # [FUNCTION DOC] HELPER: Fills an annotated dict with a thousand increasing keys and
            # returns it.
            a: Dict[int, int] = {}
            for i in range(1000):
                a[i] = i + 1
            return a

        self.checkScript(dict_ordering, ())
        di = torch.jit.script(dict_ordering)()
        res = list(di.items())
        for i in range(1000):
            key, value = res[i]
            self.assertTrue(key == i and value == i + 1)

    @skipIfTorchDynamo("TorchDynamo fails for this test for unknown reason")
    def test_optional_dict_construct(self):
        # [FUNCTION DOC] Checks a module that builds a dict of optional tensors and passes it to a
        # method, using checkModule. Skipped under TorchDynamo.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a method taking a dict of optional tensors and a
            # forward that builds one.
            def use(self, buffer: Dict[str, Optional[torch.Tensor]]):
                # [FUNCTION DOC] HELPER: Returns the entry with key prev_key.
                return buffer["prev_key"]

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Creates two random tensors, stores them in an annotated
                # dict and returns the lookup performed by the helper method.
                prev_key = torch.rand(2, 3)
                next_key = torch.rand(2, 3)
                saved_state: Dict[str, Optional[torch.Tensor]] = {
                    "prev_key": prev_key,
                    "next_key": next_key,
                }

                return self.use(saved_state)

        self.checkModule(M(), (torch.rand(2, 2),))


class TestNamedTuple(JitTestCase):
    # [CLASS DOC] JitTestCase suite for NamedTuple support in TorchScript: construction,
    # constants, returns, attributes of modules, unpacking and slicing, keyword construction,
    # forward-reference field types and name resolution.
    def test_namedtuple(self):
        # [FUNCTION DOC] Checks a scripted function constructs a NamedTuple, iterates its list
        # field and combines fields (result 18.0).
        class FeatureVector(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with float_features (float), sequence_features (list
            # of float) and time_since_first (float).
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x) -> float:
            # [FUNCTION DOC] HELPER: Scripted; builds the NamedTuple with 3.0, [3.0], 3.0,
            # adds the list items to the float field and multiplies by the last field;
            # returns the float.
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv

        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_constant(self):
        # [FUNCTION DOC] Checks a scripted function can return a NamedTuple constructed from
        # constants and it compares equal to the Python one.
        class Tup(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with two int fields a and b.
            a: int
            b: int

        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Scripted; returns the NamedTuple with values 1 and 2.
            return Tup(1, 2)

        self.assertEqual(foo(), Tup(1, 2))

    def test_return_named_tuple(self):
        # [FUNCTION DOC] Checks a scripted function returning a NamedTuple exposes its fields to
        # Python (run twice, fields 3.0, [3.0], 3.0).
        class FeatureVector(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with float_features (float), sequence_features (list
            # of float) and time_since_first (float).
            float_features: float
            sequence_features: List[float]
            time_since_first: float

        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted; ignores x and returns a NamedTuple built with 3.0,
            # [3.0], 3.0.
            fv = FeatureVector(3.0, [3.0], 3.0)
            return fv

        out = foo(torch.rand(3, 4))
        out = foo(torch.rand(3, 4))
        self.assertEqual(out.float_features, 3.0)
        self.assertEqual(out.sequence_features, [3.0])
        self.assertEqual(out.time_since_first, 3.0)

    def test_namedtuple_as_attr(self):
        # [FUNCTION DOC] Checks that a module holding a dict of NamedTuple values as an annotated
        # attribute can be scripted.
        class Config(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a single int field size.
            size: int

        class MyMod(nn.Module):
            # [CLASS DOC] HELPER: Module with a class-level annotation for a dict from int to the
            # config NamedTuple, stored from its constructor.
            configs: Dict[int, Config]

            def __init__(self, configs):
                # [FUNCTION DOC] HELPER: Calls the base constructor and stores the given configs
                # dict as an attribute.
                super().__init__()
                self.configs = configs

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the size of every stored config to x in place and
                # returns x.
                for config in self.configs.values():
                    x += config.size
                return x

        s = torch.jit.script(MyMod({0: Config(size=16)}))

    def test_namedtuple_resolution(self):
        # [FUNCTION DOC] Checks that a NamedTuple type reached by attribute access on a custom
        # module-like object resolves in a scripted return annotation.
        class TheType(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with one int field t.
            t: int

        class MyModule(types.ModuleType):
            # [CLASS DOC] HELPER: ModuleType subclass named MyModule whose attribute lookup
            # returns the NamedTuple type for any name.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module type with the name MyModule.
                super().__init__("MyModule")

            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: Returns the NamedTuple type regardless of the
                # attribute requested.
                return TheType

        some_module = MyModule()

        def fn() -> some_module.Type:
            # [FUNCTION DOC] HELPER: Declared to return the NamedTuple type resolved through the
            # module-like object and constructs it with 1.
            return some_module.Type(1)

        self.checkScript(fn, [])

    def test_namedtuple_slice_unpack(self):
        # [FUNCTION DOC] Checks tuple unpacking and slicing of a NamedTuple in a
        # scripted function.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c.
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int, b: float, c: List[int]):
            # [FUNCTION DOC] HELPER: Scripted; builds the NamedTuple, unpacks it into three values
            # and returns the first-element slice with the first and third values.
            tup = MyCoolNamedTuple(a, b, c)
            my_a, my_b, my_c = tup
            return tup[:1], my_a, my_c

        self.assertEqual(foo(3, 3.5, [6]), ((3,), 3, [6]))

    def test_namedtuple_lower(self):
        # [FUNCTION DOC] Checks that the graph contains a tuple construct for a NamedTuple and
        # that the tuple-lowering pass removes it.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c.
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(a: int):
            # [FUNCTION DOC] HELPER: Scripted; constructs and returns the NamedTuple with the
            # given int, 3.14 and a one-element list.
            tup = MyCoolNamedTuple(a, 3.14, [9])
            return tup

        FileCheck().check("TupleConstruct").run(foo.graph)
        torch._C._jit_pass_lower_all_tuples(foo.graph)
        FileCheck().check_not("TupleConstruct").run(foo.graph)

    def test_namedtuple_type_annotation(self):
        # [FUNCTION DOC] Checks a NamedTuple type declared as a global (so scripting can resolve
        # it) works as argument and return annotation.
        global MyCoolNamedTuple  # see [local resolution in python]

        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c, made global by
            # the enclosing test.
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo(x: MyCoolNamedTuple) -> MyCoolNamedTuple:
            # [FUNCTION DOC] HELPER: Scripted identity function on the NamedTuple.
            return x

        mnt = MyCoolNamedTuple(42, 420.0, [666])
        self.assertEqual(foo(mnt), mnt)

    def test_namedtuple_wrong_types(self):
        # [FUNCTION DOC] Checks that constructing a NamedTuple with strings for typed fields fails
        # to compile with an argument-type error.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c.
            a: int
            b: float
            c: List[int]

        with self.assertRaisesRegex(
            RuntimeError,
            "Expected a value of type 'int' for argument 'a'"
            " but instead found type 'str'",
        ):

            @torch.jit.script
            def foo():
                # [FUNCTION DOC] HELPER: Scripted; builds the NamedTuple from three strings
                # (expected compile failure).
                tup = MyCoolNamedTuple("foo", "bar", "baz")
                return tup

    def test_namedtuple_kwarg_construct(self):
        # [FUNCTION DOC] Checks NamedTuple construction with keyword arguments in a different
        # order from the fields.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c.
            a: int
            b: float
            c: List[int]

        @torch.jit.script
        def foo():
            # [FUNCTION DOC] HELPER: Scripted; builds the NamedTuple using keyword arguments and
            # returns it.
            tup = MyCoolNamedTuple(c=[1, 2, 3], b=3.5, a=9)
            return tup

        tup = foo()
        self.assertEqual(tup.a, 9)
        self.assertEqual(tup.b, 3.5)
        self.assertEqual(tup.c, [1, 2, 3])

    @unittest.skipIf(True, "broken while these tests were not in CI")
    def test_namedtuple_serialization(self):
        # [FUNCTION DOC] Intended to check saving and loading a module that returns a NamedTuple.
        # Always skipped (marked as broken); if run it saves to a file named foo.zip in the
        # working directory.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with int a, float b and int-list c.
            a: int
            b: float
            c: List[int]

        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule whose scripted forward returns a NamedTuple.
            @torch.jit.script_method
            def forward(self):
                # [FUNCTION DOC] HELPER: Scripted forward; returns the NamedTuple built from 3,
                # 3.5 and a list of three ints.
                return MyCoolNamedTuple(3, 3.5, [3, 4, 5])

        mm = MyMod()
        mm.save("foo.zip")
        torch.testing._internal.jit_utils.clear_class_registry()
        loaded = torch.jit.load("foo.zip")

        out = mm()
        out_loaded = loaded()

        for name in ["a", "b", "c"]:
            self.assertEqual(getattr(out_loaded, name), getattr(out, name))

    def test_namedtuple_inside_forwardref(self):
        # [FUNCTION DOC] Checks a NamedTuple whose field types are string forward references can
        # be used in a scripted function.
        class FeatureVector(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple like the feature vector but with string
            # forward-reference annotations.
            float_features: "float"
            sequence_features: "List[float]"
            time_since_first: "float"

        @torch.jit.script
        def foo(x) -> float:
            # [FUNCTION DOC] HELPER: Scripted; same computation as the basic NamedTuple test,
            # returns 18.0.
            fv = FeatureVector(3.0, [3.0], 3.0)
            rv = fv.float_features
            for val in fv.sequence_features:
                rv += val
            rv *= fv.time_since_first
            return rv

        self.assertEqual(foo(torch.rand(3, 4)), 18.0)

    def test_namedtuple_input_forwardref(self):
        # [FUNCTION DOC] Checks a global NamedTuple with forward-reference field types as a
        # scripted argument: scripted and eager outputs match.
        class MyNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with string annotations for an int, a float and
            # a tensor.
            a: "int"
            b: "float"
            c: "torch.Tensor"

        make_global(MyNamedTuple)

        nt = MyNamedTuple(4, 2.5, torch.rand((2, 2)))

        def fn(obj: MyNamedTuple):
            # [FUNCTION DOC] HELPER: Returns the sine of the tensor field plus the float field
            # raised to the int field.
            return ((obj.c + obj.b) ** obj.a).sin()

        expected = fn(nt)
        fn_s = torch.jit.script(fn)
        actual = fn_s(nt)
        self.assertEqual(expected, actual)

    # see #95858
    @unittest.expectedFailure
    def test_namedtuple_resolution_forwardref(self):
        # [FUNCTION DOC] Same as the module-like resolution test but with a forward-reference
        # field; marked as an expected failure (issue 95858).
        class TheType(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with one field t annotated by a string.
            t: "int"

        class MyModule(types.ModuleType):
            # [CLASS DOC] HELPER: ModuleType subclass named MyModule whose attribute lookup
            # returns the NamedTuple type for any name.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module type with the name MyModule.
                super().__init__("MyModule")

            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: Returns the NamedTuple type regardless of the
                # attribute requested.
                return TheType

        some_module = MyModule()

        def fn() -> some_module.Type:
            # [FUNCTION DOC] HELPER: Declared to return the NamedTuple type resolved through the
            # module-like object and constructs it with 1.
            return some_module.Type(1)

        self.checkScript(fn, [])


class TestScriptDict(JitTestCase):
    # [CLASS DOC] Tests for the ScriptDict objects returned by torch.jit.script on a Python dict,
    # which share reference semantics with TorchScript. Most tests run a function on an eager
    # dict and on its scripted counterpart and require the same result, final contents or
    # exception type.
    """
    This class contains a suite of tests for torch.jit.script, a
    function that returns a dictionary-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that objects returned
    by torch.jit.script behave like dictionaries do so that they are fungible
    in almost all circumstances with regular dictionaries.
    """

    def _script_dict_add(self, d: torch._C.ScriptDict, k: int, v: int):
        # [FUNCTION DOC] HELPER: Scripts a function that stores key k with value v into the dict
        # d, then calls it; used to show mutations made in TorchScript are visible in Python.
        """
        This is a helper function that inserts the pair (k, v) into the
        dictionary d in TorchScript. It is used for testing reference
        semantics.
        """

        @torch.jit.script
        def dict_add(d: Dict[int, int], k: int, v: int):
            # [FUNCTION DOC] HELPER: Scripted; sets the given key to the value in the
            # dict argument.
            d[k] = v

        dict_add(d, k, v)

    def _compare_eager_and_script(self, fn, input_dict, script_input_dict=None):
        # [FUNCTION DOC] HELPER: Runs fn on the eager dict and on a scripted dict (scripting the
        # input if none given), recording whether each raised. Asserts both raised or both did
        # not; on raising, asserts same exception type, otherwise asserts equal outputs and
        # equal dict contents.
        """
        This is a helper function that facilitates comparing behaviour between
        Python dictionaries and "scripted" dictionaries.

        Args:
            fn: The function to test and compare the behaviour of.
            input_dict: The input dictionary to use for the test (passed to fn).
            script_input_dict: The scripted input dictionary to use for the tests.
                                If None, input_dict is scripted with torch.jit.script
                                and used instead.
        """
        # Create ScriptDict version of input_dict if needed.
        script_input_dict = script_input_dict or torch.jit.script(input_dict)

        # Run fn with both input_dict and scripted_dict.
        eager_raised, script_raised = False, False

        try:
            eager_out = fn(input_dict)
        except Exception as e:
            eager_exception = e
            eager_raised = True

        try:
            script_out = fn(script_input_dict)
        except Exception as e:
            script_exception = e
            script_raised = True

        # Check that both calls raised or none of them raised.
        self.assertEqual(eager_raised, script_raised)

        if eager_raised:
            # If fn raised an exception, it should be the same between
            # regular and scripted dictionaries.
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            # Otherwise, make sure the outputs match and the dictionaries
            # match (the latter may not be the same as the output).
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_dict, script_input_dict)

    def test_repr(self):
        # [FUNCTION DOC] Compares repr of a dict eagerly and scripted.
        """
        Test the __repr__ method.
        """
        self._compare_eager_and_script(lambda d: repr(d), {1: 2})

    def test_bool(self):
        # [FUNCTION DOC] Compares truthiness of a non-empty and an empty dict eagerly
        # and scripted.
        """
        Test the __bool__ method. This should return True
        if the dictionary is non-empty and False otherwise.
        """
        self._compare_eager_and_script(lambda d: bool(d), {1: 2})
        self._compare_eager_and_script(lambda d: bool(d), {})

    def test_iter(self):
        # [FUNCTION DOC] Compares summing keys by iteration over eager and scripted dicts.
        """
        Test iteration over a dictionary's keys.
        """

        def sum_keys(input_dict):
            # [FUNCTION DOC] HELPER: Returns the sum of the keys of the dict.
            s = 0
            for k in input_dict:
                s += k

            return s

        self._compare_eager_and_script(sum_keys, {1: 2, 3: 4})

    def test_items(self):
        # [FUNCTION DOC] Compares iteration over items eagerly and scripted.
        """
        Test .items().
        """

        def sum_pair_product(input_dict):
            # [FUNCTION DOC] HELPER: Returns the sum of key times value over all items.
            s = 0
            for k, v in input_dict.items():
                s += k * v

            return s

        self._compare_eager_and_script(sum_pair_product, {1: 2, 3: 4})

    def test_getitem(self):
        # [FUNCTION DOC] Compares indexing for present and missing keys, including a key of the
        # wrong type, eagerly and scripted.
        """
        Test accessing dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}
        self._compare_eager_and_script(lambda d: d[1], data)
        self._compare_eager_and_script(lambda d: d[4], data)
        self._compare_eager_and_script(lambda d: d[2], data)
        self._compare_eager_and_script(lambda d: d["key"], data)

    def test_setitem(self):
        # [FUNCTION DOC] Compares item assignment eagerly and scripted, and checks that a scripted
        # dict raises TypeError for a wrong-typed key or value.
        """
        Test setting dictionary values using the [] operator.
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            # [FUNCTION DOC] HELPER: Sets the keys 1 and 3 to 10 and 11.
            input_dict[1] = 10
            input_dict[3] = 11

        self._compare_eager_and_script(fn, data)

        # Check that using improperly typed keys and values
        # throws TypeError.
        # _compare_eager_and_script cannot be used here since
        # the following uses of __setitem__ are valid in
        # Python.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            script_data["str"] = 3

        with self.assertRaises(TypeError):
            script_data[3] = "str"

    def test_contains(self):
        # [FUNCTION DOC] Compares membership tests eagerly and scripted, and checks a wrong-typed
        # key raises KeyError on the scripted dict.
        """
        Test membership checks (x in y, x not in y).
        """
        data = {1: 2, 3: 4}

        def fn(input_dict):
            # [FUNCTION DOC] HELPER: Returns four membership results for keys 1, 2, 3 and 4.
            return (
                1 in input_dict,
                2 not in input_dict,
                3 in input_dict,
                4 not in input_dict,
            )

        self._compare_eager_and_script(fn, data)

        # Check that using an improperly typed key
        # throws KeyError.
        script_data = torch.jit.script(data)

        with self.assertRaises(KeyError):
            a = "str" in script_data

    def test_delitem(self):
        # [FUNCTION DOC] Compares deletion of a present and a missing key eagerly and scripted,
        # and checks a wrong-typed key raises TypeError.
        """
        Test deletion.
        """
        data = {1: 2, 3: 4}

        def del_fn(input_dict):
            # [FUNCTION DOC] HELPER: Deletes key 1.
            del input_dict[1]

        def del_fn_raises(input_dict):
            # [FUNCTION DOC] HELPER: Deletes key 10, which is not in the dict (raises).
            del input_dict[10]

        self._compare_eager_and_script(del_fn, data)
        self._compare_eager_and_script(del_fn_raises, data)

        # Check that using an improperly typed key
        # throws TypeError.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            del script_data["str"]

    def test_len(self):
        # [FUNCTION DOC] Compares len of non-empty and empty dicts eagerly and scripted.
        """
        Test len() builtin function.
        """
        self._compare_eager_and_script(lambda d: len(d), {1: 2})
        self._compare_eager_and_script(lambda d: len(d), {})

    @unittest.skip(
        "Cannot pass until all dicts returned from TorchScript are ScriptDicts"
    )
    def test_nested(self):
        # [FUNCTION DOC] Intended to check mutation of nested scripted dicts is visible in the
        # outer one. Skipped until all dicts returned from TorchScript are ScriptDicts.
        """
        Test that reference semantics are honoured when the ScriptDict that is
        mutated using TorchScript is inside another.
        """
        nested = torch.jit.script(
            {1: {1: 2}, 2: {3: 4}}, type_hint=Dict[int, Dict[int, int]]
        )

        one = nested[1]
        two = nested[2]

        self._script_dict_add(one, 9, 10)
        self._script_dict_add(two, 11, 12)

        # The mutation should be visible in the original dictionary, nested.
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(len(nested[1]), 2)
        self.assertEqual(len(nested[2]), 2)

    def test_reference_semantics(self):
        # [FUNCTION DOC] Checks a mutation made inside a TorchScript function is visible in the
        # Python ScriptDict.
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptDict in TorchScript are visible in Python.
        """
        data = torch.jit.script({1: 2})
        self._script_dict_add(data, 3, 4)

        # The mutation should be visible in the original dictionary.
        self.assertEqual(len(data), 2)
        self.assertTrue(3 in data)
        self.assertEqual(data[3], 4)


class TestScriptList(JitTestCase):
    # [CLASS DOC] Tests for the ScriptList objects returned by torch.jit.script on a Python list,
    # with reference semantics shared with TorchScript. Most tests run a function on an eager
    # list and on its scripted counterpart and require the same result, final contents or
    # exception type.
    """
    This class contains a suite of tests for torch._C.ScriptList, a
    function that returns a list-like object that has reference
    semantics across the Python/TorchScript boundary. That is,
    it can be passed to a TorchScript function that mutates it
    and those modifications are visible in the scope of the Python
    caller of said TorchScript function.

    The vast majority of tests are for making sure that instances of
    torch._C.ScriptList behave like lists do so that they are fungible
    in almost all circumstances with regular list.
    """

    def _script_list_add(self, l: torch._C.ScriptList, e: int):
        # [FUNCTION DOC] HELPER: Scripts a function that appends element e to list l, then calls
        # it; used to show TorchScript mutations are visible in Python.
        """
        This is a helper function that inserts the element e into the
        list l in TorchScript. It is used for testing reference
        semantics.
        """

        @torch.jit.script
        def list_add(l: List[int], e: int):
            # [FUNCTION DOC] HELPER: Scripted; appends the element to the list argument.
            l.append(e)

        list_add(l, e)

    def _compare_eager_and_script(self, fn, input_list, script_input_list=None):
        # [FUNCTION DOC] HELPER: Runs fn on the eager list and on a scripted list (scripting the
        # input if none given), recording whether each raised. Asserts both raised or both did
        # not; on raising, asserts same exception type, otherwise asserts equal outputs and
        # equal list contents.
        """
        This is a helper function that facilitates comparing behaviour between
        Python lists and "scripted" lists.
        Args:
            fn: The function to test and compare the behaviour of.
            input_list: The input list to use for the test (passed to fn).
            script_input_list: The scripted input list to use for the tests.
                                If None, input_list is scripted with torch.jit.script
                                and used instead.
        """
        # Create ScriptDict version of input_list if needed.
        script_input_list = script_input_list or torch.jit.script(input_list)

        # Run fn with both input_list and scripted_dict.
        eager_raised, script_raised = False, False

        try:
            eager_out = fn(input_list)
        except Exception as e:
            eager_exception = e
            eager_raised = True

        try:
            script_out = fn(script_input_list)
        except Exception as e:
            script_exception = e
            script_raised = True

        # Check that both calls raised or none of them raised.
        self.assertEqual(eager_raised, script_raised)

        if eager_raised:
            # If fn raised an exception, it should be the same between
            # regular and scripted lists.
            self.assertEqual(type(eager_exception), type(script_exception))
        else:
            # Otherwise, make sure the outputs match and the lists
            # match (the latter may not be the same as the output).
            self.assertEqual(eager_out, script_out)
            self.assertEqual(input_list, script_input_list)

    def test_repr(self):
        # [FUNCTION DOC] Compares repr of a list eagerly and scripted.
        """
        Test the __repr__ method.
        """
        self._compare_eager_and_script(lambda l: repr(l), [1])

    def test_bool(self):
        # [FUNCTION DOC] Compares truthiness of a non-empty and an empty list eagerly
        # and scripted.
        """
        Test the __bool__ method. This should return True
        if the list is non-empty and False otherwise.
        """
        self._compare_eager_and_script(lambda l: bool(l), [1])
        self._compare_eager_and_script(lambda l: bool(l), [])

    def test_iter(self):
        # [FUNCTION DOC] Compares summing elements by iteration over eager and scripted lists.
        """
        Test iteration over a list's elements.
        """

        def sum_elements(input_list):
            # [FUNCTION DOC] HELPER: Returns the sum of the elements.
            s = 0
            for k in input_list:
                s += k

            return s

        self._compare_eager_and_script(sum_elements, [1, 2, 3, 4])

    def test_getitem(self):
        # [FUNCTION DOC] Compares indexing, slicing (including step -1) and error cases (out of
        # range index, wrong key type) eagerly and scripted.
        """
        Test accessing list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # Test regular indexing.
        self._compare_eager_and_script(lambda l: l[1], data)
        self._compare_eager_and_script(lambda l: l[3], data)
        self._compare_eager_and_script(lambda l: l[-1], data)

        # Test slicing.
        self._compare_eager_and_script(lambda l: l[1:3], data)
        self._compare_eager_and_script(lambda l: l[:], data)
        self._compare_eager_and_script(lambda l: l[1:], data)
        self._compare_eager_and_script(lambda l: l[:2], data)
        self._compare_eager_and_script(lambda l: l[-1], data)
        self._compare_eager_and_script(lambda l: l[-1::-1], data)

        # Test errors.
        self._compare_eager_and_script(lambda l: l[5], data)
        self._compare_eager_and_script(lambda l: l[-7], data)
        self._compare_eager_and_script(lambda l: l["key"], data)

    def test_setitem(self):
        # [FUNCTION DOC] Compares item and slice assignment and error cases eagerly and scripted,
        # and checks assigning a string element to the scripted int list raises TypeError.
        """
        Test setting list elements using the [] operator.
        """
        data = [1, 2, 3, 4]

        # Test regular assignment.
        def setitem(input_list):
            # [FUNCTION DOC] HELPER: Assigns 10, 11 and 12 at indices 1, 3 and -1.
            input_list[1] = 10
            input_list[3] = 11
            input_list[-1] = 12

        self._compare_eager_and_script(setitem, data.copy())

        # Test slice assignment.
        # TODO: Something like input_list[:1] = [1, 2, 3, 4, 5]
        # is allowed in Python, but pybind11/stl_bind.h does not
        # allow it. Should we?
        def setitem_slice(input_list):
            # [FUNCTION DOC] HELPER: Assigns two-element lists to a stepped slice and a
            # tail slice.
            input_list[:4:2] = [10, 11]
            input_list[-2:] = [15, 16]

        self._compare_eager_and_script(setitem_slice, data)

        # Test errors.
        def out_of_range(input_list):
            # [FUNCTION DOC] HELPER: Assigns at index 11 (raises).
            input_list[11] = 3

        def out_of_range_negative(input_list):
            # [FUNCTION DOC] HELPER: Assigns at index -11 (raises).
            input_list[-11] = 3

        def wrong_index_type(input_list):
            # [FUNCTION DOC] HELPER: Assigns using a string index (raises).
            input_list["str"] = 3

        self._compare_eager_and_script(out_of_range, data)
        self._compare_eager_and_script(out_of_range_negative, data)
        self._compare_eager_and_script(wrong_index_type, data)

        # Check that using value of an incorrect type throws TypeError.
        # _compare_eager_and_script cannot be used here since
        # the following use of __setitem__ is valid in
        # Python.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            script_data[0] = "str"

    def test_contains(self):
        # [FUNCTION DOC] Compares membership tests eagerly and scripted, and checks a wrong-typed
        # element raises TypeError on the scripted list.
        """
        Test membership checks (x in y, x not in y).
        """
        data = [1, 2, 3, 4]

        def fn(input_list):
            # [FUNCTION DOC] HELPER: Returns four membership results for values 1, 2, 3 and 4.
            return (
                1 in input_list,
                2 not in input_list,
                3 in input_list,
                4 not in input_list,
            )

        self._compare_eager_and_script(fn, data)

        # Check that using a value of an incorrect type throws a TypeError.
        script_data = torch.jit.script(data)

        with self.assertRaises(TypeError):
            a = "str" in script_data

    def test_delitem(self):
        # [FUNCTION DOC] Compares deletion by valid index, out-of-range index and wrong-typed
        # index eagerly and scripted.
        """
        Test deletion.
        """
        data = [1, 2, 3, 4]

        def del_fn(input_list):
            # [FUNCTION DOC] HELPER: Deletes index 1.
            del input_list[1]

        def del_fn_out_of_range(input_list):
            # [FUNCTION DOC] HELPER: Deletes index 10 (raises).
            del input_list[10]

        def del_fn_wrong_type(input_list):
            # [FUNCTION DOC] HELPER: Deletes using a string index (raises).
            del input_list["str"]

        self._compare_eager_and_script(del_fn, data.copy())
        self._compare_eager_and_script(del_fn_out_of_range, data)
        self._compare_eager_and_script(del_fn_wrong_type, data)

    def test_len(self):
        # [FUNCTION DOC] Compares len of non-empty and empty lists eagerly and scripted.
        """
        Test len() builtin function.
        """
        self._compare_eager_and_script(lambda l: len(l), [1, 2, 3, 4])
        self._compare_eager_and_script(lambda l: len(l), [])

    def test_count(self):
        # [FUNCTION DOC] Compares count eagerly and scripted, and checks a wrong-typed element
        # raises TypeError.
        """
        Test count method.
        """
        self._compare_eager_and_script(lambda l: l.count(3), [1, 2, 3, 3])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.count("str")

    def test_remove(self):
        # [FUNCTION DOC] Compares remove of a present and an absent element eagerly and scripted,
        # and checks a wrong-typed element raises TypeError.
        """
        Test remove method.
        """
        self._compare_eager_and_script(lambda l: l.remove(1), [1, 2, 3])
        self._compare_eager_and_script(lambda l: l.remove(10), [1, 2, 3])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.remove("str")

    def test_append(self):
        # [FUNCTION DOC] Compares append eagerly and scripted, and checks a wrong-typed element
        # raises TypeError.
        """
        Test append method.
        """
        self._compare_eager_and_script(lambda l: l.append(1), [4, 3, 2])

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.append("str")

    def test_clear(self):
        # [FUNCTION DOC] Compares clear eagerly and scripted.
        """
        Test clear.
        """
        self._compare_eager_and_script(lambda l: l.clear(), [4, 3, 2])

    def test_extend(self):
        # [FUNCTION DOC] Compares extend with a list, a dict and a custom iterable eagerly and
        # scripted, and checks wrong-typed lists or dicts raise TypeError.
        """
        Test extend.
        """

        class Iterable:
            # [CLASS DOC] HELPER: Iterator yielding consecutive integers from zero, intended to
            # stop at a limit.
            # KNOWN ISSUE: the iteration method compares against the undefined name limit instead
            # of the stored attribute, so iterating raises NameError; the extend-with-iterable
            # case therefore only compares exception types.
            def __init__(self, limit: int):
                # [FUNCTION DOC] HELPER: Stores the limit and a starting value of zero.
                self.limit = limit
                self.value = 0

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns the iterator itself.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Intended to return successive ints and stop at the limit.
                # KNOWN ISSUE: refers to the undefined name limit, so it raises NameError on the
                # first call.
                if self.value == limit:  # noqa: F821
                    raise StopIteration

                ret = self.value
                self.value += 1
                return ret

        data = [1, 2, 3]

        def extend_list(input_list):
            # [FUNCTION DOC] HELPER: Extends the list with a list of 4, 5 and 6.
            input_list.extend([4, 5, 6])

        def extend_dict(input_list):
            # [FUNCTION DOC] HELPER: Extends the list with the keys of a dict.
            input_list.extend({4: 10, 5: 11, 6: 12})

        def extend_iterable(input_list):
            # [FUNCTION DOC] HELPER: Extends the list with an instance of the custom iterable.
            input_list.extend(Iterable(3))

        self._compare_eager_and_script(extend_list, data.copy())
        self._compare_eager_and_script(extend_dict, data.copy())
        self._compare_eager_and_script(extend_iterable, data)

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.extend(["a"])

        with self.assertRaises(TypeError):
            script_data.extend({"a": 1})

    def test_insert(self):
        # [FUNCTION DOC] Compares insert at several positions eagerly and scripted.
        # KNOWN ISSUE: the type-check case calls insert with a single tuple argument, so the
        # TypeError raised may come from the argument count rather than from the element type.
        """
        Test insert.
        """
        data = [1, 2, 4]

        self._compare_eager_and_script(lambda l: l.insert(3, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(0, 3), data.copy())
        self._compare_eager_and_script(lambda l: l.insert(-2, 3), data)

        # Check that using a value of an incorrect type throws TypeError.
        script_data = torch.jit.script([1])

        with self.assertRaises(TypeError):
            script_data.insert((0, "str"))

    def test_pop(self):
        # [FUNCTION DOC] Compares pop with no index, positive and negative indices, and an
        # out-of-range index eagerly and scripted.
        """
        Test pop.
        """
        data = [1, 2, 3, 4, 5]

        # Test normal cases.
        self._compare_eager_and_script(lambda l: l.pop(), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(2), data.copy())
        self._compare_eager_and_script(lambda l: l.pop(-3), data.copy())

        # Test error cases.
        self._compare_eager_and_script(lambda l: l.pop(10), data)

    @unittest.skip(
        "Cannot pass until all list returned from TorchScript are ScriptLists"
    )
    def test_nested(self):
        # [FUNCTION DOC] Intended to check mutation of nested scripted lists is visible in the
        # outer one. Skipped until all lists returned from TorchScript are ScriptLists.
        """
        Test that reference semantics are honoured when the ScriptList that is
        mutated using TorchScript is inside another.
        """
        nested = torch.jit.script([[1], [2]], List[List[int]])

        one = nested[0]
        two = nested[1]

        self._script_list_add(one, 3)
        self._script_list_add(two, 4)

        # The mutation should be visible in the original list, nested.
        self.assertEqual(len(one), 2)
        self.assertEqual(len(two), 2)
        self.assertEqual(one[len(one) - 1], 3)
        self.assertEqual(two[len(one) - 1], 4)
        self.assertEqual(len(nested[0]), 2)
        self.assertEqual(len(nested[1]), 2)

    def test_reference_semantics(self):
        # [FUNCTION DOC] Checks a mutation made inside a TorchScript function is visible in the
        # Python ScriptList.
        """
        Test that reference semantics are honoured; that modifications made
        to a ScriptList in TorchScript are visible in Python.
        """
        l = torch.jit.script([1, 2])
        self._script_list_add(l, 3)

        self.assertEqual(len(l), 3)
        self.assertTrue(3 in l)
        self.assertEqual(l[2], 3)

    def test_defaultdict(self):
        # [FUNCTION DOC] Checks a module with a class-level annotated string-to-list-of-string
        # attribute initialized from a defaultdict can be scripted, then scripts fresh instances
        # 300 times as a flakiness smoke test (takes a couple of seconds).
        def get_dict():
            # [FUNCTION DOC] HELPER: Returns an empty defaultdict with list as the
            # default factory.
            test_dict = defaultdict(list)
            return test_dict

        class Test(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a dict-typed attribute from a defaultdict and two
            # string attributes.
            segments_groupby_col: Dict[str, List[str]]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor, stores the helper's dict as
                # an attribute and sets two string attributes.
                super().__init__()
                self.segments_groupby_col = get_dict()
                self.col1 = "a"
                self.col2 = "b"

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns 1 if the first string attribute is a key of the
                # dict, else 2.
                if self.col1 in self.segments_groupby_col:
                    return 1
                else:
                    return 2

        test = Test()
        test_script = torch.jit.script(test)
        test_script.segments_groupby_col

        # Smoketest for flakiness. Takes around 2s.
        for _ in range(300):
            test = Test()
            test_script = torch.jit.script(test)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
