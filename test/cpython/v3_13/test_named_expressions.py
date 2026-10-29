# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_named_expressions adapted to run under Dynamo
# (CPythonTestCase). Tests PEP 572 assignment expressions: invalid forms must raise SyntaxError
# with specific messages (checked by compiling snippets with exec), valid forms must bind
# variables with the right value and scope (function, comprehension, global, nonlocal, generator
# expression, name mangling). Import time effect: defines the module global GLOBAL_VAR used by
# scope tests; running the file calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_named_expressions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest

GLOBAL_VAR = None

class NamedExpressionInvalidTest(CPythonTestCase):
    # [CLASS DOC] Syntax-error tests: each method execs a small source snippet and expects a
    # SyntaxError with a given message fragment. No state beyond the TestCase.

    def test_named_expression_invalid_01(self):
        # [FUNCTION DOC] An unparenthesized walrus as a whole statement is a SyntaxError
        # (invalid syntax).
        code = """x := 0"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_02(self):
        # [FUNCTION DOC] An unparenthesized walrus on the right of a plain assignment raises
        # SyntaxError.
        code = """x = y := 0"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_03(self):
        # [FUNCTION DOC] An unparenthesized walrus statement whose value is a call raises
        # SyntaxError.
        code = """y := f(x)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_04(self):
        # [FUNCTION DOC] Chained assignment followed by an unparenthesized walrus call raises
        # SyntaxError.
        code = """y0 = y1 := f(x)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_06(self):
        # [FUNCTION DOC] A walrus whose target is a tuple raises SyntaxError saying assignment
        # expressions cannot be used with a tuple.
        code = """((a, b) := (1, 2))"""

        with self.assertRaisesRegex(SyntaxError, "cannot use assignment expressions with tuple"):
            exec(code, {}, {})

    def test_named_expression_invalid_07(self):
        # [FUNCTION DOC] An unparenthesized walrus as a parameter default value raises
        # SyntaxError.
        code = """def spam(a = b := 42): pass"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_08(self):
        # [FUNCTION DOC] An unparenthesized walrus inside a parameter annotation raises
        # SyntaxError.
        code = """def spam(a: b := 42 = 5): pass"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_09(self):
        # [FUNCTION DOC] An unparenthesized walrus as a keyword argument value raises
        # SyntaxError.
        code = """spam(a=b := 'c')"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_10(self):
        # [FUNCTION DOC] A keyword argument whose value is an unparenthesized walrus call raises
        # SyntaxError.
        code = """spam(x = y := f(x))"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_11(self):
        # [FUNCTION DOC] A bare walrus positional argument after a keyword argument raises
        # SyntaxError (positional argument follows keyword argument).
        code = """spam(a=1, b := 2)"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_12(self):
        # [FUNCTION DOC] A parenthesized walrus positional argument after a keyword argument
        # still raises SyntaxError about positional argument following keyword argument.
        code = """spam(a=1, (b := 2))"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_13(self):
        # [FUNCTION DOC] Same snippet and expected error as test 12 (a duplicate case in the
        # upstream CPython suite).
        code = """spam(a=1, (b := 2))"""

        with self.assertRaisesRegex(SyntaxError,
            "positional argument follows keyword argument"):
            exec(code, {}, {})

    def test_named_expression_invalid_14(self):
        # [FUNCTION DOC] A walrus nested unparenthesized inside a lambda body after another
        # walrus raises SyntaxError (invalid syntax).
        code = """(x := lambda: y := 1)"""

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_15(self):
        # [FUNCTION DOC] A walrus directly as a lambda body raises SyntaxError (cannot use
        # assignment expressions with lambda).
        code = """(lambda: x := 1)"""

        with self.assertRaisesRegex(SyntaxError,
            "cannot use assignment expressions with lambda"):
            exec(code, {}, {})

    def test_named_expression_invalid_16(self):
        # [FUNCTION DOC] A walrus as the unparenthesized iterable of a comprehension raises
        # SyntaxError.
        code = "[i + 1 for i in i := [1,2]]"

        with self.assertRaisesRegex(SyntaxError, "invalid syntax"):
            exec(code, {}, {})

    def test_named_expression_invalid_17(self):
        # [FUNCTION DOC] Unparenthesized walrus targets in a list display that looks like a
        # comprehension raise SyntaxError suggesting parentheses around the comprehension
        # target.
        code = "[i := 0, j := 1 for i, j in [(1, 2), (3, 4)]]"

        with self.assertRaisesRegex(SyntaxError,
                "did you forget parentheses around the comprehension target?"):
            exec(code, {}, {})

    def test_named_expression_invalid_in_class_body(self):
        # [FUNCTION DOC] A walrus inside a comprehension in a class body raises SyntaxError
        # (cannot be used in a class body comprehension).
        code = """class Foo():
            [(42, 1 + ((( j := i )))) for i in range(5)]
        """

        with self.assertRaisesRegex(SyntaxError,
            "assignment expression within a comprehension cannot be used in a class body"):
            exec(code, {}, {})

    def test_named_expression_valid_rebinding_iteration_variable(self):
        # [FUNCTION DOC] For several walrus targets that are only parts of a complex target
        # expression (subscript or attribute components) in a comprehension, exec in module,
        # class and function scopes must not raise SyntaxError; only NameError (module and class
        # scope) is expected since the names are undefined. Covers parenthesis, bracket and
        # brace comprehension forms.
        # KNOWN ISSUE: the formatted code is assigned back to the loop variable, so after the
        # first bracket kind the template has no placeholders and the later kinds re-run the
        # first form.
        # This test covers that we can reassign variables
        # that are not directly assigned in the
        # iterable part of a comprehension.
        cases = [
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: c",
                "{0}(c := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: d",
                "{0}(d := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: e",
                "{0}(e := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: f",
                "{0}(f := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: g",
                "{0}(g := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: h",
                "{0}(h := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: i",
                "{0}(i := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: j",
                "{0}(j := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
        ]
        for test_case, code in cases:
            for lpar, rpar in [('(', ')'), ('[', ']'), ('{', '}')]:
                code = code.format(lpar, rpar)
                with self.subTest(case=test_case, lpar=lpar, rpar=rpar):
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    with self.assertRaises(NameError):
                        exec(code, {}) # Module scope
                    with self.assertRaises(NameError):
                        exec(code, {}, {}) # Class scope
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_iteration_variable(self):
        # [FUNCTION DOC] Walrus targets that equal a directly assigned comprehension iteration
        # variable in a complex target raise SyntaxError (cannot rebind iteration variable) in
        # module, class and function scopes.
        # KNOWN ISSUE: the formatted code is assigned back to the loop variable, so only the
        # first bracket kind is truly exercised.
        # This test covers that we cannot reassign variables
        # that are directly assigned in the iterable part of a comprehension.
        cases = [
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{0}(a := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
            ("Complex expression: b", "b",
                "{0}(b := 1) for a, (*b, c[d+e::f(g)], h.i) in j{1}"),
        ]
        for test_case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            for lpar, rpar in [('(', ')'), ('[', ']'), ('{', '}')]:
                code = code.format(lpar, rpar)
                with self.subTest(case=test_case, lpar=lpar, rpar=rpar):
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    # Names used in snippets are not defined,
                    # but we are fine with it: just must not be a SyntaxError.
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(code, {}) # Module scope
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(code, {}, {}) # Class scope
                    with self.assertRaisesRegex(SyntaxError, msg):
                        exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_list_comprehension_iteration_variable(self):
        # [FUNCTION DOC] List comprehensions that rebind their own iteration variable with a
        # walrus (local, nested, inner loop, unpacking, in a condition, unreachable branches)
        # must raise SyntaxError naming the variable, in module, class and lambda (function)
        # scopes.
        cases = [
            ("Local reuse", 'i', "[i := 0 for i in range(5)]"),
            ("Nested reuse", 'j', "[[(j := 0) for i in range(5)] for j in range(5)]"),
            ("Reuse inner loop target", 'j', "[(j := 0) for i in range(5) for j in range(5)]"),
            ("Unpacking reuse", 'i', "[i := 0 for i, j in [(0, 1)]]"),
            ("Reuse in loop condition", 'i', "[i+1 for i in range(5) if (i := 0)]"),
            ("Unreachable reuse", 'i', "[False or (i:=0) for i in range(5)]"),
            ("Unreachable nested reuse", 'i',
                "[(i, j) for i in range(5) for j in range(5) if True or (i:=10)]"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_list_comprehension_inner_loop(self):
        # [FUNCTION DOC] A list comprehension whose inner loop target is also a walrus target
        # raises SyntaxError (inner loop cannot rebind assignment expression target), in all
        # three scopes.
        cases = [
            ("Inner reuse", 'j', "[i for i in range(5) if (j := 0) for j in range(5)]"),
            ("Inner unpacking reuse", 'j', "[i for i in range(5) if (j := 0) for j, k in [(0, 1)]]"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_list_comprehension_iterable_expression(self):
        # [FUNCTION DOC] A walrus anywhere inside a list comprehension iterable expression (top
        # level, tuple, list, lambda, inner loop, nested comprehensions) raises SyntaxError
        # (cannot be used in a comprehension iterable expression), in all three scopes.
        cases = [
            ("Top level", "[i for i in (i := range(5))]"),
            ("Inside tuple", "[i for i in (2, 3, i := range(5))]"),
            ("Inside list", "[i for i in [2, 3, i := range(5)]]"),
            ("Different name", "[i for i in (j := range(5))]"),
            ("Lambda expression", "[i for i in (lambda:(j := range(5)))()]"),
            ("Inner loop", "[i for i in range(5) for j in (i := range(5))]"),
            ("Nested comprehension", "[i for i in [j for j in (k := range(5))]]"),
            ("Nested comprehension condition", "[i for i in [j for j in range(5) if (j := True)]]"),
            ("Nested comprehension body", "[i for i in [(j := True) for j in range(5)]]"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_set_comprehension_iteration_variable(self):
        # [FUNCTION DOC] Set comprehension version of the iteration-variable rebinding tests,
        # including the complex-target regression cases; each must raise SyntaxError in all
        # three scopes.
        cases = [
            ("Local reuse", 'i', "{i := 0 for i in range(5)}"),
            ("Nested reuse", 'j', "{{(j := 0) for i in range(5)} for j in range(5)}"),
            ("Reuse inner loop target", 'j', "{(j := 0) for i in range(5) for j in range(5)}"),
            ("Unpacking reuse", 'i', "{i := 0 for i, j in {(0, 1)}}"),
            ("Reuse in loop condition", 'i', "{i+1 for i in range(5) if (i := 0)}"),
            ("Unreachable reuse", 'i', "{False or (i:=0) for i in range(5)}"),
            ("Unreachable nested reuse", 'i',
                "{(i, j) for i in range(5) for j in range(5) if True or (i:=10)}"),
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{(a := 1) for a, (*b, c[d+e::f(g)], h.i) in j}"),
            ("Complex expression: b", "b",
                "{(b := 1) for a, (*b, c[d+e::f(g)], h.i) in j}"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_set_comprehension_inner_loop(self):
        # [FUNCTION DOC] Set comprehension version of the inner-loop rebinding test; expects
        # SyntaxError in module, class and lambda scopes.
        cases = [
            ("Inner reuse", 'j', "{i for i in range(5) if (j := 0) for j in range(5)}"),
            ("Inner unpacking reuse", 'j', "{i for i in range(5) if (j := 0) for j, k in {(0, 1)}}"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_set_comprehension_iterable_expression(self):
        # [FUNCTION DOC] Set comprehension version of the iterable-expression test; a walrus in
        # the iterable raises SyntaxError in all three scopes.
        cases = [
            ("Top level", "{i for i in (i := range(5))}"),
            ("Inside tuple", "{i for i in (2, 3, i := range(5))}"),
            ("Inside list", "{i for i in {2, 3, i := range(5)}}"),
            ("Different name", "{i for i in (j := range(5))}"),
            ("Lambda expression", "{i for i in (lambda:(j := range(5)))()}"),
            ("Inner loop", "{i for i in range(5) for j in (i := range(5))}"),
            ("Nested comprehension", "{i for i in {j for j in (k := range(5))}}"),
            ("Nested comprehension condition", "{i for i in {j for j in range(5) if (j := True)}}"),
            ("Nested comprehension body", "{i for i in {(j := True) for j in range(5)}}"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_dict_comprehension_iteration_variable(self):
        # [FUNCTION DOC] Dict comprehension version of the iteration-variable rebinding tests
        # (key, value, both, nested, unpacking, conditions, complex targets); each raises
        # SyntaxError in all three scopes.
        cases = [
            ("Key reuse", 'i', "{(i := 0): 1 for i in range(5)}"),
            ("Value reuse", 'i', "{1: (i := 0) for i in range(5)}"),
            ("Both reuse", 'i', "{(i := 0): (i := 0) for i in range(5)}"),
            ("Nested reuse", 'j', "{{(j := 0): 1 for i in range(5)} for j in range(5)}"),
            ("Reuse inner loop target", 'j', "{(j := 0): 1 for i in range(5) for j in range(5)}"),
            ("Unpacking key reuse", 'i', "{(i := 0): 1 for i, j in {(0, 1)}}"),
            ("Unpacking value reuse", 'i', "{1: (i := 0) for i, j in {(0, 1)}}"),
            ("Reuse in loop condition", 'i', "{i+1: 1 for i in range(5) if (i := 0)}"),
            ("Unreachable reuse", 'i', "{(False or (i:=0)): 1 for i in range(5)}"),
            ("Unreachable nested reuse", 'i',
                "{i: j for i in range(5) for j in range(5) if True or (i:=10)}"),
            # Regression tests from https://github.com/python/cpython/issues/87447
            ("Complex expression: a", "a",
                "{(a := 1): 1 for a, (*b, c[d+e::f(g)], h.i) in j}"),
            ("Complex expression: b", "b",
                "{(b := 1): 1 for a, (*b, c[d+e::f(g)], h.i) in j}"),
        ]
        for case, target, code in cases:
            msg = f"assignment expression cannot rebind comprehension iteration variable '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_rebinding_dict_comprehension_inner_loop(self):
        # [FUNCTION DOC] Dict comprehension version of the inner-loop rebinding test; expects
        # SyntaxError in module, class and lambda scopes.
        cases = [
            ("Inner reuse", 'j', "{i: 1 for i in range(5) if (j := 0) for j in range(5)}"),
            ("Inner unpacking reuse", 'j', "{i: 1 for i in range(5) if (j := 0) for j, k in {(0, 1)}}"),
        ]
        for case, target, code in cases:
            msg = f"comprehension inner loop cannot rebind assignment expression target '{target}'"
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_dict_comprehension_iterable_expression(self):
        # [FUNCTION DOC] Dict comprehension version of the iterable-expression test; a walrus in
        # the iterable raises SyntaxError in all three scopes.
        cases = [
            ("Top level", "{i: 1 for i in (i := range(5))}"),
            ("Inside tuple", "{i: 1 for i in (2, 3, i := range(5))}"),
            ("Inside list", "{i: 1 for i in [2, 3, i := range(5)]}"),
            ("Different name", "{i: 1 for i in (j := range(5))}"),
            ("Lambda expression", "{i: 1 for i in (lambda:(j := range(5)))()}"),
            ("Inner loop", "{i: 1 for i in range(5) for j in (i := range(5))}"),
            ("Nested comprehension", "{i: 1 for i in {j: 2 for j in (k := range(5))}}"),
            ("Nested comprehension condition", "{i: 1 for i in {j: 2 for j in range(5) if (j := True)}}"),
            ("Nested comprehension body", "{i: 1 for i in {(j := True) for j in range(5)}}"),
        ]
        msg = "assignment expression cannot be used in a comprehension iterable expression"
        for case, code in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}) # Module scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(code, {}, {}) # Class scope
                with self.assertRaisesRegex(SyntaxError, msg):
                    exec(f"lambda: {code}", {}) # Function scope

    def test_named_expression_invalid_mangled_class_variables(self):
        # [FUNCTION DOC] Inside a method of a class, a walrus on a double-underscore name that
        # is also a nested comprehension iteration variable (a name subject to mangling) raises
        # SyntaxError about rebinding the iteration variable.
        code = """class Foo:
            def bar(self):
                [[(__x:=2) for _ in range(2)] for __x in range(2)]
        """

        with self.assertRaisesRegex(SyntaxError,
            "assignment expression cannot rebind comprehension iteration variable '__x'"):
            exec(code, {}, {})


class NamedExpressionAssignmentTest(CPythonTestCase):
    # [CLASS DOC] Tests that valid walrus expressions bind the expected values (plain
    # statements, conditions, loops, comprehensions, subscripts). No state beyond the TestCase.

    def test_named_expression_assignment_01(self):
        # [FUNCTION DOC] A parenthesized walrus statement binds the name to the value 10.
        (a := 10)

        self.assertEqual(a, 10)

    def test_named_expression_assignment_02(self):
        # [FUNCTION DOC] Rebinding an existing name to itself through a walrus leaves its value
        # 20.
        a = 20
        (a := a)

        self.assertEqual(a, 20)

    def test_named_expression_assignment_03(self):
        # [FUNCTION DOC] A walrus can bind the result of an arithmetic expression (3).
        (total := 1 + 2)

        self.assertEqual(total, 3)

    def test_named_expression_assignment_04(self):
        # [FUNCTION DOC] A walrus can bind a whole tuple.
        (info := (1, 2, 3))

        self.assertEqual(info, (1, 2, 3))

    def test_named_expression_assignment_05(self):
        # [FUNCTION DOC] In a parenthesized tuple display, the walrus binds only its first
        # element (1).
        (x := 1, 2)

        self.assertEqual(x, 1)

    def test_named_expression_assignment_06(self):
        # [FUNCTION DOC] Nested walruses all bind the same innermost value (0) to three names.
        (z := (y := (x := 0)))

        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_named_expression_assignment_07(self):
        # [FUNCTION DOC] A walrus can bind a tuple value to a name used later.
        (loc := (1, 2))

        self.assertEqual(loc, (1, 2))

    def test_named_expression_assignment_08(self):
        # [FUNCTION DOC] A walrus in an if condition binds the truthy string before the body
        # runs; fails the test if the else branch is taken.
        if spam := "eggs":
            self.assertEqual(spam, "eggs")
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_09(self):
        # [FUNCTION DOC] A walrus on the right of a short-circuit and binds when reached; fails
        # the test if the else branch is taken.
        if True and (spam := True):
            self.assertTrue(spam)
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_10(self):
        # [FUNCTION DOC] A walrus inside a comparison in an if condition binds 10 and the
        # comparison succeeds.
        if (match := 10) == 10:
            self.assertEqual(match, 10)
        else: self.fail("variable was not assigned using named expression")

    def test_named_expression_assignment_11(self):
        # [FUNCTION DOC] A walrus in a comprehension filter makes its value usable in the
        # element expression; checks the produced triples.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x/y) for x in input_data if (y := spam(x)) > 0]

        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])

    def test_named_expression_assignment_12(self):
        # [FUNCTION DOC] A walrus inside the element of a list comprehension is reused in the
        # same element to compute a ratio; checks the resulting pairs.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a
        res = [[y := spam(x), x/y] for x in range(1, 5)]

        self.assertEqual(res, [[1, 1.0], [2, 1.0], [3, 1.0], [4, 1.0]])

    def test_named_expression_assignment_13(self):
        # [FUNCTION DOC] A walrus inside a call argument binds the list that len measures (2).
        length = len(lines := [1, 2])

        self.assertEqual(length, 2)
        self.assertEqual(lines, [1,2])

    def test_named_expression_assignment_14(self):
        # [FUNCTION DOC] Integer n-th root (floor) iteration using a walrus in a while
        # condition, with a=9, n=2, x=3; the loop must end with a equal to 1.
        """
        Where all variables are positive integers, and a is at least as large
        as the n'th root of x, this algorithm returns the floor of the n'th
        root of x (and roughly doubling the number of accurate bits per
        iteration):
        """
        a = 9
        n = 2
        x = 3

        while a > (d := x // a**(n-1)):
            a = ((n-1)*a + d) // n

        self.assertEqual(a, 1)

    def test_named_expression_assignment_15(self):
        # [FUNCTION DOC] A walrus in a while condition that is false means the body never runs
        # (fails the test if it does) and the name is bound to False.
        while a := False:
            self.fail("While body executed")  # This will not run

        self.assertEqual(a, False)

    def test_named_expression_assignment_16(self):
        # [FUNCTION DOC] Walruses in a dict comprehension key and value generate Fibonacci-like
        # pairs; checks the resulting mapping of six entries.
        a, b = 1, 2
        fib = {(c := a): (a := b) + (b := a + c) - b for __ in range(6)}
        self.assertEqual(fib, {1: 2, 2: 3, 3: 5, 5: 8, 8: 13, 13: 21})

    def test_named_expression_assignment_17(self):
        # [FUNCTION DOC] A walrus used as a subscript index binds the index and the element is
        # looked up.
        a = [1]
        element = a[b:=0]
        self.assertEqual(b, 0)
        self.assertEqual(element, a[0])

    def test_named_expression_assignment_18(self):
        # [FUNCTION DOC] Walruses inside a multi-index subscript on a custom object bind both
        # indices; the object receives them as a tuple.
        class TwoDimensionalList:
            # [CLASS DOC] HELPER: wrapper around a list of lists whose item access takes a pair
            # of indices.
            def __init__(self, two_dimensional_list):
                # [FUNCTION DOC] HELPER: store the nested list in the attribute
                # two_dimensional_list.
                self.two_dimensional_list = two_dimensional_list

            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: return the element at the row and column given by the
                # first and second entries of the index.
                return self.two_dimensional_list[index[0]][index[1]]

        a = TwoDimensionalList([[1], [2]])
        element = a[b:=0, c:=0]
        self.assertEqual(b, 0)
        self.assertEqual(c, 0)
        self.assertEqual(element, a.two_dimensional_list[b][c])



class NamedExpressionScopeTest(CPythonTestCase):
    # [CLASS DOC] Tests the scope where a walrus target is bound (enclosing function, global,
    # nonlocal, comprehension and generator contexts). No state beyond the TestCase.

    def test_named_expression_scope_01(self):
        # [FUNCTION DOC] A name bound by a walrus inside a function is not visible outside; exec
        # of the snippet must raise NameError.
        code = """def spam():
    (a := 5)
print(a)"""

        with self.assertRaisesRegex(NameError, "name 'a' is not defined"):
            exec(code, {}, {})

    def test_named_expression_scope_02(self):
        # [FUNCTION DOC] A walrus in a comprehension element accumulates a running total, which
        # also leaks to the enclosing function.
        total = 0
        partial_sums = [total := total + v for v in range(5)]

        self.assertEqual(partial_sums, [0, 1, 3, 6, 10])
        self.assertEqual(total, 10)

    def test_named_expression_scope_03(self):
        # [FUNCTION DOC] A walrus inside a generator expression passed to any binds the last
        # examined value in the enclosing scope (stops at the first match, 1).
        containsOne = any((lastNum := num) == 1 for num in [1, 2, 3])

        self.assertTrue(containsOne)
        self.assertEqual(lastNum, 1)

    def test_named_expression_scope_04(self):
        # [FUNCTION DOC] A walrus in a list comprehension element leaves its final value (4)
        # bound in the enclosing function.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a
        res = [[y := spam(x), x/y] for x in range(1, 5)]

        self.assertEqual(y, 4)

    def test_named_expression_scope_05(self):
        # [FUNCTION DOC] A walrus in a comprehension filter leaves its last value (3) bound in
        # the enclosing function.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a
        input_data = [1, 2, 3]
        res = [(x, y, x/y) for x in input_data if (y := spam(x)) > 0]

        self.assertEqual(res, [(1, 1, 1.0), (2, 2, 1.0), (3, 3, 1.0)])
        self.assertEqual(y, 3)

    def test_named_expression_scope_06(self):
        # [FUNCTION DOC] A walrus in a nested comprehension binds the name in the enclosing
        # function.
        res = [[spam := i for i in range(3)] for j in range(2)]

        self.assertEqual(res, [[0, 1, 2], [0, 1, 2]])
        self.assertEqual(spam, 2)

    def test_named_expression_scope_07(self):
        # [FUNCTION DOC] A walrus inside a call argument binds the name in the current scope.
        len(lines := [1, 2])

        self.assertEqual(lines, [1, 2])

    def test_named_expression_scope_08(self):
        # [FUNCTION DOC] Two walruses nested in call arguments within a comprehension both leak
        # to the enclosing scope with their last values.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a

        def eggs(b):
            # [FUNCTION DOC] HELPER: return twice the argument.
            return b * 2

        res = [spam(a := eggs(b := h)) for h in range(2)]

        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)
        self.assertEqual(b, 1)

    def test_named_expression_scope_09(self):
        # [FUNCTION DOC] Nested walruses on the same name in a comprehension leave the final
        # value (2).
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a

        def eggs(b):
            # [FUNCTION DOC] HELPER: return twice the argument.
            return b * 2

        res = [spam(a := eggs(a := h)) for h in range(2)]

        self.assertEqual(res, [0, 2])
        self.assertEqual(a, 2)

    def test_named_expression_scope_10(self):
        # [FUNCTION DOC] Walruses in nested comprehensions bind names in the enclosing scope.
        res = [b := [a := 1 for i in range(2)] for j in range(2)]

        self.assertEqual(res, [[1, 1], [1, 1]])
        self.assertEqual(a, 1)
        self.assertEqual(b, [1, 1])

    def test_named_expression_scope_11(self):
        # [FUNCTION DOC] A walrus on the loop value in a comprehension element leaves the last
        # value (4).
        res = [j := i for i in range(5)]

        self.assertEqual(res, [0, 1, 2, 3, 4])
        self.assertEqual(j, 4)

    def test_named_expression_scope_17(self):
        # [FUNCTION DOC] A walrus that rebinds an existing outer name within a comprehension
        # carries the accumulated value across iterations (final 10).
        b = 0
        res = [b := i + b for i in range(5)]

        self.assertEqual(res, [0, 1, 3, 6, 10])
        self.assertEqual(b, 10)

    def test_named_expression_scope_18(self):
        # [FUNCTION DOC] An unparenthesized walrus as the sole positional call argument binds
        # the name and passes the value.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a

        res = spam(b := 2)

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_19(self):
        # [FUNCTION DOC] A parenthesized walrus as a positional call argument binds and passes
        # the value.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a

        res = spam((b := 2))

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_20(self):
        # [FUNCTION DOC] A parenthesized walrus as a keyword argument value binds and passes the
        # value.
        def spam(a):
            # [FUNCTION DOC] HELPER: return the argument unchanged.
            return a

        res = spam(a=(b := 2))

        self.assertEqual(res, 2)
        self.assertEqual(b, 2)

    def test_named_expression_scope_21(self):
        # [FUNCTION DOC] A bare walrus positional argument followed by a keyword argument works
        # (sum 3) and binds the name.
        def spam(a, b):
            # [FUNCTION DOC] HELPER: return the sum of the two arguments.
            return a + b

        res = spam(c := 2, b=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_22(self):
        # [FUNCTION DOC] A parenthesized walrus positional argument followed by a keyword
        # argument works.
        def spam(a, b):
            # [FUNCTION DOC] HELPER: return the sum of the two arguments.
            return a + b

        res = spam((c := 2), b=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_23(self):
        # [FUNCTION DOC] A parenthesized walrus keyword argument value followed by another
        # keyword argument works.
        def spam(a, b):
            # [FUNCTION DOC] HELPER: return the sum of the two arguments.
            return a + b

        res = spam(b=(c := 2), a=1)

        self.assertEqual(res, 3)
        self.assertEqual(c, 2)

    def test_named_expression_scope_24(self):
        # [FUNCTION DOC] A walrus on a name declared nonlocal in the nested function updates the
        # outer local variable.
        a = 10
        def spam():
            # [FUNCTION DOC] HELPER: rebind the enclosing variable a to 20 through a walrus
            # (declared nonlocal).
            nonlocal a
            (a := 20)
        spam()

        self.assertEqual(a, 20)

    def test_named_expression_scope_25(self):
        # [FUNCTION DOC] A walrus on a name declared global in a function updates the global of
        # the namespace mapping given to exec.
        ns = {}
        code = """a = 10
def spam():
    global a
    (a := 20)
spam()"""

        exec(code, ns, {})

        self.assertEqual(ns["a"], 20)

    def test_named_expression_variable_reuse_in_comprehensions(self):
        # [FUNCTION DOC] Builds comprehensions that rebind a non-iteration name (global or
        # nonlocal inside a lambda) via walrus in the element, filter, body and nested forms;
        # each snippet is exec'd in a fresh namespace and must end with that name equal to 2 and
        # the result [0, 1, 2].
        # The compiler is expected to raise syntax error for comprehension
        # iteration variables, but should be fine with rebinding of other
        # names (e.g. globals, nonlocals, other assignment expressions)

        # The cases are all defined to produce the same expected result
        # Each comprehension is checked at both function scope and module scope
        rebinding = "[x := i for i in range(3) if (x := i) or not x]"
        filter_ref = "[x := i for i in range(3) if x or not x]"
        body_ref = "[x for i in range(3) if (x := i) or not x]"
        nested_ref = "[j for i in range(3) if x or not x for j in range(3) if (x := i)][:-3]"
        cases = [
            ("Rebind global", f"x = 1; result = {rebinding}"),
            ("Rebind nonlocal", f"result, x = (lambda x=1: ({rebinding}, x))()"),
            ("Filter global", f"x = 1; result = {filter_ref}"),
            ("Filter nonlocal", f"result, x = (lambda x=1: ({filter_ref}, x))()"),
            ("Body global", f"x = 1; result = {body_ref}"),
            ("Body nonlocal", f"result, x = (lambda x=1: ({body_ref}, x))()"),
            ("Nested global", f"x = 1; result = {nested_ref}"),
            ("Nested nonlocal", f"result, x = (lambda x=1: ({nested_ref}, x))()"),
        ]
        for case, code in cases:
            with self.subTest(case=case):
                ns = {}
                exec(code, ns)
                self.assertEqual(ns["x"], 2)
                self.assertEqual(ns["result"], [0, 1, 2])

    def test_named_expression_global_scope(self):
        # [FUNCTION DOC] A walrus in a comprehension of a function with a global declaration
        # sets the module global GLOBAL_VAR; the global is reset to None afterwards.
        sentinel = object()
        global GLOBAL_VAR
        def f():
            # [FUNCTION DOC] HELPER: with a global declaration, bind GLOBAL_VAR to the sentinel
            # via a walrus in a comprehension and assert it holds the sentinel.
            global GLOBAL_VAR
            [GLOBAL_VAR := sentinel for _ in range(1)]
            self.assertEqual(GLOBAL_VAR, sentinel)
        try:
            f()
            self.assertEqual(GLOBAL_VAR, sentinel)
        finally:
            GLOBAL_VAR = None

    def test_named_expression_global_scope_no_global_keyword(self):
        # [FUNCTION DOC] Without a global declaration, a walrus in a comprehension binds the
        # function local; the module GLOBAL_VAR stays None.
        sentinel = object()
        def f():
            # [FUNCTION DOC] HELPER: bind a local GLOBAL_VAR to the sentinel via a walrus in a
            # comprehension and assert the local holds it.
            GLOBAL_VAR = None
            [GLOBAL_VAR := sentinel for _ in range(1)]
            self.assertEqual(GLOBAL_VAR, sentinel)
        f()
        self.assertEqual(GLOBAL_VAR, None)

    def test_named_expression_nonlocal_scope(self):
        # [FUNCTION DOC] A walrus in a comprehension inside a nested function with a nonlocal
        # declaration updates the outer function variable.
        sentinel = object()
        def f():
            # [FUNCTION DOC] HELPER: define g, call it, and assert the enclosing variable now
            # holds the sentinel.
            nonlocal_var = None
            def g():
                # [FUNCTION DOC] HELPER: with a nonlocal declaration, bind the outer variable to
                # the sentinel via a walrus in a comprehension.
                nonlocal nonlocal_var
                [nonlocal_var := sentinel for _ in range(1)]
            g()
            self.assertEqual(nonlocal_var, sentinel)
        f()

    def test_named_expression_nonlocal_scope_no_nonlocal_keyword(self):
        # [FUNCTION DOC] Without a nonlocal declaration, a walrus in a comprehension binds a
        # local of the inner function; the outer variable stays None.
        sentinel = object()
        def f():
            # [FUNCTION DOC] HELPER: define g, call it, and assert the enclosing variable is
            # still None.
            nonlocal_var = None
            def g():
                # [FUNCTION DOC] HELPER: bind a local of the same name (no nonlocal declaration)
                # via a walrus in a comprehension; the outer variable is unaffected.
                [nonlocal_var := sentinel for _ in range(1)]
            g()
            self.assertEqual(nonlocal_var, None)
        f()

    def test_named_expression_scope_in_genexp(self):
        # [FUNCTION DOC] A walrus inside a generator expression does not bind in the enclosing
        # locals before iteration; iterating the generator yields the expected sums.
        a = 1
        b = [1, 2, 3, 4]
        genexp = (c := i + a for i in b)

        self.assertNotIn("c", locals())
        for idx, elem in enumerate(genexp):
            self.assertEqual(elem, b[idx] + a)

    def test_named_expression_scope_mangled_names(self):
        # [FUNCTION DOC] Walruses on double-underscore names in comprehensions inside a class
        # method bind the global variable, both when written already mangled and when mangled by
        # the compiler; final value 2 is read through the mangled global name.
        class Foo:
            # [CLASS DOC] HELPER: class used to exercise name mangling of double-underscore
            # walrus targets.
            def f(self_):
                # [FUNCTION DOC] HELPER: declare a mangled-style global, set it to 0, then
                # rebind it to 1 and 2 via walruses in comprehensions (once with the explicit
                # mangled name, once with the unmangled one), asserting each value.
                global __x1
                __x1 = 0
                [_Foo__x1 := 1 for a in [2]]
                self.assertEqual(__x1, 1)
                [__x1 := 2 for a in [3]]
                self.assertEqual(__x1, 2)

        Foo().f()
        self.assertEqual(_Foo__x1, 2)

if __name__ == "__main__":
    run_tests()
