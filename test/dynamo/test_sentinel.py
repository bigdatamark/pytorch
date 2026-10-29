# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the Python 3.15 builtin sentinel objects: identity and equality
# comparison, attribute reads, truthiness, use as default values, dict lookup defaults, guards
# and recompilation keyed on identity, graph breaks, and creating sentinels inside compiled
# code. The whole test class is skipped on older Python versions; a module-level sentinel is
# defined at import time only on 3.15 or newer.

import builtins
import functools
import operator
import sys
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter


if sys.version_info >= (3, 15):
    MODULE_SENTINEL = builtins.sentinel("MODULE_SENTINEL")


@unittest.skipIf(sys.version_info < (3, 15), "Sentinel not supported in python < 3.15")
class SentinelTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for sentinel support in Dynamo, skipped when the interpreter is
    # older than 3.15. Most tests compile with the eager backend and full graph and compare
    # against eager results.
    def test_is(self):
        # [FUNCTION DOC] Compiles a function returning identity comparisons of two distinct
        # sentinels and of one sentinel with itself, and asserts the compiled result equals the
        # eager one.
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            # [FUNCTION DOC] HELPER: Returns a pair: whether the two sentinels are the same
            # object, and whether the first is the same object as itself.
            return s1 is s2, s1 is s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    @unittest.expectedFailure
    def test_eq(self):
        # [FUNCTION DOC] Expected failure: equality comparison of sentinels inside a compiled
        # full-graph function is not supported yet, so the compiled result is not obtained as
        # expected.
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            # [FUNCTION DOC] HELPER: Returns a pair of equality comparisons between the two
            # sentinels and of a sentinel with itself.
            return s1 == s2, s1 == s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    def test_attrs(self):
        # [FUNCTION DOC] Compiles a function that reads the name and module attributes of a
        # closed-over sentinel alongside a cosine of the input, and asserts it matches eager.
        s1 = builtins.sentinel("s1")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of x together with the sentinel's name
            # and module attributes.
            return x.cos(), s1.__name__, s1.__module__

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))

    def test_name_ro(self):
        # [FUNCTION DOC] Compiles a function that assigns to the read-only name attribute of a
        # sentinel. Asserts that Dynamo currently raises its Unsupported error; a TODO notes
        # that an AttributeError would be the correct behavior.
        s1 = builtins.sentinel("s1")

        def f():
            # [FUNCTION DOC] HELPER: Tries to assign a new string to the sentinel's name
            # attribute, which eager Python rejects; returns nothing.
            s1.__name__ = "fail"

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        # TODO should raise AttributeError
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_f()

    def test_recompile(self):
        # [FUNCTION DOC] Compiles a function that branches on identity with one sentinel. Calls
        # it with each of two sentinels and checks the results match eager and that two frames
        # were compiled, i.e. the identity check triggers a guard and recompile.
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f(x, s):
            # [FUNCTION DOC] HELPER: Returns the sine of x when the argument is the first
            # sentinel, otherwise the cosine.
            if s is s1:
                return x.sin()
            return x.cos()

        cnts = CompileCounter()
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(opt_f(x, s2), f(x, s2))
        self.assertEqual(cnts.frame_count, 2)

    def test_is_not(self):
        # [FUNCTION DOC] Same as the identity test but with the negated identity comparison;
        # compares compiled to eager results.
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f():
            # [FUNCTION DOC] HELPER: Returns a pair of negated identity comparisons: different
            # sentinels, and a sentinel against itself.
            return s1 is not s2, s1 is not s1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(opt_f(), f())

    def test_missing_default(self):
        # [FUNCTION DOC] Uses a sentinel as the default for a missing argument in a helper
        # function and checks that compiled calls with and without the optional argument match
        # eager.
        # The canonical sentinel use case: marker for "argument not provided"
        MISSING = builtins.sentinel("MISSING")

        def g(x, y=MISSING):
            # [FUNCTION DOC] HELPER: Returns the sine of x if y is the missing-marker sentinel,
            # otherwise x plus y.
            if y is MISSING:
                return x.sin()
            return x + y

        def f(x, y):
            # [FUNCTION DOC] HELPER: Calls the helper with only x and again with both x and y,
            # returning both results.
            return g(x), g(x, y)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self.assertEqual(opt_f(x, y), f(x, y))

    def test_truthiness(self):
        # [FUNCTION DOC] Checks that a sentinel is truthy in compiled code by applying both a
        # positive and a negated branch on it; asserts the compiled result equals eager.
        s1 = builtins.sentinel("s1")

        def f(x):
            # [FUNCTION DOC] HELPER: Applies sine to x if the sentinel is truthy, then adds 100
            # if it is falsy, and returns x.
            if s1:
                x = x.sin()
            if not s1:
                x = x + 100
            return x

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))

    def test_dict_get_default(self):
        # [FUNCTION DOC] Uses a sentinel as the default of a dictionary get call in compiled
        # code, with an empty dictionary and with a dictionary holding the key, and checks
        # results match eager.
        s1 = builtins.sentinel("s1")

        def f(x, d):
            # [FUNCTION DOC] HELPER: Looks up a key in d with the sentinel as default; returns
            # sine of x if missing, else x plus the value found.
            v = d.get("k", s1)
            if v is s1:
                return x.sin()
            return x + v

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, {}), f(x, {}))
        self.assertEqual(opt_f(x, {"k": x}), f(x, {"k": x}))

    def test_graph_break(self):
        # [FUNCTION DOC] Compiles without full graph a function that has a graph break and
        # returns a sentinel identity check afterwards; asserts the compiled result equals
        # eager.
        s1 = builtins.sentinel("s1")

        def f(x, s):
            # [FUNCTION DOC] HELPER: Computes sine of x, graph-breaks, and returns it with
            # whether s is the closed-over sentinel.
            y = x.sin()
            torch._dynamo.graph_break()
            return y, s is s1

        opt_f = torch.compile(f, backend="eager")
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))

    def test_return_sentinel(self):
        # [FUNCTION DOC] Returns a sentinel directly and nested inside a list from a compiled
        # function. Asserts the result equals eager and that the returned objects are identical
        # to the original sentinel.
        s1 = builtins.sentinel("s1")

        def f(x):
            # [FUNCTION DOC] HELPER: Returns sine of x, the sentinel, and a list holding the
            # sentinel.
            return x.sin(), s1, [s1]

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        ref, res = f(x), opt_f(x)
        self.assertEqual(ref, res)
        self.assertIs(res[1], s1)
        self.assertIs(res[2][0], s1)

    def test_guard_stability(self):
        # [FUNCTION DOC] Checks guards depend on sentinel identity, not name. Calling twice with
        # the same sentinel compiles one frame; a distinct sentinel with the same name triggers
        # a recompile (two frames).
        s1 = builtins.sentinel("s1")
        s1_dup = builtins.sentinel("s1")  # same name, distinct object

        def f(x, s):
            # [FUNCTION DOC] HELPER: Returns the sine of x if s is the closed-over sentinel,
            # else the cosine.
            if s is s1:
                return x.sin()
            return x.cos()

        cnts = CompileCounter()
        opt_f = torch.compile(f, backend=cnts, fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(opt_f(x, s1), f(x, s1))
        self.assertEqual(cnts.frame_count, 1)
        # identity (not name) drives specialization
        self.assertEqual(opt_f(x, s1_dup), f(x, s1_dup))
        self.assertEqual(cnts.frame_count, 2)

    def test_global_sentinel(self):
        # [FUNCTION DOC] Checks that a module-level sentinel global can be referenced in
        # compiled code and compared by identity; compares compiled to eager results.
        def f(x, s):
            # [FUNCTION DOC] HELPER: Returns the sine of x if s is the module-level sentinel,
            # else the cosine.
            if s is MODULE_SENTINEL:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, MODULE_SENTINEL), f(x, MODULE_SENTINEL))

    def test_attr_sentinel(self):
        # [FUNCTION DOC] Checks a sentinel stored as a class attribute and read through an
        # instance in compiled code; compares compiled to eager.
        class Cfg:
            # [CLASS DOC] HELPER: Plain holder class with one class attribute that is a sentinel
            # named MISSING.
            MISSING = builtins.sentinel("MISSING")

        cfg = Cfg()

        def f(x, s):
            # [FUNCTION DOC] HELPER: Returns the sine of x if s is the sentinel attribute of the
            # closed-over config instance, else the cosine.
            if s is cfg.MISSING:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, cfg.MISSING), f(x, cfg.MISSING))

    def test_container_input(self):
        # [FUNCTION DOC] Passes sentinels inside a list input and checks the compiled function
        # matches eager for lists holding each of two sentinels.
        s1 = builtins.sentinel("s1")
        s2 = builtins.sentinel("s2")

        def f(x, lst):
            # [FUNCTION DOC] HELPER: Returns the sine of x if the first element of the list is
            # the first sentinel, else the cosine.
            if lst[0] is s1:
                return x.sin()
            return x.cos()

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x, [s1]), f(x, [s1]))
        self.assertEqual(opt_f(x, [s2]), f(x, [s2]))

    def test_create_inside_graph_breaks(self):
        # [FUNCTION DOC] Asserts that creating a sentinel inside a full-graph compiled function
        # raises Dynamo's Unsupported error (out of scope; should be a graph break, not a
        # crash).
        # Creating a sentinel inside the compiled region is out of scope;
        # it should graph break, not crash.
        def f(x):
            # [FUNCTION DOC] HELPER: Creates a sentinel inside the function and returns sine of
            # x together with it.
            s = builtins.sentinel("inner")
            return x.sin(), s

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_f(x)

    def test_reduce_sentinel_default(self):
        # [FUNCTION DOC] Compiles a functools reduce over operator add of three tensors without
        # an initial value, where the C default for the initial argument is a sentinel in Python
        # 3.15; asserts the compiled result matches eager.
        # functools.reduce's `initial` C-default is a sentinel in 3.15. Unlike
        # user-function defaults (which are source-backed), it reaches Dynamo
        # through SourcelessBuilder via the reduce polyfill.
        def f(x):
            # [FUNCTION DOC] HELPER: Reduces the list of x, sine of x and cosine of x with
            # elementwise addition.
            return functools.reduce(operator.add, [x, x.sin(), x.cos()])

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(3, 3)
        self.assertEqual(opt_f(x), f(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
