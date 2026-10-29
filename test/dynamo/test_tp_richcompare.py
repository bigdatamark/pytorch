# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo's unified rich-comparison handling (equality, inequality and the
# four orderings) matches eager Python across constants, containers, user-defined classes, C
# extension types, tensors, symbolic shapes and special variable kinds. Defines an opaque test
# value class registered as a constant custom class at import time (global registration side
# effect) and one TestCase with comparison helpers; runs through the Dynamo test runner.
"""Tests for tp_richcompare_impl: unified comparison protocol in Dynamo."""

import operator
import unittest

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._library.opaque_object import register_custom_class


class _OpaqueVal(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Test-only opaque value class registered as a constant custom class. It
    # wraps a single payload in the val attribute and compares by payload equality instead of
    # identity, so Dynamo must treat instances as value-comparable constants.
    def __init__(self, val):
        # [FUNCTION DOC] HELPER: Stores the wrapped payload on the val attribute.
        self.val = val

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Returns the not-implemented sentinel when the other operand is
        # not an _OpaqueVal; otherwise returns whether the two payloads are equal.
        if not isinstance(other, _OpaqueVal):
            return NotImplemented
        return self.val == other.val

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hashes the payload so equal instances hash equally, keeping the
        # type usable in sets and as dict keys.
        return hash(self.val)

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string containing the class name and
        # the payload.
        return f"_OpaqueVal({self.val})"

    def __fx_repr__(self):
        # [FUNCTION DOC] HELPER: Returns a pair used by FX code generation: the source string that
        # rebuilds this value and a mapping from the class name to the class, so generated graph
        # code can resolve it.
        return (f"_OpaqueVal({self.val})", {"_OpaqueVal": _OpaqueVal})


register_custom_class(_OpaqueVal, typ="constant", hoist=True)


class TpRichcompareTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for Dynamo's unified rich-comparison protocol. Each test
    # compiles a function that compares values (constants, containers, user classes, C
    # types, tensors, special variables) with the eager backend and checks the result, or
    # the raised TypeError, against plain eager Python. Helpers drive all six comparison
    # operators in both operand orders.
    def _assert_cmp_equals(self, a, b, op, *, expect_type_error=None):
        # [FUNCTION DOC] HELPER: Checks one binary comparison. Evaluates the operator eagerly; if
        # eager raises TypeError, the compiled version (full graph, eager backend) must also raise
        # a TypeError whose message mentions an unsupported comparison, and when expect_type_error
        # is given it must be true. Otherwise (expect_type_error given) it must be false and the
        # compiled result must equal the eager one.
        # KNOWN ISSUE: the failure-message argument of the must-not-raise assertion is a function
        # object rather than a string, so the intended explanatory message is never shown.
        """Assert Dynamo's op(a, b) matches eager Python's op(a, b).

        If expect_type_error is set, also asserts that the eager behavior
        matches the expectation (True = must raise, False = must not raise).
        """
        try:
            expected = op(a, b)
        except TypeError as eager_exc:
            if expect_type_error is not None:
                self.assertTrue(
                    expect_type_error,
                    f"Expected {op.__name__}({a!r}, {b!r}) to succeed but "
                    f"eager raised TypeError: {eager_exc}",
                )

            def fn(_, _op=op, _a=a, _b=b):
                # [FUNCTION DOC] HELPER: Compiled wrapper taking an ignored dummy tensor. It
                # applies the captured operator to the captured operands and, on TypeError,
                # returns the message text instead of raising so the test can inspect it.
                try:
                    return _op(_a, _b)
                except TypeError as e:
                    return str(e)

            result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
            self.assertIn("not supported between", result)
            return

        if expect_type_error is not None:
            self.assertFalse(
                expect_type_error,
                lambda msg: f"{msg}\nExpected {op.__name__}({a!r}, {b!r}) to raise TypeError "
                f"but eager returned {expected!r}",
            )

        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function taking an ignored dummy tensor; returns the
            # captured operator applied to the captured operands.
            return op(a, b)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    _ALL_OPS = (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )
    _ORDERING_OPS = frozenset({operator.lt, operator.le, operator.gt, operator.ge})
    _LE_GE = frozenset({operator.le, operator.ge})

    def _assert_all_cmp_equals(self, a, b, *, error_ops=frozenset()):
        # [FUNCTION DOC] HELPER: Runs the single-comparison check for all six operators (eq, ne,
        # lt, le, gt, ge), with operands in both orders, each in its own subtest. Operators in
        # error_ops are expected to raise TypeError. Resets Dynamo state after every comparison so
        # compiles do not share cache entries.
        """Assert all 6 comparison ops match eager for (a, b) and (b, a).

        error_ops: set of operator functions expected to raise TypeError.
        Use _ORDERING_OPS for types that support eq/ne but not ordering.
        """
        for op in self._ALL_OPS:
            with self.subTest(op=op.__name__, order="a,b"):
                self._assert_cmp_equals(a, b, op, expect_type_error=(op in error_ops))
            torch._dynamo.reset()
            with self.subTest(op=op.__name__, order="b,a"):
                self._assert_cmp_equals(b, a, op, expect_type_error=(op in error_ops))
            torch._dynamo.reset()

    def _assert_sourceless_cmp_equals(
        self, make_a, make_b, op, *, expect_type_error=None
    ):
        # [FUNCTION DOC] HELPER: Like the single-comparison check but both operands are produced
        # by factory callables invoked inside the compiled region, so Dynamo sees objects without
        # a source. A TypeError in eager requires a compiled TypeError mentioning an unsupported
        # comparison; otherwise the compiled result must equal eager.
        """Like _assert_cmp_equals but for objects created inside the compile region."""
        try:
            expected = op(make_a(), make_b())
        except TypeError:
            if expect_type_error is not None:
                self.assertTrue(expect_type_error)

            def fn(_, _op=op, _make_a=make_a, _make_b=make_b):
                # [FUNCTION DOC] HELPER: Compiled wrapper with an ignored tensor argument. It
                # builds both operands via the captured factories, applies the operator, and
                # returns the TypeError message text if one is raised.
                try:
                    return _op(_make_a(), _make_b())
                except TypeError as e:
                    return str(e)

            result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
            self.assertIn("not supported between", result)
            return

        if expect_type_error is not None:
            self.assertFalse(expect_type_error)

        def fn(_, _op=op, _make_a=make_a, _make_b=make_b):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored tensor argument; builds
            # both operands via the captured factories and returns the operator result.
            return _op(_make_a(), _make_b())

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def _assert_all_sourceless_cmp_equals(
        self, make_a, make_b, *, error_ops=frozenset()
    ):
        # [FUNCTION DOC] HELPER: Runs the sourceless comparison check for all six operators (one
        # operand order only), each in a subtest, expecting TypeError for operators in error_ops,
        # and resets Dynamo state after each.
        """Like _assert_all_cmp_equals but for objects created inside the compile region."""
        for op in self._ALL_OPS:
            with self.subTest(op=op.__name__):
                self._assert_sourceless_cmp_equals(
                    make_a, make_b, op, expect_type_error=(op in error_ops)
                )
            torch._dynamo.reset()

    # =====================================================================
    # Constants (ConstantVariable)
    # =====================================================================

    def test_int_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal ints with all operators against eager.
        self._assert_all_cmp_equals(42, 42)

    def test_int_cmp_diff(self):
        # [FUNCTION DOC] Compares two different ints with all operators against eager.
        self._assert_all_cmp_equals(1, 2)

    def test_str_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal strings with all operators against eager.
        self._assert_all_cmp_equals("hello", "hello")

    def test_str_cmp_diff(self):
        # [FUNCTION DOC] Compares two different strings with all operators against eager.
        self._assert_all_cmp_equals("hello", "world")

    def test_float_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal floats with all operators against eager.
        self._assert_all_cmp_equals(1.5, 1.5)

    def test_float_cmp_diff(self):
        # [FUNCTION DOC] Compares two different floats with all operators against eager.
        self._assert_all_cmp_equals(1.5, 2.5)

    def test_bool_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal booleans with all operators against eager.
        self._assert_all_cmp_equals(True, True)

    def test_bool_cmp_diff(self):
        # [FUNCTION DOC] Compares true against false with all operators against eager.
        self._assert_all_cmp_equals(True, False)

    def test_mixed_numeric_cmp_equal(self):
        # [FUNCTION DOC] Compares an int with an equal-valued float using all operators.
        self._assert_all_cmp_equals(1, 1.0)

    def test_mixed_numeric_cmp_diff(self):
        # [FUNCTION DOC] Compares an int with a different float using all operators.
        self._assert_all_cmp_equals(1, 2.5)

    # =====================================================================
    # Cross-type comparison (identity fallback)
    # =====================================================================

    def test_cross_type_list_tuple(self):
        # [FUNCTION DOC] Compares a list with a tuple of the same items; equality operators
        # succeed and ordering operators must raise TypeError in eager and compiled code.
        self._assert_all_cmp_equals([1, 2], (1, 2), error_ops=self._ORDERING_OPS)

    def test_cross_type_int_str(self):
        # [FUNCTION DOC] Compares an int with a string; equality operators succeed (not equal),
        # ordering operators must raise TypeError.
        self._assert_all_cmp_equals(1, "1", error_ops=self._ORDERING_OPS)

    # =====================================================================
    # List / Tuple comparison (BaseListVariable)
    # =====================================================================

    def test_list_cmp(self):
        # [FUNCTION DOC] Compares two lists that differ in the last element with all operators.
        self._assert_all_cmp_equals([1, 2], [1, 3])

    def test_list_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal lists with all operators.
        self._assert_all_cmp_equals([1, 2, 3], [1, 2, 3])

    def test_list_cmp_length_mismatch(self):
        # [FUNCTION DOC] Compares a list with a longer list sharing the same prefix, with
        # all operators.
        self._assert_all_cmp_equals([1, 2], [1, 2, 3])

    def test_list_cmp_empty(self):
        # [FUNCTION DOC] Compares two empty lists with all operators.
        self._assert_all_cmp_equals([], [])

    def test_tuple_cmp(self):
        # [FUNCTION DOC] Compares two tuples that differ in the last element with all operators.
        self._assert_all_cmp_equals((1, 2), (1, 3))

    def test_tuple_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal tuples with all operators.
        self._assert_all_cmp_equals((1, 2, 3), (1, 2, 3))

    # =====================================================================
    # Dict comparison (ConstDictVariable)
    # =====================================================================

    def test_dict_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal dicts; ordering operators must raise TypeError.
        self._assert_all_cmp_equals(
            {1: 2, 3: 4}, {1: 2, 3: 4}, error_ops=self._ORDERING_OPS
        )

    def test_dict_cmp_diff(self):
        # [FUNCTION DOC] Compares two dicts with different contents; ordering operators must
        # raise TypeError.
        self._assert_all_cmp_equals({1: 2}, {3: 4}, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Set comparison (SetVariable)
    # =====================================================================

    def test_set_cmp_subset(self):
        # [FUNCTION DOC] Compares a set with a strict superset using all operators (set ordering
        # means subset tests).
        self._assert_all_cmp_equals({1, 2}, {1, 2, 3})

    def test_set_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal sets created in different order with all operators.
        self._assert_all_cmp_equals({1, 2, 3}, {3, 2, 1})

    def test_set_cmp_disjoint(self):
        # [FUNCTION DOC] Compares two disjoint sets with all operators.
        self._assert_all_cmp_equals({1, 2}, {3, 4})

    def test_frozenset_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal frozensets with all operators.
        self._assert_all_cmp_equals(frozenset({1, 2}), frozenset({2, 1}))

    def test_frozenset_cmp_diff(self):
        # [FUNCTION DOC] Compares a frozenset with a strict superset using all operators.
        self._assert_all_cmp_equals(frozenset({1, 2}), frozenset({1, 2, 3}))

    def test_set_cross_type(self):
        # [FUNCTION DOC] Compares a set with a list; equality succeeds, ordering must
        # raise TypeError.
        self._assert_all_cmp_equals({1, 2}, [1, 2], error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Range comparison (RangeVariable)
    # =====================================================================

    def test_range_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal ranges; ordering operators must raise TypeError.
        self._assert_all_cmp_equals(range(10), range(10), error_ops=self._ORDERING_OPS)

    def test_range_cmp_diff(self):
        # [FUNCTION DOC] Compares ranges of different length; ordering operators must
        # raise TypeError.
        self._assert_all_cmp_equals(range(10), range(5), error_ops=self._ORDERING_OPS)

    def test_range_cmp_with_step(self):
        # [FUNCTION DOC] Compares two equal ranges that have a step; ordering must
        # raise TypeError.
        self._assert_all_cmp_equals(
            range(0, 10, 2), range(0, 10, 2), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Slice comparison (SliceVariable)
    # =====================================================================

    def test_slice_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal slices with all operators.
        self._assert_all_cmp_equals(slice(1, 10), slice(1, 10))

    def test_slice_cmp_diff(self):
        # [FUNCTION DOC] Compares slices with different start values with all operators.
        self._assert_all_cmp_equals(slice(1, 10), slice(2, 10))

    def test_slice_cmp_with_step(self):
        # [FUNCTION DOC] Compares slices that differ only in step with all operators.
        self._assert_all_cmp_equals(slice(1, 10, 2), slice(1, 10, 3))

    # =====================================================================
    # Tensor comparison (TensorVariable)
    # =====================================================================

    def test_tensor_cmp(self):
        # [FUNCTION DOC] Compares two integer tensors that differ in one element with
        # all operators.
        self._assert_all_cmp_equals(torch.tensor([1, 2, 3]), torch.tensor([1, 0, 3]))

    def test_tensor_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal integer tensors with all operators.
        self._assert_all_cmp_equals(torch.tensor([1, 2, 3]), torch.tensor([1, 2, 3]))

    # =====================================================================
    # User-defined comparison methods
    # =====================================================================

    def test_user_defined_eq_and_lt(self):
        # [FUNCTION DOC] Checks a user class defining only equality and less-than: greater-than
        # works through the reflected less-than, while less-or-equal and greater-or-equal must
        # raise TypeError.
        """User type with __eq__ and __lt__: gt works via reflected lt, le/ge raise."""

        class MyObj:
            # [CLASS DOC] HELPER: Local class with a val attribute, value-based equality and
            # less-than, both returning the not-implemented sentinel for foreign types.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-MyObj operands, else
                # compares the stored values.
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-MyObj operands, else
                # tests whether the stored value is smaller.
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val < other.val

        self._assert_all_cmp_equals(MyObj(1), MyObj(2), error_ops=self._LE_GE)
        self._assert_all_cmp_equals(MyObj(42), MyObj(42), error_ops=self._LE_GE)

    def test_user_defined_eq_only(self):
        # [FUNCTION DOC] Checks a user class with equality but no ordering methods; every ordering
        # operator must raise TypeError.
        """Class with __eq__ but no ordering — ordering raises TypeError."""

        class MyObj:
            # [CLASS DOC] HELPER: Local class with a val attribute and value-based equality only.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-MyObj operands, else
                # compares the stored values.
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_cmp_equals(MyObj(1), MyObj(2), error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(MyObj(42), MyObj(42), error_ops=self._ORDERING_OPS)

    def test_user_defined_no_cmp(self):
        # [FUNCTION DOC] Checks a plain class without comparison methods: equality is
        # identity-based and ordering raises TypeError.
        """Type with no comparison methods — identity eq/ne, ordering raises."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty local class with no comparison methods.
            pass

        self._assert_all_cmp_equals(MyObj(), MyObj(), error_ops=self._ORDERING_OPS)

    def test_user_defined_cross_type(self):
        # [FUNCTION DOC] Compares an instance of a user class with an int and with an instance
        # of a different user class; equality falls back to identity (not equal) and ordering
        # raises TypeError.
        """UDOV vs constant and vs different UDOV — __eq__ returns NotImplemented,
        falls back to identity (False); ordering raises TypeError."""

        class Foo:
            # [CLASS DOC] HELPER: Local class with a val attribute whose equality returns
            # not-implemented for non-Foo operands.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-Foo operands, else
                # compares the stored values.
                if not isinstance(other, Foo):
                    return NotImplemented
                return self.val == other.val

        class Bar:
            # [CLASS DOC] HELPER: Same shape as Foo but a distinct type, so cross-type equality
            # returns not-implemented both ways.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-Bar operands, else
                # compares the stored values.
                if not isinstance(other, Bar):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_cmp_equals(Foo(1), 42, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(Foo(1), Bar(1), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Subclass priority
    # =====================================================================

    def test_subclass_priority(self):
        # [FUNCTION DOC] Checks that when the right operand is an instance of a subclass
        # overriding equality, the subclass method is tried first, matching eager results.
        """B(A) overrides __eq__ — A() == B() calls B.__eq__ first."""

        class A:
            # [CLASS DOC] HELPER: Base class whose equality always returns a fixed marker string.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns the fixed string for the base class instead of
                # a boolean.
                return "a_eq"

        class B(A):
            # [CLASS DOC] HELPER: Subclass of A overriding equality with a different
            # marker string.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns the fixed string for the subclass.
                return "b_eq"

        self._assert_all_cmp_equals(B(), A(), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Identity fallback
    # =====================================================================

    def test_identity_cmp(self):
        # [FUNCTION DOC] Checks identity-based equality for a plain class: an object equals itself
        # and differs from another instance; ordering raises TypeError.
        """object.__eq__/__ne__: identity-based for same and different objects."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty local class used for identity comparison.
            pass

        a = MyObj()
        self._assert_all_cmp_equals(a, a, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(a, MyObj(), error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Function / module comparison (identity-based)
    # =====================================================================

    def test_function_cmp(self):
        # [FUNCTION DOC] Checks that plain functions compare by identity and ordering
        # raises TypeError.
        """User functions use identity comparison."""

        def f():
            # [FUNCTION DOC] HELPER: Empty function used as a comparison operand.
            pass

        def g():
            # [FUNCTION DOC] HELPER: Second empty function, distinct from the first.
            pass

        self._assert_all_cmp_equals(f, f, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(f, g, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Exception comparison (identity-based)
    # =====================================================================

    def test_exception_cmp(self):
        # [FUNCTION DOC] Checks that two separate exception instances with the same message
        # compare by identity; ordering raises TypeError.
        self._assert_all_cmp_equals(
            ValueError("a"), ValueError("a"), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Consistency: operator.eq matches __eq__
    # =====================================================================

    def test_consistency_int(self):
        # [FUNCTION DOC] Verifies in compiled code that the equality operator on two ints gives
        # the same result as calling the equality dunder method directly.
        """operator.eq(a, b) matches a.__eq__(b) for ints."""

        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored tensor argument; returns
            # the operator result and the dunder-method result for two ints.
            a, b = 1, 2
            return operator.eq(a, b), a.__eq__(b)

        r1, r2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(r1, r2)

    def test_consistency_list(self):
        # [FUNCTION DOC] Verifies in compiled code that the equality operator on two equal lists
        # agrees with calling the list equality dunder directly.
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored tensor argument; returns
            # the operator result and the dunder-method result for two lists.
            a, b = [1, 2], [1, 2]
            return a == b, a.__eq__(b)

        r1, r2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(r1, r2)

    # =====================================================================
    # Graph-break tests
    # =====================================================================

    def _make_graph_break_class(self, *ops):
        # [FUNCTION DOC] HELPER: Builds a local class holding a val attribute, then attaches for
        # each requested operator name a dunder method that forces a graph break, returns the
        # not-implemented sentinel for foreign types, and otherwise applies the matching operator
        # function to the stored values. Returns the class.
        """Create a class where the given comparison ops contain a graph break."""

        class Obj:
            # [CLASS DOC] HELPER: Local class storing a val attribute; comparison methods are
            # attached dynamically by the enclosing helper.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

        for op in ops:

            def method(self, other, _op=op):
                # [FUNCTION DOC] HELPER: Comparison method attached to Obj; the operator name is
                # bound as a default argument. It calls the Dynamo graph-break function, returns
                # not-implemented for non-Obj operands, else applies the named operator to the two
                # stored values.
                torch._dynamo.graph_break()
                if not isinstance(other, Obj):
                    return NotImplemented
                return getattr(operator, _op)(self.val, other.val)

            setattr(Obj, f"__{op}__", method)
        return Obj

    def test_graph_break_in_eq_via_ne(self):
        # [FUNCTION DOC] Checks that the not-equal operator, falling back to a user equality
        # method that graph-breaks, still compiles and gives the right answer (equal objects are
        # not unequal).
        """ne traces into __eq__ (via generic_richcompare) which graph-breaks."""
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function adding one to the input and returning it
            # along with a not-equal test of two equal Obj instances.
            return x + 1, Obj(0) != Obj(0)

        _, ne = fn(torch.ones(1))
        self.assertFalse(ne)

    def test_graph_break_in_eq_via_eq(self):
        # [FUNCTION DOC] Checks that equality via a user method that graph-breaks gives true for
        # equal objects.
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and an
            # equality test of two equal Obj instances.
            return x + 1, Obj(1) == Obj(1)

        _, eq = fn(torch.ones(1))
        self.assertTrue(eq)

    def test_graph_break_in_eq_not_equal(self):
        # [FUNCTION DOC] Checks that equality via a graph-breaking method gives false for
        # different values.
        Obj = self._make_graph_break_class("eq")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and an
            # equality test of two different Obj instances.
            return x + 1, Obj(1) == Obj(2)

        _, eq = fn(torch.ones(1))
        self.assertFalse(eq)

    def test_graph_break_in_lt(self):
        # [FUNCTION DOC] Checks that less-than through a graph-breaking method returns true for a
        # smaller left value.
        Obj = self._make_graph_break_class("lt")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and a
            # less-than test of Obj one against Obj two.
            return x + 1, Obj(1) < Obj(2)

        _, lt = fn(torch.ones(1))
        self.assertTrue(lt)

    def test_graph_break_in_lt_false(self):
        # [FUNCTION DOC] Checks that less-than through a graph-breaking method returns false when
        # the left value is larger.
        Obj = self._make_graph_break_class("lt")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and a
            # less-than test of Obj three against Obj two.
            return x + 1, Obj(3) < Obj(2)

        _, lt = fn(torch.ones(1))
        self.assertFalse(lt)

    def test_graph_break_in_ne(self):
        # [FUNCTION DOC] Checks a not-equal method that graph-breaks returns true for
        # different values.
        Obj = self._make_graph_break_class("ne")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and a
            # not-equal test of Obj one against Obj two.
            return x + 1, Obj(1) != Obj(2)

        _, ne = fn(torch.ones(1))
        self.assertTrue(ne)

    def test_graph_break_in_gt(self):
        # [FUNCTION DOC] Checks a greater-than method that graph-breaks returns true when the left
        # value is larger.
        Obj = self._make_graph_break_class("gt")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning the input plus one and a
            # greater-than test of Obj three against Obj one.
            return x + 1, Obj(3) > Obj(1)

        _, gt = fn(torch.ones(1))
        self.assertTrue(gt)

    def test_graph_break_in_multiple_ops(self):
        # [FUNCTION DOC] Checks a class where both equality and less-than graph-break: equality is
        # false and less-than is true for values one and two.
        """Class where both __eq__ and __lt__ graph-break."""
        Obj = self._make_graph_break_class("eq", "lt")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function creating two Obj instances and returning
            # the input plus one with their equality and less-than results.
            a, b = Obj(1), Obj(2)
            return x + 1, a == b, a < b

        _, eq, lt = fn(torch.ones(1))
        self.assertFalse(eq)
        self.assertTrue(lt)

    # =====================================================================
    # Dunder method dispatch
    # =====================================================================

    def test_dunder_eq_int(self):
        # [FUNCTION DOC] Calls the equality dunder method directly on an int constant in a
        # full-graph compile and expects true.
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored argument; returns the
            # direct equality dunder call on two equal ints.
            return (42).__eq__(42)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    def test_dunder_ne_str(self):
        # [FUNCTION DOC] Calls the not-equal dunder method directly on a string constant in a
        # full-graph compile and expects true.
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored argument; returns the
            # direct not-equal dunder call on two different strings.
            return "hello".__ne__("world")

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    def test_dunder_lt_list(self):
        # [FUNCTION DOC] Calls the less-than dunder method directly on a list in a full-graph
        # compile and expects true.
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored argument; returns the
            # direct less-than dunder call on two lists.
            return [1, 2].__lt__([1, 3])

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertTrue(result)

    # =====================================================================
    # C extension type comparison (datetime, decimal)
    # =====================================================================

    def test_datetime_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal date objects (a C extension type) with all operators.
        import datetime

        self._assert_all_cmp_equals(
            datetime.date(2024, 1, 1), datetime.date(2024, 1, 1)
        )

    def test_datetime_cmp_diff(self):
        # [FUNCTION DOC] Compares two different date objects with all operators.
        import datetime

        self._assert_all_cmp_equals(
            datetime.date(2024, 1, 1), datetime.date(2024, 6, 1)
        )

    def test_decimal_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal Decimal objects with all operators.
        import decimal

        self._assert_all_cmp_equals(decimal.Decimal("1.5"), decimal.Decimal("1.5"))

    def test_decimal_cmp_diff(self):
        # [FUNCTION DOC] Compares two different Decimal objects with all operators.
        import decimal

        self._assert_all_cmp_equals(decimal.Decimal("1.0"), decimal.Decimal("2.0"))

    # =====================================================================
    # Enum comparison
    # =====================================================================

    def test_enum_cmp(self):
        # [FUNCTION DOC] Compares enum members (same member and different members) with all
        # operators; ordering must raise TypeError.
        from enum import Enum

        class Color(Enum):
            # [CLASS DOC] HELPER: Local two-member enum used as comparison operands.
            RED = 1
            BLUE = 2

        self._assert_all_cmp_equals(Color.RED, Color.RED, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(Color.RED, Color.BLUE, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # dict_keys / dict_items comparison (set-like)
    # =====================================================================

    def test_dict_keys_cmp(self):
        # [FUNCTION DOC] Compares dict key views created inside the compiled region (equal key
        # sets, then a subset) with all operators.
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: {2: "x", 1: "y"}.keys(),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a"}.keys(),
            lambda: {1: "x", 2: "y"}.keys(),
        )

    # =====================================================================
    # NoneType
    # =====================================================================

    def test_none_cmp(self):
        # [FUNCTION DOC] Compares None with None; ordering operators must raise TypeError.
        self._assert_all_cmp_equals(None, None, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Comparison with None (common pattern: obj == None / obj is None)
    # =====================================================================

    def test_obj_eq_none(self):
        # [FUNCTION DOC] Compares an int with None; ordering operators must raise TypeError.
        self._assert_all_cmp_equals(42, None, error_ops=self._ORDERING_OPS)

    def test_list_eq_none(self):
        # [FUNCTION DOC] Compares a list with None; ordering operators must raise TypeError.
        self._assert_all_cmp_equals([1, 2], None, error_ops=self._ORDERING_OPS)

    def test_str_eq_none(self):
        # [FUNCTION DOC] Compares a string with None; ordering operators must raise TypeError.
        self._assert_all_cmp_equals("hello", None, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Sourceless objects (created inside compile region)
    # =====================================================================

    def test_sourceless_udov_eq(self):
        # [FUNCTION DOC] Compares instances of a user class created inside the compiled region
        # (equal and unequal values); ordering must raise TypeError.
        class MyObj:
            # [CLASS DOC] HELPER: Local class with a val attribute and value-based equality.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value on the val attribute.
                self.val = val

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-MyObj operands, else
                # compares the stored values.
                if not isinstance(other, MyObj):
                    return NotImplemented
                return self.val == other.val

        self._assert_all_sourceless_cmp_equals(
            lambda: MyObj(1), lambda: MyObj(1), error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: MyObj(1), lambda: MyObj(2), error_ops=self._ORDERING_OPS
        )

    def test_sourceless_list_eq(self):
        # [FUNCTION DOC] Compares lists built inside the compiled region, equal and unequal, with
        # all operators.
        self._assert_all_sourceless_cmp_equals(lambda: [1, 2], lambda: [1, 2])
        self._assert_all_sourceless_cmp_equals(lambda: [1, 2], lambda: [3, 4])

    # =====================================================================
    # NaN comparison
    # =====================================================================

    def test_nan_cmp_self(self):
        # [FUNCTION DOC] Compares a NaN float with itself using all operators.
        nan = float("nan")
        self._assert_all_cmp_equals(nan, nan)

    def test_nan_cmp_value(self):
        # [FUNCTION DOC] Compares a NaN float with a regular float using all operators.
        nan = float("nan")
        self._assert_all_cmp_equals(nan, 1.0)

    # =====================================================================
    # torch.Size comparison (tuple subclass, common in shape ops)
    # =====================================================================

    def test_torch_size_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal torch.Size objects with all operators.
        self._assert_all_cmp_equals(torch.Size([1, 2, 3]), torch.Size([1, 2, 3]))

    def test_torch_size_cmp_diff(self):
        # [FUNCTION DOC] Compares two differing torch.Size objects with all operators.
        self._assert_all_cmp_equals(torch.Size([1, 2]), torch.Size([1, 3]))

    # =====================================================================
    # set vs frozenset cross-comparison
    # =====================================================================

    def test_set_frozenset_cmp(self):
        # [FUNCTION DOC] Compares a set with an equal frozenset using all operators.
        self._assert_all_cmp_equals({1, 2, 3}, frozenset({1, 2, 3}))

    def test_set_frozenset_cmp_diff(self):
        # [FUNCTION DOC] Compares a set with a larger frozenset using all operators.
        self._assert_all_cmp_equals({1, 2}, frozenset({1, 2, 3}))

    # =====================================================================
    # Nested container comparison
    # =====================================================================

    def test_nested_list_cmp(self):
        # [FUNCTION DOC] Compares equal lists that contain nested lists with all operators.
        self._assert_all_cmp_equals([1, [2, 3]], [1, [2, 3]])

    def test_nested_list_cmp_diff(self):
        # [FUNCTION DOC] Compares lists whose nested lists differ with all operators.
        self._assert_all_cmp_equals([1, [2, 3]], [1, [2, 4]])

    def test_nested_tuple_cmp(self):
        # [FUNCTION DOC] Compares equal tuples with nested tuples using all operators.
        self._assert_all_cmp_equals((1, (2, 3)), (1, (2, 3)))

    # =====================================================================
    # Deque comparison (DequeVariable)
    # =====================================================================

    def test_deque_cmp(self):
        # [FUNCTION DOC] Compares deques created inside the compiled region, equal and different,
        # with all operators.
        from collections import deque

        self._assert_all_sourceless_cmp_equals(
            lambda: deque([1, 2, 3]), lambda: deque([1, 2, 3])
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: deque([1, 2]), lambda: deque([1, 3])
        )

    def test_deque_cmp_cross_type(self):
        # [FUNCTION DOC] Compares a deque with a list inside the compiled region; ordering must
        # raise TypeError.
        """deque vs list: NotImplemented (different types)."""
        from collections import deque

        self._assert_all_sourceless_cmp_equals(
            lambda: deque([1, 2]), lambda: [1, 2], error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # List subclass comparison (UserDefinedListVariable)
    # =====================================================================

    def test_list_subclass_cmp(self):
        # [FUNCTION DOC] Compares instances of a list subclass without its own equality, equal and
        # different, inside the compiled region.
        """list subclass without __eq__ uses list's comparison."""

        class MyList(list):
            # [CLASS DOC] HELPER: Empty list subclass.
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MyList([1, 2, 3]), lambda: MyList([1, 2, 3])
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: MyList([1, 2]), lambda: MyList([1, 3])
        )

    def test_list_subclass_vs_list(self):
        # [FUNCTION DOC] Compares a list subclass instance with a plain list of the same items
        # inside the compiled region.
        """MyList vs plain list."""

        class MyList(list):
            # [CLASS DOC] HELPER: Empty list subclass.
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MyList([1, 2, 3]), lambda: [1, 2, 3]
        )

    def test_immutable_list_cmp(self):
        # [FUNCTION DOC] Compares FX immutable lists, equal and different, created inside the
        # compiled region.
        """torch.fx.immutable_collections.immutable_list comparison."""
        from torch.fx.immutable_collections import immutable_list

        self._assert_all_sourceless_cmp_equals(
            lambda: immutable_list([1, 2, 3]), lambda: immutable_list([1, 2, 3])
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: immutable_list([1, 2]), lambda: immutable_list([1, 3])
        )

    # =====================================================================
    # dict_items comparison (set-like)
    # =====================================================================

    def test_dict_items_cmp(self):
        # [FUNCTION DOC] Compares dict item views built in the compiled region: equal items and
        # items with differing values.
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.items(),
            lambda: {1: "a", 2: "b"}.items(),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a"}.items(),
            lambda: {1: "b"}.items(),
        )

    def test_dict_keys_vs_set(self):
        # [FUNCTION DOC] Compares a dict key view against a plain set in the compiled region, for
        # equal and for subset cases.
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: {1, 2},
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a"}.keys(),
            lambda: {1, 2},
        )

    def test_dict_keys_vs_frozenset(self):
        # [FUNCTION DOC] Compares a dict key view against an equal frozenset in the
        # compiled region.
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: frozenset({1, 2}),
        )

    def test_dict_keys_vs_custom_set(self):
        # [FUNCTION DOC] Compares a dict key view with a set subclass whose length and containment
        # methods lie; the compiled result must match eager, so the polyfill must honor those
        # overridden methods.
        """dict_keys compared against a set subclass with custom __len__/__contains__.

        CPython's dictview_richcompare calls PyObject_Size and
        PySequence_Contains which go through sq_length/sq_contains slots.
        Our polyfill must respect these overrides.
        """

        class LyingSet(set):
            # [CLASS DOC] HELPER: Set subclass whose length always reports 999 and whose
            # containment test is always true.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed wrong length.
                return 999

            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Reports every item as contained.
                return True

        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: LyingSet({1, 2}),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: "a", 2: "b"}.keys(),
            lambda: LyingSet({1}),
        )

    # =====================================================================
    # bytes comparison
    # =====================================================================

    def test_bytes_cmp(self):
        # [FUNCTION DOC] Compares two different byte strings with all operators.
        self._assert_all_cmp_equals(b"hello", b"world")

    def test_bytes_cmp_equal(self):
        # [FUNCTION DOC] Compares two equal byte strings with all operators.
        self._assert_all_cmp_equals(b"hello", b"hello")

    # =====================================================================
    # complex comparison (eq/ne only, ordering raises)
    # =====================================================================

    def test_complex_cmp(self):
        # [FUNCTION DOC] Compares two equal complex numbers; ordering operators must
        # raise TypeError.
        self._assert_all_cmp_equals(1 + 2j, 1 + 2j, error_ops=self._ORDERING_OPS)

    def test_complex_cmp_diff(self):
        # [FUNCTION DOC] Compares two different complex numbers; ordering must raise TypeError.
        self._assert_all_cmp_equals(1 + 2j, 3 + 4j, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # re.Pattern comparison (C extension type)
    # =====================================================================

    def test_re_pattern_cmp_equal(self):
        # [FUNCTION DOC] Compares two compiled regex patterns with the same source; ordering must
        # raise TypeError.
        import re

        self._assert_all_cmp_equals(
            re.compile(r"\d+"), re.compile(r"\d+"), error_ops=self._ORDERING_OPS
        )

    def test_re_pattern_cmp_diff(self):
        # [FUNCTION DOC] Compares regex patterns with different sources; ordering must
        # raise TypeError.
        import re

        self._assert_all_cmp_equals(
            re.compile(r"\d+"), re.compile(r"\w+"), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # allow_c_slot API for C extension type comparison
    # =====================================================================

    def test_allow_c_slot_api(self):
        # [FUNCTION DOC] Uses an in-memory sqlite database to get two Row objects. Before
        # registering the Row type, comparing them in a full-graph compile must raise Unsupported
        # mentioning an untraceable C comparison. After resetting Dynamo and calling the
        # allow_c_slot API for the Row type, equality comparisons compile and match eager. Side
        # effect: registers a C type globally.
        """allow_c_slot registers a C type's comparison, then it works."""
        import sqlite3

        from torch._dynamo.exc import Unsupported

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a, b)")
        conn.execute("INSERT INTO t VALUES (1, 2)")
        conn.row_factory = sqlite3.Row
        row1 = conn.execute("SELECT * FROM t").fetchone()
        row2 = conn.execute("SELECT * FROM t").fetchone()

        # Before registration: graph-breaks
        with self.assertRaisesRegex(Unsupported, "Untraceable C tp_richcompare"):
            torch.compile(lambda _: row1 == row2, backend="eager", fullgraph=True)(
                torch.tensor(0)
            )

        torch._dynamo.reset()

        # After registration: works
        torch._dynamo.allow_c_slot(sqlite3.Row)
        self._assert_cmp_equals(row1, row1, operator.eq)
        self._assert_cmp_equals(row1, row2, operator.eq)

    # =====================================================================
    # __eq__ returning NotImplemented visible to user code
    # =====================================================================

    def test_eq_returns_not_implemented(self):
        # [FUNCTION DOC] Verifies that a direct equality dunder call with an incompatible operand
        # is visible to compiled code as the not-implemented sentinel.
        """a.__eq__(b) can return NotImplemented when types are incompatible."""

        class MyObj:
            # [CLASS DOC] HELPER: Local class whose equality returns not-implemented for foreign
            # types and true otherwise.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns not-implemented for non-MyObj operands,
                # else true.
                if not isinstance(other, MyObj):
                    return NotImplemented
                return True

        obj = MyObj()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored argument; returns whether
            # calling equality with an int yields the not-implemented sentinel.
            result = obj.__eq__(42)
            return result is NotImplemented

        self.assertTrue(fn(torch.tensor(0)))

    # =====================================================================
    # FakeIdVariable comparison (id() on sourceless objects)
    # =====================================================================

    def test_fake_id_comparison_graph_breaks(self):
        # [FUNCTION DOC] Checks that comparing the id of an object created in the compiled region
        # with a number graph-breaks (fake ids are not sound) and the function still returns the
        # else branch result.
        """FakeIdVariable comparisons are not guaranteed to be sound (the
        compile-time fake id may not match runtime values), so we
        unconditionally graph-break.
        See test_id.py for additional FakeId tests.
        """

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function that builds a list, compares its id with a
            # constant and returns the input plus one or plus two accordingly.
            lst = [1, 2, 3]
            if id(lst) == 42:
                return x + 1
            return x + 2

        result = fn(torch.tensor(0.0))
        expected = torch.tensor(2.0)
        self.assertEqual(result, expected)

    # =====================================================================
    # Comparison driving control flow (guard behavior)
    # =====================================================================

    def test_comparison_guard(self):
        # [FUNCTION DOC] Checks that comparing an argument to a constant drives control flow and
        # installs a guard, so two different argument values each give the right branch result.
        """Comparison with a constant drives control flow (installs guard)."""

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, val):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one if the value
            # argument equals zero, else input plus two.
            if val == 0:
                return x + 1
            return x + 2

        self.assertEqual(fn(torch.tensor(0.0), 0), torch.tensor(1.0))
        self.assertEqual(fn(torch.tensor(0.0), 1), torch.tensor(2.0))

    # =====================================================================
    # Sourceless containers (created inside compile region)
    # =====================================================================

    def test_sourceless_dict_eq(self):
        # [FUNCTION DOC] Compares dicts built inside the compiled region, equal and different;
        # ordering must raise TypeError.
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: 2}, lambda: {1: 2}, error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: {1: 2}, lambda: {3: 4}, error_ops=self._ORDERING_OPS
        )

    def test_sourceless_set_eq(self):
        # [FUNCTION DOC] Compares sets built inside the compiled region, equal and disjoint, with
        # all operators.
        self._assert_all_sourceless_cmp_equals(lambda: {1, 2}, lambda: {2, 1})
        self._assert_all_sourceless_cmp_equals(lambda: {1, 2}, lambda: {3, 4})

    def test_sourceless_tuple_eq(self):
        # [FUNCTION DOC] Compares tuples built inside the compiled region, equal and different,
        # with all operators.
        self._assert_all_sourceless_cmp_equals(lambda: (1, 2), lambda: (1, 2))
        self._assert_all_sourceless_cmp_equals(lambda: (1, 2), lambda: (1, 3))

    def test_sourceless_range_eq(self):
        # [FUNCTION DOC] Compares ranges built inside the compiled region, equal and different;
        # ordering must raise TypeError.
        self._assert_all_sourceless_cmp_equals(
            lambda: range(10), lambda: range(10), error_ops=self._ORDERING_OPS
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: range(10), lambda: range(5), error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # Builtin function comparison (BaseBuiltinVariable)
    # =====================================================================

    def test_builtin_cmp(self):
        # [FUNCTION DOC] Compares builtin callables (same builtin, and a builtin against a type);
        # ordering must raise TypeError.
        self._assert_all_cmp_equals(len, len, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(len, int, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # Module comparison (PythonModuleVariable)
    # =====================================================================

    def test_module_cmp(self):
        # [FUNCTION DOC] Checks module objects compare by identity in compiled code: torch equals
        # itself and differs from math.
        import math

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one and the equality
            # of torch with itself and the inequality of torch with math.
            eq = torch == torch
            ne = torch != math
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # torch op comparison (BaseTorchVariable)
    # =====================================================================

    def test_torch_op_cmp(self):
        # [FUNCTION DOC] Compares torch functions (same function, different functions); ordering
        # must raise TypeError.
        self._assert_all_cmp_equals(torch.sin, torch.sin, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(torch.sin, torch.cos, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # TypingVariable comparison
    # =====================================================================

    def test_typing_cmp(self):
        # [FUNCTION DOC] Checks typing aliases in compiled code: List equals List and differs
        # from Dict.
        import typing

        List = typing.List  # noqa: UP006
        Dict = typing.Dict  # noqa: UP006

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one and the equality
            # and inequality results for the two typing aliases.
            eq = List == List
            ne = List != Dict
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # MethodWrapperVariable comparison
    # =====================================================================

    def test_method_wrapper_cmp(self):
        # [FUNCTION DOC] Compares two method-wrapper objects obtained from the same list type;
        # ordering must raise TypeError.
        self._assert_all_cmp_equals(
            [].__add__, [].__add__, error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # functools.partial comparison (identity-based)
    # =====================================================================

    def test_partial_cmp(self):
        # [FUNCTION DOC] Checks functools.partial objects compare by identity: a partial equals
        # itself and differs from an equivalent separate partial.
        import functools

        p1 = functools.partial(int, base=2)
        p2 = functools.partial(int, base=2)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one, self-equality of
            # one partial, and inequality between two partials.
            eq = p1 == p1
            ne = p1 != p2
            return x + 1, eq, ne

        _, eq, ne = fn(torch.tensor(0.0))
        self.assertTrue(eq)
        self.assertTrue(ne)

    # =====================================================================
    # NN module comparison (identity-based)
    # =====================================================================

    def test_nn_module_cmp(self):
        # [FUNCTION DOC] Compares nn.Linear modules by identity (same module, two different
        # modules); ordering must raise TypeError.
        m1 = torch.nn.Linear(3, 3)
        m2 = torch.nn.Linear(3, 3)
        self._assert_all_cmp_equals(m1, m1, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(m1, m2, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # SymNode comparison
    # =====================================================================

    def test_symnode_cmp(self):
        # [FUNCTION DOC] Checks that comparisons on a symbolic shape value compile in full-graph
        # mode and give expected truth values.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function taking a tensor; returns self-equality,
            # self-inequality, a less-than against 100 and a greater-or-equal against 1 for the
            # first dimension size.
            s = x.shape[0]
            return s == s, s != s, s < 100, s >= 1

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertTrue(result[0])
        self.assertFalse(result[1])
        self.assertTrue(result[2])
        self.assertTrue(result[3])

    # =====================================================================
    # OrderedDict comparison (UserDefinedDictVariable)
    # =====================================================================

    def test_ordered_dict_cmp(self):
        # [FUNCTION DOC] Compares OrderedDicts created in the compiled region: equal contents in
        # the same order, and the same contents in different order (order-sensitive equality);
        # ordering must raise TypeError.
        """OrderedDict structural equality (order-sensitive)."""
        from collections import OrderedDict

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedDict(a=1, b=2),
            lambda: OrderedDict(a=1, b=2),
            error_ops=self._ORDERING_OPS,
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedDict(a=1, b=2),
            lambda: OrderedDict(b=2, a=1),
            error_ops=self._ORDERING_OPS,
        )

    def test_counter_cmp(self):
        # [FUNCTION DOC] Compares Counter objects where one is a sub-multiset of the other, using
        # all operators (Counter defines Python-level ordering).
        """Counter defines Python-level ordering (subset/superset)."""
        from collections import Counter

        self._assert_all_sourceless_cmp_equals(
            lambda: Counter(a=3, b=1),
            lambda: Counter(a=3, b=1, c=1),
        )

    def test_dict_subclass_cmp_bypasses_dunders(self):
        # [FUNCTION DOC] Checks that comparing dict subclass instances does not call their
        # overridden item-access, length or containment methods (which raise errors),
        # matching eager.
        """dict comparison uses C struct access, not __getitem__/__len__."""

        class TrapDict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose item access, length and containment methods
            # raise RuntimeError if invoked.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError; must not be called
                # by comparison.
                raise RuntimeError("__getitem__ should not be called")

            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError; must not be called
                # by comparison.
                raise RuntimeError("__len__ should not be called")

            def __contains__(self, key):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError; must not be called
                # by comparison.
                raise RuntimeError("__contains__ should not be called")

        self._assert_all_sourceless_cmp_equals(
            lambda: TrapDict(a=1, b=2),
            lambda: TrapDict(a=1, b=2),
            error_ops=self._ORDERING_OPS,
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: TrapDict(a=1),
            lambda: TrapDict(a=1, b=2),
            error_ops=self._ORDERING_OPS,
        )

    def test_set_subclass_cmp_bypasses_dunders(self):
        # [FUNCTION DOC] Checks that comparing set subclass instances does not call their
        # overridden length or containment methods (which raise errors), matching eager.
        """set comparison uses C struct access, not __len__/__contains__."""

        class TrapSet(set):
            # [CLASS DOC] HELPER: Set subclass whose length and containment methods raise
            # RuntimeError if invoked.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError; must not be called
                # by comparison.
                raise RuntimeError("__len__ should not be called")

            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError; must not be called
                # by comparison.
                raise RuntimeError("__contains__ should not be called")

        self._assert_all_sourceless_cmp_equals(
            lambda: TrapSet({1, 2, 3}),
            lambda: TrapSet({3, 2, 1}),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: TrapSet({1, 2}),
            lambda: TrapSet({1, 2, 3}),
        )

    # =====================================================================
    # namedtuple comparison (UserDefinedTupleVariable)
    # =====================================================================

    def test_namedtuple_cmp(self):
        # [FUNCTION DOC] Compares namedtuple instances (equal and different) built in the compiled
        # region with all operators.
        """namedtuple structural equality."""
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        self._assert_all_sourceless_cmp_equals(lambda: Point(1, 2), lambda: Point(1, 2))
        self._assert_all_sourceless_cmp_equals(lambda: Point(1, 2), lambda: Point(3, 4))

    # =====================================================================
    # MappingProxy comparison (identity-based)
    # =====================================================================

    def test_mappingproxy_cmp(self):
        # [FUNCTION DOC] Compares a mapping proxy with itself; ordering must raise TypeError.
        from types import MappingProxyType

        d = {1: 2}
        mp = MappingProxyType(d)
        self._assert_all_cmp_equals(mp, mp, error_ops=self._ORDERING_OPS)

    def test_mappingproxy_vs_dict(self):
        # [FUNCTION DOC] Compares a plain dict with a mapping proxy of equal contents; ordering
        # must raise TypeError.
        """MappingProxyType delegates comparison to its underlying mapping."""
        from types import MappingProxyType

        d = {1: "a"}
        p = MappingProxyType({1: "a"})
        self._assert_all_cmp_equals(d, p, error_ops=self._ORDERING_OPS)

    def test_mappingproxy_vs_mappingproxy(self):
        # [FUNCTION DOC] Compares two mapping proxies with equal contents; ordering must
        # raise TypeError.
        """Two MappingProxyType with equal contents compare equal."""
        from types import MappingProxyType

        a = MappingProxyType({1: "a"})
        b = MappingProxyType({1: "a"})
        self._assert_all_cmp_equals(a, b, error_ops=self._ORDERING_OPS)

    # =====================================================================
    # dict_values comparison (identity-based, no tp_richcompare)
    # =====================================================================

    def test_dict_values_cmp(self):
        # [FUNCTION DOC] Checks that a dict values view equals itself (identity comparison, no
        # rich-compare slot) in a full-graph compile.
        """dict_values uses identity comparison (no tp_richcompare)."""

        @torch.compile(backend="eager", fullgraph=True)
        def fn(_):
            # [FUNCTION DOC] HELPER: Compiled function with an ignored argument; builds a dict,
            # takes its values view and returns whether the view equals itself.
            d = {1: "a", 2: "b"}
            v = d.values()
            return v == v

        self.assertTrue(fn(torch.tensor(0)))

    # =====================================================================
    # Tuple/list comparison with symbolic elements
    # =====================================================================

    def test_tuple_cmp_symbolic_shapes_same(self):
        # [FUNCTION DOC] Checks that a tensor shape compared with itself is true in a
        # full-graph compile.
        """Tuple comparison of symbolic shapes from the same tensor."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning whether a tensor shape
            # equals itself.
            return x.shape == x.shape

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3, 4))
        self.assertTrue(result)

    def test_tuple_cmp_symbolic_shapes_equal(self):
        # [FUNCTION DOC] Checks that shapes of two tensors with the same sizes compare equal in a
        # full-graph compile.
        """Tuple comparison where symbolic shapes happen to be equal."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning whether the shapes of two tensors
            # are equal.
            return x.shape == y.shape

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn(3, 4), torch.randn(3, 4)
        )
        self.assertTrue(result)

    def test_tuple_cmp_symbolic_shapes_ne(self):
        # [FUNCTION DOC] Checks that a not-equal test on the shapes of two same-sized tensors
        # drives a branch and the else path (input plus two) is taken.
        """Tuple != comparison with symbolic shapes."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one if the two shapes
            # differ, else input plus two.
            if x.shape != y.shape:
                return x + 1
            return x + 2

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager")(x, torch.randn(3, 4))
        self.assertEqual(result, x + 2)

    def test_tuple_cmp_symbolic_shapes_dynamic(self):
        # [FUNCTION DOC] Same as the shape not-equal branch test but compiled with dynamic shapes,
        # so the comparison is symbolic; expects the equal-shape path result.
        """Tuple comparison with fully dynamic shapes produces symbolic result."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning input plus one if the two shapes
            # differ, else input plus two.
            if x.shape != y.shape:
                return x + 1
            return x + 2

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager", dynamic=True)(x, torch.randn(3, 4))
        self.assertEqual(result, x + 2)

    def test_list_cmp_symbolic_shapes(self):
        # [FUNCTION DOC] Checks that lists built from tensor shapes compare equal and select the
        # equal-shape branch.
        """List comparison with symbolic shape elements."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function converting both shapes to lists and
            # returning input plus one if they differ, else plus two.
            a = list(x.shape)
            b = list(y.shape)
            if a != b:
                return x + 1
            return x + 2

        x = torch.randn(3, 4)
        result = torch.compile(fn, backend="eager")(x, torch.randn(3, 4))
        self.assertEqual(result, x + 2)

    def test_tuple_cmp_symbolic_shapes_fullgraph(self):
        # [FUNCTION DOC] Checks equality and inequality of shapes of two same-sized tensors
        # compile without graph breaks: equal is true and not-equal is false.
        """Shape tuple eq/ne with fullgraph=True: no graph break needed."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning the equality and inequality of
            # two tensor shapes.
            return x.shape == y.shape, x.shape != y.shape

        eq, ne = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.randn(3, 4), torch.randn(3, 4)
        )
        self.assertTrue(eq)
        self.assertFalse(ne)

    def test_tuple_cmp_symbolic_mixed_constant(self):
        # [FUNCTION DOC] Checks equality of tuples mixing a constant and a symbolic size compiles
        # in full-graph mode and gives true.
        """Tuple with mix of constant and symbolic elements."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function building two tuples of a constant and the
            # first dimension size and returning their equality.
            a = (1, x.shape[0])
            b = (1, x.shape[0])
            return a == b

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertTrue(result)

    def test_tuple_cmp_symbolic_shapes_ordering(self):
        # [FUNCTION DOC] Checks ordering operators between a tensor shape and constant tuples:
        # greater-or-equal, less-or-equal and greater-than all hold for a three by four tensor.
        """Tuple ordering (>=, <=, >, <) with symbolic shape elements."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function returning three ordering comparisons of the
            # shape against constant tuples.
            return x.shape >= (3, 4), x.shape <= (3, 4), x.shape > (2, 5)

        ge, le, gt = torch.compile(fn, backend="eager")(torch.randn(3, 4))
        self.assertTrue(ge)
        self.assertTrue(le)
        self.assertTrue(gt)

    def test_tuple_cmp_symbolic_shapes_ordering_two_tensors(self):
        # [FUNCTION DOC] Checks ordering between shapes of tensors sized three by four and three
        # by five: greater-or-equal false, less-than true, equal false.
        """Ordering between two tensor shapes."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled function returning greater-or-equal, less-than and
            # equality of two tensor shapes.
            return x.shape >= y.shape, x.shape < y.shape, x.shape == y.shape

        ge, lt, eq = torch.compile(fn, backend="eager")(
            torch.randn(3, 4), torch.randn(3, 5)
        )
        self.assertFalse(ge)
        self.assertTrue(lt)
        self.assertFalse(eq)

    # =====================================================================
    # Sourceless torch.Size
    # =====================================================================

    def test_sourceless_torch_size(self):
        # [FUNCTION DOC] Compares torch.Size objects built in the compiled region, equal and
        # different, with all operators.
        self._assert_all_sourceless_cmp_equals(
            lambda: torch.Size([1, 2, 3]),
            lambda: torch.Size([1, 2, 3]),
        )
        self._assert_all_sourceless_cmp_equals(
            lambda: torch.Size([1, 2]),
            lambda: torch.Size([1, 3]),
        )

    # =====================================================================
    # Pybind11 enum comparison (DispatchKey, TransformType)
    # =====================================================================

    def test_pybind11_enum_cmp(self):
        # [FUNCTION DOC] Compares pybind11 DispatchKey enum values (same and different); ordering
        # must raise TypeError.
        """Pybind11 enums with C-level __eq__: constant-folded via allowlist."""
        dk_cpu = torch.DispatchKey.CPU
        dk_cuda = torch.DispatchKey.CUDA
        self._assert_all_cmp_equals(dk_cpu, dk_cpu, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(dk_cpu, dk_cuda, error_ops=self._ORDERING_OPS)

    def test_pybind11_enum_non_singleton(self):
        # [FUNCTION DOC] Compares a DispatchKey returned from a C++ call (possibly not the cached
        # singleton) with the CPU key; ordering must raise TypeError.
        """Pybind11 enum values returned from C++ may not be the cached singleton."""
        ks = torch._C.DispatchKeySet(torch.DispatchKey.CPU)
        dk = ks.highestPriorityTypeId()
        self._assert_all_cmp_equals(
            dk, torch.DispatchKey.CPU, error_ops=self._ORDERING_OPS
        )

    # =====================================================================
    # OrderedSet comparison
    # =====================================================================

    def test_ordered_set_cmp_equal(self):
        # [FUNCTION DOC] Compares two OrderedSets with the same items in different order, created
        # in the compiled region, with all operators.
        """OrderedSet vs OrderedSet: structural equality via SetVariable."""
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2, 3]),
            lambda: OrderedSet([3, 2, 1]),
        )

    def test_ordered_set_cmp_subset(self):
        # [FUNCTION DOC] Compares an OrderedSet with a larger one containing it, with
        # all operators.
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2]),
            lambda: OrderedSet([1, 2, 3]),
        )

    def test_ordered_set_vs_set(self):
        # [FUNCTION DOC] Compares an OrderedSet with a plain set of the same items, with
        # all operators.
        """OrderedSet vs plain set: cross-type accepted by SetVariable."""
        from torch.utils._ordered_set import OrderedSet

        self._assert_all_sourceless_cmp_equals(
            lambda: OrderedSet([1, 2, 3]),
            lambda: {3, 2, 1},
        )

    # =====================================================================
    # UserDefinedSetVariable cross-type comparison
    # =====================================================================

    def test_user_defined_set_cmp(self):
        # [FUNCTION DOC] Compares two instances of a set subclass with equal contents, created in
        # the compiled region.
        """class MySet(set) vs MySet: unwraps to SetVariable."""

        class MySet(set):
            # [CLASS DOC] HELPER: Empty set subclass.
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2, 3}),
            lambda: MySet({3, 2, 1}),
        )

    def test_user_defined_set_vs_plain_set(self):
        # [FUNCTION DOC] Compares a set subclass instance with an equal plain set, created in the
        # compiled region.
        """class MySet(set) vs set: cross-type comparison."""

        class MySet(set):
            # [CLASS DOC] HELPER: Empty set subclass.
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2, 3}),
            lambda: {3, 2, 1},
        )

    def test_user_defined_set_vs_frozenset(self):
        # [FUNCTION DOC] Compares a set subclass instance with a larger frozenset, created in the
        # compiled region.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty set subclass.
            pass

        self._assert_all_sourceless_cmp_equals(
            lambda: MySet({1, 2}),
            lambda: frozenset({1, 2, 3}),
        )

    # =====================================================================
    # IS_OP (is / is not)
    # =====================================================================

    def test_is_op(self):
        # [FUNCTION DOC] Checks identity tests (is and is not) on a None check and on
        # self-identity in a full-graph compile; expected result is false, true, true.
        """IS_OP dispatches directly, not through COMPARE_OP's graph-break wrapper."""
        a = [1, 2, 3]

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Compiled function returning whether an object is None, is not
            # None, and whether the tensor is itself.
            return obj is None, obj is not None, x is x

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0), a)
        self.assertEqual(result, (False, True, True))

    # =====================================================================
    # Opaque object comparison (CustomClassObjectVariable)
    # =====================================================================

    def test_opaque_object_cmp(self):
        # [FUNCTION DOC] Compares opaque custom-class values: equal payloads in separate
        # instances, and different payloads; equality is value-based and ordering must
        # raise TypeError.
        """Opaque objects use value-based __eq__, not identity."""
        a1 = _OpaqueVal(1)
        a2 = _OpaqueVal(1)
        b = _OpaqueVal(2)
        self._assert_all_cmp_equals(a1, a2, error_ops=self._ORDERING_OPS)
        self._assert_all_cmp_equals(a1, b, error_ops=self._ORDERING_OPS)

    def test_opaque_object_tuple_cmp(self):
        # [FUNCTION DOC] Compares tuples of opaque values (equal and differing) with equality and
        # inequality only, resetting Dynamo after each case.
        """Tuple of opaque objects: mirrors DTensor's placements comparison."""
        p1 = (_OpaqueVal(1), _OpaqueVal(2))
        p2 = (_OpaqueVal(1), _OpaqueVal(2))
        p3 = (_OpaqueVal(1), _OpaqueVal(3))

        for op in (operator.eq, operator.ne):
            with self.subTest(op=op.__name__):
                self._assert_cmp_equals(p1, p2, op)
            torch._dynamo.reset()
            with self.subTest(op=op.__name__, args="diff"):
                self._assert_cmp_equals(p1, p3, op)
            torch._dynamo.reset()

    # =====================================================================
    # GetAttrVariable resolution
    # =====================================================================

    def test_getattr_bound_method_cmp(self):
        # [FUNCTION DOC] Checks that comparing a bound method attribute of a tensor against a
        # previously captured method is true under a counting backend.
        """tensor.data_ptr == captured_data_ptr after graph break."""
        t = torch.randn(5, 5)
        data_ptr = t.data_ptr

        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def fn(copy, dp):
            # [FUNCTION DOC] HELPER: Compiled function returning whether the data_ptr bound method
            # of the first argument equals the second argument.
            return copy.data_ptr == dp

        self.assertTrue(fn(t, data_ptr))

    # =====================================================================
    # Higher-order operator comparison (TorchHigherOrderOperatorVariable)
    # =====================================================================

    def test_hop_eq(self):
        # [FUNCTION DOC] Checks that a higher-order operator object compares equal to itself in a
        # full-graph compile and matches eager.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the flex_attention higher-order op
            # equals itself.
            return (
                torch._higher_order_ops.flex_attention
                == torch._higher_order_ops.flex_attention
            )

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    def test_hop_ne(self):
        # [FUNCTION DOC] Checks that two different higher-order operator objects compare unequal
        # in a full-graph compile and match eager.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the flex_attention operator differs from the
            # cond operator.
            return (
                torch._higher_order_ops.flex_attention != torch._higher_order_ops.cond
            )

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    # =====================================================================
    # Context manager comparison (ContextWrappingVariable)
    # =====================================================================

    def test_grad_mode_eq_self(self):
        # [FUNCTION DOC] Compares a no-grad context manager object with itself; ordering must
        # raise TypeError.
        a = torch.no_grad()
        self._assert_all_cmp_equals(a, a, error_ops=self._ORDERING_OPS)

    def test_inference_mode_eq_self(self):
        # [FUNCTION DOC] Compares an inference-mode context manager object with itself; ordering
        # must raise TypeError.
        a = torch.inference_mode()
        self._assert_all_cmp_equals(a, a, error_ops=self._ORDERING_OPS)

    def test_grad_mode_eq_different_graph_breaks(self):
        # [FUNCTION DOC] Checks that comparing two separately created no-grad managers inside a
        # full-graph compile raises Unsupported mentioning the identity fallback.
        from torch._dynamo.exc import Unsupported

        def fn():
            # [FUNCTION DOC] HELPER: Creates two no-grad context managers and returns
            # their equality.
            a = torch.no_grad()
            b = torch.no_grad()
            return a == b

        with self.assertRaisesRegex(Unsupported, "richcompare identity fallback"):
            torch.compile(fn, backend="eager", fullgraph=True)()

    def test_inference_mode_ne_different_graph_breaks(self):
        # [FUNCTION DOC] Checks that comparing two separately created inference-mode
        # managers for inequality inside a full-graph compile raises Unsupported mentioning
        # the identity fallback.
        from torch._dynamo.exc import Unsupported

        def fn():
            # [FUNCTION DOC] HELPER: Creates two inference-mode context managers and returns
            # their inequality.
            a = torch.inference_mode()
            b = torch.inference_mode()
            return a != b

        with self.assertRaisesRegex(Unsupported, "richcompare identity fallback"):
            torch.compile(fn, backend="eager", fullgraph=True)()

    # =====================================================================
    # Iterator comparison (IteratorVariable)
    # =====================================================================

    def test_zip_iterator_eq_self(self):
        # [FUNCTION DOC] Checks a zip iterator compares equal to itself and not unequal, matching
        # eager in a full-graph compile.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a zip of two inputs and returns its self-equality
            # and self-inequality.
            z1 = zip(a, b)
            return z1 == z1, z1 != z1

        expected = fn([1, 2], [3, 4])
        result = torch.compile(fn, backend="eager", fullgraph=True)([1, 2], [3, 4])
        self.assertEqual(result, expected)

    def test_map_iterator_eq_self(self):
        # [FUNCTION DOC] Checks a map iterator compares equal to itself and not unequal, matching
        # eager in a full-graph compile.
        def fn(a):
            # [FUNCTION DOC] HELPER: Builds a map iterator over the input and returns its
            # self-equality and self-inequality.
            m1 = map(lambda x: x + 1, a)  # noqa: C417
            return m1 == m1, m1 != m1

        expected = fn([1, 2])
        result = torch.compile(fn, backend="eager", fullgraph=True)([1, 2])
        self.assertEqual(result, expected)

    def test_zip_iterator_eq_different_graph_breaks(self):
        # [FUNCTION DOC] Checks equality of two distinct zip iterators in a full-graph compile
        # raises Unsupported mentioning the identity fallback.
        from torch._dynamo.exc import Unsupported

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds two zip iterators over the same inputs and returns
            # their equality.
            z1 = zip(a, b)
            z2 = zip(a, b)
            return z1 == z2

        with self.assertRaisesRegex(Unsupported, "richcompare identity fallback"):
            torch.compile(fn, backend="eager", fullgraph=True)([1, 2], [3, 4])

    def test_map_iterator_ne_different_graph_breaks(self):
        # [FUNCTION DOC] Checks inequality of two distinct map iterators in a full-graph compile
        # raises Unsupported mentioning the identity fallback.
        from torch._dynamo.exc import Unsupported

        def fn(a):
            # [FUNCTION DOC] HELPER: Builds two map iterators over the same input and returns
            # their inequality.
            m1 = map(lambda x: x + 1, a)  # noqa: C417
            m2 = map(lambda x: x + 1, a)  # noqa: C417
            return m1 != m2

        with self.assertRaisesRegex(Unsupported, "richcompare identity fallback"):
            torch.compile(fn, backend="eager", fullgraph=True)([1, 2])

    # =====================================================================
    # Event comparison (EventVariable)
    # =====================================================================

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_event_eq(self):
        # [FUNCTION DOC] Skipped without CUDA. Compares two CUDA events with equality and
        # inequality, and an event with itself, and expects compiled results to match eager.
        def fn(e1, e2):
            # [FUNCTION DOC] HELPER: Returns equality, inequality of two events and self-equality
            # of the first.
            return e1 == e2, e1 != e2, e1 == e1

        e1 = torch.cuda.Event()
        e2 = torch.cuda.Event()
        expected = fn(e1, e2)
        result = torch.compile(fn, backend="eager", fullgraph=True)(e1, e2)
        self.assertEqual(result, expected)

    # =====================================================================
    # itertools module comparison (ItertoolsVariable)
    # =====================================================================

    def test_itertools_eq(self):
        # [FUNCTION DOC] Checks the itertools module compares equal to itself in a full-graph
        # compile, matching eager.
        import itertools

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether the itertools module equals itself.
            return itertools == itertools

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    # =====================================================================
    # NaN identity in containers (followups)
    # =====================================================================

    def test_list_nan_identity(self):
        # [FUNCTION DOC] Checks that a list holding a NaN compares as expected against itself with
        # all operators (identity shortcut for elements).
        """NaN element identity: [nan] == [nan] is True, [nan] <= [nan] is True."""
        nan = float("nan")
        lst = [nan]
        self._assert_all_cmp_equals(lst, lst)

    def test_list_nan_different_objects(self):
        # [FUNCTION DOC] Checks that two lists each containing a freshly made NaN compare unequal,
        # matching eager.
        """Different NaN objects: [float('nan')] == [float('nan')] is False."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether two lists each holding a new NaN are equal.
            return [float("nan")] == [float("nan")]

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    def test_tuple_nan_identity(self):
        # [FUNCTION DOC] Checks equality and inequality of two one-element tuples sharing the same
        # NaN object, matching eager.
        """NaN element identity in tuples."""
        nan = float("nan")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns equality and inequality of the two arguments.
            return a == b, a != b

        expected = fn((nan,), (nan,))
        result = torch.compile(fn, backend="eager", fullgraph=True)((nan,), (nan,))
        self.assertEqual(result, expected)

    def test_dict_nan_identity(self):
        # [FUNCTION DOC] Checks equality and inequality of two dicts holding the same NaN object
        # as a value, matching eager.
        """Dict value NaN identity: {k: nan} == {k: nan} is True."""
        nan = float("nan")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns equality and inequality of the two arguments.
            return a == b, a != b

        expected = fn({"k": nan}, {"k": nan})
        result = torch.compile(fn, backend="eager", fullgraph=True)(
            {"k": nan}, {"k": nan}
        )
        self.assertEqual(result, expected)

    def test_dict_nan_different_objects(self):
        # [FUNCTION DOC] Checks that dicts holding different NaN objects compare unequal,
        # matching eager.
        """Different NaN value objects in dicts compare as not equal."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether two dicts each holding a new NaN are equal.
            return {"k": float("nan")} == {"k": float("nan")}

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    def test_contains_nan_identity(self):
        # [FUNCTION DOC] Checks the membership test finds a NaN that is the same object as a list
        # element, matching eager.
        """NaN in list: nan in [nan] is True (identity shortcut)."""
        nan = float("nan")

        def fn(lst, x):
            # [FUNCTION DOC] HELPER: Returns whether the item is a member of the list.
            return x in lst

        expected = fn([nan, 1, 2], nan)
        result = torch.compile(fn, backend="eager", fullgraph=True)([nan, 1, 2], nan)
        self.assertEqual(result, expected)

    def test_contains_nan_different_objects(self):
        # [FUNCTION DOC] Checks the membership test does not find a different NaN object,
        # matching eager.
        """Different NaN objects: float('nan') in [float('nan')] is False."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns whether a new NaN is a member of a list with another
            # new NaN.
            return float("nan") in [float("nan"), 1, 2]

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(result, expected)

    # =====================================================================
    # Subclass custom __eq__ (followups)
    # =====================================================================

    def test_set_subclass_custom_eq(self):
        # [FUNCTION DOC] Checks that a set subclass with custom equality and inequality methods is
        # used when comparing it with a plain set on the right; results must match eager.
        """set subclass with custom __eq__ must use the custom method."""

        class MySet(set):
            # [CLASS DOC] HELPER: Set subclass with equality and inequality returning marker
            # strings and a constant hash.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns a fixed marker string.
                return "custom_eq"

            def __ne__(self, other):
                # [FUNCTION DOC] HELPER: Returns a fixed marker string.
                return "custom_ne"

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns zero, keeping the instance hashable despite
                # custom equality.
                return 0

        def fn(s1, s2):
            # [FUNCTION DOC] HELPER: Returns the equality and inequality results of two arguments.
            return (s1 == s2, s1 != s2)

        s1 = {1, 2, 3}
        s2 = MySet({1, 2, 3})
        expected = fn(s1, s2)
        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(s1, s2)
        self.assertEqual(result, expected)

    def test_set_subclass_custom_eq_reversed(self):
        # [FUNCTION DOC] Checks the subclass equality takes priority when the subclass is the
        # right operand against a plain set; results must match eager.
        """MySet == set uses subclass priority."""

        class MySet(set):
            # [CLASS DOC] HELPER: Set subclass with custom equality returning a marker string.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns a fixed marker string.
                return "custom_eq"

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        def fn(s1, s2):
            # [FUNCTION DOC] HELPER: Returns the equality of the second argument against
            # the first.
            return s2 == s1

        s1 = MySet({1, 2, 3})
        s2 = {1, 2, 3}
        expected = fn(s1, s2)
        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(s1, s2)
        self.assertEqual(result, expected)

    def test_tuple_subclass_custom_eq(self):
        # [FUNCTION DOC] Checks a tuple subclass with custom equality is used when compared with a
        # plain tuple; results must match eager.
        """tuple subclass with custom __eq__ must use the custom method."""

        class MyTuple(tuple):  # noqa: SLOT001
            def __eq__(self, other):
            # [CLASS DOC] HELPER: Tuple subclass with custom equality returning a marker string
            # and constant hash.
                # [FUNCTION DOC] HELPER: Returns a fixed marker string.
                return "custom_eq"

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        def fn(t1, t2):
            # [FUNCTION DOC] HELPER: Returns the equality of the two arguments.
            return t1 == t2

        t1 = (1, 2, 3)
        t2 = MyTuple((1, 2, 3))
        expected = fn(t1, t2)
        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(t1, t2)
        self.assertEqual(result, expected)

    def test_tuple_subclass_inherited_eq(self):
        # [FUNCTION DOC] Checks a tuple subclass without its own equality behaves like tuple for
        # equal, unequal and less-than; results must match eager.
        """tuple subclass without __eq__ uses tuple's comparison."""

        class MyTuple(tuple):  # noqa: SLOT001
            pass

        def fn(t1, t2):
            # [CLASS DOC] HELPER: Empty tuple subclass.
            # [FUNCTION DOC] HELPER: Returns equality, inequality and less-than of the
            # two arguments.
            return t1 == t2, t1 != t2, t1 < t2

        t1 = MyTuple((1, 2, 3))
        t2 = MyTuple((1, 2, 4))
        expected = fn(t1, t2)
        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(t1, t2)
        self.assertEqual(result, expected)

    def test_set_subclass_inherited_eq(self):
        # [FUNCTION DOC] Checks a set subclass without its own equality behaves like set for
        # equality, inequality and proper-subset tests; results must match eager.
        """set subclass without __eq__ uses set's comparison."""

        class MySet(set):
            # [CLASS DOC] HELPER: Empty set subclass.
            pass

        def fn(s1, s2):
            # [FUNCTION DOC] HELPER: Returns equality, inequality and less-than of the
            # two arguments.
            return s1 == s2, s1 != s2, s1 < s2

        s1 = MySet({1, 2})
        s2 = MySet({1, 2, 3})
        expected = fn(s1, s2)
        torch._dynamo.reset()
        result = torch.compile(fn, backend="eager", fullgraph=True)(s1, s2)
        self.assertEqual(result, expected)

    def test_set_subclass_inherited_cmp_uses_internal_contents(self):
        # [FUNCTION DOC] Checks that for set and frozenset subclasses that lie through iteration,
        # length and containment methods, all twelve comparisons use the real internal contents.
        # For each case the compiled function must match eager, give the same result on a second
        # call, and compile exactly once.
        class LyingSet(set):
            # [CLASS DOC] HELPER: Set subclass whose iteration yields only a fake item and whose
            # length and containment methods raise RuntimeError.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Yields a bogus single item instead of real contents.
                return iter([99])

            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("__len__ should not be called")

            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("__contains__ should not be called")

        class LyingFrozenSet(frozenset):
            # [CLASS DOC] HELPER: Frozenset subclass with the same lying iteration and failing
            # length and containment methods.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Yields a bogus single item instead of real contents.
                return iter([99])

            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("__len__ should not be called")

            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError("__contains__ should not be called")

        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Compiled function evaluating twelve comparisons between the
            # two sets in both directions (equal, not-equal, and the four orderings) and adding to
            # the input a bitmask with one bit per true comparison.
            offset = 0
            comparisons = (
                a == b,
                b == a,
                a != b,
                b != a,
                a <= b,
                b <= a,
                a < b,
                b < a,
                a >= b,
                b >= a,
                a > b,
                b > a,
            )
            for i, pred in enumerate(comparisons):
                if pred:
                    offset += 1 << i
            return x + offset

        cases = (
            ({1, 2, 3}, LyingSet({1, 2, 3})),
            (LyingSet({1, 2, 3}), {1, 2, 3}),
            (LyingSet({1, 2}), LyingSet({1, 2, 3})),
            ({1, 2, 3}, LyingFrozenSet({1, 2, 3})),
        )

        def base_len(obj):
            # [FUNCTION DOC] HELPER: Returns the true size of a set or frozenset using the base
            # class length method, bypassing overrides; used only to label subtests.
            if isinstance(obj, set):
                return set.__len__(obj)
            return frozenset.__len__(obj)

        for a, b in cases:
            with self.subTest(a=type(a), b=type(b), sizes=(base_len(a), base_len(b))):
                x = torch.ones(())
                expected = fn(x, a, b)
                cnt = torch._dynamo.testing.CompileCounter()
                opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
                self.assertEqual(opt_fn(x, a, b), expected)
                self.assertEqual(opt_fn(x, a, b), expected)
                self.assertEqual(cnt.frame_count, 1)
            torch._dynamo.reset()

    # =====================================================================
    # Tensor vs non-proxyable types (followups)
    # =====================================================================

    def test_tensor_eq_user_defined_object(self):
        # [FUNCTION DOC] Checks that comparing a tensor with an instance of an unrelated user
        # class gives the same result as eager (identity fallback) in a full-graph compile.
        """tensor == UserDefinedObject() should return False (identity fallback)."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty local class.
            pass

        def fn(t):
            # [FUNCTION DOC] HELPER: Returns equality of the tensor with a new MyObj.
            return t == MyObj()

        t = torch.randn(3)
        expected = fn(t)
        result = torch.compile(fn, backend="eager", fullgraph=True)(t)
        self.assertEqual(result, expected)

    def test_tensor_ne_user_defined_object(self):
        # [FUNCTION DOC] Checks that inequality between a tensor and an unrelated user object
        # matches eager in a full-graph compile.
        class MyObj:
            # [CLASS DOC] HELPER: Empty local class.
            pass

        def fn(t):
            # [FUNCTION DOC] HELPER: Returns inequality of the tensor with a new MyObj.
            return t != MyObj()

        t = torch.randn(3)
        expected = fn(t)
        result = torch.compile(fn, backend="eager", fullgraph=True)(t)
        self.assertEqual(result, expected)

    # =====================================================================
    # DispatchKeySet comparison (followups)
    # =====================================================================

    def test_dispatch_key_set_eq(self):
        # [FUNCTION DOC] Checks equality of two DispatchKeySet objects with the same key matches
        # eager in a full-graph compile.
        from torch._C import DispatchKey, DispatchKeySet

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns equality of the two arguments.
            return a == b

        ks1 = DispatchKeySet(DispatchKey.CPU)
        ks2 = DispatchKeySet(DispatchKey.CPU)
        expected = fn(ks1, ks2)
        result = torch.compile(fn, backend="eager", fullgraph=True)(ks1, ks2)
        self.assertEqual(result, expected)

    def test_dispatch_key_set_ne(self):
        # [FUNCTION DOC] Checks inequality of DispatchKeySets with different keys matches eager in
        # a full-graph compile.
        from torch._C import DispatchKey, DispatchKeySet

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns inequality of the two arguments.
            return a != b

        ks1 = DispatchKeySet(DispatchKey.CPU)
        ks2 = DispatchKeySet(DispatchKey.CUDA)
        expected = fn(ks1, ks2)
        result = torch.compile(fn, backend="eager", fullgraph=True)(ks1, ks2)
        self.assertEqual(result, expected)

    def test_unbound_builtin_cmp_dunder(self):
        # [FUNCTION DOC] Checks direct calls of comparison dunders on builtin types (complex, int,
        # float, str, bool) with matching and mismatching operands, including the not-implemented
        # results, in a full-graph compile; each compiled value must be the identical object as
        # the eager one.
        """type.__cmp__(a, b) invokes only the left type's slot (may be
        NotImplemented) rather than the full comparison protocol."""

        def fn():
            # [FUNCTION DOC] HELPER: Returns a tuple of thirteen results of unbound comparison
            # dunder calls on builtin types.
            return (
                complex.__eq__(1 + 1j, 1 + 1j),
                complex.__eq__(1 + 1j, 2 + 2j),
                complex.__eq__(1 + 1j, 2),
                complex.__ne__(1 + 1j, 1 + 1j),
                complex.__eq__(1 + 1j, None),
                complex.__lt__(1 + 1j, 2 + 2j),
                int.__eq__(1, 1),
                int.__lt__(1, 2),
                int.__eq__(1, None),
                float.__eq__(1.0, 2.0),
                float.__lt__(1.0, 2.0),
                str.__eq__("a", "a"),
                bool.__eq__(True, 1),
            )

        expected = fn()
        result = torch.compile(fn, backend="eager", fullgraph=True)()
        self.assertEqual(len(expected), len(result))
        for e, r in zip(expected, result):
            self.assertIs(r, e)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
