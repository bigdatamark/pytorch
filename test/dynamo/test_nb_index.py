# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the unified index protocol (nb_index_impl): the operator module
# index function and the __index__ method on ints, bools, user-defined objects, tensors, and use
# of index conversion in list and tuple subscripts. Covers error messages, static and class method
# __index__, blocked slot and exceptions raised inside __index__. Mix of make_dynamo_test cases
# and fullgraph torch.compile cases.
"""Tests for nb_index_impl: unified __index__ / operator.index protocol in Dynamo."""

import operator

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class NbIndexTests(TestCase):
    # [CLASS DOC] Test case for index conversion in Dynamo; no setup state. Each test either runs
    # through make_dynamo_test or compiles a small function with fullgraph and checks the returned
    # value or error message.
    # --- int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_int_index(self):
        # [FUNCTION DOC] Checks operator index of the int five is five.
        self.assertEqual(operator.index(5), 5)

    @make_dynamo_test
    def test_bool_index(self):
        # [FUNCTION DOC] Checks operator index of True is one.
        self.assertEqual(operator.index(True), 1)

    @make_dynamo_test
    def test_int_dunder_index(self):
        # [FUNCTION DOC] Checks calling __index__ directly on an int constant.
        self.assertEqual((5).__index__(), 5)

    # --- TypeError for non-index types ---

    def test_str_index_raises(self):
        # [FUNCTION DOC] Compiles a function that calls operator index on a string and returns the
        # TypeError message; asserts the message says it cannot be interpreted as an integer.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on a string
            # constant.
            try:
                return operator.index("hello")
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_float_index_raises(self):
        # [FUNCTION DOC] Same as above with a float constant; message must say it cannot be
        # interpreted as an integer.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on a float
            # constant.
            try:
                return operator.index(3.14)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_list_index_raises(self):
        # [FUNCTION DOC] Same with a list of two ints.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on a list.
            try:
                return operator.index([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    def test_none_index_raises(self):
        # [FUNCTION DOC] Same with None.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on None.
            try:
                return operator.index(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    # --- UserDefinedObjectVariable with __index__ ---

    def test_user_defined_index(self):
        # [FUNCTION DOC] Compiles operator index on a captured user-defined object with __index__
        # returning the stored value; expects 42.
        class MyInt:
            # [CLASS DOC] HELPER: Test class holding an int; defines __index__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyInt(42)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns operator index applied to the captured object.
            return operator.index(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_user_defined_dunder_index(self):
        # [FUNCTION DOC] Compiles a direct call of the __index__ method on a user-defined object;
        # expects 7.
        class MyInt:
            # [CLASS DOC] HELPER: Test class holding an int; defines __index__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyInt(7)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the result of calling __index__ directly on the
            # captured object.
            return obj.__index__()

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 7
        )

    def test_user_defined_no_index_raises(self):
        # [FUNCTION DOC] Checks operator index on an object without __index__ raises TypeError
        # mentioning it cannot be interpreted as an integer.
        class NoIndex:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on the
            # captured object.
            try:
                return operator.index(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("cannot be interpreted as an integer", result)

    # --- nb_index used in list/tuple subscript (PyNumber_AsSsize_t) ---

    @make_dynamo_test
    def test_list_subscript_with_bool(self):
        # [FUNCTION DOC] Checks a list subscripted by True returns the second element (twenty).
        lst = [10, 20, 30]
        self.assertEqual(lst[True], 20)

    @make_dynamo_test
    def test_tuple_subscript_with_bool(self):
        # [FUNCTION DOC] Checks a tuple subscripted by True returns the second element (twenty).
        t = (10, 20, 30)
        self.assertEqual(t[True], 20)

    def test_list_subscript_with_user_defined_index(self):
        # [FUNCTION DOC] Compiles subscripting a list with a captured object whose __index__
        # returns one; expects twenty.
        class MyIdx:
            # [CLASS DOC] HELPER: Class whose __index__ returns one.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int one.
                return 1

        idx = MyIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a three-element list and returns the element selected
            # by the captured index object.
            lst = [10, 20, 30]
            return lst[idx]

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 20
        )

    def test_tuple_subscript_with_user_defined_index(self):
        # [FUNCTION DOC] Compiles subscripting a tuple with a captured object whose __index__
        # returns two; expects thirty.
        class MyIdx:
            # [CLASS DOC] HELPER: Class whose __index__ returns two.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int two.
                return 2

        idx = MyIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a three-element tuple and returns the element selected
            # by the captured index object.
            t = (10, 20, 30)
            return t[idx]

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 30
        )

    def test_list_subscript_with_no_index_raises(self):
        # [FUNCTION DOC] Checks subscripting a list with an object that has no __index__ raises
        # TypeError with the message that list indices must be integers or slices.
        class NoIndex:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Subscripts a list with the captured object and returns the
            # TypeError message.
            lst = [10, 20, 30]
            try:
                return lst[obj]
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("list indices must be integers or slices", result)

    def test_user_defined_staticmethod_index(self):
        # [FUNCTION DOC] Checks operator index works when __index__ is a static method taking no
        # arguments; expects three.
        class StaticIdx:
            # [CLASS DOC] HELPER: Class with a static method __index__ returning three.
            @staticmethod
            def __index__():
                # [FUNCTION DOC] HELPER: Static method; returns three.
                return 3

        obj = StaticIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns operator index applied to the captured object.
            return operator.index(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3
        )

    def test_user_defined_classmethod_index(self):
        # [FUNCTION DOC] Checks operator index works when __index__ is a class method; expects
        # four.
        class ClassIdx:
            # [CLASS DOC] HELPER: Class with a class method __index__ returning four.
            @classmethod
            def __index__(cls):
                # [FUNCTION DOC] HELPER: Class method; returns four.
                return 4

        obj = ClassIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns operator index applied to the captured object.
            return operator.index(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 4
        )

    def test_custom_getitem_with_user_defined_index(self):
        # [FUNCTION DOC] Checks that a user-defined container with its own __getitem__ receives
        # the index object unchanged and forwards it to a list, which converts it through
        # __index__ (expects twenty).
        class MyIdx:
            # [CLASS DOC] HELPER: Index class whose __index__ returns one.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int one.
                return 1

        class MyContainer:
            # [CLASS DOC] HELPER: Container wrapping a list; its __getitem__ forwards the key to
            # the list.
            def __init__(self, data):
                # [FUNCTION DOC] HELPER: Stores the data argument in the data attribute.
                self.data = data

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: Returns the element of the wrapped list at the given key.
                return self.data[idx]

        idx = MyIdx()
        container = MyContainer([10, 20, 30])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the element of the captured container selected by the
            # captured index object.
            return container[idx]

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 20
        )

    def test_dict_getitem_with_non_indexable(self):
        # [FUNCTION DOC] Checks that a dict lookup with an object lacking __index__ works as an
        # ordinary hashable key; expects forty-two.
        class NoIndex:
            # [CLASS DOC] HELPER: Empty class used as a dict key.
            pass

        obj = NoIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by the captured object and returns the
            # value looked up with the same key.
            d = {obj: 42}
            return d[obj]

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_list_getitem_non_indexable_matches_cpython(self):
        # [FUNCTION DOC] Checks a list subscripted by an object lacking __index__ raises TypeError
        # in both eager and compiled runs (the function returns the word caught), and both results
        # are compared with the expected word.
        class NoIndex:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Subscripts a two-element list with the captured object;
            # returns the word caught on TypeError.
            try:
                return [10, 20][obj]
            except TypeError:
                return "caught"

        # Both CPython and Dynamo should raise TypeError and catch it
        eager_result = fn(torch.tensor(0))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(0)
        )
        self.assertEqual(eager_result, "caught")
        self.assertEqual(compiled_result, "caught")

    def test_list_subscript_error_message_matches_cpython(self):
        # [FUNCTION DOC] Checks the TypeError message for subscripting a list with a string says
        # list indices must be integers or slices, in both eager and compiled runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Subscripts a two-element list with a string and returns the
            # TypeError message. The lint suppression silences a warning on the constant
            # subscript.
            try:
                return [10, 20]["hello"]  # noqa: RUF016
            except TypeError as e:
                return str(e)

        eager_result = fn(torch.tensor(0))
        compiled_result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(0)
        )
        self.assertIn("list indices must be integers or slices", eager_result)
        self.assertIn("list indices must be integers or slices", compiled_result)

    def test_index_returning_non_int_raises(self):
        # [FUNCTION DOC] Checks that __index__ returning a string raises TypeError mentioning a
        # non-int return.
        class Bad:
            # [CLASS DOC] HELPER: Class whose __index__ returns a string (lint suppression for the
            # invalid return).
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the string "not an int", an invalid result.
                return "not an int"  # noqa: PLE0305

        obj = Bad()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on the
            # captured object.
            try:
                return operator.index(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__index__ returned non-int", result)

    def test_negative_index_via_user_defined(self):
        # [FUNCTION DOC] Checks that a negative index from __index__ (minus one) selects the last
        # list element (thirty).
        class NegIdx:
            # [CLASS DOC] HELPER: Class whose __index__ returns minus one.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int minus one.
                return -1

        idx = NegIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Subscripts a three-element list with the captured index
            # object.
            lst = [10, 20, 30]
            return lst[idx]

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 30
        )

    def test_index_raising_exception_propagates(self):
        # [FUNCTION DOC] Checks that a ValueError raised inside __index__ propagates; the compiled
        # function returns its message.
        class RaisingIdx:
            # [CLASS DOC] HELPER: Class whose __index__ raises ValueError.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError with the message custom error.
                raise ValueError("custom error")

        obj = RaisingIdx()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the ValueError message from operator index on the
            # captured object.
            try:
                return operator.index(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    # --- Blocked slot: __index__ = None ---

    def test_user_defined_index_none_raises(self):
        # [FUNCTION DOC] Checks that setting __index__ to None blocks the slot: TypeError whose
        # message mentions NoneType, equal to the eager message.
        class NoIndex:
            # [CLASS DOC] HELPER: Class whose __index__ attribute is None.
            __index__ = None

        obj = NoIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on the
            # captured object.
            try:
                return operator.index(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("NoneType", result)
        self.assertEqual(result, eager_result)

    # --- Tensor __index__ ---

    def test_tensor_int_index(self):
        # [FUNCTION DOC] Checks operator index of an int64 scalar tensor gives five.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns operator index applied to the argument.
            return operator.index(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5)

    def test_tensor_float_index_raises(self):
        # [FUNCTION DOC] Checks operator index of a float tensor raises TypeError whose message
        # says only integer tensors can be converted.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from operator index on the
            # argument.
            try:
                return operator.index(x)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(1.5))
        self.assertIn("only integer tensors", result)

    def test_list_subscript_with_tensor(self):
        # [FUNCTION DOC] Checks subscripting a list by an int64 scalar tensor with value two
        # returns thirty.
        def fn(x):
            # [FUNCTION DOC] HELPER: Subscripts a three-element list with the argument.
            lst = [10, 20, 30]
            return lst[x]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(2))
        self.assertEqual(result, 30)


if __name__ == "__main__":
    run_tests()
