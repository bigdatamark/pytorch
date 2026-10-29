# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the unified int conversion protocol (nb_int_impl): the builtin int
# conversion and the __int__ method on ints, bools, floats, user-defined objects (including
# static, class and instance-method forms), tensors and symbolic sizes, plus the __index__
# fallback, blocked slot and error messages. Imports instancemethod from the CPython testing
# module _testcapi at import time, so the module fails to import where that module is unavailable.
"""Tests for nb_int_impl: unified __int__ / int() protocol in Dynamo."""

from _testcapi import instancemethod

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


class NbIntTests(TestCase):
    # [CLASS DOC] Test case for int conversion in Dynamo; no setup state. Tests either run through
    # make_dynamo_test or compile a small function with fullgraph and compare to eager or expected
    # values.
    # --- int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_int_int(self):
        # [FUNCTION DOC] Only compares the int five to itself; it does not call the int
        # conversion, so it is a trivial sanity check.
        self.assertEqual(5, 5)

    @make_dynamo_test
    def test_bool_int(self):
        # [FUNCTION DOC] Checks converting True and False gives one and zero.
        self.assertEqual(int(True), 1)
        self.assertEqual(int(False), 0)

    @make_dynamo_test
    def test_int_dunder_int(self):
        # [FUNCTION DOC] Checks calling __int__ directly on an int constant.
        self.assertEqual((5).__int__(), 5)

    @make_dynamo_test
    def test_bool_dunder_int(self):
        # [FUNCTION DOC] Checks calling __int__ directly on True gives one.
        self.assertEqual(True.__int__(), 1)

    # --- float (ConstantVariable) ---

    @make_dynamo_test
    def test_float_int(self):
        # [FUNCTION DOC] Checks converting 3.14 truncates to three.
        self.assertEqual(int(3.14), 3)

    @make_dynamo_test
    def test_float_negative_int(self):
        # [FUNCTION DOC] Checks converting minus 2.9 truncates toward zero to minus two.
        self.assertEqual(int(-2.9), -2)

    @make_dynamo_test
    def test_float_dunder_int(self):
        # [FUNCTION DOC] Checks calling __int__ directly on a float constant truncates.
        self.assertEqual((3.14).__int__(), 3)

    # --- TypeError for non-int types ---

    def test_complex_int_raises(self):
        # [FUNCTION DOC] Compiles a function converting a complex number and returning the
        # TypeError message; the compiled message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting a complex
            # constant.
            try:
                return int(1 + 2j)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_none_int_raises(self):
        # [FUNCTION DOC] Same as the complex case but for None.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting None.
            try:
                return int(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_list_int_raises(self):
        # [FUNCTION DOC] Same for a list of two ints.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting a list.
            try:
                return int([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_dict_int_raises(self):
        # [FUNCTION DOC] Same for an empty dict.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting an empty dict.
            try:
                return int({})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_set_int_raises(self):
        # [FUNCTION DOC] Same for an empty set.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting an empty set.
            try:
                return int(set())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_tuple_int_raises(self):
        # [FUNCTION DOC] Same for an empty tuple.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting an empty tuple.
            try:
                return int(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- str parsing (constructor, not nb_int) ---

    @make_dynamo_test
    def test_str_int_parsing(self):
        # [FUNCTION DOC] Checks int of a decimal string parses it (constructor behavior, not the
        # int slot).
        self.assertEqual(int("123"), 123)

    @make_dynamo_test
    def test_str_int_base(self):
        # [FUNCTION DOC] Checks int of a hex string with base sixteen gives 255.
        self.assertEqual(int("ff", 16), 255)

    # --- UserDefinedObjectVariable with __int__ ---

    def test_user_defined_int(self):
        # [FUNCTION DOC] Compiles int conversion of a captured user-defined object whose __int__
        # returns the stored value; expects 42.
        class MyInt:
            # [CLASS DOC] HELPER: Test class holding an int; defines __int__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyInt(42)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the captured object.
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_user_defined_dunder_int(self):
        # [FUNCTION DOC] Compiles a direct call of the __int__ method on a user-defined object;
        # expects 7.
        class MyInt:
            # [CLASS DOC] HELPER: Test class holding an int; defines __int__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyInt(7)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the result of calling __int__ directly on the
            # captured object.
            return obj.__int__()

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 7
        )

    def test_user_defined_no_int_raises(self):
        # [FUNCTION DOC] Checks that int of an object with neither __int__ nor __index__ raises
        # TypeError with the must-be-a-string-or-a-real-number style message.
        class NoInt:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoInt()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("int() argument must be a string", result)

    def test_int_returning_non_int_raises(self):
        # [FUNCTION DOC] Checks that an __int__ returning a string raises TypeError mentioning a
        # non-int return.
        class Bad:
            # [CLASS DOC] HELPER: Class whose __int__ returns a string.
            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the string "not an int", an invalid result.
                return "not an int"

        obj = Bad()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__int__ returned non-int", result)

    def test_int_returning_float_raises(self):
        # [FUNCTION DOC] Checks that an __int__ returning a float still raises TypeError; the
        # message mentions a non-int return of type float.
        # A numeric-but-wrong-type return (float is not int) must still raise.
        class Bad:
            # [CLASS DOC] HELPER: Class whose __int__ returns the float 3.0.
            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the float 3.0, an invalid result.
                return 3.0

        obj = Bad()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__int__ returned non-int (type float)", result)

    def test_int_raising_exception_propagates(self):
        # [FUNCTION DOC] Checks that a ValueError raised inside __int__ propagates; the compiled
        # function returns its message.
        class RaisingInt:
            # [CLASS DOC] HELPER: Class whose __int__ raises ValueError.
            def __int__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError with the message custom error.
                raise ValueError("custom error")

        obj = RaisingInt()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the ValueError message from converting the captured
            # object.
            try:
                return int(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    def test_user_defined_staticmethod_int(self):
        # [FUNCTION DOC] Checks int conversion works when __int__ is a static method; expects
        # three.
        class StaticInt:
            # [CLASS DOC] HELPER: Class with a static method __int__ returning three.
            @staticmethod
            def __int__():
                # [FUNCTION DOC] HELPER: Static method; returns three.
                return 3

        obj = StaticInt()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the captured object.
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3
        )

    def test_user_defined_classmethod_int(self):
        # [FUNCTION DOC] Checks int conversion works when __int__ is a class method; expects four.
        class ClassInt:
            # [CLASS DOC] HELPER: Class with a class method __int__ returning four.
            @classmethod
            def __int__(cls):
                # [FUNCTION DOC] HELPER: Class method; returns four.
                return 4

        obj = ClassInt()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the captured object.
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 4
        )

    def test_user_defined_instancemethod_int(self):
        # [FUNCTION DOC] Checks that __int__ stored as a C-level instance method wrapper (as
        # pybind11 enum members produce) is resolved without recursing back into the int
        # conversion implementation; expects zero.
        # __int__ as a PyInstanceMethod (the type pybind11 emits for
        # C-defined dunders, e.g. py::enum_ members). resolve_type_attr
        # must not route this back through nb_int_impl (infinite recursion).
        def _to_int(self):
            # [FUNCTION DOC] HELPER: Plain function returning zero, wrapped as an instance method
            # on class E.
            return 0

        class E:
            # [CLASS DOC] HELPER: Class whose __int__ attribute is an instancemethod wrapper
            # around the helper function.
            __int__ = instancemethod(_to_int)

        obj = E()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the captured object.
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 0
        )

    # --- nb_index fallback (PyNumber_Long step 3) ---

    def test_index_fallback_for_int(self):
        # [FUNCTION DOC] Checks the __index__ fallback: int of an object with only __index__
        # returning 42 gives 42.
        class HasIndex:
            # [CLASS DOC] HELPER: Class defining only __index__.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int 42.
                return 42

        obj = HasIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the captured object.
            return int(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42
        )

    def test_index_fallback_no_int_no_index_raises(self):
        # [FUNCTION DOC] Checks that an object with neither __int__ nor __index__ raises TypeError
        # with the must-be-a-string message, equal in compiled and eager runs.
        class NoSlots:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoSlots()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("int() argument must be a string", result)
        self.assertEqual(result, eager_result)

    # --- Tensor ---

    def test_tensor_int_dtype(self):
        # [FUNCTION DOC] Checks int of an int64 scalar tensor gives five.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the argument.
            return int(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5)

    def test_tensor_float_dtype(self):
        # [FUNCTION DOC] Checks int of a float scalar tensor 3.7 truncates to three.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the argument.
            return int(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(3.7))
        self.assertEqual(result, 3)

    def test_tensor_complex_raises(self):
        # [FUNCTION DOC] Checks int of a complex scalar tensor raises RuntimeError whose message
        # mentions the value cannot be converted to type int64_t without overflow, equal in both
        # runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the RuntimeError message from converting the
            # argument.
            try:
                return int(x)
            except RuntimeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(1 + 2j)
        )
        eager_result = fn(torch.tensor(1 + 2j))
        self.assertIn(
            "value cannot be converted to type int64_t without overflow", result
        )
        self.assertEqual(result, eager_result)

    def test_tensor_dunder_int(self):
        # [FUNCTION DOC] Checks calling __int__ on an int tensor gives five.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the result of calling __int__ on the argument.
            return x.__int__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5)

    # --- Blocked slot: __int__ = None ---

    def test_user_defined_int_none_raises(self):
        # [FUNCTION DOC] Checks that setting __int__ to None blocks the slot: TypeError whose
        # message mentions NoneType, equal to eager.
        class NoInt:
            # [CLASS DOC] HELPER: Class whose __int__ attribute is None.
            __int__ = None

        obj = NoInt()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return int(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("NoneType", result)
        self.assertEqual(result, eager_result)

    # --- SymNodeVariable ---

    def test_symnode_int(self):
        # [FUNCTION DOC] Checks int of the size of dimension zero of a 10 by 20 tensor gives ten.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the int conversion of the size of dimension zero.
            return int(x.size(0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(10, 20))
        self.assertEqual(result, 10)


if __name__ == "__main__":
    run_tests()
