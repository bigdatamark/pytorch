# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for the unified float conversion protocol (nb_float_impl): the builtin
# float conversion and the __float__ method on constants, user-defined objects, tensors and
# symbolic ints, including the __index__ fallback, blocked slot and error messages. Mix of
# make_dynamo_test cases and cases compiled with torch.compile (eager backend, fullgraph). Some
# tensor tests are skipped under crossref mode.
"""Tests for nb_float_impl: unified __float__ / float() protocol in Dynamo."""

import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test, skipIfCrossRef


class NbFloatTests(TestCase):
    # [CLASS DOC] Test case for float conversion in Dynamo; no setup state. Tests either run
    # through make_dynamo_test or compile a small function with fullgraph and compare to eager or
    # to expected values.
    # --- float / int / bool (ConstantVariable) ---

    @make_dynamo_test
    def test_float_float(self):
        # [FUNCTION DOC] Checks converting a float constant returns the same float (the lint
        # suppression covers the redundant conversion).
        self.assertEqual(float(3.14), 3.14)  # noqa: UP018

    @make_dynamo_test
    def test_int_float(self):
        # [FUNCTION DOC] Checks converting the int five gives the float five.
        self.assertEqual(float(5), 5.0)

    @make_dynamo_test
    def test_bool_float(self):
        # [FUNCTION DOC] Checks converting True and False gives one and zero as floats.
        self.assertEqual(float(True), 1.0)
        self.assertEqual(float(False), 0.0)

    @make_dynamo_test
    def test_float_dunder_float(self):
        # [FUNCTION DOC] Checks calling __float__ directly on a float constant.
        self.assertEqual((3.14).__float__(), 3.14)

    @make_dynamo_test
    def test_int_dunder_float(self):
        # [FUNCTION DOC] Checks calling __float__ directly on an int constant gives a float.
        self.assertEqual((5).__float__(), 5.0)

    @make_dynamo_test
    def test_bool_dunder_float(self):
        # [FUNCTION DOC] Checks calling __float__ directly on True gives one as a float.
        self.assertEqual(True.__float__(), 1.0)

    # --- TypeError for non-float types ---

    def test_complex_float_raises(self):
        # [FUNCTION DOC] Compiles a function that tries to convert a complex number and returns
        # the TypeError message; asserts the compiled message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting a complex constant,
            # caught as TypeError.
            try:
                return float(1 + 2j)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_none_float_raises(self):
        # [FUNCTION DOC] Same as the complex case but converting None; compiled TypeError message
        # equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting None, caught as
            # TypeError.
            try:
                return float(None)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_list_float_raises(self):
        # [FUNCTION DOC] Same for a list of two ints; compiled TypeError message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting a list, caught as
            # TypeError.
            try:
                return float([1, 2])
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_dict_float_raises(self):
        # [FUNCTION DOC] Same for an empty dict; compiled TypeError message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting an empty dict,
            # caught as TypeError.
            try:
                return float({})
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_set_float_raises(self):
        # [FUNCTION DOC] Same for an empty set; compiled TypeError message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting an empty set,
            # caught as TypeError.
            try:
                return float(set())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    def test_tuple_float_raises(self):
        # [FUNCTION DOC] Same for an empty tuple; compiled TypeError message equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the error message from converting an empty tuple,
            # caught as TypeError.
            try:
                return float(())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- str parsing (constructor, not nb_float) ---

    @make_dynamo_test
    def test_str_float_parsing(self):
        # [FUNCTION DOC] Checks that float of a decimal string parses it (string parsing is
        # constructor behavior, not the float slot).
        self.assertEqual(float("3.14"), 3.14)

    @make_dynamo_test
    def test_str_float_int(self):
        # [FUNCTION DOC] Checks float of an integer string gives the float value.
        self.assertEqual(float("123"), 123.0)

    # --- UserDefinedObjectVariable with __float__ ---

    def test_user_defined_float(self):
        # [FUNCTION DOC] Compiles float conversion of a captured user-defined object whose
        # __float__ returns the stored value; expects 3.14.
        class MyFloat:
            # [CLASS DOC] HELPER: Test class holding a float value; defines __float__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __float__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyFloat(3.14)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the captured object; the
            # tensor argument is unused.
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3.14
        )

    def test_user_defined_dunder_float(self):
        # [FUNCTION DOC] Compiles a direct call of the __float__ method on a user-defined object;
        # expects 2.71.
        class MyFloat:
            # [CLASS DOC] HELPER: Test class holding a float value; defines __float__.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the value argument in the v attribute.
                self.v = v

            def __float__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value.
                return self.v

        obj = MyFloat(2.71)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the result of calling __float__ directly on the
            # captured object.
            return obj.__float__()

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 2.71
        )

    def test_user_defined_no_float_raises(self):
        # [FUNCTION DOC] Checks that float of an object with no __float__ and no __index__ raises
        # TypeError whose message contains the text that the argument must be a string or a real
        # number.
        class NoFloat:
            # [CLASS DOC] HELPER: Empty class with no numeric methods.
            pass

        obj = NoFloat()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("float() argument must be a string", result)

    def test_float_returning_non_float_raises(self):
        # [FUNCTION DOC] Checks that a __float__ returning a string raises TypeError with a
        # message that mentions a non-float return.
        class Bad:
            # [CLASS DOC] HELPER: Class whose __float__ returns a string.
            def __float__(self):
                # [FUNCTION DOC] HELPER: Returns the string "not a float", an invalid result.
                return "not a float"

        obj = Bad()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__float__ returned non-float", result)

    def test_float_returning_int_raises(self):
        # [FUNCTION DOC] Checks that a __float__ returning an int (numeric but not a float) still
        # raises TypeError; the message must mention a non-float return of type int.
        # A numeric-but-wrong-type return (int is not float) must still raise.
        class Bad:
            # [CLASS DOC] HELPER: Class whose __float__ returns the int three.
            def __float__(self):
                # [FUNCTION DOC] HELPER: Returns the int three, an invalid result.
                return 3

        obj = Bad()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn("__float__ returned non-float (type int)", result)

    def test_float_raising_exception_propagates(self):
        # [FUNCTION DOC] Checks that an exception raised inside __float__ (ValueError with a
        # custom message) propagates; the compiled function returns its message.
        class RaisingFloat:
            # [CLASS DOC] HELPER: Class whose __float__ raises ValueError.
            def __float__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError with the message custom error.
                raise ValueError("custom error")

        obj = RaisingFloat()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the ValueError message from converting the captured
            # object.
            try:
                return float(obj)
            except ValueError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, "custom error")

    def test_user_defined_staticmethod_float(self):
        # [FUNCTION DOC] Checks float conversion works when __float__ is defined as a static
        # method taking no arguments; expects 3.0.
        class StaticFloat:
            # [CLASS DOC] HELPER: Class with a static method __float__ returning 3.0.
            @staticmethod
            def __float__():
                # [FUNCTION DOC] HELPER: Static method; returns 3.0.
                return 3.0

        obj = StaticFloat()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the captured object.
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 3.0
        )

    def test_user_defined_classmethod_float(self):
        # [FUNCTION DOC] Checks float conversion works when __float__ is a class method; expects
        # 4.0.
        class ClassFloat:
            # [CLASS DOC] HELPER: Class with a class method __float__ returning 4.0.
            @classmethod
            def __float__(cls):
                # [FUNCTION DOC] HELPER: Class method; returns 4.0.
                return 4.0

        obj = ClassFloat()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the captured object.
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 4.0
        )

    # --- nb_index fallback (PyNumber_Float step 3) ---

    def test_index_fallback_for_float(self):
        # [FUNCTION DOC] Checks the fallback to __index__ when no __float__ exists: float of an
        # object with only __index__ returning 42 gives 42.0.
        class HasIndex:
            # [CLASS DOC] HELPER: Class defining only __index__.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the int 42.
                return 42

        obj = HasIndex()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the captured object.
            return float(obj)

        self.assertEqual(
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0)), 42.0
        )

    def test_index_fallback_no_float_no_index_raises(self):
        # [FUNCTION DOC] Checks that an object with neither __float__ nor __index__ raises
        # TypeError with the must-be-a-string-or-real-number message, and that the compiled
        # message equals the eager one.
        class NoSlots:
            # [CLASS DOC] HELPER: Empty class.
            pass

        obj = NoSlots()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("float() argument must be a string", result)
        self.assertEqual(result, eager_result)

    # --- Tensor ---

    @skipIfCrossRef
    def test_tensor_int_dtype(self):
        # [FUNCTION DOC] Skipped under crossref. Checks float of an int64 scalar tensor gives 5.0.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the argument.
            return float(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5.0)

    @skipIfCrossRef
    def test_tensor_float_dtype(self):
        # [FUNCTION DOC] Skipped under crossref. Checks float of a float scalar tensor is about
        # 3.14 (two places).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the argument.
            return float(x)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(3.14))
        self.assertAlmostEqual(result, 3.14, places=2)

    @skipIfCrossRef
    def test_tensor_complex_raises(self):
        # [FUNCTION DOC] Skipped under crossref. Checks float of a complex scalar tensor raises
        # RuntimeError whose message mentions the value cannot be converted to type double without
        # overflow, equal in compiled and eager runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the RuntimeError message from converting the
            # argument.
            try:
                return float(x)
            except RuntimeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(1 + 2j)
        )
        eager_result = fn(torch.tensor(1 + 2j))
        self.assertIn(
            "value cannot be converted to type double without overflow", result
        )
        self.assertEqual(result, eager_result)

    @skipIfCrossRef
    def test_tensor_dunder_float(self):
        # [FUNCTION DOC] Skipped under crossref. Checks calling __float__ on an int tensor gives
        # 5.0.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the result of calling __float__ on the argument.
            return x.__float__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(5))
        self.assertEqual(result, 5.0)

    # --- Blocked slot: __float__ = None ---

    def test_user_defined_float_none_raises(self):
        # [FUNCTION DOC] Checks that setting __float__ to None blocks the slot: conversion raises
        # TypeError whose message mentions NoneType and equals eager.
        class NoFloat:
            # [CLASS DOC] HELPER: Class whose __float__ attribute is None.
            __float__ = None

        obj = NoFloat()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the TypeError message from converting the captured
            # object.
            try:
                return float(obj)
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertIn("NoneType", result)
        self.assertEqual(result, eager_result)

    # --- SymNodeVariable ---

    @skipIfCrossRef
    def test_symnode_float(self):
        # [FUNCTION DOC] Skipped under crossref. Checks float of a symbolic or concrete size
        # (first dimension of a 10 by 20 tensor) gives 10.0.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the float conversion of the size of dimension zero.
            return float(x.size(0))

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(10, 20))
        self.assertEqual(result, 10.0)


if __name__ == "__main__":
    run_tests()
