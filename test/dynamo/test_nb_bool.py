# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for truth-value conversion (nb_bool / generic is-true) through the
# builtin bool conversion. Covers constants, containers, dict subclasses, enum members, user-
# defined objects with __bool__ or __len__, a blocked __bool__ slot, metaclass __bool__, nn.Module
# containers and tensors. Defines a module-level enum used by one test. Note the torch.nn import
# sits after the enum definition. Importing registers tests only.
"""Tests for nb_bool / generic_is_true: bool() via PyObject_IsTrue in Dynamo."""

import collections
import enum

import torch


class _Color(enum.Enum):
    # [CLASS DOC] HELPER: Two-member enum (RED, BLUE) used to check that enum members are truthy.
    RED = 1
    BLUE = 2


import torch.nn
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import make_dynamo_test


@torch._dynamo.config.patch("enable_trace_unittest", True)
class NbBoolTests(TestCase):
    # [CLASS DOC] Test case for truth-value conversion in Dynamo; the class is patched with the
    # enable_trace_unittest dynamo config so assert methods can be traced. Tests marked
    # make_dynamo_test run eagerly then compiled; the others compare torch.compile results to
    # eager.
    # --- Scalar constants (ConstantVariable path) ---

    @make_dynamo_test
    def test_int(self):
        # [FUNCTION DOC] Checks bool of zero, one and minus one ints: only zero is false.
        self.assertEqual(bool(0), False)
        self.assertEqual(bool(1), True)
        self.assertEqual(bool(-1), True)

    @make_dynamo_test
    def test_float(self):
        # [FUNCTION DOC] Checks bool of floats: zero and negative zero are false, a nonzero float
        # is true.
        self.assertEqual(bool(0.0), False)
        self.assertEqual(bool(-0.0), False)
        self.assertEqual(bool(3.14), True)

    @make_dynamo_test
    def test_none(self):
        # [FUNCTION DOC] Checks bool of None is false.
        self.assertEqual(bool(None), False)

    @make_dynamo_test
    def test_str(self):
        # [FUNCTION DOC] Checks bool of strings: empty is false, non-empty true.
        self.assertEqual(bool(""), False)
        self.assertEqual(bool("nonempty"), True)

    @make_dynamo_test
    def test_bytes(self):
        # [FUNCTION DOC] Checks bool of bytes: empty is false, non-empty true.
        self.assertEqual(bool(b""), False)
        self.assertEqual(bool(b"hello"), True)

    @make_dynamo_test
    def test_bool(self):
        # [FUNCTION DOC] Only compares the bool literals False and True to themselves; it does not
        # call the bool conversion, so it is a trivial sanity check.
        self.assertEqual(False, False)
        self.assertEqual(True, True)

    @make_dynamo_test
    def test_complex_zero(self):
        # [FUNCTION DOC] Checks bool of the complex zero is false.
        self.assertEqual(bool(0j), False)

    @make_dynamo_test
    def test_complex_nonzero(self):
        # [FUNCTION DOC] Checks bool of a complex number with nonzero parts is true.
        self.assertEqual(bool(1 + 2j), True)

    @make_dynamo_test
    def test_complex_real_nonzero_imag_zero(self):
        # [FUNCTION DOC] Checks bool of a complex number with nonzero real and zero imaginary part
        # is true.
        self.assertEqual(bool(1 + 0j), True)

    @make_dynamo_test
    def test_complex_real_zero_imag_nonzero(self):
        # [FUNCTION DOC] Checks bool of a complex number with zero real and nonzero imaginary part
        # is true.
        self.assertEqual(bool(0 + 1j), True)

    # --- Containers (length fallback / _bool_from_length path) ---

    @make_dynamo_test
    def test_empty_list(self):
        # [FUNCTION DOC] Checks an empty list is false.
        self.assertEqual(bool([]), False)

    @make_dynamo_test
    def test_nonempty_list(self):
        # [FUNCTION DOC] Checks a non-empty list is true.
        self.assertEqual(bool([1, 2, 3]), True)

    @make_dynamo_test
    def test_empty_dict(self):
        # [FUNCTION DOC] Checks an empty dict is false.
        self.assertEqual(bool({}), False)

    @make_dynamo_test
    def test_nonempty_dict(self):
        # [FUNCTION DOC] Checks a non-empty dict is true.
        self.assertEqual(bool({"a": 1}), True)

    @make_dynamo_test
    def test_empty_tuple(self):
        # [FUNCTION DOC] Checks an empty tuple is false.
        self.assertEqual(bool(()), False)

    @make_dynamo_test
    def test_nonempty_tuple(self):
        # [FUNCTION DOC] Checks a one-element tuple is true.
        self.assertEqual(bool((1,)), True)

    @make_dynamo_test
    def test_empty_set(self):
        # [FUNCTION DOC] Checks an empty set is false.
        self.assertEqual(bool(set()), False)

    @make_dynamo_test
    def test_nonempty_set(self):
        # [FUNCTION DOC] Checks a non-empty set is true.
        self.assertEqual(bool({1, 2}), True)

    @make_dynamo_test
    def test_empty_frozenset(self):
        # [FUNCTION DOC] Checks an empty frozenset is false.
        self.assertEqual(bool(frozenset()), False)

    @make_dynamo_test
    def test_nonempty_frozenset(self):
        # [FUNCTION DOC] Checks a non-empty frozenset is true.
        self.assertEqual(bool(frozenset({1})), True)

    @make_dynamo_test
    def test_empty_range(self):
        # [FUNCTION DOC] Checks a zero-length range is false.
        self.assertEqual(bool(range(0)), False)

    @make_dynamo_test
    def test_nonempty_range(self):
        # [FUNCTION DOC] Checks a range of length five is true.
        self.assertEqual(bool(range(5)), True)

    # --- dict subclasses ---

    @make_dynamo_test
    def test_empty_defaultdict(self):
        # [FUNCTION DOC] Checks an empty defaultdict is false.
        d = collections.defaultdict(int)
        self.assertEqual(bool(d), False)

    @make_dynamo_test
    def test_nonempty_defaultdict(self):
        # [FUNCTION DOC] Checks a defaultdict with one item is true.
        d = collections.defaultdict(int, {"x": 1})
        self.assertEqual(bool(d), True)

    @make_dynamo_test
    def test_empty_counter(self):
        # [FUNCTION DOC] Checks an empty Counter is false.
        c = collections.Counter()
        self.assertEqual(bool(c), False)

    @make_dynamo_test
    def test_nonempty_counter(self):
        # [FUNCTION DOC] Checks a Counter built from a three-character string is true.
        c = collections.Counter("abc")
        self.assertEqual(bool(c), True)

    # --- Enum (UserDefinedClassVariable / ConstantVariable path) ---

    @make_dynamo_test
    def test_enum_member(self):
        # [FUNCTION DOC] Checks that both members of the module-level enum are true.
        self.assertEqual(bool(_Color.RED), True)
        self.assertEqual(bool(_Color.BLUE), True)

    # --- UserDefinedObjectVariable tests (torch.compile path) ---

    def test_user_defined_with_bool(self):
        # [FUNCTION DOC] Compiles a function that picks between two tensor expressions based on
        # the truth of a user-defined object with __bool__; the compiled result is compared to
        # eager for an object whose value is positive and, after a dynamo reset and recompile, for
        # a negative value.
        class MyObj:
            # [CLASS DOC] HELPER: Test object holding a value; __bool__ is true when the value is
            # positive.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores the value argument in the val attribute.
                self.val = val

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns whether the stored value is greater than zero.
                return self.val > 0

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the object is truthy, otherwise
            # the input minus one.
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(5)), compiled(x, MyObj(5)))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, MyObj(-1)), compiled(x, MyObj(-1)))

    def test_user_defined_with_len(self):
        # [FUNCTION DOC] Same as the previous test but for an object defining only __len__:
        # truthiness comes from the length being nonzero. Checks a two-element and, after reset,
        # an empty container.
        class Container:
            # [CLASS DOC] HELPER: Test container holding a list of items; defines only __len__.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores the items argument in the items attribute.
                self.items = items

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the number of stored items.
                return len(self.items)

        def fn(x, c):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the container is truthy,
            # otherwise the input minus one.
            return x + 1 if bool(c) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([1, 2])), compiled(x, Container([1, 2])))
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend="eager")
        self.assertEqual(fn(x, Container([])), compiled(x, Container([])))

    def test_user_defined_no_bool_no_len(self):
        # [FUNCTION DOC] Checks that an object with neither __bool__ nor __len__ is truthy;
        # compiled with fullgraph and compared with eager.
        class Plain:
            # [CLASS DOC] HELPER: Empty class with no truth-value methods.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the object is truthy, otherwise
            # the input minus one.
            return x + 1 if bool(obj) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, Plain()), compiled(x, Plain()))

    def test_user_defined_bool_returns_non_bool_raises(self):
        # [FUNCTION DOC] Checks that __bool__ returning a non-bool (the int one) raises TypeError
        # both in eager and when called through a compiled function.
        class BadBool:
            # [CLASS DOC] HELPER: Class whose __bool__ wrongly returns the int one.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns the int one, which is not a bool, so conversion
                # must raise TypeError.
                return 1

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the object is truthy, otherwise
            # the input minus one.
            return x + 1 if bool(obj) else x - 1

        with self.assertRaises(TypeError):
            bool(BadBool())
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(torch.randn(4), BadBool())

    def test_if_cond_user_defined_object_returns_self(self):
        # [FUNCTION DOC] Checks that using an object whose __bool__ returns itself in an if
        # condition raises TypeError inside a compiled function; the error message is caught in
        # the function and the compiled result string equals the eager one.
        class MyObj:
            # [CLASS DOC] HELPER: Class whose __bool__ returns the object itself, which is
            # invalid.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns the instance itself (not a bool), so truth
                # testing raises TypeError.
                return self

        def fn(a, obj):
            # [FUNCTION DOC] HELPER: Tries the if condition on the object; returns input plus one
            # or minus one, or on TypeError returns the error message string.
            try:
                if obj:
                    return a + 1
                return a - 1
            except TypeError as e:
                return str(e)

        x = torch.rand(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        out = compiled(x, MyObj())
        self.assertTrue(type(out) is str)
        self.assertEqual(out, fn(x, MyObj()))

    def test_bool_user_defined_object_raises_typeerror(self):
        # [FUNCTION DOC] Checks that explicit bool conversion of an int subclass instance whose
        # __bool__ returns itself raises TypeError; the compiled function returns the message
        # string equal to eager.
        class Baz(int):
            # [CLASS DOC] HELPER: Int subclass with an invalid __bool__ returning self.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns the instance itself, which is not a bool.
                return self

        def fn(obj):
            # [FUNCTION DOC] HELPER: Converts the argument with the builtin bool conversion; on
            # TypeError returns the message string.
            try:
                bool(obj)
            except TypeError as e:
                return str(e)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        out = opt_fn(Baz())
        self.assertTrue(type(out) is str)
        self.assertEqual(fn(Baz()), out)

    # --- Blocked slot: __bool__ = None ---

    def test_user_defined_bool_none_raises(self):
        # [FUNCTION DOC] Checks that setting __bool__ to None on a class blocks the slot:
        # converting the instance to bool raises TypeError (asserted inside the compiled
        # function), and the function returns the sine of its input; compiled and eager results
        # are compared.
        class NoBool:
            # [CLASS DOC] HELPER: Class whose __bool__ attribute is None (slot disabled).
            __bool__ = None

        obj = NoBool()

        def fn(x):
            # [FUNCTION DOC] HELPER: Asserts that converting the captured instance raises
            # TypeError, then returns the sine of the input.
            with self.assertRaises(TypeError):
                str(bool(obj))
            return x.sin()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        eager_result = fn(torch.tensor(0))
        self.assertEqual(result, eager_result)

    # --- Metaclass with __bool__ (UserDefinedClassVariable path) ---

    def test_metaclass_bool(self):
        # [FUNCTION DOC] Checks bool conversion of classes themselves: a class whose metaclass
        # defines __bool__ returning False is false, while a class whose metaclass defines no such
        # method is true. The function adds both bool results to the input tensor.
        class Foo(type):
            # [CLASS DOC] HELPER: Metaclass defining __bool__ that returns False for any class
            # using it.
            def __bool__(cls):
                # [FUNCTION DOC] HELPER: Returns False.
                return False

        class A(metaclass=Foo):
            # [CLASS DOC] HELPER: Class using the falsy metaclass.
            pass

        class Bar(type):
            # [CLASS DOC] HELPER: Metaclass without any truth-value method.
            pass

        class B(metaclass=Bar):
            # [CLASS DOC] HELPER: Class using the plain metaclass; truthy.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus the bool of A plus the bool of B.
            # A's metaclass defines __bool__ returning False; B's does not (truthy).
            return x + bool(A) + bool(B)

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- nn.Module (NNModuleVariable path) ---

    def test_nn_module_nonempty(self):
        # [FUNCTION DOC] Checks a non-empty nn.ModuleList is truthy under fullgraph compile,
        # matching eager.
        mod = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the captured module list is
            # truthy, otherwise minus one.
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_nn_module_empty(self):
        # [FUNCTION DOC] Checks an empty nn.ModuleList is falsy under fullgraph compile, matching
        # eager.
        mod = torch.nn.ModuleList()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one if the captured empty module list
            # is truthy, otherwise minus one.
            return x + 1 if bool(mod) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    # --- Tensor (TensorVariable path) ---

    def test_tensor_nonzero(self):
        # [FUNCTION DOC] Checks bool of a nonzero scalar tensor created inside the compiled
        # function is true.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a scalar tensor of value one and returns the input
            # plus or minus one based on its truth.
            t = torch.tensor(1)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_tensor_zero(self):
        # [FUNCTION DOC] Checks bool of a zero scalar tensor created inside the compiled function
        # is false.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a scalar tensor of value zero and returns the input
            # plus or minus one based on its truth.
            t = torch.tensor(0)
            return x + 1 if bool(t) else x - 1

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))


if __name__ == "__main__":
    run_tests()
