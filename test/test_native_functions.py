# Owner(s): ["module: unknown"]

# [MODULE DOC] End-to-end tests of native_functions.yaml features: error messages for string
# values passed where integer lists are expected, optional float and int list arguments,
# optional filled int lists, and string default arguments. Uses test-only operators in
# torch._C._nn, in eager, scripted and traced form. Defines two small wrapper modules.
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, skipIfTorchDynamo, HardwareClassification

# End-to-end tests of features in native_functions.yaml


class FloatListWrapperModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping the test operator taking a tensor and an optional float list.
    # Used in eager and scripted form; the optional list is annotated for scripting.
    def forward(self, values, incr: list[float] | None):
        # [FUNCTION DOC] Forwards the values tensor and the optional float list to the test
        # operator and returns its result.
        return torch._C._nn._test_optional_floatlist(values, incr)


class IntListWrapperModule(torch.nn.Module):
    # [CLASS DOC] Module wrapping the test operator taking a tensor and an optional int list.
    # Used in eager and scripted form; the optional list is annotated for scripting.
    def forward(self, values, incr: list[int] | None):
        # [FUNCTION DOC] Forwards the values tensor and the optional int list to the test
        # operator and returns its result.
        return torch._C._nn._test_optional_intlist(values, incr)


class TestNativeFunctions(TestCase):
    # [CLASS DOC] Test case for argument parsing of native functions with list, optional and
    # string-default arguments. Helper methods build bad inputs and share checks between eager,
    # scripted and traced variants. The class attribute marks the test as hardware generic.
    hw_classification = HardwareClassification.GENERIC

    def _lists_with_str(self):
        # [FUNCTION DOC] HELPER: Returns a list of invalid list-like arguments each containing or
        # being a string: tuples, lists and a bare string.
        return [
            ("foo",),
            (2, "foo"),
            ("foo", 3),
            ["foo"],
            [2, "foo"],
            ["foo", 3],
            "foo",
        ]

    def _test_raises_str_typeerror(self, fn):
        # [FUNCTION DOC] HELPER: For each invalid argument, asserts the given callable raises a
        # type error mentioning a string, then calls it again and prints the error message.
        for arg in self._lists_with_str():
            self.assertRaisesRegex(TypeError, "str", lambda: fn(arg))
            try:
                fn(arg)
            except TypeError as e:
                print(e)

    def test_symintlist_error(self):
        # [FUNCTION DOC] Checks that passing string-containing sizes to the pad function raises a
        # type error naming a string.
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    def test_vararg_symintlist_error(self):
        # [FUNCTION DOC] Checks the same string type error for the variadic and sequence forms of
        # the tensor creation function rand.
        self._test_raises_str_typeerror(lambda arg: torch.rand(arg))
        self._test_raises_str_typeerror(lambda arg: torch.rand(*arg))

    def test_symintlist_error_with_overload_but_is_unique(self):
        # [FUNCTION DOC] Checks the string type error for the set_ method, whose size argument
        # is a symbolic int list on an overload that is unique by arity.
        x = torch.randn(1)
        y = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.set_(y, 0, arg))

    def test_symintlist_error_with_overload(self):
        # [FUNCTION DOC] Checks the string type error for the view method, which has several
        # overloads taking a symbolic int list.
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: x.view(arg))

    def test_intlist_error_with_overload(self):
        # [FUNCTION DOC] Checks the string type error for the pad function with an int list
        # argument, in the presence of overloads.
        x = torch.randn(1)
        self._test_raises_str_typeerror(lambda arg: torch._C._nn.pad(x, arg))

    #
    # optional float list
    #

    def do_test_optional_floatlist_with_module(self, module):
        # [FUNCTION DOC] HELPER: Shared check for the optional float list operator. With no list
        # the result must alias the input (changes to the input show in the result); with a list
        # the result is the input plus the list elementwise. The module is any callable taking
        # values and the optional list.
        values = torch.tensor([1.5, 2.5], dtype=torch.float)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3.5
        self.assertEqual(values, returned)

        returned = module(values, [5.1, 4.1])
        self.assertEqual(values, torch.tensor([3.5, 2.5], dtype=torch.float))
        self.assertEqual(returned, torch.tensor([8.6, 6.6], dtype=torch.float))

    def trace_optional_floatlist(self, const):
        # [FUNCTION DOC] HELPER: Traces the optional float list operator with the given constant
        # (None or a list of floats) baked in and returns the traced function.
        def wrapper(values):
            # [FUNCTION DOC] HELPER: Calls the test operator on the values with the captured
            # constant and returns the result; traced by the enclosing method.
            return torch._C._nn._test_optional_floatlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1.5, 2.5], dtype=torch.float))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_floatlist(self):
        # [FUNCTION DOC] Runs the shared optional float list check on the eager module, the
        # scripted module, and a plain function dispatching between two traced functions (one
        # for a missing list, one for a specific list). Skipped under TorchDynamo.
        self.do_test_optional_floatlist_with_module(FloatListWrapperModule())
        self.do_test_optional_floatlist_with_module(torch.jit.script(FloatListWrapperModule()))

        traced_none = self.trace_optional_floatlist(None)
        traced_list = self.trace_optional_floatlist([5.1, 4.1])

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and [5.1, 4.1].
        def fake_module(values, const):
            # [FUNCTION DOC] HELPER: Dispatches to the traced function matching the constant
            # (None or the specific float list); raises a generic exception otherwise.
            if const is None:
                return traced_none(values)
            if const == [5.1, 4.1]:
                return traced_list(values)
            raise Exception("Invalid argument")  # noqa: TRY002

        self.do_test_optional_floatlist_with_module(fake_module)

    def test_optional_floatlist_invalid(self):
        # [FUNCTION DOC] Checks error behavior for invalid optional float list arguments: a list
        # of strings and a tensor each raise type errors in eager and runtime errors in scripted
        # form, with messages matched by pattern.
        with self.assertRaisesRegex(TypeError, "must be tuple of floats, not list"):
            FloatListWrapperModule()(torch.zeros(1), ["hi"])

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), ["hi"])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            FloatListWrapperModule()(torch.zeros(1), torch.zeros(1))

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(FloatListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional int list
    #

    def do_test_optional_intlist_with_module(self, module):
        # [FUNCTION DOC] HELPER: Shared check for the optional int list operator, same contract as
        # the float variant: aliasing when the list is missing, elementwise addition otherwise.
        values = torch.tensor([1, 2], dtype=torch.int)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3
        self.assertEqual(values, returned)

        returned = module(values, [5, 4])
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([8, 6], dtype=torch.int))

    def trace_optional_intlist(self, const):
        # [FUNCTION DOC] HELPER: Traces the optional int list operator with the given constant
        # (None or a list of ints) baked in and returns the traced function.
        def wrapper(values):
            # [FUNCTION DOC] HELPER: Calls the test operator on the values with the captured
            # constant and returns the result; traced by the enclosing method.
            return torch._C._nn._test_optional_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_intlist(self):
        # [FUNCTION DOC] Runs the shared optional int list check on the eager module, the
        # scripted module, and a function dispatching between two traced functions. Skipped under
        # TorchDynamo.
        self.do_test_optional_intlist_with_module(IntListWrapperModule())
        self.do_test_optional_intlist_with_module(torch.jit.script(IntListWrapperModule()))

        traced_none = self.trace_optional_intlist(None)
        traced_list = self.trace_optional_intlist([5, 4])

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and [5, 4].
        def fake_module(values, const):
            # [FUNCTION DOC] HELPER: Dispatches to the traced function matching the constant
            # (None or the specific int list); raises a generic exception otherwise.
            if const is None:
                return traced_none(values)
            if const == [5, 4]:
                return traced_list(values)
            raise Exception("Invalid argument")  # noqa: TRY002

        self.do_test_optional_intlist_with_module(fake_module)

    def test_optional_intlist_invalid(self):
        # [FUNCTION DOC] Checks error behavior for invalid optional int list arguments: a list of
        # floats and a tensor each raise type errors in eager and runtime errors in scripted
        # form, with messages matched by pattern.
        with self.assertRaisesRegex(TypeError, "must be .* but found"):
            IntListWrapperModule()(torch.zeros(1), [0.5])

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), [0.5])

        with self.assertRaisesRegex(TypeError, "must be .* Tensor"):
            IntListWrapperModule()(torch.zeros(1), torch.zeros(1))

        with self.assertRaisesRegex(RuntimeError, "value of type .* instead found type"):
            torch.jit.script(IntListWrapperModule())(torch.zeros(1), torch.zeros(1))

    #
    # optional filled int list
    #

    def do_test_optional_filled_intlist_with_module(self, module):
        # [FUNCTION DOC] HELPER: Shared check for the optional filled int list operator, where a
        # single int is expanded to a list. Missing value must alias the input; an int value is
        # added to every element of the input.
        values = torch.tensor([1, 2], dtype=torch.int)

        returned = module(values, None)
        self.assertEqual(values, returned)
        # Make sure that it's an alias, indicating that the operator saw a nullopt.
        values[0] = 3
        self.assertEqual(values, returned)

        returned = module(values, 10)
        self.assertEqual(values, torch.tensor([3, 2], dtype=torch.int))
        self.assertEqual(returned, torch.tensor([13, 12], dtype=torch.int))

    def trace_optional_filled_intlist(self, const):
        # [FUNCTION DOC] HELPER: Traces the optional filled int list operator with the given
        # constant (None or an int) baked in and returns the traced function.
        def wrapper(values):
            # [FUNCTION DOC] HELPER: Calls the test operator on the values with the captured
            # constant and returns the result; traced by the enclosing method.
            return torch._C._nn._test_optional_filled_intlist(values, const)
        return torch.jit.trace(wrapper, torch.tensor([1, 2], dtype=torch.int))

    @skipIfTorchDynamo("Not a suitable test for TorchDynamo")
    def test_optional_filled_intlist(self):
        # [FUNCTION DOC] Checks the filled int list operator is equivalent whether given a full
        # list or a single int, in eager and scripted form, then runs the shared check with a
        # function dispatching between two traced functions. Skipped under TorchDynamo.

        def f(n: int):
            # [FUNCTION DOC] HELPER: Calls the operator on a two-element int tensor with an
            # explicit pair and with a single int, returning both results as a tuple.
            x = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), (n, n))
            y = torch._C._nn._test_optional_filled_intlist(torch.tensor([1, 1], dtype=torch.int), n)
            return x, y

        # eager
        returned = f(10)
        self.assertEqual(returned[0], returned[1])

        # scripted
        s = torch.jit.script(f)
        returned = s(10)
        self.assertEqual(returned[0], returned[1])

        # traced
        traced_none = self.trace_optional_filled_intlist(None)
        traced_int = self.trace_optional_filled_intlist(10)

        # Not really a module, just lets us use our two traced functions to handle
        # the specific cases of passing None and 10.
        def fake_module(values, const):
            # [FUNCTION DOC] HELPER: Dispatches to the traced function matching the constant
            # (None or ten); raises a generic exception otherwise.
            if const is None:
                return traced_none(values)
            if const == 10:
                return traced_int(values)
            raise Exception("Invalid argument")  # noqa: TRY002

        self.do_test_optional_filled_intlist_with_module(fake_module)

    def test_string_defaults(self):
        # [FUNCTION DOC] Checks the string default test operator: it runs with defaults, raises a
        # runtime error mentioning A when the first string is passed empty and mentioning B when
        # the second is passed empty, and also runs when called from a scripted function.
        dummy = torch.rand(1)
        fn = torch._C._nn._test_string_default
        fn(dummy)

        with self.assertRaisesRegex(RuntimeError, "A"):
            fn(dummy, a="")

        with self.assertRaisesRegex(RuntimeError, "B"):
            fn(dummy, b="")

        def f(x):
            # [FUNCTION DOC] HELPER: Calls the string default operator with defaults; scripted by
            # the enclosing method.
            torch._C._nn._test_string_default(x)
        scripted_fn = torch.jit.script(f)
        scripted_fn(dummy)


if __name__ == '__main__':
    run_tests()
