# Owner(s): ["module: onnx"]
# [MODULE DOC] Tests for the internal ONNX symbolic registration module: the opset
# dispatch helper, the override dictionary, and the decorators that register symbolic
# functions in a global registry.
"""Unit tests for the internal registration wrapper module."""

from collections.abc import Sequence

from torch.onnx import errors
from torch.onnx._internal.torchscript_exporter import registration
from torch.testing._internal import common_utils


@common_utils.instantiate_parametrized_tests
class TestGlobalHelpers(common_utils.TestCase):
    # [CLASS DOC] Parametrized tests for module-level helper functions of the registration
    # module, currently the opset dispatch helper.
    @common_utils.parametrize(
        "available_opsets, target, expected",
        [
            ((7, 8, 9, 10, 11), 16, 11),
            ((7, 8, 9, 10, 11), 11, 11),
            ((7, 8, 9, 10, 11), 10, 10),
            ((7, 8, 9, 10, 11), 9, 9),
            ((7, 8, 9, 10, 11), 8, 8),
            ((7, 8, 9, 10, 11), 7, 7),
            ((9, 10, 16), 16, 16),
            ((9, 10, 16), 15, 10),
            ((9, 10, 16), 10, 10),
            ((9, 10, 16), 9, 9),
            ((9, 10, 16), 8, 9),
            ((9, 10, 16), 7, 9),
            ((7, 9, 10, 16), 16, 16),
            ((7, 9, 10, 16), 10, 10),
            ((7, 9, 10, 16), 9, 9),
            ((7, 9, 10, 16), 8, 9),
            ((7, 9, 10, 16), 7, 7),
            ([17], 16, None),  # New op added in 17
            ([9], 9, 9),
            ([9], 8, 9),
            ([], 16, None),
            ([], 9, None),
            ([], 8, None),
            # Ops registered at opset 1 found as a fallback when target >= 9
            ([1], 16, 1),
        ],
    )
    def test_dispatch_opset_version_returns_correct_version(
        self, available_opsets: Sequence[int], target: int, expected: int
    ):
        # [FUNCTION DOC] Parametrized test: asks the opset dispatch helper which registered
        # version serves the target opset and compares it with the expected version (none
        # when nothing applies).
        actual = registration._dispatch_opset_version(target, available_opsets)
        self.assertEqual(actual, expected)


class TestOverrideDict(common_utils.TestCase):
    # [CLASS DOC] Tests for the override dictionary that layers custom entries over base
    # entries; the fixture dictionary is recreated before each test.
    def setUp(self):
        # [FUNCTION DOC] Runs the base setup, then creates an empty override dictionary
        # mapping strings to integers as the fixture.
        super().setUp()
        self.override_dict: registration.OverrideDict[str, int] = (
            registration.OverrideDict()
        )

    def test_get_item_returns_base_value_when_no_override(self):
        # [FUNCTION DOC] Checks item lookup returns base values when nothing is overridden,
        # and that the length is two.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)

        self.assertEqual(self.override_dict["a"], 42)
        self.assertEqual(self.override_dict["b"], 0)
        self.assertEqual(len(self.override_dict), 2)

    def test_get_item_returns_overridden_value_when_override(self):
        # [FUNCTION DOC] Checks item lookup returns the override where present, the base
        # value otherwise, includes override-only keys, and that the length is three.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(self.override_dict["b"], 0)
        self.assertEqual(self.override_dict["c"], 1)
        self.assertEqual(len(self.override_dict), 3)

    def test_get_item_raises_key_error_when_not_found(self):
        # [FUNCTION DOC] Checks that item lookup of an unknown key raises a key error.
        self.override_dict.set_base("a", 42)

        with self.assertRaises(KeyError):
            self.override_dict["nonexistent_key"]

    def test_get_returns_overridden_value_when_override(self):
        # [FUNCTION DOC] Checks the get method returns the override for overridden keys,
        # base values otherwise, includes override-only keys, and that the length is three.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict.get("a"), 100)
        self.assertEqual(self.override_dict.get("b"), 0)
        self.assertEqual(self.override_dict.get("c"), 1)
        self.assertEqual(len(self.override_dict), 3)

    def test_get_returns_none_when_not_found(self):
        # [FUNCTION DOC] Checks the get method returns none for an unknown key.
        self.override_dict.set_base("a", 42)

        self.assertEqual(self.override_dict.get("nonexistent_key"), None)

    def test_in_base_returns_true_for_base_value(self):
        # [FUNCTION DOC] Checks membership covers base and overridden keys, while the
        # in-base query is true for base keys even if overridden and false for override-only
        # and unknown keys.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertIn("a", self.override_dict)
        self.assertIn("b", self.override_dict)
        self.assertIn("c", self.override_dict)

        self.assertTrue(self.override_dict.in_base("a"))
        self.assertTrue(self.override_dict.in_base("b"))
        self.assertFalse(self.override_dict.in_base("c"))
        self.assertFalse(self.override_dict.in_base("nonexistent_key"))

    def test_overridden_returns_true_for_overridden_value(self):
        # [FUNCTION DOC] Checks the overridden query is true for overridden keys (including
        # one with no base entry) and false for base-only and unknown keys.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertTrue(self.override_dict.overridden("a"))
        self.assertFalse(self.override_dict.overridden("b"))
        self.assertTrue(self.override_dict.overridden("c"))
        self.assertFalse(self.override_dict.overridden("nonexistent_key"))

    def test_remove_override_removes_overridden_value(self):
        # [FUNCTION DOC] Overrides one base key and one new key, removes both overrides, and
        # checks the base value returns for the first, the second disappears, and neither is
        # reported as overridden.
        self.override_dict.set_base("a", 42)
        self.override_dict.set_base("b", 0)
        self.override_dict.override("a", 100)
        self.override_dict.override("c", 1)

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(self.override_dict["c"], 1)

        self.override_dict.remove_override("a")
        self.override_dict.remove_override("c")
        self.assertEqual(self.override_dict["a"], 42)
        self.assertEqual(self.override_dict.get("c"), None)
        self.assertFalse(self.override_dict.overridden("a"))
        self.assertFalse(self.override_dict.overridden("c"))

    def test_remove_override_removes_overridden_key(self):
        # [FUNCTION DOC] Adds only an override, then removes it, and checks the key
        # disappears: length becomes zero and the key is no longer contained.
        self.override_dict.override("a", 100)
        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(len(self.override_dict), 1)
        self.override_dict.remove_override("a")
        self.assertEqual(len(self.override_dict), 0)
        self.assertNotIn("a", self.override_dict)

    def test_overridden_key_precedes_base_key_regardless_of_insert_order(self):
        # [FUNCTION DOC] Sets a base entry, overrides the key, then sets the base again;
        # checks the overridden value is still returned and the length stays one.
        self.override_dict.set_base("a", 42)
        self.override_dict.override("a", 100)
        self.override_dict.set_base("a", 0)

        self.assertEqual(self.override_dict["a"], 100)
        self.assertEqual(len(self.override_dict), 1)

    def test_bool_is_true_when_not_empty(self):
        # [FUNCTION DOC] Checks truthiness: an empty dictionary is false, and it becomes
        # true after adding an override only, then a base entry too, and stays true after
        # removing the override leaves only the base entry.
        if self.override_dict:
            self.fail("OverrideDict should be false when empty")
        self.override_dict.override("a", 1)
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        self.override_dict.set_base("a", 42)
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")
        self.override_dict.remove_override("a")
        if not self.override_dict:
            self.fail("OverrideDict should be true when not empty")


class TestRegistrationDecorators(common_utils.TestCase):
    # [CLASS DOC] Tests for the symbolic and custom symbolic registration decorators; each
    # test registers the test operator in the global registry, and teardown removes it.
    def tearDown(self) -> None:
        # [FUNCTION DOC] HELPER: Removes the test operator name from the global registry
        # mapping so tests do not leak registrations.
        registration.registry._registry.pop("test::test_op", None)

    def test_onnx_symbolic_registers_function(self):
        # [FUNCTION DOC] Checks the operator is unregistered, registers a function for opset
        # nine with the symbolic decorator, then checks it is registered and the group
        # returns that very function for opset nine.
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        @registration.onnx_symbolic("test::test_op", opset=9)
        def test(g, x):
            # [FUNCTION DOC] HELPER: Symbolic function that emits a test node on the graph
            # for the input.
            return g.op("test", x)

        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))
        function_group = registration.registry.get_function_group("test::test_op")
        if function_group is None:
            raise AssertionError("function_group is None")
        self.assertEqual(function_group.get(9), test)

    def test_onnx_symbolic_registers_function_applied_decorator_when_provided(self):
        # [FUNCTION DOC] Registers a function with an extra decorator and checks that the
        # registered function is the decorated wrapper: the wrapper does not run at
        # registration, and runs when the registered function is called.
        wrapper_called = False

        def decorator(func):
            # [FUNCTION DOC] HELPER: Decorator that wraps a function so the call sets
            # the enclosing flag.
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Records on the enclosing test's flag that it
                # ran, then calls the wrapped function with the same arguments and
                # returns its result.
                nonlocal wrapper_called
                wrapper_called = True
                return func(*args, **kwargs)

            return wrapper

        @registration.onnx_symbolic("test::test_op", opset=9, decorate=[decorator])
        def test():
            # [FUNCTION DOC] HELPER: Placeholder symbolic function that takes no arguments
            # and returns nothing.
            return

        function_group = registration.registry.get_function_group("test::test_op")
        if function_group is None:
            raise AssertionError("function_group is None")
        registered_function = function_group[9]
        self.assertFalse(wrapper_called)
        registered_function()
        self.assertTrue(wrapper_called)

    def test_onnx_symbolic_raises_warning_when_overriding_function(self):
        # [FUNCTION DOC] Registers a function and checks that registering another function
        # for the same name and opset emits the exporter warning that the symbolic function
        # is already registered.
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        @registration.onnx_symbolic("test::test_op", opset=9)
        def test1():
            # [FUNCTION DOC] HELPER: Placeholder symbolic function that takes no arguments and
            # returns nothing; registered first for the test operator and opset nine.
            return

        with self.assertWarnsRegex(
            errors.OnnxExporterWarning,
            "Symbolic function 'test::test_op' already registered",
        ):

            @registration.onnx_symbolic("test::test_op", opset=9)
            def test2():
                # [FUNCTION DOC] HELPER: Placeholder symbolic function that returns
                # nothing; registering it a second time for the same name and opset
                # triggers the override warning.
                return

    def test_custom_onnx_symbolic_registers_custom_function(self):
        # [FUNCTION DOC] Registers a function with the custom decorator and checks the
        # operator becomes registered for opset nine and the group returns that function.
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        @registration.custom_onnx_symbolic("test::test_op", opset=9)
        def test(g, x):
            # [FUNCTION DOC] HELPER: Symbolic function that emits a test node on the graph
            # for the input.
            return g.op("test", x)

        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))
        function_group = registration.registry.get_function_group("test::test_op")
        if function_group is None:
            raise AssertionError("function_group is None")
        self.assertEqual(function_group.get(9), test)

    def test_custom_onnx_symbolic_overrides_existing_function(self):
        # [FUNCTION DOC] Registers a regular symbolic function and then a custom one for
        # the same name and opset, and checks the registry now returns the custom function
        # for that opset.
        self.assertFalse(registration.registry.is_registered_op("test::test_op", 9))

        @registration.onnx_symbolic("test::test_op", opset=9)
        def test_original():
            # [FUNCTION DOC] HELPER: Returns the string original; registered first as a
            # regular symbolic function.
            return "original"

        self.assertTrue(registration.registry.is_registered_op("test::test_op", 9))

        @registration.custom_onnx_symbolic("test::test_op", opset=9)
        def test_custom():
            # [FUNCTION DOC] HELPER: Returns the string custom; used as a custom replacement
            # for the original function.
            return "custom"

        function_group = registration.registry.get_function_group("test::test_op")
        if function_group is None:
            raise AssertionError("function_group is None")
        self.assertEqual(function_group.get(9), test_custom)


if __name__ == "__main__":
    common_utils.run_tests()
