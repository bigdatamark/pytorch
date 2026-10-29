# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo's ID_MATCH and TYPE_MATCH guards render in the guard manager
# string as ___check_obj_id / ___check_type_id calls with an anonymized id and a readable type
# name, and that rendering guard text never calls user-defined __repr__. Inspects the cache
# entry list of a compiled function; no import-time side effects.
"""
Test for TYPE_MATCH guard and ___check_type_id function.

This test demonstrates how the TYPE_MATCH guard works in PyTorch Dynamo.
When a function is compiled, Dynamo installs guards to ensure the compiled
code remains valid. TYPE_MATCH guards ensure that values maintain their
exact type (using type identity, not just type equality).
"""

import re

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.eval_frame import _debug_get_cache_entry_list
from torch.testing._internal.common_utils import munge_exc


class TestCheckTypeId(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case that compiles small functions, reads the guard manager string
    # from the cache entry and checks the guard lines for type identity guards.
    @staticmethod
    def _find_guard_lines(guard_manager_str: str, keyword: str) -> list[str]:
        # [FUNCTION DOC] HELPER: Static method. Take the guard manager string, normalize it with
        # munge_exc, replace every run of seven or more digits with the placeholder text for a
        # type id, and return the list of lines containing the keyword (matched literally).
        # Normalize and anonymize type IDs, then return lines containing the keyword
        normalized = re.sub(
            r"\d{7,}", "<type_id>", munge_exc(guard_manager_str), flags=re.MULTILINE
        )
        pattern = re.compile(rf"^.*{re.escape(keyword)}.*$", re.MULTILINE)
        return pattern.findall(normalized)

    def test_type_match_with_different_values(self):
        # [FUNCTION DOC] Compile (eager, fullgraph) a function multiplying by an attribute of a
        # local class, check the result, then fetch the single cache entry and its ID_MATCH
        # guard line. Assert it uses the object-id check on the Config class, that the line
        # prefix equals the expected anonymized text, and that the type name suffix mentions
        # this test's local Config class.
        """
        Test that TYPE_MATCH guard correctly identifies type mismatches.

        This test compiles a function that uses a global variable and verifies:
        1. The compiled function works with values of the same type
        2. The function recompiles when the type changes
        3. The ___check_type_id/check_obj_id guard is present in the generated code
        4. The check_type_id should present the user-friendly code that specify the type
        """

        # Define a global variable that we'll guard on
        class Config:
            # [CLASS DOC] HELPER: Local class with an int class attribute multiplier set to 2.
            multiplier = 2  # int type

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x times Config.multiplier; the compiled function
            # under test.
            # This will trigger a TYPE_MATCH guard on Config.multiplier
            return x * Config.multiplier

        # Compile the function
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call - should compile and install guards
        x = torch.randn(4)
        result1 = opt_fn(x)
        expected1 = x * 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Get the cache entry to inspect guards
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        # Check that the guard string contains check_type_id
        guard_str = str(cache_entries[0].guard_manager)
        matches = self._find_guard_lines(guard_str, "ID_MATCH")
        self.assertIn("___check_obj_id", matches[0])
        self.assertIn(
            ".TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>",
            matches[0],
        )
        # Match the first part (everything before "type=")
        first_part = matches[0].split("type=")[0]
        expected_first_part = (
            "| | +- ID_MATCH: ___check_obj_id(L['Config'], <type_id>), "
        )
        self.assertEqual(first_part, expected_first_part)

        # Match the second part (the type string)
        second_part = matches[0].split("type=")[1].rstrip()
        expected_second_part = (
            "TestCheckTypeId.test_type_match_with_different_values.<locals>.Config'>"
        )
        self.assertIn(expected_second_part, second_part)

    def test_id_match_guard_text_does_not_call_repr(self):
        # [FUNCTION DOC] Verify guard text generation does not call __repr__ of guarded objects:
        # use a metaclass and an instance class that count repr calls, compile a function
        # comparing ids and using the class attribute, and assert zero repr calls, one cache
        # entry, no marker text in the guard string, and ID_MATCH guards for the instance and
        # the class whose text shows the type as a class with the expected name.
        class ReprMeta(type):
            # [CLASS DOC] HELPER: Metaclass counting __repr__ calls in a class attribute and
            # returning a fixed marker string.
            repr_calls = 0

            def __repr__(cls):
                # [FUNCTION DOC] HELPER: Increment the metaclass repr counter and return the
                # marker string.
                ReprMeta.repr_calls += 1
                return "<expensive class repr>"

        class Config(metaclass=ReprMeta):
            # [CLASS DOC] HELPER: Class created with the counting metaclass, with multiplier set
            # to 2.
            multiplier = 2

        class ReprCounter:
            # [CLASS DOC] HELPER: Class whose instances count __repr__ calls and return a marker
            # string.
            repr_calls = 0

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Increment the instance repr counter and return the
                # marker string.
                ReprCounter.repr_calls += 1
                return "<expensive instance repr>"

        obj = ReprCounter()

        def fn(x, o):
            # [FUNCTION DOC] HELPER: If the id of o equals the id of the closed-over obj return
            # x plus Config.multiplier, otherwise x plus 3.
            if id(o) == id(obj):
                return x + Config.multiplier
            return x + 3

        torch._dynamo.reset()
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.ones(1)
        self.assertTrue(torch.allclose(opt_fn(x, obj), x + 2))
        self.assertEqual(ReprCounter.repr_calls, 0)
        self.assertEqual(ReprMeta.repr_calls, 0)

        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        guard_str = str(cache_entries[0].guard_manager)
        self.assertNotIn("<expensive", guard_str)

        matches = self._find_guard_lines(guard_str, "ID_MATCH")
        object_guard = next(line for line in matches if "L['o']" in line)
        class_guard = next(line for line in matches if "Config" in line)

        self.assertIn("type=<class '", object_guard)
        self.assertIn("ReprCounter'>", object_guard)
        self.assertIn("type=<class '", class_guard)
        self.assertIn("Config'>", class_guard)

    def test_type_match_with_custom_classes(self):
        # [FUNCTION DOC] Compile a function reading attributes of a closed-over Point instance,
        # check the result, and assert the single cache entry has a TYPE_MATCH guard line whose
        # prefix is the anonymized type id check on the point local, and whose type name suffix
        # names the local Point class (the guard checks type identity).
        """
        Test TYPE_MATCH guard with custom class instances.

        Demonstrates that the guard checks type identity, not structural equality.
        """

        class Point:
            # [CLASS DOC] HELPER: Local class storing x and y.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Store x and y as attributes.
                self.x = x
                self.y = y

        class Point2D:
            # [CLASS DOC] HELPER: Second structurally identical local class; defined but unused.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Store x and y as attributes.
                self.x = x
                self.y = y

        point = Point(1, 2)

        def fn(tensor):
            # [FUNCTION DOC] HELPER: Return tensor plus point.x plus point.y.
            # Access point's attributes, triggering TYPE_MATCH guard on point
            return tensor + point.x + point.y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        # First call with Point instance
        x = torch.ones(4)
        result1 = opt_fn(x)
        expected1 = x + 1 + 2
        self.assertTrue(torch.allclose(result1, expected1))

        # Verify guard contains check_type_id
        cache_entries = _debug_get_cache_entry_list(fn.__code__)
        self.assertEqual(len(cache_entries), 1)

        guard_str = str(cache_entries[0].guard_manager)
        matches = self._find_guard_lines(guard_str, "TYPE_MATCH")
        # Match the first part (everything before "type=")
        first_part = matches[0].split("type=")[0]
        expected_first_part = (
            "| | +- TYPE_MATCH: ___check_type_id(L['point'], <type_id>), "
        )
        self.assertEqual(first_part, expected_first_part)

        # Match the second part (the type string)
        second_part = matches[0].split("type=")[1].rstrip()
        expected_second_part = (
            "TestCheckTypeId.test_type_match_with_custom_classes.<locals>.Point'>"
        )
        self.assertIn(expected_second_part, second_part)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
