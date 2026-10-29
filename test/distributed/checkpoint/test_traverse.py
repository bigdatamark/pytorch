# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Unit tests for torch.distributed.checkpoint._traverse: flattening a nested
# state dict into path/value callbacks (traverse_state_dict) and the path based helpers
# set_element and get_element. Single process, CPU tensors only. No import-time side effects.

from collections import OrderedDict
from typing import TYPE_CHECKING

import torch
import torch.distributed.checkpoint._traverse as _traverse
from torch.testing._internal.common_utils import run_tests, TestCase


if TYPE_CHECKING:
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE


class TestTraverse(TestCase):
    # [CLASS DOC] Stateless TestCase grouping the tests of the _traverse utilities; each test
    # builds its own state dict and records callback invocations in a local mapping.
    """
    Test class for util methods of _traverse
    """

    def test_traverse_shallow(self) -> None:
        # [FUNCTION DOC] Checks that scalars, lists and tensors are reported as leaves keyed by
        # a one element path tuple (lists are not descended into at depth one), while mapping
        # entries are expanded with their keys converted to strings.
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {1: 2, 2: 3},
            "key3": torch.tensor([1]),
        }

        data = {}

        def collect_data(path, value):
            # [FUNCTION DOC] HELPER: Traversal callback that stores the value in the enclosing
            # mapping under its path tuple.
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0",), data)
        self.assertEqual(data[("key0",)], 1)

        self.assertIn(("key1",), data)
        self.assertEqual(data[("key1",)], [1, 2])

        self.assertIn(("key2", "1"), data)
        self.assertEqual(data[("key2", "1")], 2)
        self.assertIn(("key2", "2"), data)
        self.assertEqual(data[("key2", "2")], 3)

        self.assertIn(("key3",), data)
        self.assertEqual(data[("key3",)], torch.tensor([1]))

    def test_traverse_nested_list(self) -> None:
        # [FUNCTION DOC] Checks that lists nested in lists are expanded with integer indices in
        # the path, but a list made only of plain numbers at the innermost level is reported as
        # one leaf rather than element by element.
        state_dict = {
            "key1": [
                torch.tensor([1]),
                [33, torch.tensor([2]), [44, 55]],
                [66, 77],
            ],
        }

        data = {}

        def collect_data(path, value):
            # [FUNCTION DOC] HELPER: Traversal callback recording value by path.
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertNotIn(("key1"), data)

        self.assertIn(("key1", 0), data)
        self.assertEqual(data[("key1", 0)], torch.tensor([1]))

        self.assertIn(("key1", 1, 0), data)
        self.assertEqual(data[("key1", 1, 0)], 33)

        self.assertIn(("key1", 1, 1), data)
        self.assertEqual(data[("key1", 1, 1)], torch.tensor([2]))

        self.assertIn(("key1", 1, 2), data)
        self.assertEqual(data[("key1", 1, 2)], [44, 55])
        self.assertNotIn(("key1", 1, 2, 0), data)

        self.assertIn(("key1", 2), data)
        self.assertEqual(data[("key1", 2)], [66, 77])

    def test_traverse_nested_dict(self) -> None:
        # [FUNCTION DOC] Checks that a mapping inside a mapping yields only leaf paths of the
        # form (outer key, inner key); the intermediate mapping itself is not reported.
        state_dict = {
            "key0": {"key1": 99, "key2": torch.tensor([1])},
        }

        data = {}

        def collect_data(path, value):
            # [FUNCTION DOC] HELPER: Traversal callback recording value by path.
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertNotIn(("key0",), data)

        self.assertIn(("key0", "key1"), data)
        self.assertEqual(data[("key0", "key1")], 99)

        self.assertIn(("key0", "key2"), data)
        self.assertEqual(data[("key0", "key2")], torch.tensor([1]))

    def test_traverse_doesnt_ignore_intermediate_collections(self) -> None:
        # [FUNCTION DOC] Checks that a tensor buried under mapping, list, mapping, mapping
        # nesting is reported with the full mixed path of keys and a list index.
        state_dict: STATE_DICT_TYPE = {"key0": [{"key1": {"key2": torch.tensor([1])}}]}

        data = {}

        def collect_data(path, value):
            # [FUNCTION DOC] HELPER: Traversal callback recording value by path.
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0", 0, "key1", "key2"), data)
        self.assertEqual(
            data[("key0", 0, "key1", "key2")],
            torch.tensor([1]),
        )

    def test_traverse_with_ordered_dict(self) -> None:
        # [FUNCTION DOC] Checks that an OrderedDict top level state dict is traversed the same
        # way as a plain dict, with list elements reported by index.
        state_dict = OrderedDict(
            {
                "key0": [
                    99,
                    torch.tensor([3]),
                ]
            }
        )

        data = {}

        def collect_data(path, value):
            # [FUNCTION DOC] HELPER: Traversal callback recording value by path.
            nonlocal data
            data[path] = value

        _traverse.traverse_state_dict(state_dict, collect_data)

        self.assertIn(("key0", 0), data)
        self.assertEqual(data[("key0", 0)], 99)

        self.assertIn(("key0", 1), data)
        self.assertEqual(data[("key0", 1)], torch.tensor([3]))

    def test_set_element(self) -> None:
        # [FUNCTION DOC] Checks that set_element creates containers on demand along a path: an
        # integer path component makes a list padded with None up to that index (and extended
        # when a larger index is set), a string component makes a mapping, and nested integer
        # components create lists of lists.
        state_dict: STATE_DICT_TYPE = {}

        _traverse.set_element(state_dict, ("k",), 10)
        self.assertEqual(state_dict["k"], 10)

        _traverse.set_element(state_dict, ("k1", 2), 1)
        self.assertEqual(state_dict["k1"], [None, None, 1])

        _traverse.set_element(state_dict, ("k1", 1), 99)
        self.assertEqual(state_dict["k1"], [None, 99, 1])

        _traverse.set_element(state_dict, ("k1", 3), 88)
        self.assertEqual(state_dict["k1"], [None, 99, 1, 88])

        _traverse.set_element(state_dict, ("k2", "k3"), 3)
        self.assertEqual(state_dict["k2"], {"k3": 3})

        _traverse.set_element(state_dict, ("k2", "k4", 0, 0), 99)
        self.assertEqual(state_dict["k2"]["k4"][0], [99])

    def test_get_element(self) -> None:
        # [FUNCTION DOC] Checks that get_element follows a path through mappings and lists and
        # returns None (rather than raising) for missing keys, out of range indices, indexing
        # into a non-container, or a wrong kind of key.
        state_dict = {"a": [0, 1], "b": [2, {"c": "d"}]}
        self.assertEqual(_traverse.get_element(state_dict, ("a",)), [0, 1])
        self.assertEqual(_traverse.get_element(state_dict, ("b", 0)), 2)
        self.assertEqual(_traverse.get_element(state_dict, ("b", 1, "c")), "d")

        self.assertIsNone(_traverse.get_element(state_dict, ("c",)))
        self.assertIsNone(_traverse.get_element(state_dict, ("a", 33)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 88)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 0, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, 2)))
        self.assertIsNone(_traverse.get_element(state_dict, ("b", 1, "d")))


if __name__ == "__main__":
    run_tests()
