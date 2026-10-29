# Owner(s): ["oncall: distributed"]
# [MODULE DOC]
# Unit tests for the distributed checkpoint nested-dict helpers flatten_state_dict and
# unflatten_state_dict. Checks the flatten/unflatten round trip and the path mapping.
# Single-process, CPU only; no import-time side effects beyond imports.

import torch
from torch.distributed.checkpoint._nested_dict import (
    flatten_state_dict,
    unflatten_state_dict,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class TestFlattening(TestCase):
    # [CLASS DOC] Stateless test case for state dict flattening; each test builds its own
    # nested state dict of ints, strings, lists, dicts and tensors.
    def test_flattening_round_trip(self) -> None:
        # [FUNCTION DOC] Flattens a nested state dict (lists, dicts, tensors, scalars),
        # unflattens it with the returned mapping, and asserts the result equals the original.
        state_dict = {
            "key0": 1,
            "key1": [1, 2],
            "key2": {"1": 2, "2": 3},
            "key3": torch.tensor([1]),
            "key4": [[torch.tensor(2), "x"], [1, 2, 3], {"key6": [44]}],
        }

        flatten_dict, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
            {
                'key0': 1,
                'key1': [1, 2],
                'key2': {'1': 2, '2': 3},
                'key3': tensor([1]),
                'key4.0.0': tensor(2),
                'key4.0.1': 'x',
                'key4.1': [1, 2, 3],
                'key4.2': {'key6': [44]}
            }
        """
        restored = unflatten_state_dict(flatten_dict, mapping)

        self.assertEqual(state_dict, restored)

    def test_mapping(self) -> None:
        # [FUNCTION DOC] Flattens a state dict and checks that the mapping from dotted flat
        # keys to original key paths (tuples of keys and list indices) is correct for
        # nested lists and dicts, including a list holding only non-tensor leaves.
        # KNOWN ISSUE: the inline docstring-style comment shows k3 kept whole as a single
        # flat key, but the assertions expect it split into k3.0, k3.1 and k3.2.0.k3.
        state_dict = {
            "k0": [1],
            "k2": [torch.tensor([1]), 99, [{"k3": torch.tensor(1)}]],
            "k3": ["x", 99, [{"k3": "y"}]],
        }

        _, mapping = flatten_state_dict(state_dict)
        """
        flatten_dict:
        {'k0': [1], 'k2.0': tensor([1]), 'k2.1': 99, 'k2.2.0.k3': tensor(1), 'k3': ['x', 99, [{'k3': 'y'}]]}
        mapping:
        {'k0': ('k0',), 'k2.0': ('k2', 0), 'k2.1': ('k2', 1), 'k2.2.0.k3': ('k2', 2, 0, 'k3'), 'k3': ('k3',)}
        """

        self.assertEqual(("k0",), mapping["k0"])
        self.assertEqual(("k2", 0), mapping["k2.0"])
        self.assertEqual(("k2", 1), mapping["k2.1"])
        self.assertEqual(("k2", 2, 0, "k3"), mapping["k2.2.0.k3"])
        self.assertEqual(("k3", 0), mapping["k3.0"])
        self.assertEqual(("k3", 1), mapping["k3.1"])
        self.assertEqual(("k3", 2, 0, "k3"), mapping["k3.2.0.k3"])


if __name__ == "__main__":
    run_tests()
