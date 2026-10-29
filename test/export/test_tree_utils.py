# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for the export tree utilities: reordering user keyword arguments to match a
# stored tree spec, and the customizable tree spec equivalence check. Defines TestTreeUtils.
from collections import OrderedDict

import torch
from torch._dynamo.test_case import TestCase
from torch.export._tree_utils import is_equivalent, reorder_kwargs
from torch.testing._internal.common_utils import run_tests
from torch.utils._pytree import tree_structure


class TestTreeUtils(TestCase):
    # [CLASS DOC] Test case for reorder_kwargs and is_equivalent.
    def test_reorder_kwargs(self):
        # [FUNCTION DOC] Builds a tree spec from original keyword arguments (a then b), reorders
        # user keywords given in the opposite order, and checks the popped key order of the result
        # matches that of the original mapping (the order is fixed by the spec).
        original_kwargs = {"a": torch.tensor(0), "b": torch.tensor(1)}
        user_kwargs = {"b": torch.tensor(2), "a": torch.tensor(3)}
        orig_spec = tree_structure(((), original_kwargs))

        reordered_kwargs = reorder_kwargs(user_kwargs, orig_spec)

        # Key ordering should be the same
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0])
        self.assertEqual(reordered_kwargs.popitem()[0], original_kwargs.popitem()[0])

    def test_equivalence_check(self):
        # [FUNCTION DOC] Checks that a plain mapping and an ordered mapping with the same keys
        # (including a None entry) are equivalent under a custom comparison that treats the two
        # mapping types alike, and that an ordered mapping with different key order is not.
        tree1 = {"a": torch.tensor(0), "b": torch.tensor(1), "c": None}
        tree2 = OrderedDict(a=torch.tensor(0), b=torch.tensor(1), c=None)
        spec1 = tree_structure(tree1)
        spec2 = tree_structure(tree2)

        def dict_ordered_dict_eq(type1, context1, type2, context2):
            # [FUNCTION DOC] HELPER: Equivalence callback for node type and context pairs. If
            # either type is missing, requires both missing and equal contexts; if both types are
            # dict or ordered dict kinds, compares contexts only; otherwise requires identical
            # types and equal contexts. Returns a bool.
            if type1 is None or type2 is None:
                return type1 is type2 and context1 == context2

            if issubclass(type1, (dict, OrderedDict)) and issubclass(
                type2, (dict, OrderedDict)
            ):
                return context1 == context2

            return type1 is type2 and context1 == context2

        self.assertTrue(is_equivalent(spec1, spec2, dict_ordered_dict_eq))

        # Wrong ordering should still fail
        tree3 = OrderedDict(b=torch.tensor(1), a=torch.tensor(0))
        spec3 = tree_structure(tree3)
        self.assertFalse(is_equivalent(spec1, spec3, dict_ordered_dict_eq))


if __name__ == "__main__":
    run_tests()
