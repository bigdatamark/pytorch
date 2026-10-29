# [MODULE DOC] Unit tests for torchgen.utils.NamespaceHelper: parsing an entity name with a
# namespace prefix, default namespace handling, and the maximum namespace depth check.
import unittest

from torchgen.utils import NamespaceHelper


class TestNamespaceHelper(unittest.TestCase):
    # [CLASS DOC] Test case for NamespaceHelper; stateless.
    def test_create_from_namespaced_tuple(self) -> None:
        # [FUNCTION DOC] Checks a namespaced entity is split into entity name and namespace.
        helper = NamespaceHelper.from_namespaced_entity("aten::add")
        self.assertEqual(helper.entity_name, "add")
        self.assertEqual(helper.get_cpp_namespace(), "aten")

    def test_default_namespace(self) -> None:
        # [FUNCTION DOC] Checks an unqualified entity has an empty namespace unless a default
        # is passed to the namespace getter.
        helper = NamespaceHelper.from_namespaced_entity("add")
        self.assertEqual(helper.entity_name, "add")
        self.assertEqual(helper.get_cpp_namespace(), "")
        self.assertEqual(helper.get_cpp_namespace("default"), "default")

    def test_namespace_levels_more_than_max(self) -> None:
        # [FUNCTION DOC] Checks that constructing a helper with more namespace levels than
        # the allowed maximum raises an assertion error.
        with self.assertRaises(AssertionError):
            NamespaceHelper(
                namespace_str="custom_1::custom_2", entity_name="", max_level=1
            )
