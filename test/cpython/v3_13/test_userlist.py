# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 UserList test. Runs the shared list
# conformance suite (list_tests.CommonTest) against collections.UserList, plus UserList
# specific checks (slicing, concatenation, copy). At import time it installs a meta path
# finder that redirects "test.*" imports (e.g. test.mapping_tests) to top-level modules,
# and defines a test runner entry point via run_tests when executed as a script.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_userlist.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase


# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    # [CLASS DOC] HELPER: Import hook that maps selected "test.xxx" module names (those listed
    # in redirect_imports) onto the standalone top-level module "xxx". Stateless; installed
    # at the front of sys.meta_path by this module.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a name in redirect_imports, imports the same name without
        # the "test." prefix, registers that module in sys.modules under the original name
        # (side effect) and returns its spec. Returns None (defers to other finders) if the
        # name is not listed or the standalone import fails with ImportError.
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

# Check every path through every method of UserList

from collections import UserList
import list_tests
import unittest
from test import support


class UserListTest(list_tests.CommonTest):
    # [CLASS DOC] Runs the generic list test suite with type2test set to UserList, and adds
    # tests for behaviors specific to UserList (slice result type, mixed operand arithmetic,
    # copy). Inherits all other tests from list_tests.CommonTest.
    type2test = UserList

    def test_getslice(self):
        # [FUNCTION DOC] Runs the inherited slice test, then checks that slicing a UserList
        # with every start/stop combination in a range of negative and out-of-range indices
        # matches slicing a plain list.
        super().test_getslice()
        l = [0, 1, 2, 3, 4]
        u = self.type2test(l)
        for i in range(-3, 6):
            self.assertEqual(u[:i], l[:i])
            self.assertEqual(u[i:], l[i:])
            for j in range(-3, 6):
                self.assertEqual(u[i:j], l[i:j])

    def test_slice_type(self):
        # [FUNCTION DOC] Checks that a full slice of a UserList is itself a UserList (same
        # class as the source) and compares equal to the original.
        l = [0, 1, 2, 3, 4]
        u = UserList(l)
        self.assertIsInstance(u[:], u.__class__)
        self.assertEqual(u[:],u)

    def test_add_specials(self):
        # [FUNCTION DOC] Adding a string on the right of a UserList built from a string
        # concatenates element-wise, giving the list of all characters of both strings.
        u = UserList("spam")
        u2 = u + "eggs"
        self.assertEqual(u2, list("spameggs"))

    def test_radd_specials(self):
        # [FUNCTION DOC] Checks reflected addition: a string plus a UserList, and a direct
        # call of the reflected-add method with another UserList, both give the combined
        # character list with the left operand first.
        u = UserList("eggs")
        u2 = "spam" + u
        self.assertEqual(u2, list("spameggs"))
        u2 = u.__radd__(UserList("spam"))
        self.assertEqual(u2, list("spameggs"))

    def test_iadd(self):
        # [FUNCTION DOC] Runs the inherited in-place add test, then checks that extending a
        # plain list in place with a UserList appends its elements.
        super().test_iadd()
        u = [0, 1]
        u += UserList([0, 1])
        self.assertEqual(u, [0, 1, 0, 1])

    def test_mixedcmp(self):
        # [FUNCTION DOC] Checks that a UserList compares equal to a plain list with the same
        # items and unequal to lists that are shorter or differ in an element.
        u = self.type2test([0, 1])
        self.assertEqual(u, [0, 1])
        self.assertNotEqual(u, [0])
        self.assertNotEqual(u, [0, 2])

    def test_mixedadd(self):
        # [FUNCTION DOC] Checks that adding a plain list to a UserList works: an empty list
        # leaves it equal, and a one-item list appends that item.
        u = self.type2test([0, 1])
        self.assertEqual(u + [], u)
        self.assertEqual(u + [2], [0, 1, 2])

    def test_getitemoverwriteiter(self):
        # [FUNCTION DOC] Defines a subclass whose item access returns the key as a string
        # with a "!!!" suffix, and checks that iteration uses the override (first item is
        # "0!!!"), i.e. the iterator goes through item access rather than internal storage.
        # Verify that __getitem__ overrides *are* recognized by __iter__
        class T(self.type2test):
            # [CLASS DOC] HELPER: UserList subclass whose item access is overridden.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the key converted to text plus "!!!".
                return str(key) + '!!!'
        self.assertEqual(next(iter(T((1,2)))), "0!!!")

    def test_userlist_copy(self):
        # [FUNCTION DOC] Checks that copying a UserList yields an equal object of the same
        # type.
        u = self.type2test([6, 8, 1, 9, 1])
        v = u.copy()
        self.assertEqual(u, v)
        self.assertEqual(type(u), type(v))

    # Decorate existing test with recursion limit, because
    # the test is for C structure, but `UserList` is a Python structure.
    # test_repr_deep = support.infinite_recursion(25)(
    #     list_tests.CommonTest.test_repr_deep,
    # )

if __name__ == "__main__":
    run_tests()
