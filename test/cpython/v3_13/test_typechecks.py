# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
#
# [MODULE DOC] Dynamo-adapted copy of CPython 3.13 test_typechecks: checks that custom
# __instancecheck__ / __subclasscheck__ on a metaclass drive isinstance and issubclass,
# including tuple arguments and subclasses. Defines a metaclass ABC (checks against a
# "__subclass__" set plus the class itself via the MRO), Integer and SubInt classes, and one
# CPythonTestCase. No import-time side effects beyond imports; run_tests() under __main__.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_typechecks.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Unit tests for __instancecheck__ and __subclasscheck__."""

import unittest


class ABC(type):
    # [CLASS DOC] HELPER: Metaclass overriding instance/subclass checks. A class is a
    # "subclass" of cls if any class in its MRO is cls or is in cls's "__subclass__" set.

    def __instancecheck__(cls, inst):
        # [FUNCTION DOC] HELPER: True if the type of inst or its __class__ passes the
        # subclass check against cls.
        """Implement isinstance(inst, cls)."""
        return any(cls.__subclasscheck__(c)
                   for c in {type(inst), inst.__class__})

    def __subclasscheck__(cls, sub):
        # [FUNCTION DOC] HELPER: Looks up the optional "__subclass__" set in cls's own
        # namespace (default empty), adds cls, and returns True if any class in sub's MRO is
        # in that set.
        """Implement issubclass(sub, cls)."""
        candidates = cls.__dict__.get("__subclass__", set()) | {cls}
        return any(c in candidates for c in sub.mro())


class Integer(metaclass=ABC):
    # [CLASS DOC] HELPER: Test class using ABC; registers the builtin int as a virtual subclass.
    __subclass__ = {int}


class SubInt(Integer):
    # [CLASS DOC] HELPER: Real subclass of Integer, used to check inheritance interplay.
    pass


class TypeChecksTest(CPythonTestCase):
    # [CLASS DOC] Test case verifying isinstance/issubclass delegate to the ABC metaclass hooks.

    def testIsSubclassInternal(self):
        # [FUNCTION DOC] Calls the subclass-check hook directly: int is accepted, float is not.
        self.assertEqual(Integer.__subclasscheck__(int), True)
        self.assertEqual(Integer.__subclasscheck__(float), False)

    def testIsSubclassBuiltin(self):
        # [FUNCTION DOC] issubclass with Integer (alone and in a tuple): int True, float False.
        self.assertEqual(issubclass(int, Integer), True)
        self.assertEqual(issubclass(int, (Integer,)), True)
        self.assertEqual(issubclass(float, Integer), False)
        self.assertEqual(issubclass(float, (Integer,)), False)

    def testIsInstanceBuiltin(self):
        # [FUNCTION DOC] isinstance with Integer (alone and in a tuple): 42 True, 3.14 False.
        self.assertEqual(isinstance(42, Integer), True)
        self.assertEqual(isinstance(42, (Integer,)), True)
        self.assertEqual(isinstance(3.14, Integer), False)
        self.assertEqual(isinstance(3.14, (Integer,)), False)

    def testIsInstanceActual(self):
        # [FUNCTION DOC] An actual Integer instance is an instance of Integer, bare or in a tuple.
        self.assertEqual(isinstance(Integer(), Integer), True)
        self.assertEqual(isinstance(Integer(), (Integer,)), True)

    def testIsSubclassActual(self):
        # [FUNCTION DOC] Integer is a subclass of itself, bare or in a tuple.
        self.assertEqual(issubclass(Integer, Integer), True)
        self.assertEqual(issubclass(Integer, (Integer,)), True)

    def testSubclassBehavior(self):
        # [FUNCTION DOC] SubInt is a subclass/instance of Integer and SubInt; Integer, int and 42
        # are not subclasses/instances of SubInt (bare and tuple forms).
        self.assertEqual(issubclass(SubInt, Integer), True)
        self.assertEqual(issubclass(SubInt, (Integer,)), True)
        self.assertEqual(issubclass(SubInt, SubInt), True)
        self.assertEqual(issubclass(SubInt, (SubInt,)), True)
        self.assertEqual(issubclass(Integer, SubInt), False)
        self.assertEqual(issubclass(Integer, (SubInt,)), False)
        self.assertEqual(issubclass(int, SubInt), False)
        self.assertEqual(issubclass(int, (SubInt,)), False)
        self.assertEqual(isinstance(SubInt(), Integer), True)
        self.assertEqual(isinstance(SubInt(), (Integer,)), True)
        self.assertEqual(isinstance(SubInt(), SubInt), True)
        self.assertEqual(isinstance(SubInt(), (SubInt,)), True)
        self.assertEqual(isinstance(42, SubInt), False)
        self.assertEqual(isinstance(42, (SubInt,)), False)


if __name__ == "__main__":
    run_tests()
