# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_isinstance suite. Exercises corner
# cases of isinstance() and issubclass(): exceptions raised while reading __bases__ or __class__
# (masked only when AttributeError), abstract classes and instances built from properties, tuple
# and union arguments, and recursion limits. Defines module-level helper classes and the
# blowstack function.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.13/Lib/test/test_isinstance.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase
# ======= END DYNAMO PATCH =======

# Tests some corner cases with isinstance() and issubclass().  While these
# tests use new style classes and properties, they actually do whitebox
# testing of error conditions uncovered when using extension types.

import unittest
import typing
from test import support



class TestIsInstanceExceptions(__TestCase):
    # [CLASS DOC] Tests how isinstance() treats errors raised while reading __bases__ or __class__
    # of its arguments: AttributeError is masked (False or TypeError), other exceptions propagate.
    # Test to make sure that an AttributeError when accessing the instance's
    # class's bases is masked.  This was actually a bug in Python 2.2 and
    # 2.2.1 where the exception wasn't caught but it also wasn't being cleared
    # (leading to an "undetected error" in the debug build).  Set up is,
    # isinstance(inst, cls) where:
    #
    # - cls isn't a type, or a tuple
    # - cls has a __bases__ attribute
    # - inst has a __class__ attribute
    # - inst.__class__ as no __bases__ attribute
    #
    # Sounds complicated, I know, but this mimics a situation where an
    # extension type raises an AttributeError when its __bases__ attribute is
    # gotten.  In that case, isinstance() should return False.
    def test_class_has_no_bases(self):
        # [FUNCTION DOC] An instance whose class has no __bases__ makes isinstance against an
        # abstract class return False.
        class I(object):
            # [CLASS DOC] HELPER: Class whose __class__ property returns None (no __bases__).
            def getclass(self):
                # [FUNCTION DOC] HELPER: Return None so the reported class has no __bases__.
                # This must return an object that has no __bases__ attribute
                return None
            __class__ = property(getclass)

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property returns an empty tuple.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Return an empty tuple.
                return ()
            __bases__ = property(getbases)

        self.assertEqual(False, isinstance(I(), C()))

    # Like above except that inst.__class__.__bases__ raises an exception
    # other than AttributeError
    def test_bases_raises_other_than_attribute_error(self):
        # [FUNCTION DOC] If reading the instance class __bases__ raises RuntimeError, isinstance
        # propagates it.
        class E(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises RuntimeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise RuntimeError.
                raise RuntimeError
            __bases__ = property(getbases)

        class I(object):
            # [CLASS DOC] HELPER: Class whose __class__ property returns an E instance.
            def getclass(self):
                # [FUNCTION DOC] HELPER: Return a new E instance.
                return E()
            __class__ = property(getclass)

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property returns an empty tuple.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Return an empty tuple.
                return ()
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, isinstance, I(), C())

    # Here's a situation where getattr(cls, '__bases__') raises an exception.
    # If that exception is not AttributeError, it should not get masked
    def test_dont_mask_non_attribute_error(self):
        # [FUNCTION DOC] A RuntimeError raised by the second argument __bases__ property is not
        # masked by isinstance.
        class I: pass

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises RuntimeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise RuntimeError.
                raise RuntimeError
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, isinstance, I(), C())

    # Like above, except that getattr(cls, '__bases__') raises an
    # AttributeError, which /should/ get masked as a TypeError
    def test_mask_attribute_error(self):
        # [FUNCTION DOC] An AttributeError raised by the second argument __bases__ property is
        # converted to TypeError.
        class I: pass

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises AttributeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise AttributeError.
                raise AttributeError
            __bases__ = property(getbases)

        self.assertRaises(TypeError, isinstance, I(), C())

    # check that we don't mask non AttributeErrors
    # see: http://bugs.python.org/issue1574217
    def test_isinstance_dont_mask_non_attribute_error(self):
        # [FUNCTION DOC] A RuntimeError raised by a __class__ property propagates out of
        # isinstance for both a builtin type and an old-style class as second argument
        # (issue 1574217).
        class C(object):
            # [CLASS DOC] HELPER: Class whose __class__ property raises RuntimeError.
            def getclass(self):
                # [FUNCTION DOC] HELPER: Raise RuntimeError.
                raise RuntimeError
            __class__ = property(getclass)

        c = C()
        self.assertRaises(RuntimeError, isinstance, c, bool)

        # test another code path
        class D: pass
        self.assertRaises(RuntimeError, isinstance, c, D)


# These tests are similar to above, but tickle certain code paths in
# issubclass() instead of isinstance() -- really PyObject_IsSubclass()
# vs. PyObject_IsInstance().
class TestIsSubclassExceptions(__TestCase):
    # [CLASS DOC] Tests that issubclass() masks AttributeError from __bases__ (as TypeError) but
    # lets other exceptions through, for both the first and the class arguments.
    def test_dont_mask_non_attribute_error(self):
        # [FUNCTION DOC] A RuntimeError from __bases__ of the first argument propagates out
        # of issubclass.
        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises RuntimeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise RuntimeError.
                raise RuntimeError
            __bases__ = property(getbases)

        class S(C): pass

        self.assertRaises(RuntimeError, issubclass, C(), S())

    def test_mask_attribute_error(self):
        # [FUNCTION DOC] An AttributeError from __bases__ of the first argument makes issubclass
        # raise TypeError.
        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises AttributeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise AttributeError.
                raise AttributeError
            __bases__ = property(getbases)

        class S(C): pass

        self.assertRaises(TypeError, issubclass, C(), S())

    # Like above, but test the second branch, where the __bases__ of the
    # second arg (the cls arg) is tested.  This means the first arg must
    # return a valid __bases__, and it's okay for it to be a normal --
    # unrelated by inheritance -- class.
    def test_dont_mask_non_attribute_error_in_cls_arg(self):
        # [FUNCTION DOC] A RuntimeError from __bases__ of the class argument propagates out
        # of issubclass.
        class B: pass

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises RuntimeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise RuntimeError.
                raise RuntimeError
            __bases__ = property(getbases)

        self.assertRaises(RuntimeError, issubclass, B, C())

    def test_mask_attribute_error_in_cls_arg(self):
        # [FUNCTION DOC] An AttributeError from __bases__ of the class argument makes issubclass
        # raise TypeError.
        class B: pass

        class C(object):
            # [CLASS DOC] HELPER: Class whose __bases__ property raises AttributeError.
            def getbases(self):
                # [FUNCTION DOC] HELPER: Raise AttributeError.
                raise AttributeError
            __bases__ = property(getbases)

        self.assertRaises(TypeError, issubclass, B, C())



# meta classes for creating abstract classes and instances
class AbstractClass(object):
    # [CLASS DOC] HELPER: Fake class object: stores a bases tuple, exposes it via __bases__ and
    # creates an AbstractInstance when called.
    def __init__(self, bases):
        # [FUNCTION DOC] HELPER: Store the bases tuple in the bases attribute.
        self.bases = bases

    def getbases(self):
        # [FUNCTION DOC] HELPER: Return the stored bases tuple (used by the __bases__ property).
        return self.bases
    __bases__ = property(getbases)

    def __call__(self):
        # [FUNCTION DOC] HELPER: Return a new AbstractInstance whose class is this object.
        return AbstractInstance(self)

class AbstractInstance(object):
    # [CLASS DOC] HELPER: Fake instance whose __class__ property reports the AbstractClass given
    # at construction.
    def __init__(self, klass):
        # [FUNCTION DOC] HELPER: Store the given class object in klass.
        self.klass = klass

    def getclass(self):
        # [FUNCTION DOC] HELPER: Return the stored klass (used by the __class__ property).
        return self.klass
    __class__ = property(getclass)

# abstract classes
AbstractSuper = AbstractClass(bases=())

AbstractChild = AbstractClass(bases=(AbstractSuper,))

# normal classes
class Super:
    # [CLASS DOC] HELPER: Ordinary base class.
    pass

class Child(Super):
    # [CLASS DOC] HELPER: Ordinary subclass of Super.
    pass

class TestIsInstanceIsSubclass(__TestCase):
    # [CLASS DOC] Checks isinstance and issubclass return proper booleans (not TypeError) for all
    # mixes of normal and abstract classes/instances, tuples, unions and recursion limits.
    # Tests to ensure that isinstance and issubclass work on abstract
    # classes and instances.  Before the 2.2 release, TypeErrors were
    # raised when boolean values should have been returned.  The bug was
    # triggered by mixing 'normal' classes and instances were with
    # 'abstract' classes and instances.  This case tries to test all
    # combinations.

    def test_isinstance_normal(self):
        # [FUNCTION DOC] isinstance on normal instances against normal and abstract classes gives
        # the expected booleans.
        # normal instances
        self.assertEqual(True, isinstance(Super(), Super))
        self.assertEqual(False, isinstance(Super(), Child))
        self.assertEqual(False, isinstance(Super(), AbstractSuper))
        self.assertEqual(False, isinstance(Super(), AbstractChild))

        self.assertEqual(True, isinstance(Child(), Super))
        self.assertEqual(False, isinstance(Child(), AbstractSuper))

    def test_isinstance_abstract(self):
        # [FUNCTION DOC] isinstance on abstract instances against abstract and normal classes
        # follows the abstract inheritance chain.
        # abstract instances
        self.assertEqual(True, isinstance(AbstractSuper(), AbstractSuper))
        self.assertEqual(False, isinstance(AbstractSuper(), AbstractChild))
        self.assertEqual(False, isinstance(AbstractSuper(), Super))
        self.assertEqual(False, isinstance(AbstractSuper(), Child))

        self.assertEqual(True, isinstance(AbstractChild(), AbstractChild))
        self.assertEqual(True, isinstance(AbstractChild(), AbstractSuper))
        self.assertEqual(False, isinstance(AbstractChild(), Super))
        self.assertEqual(False, isinstance(AbstractChild(), Child))

    def test_isinstance_with_or_union(self):
        # [FUNCTION DOC] isinstance accepts union types of classes and typing aliases;
        # parameterized generics such as a list of int in the check raise TypeError.
        self.assertTrue(isinstance(Super(), Super | int))
        self.assertFalse(isinstance(None, str | int))
        self.assertTrue(isinstance(3, str | int))
        self.assertTrue(isinstance("", str | int))
        self.assertTrue(isinstance([], typing.List | typing.Tuple))
        self.assertTrue(isinstance(2, typing.List | int))
        self.assertFalse(isinstance(2, typing.List | typing.Tuple))
        self.assertTrue(isinstance(None, int | None))
        self.assertFalse(isinstance(3.14, int | str))
        with self.assertRaises(TypeError):
            isinstance(2, list[int])
        with self.assertRaises(TypeError):
            isinstance(2, list[int] | int)
        with self.assertRaises(TypeError):
            isinstance(2, float | str | list[int] | int)



    def test_subclass_normal(self):
        # [FUNCTION DOC] issubclass on normal classes, including unions of typing aliases, gives
        # expected booleans.
        # normal classes
        self.assertEqual(True, issubclass(Super, Super))
        self.assertEqual(False, issubclass(Super, AbstractSuper))
        self.assertEqual(False, issubclass(Super, Child))

        self.assertEqual(True, issubclass(Child, Child))
        self.assertEqual(True, issubclass(Child, Super))
        self.assertEqual(False, issubclass(Child, AbstractSuper))
        self.assertTrue(issubclass(typing.List, typing.List|typing.Tuple))
        self.assertFalse(issubclass(int, typing.List|typing.Tuple))

    def test_subclass_abstract(self):
        # [FUNCTION DOC] issubclass on abstract classes follows the abstract inheritance chain.
        # abstract classes
        self.assertEqual(True, issubclass(AbstractSuper, AbstractSuper))
        self.assertEqual(False, issubclass(AbstractSuper, AbstractChild))
        self.assertEqual(False, issubclass(AbstractSuper, Child))

        self.assertEqual(True, issubclass(AbstractChild, AbstractChild))
        self.assertEqual(True, issubclass(AbstractChild, AbstractSuper))
        self.assertEqual(False, issubclass(AbstractChild, Super))
        self.assertEqual(False, issubclass(AbstractChild, Child))

    def test_subclass_tuple(self):
        # [FUNCTION DOC] issubclass accepts a tuple, possibly nested or empty, as the
        # second argument.
        # test with a tuple as the second argument classes
        self.assertEqual(True, issubclass(Child, (Child,)))
        self.assertEqual(True, issubclass(Child, (Super,)))
        self.assertEqual(False, issubclass(Super, (Child,)))
        self.assertEqual(True, issubclass(Super, (Child, Super)))
        self.assertEqual(False, issubclass(Child, ()))
        self.assertEqual(True, issubclass(Super, (Child, (Super,))))

        self.assertEqual(True, issubclass(int, (int, (float, int))))
        self.assertEqual(True, issubclass(str, (str, (Child, str))))

    def test_subclass_recursion_limit(self):
        # [FUNCTION DOC] A deeply nested tuple argument makes issubclass raise RecursionError
        # rather than crashing.
        # make sure that issubclass raises RecursionError before the C stack is
        # blown
        with support.infinite_recursion():
            self.assertRaises(RecursionError, blowstack, issubclass, str, str)

    def test_isinstance_recursion_limit(self):
        # [FUNCTION DOC] A deeply nested tuple argument makes isinstance raise RecursionError
        # rather than crashing.
        # KNOWN ISSUE: the inline comment inside says issubclass, but the test calls isinstance.
        # make sure that issubclass raises RecursionError before the C stack is
        # blown
        with support.infinite_recursion():
            self.assertRaises(RecursionError, blowstack, isinstance, '', str)

    def test_subclass_with_union(self):
        # [FUNCTION DOC] issubclass accepts union types; a non-class first argument or a
        # parameterized generic raises TypeError.
        self.assertTrue(issubclass(int, int | float | int))
        self.assertTrue(issubclass(str, str | Child | str))
        self.assertFalse(issubclass(dict, float|str))
        self.assertFalse(issubclass(object, float|str))
        with self.assertRaises(TypeError):
            issubclass(2, Child | Super)
        with self.assertRaises(TypeError):
            issubclass(int, list[int] | Child)

    def test_issubclass_refcount_handling(self):
        # [FUNCTION DOC] Regression test (bpo 39382) where a __bases__ property returns a fresh
        # tuple; issubclass must keep references alive and give True.
        # bpo-39382: abstract_issubclass() didn't hold item reference while
        # peeking in the bases tuple, in the single inheritance case.
        class A:
            # [CLASS DOC] HELPER: Class whose __bases__ property returns a tuple holding the
            # int type.
            @property
            def __bases__(self):
                # [FUNCTION DOC] HELPER: Return a tuple containing int.
                return (int, )

        class B:
            # [CLASS DOC] HELPER: Class whose __bases__ property returns a tuple with a new
            # A instance.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set attribute x to 1 (alters memory layout to expose the
                # original bug).
                # setting this here increases the chances of exhibiting the bug,
                # probably due to memory layout changes.
                self.x = 1

            @property
            def __bases__(self):
                # [FUNCTION DOC] HELPER: Return a tuple containing a new A instance.
                return (A(), )

        self.assertEqual(True, issubclass(B(), int))

    def test_infinite_recursion_in_bases(self):
        # [FUNCTION DOC] A __bases__ property that returns itself recursively makes issubclass and
        # isinstance raise RecursionError, within a lowered recursion limit.
        class X:
            # [CLASS DOC] HELPER: Class whose __bases__ property recurses forever.
            @property
            def __bases__(self):
                # [FUNCTION DOC] HELPER: Return the same property value, recursing without end.
                return self.__bases__
        with support.infinite_recursion(25):
            self.assertRaises(RecursionError, issubclass, X(), int)
            self.assertRaises(RecursionError, issubclass, int, X())
            self.assertRaises(RecursionError, isinstance, 1, X())

    def test_infinite_recursion_via_bases_tuple(self):
        # [FUNCTION DOC] Regression test for bpo 30570: attribute lookup returning a tuple holding
        # the object itself makes issubclass raise RecursionError.
        """Regression test for bpo-30570."""
        class Failure(object):
            # [CLASS DOC] HELPER: Class whose attribute lookup always returns a tuple of itself
            # and None.
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: Return a tuple (self, None) for any attribute name.
                return (self, None)
        with support.infinite_recursion():
            with self.assertRaises(RecursionError):
                issubclass(Failure(), int)

    def test_infinite_cycle_in_bases(self):
        # [FUNCTION DOC] Regression test for bpo 30570: a __bases__ cycle makes issubclass
        # raise RecursionError.
        """Regression test for bpo-30570."""
        class X:
            # [CLASS DOC] HELPER: Class whose __bases__ property returns a tuple of itself
            # three times.
            @property
            def __bases__(self):
                # [FUNCTION DOC] HELPER: Return a tuple containing self three times.
                return (self, self, self)
        with support.infinite_recursion():
            self.assertRaises(RecursionError, issubclass, X(), int)

    def test_infinitely_many_bases(self):
        # [FUNCTION DOC] Regression test for bpo 30570: an unbounded tree of bases makes
        # issubclass raise RecursionError.
        """Regression test for bpo-30570."""
        class X:
            # [CLASS DOC] HELPER: Class whose attribute lookup creates two new classes that share
            # its lookup hook.
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: Create classes A and B sharing this hook and return a
                # tuple of one instance of each.
                # KNOWN ISSUE: the assertion calls assertEqual on self, which is not a TestCase;
                #   the lookup of it re-enters this hook until RecursionError, so the test passes
                #   by accident.
                self.assertEqual(attr, "__bases__")
                class A:
                    # [CLASS DOC] HELPER: Empty helper class.
                    pass
                class B:
                    # [CLASS DOC] HELPER: Empty helper class.
                    pass
                A.__getattr__ = B.__getattr__ = X.__getattr__
                return (A(), B())
        with support.infinite_recursion(25):
            self.assertRaises(RecursionError, issubclass, X(), int)


def blowstack(fxn, arg, compare_to):
    # [FUNCTION DOC] HELPER: Call fxn with arg and a nested tuple whose depth grows each loop
    # iteration, up to the recursion limit estimate, so a RecursionError is expected from fxn.
    # Make sure that calling isinstance with a deeply nested tuple for its
    # argument will raise RecursionError eventually.
    tuple_arg = (compare_to,)
    for cnt in range(support.exceeds_recursion_limit()):
        tuple_arg = (tuple_arg,)
        fxn(arg, tuple_arg)


if __name__ == '__main__':
    run_tests()
