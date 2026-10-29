# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_subclassinit ported to run under Dynamo's CPythonTestCase. Checks
# PEP 487 behavior: __init_subclass__ hooks (kwargs, errors, MRO/diamond order), __set_name__
# descriptor notification (metaclass, errors with notes), and class-creation errors from
# metaclasses and the type constructor. Running the file calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_subclassinit.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import types
import unittest


class Test(CPythonTestCase):
    # [CLASS DOC] Single test case holding all PEP 487 subclass-initialization checks. Stateless;
    # each test builds its own throwaway classes and asserts on the resulting class attributes.
    def test_init_subclass(self):
        # [FUNCTION DOC] Verifies __init_subclass__ runs for a subclass B of A (sets a flag on
        # B only).
        class A:
            # [CLASS DOC] HELPER: base class whose __init_subclass__ marks each subclass
            # as initialized.
            initialized = False

            def __init_subclass__(cls):
                # [FUNCTION DOC] HELPER: calls the parent hook then sets the initialized flag on
                # the new subclass.
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            # [CLASS DOC] HELPER: empty subclass used to trigger the parent
            # __init_subclass__ hook.
            pass

        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_dict(self):
        # [FUNCTION DOC] Same as the plain case but the base derives from dict, so built-in bases
        # also run the hook.
        class A(dict):
            # [CLASS DOC] HELPER: dict subclass with an __init_subclass__ hook marking subclasses.
            initialized = False

            def __init_subclass__(cls):
                # [FUNCTION DOC] HELPER: sets the initialized flag on the newly created subclass.
                super().__init_subclass__()
                cls.initialized = True

        class B(A):
            # [CLASS DOC] HELPER: empty dict-derived subclass that triggers the hook.
            pass

        self.assertFalse(A.initialized)
        self.assertTrue(B.initialized)

    def test_init_subclass_kwargs(self):
        # [FUNCTION DOC] Checks keyword arguments in the class statement are forwarded to
        # __init_subclass__ (B gets kwargs mapping with x equal to 3).
        class A:
            # [CLASS DOC] HELPER: base whose hook stores received keyword arguments on
            # the subclass.
            def __init_subclass__(cls, **kwargs):
                # [FUNCTION DOC] HELPER: records the keyword arguments passed by the
                # class statement.
                cls.kwargs = kwargs

        class B(A, x=3):
            # [CLASS DOC] HELPER: subclass created with keyword x=3.
            pass

        self.assertEqual(B.kwargs, dict(x=3))

    def test_init_subclass_error(self):
        # [FUNCTION DOC] Checks an exception raised inside __init_subclass__ propagates out of the
        # class statement (RuntimeError expected).
        class A:
            # [CLASS DOC] HELPER: base whose hook always raises RuntimeError.
            def __init_subclass__(cls):
                # [FUNCTION DOC] HELPER: unconditionally raises RuntimeError.
                raise RuntimeError

        with self.assertRaises(RuntimeError):
            class B(A):
                # [CLASS DOC] HELPER: subclass whose creation must fail.
                pass

    def test_init_subclass_wrong(self):
        # [FUNCTION DOC] Checks that a hook requiring an argument that is not supplied makes class
        # creation raise TypeError.
        class A:
            # [CLASS DOC] HELPER: base whose hook needs a mandatory argument.
            def __init_subclass__(cls, whatever):
                # [FUNCTION DOC] HELPER: hook with a required parameter; body does nothing.
                pass

        with self.assertRaises(TypeError):
            class B(A):
                # [CLASS DOC] HELPER: subclass created without the required argument.
                pass

    def test_init_subclass_skipped(self):
        # [FUNCTION DOC] Checks the hook is invoked for each new class, including through an
        # intermediate class that does not define it, with the cls argument being that new class.
        class BaseWithInit:
            # [CLASS DOC] HELPER: base whose hook stores the new class on itself.
            def __init_subclass__(cls, **kwargs):
                # [FUNCTION DOC] HELPER: chains to the parent hook then stores cls as the
                # initialized attribute.
                super().__init_subclass__(**kwargs)
                cls.initialized = cls

        class BaseWithoutInit(BaseWithInit):
            # [CLASS DOC] HELPER: intermediate class with no hook of its own.
            pass

        class A(BaseWithoutInit):
            # [CLASS DOC] HELPER: leaf class created below the intermediate class.
            pass

        self.assertIs(A.initialized, A)
        self.assertIs(BaseWithoutInit.initialized, BaseWithoutInit)

    def test_init_subclass_diamond(self):
        # [FUNCTION DOC] Checks cooperative __init_subclass__ calls in a diamond hierarchy: the
        # final class gets calls recorded in method resolution order (right, then middle), while
        # the parents Left and Right keep an empty list.
        class Base:
            # [CLASS DOC] HELPER: root of the diamond; its hook resets a calls list on
            # the subclass.
            def __init_subclass__(cls, **kwargs):
                # [FUNCTION DOC] HELPER: forwards keyword arguments up the chain and gives cls an
                # empty list.
                super().__init_subclass__(**kwargs)
                cls.calls = []

        class Left(Base):
            # [CLASS DOC] HELPER: left diamond branch with no hook.
            pass

        class Middle:
            # [CLASS DOC] HELPER: mixin whose hook requires a middle keyword and appends it
            # to calls.
            def __init_subclass__(cls, middle, **kwargs):
                # [FUNCTION DOC] HELPER: forwards remaining keywords, then appends the middle
                # value to calls.
                super().__init_subclass__(**kwargs)
                cls.calls += [middle]

        class Right(Base):
            # [CLASS DOC] HELPER: right diamond branch whose hook appends a default right value.
            def __init_subclass__(cls, right="right", **kwargs):
                # [FUNCTION DOC] HELPER: forwards remaining keywords, then appends the right value
                # to calls.
                super().__init_subclass__(**kwargs)
                cls.calls += [right]

        class A(Left, Middle, Right, middle="middle"):
            # [CLASS DOC] HELPER: class combining Left, Middle and Right, passing middle as
            # a keyword.
            pass

        self.assertEqual(A.calls, ["right", "middle"])
        self.assertEqual(Left.calls, [])
        self.assertEqual(Right.calls, [])

    def test_set_name(self):
        # [FUNCTION DOC] Checks __set_name__ is called on a class attribute descriptor with the
        # owner class and attribute name.
        class Descriptor:
            # [CLASS DOC] HELPER: descriptor that records its owner and name in __set_name__.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: stores owner and name on the descriptor.
                self.owner = owner
                self.name = name

        class A:
            # [CLASS DOC] HELPER: class holding one descriptor attribute named d.
            d = Descriptor()

        self.assertEqual(A.d.name, "d")
        self.assertIs(A.d.owner, A)

    def test_set_name_metaclass(self):
        # [FUNCTION DOC] Checks descriptors are named before the metaclass __new__ returns, even
        # when the metaclass returns a non-class value (the class name is then bound to zero).
        class Meta(type):
            # [CLASS DOC] HELPER: metaclass asserting descriptor info after type creation and
            # returning 0.
            def __new__(cls, name, bases, ns):
                # [FUNCTION DOC] HELPER: builds the class, asserts name and owner of attribute d
                # were set, and returns zero instead of the class.
                ret = super().__new__(cls, name, bases, ns)
                self.assertEqual(ret.d.name, "d")
                self.assertIs(ret.d.owner, ret)
                return 0

        class Descriptor:
            # [CLASS DOC] HELPER: descriptor recording owner and name.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: stores owner and name on the descriptor.
                self.owner = owner
                self.name = name

        class A(metaclass=Meta):
            # [CLASS DOC] HELPER: class built by the zero-returning metaclass.
            d = Descriptor()
        self.assertEqual(A, 0)

    def test_set_name_error(self):
        # [FUNCTION DOC] Checks an exception inside __set_name__ propagates (ZeroDivisionError)
        # and that the exception notes mention the class name, attribute name and descriptor type.
        class Descriptor:
            # [CLASS DOC] HELPER: descriptor whose __set_name__ divides by zero.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: raises ZeroDivisionError.
                1/0

        with self.assertRaises(ZeroDivisionError) as cm:
            class NotGoingToWork:
                # [CLASS DOC] HELPER: class whose creation fails during descriptor naming.
                attr = Descriptor()

        notes = cm.exception.__notes__
        self.assertRegex(str(notes), r'\bNotGoingToWork\b')
        self.assertRegex(str(notes), r'\battr\b')
        self.assertRegex(str(notes), r'\bDescriptor\b')

    def test_set_name_wrong(self):
        # [FUNCTION DOC] Checks a __set_name__ with the wrong signature raises TypeError whose
        # notes name the class, attribute and descriptor type.
        class Descriptor:
            # [CLASS DOC] HELPER: descriptor whose __set_name__ takes no owner or name arguments.
            def __set_name__(self):
                # [FUNCTION DOC] HELPER: wrong-arity hook; does nothing.
                pass

        with self.assertRaises(TypeError) as cm:
            class NotGoingToWork:
                # [CLASS DOC] HELPER: class whose creation fails due to the bad hook.
                attr = Descriptor()

        notes = cm.exception.__notes__
        self.assertRegex(str(notes), r'\bNotGoingToWork\b')
        self.assertRegex(str(notes), r'\battr\b')
        self.assertRegex(str(notes), r'\bDescriptor\b')

    def test_set_name_lookup(self):
        # [FUNCTION DOC] Checks __set_name__ is looked up on the type, not the instance, so a
        # descriptor-like object with __getattr__ is never asked for it.
        resolved = []
        class NonDescriptor:
            # [CLASS DOC] HELPER: object whose __getattr__ records the attribute names requested.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: appends the requested name to the shared resolved list.
                resolved.append(name)

        class A:
            # [CLASS DOC] HELPER: class holding a NonDescriptor attribute.
            d = NonDescriptor()

        self.assertNotIn('__set_name__', resolved,
                         '__set_name__ is looked up in instance dict')

    def test_set_name_init_subclass(self):
        # [FUNCTION DOC] Checks __set_name__ runs before __init_subclass__ and before the
        # metaclass sees the class, so both can read the owner and name from the descriptor.
        class Descriptor:
            # [CLASS DOC] HELPER: descriptor recording owner and name.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: stores owner and name on the descriptor.
                self.owner = owner
                self.name = name

        class Meta(type):
            # [CLASS DOC] HELPER: metaclass copying owner and name attributes onto the class.
            def __new__(cls, name, bases, ns):
                # [FUNCTION DOC] HELPER: creates the class and copies its owner and name into
                # meta_ attributes.
                self = super().__new__(cls, name, bases, ns)
                self.meta_owner = self.owner
                self.meta_name = self.name
                return self

        class A:
            # [CLASS DOC] HELPER: base whose hook copies owner and name from descriptor d to
            # the subclass.
            def __init_subclass__(cls):
                # [FUNCTION DOC] HELPER: reads the descriptor d of cls and stores its owner and
                # name on cls.
                cls.owner = cls.d.owner
                cls.name = cls.d.name

        class B(A, metaclass=Meta):
            # [CLASS DOC] HELPER: class using the metaclass and base above with descriptor d.
            d = Descriptor()

        self.assertIs(B.owner, B)
        self.assertEqual(B.name, 'd')
        self.assertIs(B.meta_owner, B)
        self.assertEqual(B.name, 'd')

    def test_set_name_modifying_dict(self):
        # [FUNCTION DOC] Checks that __set_name__ may add attributes to the owner while the class
        # namespace is processed; all five descriptors must still be notified exactly once.
        notified = []
        class Descriptor:
            # [CLASS DOC] HELPER: descriptor that adds a new attribute to its owner and records
            # its name.
            def __set_name__(self, owner, name):
                # [FUNCTION DOC] HELPER: sets an extra attribute on owner and appends name
                # to notified.
                setattr(owner, name + 'x', None)
                notified.append(name)

        class A:
            # [CLASS DOC] HELPER: class with five descriptor attributes a to e.
            a = Descriptor()
            b = Descriptor()
            c = Descriptor()
            d = Descriptor()
            e = Descriptor()

        self.assertCountEqual(notified, ['a', 'b', 'c', 'd', 'e'])

    def test_errors(self):
        # [FUNCTION DOC] Checks errors for extra class keywords: the default metaclass rejects
        # them with TypeError (via class statement and types.new_class), types.prepare_class
        # accepts them, a metaclass __init__ alone is not enough, and a metaclass accepting the
        # keyword in both __new__ and __init__ stores it.
        class MyMeta(type):
            # [CLASS DOC] HELPER: plain metaclass that accepts no extra keywords.
            pass

        with self.assertRaises(TypeError):
            class MyClass(metaclass=MyMeta, otherarg=1):
                # [CLASS DOC] HELPER: class whose creation must fail on the unknown keyword.
                pass

        with self.assertRaises(TypeError):
            types.new_class("MyClass", (object,),
                            dict(metaclass=MyMeta, otherarg=1))
        types.prepare_class("MyClass", (object,),
                            dict(metaclass=MyMeta, otherarg=1))

        class MyMeta(type):
            # [CLASS DOC] HELPER: metaclass whose __init__ takes otherarg but __new__ does not.
            def __init__(self, name, bases, namespace, otherarg):
                # [FUNCTION DOC] HELPER: forwards only name, bases and namespace to type.__init__.
                super().__init__(name, bases, namespace)

        with self.assertRaises(TypeError):
            class MyClass2(metaclass=MyMeta, otherarg=1):
                # [CLASS DOC] HELPER: class whose creation must fail with TypeError.
                pass

        class MyMeta(type):
            # [CLASS DOC] HELPER: metaclass accepting otherarg in __new__ and __init__.
            def __new__(cls, name, bases, namespace, otherarg):
                # [FUNCTION DOC] HELPER: creates the class, dropping the otherarg keyword.
                return super().__new__(cls, name, bases, namespace)

            def __init__(self, name, bases, namespace, otherarg):
                # [FUNCTION DOC] HELPER: initializes the class and stores otherarg on it.
                super().__init__(name, bases, namespace)
                self.otherarg = otherarg

        class MyClass3(metaclass=MyMeta, otherarg=1):
            # [CLASS DOC] HELPER: class created with otherarg=1; must expose it as an attribute.
            pass

        self.assertEqual(MyClass3.otherarg, 1)

    def test_errors_changed_pep487(self):
        # [FUNCTION DOC] Checks PEP 487 behavior: a metaclass __new__ passing name, bases and dict
        # keywords to type.__new__ raises TypeError, while one accepting otherarg works.
        # These tests failed before Python 3.6, PEP 487
        class MyMeta(type):
            # [CLASS DOC] HELPER: metaclass that calls type.__new__ with a wrong keyword form.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: calls type.__new__ with keyword arguments, which
                # is rejected.
                return super().__new__(cls, name=name, bases=bases,
                                       dict=namespace)

        with self.assertRaises(TypeError):
            class MyClass(metaclass=MyMeta):
                # [CLASS DOC] HELPER: class whose creation fails because of the metaclass.
                pass

        class MyMeta(type):
            # [CLASS DOC] HELPER: metaclass that stores otherarg on the created class.
            def __new__(cls, name, bases, namespace, otherarg):
                # [FUNCTION DOC] HELPER: creates the class normally and records otherarg on it.
                self = super().__new__(cls, name, bases, namespace)
                self.otherarg = otherarg
                return self

        class MyClass2(metaclass=MyMeta, otherarg=1):
            # [CLASS DOC] HELPER: class created with otherarg=1.
            pass

        self.assertEqual(MyClass2.otherarg, 1)

    def test_type(self):
        # [FUNCTION DOC] Checks the three-argument call to type builds a class with the given
        # name, and that passing the arguments as keywords raises TypeError.
        t = type('NewClass', (object,), {})
        self.assertIsInstance(t, type)
        self.assertEqual(t.__name__, 'NewClass')

        with self.assertRaises(TypeError):
            type(name='NewClass', bases=(object,), dict={})


if __name__ == "__main__":
    run_tests()
