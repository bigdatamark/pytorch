# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-adapted copy of CPython 3.13 test_super: exercises zero-argument super(),
# the implicit __class__ cell and __classcell__ propagation through metaclasses, error messages
# for malformed super() calls, shadowing of the name super, and a threaded __class__ swap. Defines
# a diamond hierarchy A..G used by the tests and one CPythonTestCase. Imports test.support
# helpers; no other import-time side effects. run_tests() runs under __main__.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_super.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

"""Unit tests for zero-argument super() & related machinery."""

import textwrap
import threading
import unittest
from unittest.mock import patch
from test.support import import_helper, threading_helper


ADAPTIVE_WARMUP_DELAY = 2


class A:
    # [CLASS DOC] HELPER: Root of the test hierarchy; f returns 'A' and cm is a classmethod
    # returning (cls, 'A').
    def f(self):
        # [FUNCTION DOC] HELPER: Returns the string 'A'.
        return 'A'
    @classmethod
    def cm(cls):
        # [FUNCTION DOC] HELPER: Returns a tuple of the class it is called on and the string 'A'.
        return (cls, 'A')

class B(A):
    # [CLASS DOC] HELPER: Subclass of A whose methods call super() and append 'B' (f) or wrap in
    # 'B' (cm).
    def f(self):
        # [FUNCTION DOC] HELPER: Returns the result of the next f in the MRO with 'B' appended.
        return super().f() + 'B'
    @classmethod
    def cm(cls):
        # [FUNCTION DOC] HELPER: Returns a tuple of cls, the next cm in the MRO, and 'B'.
        return (cls, super().cm(), 'B')

class C(A):
    # [CLASS DOC] HELPER: Sibling of B under A, with the same cooperative super() pattern and
    # tag 'C'.
    def f(self):
        # [FUNCTION DOC] HELPER: Returns the next f in the MRO with 'C' appended.
        return super().f() + 'C'
    @classmethod
    def cm(cls):
        # [FUNCTION DOC] HELPER: Returns a tuple of cls, the next cm in the MRO, and 'C'.
        return (cls, super().cm(), 'C')

class D(C, B):
    # [CLASS DOC] HELPER: Diamond subclass of C and B (MRO D, C, B, A) with cooperative f and cm.
    def f(self):
        # [FUNCTION DOC] HELPER: Returns the next f in the MRO with 'D' appended; full chain
        # gives 'ABCD'.
        return super().f() + 'D'
    def cm(cls):
        # [FUNCTION DOC] HELPER: Not decorated as a classmethod, so the first argument is the
        # instance; returns a tuple of it, the next cm via super(), and 'D'.
        return (cls, super().cm(), 'D')

class E(D):
    # [CLASS DOC] HELPER: Empty subclass of D, checks inherited methods keep working with super().
    pass

class F(E):
    # [CLASS DOC] HELPER: Subclass of E that rebinds f to the plain function E.f
    # (unbound transfer).
    f = E.f

class G(A):
    # [CLASS DOC] HELPER: Plain subclass of A with no overrides, used for classmethod tests.
    pass


class TestSuper(CPythonTestCase):
    # [CLASS DOC] Test case for zero-argument super() and related compiler/runtime machinery.

    def tearDown(self):
        # [FUNCTION DOC] Restores the module-level __class__ binding that the pathologies test
        # rewrote to TestSuper, then calls the parent tearDown.
        # This fixes the damage that test_various___class___pathologies does.
        nonlocal __class__
        __class__ = TestSuper
        super().tearDown()

    def test_basics_working(self):
        # [FUNCTION DOC] D().f() walks the whole MRO and yields 'ABCD'.
        self.assertEqual(D().f(), 'ABCD')

    def test_class_getattr_working(self):
        # [FUNCTION DOC] Calling D.f on an explicit instance via the class yields 'ABCD'.
        self.assertEqual(D.f(D()), 'ABCD')

    def test_subclass_no_override_working(self):
        # [FUNCTION DOC] E inherits f without override: call through instance and through class
        # give 'ABCD'.
        self.assertEqual(E().f(), 'ABCD')
        self.assertEqual(E.f(E()), 'ABCD')

    def test_unbound_method_transfer_working(self):
        # [FUNCTION DOC] F.f is the E.f function transferred to a new class; super() still
        # resolves to 'ABCD'.
        self.assertEqual(F().f(), 'ABCD')
        self.assertEqual(F.f(F()), 'ABCD')

    def test_class_methods_still_working(self):
        # [FUNCTION DOC] Plain classmethod calls on A and G, via class and instance, return the
        # right cls and 'A'.
        self.assertEqual(A.cm(), (A, 'A'))
        self.assertEqual(A().cm(), (A, 'A'))
        self.assertEqual(G.cm(), (G, 'A'))
        self.assertEqual(G().cm(), (G, 'A'))

    def test_super_in_class_methods_working(self):
        # [FUNCTION DOC] cm on D and E instances (not a classmethod in D) nests the super()
        # results as expected; asserts the exact nested tuple for each.
        d = D()
        self.assertEqual(d.cm(), (d, (D, (D, (D, 'A'), 'B'), 'C'), 'D'))
        e = E()
        self.assertEqual(e.cm(), (e, (E, (E, (E, 'A'), 'B'), 'C'), 'D'))

    def test_super_with_closure(self):
        # [FUNCTION DOC] Regression for issue 4360: super() works in a method that also defines a
        # nested closure over self. Expects the string 'AE'.
        # Issue4360: super() did not work in a function that
        # contains a closure
        class E(A):
            # [CLASS DOC] HELPER: Local subclass of A whose f contains a closure and
            # calls super().
            def f(self):
                # [FUNCTION DOC] HELPER: Defines an unused nested function capturing self, returns
                # super f plus 'E'.
                def nested():
                    # [FUNCTION DOC] HELPER: Does nothing but reference self so the method has
                    # a closure.
                    self
                return super().f() + 'E'

        self.assertEqual(E().f(), 'AE')

    def test_various___class___pathologies(self):
        # [FUNCTION DOC] Issue 12370: checks odd uses of the name __class__. A class attribute
        # __class__ does not break super(); a class-body __class__ reference sees the enclosing
        # test class; evaluating it in an exec'd class raises NameError (not UnboundLocalError);
        # global and nonlocal declarations of __class__ in a class body bind the module global and
        # enclosing variable respectively.
        # See issue #12370
        class X(A):
            # [CLASS DOC] HELPER: Subclass of A that sets a class attribute __class__ to 413.
            def f(self):
                # [FUNCTION DOC] HELPER: Returns super().f() (still works despite the
                # __class__ attribute).
                return super().f()
            __class__ = 413
        x = X()
        self.assertEqual(x.f(), 'A')
        self.assertEqual(x.__class__, 413)
        class X:
            # [CLASS DOC] HELPER: Class whose body reads __class__ from the enclosing scope.
            x = __class__
            def f():
                # [FUNCTION DOC] HELPER: Method that references __class__ (never called).
                __class__
        self.assertIs(X.x, type(self))
        with self.assertRaises(NameError) as e:
            exec("""class X:
                __class__
                def f():
                    __class__""", globals(), {})
        self.assertIs(type(e.exception), NameError) # Not UnboundLocalError
        class X:
            # [CLASS DOC] HELPER: Class whose body declares __class__ global and assigns 42 to it.
            global __class__
            __class__ = 42
            def f():
                # [FUNCTION DOC] HELPER: Method that references __class__ (never called).
                __class__
        self.assertEqual(globals()["__class__"], 42)
        del globals()["__class__"]
        self.assertNotIn("__class__", X.__dict__)
        class X:
            # [CLASS DOC] HELPER: Class whose body declares __class__ nonlocal and assigns 42
            # to it.
            nonlocal __class__
            __class__ = 42
            def f():
                # [FUNCTION DOC] HELPER: Method that references __class__ (never called).
                __class__
        self.assertEqual(__class__, 42)

    def test___class___instancemethod(self):
        # [FUNCTION DOC] Issue 14857: an instance method returning __class__ yields the
        # defining class.
        # See issue #14857
        class X:
            # [CLASS DOC] HELPER: Class with an instance method that returns __class__.
            def f(self):
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__
        self.assertIs(X().f(), X)

    def test___class___classmethod(self):
        # [FUNCTION DOC] Issue 14857: same as above for a classmethod.
        # See issue #14857
        class X:
            # [CLASS DOC] HELPER: Class with a classmethod that returns __class__.
            @classmethod
            def f(cls):
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__
        self.assertIs(X.f(), X)

    def test___class___staticmethod(self):
        # [FUNCTION DOC] Issue 14857: same as above for a staticmethod.
        # See issue #14857
        class X:
            # [CLASS DOC] HELPER: Class with a staticmethod that returns __class__.
            @staticmethod
            def f():
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__
        self.assertIs(X.f(), X)

    def test___class___new(self):
        # [FUNCTION DOC] Issue 23722: the __class__ cell is already filled when the metaclass
        # __new__ finishes type creation, so a static method can be called there and returns the
        # new class.
        # See issue #23722
        # Ensure zero-arg super() works as soon as type.__new__() is completed
        test_class = None

        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that calls f on the freshly created class and records
            # the result.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: Creates the class, stores the result of calling its f in
                # test_class, returns the class.
                nonlocal test_class
                self = super().__new__(cls, name, bases, namespace)
                test_class = self.f()
                return self

        class A(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using Meta; its static f returns __class__.
            @staticmethod
            def f():
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__

        self.assertIs(test_class, A)

    def test___class___delayed(self):
        # [FUNCTION DOC] Issue 23722: if the metaclass returns None, the class name is bound to
        # None; creating a class later from the saved namespace via type() fills the cell so f
        # returns that class.
        # See issue #23722
        test_namespace = None

        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that saves the namespace and returns None instead of
            # a class.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: Records the namespace in test_namespace and returns None.
                nonlocal test_namespace
                test_namespace = namespace
                return None

        class A(metaclass=Meta):
            # [CLASS DOC] HELPER: Class whose creation yields None because of Meta; has a static f
            # using __class__.
            @staticmethod
            def f():
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__

        self.assertIs(A, None)

        B = type("B", (), test_namespace)
        self.assertIs(B.f(), B)

    def test___class___mro(self):
        # [FUNCTION DOC] Issue 23722: the cell is populated before the metaclass mro hook runs, so
        # a function stored in the class dict can read __class__ from within mro.
        # See issue #23722
        test_class = None

        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose mro override calls the class's f first, then
            # defers to type.
            def mro(self):
                # [FUNCTION DOC] HELPER: Calls the f found directly in the class dict, then
                # returns the normal mro.
                # self.f() doesn't work yet...
                self.__dict__["f"]()
                return super().mro()

        class A(metaclass=Meta):
            # [CLASS DOC] HELPER: Class whose f records __class__ into test_class.
            def f():
                # [FUNCTION DOC] HELPER: Stores the implicit __class__ cell value into the
                # enclosing test_class.
                nonlocal test_class
                test_class = __class__

        self.assertIs(test_class, A)

    def test___classcell___expected_behaviour(self):
        # [FUNCTION DOC] Issue 23722: the compiler adds __classcell__ to the namespace only when a
        # method needs it; it is the very cell found in the method closure and is not exposed as a
        # class attribute.
        # See issue #23722
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that snapshots a copy of the class namespace before
            # creating the class.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: Saves a copy of namespace in namespace_snapshot and
                # creates the class normally.
                nonlocal namespace_snapshot
                namespace_snapshot = namespace.copy()
                return super().__new__(cls, name, bases, namespace)

        # __classcell__ is injected into the class namespace by the compiler
        # when at least one method needs it, and should be omitted otherwise
        namespace_snapshot = None
        class WithoutClassRef(metaclass=Meta):
            # [CLASS DOC] HELPER: Class with no __class__ use, so no __classcell__ is injected.
            pass
        self.assertNotIn("__classcell__", namespace_snapshot)

        # With zero-arg super() or an explicit __class__ reference,
        # __classcell__ is the exact cell reference to be populated by
        # type.__new__
        namespace_snapshot = None
        class WithClassRef(metaclass=Meta):
            # [CLASS DOC] HELPER: Class whose method references __class__, so __classcell__
            # is injected.
            def f(self):
                # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                return __class__

        class_cell = namespace_snapshot["__classcell__"]
        method_closure = WithClassRef.f.__closure__
        self.assertEqual(len(method_closure), 1)
        self.assertIs(class_cell, method_closure[0])
        # Ensure the cell reference *doesn't* get turned into an attribute
        with self.assertRaises(AttributeError):
            WithClassRef.__classcell__

    def test___classcell___missing(self):
        # [FUNCTION DOC] Issue 23722: a metaclass that drops __classcell__ is fine for
        # classes that do not need it, but classes using __class__ or zero-arg super()
        # raise a RuntimeError saying the cell was not set and asking whether __classcell__
        # was propagated.
        # See issue #23722
        # Some metaclasses may not pass the original namespace to type.__new__
        # We test that case here by forcibly deleting __classcell__
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that removes __classcell__ from the namespace before
            # type creation.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: Pops __classcell__ (if present) and creates the class.
                namespace.pop('__classcell__', None)
                return super().__new__(cls, name, bases, namespace)

        # The default case should continue to work without any errors
        class WithoutClassRef(metaclass=Meta):
            # [CLASS DOC] HELPER: Class needing no cell; must still be created successfully.
            pass

        # With zero-arg super() or an explicit __class__ reference, we expect
        # __build_class__ to raise a RuntimeError complaining that
        # __class__ was not set, and asking if __classcell__ was propagated
        # to type.__new__.
        expected_error = '__class__ not set.*__classcell__ propagated'
        with self.assertRaisesRegex(RuntimeError, expected_error):
            class WithClassRef(metaclass=Meta):
                # [CLASS DOC] HELPER: Class needing the cell; creation is expected to
                # raise RuntimeError.
                def f(self):
                    # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                    return __class__

    def test___classcell___overwrite(self):
        # [FUNCTION DOC] Issue 23722: a metaclass that replaces __classcell__ with a non-cell
        # value (None, 0, empty string, plain object) makes class creation raise TypeError,
        # checked per value.
        # See issue #23722
        # Overwriting __classcell__ with nonsense is explicitly prohibited
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that overwrites __classcell__ in the namespace with
            # the cell keyword.
            def __new__(cls, name, bases, namespace, cell):
                # [FUNCTION DOC] HELPER: Sets __classcell__ to the supplied cell value and creates
                # the class.
                namespace['__classcell__'] = cell
                return super().__new__(cls, name, bases, namespace)

        for bad_cell in (None, 0, "", object()):
            with self.subTest(bad_cell=bad_cell):
                with self.assertRaises(TypeError):
                    class A(metaclass=Meta, cell=bad_cell):
                        # [CLASS DOC] HELPER: Empty class created with an invalid __classcell__
                        # injected by Meta.
                        pass

    def test___classcell___wrong_cell(self):
        # [FUNCTION DOC] Issue 23722: creating a second class from the same namespace (so the cell
        # would point to the wrong class) makes class creation raise TypeError.
        # See issue #23722
        # Pointing the cell reference at the wrong class is also prohibited
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass that creates the class, then builds an extra type from
            # the same namespace.
            def __new__(cls, name, bases, namespace):
                # [FUNCTION DOC] HELPER: Creates the class and a second class sharing the
                # namespace, returns the first.
                cls = super().__new__(cls, name, bases, namespace)
                B = type("B", (), namespace)
                return cls

        with self.assertRaises(TypeError):
            class A(metaclass=Meta):
                # [CLASS DOC] HELPER: Class using the cell; creation is expected to
                # raise TypeError.
                def f(self):
                    # [FUNCTION DOC] HELPER: Returns the implicit __class__ cell value.
                    return __class__

    def test_obscure_super_errors(self):
        # [FUNCTION DOC] Checks RuntimeError messages from zero-arg super() in unusual frames: a
        # function with no arguments, a method with no arguments, a deleted first argument, and an
        # emptied __class__ cell.
        def f():
            # [FUNCTION DOC] HELPER: Calls zero-arg super() with no enclosing class or arguments;
            # raises RuntimeError.
            super()
        with self.assertRaisesRegex(RuntimeError, r"no arguments"):
            f()

        class C:
            def f():
                # [FUNCTION DOC] HELPER: Method without parameters that calls super();
                # raises RuntimeError.
                super()
        with self.assertRaisesRegex(RuntimeError, r"no arguments"):
            C.f()

        def f(x):
            # [FUNCTION DOC] HELPER: Deletes its only argument then calls super();
            # raises RuntimeError.
            del x
            super()
        with self.assertRaisesRegex(RuntimeError, r"arg\[0\] deleted"):
            f(None)

        class X:
            def f(x):
                # [FUNCTION DOC] HELPER: Deletes the __class__ cell then calls super();
                # raises RuntimeError.
                nonlocal __class__
                del __class__
                super()
        with self.assertRaisesRegex(RuntimeError, r"empty __class__ cell"):
            X().f()

    def test_cell_as_self(self):
        # [FUNCTION DOC] A closure cell object passed as self to a method using zero-arg super()
        # raises TypeError rather than crashing.
        class X:
            # [CLASS DOC] HELPER: Class whose method calls zero-arg super().
            def meth(self):
                # [FUNCTION DOC] HELPER: Calls super() with no arguments.
                super()

        def f():
            # [FUNCTION DOC] HELPER: Builds a closure over an X instance and returns it.
            k = X()
            def g():
                # [FUNCTION DOC] HELPER: Returns the captured k.
                return k
            return g
        c = f().__closure__[0]
        self.assertRaises(TypeError, X.meth, c)

    def test_super_init_leaks(self):
        # [FUNCTION DOC] Issue 26718: calling super.__init__ repeatedly (1000 times) on one super
        # object must not leak memory; detected with regrtest refleak mode, no explicit assertion.
        # Issue #26718: super.__init__ leaked memory if called multiple times.
        # This will be caught by regrtest.py -R if this leak.
        # NOTE: Despite the use in the test a direct call of super.__init__
        # is not endorsed.
        sp = super(float, 1.0)
        for i in range(1000):
            super.__init__(sp, int, i)

    def test_super_argcount(self):
        # [FUNCTION DOC] super called with three arguments raises TypeError mentioning 'expected
        # at most'.
        with self.assertRaisesRegex(TypeError, "expected at most"):
            super(int, int, int)

    def test_super_argtype(self):
        # [FUNCTION DOC] super with a non-type first argument raises TypeError about argument 1
        # being a type.
        with self.assertRaisesRegex(TypeError, "argument 1 must be a type"):
            super(1, int)

    def test_shadowed_global(self):
        # [FUNCTION DOC] Writes a module source that defines its own class named super and uses it
        # without arguments; imports it temporarily and checks the shadowing class is used, then
        # unloads it.
        source = textwrap.dedent(
            """
            class super:
                msg = "truly super"

            class C:
                def method(self):
                    return super().msg
            """,
        )
        with import_helper.ready_to_import(name="shadowed_super", source=source):
            import shadowed_super
        self.assertEqual(shadowed_super.C().method(), "truly super")
        import_helper.unload("shadowed_super")

    def test_shadowed_local(self):
        # [FUNCTION DOC] A local class named super shadows the builtin in a method; the method
        # returns its msg.
        class super:
            # [CLASS DOC] HELPER: Local class named super with a msg attribute.
            msg = "quite super"

        class C:
            # [CLASS DOC] HELPER: Class whose method calls the shadowed super and reads msg.
            def method(self):
                # [FUNCTION DOC] HELPER: Returns msg from the object produced by calling the
                # name super.
                return super().msg

        self.assertEqual(C().method(), "quite super")

    def test_shadowed_dynamic(self):
        # [FUNCTION DOC] Patching a module-level name super with another class makes a method
        # use it.
        class MySuper:
            # [CLASS DOC] HELPER: Stand-in class with a msg attribute, patched in for super.
            msg = "super super"

        class C:
            # [CLASS DOC] HELPER: Class whose method calls the (patched) name super and reads msg.
            def method(self):
                # [FUNCTION DOC] HELPER: Returns msg from the object produced by calling super.
                return super().msg

        with patch(f"{__name__}.super", MySuper) as m:
            self.assertEqual(C().method(), "super super")

    def test_shadowed_dynamic_two_arg(self):
        # [FUNCTION DOC] Like the dynamic shadowing test but with a two-argument call; the stand-
        # in must receive exactly the arguments passed, which are recorded in a list and compared.
        call_args = []
        class MySuper:
            # [CLASS DOC] HELPER: Stand-in for super that records its constructor arguments
            # in call_args.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Appends the tuple of positional arguments to call_args.
                call_args.append(args)
            msg = "super super"

        class C:
            # [CLASS DOC] HELPER: Class whose method calls super with two literal arguments.
            def method(self):
                # [FUNCTION DOC] HELPER: Returns msg from super(1, 2).
                return super(1, 2).msg

        with patch(f"{__name__}.super", MySuper) as m:
            self.assertEqual(C().method(), "super super")
            self.assertEqual(call_args, [(1, 2)])

    def test_attribute_error(self):
        # [FUNCTION DOC] Missing attribute on a super object raises AttributeError naming 'super'
        # and the attribute.
        class C:
            # [CLASS DOC] HELPER: Class whose method reads an attribute that does not exist
            # on super().
            def method(self):
                # [FUNCTION DOC] HELPER: Returns the nonexistent msg attribute of the
                # super object.
                return super().msg

        with self.assertRaisesRegex(AttributeError, "'super' object has no attribute 'msg'"):
            C().method()

    def test_bad_first_arg(self):
        # [FUNCTION DOC] super with a non-type first argument raises TypeError when evaluated
        # inside a method.
        class C:
            # [CLASS DOC] HELPER: Class whose method calls super with an integer as the
            # first argument.
            def method(self):
                # [FUNCTION DOC] HELPER: Calls super(1, self).method().
                return super(1, self).method()

        with self.assertRaisesRegex(TypeError, "argument 1 must be a type"):
            C().method()

    def test_supercheck_fail(self):
        # [FUNCTION DOC] Checks the TypeError text of a failed super type check for three cases:
        # obj of unrelated type, an instance of a different class, and a type object, with the
        # right wording each.
        class C:
            # [CLASS DOC] HELPER: Class whose method forwards caller-chosen type and object
            # to super.
            def method(self, type_, obj):
                # [FUNCTION DOC] HELPER: Returns super(type_, obj).method().
                return super(type_, obj).method()

        c = C()
        err_msg = (
            r"super\(type, obj\): obj \({} {}\) is not "
            r"an instance or subtype of type \({}\)."
        )

        cases = (
            (int, c, int.__name__, C.__name__, "instance of"),
            # obj is instance of type
            (C, list(), C.__name__, list.__name__, "instance of"),
            # obj is type itself
            (C, list, C.__name__, list.__name__, "type"),
        )

        for case in cases:
            with self.subTest(case=case):
                type_, obj, type_str, obj_str, instance_or_type = case
                regex = err_msg.format(instance_or_type, obj_str, type_str)

                with self.assertRaisesRegex(TypeError, regex):
                    c.method(type_, obj)

    def test_super___class__(self):
        # [FUNCTION DOC] The __class__ attribute of a super object is the builtin super.
        class C:
            # [CLASS DOC] HELPER: Class whose method returns the __class__ of a super object.
            def method(self):
                # [FUNCTION DOC] HELPER: Returns super().__class__.
                return super().__class__

        self.assertEqual(C().method(), super)

    def test_super_subclass___class__(self):
        # [FUNCTION DOC] The __class__ of an instance of a super subclass is that subclass.
        class mysuper(super):
            # [CLASS DOC] HELPER: Trivial subclass of the builtin super.
            pass

        class C:
            # [CLASS DOC] HELPER: Class whose method returns the __class__ of a mysuper instance.
            def method(self):
                # [FUNCTION DOC] HELPER: Returns mysuper(C, self).__class__.
                return mysuper(C, self).__class__

        self.assertEqual(C().method(), mysuper)

    def test_unusual_getattro(self):
        # [FUNCTION DOC] Uses super with a metaclass-derived metaclass to call type.__setattr__ on
        # a class and verifies the attribute is set; repeated ADAPTIVE_WARMUP_DELAY times to hit
        # specialized paths.
        class MyType(type):
            # [CLASS DOC] HELPER: Empty metaclass.
            pass

        def test(name):
            # [FUNCTION DOC] HELPER: Creates a class whose metaclass is MyType, sets attribute bar
            # through super, asserts it is 1.
            mytype = MyType(name, (MyType,), {})
            super(MyType, type(mytype)).__setattr__(mytype, "bar", 1)
            self.assertEqual(mytype.bar, 1)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            test("foo1")

    def test_reassigned_new(self):
        # [FUNCTION DOC] A __new__ reassigned to itself in __init_subclass__ must still work with
        # super().__new__ in a subclass; constructs C repeatedly to warm up specialization.
        class A:
            # [CLASS DOC] HELPER: Base whose __new__ does nothing and whose __init_subclass__
            # copies __new__ into subclasses.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns nothing.
                pass

            def __init_subclass__(cls):
                # [FUNCTION DOC] HELPER: Assigns the inherited __new__ onto the subclass if not
                # defined in its dict.
                if "__new__" not in cls.__dict__:
                    cls.__new__ = cls.__new__

        class B(A):
            # [CLASS DOC] HELPER: Empty subclass of A.
            pass

        class C(B):
            # [CLASS DOC] HELPER: Subclass of B whose __new__ calls super().__new__.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Returns super().__new__(cls).
                return super().__new__(cls)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            C()

    def test_mixed_staticmethod_hierarchy(self):
        # [FUNCTION DOC] Desugared version of the reassigned-new test using a mix of staticmethods
        # and plain functions along the hierarchy; calls C.some repeatedly.
        # This test is just a desugared version of `test_reassigned_new`
        class A:
            # [CLASS DOC] HELPER: Base with a staticmethod some asserting no extra arguments
            # are passed.
            @staticmethod
            def some(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Asserts that args and kwargs are empty.
                self.assertFalse(args)
                self.assertFalse(kwargs)

        class B(A):
            # [CLASS DOC] HELPER: Subclass with a plain-function some that forwards to super.
            def some(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Returns super().some(cls, args, kwargs) forwarded.
                return super().some(cls, *args, **kwargs)

        class C(B):
            # [CLASS DOC] HELPER: Subclass with a staticmethod some that calls super.
            @staticmethod
            def some(cls):
                # [FUNCTION DOC] HELPER: Returns super().some(cls).
                return super().some(cls)

        for _ in range(ADAPTIVE_WARMUP_DELAY):
            C.some(C)

    @threading_helper.requires_working_threading()
    def test___class___modification_multithreaded(self):
        # [FUNCTION DOC] Starts six threads that each flip an object's __class__ between two
        # classes 200 times, then joins them. Asserts nothing; it relies on an audit hook crashing
        # on old Pythons.
        """ Note: this test isn't actually testing anything on its own.
        It requires a sys audithook to be set to crash on older Python.
        This should be the case anyways as our test suite sets
        an audit hook.
        """

        class Foo:
            # [CLASS DOC] HELPER: Empty class, one of two swapped between.
            pass

        class Bar:
            # [CLASS DOC] HELPER: Empty class, the other swapped to.
            pass

        thing = Foo()
        def work():
            # [FUNCTION DOC] HELPER: Thread body: swaps the shared object's __class__ back and
            # forth 200 times.
            foo = thing
            for _ in range(200):
                foo.__class__ = Bar
                type(foo)
                foo.__class__ = Foo
                type(foo)


        threads = []
        for _ in range(6):
            thread = threading.Thread(target=work)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()


if __name__ == "__main__":
    run_tests()
