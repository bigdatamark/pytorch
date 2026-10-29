# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_property suite. Defines helper
# property-using classes (getter/setter/deleter raising subclasses, doc-string
# inheritance cases, property subclasses with and without __slots__) and test cases
# checking property decorators, __doc__ propagation, __name__, __set_name__ and the
# error messages for properties lacking getter/setter/deleter. Importing has no side
# effects beyond class creation; run_tests is called only when run as a script.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_property.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

# Test case for property
# more tests are in test_descr

import sys
import unittest
from test import support

class PropertyBase(Exception):
    # [CLASS DOC] HELPER: Base exception for the markers raised by property accessors below.
    pass

class PropertyGet(PropertyBase):
    # [CLASS DOC] HELPER: Raised by the overriding getter of SubClass.
    pass

class PropertySet(PropertyBase):
    # [CLASS DOC] HELPER: Raised by the overriding setter of SubClass.
    pass

class PropertyDel(PropertyBase):
    # [CLASS DOC] HELPER: Raised by the overriding deleter of SubClass.
    pass

class BaseClass(object):
    # [CLASS DOC] HELPER: Fixture with a read/write/delete property "spam" backed by the
    # instance attribute _spam (initialised to 5); the getter has a docstring.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialise the backing attribute _spam to 5.
        self._spam = 5

    @property
    def spam(self):
        # [FUNCTION DOC] HELPER: Property getter; returns the stored _spam value.
        """BaseClass.getter"""
        return self._spam

    @spam.setter
    def spam(self, value):
        # [FUNCTION DOC] HELPER: Property setter; stores the given value in _spam.
        self._spam = value

    @spam.deleter
    def spam(self):
        # [FUNCTION DOC] HELPER: Property deleter; removes the _spam attribute.
        del self._spam

class SubClass(BaseClass):
    # [CLASS DOC] HELPER: Subclass of BaseClass whose property accessors (via getter/setter/
    # deleter on the inherited property) all raise PropertyGet/Set/Del carrying _spam.

    @BaseClass.spam.getter
    def spam(self):
        # [FUNCTION DOC] HELPER: Overriding getter; always raises PropertyGet with the _spam value.
        """SubClass.getter"""
        raise PropertyGet(self._spam)

    @spam.setter
    def spam(self, value):
        # [FUNCTION DOC] HELPER: Overriding setter; always raises PropertySet with the _spam value.
        raise PropertySet(self._spam)

    @spam.deleter
    def spam(self):
        # [FUNCTION DOC] HELPER: Overriding deleter; always raises PropertyDel with
        # the _spam value.
        raise PropertyDel(self._spam)

class PropertyDocBase(object):
    # [CLASS DOC] HELPER: Fixture whose property "spam" is built from a plain function with an
    # explicit doc argument; class attribute _spam is 1.
    _spam = 1
    def _get_spam(self):
        # [FUNCTION DOC] HELPER: Getter function used to build the property; returns _spam.
        return self._spam
    spam = property(_get_spam, doc="spam spam spam")

class PropertyDocSub(PropertyDocBase):
    # [CLASS DOC] HELPER: Subclass that replaces the getter of the inherited documented
    # property; used to show the explicit doc is kept over the new getter's docstring.
    @PropertyDocBase.spam.getter
    def spam(self):
        # [FUNCTION DOC] HELPER: New getter; returns _spam. Its docstring must not become the doc.
        """The decorator does not use this doc string"""
        return self._spam

class PropertySubNewGetter(BaseClass):
    # [CLASS DOC] HELPER: Subclass of BaseClass that swaps in a getter with a new docstring
    # and a constant result of 5.
    @BaseClass.spam.getter
    def spam(self):
        # [FUNCTION DOC] HELPER: Replacement getter; returns the constant 5.
        """new docstring"""
        return 5

class PropertyNewGetter(object):
    # [CLASS DOC] HELPER: Fixture that redefines its property getter after creation, so the
    # final getter (returning 8, new docstring) replaces the original (returning 1).
    @property
    def spam(self):
        # [FUNCTION DOC] HELPER: Original getter; returns 1 (replaced later in the class body).
        """original docstring"""
        return 1
    @spam.getter
    def spam(self):
        # [FUNCTION DOC] HELPER: Replacement getter; returns 8 and carries the new docstring.
        """new docstring"""
        return 8

class PropertyTests(CPythonTestCase):
    # [CLASS DOC] Test case for basic property decorator behavior: inheritance of accessors,
    # docstring handling, abstractmethod flag, reference-leak checks, __name__ and
    # __set_name__ semantics.
    def test_property_decorator_baseclass(self):
        # [FUNCTION DOC] Checks get, set and delete on a base-class property; after deleting,
        # both the property and backing attribute are gone, and it can be set again.
        # see #1620
        base = BaseClass()
        self.assertEqual(base.spam, 5)
        self.assertEqual(base._spam, 5)
        base.spam = 10
        self.assertEqual(base.spam, 10)
        self.assertEqual(base._spam, 10)
        delattr(base, "spam")
        self.assertTrue(not hasattr(base, "spam"))
        self.assertTrue(not hasattr(base, "_spam"))
        base.spam = 20
        self.assertEqual(base.spam, 20)
        self.assertEqual(base._spam, 20)

    def test_property_decorator_subclass(self):
        # [FUNCTION DOC] Checks that the subclass accessors raise PropertyGet, PropertySet and
        # PropertyDel when accessed through getattr, setattr and delattr.
        # see #1620
        sub = SubClass()
        self.assertRaises(PropertyGet, getattr, sub, "spam")
        self.assertRaises(PropertySet, setattr, sub, "spam", None)
        self.assertRaises(PropertyDel, delattr, sub, "spam")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_subclass_doc(self):
        # [FUNCTION DOC] Checks the subclass property doc is the new getter's docstring (skipped
        # when running with -OO).
        sub = SubClass()
        self.assertEqual(sub.__class__.spam.__doc__, "SubClass.getter")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_baseclass_doc(self):
        # [FUNCTION DOC] Checks the base-class property doc is the getter's docstring (skipped
        # with -OO).
        base = BaseClass()
        self.assertEqual(base.__class__.spam.__doc__, "BaseClass.getter")

    def test_property_decorator_doc(self):
        # [FUNCTION DOC] Checks the explicit doc argument is kept by the base and by a subclass
        # that only replaces the getter.
        base = PropertyDocBase()
        sub = PropertyDocSub()
        self.assertEqual(base.__class__.spam.__doc__, "spam spam spam")
        self.assertEqual(sub.__class__.spam.__doc__, "spam spam spam")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_getter_doc_override(self):
        # [FUNCTION DOC] Checks that replacing a getter (in a subclass and in the same class)
        # makes the new getter's docstring the property doc (skipped with -OO).
        newgettersub = PropertySubNewGetter()
        self.assertEqual(newgettersub.spam, 5)
        self.assertEqual(newgettersub.__class__.spam.__doc__, "new docstring")
        newgetter = PropertyNewGetter()
        self.assertEqual(newgetter.spam, 8)
        self.assertEqual(newgetter.__class__.spam.__doc__, "new docstring")

    def test_property___isabstractmethod__descriptor(self):
        # [FUNCTION DOC] Checks the property __isabstractmethod__ mirrors the truthiness of the
        # wrapped function's flag for several values, and that an exception from truth
        # testing the flag propagates as ValueError.
        for val in (True, False, [], [1], '', '1'):
            class C(object):
                # [CLASS DOC] HELPER: Wraps a function carrying the loop value as its
                # abstract flag.
                def foo(self):
                    # [FUNCTION DOC] HELPER: Empty method wrapped by the property.
                    pass
                foo.__isabstractmethod__ = val
                foo = property(foo)
            self.assertIs(C.foo.__isabstractmethod__, bool(val))

        # check that the property's __isabstractmethod__ descriptor does the
        # right thing when presented with a value that fails truth testing:
        class NotBool(object):
            # [CLASS DOC] HELPER: Object whose truth test (bool and len) raises ValueError.
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError; also aliased as the
                # length method.
                raise ValueError()
            __len__ = __bool__
        with self.assertRaises(ValueError):
            class C(object):
                # [CLASS DOC] HELPER: Wraps a function whose abstract flag fails truth testing.
                def foo(self):
                    # [FUNCTION DOC] HELPER: Empty method wrapped by the property.
                    pass
                foo.__isabstractmethod__ = NotBool()
                foo = property(foo)
            C.foo.__isabstractmethod__

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_builtin_doc_writable(self):
        # [FUNCTION DOC] Checks the doc of a property built with a doc argument can be reassigned
        # (skipped with -OO).
        p = property(doc='basic')
        self.assertEqual(p.__doc__, 'basic')
        p.__doc__ = 'extended'
        self.assertEqual(p.__doc__, 'extended')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_decorator_doc_writable(self):
        # [FUNCTION DOC] Checks the doc of a decorator-created property can be reassigned on the
        # class-level property (skipped with -OO).
        class PropertyWritableDoc(object):
            # [CLASS DOC] HELPER: Holds a property with docstring "Eggs".

            @property
            def spam(self):
                # [FUNCTION DOC] HELPER: Getter returning the string eggs.
                """Eggs"""
                return "eggs"

        sub = PropertyWritableDoc()
        self.assertEqual(sub.__class__.spam.__doc__, 'Eggs')
        sub.__class__.spam.__doc__ = 'Spam'
        self.assertEqual(sub.__class__.spam.__doc__, 'Spam')

    @support.refcount_test
    def test_refleaks_in___init__(self):
        # [FUNCTION DOC] Checks that re-calling property.__init__ 100 times does not leak
        # references (total refcount delta within 10). Needs a debug build, via refcount_test.
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        fake_prop = property('fget', 'fset', 'fdel', 'doc')
        refs_before = gettotalrefcount()
        for i in range(100):
            fake_prop.__init__('fget', 'fset', 'fdel', 'doc')
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.refcount_test
    def test_gh_115618(self):
        # [FUNCTION DOC] Regression test for gh-115618: getter/setter/deleter called with None
        # 100 times each must leave fget/fset/fdel as None and not change the total refcount
        # by more than 10. Needs a debug build.
        # Py_XDECREF() was improperly called for None argument
        # in property methods.
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        prop = property()
        refs_before = gettotalrefcount()
        for i in range(100):
            prop = prop.getter(None)
        self.assertIsNone(prop.fget)
        for i in range(100):
            prop = prop.setter(None)
        self.assertIsNone(prop.fset)
        for i in range(100):
            prop = prop.deleter(None)
        self.assertIsNone(prop.fdel)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    def test_property_name(self):
        # [FUNCTION DOC] Checks property __name__: taken from the getter, or setter when there
        # is no getter named via the class attribute; can be set (including to None);
        # raises AttributeError when no getter name exists; and an error raised while
        # looking up the getter's name propagates until a name is assigned.
        def getter(self):
            # [FUNCTION DOC] HELPER: Getter returning 42.
            return 42

        def setter(self, value):
            # [FUNCTION DOC] HELPER: Setter that does nothing.
            pass

        class A:
            # [CLASS DOC] HELPER: Class with properties made via decorator, setter-rename, and
            # direct property calls with getter only or setter only.
            @property
            def foo(self):
                # [FUNCTION DOC] HELPER: Getter returning 1.
                return 1

            @foo.setter
            def oof(self, value):
                # [FUNCTION DOC] HELPER: Setter, bound to the name oof; does nothing.
                pass

            bar = property(getter)
            baz = property(None, setter)

        self.assertEqual(A.foo.__name__, 'foo')
        self.assertEqual(A.oof.__name__, 'oof')
        self.assertEqual(A.bar.__name__, 'bar')
        self.assertEqual(A.baz.__name__, 'baz')

        A.quux = property(getter)
        self.assertEqual(A.quux.__name__, 'getter')
        A.quux.__name__ = 'myquux'
        self.assertEqual(A.quux.__name__, 'myquux')
        self.assertEqual(A.bar.__name__, 'bar')  # not affected
        A.quux.__name__ = None
        self.assertIsNone(A.quux.__name__)

        with self.assertRaisesRegex(
            AttributeError, "'property' object has no attribute '__name__'"
        ):
            property(None, setter).__name__

        with self.assertRaisesRegex(
            AttributeError, "'property' object has no attribute '__name__'"
        ):
            property(1).__name__

        class Err:
            # [CLASS DOC] HELPER: Object whose attribute lookup always raises RuntimeError.
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError with message fail.
                raise RuntimeError('fail')

        p = property(Err())
        with self.assertRaisesRegex(RuntimeError, 'fail'):
            p.__name__

        p.__name__ = 'not_fail'
        self.assertEqual(p.__name__, 'not_fail')

    def test_property_set_name_incorrect_args(self):
        # [FUNCTION DOC] Checks __set_name__ called with 0, 1 or 3 positional args raises
        # TypeError with a message about taking 2 positional arguments.
        p = property()

        for i in (0, 1, 3):
            with self.assertRaisesRegex(
                TypeError,
                fr'^__set_name__\(\) takes 2 positional arguments but {i} were given$'
            ):
                p.__set_name__(*([0] * i))

    def test_property_setname_on_property_subclass(self):
        # [FUNCTION DOC] Regression test for cpython issue 100942: calling __set_name__ and then
        # getter on a property whose type's __new__ returned a non-property must not crash.
        # https://github.com/python/cpython/issues/100942
        # Copy was setting the name field without first
        # verifying that the copy was an actual property
        # instance.  As a result, the code below was
        # causing a segfault.

        class pro(property):
            # [CLASS DOC] HELPER: property subclass whose __new__ returns a string instead of an
            # instance.
            def __new__(typ, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Ignores its arguments and returns the string abcdef.
                return "abcdef"

        class A:
            # [CLASS DOC] HELPER: Empty class used as owner argument for __set_name__.
            pass

        p = property.__new__(pro)
        p.__set_name__(A, 1)
        np = p.getter(lambda self: 1)

# Issue 5890: subclasses of property do not preserve method __doc__ strings
class PropertySub(property):
    # [CLASS DOC] HELPER: Property subclass with a docstring and an instance __dict__.
    """This is a subclass of property"""

class PropertySubWoDoc(property):
    # [CLASS DOC] HELPER: Property subclass without a docstring.
    pass

class PropertySubSlots(property):
    # [CLASS DOC] HELPER: Property subclass with empty __slots__ (no instance dict).
    """This is a subclass of property that defines __slots__"""
    __slots__ = ()

class PropertySubclassTests(CPythonTestCase):
    # [CLASS DOC] Test case for docstring handling of property subclasses (issue 5890,
    # 41287, 98963): how docs are copied from getters, or dropped when slots forbid it.

    @support.requires_docstrings
    def test_slots_docstring_copy_exception(self):
        # [FUNCTION DOC] Checks that using a slotted property subclass as a decorator on a
        # function with a docstring raises AttributeError, since the doc cannot be stored.
        # A special case error that we preserve despite the GH-98963 behavior
        # that would otherwise silently ignore this error.
        # This came from commit b18500d39d791c879e9904ebac293402b4a7cd34
        # as part of https://bugs.python.org/issue5890 which allowed docs to
        # be set via property subclasses in the first place.
        with self.assertRaises(AttributeError):
            class Foo(object):
                # [CLASS DOC] HELPER: Class body expected to raise AttributeError on definition.
                @PropertySubSlots
                def spam(self):
                    # [FUNCTION DOC] HELPER: Documented getter returning 1.
                    """Trying to copy this docstring will raise an exception"""
                    return 1

    def test_property_with_slots_no_docstring(self):
        # [FUNCTION DOC] Checks a slotted property subclass without a doc slot and getters without
        # docstrings leaves __doc__ as None and raises nothing.
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            # [CLASS DOC] HELPER: property subclass with only a foo slot, no doc slot.
            __slots__ = ("foo",)

        p = slotted_prop()  # no AttributeError
        self.assertIsNone(getattr(p, "__doc__", None))

        def undocumented_getter():
            # [FUNCTION DOC] HELPER: Getter without docstring, returns 4.
            return 4

        p = slotted_prop(undocumented_getter)  # New in 3.12: no AttributeError
        self.assertIsNone(getattr(p, "__doc__", None))

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_docstring_silently_dropped(self):
        # [FUNCTION DOC] Checks that a doc argument is silently dropped (doc is None) for a
        # slotted property without a doc slot, but a getter with a docstring raises
        # AttributeError.
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            # [CLASS DOC] HELPER: property subclass with only a foo slot.
            __slots__ = ("foo",)

        p = slotted_prop(doc="what's up")  # no AttributeError
        self.assertIsNone(p.__doc__)

        def documented_getter():
            # [FUNCTION DOC] HELPER: Getter with a docstring, returns 4.
            """getter doc."""
            return 4

        # Historical behavior: A docstring from a getter always raises.
        # (matches test_slots_docstring_copy_exception above).
        with self.assertRaises(AttributeError):
            p = slotted_prop(documented_getter)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_with_slots_and_doc_slot_docstring_present(self):
        # [FUNCTION DOC] Checks that when the slotted subclass has a doc slot, the doc argument
        # and the getter's docstring are stored and visible via __doc__.
        # https://github.com/python/cpython/issues/98963#issuecomment-1574413319
        class slotted_prop(property):
            # [CLASS DOC] HELPER: property subclass with foo and __doc__ slots.
            __slots__ = ("foo", "__doc__")

        p = slotted_prop(doc="what's up")
        self.assertEqual("what's up", p.__doc__)  # new in 3.12: This gets set.

        def documented_getter():
            # [FUNCTION DOC] HELPER: Getter with a docstring, returns 4.
            """what's up getter doc?"""
            return 4

        p = slotted_prop(documented_getter)
        self.assertEqual("what's up getter doc?", p.__doc__)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_issue41287(self):
        # [FUNCTION DOC] Regression test for issue 41287: for property and its subclasses the
        # explicit doc argument wins over the getter docstring, the getter docstring is used
        # when doc is None, and the result is None if neither exists.

        self.assertEqual(PropertySub.__doc__, "This is a subclass of property",
                         "Docstring of `property` subclass is ignored")

        doc = PropertySub(None, None, None, "issue 41287 is fixed").__doc__
        self.assertEqual(doc, "issue 41287 is fixed",
                         "Subclasses of `property` ignores `doc` constructor argument")

        def getter(x):
            # [FUNCTION DOC] HELPER: Getter with a docstring; returns None.
            """Getter docstring"""

        def getter_wo_doc(x):
            # [FUNCTION DOC] HELPER: Getter without a docstring; returns None.
            pass

        for ps in property, PropertySub, PropertySubWoDoc:
            doc = ps(getter, None, None, "issue 41287 is fixed").__doc__
            self.assertEqual(doc, "issue 41287 is fixed",
                             "Getter overrides explicit property docstring (%s)" % ps.__name__)

            doc = ps(getter, None, None, None).__doc__
            self.assertEqual(doc, "Getter docstring", "Getter docstring is not picked-up (%s)" % ps.__name__)

            doc = ps(getter_wo_doc, None, None, "issue 41287 is fixed").__doc__
            self.assertEqual(doc, "issue 41287 is fixed",
                             "Getter overrides explicit property docstring (%s)" % ps.__name__)

            doc = ps(getter_wo_doc, None, None, None).__doc__
            self.assertIsNone(doc, "Property class doc appears in instance __doc__ (%s)" % ps.__name__)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_docstring_copy(self):
        # [FUNCTION DOC] Checks that decorating a documented function with a property subclass
        # copies its docstring to the property.
        class Foo(object):
            # [CLASS DOC] HELPER: Holder of a PropertySub-decorated documented method.
            @PropertySub
            def spam(self):
                # [FUNCTION DOC] HELPER: Documented getter returning 1.
                """spam wrapped in property subclass"""
                return 1
        self.assertEqual(
            Foo.spam.__doc__,
            "spam wrapped in property subclass")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_docstring_copy2(self):
        # [FUNCTION DOC] Checks the five docstring-preservation cases when copying a property
        # with a new getter: explicit doc kept, getter doc taken if none given, and a doc
        # assigned after construction survives only when the old getter had no docstring.
        """
        Property tries to provide the best docstring it finds for its instances.
        If a user-provided docstring is available, it is preserved on copies.
        If no docstring is available during property creation, the property
        will utilize the docstring from the getter if available.
        """
        def getter1(self):
            # [FUNCTION DOC] HELPER: Getter without docstring, returns 1.
            return 1
        def getter2(self):
            # [FUNCTION DOC] HELPER: Getter with doc "doc 2", returns 2.
            """doc 2"""
            return 2
        def getter3(self):
            # [FUNCTION DOC] HELPER: Getter with doc "doc 3", returns 3.
            """doc 3"""
            return 3

        # Case-1: user-provided doc is preserved in copies
        #         of property with undocumented getter
        p = property(getter1, None, None, "doc-A")

        p2 = p.getter(getter2)
        self.assertEqual(p.__doc__, "doc-A")
        self.assertEqual(p2.__doc__, "doc-A")

        # Case-2: user-provided doc is preserved in copies
        #         of property with documented getter
        p = property(getter2, None, None, "doc-A")

        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "doc-A")
        self.assertEqual(p2.__doc__, "doc-A")

        # Case-3: with no user-provided doc new getter doc
        #         takes precedence
        p = property(getter2, None, None, None)

        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "doc 2")
        self.assertEqual(p2.__doc__, "doc 3")

        # Case-4: A user-provided doc is assigned after property construction
        #         with documented getter. The doc IS NOT preserved.
        #         It's an odd behaviour, but it's a strange enough
        #         use case with no easy solution.
        p = property(getter2, None, None, None)
        p.__doc__ = "user"
        p2 = p.getter(getter3)
        self.assertEqual(p.__doc__, "user")
        self.assertEqual(p2.__doc__, "doc 3")

        # Case-5: A user-provided doc is assigned after property construction
        #         with UNdocumented getter. The doc IS preserved.
        p = property(getter1, None, None, None)
        p.__doc__ = "user"
        p2 = p.getter(getter2)
        self.assertEqual(p.__doc__, "user")
        self.assertEqual(p2.__doc__, "user")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_prefer_explicit_doc(self):
        # [FUNCTION DOC] Regression test for issue 25757: an explicit doc argument is preferred
        # on property and subclass, including when fget is passed as keyword.
        # Issue 25757: subclasses of property lose docstring
        self.assertEqual(property(doc="explicit doc").__doc__, "explicit doc")
        self.assertEqual(PropertySub(doc="explicit doc").__doc__, "explicit doc")

        class Foo:
            # [CLASS DOC] HELPER: Holds PropertySub properties created with explicit doc arguments.
            spam = PropertySub(doc="spam explicit doc")

            @spam.getter
            def spam(self):
                # [FUNCTION DOC] HELPER: Getter whose docstring is ignored because a doc was given.
                """ignored as doc already set"""
                return 1

            def _stuff_getter(self):
                # [FUNCTION DOC] HELPER: Getter used as fget; its docstring is ignored
                # because an explicit doc was given.
                """ignored as doc set directly"""
            stuff = PropertySub(doc="stuff doc argument", fget=_stuff_getter)

        #self.assertEqual(Foo.spam.__doc__, "spam explicit doc")
        self.assertEqual(Foo.stuff.__doc__, "stuff doc argument")

    def test_property_no_doc_on_getter(self):
        # [FUNCTION DOC] Checks that when the getter's __doc__ access raises AttributeError the
        # doc of both property and subclass is None.
        # If a property's getter has no __doc__ then the property's doc should
        # be None; test that this is consistent with subclasses as well; see
        # GH-2487
        class NoDoc:
            # [CLASS DOC] HELPER: Object whose __doc__ is a property that raises AttributeError.
            @property
            def __doc__(self):
                # [FUNCTION DOC] HELPER: Always raises AttributeError.
                raise AttributeError

        self.assertEqual(property(NoDoc()).__doc__, None)
        self.assertEqual(PropertySub(NoDoc()).__doc__, None)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_setter_copies_getter_docstring(self):
        # [FUNCTION DOC] Checks that using a setter on a property subclass keeps the getter's
        # docstring (also in a derived class), and that set/get values are correct (skipped
        # with -OO).
        class Foo(object):
            # [CLASS DOC] HELPER: Class with a PropertySub property having a getter
            # and a setter over _spam.
            # [FUNCTION DOC] HELPER: Sets _spam to 1. (Single-line def, so this block
            # sits above it.)
            def __init__(self): self._spam = 1
            @PropertySub
            def spam(self):
                # [FUNCTION DOC] HELPER: Getter returning _spam.
                """spam wrapped in property subclass"""
                return self._spam
            @spam.setter
            def spam(self, value):
                # [FUNCTION DOC] HELPER: Setter storing the value in _spam; docstring ignored.
                """this docstring is ignored"""
                self._spam = value
        foo = Foo()
        self.assertEqual(foo.spam, 1)
        foo.spam = 2
        self.assertEqual(foo.spam, 2)
        self.assertEqual(
            Foo.spam.__doc__,
            "spam wrapped in property subclass")
        class FooSub(Foo):
            # [CLASS DOC] HELPER: Subclass whose setter ignores the value and stores
            # the string eggs.
            @Foo.spam.setter
            def spam(self, value):
                # [FUNCTION DOC] HELPER: Setter that always stores the string eggs in _spam.
                """another ignored docstring"""
                self._spam = 'eggs'
        foosub = FooSub()
        self.assertEqual(foosub.spam, 1)
        foosub.spam = 7
        self.assertEqual(foosub.spam, 'eggs')
        self.assertEqual(
            FooSub.spam.__doc__,
            "spam wrapped in property subclass")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_property_new_getter_new_docstring(self):
        # [FUNCTION DOC] Checks that replacing the getter of a PropertySub property, in the same
        # class and in a subclass, yields the new getter's docstring (skipped with -OO).
        # KNOWN ISSUE: the final assertion re-checks Foo rather than
        # Foo2, so the subclass case is untested.

        class Foo(object):
            # [CLASS DOC] HELPER: Redefines the getter of a PropertySub property in the same class.
            @PropertySub
            def spam(self):
                # [FUNCTION DOC] HELPER: Original getter returning 1.
                """a docstring"""
                return 1
            @spam.getter
            def spam(self):
                # [FUNCTION DOC] HELPER: Replacement getter returning 2.
                """a new docstring"""
                return 2
        self.assertEqual(Foo.spam.__doc__, "a new docstring")
        class FooBase(object):
            # [CLASS DOC] HELPER: Base holding a PropertySub property.
            @PropertySub
            def spam(self):
                # [FUNCTION DOC] HELPER: Getter returning 1.
                """a docstring"""
                return 1
        class Foo2(FooBase):
            # [CLASS DOC] HELPER: Subclass replacing the inherited getter.
            @FooBase.spam.getter
            def spam(self):
                # [FUNCTION DOC] HELPER: Replacement getter returning 2.
                """a new docstring"""
                return 2
        self.assertEqual(Foo.spam.__doc__, "a new docstring")


class _PropertyUnreachableAttribute:
    # [CLASS DOC] HELPER: Mixin for tests of properties lacking an accessor. Subclasses set
    # msg_format (a template with one slot) and cls (a class with a foo property); the
    # setUpClass hook creates the instance obj. Tests expect AttributeError messages.
    msg_format = None
    obj = None
    cls = None

    def _format_exc_msg(self, msg):
        # [FUNCTION DOC] HELPER: Return msg_format filled with the given text.
        return self.msg_format.format(msg)

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Call the parent setUpClass and create the shared instance obj.
        super().setUpClass()
        cls.obj = cls.cls()

    def test_get_property(self):
        # [FUNCTION DOC] Reading the property without a getter must raise
        # AttributeError with text "has no getter".
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no getter")):
            self.obj.foo

    def test_set_property(self):
        # [FUNCTION DOC] Setting the property without a setter must raise
        # AttributeError with text "has no setter".
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no setter")):
            self.obj.foo = None

    def test_del_property(self):
        # [FUNCTION DOC] Deleting the property without a deleter must raise
        # AttributeError with text "has no deleter".
        with self.assertRaisesRegex(AttributeError, self._format_exc_msg("has no deleter")):
            del self.obj.foo


class PropertyUnreachableAttributeWithName(_PropertyUnreachableAttribute, CPythonTestCase):
    # [CLASS DOC] Runs the unreachable-accessor tests where the property was named by the
    # class body, so messages include the property name foo.
    msg_format = r"^property 'foo' of 'PropertyUnreachableAttributeWithName\.cls' object {}$"

    class cls:
        # [CLASS DOC] HELPER: Holds a property foo with no accessors.
        foo = property()


class PropertyUnreachableAttributeNoName(_PropertyUnreachableAttribute, CPythonTestCase):
    # [CLASS DOC] Runs the unreachable-accessor tests where the property was attached after
    # class creation, so messages omit the property name.
    msg_format = r"^property of 'PropertyUnreachableAttributeNoName\.cls' object {}$"

    class cls:
        # [CLASS DOC] HELPER: Empty class; the property is attached to it just after the body.
        pass

    cls.foo = property()


if __name__ == '__main__':
    run_tests()
