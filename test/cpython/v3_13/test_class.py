# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_class suite: checks that instances
# of user classes dispatch operators and special methods (arithmetic, comparison, container,
# attribute access, hash/str/repr, construction errors) to the expected dunder methods. A
# synthetic AllTests class records each dunder call in the module-level list callLst.
# Import side effects: builds AllTests via exec of generated method source, then deletes
# the temporary names. Also defines inline-values tests, skipped under Dynamo.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_class.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo

# ======= END DYNAMO PATCH =======

"Test the functionality of Python classes implementing operators."

import unittest
from test import support
from test.support import cpython_only, import_helper, script_helper

testmeths = [

# Binary operations
    "add",
    "radd",
    "sub",
    "rsub",
    "mul",
    "rmul",
    "matmul",
    "rmatmul",
    "truediv",
    "rtruediv",
    "floordiv",
    "rfloordiv",
    "mod",
    "rmod",
    "divmod",
    "rdivmod",
    "pow",
    "rpow",
    "rshift",
    "rrshift",
    "lshift",
    "rlshift",
    "and",
    "rand",
    "or",
    "ror",
    "xor",
    "rxor",

# List/dict operations
    "contains",
    "getitem",
    "setitem",
    "delitem",

# Unary operations
    "neg",
    "pos",
    "abs",

# generic operations
    "init",
    ]

# These need to return something other than None
#    "hash",
#    "str",
#    "repr",
#    "int",
#    "float",

# These are separate because they can influence the test of other methods.
#    "getattr",
#    "setattr",
#    "delattr",

callLst = []
def trackCall(f):
    # [FUNCTION DOC] HELPER: decorator factory: wraps function f so each call appends a tuple
    # of the function name and positional args to the global callLst, then calls f and returns
    # its result. Keyword arguments are forwarded but not recorded.
    def track(*args, **kwargs):
        # [FUNCTION DOC] HELPER: wrapper created by trackCall; records (name, args) in callLst
        # and delegates.
        callLst.append((f.__name__, args))
        return f(*args, **kwargs)
    return track

statictests = """
@trackCall
def __hash__(self, *args):
    return hash(id(self))

@trackCall
def __str__(self, *args):
    return "AllTests"

@trackCall
def __repr__(self, *args):
    return "AllTests"

@trackCall
def __int__(self, *args):
    return 1

@trackCall
def __index__(self, *args):
    return 1

@trackCall
def __float__(self, *args):
    return 1.0

@trackCall
def __eq__(self, *args):
    return True

@trackCall
def __ne__(self, *args):
    return False

@trackCall
def __lt__(self, *args):
    return False

@trackCall
def __le__(self, *args):
    return True

@trackCall
def __gt__(self, *args):
    return False

@trackCall
def __ge__(self, *args):
    return True
"""

# Synthesize all the other AllTests methods from the names in testmeths.

method_template = """\
@trackCall
def __%s__(self, *args):
    pass
"""

d = {}
exec(statictests, globals(), d)
for method in testmeths:
    exec(method_template % method, globals(), d)
AllTests = type("AllTests", (object,), d)
del d, statictests, method, method_template

class ClassTests(CPythonTestCase):
    # [CLASS DOC] Tests that operators and builtins invoke the matching dunder methods
    # on AllTests. Each test clears callLst, triggers an operation, and compares the
    # recorded calls.
    def setUp(self):
        # [FUNCTION DOC] HELPER: calls the parent setUp, then empties the global callLst
        # in place.
        super().setUp()
        callLst[:] = []

    def assertCallStack(self, expected_calls):
        # [FUNCTION DOC] HELPER: fails the test unless a copy of callLst equals
        # expected_calls; the failure message shows both the expected and actual call lists.
        # The copy is taken first because the comparison itself could add entries to callLst.
        actualCallList = callLst[:]  # need to copy because the comparison below will add
                                     # additional calls to callLst
        if expected_calls != actualCallList:
            self.fail("Expected call list:\n  %s\ndoes not match actual call list\n  %s" %
                      (expected_calls, actualCallList))

    def testInit(self):
        # [FUNCTION DOC] Constructing AllTests must record exactly one call to __init__ with
        # the new instance.
        foo = AllTests()
        self.assertCallStack([("__init__", (foo,))])

    def testBinaryOps(self):
        # [FUNCTION DOC] For every binary operator (arithmetic, shift, bitwise, matmul,
        # divmod, pow), checks the forward and reflected form dispatches to the matching
        # dunder with args (testme, 1).
        testme = AllTests()
        # Binary operations

        callLst[:] = []
        testme + 1
        self.assertCallStack([("__add__", (testme, 1))])

        callLst[:] = []
        1 + testme
        self.assertCallStack([("__radd__", (testme, 1))])

        callLst[:] = []
        testme - 1
        self.assertCallStack([("__sub__", (testme, 1))])

        callLst[:] = []
        1 - testme
        self.assertCallStack([("__rsub__", (testme, 1))])

        callLst[:] = []
        testme * 1
        self.assertCallStack([("__mul__", (testme, 1))])

        callLst[:] = []
        1 * testme
        self.assertCallStack([("__rmul__", (testme, 1))])

        callLst[:] = []
        testme @ 1
        self.assertCallStack([("__matmul__", (testme, 1))])

        callLst[:] = []
        1 @ testme
        self.assertCallStack([("__rmatmul__", (testme, 1))])

        callLst[:] = []
        testme / 1
        self.assertCallStack([("__truediv__", (testme, 1))])


        callLst[:] = []
        1 / testme
        self.assertCallStack([("__rtruediv__", (testme, 1))])

        callLst[:] = []
        testme // 1
        self.assertCallStack([("__floordiv__", (testme, 1))])


        callLst[:] = []
        1 // testme
        self.assertCallStack([("__rfloordiv__", (testme, 1))])

        callLst[:] = []
        testme % 1
        self.assertCallStack([("__mod__", (testme, 1))])

        callLst[:] = []
        1 % testme
        self.assertCallStack([("__rmod__", (testme, 1))])


        callLst[:] = []
        divmod(testme,1)
        self.assertCallStack([("__divmod__", (testme, 1))])

        callLst[:] = []
        divmod(1, testme)
        self.assertCallStack([("__rdivmod__", (testme, 1))])

        callLst[:] = []
        testme ** 1
        self.assertCallStack([("__pow__", (testme, 1))])

        callLst[:] = []
        1 ** testme
        self.assertCallStack([("__rpow__", (testme, 1))])

        callLst[:] = []
        testme >> 1
        self.assertCallStack([("__rshift__", (testme, 1))])

        callLst[:] = []
        1 >> testme
        self.assertCallStack([("__rrshift__", (testme, 1))])

        callLst[:] = []
        testme << 1
        self.assertCallStack([("__lshift__", (testme, 1))])

        callLst[:] = []
        1 << testme
        self.assertCallStack([("__rlshift__", (testme, 1))])

        callLst[:] = []
        testme & 1
        self.assertCallStack([("__and__", (testme, 1))])

        callLst[:] = []
        1 & testme
        self.assertCallStack([("__rand__", (testme, 1))])

        callLst[:] = []
        testme | 1
        self.assertCallStack([("__or__", (testme, 1))])

        callLst[:] = []
        1 | testme
        self.assertCallStack([("__ror__", (testme, 1))])

        callLst[:] = []
        testme ^ 1
        self.assertCallStack([("__xor__", (testme, 1))])

        callLst[:] = []
        1 ^ testme
        self.assertCallStack([("__rxor__", (testme, 1))])

    def testListAndDictOps(self):
        # [FUNCTION DOC] Checks container protocol: membership test on a class without
        # __contains__ raises TypeError; on AllTests, in/getitem/setitem/delitem (including
        # slice and extended-slice keys) record the matching dunder calls with the expected
        # key arguments.
        testme = AllTests()

        # List/dict operations

        class Empty: pass

        try:
            1 in Empty()
            self.fail('failed, should have raised TypeError')
        except TypeError:
            pass

        callLst[:] = []
        1 in testme
        self.assertCallStack([('__contains__', (testme, 1))])

        callLst[:] = []
        testme[1]
        self.assertCallStack([('__getitem__', (testme, 1))])

        callLst[:] = []
        testme[1] = 1
        self.assertCallStack([('__setitem__', (testme, 1, 1))])

        callLst[:] = []
        del testme[1]
        self.assertCallStack([('__delitem__', (testme, 1))])

        callLst[:] = []
        testme[:42]
        self.assertCallStack([('__getitem__', (testme, slice(None, 42)))])

        callLst[:] = []
        testme[:42] = "The Answer"
        self.assertCallStack([('__setitem__', (testme, slice(None, 42),
                                               "The Answer"))])

        callLst[:] = []
        del testme[:42]
        self.assertCallStack([('__delitem__', (testme, slice(None, 42)))])

        callLst[:] = []
        testme[2:1024:10]
        self.assertCallStack([('__getitem__', (testme, slice(2, 1024, 10)))])

        callLst[:] = []
        testme[2:1024:10] = "A lot"
        self.assertCallStack([('__setitem__', (testme, slice(2, 1024, 10),
                                                                    "A lot"))])
        callLst[:] = []
        del testme[2:1024:10]
        self.assertCallStack([('__delitem__', (testme, slice(2, 1024, 10)))])

        callLst[:] = []
        testme[:42, ..., :24:, 24, 100]
        self.assertCallStack([('__getitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100)))])
        callLst[:] = []
        testme[:42, ..., :24:, 24, 100] = "Strange"
        self.assertCallStack([('__setitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100), "Strange"))])
        callLst[:] = []
        del testme[:42, ..., :24:, 24, 100]
        self.assertCallStack([('__delitem__', (testme, (slice(None, 42, None),
                                                        Ellipsis,
                                                        slice(None, 24, None),
                                                        24, 100)))])

    def testUnaryOps(self):
        # [FUNCTION DOC] Checks unary neg/pos/abs and the conversions int, float, oct and hex
        # dispatch to __neg__, __pos__, __abs__, __int__, __float__ and __index__ (oct and hex
        # use __index__).
        testme = AllTests()

        callLst[:] = []
        -testme
        self.assertCallStack([('__neg__', (testme,))])
        callLst[:] = []
        +testme
        self.assertCallStack([('__pos__', (testme,))])
        callLst[:] = []
        abs(testme)
        self.assertCallStack([('__abs__', (testme,))])
        callLst[:] = []
        int(testme)
        self.assertCallStack([('__int__', (testme,))])
        callLst[:] = []
        float(testme)
        self.assertCallStack([('__float__', (testme,))])
        callLst[:] = []
        oct(testme)
        self.assertCallStack([('__index__', (testme,))])
        callLst[:] = []
        hex(testme)
        self.assertCallStack([('__index__', (testme,))])


    def testMisc(self):
        # [FUNCTION DOC] Checks hash, repr, str and the comparison operators (including
        # reflected forms, where 1 < testme calls __gt__) dispatch to the matching
        # dunder methods.
        testme = AllTests()

        callLst[:] = []
        hash(testme)
        self.assertCallStack([('__hash__', (testme,))])

        callLst[:] = []
        repr(testme)
        self.assertCallStack([('__repr__', (testme,))])

        callLst[:] = []
        str(testme)
        self.assertCallStack([('__str__', (testme,))])

        callLst[:] = []
        testme == 1
        self.assertCallStack([('__eq__', (testme, 1))])

        callLst[:] = []
        testme < 1
        self.assertCallStack([('__lt__', (testme, 1))])

        callLst[:] = []
        testme > 1
        self.assertCallStack([('__gt__', (testme, 1))])

        callLst[:] = []
        testme != 1
        self.assertCallStack([('__ne__', (testme, 1))])

        callLst[:] = []
        1 == testme
        self.assertCallStack([('__eq__', (1, testme))])

        callLst[:] = []
        1 < testme
        self.assertCallStack([('__gt__', (1, testme))])

        callLst[:] = []
        1 > testme
        self.assertCallStack([('__lt__', (1, testme))])

        callLst[:] = []
        1 != testme
        self.assertCallStack([('__ne__', (1, testme))])


    def testGetSetAndDel(self):
        # [FUNCTION DOC] Checks that attribute get/set/delete on a subclass defining
        # __getattr__, __setattr__ and __delattr__ calls those hooks with the instance and
        # attribute name.
        # Interfering tests
        class ExtraTests(AllTests):
            # [CLASS DOC] HELPER: AllTests subclass whose attribute hooks return a fixed value
            # or do nothing.
            @trackCall
            def __getattr__(self, *args):
                # [FUNCTION DOC] HELPER: tracked __getattr__ hook; returns the constant
                # string SomeVal.
                return "SomeVal"

            @trackCall
            def __setattr__(self, *args):
                # [FUNCTION DOC] HELPER: tracked __setattr__ hook; ignores the assignment
                # (nothing is stored).
                pass

            @trackCall
            def __delattr__(self, *args):
                # [FUNCTION DOC] HELPER: tracked __delattr__ hook; does nothing.
                pass

        testme = ExtraTests()

        callLst[:] = []
        testme.spam
        self.assertCallStack([('__getattr__', (testme, "spam"))])

        callLst[:] = []
        testme.eggs = "spam, spam, spam and ham"
        self.assertCallStack([('__setattr__', (testme, "eggs",
                                               "spam, spam, spam and ham"))])

        callLst[:] = []
        del testme.cardinal
        self.assertCallStack([('__delattr__', (testme, "cardinal"))])

    def testHasAttrString(self):
        # [FUNCTION DOC] Imports _testlimitedcapi (skips if absent) and checks
        # object_hasattrstring returns 1 for an existing attribute, 0 for a missing one, and
        # leaves no pending exception.
        import sys
        from test.support import import_helper
        _testlimitedcapi = import_helper.import_module('_testlimitedcapi')

        class A:
            # [CLASS DOC] HELPER: plain class whose instances get an attribute named attr set
            # to 1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: sets the attribute attr to 1.
                self.attr = 1

        a = A()
        self.assertEqual(_testlimitedcapi.object_hasattrstring(a, b"attr"), 1)
        self.assertEqual(_testlimitedcapi.object_hasattrstring(a, b"noattr"), 0)
        self.assertIsNone(sys.exception())

    def testDel(self):
        # [FUNCTION DOC] Checks that deleting the last reference to an instance runs __del__
        # (after a forced garbage collection) by observing a message appended to a local list.
        x = []

        class DelTest:
            # [CLASS DOC] HELPER: class whose __del__ appends a marker string to the
            # enclosing list.
            def __del__(self):
                # [FUNCTION DOC] HELPER: destructor hook; appends the marker string to the
                # enclosing list x.
                x.append("crab people, crab people")
        testme = DelTest()
        del testme
        import gc
        gc.collect()
        self.assertEqual(["crab people, crab people"], x)

    def testBadTypeReturned(self):
        # [FUNCTION DOC] Checks that float, complex, str, repr, bytes, bin, oct, hex, bool
        # and a helper that indexes a list all raise TypeError when the dunder method
        # returns None.
        # return values of some method are type-checked
        class BadTypeClass:
            # [CLASS DOC] HELPER: class whose conversion dunders all return None (an invalid
            # result type).
            def __int__(self):
                # [FUNCTION DOC] HELPER: returns None; aliased to the other conversion dunders
                # of the class.
                return None
            __float__ = __int__
            __complex__ = __int__
            __str__ = __int__
            __repr__ = __int__
            __bytes__ = __int__
            __bool__ = __int__
            __index__ = __int__
        def index(x):
            # [FUNCTION DOC] HELPER: indexes an empty list with x, triggering __index__ on x;
            # always fails or raises.
            return [][x]

        for f in [float, complex, str, repr, bytes, bin, oct, hex, bool, index]:
            self.assertRaises(TypeError, f, BadTypeClass())

    def testHashStuff(self):
        # [FUNCTION DOC] Checks hash works on a plain class and raises TypeError for a class
        # that defines __eq__ without __hash__.
        # Test correct errors from hash() on objects with comparisons but
        #  no __hash__

        class C0:
            # [CLASS DOC] HELPER: plain class with default hashing.
            pass

        hash(C0()) # This should work; the next two should raise TypeError

        class C2:
            # [CLASS DOC] HELPER: class defining only __eq__ (a one-line def), making its
            # instances unhashable.
            def __eq__(self, other): return 1

        self.assertRaises(TypeError, hash, C2())

    def testPredefinedAttrs(self):
        # [FUNCTION DOC] Checks that standard object dunder methods are callable on
        # object, an object instance, a custom class and its instance; that many others
        # (numeric, container, iterator, in-place) are absent; and that __call__ is not
        # found on instances.
        o = object()

        class Custom:
            # [CLASS DOC] HELPER: empty class used to probe inherited attributes.
            pass

        c = Custom()

        methods = (
            '__class__', '__delattr__', '__dir__', '__eq__', '__format__',
            '__ge__', '__getattribute__', '__getstate__', '__gt__', '__hash__',
            '__init__', '__init_subclass__', '__le__', '__lt__', '__ne__',
            '__new__', '__reduce__', '__reduce_ex__', '__repr__',
            '__setattr__', '__sizeof__', '__str__', '__subclasshook__'
        )
        for name in methods:
            with self.subTest(name):
                self.assertTrue(callable(getattr(object, name, None)))
                self.assertTrue(callable(getattr(o, name, None)))
                self.assertTrue(callable(getattr(Custom, name, None)))
                self.assertTrue(callable(getattr(c, name, None)))

        not_defined = [
            '__abs__', '__aenter__', '__aexit__', '__aiter__', '__anext__',
            '__await__', '__bool__', '__bytes__', '__ceil__',
            '__complex__', '__contains__', '__del__', '__delete__',
            '__delitem__', '__divmod__', '__enter__', '__exit__',
            '__float__', '__floor__', '__get__', '__getattr__', '__getitem__',
            '__index__', '__int__', '__invert__', '__iter__', '__len__',
            '__length_hint__', '__missing__', '__neg__', '__next__',
            '__objclass__', '__pos__', '__rdivmod__', '__reversed__',
            '__round__', '__set__', '__setitem__', '__trunc__'
        ]
        augment = (
            'add', 'and', 'floordiv', 'lshift', 'matmul', 'mod', 'mul', 'pow',
            'rshift', 'sub', 'truediv', 'xor'
        )
        not_defined.extend(map("__{}__".format, augment))
        not_defined.extend(map("__r{}__".format, augment))
        not_defined.extend(map("__i{}__".format, augment))
        for name in not_defined:
            with self.subTest(name):
                self.assertFalse(hasattr(object, name))
                self.assertFalse(hasattr(o, name))
                self.assertFalse(hasattr(Custom, name))
                self.assertFalse(hasattr(c, name))

        # __call__() is defined on the metaclass but not the class
        self.assertFalse(hasattr(o, "__call__"))
        self.assertFalse(hasattr(c, "__call__"))

    def testSFBug532646(self):
        # [FUNCTION DOC] Regression test: a class whose __call__ attribute is an instance of
        # itself must raise RecursionError when called instead of crashing.
        # Test for SF bug 532646

        class A:
            # [CLASS DOC] HELPER: empty class; its __call__ is later set to an instance
            # of itself.
            pass
        A.__call__ = A()
        a = A()

        try:
            a() # This should not segfault
        except RecursionError:
            pass
        else:
            self.fail("Failed to raise RecursionError")

    def testForExceptionsRaisedInInstanceGetattr2(self):
        # [FUNCTION DOC] Checks that AttributeError raised inside a property getter is
        # not masked: via attribute access, via __eq__ comparison and via __init__
        # during construction.
        # Tests for exceptions raised in instance_getattr2().

        def booh(self):
            # [FUNCTION DOC] HELPER: property getter that always raises AttributeError with
            # message booh.
            raise AttributeError("booh")

        class A:
            # [CLASS DOC] HELPER: class exposing attribute a as a property whose
            # getter raises.
            a = property(booh)
        try:
            A().a # Raised AttributeError: A instance has no attribute 'a'
        except AttributeError as x:
            if str(x) != "booh":
                self.fail("attribute error for A().a got masked: %s" % x)

        class E:
            # [CLASS DOC] HELPER: class whose __eq__ is a property whose getter
            # raises AttributeError.
            __eq__ = property(booh)
        E() == E() # In debug mode, caused a C-level assert() to fail

        class I:
            # [CLASS DOC] HELPER: class whose __init__ is a property whose getter
            # raises AttributeError.
            __init__ = property(booh)
        try:
            # In debug mode, printed XXX undetected error and
            #  raises AttributeError
            I()
        except AttributeError:
            pass
        else:
            self.fail("attribute error for I.__init__ got masked")

    def assertNotOrderable(self, a, b):
        # [FUNCTION DOC] HELPER: asserts that all four ordering comparisons between a and b
        # raise TypeError.
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a >= b

    def testHashComparisonOfMethods(self):
        # [FUNCTION DOC] Checks equality, inequality, hashing and non-orderability of bound
        # and unbound methods: bound methods compare by function and by the instance's __eq__,
        # and hashing a bound method hashes its function plus the instance identity.
        # Test comparison and hash of methods
        class A:
            # [CLASS DOC] HELPER: class with always-equal __eq__ and a __hash__ that
            # raises TypeError.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: stores the argument x on the instance.
                self.x = x
            def f(self):
                # [FUNCTION DOC] HELPER: no-op method used for method comparison.
                pass
            def g(self):
                # [FUNCTION DOC] HELPER: second no-op method used for method comparison.
                pass
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: equality hook; always returns True.
                return True
            def __hash__(self):
                # [FUNCTION DOC] HELPER: hash hook; always raises TypeError.
                raise TypeError
        class B(A):
            # [CLASS DOC] HELPER: empty subclass of A used to compare inherited
            # unbound methods.
            pass

        a1 = A(1)
        a2 = A(1)
        self.assertTrue(a1.f == a1.f)
        self.assertFalse(a1.f != a1.f)
        self.assertFalse(a1.f == a2.f)
        self.assertTrue(a1.f != a2.f)
        self.assertFalse(a1.f == a1.g)
        self.assertTrue(a1.f != a1.g)
        self.assertNotOrderable(a1.f, a1.f)
        self.assertEqual(hash(a1.f), hash(a1.f))

        self.assertFalse(A.f == a1.f)
        self.assertTrue(A.f != a1.f)
        self.assertFalse(A.f == A.g)
        self.assertTrue(A.f != A.g)
        self.assertTrue(B.f == A.f)
        self.assertFalse(B.f != A.f)
        self.assertNotOrderable(A.f, A.f)
        self.assertEqual(hash(B.f), hash(A.f))

        # the following triggers a SystemError in 2.4
        a = A(hash(A.f)^(-1))
        hash(a.f)

    def testSetattrWrapperNameIntern(self):
        # [FUNCTION DOC] Regression test: setting a dunder attribute on a class via a non-
        # interned name string must still install the operator (A() + 1 works); deleting it
        # removes it again.
        # Issue #25794: __setattr__ should intern the attribute name
        class A:
            # [CLASS DOC] HELPER: empty class receiving a dynamically assigned __add__.
            pass

        def add(self, other):
            # [FUNCTION DOC] HELPER: replacement __add__; returns the string summa.
            return 'summa'

        name = str(b'__add__', 'ascii')  # shouldn't be optimized
        self.assertIsNot(name, '__add__')  # not interned
        type.__setattr__(A, name, add)
        self.assertEqual(A() + 1, 'summa')

        name2 = str(b'__add__', 'ascii')
        self.assertIsNot(name2, '__add__')
        self.assertIsNot(name2, name)
        type.__delattr__(A, name2)
        with self.assertRaises(TypeError):
            A() + 1

    def testSetattrNonStringName(self):
        # [FUNCTION DOC] Checks that setting a class attribute with a bytes (non-str) name
        # raises TypeError.
        class A:
            # [CLASS DOC] HELPER: empty class used as the target of the failing assignment.
            pass

        with self.assertRaises(TypeError):
            type.__setattr__(A, b'x', None)

    def testTypeAttributeAccessErrorMessages(self):
        # [FUNCTION DOC] Checks the AttributeError message for reading or deleting a missing
        # attribute on a class object.
        class A:
            # [CLASS DOC] HELPER: empty class.
            pass

        error_msg = "type object 'A' has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            A.x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del A.x

    def testObjectAttributeAccessErrorMessages(self):
        # [FUNCTION DOC] Checks AttributeError messages for missing, read-only and slot-
        # restricted instance attributes (get, set and delete) on classes with and without
        # __dict__ or slots.
        class A:
            # [CLASS DOC] HELPER: empty class with a regular instance dict.
            pass
        class B:
            # [CLASS DOC] HELPER: class with slots restricted to z and a class attribute y.
            y = 0
            __slots__ = ('z',)
        class C:
            # [CLASS DOC] HELPER: slotted class whose __setattr__ maps assignments to z onto
            # slot y.
            __slots__ = ("y",)

            def __setattr__(self, name, value) -> None:
                # [FUNCTION DOC] HELPER: stores 1 in slot y when name is z, otherwise
                # delegates to the parent.
                if name == "z":
                    super().__setattr__("y", 1)
                else:
                    super().__setattr__(name, value)

        error_msg = "'A' object has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            A().x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del A().x

        error_msg = "'B' object has no attribute 'x'"
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().x
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().x
        with self.assertRaisesRegex(
            AttributeError,
            "'B' object has no attribute 'x' and no __dict__ for setting new attributes"
        ):
            B().x = 0
        with self.assertRaisesRegex(
            AttributeError,
            "'C' object has no attribute 'x'"
        ):
            C().x = 0

        error_msg = "'B' object attribute 'y' is read-only"
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().y
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().y = 0

        error_msg = 'z'
        with self.assertRaisesRegex(AttributeError, error_msg):
            B().z
        with self.assertRaisesRegex(AttributeError, error_msg):
            del B().z

    def testConstructorErrorMessages(self):
        # [FUNCTION DOC] Checks TypeError messages from constructing instances and calling
        # __new__ or __init__ directly with extra arguments, for classes with no overrides,
        # with both __init__ and __new__ overridden, and with only __init__ overridden.
        # bpo-31506: Improves the error message logic for object_new & object_init

        # Class without any method overrides
        class C:
            # [CLASS DOC] HELPER: class without any method overrides.
            pass

        error_msg = r'C.__init__\(\) takes exactly one argument \(the instance to initialize\)'

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            C(42)

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            C.__new__(C, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            C().__init__(42)

        with self.assertRaisesRegex(TypeError, r'C\(\) takes no arguments'):
            object.__new__(C, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__init__(C(), 42)

        # Class with both `__init__` & `__new__` method overridden
        class D:
            # [CLASS DOC] HELPER: class overriding both __new__ and __init__, forwarding
            # extra arguments.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: forwards all arguments to the parent __new__
                # (returns nothing).
                super().__new__(cls, *args, **kwargs)
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: forwards all arguments to the parent __init__.
                super().__init__(*args, **kwargs)

        error_msg =  r'object.__new__\(\) takes exactly one argument \(the type to instantiate\)'

        with self.assertRaisesRegex(TypeError, error_msg):
            D(42)

        with self.assertRaisesRegex(TypeError, error_msg):
            D.__new__(D, 42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__new__(D, 42)

        # Class that only overrides __init__
        class E:
            # [CLASS DOC] HELPER: class overriding only __init__, forwarding extra arguments.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: forwards all arguments to the parent __init__.
                super().__init__(*args, **kwargs)

        error_msg = r'object.__init__\(\) takes exactly one argument \(the instance to initialize\)'

        with self.assertRaisesRegex(TypeError, error_msg):
            E().__init__(42)

        with self.assertRaisesRegex(TypeError, error_msg):
            object.__init__(E(), 42)

    def testClassWithExtCall(self):
        # [FUNCTION DOC] Checks class statements using keyword unpacking and many positional
        # bases with a custom metaclass returning (bases, kwargs); verifies the bases tuple
        # and kwargs mapping.
        class Meta(int):
            # [CLASS DOC] HELPER: int-derived metaclass whose __new__ returns the bases and
            # keyword arguments.
            def __init__(*args, **kwargs):
                # [FUNCTION DOC] HELPER: no-op initializer.
                pass

            def __new__(cls, name, bases, attrs, **kwargs):
                # [FUNCTION DOC] HELPER: returns a tuple of the bases and the extra keyword
                # arguments instead of a class.
                return bases, kwargs

        d = {'metaclass': Meta}

        class A(**d): pass
        self.assertEqual(A, ((), {}))
        class A(0, 1, 2, 3, 4, 5, 6, 7, **d): pass
        self.assertEqual(A, (tuple(range(8)), {}))
        class A(0, *range(1, 8), **d, foo='bar'): pass
        self.assertEqual(A, (tuple(range(8)), {'foo': 'bar'}))

    def testClassCallRecursionLimit(self):
        # [FUNCTION DOC] Checks that infinitely recursive construction raises RecursionError,
        # both directly and when started one call level deeper through a helper.
        class C:
            # [CLASS DOC] HELPER: class whose __init__ constructs another instance of
            # itself forever.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructs a new instance of C, recursing
                # until RecursionError.
                self.c = C()

        with self.assertRaises(RecursionError):
            C()

        def add_one_level():
            # [FUNCTION DOC] HELPER: calls the constructor once more from a deeper
            # stack frame.
            #Each call to C() consumes 2 levels, so offset by 1.
            C()

        with self.assertRaises(RecursionError):
            add_one_level()

    def testMetaclassCallOptimization(self):
        # [FUNCTION DOC] Checks that a metaclass __call__ override is invoked once per
        # instantiation: constructs 100 instances and expects the counter to equal 100.
        calls = 0

        class TypeMetaclass(type):
            # [CLASS DOC] HELPER: metaclass whose __call__ increments an enclosing counter
            # and delegates.
            def __call__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: counts the call via a nonlocal counter, then
                # delegates to type.__call__.
                nonlocal calls
                calls += 1
                return type.__call__(cls, *args, **kwargs)

        class Type(metaclass=TypeMetaclass):
            # [CLASS DOC] HELPER: class using TypeMetaclass; stores the constructor argument.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: stores obj on the instance as _obj.
                self._obj = obj

        for i in range(100):
            Type(i)
        self.assertEqual(calls, 100)

    def test_specialization_class_call_doesnt_crash(self):
        # [FUNCTION DOC] Regression test (gh-123185): repeatedly calling a class constructor
        # with a missing argument (raising TypeError each time) must not crash the
        # specializing interpreter.
        # gh-123185

        class Foo:
            # [CLASS DOC] HELPER: class whose __init__ requires one argument.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: accepts and ignores arg.
                pass

        for _ in range(8):
            try:
                Foo()
            except:
                pass


try:
    from _testinternalcapi import has_inline_values
except ImportError:
    has_inline_values = None

Py_TPFLAGS_MANAGED_DICT = (1 << 2)

class Plain:
    # [CLASS DOC] HELPER: empty class used by the inline-values tests.
    pass


class WithAttrs:
    # [CLASS DOC] HELPER: class setting four integer attributes a to d in __init__.

    def __init__(self):
        # [FUNCTION DOC] HELPER: sets attributes a, b, c, d to 1, 2, 3, 4.
        self.a = 1
        self.b = 2
        self.c = 3
        self.d = 4


@skipIfTorchDynamo("CPython only")
class TestInlineValues(CPythonTestCase):
    # [CLASS DOC] Tests of CPython inline (managed) instance-dict storage via
    # has_inline_values from _testinternalcapi (None when unavailable). Skipped under Dynamo
    # via skipIfTorchDynamo.

    def test_flags(self):
        # [FUNCTION DOC] Checks Plain and WithAttrs have the managed-dict type flag set.
        self.assertEqual(Plain.__flags__ & Py_TPFLAGS_MANAGED_DICT, Py_TPFLAGS_MANAGED_DICT)
        self.assertEqual(WithAttrs.__flags__ & Py_TPFLAGS_MANAGED_DICT, Py_TPFLAGS_MANAGED_DICT)

    def test_has_inline_values(self):
        # [FUNCTION DOC] Checks a fresh instance has inline values and loses them after its
        # dict is deleted.
        c = Plain()
        self.assertTrue(has_inline_values(c))
        del c.__dict__
        self.assertFalse(has_inline_values(c))

    def test_instances(self):
        # [FUNCTION DOC] Checks fresh instances of Plain and WithAttrs have inline values.
        self.assertTrue(has_inline_values(Plain()))
        self.assertTrue(has_inline_values(WithAttrs()))

    def test_inspect_dict(self):
        # [FUNCTION DOC] Checks that merely reading an instance's dict keeps inline values.
        for cls in (Plain, WithAttrs):
            c = cls()
            c.__dict__
            self.assertTrue(has_inline_values(c))

    def test_update_dict(self):
        # [FUNCTION DOC] Checks that updating an instance's dict with new keys keeps
        # inline values.
        d = { "e": 5, "f": 6 }
        for cls in (Plain, WithAttrs):
            c = cls()
            c.__dict__.update(d)
            self.assertTrue(has_inline_values(c))

    @staticmethod
    def set_100(obj):
        # [FUNCTION DOC] HELPER: static helper: sets 100 attributes named a0 to a99 on obj
        # with values 0 to 99.
        for i in range(100):
            setattr(obj, f"a{i}", i)

    def check_100(self, obj):
        # [FUNCTION DOC] HELPER: asserts obj has attributes a0 to a99 with values 0 to 99.
        for i in range(100):
            self.assertEqual(getattr(obj, f"a{i}"), i)

    def test_many_attributes(self):
        # [FUNCTION DOC] Checks inline values are dropped once 100 attributes are added to an
        # instance, the values stay readable, and a new instance of the same class again has
        # inline values.
        class C: pass
        c = C()
        self.assertTrue(has_inline_values(c))
        self.set_100(c)
        self.assertFalse(has_inline_values(c))
        self.check_100(c)
        c = C()
        self.assertTrue(has_inline_values(c))

    def test_many_attributes_with_dict(self):
        # [FUNCTION DOC] Like the previous test but with the instance dict already
        # materialized first.
        class C: pass
        c = C()
        d = c.__dict__
        self.assertTrue(has_inline_values(c))
        self.set_100(c)
        self.assertFalse(has_inline_values(c))
        self.check_100(c)

    def test_bug_117750(self):
        # [FUNCTION DOC] Regression test for bug 117750 (abort on 3.13a6): clearing the dict
        # inside __init__ then assigning a new attribute must not crash and the dict must hold
        # the new item.
        "Aborted on 3.13a6"
        class C:
            # [CLASS DOC] HELPER: class whose __init__ clears its own instance dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: clears the instance dict.
                self.__dict__.clear()

        obj = C()
        self.assertEqual(obj.__dict__, {})
        obj.foo = None # Aborted here
        self.assertEqual(obj.__dict__, {"foo":None})

    def test_store_attr_deleted_dict(self):
        # [FUNCTION DOC] Checks that after deleting an instance dict, attribute assignment
        # still works.
        class Foo:
            # [CLASS DOC] HELPER: empty class.
            pass

        f = Foo()
        del f.__dict__
        f.a = 3
        self.assertEqual(f.a, 3)

    def test_rematerialize_object_dict(self):
        # [FUNCTION DOC] Regression test for gh-121860: re-materializing a deleted managed
        # dict (via a str subclass attribute lookup or a __class__ change) must not crash;
        # dict ends up empty.
        # gh-121860: rematerializing an object's managed dictionary after it
        # had been deleted caused a crash.
        class Foo: pass
        f = Foo()
        f.__dict__["attr"] = 1
        del f.__dict__

        # Using a str subclass is a way to trigger the re-materialization
        class StrSubclass(str): pass
        self.assertFalse(hasattr(f, StrSubclass("attr")))

        # Changing the __class__ also triggers the re-materialization
        class Bar: pass
        f.__class__ = Bar
        self.assertIsInstance(f, Bar)
        self.assertEqual(f.__dict__, {})

    def test_store_attr_type_cache(self):
        # [FUNCTION DOC] Checks the type attribute cache stays consistent with the class dict
        # when a destructor of the replaced value reads the attribute during reassignment.
        """Verifies that the type cache doesn't provide a value which  is
        inconsistent from the dict."""
        class X:
            # [CLASS DOC] HELPER: class whose destructor reads C.a and asserts it matches the
            # class dict.
            def __del__(inner_self):
                # [FUNCTION DOC] HELPER: destructor; reads the attribute from C and checks it
                # equals the dict entry.
                v = C.a
                self.assertEqual(v, C.__dict__['a'])

        class C:
            # [CLASS DOC] HELPER: class holding an X instance in class attribute a.
            a = X()

        # prime the cache
        C.a
        C.a

        # destructor shouldn't be able to see inconsistent state
        C.a = X()
        C.a = X()

    @cpython_only
    def test_detach_materialized_dict_no_memory(self):
        # [FUNCTION DOC] CPython-only: skipped without _testcapi. Runs a subprocess script
        # that simulates an allocation failure while an object with a materialized dict is
        # deleted, expecting an unraisable MemoryError and a surviving dict. Asserts exit code
        # 0 and empty stdout and stderr.
        # Skip test if _testcapi is not available:
        import_helper.import_module('_testcapi')

        code = """if 1:
            import test.support
            import _testcapi

            class A:
                def __init__(self):
                    self.a = 1
                    self.b = 2
            a = A()
            d = a.__dict__
            with test.support.catch_unraisable_exception() as ex:
                _testcapi.set_nomemory(0, 1)
                del a
                assert ex.unraisable.exc_type is MemoryError
            try:
                d["a"]
            except KeyError:
                pass
            else:
                assert False, "KeyError not raised"
        """
        rc, out, err = script_helper.assert_python_ok("-c", code)
        self.assertEqual(rc, 0)
        self.assertFalse(out, msg=out.decode('utf-8'))
        self.assertFalse(err, msg=err.decode('utf-8'))

if __name__ == "__main__":
    run_tests()
