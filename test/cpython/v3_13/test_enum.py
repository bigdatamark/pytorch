# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-adapted copy of CPython 3.13's test_enum: the standard library test suite
# for the enum module (Enum, IntEnum, StrEnum, Flag, IntFlag, ReprEnum, EnumDict, verify/unique,
# global_enum, _simple_enum, pickling, repr/str/format, boundaries, etc). The only change from
# upstream is the header that swaps the base class for CPythonTestCase (aliased as __TestCase)
# so the tests run under the Dynamo harness. At import time it defines module level sample enums
# used for pickle, subclass, repr and doctest tests; failures to build any of them are stored as
# the exception object in that name instead of aborting the import. Ends with run_tests when
# executed as a script.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.12/Lib/test/test_enum.py

import torch
import torch._dynamo.test_case
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

__TestCase = CPythonTestCase
# ======= END DYNAMO PATCH =======

import copy
import enum
import doctest
import inspect
import os
import pydoc
import sys
import unittest
import threading
import typing
import builtins as bltns
from collections import OrderedDict
from datetime import date
from functools import partial
from enum import Enum, EnumMeta, IntEnum, StrEnum, EnumType, Flag, IntFlag, unique, auto
from enum import STRICT, CONFORM, EJECT, KEEP, _simple_enum, _test_simple_enum
from enum import verify, UNIQUE, CONTINUOUS, NAMED_FLAGS, ReprEnum
from enum import member, nonmember, _iter_bits_lsb, EnumDict
from io import StringIO
from pickle import dumps, loads, PicklingError, HIGHEST_PROTOCOL
from test import support
from test.support import ALWAYS_EQ, REPO_ROOT
from test.support import threading_helper
from datetime import timedelta

python_version = sys.version_info[:2]


def reraise_if_not_enum(*enum_types_or_exceptions):
    # [FUNCTION DOC] HELPER: Decorator factory used by tests whose enum definitions may have
    # failed at import time. Takes the enum types (or stored exception objects) and returns a
    # decorator. The wrapped test, when called, collects the arguments that are Exception
    # instances: one is re-raised as is, several are raised together as an ExceptionGroup named
    # 'Enum Exceptions'; if none, the original test function is called normally.
    from functools import wraps

    def decorator(func):
        # [FUNCTION DOC] HELPER: Decorator returned by reraise_if_not_enum; wraps func
        # (preserving its metadata) with the checking wrapper below and returns the wrapper.
        @wraps(func)
        def inner(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper that raises the stored import-time enum
            # exception(s) if any exist, otherwise forwards all arguments to the wrapped test
            # and returns its result.
            excs = [
                e
                for e in enum_types_or_exceptions
                if isinstance(e, Exception)
            ]
            if len(excs) == 1:
                raise excs[0]
            elif excs:
                raise ExceptionGroup('Enum Exceptions', excs)
            return func(*args, **kwargs)
        return inner
    return decorator

MODULE = __name__
SHORT_MODULE = MODULE.split('.')[-1]

# for pickle tests
try:
    class Stooges(Enum):
        # [CLASS DOC] HELPER: Plain Enum fixture (LARRY=1, CURLY=2, MOE=3) used by pickle tests.
        LARRY = 1
        CURLY = 2
        MOE = 3
except Exception as exc:
    Stooges = exc

try:
    class IntStooges(int, Enum):
        # [CLASS DOC] HELPER: Int-mixin Enum fixture with members 1, 2, 3 for pickle tests.
        LARRY = 1
        CURLY = 2
        MOE = 3
except Exception as exc:
    IntStooges = exc

try:
    class FloatStooges(float, Enum):
        # [CLASS DOC] HELPER: Float-mixin Enum fixture (1.39, 2.72, 3.142596) for pickle tests.
        LARRY = 1.39
        CURLY = 2.72
        MOE = 3.142596
except Exception as exc:
    FloatStooges = exc

try:
    class FlagStooges(Flag):
        # [CLASS DOC] HELPER: Flag fixture with single-bit members 1, 2, 4 plus a composite
        # alias BIG=389, used by pickle and repr tests.
        LARRY = 1
        CURLY = 2
        MOE = 4
        BIG = 389
except Exception as exc:
    FlagStooges = exc

try:
    class FlagStoogesWithZero(Flag):
        # [CLASS DOC] HELPER: Same as FlagStooges but with an explicit zero-valued
        # NOFLAG member.
        NOFLAG = 0
        LARRY = 1
        CURLY = 2
        MOE = 4
        BIG = 389
except Exception as exc:
    FlagStoogesWithZero = exc

try:
    class IntFlagStooges(IntFlag):
        # [CLASS DOC] HELPER: IntFlag fixture mirroring FlagStooges (members 1, 2, 4
        # and BIG=389).
        LARRY = 1
        CURLY = 2
        MOE = 4
        BIG = 389
except Exception as exc:
    IntFlagStooges = exc

try:
    class IntFlagStoogesWithZero(IntFlag):
        # [CLASS DOC] HELPER: IntFlag fixture mirroring FlagStooges with a zero-valued
        # NOFLAG member.
        NOFLAG = 0
        LARRY = 1
        CURLY = 2
        MOE = 4
        BIG = 389
except Exception as exc:
    IntFlagStoogesWithZero = exc

# for pickle test and subclass tests
try:
    class Name(StrEnum):
        # [CLASS DOC] HELPER: StrEnum fixture whose members' values are the strings 'Guido van
        # Rossum' and 'Barry Warsaw'; used for pickle and subclass tests.
        BDFL = 'Guido van Rossum'
        FLUFL = 'Barry Warsaw'
except Exception as exc:
    Name = exc

try:
    Question = Enum('Question', 'who what when where why', module=__name__)
except Exception as exc:
    Question = exc

try:
    Answer = Enum('Answer', 'him this then there because')
except Exception as exc:
    Answer = exc

try:
    Theory = Enum('Theory', 'rule law supposition', qualname='spanish_inquisition')
except Exception as exc:
    Theory = exc

# for doctests
try:
    class Fruit(Enum):
        # [CLASS DOC] HELPER: Plain Enum fixture (TOMATO, BANANA, CHERRY) referenced
        # by doctests.
        TOMATO = 1
        BANANA = 2
        CHERRY = 3
except Exception:
    pass

def test_pickle_dump_load(assertion, source, target=None):
    # [FUNCTION DOC] HELPER: Pickles source with every protocol from 0 to HIGHEST_PROTOCOL,
    # unpickles it, and calls assertion with the round-tripped object and the expected target
    # (defaults to source). Marked not-a-test via the __test__ attribute set after the def.
    if target is None:
        target = source
    for protocol in range(HIGHEST_PROTOCOL + 1):
        assertion(loads(dumps(source, protocol=protocol)), target)
test_pickle_dump_load.__test__ = False

def test_pickle_exception(assertion, exception, obj):
    # [FUNCTION DOC] HELPER: For every pickle protocol, asserts (through the supplied context
    # manager factory, e.g. assertRaises) that pickling obj raises the given exception. Marked
    # not-a-test via the __test__ attribute set after the def.
    for protocol in range(HIGHEST_PROTOCOL + 1):
        with assertion(exception):
            dumps(obj, protocol=protocol)
test_pickle_exception.__test__ = False

class TestHelpers(__TestCase):
    # [CLASS DOC] Tests for the private enum module predicates _is_descriptor, _is_sunder,
    # _is_dunder, _is_private and _iter_bits_lsb. Class attributes hold name samples (sunder,
    # dunder, private, private-and-sunder and random names) used to check each classifier.
    # _is_descriptor, _is_sunder, _is_dunder

    sunder_names = '_bad_', '_good_', '_what_ho_'
    dunder_names = '__mal__', '__bien__', '__que_que__'
    private_names = '_MyEnum__private', '_MyEnum__still_private', '_MyEnum___triple_private'
    private_and_sunder_names = '_MyEnum__private_', '_MyEnum__also_private_'
    random_names = 'okay', '_semi_private', '_weird__', '_MyEnum__'

    def test_is_descriptor(self):
        # [FUNCTION DOC] Checks _is_descriptor is False for a plain object and becomes True once
        # any one of __get__, __set__ or __delete__ is set on it.
        class foo:
            # [CLASS DOC] HELPER: Empty class whose instances get descriptor attributes added by
            # the test.
            pass
        for attr in ('__get__','__set__','__delete__'):
            obj = foo()
            self.assertFalse(enum._is_descriptor(obj))
            setattr(obj, attr, 1)
            self.assertTrue(enum._is_descriptor(obj))

    def test_sunder(self):
        # [FUNCTION DOC] Checks _is_sunder is True for sunder and private-and-sunder names and
        # False for dunder, private and ordinary names, plus a set of edge strings (underscores
        # only, etc).
        for name in self.sunder_names + self.private_and_sunder_names:
            self.assertTrue(enum._is_sunder(name), '%r is a not sunder name?' % name)
        for name in self.dunder_names + self.private_names + self.random_names:
            self.assertFalse(enum._is_sunder(name), '%r is a sunder name?' % name)
        for s in ('_a_', '_aa_'):
            self.assertTrue(enum._is_sunder(s))
        for s in ('a', 'a_', '_a', '__a', 'a__', '__a__', '_a__', '__a_', '_',
                '__', '___', '____', '_____',):
            self.assertFalse(enum._is_sunder(s))

    def test_dunder(self):
        # [FUNCTION DOC] Checks _is_dunder accepts only double-underscore bracketed names and
        # rejects sunder, private and random names and short all-underscore strings.
        for name in self.dunder_names:
            self.assertTrue(enum._is_dunder(name), '%r is a not dunder name?' % name)
        for name in self.sunder_names + self.private_names + self.private_and_sunder_names + self.random_names:
            self.assertFalse(enum._is_dunder(name), '%r is a dunder name?' % name)
        for s in ('__a__', '__aa__'):
            self.assertTrue(enum._is_dunder(s))
        for s in ('a', 'a_', '_a', '__a', 'a__', '_a_', '_a__', '__a_', '_',
                '__', '___', '____', '_____',):
            self.assertFalse(enum._is_dunder(s))


    def test_is_private(self):
        # [FUNCTION DOC] Checks _is_private for the class name MyEnum is True for mangled
        # private names and False for sunder, dunder and random names.
        # KNOWN ISSUE: the assertion messages contain a %r placeholder that is never formatted.
        for name in self.private_names + self.private_and_sunder_names:
            self.assertTrue(enum._is_private('MyEnum', name), '%r is a not private name?')
        for name in self.sunder_names + self.dunder_names + self.random_names:
            self.assertFalse(enum._is_private('MyEnum', name), '%r is a private name?')

    def test_iter_bits_lsb(self):
        # [FUNCTION DOC] Checks _iter_bits_lsb yields the set bits of 7 as 1, 2, 4 in
        # ascending order and that a negative input raises ValueError mentioning it is not a
        # positive integer.
        self.assertEqual(list(_iter_bits_lsb(7)), [1, 2, 4])
        self.assertRaisesRegex(ValueError, '-8 is not a positive integer', list, _iter_bits_lsb(-8))


# for subclassing tests

class classproperty:
    # [CLASS DOC] HELPER: Minimal read-only descriptor that calls fget with the owner class,
    # used to test that enums tolerate class-level properties in subclass tests.

    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        # [FUNCTION DOC] HELPER: Stores fget, fset, fdel and sets __doc__ to doc, falling back
        # to the doc of fget when doc is not given.
        self.fget = fget
        self.fset = fset
        self.fdel = fdel
        if doc is None and fget is not None:
            doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, instance, ownerclass):
        # [FUNCTION DOC] HELPER: Descriptor protocol hook; ignores the instance and returns the
        # result of calling fget with the owning class.
        return self.fget(ownerclass)

# for global repr tests

try:
    @enum.global_enum
    class HeadlightsK(IntFlag, boundary=enum.KEEP):
        # [CLASS DOC] HELPER: IntFlag fixture with KEEP boundary (OFF_K=0 and three auto
        # members), registered into module globals via global_enum for global repr tests.
        OFF_K = 0
        LOW_BEAM_K = auto()
        HIGH_BEAM_K = auto()
        FOG_K = auto()
except Exception as exc:
    HeadlightsK = exc


try:
    @enum.global_enum
    class HeadlightsC(IntFlag, boundary=enum.CONFORM):
        # [CLASS DOC] HELPER: IntFlag fixture with CONFORM boundary (OFF_C=0 plus three auto
        # members), registered via global_enum for global repr tests.
        OFF_C = 0
        LOW_BEAM_C = auto()
        HIGH_BEAM_C = auto()
        FOG_C = auto()
except Exception as exc:
    HeadlightsC = exc


try:
    @enum.global_enum
    class NoName(Flag):
        # [CLASS DOC] HELPER: Flag fixture (ONE=1, TWO=2) registered via global_enum, used to
        # test repr of members of a global enum without a zero (no-name) member.
        ONE = 1
        TWO = 2
except Exception as exc:
    NoName = exc


# tests

class _EnumTests:
    # [CLASS DOC] Mixin holding tests common to all enum flavors (plain, Int, Str, Flag, mixed
    # types, class syntax and functional API). Concrete subclasses supply enum_type and combine
    # with an output mixin and the test case base. setUp builds, per flavor, the fixtures
    # MainEnum, NewStrEnum, NewFormatEnum, NewStrFormatEnum, NewBaseEnum, NewSubEnum, LazyGNV
    # and BusyGNV, plus names, values, source_values and is_flag.
    """
    Test for behavior that is the same across the different types of enumerations.
    """

    values = None

    def setUp(self):
        # [FUNCTION DOC] Builds the per-test fixture enums. Chooses the construction style from
        # the test class name: names ending in 'Class' use class statements, names ending in
        # 'Function' use the functional API with equivalent definitions; any other name raises
        # ValueError. Sets self.MainEnum, NewStrEnum, NewFormatEnum, NewStrFormatEnum,
        # NewBaseEnum, NewSubEnum, LazyGNV, BusyGNV, is_flag, names, values (depends on
        # StrEnum/str mixin/Flag) and source_values; for Flag types also dupe2 as the member
        # with value 5.
        super().setUp()
        if self.__class__.__name__[-5:] == 'Class':
            class BaseEnum(self.enum_type):
                # [CLASS DOC] HELPER: Base enum of self.enum_type that defines an enum.property
                # named first returning a string with the member name; MainEnum overrides it
                # with a member.
                @enum.property
                def first(self):
                    # [FUNCTION DOC] HELPER: enum.property getter returning the text "<name> is
                    # first!" for a member.
                    return '%s is first!' % self.name
            class MainEnum(BaseEnum):
                # [CLASS DOC] HELPER: Fixture enum with members first, second, third (auto) and
                # dupe, an alias of third (value 3 for Flag types, which aliases the combined
                # flag instead).
                first = auto()
                second = auto()
                third = auto()
                if issubclass(self.enum_type, Flag):
                    dupe = 3
                else:
                    dupe = third
            self.MainEnum = MainEnum
            #
            class NewStrEnum(self.enum_type):
                # [CLASS DOC] HELPER: Fixture enum whose __str__ returns the upper-cased
                # member name.
                def __str__(self):
                    # [FUNCTION DOC] HELPER: Returns the upper-cased member name.
                    return self.name.upper()
                first = auto()
            self.NewStrEnum = NewStrEnum
            #
            class NewFormatEnum(self.enum_type):
                # [CLASS DOC] HELPER: Fixture enum overriding __format__ to return the
                # upper-cased name.
                def __format__(self, spec):
                    # [FUNCTION DOC] HELPER: Ignores the spec and returns the upper-cased
                    # member name.
                    return self.name.upper()
                first = auto()
            self.NewFormatEnum = NewFormatEnum
            #
            class NewStrFormatEnum(self.enum_type):
                # [CLASS DOC] HELPER: Fixture enum overriding both __str__ (title-cased name)
                # and __format__ (reversed name).
                def __str__(self):
                    # [FUNCTION DOC] HELPER: Returns the member name title-cased.
                    return self.name.title()
                def __format__(self, spec):
                    # [FUNCTION DOC] HELPER: Ignores the spec and returns the member
                    # name reversed.
                    return ''.join(reversed(self.name))
                first = auto()
            self.NewStrFormatEnum = NewStrFormatEnum
            #
            class NewBaseEnum(self.enum_type):
                # [CLASS DOC] HELPER: Member-less base enum overriding __str__ (title case) and
                # __format__ (reversed name); subclassed by NewSubEnum to test inheritance of
                # the overrides.
                def __str__(self):
                    # [FUNCTION DOC] HELPER: Returns the member name title-cased.
                    return self.name.title()
                def __format__(self, spec):
                    # [FUNCTION DOC] HELPER: Ignores the spec and returns the member
                    # name reversed.
                    return ''.join(reversed(self.name))
            self.NewBaseEnum = NewBaseEnum
            class NewSubEnum(NewBaseEnum):
                # [CLASS DOC] HELPER: Subclass of NewBaseEnum with one auto member 'first'.
                first = auto()
            self.NewSubEnum = NewSubEnum
            #
            class LazyGNV(self.enum_type):
                # [CLASS DOC] HELPER: Enum whose _generate_next_value_ is a plain function
                # (not decorated), returning None; used to test that it is treated as
                # implicitly static.
                def _generate_next_value_(name, start, last, values):
                    # [FUNCTION DOC] HELPER: Placeholder next-value generator taking (name,
                    # start, last, values) and returning None.
                    pass
            self.LazyGNV = LazyGNV
            #
            class BusyGNV(self.enum_type):
                # [CLASS DOC] HELPER: Enum whose _generate_next_value_ is explicitly a
                # staticmethod returning None.
                @staticmethod
                def _generate_next_value_(name, start, last, values):
                    # [FUNCTION DOC] HELPER: Placeholder next-value generator returning None.
                    pass
            self.BusyGNV = BusyGNV
            #
            self.is_flag = False
            self.names = ['first', 'second', 'third']
            if issubclass(MainEnum, StrEnum):
                self.values = self.names
            elif MainEnum._member_type_ is str:
                self.values = ['1', '2', '3']
            elif issubclass(self.enum_type, Flag):
                self.values = [1, 2, 4]
                self.is_flag = True
                self.dupe2 = MainEnum(5)
            else:
                self.values = self.values or [1, 2, 3]
            #
            if not getattr(self, 'source_values', False):
                self.source_values = self.values
        elif self.__class__.__name__[-8:] == 'Function':
            @enum.property
            def first(self):
                # [FUNCTION DOC] HELPER: enum.property getter returning "<name> is first!"
                # (functional-API variant).
                return '%s is first!' % self.name
            BaseEnum = self.enum_type('BaseEnum', {'first':first})
            #
            first = auto()
            second = auto()
            third = auto()
            if issubclass(self.enum_type, Flag):
                dupe = 3
            else:
                dupe = third
            self.MainEnum = MainEnum = BaseEnum('MainEnum', dict(first=first, second=second, third=third, dupe=dupe))
            #
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the upper-cased member name; injected as
                # __str__ in functional-API enums.
                return self.name.upper()
            first = auto()
            self.NewStrEnum = self.enum_type('NewStrEnum', (('first',first),('__str__',__str__)))
            #
            def __format__(self, spec):
                # [FUNCTION DOC] HELPER: Ignores spec and returns the upper-cased name; injected
                # as __format__.
                return self.name.upper()
            first = auto()
            self.NewFormatEnum = self.enum_type('NewFormatEnum', [('first',first),('__format__',__format__)])
            #
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the title-cased member name; injected
                # as __str__.
                return self.name.title()
            def __format__(self, spec):
                # [FUNCTION DOC] HELPER: Returns the member name reversed, ignoring spec;
                # injected as __format__.
                return ''.join(reversed(self.name))
            first = auto()
            self.NewStrFormatEnum = self.enum_type('NewStrFormatEnum', dict(first=first, __format__=__format__, __str__=__str__))
            #
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the title-cased member name; injected
                # as __str__.
                return self.name.title()
            def __format__(self, spec):
                # [FUNCTION DOC] HELPER: Returns the member name reversed, ignoring spec;
                # injected as __format__.
                return ''.join(reversed(self.name))
            self.NewBaseEnum = self.enum_type('NewBaseEnum', dict(__format__=__format__, __str__=__str__))
            self.NewSubEnum = self.NewBaseEnum('NewSubEnum', 'first')
            #
            def _generate_next_value_(name, start, last, values):
                # [FUNCTION DOC] HELPER: Placeholder next-value generator (plain function)
                # returning None.
                pass
            self.LazyGNV = self.enum_type('LazyGNV', {'_generate_next_value_':_generate_next_value_})
            #
            @staticmethod
            def _generate_next_value_(name, start, last, values):
                # [FUNCTION DOC] HELPER: Placeholder next-value generator (staticmethod)
                # returning None.
                pass
            self.BusyGNV = self.enum_type('BusyGNV', {'_generate_next_value_':_generate_next_value_})
            #
            self.is_flag = False
            self.names = ['first', 'second', 'third']
            if issubclass(MainEnum, StrEnum):
                self.values = self.names
            elif MainEnum._member_type_ is str:
                self.values = ['1', '2', '3']
            elif issubclass(self.enum_type, Flag):
                self.values = [1, 2, 4]
                self.is_flag = True
                self.dupe2 = MainEnum(5)
            else:
                self.values = self.values or [1, 2, 3]
            #
            if not getattr(self, 'source_values', False):
                self.source_values = self.values
        else:
            raise ValueError('unknown enum style: %r' % self.__class__.__name__)

    def assertFormatIsValue(self, spec, member):
        # [FUNCTION DOC] HELPER: Asserts formatting the member with the format spec gives the
        # same string as formatting its underlying value with that spec.
        self.assertEqual(spec.format(member), spec.format(member.value))

    def assertFormatIsStr(self, spec, member):
        # [FUNCTION DOC] HELPER: Asserts formatting the member with the format spec gives the
        # same string as formatting str(member) with that spec.
        self.assertEqual(spec.format(member), spec.format(str(member)))

    def test_attribute_deletion(self):
        # [FUNCTION DOC] Checks that a user method on an enum class can be deleted (hasattr
        # turns False) but deleting a member, a nonexistent attribute, or a member's name
        # raises AttributeError.
        class Season(self.enum_type):
            # [CLASS DOC] HELPER: Enum of self.enum_type with three auto members and a
            # method spam.
            SPRING = auto()
            SUMMER = auto()
            AUTUMN = auto()
            #
            def spam(cls):
                # [FUNCTION DOC] HELPER: Empty method used as the deletable attribute.
                pass
        #
        self.assertTrue(hasattr(Season, 'spam'))
        del Season.spam
        self.assertFalse(hasattr(Season, 'spam'))
        #
        with self.assertRaises(AttributeError):
            del Season.SPRING
        with self.assertRaises(AttributeError):
            del Season.DRY
        with self.assertRaises(AttributeError):
            del Season.SPRING.name

    def test_bad_new_super(self):
        # [FUNCTION DOC] Checks that defining a __new__ that calls super().__new__ with a value
        # raises TypeError telling the user not to use super().__new__.
        with self.assertRaisesRegex(
                TypeError,
                'do not use .super...__new__;',
            ):
            class BadSuper(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum whose __new__ calls super().__new__;
                # definition must fail.
                def __new__(cls, value):
                    # [FUNCTION DOC] HELPER: Offending __new__ that calls the parent __new__ and
                    # returns the object.
                    obj = super().__new__(cls, value)
                    return obj
                failed = 1

    def test_basics(self):
        # [FUNCTION DOC] Checks basic enum behavior of MainEnum: class repr/str/format ("enum"
        # or "flag" wording), iteration order, names and values, lookups and alias handling for
        # both plain and flag types.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(repr(TE), "<flag 'MainEnum'>")
            self.assertEqual(str(TE), "<flag 'MainEnum'>")
            self.assertEqual(format(TE), "<flag 'MainEnum'>")
            self.assertTrue(TE(5) is self.dupe2)
            self.assertTrue(7 in TE)
        else:
            self.assertEqual(repr(TE), "<enum 'MainEnum'>")
            self.assertEqual(str(TE), "<enum 'MainEnum'>")
            self.assertEqual(format(TE), "<enum 'MainEnum'>")
        self.assertEqual(list(TE), [TE.first, TE.second, TE.third])
        self.assertEqual(
                [m.name for m in TE],
                self.names,
                )
        self.assertEqual(
                [m.value for m in TE],
                self.values,
                )
        self.assertEqual(
                [m.first for m in TE],
                ['first is first!', 'second is first!', 'third is first!']
                )
        for member, name in zip(TE, self.names, strict=True):
            self.assertIs(TE[name], member)
        for member, value in zip(TE, self.values, strict=True):
            self.assertIs(TE(value), member)
        if issubclass(TE, StrEnum):
            self.assertTrue(TE.dupe is TE('third') is TE['dupe'])
        elif TE._member_type_ is str:
            self.assertTrue(TE.dupe is TE('3') is TE['dupe'])
        elif issubclass(TE, Flag):
            self.assertTrue(TE.dupe is TE(3) is TE['dupe'])
        else:
            self.assertTrue(TE.dupe is TE(self.values[2]) is TE['dupe'])

    def test_bool_is_true(self):
        # [FUNCTION DOC] Checks that an empty enum class, MainEnum and every MainEnum member
        # are truthy.
        class Empty(self.enum_type):
            # [CLASS DOC] HELPER: Member-less enum of self.enum_type, used to check
            # class truthiness.
            pass
        self.assertTrue(Empty)
        #
        self.assertTrue(self.MainEnum)
        for member in self.MainEnum:
            self.assertTrue(member)

    def test_changing_member_fails(self):
        # [FUNCTION DOC] Checks that assigning a new value to an existing member name on the
        # enum class raises AttributeError.
        MainEnum = self.MainEnum
        with self.assertRaises(AttributeError):
            self.MainEnum.second = 'really first'

    def test_contains_tf(self):
        # [FUNCTION DOC] Checks membership (in) semantics: members and their raw values are
        # contained (strings only for StrEnum), aliases are contained, NaN and members of an
        # unrelated Enum are not. For object-based enums members built from equal values in
        # another enum are not contained; for mixed-type enums they are.
        MainEnum = self.MainEnum
        self.assertIn(MainEnum.first, MainEnum)
        self.assertTrue(self.values[0] in MainEnum)
        if type(self) not in (TestStrEnumClass, TestStrEnumFunction):
            self.assertFalse('first' in MainEnum)
        val = MainEnum.dupe
        self.assertIn(val, MainEnum)
        self.assertNotIn(float('nan'), MainEnum)
        #
        class OtherEnum(Enum):
            # [CLASS DOC] HELPER: Unrelated plain Enum with members one and two.
            one = auto()
            two = auto()
        self.assertNotIn(OtherEnum.two, MainEnum)
        #
        if MainEnum._member_type_ is object:
            # enums without mixed data types will always be False
            class NotEqualEnum(self.enum_type):
                # [CLASS DOC] HELPER: Enum of self.enum_type whose members reuse the first two
                # source values.
                this = self.source_values[0]
                that = self.source_values[1]
            self.assertNotIn(NotEqualEnum.this, MainEnum)
            self.assertNotIn(NotEqualEnum.that, MainEnum)
        else:
            # enums with mixed data types may be True
            class EqualEnum(self.enum_type):
                # [CLASS DOC] HELPER: Enum of self.enum_type with the same shape as
                # NotEqualEnum, used in the mixed-data-type branch where its members compare
                # equal to MainEnum members.
                this = self.source_values[0]
                that = self.source_values[1]
            self.assertIn(EqualEnum.this, MainEnum)
            self.assertIn(EqualEnum.that, MainEnum)

    def test_contains_same_name_diff_enum_diff_values(self):
        # [FUNCTION DOC] Checks that members with identical names but different values in two
        # enums are only contained in their own enum, never in the other.
        MainEnum = self.MainEnum
        #
        class OtherEnum(Enum):
            # [CLASS DOC] HELPER: Plain Enum with members first, second, third holding
            # string values.
            first = "brand"
            second = "new"
            third = "values"
        #
        self.assertIn(MainEnum.first, MainEnum)
        self.assertIn(MainEnum.second, MainEnum)
        self.assertIn(MainEnum.third, MainEnum)
        self.assertNotIn(MainEnum.first, OtherEnum)
        self.assertNotIn(MainEnum.second, OtherEnum)
        self.assertNotIn(MainEnum.third, OtherEnum)
        #
        self.assertIn(OtherEnum.first, OtherEnum)
        self.assertIn(OtherEnum.second, OtherEnum)
        self.assertIn(OtherEnum.third, OtherEnum)
        self.assertNotIn(OtherEnum.first, MainEnum)
        self.assertNotIn(OtherEnum.second, MainEnum)
        self.assertNotIn(OtherEnum.third, MainEnum)

    def test_dir_on_class(self):
        # [FUNCTION DOC] Checks dir() of the enum class equals the set computed by the local
        # enum_dir helper.
        TE = self.MainEnum
        self.assertEqual(set(dir(TE)), set(enum_dir(TE)))

    def test_dir_on_item(self):
        # [FUNCTION DOC] Checks dir() of a member equals the set computed by the local
        # member_dir helper.
        TE = self.MainEnum
        self.assertEqual(set(dir(TE.first)), set(member_dir(TE.first)))

    def test_dir_with_added_behavior(self):
        # [FUNCTION DOC] Checks a user method appears in dir() of a member but not in dir() of
        # the class.
        class Test(self.enum_type):
            # [CLASS DOC] HELPER: Enum with two auto members and a method wowser.
            this = auto()
            these = auto()
            def wowser(self):
                # [FUNCTION DOC] HELPER: Returns a greeting string containing the member name.
                return ("Wowser! I'm %s!" % self.name)
        self.assertTrue('wowser' not in dir(Test))
        self.assertTrue('wowser' in dir(Test.this))

    def test_dir_on_sub_with_behavior_on_super(self):
        # [FUNCTION DOC] Regression test (issue 22506): a method defined on a member-less base
        # enum is hidden from dir() of the subclass but shown in dir() of its members.
        # see issue22506
        class SuperEnum(self.enum_type):
            # [CLASS DOC] HELPER: Member-less base enum defining the method invisible.
            def invisible(self):
                # [FUNCTION DOC] HELPER: Returns a fixed string.
                return "did you see me?"
        class SubEnum(SuperEnum):
            # [CLASS DOC] HELPER: Subclass of SuperEnum with one auto member.
            sample = auto()
        self.assertTrue('invisible' not in dir(SubEnum))
        self.assertTrue('invisible' in dir(SubEnum.sample))

    def test_dir_on_sub_with_behavior_including_instance_dict_on_super(self):
        # [FUNCTION DOC] Regression test (issue 40084): an attribute set on members in a base
        # class __new__ (description) shows up in dir() of members of a subclass but not in
        # dir() of the class.
        # see issue40084
        class SuperEnum(self.enum_type):
            # [CLASS DOC] HELPER: Base enum with a custom __new__ that builds the member via the
            # member type's __new__, sets _value_ and an instance attribute description.
            def __new__(cls, *value, **kwds):
                # [FUNCTION DOC] HELPER: Creates a member: uses the plain object constructor for
                # object enums, otherwise unpacks the (possibly tuple) value for the mixin type;
                # sets _value_ to the single value or the tuple of values, adds description, and
                # returns the new member.
                new = self.enum_type._member_type_.__new__
                if self.enum_type._member_type_ is object:
                    obj = new(cls)
                else:
                    if isinstance(value[0], tuple):
                        create_value ,= value[0]
                    else:
                        create_value = value
                    obj = new(cls, *create_value)
                obj._value_ = value[0] if len(value) == 1 else value
                obj.description = 'test description'
                return obj
        class SubEnum(SuperEnum):
            # [CLASS DOC] HELPER: Subclass with one member whose value is the second
            # source value.
            sample = self.source_values[1]
        self.assertTrue('description' not in dir(SubEnum))
        self.assertTrue('description' in dir(SubEnum.sample), dir(SubEnum.sample))

    def test_empty_enum_has_no_values(self):
        # [FUNCTION DOC] Checks calling a member-less enum (NewBaseEnum) with a value raises
        # TypeError saying the class has no members.
        with self.assertRaisesRegex(TypeError, "<.... 'NewBaseEnum'> has no members"):
            self.NewBaseEnum(7)

    def test_enum_in_enum_out(self):
        # [FUNCTION DOC] Checks calling the enum with an existing member returns that
        # same member.
        Main = self.MainEnum
        self.assertIs(Main(Main.first), Main.first)

    def test_gnv_is_static(self):
        # [FUNCTION DOC] Checks _generate_next_value_ is stored as a staticmethod in the class
        # dict for both the plain-function and the explicit-staticmethod fixtures.
        lazy = self.LazyGNV
        busy = self.BusyGNV
        self.assertTrue(type(lazy.__dict__['_generate_next_value_']) is staticmethod)
        self.assertTrue(type(busy.__dict__['_generate_next_value_']) is staticmethod)

    def test_hash(self):
        # [FUNCTION DOC] Checks members are hashable and usable as dictionary keys.
        MainEnum = self.MainEnum
        mapping = {}
        mapping[MainEnum.first] = '1225'
        mapping[MainEnum.second] = '0315'
        mapping[MainEnum.third] = '0704'
        self.assertEqual(mapping[MainEnum.second], '0315')

    def test_invalid_names(self):
        # [FUNCTION DOC] Checks that reserved names (mro, _create_, _get_mixins_, _find_new_ and
        # an unknown sunder name) used as member names raise ValueError at class creation.
        with self.assertRaises(ValueError):
            class Wrong(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum using reserved name mro; definition
                # must raise.
                mro = 9
        with self.assertRaises(ValueError):
            class Wrong(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum using reserved name _create_; definition
                # must raise.
                _create_= 11
        with self.assertRaises(ValueError):
            class Wrong(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum using reserved name _get_mixins_; definition
                # must raise.
                _get_mixins_ = 9
        with self.assertRaises(ValueError):
            class Wrong(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum using reserved name _find_new_; definition
                # must raise.
                _find_new_ = 1
        with self.assertRaises(ValueError):
            class Wrong(self.enum_type):
                # [CLASS DOC] HELPER: Invalid enum using unknown sunder name _any_name_;
                # definition must raise.
                _any_name_ = 9

    def test_object_str_override(self):
        # [FUNCTION DOC] Checks assigning object.__str__ as __str__ in an enum is kept and not
        # reset to the Enum default, so str() uses the overridden __repr__ text.
        "check that setting __str__ to object's is not reset to Enum's"
        class Generic(self.enum_type):
            # [CLASS DOC] HELPER: Enum with custom __repr__ and __str__ set to object's __str__.
            item = self.source_values[2]
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the member name followed by ".test".
                return "%s.test" % (self._name_, )
            __str__ = object.__str__
        self.assertEqual(str(Generic.item), 'item.test')

    def test_overridden_str(self):
        # [FUNCTION DOC] Checks str() and format() of NewStrEnum members give the
        # upper-cased name.
        NS = self.NewStrEnum
        self.assertEqual(str(NS.first), NS.first.name.upper())
        self.assertEqual(format(NS.first), NS.first.name.upper())

    def test_overridden_str_format(self):
        # [FUNCTION DOC] Checks NewStrFormatEnum: str gives title-cased name and format the
        # reversed name.
        NSF = self.NewStrFormatEnum
        self.assertEqual(str(NSF.first), NSF.first.name.title())
        self.assertEqual(format(NSF.first), ''.join(reversed(NSF.first.name)))

    def test_overridden_str_format_inherited(self):
        # [FUNCTION DOC] Checks a subclass inherits the __str__ and __format__ overrides of its
        # base enum.
        NSE = self.NewSubEnum
        self.assertEqual(str(NSE.first), NSE.first.name.title())
        self.assertEqual(format(NSE.first), ''.join(reversed(NSE.first.name)))

    def test_programmatic_function_string(self):
        # [FUNCTION DOC] Checks the functional API with a whitespace separated name string:
        # member count, order, names, auto values, equality with values (only for mixin types),
        # membership, member type and lookup by value.
        MinorEnum = self.enum_type('MinorEnum', 'june july august')
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        values = self.values
        if self.enum_type is StrEnum:
            values = ['june','july','august']
        for month, av in zip('june july august'.split(), values):
            e = MinorEnum[month]
            self.assertEqual(e.value, av, list(MinorEnum))
            self.assertEqual(e.name, month)
            if MinorEnum._member_type_ is not object and issubclass(MinorEnum, MinorEnum._member_type_):
                self.assertEqual(e, av)
            else:
                self.assertNotEqual(e, av)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)
            self.assertIs(e, MinorEnum(av))

    def test_programmatic_function_string_list(self):
        # [FUNCTION DOC] Same as the string variant of the functional API test but names are
        # given as a list.
        MinorEnum = self.enum_type('MinorEnum', ['june', 'july', 'august'])
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        values = self.values
        if self.enum_type is StrEnum:
            values = ['june','july','august']
        for month, av in zip('june july august'.split(), values):
            e = MinorEnum[month]
            self.assertEqual(e.value, av)
            self.assertEqual(e.name, month)
            if MinorEnum._member_type_ is not object and issubclass(MinorEnum, MinorEnum._member_type_):
                self.assertEqual(e, av)
            else:
                self.assertNotEqual(e, av)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)
            self.assertIs(e, MinorEnum(av))

    def test_programmatic_function_iterable(self):
        # [FUNCTION DOC] Checks the functional API with an iterable of (name, value)
        # pairs: count, order, names, values, equality with mixin values, membership and
        # lookup by value.
        MinorEnum = self.enum_type(
                'MinorEnum',
                (('june', self.source_values[0]), ('july', self.source_values[1]), ('august', self.source_values[2]))
                )
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for month, av in zip('june july august'.split(), self.values):
            e = MinorEnum[month]
            self.assertEqual(e.value, av)
            self.assertEqual(e.name, month)
            if MinorEnum._member_type_ is not object and issubclass(MinorEnum, MinorEnum._member_type_):
                self.assertEqual(e, av)
            else:
                self.assertNotEqual(e, av)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)
            self.assertIs(e, MinorEnum(av))

    def test_programmatic_function_from_dict(self):
        # [FUNCTION DOC] Checks the functional API with an ordered mapping of names to values:
        # count, order, equality with mixin values, membership, type and lookup by value.
        MinorEnum = self.enum_type(
                'MinorEnum',
                OrderedDict((('june', self.source_values[0]), ('july', self.source_values[1]), ('august', self.source_values[2])))
                )
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for month, av in zip('june july august'.split(), self.values):
            e = MinorEnum[month]
            if MinorEnum._member_type_ is not object and issubclass(MinorEnum, MinorEnum._member_type_):
                self.assertEqual(e, av)
            else:
                self.assertNotEqual(e, av)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)
            self.assertIs(e, MinorEnum(av))

    def test_repr(self):
        # [FUNCTION DOC] Checks member repr strings: flags show zero and combined names with
        # value, StrEnum shows quoted value, others show the value repr; aliases display under
        # the canonical name.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(repr(TE(0)), "<MainEnum: 0>")
            self.assertEqual(repr(TE.dupe), "<MainEnum.dupe: 3>")
            self.assertEqual(repr(self.dupe2), "<MainEnum.first|third: 5>")
        elif issubclass(TE, StrEnum):
            self.assertEqual(repr(TE.dupe), "<MainEnum.third: 'third'>")
        else:
            self.assertEqual(repr(TE.dupe), "<MainEnum.third: %r>" % (self.values[2], ), TE._value_repr_)
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(repr(member), "<MainEnum.%s: %r>" % (member.name, member.value))

    def test_repr_override(self):
        # [FUNCTION DOC] Checks a user-defined __repr__ on the enum is used for its members.
        class Generic(self.enum_type):
            # [CLASS DOC] HELPER: Enum with three auto members and a custom __repr__.
            first = auto()
            second = auto()
            third = auto()
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a sentence containing the member name.
                return "don't you just love shades of %s?" % self.name
        self.assertEqual(
                repr(Generic.third),
                "don't you just love shades of third?",
                )

    def test_inherited_repr(self):
        # [FUNCTION DOC] Checks a subclass of an enum that defines __repr__ inherits
        # that __repr__.
        class MyEnum(self.enum_type):
            # [CLASS DOC] HELPER: Member-less base enum with a custom __repr__.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns "My name is <name>.".
                return "My name is %s." % self.name
        class MySubEnum(MyEnum):
            # [CLASS DOC] HELPER: Subclass of MyEnum with members this, that, theother.
            this = auto()
            that = auto()
            theother = auto()
        self.assertEqual(repr(MySubEnum.that), "My name is that.")

    def test_multiple_superclasses_repr(self):
        # [FUNCTION DOC] Checks repr of a member of an enum whose bases include a non-Enum class
        # that uses EnumMeta as metaclass is the standard "<E.A: 1>" form.
        class _EnumSuperClass(metaclass=EnumMeta):
            # [CLASS DOC] HELPER: Empty plain class using EnumMeta as its metaclass.
            pass
        class E(_EnumSuperClass, Enum):
            # [CLASS DOC] HELPER: Enum combining the EnumMeta based class with Enum, one
            # member A=1.
            A = 1
        self.assertEqual(repr(E.A), "<E.A: 1>")

    def test_reversed_iteration_order(self):
        # [FUNCTION DOC] Checks reversed() over an enum yields members in the reverse of
        # definition order.
        self.assertEqual(
                list(reversed(self.MainEnum)),
                [self.MainEnum.third, self.MainEnum.second, self.MainEnum.first],
                )

class _PlainOutputTests:
    # [CLASS DOC] Mixin with str/format expectations for enums that print as ClassName.member
    # (plain Enum and Flag types); relies on fixtures from _EnumTests.setUp.

    def test_str(self):
        # [FUNCTION DOC] Checks str() gives "MainEnum.name" (flags show zero as "MainEnum(0)"
        # and combined as joined names); aliases print under their canonical name.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(str(TE(0)), "MainEnum(0)")
            self.assertEqual(str(TE.dupe), "MainEnum.dupe")
            self.assertEqual(str(self.dupe2), "MainEnum.first|third")
        else:
            self.assertEqual(str(TE.dupe), "MainEnum.third")
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(str(member), "MainEnum.%s" % (member.name, ))

    def test_format(self):
        # [FUNCTION DOC] Checks format() with no spec gives the same "MainEnum.name" text
        # as str().
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(format(TE.dupe), "MainEnum.dupe")
            self.assertEqual(format(self.dupe2), "MainEnum.first|third")
        else:
            self.assertEqual(format(TE.dupe), "MainEnum.third")
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(format(member), "MainEnum.%s" % (member.name, ))

    def test_overridden_format(self):
        # [FUNCTION DOC] Checks that a __format__ override changes format() ("FIRST") but not
        # str() ("NewFormatEnum.first").
        NF = self.NewFormatEnum
        self.assertEqual(str(NF.first), "NewFormatEnum.first", '%s %r' % (NF.__str__, NF.first))
        self.assertEqual(format(NF.first), "FIRST")

    def test_format_specs(self):
        # [FUNCTION DOC] Checks that format specs (empty, width, alignment, precision) on a
        # member produce the same result as the spec applied to str(member).
        TE = self.MainEnum
        self.assertFormatIsStr('{}', TE.second)
        self.assertFormatIsStr('{:}', TE.second)
        self.assertFormatIsStr('{:20}', TE.second)
        self.assertFormatIsStr('{:^20}', TE.second)
        self.assertFormatIsStr('{:>20}', TE.second)
        self.assertFormatIsStr('{:<20}', TE.second)
        self.assertFormatIsStr('{:5.2}', TE.second)


class _MixedOutputTests:
    # [CLASS DOC] Mixin with str/format expectations for enums mixing in another data type but
    # not ReprEnum; str and format both use ClassName.member. Relies on _EnumTests fixtures.

    def test_str(self):
        # [FUNCTION DOC] Checks str() is "MainEnum.name" for each member, flag aliases and the
        # dupe alias.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(str(TE.dupe), "MainEnum.dupe")
            self.assertEqual(str(self.dupe2), "MainEnum.first|third")
        else:
            self.assertEqual(str(TE.dupe), "MainEnum.third")
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(str(member), "MainEnum.%s" % (member.name, ))

    def test_format(self):
        # [FUNCTION DOC] Checks format() with no spec is "MainEnum.name" for members
        # and aliases.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(format(TE.dupe), "MainEnum.dupe")
            self.assertEqual(format(self.dupe2), "MainEnum.first|third")
        else:
            self.assertEqual(format(TE.dupe), "MainEnum.third")
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(format(member), "MainEnum.%s" % (member.name, ))

    def test_overridden_format(self):
        # [FUNCTION DOC] Checks the __format__ override returns "FIRST" while str()
        # stays "NewFormatEnum.first".
        NF = self.NewFormatEnum
        self.assertEqual(str(NF.first), "NewFormatEnum.first")
        self.assertEqual(format(NF.first), "FIRST")

    def test_format_specs(self):
        # [FUNCTION DOC] Checks format specs on a member behave like the same spec
        # on str(member).
        TE = self.MainEnum
        self.assertFormatIsStr('{}', TE.first)
        self.assertFormatIsStr('{:}', TE.first)
        self.assertFormatIsStr('{:20}', TE.first)
        self.assertFormatIsStr('{:^20}', TE.first)
        self.assertFormatIsStr('{:>20}', TE.first)
        self.assertFormatIsStr('{:<20}', TE.first)
        self.assertFormatIsStr('{:5.2}', TE.first)


class _MinimalOutputTests:
    # [CLASS DOC] Mixin with expectations for enums using ReprEnum behavior (IntEnum, StrEnum,
    # IntFlag, ReprEnum mixins): str and format use the plain value. Also tests copying.

    def test_str(self):
        # [FUNCTION DOC] Checks str() of every member equals str() of its value (flag alias
        # values 3 and 5).
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(str(TE.dupe), "3")
            self.assertEqual(str(self.dupe2), "5")
        else:
            self.assertEqual(str(TE.dupe), str(self.values[2]))
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(str(member), str(value))

    def test_format(self):
        # [FUNCTION DOC] Checks format() of members equals format() of their values.
        TE = self.MainEnum
        if self.is_flag:
            self.assertEqual(format(TE.dupe), "3")
            self.assertEqual(format(self.dupe2), "5")
        else:
            self.assertEqual(format(TE.dupe), format(self.values[2]))
        for name, value, member in zip(self.names, self.values, TE, strict=True):
            self.assertEqual(format(member), format(value))

    def test_overridden_format(self):
        # [FUNCTION DOC] Checks str() yields the value text while the __format__ override
        # yields "FIRST".
        NF = self.NewFormatEnum
        self.assertEqual(str(NF.first), str(self.values[0]))
        self.assertEqual(format(NF.first), "FIRST")

    def test_format_specs(self):
        # [FUNCTION DOC] Checks format specs on a member match the same specs on its value; for
        # float mixins also the n, 5.2 and f specs.
        TE = self.MainEnum
        self.assertFormatIsValue('{}', TE.third)
        self.assertFormatIsValue('{:}', TE.third)
        self.assertFormatIsValue('{:20}', TE.third)
        self.assertFormatIsValue('{:^20}', TE.third)
        self.assertFormatIsValue('{:>20}', TE.third)
        self.assertFormatIsValue('{:<20}', TE.third)
        if TE._member_type_ is float:
            self.assertFormatIsValue('{:n}', TE.third)
            self.assertFormatIsValue('{:5.2}', TE.third)
            self.assertFormatIsValue('{:f}', TE.third)

    def test_copy(self):
        # [FUNCTION DOC] Checks copy.copy and copy.deepcopy of the enum class return the
        # class itself.
        TE = self.MainEnum
        copied = copy.copy(TE)
        self.assertEqual(copied, TE)
        self.assertIs(copied, TE)
        deep = copy.deepcopy(TE)
        self.assertEqual(deep, TE)
        self.assertIs(deep, TE)

    def test_copy_member(self):
        # [FUNCTION DOC] Checks copy.copy and copy.deepcopy of a member return the same
        # member object.
        TE = self.MainEnum
        copied = copy.copy(TE.first)
        self.assertIs(copied, TE.first)
        deep = copy.deepcopy(TE.first)
        self.assertIs(deep, TE.first)

class _FlagTests:
    # [CLASS DOC] Mixin with tests specific to Flag-based enums (default-missing error context
    # and inversion of closed/open flag sets); uses self.enum_type.

    def test_default_missing_with_wrong_type_value(self):
        # [FUNCTION DOC] Checks calling the flag enum with a non-integer value ('RED') raises
        # ValueError about an invalid value and that the exception has no chained __context__.
        with self.assertRaisesRegex(
            ValueError,
            "'RED' is not a valid ",
            ) as ctx:
            self.MainEnum('RED')
        self.assertIs(ctx.exception.__context__, None)

    def test_closed_invert_expectations(self):
        # [FUNCTION DOC] Checks inversion on closed flag enums (masks equal to all defined
        # bits): the complement of a member or combination is the remaining members and
        # inverting zero gives all bits.
        class ClosedAB(self.enum_type):
            # [CLASS DOC] HELPER: Flag enum with A=1, B=2 and MASK=3.
            A = 1
            B = 2
            MASK = 3
        A, B = ClosedAB
        AB_MASK = ClosedAB.MASK
        #
        self.assertIs(~A, B)
        self.assertIs(~B, A)
        self.assertIs(~(A|B), ClosedAB(0))
        self.assertIs(~AB_MASK, ClosedAB(0))
        self.assertIs(~ClosedAB(0), (A|B))
        #
        class ClosedXYZ(self.enum_type):
            # [CLASS DOC] HELPER: Flag enum with X=4, Y=2, Z=1 and MASK=7.
            X = 4
            Y = 2
            Z = 1
            MASK = 7
        X, Y, Z = ClosedXYZ
        XYZ_MASK = ClosedXYZ.MASK
        #
        self.assertIs(~X, Y|Z)
        self.assertIs(~Y, X|Z)
        self.assertIs(~Z, X|Y)
        self.assertIs(~(X|Y), Z)
        self.assertIs(~(X|Z), Y)
        self.assertIs(~(Y|Z), X)
        self.assertIs(~(X|Y|Z), ClosedXYZ(0))
        self.assertIs(~XYZ_MASK, ClosedXYZ(0))
        self.assertIs(~ClosedXYZ(0), (X|Y|Z))

    def test_open_invert_expectations(self):
        # [FUNCTION DOC] Checks inversion on open flag enums (mask 255 or 31 beyond defined
        # bits): with EJECT or KEEP boundaries the result includes undefined bits; otherwise it
        # is confined to defined members.
        class OpenAB(self.enum_type):
            # [CLASS DOC] HELPER: Flag enum with A=1, B=2 and MASK=255.
            A = 1
            B = 2
            MASK = 255
        A, B = OpenAB
        AB_MASK = OpenAB.MASK
        #
        if OpenAB._boundary_ in (EJECT, KEEP):
            self.assertIs(~A, OpenAB(254))
            self.assertIs(~B, OpenAB(253))
            self.assertIs(~(A|B), OpenAB(252))
            self.assertIs(~AB_MASK, OpenAB(0))
            self.assertIs(~OpenAB(0), AB_MASK)
        else:
            self.assertIs(~A, B)
            self.assertIs(~B, A)
            self.assertIs(~(A|B), OpenAB(0))
            self.assertIs(~AB_MASK, OpenAB(0))
            self.assertIs(~OpenAB(0), (A|B))
        #
        class OpenXYZ(self.enum_type):
            # [CLASS DOC] HELPER: Flag enum with X=4, Y=2, Z=1 and MASK=31.
            X = 4
            Y = 2
            Z = 1
            MASK = 31
        X, Y, Z = OpenXYZ
        XYZ_MASK = OpenXYZ.MASK
        #
        if OpenXYZ._boundary_ in (EJECT, KEEP):
            self.assertIs(~X, OpenXYZ(27))
            self.assertIs(~Y, OpenXYZ(29))
            self.assertIs(~Z, OpenXYZ(30))
            self.assertIs(~(X|Y), OpenXYZ(25))
            self.assertIs(~(X|Z), OpenXYZ(26))
            self.assertIs(~(Y|Z), OpenXYZ(28))
            self.assertIs(~(X|Y|Z), OpenXYZ(24))
            self.assertIs(~XYZ_MASK, OpenXYZ(0))
            self.assertTrue(~OpenXYZ(0), XYZ_MASK)
        else:
            self.assertIs(~X, Y|Z)
            self.assertIs(~Y, X|Z)
            self.assertIs(~Z, X|Y)
            self.assertIs(~(X|Y), Z)
            self.assertIs(~(X|Z), Y)
            self.assertIs(~(Y|Z), X)
            self.assertIs(~(X|Y|Z), OpenXYZ(0))
            self.assertIs(~XYZ_MASK, OpenXYZ(0))
            self.assertTrue(~OpenXYZ(0), (X|Y|Z))


class TestPlainEnumClass(_EnumTests, _PlainOutputTests, __TestCase):
    # [CLASS DOC] Runs the shared _EnumTests with the plain-output expectations for plain Enum
    # built with class statements (enum_type is Enum).
    enum_type = Enum

class TestPlainEnumFunction(_EnumTests, _PlainOutputTests, __TestCase):
    # [CLASS DOC] Same as TestPlainEnumClass but the fixtures are built with the functional API.
    enum_type = Enum


class TestPlainFlagClass(_EnumTests, _PlainOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Runs shared enum tests, plain output tests and Flag tests for Flag defined
    # with class statements (enum_type is Flag); adds a test for a None-valued member.
    enum_type = Flag

    def test_none_member(self):
        # [FUNCTION DOC] Checks a Flag member whose value is None keeps value None and raises
        # TypeError when combined with another flag using or, and, xor, or when inverted
        # (messages name the member).
        class FlagWithNoneMember(Flag):
            # [CLASS DOC] HELPER: Flag with a normal member A=1 and a member E whose value
            # is None.
            A = 1
            E = None

        self.assertEqual(FlagWithNoneMember.A.value, 1)
        self.assertIs(FlagWithNoneMember.E.value, None)
        with self.assertRaisesRegex(TypeError, r"'FlagWithNoneMember.E' cannot be combined with other flags with |"):
            FlagWithNoneMember.A | FlagWithNoneMember.E
        with self.assertRaisesRegex(TypeError, r"'FlagWithNoneMember.E' cannot be combined with other flags with &"):
            FlagWithNoneMember.E & FlagWithNoneMember.A
        with self.assertRaisesRegex(TypeError, r"'FlagWithNoneMember.E' cannot be combined with other flags with \^"):
            FlagWithNoneMember.A ^ FlagWithNoneMember.E
        with self.assertRaisesRegex(TypeError, r"'FlagWithNoneMember.E' cannot be inverted"):
            ~FlagWithNoneMember.E


class TestPlainFlagFunction(_EnumTests, _PlainOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Same as TestPlainFlagClass shared checks, with fixtures built via the
    # functional API.
    enum_type = Flag


class TestIntEnumClass(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Runs shared enum tests with minimal (ReprEnum style) output for IntEnum
    # defined by class statements; adds a test for member attributes shadowed by int attributes.
    enum_type = IntEnum
    #
    def test_shadowed_attr(self):
        # [FUNCTION DOC] Checks that members named divisor and numerator, which shadow int
        # attributes, still resolve: the int attribute on a member wins for the value (1) and a
        # member attribute lookup through another member returns the member.
        class Number(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members named divisor (1) and numerator (2).
            divisor = 1
            numerator = 2
        #
        self.assertEqual(Number.divisor.numerator, 1)
        self.assertIs(Number.numerator.divisor, Number.divisor)


class TestIntEnumFunction(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Same as TestIntEnumClass but with fixtures created by the functional API.
    enum_type = IntEnum
    #
    def test_shadowed_attr(self):
        # [FUNCTION DOC] Functional-API version of the shadowed-attribute check for IntEnum:
        # builds Number from the names divisor and numerator and checks attribute resolution
        # through members.
        Number = IntEnum('Number', ('divisor', 'numerator'))
        #
        self.assertEqual(Number.divisor.numerator, 1)
        self.assertIs(Number.numerator.divisor, Number.divisor)


class TestStrEnumClass(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Runs shared enum tests with minimal output for StrEnum defined with class
    # statements; adds a test for members that shadow str methods.
    enum_type = StrEnum
    #
    def test_shadowed_attr(self):
        # [FUNCTION DOC] Checks members named author and title: calling title() on a member
        # still uses the str method (giving the capitalized text) while attribute access via a
        # member returns the member.
        class Book(StrEnum):
            # [CLASS DOC] HELPER: StrEnum with members author and title with
            # equal-string values.
            author = 'author'
            title = 'title'
        #
        self.assertEqual(Book.author.title(), 'Author')
        self.assertEqual(Book.title.title(), 'Title')
        self.assertIs(Book.title.author, Book.author)


class TestStrEnumFunction(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Same as TestStrEnumClass but with the functional API.
    enum_type = StrEnum
    #
    def test_shadowed_attr(self):
        # [FUNCTION DOC] Functional-API version of the shadowed str attribute check using names
        # author and title.
        Book = StrEnum('Book', ('author', 'title'))
        #
        self.assertEqual(Book.author.title(), 'Author')
        self.assertEqual(Book.title.title(), 'Title')
        self.assertIs(Book.title.author, Book.author)


class TestIntFlagClass(_EnumTests, _MinimalOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Runs shared enum and Flag tests with minimal output for IntFlag defined by
    # class statements.
    enum_type = IntFlag


class TestIntFlagFunction(_EnumTests, _MinimalOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Same as TestIntFlagClass with the functional API.
    enum_type = IntFlag


class TestMixedIntClass(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Runs shared enum tests with mixed-output expectations for an Enum with an int
    # mixin defined via class syntax (nested enum_type is an empty int Enum subclass).
    class enum_type(int, Enum): pass


class TestMixedIntFunction(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Same as TestMixedIntClass with enum_type created through the functional
    # API (type=int).
    enum_type = Enum('enum_type', type=int)


class TestMixedStrClass(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum tests for an Enum with a str mixin (class syntax).
    class enum_type(str, Enum): pass


class TestMixedStrFunction(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum tests for a str-mixin Enum made via the
    # functional API.
    enum_type = Enum('enum_type', type=str)


class TestMixedIntFlagClass(_EnumTests, _MixedOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum and Flag tests for a Flag with an int mixin
    # (class syntax).
    class enum_type(int, Flag): pass


class TestMixedIntFlagFunction(_EnumTests, _MixedOutputTests, _FlagTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum and Flag tests for an int-mixin Flag built via the
    # functional API.
    enum_type = Flag('enum_type', type=int)


class TestMixedDateClass(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum tests for an Enum mixed with datetime.date. Class
    # attributes give the expected member values (date objects) and the tuple source values used
    # to construct them; enum_type generates tuples via _generate_next_value_.
    #
    values = [date(2021, 12, 25), date(2020, 3, 15), date(2019, 11, 27)]
    source_values = [(2021, 12, 25), (2020, 3, 15), (2019, 11, 27)]
    #
    class enum_type(date, Enum):
        # [CLASS DOC] HELPER: Enum with date mixin whose auto values are three fixed (year,
        # month, day) tuples.
        @staticmethod
        def _generate_next_value_(name, start, count, last_values):
            # [FUNCTION DOC] HELPER: Returns the date tuple at index count from a fixed list of
            # three tuples.
            values = [(2021, 12, 25), (2020, 3, 15), (2019, 11, 27)]
            return values[count]


class TestMixedDateFunction(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Same checks as TestMixedDateClass with enum_type built by the functional API
    # (type=date); the plain function below is made static by EnumType.
    #
    values = [date(2021, 12, 25), date(2020, 3, 15), date(2019, 11, 27)]
    source_values = [(2021, 12, 25), (2020, 3, 15), (2019, 11, 27)]
    #
    # staticmethod decorator will be added by EnumType if not present
    def _generate_next_value_(name, start, count, last_values):
        # [FUNCTION DOC] HELPER: Next-value generator returning the date tuple at index count
        # from three fixed tuples.
        values = [(2021, 12, 25), (2020, 3, 15), (2019, 11, 27)]
        return values[count]
    #
    enum_type = Enum('enum_type', {'_generate_next_value_':_generate_next_value_}, type=date)


class TestMinimalDateClass(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Minimal-output shared enum tests for a ReprEnum with date mixin (class
    # syntax), with its own expected date values and source tuples.
    #
    values = [date(2023, 12, 1), date(2016, 2, 29), date(2009, 1, 1)]
    source_values = [(2023, 12, 1), (2016, 2, 29), (2009, 1, 1)]
    #
    class enum_type(date, ReprEnum):
        # [CLASS DOC] HELPER: ReprEnum with date mixin that generates tuples for auto values.
        # staticmethod decorator will be added by EnumType if absent
        def _generate_next_value_(name, start, count, last_values):
            # [FUNCTION DOC] HELPER: Returns the date tuple at index count; made static by
            # EnumType if needed.
            values = [(2023, 12, 1), (2016, 2, 29), (2009, 1, 1)]
            return values[count]


class TestMinimalDateFunction(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Same as TestMinimalDateClass with enum_type built from the functional API.
    #
    values = [date(2023, 12, 1), date(2016, 2, 29), date(2009, 1, 1)]
    source_values = [(2023, 12, 1), (2016, 2, 29), (2009, 1, 1)]
    #
    @staticmethod
    def _generate_next_value_(name, start, count, last_values):
        # [FUNCTION DOC] HELPER: Static next-value generator returning the date tuple at
        # index count.
        values = [(2023, 12, 1), (2016, 2, 29), (2009, 1, 1)]
        return values[count]
    #
    enum_type = ReprEnum('enum_type', {'_generate_next_value_':_generate_next_value_}, type=date)


class TestMixedFloatClass(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Mixed-output shared enum tests for an Enum with a float mixin (class syntax)
    # whose auto values are 1.1, 2.2, 3.3.
    #
    values = [1.1, 2.2, 3.3]
    #
    class enum_type(float, Enum):
        # [CLASS DOC] HELPER: Float-mixin Enum whose generated values come from a fixed list.
        def _generate_next_value_(name, start, count, last_values):
            # [FUNCTION DOC] HELPER: Returns the float at index count from the list 1.1,
            # 2.2, 3.3.
            values = [1.1, 2.2, 3.3]
            return values[count]


class TestMixedFloatFunction(_EnumTests, _MixedOutputTests, __TestCase):
    # [CLASS DOC] Same as TestMixedFloatClass using the functional API.
    #
    values = [1.1, 2.2, 3.3]
    #
    def _generate_next_value_(name, start, count, last_values):
        # [FUNCTION DOC] HELPER: Returns the float at index count from the list 1.1, 2.2, 3.3.
        values = [1.1, 2.2, 3.3]
        return values[count]
    #
    enum_type = Enum('enum_type', {'_generate_next_value_':_generate_next_value_}, type=float)


class TestMinimalFloatClass(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Minimal-output shared enum tests for a float-mixin ReprEnum (class syntax)
    # with auto values 4.4, 5.5, 6.6.
    #
    values = [4.4, 5.5, 6.6]
    #
    class enum_type(float, ReprEnum):
        # [CLASS DOC] HELPER: Float-mixin ReprEnum generating values from a fixed list.
        def _generate_next_value_(name, start, count, last_values):
            # [FUNCTION DOC] HELPER: Returns the float at index count from the list 4.4,
            # 5.5, 6.6.
            values = [4.4, 5.5, 6.6]
            return values[count]


class TestMinimalFloatFunction(_EnumTests, _MinimalOutputTests, __TestCase):
    # [CLASS DOC] Same as TestMinimalFloatClass using the functional API.
    #
    values = [4.4, 5.5, 6.6]
    #
    def _generate_next_value_(name, start, count, last_values):
        # [FUNCTION DOC] HELPER: Returns the float at index count from the list 4.4, 5.5, 6.6.
        values = [4.4, 5.5, 6.6]
        return values[count]
    #
    enum_type = ReprEnum('enum_type', {'_generate_next_value_':_generate_next_value_}, type=float)


class TestSpecial(__TestCase):
    # [CLASS DOC] Tests for behaviors specific to certain enums (bool, comparisons, duplicate
    # names, nonmember/member, partial, mixed types, functional API options, etc). setUp
    # provides Season, Grades (IntEnum), Directional (str Enum) and Holiday (date Enum).
    """
    various operations that are not attributable to every possible enum
    """

    def setUp(self):
        # [FUNCTION DOC] Builds fixture enums on the test instance: Season (plain, four
        # members), Grades (IntEnum A..F), Directional (str mixin) and Holiday (date mixin with
        # tuple values).
        super().setUp()
        class Season(Enum):
            # [CLASS DOC] HELPER: Plain Enum with SPRING, SUMMER, AUTUMN, WINTER.
            SPRING = 1
            SUMMER = 2
            AUTUMN = 3
            WINTER = 4
        self.Season = Season
        #
        class Grades(IntEnum):
            # [CLASS DOC] HELPER: IntEnum grades A=5, B=4, C=3, D=2, F=0.
            A = 5
            B = 4
            C = 3
            D = 2
            F = 0
        self.Grades = Grades
        #
        class Directional(str, Enum):
            # [CLASS DOC] HELPER: str-mixin Enum with compass direction strings.
            EAST = 'east'
            WEST = 'west'
            NORTH = 'north'
            SOUTH = 'south'
        self.Directional = Directional
        #
        from datetime import date
        class Holiday(date, Enum):
            # [CLASS DOC] HELPER: date-mixin Enum with NEW_YEAR and IDES_OF_MARCH given
            # as tuples.
            NEW_YEAR = 2013, 1, 1
            IDES_OF_MARCH = 2013, 3, 15
        self.Holiday = Holiday

    def test_bool(self):
        # [FUNCTION DOC] Checks truthiness: plain Enum members are always true, a custom
        # __bool__ using the value takes effect, and mixed-in int members follow the int
        # truth value.
        # plain Enum members are always True
        class Logic(Enum):
            # [CLASS DOC] HELPER: Plain Enum with boolean values true and false.
            true = True
            false = False
        self.assertTrue(Logic.true)
        self.assertTrue(Logic.false)
        # unless overridden
        class RealLogic(Enum):
            # [CLASS DOC] HELPER: Enum with boolean values and __bool__ returning bool of
            # the value.
            true = True
            false = False
            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns the truth value of the member's value.
                return bool(self._value_)
        self.assertTrue(RealLogic.true)
        self.assertFalse(RealLogic.false)
        # mixed Enums depend on mixed-in type
        class IntLogic(int, Enum):
            # [CLASS DOC] HELPER: int-mixin Enum with members true=1 and false=0.
            true = 1
            false = 0
        self.assertTrue(IntLogic.true)
        self.assertFalse(IntLogic.false)

    def test_comparisons(self):
        # [FUNCTION DOC] Checks that members of plain Enums do not support ordering (TypeError)
        # and are not equal to raw values or to same-named members of another enum.
        Season = self.Season
        with self.assertRaises(TypeError):
            Season.SPRING < Season.WINTER
        with self.assertRaises(TypeError):
            Season.SPRING > 4
        #
        self.assertNotEqual(Season.SPRING, 1)
        #
        class Part(Enum):
            # [CLASS DOC] HELPER: Plain Enum with SPRING, CLIP, BARREL members.
            SPRING = 1
            CLIP = 2
            BARREL = 3
        #
        self.assertNotEqual(Season.SPRING, Part.SPRING)
        with self.assertRaises(TypeError):
            Season.SPRING < Part.CLIP

    @unittest.skip('to-do list')
    def test_dir_with_custom_dunders(self):
        # [FUNCTION DOC] Skipped to-do test: dir() of a plain Enum should omit custom dunders,
        # but include __repr__, __str__, __format__ and __init__ when the enum defines them.
        class PlainEnum(Enum):
            # [CLASS DOC] HELPER: Empty plain Enum.
            pass
        cls_dir = dir(PlainEnum)
        self.assertNotIn('__repr__', cls_dir)
        self.assertNotIn('__str__', cls_dir)
        self.assertNotIn('__format__', cls_dir)
        self.assertNotIn('__init__', cls_dir)
        #
        class MyEnum(Enum):
            # [CLASS DOC] HELPER: Enum that defines __repr__, __str__, __format__ and
            # __init__ methods.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the default object repr of the member.
                return object.__repr__(self)
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the default object repr of the member.
                return object.__repr__(self)
            def __format__(self):
                # [FUNCTION DOC] HELPER: Returns the default object repr of the member (takes no
                # spec argument).
                return object.__repr__(self)
            def __init__(self):
                # [FUNCTION DOC] HELPER: Empty initializer.
                pass
        cls_dir = dir(MyEnum)
        self.assertIn('__repr__', cls_dir)
        self.assertIn('__str__', cls_dir)
        self.assertIn('__format__', cls_dir)
        self.assertIn('__init__', cls_dir)

    def test_duplicate_name_error(self):
        # [FUNCTION DOC] Checks that reusing a member name (second assignment, a method with the
        # same name, or a property followed by a member) raises TypeError at class creation.
        with self.assertRaises(TypeError):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum redefining the member red; creation must
                # raise TypeError.
                red = 1
                green = 2
                blue = 3
                red = 4
        #
        with self.assertRaises(TypeError):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum where a method named red replaces the
                # member red.
                red = 1
                green = 2
                blue = 3
                def red(self):  # noqa: F811
                    # [FUNCTION DOC] HELPER: Method returning 'red'; collides with a
                    # member name.
                    return 'red'
        #
        with self.assertRaises(TypeError):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum where a property named red is followed by a
                # member red.
                @enum.property
                def red(self):
                    # [FUNCTION DOC] HELPER: Property getter returning 'redder'; collides with a
                    # member name.
                    return 'redder'
                red = 1  # noqa: F811
                green = 2
                blue = 3

    @reraise_if_not_enum(Theory)
    def test_enum_function_with_qualname(self):
        # [FUNCTION DOC] Checks the enum created via the functional API with a qualname keyword
        # has that __qualname__; reraises the stored creation exception if the module level
        # fixture failed.
        self.assertEqual(Theory.__qualname__, 'spanish_inquisition')

    def test_enum_of_types(self):
        # [FUNCTION DOC] Checks enums may hold types (int, float, str, user classes) as
        # member values.
        """Support using Enum to refer to types deliberately."""
        class MyTypes(Enum):
            # [CLASS DOC] HELPER: Enum whose values are builtin types.
            i = int
            f = float
            s = str
        self.assertEqual(MyTypes.i.value, int)
        self.assertEqual(MyTypes.f.value, float)
        self.assertEqual(MyTypes.s.value, str)
        class Foo:
            # [CLASS DOC] HELPER: Empty plain class used as an enum value.
            pass
        class Bar:
            # [CLASS DOC] HELPER: Empty plain class used as an enum value.
            pass
        class MyTypes2(Enum):
            # [CLASS DOC] HELPER: Enum whose values are the classes Foo and Bar.
            a = Foo
            b = Bar
        self.assertEqual(MyTypes2.a.value, Foo)
        self.assertEqual(MyTypes2.b.value, Bar)
        class SpamEnumNotInner:
            # [CLASS DOC] HELPER: Empty plain class used as an enum value.
            pass
        class SpamEnum(Enum):
            # [CLASS DOC] HELPER: Enum with one member whose value is a class defined outside
            # the enum.
            spam = SpamEnumNotInner
        self.assertEqual(SpamEnum.spam.value, SpamEnumNotInner)

    def test_enum_of_generic_aliases(self):
        # [FUNCTION DOC] Checks generic alias objects (typing List of int, list of int) can be
        # member values and show up in the member repr.
        class E(Enum):
            # [CLASS DOC] HELPER: Enum with two generic alias values.
            a = typing.List[int]
            b = list[int]
        self.assertEqual(E.a.value, typing.List[int])
        self.assertEqual(E.b.value, list[int])
        self.assertEqual(repr(E.a), '<E.a: typing.List[int]>')
        self.assertEqual(repr(E.b), '<E.b: list[int]>')

    @unittest.skipIf(
            python_version >= (3, 13),
            'inner classes are not members',
            )
    def test_nested_classes_in_enum_are_members(self):
        # [FUNCTION DOC] Skipped on Python 3.13 or later. Checks that before 3.13 a nested class
        # in an enum body emits a DeprecationWarning and becomes a member whose value is the
        # nested enum.
        """
        Check for warnings pre-3.13
        """
        with self.assertWarnsRegex(DeprecationWarning, 'will not become a member'):
            class Outer(Enum):
                # [CLASS DOC] HELPER: Enum with members a, b and a nested enum class Inner.
                a = 1
                b = 2
                class Inner(Enum):
                    # [CLASS DOC] HELPER: Nested Enum with members foo=10 and bar=11.
                    foo = 10
                    bar = 11
        self.assertTrue(isinstance(Outer.Inner, Outer))
        self.assertEqual(Outer.a.value, 1)
        self.assertEqual(Outer.Inner.value.foo.value, 10)
        self.assertEqual(
            list(Outer.Inner.value),
            [Outer.Inner.value.foo, Outer.Inner.value.bar],
            )
        self.assertEqual(
            list(Outer),
            [Outer.a, Outer.b, Outer.Inner],
            )

    @unittest.skipIf(
            python_version < (3, 13),
            'inner classes are still members',
            )
    def test_nested_classes_in_enum_are_not_members(self):
        # [FUNCTION DOC] Skipped before 3.13. Checks that from 3.13 on a nested class in an enum
        # body is not a member; Outer lists only a and b while Inner stays a normal class.
        """Support locally-defined nested classes."""
        class Outer(Enum):
            # [CLASS DOC] HELPER: Enum with members a, b and a nested enum class Inner (not
            # a member).
            a = 1
            b = 2
            class Inner(Enum):
                # [CLASS DOC] HELPER: Nested Enum with foo=10 and bar=11.
                foo = 10
                bar = 11
        self.assertTrue(isinstance(Outer.Inner, type))
        self.assertEqual(Outer.a.value, 1)
        self.assertEqual(Outer.Inner.foo.value, 10)
        self.assertEqual(
            list(Outer.Inner),
            [Outer.Inner.foo, Outer.Inner.bar],
            )
        self.assertEqual(
            list(Outer),
            [Outer.a, Outer.b],
            )

    def test_nested_classes_in_enum_with_nonmember(self):
        # [FUNCTION DOC] Checks that decorating a nested class with nonmember keeps it a plain
        # class and out of the member list.
        class Outer(Enum):
            # [CLASS DOC] HELPER: Enum with a, b and a nonmember-decorated nested enum Inner.
            a = 1
            b = 2
            @nonmember
            class Inner(Enum):
                # [CLASS DOC] HELPER: Nested Enum with foo=10 and bar=11.
                foo = 10
                bar = 11
        self.assertTrue(isinstance(Outer.Inner, type))
        self.assertEqual(Outer.a.value, 1)
        self.assertEqual(Outer.Inner.foo.value, 10)
        self.assertEqual(
            list(Outer.Inner),
            [Outer.Inner.foo, Outer.Inner.bar],
            )
        self.assertEqual(
            list(Outer),
            [Outer.a, Outer.b],
            )

    def test_enum_of_types_with_nonmember(self):
        # [FUNCTION DOC] Checks that wrapping a value in nonmember leaves the raw object as a
        # class attribute (not a member) while other types remain members.
        """Support using Enum to refer to types deliberately."""
        class MyTypes(Enum):
            # [CLASS DOC] HELPER: Enum with type values where f is wrapped in nonmember.
            i = int
            f = nonmember(float)
            s = str
        self.assertEqual(MyTypes.i.value, int)
        self.assertTrue(MyTypes.f is float)
        self.assertEqual(MyTypes.s.value, str)
        class Foo:
            # [CLASS DOC] HELPER: Empty class used as a value.
            pass
        class Bar:
            # [CLASS DOC] HELPER: Empty class used as a value.
            pass
        class MyTypes2(Enum):
            # [CLASS DOC] HELPER: Enum with member a=Foo and nonmember b=Bar.
            a = Foo
            b = nonmember(Bar)
        self.assertEqual(MyTypes2.a.value, Foo)
        self.assertTrue(MyTypes2.b is Bar)
        class SpamEnumIsInner:
            # [CLASS DOC] HELPER: Empty class used as a nonmember attribute value.
            pass
        class SpamEnum(Enum):
            # [CLASS DOC] HELPER: Enum with a single nonmember spam attribute.
            spam = nonmember(SpamEnumIsInner)
        self.assertTrue(SpamEnum.spam is SpamEnumIsInner)

    def test_using_members_as_nonmember(self):
        # [FUNCTION DOC] Checks a nonmember attribute computed from earlier flag members (their
        # union) stays a plain int attribute equal to 3 and not a Flag member, with explicit and
        # auto values.
        class Example(Flag):
            # [CLASS DOC] HELPER: Flag with A=1, B=2 and ALL as nonmember union.
            A = 1
            B = 2
            ALL = nonmember(A | B)

        self.assertEqual(Example.A.value, 1)
        self.assertEqual(Example.B.value, 2)
        self.assertEqual(Example.ALL, 3)
        self.assertIs(type(Example.ALL), int)

        class Example(Flag):
            # [CLASS DOC] HELPER: Flag with auto members A, B and ALL as nonmember union.
            A = auto()
            B = auto()
            ALL = nonmember(A | B)

        self.assertEqual(Example.A.value, 1)
        self.assertEqual(Example.B.value, 2)
        self.assertEqual(Example.ALL, 3)
        self.assertIs(type(Example.ALL), int)

    def test_nested_classes_in_enum_with_member(self):
        # [FUNCTION DOC] Checks that decorating a nested class with member makes it a regular
        # member whose value is the nested enum class, and it is listed among members.
        """Support locally-defined nested classes."""
        class Outer(Enum):
            # [CLASS DOC] HELPER: Enum with a, b and a member-decorated nested enum Inner.
            a = 1
            b = 2
            @member
            class Inner(Enum):
                # [CLASS DOC] HELPER: Nested Enum with foo=10 and bar=11.
                foo = 10
                bar = 11
        self.assertTrue(isinstance(Outer.Inner, Outer))
        self.assertEqual(Outer.a.value, 1)
        self.assertEqual(Outer.Inner.value.foo.value, 10)
        self.assertEqual(
            list(Outer.Inner.value),
            [Outer.Inner.value.foo, Outer.Inner.value.bar],
            )
        self.assertEqual(
            list(Outer),
            [Outer.a, Outer.b, Outer.Inner],
            )

    def test_partial(self):
        # [FUNCTION DOC] Checks that a functools.partial in an enum body emits a FutureWarning
        # (with the warning attributed to this file) and stays a callable attribute, not a
        # member; accessing it through a member warns again.
        def func(a, b=5):
            # [FUNCTION DOC] HELPER: Returns the tuple of its two arguments, b defaulting to 5.
            return a, b
        with self.assertWarnsRegex(FutureWarning, r'partial.*enum\.member') as cm:
            class E(Enum):
                # [CLASS DOC] HELPER: Enum with member a and a partial attribute b.
                a = 1
                b = partial(func)
        self.assertEqual(cm.filename, __file__)
        self.assertIsInstance(E.b, partial)
        self.assertEqual(E.b(2), (2, 5))
        with self.assertWarnsRegex(FutureWarning, 'partial'):
            self.assertEqual(E.a.b(2), (2, 5))

    def test_enum_with_value_name(self):
        # [FUNCTION DOC] Checks members named name and value are allowed; they are real members
        # whose own name and value attributes still work.
        class Huh(Enum):
            # [CLASS DOC] HELPER: Enum with members named name (1) and value (2).
            name = 1
            value = 2
        self.assertEqual(list(Huh), [Huh.name, Huh.value])
        self.assertIs(type(Huh.name), Huh)
        self.assertEqual(Huh.name.name, 'name')
        self.assertEqual(Huh.name.value, 1)

    def test_contains_name_and_value_overlap(self):
        # [FUNCTION DOC] Checks containment among IntEnums: a member is contained in another
        # IntEnum with an equal value regardless of names, and not when the value differs.
        class IntEnum1(IntEnum):
            # [CLASS DOC] HELPER: IntEnum X=1.
            X = 1
        class IntEnum2(IntEnum):
            # [CLASS DOC] HELPER: IntEnum X=1 (same value as IntEnum1).
            X = 1
        class IntEnum3(IntEnum):
            # [CLASS DOC] HELPER: IntEnum X=2.
            X = 2
        class IntEnum4(IntEnum):
            # [CLASS DOC] HELPER: IntEnum Y=1.
            Y = 1
        self.assertIn(IntEnum1.X, IntEnum1)
        self.assertIn(IntEnum1.X, IntEnum2)
        self.assertNotIn(IntEnum1.X, IntEnum3)
        self.assertIn(IntEnum1.X, IntEnum4)

    def test_contains_different_types_same_members(self):
        # [FUNCTION DOC] Checks an IntEnum member is contained in an IntFlag with the same value
        # and vice versa.
        class IntEnum1(IntEnum):
            # [CLASS DOC] HELPER: IntEnum X=1.
            X = 1
        class IntFlag1(IntFlag):
            # [CLASS DOC] HELPER: IntFlag X=1.
            X = 1
        self.assertIn(IntEnum1.X, IntFlag1)
        self.assertIn(IntFlag1.X, IntEnum1)

    def test_contains_does_not_call_missing(self):
        # [FUNCTION DOC] Checks that the in operator does not invoke _missing_: None and 3 are
        # contained, 7 is not even though _missing_ would return a member.
        class AnEnum(Enum):
            # [CLASS DOC] HELPER: Enum with UNKNOWN=None, LUCKY=3 and a _missing_ hook
            # returning UNKNOWN.
            UNKNOWN = None
            LUCKY = 3
            @classmethod
            def _missing_(cls, *values):
                # [FUNCTION DOC] HELPER: Classmethod returning the UNKNOWN member for any value.
                return cls.UNKNOWN
        self.assertTrue(None in AnEnum)
        self.assertTrue(3 in AnEnum)
        self.assertFalse(7 in AnEnum)

    def test_inherited_data_type(self):
        # [FUNCTION DOC] Checks enums with user-defined int mixins: repr uses the mixin repr
        # (hex), the mixin type is recorded as _member_type_, and members and classes survive a
        # pickle round trip (classes are registered in module globals for lookup), including a
        # mixin with its own __new__.
        class HexInt(int):
            # [CLASS DOC] HELPER: int subclass whose repr is hexadecimal.
            __qualname__ = 'HexInt'
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the hex text of the integer.
                return hex(self)
        class MyEnum(HexInt, enum.Enum):
            # [CLASS DOC] HELPER: Enum with HexInt mixin and members A, B, C.
            __qualname__ = 'MyEnum'
            A = 1
            B = 2
            C = 3
        self.assertEqual(repr(MyEnum.A), '<MyEnum.A: 0x1>')
        globals()['HexInt'] = HexInt
        globals()['MyEnum'] = MyEnum
        test_pickle_dump_load(self.assertIs, MyEnum.A)
        test_pickle_dump_load(self.assertIs, MyEnum)
        #
        class SillyInt(HexInt):
            # [CLASS DOC] HELPER: Empty subclass of HexInt.
            __qualname__ = 'SillyInt'
        class MyOtherEnum(SillyInt, enum.Enum):
            # [CLASS DOC] HELPER: Enum with SillyInt mixin and members D, E, F.
            __qualname__ = 'MyOtherEnum'
            D = 4
            E = 5
            F = 6
        self.assertIs(MyOtherEnum._member_type_, SillyInt)
        globals()['SillyInt'] = SillyInt
        globals()['MyOtherEnum'] = MyOtherEnum
        test_pickle_dump_load(self.assertIs, MyOtherEnum.E)
        test_pickle_dump_load(self.assertIs, MyOtherEnum)
        #
        # This did not work in 3.10, but does now with pickling by name
        class UnBrokenInt(int):
            # [CLASS DOC] HELPER: int subclass with a custom __new__ forwarding to int.__new__.
            __qualname__ = 'UnBrokenInt'
            def __new__(cls, value):
                # [FUNCTION DOC] HELPER: Builds an int instance of cls from value.
                return int.__new__(cls, value)
        class MyUnBrokenEnum(UnBrokenInt, Enum):
            # [CLASS DOC] HELPER: Enum with UnBrokenInt mixin and members G, H, I.
            __qualname__ = 'MyUnBrokenEnum'
            G = 7
            H = 8
            I = 9
        self.assertIs(MyUnBrokenEnum._member_type_, UnBrokenInt)
        self.assertIs(MyUnBrokenEnum(7), MyUnBrokenEnum.G)
        globals()['UnBrokenInt'] = UnBrokenInt
        globals()['MyUnBrokenEnum'] = MyUnBrokenEnum
        test_pickle_dump_load(self.assertIs, MyUnBrokenEnum.I)
        test_pickle_dump_load(self.assertIs, MyUnBrokenEnum)

    @reraise_if_not_enum(FloatStooges)
    def test_floatenum_fromhex(self):
        # [FUNCTION DOC] Checks the float-mixin enum's inherited fromhex classmethod returns the
        # member for an exact hex of a member value and raises ValueError for a non-member
        # value. Reraises stored import error if FloatStooges failed.
        h = float.hex(FloatStooges.MOE.value)
        self.assertIs(FloatStooges.fromhex(h), FloatStooges.MOE)
        h = float.hex(FloatStooges.MOE.value + 0.01)
        with self.assertRaises(ValueError):
            FloatStooges.fromhex(h)

    def test_programmatic_function_type(self):
        # [FUNCTION DOC] Checks the functional API with type=int: count, order, members equal 1,
        # 2, 3 with matching names, membership and member type.
        MinorEnum = Enum('MinorEnum', 'june july august', type=int)
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 1):
            e = MinorEnum(i)
            self.assertEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_string_with_start(self):
        # [FUNCTION DOC] Checks the functional API with start=10: values begin at 10 but members
        # do not compare equal to the ints.
        MinorEnum = Enum('MinorEnum', 'june july august', start=10)
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 10):
            e = MinorEnum(i)
            self.assertEqual(int(e.value), i)
            self.assertNotEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_type_with_start(self):
        # [FUNCTION DOC] Checks the functional API with type=int and start=30: members equal 30,
        # 31, 32.
        MinorEnum = Enum('MinorEnum', 'june july august', type=int, start=30)
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 30):
            e = MinorEnum(i)
            self.assertEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_string_list_with_start(self):
        # [FUNCTION DOC] Checks the functional API with a list of names and start=20: values
        # begin at 20 and members do not compare equal to the ints.
        MinorEnum = Enum('MinorEnum', ['june', 'july', 'august'], start=20)
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 20):
            e = MinorEnum(i)
            self.assertEqual(int(e.value), i)
            self.assertNotEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_type_from_subclass(self):
        # [FUNCTION DOC] Checks the functional API on IntEnum without type argument yields int
        # members 1..3.
        MinorEnum = IntEnum('MinorEnum', 'june july august')
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 1):
            e = MinorEnum(i)
            self.assertEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_type_from_subclass_with_start(self):
        # [FUNCTION DOC] Checks the functional API on IntEnum with start=40: members equal 40,
        # 41, 42.
        MinorEnum = IntEnum('MinorEnum', 'june july august', start=40)
        lst = list(MinorEnum)
        self.assertEqual(len(lst), len(MinorEnum))
        self.assertEqual(len(MinorEnum), 3, MinorEnum)
        self.assertEqual(
                [MinorEnum.june, MinorEnum.july, MinorEnum.august],
                lst,
                )
        for i, month in enumerate('june july august'.split(), 40):
            e = MinorEnum(i)
            self.assertEqual(e, i)
            self.assertEqual(e.name, month)
            self.assertIn(e, MinorEnum)
            self.assertIs(type(e), MinorEnum)

    def test_programmatic_function_is_value_call(self):
        # [FUNCTION DOC] Checks that for tuple-valued members calling the enum with the tuple
        # or with the unpacked arguments finds the member, while a single non-tuple argument
        # raises ValueError.
        class TwoPart(Enum):
            # [CLASS DOC] HELPER: Enum with two-element tuple values.
            ONE = 1, 1.0
            TWO = 2, 2.0
            THREE = 3, 3.0
        self.assertRaisesRegex(ValueError, '1 is not a valid .*TwoPart', TwoPart, 1)
        self.assertIs(TwoPart((1, 1.0)), TwoPart.ONE)
        self.assertIs(TwoPart(1, 1.0), TwoPart.ONE)
        class ThreePart(Enum):
            # [CLASS DOC] HELPER: Enum with three-element tuple values.
            ONE = 1, 1.0, 'one'
            TWO = 2, 2.0, 'two'
            THREE = 3, 3.0, 'three'
        self.assertIs(ThreePart((3, 3.0, 'three')), ThreePart.THREE)
        self.assertIs(ThreePart(3, 3.0, 'three'), ThreePart.THREE)

    @reraise_if_not_enum(IntStooges)
    def test_intenum_from_bytes(self):
        # [FUNCTION DOC] Checks the IntEnum classmethod from_bytes returns the member for a
        # valid big-endian value and raises ValueError for an unknown value. Reraises the stored
        # error if IntStooges failed.
        self.assertIs(IntStooges.from_bytes(b'\x00\x03', 'big'), IntStooges.MOE)
        with self.assertRaises(ValueError):
            IntStooges.from_bytes(b'\x00\x05', 'big')

    def test_reserved_sunder_error(self):
        # [FUNCTION DOC] Checks that defining a member with an unknown sunder name raises
        # ValueError saying sunder names are reserved.
        with self.assertRaisesRegex(
                ValueError,
                '_sunder_ names, such as ._bad_., are reserved',
            ):
            class Bad(Enum):
                # [CLASS DOC] HELPER: Invalid enum with sunder member _bad_.
                _bad_ = 1

    def test_too_many_data_types(self):
        # [FUNCTION DOC] Checks that mixing two data types (str and int, or two user subclasses)
        # in the bases of an enum raises TypeError ("too many data types").
        with self.assertRaisesRegex(TypeError, 'too many data types'):
            class Huh(str, int, Enum):
                # [CLASS DOC] HELPER: Invalid enum with bases str and int.
                One = 1

        class MyStr(str):
            # [CLASS DOC] HELPER: str subclass with a hello method.
            def hello(self):
                # [FUNCTION DOC] HELPER: Returns a greeting containing the string.
                return 'hello, %s' % self
        class MyInt(int):
            # [CLASS DOC] HELPER: int subclass with a method named repr.
            def repr(self):
                # [FUNCTION DOC] HELPER: Returns the hex text of the integer.
                return hex(self)
        with self.assertRaisesRegex(TypeError, 'too many data types'):
            class Huh(MyStr, MyInt, Enum):
                # [CLASS DOC] HELPER: Invalid enum with bases MyStr and MyInt.
                One = 1

    @reraise_if_not_enum(Stooges)
    def test_pickle_enum(self):
        # [FUNCTION DOC] Pickle round trip for a member and the class of the module-level
        # Stooges enum.
        test_pickle_dump_load(self.assertIs, Stooges.CURLY)
        test_pickle_dump_load(self.assertIs, Stooges)

    @reraise_if_not_enum(IntStooges)
    def test_pickle_int(self):
        # [FUNCTION DOC] Pickle round trip for a member and the class of IntStooges.
        test_pickle_dump_load(self.assertIs, IntStooges.CURLY)
        test_pickle_dump_load(self.assertIs, IntStooges)

    @reraise_if_not_enum(FloatStooges)
    def test_pickle_float(self):
        # [FUNCTION DOC] Pickle round trip for a member and the class of FloatStooges.
        test_pickle_dump_load(self.assertIs, FloatStooges.CURLY)
        test_pickle_dump_load(self.assertIs, FloatStooges)

    @reraise_if_not_enum(Answer)
    def test_pickle_enum_function(self):
        # [FUNCTION DOC] Pickle round trip for a member and the class of the functional-API enum
        # Answer (no module given; module is found by frame inspection).
        test_pickle_dump_load(self.assertIs, Answer.him)
        test_pickle_dump_load(self.assertIs, Answer)

    @reraise_if_not_enum(Question)
    def test_pickle_enum_function_with_module(self):
        # [FUNCTION DOC] Pickle round trip for a member and the class of the functional-API enum
        # Question (module passed explicitly).
        test_pickle_dump_load(self.assertIs, Question.who)
        test_pickle_dump_load(self.assertIs, Question)

    def test_pickle_nested_class(self):
        # [FUNCTION DOC] Checks pickling a member of an enum nested in a test class. Defines
        # NestedEnum, attaches it to the test class (side effect on the class) with a qualified
        # name, and round trips a member.
        # would normally just have this directly in the class namespace
        class NestedEnum(Enum):
            # [CLASS DOC] HELPER: Enum with members twigs and shiny with string values.
            twigs = 'common'
            shiny = 'rare'

        self.__class__.NestedEnum = NestedEnum
        self.NestedEnum.__qualname__ = '%s.NestedEnum' % self.__class__.__name__
        test_pickle_dump_load(self.assertIs, self.NestedEnum.twigs)

    def test_pickle_by_name(self):
        # [FUNCTION DOC] Checks setting __reduce_ex__ to the by-global-name reducer makes
        # reduction of a member return just its name for every protocol below the highest.
        class ReplaceGlobalInt(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with ONE and TWO.
            ONE = 1
            TWO = 2
        ReplaceGlobalInt.__reduce_ex__ = enum._reduce_ex_by_global_name
        for proto in range(HIGHEST_PROTOCOL):
            self.assertEqual(ReplaceGlobalInt.TWO.__reduce_ex__(proto), 'TWO')

    def test_pickle_explodes(self):
        # [FUNCTION DOC] Creates enum BadPickle via the functional API, registers it in module
        # globals, marks it unpicklable, and checks pickling a member raises TypeError and
        # pickling the class raises PicklingError.
        BadPickle = Enum(
                'BadPickle', 'dill sweet bread-n-butter', module=__name__)
        globals()['BadPickle'] = BadPickle
        # now break BadPickle to test exception raising
        enum._make_class_unpicklable(BadPickle)
        test_pickle_exception(self.assertRaises, TypeError, BadPickle.dill)
        test_pickle_exception(self.assertRaises, PicklingError, BadPickle)

    def test_string_enum(self):
        # [FUNCTION DOC] Checks a str-mixin enum member equals its string value.
        class SkillLevel(str, Enum):
            # [CLASS DOC] HELPER: str-mixin Enum with three skill level phrases.
            master = 'what is the sound of one hand clapping?'
            journeyman = 'why did the chicken cross the road?'
            apprentice = 'knock, knock!'
        self.assertEqual(SkillLevel.apprentice, 'knock, knock!')

    def test_getattr_getitem(self):
        # [FUNCTION DOC] Checks lookup three ways: by value call, by getattr and by item name.
        class Period(Enum):
            # [CLASS DOC] HELPER: Enum with morning, noon, evening, night.
            morning = 1
            noon = 2
            evening = 3
            night = 4
        self.assertIs(Period(2), Period.noon)
        self.assertIs(getattr(Period, 'night'), Period.night)
        self.assertIs(Period['morning'], Period.morning)

    def test_getattr_dunder(self):
        # [FUNCTION DOC] Checks that a dunder attribute (__eq__) can be fetched from an enum
        # class via getattr.
        Season = self.Season
        self.assertTrue(getattr(Season, '__eq__'))

    def test_iteration_order(self):
        # [FUNCTION DOC] Checks that iteration order follows definition order, not value order.
        class Season(Enum):
            # [CLASS DOC] HELPER: Enum defined in the order SUMMER=2, WINTER=4,
            # AUTUMN=3, SPRING=1.
            SUMMER = 2
            WINTER = 4
            AUTUMN = 3
            SPRING = 1
        self.assertEqual(
                list(Season),
                [Season.SUMMER, Season.WINTER, Season.AUTUMN, Season.SPRING],
                )

    @reraise_if_not_enum(Name)
    def test_subclassing(self):
        # [FUNCTION DOC] Checks the module-level StrEnum Name: member equals its string value,
        # value lookup returns the member, attribute lookup matches, and it round-trips
        # through pickle.
        self.assertEqual(Name.BDFL, 'Guido van Rossum')
        self.assertTrue(Name.BDFL, Name('Guido van Rossum'))
        self.assertIs(Name.BDFL, getattr(Name, 'BDFL'))
        test_pickle_dump_load(self.assertIs, Name.BDFL)

    def test_extending(self):
        # [FUNCTION DOC] Checks that subclassing an enum with members raises TypeError (also
        # when mixing IntEnum, with a message "cannot extend") and that calling the enum with
        # extra args for a bad value raises ValueError.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with red, green, blue.
            red = 1
            green = 2
            blue = 3
        #
        with self.assertRaises(TypeError):
            class MoreColor(Color):
                # [CLASS DOC] HELPER: Invalid subclass of Color that adds members.
                cyan = 4
                magenta = 5
                yellow = 6
        #
        with self.assertRaisesRegex(TypeError, "<enum .EvenMoreColor.> cannot extend <enum .Color.>"):
            class EvenMoreColor(Color, IntEnum):
                # [CLASS DOC] HELPER: Invalid subclass of Color and IntEnum.
                chartruese = 7
        #
        with self.assertRaisesRegex(ValueError, r"\(.Foo., \(.pink., .black.\)\) is not a valid .*Color"):
            Color('Foo', ('pink', 'black'))

    def test_exclude_methods(self):
        # [FUNCTION DOC] Checks that a method defined in an enum body is not a member and works
        # on members.
        class whatever(Enum):
            # [CLASS DOC] HELPER: Enum with members this/these and a method really.
            this = 'that'
            these = 'those'
            def really(self):
                # [FUNCTION DOC] HELPER: Returns a string containing the member value.
                return 'no, not %s' % self.value
        self.assertIsNot(type(whatever.really), whatever)
        self.assertEqual(whatever.this.really(), 'no, not that')

    def test_wrong_inheritance_order(self):
        # [FUNCTION DOC] Checks that putting the data type after Enum in the bases (Enum, str)
        # raises TypeError.
        with self.assertRaises(TypeError):
            class Wrong(Enum, str):
                # [CLASS DOC] HELPER: Invalid enum with wrong base order.
                NotHere = 'error before this point'

    def test_raise_custom_error_on_creation(self):
        # [FUNCTION DOC] Checks that an exception raised from a member __init__ propagates
        # unchanged (custom ValueError subclass) out of the class creation.
        class InvalidRgbColorError(ValueError):
            # [CLASS DOC] HELPER: ValueError subclass storing r, g, b with a formatted message.
            def __init__(self, r, g, b):
                # [FUNCTION DOC] HELPER: Stores the three channel values and calls the parent
                # initializer with a message.
                self.r = r
                self.g = g
                self.b = b
                super().__init__(f'({r}, {g}, {b}) is not a valid RGB color')

        with self.assertRaises(InvalidRgbColorError):
            class RgbColor(Enum):
                # [CLASS DOC] HELPER: Enum of RGB tuples whose __init__ validates each channel
                # is within 0..255.
                RED = (255, 0, 0)
                GREEN = (0, 255, 0)
                BLUE = (0, 0, 255)
                INVALID = (256, 0, 0)

                def __init__(self, r, g, b):
                    # [FUNCTION DOC] HELPER: Raises InvalidRgbColorError if any channel is
                    # outside 0 to 255.
                    if not all(0 <= val <= 255 for val in (r, g, b)):
                        raise InvalidRgbColorError(r, g, b)

    def test_intenum_transitivity(self):
        # [FUNCTION DOC] Checks members of two different IntEnums with equal values
        # compare equal.
        class number(IntEnum):
            # [CLASS DOC] HELPER: IntEnum one, two, three.
            one = 1
            two = 2
            three = 3
        class numero(IntEnum):
            # [CLASS DOC] HELPER: IntEnum uno, dos, tres with values 1..3.
            uno = 1
            dos = 2
            tres = 3
        self.assertEqual(number.one, numero.uno)
        self.assertEqual(number.two, numero.dos)
        self.assertEqual(number.three, numero.tres)

    def test_wrong_enum_in_call(self):
        # [FUNCTION DOC] Checks calling a plain Enum with a member of a different enum
        # raises ValueError.
        class Monochrome(Enum):
            # [CLASS DOC] HELPER: Enum black=0, white=1.
            black = 0
            white = 1
        class Gender(Enum):
            # [CLASS DOC] HELPER: Enum male=0, female=1.
            male = 0
            female = 1
        self.assertRaises(ValueError, Monochrome, Gender.male)

    def test_wrong_enum_in_mixed_call(self):
        # [FUNCTION DOC] Checks calling an IntEnum with a member of an unrelated plain Enum
        # raises ValueError.
        class Monochrome(IntEnum):
            # [CLASS DOC] HELPER: IntEnum black=0, white=1.
            black = 0
            white = 1
        class Gender(Enum):
            # [CLASS DOC] HELPER: Plain Enum male=0, female=1.
            male = 0
            female = 1
        self.assertRaises(ValueError, Monochrome, Gender.male)

    def test_mixed_enum_in_call_1(self):
        # [FUNCTION DOC] Checks calling an IntEnum with a member of another IntEnum finds the
        # member by value.
        class Monochrome(IntEnum):
            # [CLASS DOC] HELPER: IntEnum black=0, white=1.
            black = 0
            white = 1
        class Gender(IntEnum):
            # [CLASS DOC] HELPER: IntEnum male=0, female=1.
            male = 0
            female = 1
        self.assertIs(Monochrome(Gender.female), Monochrome.white)

    def test_mixed_enum_in_call_2(self):
        # [FUNCTION DOC] Checks calling a plain Enum with an IntEnum member finds the member by
        # value equality.
        class Monochrome(Enum):
            # [CLASS DOC] HELPER: Plain Enum black=0, white=1.
            black = 0
            white = 1
        class Gender(IntEnum):
            # [CLASS DOC] HELPER: IntEnum male=0, female=1.
            male = 0
            female = 1
        self.assertIs(Monochrome(Gender.male), Monochrome.black)

    def test_flufl_enum(self):
        # [FUNCTION DOC] Checks an intermediate member-less Enum with an __int__ method is
        # inherited so int() of a member of the subclass uses the value.
        class Fluflnum(Enum):
            # [CLASS DOC] HELPER: Member-less base enum defining __int__.
            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the integer form of the member value.
                return int(self.value)
        class MailManOptions(Fluflnum):
            # [CLASS DOC] HELPER: Subclass with three members.
            option1 = 1
            option2 = 2
            option3 = 3
        self.assertEqual(int(MailManOptions.option1), 1)

    def test_introspection(self):
        # [FUNCTION DOC] Checks _member_type_ is int for IntEnum, str for a str-mixin enum and
        # object for a plain enum, on both the class and its members.
        class Number(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with values 100 and 200.
            one = 100
            two = 200
        self.assertIs(Number.one._member_type_, int)
        self.assertIs(Number._member_type_, int)
        class String(str, Enum):
            # [CLASS DOC] HELPER: str-mixin Enum with three string members.
            yarn = 'soft'
            rope = 'rough'
            wire = 'hard'
        self.assertIs(String.yarn._member_type_, str)
        self.assertIs(String._member_type_, str)
        class Plain(Enum):
            # [CLASS DOC] HELPER: plain Enum with vanilla (string) and one (int) members.
            vanilla = 'white'
            one = 1
        self.assertIs(Plain.vanilla._member_type_, object)
        self.assertIs(Plain._member_type_, object)

    def test_no_such_enum_member(self):
        # [FUNCTION DOC] Checks that calling with an unknown value raises ValueError and lookup
        # by an unknown name raises KeyError.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum red, green, blue.
            red = 1
            green = 2
            blue = 3
        with self.assertRaises(ValueError):
            Color(4)
        with self.assertRaises(KeyError):
            Color['chartreuse']

    # tests that need to be evalualted for moving

    def test_multiple_mixin_mro(self):
        # [FUNCTION DOC] Checks a custom metaclass derived from the Enum metaclass can combine
        # with Enum and IntEnum mixins: builds the metaclass that replaces Ellipsis values by
        # running numbers, creates two base enums and two subclasses. Only checks that
        # creation succeeds.
        class auto_enum(type(Enum)):
            # [CLASS DOC] HELPER: Metaclass subclassing the Enum metaclass; rewrites the class
            # namespace so member values equal to Ellipsis become consecutive integers.
            def __new__(metacls, cls, bases, classdict):
                # [FUNCTION DOC] HELPER: Builds a new namespace of the same type, copies members
                # (replacing Ellipsis with the previous value plus one, starting at zero) and
                # then all non-member entries, and delegates to the parent metaclass __new__.
                temp = type(classdict)()
                temp._cls_name = cls
                names = set(classdict._member_names)
                i = 0
                for k in classdict._member_names:
                    v = classdict[k]
                    if v is Ellipsis:
                        v = i
                    else:
                        i = v
                    i += 1
                    temp[k] = v
                for k, v in classdict.items():
                    if k not in names:
                        temp[k] = v
                return super(auto_enum, metacls).__new__(
                        metacls, cls, bases, temp)

        class AutoNumberedEnum(Enum, metaclass=auto_enum):
            # [CLASS DOC] HELPER: Empty plain Enum base using the auto-numbering metaclass.
            pass

        class AutoIntEnum(IntEnum, metaclass=auto_enum):
            # [CLASS DOC] HELPER: Empty IntEnum base using the auto-numbering metaclass.
            pass

        class TestAutoNumber(AutoNumberedEnum):
            # [CLASS DOC] HELPER: Enum with members a=..., b=3, c=... numbered automatically.
            a = ...
            b = 3
            c = ...

        class TestAutoInt(AutoIntEnum):
            # [CLASS DOC] HELPER: IntEnum with members a=..., b=3, c=... numbered automatically.
            a = ...
            b = 3
            c = ...

    def test_subclasses_with_getnewargs(self):
        # [FUNCTION DOC] Checks an int-mixin enum whose mixin defines __getnewargs__:
        # Enum.__new__ is kept, the add operator and repr propagate names, and pickling works
        # for numbers, members and class.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'       # needed for pickle protocol 4
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            def __getnewargs__(self):
                # [FUNCTION DOC] HELPER: Pickle hook (__getnewargs__) returning the stored
                # constructor arguments tuple.
                return self._args
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp,
                        )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'      # needed for pickle protocol 4
            x = ('the-x', 1)
            y = ('the-y', 2)


        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        test_pickle_dump_load(self.assertEqual, NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_subclasses_with_getnewargs_ex(self):
        # [FUNCTION DOC] Same as the __getnewargs__ test but the mixin
        # defines __getnewargs_ex__.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'       # needed for pickle protocol 4
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: Pickle hook (__getnewargs_ex__) returning the stored
                # argument tuple and an empty keyword dict.
                return self._args, {}
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp,
                        )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'      # needed for pickle protocol 4
            x = ('the-x', 1)
            y = ('the-y', 2)


        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        test_pickle_dump_load(self.assertEqual, NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_subclasses_with_reduce(self):
        # [FUNCTION DOC] Same checks with a mixin defining __reduce__ for pickling.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'       # needed for pickle protocol 4
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            def __reduce__(self):
                # [FUNCTION DOC] HELPER: Pickle hook (__reduce__) returning the class and the
                # stored constructor arguments.
                return self.__class__, self._args
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp,
                        )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'      # needed for pickle protocol 4
            x = ('the-x', 1)
            y = ('the-y', 2)


        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        test_pickle_dump_load(self.assertEqual, NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_subclasses_with_reduce_ex(self):
        # [FUNCTION DOC] Same checks with a mixin defining __reduce_ex__ for pickling.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'       # needed for pickle protocol 4
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            def __reduce_ex__(self, proto):
                # [FUNCTION DOC] HELPER: Pickle hook (__reduce_ex__) returning the class and
                # stored arguments, ignoring the protocol.
                return self.__class__, self._args
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp,
                        )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'      # needed for pickle protocol 4
            x = ('the-x', 1)
            y = ('the-y', 2)

        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        test_pickle_dump_load(self.assertEqual, NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_subclasses_without_direct_pickle_support(self):
        # [FUNCTION DOC] Checks that a mixin without pickle hooks makes pickling members fail
        # with TypeError (missing name/value), and that setting the enum's __reduce_ex__ to
        # pickle_by_enum_name afterward fixes member and class pickling.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'
            x = ('the-x', 1)
            y = ('the-y', 2)
        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        with self.assertRaisesRegex(TypeError, "name and value must be specified"):
            test_pickle_dump_load(self.assertIs, NEI.y)
        # fix pickle support and try again
        NEI.__reduce_ex__ = enum.pickle_by_enum_name
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_subclasses_with_direct_pickle_support(self):
        # [FUNCTION DOC] Checks a member class that defines its own __reduce_ex__ via getattr
        # lookup can be pickled: members and class round trip with identity preserved.
        class NamedInt(int):
            # [CLASS DOC] HELPER: int subclass carrying a name and stored constructor args.
            # Its __new__ requires a name plus value; repr shows the class, name and int
            # value; str is the plain int text; adding two instances builds a new instance
            # named by the expression.
            __qualname__ = 'NamedInt'
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Splits the arguments into a name and value arguments,
                # raises TypeError when no value is given, builds the int, and stores the name
                # (_intname) and all arguments (_args).
                _args = args
                name, *args = args
                if len(args) == 0:
                    raise TypeError("name and value must be specified")
                self = int.__new__(cls, *args)
                self._intname = name
                self._args = _args
                return self
            @bltns.property
            def __name__(self):
                # [FUNCTION DOC] HELPER: Property returning the stored name of the number.
                return self._intname
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the text of class name, name and integer value.
                # repr() is updated to include the name and type info
                return "{}({!r}, {})".format(
                        type(self).__name__,
                        self.__name__,
                        int.__repr__(self),
                        )
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the plain int text, falling back to the int
                # repr if int str is the generic object one.
                # str() is unchanged, even if it relies on the repr() fallback
                base = int
                base_str = base.__str__
                if base_str.__objclass__ is object:
                    return base.__repr__(self)
                return base_str(self)
            # for simplicity, we only define one operator that
            # propagates expressions
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Adds as ints; when both operands are NamedInt returns a
                # NamedInt named with the parenthesized expression of both names, otherwise
                # returns the plain int sum.
                temp = int(self) + int( other)
                if isinstance(self, NamedInt) and isinstance(other, NamedInt):
                    return NamedInt(
                        '({0} + {1})'.format(self.__name__, other.__name__),
                        temp,
                        )
                else:
                    return temp

        class NEI(NamedInt, Enum):
            # [CLASS DOC] HELPER: Enum with NamedInt mixin and members x=(the-x, 1) and
            # y=(the-y, 2).
            __qualname__ = 'NEI'
            x = ('the-x', 1)
            y = ('the-y', 2)
            def __reduce_ex__(self, proto):
                # [FUNCTION DOC] HELPER: Pickle hook returning getattr with the class and member
                # name, so unpickling looks up the member by name.
                return getattr, (self.__class__, self._name_)

        self.assertIs(NEI.__new__, Enum.__new__)
        self.assertEqual(repr(NEI.x + NEI.y), "NamedInt('(the-x + the-y)', 3)")
        globals()['NamedInt'] = NamedInt
        globals()['NEI'] = NEI
        NI5 = NamedInt('test', 5)
        self.assertEqual(NI5, 5)
        self.assertEqual(NEI.y.value, 2)
        test_pickle_dump_load(self.assertIs, NEI.y)
        test_pickle_dump_load(self.assertIs, NEI)

    def test_tuple_subclass(self):
        # [FUNCTION DOC] Checks an Enum mixed with tuple: members are instances of the enum and
        # of tuple, equal their tuple values, and a member pickles by identity (class is
        # registered in globals).
        class SomeTuple(tuple, Enum):
            # [CLASS DOC] HELPER: tuple-mixin Enum with first/second/third tuple values.
            __qualname__ = 'SomeTuple'      # needed for pickle protocol 4
            first = (1, 'for the money')
            second = (2, 'for the show')
            third = (3, 'for the music')
        self.assertIs(type(SomeTuple.first), SomeTuple)
        self.assertIsInstance(SomeTuple.second, tuple)
        self.assertEqual(SomeTuple.third, (3, 'for the music'))
        globals()['SomeTuple'] = SomeTuple
        test_pickle_dump_load(self.assertIs, SomeTuple.first)

    def test_tuple_subclass_with_auto_1(self):
        # [FUNCTION DOC] Checks namedtuple mixin with auto() inside the tuple values: auto fills
        # the index field with 1, 2, 3, value is a namedtuple, fields are accessible on members,
        # and pickling works.
        from collections import namedtuple
        T = namedtuple('T', 'index desc')
        class SomeEnum(T, Enum):
            # [CLASS DOC] HELPER: Enum mixed with namedtuple T (index, desc) using auto
            # for index.
            __qualname__ = 'SomeEnum'      # needed for pickle protocol 4
            first = auto(), 'for the money'
            second = auto(), 'for the show'
            third = auto(), 'for the music'
        self.assertIs(type(SomeEnum.first), SomeEnum)
        self.assertEqual(SomeEnum.third.value, (3, 'for the music'))
        self.assertIsInstance(SomeEnum.third.value, T)
        self.assertEqual(SomeEnum.first.index, 1)
        self.assertEqual(SomeEnum.second.desc, 'for the show')
        globals()['SomeEnum'] = SomeEnum
        globals()['T'] = T
        test_pickle_dump_load(self.assertIs, SomeEnum.first)

    def test_tuple_subclass_with_auto_2(self):
        # [FUNCTION DOC] Checks namedtuple values created with auto() as a field: the value is
        # an instance of the namedtuple with filled-in index, and members pickle by identity.
        from collections import namedtuple
        T = namedtuple('T', 'index desc')
        class SomeEnum(Enum):
            # [CLASS DOC] HELPER: plain Enum whose values are namedtuples with an auto index.
            __qualname__ = 'SomeEnum'      # needed for pickle protocol 4
            first = T(auto(), 'for the money')
            second = T(auto(), 'for the show')
            third = T(auto(), 'for the music')
        self.assertIs(type(SomeEnum.first), SomeEnum)
        self.assertEqual(SomeEnum.third.value, (3, 'for the music'))
        self.assertIsInstance(SomeEnum.third.value, T)
        self.assertEqual(SomeEnum.first.value.index, 1)
        self.assertEqual(SomeEnum.second.value.desc, 'for the show')
        globals()['SomeEnum'] = SomeEnum
        globals()['T'] = T
        test_pickle_dump_load(self.assertIs, SomeEnum.first)

    def test_duplicate_values_give_unique_enum_items(self):
        # [FUNCTION DOC] Checks that members with identical (empty tuple) definitions stay
        # distinct when a custom __new__ assigns values from the member count, and lookup by the
        # computed value works.
        class AutoNumber(Enum):
            # [CLASS DOC] HELPER: Enum with three members given empty tuples and an
            # auto-numbering __new__.
            first = ()
            second = ()
            third = ()
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates a member with value equal to the number of
                # existing members plus one.
                value = len(cls.__members__) + 1
                obj = object.__new__(cls)
                obj._value_ = value
                return obj
            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the integer form of the member value.
                return int(self._value_)
        self.assertEqual(
                list(AutoNumber),
                [AutoNumber.first, AutoNumber.second, AutoNumber.third],
                )
        self.assertEqual(int(AutoNumber.second), 2)
        self.assertEqual(AutoNumber.third.value, 3)
        self.assertIs(AutoNumber(1), AutoNumber.first)

    def test_inherited_new_from_enhanced_enum(self):
        # [FUNCTION DOC] Checks a subclass inherits __new__ and __int__ from a member-less
        # enhanced base enum.
        class AutoNumber(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with an auto-numbering __new__ and __int__.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates a member whose value is the count of existing
                # members plus one.
                value = len(cls.__members__) + 1
                obj = object.__new__(cls)
                obj._value_ = value
                return obj
            def __int__(self):
                # [FUNCTION DOC] HELPER: Returns the integer form of the member value.
                return int(self._value_)
        class Color(AutoNumber):
            # [CLASS DOC] HELPER: Subclass with red, green, blue given empty tuples.
            red = ()
            green = ()
            blue = ()
        self.assertEqual(list(Color), [Color.red, Color.green, Color.blue])
        self.assertEqual(list(map(int, Color)), [1, 2, 3])

    def test_inherited_new_from_mixed_enum(self):
        # [FUNCTION DOC] Same as the previous test but the base is a member-less IntEnum with
        # custom __new__.
        class AutoNumber(IntEnum):
            # [CLASS DOC] HELPER: Member-less IntEnum with an auto-numbering __new__.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates an int member whose value is the count of
                # existing members plus one.
                value = len(cls.__members__) + 1
                obj = int.__new__(cls, value)
                obj._value_ = value
                return obj
        class Color(AutoNumber):
            # [CLASS DOC] HELPER: Subclass with three members from empty tuples.
            red = ()
            green = ()
            blue = ()
        self.assertEqual(list(Color), [Color.red, Color.green, Color.blue])
        self.assertEqual(list(map(int, Color)), [1, 2, 3])

    def test_equality(self):
        # [FUNCTION DOC] Checks that an enum member compares equal to the ALWAYS_EQ test object
        # in both operand orders.
        class OrdinaryEnum(Enum):
            # [CLASS DOC] HELPER: Enum with a single member a=1.
            a = 1
        self.assertEqual(ALWAYS_EQ, OrdinaryEnum.a)
        self.assertEqual(OrdinaryEnum.a, ALWAYS_EQ)

    def test_ordered_mixin(self):
        # [FUNCTION DOC] Checks rich comparison methods defined in a mixin base work on members
        # of a subclass (greater, less-equal, less, greater-equal, equal and not-equal).
        class OrderedEnum(Enum):
            # [CLASS DOC] HELPER: Member-less Enum defining ordering methods by comparing values
            # for same-class operands.
            def __ge__(self, other):
                # [FUNCTION DOC] HELPER: Greater-or-equal on values for same class,
                # otherwise NotImplemented.
                if self.__class__ is other.__class__:
                    return self._value_ >= other._value_
                return NotImplemented
            def __gt__(self, other):
                # [FUNCTION DOC] HELPER: Greater-than on values for same class,
                # otherwise NotImplemented.
                if self.__class__ is other.__class__:
                    return self._value_ > other._value_
                return NotImplemented
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: Less-or-equal on values for same class,
                # otherwise NotImplemented.
                if self.__class__ is other.__class__:
                    return self._value_ <= other._value_
                return NotImplemented
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Less-than on values for same class,
                # otherwise NotImplemented.
                if self.__class__ is other.__class__:
                    return self._value_ < other._value_
                return NotImplemented
        class Grade(OrderedEnum):
            # [CLASS DOC] HELPER: Ordered enum with grades A=5 down to F=1.
            A = 5
            B = 4
            C = 3
            D = 2
            F = 1
        self.assertGreater(Grade.A, Grade.B)
        self.assertLessEqual(Grade.F, Grade.C)
        self.assertLess(Grade.D, Grade.A)
        self.assertGreaterEqual(Grade.B, Grade.B)
        self.assertEqual(Grade.B, Grade.B)
        self.assertNotEqual(Grade.C, Grade.D)

    def test_extending2(self):
        # [FUNCTION DOC] Checks a subclass of a member-less base can define members but
        # extending an enum that already has members raises TypeError.
        class Shade(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a shade method.
            def shade(self):
                # [FUNCTION DOC] HELPER: Prints the member name.
                print(self.name)
        class Color(Shade):
            # [CLASS DOC] HELPER: Subclass of Shade with red, green, blue.
            red = 1
            green = 2
            blue = 3
        with self.assertRaises(TypeError):
            class MoreColor(Color):
                # [CLASS DOC] HELPER: Invalid subclass of Color adding members.
                cyan = 4
                magenta = 5
                yellow = 6

    def test_extending3(self):
        # [FUNCTION DOC] Checks several levels of member-less enum inheritance work: a method
        # defined in the middle class can be called on members of the final subclass.
        class Shade(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a shade method.
            def shade(self):
                # [FUNCTION DOC] HELPER: Returns the member name.
                return self.name
        class Color(Shade):
            # [CLASS DOC] HELPER: Member-less subclass adding a hex method.
            def hex(self):
                # [FUNCTION DOC] HELPER: Returns the value followed by a fixed suffix text.
                return '%s hexlified!' % self.value
        class MoreColor(Color):
            # [CLASS DOC] HELPER: Concrete subclass with cyan, magenta, yellow.
            cyan = 4
            magenta = 5
            yellow = 6
        self.assertEqual(MoreColor.magenta.hex(), '5 hexlified!')

    def test_subclass_duplicate_name(self):
        # [FUNCTION DOC] Checks a member can reuse the name of a method inherited from a base
        # enum, so the attribute on the subclass is the member.
        class Base(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a method named test.
            def test(self):
                # [FUNCTION DOC] HELPER: Empty method.
                pass
        class Test(Base):
            # [CLASS DOC] HELPER: Subclass with a member named test.
            test = 1
        self.assertIs(type(Test.test), Test)

    def test_subclass_duplicate_name_dynamic(self):
        # [FUNCTION DOC] Checks a member can shadow an inherited DynamicClassAttribute or
        # enum.property of a base enum; the member keeps its value and the property is still
        # reachable through the member.
        from types import DynamicClassAttribute
        class Base(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a DynamicClassAttribute named test.
            @DynamicClassAttribute
            def test(self):
                # [FUNCTION DOC] HELPER: Returns the string dynamic.
                return 'dynamic'
        class Test(Base):
            # [CLASS DOC] HELPER: Subclass with member named test.
            test = 1
        self.assertEqual(Test.test.test, 'dynamic')
        self.assertEqual(Test.test.value, 1)
        class Base2(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with an enum.property named flash.
            @enum.property
            def flash(self):
                # [FUNCTION DOC] HELPER: Returns the string flashy dynamic.
                return 'flashy dynamic'
        class Test(Base2):
            # [CLASS DOC] HELPER: Subclass with member named flash.
            flash = 1
        self.assertEqual(Test.flash.flash, 'flashy dynamic')
        self.assertEqual(Test.flash.value, 1)

    def test_no_duplicates(self):
        # [FUNCTION DOC] Checks a custom __init__ that rejects duplicate values prevents
        # aliases: the valid enum is created, the one with a duplicate value raises ValueError.
        class UniqueEnum(Enum):
            # [CLASS DOC] HELPER: Member-less base whose __init__ raises ValueError if a member
            # value already exists.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Raises ValueError, naming the alias and original, when
                # the new member value equals the value of an existing member.
                cls = self.__class__
                if any(self.value == e.value for e in cls):
                    a = self.name
                    e = cls(self.value).name
                    raise ValueError(
                            "aliases not allowed in UniqueEnum:  %r --> %r"
                            % (a, e)
                            )
        class Color(UniqueEnum):
            # [CLASS DOC] HELPER: Valid UniqueEnum subclass with three distinct values.
            red = 1
            green = 2
            blue = 3
        with self.assertRaises(ValueError):
            class Color(UniqueEnum):
                # [CLASS DOC] HELPER: Invalid UniqueEnum subclass with a duplicate
                # value (grene=2).
                red = 1
                green = 2
                blue = 3
                grene = 2

    def test_init(self):
        # [FUNCTION DOC] Checks a planet enum with tuple values unpacked by __init__ into mass
        # and radius, and a computed property surface_gravity (checked for Earth).
        class Planet(Enum):
            # [CLASS DOC] HELPER: Enum of planets with (mass, radius) tuple values, an __init__
            # and a gravity property.
            MERCURY = (3.303e+23, 2.4397e6)
            VENUS   = (4.869e+24, 6.0518e6)
            EARTH   = (5.976e+24, 6.37814e6)
            MARS    = (6.421e+23, 3.3972e6)
            JUPITER = (1.9e+27,   7.1492e7)
            SATURN  = (5.688e+26, 6.0268e7)
            URANUS  = (8.686e+25, 2.5559e7)
            NEPTUNE = (1.024e+26, 2.4746e7)
            def __init__(self, mass, radius):
                # [FUNCTION DOC] HELPER: Stores the mass in kilograms and radius in meters on
                # the member.
                self.mass = mass       # in kilograms
                self.radius = radius   # in meters
            @enum.property
            def surface_gravity(self):
                # [FUNCTION DOC] HELPER: Computes surface gravity from mass and radius using the
                # gravitational constant.
                # universal gravitational constant  (m3 kg-1 s-2)
                G = 6.67300E-11
                return G * self.mass / (self.radius * self.radius)
        self.assertEqual(round(Planet.EARTH.surface_gravity, 2), 9.80)
        self.assertEqual(Planet.EARTH.value, (5.976e+24, 6.37814e6))

    def test_ignore(self):
        # [FUNCTION DOC] Checks the _ignore_ mechanism: names listed there are removed from the
        # class and temporary namespace entries generate month/week/day members, with equal
        # values becoming aliases.
        class Period(timedelta, Enum):
            # [CLASS DOC] HELPER: timedelta-mixin Enum generating month, week and day members
            # through _ignore_.
            '''
            different lengths of time
            '''
            def __new__(cls, value, period):
                # [FUNCTION DOC] HELPER: Builds a timedelta member from the number of days and
                # records the period label.
                obj = timedelta.__new__(cls, value)
                obj._value_ = value
                obj.period = period
                return obj
            _ignore_ = 'Period i'
            Period = vars()
            for i in range(13):
                Period['month_%d' % i] = i*30, 'month'
            for i in range(53):
                Period['week_%d' % i] = i*7, 'week'
            for i in range(32):
                Period['day_%d' % i] = i, 'day'
            OneDay = day_1
            OneWeek = week_1
            OneMonth = month_1
        self.assertFalse(hasattr(Period, '_ignore_'))
        self.assertFalse(hasattr(Period, 'Period'))
        self.assertFalse(hasattr(Period, 'i'))
        self.assertTrue(isinstance(Period.day_1, timedelta))
        self.assertTrue(Period.month_1 is Period.day_30)
        self.assertTrue(Period.week_4 is Period.day_28)

    def test_nonhash_value(self):
        # [FUNCTION DOC] Checks that members with unhashable values (lists) work: lookup by an
        # equal list returns the member.
        class AutoNumberInAList(Enum):
            # [CLASS DOC] HELPER: Member-less base whose __new__ assigns list values.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Creates a member whose value is a one-element list with
                # the member count plus one.
                value = [len(cls.__members__) + 1]
                obj = object.__new__(cls)
                obj._value_ = value
                return obj
        class ColorInAList(AutoNumberInAList):
            # [CLASS DOC] HELPER: Subclass with three members from empty tuples.
            red = ()
            green = ()
            blue = ()
        self.assertEqual(list(ColorInAList), [ColorInAList.red, ColorInAList.green, ColorInAList.blue])
        for enum, value in zip(ColorInAList, range(3)):
            value += 1
            self.assertEqual(enum.value, [value])
            self.assertIs(ColorInAList([value]), enum)

    def test_conflicting_types_resolved_in_new(self):
        # [FUNCTION DOC] Checks a custom __new__ on a base int-mixin enum resolves the int
        # and label from a (value, label) tuple, and is inherited by subclasses for iteration
        # and lookup.
        class LabelledIntEnum(int, Enum):
            # [CLASS DOC] HELPER: Member-less int-mixin Enum whose __new__ takes (value, label).
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Creates an int member from the value, stores the label
                # and sets _value_.
                value, label = args
                obj = int.__new__(cls, value)
                obj.label = label
                obj._value_ = value
                return obj

        class LabelledList(LabelledIntEnum):
            # [CLASS DOC] HELPER: Subclass with two labelled int members.
            unprocessed = (1, "Unprocessed")
            payment_complete = (2, "Payment Complete")

        self.assertEqual(list(LabelledList), [LabelledList.unprocessed, LabelledList.payment_complete])
        self.assertEqual(LabelledList.unprocessed, 1)
        self.assertEqual(LabelledList(1), LabelledList.unprocessed)

    def test_default_missing_no_chained_exception(self):
        # [FUNCTION DOC] Checks that the ValueError from a failed lookup has no chained
        # context exception.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum RED, GREEN, BLUE.
            RED = 1
            GREEN = 2
            BLUE = 3
        try:
            Color(7)
        except ValueError as exc:
            self.assertTrue(exc.__context__ is None)
        else:
            raise Exception('Exception not raised.')

    def test_missing_override(self):
        # [FUNCTION DOC] Checks _missing_ overrides: returning a member works; returning None
        # gives ValueError with no context; a non-member return gives TypeError chained to
        # ValueError; raising propagates the error chained to ValueError.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with a _missing_ hook handling 'three', 'bad return',
            # 'error out' specially.
            red = 1
            green = 2
            blue = 3
            @classmethod
            def _missing_(cls, item):
                # [FUNCTION DOC] HELPER: Returns blue for 'three', a bad non-member value for
                # 'bad return', raises ZeroDivisionError for 'error out', and None otherwise.
                if item == 'three':
                    return cls.blue
                elif item == 'bad return':
                    # trigger internal error
                    return 5
                elif item == 'error out':
                    raise ZeroDivisionError
                else:
                    # trigger not found
                    return None
        self.assertIs(Color('three'), Color.blue)
        try:
            Color(7)
        except ValueError as exc:
            self.assertTrue(exc.__context__ is None)
        else:
            raise Exception('Exception not raised.')
        try:
            Color('bad return')
        except TypeError as exc:
            self.assertTrue(isinstance(exc.__context__, ValueError))
        else:
            raise Exception('Exception not raised.')
        try:
            Color('error out')
        except ZeroDivisionError as exc:
            self.assertTrue(isinstance(exc.__context__, ValueError))
        else:
            raise Exception('Exception not raised.')

    def test_missing_exceptions_reset(self):
        # [FUNCTION DOC] Checks that a failed enum lookup caught inside __init__ does not
        # leave a reference cycle: weak references to instances of the helper classes die
        # after gc.collect.
        import gc
        import weakref
        #
        class TestEnum(enum.Enum):
            # [CLASS DOC] HELPER: Enum with two members, excluded from test collection.
            VAL1 = 'val1'
            VAL2 = 'val2'
            __test__ = False
        #
        class Class1:
            # [CLASS DOC] HELPER: Class whose initializer catches its own ValueError.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Raises and swallows a ValueError.
                # Gracefully handle an exception of our own making
                try:
                    raise ValueError()
                except ValueError:
                    pass
        #
        class Class2:
            # [CLASS DOC] HELPER: Class whose initializer catches an Enum lookup ValueError.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Looks up an invalid value in TestEnum and swallows the
                # resulting ValueError.
                # Gracefully handle an exception of Enum's making
                try:
                    TestEnum('invalid_value')
                except ValueError:
                    pass
        # No strong refs here so these are free to die.
        class_1_ref = weakref.ref(Class1())
        class_2_ref = weakref.ref(Class2())
        #
        # The exception raised by Enum used to create a reference loop and thus
        # Class2 instances would stick around until the next garbage collection
        # cycle, unlike Class1.  Verify Class2 no longer does this.
        gc.collect()  # For PyPy or other GCs.
        self.assertIs(class_1_ref(), None)
        self.assertIs(class_2_ref(), None)

    def test_multiple_mixin(self):
        # [FUNCTION DOC] Checks combining mixins with enums in various base orders: MaxMixin
        # gives a class-level MAX property (count of members); StrMixin provides a lowercase
        # __str__ (re-assigned explicitly since 3.11); behavior and social methods resolve
        # according to MRO; an IntEnum can also be in bases.
        class MaxMixin:
            # [CLASS DOC] HELPER: Plain mixin with a classproperty MAX that computes and caches
            # the member count.
            @classproperty
            def MAX(cls):
                # [FUNCTION DOC] HELPER: Computes the number of members, stores it on the class
                # as MAX and returns it.
                max = len(cls)
                cls.MAX = max
                return max
        class StrMixin:
            # [CLASS DOC] HELPER: Plain mixin providing a __str__ returning the lowercase
            # member name.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the lowercase member name.
                return self._name_.lower()
        class SomeEnum(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with behavior method returning booyah.
            def behavior(self):
                # [FUNCTION DOC] HELPER: Returns booyah.
                return 'booyah'
        class AnotherEnum(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with behavior (nuhuh!) and social methods.
            def behavior(self):
                # [FUNCTION DOC] HELPER: Returns nuhuh!.
                return 'nuhuh!'
            def social(self):
                # [FUNCTION DOC] HELPER: Returns a greeting phrase.
                return "what's up?"
        class Color(MaxMixin, Enum):
            # [CLASS DOC] HELPER: Enum with MaxMixin and members RED, GREEN, BLUE.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 3)
        self.assertEqual(Color.MAX, 3)
        self.assertEqual(str(Color.BLUE), 'Color.BLUE')
        class Color(MaxMixin, StrMixin, Enum):
            # [CLASS DOC] HELPER: Enum with MaxMixin and StrMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 3)
        self.assertEqual(Color.MAX, 3)
        self.assertEqual(str(Color.BLUE), 'blue')
        class Color(StrMixin, MaxMixin, Enum):
            # [CLASS DOC] HELPER: Enum with StrMixin then MaxMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 3)
        self.assertEqual(Color.MAX, 3)
        self.assertEqual(str(Color.BLUE), 'blue')
        class CoolColor(StrMixin, SomeEnum, Enum):
            # [CLASS DOC] HELPER: Enum with StrMixin and SomeEnum.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(CoolColor.RED.value, 1)
        self.assertEqual(CoolColor.GREEN.value, 2)
        self.assertEqual(CoolColor.BLUE.value, 3)
        self.assertEqual(str(CoolColor.BLUE), 'blue')
        self.assertEqual(CoolColor.RED.behavior(), 'booyah')
        class CoolerColor(StrMixin, AnotherEnum, Enum):
            # [CLASS DOC] HELPER: Enum with StrMixin and AnotherEnum.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(CoolerColor.RED.value, 1)
        self.assertEqual(CoolerColor.GREEN.value, 2)
        self.assertEqual(CoolerColor.BLUE.value, 3)
        self.assertEqual(str(CoolerColor.BLUE), 'blue')
        self.assertEqual(CoolerColor.RED.behavior(), 'nuhuh!')
        self.assertEqual(CoolerColor.RED.social(), "what's up?")
        class CoolestColor(StrMixin, SomeEnum, AnotherEnum):
            # [CLASS DOC] HELPER: Enum with StrMixin, SomeEnum and AnotherEnum in that
            # base order.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(CoolestColor.RED.value, 1)
        self.assertEqual(CoolestColor.GREEN.value, 2)
        self.assertEqual(CoolestColor.BLUE.value, 3)
        self.assertEqual(str(CoolestColor.BLUE), 'blue')
        self.assertEqual(CoolestColor.RED.behavior(), 'booyah')
        self.assertEqual(CoolestColor.RED.social(), "what's up?")
        class ConfusedColor(StrMixin, AnotherEnum, SomeEnum):
            # [CLASS DOC] HELPER: Enum with StrMixin, AnotherEnum and SomeEnum in that
            # base order.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(ConfusedColor.RED.value, 1)
        self.assertEqual(ConfusedColor.GREEN.value, 2)
        self.assertEqual(ConfusedColor.BLUE.value, 3)
        self.assertEqual(str(ConfusedColor.BLUE), 'blue')
        self.assertEqual(ConfusedColor.RED.behavior(), 'nuhuh!')
        self.assertEqual(ConfusedColor.RED.social(), "what's up?")
        class ReformedColor(StrMixin, IntEnum, SomeEnum, AnotherEnum):
            # [CLASS DOC] HELPER: Enum with StrMixin, IntEnum, SomeEnum and AnotherEnum.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__          # needed as of 3.11
        self.assertEqual(ReformedColor.RED.value, 1)
        self.assertEqual(ReformedColor.GREEN.value, 2)
        self.assertEqual(ReformedColor.BLUE.value, 3)
        self.assertEqual(str(ReformedColor.BLUE), 'blue')
        self.assertEqual(ReformedColor.RED.behavior(), 'booyah')
        self.assertEqual(ConfusedColor.RED.social(), "what's up?")
        self.assertTrue(issubclass(ReformedColor, int))

    def test_multiple_inherited_mixin(self):
        # [FUNCTION DOC] Checks the unique decorator accepts a StrEnum and a subclass of an
        # empty StrEnum with distinct members.
        @unique
        class Decision1(StrEnum):
            # [CLASS DOC] HELPER: StrEnum with REVERT, REVERT_ALL, RETRY.
            REVERT = "REVERT"
            REVERT_ALL = "REVERT_ALL"
            RETRY = "RETRY"
        class MyEnum(StrEnum):
            # [CLASS DOC] HELPER: Empty StrEnum base.
            pass
        @unique
        class Decision2(MyEnum):
            # [CLASS DOC] HELPER: Subclass of MyEnum with three distinct string members.
            REVERT = "REVERT"
            REVERT_ALL = "REVERT_ALL"
            RETRY = "RETRY"

    def test_multiple_mixin_inherited(self):
        # [FUNCTION DOC] Checks a mixin class defining __repr__ combined with an int subclass
        # and Enum is honored, and a subclass can override __new__ to alter the value.
        class MyInt(int):
            # [CLASS DOC] HELPER: int subclass with __new__ forwarding to int.
            def __new__(cls, value):
                # [FUNCTION DOC] HELPER: Creates the int instance.
                return super().__new__(cls, value)

        class HexMixin:
            # [CLASS DOC] HELPER: Plain mixin with a hex based __repr__.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the hexadecimal text of the integer.
                return hex(self)

        class MyIntEnum(HexMixin, MyInt, enum.Enum):
            # [CLASS DOC] HELPER: Member-less Enum with HexMixin and MyInt that keeps the
            # hex __repr__.
            __repr__ = HexMixin.__repr__

        class Foo(MyIntEnum):
            # [CLASS DOC] HELPER: Subclass with member TEST=1.
            TEST = 1
        self.assertTrue(isinstance(Foo.TEST, MyInt))
        self.assertEqual(Foo._member_type_, MyInt)
        self.assertEqual(repr(Foo.TEST), "0x1")

        class Fee(MyIntEnum):
            # [CLASS DOC] HELPER: Subclass with member TEST and a __new__ that adds one to
            # the value.
            TEST = 1
            def __new__(cls, value):
                # [FUNCTION DOC] HELPER: Creates an int member with value plus one and sets
                # _value_ to it.
                value += 1
                member = int.__new__(cls, value)
                member._value_ = value
                return member
        self.assertEqual(Fee.TEST, 2)

    def test_multiple_mixin_with_common_data_type(self):
        # [FUNCTION DOC] Checks two str-mixin bases cooperating: a case-insensitive _missing_
        # and a lenient one that creates new invalid pseudo-members; known members still iterate
        # unchanged and unknown values yield members flagged as not valid.
        class CaseInsensitiveStrEnum(str, Enum):
            # [CLASS DOC] HELPER: str-mixin Enum whose _missing_ finds
            # members case-insensitively.
            @classmethod
            def _missing_(cls, value):
                # [FUNCTION DOC] HELPER: Classmethod looking for a member whose value matches
                # ignoring case, else defers to the parent.
                for member in cls._member_map_.values():
                    if member._value_.lower() == value.lower():
                        return member
                return super()._missing_(value)
        #
        class LenientStrEnum(str, Enum):
            # [CLASS DOC] HELPER: str-mixin Enum with a valid flag; _missing_ builds an
            # invalid pseudo-member.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: Marks the member as valid.
                self._valid = True
            @classmethod
            def _missing_(cls, value):
                # [FUNCTION DOC] HELPER: Classmethod creating a pseudo-member with valid
                # False, upper-cased name and given value, storing it in the member map, and
                # returning it.
                unknown = cls._member_type_.__new__(cls, value)
                unknown._valid = False
                unknown._name_ = value.upper()
                unknown._value_ = value
                cls._member_map_[value] = unknown
                return unknown
            @enum.property
            def valid(self):
                # [FUNCTION DOC] HELPER: Property returning the member valid flag.
                return self._valid
        #
        class JobStatus(CaseInsensitiveStrEnum, LenientStrEnum):
            # [CLASS DOC] HELPER: Enum combining the two str mixins with ACTIVE,
            # PENDING, TERMINATED.
            ACTIVE = "active"
            PENDING = "pending"
            TERMINATED = "terminated"
        #
        JS = JobStatus
        self.assertEqual(list(JobStatus), [JS.ACTIVE, JS.PENDING, JS.TERMINATED])
        self.assertEqual(JS.ACTIVE, 'active')
        self.assertEqual(JS.ACTIVE.value, 'active')
        self.assertIs(JS('Active'), JS.ACTIVE)
        self.assertTrue(JS.ACTIVE.valid)
        missing = JS('missing')
        self.assertEqual(list(JobStatus), [JS.ACTIVE, JS.PENDING, JS.TERMINATED])
        self.assertEqual(JS.ACTIVE, 'active')
        self.assertEqual(JS.ACTIVE.value, 'active')
        self.assertIs(JS('Active'), JS.ACTIVE)
        self.assertTrue(JS.ACTIVE.valid)
        self.assertTrue(isinstance(missing, JS))
        self.assertFalse(missing.valid)

    def test_empty_globals(self):
        # [FUNCTION DOC] Regression test (bpo-35717): creating an Enum through compiled code run
        # with empty globals must not fail with KeyError. Compiles and runs the code with Dynamo
        # graph-break errors disabled.
        # bpo-35717: sys._getframe(2).f_globals['__name__'] fails with KeyError
        # when using compile and exec because f_globals is empty
        code = "from enum import Enum; Enum('Animal', 'ANT BEE CAT DOG')"
        with torch._dynamo.error_on_graph_break(False):
            code = compile(code, "<string>", "exec")
        global_ns = {}
        local_ls = {}
        with torch._dynamo.error_on_graph_break(False):
            exec(code, global_ns, local_ls)

    def test_strenum(self):
        # [FUNCTION DOC] Checks StrEnum: str, format and equality use the plain string; bytes
        # with encoding and errors are decoded; extra mixin classes are allowed; a custom
        # __str__ may be re-assigned; non-string values or bad encoding/errors arguments
        # raise TypeError.
        class GoodStrEnum(StrEnum):
            # [CLASS DOC] HELPER: StrEnum with string and decoded-bytes members.
            one = '1'
            two = '2'
            three = b'3', 'ascii'
            four = b'4', 'latin1', 'strict'
        self.assertEqual(GoodStrEnum.one, '1')
        self.assertEqual(str(GoodStrEnum.one), '1')
        self.assertEqual('{}'.format(GoodStrEnum.one), '1')
        self.assertEqual(GoodStrEnum.one, str(GoodStrEnum.one))
        self.assertEqual(GoodStrEnum.one, '{}'.format(GoodStrEnum.one))
        self.assertEqual(repr(GoodStrEnum.one), "<GoodStrEnum.one: '1'>")
        #
        class DumbMixin:
            # [CLASS DOC] HELPER: Plain mixin with a __str__ returning a fixed text.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed string.
                return "don't do this"
        class DumbStrEnum(DumbMixin, StrEnum):
            # [CLASS DOC] HELPER: StrEnum with DumbMixin and members five, six, seven.
            five = '5'
            six = '6'
            seven = '7'
            __str__ = DumbMixin.__str__             # needed as of 3.11
        self.assertEqual(DumbStrEnum.seven, '7')
        self.assertEqual(str(DumbStrEnum.seven), "don't do this")
        #
        class EnumMixin(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a hello method.
            def hello(self):
                # [FUNCTION DOC] HELPER: Prints a greeting naming the member.
                print('hello from %s' % (self, ))
        class HelloEnum(EnumMixin, StrEnum):
            # [CLASS DOC] HELPER: StrEnum with EnumMixin and member eight.
            eight = '8'
        self.assertEqual(HelloEnum.eight, '8')
        self.assertEqual(HelloEnum.eight, str(HelloEnum.eight))
        #
        class GoodbyeMixin:
            # [CLASS DOC] HELPER: Plain mixin with a goodbye method.
            def goodbye(self):
                # [FUNCTION DOC] HELPER: Prints a farewell message.
                print('%s wishes you a fond farewell')
        class GoodbyeEnum(GoodbyeMixin, EnumMixin, StrEnum):
            # [CLASS DOC] HELPER: StrEnum with GoodbyeMixin and EnumMixin and member nine.
            nine = '9'
        self.assertEqual(GoodbyeEnum.nine, '9')
        self.assertEqual(GoodbyeEnum.nine, str(GoodbyeEnum.nine))
        #
        with self.assertRaisesRegex(TypeError, '1 is not a string'):
            class FirstFailedStrEnum(StrEnum):
                # [CLASS DOC] HELPER: Invalid StrEnum with a non-string value 1.
                one = 1
                two = '2'
        with self.assertRaisesRegex(TypeError, "2 is not a string"):
            class SecondFailedStrEnum(StrEnum):
                # [CLASS DOC] HELPER: Invalid StrEnum with a tuple value starting with an int.
                one = '1'
                two = 2,
                three = '3'
        with self.assertRaisesRegex(TypeError, '2 is not a string'):
            class ThirdFailedStrEnum(StrEnum):
                # [CLASS DOC] HELPER: Invalid StrEnum with the non-string value 2.
                one = '1'
                two = 2
        with self.assertRaisesRegex(TypeError, 'encoding must be a string, not %r' % (sys.getdefaultencoding, )):
            class ThirdFailedStrEnum(StrEnum):
                # [CLASS DOC] HELPER: Invalid StrEnum with a non-string encoding argument.
                one = '1'
                two = b'2', sys.getdefaultencoding
        with self.assertRaisesRegex(TypeError, 'errors must be a string, not 9'):
            class ThirdFailedStrEnum(StrEnum):
                # [CLASS DOC] HELPER: Invalid StrEnum with a non-string errors argument.
                one = '1'
                two = b'2', 'ascii', 9

    def test_custom_strenum(self):
        # [FUNCTION DOC] Checks a user-made (str, Enum) base: members accept string and
        # decoded-bytes values, equal their string, but str and format use the ClassName.member
        # form while repr shows the value.
        class CustomStrEnum(str, Enum):
            # [CLASS DOC] HELPER: Empty str-mixin Enum.
            pass
        class OkayEnum(CustomStrEnum):
            # [CLASS DOC] HELPER: Subclass with string and decoded-bytes members.
            one = '1'
            two = '2'
            three = b'3', 'ascii'
            four = b'4', 'latin1', 'strict'
        self.assertEqual(OkayEnum.one, '1')
        self.assertEqual(str(OkayEnum.one), 'OkayEnum.one')
        self.assertEqual('{}'.format(OkayEnum.one), 'OkayEnum.one')
        self.assertEqual(repr(OkayEnum.one), "<OkayEnum.one: '1'>")
        #
        class DumbMixin:
            # [CLASS DOC] HELPER: Plain mixin with a __str__ returning a fixed text.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed string.
                return "don't do this"
        class DumbStrEnum(DumbMixin, CustomStrEnum):
            # [CLASS DOC] HELPER: Enum with DumbMixin and CustomStrEnum and members five,
            # six, seven.
            five = '5'
            six = '6'
            seven = '7'
            __str__ = DumbMixin.__str__         # needed as of 3.11
        self.assertEqual(DumbStrEnum.seven, '7')
        self.assertEqual(str(DumbStrEnum.seven), "don't do this")
        #
        class EnumMixin(Enum):
            # [CLASS DOC] HELPER: Member-less Enum with a hello method.
            def hello(self):
                # [FUNCTION DOC] HELPER: Prints a greeting naming the member.
                print('hello from %s' % (self, ))
        class HelloEnum(EnumMixin, CustomStrEnum):
            # [CLASS DOC] HELPER: Enum with EnumMixin and CustomStrEnum and member eight.
            eight = '8'
        self.assertEqual(HelloEnum.eight, '8')
        self.assertEqual(str(HelloEnum.eight), 'HelloEnum.eight')
        #
        class GoodbyeMixin:
            # [CLASS DOC] HELPER: Plain mixin with a goodbye method.
            def goodbye(self):
                # [FUNCTION DOC] HELPER: Prints a farewell message.
                print('%s wishes you a fond farewell')
        class GoodbyeEnum(GoodbyeMixin, EnumMixin, CustomStrEnum):
            # [CLASS DOC] HELPER: Enum with GoodbyeMixin, EnumMixin and CustomStrEnum and
            # member nine.
            nine = '9'
        self.assertEqual(GoodbyeEnum.nine, '9')
        self.assertEqual(str(GoodbyeEnum.nine), 'GoodbyeEnum.nine')
        #
        class FirstFailedStrEnum(CustomStrEnum):
            # [CLASS DOC] HELPER: Enum on CustomStrEnum whose int value is coerced to the
            # text '1'.
            one = 1   # this will become '1'
            two = '2'
        class SecondFailedStrEnum(CustomStrEnum):
            # [CLASS DOC] HELPER: Enum on CustomStrEnum whose tuple value is coerced to the
            # text '2'.
            one = '1'
            two = 2,  # this will become '2'
            three = '3'
        class ThirdFailedStrEnum(CustomStrEnum):
            # [CLASS DOC] HELPER: Enum on CustomStrEnum whose int value is coerced to the
            # text '2'.
            one = '1'
            two = 2  # this will become '2'
        with self.assertRaisesRegex(TypeError,
                r"argument (2|'encoding') must be str, not "):
            class ThirdFailedStrEnum(CustomStrEnum):
                # [CLASS DOC] HELPER: Invalid enum passing a non-str encoding argument; creation
                # must raise TypeError.
                one = '1'
                two = b'2', sys.getdefaultencoding
        with self.assertRaisesRegex(TypeError,
                r"argument (3|'errors') must be str, not "):
            class ThirdFailedStrEnum(CustomStrEnum):
                # [CLASS DOC] HELPER: Invalid enum passing a non-str errors argument; creation
                # must raise TypeError.
                one = '1'
                two = b'2', 'ascii', 9

    def test_missing_value_error(self):
        # [FUNCTION DOC] Checks that a custom __new__ which never sets _value_ makes class
        # creation raise TypeError saying _value_ was not set in __new__.
        with self.assertRaisesRegex(TypeError, "_value_ not set in __new__"):
            class Combined(str, Enum):
                # [CLASS DOC] HELPER: Invalid str-mixin Enum whose __new__ builds fields from a
                # (value, sequence) tuple but forgets _value_.
                #
                def __new__(cls, value, sequence):
                    # [FUNCTION DOC] HELPER: Builds a str member, parses the text before and
                    # inside parentheses into fis_name and segment, stores the sequence, and
                    # returns the member without setting _value_.
                    enum = str.__new__(cls, value)
                    if '(' in value:
                        fis_name, segment = value.split('(', 1)
                        segment = segment.strip(' )')
                    else:
                        fis_name = value
                        segment = None
                    enum.fis_name = fis_name
                    enum.segment = segment
                    enum.sequence = sequence
                    return enum
                #
                def __repr__(self):
                    # [FUNCTION DOC] HELPER: Returns class and member name wrapped in
                    # angle brackets.
                    return "<%s.%s>" % (self.__class__.__name__, self._name_)
                #
                key_type      = 'An$(1,2)', 0
                company_id    = 'An$(3,2)', 1
                code          = 'An$(5,1)', 2
                description   = 'Bn$',      3


    def test_private_variable_is_normal_attribute(self):
        # [FUNCTION DOC] Checks that double-underscore (private) names in an enum body become
        # normal mangled class attributes, not members.
        class Private(Enum):
            # [CLASS DOC] HELPER: Enum whose only attributes are name-mangled private names.
            __corporal = 'Radar'
            __major_ = 'Hoolihan'
        self.assertEqual(Private._Private__corporal, 'Radar')
        self.assertEqual(Private._Private__major_, 'Hoolihan')

    def test_member_from_member_access(self):
        # [FUNCTION DOC] Checks that a member reached through another member returns the
        # class member, and that a member named name works with item lookup and its own
        # name attribute.
        class Di(Enum):
            # [CLASS DOC] HELPER: Enum with YES=1, NO=0 and a member named name.
            YES = 1
            NO = 0
            name = 3
        warn = Di.YES.NO
        self.assertIs(warn, Di.NO)
        self.assertIs(Di.name, Di['name'])
        self.assertEqual(Di.name.name, 'name')

    def test_dynamic_members_with_static_methods(self):
        # [FUNCTION DOC] Checks members created dynamically by updating the class namespace from
        # a filtered mapping and a regular method on the enum; defining a name twice with
        # different values raises TypeError.
        #
        foo_defines = {'FOO_CAT': 'aloof', 'BAR_DOG': 'friendly', 'FOO_HORSE': 'big'}
        class Foo(Enum):
            # [CLASS DOC] HELPER: Enum whose members come from entries of a mapping with the
            # FOO_ prefix, with an upper method.
            vars().update({
                    k: v
                    for k, v in foo_defines.items()
                    if k.startswith('FOO_')
                    })
            def upper(self):
                # [FUNCTION DOC] HELPER: Returns the upper-cased member value.
                return self.value.upper()
        self.assertEqual(list(Foo), [Foo.FOO_CAT, Foo.FOO_HORSE])
        self.assertEqual(Foo.FOO_CAT.value, 'aloof')
        self.assertEqual(Foo.FOO_HORSE.upper(), 'BIG')
        #
        with self.assertRaisesRegex(TypeError, "'FOO_CAT' already defined as 'aloof'"):
            class FooBar(Enum):
                # [CLASS DOC] HELPER: Invalid enum that updates the namespace twice with
                # FOO_CAT, which must raise TypeError.
                vars().update({
                        k: v
                        for k, v in foo_defines.items()
                        if k.startswith('FOO_')
                        },
                        **{'FOO_CAT': 'small'},
                        )
                def upper(self):
                    # [FUNCTION DOC] HELPER: Returns the upper-cased member value.
                    return self.value.upper()

    def test_repr_with_dataclass(self):
        # [FUNCTION DOC] Checks the member repr for dataclass mixins: an overridden dataclass
        # __repr__ is used, an auto-generated dataclass repr is not used (field text shown
        # instead), an inherited repr is used, and without any custom repr the default object
        # repr shows.
        "ensure dataclass-mixin has correct repr()"
        #
        # check overridden dataclass __repr__ is used
        #
        from dataclasses import dataclass, field
        @dataclass(repr=False)
        class Foo:
            # [CLASS DOC] HELPER: Dataclass with field a and a custom __repr__ returning a
            # fixed text.
            __qualname__ = 'Foo'
            a: int
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed string.
                return 'ha hah!'
        class Entries(Foo, Enum):
            # [CLASS DOC] HELPER: Enum mixing Foo with member ENTRY1.
            ENTRY1 = 1
        self.assertEqual(repr(Entries.ENTRY1), '<Entries.ENTRY1: ha hah!>')
        self.assertTrue(Entries.ENTRY1.value == Foo(1), Entries.ENTRY1.value)
        self.assertTrue(isinstance(Entries.ENTRY1, Foo))
        self.assertTrue(Entries._member_type_ is Foo, Entries._member_type_)
        #
        # check auto-generated dataclass __repr__ is not used
        #
        @dataclass
        class CreatureDataMixin:
            # [CLASS DOC] HELPER: Dataclass mixin with size, legs and a tail field excluded
            # from repr.
            __qualname__ = 'CreatureDataMixin'
            size: str
            legs: int
            tail: bool = field(repr=False, default=True)
        class Creature(CreatureDataMixin, Enum):
            # [CLASS DOC] HELPER: Enum with the dataclass mixin and BEETLE and DOG
            # tuple members.
            __qualname__ = 'Creature'
            BEETLE = ('small', 6)
            DOG = ('medium', 4)
        self.assertEqual(repr(Creature.DOG), "<Creature.DOG: size='medium', legs=4>")
        #
        # check inherited repr used
        #
        class Huh:
            # [CLASS DOC] HELPER: Plain class whose repr returns inherited.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the string inherited.
                return 'inherited'
        @dataclass(repr=False)
        class CreatureDataMixin(Huh):
            # [CLASS DOC] HELPER: Dataclass mixin without its own repr, inheriting Huh.
            __qualname__ = 'CreatureDataMixin'
            size: str
            legs: int
            tail: bool = field(repr=False, default=True)
        class Creature(CreatureDataMixin, Enum):
            # [CLASS DOC] HELPER: Enum using the inheriting dataclass mixin.
            __qualname__ = 'Creature'
            BEETLE = ('small', 6)
            DOG = ('medium', 4)
        self.assertEqual(repr(Creature.DOG), "<Creature.DOG: inherited>")
        #
        # check default object.__repr__ used if nothing provided
        #
        @dataclass(repr=False)
        class CreatureDataMixin:
            # [CLASS DOC] HELPER: Dataclass mixin with repr disabled and no inherited repr.
            __qualname__ = 'CreatureDataMixin'
            size: str
            legs: int
            tail: bool = field(repr=False, default=True)
        class Creature(CreatureDataMixin, Enum):
            # [CLASS DOC] HELPER: Enum using the dataclass mixin without any custom repr.
            __qualname__ = 'Creature'
            BEETLE = ('small', 6)
            DOG = ('medium', 4)
        self.assertRegex(repr(Creature.DOG), "<Creature.DOG: .*CreatureDataMixin object at .*>")

    def test_repr_with_init_mixin(self):
        # [FUNCTION DOC] Checks the mixin __repr__ is used for members of an enum whose mixin
        # has an __init__.
        class Foo:
            # [CLASS DOC] HELPER: Mixin with an initializer taking a and a repr showing it.
            def __init__(self, a):
                # [FUNCTION DOC] HELPER: Stores a on the instance.
                self.a = a
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns text with the stored a.
                return f'Foo(a={self.a!r})'
        class Entries(Foo, Enum):
            # [CLASS DOC] HELPER: Enum with Foo mixin and one member.
            ENTRY1 = 1
        #
        self.assertEqual(repr(Entries.ENTRY1), 'Foo(a=1)')

    def test_repr_and_str_with_no_init_mixin(self):
        # [FUNCTION DOC] Checks that repr and str come from a mixin that defines no __new__ and
        # both are overridden there.
        # non-data_type is a mixin that doesn't define __new__
        class Foo:
            # [CLASS DOC] HELPER: Mixin without __new__ defining __repr__ (Foo) and
            # __str__ (ooF).
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns the string Foo.
                return 'Foo'
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the string ooF.
                return 'ooF'
        class Entries(Foo, Enum):
            # [CLASS DOC] HELPER: Enum with that mixin and one member.
            ENTRY1 = 1
        #
        self.assertEqual(repr(Entries.ENTRY1), 'Foo')
        self.assertEqual(str(Entries.ENTRY1), 'ooF')

    def test_value_backup_assign(self):
        # [FUNCTION DOC] Checks that when a custom __new__ does not set _value_, the enum
        # assigns the original tuple/argument value.
        # check that enum will add missing values when custom __new__ does not
        class Some(Enum):
            # [CLASS DOC] HELPER: Enum whose __new__ creates a plain object without setting
            # a value.
            def __new__(cls, val):
                # [FUNCTION DOC] HELPER: Creates an object member ignoring the supplied value.
                return object.__new__(cls)
            x = 1
            y = 2
        self.assertEqual(Some.x.value, 1)
        self.assertEqual(Some.y.value, 2)

    def test_custom_flag_bitwise(self):
        # [FUNCTION DOC] Checks bitwise or of two members of an (int, Flag) enum, and with a
        # plain int, yields an instance of that class.
        class MyIntFlag(int, Flag):
            # [CLASS DOC] HELPER: Flag with int mixin and members 1, 2, 4.
            ONE = 1
            TWO = 2
            FOUR = 4
        self.assertTrue(isinstance(MyIntFlag.ONE | MyIntFlag.TWO, MyIntFlag), MyIntFlag.ONE | MyIntFlag.TWO)
        self.assertTrue(isinstance(MyIntFlag.ONE | 2, MyIntFlag))

    def test_int_flags_copy(self):
        # [FUNCTION DOC] Checks copy and deepcopy of IntFlag combinations (including one with an
        # extra undefined bit) are equal to the original and keep the value.
        class MyIntFlag(IntFlag):
            # [CLASS DOC] HELPER: IntFlag with members 1, 2, 4.
            ONE = 1
            TWO = 2
            FOUR = 4

        flags = MyIntFlag.ONE | MyIntFlag.TWO
        copied = copy.copy(flags)
        deep = copy.deepcopy(flags)
        self.assertEqual(copied, flags)
        self.assertEqual(deep, flags)

        flags = MyIntFlag.ONE | MyIntFlag.TWO | 8
        copied = copy.copy(flags)
        deep = copy.deepcopy(flags)
        self.assertEqual(copied, flags)
        self.assertEqual(deep, flags)
        self.assertEqual(copied.value, 1 | 2 | 8)

    def test_namedtuple_as_value(self):
        # [FUNCTION DOC] Checks namedtuple values: as a plain Enum value the tuple is stored
        # whole and its fields are not exposed on the member; as a mixin the fields are
        # accessible; with a custom __new__ the fields are set by hand.
        from collections import namedtuple
        TTuple = namedtuple('TTuple', 'id a blist')
        class NTEnum(Enum):
            # [CLASS DOC] HELPER: Plain Enum with namedtuple values.
            NONE = TTuple(0, 0, [])
            A = TTuple(1, 2, [4])
            B = TTuple(2, 4, [0, 1, 2])
        self.assertEqual(repr(NTEnum.NONE), "<NTEnum.NONE: TTuple(id=0, a=0, blist=[])>")
        self.assertEqual(NTEnum.NONE.value, TTuple(id=0, a=0, blist=[]))
        self.assertEqual(
                [x.value for x in NTEnum],
                [TTuple(id=0, a=0, blist=[]), TTuple(id=1, a=2, blist=[4]), TTuple(id=2, a=4, blist=[0, 1, 2])],
                )

        self.assertRaises(AttributeError, getattr, NTEnum.NONE, 'id')
        #
        class NTCEnum(TTuple, Enum):
            # [CLASS DOC] HELPER: Enum with the namedtuple class as mixin; members built from
            # plain tuples.
            NONE = 0, 0, []
            A = 1, 2, [4]
            B = 2, 4, [0, 1, 2]
        self.assertEqual(repr(NTCEnum.NONE), "<NTCEnum.NONE: TTuple(id=0, a=0, blist=[])>")
        self.assertEqual(NTCEnum.NONE.value, TTuple(id=0, a=0, blist=[]))
        self.assertEqual(NTCEnum.NONE.id, 0)
        self.assertEqual(NTCEnum.A.a, 2)
        self.assertEqual(NTCEnum.B.blist, [0, 1 ,2])
        self.assertEqual(
                [x.value for x in NTCEnum],
                [TTuple(id=0, a=0, blist=[]), TTuple(id=1, a=2, blist=[4]), TTuple(id=2, a=4, blist=[0, 1, 2])],
                )
        #
        class NTDEnum(Enum):
            # [CLASS DOC] HELPER: Enum with namedtuple values and a __new__ that unpacks them
            # into id, a and blist.
            def __new__(cls, id, a, blist):
                # [FUNCTION DOC] HELPER: Creates a member and sets id, a and blist attributes.
                member = object.__new__(cls)
                member.id = id
                member.a = a
                member.blist = blist
                return member
            NONE = TTuple(0, 0, [])
            A = TTuple(1, 2, [4])
            B = TTuple(2, 4, [0, 1, 2])
        self.assertEqual(repr(NTDEnum.NONE), "<NTDEnum.NONE: TTuple(id=0, a=0, blist=[])>")
        self.assertEqual(NTDEnum.NONE.id, 0)
        self.assertEqual(NTDEnum.A.a, 2)
        self.assertEqual(NTDEnum.B.blist, [0, 1 ,2])

    def test_flag_with_custom_new(self):
        # [FUNCTION DOC] Checks a Flag/IntFlag with a custom __new__ that turns a character code
        # into a bit position: _all_bits_ and _flag_mask_ are computed from those large values,
        # with and without aliases, and with the KEEP boundary.
        class FlagFromChar(IntFlag):
            # [CLASS DOC] HELPER: IntFlag with a __new__ turning the character code into a
            # one-bit value; member a.
            def __new__(cls, c):
                # [FUNCTION DOC] HELPER: Creates an int member whose value is one shifted left
                # by the code.
                value = 1 << c
                self = int.__new__(cls, value)
                self._value_ = value
                return self
            #
            a = ord('a')
        #
        self.assertEqual(FlagFromChar._all_bits_, 316912650057057350374175801343)
        self.assertEqual(FlagFromChar._flag_mask_, 158456325028528675187087900672)
        self.assertEqual(FlagFromChar.a, 158456325028528675187087900672)
        self.assertEqual(FlagFromChar.a|1, 158456325028528675187087900673)
        #
        #
        class FlagFromChar(Flag):
            # [CLASS DOC] HELPER: Flag with an object based __new__ shifting one by the code;
            # members a and z.
            def __new__(cls, c):
                # [FUNCTION DOC] HELPER: Creates an object member whose value is one shifted
                # left by the code.
                value = 1 << c
                self = object.__new__(cls)
                self._value_ = value
                return self
            #
            a = ord('a')
            z = 1
        #
        self.assertEqual(FlagFromChar._all_bits_, 316912650057057350374175801343)
        self.assertEqual(FlagFromChar._flag_mask_, 158456325028528675187087900674)
        self.assertEqual(FlagFromChar.a.value, 158456325028528675187087900672)
        self.assertEqual((FlagFromChar.a|FlagFromChar.z).value, 158456325028528675187087900674)
        #
        #
        class FlagFromChar(int, Flag, boundary=KEEP):
            # [CLASS DOC] HELPER: (int, Flag) with KEEP boundary and a shifting __new__;
            # member a.
            def __new__(cls, c):
                # [FUNCTION DOC] HELPER: Creates an int member whose value is one shifted left
                # by the code.
                value = 1 << c
                self = int.__new__(cls, value)
                self._value_ = value
                return self
            #
            a = ord('a')
        #
        self.assertEqual(FlagFromChar._all_bits_, 316912650057057350374175801343)
        self.assertEqual(FlagFromChar._flag_mask_, 158456325028528675187087900672)
        self.assertEqual(FlagFromChar.a, 158456325028528675187087900672)
        self.assertEqual(FlagFromChar.a|1, 158456325028528675187087900673)

    def test_init_exception(self):
        # [FUNCTION DOC] Checks that errors from mixin __init__ propagate: a TypeError for the
        # mismatched initializer signature, and the mixin's ValueError when __new__ calls it.
        class Base:
            # [CLASS DOC] HELPER: Mixin whose __new__ makes a plain object and whose __init__
            # always raises ValueError.
            def __new__(cls, *args):
                # [FUNCTION DOC] HELPER: Creates an instance without arguments.
                return object.__new__(cls)
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Always raises ValueError mentioning the argument.
                raise ValueError("I don't like", x)
        with self.assertRaises(TypeError):
            class MyEnum(Base, enum.Enum):
                # [CLASS DOC] HELPER: Invalid Enum with Base whose __init__ takes one argument
                # but is called with a value; must raise.
                A = 'a'
                def __init__(self, y):
                    # [FUNCTION DOC] HELPER: Stores y on the member.
                    self.y = y
        with self.assertRaises(ValueError):
            class MyEnum(Base, enum.Enum):
                # [CLASS DOC] HELPER: Enum with Base whose __new__ builds a Base value, which
                # makes the base initializer raise.
                A = 'a'
                def __init__(self, y):
                    # [FUNCTION DOC] HELPER: Stores y on the member.
                    self.y = y
                def __new__(cls, value):
                    # [FUNCTION DOC] HELPER: Creates a member whose value is a Base object built
                    # from value (raises ValueError).
                    member = Base.__new__(cls)
                    member._value_ = Base(value)
                    return member

    def test_extra_member_creation(self):
        # [FUNCTION DOC] Checks a metaclass can add extra members by inserting entries into the
        # class namespace before creation; the added _DESC members appear after the originals in
        # iteration order.
        class IDEnumMeta(EnumMeta):
            # [CLASS DOC] HELPER: EnumMeta subclass that adds NAME_DESC entries for each
            # member name.
            def __new__(metacls, cls, bases, classdict, **kwds):
                # [FUNCTION DOC] HELPER: For every member name adds an entry named with _DESC
                # and a minus-prefixed value to the class namespace, then delegates to the
                # parent metaclass __new__.
                # add new entries to classdict
                for name in classdict.member_names:
                    classdict[f'{name}_DESC'] = f'-{classdict[name]}'
                return super().__new__(metacls, cls, bases, classdict, **kwds)
        class IDEnum(StrEnum, metaclass=IDEnumMeta):
            # [CLASS DOC] HELPER: Empty StrEnum using the IDEnumMeta metaclass.
            pass
        class MyEnum(IDEnum):
            # [CLASS DOC] HELPER: Subclass with members ID and NAME.
            ID = 'id'
            NAME = 'name'
        self.assertEqual(list(MyEnum), [MyEnum.ID, MyEnum.NAME, MyEnum.ID_DESC, MyEnum.NAME_DESC])

    def test_add_alias(self):
        # [FUNCTION DOC] Checks _add_alias_ makes a new name resolve to an existing member,
        # including a name that collides with a mixin property (the alias on the class wins for
        # class lookup while members keep the property value).
        class mixin:
            # [CLASS DOC] HELPER: Plain mixin with a property ORG returning huh.
            @property
            def ORG(self):
                # [FUNCTION DOC] HELPER: Property returning the string huh.
                return 'huh'
        class Color(mixin, Enum):
            # [CLASS DOC] HELPER: Enum with the mixin and RED, GREEN, BLUE.
            RED = 1
            GREEN = 2
            BLUE = 3
        Color.RED._add_alias_('ROJO')
        self.assertIs(Color.RED, Color['ROJO'])
        self.assertIs(Color.RED, Color.ROJO)
        Color.BLUE._add_alias_('ORG')
        self.assertIs(Color.BLUE, Color['ORG'])
        self.assertIs(Color.BLUE, Color.ORG)
        self.assertEqual(Color.RED.ORG, 'huh')
        self.assertEqual(Color.GREEN.ORG, 'huh')
        self.assertEqual(Color.BLUE.ORG, 'huh')
        self.assertEqual(Color.ORG.ORG, 'huh')

    def test_add_value_alias_after_creation(self):
        # [FUNCTION DOC] Checks _add_value_alias_ added after class creation lets calling with
        # the extra value return the member.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum RED, GREEN, BLUE.
            RED = 1
            GREEN = 2
            BLUE = 3
        Color.RED._add_value_alias_(5)
        self.assertIs(Color.RED, Color(5))

    def test_add_value_alias_during_creation(self):
        # [FUNCTION DOC] Checks adding value aliases inside __new__ during class creation: extra
        # tuple elements become alternative lookup values.
        class Types(Enum):
            # [CLASS DOC] HELPER: Enum with members given as an int plus optional string aliases
            # and a __new__ registering them.
            Unknown = 0,
            Source  = 1, 'src'
            NetList = 2, 'nl'
            def __new__(cls, int_value, *value_aliases):
                # [FUNCTION DOC] HELPER: Creates a member with the int as value and registers
                # each extra argument as a value alias.
                member = object.__new__(cls)
                member._value_ = int_value
                for alias in value_aliases:
                    member._add_value_alias_(alias)
                return member
        self.assertIs(Types(0), Types.Unknown)
        self.assertIs(Types(1), Types.Source)
        self.assertIs(Types('src'), Types.Source)
        self.assertIs(Types(2), Types.NetList)
        self.assertIs(Types('nl'), Types.NetList)

    def test_second_tuple_item_is_falsey(self):
        # [FUNCTION DOC] Checks lookup by unpacked tuple arguments works even when the second
        # element is falsy (zero).
        class Cardinal(Enum):
            # [CLASS DOC] HELPER: Enum of direction tuples such as (1, 0) and (0, 1).
            RIGHT = (1, 0)
            UP = (0, 1)
            LEFT = (-1, 0)
            DOWN = (0, -1)
        self.assertIs(Cardinal(1, 0), Cardinal.RIGHT)
        self.assertIs(Cardinal(-1, 0), Cardinal.LEFT)

    def test_no_members(self):
        # [FUNCTION DOC] Checks calling Enum or Flag directly with a value raises TypeError
        # about having no members.
        with self.assertRaisesRegex(
                TypeError,
                'has no members',
            ):
            Enum(7)
        with self.assertRaisesRegex(
                TypeError,
                'has no members',
            ):
            Flag(7)

    def test_empty_names(self):
        # [FUNCTION DOC] Checks the functional API with empty names (string, list, mapping) and
        # type None or int gives an empty enum that raises TypeError when called; a non-iterable
        # names argument raises TypeError.
        for nothing in '', [], {}:
            for e_type in None, int:
                empty_enum = Enum('empty_enum', nothing, type=e_type)
                self.assertEqual(len(empty_enum), 0)
                self.assertRaisesRegex(TypeError, 'has no members', empty_enum, 0)
        self.assertRaisesRegex(TypeError, '.int. object is not iterable', Enum, 'bad_enum', names=0)
        self.assertRaisesRegex(TypeError, '.int. object is not iterable', Enum, 'bad_enum', 0, type=int)

    def test_nonhashable_matches_hashable(self):    # issue 125710
        # [FUNCTION DOC] Regression test (issue 125710): lookup with an unhashable set finds the
        # member whose value is an equal frozenset.
        class Directions(Enum):
            # [CLASS DOC] HELPER: Enum with frozenset values.
            DOWN_ONLY = frozenset({"sc"})
            UP_ONLY = frozenset({"cs"})
            UNRESTRICTED = frozenset({"sc", "cs"})
        self.assertIs(Directions({"sc"}), Directions.DOWN_ONLY)


class TestOrder(__TestCase):
    # [CLASS DOC] Tests the _order_ attribute that declares the expected member order;
    # mismatches raise TypeError.
    "test usage of the `_order_` attribute"

    def test_same_members(self):
        # [FUNCTION DOC] Checks a matching _order_ is accepted.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum red/green/blue with matching _order_.
            _order_ = 'red green blue'
            red = 1
            green = 2
            blue = 3

    def test_same_members_with_aliases(self):
        # [FUNCTION DOC] Checks a matching _order_ is accepted when the enum has an alias.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with matching _order_ and alias verde.
            _order_ = 'red green blue'
            red = 1
            green = 2
            blue = 3
            verde = green

    def test_same_members_wrong_order(self):
        # [FUNCTION DOC] Checks members defined in a different order than _order_
        # raise TypeError.
        with self.assertRaisesRegex(TypeError, 'member order does not match _order_'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum with members in the wrong order.
                _order_ = 'red green blue'
                red = 1
                blue = 3
                green = 2

    def test_order_has_extra_members(self):
        # [FUNCTION DOC] Checks _order_ listing a name that is not a member raises TypeError.
        with self.assertRaisesRegex(TypeError, 'member order does not match _order_'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum with _order_ having an extra name.
                _order_ = 'red green blue purple'
                red = 1
                green = 2
                blue = 3

    def test_order_has_extra_members_with_aliases(self):
        # [FUNCTION DOC] Checks _order_ with an extra name raises TypeError even when
        # aliases exist.
        with self.assertRaisesRegex(TypeError, 'member order does not match _order_'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum with extra name in _order_ and an alias.
                _order_ = 'red green blue purple'
                red = 1
                green = 2
                blue = 3
                verde = green

    def test_enum_has_extra_members(self):
        # [FUNCTION DOC] Checks an enum with a member missing from _order_ raises TypeError.
        with self.assertRaisesRegex(TypeError, 'member order does not match _order_'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum with an extra member not in _order_.
                _order_ = 'red green blue'
                red = 1
                green = 2
                blue = 3
                purple = 4

    def test_enum_has_extra_members_with_aliases(self):
        # [FUNCTION DOC] Checks a member missing from _order_ raises TypeError even with
        # an alias.
        with self.assertRaisesRegex(TypeError, 'member order does not match _order_'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid enum with an extra member and an alias.
                _order_ = 'red green blue'
                red = 1
                green = 2
                blue = 3
                purple = 4
                verde = green


class OldTestFlag(__TestCase):
    # [CLASS DOC] Older style tests of Flag: bitwise ops, boundaries, iteration, functional API
    # and more. Class attributes Perm, Open and Color provide fixture flag enums.
    """Tests of the Flags."""

    class Perm(Flag):
        # [CLASS DOC] HELPER: Flag with R=4, W=2, X=1.
        R, W, X = 4, 2, 1

    class Open(Flag):
        # [CLASS DOC] HELPER: Flag with RO=0, WO=1, RW=2, AC=3 and CE=1 shifted left by 19.
        RO = 0
        WO = 1
        RW = 2
        AC = 3
        CE = 1<<19

    class Color(Flag):
        # [CLASS DOC] HELPER: Flag with BLACK=0, RED, ROJO (alias of RED), GREEN, BLUE, PURPLE,
        # WHITE and alias BLANCO.
        BLACK = 0
        RED = 1
        ROJO = 1
        GREEN = 2
        BLUE = 4
        PURPLE = RED|BLUE
        WHITE = RED|GREEN|BLUE
        BLANCO = RED|GREEN|BLUE

    def test_or(self):
        # [FUNCTION DOC] Checks or on Perm members equals the value-level or, has type Perm, is
        # idempotent, and that a zero member or'd with another gives the other.
        Perm = self.Perm
        for i in Perm:
            for j in Perm:
                self.assertEqual((i | j), Perm(i.value | j.value))
                self.assertEqual((i | j).value, i.value | j.value)
                self.assertIs(type(i | j), Perm)
        for i in Perm:
            self.assertIs(i | i, i)
        Open = self.Open
        self.assertIs(Open.RO | Open.CE, Open.CE)

    def test_and(self):
        # [FUNCTION DOC] Checks and on all Perm members and composites gives value-level and, of
        # type Perm, with identity results for a member and the full mask, and zero for a zero
        # member and CE.
        Perm = self.Perm
        RW = Perm.R | Perm.W
        RX = Perm.R | Perm.X
        WX = Perm.W | Perm.X
        RWX = Perm.R | Perm.W | Perm.X
        values = list(Perm) + [RW, RX, WX, RWX, Perm(0)]
        for i in values:
            for j in values:
                self.assertEqual((i & j).value, i.value & j.value)
                self.assertIs(type(i & j), Perm)
        for i in Perm:
            self.assertIs(i & i, i)
            self.assertIs(i & RWX, i)
            self.assertIs(RWX & i, i)
        Open = self.Open
        self.assertIs(Open.RO & Open.CE, Open.RO)

    def test_xor(self):
        # [FUNCTION DOC] Checks xor on Perm members gives value-level xor, xor with zero is
        # identity, and xor of identical values gives zero.
        Perm = self.Perm
        for i in Perm:
            for j in Perm:
                self.assertEqual((i ^ j).value, i.value ^ j.value)
                self.assertIs(type(i ^ j), Perm)
        for i in Perm:
            self.assertIs(i ^ Perm(0), i)
            self.assertIs(Perm(0) ^ i, i)
        Open = self.Open
        self.assertIs(Open.RO ^ Open.CE, Open.CE)
        self.assertIs(Open.CE ^ Open.CE, Open.RO)

    def test_bool(self):
        # [FUNCTION DOC] Checks Perm members are truthy and Open members are truthy exactly when
        # their value is non-zero.
        Perm = self.Perm
        for f in Perm:
            self.assertTrue(f)
        Open = self.Open
        for f in Open:
            self.assertEqual(bool(f.value), bool(f))

    def test_boundary(self):
        # [FUNCTION DOC] Checks the boundary option: the default is STRICT, explicit CONFORM,
        # STRICT, EJECT, KEEP are recorded; invalid values raise ValueError, are masked
        # (CONFORM), returned as ints (EJECT) or kept (KEEP); also checks multi-bit aliases and
        # composite members with skipped bits.
        self.assertIs(enum.Flag._boundary_, STRICT)
        class Iron(Flag, boundary=CONFORM):
            # [CLASS DOC] HELPER: Flag with CONFORM boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Iron._boundary_, CONFORM)
        #
        class Water(Flag, boundary=STRICT):
            # [CLASS DOC] HELPER: Flag with STRICT boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Water._boundary_, STRICT)
        #
        class Space(Flag, boundary=EJECT):
            # [CLASS DOC] HELPER: Flag with EJECT boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Space._boundary_, EJECT)
        #
        class Bizarre(Flag, boundary=KEEP):
            # [CLASS DOC] HELPER: Flag with KEEP boundary and multi-bit members b=3, c=4, d=6.
            b = 3
            c = 4
            d = 6
        #
        self.assertRaisesRegex(ValueError, 'invalid value 7', Water, 7)
        #
        self.assertIs(Iron(7), Iron.ONE|Iron.TWO)
        self.assertIs(Iron(~9), Iron.TWO)
        #
        self.assertEqual(Space(7), 7)
        self.assertTrue(type(Space(7)) is int)
        #
        self.assertEqual(list(Bizarre), [Bizarre.c])
        self.assertIs(Bizarre(3), Bizarre.b)
        self.assertIs(Bizarre(6), Bizarre.d)
        #
        class SkipFlag(enum.Flag):
            # [CLASS DOC] HELPER: Flag where C combines bit 4 and member B (skips bit values).
            A = 1
            B = 2
            C = 4 | B
        #
        self.assertTrue(SkipFlag.C in (SkipFlag.A|SkipFlag.C))
        self.assertRaisesRegex(ValueError, 'SkipFlag.. invalid value 42', SkipFlag, 42)
        #
        class SkipIntFlag(enum.IntFlag):
            # [CLASS DOC] HELPER: IntFlag where C combines bit 4 and member B.
            A = 1
            B = 2
            C = 4 | B
        #
        self.assertTrue(SkipIntFlag.C in (SkipIntFlag.A|SkipIntFlag.C))
        self.assertEqual(SkipIntFlag(42).value, 42)
        #
        class MethodHint(Flag):
            # [CLASS DOC] HELPER: Flag with a mask member OnlyMask covering several
            # other members.
            HiddenText = 0x10
            DigitsOnly = 0x01
            LettersOnly = 0x02
            OnlyMask = 0x0f
        #
        self.assertEqual(str(MethodHint.HiddenText|MethodHint.OnlyMask), 'MethodHint.HiddenText|DigitsOnly|LettersOnly|OnlyMask')


    def test_iter(self):
        # [FUNCTION DOC] Checks iterating a Flag yields only single-bit members, skipping zero,
        # aliases and composites.
        Color = self.Color
        Open = self.Open
        self.assertEqual(list(Color), [Color.RED, Color.GREEN, Color.BLUE])
        self.assertEqual(list(Open), [Open.WO, Open.RW, Open.CE])

    def test_programatic_function_string(self):
        # [FUNCTION DOC] Checks Flag functional API with a name string: three members, values 1,
        # 2, 4 as plain ints, names match.
        Perm = Flag('Perm', 'R W X')
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_string_with_start(self):
        # [FUNCTION DOC] Checks Flag functional API with start=8: values 8, 16, 32.
        Perm = Flag('Perm', 'R W X', start=8)
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 8<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_string_list(self):
        # [FUNCTION DOC] Checks Flag functional API with a list of names: values 1, 2, 4.
        Perm = Flag('Perm', ['R', 'W', 'X'])
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_iterable(self):
        # [FUNCTION DOC] Checks Flag functional API with (name, value) pairs 2, 8, 32: values
        # preserved as plain ints, names match.
        Perm = Flag('Perm', (('R', 2), ('W', 8), ('X', 32)))
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<(2*i+1)
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_from_dict(self):
        # [FUNCTION DOC] Checks Flag functional API with an ordered mapping of names to 2,
        # 8, 32.
        Perm = Flag('Perm', OrderedDict((('R', 2), ('W', 8), ('X', 32))))
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<(2*i+1)
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    @reraise_if_not_enum(
        FlagStooges,
        FlagStoogesWithZero,
        IntFlagStooges,
        IntFlagStoogesWithZero,
    )
    def test_pickle(self):
        # [FUNCTION DOC] Pickle round trips for Flag and IntFlag module fixtures (with and
        # without a zero member): single and combined members, zero results, composite aliases
        # and the classes. Reraises stored creation errors if any fixture failed.
        test_pickle_dump_load(self.assertIs, FlagStooges.CURLY)
        test_pickle_dump_load(self.assertEqual,
                        FlagStooges.CURLY|FlagStooges.MOE)
        test_pickle_dump_load(self.assertEqual,
                        FlagStooges.CURLY&~FlagStooges.CURLY)
        test_pickle_dump_load(self.assertIs, FlagStooges)
        test_pickle_dump_load(self.assertEqual, FlagStooges.BIG)
        test_pickle_dump_load(self.assertEqual,
                        FlagStooges.CURLY|FlagStooges.BIG)

        test_pickle_dump_load(self.assertIs, FlagStoogesWithZero.CURLY)
        test_pickle_dump_load(self.assertEqual,
                        FlagStoogesWithZero.CURLY|FlagStoogesWithZero.MOE)
        test_pickle_dump_load(self.assertIs, FlagStoogesWithZero.NOFLAG)
        test_pickle_dump_load(self.assertEqual, FlagStoogesWithZero.BIG)
        test_pickle_dump_load(self.assertEqual,
                        FlagStoogesWithZero.CURLY|FlagStoogesWithZero.BIG)

        test_pickle_dump_load(self.assertIs, IntFlagStooges.CURLY)
        test_pickle_dump_load(self.assertEqual,
                        IntFlagStooges.CURLY|IntFlagStooges.MOE)
        test_pickle_dump_load(self.assertEqual,
                        IntFlagStooges.CURLY|IntFlagStooges.MOE|0x30)
        test_pickle_dump_load(self.assertEqual, IntFlagStooges(0))
        test_pickle_dump_load(self.assertEqual, IntFlagStooges(0x30))
        test_pickle_dump_load(self.assertIs, IntFlagStooges)
        test_pickle_dump_load(self.assertEqual, IntFlagStooges.BIG)
        test_pickle_dump_load(self.assertEqual, IntFlagStooges.BIG|1)
        test_pickle_dump_load(self.assertEqual,
                        IntFlagStooges.CURLY|IntFlagStooges.BIG)

        test_pickle_dump_load(self.assertIs, IntFlagStoogesWithZero.CURLY)
        test_pickle_dump_load(self.assertEqual,
                        IntFlagStoogesWithZero.CURLY|IntFlagStoogesWithZero.MOE)
        test_pickle_dump_load(self.assertIs, IntFlagStoogesWithZero.NOFLAG)
        test_pickle_dump_load(self.assertEqual, IntFlagStoogesWithZero.BIG)
        test_pickle_dump_load(self.assertEqual, IntFlagStoogesWithZero.BIG|1)
        test_pickle_dump_load(self.assertEqual,
                        IntFlagStoogesWithZero.CURLY|IntFlagStoogesWithZero.BIG)

    def test_contains_tf(self):
        # [FUNCTION DOC] Checks the in operator for Flag classes: members of a different Flag
        # and strings are not contained, own members and the raw value 1 are.
        Open = self.Open
        Color = self.Color
        self.assertFalse(Color.BLACK in Open)
        self.assertFalse(Open.RO in Color)
        self.assertFalse('BLACK' in Color)
        self.assertFalse('RO' in Open)
        self.assertTrue(Color.BLACK in Color)
        self.assertTrue(Open.RO in Open)
        self.assertTrue(1 in Color)
        self.assertTrue(1 in Open)

    def test_member_contains(self):
        # [FUNCTION DOC] Checks that a Perm member is contained in a composite exactly when its
        # bit is set in it.
        Perm = self.Perm
        R, W, X = Perm
        RW = R | W
        RX = R | X
        WX = W | X
        RWX = R | W | X
        self.assertTrue(R in RW)
        self.assertTrue(R in RX)
        self.assertTrue(R in RWX)
        self.assertTrue(W in RW)
        self.assertTrue(W in WX)
        self.assertTrue(W in RWX)
        self.assertTrue(X in RX)
        self.assertTrue(X in WX)
        self.assertTrue(X in RWX)
        self.assertFalse(R in WX)
        self.assertFalse(W in RX)
        self.assertFalse(X in RW)

    def test_member_iter(self):
        # [FUNCTION DOC] Checks iterating a composite Color member yields its single-bit members
        # in order (empty for zero).
        Color = self.Color
        self.assertEqual(list(Color.BLACK), [])
        self.assertEqual(list(Color.PURPLE), [Color.RED, Color.BLUE])
        self.assertEqual(list(Color.BLUE), [Color.BLUE])
        self.assertEqual(list(Color.GREEN), [Color.GREEN])
        self.assertEqual(list(Color.WHITE), [Color.RED, Color.GREEN, Color.BLUE])
        self.assertEqual(list(Color.WHITE), [Color.RED, Color.GREEN, Color.BLUE])

    def test_member_length(self):
        # [FUNCTION DOC] Checks the length of a Color member equals its number of set bits (0,
        # 1, 2, 3).
        self.assertEqual(self.Color.__len__(self.Color.BLACK), 0)
        self.assertEqual(self.Color.__len__(self.Color.GREEN), 1)
        self.assertEqual(self.Color.__len__(self.Color.PURPLE), 2)
        self.assertEqual(self.Color.__len__(self.Color.BLANCO), 3)

    def test_number_reset_and_order_cleanup(self):
        # [FUNCTION DOC] Checks that an alias with an explicit value resets auto numbering for
        # later Flag members and that _order_ ignoring aliases passes: DOS aliases TWO and later
        # members continue as 8 and 16.
        class Confused(Flag):
            # [CLASS DOC] HELPER: Flag with auto members and an explicit alias DOS=2 in the
            # middle, with an _order_.
            _order_ = 'ONE TWO FOUR DOS EIGHT SIXTEEN'
            ONE = auto()
            TWO = auto()
            FOUR = auto()
            DOS = 2
            EIGHT = auto()
            SIXTEEN = auto()
        self.assertEqual(
                list(Confused),
                [Confused.ONE, Confused.TWO, Confused.FOUR, Confused.EIGHT, Confused.SIXTEEN])
        self.assertIs(Confused.TWO, Confused.DOS)
        self.assertEqual(Confused.DOS._value_, 2)
        self.assertEqual(Confused.EIGHT._value_, 8)
        self.assertEqual(Confused.SIXTEEN._value_, 16)

    def test_aliases(self):
        # [FUNCTION DOC] Checks aliases resolve by value and name to the canonical member (ROJO
        # to RED, BLANCO to WHITE) and AC in Open.
        Color = self.Color
        self.assertEqual(Color(1).name, 'RED')
        self.assertEqual(Color['ROJO'].name, 'RED')
        self.assertEqual(Color(7).name, 'WHITE')
        self.assertEqual(Color['BLANCO'].name, 'WHITE')
        self.assertIs(Color.BLANCO, Color.WHITE)
        Open = self.Open
        self.assertIs(Open['AC'], Open.AC)

    def test_auto_number(self):
        # [FUNCTION DOC] Checks Flag auto numbering gives powers of two 1, 2, 4.
        class Color(Flag):
            # [CLASS DOC] HELPER: Flag with three auto members.
            red = auto()
            blue = auto()
            green = auto()

        self.assertEqual(list(Color), [Color.red, Color.blue, Color.green])
        self.assertEqual(Color.red.value, 1)
        self.assertEqual(Color.blue.value, 2)
        self.assertEqual(Color.green.value, 4)

    def test_auto_number_garbage(self):
        # [FUNCTION DOC] Checks a non-int Flag value raises TypeError about an invalid
        # flag value.
        with self.assertRaisesRegex(TypeError, 'invalid flag value .not an int.'):
            class Color(Flag):
                # [CLASS DOC] HELPER: Invalid Flag with a string value.
                red = 'not an int'
                blue = auto()

    def test_duplicate_auto(self):
        # [FUNCTION DOC] Checks that a chained assignment with auto creates one member plus an
        # alias and later auto members continue sequentially.
        class Dupes(Enum):
            # [CLASS DOC] HELPER: Enum with first and alias primero sharing one auto value, then
            # second and third.
            first = primero = auto()
            second = auto()
            third = auto()
        self.assertEqual([Dupes.first, Dupes.second, Dupes.third], list(Dupes))

    def test_multiple_mixin(self):
        # [FUNCTION DOC] Checks Flag combined with mixins in different base orders: AllMixin
        # gives a class-level ALL property computing the union of all members; StrMixin
        # (re-assigned __str__) gives lowercase names; plain Flag str is ClassName.member.
        class AllMixin:
            # [CLASS DOC] HELPER: Mixin with a classproperty ALL that ORs all members and caches
            # it on the class.
            @classproperty
            def ALL(cls):
                # [FUNCTION DOC] HELPER: Returns the union of all members of the class (None if
                # empty) and stores it as ALL.
                members = list(cls)
                all_value = None
                if members:
                    all_value = members[0]
                    for member in members[1:]:
                        all_value |= member
                cls.ALL = all_value
                return all_value
        class StrMixin:
            # [CLASS DOC] HELPER: Mixin with __str__ returning the lowercase member name.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the lowercase member name.
                return self._name_.lower()
        class Color(AllMixin, Flag):
            # [CLASS DOC] HELPER: Flag with AllMixin and RED, GREEN, BLUE.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), 'Color.BLUE')
        class Color(AllMixin, StrMixin, Flag):
            # [CLASS DOC] HELPER: Flag with AllMixin and StrMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), 'blue')
        class Color(StrMixin, AllMixin, Flag):
            # [CLASS DOC] HELPER: Flag with StrMixin and AllMixin in that order.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), 'blue')

    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_unique_composite(self):
        # [FUNCTION DOC] Concurrency check: eight threads each look up values 0 to 255 in a Flag
        # that defines identity based equality and a value hash; no thread may fail and 256
        # distinct members must be seen. Skipped without working threading.
        # KNOWN ISSUE: an inline comment says 248 members while the assertion expects 256.
        # override __eq__ to be identity only
        class TestFlag(Flag):
            # [CLASS DOC] HELPER: Flag with eight auto members, __eq__ by identity and __hash__
            # by value.
            one = auto()
            two = auto()
            three = auto()
            four = auto()
            five = auto()
            six = auto()
            seven = auto()
            eight = auto()
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Identity comparison.
                return self is other
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the member value.
                return hash(self._value_)
        # have multiple threads competing to complete the composite members
        seen = set()
        failed = False
        def cycle_enum():
            # [FUNCTION DOC] HELPER: Thread target; for 256 values looks up the flag and adds it
            # to a shared set, setting the failed flag in the enclosing scope on any exception.
            nonlocal failed
            try:
                for i in range(256):
                    seen.add(TestFlag(i))
            except Exception:
                failed = True
        threads = [
                threading.Thread(target=cycle_enum)
                for _ in range(8)
                ]
        with threading_helper.start_threads(threads):
            pass
        # check that only 248 members were created
        self.assertFalse(
                failed,
                'at least one thread failed while creating composite members')
        self.assertEqual(256, len(seen), 'too many composite members created')

    def test_init_subclass(self):
        # [FUNCTION DOC] Checks __init_subclass__ chaining on Flag subclasses: hooks run in the
        # right order and define attributes only on classes where the chain was not cut; a
        # subclass whose hook does nothing stops the chain.
        class MyEnum(Flag):
            # [CLASS DOC] HELPER: Flag base whose __init_subclass__ asserts no _test attribute
            # exists and sets _test1.
            def __init_subclass__(cls, **kwds):
                # [FUNCTION DOC] HELPER: Calls the parent hook, asserts the class has no own
                # _test attribute, sets _test1.
                super().__init_subclass__(**kwds)
                self.assertFalse(cls.__dict__.get('_test', False))
                cls._test1 = 'MyEnum'
        #
        class TheirEnum(MyEnum):
            # [CLASS DOC] HELPER: Subclass whose hook calls up the chain and sets _test2.
            def __init_subclass__(cls, **kwds):
                # [FUNCTION DOC] HELPER: Calls the parent hook and sets _test2 to TheirEnum.
                super(TheirEnum, cls).__init_subclass__(**kwds)
                cls._test2 = 'TheirEnum'
        class WhoseEnum(TheirEnum):
            # [CLASS DOC] HELPER: Subclass with a hook that does nothing (breaks the chain).
            def __init_subclass__(cls, **kwds):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass
        class NoEnum(WhoseEnum):
            # [CLASS DOC] HELPER: Concrete subclass with member ONE.
            ONE = 1
        self.assertEqual(TheirEnum.__dict__['_test1'], 'MyEnum')
        self.assertEqual(WhoseEnum.__dict__['_test1'], 'MyEnum')
        self.assertEqual(WhoseEnum.__dict__['_test2'], 'TheirEnum')
        self.assertFalse(NoEnum.__dict__.get('_test1', False))
        self.assertFalse(NoEnum.__dict__.get('_test2', False))
        #
        class OurEnum(MyEnum):
            # [CLASS DOC] HELPER: Subclass whose hook sets _test2 without calling the
            # parent hook.
            def __init_subclass__(cls, **kwds):
                # [FUNCTION DOC] HELPER: Sets _test2 to OurEnum without calling the parent.
                cls._test2 = 'OurEnum'
        class WhereEnum(OurEnum):
            # [CLASS DOC] HELPER: Subclass with a hook that does nothing.
            def __init_subclass__(cls, **kwds):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass
        class NeverEnum(WhereEnum):
            # [CLASS DOC] HELPER: Concrete subclass with member ONE.
            ONE = 1
        self.assertEqual(OurEnum.__dict__['_test1'], 'MyEnum')
        self.assertFalse(WhereEnum.__dict__.get('_test1', False))
        self.assertEqual(WhereEnum.__dict__['_test2'], 'OurEnum')
        self.assertFalse(NeverEnum.__dict__.get('_test1', False))
        self.assertFalse(NeverEnum.__dict__.get('_test2', False))


class OldTestIntFlag(__TestCase):
    # [CLASS DOC] Older style tests of IntFlag: int behavior, bitwise ops with ints, invert,
    # boundaries, functional API and global repr. Class attributes Perm, Open, Color and Skip
    # give fixture enums.
    """Tests of the IntFlags."""

    class Perm(IntFlag):
        # [CLASS DOC] HELPER: IntFlag with R=4, W=2, X=1.
        R = 1 << 2
        W = 1 << 1
        X = 1 << 0

    class Open(IntFlag):
        # [CLASS DOC] HELPER: IntFlag with RO=0, WO=1, RW=2, AC=3, CE shifted 19 bits.
        RO = 0
        WO = 1
        RW = 2
        AC = 3
        CE = 1<<19

    class Color(IntFlag):
        # [CLASS DOC] HELPER: IntFlag with BLACK=0, RED, ROJO alias, GREEN, BLUE, PURPLE, WHITE
        # and alias BLANCO.
        BLACK = 0
        RED = 1
        ROJO = 1
        GREEN = 2
        BLUE = 4
        PURPLE = RED|BLUE
        WHITE = RED|GREEN|BLUE
        BLANCO = RED|GREEN|BLUE

    class Skip(IntFlag):
        # [CLASS DOC] HELPER: IntFlag with FIRST=1, SECOND=2, EIGHTH=8 (a gap in bits).
        FIRST = 1
        SECOND = 2
        EIGHTH = 8

    def test_type(self):
        # [FUNCTION DOC] Checks IntFlag members are int-based instances of their class, equal
        # their values, and combinations stay in the class.
        Perm = self.Perm
        self.assertTrue(Perm._member_type_ is int)
        Open = self.Open
        for f in Perm:
            self.assertTrue(isinstance(f, Perm))
            self.assertEqual(f, f.value)
        self.assertTrue(isinstance(Perm.W | Perm.X, Perm))
        self.assertEqual(Perm.W | Perm.X, 3)
        for f in Open:
            self.assertTrue(isinstance(f, Open))
            self.assertEqual(f, f.value)
        self.assertTrue(isinstance(Open.WO | Open.RW, Open))
        self.assertEqual(Open.WO | Open.RW, 3)

    @reraise_if_not_enum(HeadlightsK)
    def test_global_repr_keep(self):
        # [FUNCTION DOC] Checks the repr of the global KEEP-boundary IntFlag HeadlightsK for
        # zero, mixed known and unknown bits and an unknown bit alone.
        self.assertEqual(
                repr(HeadlightsK(0)),
                '%s.OFF_K' % SHORT_MODULE,
                )
        self.assertEqual(
                repr(HeadlightsK(2**0 + 2**2 + 2**3)),
                '%(m)s.LOW_BEAM_K|%(m)s.FOG_K|8' % {'m': SHORT_MODULE},
                )
        self.assertEqual(
                repr(HeadlightsK(2**3)),
                '%(m)s.HeadlightsK(8)' % {'m': SHORT_MODULE},
                )

    @reraise_if_not_enum(HeadlightsC)
    def test_global_repr_conform1(self):
        # [FUNCTION DOC] Checks the repr of the global CONFORM-boundary IntFlag HeadlightsC:
        # unknown bits are dropped, leaving the zero member.
        self.assertEqual(
                repr(HeadlightsC(0)),
                '%s.OFF_C' % SHORT_MODULE,
                )
        self.assertEqual(
                repr(HeadlightsC(2**0 + 2**2 + 2**3)),
                '%(m)s.LOW_BEAM_C|%(m)s.FOG_C' % {'m': SHORT_MODULE},
                )
        self.assertEqual(
                repr(HeadlightsC(2**3)),
                '%(m)s.OFF_C' % {'m': SHORT_MODULE},
                )

    @reraise_if_not_enum(NoName)
    @unittest.skip('Fails because repr(NoName.ONE) == "__main__.ONE"')
    def test_global_enum_str(self):
        # [FUNCTION DOC] Skipped test of repr and str for the global Flag NoName; the
        # expectations assume the module is named test_enum.
        self.assertEqual(repr(NoName.ONE), 'test_enum.ONE')
        self.assertEqual(repr(NoName(0)), 'test_enum.NoName(0)')
        self.assertEqual(str(NoName.ONE & NoName.TWO), 'NoName(0)')
        self.assertEqual(str(NoName(0)), 'NoName(0)')

    def test_format(self):
        # [FUNCTION DOC] Checks format with an empty spec gives the numeric value for IntFlag,
        # and uses the member names when __str__ is overridden to return the name.
        Perm = self.Perm
        self.assertEqual(format(Perm.R, ''), '4')
        self.assertEqual(format(Perm.R | Perm.X, ''), '5')
        #
        class NewPerm(IntFlag):
            # [CLASS DOC] HELPER: IntFlag R, W, X whose __str__ returns the member name.
            R = 1 << 2
            W = 1 << 1
            X = 1 << 0
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the member name.
                return self._name_
        self.assertEqual(format(NewPerm.R, ''), 'R')
        self.assertEqual(format(NewPerm.R | Perm.X, ''), 'R|X')

    def test_or(self):
        # [FUNCTION DOC] Checks or between IntFlag members and ints in both orders: values equal
        # int or, results are of type Perm, idempotence and zero identity.
        Perm = self.Perm
        for i in Perm:
            for j in Perm:
                self.assertEqual(i | j, i.value | j.value)
                self.assertEqual((i | j).value, i.value | j.value)
                self.assertIs(type(i | j), Perm)
            for j in range(8):
                self.assertEqual(i | j, i.value | j)
                self.assertEqual((i | j).value, i.value | j)
                self.assertIs(type(i | j), Perm)
                self.assertEqual(j | i, j | i.value)
                self.assertEqual((j | i).value, j | i.value)
                self.assertIs(type(j | i), Perm)
        for i in Perm:
            self.assertIs(i | i, i)
            self.assertIs(i | 0, i)
            self.assertIs(0 | i, i)
        Open = self.Open
        self.assertIs(Open.RO | Open.CE, Open.CE)

    def test_and(self):
        # [FUNCTION DOC] Checks and between IntFlag members/composites and ints in both orders
        # equals value-level and, with Perm results.
        Perm = self.Perm
        RW = Perm.R | Perm.W
        RX = Perm.R | Perm.X
        WX = Perm.W | Perm.X
        RWX = Perm.R | Perm.W | Perm.X
        values = list(Perm) + [RW, RX, WX, RWX, Perm(0)]
        for i in values:
            for j in values:
                self.assertEqual(i & j, i.value & j.value, 'i is %r, j is %r' % (i, j))
                self.assertEqual((i & j).value, i.value & j.value, 'i is %r, j is %r' % (i, j))
                self.assertIs(type(i & j), Perm, 'i is %r, j is %r' % (i, j))
            for j in range(8):
                self.assertEqual(i & j, i.value & j)
                self.assertEqual((i & j).value, i.value & j)
                self.assertIs(type(i & j), Perm)
                self.assertEqual(j & i, j & i.value)
                self.assertEqual((j & i).value, j & i.value)
                self.assertIs(type(j & i), Perm)
        for i in Perm:
            self.assertIs(i & i, i)
            self.assertIs(i & 7, i)
            self.assertIs(7 & i, i)
        Open = self.Open
        self.assertIs(Open.RO & Open.CE, Open.RO)

    def test_xor(self):
        # [FUNCTION DOC] Checks xor between IntFlag members and ints in both orders equals
        # value-level xor with Perm results; xor with zero is identity.
        Perm = self.Perm
        for i in Perm:
            for j in Perm:
                self.assertEqual(i ^ j, i.value ^ j.value)
                self.assertEqual((i ^ j).value, i.value ^ j.value)
                self.assertIs(type(i ^ j), Perm)
            for j in range(8):
                self.assertEqual(i ^ j, i.value ^ j)
                self.assertEqual((i ^ j).value, i.value ^ j)
                self.assertIs(type(i ^ j), Perm)
                self.assertEqual(j ^ i, j ^ i.value)
                self.assertEqual((j ^ i).value, j ^ i.value)
                self.assertIs(type(j ^ i), Perm)
        for i in Perm:
            self.assertIs(i ^ 0, i)
            self.assertIs(0 ^ i, i)
        Open = self.Open
        self.assertIs(Open.RO ^ Open.CE, Open.CE)
        self.assertIs(Open.CE ^ Open.CE, Open.RO)

    def test_invert(self):
        # [FUNCTION DOC] Checks inversion yields a Perm equal to the inverted value, double
        # inversion returns the original, and clearing a bit by and-with-invert works.
        Perm = self.Perm
        RW = Perm.R | Perm.W
        RX = Perm.R | Perm.X
        WX = Perm.W | Perm.X
        RWX = Perm.R | Perm.W | Perm.X
        values = list(Perm) + [RW, RX, WX, RWX, Perm(0)]
        for i in values:
            self.assertEqual(~i, (~i).value)
            self.assertIs(type(~i), Perm)
            self.assertEqual(~~i, i)
        for i in Perm:
            self.assertIs(~~i, i)
        Open = self.Open
        self.assertIs(Open.WO & ~Open.WO, Open.RO)
        self.assertIs((Open.WO|Open.CE) & ~Open.WO, Open.CE)

    def test_boundary(self):
        # [FUNCTION DOC] Checks IntFlag boundaries: default KEEP; STRICT raises ValueError on
        # invalid bits, CONFORM masks them, EJECT returns plain ints, KEEP keeps them; combining
        # members of different flags gives the left class with ints.
        self.assertIs(enum.IntFlag._boundary_, KEEP)
        class Simple(IntFlag, boundary=KEEP):
            # [CLASS DOC] HELPER: IntFlag with KEEP boundary and member SINGLE.
            SINGLE = 1
        #
        class Iron(IntFlag, boundary=STRICT):
            # [CLASS DOC] HELPER: IntFlag with STRICT boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Iron._boundary_, STRICT)
        #
        class Water(IntFlag, boundary=CONFORM):
            # [CLASS DOC] HELPER: IntFlag with CONFORM boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Water._boundary_, CONFORM)
        #
        class Space(IntFlag, boundary=EJECT):
            # [CLASS DOC] HELPER: IntFlag with EJECT boundary.
            ONE = 1
            TWO = 2
            EIGHT = 8
        self.assertIs(Space._boundary_, EJECT)
        #
        class Bizarre(IntFlag, boundary=KEEP):
            # [CLASS DOC] HELPER: IntFlag with KEEP boundary and multi-bit members.
            b = 3
            c = 4
            d = 6
        #
        self.assertRaisesRegex(ValueError, 'invalid value 5', Iron, 5)
        #
        self.assertIs(Water(7), Water.ONE|Water.TWO)
        self.assertIs(Water(~9), Water.TWO)
        #
        self.assertEqual(Space(7), 7)
        self.assertTrue(type(Space(7)) is int)
        #
        self.assertEqual(list(Bizarre), [Bizarre.c])
        self.assertIs(Bizarre(3), Bizarre.b)
        self.assertIs(Bizarre(6), Bizarre.d)
        #
        simple = Simple.SINGLE | Iron.TWO
        self.assertEqual(simple, 3)
        self.assertIsInstance(simple, Simple)
        self.assertEqual(repr(simple), '<Simple.SINGLE|<Iron.TWO: 2>: 3>')
        self.assertEqual(str(simple), '3')

    def test_iter(self):
        # [FUNCTION DOC] Checks iterating IntFlag classes yields single-bit members only.
        Color = self.Color
        Open = self.Open
        self.assertEqual(list(Color), [Color.RED, Color.GREEN, Color.BLUE])
        self.assertEqual(list(Open), [Open.WO, Open.RW, Open.CE])

    def test_programatic_function_string(self):
        # [FUNCTION DOC] Checks IntFlag functional API with a name string: values 1, 2, 4,
        # members equal ints.
        Perm = IntFlag('Perm', 'R W X')
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e, v)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_string_with_start(self):
        # [FUNCTION DOC] Checks IntFlag functional API with start=8: values 8, 16, 32.
        Perm = IntFlag('Perm', 'R W X', start=8)
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 8<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e, v)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_string_list(self):
        # [FUNCTION DOC] Checks IntFlag functional API with a list of names: values 1, 2, 4.
        Perm = IntFlag('Perm', ['R', 'W', 'X'])
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<i
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e, v)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_iterable(self):
        # [FUNCTION DOC] Checks IntFlag functional API with (name, value) pairs 2, 8, 32.
        Perm = IntFlag('Perm', (('R', 2), ('W', 8), ('X', 32)))
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<(2*i+1)
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e, v)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)

    def test_programatic_function_from_dict(self):
        # [FUNCTION DOC] Checks IntFlag functional API with an ordered mapping 2, 8, 32.
        Perm = IntFlag('Perm', OrderedDict((('R', 2), ('W', 8), ('X', 32))))
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 3, Perm)
        self.assertEqual(lst, [Perm.R, Perm.W, Perm.X])
        for i, n in enumerate('R W X'.split()):
            v = 1<<(2*i+1)
            e = Perm(v)
            self.assertEqual(e.value, v)
            self.assertEqual(type(e.value), int)
            self.assertEqual(e, v)
            self.assertEqual(e.name, n)
            self.assertIn(e, Perm)
            self.assertIs(type(e), Perm)


    def test_programatic_function_from_empty_list(self):
        # [FUNCTION DOC] Checks the functional API with an empty list produces empty IntFlag and
        # Enum classes.
        Perm = enum.IntFlag('Perm', [])
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 0, Perm)
        Thing = enum.Enum('Thing', [])
        lst = list(Thing)
        self.assertEqual(len(lst), len(Thing))
        self.assertEqual(len(Thing), 0, Thing)


    def test_programatic_function_from_empty_tuple(self):
        # [FUNCTION DOC] Checks the functional API with an empty tuple produces empty IntFlag
        # and Enum classes.
        Perm = enum.IntFlag('Perm', ())
        lst = list(Perm)
        self.assertEqual(len(lst), len(Perm))
        self.assertEqual(len(Perm), 0, Perm)
        Thing = enum.Enum('Thing', ())
        self.assertEqual(len(lst), len(Thing))
        self.assertEqual(len(Thing), 0, Thing)

    def test_contains_tf(self):
        # [FUNCTION DOC] Checks in operator for IntFlag classes: own members and ints are
        # contained, name strings are not.
        Open = self.Open
        Color = self.Color
        self.assertTrue(Color.GREEN in Color)
        self.assertTrue(Open.RW in Open)
        self.assertFalse('GREEN' in Color)
        self.assertFalse('RW' in Open)
        self.assertTrue(2 in Color)
        self.assertTrue(2 in Open)

    def test_member_contains(self):
        # [FUNCTION DOC] Checks a member is in a composite exactly when its bit is set; a string
        # operand raises TypeError.
        Perm = self.Perm
        R, W, X = Perm
        RW = R | W
        RX = R | X
        WX = W | X
        RWX = R | W | X
        self.assertTrue(R in RW)
        self.assertTrue(R in RX)
        self.assertTrue(R in RWX)
        self.assertTrue(W in RW)
        self.assertTrue(W in WX)
        self.assertTrue(W in RWX)
        self.assertTrue(X in RX)
        self.assertTrue(X in WX)
        self.assertTrue(X in RWX)
        self.assertFalse(R in WX)
        self.assertFalse(W in RX)
        self.assertFalse(X in RW)
        with self.assertRaises(TypeError):
            self.assertFalse('test' in RW)

    def test_member_iter(self):
        # [FUNCTION DOC] Checks iterating a composite IntFlag member yields its
        # single-bit members.
        Color = self.Color
        self.assertEqual(list(Color.BLACK), [])
        self.assertEqual(list(Color.PURPLE), [Color.RED, Color.BLUE])
        self.assertEqual(list(Color.BLUE), [Color.BLUE])
        self.assertEqual(list(Color.GREEN), [Color.GREEN])
        self.assertEqual(list(Color.WHITE), [Color.RED, Color.GREEN, Color.BLUE])

    def test_member_length(self):
        # [FUNCTION DOC] Checks the length of an IntFlag member equals its number of set bits.
        self.assertEqual(self.Color.__len__(self.Color.BLACK), 0)
        self.assertEqual(self.Color.__len__(self.Color.GREEN), 1)
        self.assertEqual(self.Color.__len__(self.Color.PURPLE), 2)
        self.assertEqual(self.Color.__len__(self.Color.BLANCO), 3)

    def test_aliases(self):
        # [FUNCTION DOC] Checks aliases resolve to canonical names by value and by name.
        Color = self.Color
        self.assertEqual(Color(1).name, 'RED')
        self.assertEqual(Color['ROJO'].name, 'RED')
        self.assertEqual(Color(7).name, 'WHITE')
        self.assertEqual(Color['BLANCO'].name, 'WHITE')
        self.assertIs(Color.BLANCO, Color.WHITE)
        Open = self.Open
        self.assertIs(Open['AC'], Open.AC)

    def test_bool(self):
        # [FUNCTION DOC] Checks truthiness of IntFlag members follows their integer value.
        Perm = self.Perm
        for f in Perm:
            self.assertTrue(f)
        Open = self.Open
        for f in Open:
            self.assertEqual(bool(f.value), bool(f))


    def test_multiple_mixin(self):
        # [FUNCTION DOC] Checks IntFlag with mixins in different base orders: ALL property
        # from AllMixin, plain str is the number text, and with StrMixin re-assigned the
        # lowercase name.
        class AllMixin:
            # [CLASS DOC] HELPER: Mixin with a classproperty ALL that ORs all members and
            # caches it.
            @classproperty
            def ALL(cls):
                # [FUNCTION DOC] HELPER: Returns the union of all members (None if empty) and
                # stores it as ALL.
                members = list(cls)
                all_value = None
                if members:
                    all_value = members[0]
                    for member in members[1:]:
                        all_value |= member
                cls.ALL = all_value
                return all_value
        class StrMixin:
            # [CLASS DOC] HELPER: Mixin with __str__ returning the lowercase member name.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Returns the lowercase member name.
                return self._name_.lower()
        class Color(AllMixin, IntFlag):
            # [CLASS DOC] HELPER: IntFlag with AllMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), '4')
        class Color(AllMixin, StrMixin, IntFlag):
            # [CLASS DOC] HELPER: IntFlag with AllMixin and StrMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), 'blue')
        class Color(StrMixin, AllMixin, IntFlag):
            # [CLASS DOC] HELPER: IntFlag with StrMixin and AllMixin.
            RED = auto()
            GREEN = auto()
            BLUE = auto()
            __str__ = StrMixin.__str__
        self.assertEqual(Color.RED.value, 1)
        self.assertEqual(Color.GREEN.value, 2)
        self.assertEqual(Color.BLUE.value, 4)
        self.assertEqual(Color.ALL.value, 7)
        self.assertEqual(str(Color.BLUE), 'blue')

    @threading_helper.reap_threads
    @threading_helper.requires_working_threading()
    def test_unique_composite(self):
        # [FUNCTION DOC] Concurrency check for IntFlag: eight threads each look up values 0 to
        # 255 in an IntFlag with identity equality; no thread may fail and 256 distinct members
        # must be seen. Skipped without working threading.
        # KNOWN ISSUE: an inline comment says 248 members while the assertion expects 256.
        # override __eq__ to be identity only
        class TestFlag(IntFlag):
            # [CLASS DOC] HELPER: IntFlag with eight auto members, __eq__ by identity and
            # __hash__ by value.
            one = auto()
            two = auto()
            three = auto()
            four = auto()
            five = auto()
            six = auto()
            seven = auto()
            eight = auto()
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Identity comparison.
                return self is other
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the member value.
                return hash(self._value_)
        # have multiple threads competing to complete the composite members
        seen = set()
        failed = False
        def cycle_enum():
            # [FUNCTION DOC] HELPER: Thread target; looks up 256 values and adds each to a
            # shared set, setting the failed flag on any exception.
            nonlocal failed
            try:
                for i in range(256):
                    seen.add(TestFlag(i))
            except Exception:
                failed = True
        threads = [
                threading.Thread(target=cycle_enum)
                for _ in range(8)
                ]
        with threading_helper.start_threads(threads):
            pass
        # check that only 248 members were created
        self.assertFalse(
                failed,
                'at least one thread failed while creating composite members')
        self.assertEqual(256, len(seen), 'too many composite members created')


class TestEmptyAndNonLatinStrings(__TestCase):
    # [CLASS DOC] Tests for enum member names that are empty or use non-Latin characters in the
    # functional API.

    def test_empty_string(self):
        # [FUNCTION DOC] Checks that an empty member name in the functional API
        # raises ValueError.
        with self.assertRaises(ValueError):
            empty_abc = Enum('empty_abc', ('', 'B', 'C'))

    def test_non_latin_character_string(self):
        # [FUNCTION DOC] Checks a Greek letter name is accepted as an attribute and gets
        # value 1.
        greek_abc = Enum('greek_abc', ('\u03B1', 'B', 'C'))
        item = getattr(greek_abc, '\u03B1')
        self.assertEqual(item.value, 1)

    def test_non_latin_number_string(self):
        # [FUNCTION DOC] Checks a Hebrew letter name is accepted as an attribute and gets
        # value 1.
        hebrew_123 = Enum('hebrew_123', ('\u05D0', '2', '3'))
        item = getattr(hebrew_123, '\u05D0')
        self.assertEqual(item.value, 1)


class TestUnique(__TestCase):
    # [CLASS DOC] Tests for the unique decorator, which rejects enums with aliases.

    def test_unique_clean(self):
        # [FUNCTION DOC] Checks unique accepts enums with no duplicate values (including a float
        # and a string value).
        @unique
        class Clean(Enum):
            # [CLASS DOC] HELPER: Enum with distinct values one, two, tres.
            one = 1
            two = 'dos'
            tres = 4.0
        #
        @unique
        class Cleaner(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with distinct values 1, 2, 3.
            single = 1
            double = 2
            triple = 3

    def test_unique_dirty(self):
        # [FUNCTION DOC] Checks unique raises ValueError naming aliases and their canonical
        # members for duplicates.
        with self.assertRaisesRegex(ValueError, 'tres.*one'):
            @unique
            class Dirty(Enum):
                # [CLASS DOC] HELPER: Enum where tres duplicates one.
                one = 1
                two = 'dos'
                tres = 1
        with self.assertRaisesRegex(
                ValueError,
                'double.*single.*turkey.*triple',
                ):
            @unique
            class Dirtier(IntEnum):
                # [CLASS DOC] HELPER: IntEnum with two pairs of duplicate values.
                single = 1
                double = 1
                triple = 3
                turkey = 3

    def test_unique_with_name(self):
        # [FUNCTION DOC] Checks the UNIQUE verifier accepts enums whose members are named name
        # and value.
        @verify(UNIQUE)
        class Silly(Enum):
            # [CLASS DOC] HELPER: Enum with a member named name.
            one = 1
            two = 'dos'
            name = 3
        #
        @verify(UNIQUE)
        class Sillier(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members named name and value.
            single = 1
            name = 2
            triple = 3
            value = 4

class TestVerify(__TestCase):
    # [CLASS DOC] Tests for the verify decorator with the CONTINUOUS, UNIQUE and
    # NAMED_FLAGS checks.

    def test_continuous(self):
        # [FUNCTION DOC] Checks CONTINUOUS accepts gapless auto or manual sequences and raises
        # ValueError listing the missing values for enums and the missing flag bits for Flags.
        @verify(CONTINUOUS)
        class Auto(Enum):
            # [CLASS DOC] HELPER: Enum with four auto members.
            FIRST = auto()
            SECOND = auto()
            THIRD = auto()
            FORTH = auto()
        #
        @verify(CONTINUOUS)
        class Manual(Enum):
            # [CLASS DOC] HELPER: Enum with consecutive values 3 to 6.
            FIRST = 3
            SECOND = 4
            THIRD = 5
            FORTH = 6
        #
        with self.assertRaisesRegex(ValueError, 'invalid enum .Missing.: missing values 5, 6, 7, 8, 9, 10, 12'):
            @verify(CONTINUOUS)
            class Missing(Enum):
                # [CLASS DOC] HELPER: Invalid Enum with gaps in values.
                FIRST = 3
                SECOND = 4
                THIRD = 11
                FORTH = 13
        #
        with self.assertRaisesRegex(ValueError, 'invalid flag .Incomplete.: missing values 32'):
            @verify(CONTINUOUS)
            class Incomplete(Flag):
                # [CLASS DOC] HELPER: Invalid Flag skipping bit 32.
                FIRST = 4
                SECOND = 8
                THIRD = 16
                FORTH = 64
        #
        with self.assertRaisesRegex(ValueError, 'invalid flag .StillIncomplete.: missing values 16'):
            @verify(CONTINUOUS)
            class StillIncomplete(Flag):
                # [CLASS DOC] HELPER: Invalid Flag with a composite value 11 leaving bit
                # 16 missing.
                FIRST = 4
                SECOND = 8
                THIRD = 11
                FORTH = 32


    def test_composite(self):
        # [FUNCTION DOC] Checks NAMED_FLAGS: multi-bit members whose bits are not individually
        # named raise ValueError describing the missing aliases, while show_flag_values lists
        # the single bits of a value.
        class Bizarre(Flag):
            # [CLASS DOC] HELPER: Flag with members b=3, c=4, d=6 (unnamed bits).
            b = 3
            c = 4
            d = 6
        self.assertEqual(list(Bizarre), [Bizarre.c])
        self.assertEqual(Bizarre.b.value, 3)
        self.assertEqual(Bizarre.c.value, 4)
        self.assertEqual(Bizarre.d.value, 6)
        with self.assertRaisesRegex(
                ValueError,
                "invalid Flag 'Bizarre': aliases b and d are missing combined values of 0x3 .use enum.show_flag_values.value. for details.",
            ):
            @verify(NAMED_FLAGS)
            class Bizarre(Flag):
                # [CLASS DOC] HELPER: Same Flag verified with NAMED_FLAGS; must raise.
                b = 3
                c = 4
                d = 6
        #
        self.assertEqual(enum.show_flag_values(3), [1, 2])
        class Bizarre(IntFlag):
            # [CLASS DOC] HELPER: IntFlag with members b=3, c=4, d=6.
            b = 3
            c = 4
            d = 6
        self.assertEqual(list(Bizarre), [Bizarre.c])
        self.assertEqual(Bizarre.b.value, 3)
        self.assertEqual(Bizarre.c.value, 4)
        self.assertEqual(Bizarre.d.value, 6)
        with self.assertRaisesRegex(
                ValueError,
                "invalid Flag 'Bizarre': alias d is missing value 0x2 .use enum.show_flag_values.value. for details.",
            ):
            @verify(NAMED_FLAGS)
            class Bizarre(IntFlag):
                # [CLASS DOC] HELPER: IntFlag verified with NAMED_FLAGS where d=6 lacks the bit
                # 2; must raise.
                c = 4
                d = 6
        self.assertEqual(enum.show_flag_values(2), [2])

    def test_unique_clean(self):
        # [FUNCTION DOC] Checks UNIQUE verify accepts enums with distinct values.
        @verify(UNIQUE)
        class Clean(Enum):
            # [CLASS DOC] HELPER: Enum with distinct values.
            one = 1
            two = 'dos'
            tres = 4.0
        #
        @verify(UNIQUE)
        class Cleaner(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with distinct values.
            single = 1
            double = 2
            triple = 3

    def test_unique_dirty(self):
        # [FUNCTION DOC] Checks UNIQUE verify raises ValueError naming duplicated members.
        with self.assertRaisesRegex(ValueError, 'tres.*one'):
            @verify(UNIQUE)
            class Dirty(Enum):
                # [CLASS DOC] HELPER: Enum where tres duplicates one.
                one = 1
                two = 'dos'
                tres = 1
        with self.assertRaisesRegex(
                ValueError,
                'double.*single.*turkey.*triple',
                ):
            @verify(UNIQUE)
            class Dirtier(IntEnum):
                # [CLASS DOC] HELPER: IntEnum with two pairs of duplicates.
                single = 1
                double = 1
                triple = 3
                turkey = 3

    def test_unique_with_name(self):
        # [FUNCTION DOC] Checks UNIQUE verify accepts members named name and value.
        @verify(UNIQUE)
        class Silly(Enum):
            # [CLASS DOC] HELPER: Enum with member name.
            one = 1
            two = 'dos'
            name = 3
        #
        @verify(UNIQUE)
        class Sillier(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members name and value.
            single = 1
            name = 2
            triple = 3
            value = 4

    def test_negative_alias(self):
        # [FUNCTION DOC] Checks NAMED_FLAGS accepts a Flag with a negative alias (-1) as long as
        # all bits are named.
        @verify(NAMED_FLAGS)
        class Color(Flag):
            # [CLASS DOC] HELPER: Flag with RED, GREEN, BLUE and WHITE=-1.
            RED = 1
            GREEN = 2
            BLUE = 4
            WHITE = -1
        # no error means success


class TestInternals(__TestCase):
    # [CLASS DOC] Tests of enum internals: name classifiers and auto() value generation. Class
    # attributes list sample sunder, dunder, private and random names.

    sunder_names = '_bad_', '_good_', '_what_ho_'
    dunder_names = '__mal__', '__bien__', '__que_que__'
    private_names = '_MyEnum__private', '_MyEnum__still_private'
    private_and_sunder_names = '_MyEnum__private_', '_MyEnum__also_private_'
    random_names = 'okay', '_semi_private', '_weird__', '_MyEnum__'

    def test_sunder(self):
        # [FUNCTION DOC] Checks _is_sunder on the sample name sets.
        for name in self.sunder_names + self.private_and_sunder_names:
            self.assertTrue(enum._is_sunder(name), '%r is a not sunder name?' % name)
        for name in self.dunder_names + self.private_names + self.random_names:
            self.assertFalse(enum._is_sunder(name), '%r is a sunder name?' % name)

    def test_dunder(self):
        # [FUNCTION DOC] Checks _is_dunder on the sample name sets.
        for name in self.dunder_names:
            self.assertTrue(enum._is_dunder(name), '%r is a not dunder name?' % name)
        for name in self.sunder_names + self.private_names + self.private_and_sunder_names + self.random_names:
            self.assertFalse(enum._is_dunder(name), '%r is a dunder name?' % name)

    def test_is_private(self):
        # [FUNCTION DOC] Checks _is_private for class MyEnum on the sample name sets.
        # KNOWN ISSUE: the assertion messages contain a %r placeholder that is never formatted.
        for name in self.private_names + self.private_and_sunder_names:
            self.assertTrue(enum._is_private('MyEnum', name), '%r is a not private name?')
        for name in self.sunder_names + self.dunder_names + self.random_names:
            self.assertFalse(enum._is_private('MyEnum', name), '%r is a private name?')

    def test_auto_number(self):
        # [FUNCTION DOC] Checks auto() gives 1, 2, 3 for a plain Enum.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with three auto members.
            red = auto()
            blue = auto()
            green = auto()

        self.assertEqual(list(Color), [Color.red, Color.blue, Color.green])
        self.assertEqual(Color.red.value, 1)
        self.assertEqual(Color.blue.value, 2)
        self.assertEqual(Color.green.value, 3)

    def test_auto_name(self):
        # [FUNCTION DOC] Checks a custom _generate_next_value_ returning the member name drives
        # auto values.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum whose next-value generator returns the member name.
            def _generate_next_value_(name, start, count, last):
                # [FUNCTION DOC] HELPER: Returns the member name as the value.
                return name
            red = auto()
            blue = auto()
            green = auto()

        self.assertEqual(list(Color), [Color.red, Color.blue, Color.green])
        self.assertEqual(Color.red.value, 'red')
        self.assertEqual(Color.blue.value, 'blue')
        self.assertEqual(Color.green.value, 'green')

    def test_auto_name_inherit(self):
        # [FUNCTION DOC] Checks the custom next-value generator is inherited by subclasses.
        class AutoNameEnum(Enum):
            # [CLASS DOC] HELPER: Member-less base with a name-returning generator.
            def _generate_next_value_(name, start, count, last):
                # [FUNCTION DOC] HELPER: Returns the member name as the value.
                return name
        class Color(AutoNameEnum):
            # [CLASS DOC] HELPER: Subclass using auto members.
            red = auto()
            blue = auto()
            green = auto()

        self.assertEqual(list(Color), [Color.red, Color.blue, Color.green])
        self.assertEqual(Color.red.value, 'red')
        self.assertEqual(Color.blue.value, 'blue')
        self.assertEqual(Color.green.value, 'green')

    @unittest.skipIf(
            python_version >= (3, 13),
            'mixed types with auto() no longer supported',
            )
    def test_auto_garbage_ok(self):
        # [FUNCTION DOC] Skipped on 3.13 and later. Checks mixing a string value with auto()
        # emits a DeprecationWarning and uses 1.
        with self.assertWarnsRegex(DeprecationWarning, 'will require all values to be sortable'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Enum mixing a string value and auto().
                red = 'red'
                blue = auto()
        self.assertEqual(Color.blue.value, 1)

    @unittest.skipIf(
            python_version >= (3, 13),
            'mixed types with auto() no longer supported',
            )
    def test_auto_garbage_corrected_ok(self):
        # [FUNCTION DOC] Skipped on 3.13 and later. Checks mixed value types with auto() warn
        # and continue after the number.
        with self.assertWarnsRegex(DeprecationWarning, 'will require all values to be sortable'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Enum with a string, an int and an auto member.
                red = 'red'
                blue = 2
                green = auto()

        self.assertEqual(list(Color), [Color.red, Color.blue, Color.green])
        self.assertEqual(Color.red.value, 'red')
        self.assertEqual(Color.blue.value, 2)
        self.assertEqual(Color.green.value, 3)

    @unittest.skipIf(
            python_version < (3, 13),
            'mixed types with auto() will raise in 3.13',
            )
    def test_auto_garbage_fail(self):
        # [FUNCTION DOC] Skipped before 3.13. Checks auto() after a string value raises
        # TypeError (cannot increment).
        with self.assertRaisesRegex(TypeError, "unable to increment 'red'"):
            class Color(Enum):
                # [CLASS DOC] HELPER: Enum with a string value followed by auto().
                red = 'red'
                blue = auto()

    @unittest.skipIf(
            python_version < (3, 13),
            'mixed types with auto() will raise in 3.13',
            )
    def test_auto_garbage_corrected_fail(self):
        # [FUNCTION DOC] Skipped before 3.13. Checks auto() after mixed string and int values
        # raises TypeError (cannot sort).
        with self.assertRaisesRegex(TypeError, 'unable to sort non-numeric values'):
            class Color(Enum):
                # [CLASS DOC] HELPER: Enum with string, int and auto members.
                red = 'red'
                blue = 2
                green = auto()

    def test_auto_order(self):
        # [FUNCTION DOC] Checks defining _generate_next_value_ after auto members have been used
        # raises TypeError.
        with self.assertRaises(TypeError):
            class Color(Enum):
                # [CLASS DOC] HELPER: Invalid Enum defining the generator after auto members.
                red = auto()
                green = auto()
                blue = auto()
                def _generate_next_value_(name, start, count, last):
                    # [FUNCTION DOC] HELPER: Returns the member name.
                    return name

    def test_auto_order_wierd(self):
        # [FUNCTION DOC] Checks an auto object whose value was set by hand before the generator
        # is defined keeps that value while later auto members use the generator.
        weird_auto = auto()
        weird_auto.value = 'pathological case'
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with a pre-valued auto member, a generator and a later
            # auto member.
            red = weird_auto
            def _generate_next_value_(name, start, count, last):
                # [FUNCTION DOC] HELPER: Returns the member name.
                return name
            blue = auto()
        self.assertEqual(list(Color), [Color.red, Color.blue])
        self.assertEqual(Color.red.value, 'pathological case')
        self.assertEqual(Color.blue.value, 'blue')

    @unittest.skipIf(
            python_version < (3, 13),
            'auto() will return highest value + 1 in 3.13',
            )
    def test_auto_with_aliases(self):
        # [FUNCTION DOC] Skipped before 3.13. Checks aliases of auto members resolve to the
        # original and later auto members get new members.
        class Color(Enum):
            # [CLASS DOC] HELPER: Enum with auto members and aliases oxford and crimson.
            red = auto()
            blue = auto()
            oxford = blue
            crimson = red
            green = auto()
        self.assertIs(Color.crimson, Color.red)
        self.assertIs(Color.oxford, Color.blue)
        self.assertIsNot(Color.green, Color.red)
        self.assertIsNot(Color.green, Color.blue)

    def test_duplicate_auto(self):
        # [FUNCTION DOC] Checks chained assignment of auto creates an alias and later
        # members continue.
        class Dupes(Enum):
            # [CLASS DOC] HELPER: Enum with first and alias primero sharing an auto value.
            first = primero = auto()
            second = auto()
            third = auto()
        self.assertEqual([Dupes.first, Dupes.second, Dupes.third], list(Dupes))

    def test_multiple_auto_on_line(self):
        # [FUNCTION DOC] Checks several auto() calls in one tuple get successive values, with a
        # custom __new__ that derives an abbreviation, and with a count based generator that
        # repeats the same number.
        class Huh(Enum):
            # [CLASS DOC] HELPER: Enum with tuple values built from multiple auto() calls.
            ONE = auto()
            TWO = auto(), auto()
            THREE = auto(), auto(), auto()
        self.assertEqual(Huh.ONE.value, 1)
        self.assertEqual(Huh.TWO.value, (2, 3))
        self.assertEqual(Huh.THREE.value, (4, 5, 6))
        #
        class Hah(Enum):
            # [CLASS DOC] HELPER: Enum with name-based values and an optional abbreviation
            # via __new__.
            def __new__(cls, value, abbr=None):
                # [FUNCTION DOC] HELPER: Creates a member with the value and an abbreviation
                # (lowercased first three letters unless given).
                member = object.__new__(cls)
                member._value_ = value
                member.abbr = abbr or value[:3].lower()
                return member
            def _generate_next_value_(name, start, count, last):
                # [FUNCTION DOC] HELPER: Returns the member name.
                return name
            #
            MONDAY = auto()
            TUESDAY = auto()
            WEDNESDAY = auto(), 'WED'
            THURSDAY = auto(), 'Thu'
            FRIDAY = auto()
        self.assertEqual(Hah.MONDAY.value, 'MONDAY')
        self.assertEqual(Hah.MONDAY.abbr, 'mon')
        self.assertEqual(Hah.TUESDAY.value, 'TUESDAY')
        self.assertEqual(Hah.TUESDAY.abbr, 'tue')
        self.assertEqual(Hah.WEDNESDAY.value, 'WEDNESDAY')
        self.assertEqual(Hah.WEDNESDAY.abbr, 'WED')
        self.assertEqual(Hah.THURSDAY.value, 'THURSDAY')
        self.assertEqual(Hah.THURSDAY.abbr, 'Thu')
        self.assertEqual(Hah.FRIDAY.value, 'FRIDAY')
        self.assertEqual(Hah.FRIDAY.abbr, 'fri')
        #
        class Huh(Enum):
            # [CLASS DOC] HELPER: Enum whose generator returns count plus one.
            def _generate_next_value_(name, start, count, last):
                # [FUNCTION DOC] HELPER: Returns the count plus one.
                return count+1
            ONE = auto()
            TWO = auto(), auto()
            THREE = auto(), auto(), auto()
        self.assertEqual(Huh.ONE.value, 1)
        self.assertEqual(Huh.TWO.value, (2, 2))
        self.assertEqual(Huh.THREE.value, (3, 3, 3))


expected_help_output_with_docs = """\
Help on class Color in module %s:

class Color(enum.Enum)
 |  Color(*values)
 |
 |  Method resolution order:
 |      Color
 |      enum.Enum
 |      builtins.object
 |
 |  Data and other attributes defined here:
 |
 |  CYAN = <Color.CYAN: 1>
 |
 |  MAGENTA = <Color.MAGENTA: 2>
 |
 |  YELLOW = <Color.YELLOW: 3>
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from enum.Enum:
 |
 |  name
 |      The name of the Enum member.
 |
 |  value
 |      The value of the Enum member.
 |
 |  ----------------------------------------------------------------------
 |  Static methods inherited from enum.EnumType:
 |
 |  __contains__(value)
 |      Return True if `value` is in `cls`.
 |
 |      `value` is in `cls` if:
 |      1) `value` is a member of `cls`, or
 |      2) `value` is the value of one of the `cls`'s members.
 |      3) `value` is a pseudo-member (flags)
 |
 |  __getitem__(name)
 |      Return the member matching `name`.
 |
 |  __iter__()
 |      Return members in definition order.
 |
 |  __len__()
 |      Return the number of members (no aliases)
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from enum.EnumType:
 |
 |  __members__
 |      Returns a mapping of member name->value.
 |
 |      This mapping lists all enum members, including aliases. Note that this
 |      is a read-only view of the internal mapping."""

expected_help_output_without_docs = """\
Help on class Color in module %s:

class Color(enum.Enum)
 |  Color(*values)
 |
 |  Method resolution order:
 |      Color
 |      enum.Enum
 |      builtins.object
 |
 |  Data and other attributes defined here:
 |
 |  CYAN = <Color.CYAN: 1>
 |
 |  MAGENTA = <Color.MAGENTA: 2>
 |
 |  YELLOW = <Color.YELLOW: 3>
 |
 |  ----------------------------------------------------------------------
 |  Data descriptors inherited from enum.Enum:
 |
 |  name
 |
 |  value
 |
 |  ----------------------------------------------------------------------
 |  Static methods inherited from enum.EnumType:
 |
 |  __contains__(value)
 |
 |  __getitem__(name)
 |
 |  __iter__()
 |
 |  __len__()
 |
 |  ----------------------------------------------------------------------
 |  Readonly properties inherited from enum.EnumType:
 |
 |  __members__"""

class TestStdLib(__TestCase):
    # [CLASS DOC] Tests compatibility of enums with stdlib introspection tools pydoc
    # and inspect, and with _simple_enum. Class attribute Color is a three-member plain
    # Enum fixture.

    maxDiff = None

    class Color(Enum):
        # [CLASS DOC] HELPER: Enum CYAN=1, MAGENTA=2, YELLOW=3 used for introspection tests.
        CYAN = 1
        MAGENTA = 2
        YELLOW = 3

    def test_pydoc(self):
        # [FUNCTION DOC] Renders pydoc help for Color into a string buffer and compares it
        # with the expected text (with or without docstrings depending on whether StrEnum has
        # a docstring).
        # indirectly test __objclass__
        if StrEnum.__doc__ is None:
            expected_text = expected_help_output_without_docs % __name__
        else:
            expected_text = expected_help_output_with_docs % __name__
        output = StringIO()
        helper = pydoc.Helper(output=output)
        helper(self.Color)
        result = output.getvalue().strip()
        self.assertEqual(result, expected_text, result)

    def test_inspect_getmembers(self):
        # [FUNCTION DOC] Checks the set of names from inspect.getmembers on Color and the values
        # of all except __doc__, printing differences and failing if any differ.
        values = dict((
                ('__class__', EnumType),
                ('__doc__', '...'),
                ('__members__', self.Color.__members__),
                ('__module__', __name__),
                ('YELLOW', self.Color.YELLOW),
                ('MAGENTA', self.Color.MAGENTA),
                ('CYAN', self.Color.CYAN),
                ('name', Enum.__dict__['name']),
                ('value', Enum.__dict__['value']),
                ('__len__', self.Color.__len__),
                ('__contains__', self.Color.__contains__),
                ('__name__', 'Color'),
                ('__getitem__', self.Color.__getitem__),
                ('__qualname__', 'TestStdLib.Color'),
                ('__init_subclass__', getattr(self.Color, '__init_subclass__')),
                ('__iter__', self.Color.__iter__),
                ))
        result = dict(inspect.getmembers(self.Color))
        self.assertEqual(set(values.keys()), set(result.keys()))
        failed = False
        for k in values.keys():
            if k == '__doc__':
                # __doc__ is huge, not comparing
                continue
            if result[k] != values[k]:
                print()
                print('\n%s\n     key: %s\n  result: %s\nexpected: %s\n%s\n' %
                        ('=' * 75, k, result[k], values[k], '=' * 75), sep='')
                failed = True
        if failed:
            self.fail("result does not equal expected, see print above")

    def test_inspect_classify_class_attrs(self):
        # [FUNCTION DOC] Checks inspect.classify_class_attrs on Color against an expected list
        # of Attribute records (kind, defining class, object), comparing only metadata for
        # __init_subclass__ and __doc__.
        # indirectly test __objclass__
        from inspect import Attribute
        values = [
                Attribute(name='__class__', kind='data',
                    defining_class=object, object=EnumType),
                Attribute(name='__contains__', kind='method',
                    defining_class=EnumType, object=self.Color.__contains__),
                Attribute(name='__doc__', kind='data',
                    defining_class=self.Color, object='...'),
                Attribute(name='__getitem__', kind='method',
                    defining_class=EnumType, object=self.Color.__getitem__),
                Attribute(name='__iter__', kind='method',
                    defining_class=EnumType, object=self.Color.__iter__),
                Attribute(name='__init_subclass__', kind='class method',
                    defining_class=object, object=getattr(self.Color, '__init_subclass__')),
                Attribute(name='__len__', kind='method',
                    defining_class=EnumType, object=self.Color.__len__),
                Attribute(name='__members__', kind='property',
                    defining_class=EnumType, object=EnumType.__members__),
                Attribute(name='__module__', kind='data',
                    defining_class=self.Color, object=__name__),
                Attribute(name='__name__', kind='data',
                    defining_class=self.Color, object='Color'),
                Attribute(name='__qualname__', kind='data',
                    defining_class=self.Color, object='TestStdLib.Color'),
                Attribute(name='YELLOW', kind='data',
                    defining_class=self.Color, object=self.Color.YELLOW),
                Attribute(name='MAGENTA', kind='data',
                    defining_class=self.Color, object=self.Color.MAGENTA),
                Attribute(name='CYAN', kind='data',
                    defining_class=self.Color, object=self.Color.CYAN),
                Attribute(name='name', kind='data',
                    defining_class=Enum, object=Enum.__dict__['name']),
                Attribute(name='value', kind='data',
                    defining_class=Enum, object=Enum.__dict__['value']),
                ]
        for v in values:
            try:
                v.name
            except AttributeError:
                print(v)
        values.sort(key=lambda item: item.name)
        result = list(inspect.classify_class_attrs(self.Color))
        result.sort(key=lambda item: item.name)
        self.assertEqual(
                len(values), len(result),
                "%s != %s" % ([a.name for a in values], [a.name for a in result])
                )
        failed = False
        for v, r in zip(values, result):
            if r.name in ('__init_subclass__', '__doc__'):
                # not sure how to make the __init_subclass_ Attributes match
                # so as long as there is one, call it good
                # __doc__ is too big to check exactly, so treat the same as __init_subclass__
                for name in ('name','kind','defining_class'):
                    if getattr(v, name) != getattr(r, name):
                        print('\n%s\n%s\n%s\n%s\n' % ('=' * 75, r, v, '=' * 75), sep='')
                        failed = True
            elif r != v:
                print('\n%s\n%s\n%s\n%s\n' % ('=' * 75, r, v, '=' * 75), sep='')
                failed = True
        if failed:
            self.fail("result does not equal expected, see print above")

    def test_inspect_signatures(self):
        # [FUNCTION DOC] Checks the inspect signatures of Enum (positional-only class name,
        # names, keyword-only options) and of FlagBoundary (variadic values).
        from inspect import signature, Signature, Parameter
        self.assertEqual(
                signature(Enum),
                Signature([
                    Parameter('new_class_name', Parameter.POSITIONAL_ONLY),
                    Parameter('names', Parameter.POSITIONAL_OR_KEYWORD),
                    Parameter('module', Parameter.KEYWORD_ONLY, default=None),
                    Parameter('qualname', Parameter.KEYWORD_ONLY, default=None),
                    Parameter('type', Parameter.KEYWORD_ONLY, default=None),
                    Parameter('start', Parameter.KEYWORD_ONLY, default=1),
                    Parameter('boundary', Parameter.KEYWORD_ONLY, default=None),
                    ]),
                )
        self.assertEqual(
                signature(enum.FlagBoundary),
                Signature([
                    Parameter('values', Parameter.VAR_POSITIONAL),
                    ]),
                )

    def test_test_simple_enum(self):
        # [FUNCTION DOC] Checks the _test_simple_enum helper compares a normally defined enum
        # against one made with the _simple_enum decorator: matching enums pass, a changed value
        # raises TypeError (enum mismatch); covers flags with KEEP boundary, unhashable values,
        # and members built through custom __new__.
        @_simple_enum(Enum)
        class SimpleColor:
            # [CLASS DOC] HELPER: Plain class converted to Enum by the _simple_enum decorator.
            CYAN = 1
            MAGENTA = 2
            YELLOW = 3
            @bltns.property
            def zeroth(self):
                # [FUNCTION DOC] HELPER: Property returning the text zeroed with the
                # member name.
                return 'zeroed %s' % self.name
        class CheckedColor(Enum):
            # [CLASS DOC] HELPER: Normal Enum equal to SimpleColor, used as the reference.
            CYAN = 1
            MAGENTA = 2
            YELLOW = 3
            @bltns.property
            def zeroth(self):
                # [FUNCTION DOC] HELPER: Property returning the text zeroed with the
                # member name.
                return 'zeroed %s' % self.name
        _test_simple_enum(CheckedColor, SimpleColor)
        SimpleColor.MAGENTA._value_ = 9
        self.assertRaisesRegex(
                TypeError, "enum mismatch",
                _test_simple_enum, CheckedColor, SimpleColor,
                )
        #
        #
        class CheckedMissing(IntFlag, boundary=KEEP):
            # [CLASS DOC] HELPER: IntFlag with KEEP boundary and a mask member ALL of
            # mixed bits.
            SIXTY_FOUR = 64
            ONE_TWENTY_EIGHT = 128
            TWENTY_FORTY_EIGHT = 2048
            ALL = 2048 + 128 + 64 + 12
        CM = CheckedMissing
        self.assertEqual(list(CheckedMissing), [CM.SIXTY_FOUR, CM.ONE_TWENTY_EIGHT, CM.TWENTY_FORTY_EIGHT])
        #
        @_simple_enum(IntFlag, boundary=KEEP)
        class Missing:
            # [CLASS DOC] HELPER: Plain class converted to IntFlag with KEEP boundary
            # by _simple_enum.
            SIXTY_FOUR = 64
            ONE_TWENTY_EIGHT = 128
            TWENTY_FORTY_EIGHT = 2048
            ALL = 2048 + 128 + 64 + 12
        M = Missing
        self.assertEqual(list(CheckedMissing), [M.SIXTY_FOUR, M.ONE_TWENTY_EIGHT, M.TWENTY_FORTY_EIGHT])
        _test_simple_enum(CheckedMissing, Missing)
        #
        #
        class CheckedUnhashable(Enum):
            # [CLASS DOC] HELPER: Enum with unhashable values (dict, set) and a member
            # named name.
            ONE = dict()
            TWO = set()
            name = 'python'
        self.assertIn(dict(), CheckedUnhashable)
        self.assertIn('python', CheckedUnhashable)
        self.assertEqual(CheckedUnhashable.name.value, 'python')
        self.assertEqual(CheckedUnhashable.name.name, 'name')
        #
        @_simple_enum()
        class Unhashable:
            # [CLASS DOC] HELPER: Plain class converted by _simple_enum, same unhashable values.
            ONE = dict()
            TWO = set()
            name = 'python'
        self.assertIn(dict(), Unhashable)
        self.assertIn('python', Unhashable)
        self.assertEqual(Unhashable.name.value, 'python')
        self.assertEqual(Unhashable.name.name, 'name')
        _test_simple_enum(CheckedUnhashable, Unhashable)
        ##
        class CheckedComplexStatus(IntEnum):
            # [CLASS DOC] HELPER: IntEnum with HTTP-like codes, phrases and a custom __new__;
            # has a duplicate value.
            def __new__(cls, value, phrase, description=''):
                # [FUNCTION DOC] HELPER: Creates an int member with value, phrase and
                # optional description.
                obj = int.__new__(cls, value)
                obj._value_ = value
                obj.phrase = phrase
                obj.description = description
                return obj
            CONTINUE = 100, 'Continue', 'Request received, please continue'
            PROCESSING = 102, 'Processing'
            EARLY_HINTS = 103, 'Early Hints'
            SOME_HINTS = 103, 'Some Early Hints'
        #
        @_simple_enum(IntEnum)
        class ComplexStatus:
            # [CLASS DOC] HELPER: Plain class converted to IntEnum by _simple_enum, same shape
            # as the checked class.
            def __new__(cls, value, phrase, description=''):
                # [FUNCTION DOC] HELPER: Creates an int member with value, phrase and
                # optional description.
                obj = int.__new__(cls, value)
                obj._value_ = value
                obj.phrase = phrase
                obj.description = description
                return obj
            CONTINUE = 100, 'Continue', 'Request received, please continue'
            PROCESSING = 102, 'Processing'
            EARLY_HINTS = 103, 'Early Hints'
            SOME_HINTS = 103, 'Some Early Hints'
        _test_simple_enum(CheckedComplexStatus, ComplexStatus)
        #
        #
        class CheckedComplexFlag(IntFlag):
            # [CLASS DOC] HELPER: IntFlag with (value, label) tuple members and a custom
            # __new__; SHIRT and VEST share a value.
            def __new__(cls, value, label):
                # [FUNCTION DOC] HELPER: Creates an int member with the value and a
                # label attribute.
                obj = int.__new__(cls, value)
                obj._value_ = value
                obj.label = label
                return obj
            SHIRT = 1, 'upper half'
            VEST = 1, 'outer upper half'
            PANTS = 2, 'lower half'
        self.assertIs(CheckedComplexFlag.SHIRT, CheckedComplexFlag.VEST)
        #
        @_simple_enum(IntFlag)
        class ComplexFlag:
            # [CLASS DOC] HELPER: Plain class converted to IntFlag by _simple_enum with the
            # same shape.
            def __new__(cls, value, label):
                # [FUNCTION DOC] HELPER: Creates an int member with the value and a
                # label attribute.
                obj = int.__new__(cls, value)
                obj._value_ = value
                obj.label = label
                return obj
            SHIRT = 1, 'upper half'
            VEST = 1, 'uppert half'
            PANTS = 2, 'lower half'
        _test_simple_enum(CheckedComplexFlag, ComplexFlag)


class MiscTestCase(__TestCase):
    # [CLASS DOC] Miscellaneous enum module tests: the exported names and the default __doc__
    # of enums.

    def test__all__(self):
        # [FUNCTION DOC] Checks the names in enum.__all__ against the module attributes,
        # excluding bin and show_flag_values.
        support.check__all__(self, enum, not_exported={'bin', 'show_flag_values'})

    def test_doc_1(self):
        # [FUNCTION DOC] Checks an enum with one member has no docstring (None).
        class Single(Enum):
            # [CLASS DOC] HELPER: Enum with one member.
            ONE = 1
        self.assertEqual(Single.__doc__, None)

    def test_doc_2(self):
        # [FUNCTION DOC] Checks an enum with two members has no docstring.
        class Double(Enum):
            # [CLASS DOC] HELPER: Enum with two members.
            ONE = 1
            TWO = 2
        self.assertEqual(Double.__doc__, None)

    def test_doc_3(self):
        # [FUNCTION DOC] Checks an enum with three members has no docstring.
        class Triple(Enum):
            # [CLASS DOC] HELPER: Enum with three members.
            ONE = 1
            TWO = 2
            THREE = 3
        self.assertEqual(Triple.__doc__, None)

    def test_doc_4(self):
        # [FUNCTION DOC] Checks an enum with four members has no docstring.
        class Quadruple(Enum):
            # [CLASS DOC] HELPER: Enum with four members.
            ONE = 1
            TWO = 2
            THREE = 3
            FOUR = 4
        self.assertEqual(Quadruple.__doc__, None)


# These are unordered here on purpose to ensure that declaration order
# makes no difference.
CONVERT_TEST_NAME_D = 5
CONVERT_TEST_NAME_C = 5
CONVERT_TEST_NAME_B = 5
CONVERT_TEST_NAME_A = 5  # This one should sort first.
CONVERT_TEST_NAME_E = 5
CONVERT_TEST_NAME_F = 5

CONVERT_STRING_TEST_NAME_D = 5
CONVERT_STRING_TEST_NAME_C = 5
CONVERT_STRING_TEST_NAME_B = 5
CONVERT_STRING_TEST_NAME_A = 5  # This one should sort first.
CONVERT_STRING_TEST_NAME_E = 5
CONVERT_STRING_TEST_NAME_F = 5

# global names for StrEnum._convert_ test
CONVERT_STR_TEST_2 = 'goodbye'
CONVERT_STR_TEST_1 = 'hello'

# We also need values that cannot be compared:
UNCOMPARABLE_A = 5
UNCOMPARABLE_C = (9, 1)  # naming order is broken on purpose
UNCOMPARABLE_B = 'value'

COMPLEX_C = 1j
COMPLEX_A = 2j
COMPLEX_B = 3j

class TestConvert(__TestCase):
    # [CLASS DOC] Tests for the private _convert_ classmethod that builds an enum from module
    # level constants whose names match a filter. Uses module globals CONVERT_*, UNCOMPARABLE_*
    # and COMPLEX_*.
    def tearDown(self):
        # [FUNCTION DOC] Restores the module level constants that the tests may overwrite, so
        # later tests see the original values.
        # KNOWN ISSUE: UNCOMPARABLE_B and UNCOMPARABLE_C are reset to swapped values compared
        #   with their module definition (B gets the tuple, C gets the string).
        # Reset the module-level test variables to their original integer
        # values, otherwise the already created enum values get converted
        # instead.
        g = globals()
        for suffix in ['A', 'B', 'C', 'D', 'E', 'F']:
            g['CONVERT_TEST_NAME_%s' % suffix] = 5
            g['CONVERT_STRING_TEST_NAME_%s' % suffix] = 5
        for suffix, value in (('A', 5), ('B', (9, 1)), ('C', 'value')):
            g['UNCOMPARABLE_%s' % suffix] = value
        for suffix, value in (('A', 2j), ('B', 3j), ('C', 1j)):
            g['COMPLEX_%s' % suffix] = value
        for suffix, value in (('1', 'hello'), ('2', 'goodbye')):
            g['CONVERT_STR_TEST_%s' % suffix] = value

    def test_convert_value_lookup_priority(self):
        # [FUNCTION DOC] Checks that when several names share a value, the reverse lookup by
        # value returns the first name in sorted order.
        test_type = enum.IntEnum._convert_(
                'UnittestConvert',
                MODULE,
                filter=lambda x: x.startswith('CONVERT_TEST_'))
        # We don't want the reverse lookup value to vary when there are
        # multiple possible names for a given value.  It should always
        # report the first lexicographical name in that case.
        self.assertEqual(test_type(5).name, 'CONVERT_TEST_NAME_A')

    def test_convert_int(self):
        # [FUNCTION DOC] Checks IntEnum._convert_ picks up only filtered names and values, with
        # all aliases equal to 5, and dir contains no extra or missing names.
        test_type = enum.IntEnum._convert_(
                'UnittestConvert',
                MODULE,
                filter=lambda x: x.startswith('CONVERT_TEST_'))
        # Ensure that test_type has all of the desired names and values.
        self.assertEqual(test_type.CONVERT_TEST_NAME_F,
                         test_type.CONVERT_TEST_NAME_A)
        self.assertEqual(test_type.CONVERT_TEST_NAME_B, 5)
        self.assertEqual(test_type.CONVERT_TEST_NAME_C, 5)
        self.assertEqual(test_type.CONVERT_TEST_NAME_D, 5)
        self.assertEqual(test_type.CONVERT_TEST_NAME_E, 5)
        # Ensure that test_type only picked up names matching the filter.
        extra = [name for name in dir(test_type) if name not in enum_dir(test_type)]
        missing = [name for name in enum_dir(test_type) if name not in dir(test_type)]
        self.assertEqual(
                extra + missing,
                [],
                msg='extra names: %r;  missing names: %r' % (extra, missing),
                )


    def test_convert_uncomparable(self):
        # [FUNCTION DOC] Checks _convert_ on values that cannot be compared (int, tuple, string)
        # orders members by name.
        uncomp = enum.Enum._convert_(
                'Uncomparable',
                MODULE,
                filter=lambda x: x.startswith('UNCOMPARABLE_'))
        # Should be ordered by `name` only:
        self.assertEqual(
            list(uncomp),
            [uncomp.UNCOMPARABLE_A, uncomp.UNCOMPARABLE_B, uncomp.UNCOMPARABLE_C],
            )

    def test_convert_complex(self):
        # [FUNCTION DOC] Checks _convert_ on complex numbers orders members by name.
        uncomp = enum.Enum._convert_(
            'Uncomparable',
            MODULE,
            filter=lambda x: x.startswith('COMPLEX_'))
        # Should be ordered by `name` only:
        self.assertEqual(
            list(uncomp),
            [uncomp.COMPLEX_A, uncomp.COMPLEX_B, uncomp.COMPLEX_C],
            )

    def test_convert_str(self):
        # [FUNCTION DOC] Checks StrEnum._convert_ with as_global true: members hold the strings,
        # dir is consistent, repr uses the module short name, str and format give the string.
        test_type = enum.StrEnum._convert_(
                'UnittestConvert',
                MODULE,
                filter=lambda x: x.startswith('CONVERT_STR_'),
                as_global=True)
        # Ensure that test_type has all of the desired names and values.
        self.assertEqual(test_type.CONVERT_STR_TEST_1, 'hello')
        self.assertEqual(test_type.CONVERT_STR_TEST_2, 'goodbye')
        # Ensure that test_type only picked up names matching the filter.
        extra = [name for name in dir(test_type) if name not in enum_dir(test_type)]
        missing = [name for name in enum_dir(test_type) if name not in dir(test_type)]
        self.assertEqual(
                extra + missing,
                [],
                msg='extra names: %r;  missing names: %r' % (extra, missing),
                )
        self.assertEqual(repr(test_type.CONVERT_STR_TEST_1), '%s.CONVERT_STR_TEST_1' % SHORT_MODULE)
        self.assertEqual(str(test_type.CONVERT_STR_TEST_2), 'goodbye')
        self.assertEqual(format(test_type.CONVERT_STR_TEST_1), 'hello')

    def test_convert_raise(self):
        # [FUNCTION DOC] Checks calling the misspelled _convert (no trailing underscore)
        # raises AttributeError.
        with self.assertRaises(AttributeError):
            enum.IntEnum._convert(
                'UnittestConvert',
                MODULE,
                filter=lambda x: x.startswith('CONVERT_TEST_'))

    def test_convert_repr_and_str(self):
        # [FUNCTION DOC] Checks a converted IntEnum registered as global shows module-qualified
        # repr, but str and format give the number.
        test_type = enum.IntEnum._convert_(
                'UnittestConvert',
                MODULE,
                filter=lambda x: x.startswith('CONVERT_STRING_TEST_'),
                as_global=True)
        self.assertEqual(repr(test_type.CONVERT_STRING_TEST_NAME_A), '%s.CONVERT_STRING_TEST_NAME_A' % SHORT_MODULE)
        self.assertEqual(str(test_type.CONVERT_STRING_TEST_NAME_A), '5')
        self.assertEqual(format(test_type.CONVERT_STRING_TEST_NAME_A), '5')


class TestEnumDict(__TestCase):
    # [CLASS DOC] Tests for EnumDict used as a class namespace and as a standalone mapping.
    def test_enum_dict_in_metaclass(self):
        # [FUNCTION DOC] Checks EnumDict works as the namespace returned by a metaclass
        # __prepare__: a duplicate name raises TypeError and a sunder name raises ValueError
        # inside the class body.
        """Test that EnumDict is usable as a class namespace"""
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose __prepare__ returns an EnumDict.
            @classmethod
            def __prepare__(metacls, cls, bases, **kwds):
                # [FUNCTION DOC] HELPER: Classmethod returning an EnumDict named by the class.
                return EnumDict(cls)

        class MyClass(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using Meta; body assigns a, then expects TypeError for a
            # duplicate a and ValueError for a sunder name.
            a = 1

            with self.assertRaises(TypeError):
                a = 2  # duplicate

            with self.assertRaises(ValueError):
                _a_sunder_ = 3

    def test_enum_dict_standalone(self):
        # [FUNCTION DOC] Checks a standalone EnumDict: setting an existing name to a different
        # value raises TypeError, but the in-place or operator (MutableMapping update) replaces
        # the value.
        """Test that EnumDict is usable on its own"""
        enumdict = EnumDict()
        enumdict['a'] = 1

        with self.assertRaises(TypeError):
            enumdict['a'] = 'other value'

        # Only MutableMapping interface is overridden for now.
        # If this stops passing, update the documentation.
        enumdict |= {'a': 'other value'}
        self.assertEqual(enumdict['a'], 'other value')


# helpers

def enum_dir(cls):
    # [FUNCTION DOC] HELPER: Returns the sorted list of names expected from dir() of an enum
    # class: fixed class-level dunders plus member names, with __new__ and __init_subclass__ if
    # customized, and the names of the mixed-in data type if any.
    interesting = set([
            '__class__', '__contains__', '__doc__', '__getitem__',
            '__iter__', '__len__', '__members__', '__module__',
            '__name__', '__qualname__',
            ]
            + cls._member_names_
            )
    if cls._new_member_ is not object.__new__:
        interesting.add('__new__')
    if cls.__init_subclass__ is not object.__init_subclass__:
        interesting.add('__init_subclass__')
    if cls._member_type_ is object:
        return sorted(interesting)
    else:
        # return whatever mixed-in data type has
        return sorted(set(dir(cls._member_type_)) | interesting)

def member_dir(member):
    # [FUNCTION DOC] HELPER: Returns the sorted list of names expected from dir() of a member: a
    # fixed set for plain enums or dir of the member for mixed types, then adjusted with public
    # names found along the class hierarchy (properties with getters added, member-named
    # properties removed, other non-member names added).
    if member.__class__._member_type_ is object:
        allowed = set(['__class__', '__doc__', '__eq__', '__hash__', '__module__', 'name', 'value'])
    else:
        allowed = set(dir(member))
    for cls in member.__class__.mro():
        for name, obj in cls.__dict__.items():
            if name[0] == '_':
                continue
            if isinstance(obj, enum.property):
                if obj.fget is not None or name not in member._member_map_:
                    allowed.add(name)
                else:
                    allowed.discard(name)
            elif name not in member._member_map_:
                allowed.add(name)
    return sorted(allowed)


if __name__ == '__main__':
    run_tests()
