# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-adapted copy of the CPython test_collections test suite. Tests
# UserDict/UserList/UserString, ChainMap, namedtuple, the collections.abc abstract base classes
# and Counter. Has no import-time side effects beyond imports; test classes derive from the
# Dynamo CPython test case. Run via the PyTorch test runner.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_collections.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, slowTest

__TestCase = CPythonTestCase

# ======= END DYNAMO PATCH =======

"""Unit tests for collections.py."""

import array
import collections
import copy
import doctest
import inspect
import operator
import pickle
from random import choice, randrange
from itertools import product, chain, combinations
import string
import sys
from test import support
import types
import unittest

from collections import namedtuple, Counter, OrderedDict, _count_elements
from collections import UserDict, UserString, UserList
from collections import ChainMap
from collections import deque
from collections.abc import Awaitable, Coroutine
from collections.abc import AsyncIterator, AsyncIterable, AsyncGenerator
from collections.abc import Hashable, Iterable, Iterator, Generator, Reversible
from collections.abc import Sized, Container, Callable, Collection
from collections.abc import Set, MutableSet
from collections.abc import Mapping, MutableMapping, KeysView, ItemsView, ValuesView
from collections.abc import Sequence, MutableSequence
from collections.abc import ByteString, Buffer


class TestUserObjects(__TestCase):
    # [CLASS DOC] Tests that UserString, UserList and UserDict expose every public
    # attribute of the builtin they wrap and that their copy support duplicates data while
    # sharing instance attributes.
    def _superset_test(self, a, b):
        # [FUNCTION DOC] HELPER: Asserts the attribute names of a are a superset of those of b.
        self.assertGreaterEqual(
            set(dir(a)),
            set(dir(b)),
            '{a} should have all the methods of {b}'.format(
                a=a.__name__,
                b=b.__name__,
            ),
        )

    def _copy_test(self, obj):
        # [FUNCTION DOC] HELPER: Checks the copy method and copy.copy both produce a new
        # data container equal to the original, while an extra instance attribute is
        # shared by reference.
        # Test internal copy
        obj_copy = obj.copy()
        self.assertIsNot(obj.data, obj_copy.data)
        self.assertEqual(obj.data, obj_copy.data)

        # Test copy.copy
        obj.test = [1234]  # Make sure instance vars are also copied.
        obj_copy = copy.copy(obj)
        self.assertIsNot(obj.data, obj_copy.data)
        self.assertEqual(obj.data, obj_copy.data)
        self.assertIs(obj.test, obj_copy.test)

    def test_str_protocol(self):
        # [FUNCTION DOC] Checks UserString has all attributes of str.
        self._superset_test(UserString, str)

    def test_list_protocol(self):
        # [FUNCTION DOC] Checks UserList has all attributes of list.
        self._superset_test(UserList, list)

    def test_dict_protocol(self):
        # [FUNCTION DOC] Checks UserDict has all attributes of dict.
        self._superset_test(UserDict, dict)

    def test_list_copy(self):
        # [FUNCTION DOC] Checks copying a UserList object via the copy helper.
        obj = UserList()
        obj.append(123)
        self._copy_test(obj)

    def test_dict_copy(self):
        # [FUNCTION DOC] Checks copying a UserDict object via the copy helper.
        obj = UserDict()
        obj[123] = "abc"
        self._copy_test(obj)

    def test_dict_missing(self):
        # [FUNCTION DOC] Checks UserDict subscripting uses the missing-key hook of a subclass,
        # while get ignores it and returns None.
        class A(UserDict):
            # [CLASS DOC] HELPER: UserDict subclass whose missing-key hook returns a constant.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Returns 456 for any missing key.
                return 456
        self.assertEqual(A()[123], 456)
        # get() ignores __missing__ on dict
        self.assertIs(A().get(123), None)


################################################################################
### ChainMap (helper class for configparser and the string module)
################################################################################

class TestChainMap(__TestCase):
    # [CLASS DOC] Tests ChainMap: lookup order across maps, mutation of only the first
    # map, construction, truthiness, missing hook, iteration order, child maps, copying
    # and union operators.

    def test_basics(self):
        # [FUNCTION DOC] Checks basic ChainMap behavior: lookup, shadowing, keys/items/values
        # views, get, set, delete, pop, popitem, clear affecting only the first mapping,
        # parents, repr, copy, fromkeys and pickling.
        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        d = c.new_child()
        d['b'] = 20
        d['c'] = 30
        self.assertEqual(d.maps, [{'b':20, 'c':30}, {'a':1, 'b':2}])  # check internal state
        self.assertEqual(d.items(), dict(a=1, b=20, c=30).items())    # check items/iter/getitem
        self.assertEqual(len(d), 3)                                   # check len
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, b=20, c=30, z=100).items():             # check get
            self.assertEqual(d.get(k, 100), v)

        del d['b']                                                    # unmask a value
        self.assertEqual(d.maps, [{'c':30}, {'a':1, 'b':2}])          # check internal state
        self.assertEqual(d.items(), dict(a=1, b=2, c=30).items())     # check items/iter/getitem
        self.assertEqual(len(d), 3)                                   # check len
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, b=2, c=30, z=100).items():              # check get
            self.assertEqual(d.get(k, 100), v)
        self.assertIn(repr(d), [                                      # check repr
            type(d).__name__ + "({'c': 30}, {'a': 1, 'b': 2})",
            type(d).__name__ + "({'c': 30}, {'b': 2, 'a': 1})"
        ])

        for e in d.copy(), copy.copy(d):                               # check shallow copies
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            self.assertIsNot(d.maps[0], e.maps[0])
            for m1, m2 in zip(d.maps[1:], e.maps[1:]):
                self.assertIs(m1, m2)

        # check deep copies
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            e = pickle.loads(pickle.dumps(d, proto))
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            for m1, m2 in zip(d.maps, e.maps):
                self.assertIsNot(m1, m2, e)
        for e in [copy.deepcopy(d),
                  eval(repr(d))
                ]:
            self.assertEqual(d, e)
            self.assertEqual(d.maps, e.maps)
            self.assertIsNot(d, e)
            for m1, m2 in zip(d.maps, e.maps):
                self.assertIsNot(m1, m2, e)

        f = d.new_child()
        f['b'] = 5
        self.assertEqual(f.maps, [{'b': 5}, {'c':30}, {'a':1, 'b':2}])
        self.assertEqual(f.parents.maps, [{'c':30}, {'a':1, 'b':2}])   # check parents
        self.assertEqual(f['b'], 5)                                    # find first in chain
        self.assertEqual(f.parents['b'], 2)                            # look beyond maps[0]

    def test_ordering(self):
        # [FUNCTION DOC] Checks the ordering of keys, values and items follows reverse mapping
        # order with first-seen ordering.
        # Combined order matches a series of dict updates from last to first.
        # This test relies on the ordering of the underlying dicts.

        baseline = {'music': 'bach', 'art': 'rembrandt'}
        adjustments = {'art': 'van gogh', 'opera': 'carmen'}

        cm = ChainMap(adjustments, baseline)

        combined = baseline.copy()
        combined.update(adjustments)

        self.assertEqual(list(combined.items()), list(cm.items()))

    def test_constructor(self):
        # [FUNCTION DOC] Checks the constructor with no args creates one empty map.
        self.assertEqual(ChainMap().maps, [{}])                        # no-args --> one new dict
        self.assertEqual(ChainMap({1:2}).maps, [{1:2}])                # 1 arg --> list

    def test_bool(self):
        # [FUNCTION DOC] Checks truthiness is true only if some underlying map is non-empty.
        self.assertFalse(ChainMap())
        self.assertFalse(ChainMap({}, {}))
        self.assertTrue(ChainMap({1:2}, {}))
        self.assertTrue(ChainMap({}, {1:2}))

    def test_missing(self):
        # [FUNCTION DOC] Checks a subclass missing hook is used by subscripting, not by get, and
        # its exceptions propagate.
        class DefaultChainMap(ChainMap):
            # [CLASS DOC] HELPER: ChainMap subclass with a custom missing hook.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Returns a derived value from the key.
                return 999
        d = DefaultChainMap(dict(a=1, b=2), dict(b=20, c=30))
        for k, v in dict(a=1, b=2, c=30, d=999).items():
            self.assertEqual(d[k], v)                                  # check __getitem__ w/missing
        for k, v in dict(a=1, b=2, c=30, d=77).items():
            self.assertEqual(d.get(k, 77), v)                          # check get() w/ missing
        for k, v in dict(a=True, b=True, c=True, d=False).items():
            self.assertEqual(k in d, v)                                # check __contains__ w/missing
        self.assertEqual(d.pop('a', 1001), 1, d)
        self.assertEqual(d.pop('a', 1002), 1002)                       # check pop() w/missing
        self.assertEqual(d.popitem(), ('b', 2))                        # check popitem() w/missing
        with self.assertRaises(KeyError):
            d.popitem()

    def test_order_preservation(self):
        # [FUNCTION DOC] Checks insertion order of keys is preserved and updates keep position.
        d = ChainMap(
                OrderedDict(j=0, h=88888),
                OrderedDict(),
                OrderedDict(i=9999, d=4444, c=3333),
                OrderedDict(f=666, b=222, g=777, c=333, h=888),
                OrderedDict(),
                OrderedDict(e=55, b=22),
                OrderedDict(a=1, b=2, c=3, d=4, e=5),
                OrderedDict(),
            )
        self.assertEqual(''.join(d), 'abcdefghij')
        self.assertEqual(list(d.items()),
            [('a', 1), ('b', 222), ('c', 3333), ('d', 4444),
             ('e', 55), ('f', 666), ('g', 777), ('h', 88888),
             ('i', 9999), ('j', 0)])

    def test_iter_not_calling_getitem_on_maps(self):
        # [FUNCTION DOC] Checks that iterating a ChainMap does not call the getitem override of
        # its underlying UserDict maps.
        class DictWithGetItem(UserDict):
            # [CLASS DOC] HELPER: UserDict subclass recording that getitem was called.
            def __init__(self, *args, **kwds):
                # [FUNCTION DOC] HELPER: Initializes the dict and a flag marking getitem
                # as not called.
                self.called = False
                UserDict.__init__(self, *args, **kwds)
            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: Subscript override that raises an error to
                # detect unwanted calls.
                self.called = True
                UserDict.__getitem__(self, item)

        d = DictWithGetItem(a=1)
        c = ChainMap(d)
        d.called = False

        set(c)  # iterate over chain map
        self.assertFalse(d.called, '__getitem__ was called')

    def test_dict_coercion(self):
        # [FUNCTION DOC] Checks converting a ChainMap to a plain dict gives the merged mapping.
        d = ChainMap(dict(a=1, b=2), dict(b=20, c=30))
        self.assertEqual(dict(d), dict(a=1, b=2, c=30))
        self.assertEqual(dict(d.items()), dict(a=1, b=2, c=30))

    def test_new_child(self):
        # [FUNCTION DOC] Checks new_child with and without a given mapping, with keyword
        # arguments, and with dict subclasses and type preservation.
        'Tests for changes for issue #16613.'
        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        m = {'b':20, 'c': 30}
        d = c.new_child(m)
        self.assertEqual(d.maps, [{'b':20, 'c':30}, {'a':1, 'b':2}])  # check internal state
        self.assertIs(m, d.maps[0])

        # Use a different map than a dict
        class lowerdict(dict):
            # [CLASS DOC] HELPER: dict subclass with overridden lookup and containment to show
            # new_child works with custom mappings.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Subscript override returning a transformed result.
                if isinstance(key, str):
                    key = key.lower()
                return dict.__getitem__(self, key)
            def __contains__(self, key):
                # [FUNCTION DOC] HELPER: Containment override matching
                # the lookup transformation.
                if isinstance(key, str):
                    key = key.lower()
                return dict.__contains__(self, key)

        c = ChainMap()
        c['a'] = 1
        c['b'] = 2
        m = lowerdict(b=20, c=30)
        d = c.new_child(m)
        self.assertIs(m, d.maps[0])
        for key in 'abc':                                             # check contains
            self.assertIn(key, d)
        for k, v in dict(a=1, B=20, C=30, z=100).items():             # check get
            self.assertEqual(d.get(k, 100), v)

        c = ChainMap({'a': 1, 'b': 2})
        d = c.new_child(b=20, c=30)
        self.assertEqual(d.maps, [{'b': 20, 'c': 30}, {'a': 1, 'b': 2}])

    def test_union_operators(self):
        # [FUNCTION DOC] Checks the or and in-place or operators between ChainMap and mappings
        # or iterables of pairs, including subclasses, reflected or, and NotImplemented results.
        cm1 = ChainMap(dict(a=1, b=2), dict(c=3, d=4))
        cm2 = ChainMap(dict(a=10, e=5), dict(b=20, d=4))
        cm3 = cm1.copy()
        d = dict(a=10, c=30)
        pairs = [('c', 3), ('p',0)]

        tmp = cm1 | cm2 # testing between chainmaps
        self.assertEqual(tmp.maps, [cm1.maps[0] | dict(cm2), *cm1.maps[1:]])
        cm1 |= cm2
        self.assertEqual(tmp, cm1)

        tmp = cm2 | d # testing between chainmap and mapping
        self.assertEqual(tmp.maps, [cm2.maps[0] | d, *cm2.maps[1:]])
        self.assertEqual((d | cm2).maps, [d | dict(cm2)])
        cm2 |= d
        self.assertEqual(tmp, cm2)

        # testing behavior between chainmap and iterable key-value pairs
        with self.assertRaises(TypeError):
            cm3 | pairs
        tmp = cm3.copy()
        cm3 |= pairs
        self.assertEqual(cm3.maps, [tmp.maps[0] | dict(pairs), *tmp.maps[1:]])

        # testing proper return types for ChainMap and it's subclasses
        class Subclass(ChainMap):
            # [CLASS DOC] HELPER: ChainMap subclass used to check operator results keep
            # the subclass type.
            pass

        class SubclassRor(ChainMap):
            # [CLASS DOC] HELPER: ChainMap subclass that defines a reflected or operator.
            def __ror__(self, other):
                # [FUNCTION DOC] HELPER: Reflected or override used to test dispatch priority.
                return super().__ror__(other)

        tmp = ChainMap() | ChainMap()
        self.assertIs(type(tmp), ChainMap)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = ChainMap() | Subclass()
        self.assertIs(type(tmp), ChainMap)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = Subclass() | ChainMap()
        self.assertIs(type(tmp), Subclass)
        self.assertIs(type(tmp.maps[0]), dict)
        tmp = ChainMap() | SubclassRor()
        self.assertIs(type(tmp), SubclassRor)
        self.assertIs(type(tmp.maps[0]), dict)


################################################################################
### Named Tuples
################################################################################

TestNT = namedtuple('TestNT', 'x y z')    # type used for pickle tests

class TestNamedTuple(__TestCase):
    # [CLASS DOC] Tests namedtuple: factory arguments, defaults, immutability, docs, repr,
    # instances, tuple behavior, sizes, pickling, copying, name conflicts, subclassing,
    # descriptors and match args.

    def test_factory(self):
        # [FUNCTION DOC] Checks the factory with various field specs (string, list,
        # comma- separated), invalid names and keywords raising ValueError or TypeError,
        # and verbose handling.
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__name__, 'Point')
        self.assertEqual(Point.__slots__, ())
        self.assertEqual(Point.__module__, __name__)
        self.assertEqual(Point.__getitem__, tuple.__getitem__)
        self.assertEqual(Point._fields, ('x', 'y'))

        self.assertRaises(ValueError, namedtuple, 'abc%', 'efg ghi')       # type has non-alpha char
        self.assertRaises(ValueError, namedtuple, 'class', 'efg ghi')      # type has keyword
        self.assertRaises(ValueError, namedtuple, '9abc', 'efg ghi')       # type starts with digit

        self.assertRaises(ValueError, namedtuple, 'abc', 'efg g%hi')       # field with non-alpha char
        self.assertRaises(ValueError, namedtuple, 'abc', 'abc class')      # field has keyword
        self.assertRaises(ValueError, namedtuple, 'abc', '8efg 9ghi')      # field starts with digit
        self.assertRaises(ValueError, namedtuple, 'abc', '_efg ghi')       # field with leading underscore
        self.assertRaises(ValueError, namedtuple, 'abc', 'efg efg ghi')    # duplicate field

        namedtuple('Point0', 'x1 y2')   # Verify that numbers are allowed in names
        namedtuple('_', 'a b c')        # Test leading underscores in a typename

        nt = namedtuple('nt', 'the quick brown fox')                       # check unicode input
        self.assertNotIn("u'", repr(nt._fields))
        nt = namedtuple('nt', ('the', 'quick'))                           # check unicode input
        self.assertNotIn("u'", repr(nt._fields))

        self.assertRaises(TypeError, Point._make, [11])                     # catch too few args
        self.assertRaises(TypeError, Point._make, [11, 22, 33])             # catch too many args

    def test_defaults(self):
        # [FUNCTION DOC] Checks the defaults parameter: applies to rightmost fields, replace and
        # field defaults, errors when too many defaults or non-iterable.
        Point = namedtuple('Point', 'x y', defaults=(10, 20))              # 2 defaults
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

        Point = namedtuple('Point', 'x y', defaults=(20,))                 # 1 default
        self.assertEqual(Point._field_defaults, {'y': 20})
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))

        Point = namedtuple('Point', 'x y', defaults=())                     # 0 defaults
        self.assertEqual(Point._field_defaults, {})
        self.assertEqual(Point(1, 2), (1, 2))
        with self.assertRaises(TypeError):
            Point(1)

        with self.assertRaises(TypeError):                                  # catch too few args
            Point()
        with self.assertRaises(TypeError):                                  # catch too many args
            Point(1, 2, 3)
        with self.assertRaises(TypeError):                                  # too many defaults
            Point = namedtuple('Point', 'x y', defaults=(10, 20, 30))
        with self.assertRaises(TypeError):                                  # non-iterable defaults
            Point = namedtuple('Point', 'x y', defaults=10)
        with self.assertRaises(TypeError):                                  # another non-iterable default
            Point = namedtuple('Point', 'x y', defaults=False)

        Point = namedtuple('Point', 'x y', defaults=None)                   # default is None
        self.assertEqual(Point._field_defaults, {})
        self.assertIsNone(Point.__new__.__defaults__, None)
        self.assertEqual(Point(10, 20), (10, 20))
        with self.assertRaises(TypeError):                                  # catch too few args
            Point(10)

        Point = namedtuple('Point', 'x y', defaults=[10, 20])               # allow non-tuple iterable
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point.__new__.__defaults__, (10, 20))
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

        Point = namedtuple('Point', 'x y', defaults=iter([10, 20]))         # allow plain iterator
        self.assertEqual(Point._field_defaults, {'x': 10, 'y': 20})
        self.assertEqual(Point.__new__.__defaults__, (10, 20))
        self.assertEqual(Point(1, 2), (1, 2))
        self.assertEqual(Point(1), (1, 20))
        self.assertEqual(Point(), (10, 20))

    def test_readonly(self):
        # [FUNCTION DOC] Checks fields are read-only: attribute assignment and
        # deletion raise AttributeError.
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        with self.assertRaises(AttributeError):
            p.x = 33
        with self.assertRaises(AttributeError):
            del p.x
        with self.assertRaises(TypeError):
            p[0] = 33
        with self.assertRaises(TypeError):
            del p[0]
        self.assertEqual(p.x, 11)
        self.assertEqual(p[0], 11)

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_factory_doc_attr(self):
        # [FUNCTION DOC] Checks the docstring of the generated class and its fields.
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__doc__, 'Point(x, y)')
        Point.__doc__ = '2D point'
        self.assertEqual(Point.__doc__, '2D point')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_field_doc(self):
        # [FUNCTION DOC] Checks per-field doc attributes can be set and read.
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.x.__doc__, 'Alias for field number 0')
        self.assertEqual(Point.y.__doc__, 'Alias for field number 1')
        Point.x.__doc__ = 'docstring for Point.x'
        self.assertEqual(Point.x.__doc__, 'docstring for Point.x')
        # namedtuple can mutate doc of descriptors independently
        Vector = namedtuple('Vector', 'x y')
        self.assertEqual(Vector.x.__doc__, 'Alias for field number 0')
        Vector.x.__doc__ = 'docstring for Vector.x'
        self.assertEqual(Vector.x.__doc__, 'docstring for Vector.x')

    @support.cpython_only
    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_field_doc_reuse(self):
        # [FUNCTION DOC] Checks field doc objects are reused between namedtuple classes with the
        # same field count.
        P = namedtuple('P', ['m', 'n'])
        Q = namedtuple('Q', ['o', 'p'])
        self.assertIs(P.m.__doc__, Q.o.__doc__)
        self.assertIs(P.n.__doc__, Q.p.__doc__)

    @support.cpython_only
    def test_field_repr(self):
        # [FUNCTION DOC] Checks the repr of field accessors and instances.
        Point = namedtuple('Point', 'x y')
        self.assertEqual(repr(Point.x), "_tuplegetter(0, 'Alias for field number 0')")
        self.assertEqual(repr(Point.y), "_tuplegetter(1, 'Alias for field number 1')")

        Point.x.__doc__ = 'The x-coordinate'
        Point.y.__doc__ = 'The y-coordinate'

        self.assertEqual(repr(Point.x), "_tuplegetter(0, 'The x-coordinate')")
        self.assertEqual(repr(Point.y), "_tuplegetter(1, 'The y-coordinate')")

    def test_name_fixer(self):
        # [FUNCTION DOC] Checks invalid field names are renamed to positional names when
        # rename is enabled.
        for spec, renamed in [
            [('efg', 'g%hi'),  ('efg', '_1')],                              # field with non-alpha char
            [('abc', 'class'), ('abc', '_1')],                              # field has keyword
            [('8efg', '9ghi'), ('_0', '_1')],                               # field starts with digit
            [('abc', '_efg'), ('abc', '_1')],                               # field with leading underscore
            [('abc', 'efg', 'efg', 'ghi'), ('abc', 'efg', '_2', 'ghi')],    # duplicate field
            [('abc', '', 'x'), ('abc', '_1', 'x')],                         # fieldname is a space
        ]:
            self.assertEqual(namedtuple('NT', spec, rename=True)._fields, renamed)

    def test_module_parameter(self):
        # [FUNCTION DOC] Checks the module argument sets the module of the class.
        NT = namedtuple('NT', ['x', 'y'], module=collections)
        self.assertEqual(NT.__module__, collections)

    def test_instance(self):
        # [FUNCTION DOC] Checks instance construction, indexing, unpacking, asdict, replace,
        # repr, equality, hashing and error handling for bad arguments.
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        self.assertEqual(p, Point(x=11, y=22))
        self.assertEqual(p, Point(11, y=22))
        self.assertEqual(p, Point(y=22, x=11))
        self.assertEqual(p, Point(*(11, 22)))
        self.assertEqual(p, Point(**dict(x=11, y=22)))
        self.assertRaises(TypeError, Point, 1)          # too few args
        self.assertRaises(TypeError, Point, 1, 2, 3)    # too many args
        with self.assertRaises(TypeError):              # wrong keyword argument
            Point(XXX=1, y=2)
        with self.assertRaises(TypeError):              # missing keyword argument
            Point(x=1)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')
        self.assertNotIn('__weakref__', dir(p))
        self.assertEqual(p, Point._make([11, 22]))      # test _make classmethod
        self.assertEqual(p._fields, ('x', 'y'))         # test _fields attribute
        self.assertEqual(p._replace(x=1), (1, 22))      # test _replace method
        self.assertEqual(p._asdict(), dict(x=11, y=22)) # test _asdict method

        with self.assertRaises(TypeError):
            p._replace(x=1, error=2)

        # verify that field string can have commas
        Point = namedtuple('Point', 'x, y')
        p = Point(x=11, y=22)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')

        # verify that fieldspec can be a non-string sequence
        Point = namedtuple('Point', ('x', 'y'))
        p = Point(x=11, y=22)
        self.assertEqual(repr(p), 'Point(x=11, y=22)')

    def test_tupleness(self):
        # [FUNCTION DOC] Checks instances behave as tuples: isinstance, length, slicing,
        # concatenation and ordering comparisons.
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)

        self.assertIsInstance(p, tuple)
        self.assertEqual(p, (11, 22))                                       # matches a real tuple
        self.assertEqual(tuple(p), (11, 22))                                # coercible to a real tuple
        self.assertEqual(list(p), [11, 22])                                 # coercible to a list
        self.assertEqual(max(p), 22)                                        # iterable
        self.assertEqual(max(*p), 22)                                       # star-able
        x, y = p
        self.assertEqual(p, (x, y))                                         # unpacks like a tuple
        self.assertEqual((p[0], p[1]), (11, 22))                            # indexable like a tuple
        with self.assertRaises(IndexError):
            p[3]
        self.assertEqual(p[-1], 22)
        self.assertEqual(hash(p), hash((11, 22)))

        self.assertEqual(p.x, x)
        self.assertEqual(p.y, y)
        with self.assertRaises(AttributeError):
            p.z

    def test_odd_sizes(self):
        # [FUNCTION DOC] Checks namedtuples with zero to many fields.
        Zero = namedtuple('Zero', '')
        self.assertEqual(Zero(), ())
        self.assertEqual(Zero._make([]), ())
        self.assertEqual(repr(Zero()), 'Zero()')
        self.assertEqual(Zero()._asdict(), {})
        self.assertEqual(Zero()._fields, ())

        Dot = namedtuple('Dot', 'd')
        self.assertEqual(Dot(1), (1,))
        self.assertEqual(Dot._make([1]), (1,))
        self.assertEqual(Dot(1).d, 1)
        self.assertEqual(repr(Dot(1)), 'Dot(d=1)')
        self.assertEqual(Dot(1)._asdict(), {'d':1})
        self.assertEqual(Dot(1)._replace(d=999), (999,))
        self.assertEqual(Dot(1)._fields, ('d',))

    @support.requires_resource('cpu')
    def test_large_size(self):
        # [FUNCTION DOC] Checks creating a namedtuple with a very large number of fields.
        n = support.exceeds_recursion_limit()
        names = list(set(''.join([choice(string.ascii_letters)
                                  for j in range(10)]) for i in range(n)))
        n = len(names)
        Big = namedtuple('Big', names)
        b = Big(*range(n))
        self.assertEqual(b, tuple(range(n)))
        self.assertEqual(Big._make(range(n)), tuple(range(n)))
        for pos, name in enumerate(names):
            self.assertEqual(getattr(b, name), pos)
        repr(b)                                 # make sure repr() doesn't blow-up
        d = b._asdict()
        d_expected = dict(zip(names, range(n)))
        self.assertEqual(d, d_expected)
        b2 = b._replace(**dict([(names[1], 999),(names[-5], 42)]))
        b2_expected = list(range(n))
        b2_expected[1] = 999
        b2_expected[-5] = 42
        self.assertEqual(b2, tuple(b2_expected))
        self.assertEqual(b._fields, tuple(names))

    def test_pickle(self):
        # [FUNCTION DOC] Checks pickling and unpickling at all protocols preserves
        # type and value.
        p = TestNT(x=10, y=20, z=30)
        for module in (pickle,):
            loads = getattr(module, 'loads')
            dumps = getattr(module, 'dumps')
            for protocol in range(-1, module.HIGHEST_PROTOCOL + 1):
                q = loads(dumps(p, protocol))
                self.assertEqual(p, q)
                self.assertEqual(p._fields, q._fields)
                self.assertNotIn(b'OrderedDict', dumps(p, protocol))

    def test_copy(self):
        # [FUNCTION DOC] Checks copy and deepcopy return equal namedtuples.
        p = TestNT(x=10, y=20, z=30)
        for copier in copy.copy, copy.deepcopy:
            q = copier(p)
            self.assertEqual(p, q)
            self.assertEqual(p._fields, q._fields)

    @unittest.skip("https://github.com/pytorch/pytorch/issues/179778")
    def test_name_conflicts(self):
        # [FUNCTION DOC] Checks fields named like class attributes (such as map or replace) work
        # or are rejected correctly.
        # Some names like "self", "cls", "tuple", "itemgetter", and "property"
        # failed when used as field names.  Test to make sure these now work.
        T = namedtuple('T', 'itemgetter property self cls tuple')
        t = T(1, 2, 3, 4, 5)
        self.assertEqual(t, (1,2,3,4,5))
        newt = t._replace(itemgetter=10, property=20, self=30, cls=40, tuple=50)
        self.assertEqual(newt, (10,20,30,40,50))

       # Broader test of all interesting names taken from the code, old
       # template, and an example
        words = {'Alias', 'At', 'AttributeError', 'Build', 'Bypass', 'Create',
        'Encountered', 'Expected', 'Field', 'For', 'Got', 'Helper',
        'IronPython', 'Jython', 'KeyError', 'Make', 'Modify', 'Note',
        'OrderedDict', 'Point', 'Return', 'Returns', 'Type', 'TypeError',
        'Used', 'Validate', 'ValueError', 'Variables', 'a', 'accessible', 'add',
        'added', 'all', 'also', 'an', 'arg_list', 'args', 'arguments',
        'automatically', 'be', 'build', 'builtins', 'but', 'by', 'cannot',
        'class_namespace', 'classmethod', 'cls', 'collections', 'convert',
        'copy', 'created', 'creation', 'd', 'debugging', 'defined', 'dict',
        'dictionary', 'doc', 'docstring', 'docstrings', 'duplicate', 'effect',
        'either', 'enumerate', 'environments', 'error', 'example', 'exec', 'f',
        'f_globals', 'field', 'field_names', 'fields', 'formatted', 'frame',
        'function', 'functions', 'generate', 'get', 'getter', 'got', 'greater',
        'has', 'help', 'identifiers', 'index', 'indexable', 'instance',
        'instantiate', 'interning', 'introspection', 'isidentifier',
        'isinstance', 'itemgetter', 'iterable', 'join', 'keyword', 'keywords',
        'kwds', 'len', 'like', 'list', 'map', 'maps', 'message', 'metadata',
        'method', 'methods', 'module', 'module_name', 'must', 'name', 'named',
        'namedtuple', 'namedtuple_', 'names', 'namespace', 'needs', 'new',
        'nicely', 'num_fields', 'number', 'object', 'of', 'operator', 'option',
        'p', 'particular', 'pickle', 'pickling', 'plain', 'pop', 'positional',
        'property', 'r', 'regular', 'rename', 'replace', 'replacing', 'repr',
        'repr_fmt', 'representation', 'result', 'reuse_itemgetter', 's', 'seen',
        'self', 'sequence', 'set', 'side', 'specified', 'split', 'start',
        'startswith', 'step', 'str', 'string', 'strings', 'subclass', 'sys',
        'targets', 'than', 'the', 'their', 'this', 'to', 'tuple', 'tuple_new',
        'type', 'typename', 'underscore', 'unexpected', 'unpack', 'up', 'use',
        'used', 'user', 'valid', 'values', 'variable', 'verbose', 'where',
        'which', 'work', 'x', 'y', 'z', 'zip'}
        T = namedtuple('T', words)
        # test __new__
        values = tuple(range(len(words)))
        t = T(*values)
        self.assertEqual(t, values)
        t = T(**dict(zip(T._fields, values)))
        self.assertEqual(t, values)
        # test _make
        t = T._make(values)
        self.assertEqual(t, values)
        # exercise __repr__
        repr(t)
        # test _asdict
        self.assertEqual(t._asdict(), dict(zip(T._fields, values)))
        # test _replace
        t = T._make(values)
        newvalues = tuple(v*10 for v in values)
        newt = t._replace(**dict(zip(T._fields, newvalues)))
        self.assertEqual(newt, newvalues)
        # test _fields
        self.assertEqual(T._fields, tuple(words))
        # test __getnewargs__
        self.assertEqual(t.__getnewargs__(), values)

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of a namedtuple and of a subclass.
        A = namedtuple('A', 'x')
        self.assertEqual(repr(A(1)), 'A(x=1)')
        # repr should show the name of the subclass
        class B(A):
            # [CLASS DOC] HELPER: Subclass of a namedtuple used for repr checks.
            pass
        self.assertEqual(repr(B(1)), 'B(x=1)')

    def test_keyword_only_arguments(self):
        # [FUNCTION DOC] Checks keyword-only construction and errors for missing or
        # extra keyword arguments.
        # See issue 25628
        with self.assertRaises(TypeError):
            NT = namedtuple('NT', ['x', 'y'], True)

        NT = namedtuple('NT', ['abc', 'def'], rename=True)
        self.assertEqual(NT._fields, ('abc', '_1'))
        with self.assertRaises(TypeError):
            NT = namedtuple('NT', ['abc', 'def'], False, True)

    def test_namedtuple_subclass_issue_24931(self):
        # [FUNCTION DOC] Regression test for issue 24931: a namedtuple subclass has no instance
        # dict and asdict still works.
        class Point(namedtuple('_Point', ['x', 'y'])):
            # [CLASS DOC] HELPER: Subclass of a namedtuple base for the regression test.
            pass

        a = Point(3, 4)
        self.assertEqual(a._asdict(), OrderedDict([('x', 3), ('y', 4)]))

        a.w = 5
        self.assertEqual(a.__dict__, {'w': 5})

    @support.cpython_only
    def test_field_descriptor(self):
        # [FUNCTION DOC] Checks field accessors are descriptors that can be used on other
        # classes and raise on wrong instance types.
        Point = namedtuple('Point', 'x y')
        p = Point(11, 22)
        self.assertTrue(inspect.isdatadescriptor(Point.x))
        self.assertEqual(Point.x.__get__(p), 11)
        self.assertRaises(AttributeError, Point.x.__set__, p, 33)
        self.assertRaises(AttributeError, Point.x.__delete__, p)

        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                class NewPoint(tuple):
                    # [CLASS DOC] HELPER: tuple subclass using namedtuple field descriptors.
                    x = pickle.loads(pickle.dumps(Point.x, proto))
                    y = pickle.loads(pickle.dumps(Point.y, proto))

                np = NewPoint([1, 2])

                self.assertEqual(np.x, 1)
                self.assertEqual(np.y, 2)

    def test_new_builtins_issue_43102(self):
        # [FUNCTION DOC] Regression test for issue 43102: namedtuple works when a field name
        # shadows a builtin.
        obj = namedtuple('C', ())
        new_func = obj.__new__
        self.assertEqual(new_func.__globals__['__builtins__'], {})
        self.assertEqual(new_func.__builtins__, {})

    def test_match_args(self):
        # [FUNCTION DOC] Checks the match args attribute lists the field names.
        Point = namedtuple('Point', 'x y')
        self.assertEqual(Point.__match_args__, ('x', 'y'))

    def test_non_generic_subscript(self):
        # [FUNCTION DOC] Checks subscripting a non-generic namedtuple class raises TypeError.
        # For backward compatibility, subscription works
        # on arbitrary named tuple types.
        Group = collections.namedtuple('Group', 'key group')
        A = Group[int, list[int]]
        self.assertEqual(A.__origin__, Group)
        self.assertEqual(A.__parameters__, ())
        self.assertEqual(A.__args__, (int, list[int]))
        a = A(1, [2])
        self.assertIs(type(a), Group)
        self.assertEqual(a, (1, [2]))


################################################################################
### Abstract Base Classes
################################################################################

class ABCTestCase(__TestCase):
    # [CLASS DOC] Base test case with helpers for validating ABC behavior: abstract methods,
    # isinstance checks and comparison operators.

    def validate_abstract_methods(self, abc, *names):
        # [FUNCTION DOC] HELPER: Checks that the abc named in abc is abstract with
        # exactly the given abstract methods, and that subclasses overriding all of them
        # can be instantiated.
        methodstubs = dict.fromkeys(names, lambda s, *args: 0)

        # everything should work will all required methods are present
        C = type('C', (abc,), methodstubs)
        C()

        # Dynamo raises a hard error here that we can't easily capture
        # Commenting this part as this would also fail in eager if a user
        # attempt to run the same code

        # instantiation should fail if a required method is missing
        # for name in names:
        #     stubs = methodstubs.copy()
        #     del stubs[name]
        #     C = type('C', (abc,), stubs)
        #     self.assertRaises(TypeError, C, name)

    def validate_isinstance(self, abc, name):
        # [FUNCTION DOC] HELPER: Checks isinstance of the ABC for instances of the builtin type
        # with the given name.
        stub = lambda s, *args: 0

        C = type('C', (object,), {'__hash__': None})
        setattr(C, name, stub)
        self.assertIsInstance(C(), abc)
        self.assertTrue(issubclass(C, abc))

        C = type('C', (object,), {'__hash__': None})
        self.assertNotIsInstance(C(), abc)
        self.assertFalse(issubclass(C, abc))

    def validate_comparison(self, instance):
        # [FUNCTION DOC] HELPER: Checks comparison operators of an instance against itself and
        # against a custom object returning NotImplemented.
        ops = ['lt', 'gt', 'le', 'ge', 'ne', 'or', 'and', 'xor', 'sub']
        operators = {}
        for op in ops:
            name = '__' + op + '__'
            operators[name] = getattr(operator, name)

        class Other:
            # [CLASS DOC] HELPER: Class whose equality comparison reports being called,
            # to check delegation.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the called flag to false.
                self.right_side = False
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Records that equality was called
                # and returns NotImplemented.
                self.right_side = True
                return True
            __lt__ = __eq__
            __gt__ = __eq__
            __le__ = __eq__
            __ge__ = __eq__
            __ne__ = __eq__
            __ror__ = __eq__
            __rand__ = __eq__
            __rxor__ = __eq__
            __rsub__ = __eq__

        for name, op in operators.items():
            if not hasattr(instance, name):
                continue
            other = Other()
            op(instance, other)
            self.assertTrue(other.right_side,'Right side not called for %s.%s'
                            % (type(instance), name))

def _test_gen():
    # [FUNCTION DOC] HELPER: Generator function yielding a few values, used as a sample
    # iterator or generator.
    yield

class TestOneTrickPonyABCs(ABCTestCase):
    # [CLASS DOC] Tests the single-method ABCs (Awaitable, Coroutine, Hashable, AsyncIterable,
    # AsyncIterator, Iterable, Reversible, Collection, Iterator, Generator, AsyncGenerator,
    # Sized, Container, Callable): structural subclass hooks, abstract methods and registration.

    def test_Awaitable(self):
        # [FUNCTION DOC] Checks Awaitable isinstance and subclass hook against objects with and
        # without an await method, and subclass instantiation.
        def gen():
            # [FUNCTION DOC] HELPER: Plain generator function used as a non-awaitable sample.
            yield

        @types.coroutine
        def coro():
            # [FUNCTION DOC] HELPER: Generator-based coroutine function sample.
            yield

        async def new_coro():
            # [FUNCTION DOC] HELPER: Native async function sample.
            pass

        class Bar:
            # [CLASS DOC] HELPER: Class with an await method.
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await method returning an empty iterator.
                yield

        class MinimalCoro(Coroutine):
            # [CLASS DOC] HELPER: Minimal Coroutine subclass implementing send, throw and await.
            def send(self, value):
                # [FUNCTION DOC] HELPER: Send stub returning the given value.
                return value
            def throw(self, typ, val=None, tb=None):
                # [FUNCTION DOC] HELPER: Throw stub.
                super().throw(typ, val, tb)
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await stub.
                yield

        self.validate_abstract_methods(Awaitable, '__await__')

        non_samples = [None, int(), gen(), object()]
        for x in non_samples:
            self.assertNotIsInstance(x, Awaitable)
            self.assertFalse(issubclass(type(x), Awaitable), repr(type(x)))

        samples = [Bar(), MinimalCoro()]
        for x in samples:
            self.assertIsInstance(x, Awaitable)
            self.assertTrue(issubclass(type(x), Awaitable))

        c = coro()
        # Iterable coroutines (generators with CO_ITERABLE_COROUTINE
        # flag don't have '__await__' method, hence can't be instances
        # of Awaitable. Use inspect.isawaitable to detect them.
        self.assertNotIsInstance(c, Awaitable)

        c = new_coro()
        self.assertIsInstance(c, Awaitable)
        c.close() # avoid RuntimeWarning that coro() was not awaited

        # [CLASS DOC] HELPER: Class lacking the required methods, used for negative checks.
        class CoroLike: pass
        Coroutine.register(CoroLike)
        self.assertTrue(isinstance(CoroLike(), Awaitable))
        self.assertTrue(issubclass(CoroLike, Awaitable))
        CoroLike = None
        support.gc_collect() # Kill CoroLike to clean-up ABCMeta cache

    def test_Coroutine(self):
        # [FUNCTION DOC] Checks Coroutine isinstance and subclass hook, the default close method
        # and its error cases.
        def gen():
            # [FUNCTION DOC] HELPER: Plain generator function used as a non-coroutine sample.
            yield

        @types.coroutine
        def coro():
            # [FUNCTION DOC] HELPER: Generator-based coroutine sample.
            yield

        async def new_coro():
            # [FUNCTION DOC] HELPER: Native async function sample.
            pass

        class Bar:
            # [CLASS DOC] HELPER: Class with only an await method (not a coroutine).
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await stub.
                yield

        class MinimalCoro(Coroutine):
            # [CLASS DOC] HELPER: Minimal Coroutine subclass.
            def send(self, value):
                # [FUNCTION DOC] HELPER: Send stub.
                return value
            def throw(self, typ, val=None, tb=None):
                # [FUNCTION DOC] HELPER: Throw stub.
                super().throw(typ, val, tb)
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await stub.
                yield

        self.validate_abstract_methods(Coroutine, '__await__', 'send', 'throw')

        non_samples = [None, int(), gen(), object(), Bar()]
        for x in non_samples:
            self.assertNotIsInstance(x, Coroutine)
            self.assertFalse(issubclass(type(x), Coroutine), repr(type(x)))

        samples = [MinimalCoro()]
        for x in samples:
            self.assertIsInstance(x, Awaitable)
            self.assertTrue(issubclass(type(x), Awaitable))

        c = coro()
        # Iterable coroutines (generators with CO_ITERABLE_COROUTINE
        # flag don't have '__await__' method, hence can't be instances
        # of Coroutine. Use inspect.isawaitable to detect them.
        self.assertNotIsInstance(c, Coroutine)

        c = new_coro()
        self.assertIsInstance(c, Coroutine)
        c.close() # avoid RuntimeWarning that coro() was not awaited

        class CoroLike:
            # [CLASS DOC] HELPER: Class with send, throw, close and await methods, recognized
            # structurally as a coroutine.
            def send(self, value):
                # [FUNCTION DOC] HELPER: Send stub.
                pass
            def throw(self, typ, val=None, tb=None):
                # [FUNCTION DOC] HELPER: Throw stub.
                pass
            def close(self):
                # [FUNCTION DOC] HELPER: Close stub.
                pass
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await stub.
                pass
        self.assertTrue(isinstance(CoroLike(), Coroutine))
        self.assertTrue(issubclass(CoroLike, Coroutine))

        class CoroLike:
            # [CLASS DOC] HELPER: Class missing throw, so not recognized as a coroutine.
            def send(self, value):
                # [FUNCTION DOC] HELPER: Send stub.
                pass
            def close(self):
                # [FUNCTION DOC] HELPER: Close stub.
                pass
            def __await__(self):
                # [FUNCTION DOC] HELPER: Await stub.
                pass
        self.assertFalse(isinstance(CoroLike(), Coroutine))
        self.assertFalse(issubclass(CoroLike, Coroutine))

    def test_Hashable(self):
        # [FUNCTION DOC] Checks Hashable for hashable and unhashable builtins and for
        # classes disabling hash.
        # Check some non-hashables
        non_samples = [bytearray(), list(), set(), dict()]
        for x in non_samples:
            self.assertNotIsInstance(x, Hashable)
            self.assertFalse(issubclass(type(x), Hashable), repr(type(x)))
        # Check some hashables
        samples = [None,
                   int(), float(), complex(),
                   str(),
                   tuple(), frozenset(),
                   int, list, object, type, bytes()
                   ]
        for x in samples:
            self.assertIsInstance(x, Hashable)
            self.assertTrue(issubclass(type(x), Hashable), repr(type(x)))
        self.assertRaises(TypeError, Hashable)
        # Check direct subclassing
        class H(Hashable):
            # [CLASS DOC] HELPER: Hashable subclass.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns a fixed hash value.
                return super().__hash__()
        self.assertEqual(hash(H()), 0)
        self.assertFalse(issubclass(int, H))
        self.validate_abstract_methods(Hashable, '__hash__')
        self.validate_isinstance(Hashable, '__hash__')

    def test_AsyncIterable(self):
        # [FUNCTION DOC] Checks AsyncIterable recognition via the async iteration method.
        class AI:
            # [CLASS DOC] HELPER: Class with an async iteration method.
            def __aiter__(self):
                # [FUNCTION DOC] HELPER: Returns an async iterator.
                return self
        self.assertTrue(isinstance(AI(), AsyncIterable))
        self.assertTrue(issubclass(AI, AsyncIterable))
        # Check some non-iterables
        non_samples = [None, object, []]
        for x in non_samples:
            self.assertNotIsInstance(x, AsyncIterable)
            self.assertFalse(issubclass(type(x), AsyncIterable), repr(type(x)))
        self.validate_abstract_methods(AsyncIterable, '__aiter__')
        self.validate_isinstance(AsyncIterable, '__aiter__')

    def test_AsyncIterator(self):
        # [FUNCTION DOC] Checks AsyncIterator recognition requires both async iteration
        # and next-item methods.
        class AI:
            # [CLASS DOC] HELPER: Class implementing the async iterator protocol.
            def __aiter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            async def __anext__(self):
                # [FUNCTION DOC] HELPER: Async next-item stub.
                raise StopAsyncIteration
        self.assertTrue(isinstance(AI(), AsyncIterator))
        self.assertTrue(issubclass(AI, AsyncIterator))
        non_samples = [None, object, []]
        # Check some non-iterables
        for x in non_samples:
            self.assertNotIsInstance(x, AsyncIterator)
            self.assertFalse(issubclass(type(x), AsyncIterator), repr(type(x)))
        # Similarly to regular iterators (see issue 10565)
        class AnextOnly:
            # [CLASS DOC] HELPER: Class with only an async next- item method (not
            # an async iterator).
            async def __anext__(self):
                # [FUNCTION DOC] HELPER: Async next-item stub.
                raise StopAsyncIteration
        self.assertNotIsInstance(AnextOnly(), AsyncIterator)
        self.validate_abstract_methods(AsyncIterator, '__anext__', '__aiter__')

    def test_Iterable(self):
        # [FUNCTION DOC] Checks Iterable for builtins, generators and classes with or without
        # the iteration method, including blocking by setting it to None.
        # Check some non-iterables
        non_samples = [None, 42, 3.14, 1j]
        for x in non_samples:
            self.assertNotIsInstance(x, Iterable)
            self.assertFalse(issubclass(type(x), Iterable), repr(type(x)))
        # Check some iterables
        samples = [bytes(), str(),
                   tuple(), list(), set(), frozenset(), dict(),
                   dict().keys(), dict().items(), dict().values(),
                   _test_gen(),
                   (x for x in []),
                   ]
        for x in samples:
            self.assertIsInstance(x, Iterable)
            self.assertTrue(issubclass(type(x), Iterable), repr(type(x)))
        # Check direct subclassing
        class I(Iterable):
            # [CLASS DOC] HELPER: Iterable subclass.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over nothing.
                return super().__iter__()
        self.assertEqual(list(I()), [])
        self.assertFalse(issubclass(str, I))
        self.validate_abstract_methods(Iterable, '__iter__')
        self.validate_isinstance(Iterable, '__iter__')
        # Check None blocking
        class It:
            # [CLASS DOC] HELPER: Class with an iteration method. [FUNCTION DOC] HELPER: Returns
            # an empty iterator (one-line method).
            def __iter__(self): return iter([])
        class ItBlocked(It):
            # [CLASS DOC] HELPER: Subclass that blocks iteration by setting the method to None.
            __iter__ = None
        self.assertTrue(issubclass(It, Iterable))
        self.assertTrue(isinstance(It(), Iterable))
        self.assertFalse(issubclass(ItBlocked, Iterable))
        self.assertFalse(isinstance(ItBlocked(), Iterable))

    def test_Reversible(self):
        # [FUNCTION DOC] Checks Reversible for sequences, mappings and custom classes, including
        # blocking via None.
        # Check some non-reversibles
        non_samples = [None, 42, 3.14, 1j, set(), frozenset()]
        for x in non_samples:
            self.assertNotIsInstance(x, Reversible)
            self.assertFalse(issubclass(type(x), Reversible), repr(type(x)))
        # Check some non-reversible iterables
        non_reversibles = [_test_gen(), (x for x in []), iter([]), reversed([])]
        for x in non_reversibles:
            self.assertNotIsInstance(x, Reversible)
            self.assertFalse(issubclass(type(x), Reversible), repr(type(x)))
        # Check some reversible iterables
        samples = [bytes(), str(), tuple(), list(), OrderedDict(),
                   OrderedDict().keys(), OrderedDict().items(),
                   OrderedDict().values(), Counter(), Counter().keys(),
                   Counter().items(), Counter().values(), dict(),
                   dict().keys(), dict().items(), dict().values()]
        for x in samples:
            self.assertIsInstance(x, Reversible)
            self.assertTrue(issubclass(type(x), Reversible), repr(type(x)))
        # Check also Mapping, MutableMapping, and Sequence
        self.assertTrue(issubclass(Sequence, Reversible), repr(Sequence))
        self.assertFalse(issubclass(Mapping, Reversible), repr(Mapping))
        self.assertFalse(issubclass(MutableMapping, Reversible), repr(MutableMapping))
        # Check direct subclassing
        class R(Reversible):
            # [CLASS DOC] HELPER: Reversible subclass.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(list())
            def __reversed__(self):
                # [FUNCTION DOC] HELPER: Reverse iteration stub.
                return iter(list())
        self.assertEqual(list(reversed(R())), [])
        self.assertFalse(issubclass(float, R))
        self.validate_abstract_methods(Reversible, '__reversed__', '__iter__')
        # Check reversible non-iterable (which is not Reversible)
        class RevNoIter:
            # [CLASS DOC] HELPER: Class with only a reversed method. [FUNCTION DOC] HELPER:
            # Returns an empty reversed iterator.
            def __reversed__(self): return reversed([])
        class RevPlusIter(RevNoIter):
            # [CLASS DOC] HELPER: Subclass adding an iteration method. [FUNCTION DOC] HELPER:
            # Returns an empty iterator.
            def __iter__(self): return iter([])
        self.assertFalse(issubclass(RevNoIter, Reversible))
        self.assertFalse(isinstance(RevNoIter(), Reversible))
        self.assertTrue(issubclass(RevPlusIter, Reversible))
        self.assertTrue(isinstance(RevPlusIter(), Reversible))
        # Check None blocking
        class Rev:
            # [CLASS DOC] HELPER: Class with both iteration and reversed methods. [FUNCTION DOC]
            # HELPER: Returns an empty iterator.
            def __iter__(self): return iter([])
            # [FUNCTION DOC] HELPER: Returns an empty reversed iterator.
            def __reversed__(self): return reversed([])
        class RevItBlocked(Rev):
            # [CLASS DOC] HELPER: Subclass that blocks iteration.
            __iter__ = None
        class RevRevBlocked(Rev):
            # [CLASS DOC] HELPER: Subclass that blocks reversal.
            __reversed__ = None
        self.assertTrue(issubclass(Rev, Reversible))
        self.assertTrue(isinstance(Rev(), Reversible))
        self.assertFalse(issubclass(RevItBlocked, Reversible))
        self.assertFalse(isinstance(RevItBlocked(), Reversible))
        self.assertFalse(issubclass(RevRevBlocked, Reversible))
        self.assertFalse(isinstance(RevRevBlocked(), Reversible))

    def test_Collection(self):
        # [FUNCTION DOC] Checks Collection for builtins and for classes with various
        # combinations of iteration, length and containment methods, including blocking by None.
        # Check some non-collections
        non_collections = [None, 42, 3.14, 1j, lambda x: 2*x]
        for x in non_collections:
            self.assertNotIsInstance(x, Collection)
            self.assertFalse(issubclass(type(x), Collection), repr(type(x)))
        # Check some non-collection iterables
        non_col_iterables = [_test_gen(), iter(b''), iter(bytearray()),
                             (x for x in [])]
        for x in non_col_iterables:
            self.assertNotIsInstance(x, Collection)
            self.assertFalse(issubclass(type(x), Collection), repr(type(x)))
        # Check some collections
        samples = [set(), frozenset(), dict(), bytes(), str(), tuple(),
                   list(), dict().keys(), dict().items(), dict().values()]
        for x in samples:
            self.assertIsInstance(x, Collection)
            self.assertTrue(issubclass(type(x), Collection), repr(type(x)))
        # Check also Mapping, MutableMapping, etc.
        self.assertTrue(issubclass(Sequence, Collection), repr(Sequence))
        self.assertTrue(issubclass(Mapping, Collection), repr(Mapping))
        self.assertTrue(issubclass(MutableMapping, Collection),
                                    repr(MutableMapping))
        self.assertTrue(issubclass(Set, Collection), repr(Set))
        self.assertTrue(issubclass(MutableSet, Collection), repr(MutableSet))
        self.assertTrue(issubclass(Sequence, Collection), repr(MutableSet))
        # Check direct subclassing
        class Col(Collection):
            # [CLASS DOC] HELPER: Collection subclass with all three methods.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(list())
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Containment stub.
                return False
        # [CLASS DOC] HELPER: Subclass of the collection with no changes.
        class DerCol(Col): pass
        self.assertEqual(list(iter(Col())), [])
        self.assertFalse(issubclass(list, Col))
        self.assertFalse(issubclass(set, Col))
        self.assertFalse(issubclass(float, Col))
        self.assertEqual(list(iter(DerCol())), [])
        self.assertFalse(issubclass(list, DerCol))
        self.assertFalse(issubclass(set, DerCol))
        self.assertFalse(issubclass(float, DerCol))
        self.validate_abstract_methods(Collection, '__len__', '__iter__',
                                                   '__contains__')
        # Check sized container non-iterable (which is not Collection) etc.
        class ColNoIter:
            # [CLASS DOC] HELPER: Class lacking iteration. [FUNCTION DOC] HELPER: Returns
            # zero (one-line method).
            def __len__(self): return 0
            # [FUNCTION DOC] HELPER: Returns false (one-line method).
            def __contains__(self, item): return False
        class ColNoSize:
            # [CLASS DOC] HELPER: Class lacking length. [FUNCTION DOC] HELPER: Returns
            # an empty iterator.
            def __iter__(self): return iter([])
            # [FUNCTION DOC] HELPER: Returns false.
            def __contains__(self, item): return False
        class ColNoCont:
            # [CLASS DOC] HELPER: Class lacking containment. [FUNCTION DOC] HELPER: Returns
            # an empty iterator.
            def __iter__(self): return iter([])
            # [FUNCTION DOC] HELPER: Returns zero.
            def __len__(self): return 0
        self.assertFalse(issubclass(ColNoIter, Collection))
        self.assertFalse(isinstance(ColNoIter(), Collection))
        self.assertFalse(issubclass(ColNoSize, Collection))
        self.assertFalse(isinstance(ColNoSize(), Collection))
        self.assertFalse(issubclass(ColNoCont, Collection))
        self.assertFalse(isinstance(ColNoCont(), Collection))

        # Check None blocking
        class SizeBlock:
            # [CLASS DOC] HELPER: Class whose containment method has the wrong signature, with
            # length blocked. [FUNCTION DOC] HELPER: Returns an empty iterator.
            def __iter__(self): return iter([])
            # [FUNCTION DOC] HELPER: Containment stub taking no argument.
            def __contains__(self): return False
            __len__ = None
        class IterBlock:
            # [CLASS DOC] HELPER: Class with iteration blocked. [FUNCTION DOC]
            # HELPER: Returns zero.
            def __len__(self): return 0
            # [FUNCTION DOC] HELPER: Returns true.
            def __contains__(self): return True
            __iter__ = None
        self.assertFalse(issubclass(SizeBlock, Collection))
        self.assertFalse(isinstance(SizeBlock(), Collection))
        self.assertFalse(issubclass(IterBlock, Collection))
        self.assertFalse(isinstance(IterBlock(), Collection))
        # Check None blocking in subclass
        class ColImpl:
            # [CLASS DOC] HELPER: Fully featured collection implementation.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(list())
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Containment stub.
                return False
        class NonCol(ColImpl):
            # [CLASS DOC] HELPER: Subclass that blocks a method, so it is not a Collection.
            __contains__ = None
        self.assertFalse(issubclass(NonCol, Collection))
        self.assertFalse(isinstance(NonCol(), Collection))


    def test_Iterator(self):
        # [FUNCTION DOC] Checks Iterator for builtin iterators, generators and classes
        # having next only.
        non_samples = [None, 42, 3.14, 1j, b"", "", (), [], {}, set()]
        for x in non_samples:
            self.assertNotIsInstance(x, Iterator)
            self.assertFalse(issubclass(type(x), Iterator), repr(type(x)))
        samples = [iter(bytes()), iter(str()),
                   iter(tuple()), iter(list()), iter(dict()),
                   iter(set()), iter(frozenset()),
                   iter(dict().keys()), iter(dict().items()),
                   iter(dict().values()),
                   _test_gen(),
                   (x for x in []),
                   ]
        for x in samples:
            self.assertIsInstance(x, Iterator)
            self.assertTrue(issubclass(type(x), Iterator), repr(type(x)))
        self.validate_abstract_methods(Iterator, '__next__', '__iter__')

        # Issue 10565
        class NextOnly:
            # [CLASS DOC] HELPER: Class with only a next-item method.
            def __next__(self):
                # [FUNCTION DOC] HELPER: Next-item stub.
                yield 1
                return
        self.assertNotIsInstance(NextOnly(), Iterator)

    def test_Generator(self):
        # [FUNCTION DOC] Checks Generator: structural recognition needs iteration, next, close,
        # send and throw; mixin methods close and iteration; close raises RuntimeError if the
        # throw hook fails or ignores GeneratorExit.
        class NonGen1:
            # [CLASS DOC] HELPER: Class missing send. [FUNCTION DOC] HELPER: Returns
            # self (one-line method).
            def __iter__(self): return self
            # [FUNCTION DOC] HELPER: Returns None.
            def __next__(self): return None
            # [FUNCTION DOC] HELPER: Close stub.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Throw stub.
            def throw(self, typ, val=None, tb=None): pass

        class NonGen2:
            # [CLASS DOC] HELPER: Class missing throw. [FUNCTION DOC] HELPER: Returns self.
            def __iter__(self): return self
            # [FUNCTION DOC] HELPER: Returns None.
            def __next__(self): return None
            # [FUNCTION DOC] HELPER: Close stub.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def send(self, value): return value

        class NonGen3:
            # [CLASS DOC] HELPER: Class missing iteration and next. [FUNCTION DOC]
            # HELPER: Close stub.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def send(self, value): return value
            # [FUNCTION DOC] HELPER: Throw stub.
            def throw(self, typ, val=None, tb=None): pass

        non_samples = [
            None, 42, 3.14, 1j, b"", "", (), [], {}, set(),
            iter(()), iter([]), NonGen1(), NonGen2(), NonGen3()]
        for x in non_samples:
            self.assertNotIsInstance(x, Generator)
            self.assertFalse(issubclass(type(x), Generator), repr(type(x)))

        class Gen:
            # [CLASS DOC] HELPER: Class implementing the generator protocol structurally.
            # [FUNCTION DOC] HELPER: Returns self.
            def __iter__(self): return self
            # [FUNCTION DOC] HELPER: Returns None.
            def __next__(self): return None
            # [FUNCTION DOC] HELPER: Close stub.
            def close(self): pass
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def send(self, value): return value
            # [FUNCTION DOC] HELPER: Throw stub.
            def throw(self, typ, val=None, tb=None): pass

        class MinimalGen(Generator):
            # [CLASS DOC] HELPER: Generator subclass implementing send and throw only.
            def send(self, value):
                # [FUNCTION DOC] HELPER: Send stub.
                return value
            def throw(self, typ, val=None, tb=None):
                # [FUNCTION DOC] HELPER: Throw stub.
                super().throw(typ, val, tb)

        def gen():
            # [FUNCTION DOC] HELPER: Sample generator function.
            yield 1

        samples = [gen(), (lambda: (yield))(), Gen(), MinimalGen()]
        for x in samples:
            self.assertIsInstance(x, Iterator)
            self.assertIsInstance(x, Generator)
            self.assertTrue(issubclass(type(x), Generator), repr(type(x)))
        self.validate_abstract_methods(Generator, 'send', 'throw')

        # mixin tests
        mgen = MinimalGen()
        self.assertIs(mgen, iter(mgen))
        self.assertIs(mgen.send(None), next(mgen))
        self.assertEqual(2, mgen.send(2))
        self.assertIsNone(mgen.close())
        self.assertRaises(ValueError, mgen.throw, ValueError)
        self.assertRaisesRegex(ValueError, "^huhu$",
                               mgen.throw, ValueError, ValueError("huhu"))
        self.assertRaises(StopIteration, mgen.throw, StopIteration())

        class FailOnClose(Generator):
            # [CLASS DOC] HELPER: Generator subclass whose throw fails with ValueError.
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def send(self, value): return value
            # [FUNCTION DOC] HELPER: Throw stub raising ValueError.
            def throw(self, *args): raise ValueError

        self.assertRaises(ValueError, FailOnClose().close)

        class IgnoreGeneratorExit(Generator):
            # [CLASS DOC] HELPER: Generator subclass whose throw swallows the exit exception.
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def send(self, value): return value
            # [FUNCTION DOC] HELPER: Throw stub doing nothing.
            def throw(self, *args): pass

        self.assertRaises(RuntimeError, IgnoreGeneratorExit().close)

    def test_AsyncGenerator(self):
        # [FUNCTION DOC] Checks AsyncGenerator: structural recognition needs aiter, anext,
        # aclose, asend and athrow; mixin aclose behaves correctly on failure or ignored exit,
        # using a small runner for awaitables.
        class NonAGen1:
            # [CLASS DOC] HELPER: Class missing asend. [FUNCTION DOC] HELPER: Returns self.
            def __aiter__(self): return self
            # [FUNCTION DOC] HELPER: Returns None.
            def __anext__(self): return None
            # [FUNCTION DOC] HELPER: Close stub.
            def aclose(self): pass
            # [FUNCTION DOC] HELPER: Throw stub.
            def athrow(self, typ, val=None, tb=None): pass

        class NonAGen2:
            # [CLASS DOC] HELPER: Class missing athrow. [FUNCTION DOC] HELPER: Returns self.
            def __aiter__(self): return self
            # [FUNCTION DOC] HELPER: Returns None.
            def __anext__(self): return None
            # [FUNCTION DOC] HELPER: Close stub.
            def aclose(self): pass
            # [FUNCTION DOC] HELPER: Send stub returning the value.
            def asend(self, value): return value

        class NonAGen3:
            # [CLASS DOC] HELPER: Class missing aiter and anext. [FUNCTION DOC]
            # HELPER: Close stub.
            def aclose(self): pass
            # [FUNCTION DOC] HELPER: Send stub.
            def asend(self, value): return value
            # [FUNCTION DOC] HELPER: Throw stub.
            def athrow(self, typ, val=None, tb=None): pass

        non_samples = [
            None, 42, 3.14, 1j, b"", "", (), [], {}, set(),
            iter(()), iter([]), NonAGen1(), NonAGen2(), NonAGen3()]
        for x in non_samples:
            self.assertNotIsInstance(x, AsyncGenerator)
            self.assertFalse(issubclass(type(x), AsyncGenerator), repr(type(x)))

        class Gen:
            # [CLASS DOC] HELPER: Class implementing the async generator protocol. [FUNCTION
            # DOC] HELPER: Returns self.
            def __aiter__(self): return self
            # [FUNCTION DOC] HELPER: Async next stub.
            async def __anext__(self): return None
            # [FUNCTION DOC] HELPER: Async close stub.
            async def aclose(self): pass
            # [FUNCTION DOC] HELPER: Async send stub.
            async def asend(self, value): return value
            # [FUNCTION DOC] HELPER: Async throw stub.
            async def athrow(self, typ, val=None, tb=None): pass

        class MinimalAGen(AsyncGenerator):
            # [CLASS DOC] HELPER: AsyncGenerator subclass with send and throw only.
            async def asend(self, value):
                # [FUNCTION DOC] HELPER: Async send stub.
                return value
            async def athrow(self, typ, val=None, tb=None):
                # [FUNCTION DOC] HELPER: Async throw stub.
                await super().athrow(typ, val, tb)

        async def gen():
            # [FUNCTION DOC] HELPER: Sample async generator function.
            yield 1

        samples = [gen(), Gen(), MinimalAGen()]
        for x in samples:
            self.assertIsInstance(x, AsyncIterator)
            self.assertIsInstance(x, AsyncGenerator)
            self.assertTrue(issubclass(type(x), AsyncGenerator), repr(type(x)))
        self.validate_abstract_methods(AsyncGenerator, 'asend', 'athrow')

        def run_async(coro):
            # [FUNCTION DOC] HELPER: Drives a coroutine to completion by repeatedly sending None
            # and returns its result from the stop signal.
            result = None
            while True:
                try:
                    coro.send(None)
                except StopIteration as ex:
                    result = ex.args[0] if ex.args else None
                    break
            return result

        # mixin tests
        mgen = MinimalAGen()
        self.assertIs(mgen, mgen.__aiter__())
        self.assertIs(run_async(mgen.asend(None)), run_async(mgen.__anext__()))
        self.assertEqual(2, run_async(mgen.asend(2)))
        self.assertIsNone(run_async(mgen.aclose()))
        with self.assertRaises(ValueError):
            run_async(mgen.athrow(ValueError))

        class FailOnClose(AsyncGenerator):
            # [CLASS DOC] HELPER: AsyncGenerator subclass whose throw fails with ValueError.
            # [FUNCTION DOC] HELPER: Async send stub.
            async def asend(self, value): return value
            # [FUNCTION DOC] HELPER: Async throw stub raising ValueError.
            async def athrow(self, *args): raise ValueError

        with self.assertRaises(ValueError):
            run_async(FailOnClose().aclose())

        class IgnoreGeneratorExit(AsyncGenerator):
            # [CLASS DOC] HELPER: AsyncGenerator subclass whose throw swallows the exit
            # exception. [FUNCTION DOC] HELPER: Async send stub.
            async def asend(self, value): return value
            # [FUNCTION DOC] HELPER: Async throw stub doing nothing.
            async def athrow(self, *args): pass

        with self.assertRaises(RuntimeError):
            run_async(IgnoreGeneratorExit().aclose())

    def test_Sized(self):
        # [FUNCTION DOC] Checks Sized recognition via the length method and blocking.
        non_samples = [None, 42, 3.14, 1j,
                       _test_gen(),
                       (x for x in []),
                       ]
        for x in non_samples:
            self.assertNotIsInstance(x, Sized)
            self.assertFalse(issubclass(type(x), Sized), repr(type(x)))
        samples = [bytes(), str(),
                   tuple(), list(), set(), frozenset(), dict(),
                   dict().keys(), dict().items(), dict().values(),
                   ]
        for x in samples:
            self.assertIsInstance(x, Sized)
            self.assertTrue(issubclass(type(x), Sized), repr(type(x)))
        self.validate_abstract_methods(Sized, '__len__')
        self.validate_isinstance(Sized, '__len__')

    def test_Container(self):
        # [FUNCTION DOC] Checks Container recognition via the containment method and blocking.
        non_samples = [None, 42, 3.14, 1j,
                       _test_gen(),
                       (x for x in []),
                       ]
        for x in non_samples:
            self.assertNotIsInstance(x, Container)
            self.assertFalse(issubclass(type(x), Container), repr(type(x)))
        samples = [bytes(), str(),
                   tuple(), list(), set(), frozenset(), dict(),
                   dict().keys(), dict().items(),
                   ]
        for x in samples:
            self.assertIsInstance(x, Container)
            self.assertTrue(issubclass(type(x), Container), repr(type(x)))
        self.validate_abstract_methods(Container, '__contains__')
        self.validate_isinstance(Container, '__contains__')

    def test_Callable(self):
        # [FUNCTION DOC] Checks Callable recognition via the call method, including plain
        # functions and classes.
        non_samples = [None, 42, 3.14, 1j,
                       "", b"", (), [], {}, set(),
                       _test_gen(),
                       (x for x in []),
                       ]
        for x in non_samples:
            self.assertNotIsInstance(x, Callable)
            self.assertFalse(issubclass(type(x), Callable), repr(type(x)))
        samples = [lambda: None,
                   type, int, object,
                   len,
                   list.append, [].append,
                   ]
        for x in samples:
            self.assertIsInstance(x, Callable)
            self.assertTrue(issubclass(type(x), Callable), repr(type(x)))
        self.validate_abstract_methods(Callable, '__call__')
        self.validate_isinstance(Callable, '__call__')

    def test_direct_subclassing(self):
        # [FUNCTION DOC] Checks that direct subclassing of ABCs without implementing abstract
        # methods leaves the subclass abstract (instantiation raises TypeError).
        for B in Hashable, Iterable, Iterator, Reversible, Sized, Container, Callable:
            class C(B):
                # [CLASS DOC] HELPER: Subclass of the ABC under test.
                pass
            self.assertTrue(issubclass(C, B))
            self.assertFalse(issubclass(int, C))

    def test_registration(self):
        # [FUNCTION DOC] Checks registering a class as virtual subclass of each ABC makes
        # issubclass and isinstance succeed.
        for B in Hashable, Iterable, Iterator, Reversible, Sized, Container, Callable:
            class C:
                # [CLASS DOC] HELPER: Plain class registered with the ABC.
                __hash__ = None  # Make sure it isn't hashable by default
            self.assertFalse(issubclass(C, B), B.__name__)
            B.register(C)
            self.assertTrue(issubclass(C, B))

class WithSet(MutableSet):
    # [CLASS DOC] HELPER: Minimal concrete MutableSet implementation backed by a list, used to
    # test mixin methods.

    def __init__(self, it=()):
        # [FUNCTION DOC] HELPER: Builds the backing list from an iterable.
        self.data = set(it)

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the number of stored items.
        return len(self.data)

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Iterates over the stored items.
        return iter(self.data)

    def __contains__(self, item):
        # [FUNCTION DOC] HELPER: Returns whether the item is stored.
        return item in self.data

    def add(self, item):
        # [FUNCTION DOC] HELPER: Adds the item if not already stored.
        self.data.add(item)

    def discard(self, item):
        # [FUNCTION DOC] HELPER: Removes the item if present.
        self.data.discard(item)

class TestCollectionABCs(ABCTestCase):
    # [CLASS DOC] Tests the collection ABCs Set, MutableSet, Mapping, MutableMapping,
    # Sequence, MutableSequence, ByteString and Buffer: mixin methods, interoperability with
    # builtins and regression issues.

    # XXX For now, we only test some virtual inheritance properties.
    # We should also test the proper behavior of the collection ABCs
    # as real base classes or mix-in classes.

    def test_Set(self):
        # [FUNCTION DOC] Checks Set abstract methods, isinstance of builtins, comparisons and
        # a minimal subclass.
        for sample in [set, frozenset]:
            self.assertIsInstance(sample(), Set)
            self.assertTrue(issubclass(sample, Set))
        self.validate_abstract_methods(Set, '__contains__', '__iter__', '__len__')
        class MySet(Set):
            # [CLASS DOC] HELPER: Minimal Set implementation.
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return False
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter([])
        self.validate_comparison(MySet())

    def test_hash_Set(self):
        # [FUNCTION DOC] Checks Set hash mixin produces equal hashes for equal sets and a custom
        # hash override is used.
        class OneTwoThreeSet(Set):
            # [CLASS DOC] HELPER: Set with fixed contents and overridden hash.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes the instance.
                self.contents = [1, 2, 3]
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return x in self.contents
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self.contents)
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.contents)
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash stub returning a fixed value.
                return self._hash()
        a, b = OneTwoThreeSet(), OneTwoThreeSet()
        self.assertTrue(hash(a) == hash(b))

    def test_isdisjoint_Set(self):
        # [FUNCTION DOC] Checks the isdisjoint mixin with disjoint, overlapping
        # and non-set arguments.
        class MySet(Set):
            # [CLASS DOC] HELPER: Set backed by an iterable.
            def __init__(self, itr):
                # [FUNCTION DOC] HELPER: Stores the items.
                self.contents = itr
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return x in self.contents
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.contents)
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len([x for x in self.contents])
        s1 = MySet((1, 2, 3))
        s2 = MySet((4, 5, 6))
        s3 = MySet((1, 5, 6))
        self.assertTrue(s1.isdisjoint(s2))
        self.assertFalse(s1.isdisjoint(s3))

    def test_equality_Set(self):
        # [FUNCTION DOC] Checks the equality mixin between custom sets and builtin sets,
        # including against non-sets.
        class MySet(Set):
            # [CLASS DOC] HELPER: Set backed by an iterable.
            def __init__(self, itr):
                # [FUNCTION DOC] HELPER: Stores the items.
                self.contents = itr
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return x in self.contents
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.contents)
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len([x for x in self.contents])
        s1 = MySet((1,))
        s2 = MySet((1, 2))
        s3 = MySet((3, 4))
        s4 = MySet((3, 4))
        self.assertTrue(s2 > s1)
        self.assertTrue(s1 < s2)
        self.assertFalse(s2 <= s1)
        self.assertFalse(s2 <= s3)
        self.assertFalse(s1 >= s2)
        self.assertEqual(s3, s4)
        self.assertNotEqual(s2, s3)

    def test_arithmetic_Set(self):
        # [FUNCTION DOC] Checks arithmetic operators (and, or, sub, xor) on custom
        # sets with various operands, including non-set operands raising TypeError
        # or returning NotImplemented.
        class MySet(Set):
            # [CLASS DOC] HELPER: Set backed by an iterable.
            def __init__(self, itr):
                # [FUNCTION DOC] HELPER: Stores the items.
                self.contents = itr
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return x in self.contents
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.contents)
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len([x for x in self.contents])
        s1 = MySet((1, 2, 3))
        s2 = MySet((3, 4, 5))
        s3 = s1 & s2
        self.assertEqual(s3, MySet((3,)))

    def test_MutableSet(self):
        # [FUNCTION DOC] Checks MutableSet abstract methods and mixins on the
        # concrete helper class.
        self.assertIsInstance(set(), MutableSet)
        self.assertTrue(issubclass(set, MutableSet))
        self.assertNotIsInstance(frozenset(), MutableSet)
        self.assertFalse(issubclass(frozenset, MutableSet))
        self.validate_abstract_methods(MutableSet, '__contains__', '__iter__', '__len__',
            'add', 'discard')

    def test_issue_5647(self):
        # [FUNCTION DOC] Regression test for issue 5647: in- place subtraction of a
        # set from itself.
        # MutableSet.__iand__ mutated the set during iteration
        s = WithSet('abcd')
        s &= WithSet('cdef')            # This used to fail
        self.assertEqual(set(s), set('cd'))

    def test_issue_4920(self):
        # [FUNCTION DOC] Regression test for issue 4920: in-place operators on a custom
        # MutableSet must not alter the set during iteration.
        # MutableSet.pop() method did not work
        class MySet(MutableSet):
            # [CLASS DOC] HELPER: MutableSet implementation with a list-like store
            # and recording representation.
            __slots__=['__s']
            def __init__(self,items=None):
                # [FUNCTION DOC] HELPER: Initializes the store from items.
                if items is None:
                    items=[]
                self.__s=set(items)
            def __contains__(self,v):
                # [FUNCTION DOC] HELPER: Containment stub.
                return v in self.__s
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.__s)
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self.__s)
            def add(self,v):
                # [FUNCTION DOC] HELPER: Adds a value if missing.
                result=v not in self.__s
                self.__s.add(v)
                return result
            def discard(self,v):
                # [FUNCTION DOC] HELPER: Removes a value.
                result=v in self.__s
                self.__s.discard(v)
                return result
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Representation showing the contents.
                return "MySet(%s)" % repr(list(self))
        items = [5,43,2,1]
        s = MySet(items)
        r = s.pop()
        self.assertEqual(len(s), len(items) - 1)
        self.assertNotIn(r, s)
        self.assertIn(r, items)

    def test_issue8750(self):
        # [FUNCTION DOC] Regression test for issue 8750: in- place or on a
        # MutableSet with itself.
        empty = WithSet()
        full = WithSet(range(10))
        s = WithSet(full)
        s -= s
        self.assertEqual(s, empty)
        s = WithSet(full)
        s ^= s
        self.assertEqual(s, empty)
        s = WithSet(full)
        s &= s
        self.assertEqual(s, full)
        s |= s
        self.assertEqual(s, full)

    def test_issue16373(self):
        # [FUNCTION DOC] Regression test for issue 16373: comparisons between sets where the
        # other side returns NotImplemented or defines its own ordering.
        # Recursion error comparing comparable and noncomparable
        # Set instances
        class MyComparableSet(Set):
            # [CLASS DOC] HELPER: Comparable Set implementation.
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return False
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter([])
        class MyNonComparableSet(Set):
            # [CLASS DOC] HELPER: Set whose ordering comparisons return NotImplemented.
            def __contains__(self, x):
                # [FUNCTION DOC] HELPER: Containment stub.
                return False
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter([])
            def __le__(self, x):
                # [FUNCTION DOC] HELPER: Less-or-equal stub returning NotImplemented.
                return NotImplemented
            def __lt__(self, x):
                # [FUNCTION DOC] HELPER: Less-than stub returning NotImplemented.
                return NotImplemented

        cs = MyComparableSet()
        ncs = MyNonComparableSet()
        self.assertFalse(ncs < cs)
        self.assertTrue(ncs <= cs)
        self.assertFalse(ncs > cs)
        self.assertTrue(ncs >= cs)

    def test_issue26915(self):
        # [FUNCTION DOC] Regression test for issue 26915: the contains, index and count mixins
        # of Sequence use identity or equality consistently.
        # Container membership test should check identity first
        class CustomSequence(Sequence):
            # [CLASS DOC] HELPER: Sequence implementation wrapping a list.
            def __init__(self, seq):
                # [FUNCTION DOC] HELPER: Stores the sequence.
                self._seq = seq
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Index access stub.
                return self._seq[index]
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self._seq)

        nan = float('nan')
        obj = support.NEVER_EQ
        seq = CustomSequence([nan, obj, nan])
        containers = [
            seq,
            ItemsView({1: nan, 2: obj}),
            KeysView({1: nan, 2: obj}),
            ValuesView({1: nan, 2: obj})
        ]
        for container in containers:
            for elem in container:
                self.assertIn(elem, container)
        self.assertEqual(seq.index(nan), 0)
        self.assertEqual(seq.index(obj), 1)
        self.assertEqual(seq.count(nan), 2)
        self.assertEqual(seq.count(obj), 1)

    def assertSameSet(self, s1, s2):
        # [FUNCTION DOC] HELPER: Asserts two sets have equal contents using exact
        # type- independent comparison.
        # coerce both to a real set then check equality
        self.assertSetEqual(set(s1), set(s2))

    def test_Set_from_iterable(self):
        # [FUNCTION DOC] Checks set operations on MutableSet create results via the from-
        # iterable hook of the instance, using a subclass that records its creator.
        """Verify _from_iterable overridden to an instance method works."""
        class SetUsingInstanceFromIterable(MutableSet):
            # [CLASS DOC] HELPER: MutableSet whose from-iterable hook is an instance method
            # recording who created each result.
            def __init__(self, values, created_by):
                # [FUNCTION DOC] HELPER: Stores values and the creator tag.
                if not created_by:
                    raise ValueError('created_by must be specified')
                self.created_by = created_by
                self._values = set(values)

            def _from_iterable(self, values):
                # [FUNCTION DOC] HELPER: Builds a new instance tagged with the creator
                # of the receiver.
                return type(self)(values, 'from_iterable')

            def __contains__(self, value):
                # [FUNCTION DOC] HELPER: Containment stub.
                return value in self._values

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                yield from self._values

            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self._values)

            def add(self, value):
                # [FUNCTION DOC] HELPER: Adds a value.
                self._values.add(value)

            def discard(self, value):
                # [FUNCTION DOC] HELPER: Removes a value.
                self._values.discard(value)

        impl = SetUsingInstanceFromIterable([1, 2, 3], 'test')

        actual = impl - {1}
        self.assertIsInstance(actual, SetUsingInstanceFromIterable)
        self.assertEqual('from_iterable', actual.created_by)
        self.assertEqual({2, 3}, actual)

        actual = impl | {4}
        self.assertIsInstance(actual, SetUsingInstanceFromIterable)
        self.assertEqual('from_iterable', actual.created_by)
        self.assertEqual({1, 2, 3, 4}, actual)

        actual = impl & {2}
        self.assertIsInstance(actual, SetUsingInstanceFromIterable)
        self.assertEqual('from_iterable', actual.created_by)
        self.assertEqual({2}, actual)

        actual = impl ^ {3, 4}
        self.assertIsInstance(actual, SetUsingInstanceFromIterable)
        self.assertEqual('from_iterable', actual.created_by)
        self.assertEqual({1, 2, 4}, actual)

        # NOTE: ixor'ing with a list is important here: internally, __ixor__
        # only calls _from_iterable if the other value isn't already a Set.
        impl ^= [3, 4]
        self.assertIsInstance(impl, SetUsingInstanceFromIterable)
        self.assertEqual('test', impl.created_by)
        self.assertEqual({1, 2, 4}, impl)

    def test_Set_interoperability_with_real_sets(self):
        # [FUNCTION DOC] Checks Set interoperability with builtin sets across all operators and
        # comparisons, using a list-based Set implementation, including reflected operations.
        # Issue: 8743
        class ListSet(Set):
            # [CLASS DOC] HELPER: Set implementation backed by a list without duplicates.
            def __init__(self, elements=()):
                # [FUNCTION DOC] HELPER: Fills the list with unique elements.
                self.data = []
                for elem in elements:
                    if elem not in self.data:
                        self.data.append(elem)
            def __contains__(self, elem):
                # [FUNCTION DOC] HELPER: Containment stub.
                return elem in self.data
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(self.data)
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self.data)
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Representation stub.
                return 'Set({!r})'.format(self.data)

        r1 = set('abc')
        r2 = set('bcd')
        r3 = set('abcde')
        f1 = ListSet('abc')
        f2 = ListSet('bcd')
        f3 = ListSet('abcde')
        l1 = list('abccba')
        l2 = list('bcddcb')
        l3 = list('abcdeedcba')

        target = r1 & r2
        self.assertSameSet(f1 & f2, target)
        self.assertSameSet(f1 & r2, target)
        self.assertSameSet(r2 & f1, target)
        self.assertSameSet(f1 & l2, target)

        target = r1 | r2
        self.assertSameSet(f1 | f2, target)
        self.assertSameSet(f1 | r2, target)
        self.assertSameSet(r2 | f1, target)
        self.assertSameSet(f1 | l2, target)

        fwd_target = r1 - r2
        rev_target = r2 - r1
        self.assertSameSet(f1 - f2, fwd_target)
        self.assertSameSet(f2 - f1, rev_target)
        self.assertSameSet(f1 - r2, fwd_target)
        self.assertSameSet(f2 - r1, rev_target)
        self.assertSameSet(r1 - f2, fwd_target)
        self.assertSameSet(r2 - f1, rev_target)
        self.assertSameSet(f1 - l2, fwd_target)
        self.assertSameSet(f2 - l1, rev_target)

        target = r1 ^ r2
        self.assertSameSet(f1 ^ f2, target)
        self.assertSameSet(f1 ^ r2, target)
        self.assertSameSet(r2 ^ f1, target)
        self.assertSameSet(f1 ^ l2, target)

        # Don't change the following to use assertLess or other
        # "more specific" unittest assertions.  The current
        # assertTrue/assertFalse style makes the pattern of test
        # case combinations clear and allows us to know for sure
        # the exact operator being invoked.

        # proper subset
        self.assertTrue(f1 < f3)
        self.assertFalse(f1 < f1)
        self.assertFalse(f1 < f2)
        self.assertTrue(r1 < f3)
        self.assertFalse(r1 < f1)
        self.assertFalse(r1 < f2)
        self.assertTrue(r1 < r3)
        self.assertFalse(r1 < r1)
        self.assertFalse(r1 < r2)
        with self.assertRaises(TypeError):
            f1 < l3
        with self.assertRaises(TypeError):
            f1 < l1
        with self.assertRaises(TypeError):
            f1 < l2

        # any subset
        self.assertTrue(f1 <= f3)
        self.assertTrue(f1 <= f1)
        self.assertFalse(f1 <= f2)
        self.assertTrue(r1 <= f3)
        self.assertTrue(r1 <= f1)
        self.assertFalse(r1 <= f2)
        self.assertTrue(r1 <= r3)
        self.assertTrue(r1 <= r1)
        self.assertFalse(r1 <= r2)
        with self.assertRaises(TypeError):
            f1 <= l3
        with self.assertRaises(TypeError):
            f1 <= l1
        with self.assertRaises(TypeError):
            f1 <= l2

        # proper superset
        self.assertTrue(f3 > f1)
        self.assertFalse(f1 > f1)
        self.assertFalse(f2 > f1)
        self.assertTrue(r3 > r1)
        self.assertFalse(f1 > r1)
        self.assertFalse(f2 > r1)
        self.assertTrue(r3 > r1)
        self.assertFalse(r1 > r1)
        self.assertFalse(r2 > r1)
        with self.assertRaises(TypeError):
            f1 > l3
        with self.assertRaises(TypeError):
            f1 > l1
        with self.assertRaises(TypeError):
            f1 > l2

        # any superset
        self.assertTrue(f3 >= f1)
        self.assertTrue(f1 >= f1)
        self.assertFalse(f2 >= f1)
        self.assertTrue(r3 >= r1)
        self.assertTrue(f1 >= r1)
        self.assertFalse(f2 >= r1)
        self.assertTrue(r3 >= r1)
        self.assertTrue(r1 >= r1)
        self.assertFalse(r2 >= r1)
        with self.assertRaises(TypeError):
            f1 >= l3
        with self.assertRaises(TypeError):
            f1 >=l1
        with self.assertRaises(TypeError):
            f1 >= l2

        # equality
        self.assertTrue(f1 == f1)
        self.assertTrue(r1 == f1)
        self.assertTrue(f1 == r1)
        self.assertFalse(f1 == f3)
        self.assertFalse(r1 == f3)
        self.assertFalse(f1 == r3)
        self.assertFalse(f1 == l3)
        self.assertFalse(f1 == l1)
        self.assertFalse(f1 == l2)

        # inequality
        self.assertFalse(f1 != f1)
        self.assertFalse(r1 != f1)
        self.assertFalse(f1 != r1)
        self.assertTrue(f1 != f3)
        self.assertTrue(r1 != f3)
        self.assertTrue(f1 != r3)
        self.assertTrue(f1 != l3)
        self.assertTrue(f1 != l1)
        self.assertTrue(f1 != l2)

    def test_Set_hash_matches_frozenset(self):
        # [FUNCTION DOC] Checks the hash of a Set mixin matches the hash of a frozenset of
        # the same elements.
        sets = [
            {}, {1}, {None}, {-1}, {0.0}, {"abc"}, {1, 2, 3},
            {10**100, 10**101}, {"a", "b", "ab", ""}, {False, True},
            {object(), object(), object()}, {float("nan")},  {frozenset()},
            {*range(1000)}, {*range(1000)} - {100, 200, 300},
            {*range(sys.maxsize - 10, sys.maxsize + 10)},
        ]
        for s in sets:
            fs = frozenset(s)
            self.assertEqual(hash(fs), Set._hash(fs), msg=s)

    def test_Mapping(self):
        # [FUNCTION DOC] Checks Mapping abstract methods and a minimal mapping subclass.
        for sample in [dict]:
            self.assertIsInstance(sample(), Mapping)
            self.assertTrue(issubclass(sample, Mapping))
        self.validate_abstract_methods(Mapping, '__contains__', '__iter__', '__len__',
            '__getitem__')
        class MyMapping(Mapping):
            # [CLASS DOC] HELPER: Minimal Mapping implementation.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return 0
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Item lookup stub.
                raise IndexError
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Iteration stub.
                return iter(())
        self.validate_comparison(MyMapping())
        self.assertRaises(TypeError, reversed, MyMapping())

    def test_MutableMapping(self):
        # [FUNCTION DOC] Checks MutableMapping abstract methods and builtin dict as an instance.
        for sample in [dict]:
            self.assertIsInstance(sample(), MutableMapping)
            self.assertTrue(issubclass(sample, MutableMapping))
        self.validate_abstract_methods(MutableMapping, '__iter__', '__len__',
            '__getitem__', '__setitem__', '__delitem__')

    def test_MutableMapping_subclass(self):
        # [FUNCTION DOC] Checks the mixin methods of a MutableMapping subclass: setdefault, pop,
        # popitem, update, clear.
        # Test issue 9214
        mymap = UserDict()
        mymap['red'] = 5
        self.assertIsInstance(mymap.keys(), Set)
        self.assertIsInstance(mymap.keys(), KeysView)
        self.assertIsInstance(mymap.values(), Collection)
        self.assertIsInstance(mymap.values(), ValuesView)
        self.assertIsInstance(mymap.items(), Set)
        self.assertIsInstance(mymap.items(), ItemsView)

        mymap = UserDict()
        mymap['red'] = 5
        z = mymap.keys() | {'orange'}
        self.assertIsInstance(z, set)
        list(z)
        mymap['blue'] = 7               # Shouldn't affect 'z'
        self.assertEqual(sorted(z), ['orange', 'red'])

        mymap = UserDict()
        mymap['red'] = 5
        z = mymap.items() | {('orange', 3)}
        self.assertIsInstance(z, set)
        list(z)
        mymap['blue'] = 7               # Shouldn't affect 'z'
        self.assertEqual(z, {('orange', 3), ('red', 5)})

    def test_Sequence(self):
        # [FUNCTION DOC] Checks Sequence abstract methods and isinstance of builtins.
        for sample in [tuple, list, bytes, str]:
            self.assertIsInstance(sample(), Sequence)
            self.assertTrue(issubclass(sample, Sequence))
        self.assertIsInstance(range(10), Sequence)
        self.assertTrue(issubclass(range, Sequence))
        self.assertIsInstance(memoryview(b""), Sequence)
        self.assertTrue(issubclass(memoryview, Sequence))
        self.assertTrue(issubclass(str, Sequence))
        self.validate_abstract_methods(Sequence, '__contains__', '__iter__', '__len__',
            '__getitem__')

    def test_Sequence_mixins(self):
        # [FUNCTION DOC] Checks Sequence mixins (contains, iter, reversed, index, count) on a
        # subclass, with a helper comparing index results across arguments.
        class SequenceSubclass(Sequence):
            # [CLASS DOC] HELPER: Sequence implementation wrapping a list.
            def __init__(self, seq=()):
                # [FUNCTION DOC] HELPER: Stores the sequence.
                self.seq = seq

            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Index access stub.
                return self.seq[index]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self.seq)

        # Compare Sequence.index() behavior to (list|str).index() behavior
        def assert_index_same(seq1, seq2, index_args):
            # [FUNCTION DOC] HELPER: Asserts that index on both sequences gives the same result
            # or raises the same error for the given arguments.
            try:
                expected = seq1.index(*index_args)
            except ValueError:
                with self.assertRaises(ValueError):
                    seq2.index(*index_args)
            else:
                actual = seq2.index(*index_args)
                self.assertEqual(
                    actual, expected, '%r.index%s' % (seq1, index_args))

        for ty in list, str:
            nativeseq = ty('abracadabra')
            indexes = [-10000, -9999] + list(range(-3, len(nativeseq) + 3))
            seqseq = SequenceSubclass(nativeseq)
            for letter in set(nativeseq) | {'z'}:
                assert_index_same(nativeseq, seqseq, (letter,))
                for start in range(-3, len(nativeseq) + 3):
                    assert_index_same(nativeseq, seqseq, (letter, start))
                    for stop in range(-3, len(nativeseq) + 3):
                        assert_index_same(
                            nativeseq, seqseq, (letter, start, stop))

    def test_ByteString(self):
        # [FUNCTION DOC] Checks ByteString classification of bytes types and that subclassing
        # ByteString emits a deprecation warning.
        for sample in [bytes, bytearray]:
            with self.assertWarns(DeprecationWarning):
                self.assertIsInstance(sample(), ByteString)
            self.assertTrue(issubclass(sample, ByteString))
        for sample in [str, list, tuple]:
            with self.assertWarns(DeprecationWarning):
                self.assertNotIsInstance(sample(), ByteString)
            self.assertFalse(issubclass(sample, ByteString))
        with self.assertWarns(DeprecationWarning):
            self.assertNotIsInstance(memoryview(b""), ByteString)
        self.assertFalse(issubclass(memoryview, ByteString))
        with self.assertWarns(DeprecationWarning):
            self.validate_abstract_methods(ByteString, '__getitem__', '__len__')

        with self.assertWarns(DeprecationWarning):
            # [CLASS DOC] HELPER: Subclass of ByteString used to check the deprecation warning.
            class X(ByteString): pass

        with self.assertWarns(DeprecationWarning):
            # No metaclass conflict
            # [CLASS DOC] HELPER: Subclass mixing ByteString with another ABC, same purpose.
            class Z(ByteString, Awaitable): pass

    def test_Buffer(self):
        # [FUNCTION DOC] Checks Buffer recognition for bytes-like objects and the buffer method.
        for sample in [bytes, bytearray, memoryview]:
            self.assertIsInstance(sample(b"x"), Buffer)
            self.assertTrue(issubclass(sample, Buffer))
        for sample in [str, list, tuple]:
            self.assertNotIsInstance(sample(), Buffer)
            self.assertFalse(issubclass(sample, Buffer))
        self.validate_abstract_methods(Buffer, '__buffer__')

    def test_MutableSequence(self):
        # [FUNCTION DOC] Checks MutableSequence abstract methods and isinstance of builtins.
        for sample in [tuple, str, bytes]:
            self.assertNotIsInstance(sample(), MutableSequence)
            self.assertFalse(issubclass(sample, MutableSequence))
        for sample in [list, bytearray, deque]:
            self.assertIsInstance(sample(), MutableSequence)
            self.assertTrue(issubclass(sample, MutableSequence))
        self.assertTrue(issubclass(array.array, MutableSequence))
        self.assertFalse(issubclass(str, MutableSequence))
        self.validate_abstract_methods(MutableSequence, '__contains__', '__iter__',
            '__len__', '__getitem__', '__setitem__', '__delitem__', 'insert')

    def test_MutableSequence_mixins(self):
        # [FUNCTION DOC] Checks MutableSequence mixins (append, reverse, extend, pop, remove,
        # in-place add) on a list-backed subclass.
        # Test the mixins of MutableSequence by creating a minimal concrete
        # class inherited from it.
        class MutableSequenceSubclass(MutableSequence):
            # [CLASS DOC] HELPER: MutableSequence implementation wrapping a list.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initializes an empty backing list.
                self.lst = []

            def __setitem__(self, index, value):
                # [FUNCTION DOC] HELPER: Item assignment stub.
                self.lst[index] = value

            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Item access stub.
                return self.lst[index]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Length stub.
                return len(self.lst)

            def __delitem__(self, index):
                # [FUNCTION DOC] HELPER: Item deletion stub.
                del self.lst[index]

            def insert(self, index, value):
                # [FUNCTION DOC] HELPER: Insert stub.
                self.lst.insert(index, value)

        mss = MutableSequenceSubclass()
        mss.append(0)
        mss.extend((1, 2, 3, 4))
        self.assertEqual(len(mss), 5)
        self.assertEqual(mss[3], 3)
        mss.reverse()
        self.assertEqual(mss[3], 1)
        mss.pop()
        self.assertEqual(len(mss), 4)
        mss.remove(3)
        self.assertEqual(len(mss), 3)
        mss += (10, 20, 30)
        self.assertEqual(len(mss), 6)
        self.assertEqual(mss[-1], 30)
        mss.clear()
        self.assertEqual(len(mss), 0)

        # issue 34427
        # extending self should not cause infinite loop
        items = 'ABCD'
        mss2 = MutableSequenceSubclass()
        mss2.extend(items + items)
        mss.clear()
        mss.extend(items)
        mss.extend(mss)
        self.assertEqual(len(mss), len(mss2))
        self.assertEqual(list(mss), list(mss2))

    def test_illegal_patma_flags(self):
        # [FUNCTION DOC] Checks that creating a Collection subclass with both sequence and
        # mapping pattern-matching flags raises TypeError.
        with self.assertRaises(TypeError):
            class Both(Collection):
                # [CLASS DOC] HELPER: Collection subclass declared with conflicting match flags.
                __abc_tpflags__ = (Sequence.__flags__ | Mapping.__flags__)



################################################################################
### Counter
################################################################################

class CounterSubclassWithSetItem(Counter):
    # [CLASS DOC] HELPER: Counter subclass overriding item assignment to record that
    # it was called.
    # Test a counter subclass that overrides __setitem__
    def __init__(self, *args, **kwds):
        # [FUNCTION DOC] HELPER: Initializes the counter and a record of assignments.
        self.called = False
        Counter.__init__(self, *args, **kwds)
    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Records the key and delegates to the base assignment.
        self.called = True
        Counter.__setitem__(self, key, value)

class CounterSubclassWithGet(Counter):
    # [CLASS DOC] HELPER: Counter subclass overriding get to record that it was called.
    # Test a counter subclass that overrides get()
    def __init__(self, *args, **kwds):
        # [FUNCTION DOC] HELPER: Initializes the counter and a record of lookups.
        self.called = False
        Counter.__init__(self, *args, **kwds)
    def get(self, key, default):
        # [FUNCTION DOC] HELPER: Records the key and delegates to the base lookup.
        self.called = True
        return Counter.get(self, key, default)

class TestCounter(__TestCase):
    # [CLASS DOC] Tests collections.Counter: construction, counting, ordering, update,
    # copy, conversions, multiset operators, in-place operators, subtract, unary operators,
    # repr and comparisons.

    def test_basics(self):
        # [FUNCTION DOC] Checks basic counting: construction from strings/mappings, missing keys
        # as zero, elements, most_common, deletion and the equality/total behavior.
        c = Counter('abcaba')
        self.assertEqual(c, Counter({'a':3 , 'b': 2, 'c': 1}))
        self.assertEqual(c, Counter(a=3, b=2, c=1))
        self.assertIsInstance(c, dict)
        self.assertIsInstance(c, Mapping)
        self.assertTrue(issubclass(Counter, dict))
        self.assertTrue(issubclass(Counter, Mapping))
        self.assertEqual(len(c), 3)
        self.assertEqual(sum(c.values()), 6)
        self.assertEqual(list(c.values()), [3, 2, 1])
        self.assertEqual(list(c.keys()), ['a', 'b', 'c'])
        self.assertEqual(list(c), ['a', 'b', 'c'])
        self.assertEqual(list(c.items()),
                         [('a', 3), ('b', 2), ('c', 1)])
        self.assertEqual(c['b'], 2)
        self.assertEqual(c['z'], 0)
        self.assertEqual(c.__contains__('c'), True)
        self.assertEqual(c.__contains__('z'), False)
        self.assertEqual(c.get('b', 10), 2)
        self.assertEqual(c.get('z', 10), 10)
        self.assertEqual(c, dict(a=3, b=2, c=1))
        self.assertEqual(repr(c), "Counter({'a': 3, 'b': 2, 'c': 1})")
        self.assertEqual(c.most_common(), [('a', 3), ('b', 2), ('c', 1)])
        for i in range(5):
            self.assertEqual(c.most_common(i),
                             [('a', 3), ('b', 2), ('c', 1)][:i])
        self.assertEqual(''.join(c.elements()), 'aaabbc')
        c['a'] += 1         # increment an existing value
        c['b'] -= 2         # sub existing value to zero
        del c['c']          # remove an entry
        del c['c']          # make sure that del doesn't raise KeyError
        c['d'] -= 2         # sub from a missing value
        c['e'] = -5         # directly assign a missing value
        c['f'] += 4         # add to a missing value
        self.assertEqual(c, dict(a=4, b=0, d=-2, e=-5, f=4))
        self.assertEqual(''.join(c.elements()), 'aaaaffff')
        self.assertEqual(c.pop('f'), 4)
        self.assertNotIn('f', c)
        for i in range(3):
            elem, cnt = c.popitem()
            self.assertNotIn(elem, c)
        c.clear()
        self.assertEqual(c, {})
        self.assertEqual(repr(c), 'Counter()')
        self.assertRaises(NotImplementedError, Counter.fromkeys, 'abc')
        self.assertRaises(TypeError, hash, c)
        c.update(dict(a=5, b=3))
        c.update(c=1)
        c.update(Counter('a' * 50 + 'b' * 30))
        c.update()          # test case with no args
        c.__init__('a' * 500 + 'b' * 300)
        c.__init__('cdc')
        c.__init__()
        self.assertEqual(c, dict(a=555, b=333, c=3, d=1))
        self.assertEqual(c.setdefault('d', 5), 1)
        self.assertEqual(c['d'], 1)
        self.assertEqual(c.setdefault('e', 5), 5)
        self.assertEqual(c['e'], 5)

    def test_init(self):
        # [FUNCTION DOC] Checks init with iterable, mapping and keyword arguments and error
        # cases for bad arguments.
        self.assertEqual(list(Counter(self=42).items()), [('self', 42)])
        self.assertEqual(list(Counter(iterable=42).items()), [('iterable', 42)])
        self.assertEqual(list(Counter(iterable=None).items()), [('iterable', None)])
        self.assertRaises(TypeError, Counter, 42)
        self.assertRaises(TypeError, Counter, (), ())
        self.assertRaises(TypeError, Counter.__init__)

    def test_total(self):
        # [FUNCTION DOC] Checks total sums counts and tolerates negative counts.
        c = Counter(a=10, b=5, c=0)
        self.assertEqual(c.total(), 15)

    def test_order_preservation(self):
        # [FUNCTION DOC] Checks insertion order of elements and most_common tie ordering follow
        # first occurrence, comparing to a reference ordering.
        # Input order dictates items() order
        self.assertEqual(list(Counter('abracadabra').items()),
               [('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1)])
        # letters with same count:   ^----------^         ^---------^

        # Verify retention of order even when all counts are equal
        self.assertEqual(list(Counter('xyzpdqqdpzyx').items()),
               [('x', 2), ('y', 2), ('z', 2), ('p', 2), ('d', 2), ('q', 2)])

        # Input order dictates elements() order
        self.assertEqual(list(Counter('abracadabra simsalabim').elements()),
                ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b','r',
                 'r', 'c', 'd', ' ', 's', 's', 'i', 'i', 'm', 'm', 'l'])

        # Math operations order first by the order encountered in the left
        # operand and then by the order encountered in the right operand.
        ps = 'aaabbcdddeefggghhijjjkkl'
        qs = 'abbcccdeefffhkkllllmmnno'
        order = {letter: i for i, letter in enumerate(dict.fromkeys(ps + qs))}
        def correctly_ordered(seq):
            # [FUNCTION DOC] HELPER: Checks that a sequence lists elements in non-
            # increasing count order.
            'Return true if the letters occur in the expected order'
            positions = [order[letter] for letter in seq]
            return positions == sorted(positions)

        p, q = Counter(ps), Counter(qs)
        self.assertTrue(correctly_ordered(+p))
        self.assertTrue(correctly_ordered(-p))
        self.assertTrue(correctly_ordered(p + q))
        self.assertTrue(correctly_ordered(p - q))
        self.assertTrue(correctly_ordered(p | q))
        self.assertTrue(correctly_ordered(p & q))

        p, q = Counter(ps), Counter(qs)
        p += q
        self.assertTrue(correctly_ordered(p))

        p, q = Counter(ps), Counter(qs)
        p -= q
        self.assertTrue(correctly_ordered(p))

        p, q = Counter(ps), Counter(qs)
        p |= q
        self.assertTrue(correctly_ordered(p))

        p, q = Counter(ps), Counter(qs)
        p &= q
        self.assertTrue(correctly_ordered(p))

        p, q = Counter(ps), Counter(qs)
        p.update(q)
        self.assertTrue(correctly_ordered(p))

        p, q = Counter(ps), Counter(qs)
        p.subtract(q)
        self.assertTrue(correctly_ordered(p))

    def test_update(self):
        # [FUNCTION DOC] Checks update with iterables and mappings, keywords, and errors for bad
        # arguments; subclass overrides are used.
        c = Counter()
        c.update(self=42)
        self.assertEqual(list(c.items()), [('self', 42)])
        c = Counter()
        c.update(iterable=42)
        self.assertEqual(list(c.items()), [('iterable', 42)])
        c = Counter()
        c.update(iterable=None)
        self.assertEqual(list(c.items()), [('iterable', None)])
        self.assertRaises(TypeError, Counter().update, 42)
        self.assertRaises(TypeError, Counter().update, {}, {})
        self.assertRaises(TypeError, Counter.update)

    def test_copying(self):
        # [FUNCTION DOC] Checks copy produces an equal independent counter.
        # Check that counters are copyable, deepcopyable, picklable, and
        #have a repr/eval round-trip
        words = Counter('which witch had which witches wrist watch'.split())
        def check(dup):
            # [FUNCTION DOC] HELPER: Asserts that a copy is a different object
            # with equal contents.
            msg = "\ncopy: %s\nwords: %s" % (dup, words)
            self.assertIsNot(dup, words, msg)
            self.assertEqual(dup, words)
        check(words.copy())
        check(copy.copy(words))
        check(copy.deepcopy(words))
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            with self.subTest(proto=proto):
                check(pickle.loads(pickle.dumps(words, proto)))
        check(eval(repr(words)))
        update_test = Counter()
        update_test.update(words)
        check(update_test)
        check(Counter(words))

    def test_copy_subclass(self):
        # [FUNCTION DOC] Checks copying a Counter subclass keeps the subclass type
        # and extra attributes.
        class MyCounter(Counter):
            # [CLASS DOC] HELPER: Counter subclass for the copy test.
            pass
        c = MyCounter('slartibartfast')
        d = c.copy()
        self.assertEqual(d, c)
        self.assertEqual(len(d), len(c))
        self.assertEqual(type(d), type(c))

    def test_conversions(self):
        # [FUNCTION DOC] Checks conversion to list, set, dict and construction back.
        # Convert to: set, list, dict
        s = 'she sells sea shells by the sea shore'
        self.assertEqual(sorted(Counter(s).elements()), sorted(s))
        self.assertEqual(sorted(Counter(s)), sorted(set(s)))
        self.assertEqual(dict(Counter(s)), dict(Counter(s).items()))
        self.assertEqual(set(Counter(s)), set(s))

    def test_invariant_for_the_in_operator(self):
        # [FUNCTION DOC] Checks the in operator reports membership only for present keys,
        # ignoring zero counts.
        c = Counter(a=10, b=-2, c=0)
        for elem in c:
            self.assertTrue(elem in c)
            self.assertIn(elem, c)

    def test_multiset_operations(self):
        # [FUNCTION DOC] Checks multiset operators add, subtract, union and intersection keep
        # only positive counts, return Counter, and raise TypeError for non-Counter operands.
        # Verify that adding a zero counter will strip zeros and negatives
        c = Counter(a=10, b=-2, c=0) + Counter()
        self.assertEqual(dict(c), dict(a=10))

        elements = 'abcd'
        for i in range(1000):
            # test random pairs of multisets
            p = Counter(dict((elem, randrange(-2,4)) for elem in elements))
            p.update(e=1, f=-1, g=0)
            q = Counter(dict((elem, randrange(-2,4)) for elem in elements))
            q.update(h=1, i=-1, j=0)
            for counterop, numberop in [
                (Counter.__add__, lambda x, y: max(0, x+y)),
                (Counter.__sub__, lambda x, y: max(0, x-y)),
                (Counter.__or__, lambda x, y: max(0,x,y)),
                (Counter.__and__, lambda x, y: max(0, min(x,y))),
            ]:
                result = counterop(p, q)
                for x in elements:
                    self.assertEqual(numberop(p[x], q[x]), result[x],
                                     (counterop, x, p, q))
                # verify that results exclude non-positive counts
                self.assertTrue(x>0 for x in result.values())

        elements = 'abcdef'
        for i in range(100):
            # verify that random multisets with no repeats are exactly like sets
            p = Counter(dict((elem, randrange(0, 2)) for elem in elements))
            q = Counter(dict((elem, randrange(0, 2)) for elem in elements))
            for counterop, setop in [
                (Counter.__sub__, set.__sub__),
                (Counter.__or__, set.__or__),
                (Counter.__and__, set.__and__),
            ]:
                counter_result = counterop(p, q)
                set_result = setop(set(p.elements()), set(q.elements()))
                self.assertEqual(counter_result, dict.fromkeys(set_result, 1))

    def test_inplace_operations(self):
        # [FUNCTION DOC] Checks in-place add, subtract, union and intersection keep positive
        # counts and update the left operand.
        elements = 'abcd'
        for i in range(1000):
            # test random pairs of multisets
            p = Counter(dict((elem, randrange(-2,4)) for elem in elements))
            p.update(e=1, f=-1, g=0)
            q = Counter(dict((elem, randrange(-2,4)) for elem in elements))
            q.update(h=1, i=-1, j=0)
            for inplace_op, regular_op in [
                (Counter.__iadd__, Counter.__add__),
                (Counter.__isub__, Counter.__sub__),
                (Counter.__ior__, Counter.__or__),
                (Counter.__iand__, Counter.__and__),
            ]:
                c = p.copy()
                c_id = id(c)
                regular_result = regular_op(c, q)
                inplace_result = inplace_op(c, q)
                self.assertEqual(inplace_result, regular_result)
                self.assertEqual(id(inplace_result), c_id)

    def test_subtract(self):
        # [FUNCTION DOC] Checks subtract with iterables, mappings and keywords,
        # leaving negative counts.
        c = Counter(a=-5, b=0, c=5, d=10, e=15,g=40)
        c.subtract(a=1, b=2, c=-3, d=10, e=20, f=30, h=-50)
        self.assertEqual(c, Counter(a=-6, b=-2, c=8, d=0, e=-5, f=-30, g=40, h=50))
        c = Counter(a=-5, b=0, c=5, d=10, e=15,g=40)
        c.subtract(Counter(a=1, b=2, c=-3, d=10, e=20, f=30, h=-50))
        self.assertEqual(c, Counter(a=-6, b=-2, c=8, d=0, e=-5, f=-30, g=40, h=50))
        c = Counter('aaabbcd')
        c.subtract('aaaabbcce')
        self.assertEqual(c, Counter(a=-1, b=0, c=-1, d=1, e=-1))

        c = Counter()
        c.subtract(self=42)
        self.assertEqual(list(c.items()), [('self', -42)])
        c = Counter()
        c.subtract(iterable=42)
        self.assertEqual(list(c.items()), [('iterable', -42)])
        self.assertRaises(TypeError, Counter().subtract, 42)
        self.assertRaises(TypeError, Counter().subtract, {}, {})
        self.assertRaises(TypeError, Counter.subtract)

    def test_unary(self):
        # [FUNCTION DOC] Checks unary plus keeps positive counts and unary minus negates and
        # keeps positive results.
        c = Counter(a=-5, b=0, c=5, d=10, e=15,g=40)
        self.assertEqual(dict(+c), dict(c=5, d=10, e=15, g=40))
        self.assertEqual(dict(-c), dict(a=5))

    def test_repr_nonsortable(self):
        # [FUNCTION DOC] Checks repr works for counters with keys that cannot be sorted.
        c = Counter(a=2, b=None)
        r = repr(c)
        self.assertIn("'a': 2", r)
        self.assertIn("'b': None", r)

    def test_helper_function(self):
        # [FUNCTION DOC] Checks the internal counting helper on iterables with the default dict-
        # like behavior and with custom mapping subclasses overriding item assignment or get.
        # two paths, one for real dicts and one for other mappings
        elems = list('abracadabra')

        d = dict()
        _count_elements(d, elems)
        self.assertEqual(d, {'a': 5, 'r': 2, 'b': 2, 'c': 1, 'd': 1})

        m = OrderedDict()
        _count_elements(m, elems)
        self.assertEqual(m,
             OrderedDict([('a', 5), ('b', 2), ('r', 2), ('c', 1), ('d', 1)]))

        # test fidelity to the pure python version
        c = CounterSubclassWithSetItem('abracadabra')
        self.assertTrue(c.called)
        self.assertEqual(dict(c), {'a': 5, 'b': 2, 'c': 1, 'd': 1, 'r':2 })
        c = CounterSubclassWithGet('abracadabra')
        self.assertTrue(c.called)
        self.assertEqual(dict(c), {'a': 5, 'b': 2, 'c': 1, 'd': 1, 'r':2 })

    @slowTest
    def test_multiset_operations_equivalent_to_set_operations(self):
        # [FUNCTION DOC] Checks randomized multiset operations agree with set operations when
        # counts are 0 or 1.
        # When the multiplicities are all zero or one, multiset operations
        # are guaranteed to be equivalent to the corresponding operations
        # for regular sets.
        s = list(product(('a', 'b', 'c'), range(2)))
        powerset = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        counters = [Counter(dict(groups)) for groups in powerset]
        for cp, cq in product(counters, repeat=2):
            sp = set(cp.elements())
            sq = set(cq.elements())
            self.assertEqual(set(cp + cq), sp | sq)
            self.assertEqual(set(cp - cq), sp - sq)
            self.assertEqual(set(cp | cq), sp | sq)
            self.assertEqual(set(cp & cq), sp & sq)
            self.assertEqual(cp == cq, sp == sq)
            self.assertEqual(cp != cq, sp != sq)
            self.assertEqual(cp <= cq, sp <= sq)
            self.assertEqual(cp >= cq, sp >= sq)
            self.assertEqual(cp < cq, sp < sq)
            self.assertEqual(cp > cq, sp > sq)

    def test_eq(self):
        # [FUNCTION DOC] Checks equality of counters treats missing and zero counts the same.
        self.assertEqual(Counter(a=3, b=2, c=0), Counter('ababa'))
        self.assertNotEqual(Counter(a=3, b=2), Counter('babab'))

    def test_le(self):
        # [FUNCTION DOC] Checks less-or-equal compares counts element-wise.
        self.assertTrue(Counter(a=3, b=2, c=0) <= Counter('ababa'))
        self.assertFalse(Counter(a=3, b=2) <= Counter('babab'))

    def test_lt(self):
        # [FUNCTION DOC] Checks strict less-than comparison of counters.
        self.assertTrue(Counter(a=3, b=1, c=0) < Counter('ababa'))
        self.assertFalse(Counter(a=3, b=2, c=0) < Counter('ababa'))

    def test_ge(self):
        # [FUNCTION DOC] Checks greater-or-equal comparison of counters.
        self.assertTrue(Counter(a=2, b=1, c=0) >= Counter('aab'))
        self.assertFalse(Counter(a=3, b=2, c=0) >= Counter('aabd'))

    def test_gt(self):
        # [FUNCTION DOC] Checks strict greater-than comparison of counters.
        self.assertTrue(Counter(a=3, b=2, c=0) > Counter('aab'))
        self.assertFalse(Counter(a=2, b=1, c=0) > Counter('aab'))


if __name__ == "__main__":
    run_tests()
