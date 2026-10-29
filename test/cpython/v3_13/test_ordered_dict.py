# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] CPython 3.13 test_ordered_dict adapted to run under Dynamo (CPythonTestCase).
# Tests collections.OrderedDict in its pure-Python and C implementations (plus subclasses, slots
# subclasses, the generic mapping protocol suite, builtin dict reuse of selected tests, and an
# LRU cache built on OrderedDict). Import time side effects: inserts a custom meta path finder
# at the front of sys.meta_path that redirects a fixed list of test.* module names to top-level
# modules of the same name (so the mapping_tests helper etc. resolve), and creates two fresh
# collections module copies py_coll (C module blocked) and c_coll (None if the C module is
# unavailable). Running the file calls run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_ordered_dict.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    xfailIfTorchDynamo,
)

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
    # [CLASS DOC] HELPER: import hook that maps the module names listed in redirect_imports
    # (test.mapping_tests, test.test_math, etc.) to the same-named top-level modules without the
    # test prefix. Stateless; installed once at import time at the front of sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: for a full module name in redirect_imports, import the top
        # level module with the test prefix removed, register it in the module registry under
        # the original name and return its spec; return None for other names or when the import
        # fails (ImportError). Side effect: writes to the module registry.
        # KNOWN ISSUE: relies on importlib.util being available although only importlib.abc is
        # imported explicitly.
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

import builtins
import contextlib
import copy
import gc
import operator
import pickle
import re
from random import randrange, shuffle
import struct
import sys
import unittest
import weakref
from collections.abc import MutableMapping
from test import mapping_tests, support
from test.support import import_helper, suppress_immortalization


py_coll = import_helper.import_fresh_module('collections',
                                            blocked=['_collections'])
c_coll = import_helper.import_fresh_module('collections',
                                           fresh=['_collections'])


@contextlib.contextmanager
def replaced_module(name, replacement):
    # [FUNCTION DOC] HELPER: context manager that temporarily installs the replacement module in
    # the module registry under the given name and restores the original on exit, even if the
    # body raises. Used so pickle finds the right collections implementation.
    original_module = sys.modules[name]
    sys.modules[name] = replacement
    try:
        yield
    finally:
        sys.modules[name] = original_module


class OrderedDictTests:
    # [CLASS DOC] Mixin (not a TestCase) with the behavior tests for OrderedDict. Concrete
    # subclasses set module (the collections module copy under test) and OrderedDict (the class
    # under test). Some tests are also borrowed by the builtin dict suite.

    def test_init(self):
        # [FUNCTION DOC] Check construction from a dict, keyword arguments, pairs and a mix
        # (keywords override pairs but keep first position), keyword names self and other are
        # plain keys, bad argument counts or types raise TypeError, and calling the init method
        # directly on a populated mapping updates it without clearing or reordering existing
        # keys.
        OrderedDict = self.OrderedDict
        with self.assertRaises(TypeError):
            OrderedDict([('a', 1), ('b', 2)], None)                                 # too many args
        pairs = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
        self.assertEqual(sorted(OrderedDict(dict(pairs)).items()), pairs)           # dict input
        self.assertEqual(sorted(OrderedDict(**dict(pairs)).items()), pairs)         # kwds input
        self.assertEqual(list(OrderedDict(pairs).items()), pairs)                   # pairs input
        self.assertEqual(list(OrderedDict([('a', 1), ('b', 2), ('c', 9), ('d', 4)],
                                          c=3, e=5).items()), pairs)                # mixed input

        # make sure no positional args conflict with possible kwdargs
        self.assertEqual(list(OrderedDict(self=42).items()), [('self', 42)])
        self.assertEqual(list(OrderedDict(other=42).items()), [('other', 42)])
        self.assertRaises(TypeError, OrderedDict, 42)
        self.assertRaises(TypeError, OrderedDict, (), ())
        self.assertRaises(TypeError, OrderedDict.__init__)

        # Make sure that direct calls to __init__ do not clear previous contents
        d = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 44), ('e', 55)])
        d.__init__([('e', 5), ('f', 6)], g=7, d=4)
        self.assertEqual(list(d.items()),
            [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)])

    def test_468(self):
        # [FUNCTION DOC] Check that keyword arguments built from an ordered mapping keep
        # insertion order (random shuffled input).
        OrderedDict = self.OrderedDict
        items = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)]
        shuffle(items)
        argdict = OrderedDict(items)
        d = OrderedDict(**argdict)
        self.assertEqual(list(d.items()), items)

    def test_update(self):
        # [FUNCTION DOC] Check update from dict, keywords, pairs and a mix; keywords named self
        # or other are ordinary keys; updating existing keys keeps their position and does not
        # clear the mapping; bad arguments raise TypeError.
        # KNOWN ISSUE: the last three TypeError assertions are repeated twice verbatim
        # (duplicate in upstream).
        OrderedDict = self.OrderedDict
        with self.assertRaises(TypeError):
            OrderedDict().update([('a', 1), ('b', 2)], None)                        # too many args
        pairs = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
        od = OrderedDict()
        od.update(dict(pairs))
        self.assertEqual(sorted(od.items()), pairs)                                 # dict input
        od = OrderedDict()
        od.update(**dict(pairs))
        self.assertEqual(sorted(od.items()), pairs)                                 # kwds input
        od = OrderedDict()
        od.update(pairs)
        self.assertEqual(list(od.items()), pairs)                                   # pairs input
        od = OrderedDict()
        od.update([('a', 1), ('b', 2), ('c', 9), ('d', 4)], c=3, e=5)
        self.assertEqual(list(od.items()), pairs)                                   # mixed input

        # Issue 9137: Named argument called 'other' or 'self'
        # shouldn't be treated specially.
        od = OrderedDict()
        od.update(self=23)
        self.assertEqual(list(od.items()), [('self', 23)])
        od = OrderedDict()
        od.update(other={})
        self.assertEqual(list(od.items()), [('other', {})])
        od = OrderedDict()
        od.update(red=5, blue=6, other=7, self=8)
        self.assertEqual(sorted(list(od.items())),
                         [('blue', 6), ('other', 7), ('red', 5), ('self', 8)])

        # Make sure that direct calls to update do not clear previous contents
        # add that updates items are not moved to the end
        d = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', 44), ('e', 55)])
        d.update([('e', 5), ('f', 6)], g=7, d=4)
        self.assertEqual(list(d.items()),
            [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5), ('f', 6), ('g', 7)])

        self.assertRaises(TypeError, OrderedDict().update, 42)
        self.assertRaises(TypeError, OrderedDict().update, (), ())
        self.assertRaises(TypeError, OrderedDict.update)

        self.assertRaises(TypeError, OrderedDict().update, 42)
        self.assertRaises(TypeError, OrderedDict().update, (), ())
        self.assertRaises(TypeError, OrderedDict.update)

    def test_init_calls(self):
        # [FUNCTION DOC] Check that constructing from an object having keys and items methods
        # uses only the keys method (the recorded call list is just keys).
        calls = []
        class Spam:
            # [CLASS DOC] HELPER: mapping-like object that records which of keys or items was
            # called into the enclosing list calls and returns empty tuples.
            def keys(self):
                # [FUNCTION DOC] HELPER: append keys to the shared call list and return an empty
                # tuple.
                calls.append('keys')
                return ()
            def items(self):
                # [FUNCTION DOC] HELPER: append items to the shared call list and return an
                # empty tuple.
                calls.append('items')
                return ()

        self.OrderedDict(Spam())
        self.assertEqual(calls, ['keys'])

    def test_overridden_init(self):
        # [FUNCTION DOC] Check a subclass whose init does nothing still works (internal state is
        # created at allocation time): item assignment must not fail.
        # Sync-up pure Python OD class with C class where
        # a consistent internal state is created in __new__
        # rather than __init__.
        OrderedDict = self.OrderedDict
        class ODNI(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose init method does nothing.
            def __init__(*args, **kwargs):
                # [FUNCTION DOC] HELPER: do nothing; bypasses the base initializer.
                pass
        od = ODNI()
        od['a'] = 1  # This used to fail because __init__ was bypassed

    def test_fromkeys(self):
        # [FUNCTION DOC] Check fromkeys keeps the iteration order of the keys and fills values
        # with None by default or with the given value.
        OrderedDict = self.OrderedDict
        od = OrderedDict.fromkeys('abc')
        self.assertEqual(list(od.items()), [(c, None) for c in 'abc'])
        od = OrderedDict.fromkeys('abc', value=None)
        self.assertEqual(list(od.items()), [(c, None) for c in 'abc'])
        od = OrderedDict.fromkeys('abc', value=0)
        self.assertEqual(list(od.items()), [(c, 0) for c in 'abc'])

    def test_abc(self):
        # [FUNCTION DOC] Check OrderedDict instances and the class are registered as
        # MutableMapping.
        OrderedDict = self.OrderedDict
        self.assertIsInstance(OrderedDict(), MutableMapping)
        self.assertTrue(issubclass(OrderedDict, MutableMapping))

    def test_clear(self):
        # [FUNCTION DOC] Check clear empties a shuffled mapping (length drops to zero).
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od = OrderedDict(pairs)
        self.assertEqual(len(od), len(pairs))
        od.clear()
        self.assertEqual(len(od), 0)

    def test_delitem(self):
        # [FUNCTION DOC] Check deleting a key removes it, deleting it again raises KeyError, and
        # the other items keep their order.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        od = OrderedDict(pairs)
        del od['a']
        self.assertNotIn('a', od)
        with self.assertRaises(KeyError):
            del od['a']
        self.assertEqual(list(od.items()), pairs[:2] + pairs[3:])

    def test_setitem(self):
        # [FUNCTION DOC] Check assigning an existing key keeps its position while a new key is
        # appended at the end.
        OrderedDict = self.OrderedDict
        od = OrderedDict([('d', 1), ('b', 2), ('c', 3), ('a', 4), ('e', 5)])
        od['c'] = 10           # existing element
        od['f'] = 20           # new element
        self.assertEqual(list(od.items()),
                         [('d', 1), ('b', 2), ('c', 10), ('a', 4), ('e', 5), ('f', 20)])

    def test_iterators(self):
        # [FUNCTION DOC] Check iteration over the mapping, keys, values, items and their
        # reversed forms follow insertion order (shuffled input).
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od = OrderedDict(pairs)
        self.assertEqual(list(od), [t[0] for t in pairs])
        self.assertEqual(list(od.keys()), [t[0] for t in pairs])
        self.assertEqual(list(od.values()), [t[1] for t in pairs])
        self.assertEqual(list(od.items()), pairs)
        self.assertEqual(list(reversed(od)),
                         [t[0] for t in reversed(pairs)])
        self.assertEqual(list(reversed(od.keys())),
                         [t[0] for t in reversed(pairs)])
        self.assertEqual(list(reversed(od.values())),
                         [t[1] for t in reversed(pairs)])
        self.assertEqual(list(reversed(od.items())), list(reversed(pairs)))

    def test_detect_deletion_during_iteration(self):
        # [FUNCTION DOC] Check that deleting the key just yielded makes the next step of the
        # iterator fail (the exact exception type is unspecified).
        OrderedDict = self.OrderedDict
        od = OrderedDict.fromkeys('abc')
        it = iter(od)
        key = next(it)
        del od[key]
        with self.assertRaises(Exception):
            # Note, the exact exception raised is not guaranteed
            # The only guarantee that the next() will not succeed
            next(it)

    def test_sorted_iterators(self):
        # [FUNCTION DOC] Check sorting the mapping, keys, values, items and reversed iteration
        # gives the expected sorted lists; a stray extra argument to the constructor raises
        # TypeError.
        OrderedDict = self.OrderedDict
        with self.assertRaises(TypeError):
            OrderedDict([('a', 1), ('b', 2)], None)
        pairs = [('a', 1), ('b', 2), ('c', 3), ('d', 4), ('e', 5)]
        od = OrderedDict(pairs)
        self.assertEqual(sorted(od), [t[0] for t in pairs])
        self.assertEqual(sorted(od.keys()), [t[0] for t in pairs])
        self.assertEqual(sorted(od.values()), [t[1] for t in pairs])
        self.assertEqual(sorted(od.items()), pairs)
        self.assertEqual(sorted(reversed(od)),
                         sorted([t[0] for t in reversed(pairs)]))

    def test_iterators_empty(self):
        # [FUNCTION DOC] Check all iteration forms (plain, keys, values, items, and their
        # reversed forms) of an empty mapping yield nothing.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        empty = []
        self.assertEqual(list(od), empty)
        self.assertEqual(list(od.keys()), empty)
        self.assertEqual(list(od.values()), empty)
        self.assertEqual(list(od.items()), empty)
        self.assertEqual(list(reversed(od)), empty)
        self.assertEqual(list(reversed(od.keys())), empty)
        self.assertEqual(list(reversed(od.values())), empty)
        self.assertEqual(list(reversed(od.items())), empty)

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem removes items from the end (last in, first out) until
        # empty, then raises KeyError.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od = OrderedDict(pairs)
        while pairs:
            self.assertEqual(od.popitem(), pairs.pop())
        with self.assertRaises(KeyError):
            od.popitem()
        self.assertEqual(len(od), 0)

    def test_popitem_last(self):
        # [FUNCTION DOC] Check popitem accepts the last flag positionally and by keyword,
        # removing the newest items (10 popped from 30 leaves 20).
        OrderedDict = self.OrderedDict
        pairs = [(i, i) for i in range(30)]

        obj = OrderedDict(pairs)
        for i in range(8):
            obj.popitem(True)
        obj.popitem(True)
        obj.popitem(last=True)
        self.assertEqual(len(obj), 20)

    def test_pop(self):
        # [FUNCTION DOC] Check pop returns and removes values, raises KeyError for a missing key
        # without default, returns a given default, accepts the default by keyword, and still
        # behaves the same when a subclass defines the missing hook.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od = OrderedDict(pairs)
        shuffle(pairs)
        while pairs:
            k, v = pairs.pop()
            self.assertEqual(od.pop(k), v)
        with self.assertRaises(KeyError):
            od.pop('xyz')
        self.assertEqual(len(od), 0)
        self.assertEqual(od.pop(k, 12345), 12345)

        # make sure pop still works when __missing__ is defined
        class Missing(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose missing hook returns 0.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: return 0 for any missing key.
                return 0
        m = Missing(a=1)
        self.assertEqual(m.pop('b', 5), 5)
        self.assertEqual(m.pop('a', 6), 1)
        self.assertEqual(m.pop('a', 6), 6)
        self.assertEqual(m.pop('a', default=6), 6)
        with self.assertRaises(KeyError):
            m.pop('a')

    def test_equality(self):
        # [FUNCTION DOC] Check two mappings with the same order are equal, different order makes
        # them unequal, comparison with a plain dict ignores order in both directions, and
        # different lengths are unequal.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od1 = OrderedDict(pairs)
        od2 = OrderedDict(pairs)
        self.assertEqual(od1, od2)          # same order implies equality
        pairs = pairs[2:] + pairs[:2]
        od2 = OrderedDict(pairs)
        self.assertNotEqual(od1, od2)       # different order implies inequality
        # comparison to regular dict is not order sensitive
        self.assertEqual(od1, dict(od2))
        self.assertEqual(dict(od2), od1)
        # different length implied inequality
        self.assertNotEqual(od1, OrderedDict(pairs[:-1]))

    def test_copying(self):
        # [FUNCTION DOC] Check copy method, shallow copy, deep copy, pickling at every protocol
        # (with the module registry patched to the implementation under test), repr/eval round
        # trip, update from itself and the constructor all give an equal independent mapping of
        # the same type; instance attributes are shared by shallow copy and duplicated by deep
        # copy.
        OrderedDict = self.OrderedDict
        # Check that ordered dicts are copyable, deepcopyable, picklable,
        # and have a repr/eval round-trip
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        od = OrderedDict(pairs)
        od.x = ['x']
        od.z = ['z']
        def check(dup):
            # [FUNCTION DOC] HELPER: assert dup is a distinct object equal to od with the same
            # item order, length and type.
            msg = "\ncopy: %s\nod: %s" % (dup, od)
            self.assertIsNot(dup, od, msg)
            self.assertEqual(dup, od)
            self.assertEqual(list(dup.items()), list(od.items()))
            self.assertEqual(len(dup), len(od))
            self.assertEqual(type(dup), type(od))
        check(od.copy())
        dup = copy.copy(od)
        check(dup)
        self.assertIs(dup.x, od.x)
        self.assertIs(dup.z, od.z)
        self.assertFalse(hasattr(dup, 'y'))
        dup = copy.deepcopy(od)
        check(dup)
        self.assertEqual(dup.x, od.x)
        self.assertIsNot(dup.x, od.x)
        self.assertEqual(dup.z, od.z)
        self.assertIsNot(dup.z, od.z)
        self.assertFalse(hasattr(dup, 'y'))
        # pickle directly pulls the module, so we have to fake it
        with replaced_module('collections', self.module):
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(proto=proto):
                    dup = pickle.loads(pickle.dumps(od, proto))
                    check(dup)
                    self.assertEqual(dup.x, od.x)
                    self.assertEqual(dup.z, od.z)
                    self.assertFalse(hasattr(dup, 'y'))
        check(eval(repr(od)))
        update_test = OrderedDict()
        update_test.update(od)
        check(update_test)
        check(OrderedDict(od))

    def test_yaml_linkage(self):
        # [FUNCTION DOC] Check the reduce result lists its items as lists (not tuples) so YAML
        # dump works.
        OrderedDict = self.OrderedDict
        # Verify that __reduce__ is setup in a way that supports PyYAML's dump() feature.
        # In yaml, lists are native but tuples are not.
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        od = OrderedDict(pairs)
        # yaml.dump(od) -->
        # '!!python/object/apply:__main__.OrderedDict\n- - [a, 1]\n  - [b, 2]\n'
        self.assertTrue(all(type(pair)==list for pair in od.__reduce__()[1]))

    def test_reduce_not_too_fat(self):
        # [FUNCTION DOC] Check the reduce result has no instance state (None) until an instance
        # attribute is set, after which the state is that attribute mapping.
        OrderedDict = self.OrderedDict
        # do not save instance dictionary if not needed
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        od = OrderedDict(pairs)
        self.assertIsInstance(od.__dict__, dict)
        self.assertIsNone(od.__reduce__()[2])
        od.x = 10
        self.assertEqual(od.__dict__['x'], 10)
        self.assertEqual(od.__reduce__()[2], {'x': 10})

    def test_pickle_recursive(self):
        # [FUNCTION DOC] Check a mapping containing itself survives a pickle round trip at every
        # protocol from the default negative one, keeping the self reference (module registry
        # patched).
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od[1] = od

        # pickle directly pulls the module, so we have to fake it
        with replaced_module('collections', self.module):
            for proto in range(-1, pickle.HIGHEST_PROTOCOL + 1):
                dup = pickle.loads(pickle.dumps(od, proto))
                self.assertIsNot(dup, od)
                self.assertEqual(list(dup.keys()), [1])
                self.assertIs(dup[1], dup)

    def test_repr(self):
        # [FUNCTION DOC] Check repr output text for a populated and an empty mapping, and that
        # eval of the repr gives an equal mapping.
        OrderedDict = self.OrderedDict
        od = OrderedDict([('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)])
        self.assertEqual(repr(od),
            "OrderedDict({'c': 1, 'b': 2, 'a': 3, 'd': 4, 'e': 5, 'f': 6})")
        self.assertEqual(eval(repr(od)), od)
        self.assertEqual(repr(OrderedDict()), "OrderedDict()")

    def test_repr_recursive(self):
        # [FUNCTION DOC] Check a self-referencing mapping prints an ellipsis for the recursive
        # value.
        OrderedDict = self.OrderedDict
        # See issue #9826
        od = OrderedDict.fromkeys('abc')
        od['x'] = od
        self.assertEqual(repr(od),
            "OrderedDict({'a': None, 'b': None, 'c': None, 'x': ...})")

    def test_repr_recursive_values(self):
        # [FUNCTION DOC] Check repr does not raise when a value is the values or items view of
        # the mapping itself; only a str result is asserted.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od[42] = od.values()
        r = repr(od)
        # Cannot perform a stronger test, as the contents of the repr
        # are implementation-dependent.  All we can say is that we
        # want a str result, not an exception of any sort.
        self.assertIsInstance(r, str)
        od[42] = od.items()
        r = repr(od)
        # Again.
        self.assertIsInstance(r, str)

    def test_setdefault(self):
        # [FUNCTION DOC] Check setdefault returns the existing value without reordering, appends
        # a new key at the end, accepts the default by keyword, and works for a subclass
        # defining the missing hook.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        shuffle(pairs)
        od = OrderedDict(pairs)
        pair_order = list(od.items())
        self.assertEqual(od.setdefault('a', 10), 3)
        # make sure order didn't change
        self.assertEqual(list(od.items()), pair_order)
        self.assertEqual(od.setdefault('x', 10), 10)
        # make sure 'x' is added to the end
        self.assertEqual(list(od.items())[-1], ('x', 10))
        self.assertEqual(od.setdefault('g', default=9), 9)

        # make sure setdefault still works when __missing__ is defined
        class Missing(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose missing hook returns 0.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: return 0 for any missing key.
                return 0
        self.assertEqual(Missing().setdefault(5, 9), 9)

    def test_reinsert(self):
        # [FUNCTION DOC] Check that deleting a key and inserting it again places it after the
        # other keys.
        OrderedDict = self.OrderedDict
        # Given insert a, insert b, delete a, re-insert a,
        # verify that a is now later than b.
        od = OrderedDict()
        od['a'] = 1
        od['b'] = 2
        del od['a']
        self.assertEqual(list(od.items()), [('b', 2)])
        od['a'] = 1
        self.assertEqual(list(od.items()), [('b', 2), ('a', 1)])

    def test_move_to_end(self):
        # [FUNCTION DOC] Check move_to_end moves a key to the end by default or to the start
        # with the last flag false (positional or keyword), is a no-op when already in place,
        # and raises KeyError for a missing key.
        OrderedDict = self.OrderedDict
        od = OrderedDict.fromkeys('abcde')
        self.assertEqual(list(od), list('abcde'))
        od.move_to_end('c')
        self.assertEqual(list(od), list('abdec'))
        od.move_to_end('c', False)
        self.assertEqual(list(od), list('cabde'))
        od.move_to_end('c', False)
        self.assertEqual(list(od), list('cabde'))
        od.move_to_end('e')
        self.assertEqual(list(od), list('cabde'))
        od.move_to_end('b', last=False)
        self.assertEqual(list(od), list('bcade'))
        with self.assertRaises(KeyError):
            od.move_to_end('x')
        with self.assertRaises(KeyError):
            od.move_to_end('x', False)

    def test_move_to_end_issue25406(self):
        # [FUNCTION DOC] Regression check of move_to_end to the front and back on three keys,
        # verifying the resulting order each time.
        OrderedDict = self.OrderedDict
        od = OrderedDict.fromkeys('abc')
        od.move_to_end('c', last=False)
        self.assertEqual(list(od), list('cab'))
        od.move_to_end('a', last=False)
        self.assertEqual(list(od), list('acb'))

        od = OrderedDict.fromkeys('abc')
        od.move_to_end('a')
        self.assertEqual(list(od), list('bca'))
        od.move_to_end('c')
        self.assertEqual(list(od), list('bac'))

    def test_sizeof(self):
        # [FUNCTION DOC] Check the reported size of an OrderedDict is larger than that of a
        # plain dict with the same content.
        OrderedDict = self.OrderedDict
        # Wimpy test: Just verify the reported size is larger than a regular dict
        d = dict(a=1)
        od = OrderedDict(**d)
        self.assertGreater(sys.getsizeof(od), sys.getsizeof(d))

    def test_views(self):
        # [FUNCTION DOC] Check keys and items views compare equal to those of the equivalent
        # plain dict.
        OrderedDict = self.OrderedDict
        # See http://bugs.python.org/issue24286
        s = 'the quick brown fox jumped over a lazy dog yesterday before dawn'.split()
        od = OrderedDict.fromkeys(s)
        self.assertEqual(od.keys(), dict(od).keys())
        self.assertEqual(od.items(), dict(od).items())

    def test_override_update(self):
        # [FUNCTION DOC] Check a subclass overriding update to raise does not break construction
        # from items.
        OrderedDict = self.OrderedDict
        # Verify that subclasses can override update() without breaking __init__()
        class MyOD(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose update method always raises.
            def update(self, *args, **kwds):
                # [FUNCTION DOC] HELPER: always raise a generic Exception.
                raise Exception()
        items = [('a', 1), ('c', 3), ('b', 2)]
        self.assertEqual(list(MyOD(items).items()), items)

    def test_highly_nested(self):
        # [FUNCTION DOC] Check that dropping a chain of 1000 nested mappings and collecting
        # garbage does not crash (deallocation recursion guard).
        # Issues 25395 and 35983: test that the trashcan mechanism works
        # correctly for OrderedDict: deleting a highly nested OrderDict
        # should not crash Python.
        OrderedDict = self.OrderedDict
        obj = None
        for _ in range(1000):
            obj = OrderedDict([(None, obj)])
        del obj
        support.gc_collect()

    def test_highly_nested_subclass(self):
        # [FUNCTION DOC] Check that a chain of 100 nested subclass instances with a finalizer is
        # finalized from the outermost to the innermost: the recorded order is descending.
        # Issues 25395 and 35983: test that the trashcan mechanism works
        # correctly for OrderedDict: deleting a highly nested OrderDict
        # should not crash Python.
        OrderedDict = self.OrderedDict
        deleted = []
        class MyOD(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose finalizer records the instance
            # attribute i in the shared list deleted.
            def __del__(self):
                # [FUNCTION DOC] HELPER: append the instance attribute i to the shared list
                # deleted.
                deleted.append(self.i)
        obj = None
        for i in range(100):
            obj = MyOD([(None, obj)])
            obj.i = i
        del obj
        support.gc_collect()
        self.assertEqual(deleted, list(reversed(range(100))))

    def test_delitem_hash_collision(self):
        # [FUNCTION DOC] Check deleting keys that collide in the dict hash table (a colliding
        # key, a blocking key and another colliding key) keeps the remaining items and their
        # order intact.
        OrderedDict = self.OrderedDict

        class Key:
            # [CLASS DOC] HELPER: key object with a fixed hash and a random string value used as
            # identity for equality; used to force hash collisions.
            def __init__(self, hash):
                # [FUNCTION DOC] HELPER: store the forced hash and a value string derived from
                # the object id.
                self._hash = hash
                self.value = str(id(self))
            def __hash__(self):
                # [FUNCTION DOC] HELPER: return the stored hash.
                return self._hash
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: equal if the other object has the same value attribute;
                # False if it has none.
                try:
                    return self.value == other.value
                except AttributeError:
                    return False
            def __repr__(self):
                # [FUNCTION DOC] HELPER: return the value string.
                return self.value

        def blocking_hash(hash):
            # [FUNCTION DOC] HELPER: compute a hash that lands on the probe slot following the
            # given hash in a minimum-size dict table (mirrors the collision handling of the C
            # dict lookup).
            # See the collision-handling in lookdict (in Objects/dictobject.c).
            MINSIZE = 8
            i = (hash & MINSIZE-1)
            return (i << 2) + i + hash + 1

        COLLIDING = 1

        key = Key(COLLIDING)
        colliding = Key(COLLIDING)
        blocking = Key(blocking_hash(COLLIDING))

        od = OrderedDict()
        od[key] = ...
        od[blocking] = ...
        od[colliding] = ...
        od['after'] = ...

        del od[blocking]
        del od[colliding]
        self.assertEqual(list(od.items()), [(key, ...), ('after', ...)])

    def test_issue24347(self):
        # [FUNCTION DOC] Check keys with random (unstable) hashes cause KeyError, not a crash,
        # when taking values, items, repr or copy.
        OrderedDict = self.OrderedDict

        class Key:
            # [CLASS DOC] HELPER: key whose hash is random on every call, so lookups fail.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: return a random integer below 100000.
                return randrange(100000)

        od = OrderedDict()
        for i in range(100):
            key = Key()
            od[key] = i

        # These should not crash.
        with self.assertRaises(KeyError):
            list(od.values())
        with self.assertRaises(KeyError):
            list(od.items())
        with self.assertRaises(KeyError):
            repr(od)
        with self.assertRaises(KeyError):
            od.copy()

    def test_issue24348(self):
        # [FUNCTION DOC] Check popitem on a mapping with a single key does not crash.
        OrderedDict = self.OrderedDict

        class Key:
            # [CLASS DOC] HELPER: key with constant hash 1.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: return 1.
                return 1

        od = OrderedDict()
        od[Key()] = 0
        # This should not crash.
        od.popitem()

    def test_issue24667(self):
        # [FUNCTION DOC] Check repeated popitem from the front on a sparse mapping while
        # inserting keeps working across underlying dict resizes without raising KeyError.
        """
        dict resizes after a certain number of insertion operations,
        whether or not there were deletions that freed up slots in the
        hash table.  During fast node lookup, OrderedDict must correctly
        respond to all resizes, even if the current "size" is the same
        as the old one.  We verify that here by forcing a dict resize
        on a sparse odict and then perform an operation that should
        trigger an odict resize (e.g. popitem).  One key aspect here is
        that we will keep the size of the odict the same at each popitem
        call.  This verifies that we handled the dict resize properly.
        """
        OrderedDict = self.OrderedDict

        od = OrderedDict()
        for c0 in '0123456789ABCDEF':
            for c1 in '0123456789ABCDEF':
                if len(od) == 4:
                    # This should not raise a KeyError.
                    od.popitem(last=False)
                key = c0 + c1
                od[key] = key

    # Direct use of dict methods

    def test_dict_setitem(self):
        # [FUNCTION DOC] Check dict-level set item on an ordered mapping does not corrupt repr
        # (no NULL appears).
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        dict.__setitem__(od, 'spam', 1)
        self.assertNotIn('NULL', repr(od))

    def test_dict_delitem(self):
        # [FUNCTION DOC] Check dict-level delete of a key makes repr raise KeyError (the linked
        # order is inconsistent).
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od['spam'] = 1
        od['ham'] = 2
        dict.__delitem__(od, 'spam')
        with self.assertRaises(KeyError):
            repr(od)

    def test_dict_clear(self):
        # [FUNCTION DOC] Check dict-level clear does not corrupt repr (no NULL appears).
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od['spam'] = 1
        od['ham'] = 2
        dict.clear(od)
        self.assertNotIn('NULL', repr(od))

    def test_dict_pop(self):
        # [FUNCTION DOC] Check dict-level pop of a key makes repr raise KeyError.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od['spam'] = 1
        od['ham'] = 2
        dict.pop(od, 'spam')
        with self.assertRaises(KeyError):
            repr(od)

    def test_dict_popitem(self):
        # [FUNCTION DOC] Check dict-level popitem makes repr raise KeyError.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        od['spam'] = 1
        od['ham'] = 2
        dict.popitem(od)
        with self.assertRaises(KeyError):
            repr(od)

    def test_dict_setdefault(self):
        # [FUNCTION DOC] Check dict-level setdefault does not corrupt repr.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        dict.setdefault(od, 'spam', 1)
        self.assertNotIn('NULL', repr(od))

    def test_dict_update(self):
        # [FUNCTION DOC] Check dict-level update does not corrupt repr.
        OrderedDict = self.OrderedDict
        od = OrderedDict()
        dict.update(od, [('spam', 1)])
        self.assertNotIn('NULL', repr(od))

    @suppress_immortalization()
    def test_reference_loop(self):
        # [FUNCTION DOC] Check a class referencing an ordered mapping that has the class itself
        # as a key is collectable: after delete and gc the weak reference is dead.
        # Issue 25935
        OrderedDict = self.OrderedDict
        class A:
            # [CLASS DOC] HELPER: class holding an OrderedDict as class attribute od.
            od = OrderedDict()
        A.od[A] = None
        r = weakref.ref(A)
        del A
        gc.collect()
        self.assertIsNone(r())

    def test_free_after_iterating(self):
        # [FUNCTION DOC] Check, using the support helper, that objects are freed after iterating
        # the mapping, its keys, values and items.
        support.check_free_after_iterating(self, iter, self.OrderedDict)
        support.check_free_after_iterating(self, lambda d: iter(d.keys()), self.OrderedDict)
        support.check_free_after_iterating(self, lambda d: iter(d.values()), self.OrderedDict)
        support.check_free_after_iterating(self, lambda d: iter(d.items()), self.OrderedDict)

    def test_merge_operator(self):
        # [FUNCTION DOC] Check the union operator and in-place union with mappings and pair
        # lists: right side wins, results keep OrderedDict type, empty tuple or string is
        # accepted in-place, and bad operands raise TypeError (or ValueError for a bad in-place
        # string).
        OrderedDict = self.OrderedDict

        a = OrderedDict({0: 0, 1: 1, 2: 1})
        b = OrderedDict({1: 1, 2: 2, 3: 3})

        c = a.copy()
        d = a.copy()
        c |= b
        d |= list(b.items())
        expected = OrderedDict({0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(a | dict(b), expected)
        self.assertEqual(a | b, expected)
        self.assertEqual(c, expected)
        self.assertEqual(d, expected)

        c = b.copy()
        c |= a
        expected = OrderedDict({1: 1, 2: 1, 3: 3, 0: 0})
        self.assertEqual(dict(b) | a, expected)
        self.assertEqual(b | a, expected)
        self.assertEqual(c, expected)

        self.assertIs(type(a | b), OrderedDict)
        self.assertIs(type(dict(a) | b), OrderedDict)
        self.assertIs(type(a | dict(b)), OrderedDict)

        expected = a.copy()
        a |= ()
        a |= ""
        self.assertEqual(a, expected)

        with self.assertRaises(TypeError):
            a | None
        with self.assertRaises(TypeError):
            a | ()
        with self.assertRaises(TypeError):
            a | "BAD"
        with self.assertRaises(TypeError):
            a | ""
        with self.assertRaises(ValueError):
            a |= "BAD"

    @support.cpython_only
    def test_ordered_dict_items_result_gc(self):
        # [FUNCTION DOC] CPython only. Check the recycled result tuple of the items iterator is
        # tracked by the garbage collector again after being refilled, even after a collection
        # untracked it.
        # bpo-42536: OrderedDict.items's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = iter(self.OrderedDict({None: []}).items())
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None, None). Make sure it's re-tracked
        # when it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))


class _TriggerSideEffectOnEqual:
    # [CLASS DOC] HELPER: base for keys that trigger a side effect during comparison. Class
    # attributes: count (number of equality calls so far) and trigger (call number at which the
    # side effect runs). All instances hash to -1 and compare equal.
    count = 0   # number of calls to __eq__
    trigger = 1 # count value when to trigger side effect

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: run side_effect when the class-level count equals trigger,
        # increment the count and return True.
        if self.__class__.count == self.__class__.trigger:
            self.side_effect()
        self.__class__.count += 1
        return True

    def __hash__(self):
        # [FUNCTION DOC] HELPER: return the constant -1 so all instances act as the same key.
        # all instances represent the same key
        return -1

    def side_effect(self):
        # [FUNCTION DOC] HELPER: hook for subclasses; the base raises NotImplementedError.
        raise NotImplementedError

class PurePythonOrderedDictTests(OrderedDictTests, __TestCase):
    # [CLASS DOC] Runs OrderedDictTests against the pure-Python collections copy, plus a
    # regression test for mutation during comparison.

    module = py_coll
    OrderedDict = py_coll.OrderedDict

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_attribute_error(self):
        # [FUNCTION DOC] Skipped under Dynamo. A key deletes another key of the first mapping
        # while the two mappings are compared; the pure-Python version must raise AttributeError
        # about a missing key attribute of a None link, after exactly two equality calls,
        # leaving the mappings as expected.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key whose side effect deletes the marked key from dict1.
            def side_effect(self):
                # [FUNCTION DOC] HELPER: delete the TODEL key from dict1.
                del dict1[TODEL]

        TODEL = Key()
        dict1 = self.OrderedDict(dict.fromkeys((0, TODEL, 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        # This causes an AttributeError due to the linked list being changed
        msg = re.escape("'NoneType' object has no attribute 'key'")
        self.assertRaisesRegex(AttributeError, msg, operator.eq, dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, dict.fromkeys((0, 4.2)))
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))


class CPythonBuiltinDictTests(__TestCase):
    # [CLASS DOC] Reuses a selected list of OrderedDictTests methods (init, update, abc, clear,
    # deletion, setitem, popitem, reinsert, override update, nesting and hash collision tests)
    # for the builtin dict, which keeps insertion order. Methods are copied onto the class by
    # the loop after it.
    """Builtin dict preserves insertion order.

    Reuse some of tests in OrderedDict selectively.
    """

    module = builtins
    OrderedDict = dict

for method in (
    "test_init test_update test_abc test_clear test_delitem " +
    "test_setitem test_detect_deletion_during_iteration " +
    "test_popitem test_reinsert test_override_update " +
    "test_highly_nested test_highly_nested_subclass " +
    "test_delitem_hash_collision ").split():
    setattr(CPythonBuiltinDictTests, method, getattr(OrderedDictTests, method))
del method



class CPythonOrderedDictSideEffects:
    # [CLASS DOC] Mixin of regression tests (issue 119004) for the C OrderedDict where a key's
    # equality hook mutates the mappings during comparison. Requires attribute OrderedDict from
    # the concrete class.

    def check_runtime_error_issue119004(self, dict1, dict2):
        # [FUNCTION DOC] HELPER: assert comparing the two mappings with the equality operator
        # raises RuntimeError saying the mapping was mutated during iteration.
        msg = re.escape("OrderedDict mutated during iteration")
        self.assertRaisesRegex(RuntimeError, msg, operator.eq, dict1, dict2)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_change_size_by_clear(self):
        # [FUNCTION DOC] Skipped under Dynamo. A key side effect clears the first mapping during
        # comparison: expect RuntimeError, two equality calls, first mapping empty and second
        # unchanged.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key whose side effect clears dict1.
            def side_effect(self):
                # [FUNCTION DOC] HELPER: clear dict1.
                dict1.clear()

        dict1 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        self.check_runtime_error_issue119004(dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, {})
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_change_size_by_delete_key(self):
        # [FUNCTION DOC] Skipped under Dynamo. A key side effect deletes a key from the first
        # mapping during comparison: expect RuntimeError, two equality calls, and the first
        # mapping missing that key.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key whose side effect deletes the TODEL key from dict1.
            def side_effect(self):
                # [FUNCTION DOC] HELPER: delete the TODEL key from dict1.
                del dict1[TODEL]

        TODEL = Key()
        dict1 = self.OrderedDict(dict.fromkeys((0, TODEL, 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        self.check_runtime_error_issue119004(dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, dict.fromkeys((0, 4.2)))
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_change_linked_list_by_clear(self):
        # [FUNCTION DOC] Skipped under Dynamo. A key side effect clears dict1 and adds two new
        # keys during comparison (changing the internal order list while keeping size): expect
        # RuntimeError and the first mapping holding only the new keys.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key whose side effect clears dict1 then adds keys a and b.
            def side_effect(self):
                # [FUNCTION DOC] HELPER: clear dict1 and set keys a and b to the value c.
                dict1.clear()
                dict1['a'] = dict1['b'] = 'c'

        dict1 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        self.check_runtime_error_issue119004(dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, dict.fromkeys(('a', 'b'), 'c'))
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_change_linked_list_by_delete_key(self):
        # [FUNCTION DOC] Skipped under Dynamo. A key side effect deletes one key and adds
        # another during comparison: expect RuntimeError and the first mapping to contain the
        # replaced key set.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key whose side effect deletes the TODEL key from dict1 and
            # adds key a.
            def side_effect(self):
                # [FUNCTION DOC] HELPER: delete TODEL from dict1 and set key a to c.
                del dict1[TODEL]
                dict1['a'] = 'c'

        TODEL = Key()
        dict1 = self.OrderedDict(dict.fromkeys((0, TODEL, 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        self.check_runtime_error_issue119004(dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, {0: None, 'a': 'c', 4.2: None})
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_issue119004_change_size_by_delete_key_in_dict_eq(self):
        # [FUNCTION DOC] Skipped under Dynamo. With the side effect triggered on the first
        # equality call (inside the dict-level comparison), the mappings compare unequal without
        # raising; two equality calls happen and the first mapping lost the key.
        class Key(_TriggerSideEffectOnEqual):
            # [CLASS DOC] HELPER: key triggering its side effect on the first equality call
            # (trigger is 0).
            trigger = 0
            def side_effect(self):
                # [FUNCTION DOC] HELPER: delete the TODEL key from dict1.
                del dict1[TODEL]

        TODEL = Key()
        dict1 = self.OrderedDict(dict.fromkeys((0, TODEL, 4.2)))
        dict2 = self.OrderedDict(dict.fromkeys((0, Key(), 4.2)))
        self.assertEqual(Key.count, 0)
        # the side effect is in dict.__eq__ and modifies the length
        self.assertNotEqual(dict1, dict2)
        self.assertEqual(Key.count, 2)
        self.assertDictEqual(dict1, dict.fromkeys((0, 4.2)))
        self.assertDictEqual(dict2, dict.fromkeys((0, Key(), 4.2)))


@unittest.skipUnless(c_coll, 'requires the C version of the collections module')
class CPythonOrderedDictTests(OrderedDictTests,
                              CPythonOrderedDictSideEffects,
                              __TestCase):
    # [CLASS DOC] Runs OrderedDictTests and the side effect tests against the C collections
    # module (skipped when unavailable). Attribute check_sizeof is the support size-check
    # helper.

    module = c_coll
    OrderedDict = c_coll.OrderedDict
    check_sizeof = support.check_sizeof

    @support.cpython_only
    def test_sizeof_exact(self):
        # [FUNCTION DOC] CPython only. Compute expected byte sizes from struct layouts and check
        # the exact reported size of an empty mapping, after adding an instance attribute, after
        # growing to 3 and then 10 entries, and of keys/items/values views and their iterators.
        OrderedDict = self.OrderedDict
        calcsize = struct.calcsize
        size = support.calcobjsize
        check = self.check_sizeof

        basicsize = size('nQ2P' + '3PnPn2P')
        keysize = calcsize('n2BI2n')

        entrysize = calcsize('n2P')
        p = calcsize('P')
        nodesize = calcsize('Pn2P')

        od = OrderedDict()
        check(od, basicsize)  # 8byte indices + 8*2//3 * entry table
        od.x = 1
        check(od, basicsize)
        od.update([(i, i) for i in range(3)])
        check(od, basicsize + keysize + 8*p + 8 + 5*entrysize + 3*nodesize)
        od.update([(i, i) for i in range(3, 10)])
        check(od, basicsize + keysize + 16*p + 16 + 10*entrysize + 10*nodesize)

        check(od.keys(), size('P'))
        check(od.items(), size('P'))
        check(od.values(), size('P'))

        itersize = size('iP2n2P')
        check(iter(od), itersize)
        check(iter(od.keys()), itersize)
        check(iter(od.items()), itersize)
        check(iter(od.values()), itersize)

    def test_key_change_during_iteration(self):
        # [FUNCTION DOC] Check that reordering, adding or deleting keys while iterating raises
        # RuntimeError, and that the final order is as expected after those operations.
        OrderedDict = self.OrderedDict

        od = OrderedDict.fromkeys('abcde')
        self.assertEqual(list(od), list('abcde'))
        with self.assertRaises(RuntimeError):
            for i, k in enumerate(od):
                od.move_to_end(k)
                self.assertLess(i, 5)
        with self.assertRaises(RuntimeError):
            for k in od:
                od['f'] = None
        with self.assertRaises(RuntimeError):
            for k in od:
                del od['c']
        self.assertEqual(list(od), list('bdeaf'))

    def test_iterators_pickling(self):
        # [FUNCTION DOC] Check keys, values and items iterators, partly consumed by one element,
        # pickle at every protocol and the unpickled iterator and the original both yield the
        # remaining items.
        OrderedDict = self.OrderedDict
        pairs = [('c', 1), ('b', 2), ('a', 3), ('d', 4), ('e', 5), ('f', 6)]
        od = OrderedDict(pairs)

        for method_name in ('keys', 'values', 'items'):
            meth = getattr(od, method_name)
            expected = list(meth())[1:]
            for i in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(method_name=method_name, protocol=i):
                    it = iter(meth())
                    next(it)
                    p = pickle.dumps(it, i)
                    unpickled = pickle.loads(p)
                    self.assertEqual(list(unpickled), expected)
                    self.assertEqual(list(it), expected)

    @support.cpython_only
    def test_weakref_list_is_not_traversed(self):
        # [FUNCTION DOC] CPython only. Check garbage collection of a mapping in a reference
        # cycle that also has a weak reference on a cycle list completes without traversing the
        # weakref list (bpo-39778).
        # Check that the weakref list is not traversed when collecting
        # OrderedDict objects. See bpo-39778 for more information.

        gc.collect()

        x = self.OrderedDict()
        x.cycle = x

        cycle = []
        cycle.append(cycle)

        x_ref = weakref.ref(x)
        cycle.append(x_ref)

        del x, cycle, x_ref

        gc.collect()


class PurePythonOrderedDictSubclassTests(PurePythonOrderedDictTests):
    # [CLASS DOC] Re-runs the pure-Python tests with OrderedDict replaced by a trivial subclass.

    module = py_coll
    class OrderedDict(py_coll.OrderedDict):
        # [CLASS DOC] HELPER: empty subclass of the pure-Python OrderedDict.
        pass


class CPythonOrderedDictSubclassTests(CPythonOrderedDictTests):
    # [CLASS DOC] Re-runs the C tests with OrderedDict replaced by a trivial subclass.

    module = c_coll
    class OrderedDict(c_coll.OrderedDict):
        # [CLASS DOC] HELPER: empty subclass of the C OrderedDict.
        pass


class PurePythonOrderedDictWithSlotsCopyingTests(__TestCase):
    # [CLASS DOC] Runs only the copying test with a pure-Python OrderedDict subclass that
    # declares slots x and y.

    module = py_coll
    class OrderedDict(py_coll.OrderedDict):
        # [CLASS DOC] HELPER: pure-Python subclass declaring slots x and y.
        __slots__ = ('x', 'y')
    test_copying = OrderedDictTests.test_copying


@unittest.skipUnless(c_coll, 'requires the C version of the collections module')
class CPythonOrderedDictWithSlotsCopyingTests(__TestCase):
    # [CLASS DOC] Runs only the copying test with a C OrderedDict subclass that declares slots x
    # and y (skipped without the C module).

    module = c_coll
    class OrderedDict(c_coll.OrderedDict):
        # [CLASS DOC] HELPER: C subclass declaring slots x and y.
        __slots__ = ('x', 'y')
    test_copying = OrderedDictTests.test_copying


class PurePythonGeneralMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol test suite against the pure-Python
    # OrderedDict, overriding popitem on empty mappings.

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Set type2test to the pure-Python OrderedDict, then run the parent class
        # setup.
        cls.type2test = py_coll.OrderedDict
        super().setUpClass()

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)


@unittest.skipUnless(c_coll, 'requires the C version of the collections module')
class CPythonGeneralMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol suite against the C OrderedDict (skipped
    # when unavailable).

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Set type2test to the C OrderedDict, then run the parent class setup.
        cls.type2test = c_coll.OrderedDict
        super().setUpClass()

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)


class PurePythonSubclassMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol suite against a subclass of the pure-Python
    # OrderedDict.

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Define an empty subclass of the pure-Python OrderedDict, set it as
        # type2test and run the parent class setup.
        class MyOrderedDict(py_coll.OrderedDict):
            # [CLASS DOC] HELPER: empty subclass of the pure-Python OrderedDict.
            pass
        cls.type2test = MyOrderedDict
        super().setUpClass()

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)


@unittest.skipUnless(c_coll, 'requires the C version of the collections module')
class CPythonSubclassMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol suite against a subclass of the C
    # OrderedDict (skipped when unavailable).

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Define an empty subclass of the C OrderedDict, set it as type2test and
        # run the parent class setup.
        class MyOrderedDict(c_coll.OrderedDict):
            # [CLASS DOC] HELPER: empty subclass of the C OrderedDict.
            pass
        cls.type2test = MyOrderedDict
        super().setUpClass()

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)


class SimpleLRUCache:
    # [CLASS DOC] HELPER: mixin placed before OrderedDict in a subclass to build a small LRU
    # cache. State: size (capacity) and counts (mapping of get, set and del to call counts).
    # Reads move the item to the end; writes evict from the front when full.

    def __init__(self, size):
        # [FUNCTION DOC] HELPER: call the parent initializer, store the capacity and zero the
        # operation counters.
        super().__init__()
        self.size = size
        self.counts = dict.fromkeys(('get', 'set', 'del'), 0)

    def __getitem__(self, item):
        # [FUNCTION DOC] HELPER: count a get, look up the value through the parent, move the key
        # to the end (most recent) and return the value.
        self.counts['get'] += 1
        value = super().__getitem__(item)
        self.move_to_end(item)
        return value

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: count a set, evict oldest items while the key is new and the
        # cache is full, store the value and move the key to the end.
        self.counts['set'] += 1
        while key not in self and len(self) >= self.size:
            self.popitem(last=False)
        super().__setitem__(key, value)
        self.move_to_end(key)

    def __delitem__(self, key):
        # [FUNCTION DOC] HELPER: count a delete and remove the key through the parent.
        self.counts['del'] += 1
        super().__delitem__(key)


class SimpleLRUCacheTests:
    # [CLASS DOC] Mixin of tests for the LRU cache; the concrete class sets type2test to a cache
    # class built over an OrderedDict.

    def test_add_after_full(self):
        # [FUNCTION DOC] Check adding a third item to a capacity-2 cache evicts the oldest, and
        # only set operations are counted.
        c = self.type2test(2)
        c['t1'] = 1
        c['t2'] = 2
        c['t3'] = 3
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})
        self.assertEqual(list(c), ['t2', 't3'])
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem from both ends on a full cache returns the oldest and
        # newest items without counting gets or deletes.
        c = self.type2test(3)
        for i in range(1, 4):
            c[i] = i
        self.assertEqual(c.popitem(last=False), (1, 1))
        self.assertEqual(c.popitem(last=True), (3, 3))
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})

    def test_pop(self):
        # [FUNCTION DOC] Check pop returns a present value, honors a default, raises KeyError
        # when missing, and never calls the overridden delete or get hooks (counters unchanged).
        c = self.type2test(3)
        for i in range(1, 4):
            c[i] = i
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})
        self.assertEqual(c.pop(2), 2)
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})
        self.assertEqual(c.pop(4, 0), 0)
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})
        self.assertRaises(KeyError, c.pop, 4)
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})

    def test_change_order_on_get(self):
        # [FUNCTION DOC] Check reading an item moves it to the end and counts one get.
        c = self.type2test(3)
        for i in range(1, 4):
            c[i] = i
        self.assertEqual(list(c), list(range(1, 4)))
        self.assertEqual(c.counts, {'get': 0, 'set': 3, 'del': 0})
        self.assertEqual(c[2], 2)
        self.assertEqual(c.counts, {'get': 1, 'set': 3, 'del': 0})
        self.assertEqual(list(c), [1, 3, 2])


class PySimpleLRUCacheTests(SimpleLRUCacheTests, __TestCase):
    # [CLASS DOC] Runs the LRU cache tests with a cache class derived from the pure-Python
    # OrderedDict.

    class type2test(SimpleLRUCache, py_coll.OrderedDict):
        # [CLASS DOC] HELPER: LRU cache class over the pure-Python OrderedDict.
        pass


@unittest.skipUnless(c_coll, 'requires the C version of the collections module')
class CSimpleLRUCacheTests(SimpleLRUCacheTests, __TestCase):
    # [CLASS DOC] Runs the LRU cache tests with a cache class derived from the C OrderedDict
    # (skipped when unavailable).

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Build the LRU cache class over the C OrderedDict, store it as type2test
        # and run the parent class setup.
        class type2test(SimpleLRUCache, c_coll.OrderedDict):
            # [CLASS DOC] HELPER: LRU cache class over the C OrderedDict.
            pass
        cls.type2test = type2test
        super().setUpClass()


if __name__ == "__main__":
    run_tests()
