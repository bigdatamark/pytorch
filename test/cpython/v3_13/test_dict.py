# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Port of CPython 3.13 test_dict run under the Dynamo CPython test harness.
# Installs an import finder on the meta path at import time (side effect) to redirect some
# test.* imports to top-level modules. Defines DictTest (dict API behavior), CAPITest, and
# mapping protocol tests for dict and a subclass. Run via run_tests.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_dict.py

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
    # [CLASS DOC] HELPER: import finder placed first on the meta path; redirects the listed
    # test.* module names to their top-level equivalents.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: for names in the redirect list, imports the module without the
        # test prefix, registers it in sys.modules under the full name and returns its spec;
        # returns None otherwise or on ImportError.
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

import collections
import collections.abc
import gc
import pickle
import random
import string
import sys
import unittest
import weakref
from test import support
from test.support import import_helper, get_c_recursion_limit


class DictTest(__TestCase):
    # [CLASS DOC] Test case for built-in dict behavior (CPython 3.13 test_dict port):
    # constructors, views, mutation, errors, GC tracking, pickling of iterators. Stateless; each
    # method is an independent test run under the Dynamo CPython test harness.

    def test_invalid_keyword_arguments(self):
        # [FUNCTION DOC] Checks that building or updating a dict with unpacked keyword
        # arguments raises TypeError when the mapping has non-string keys, for both a plain dict
        # and a dict subclass.
        class Custom(dict):
            # [CLASS DOC] HELPER: empty dict subclass used to check keyword unpacking
            # of subclass instances.
            pass
        for invalid in {1 : 2}, Custom({1 : 2}):
            with self.assertRaises(TypeError):
                dict(**invalid)
            with self.assertRaises(TypeError):
                {}.update(**invalid)

    def test_constructor(self):
        # [FUNCTION DOC] Checks that calling dict with no arguments returns a new empty dict,
        # distinct from a literal.
        # calling built-in types without argument must return empty
        self.assertEqual(dict(), {})
        self.assertIsNot(dict(), {})

    def test_literal_constructor(self):
        # [FUNCTION DOC] Builds dict literals of several sizes as source text with shuffled
        # random items, evaluates the text, and checks the result equals a dict built from the
        # same items. Uses random state.
        # check literal constructor for different sized dicts
        # (to exercise the BUILD_MAP oparg).
        for n in (0, 1, 6, 256, 400):
            items = [(''.join(random.sample(string.ascii_letters, 8)), i)
                     for i in range(n)]
            random.shuffle(items)
            formatted_items = ('{!r}: {:d}'.format(k, v) for k, v in items)
            dictliteral = '{' + ', '.join(formatted_items) + '}'
            self.assertEqual(eval(dictliteral), dict(items))

    def test_merge_operator(self):
        # [FUNCTION DOC] Checks the merge operators: binary merge returns a new dict with right-
        # hand values winning, in-place merge updates the left dict and accepts iterables of
        # pairs, binary merge returns NotImplemented for non-mappings, and in-place merge raises
        # TypeError for None and ValueError for a bad string.

        a = {0: 0, 1: 1, 2: 1}
        b = {1: 1, 2: 2, 3: 3}

        c = a.copy()
        c |= b

        self.assertEqual(a | b, {0: 0, 1: 1, 2: 2, 3: 3})
        self.assertEqual(c, {0: 0, 1: 1, 2: 2, 3: 3})

        c = b.copy()
        c |= a

        self.assertEqual(b | a, {1: 1, 2: 1, 3: 3, 0: 0})
        self.assertEqual(c, {1: 1, 2: 1, 3: 3, 0: 0})

        c = a.copy()
        c |= [(1, 1), (2, 2), (3, 3)]

        self.assertEqual(c, {0: 0, 1: 1, 2: 2, 3: 3})

        self.assertIs(a.__or__(None), NotImplemented)
        self.assertIs(a.__or__(()), NotImplemented)
        self.assertIs(a.__or__("BAD"), NotImplemented)
        self.assertIs(a.__or__(""), NotImplemented)

        self.assertRaises(TypeError, a.__ior__, None)
        self.assertEqual(a.__ior__(()), {0: 0, 1: 1, 2: 1})
        self.assertRaises(ValueError, a.__ior__, "BAD")
        self.assertEqual(a.__ior__(""), {0: 0, 1: 1, 2: 1})

    def test_bool(self):
        # [FUNCTION DOC] Checks truthiness: an empty dict is false, a non-empty dict is true.
        self.assertIs(not {}, True)
        self.assertTrue({1: 2})
        self.assertIs(bool({}), False)
        self.assertIs(bool({1: 2}), True)

    def test_keys(self):
        # [FUNCTION DOC] Checks the keys view: contents, membership, TypeError when called with
        # an argument, and its repr.
        d = {}
        self.assertEqual(set(d.keys()), set())
        d = {'a': 1, 'b': 2}
        k = d.keys()
        self.assertEqual(set(k), {'a', 'b'})
        self.assertIn('a', k)
        self.assertIn('b', k)
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertRaises(TypeError, d.keys, None)
        self.assertEqual(repr(dict(a=1).keys()), "dict_keys(['a'])")

    def test_values(self):
        # [FUNCTION DOC] Checks the values view: contents, TypeError when called with an
        # argument, and its repr.
        d = {}
        self.assertEqual(set(d.values()), set())
        d = {1:2}
        self.assertEqual(set(d.values()), {2})
        self.assertRaises(TypeError, d.values, None)
        self.assertEqual(repr(dict(a=1).values()), "dict_values([1])")

    def test_items(self):
        # [FUNCTION DOC] Checks the items view: contents, TypeError when called with an
        # argument, and its repr.
        d = {}
        self.assertEqual(set(d.items()), set())

        d = {1:2}
        self.assertEqual(set(d.items()), {(1, 2)})
        self.assertRaises(TypeError, d.items, None)
        self.assertEqual(repr(dict(a=1).items()), "dict_items([('a', 1)])")

    def test_views_mapping(self):
        # [FUNCTION DOC] Checks that the mapping attribute of keys, values and items views is a
        # read-only mapping proxy equal to the dict, for dict and a subclass, and that it
        # reflects later insertions.
        mappingproxy = type(type.__dict__)
        class Dict(dict):
            # [CLASS DOC] HELPER: empty dict subclass used to test view mapping
            # attributes on subclasses.
            pass
        for cls in [dict, Dict]:
            d = cls()
            m1 = d.keys().mapping
            m2 = d.values().mapping
            m3 = d.items().mapping

            for m in [m1, m2, m3]:
                self.assertIsInstance(m, mappingproxy)
                self.assertEqual(m, d)

            d["foo"] = "bar"

            for m in [m1, m2, m3]:
                self.assertIsInstance(m, mappingproxy)
                self.assertEqual(m, d)

    def test_contains(self):
        # [FUNCTION DOC] Checks membership tests on a dict, and that calling the contains method
        # with no argument raises TypeError.
        d = {}
        self.assertNotIn('a', d)
        self.assertFalse('a' in d)
        self.assertTrue('a' not in d)
        d = {'a': 1, 'b': 2}
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertNotIn('c', d)

        self.assertRaises(TypeError, d.__contains__)

    def test_len(self):
        # [FUNCTION DOC] Checks the length of an empty dict and a two-item dict.
        d = {}
        self.assertEqual(len(d), 0)
        d = {'a': 1, 'b': 2}
        self.assertEqual(len(d), 2)

    def test_getitem(self):
        # [FUNCTION DOC] Checks item get, set and delete, TypeError for a missing argument,
        # KeyError lookup against a key whose equality raises, and propagation of an exception
        # raised by a key's hash.
        d = {'a': 1, 'b': 2}
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        d['c'] = 3
        d['a'] = 4
        self.assertEqual(d['c'], 3)
        self.assertEqual(d['a'], 4)
        del d['b']
        self.assertEqual(d, {'a': 4, 'c': 3})

        self.assertRaises(TypeError, d.__getitem__)

        class BadEq(object):
            # [CLASS DOC] HELPER: key class with constant hash 24 whose equality
            # comparison raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns constant hash 24.
                return 24

        d = {}
        d[BadEq()] = 42
        self.assertRaises(KeyError, d.__getitem__, 23)

        class Exc(Exception): pass

        class BadHash(object):
            # [CLASS DOC] HELPER: key class whose hash method raises Exc when the fail flag is
            # set, else returns 42.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 42, or raises Exc when the instance fail
                # flag is true.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.__getitem__, x)

    def test_clear(self):
        # [FUNCTION DOC] Checks clear empties the dict and raises TypeError when
        # given an argument.
        d = {1:1, 2:2, 3:3}
        d.clear()
        self.assertEqual(d, {})

        self.assertRaises(TypeError, d.clear, None)

    def test_update(self):
        # [FUNCTION DOC] Checks update with dicts and no arguments, TypeError or AttributeError
        # for None, support for objects exposing keys and item access, and that exceptions from
        # keys, from the key iterator, from item access, from a failing sequence, and a
        # ValueError for a wrong-length pair are propagated.
        d = {}
        d.update({1:100})
        d.update({2:20})
        d.update({1:1, 2:2, 3:3})
        self.assertEqual(d, {1:1, 2:2, 3:3})

        d.update()
        self.assertEqual(d, {1:1, 2:2, 3:3})

        self.assertRaises((TypeError, AttributeError), d.update, None)

        class SimpleUserDict:
            # [CLASS DOC] HELPER: minimal mapping-like object wrapping a dict with keys
            # and item access.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the wrapped dict with entries 1, 2, 3.
                self.d = {1:1, 2:2, 3:3}
            def keys(self):
                # [FUNCTION DOC] HELPER: returns the keys of the wrapped dict.
                return self.d.keys()
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: returns the wrapped value for the given key.
                return self.d[i]
        d.clear()
        d.update(SimpleUserDict())
        self.assertEqual(d, {1:1, 2:2, 3:3})

        class Exc(Exception): pass

        d.clear()

        class FailingUserDict:
            # [CLASS DOC] HELPER: mapping-like object whose keys method raises Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            # [CLASS DOC] HELPER: mapping-like object whose keys iterator yields one key
            # then raises Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: returns a one-shot iterator that yields 'a' once and
                # then raises Exc.
                class BogonIter:
                    # [CLASS DOC] HELPER: iterator that yields 'a' once and then raises Exc.
                    def __init__(self):
                        # [FUNCTION DOC] HELPER: sets a counter so exactly one item is produced.
                        self.i = 1
                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: returns itself as the iterator.
                        return self
                    def __next__(self):
                        # [FUNCTION DOC] HELPER: returns 'a' on the first call, then raises Exc.
                        if self.i:
                            self.i = 0
                            return 'a'
                        raise Exc
                return BogonIter()
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: returns the key itself as its value.
                return key
        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            # [CLASS DOC] HELPER: mapping-like object whose keys are letters a to z but whose
            # item access raises Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: returns an iterator over the letters a to z.
                class BogonIter:
                    # [CLASS DOC] HELPER: iterator over the letters a to z.
                    def __init__(self):
                        # [FUNCTION DOC] HELPER: starts the position at the code of 'a'.
                        self.i = ord('a')
                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: returns itself as the iterator.
                        return self
                    def __next__(self):
                        # [FUNCTION DOC] HELPER: returns the next letter up to 'z',
                        # then raises StopIteration.
                        if self.i <= ord('z'):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration
                return BogonIter()
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())


        class badseq(object):
            # [CLASS DOC] HELPER: iterable whose iteration raises Exc on the first step.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: returns itself as the iterator.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()

        self.assertRaises(Exc, {}.update, badseq())

        self.assertRaises(ValueError, {}.update, [(1, 2, 3)])

    def test_fromkeys(self):
        # [FUNCTION DOC] Checks fromkeys: default None values, explicit value, empty and
        # generator inputs, TypeError for non-iterables, subclass and __new__-overriding
        # constructors, exceptions from constructor, iteration and setitem, and the fast paths
        # for dict and set inputs versus the slow path for other iterables.
        self.assertEqual(dict.fromkeys('abc'), {'a':None, 'b':None, 'c':None})
        d = {}
        self.assertIsNot(d.fromkeys('abc'), d)
        self.assertEqual(d.fromkeys('abc'), {'a':None, 'b':None, 'c':None})
        self.assertEqual(d.fromkeys((4,5),0), {4:0, 5:0})
        self.assertEqual(d.fromkeys([]), {})
        def g():
            # [FUNCTION DOC] HELPER: generator yielding the single value 1.
            yield 1
        self.assertEqual(d.fromkeys(g()), {1:None})
        self.assertRaises(TypeError, {}.fromkeys, 3)
        class dictlike(dict): pass
        self.assertEqual(dictlike.fromkeys('a'), {'a':None})
        self.assertEqual(dictlike().fromkeys('a'), {'a':None})
        self.assertIsInstance(dictlike.fromkeys('a'), dictlike)
        self.assertIsInstance(dictlike().fromkeys('a'), dictlike)
        class mydict(dict):
            # [CLASS DOC] HELPER: dict subclass whose __new__ returns a UserDict.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: ignores the class and returns a new UserDict.
                return collections.UserDict()
        ud = mydict.fromkeys('ab')
        self.assertEqual(ud, {'a':None, 'b':None})
        self.assertIsInstance(ud, collections.UserDict)
        self.assertRaises(TypeError, dict.fromkeys)

        class Exc(Exception): pass

        class baddict1(dict):
            # [CLASS DOC] HELPER: dict subclass whose constructor raises Exc.
            def __init__(self):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()

        self.assertRaises(Exc, baddict1.fromkeys, [1])

        class BadSeq(object):
            # [CLASS DOC] HELPER: iterable whose iteration raises Exc on the first step.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: returns itself as the iterator.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()

        self.assertRaises(Exc, dict.fromkeys, BadSeq())

        class baddict2(dict):
            # [CLASS DOC] HELPER: dict subclass whose setitem raises Exc.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()

        self.assertRaises(Exc, baddict2.fromkeys, [1])

        # test fast path for dictionary inputs
        res = dict(zip(range(6), [0]*6))
        d = dict(zip(range(6), range(6)))
        self.assertEqual(dict.fromkeys(d, 0), res)
        # test fast path for set inputs
        d = set(range(6))
        self.assertEqual(dict.fromkeys(d, 0), res)
        # test slow path for other iterable inputs
        d = list(range(6))
        self.assertEqual(dict.fromkeys(d, 0), res)

        # test fast path when object's constructor returns large non-empty dict
        class baddict3(dict):
            # [CLASS DOC] HELPER: dict subclass whose __new__ returns an existing
            # large non-empty dict.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: returns the enclosing test's dict d instead of
                # a new instance.
                return d
        d = {i : i for i in range(1000)}
        res = d.copy()
        res.update(a=None, b=None, c=None)
        self.assertEqual(baddict3.fromkeys({"a", "b", "c"}), res)

        # test slow path when object is a proper subclass of dict
        class baddict4(dict):
            # [CLASS DOC] HELPER: dict subclass whose constructor fills itself from the
            # enclosing test's dict d.
            def __init__(self):
                # [FUNCTION DOC] HELPER: initializes the base dict from d.
                dict.__init__(self, d)
        d = {i : i for i in range(1000)}
        res = d.copy()
        res.update(a=None, b=None, c=None)
        self.assertEqual(baddict4.fromkeys({"a", "b", "c"}), res)

    def test_copy(self):
        # [FUNCTION DOC] Checks copy returns an equal, distinct, independent dict and raises
        # TypeError with an argument.
        d = {1: 1, 2: 2, 3: 3}
        self.assertIsNot(d.copy(), d)
        self.assertEqual(d.copy(), d)
        self.assertEqual(d.copy(), {1: 1, 2: 2, 3: 3})

        copy = d.copy()
        d[4] = 4
        self.assertNotEqual(copy, d)

        self.assertEqual({}.copy(), {})
        self.assertRaises(TypeError, d.copy, None)

    def test_copy_fuzz(self):
        # [FUNCTION DOC] Copies randomly sized dicts of growing magnitude and checks the copy is
        # distinct, equal, and independent when modified. Uses random state.
        for dict_size in [10, 100, 1000, 10000, 100000]:
            dict_size = random.randrange(
                dict_size // 2, dict_size + dict_size // 2)
            with self.subTest(dict_size=dict_size):
                d = {}
                for i in range(dict_size):
                    d[i] = i

                d2 = d.copy()
                self.assertIsNot(d2, d)
                self.assertEqual(d, d2)
                d2['key'] = 'value'
                self.assertNotEqual(d, d2)
                self.assertEqual(len(d2), len(d) + 1)

    def test_copy_maintains_tracking(self):
        # [FUNCTION DOC] Checks that copying keeps the garbage-collector tracking state of
        # the original dict.
        class A:
            # [CLASS DOC] HELPER: empty class whose instance is a trackable dict key.
            pass

        key = A()

        for d in ({}, {'a': 1}, {key: 'val'}):
            d2 = d.copy()
            self.assertEqual(gc.is_tracked(d), gc.is_tracked(d2))

    def test_copy_noncompact(self):
        # [FUNCTION DOC] Checks copy of a dict with many deleted entries (compacting slow path)
        # equals the original.
        # Dicts don't compact themselves on del/pop operations.
        # Copy will use a slow merging strategy that produces
        # a compacted copy when roughly 33% of dict is a non-used
        # keys-space (to optimize memory footprint).
        # In this test we want to hit the slow/compacting
        # branch of dict.copy() and make sure it works OK.
        d = {k: k for k in range(1000)}
        for k in range(950):
            del d[k]
        d2 = d.copy()
        self.assertEqual(d2, d)

    def test_get(self):
        # [FUNCTION DOC] Checks get with and without default, and TypeError for
        # wrong argument counts.
        d = {}
        self.assertIs(d.get('c'), None)
        self.assertEqual(d.get('c', 3), 3)
        d = {'a': 1, 'b': 2}
        self.assertIs(d.get('c'), None)
        self.assertEqual(d.get('c', 3), 3)
        self.assertEqual(d.get('a'), 1)
        self.assertEqual(d.get('a', 3), 1)
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        # [FUNCTION DOC] Checks setdefault inserting and returning defaults, in-
        # place mutation of the stored value, TypeError without arguments, and propagation of a
        # key hash exception.
        # dict.setdefault()
        d = {}
        self.assertIs(d.setdefault('key0'), None)
        d.setdefault('key0', [])
        self.assertIs(d.setdefault('key0'), None)
        d.setdefault('key', []).append(3)
        self.assertEqual(d['key'][0], 3)
        d.setdefault('key', []).append(4)
        self.assertEqual(len(d['key']), 2)
        self.assertRaises(TypeError, d.setdefault)


        class Exc(Exception): pass

        class BadHash(object):
            # [CLASS DOC] HELPER: key class whose hash method raises Exc when the fail flag is
            # set, else returns 42.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 42, or raises Exc when the instance fail
                # flag is true.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.setdefault, x, [])

    def test_setdefault_atomic(self):
        # [FUNCTION DOC] Checks setdefault calls hash once per key and equality only once
        # overall (issue 13521).
        # Issue #13521: setdefault() calls __hash__ and __eq__ only once.
        class Hashed(object):
            # [CLASS DOC] HELPER: key counting hash and equality calls, with constant hash 42.
            def __init__(self):
                # [FUNCTION DOC] HELPER: initializes hash_count and eq_count counters to zero.
                self.hash_count = 0
                self.eq_count = 0
            def __hash__(self):
                # [FUNCTION DOC] HELPER: increments hash_count and returns constant 42.
                self.hash_count += 1
                return 42
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: increments eq_count and returns identity equality with
                # the other object.
                self.eq_count += 1
                return id(self) == id(other)
        hashed1 = Hashed()
        y = {hashed1: 5}
        hashed2 = Hashed()
        y.setdefault(hashed2, [])
        self.assertEqual(hashed1.hash_count, 1)
        self.assertEqual(hashed2.hash_count, 1)
        self.assertEqual(hashed1.eq_count + hashed2.eq_count, 1)

    def test_setitem_atomic_at_resize(self):
        # [FUNCTION DOC] Checks that inserting a key that triggers a resize calls hash once per
        # key and equality only once overall.
        class Hashed(object):
            # [CLASS DOC] HELPER: key counting hash and equality calls, with constant hash 42.
            def __init__(self):
                # [FUNCTION DOC] HELPER: initializes hash_count and eq_count counters to zero.
                self.hash_count = 0
                self.eq_count = 0
            def __hash__(self):
                # [FUNCTION DOC] HELPER: increments hash_count and returns constant 42.
                self.hash_count += 1
                return 42
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: increments eq_count and returns identity equality with
                # the other object.
                self.eq_count += 1
                return id(self) == id(other)
        hashed1 = Hashed()
        # 5 items
        y = {hashed1: 5, 0: 0, 1: 1, 2: 2, 3: 3}
        hashed2 = Hashed()
        # 6th item forces a resize
        y[hashed2] = []
        self.assertEqual(hashed1.hash_count, 1)
        self.assertEqual(hashed2.hash_count, 1)
        self.assertEqual(hashed1.eq_count + hashed2.eq_count, 1)

    def test_popitem(self):
        # [FUNCTION DOC] Checks popitem drains dicts of several sizes (original and copy),
        # returning consistent pairs, and raises KeyError on an empty dict.
        # dict.popitem()
        for copymode in -1, +1:
            # -1: b has same structure as a
            # +1: b is a.copy()
            for log2size in range(4):
                size = 2**log2size
                a = {}
                b = {}
                for i in range(size):
                    a[repr(i)] = i
                    if copymode < 0:
                        b[repr(i)] = i
                if copymode > 0:
                    b = a.copy()
                for i in range(size):
                    ka, va = ta = a.popitem()
                    self.assertEqual(va, int(ka))
                    kb, vb = tb = b.popitem()
                    self.assertEqual(vb, int(kb))
                    self.assertFalse(copymode < 0 and ta != tb)
                self.assertFalse(a)
                self.assertFalse(b)

        d = {}
        self.assertRaises(KeyError, d.popitem)

    def test_pop(self):
        # [FUNCTION DOC] Checks pop with and without default, KeyError for missing keys,
        # TypeError without arguments, and propagation of a key hash exception.
        # Tests for pop with specified key
        d = {}
        k, v = 'abc', 'def'
        d[k] = v
        self.assertRaises(KeyError, d.pop, 'ghi')

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)

        self.assertEqual(d.pop(k, v), v)
        d[k] = v
        self.assertEqual(d.pop(k, 1), v)

        self.assertRaises(TypeError, d.pop)

        class Exc(Exception): pass

        class BadHash(object):
            # [CLASS DOC] HELPER: key class whose hash method raises Exc when the fail flag is
            # set, else returns 42.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 42, or raises Exc when the instance fail
                # flag is true.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.pop, x)

    def test_mutating_iteration(self):
        # [FUNCTION DOC] Checks that growing a dict while iterating it raises RuntimeError.
        # changing dict size during iteration
        d = {}
        d[1] = 1
        with self.assertRaises(RuntimeError):
            for i in d:
                d[i+1] = 1

    def test_mutating_iteration_delete(self):
        # [FUNCTION DOC] Checks that deleting and re-adding a key during key
        # iteration raises RuntimeError.
        # change dict content during iteration
        d = {}
        d[0] = 0
        with self.assertRaises(RuntimeError):
            for i in d:
                del d[0]
                d[0] = 0

    def test_mutating_iteration_delete_over_values(self):
        # [FUNCTION DOC] Checks that deleting and re-adding a key during value
        # iteration raises RuntimeError.
        # change dict content during iteration
        d = {}
        d[0] = 0
        with self.assertRaises(RuntimeError):
            for i in d.values():
                del d[0]
                d[0] = 0

    def test_mutating_iteration_delete_over_items(self):
        # [FUNCTION DOC] Checks that deleting and re-adding a key during item
        # iteration raises RuntimeError.
        # change dict content during iteration
        d = {}
        d[0] = 0
        with self.assertRaises(RuntimeError):
            for i in d.items():
                del d[0]
                d[0] = 0

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_mutating_lookup(self):
        # [FUNCTION DOC] Checks that a dict mutated from inside a key equality call
        # during a colliding insertion ends with only the new key. Skipped under Dynamo (class
        # with closed-over objects).
        # changing dict during a lookup (issue #14417)
        class NastyKey:
            # [CLASS DOC] HELPER: key with constant hash whose equality can delete another entry
            # from a dict stored in the class attribute mutate_dict, once.
            mutate_dict = None

            def __init__(self, value):
                # [FUNCTION DOC] HELPER: stores the key value.
                self.value = value

            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns constant 1 to force a hash collision.
                # hash collision!
                return 1

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: if a pending dict and key are set, clears the setting
                # and deletes that key from the dict; then compares values.
                if NastyKey.mutate_dict:
                    mydict, key = NastyKey.mutate_dict
                    NastyKey.mutate_dict = None
                    del mydict[key]
                return self.value == other.value

        key1 = NastyKey(1)
        key2 = NastyKey(2)
        d = {key1: 1}
        NastyKey.mutate_dict = (d, key1)
        d[key2] = 2
        self.assertEqual(d, {key2: 2})

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of empty, simple and self-referential dicts, and that an
        # exception from an element repr propagates.
        d = {}
        self.assertEqual(repr(d), '{}')
        d[1] = 2
        self.assertEqual(repr(d), '{1: 2}')
        d = {}
        d[1] = d
        self.assertEqual(repr(d), '{1: {...}}')

        class Exc(Exception): pass

        class BadRepr(object):
            # [CLASS DOC] HELPER: object whose repr raises Exc.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()

        d = {1: BadRepr()}
        self.assertRaises(Exc, repr, d)

    def test_repr_deep(self):
        # [FUNCTION DOC] Checks repr of a deeply nested dict raises RecursionError.
        d = {}
        for i in range(get_c_recursion_limit() + 1):
            d = {1: d}
        self.assertRaises(RecursionError, repr, d)

    def test_eq(self):
        # [FUNCTION DOC] Checks dict equality for empty and simple dicts, and that an exception
        # from element equality propagates.
        self.assertEqual({}, {})
        self.assertEqual({1: 2}, {1: 2})

        class Exc(Exception): pass

        class BadCmp(object):
            # [CLASS DOC] HELPER: key with hash 1 whose equality raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc()
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns constant 1.
                return 1

        d1 = {BadCmp(): 1}
        d2 = {1: 1}

        with self.assertRaises(Exc):
            d1 == d2

    def test_keys_contained(self):
        # [FUNCTION DOC] Runs the view comparison helper against both keys views
        # and items views.
        self.helper_keys_contained(lambda x: x.keys())
        self.helper_keys_contained(lambda x: x.items())

    def helper_keys_contained(self, fn):
        # [FUNCTION DOC] HELPER: given a function producing a dict view, checks ordering and
        # equality comparisons between views behave like set comparisons (subset, superset,
        # strictness, empty views, equal-size elementwise comparison).
        # Test rich comparisons against dict key views, which should behave the
        # same as sets.
        empty = fn(dict())
        empty2 = fn(dict())
        smaller = fn({1:1, 2:2})
        larger = fn({1:1, 2:2, 3:3})
        larger2 = fn({1:1, 2:2, 3:3})
        larger3 = fn({4:1, 2:2, 3:3})

        self.assertTrue(smaller <  larger)
        self.assertTrue(smaller <= larger)
        self.assertTrue(larger >  smaller)
        self.assertTrue(larger >= smaller)

        self.assertFalse(smaller >= larger)
        self.assertFalse(smaller >  larger)
        self.assertFalse(larger  <= smaller)
        self.assertFalse(larger  <  smaller)

        self.assertFalse(smaller <  larger3)
        self.assertFalse(smaller <= larger3)
        self.assertFalse(larger3 >  smaller)
        self.assertFalse(larger3 >= smaller)

        # Inequality strictness
        self.assertTrue(larger2 >= larger)
        self.assertTrue(larger2 <= larger)
        self.assertFalse(larger2 > larger)
        self.assertFalse(larger2 < larger)

        self.assertTrue(larger == larger2)
        self.assertTrue(smaller != larger)

        # There is an optimization on the zero-element case.
        self.assertTrue(empty == empty2)
        self.assertFalse(empty != empty2)
        self.assertFalse(empty == smaller)
        self.assertTrue(empty != smaller)

        # With the same size, an elementwise compare happens
        self.assertTrue(larger != larger3)
        self.assertFalse(larger == larger3)

    def test_errors_in_view_containment_check(self):
        # [FUNCTION DOC] Checks that exceptions raised by element equality are propagated by all
        # comparisons of items views.
        class C:
            # [CLASS DOC] HELPER: value whose equality raises RuntimeError.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raises RuntimeError.
                raise RuntimeError

        d1 = {1: C()}
        d2 = {1: C()}
        with self.assertRaises(RuntimeError):
            d1.items() == d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() != d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() <= d2.items()
        with self.assertRaises(RuntimeError):
            d1.items() >= d2.items()

        d3 = {1: C(), 2: C()}
        with self.assertRaises(RuntimeError):
            d2.items() < d3.items()
        with self.assertRaises(RuntimeError):
            d3.items() > d2.items()

    def test_dictview_set_operations_on_keys(self):
        # [FUNCTION DOC] Checks difference, intersection, union and symmetric difference
        # between keys views.
        k1 = {1:1, 2:2}.keys()
        k2 = {1:1, 2:2, 3:3}.keys()
        k3 = {4:4}.keys()

        self.assertEqual(k1 - k2, set())
        self.assertEqual(k1 - k3, {1,2})
        self.assertEqual(k2 - k1, {3})
        self.assertEqual(k3 - k1, {4})
        self.assertEqual(k1 & k2, {1,2})
        self.assertEqual(k1 & k3, set())
        self.assertEqual(k1 | k2, {1,2,3})
        self.assertEqual(k1 ^ k2, {3})
        self.assertEqual(k1 ^ k3, {1,2,4})

    def test_dictview_set_operations_on_items(self):
        # [FUNCTION DOC] Checks difference, intersection, union and symmetric difference
        # between items views.
        k1 = {1:1, 2:2}.items()
        k2 = {1:1, 2:2, 3:3}.items()
        k3 = {4:4}.items()

        self.assertEqual(k1 - k2, set())
        self.assertEqual(k1 - k3, {(1,1), (2,2)})
        self.assertEqual(k2 - k1, {(3,3)})
        self.assertEqual(k3 - k1, {(4,4)})
        self.assertEqual(k1 & k2, {(1,1), (2,2)})
        self.assertEqual(k1 & k3, set())
        self.assertEqual(k1 | k2, {(1,1), (2,2), (3,3)})
        self.assertEqual(k1 ^ k2, {(3,3)})
        self.assertEqual(k1 ^ k3, {(1,1), (2,2), (4,4)})

    def test_items_symmetric_difference(self):
        # [FUNCTION DOC] Checks symmetric difference of items views against the set result on
        # random dicts. Uses random state.
        rr = random.randrange
        for _ in range(100):
            left = {x:rr(3) for x in range(20) if rr(2)}
            right = {x:rr(3) for x in range(20) if rr(2)}
            with self.subTest(left=left, right=right):
                expected = set(left.items()) ^ set(right.items())
                actual = left.items() ^ right.items()
                self.assertEqual(actual, expected)

    def test_dictview_mixed_set_operations(self):
        # [FUNCTION DOC] Checks comparisons and unions between dict views and plain sets.
        # Just a few for .keys()
        self.assertTrue({1:1}.keys() == {1})
        self.assertTrue({1} == {1:1}.keys())
        self.assertEqual({1:1}.keys() | {2}, {1, 2})
        self.assertEqual({2} | {1:1}.keys(), {1, 2})
        # And a few for .items()
        self.assertTrue({1:1}.items() == {(1,1)})
        self.assertTrue({(1,1)} == {1:1}.items())
        self.assertEqual({1:1}.items() | {2}, {(1,1), 2})
        self.assertEqual({2} | {1:1}.items(), {(1,1), 2})

    def test_missing(self):
        # [FUNCTION DOC] Checks the __missing__ hook: dict has none; a subclass hook supplies
        # values, raises, is ignored when set on the instance, or is absent giving KeyError
        # carrying the key.
        # Make sure dict doesn't have a __missing__ method
        self.assertFalse(hasattr(dict, "__missing__"))
        self.assertFalse(hasattr({}, "__missing__"))
        # Test several cases:
        # (D) subclass defines __missing__ method returning a value
        # (E) subclass defines __missing__ method raising RuntimeError
        # (F) subclass sets __missing__ instance variable (no effect)
        # (G) subclass doesn't define __missing__ at all
        class D(dict):
            # [CLASS DOC] HELPER: dict subclass whose __missing__ returns 42.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: returns 42 for any missing key.
                return 42
        d = D({1: 2, 3: 4})
        self.assertEqual(d[1], 2)
        self.assertEqual(d[3], 4)
        self.assertNotIn(2, d)
        self.assertNotIn(2, d.keys())
        self.assertEqual(d[2], 42)

        class E(dict):
            # [CLASS DOC] HELPER: dict subclass whose __missing__ raises RuntimeError.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: raises RuntimeError with the key.
                raise RuntimeError(key)
        e = E()
        with self.assertRaises(RuntimeError) as c:
            e[42]
        self.assertEqual(c.exception.args, (42,))

        class F(dict):
            # [CLASS DOC] HELPER: dict subclass that sets an instance attribute __missing__,
            # which must have no effect.
            def __init__(self):
                # [FUNCTION DOC] HELPER: assigns an instance-level __missing__
                # function returning None.
                # An instance variable __missing__ should have no effect
                self.__missing__ = lambda key: None
        f = F()
        with self.assertRaises(KeyError) as c:
            f[42]
        self.assertEqual(c.exception.args, (42,))

        class G(dict):
            # [CLASS DOC] HELPER: plain dict subclass with no __missing__.
            pass
        g = G()
        with self.assertRaises(KeyError) as c:
            g[42]
        self.assertEqual(c.exception.args, (42,))

    def test_tuple_keyerror(self):
        # [FUNCTION DOC] Checks a missing tuple key raises KeyError whose argument is the tuple.
        # SF #1576657
        d = {}
        with self.assertRaises(KeyError) as c:
            d[(1,)]
        self.assertEqual(c.exception.args, ((1,),))

    def test_bad_key(self):
        # [FUNCTION DOC] Checks that lookups, assignment, contains, get, setdefault, pop and
        # update all propagate an exception raised by key equality. Uses dynamo
        # error_on_graph_break disabled around exec.
        # Dictionary lookups should fail if __eq__() raises an exception.
        class CustomException(Exception):
            # [CLASS DOC] HELPER: exception type raised by the bad key.
            pass

        class BadDictKey:
            # [CLASS DOC] HELPER: key with class-based hash whose equality against another
            # instance raises CustomException.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns the hash of the class.
                return hash(self.__class__)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: raises CustomException against instances of the class,
                # else returns the other object.
                if isinstance(other, self.__class__):
                    raise CustomException
                return other

        d = {}
        x1 = BadDictKey()
        x2 = BadDictKey()
        d[x1] = 1
        for stmt in ['d[x2] = 2',
                     'z = d[x2]',
                     'x2 in d',
                     'd.get(x2)',
                     'd.setdefault(x2, 42)',
                     'd.pop(x2)',
                     'd.update({x2: 2})']:
            with self.assertRaises(CustomException):
                with torch._dynamo.error_on_graph_break(False):
                    exec(stmt, locals())

    def test_resize1(self):
        # [FUNCTION DOC] Regression test for a dict resize bug: inserts, deletes, and re-inserts
        # keys; passes if nothing crashes or hangs.
        # Dict resizing bug, found by Jack Jansen in 2.2 CVS development.
        # This version got an assert failure in debug build, infinite loop in
        # release build.  Unfortunately, provoking this kind of stuff requires
        # a mix of inserts and deletes hitting exactly the right hash codes in
        # exactly the right order, and I can't think of a randomized approach
        # that would be *likely* to hit a failing case in reasonable time.

        d = {}
        for i in range(5):
            d[i] = i
        for i in range(5):
            del d[i]
        for i in range(5, 9):  # i==8 was the problem
            d[i] = i

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_resize2(self):
        # [FUNCTION DOC] Regression test for a resize bug: a key whose equality clears the dict
        # during resize must not crash. Skipped under Dynamo.
        # Another dict resizing bug (SF bug #1456209).
        # This caused Segmentation faults or Illegal instructions.

        class X(object):
            # [CLASS DOC] HELPER: key with constant hash 5 whose equality clears the enclosing
            # dict d when the enclosing flag resizing is set.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns constant 5.
                return 5
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears d if resizing is set; always returns False.
                if resizing:
                    d.clear()
                return False
        d = {}
        resizing = False
        d[X()] = 1
        d[X()] = 2
        d[X()] = 3
        d[X()] = 4
        d[X()] = 5
        # now trigger a resize
        resizing = True
        d[9] = 6

    def test_empty_presized_dict_in_freelist(self):
        # [FUNCTION DOC] Regression test: building a presized dict literal that raises
        # ZeroDivisionError midway must not hit an assertion.
        # Bug #3537: if an empty but presized dict with a size larger
        # than 7 was in the freelist, it triggered an assertion failure
        with self.assertRaises(ZeroDivisionError):
            d = {'a': 1 // 0, 'b': None, 'c': None, 'd': None, 'e': None,
                 'f': None, 'g': None, 'h': None}
        d = {}

    def test_container_iterator(self):
        # [FUNCTION DOC] Checks that cycles between an object and dict views or iterators over a
        # dict containing it are collected by the garbage collector.
        # Bug #3680: tp_traverse was not implemented for dictiter and
        # dictview objects.
        class C(object):
            # [CLASS DOC] HELPER: empty class for creating a collectable object.
            pass
        views = (dict.items, dict.values, dict.keys)
        for v in views:
            obj = C()
            ref = weakref.ref(obj)
            container = {obj: 1}
            obj.v = v(container)
            obj.x = iter(obj.v)
            del obj, container
            gc.collect()
            self.assertIs(ref(), None, "Cycle was not collected")

    def _not_tracked(self, t):
        # [FUNCTION DOC] HELPER: collects garbage twice, then asserts the object is not tracked
        # by the garbage collector.
        # Nested containers can take several collections to untrack
        gc.collect()
        gc.collect()
        self.assertFalse(gc.is_tracked(t), t)

    def _tracked(self, t):
        # [FUNCTION DOC] HELPER: asserts the object is tracked by the garbage collector, before
        # and after two collections.
        self.assertTrue(gc.is_tracked(t), t)
        gc.collect()
        gc.collect()
        self.assertTrue(gc.is_tracked(t), t)

    def test_string_keys_can_track_values(self):
        # [FUNCTION DOC] Builds dicts whose values include themselves with string keys; passes
        # if nothing leaks or crashes.
        # Test that this doesn't leak.
        for i in range(10):
            d = {}
            for j in range(10):
                d[str(j)] = j
            d["foo"] = d

    @support.cpython_only
    def test_track_literals(self):
        # [FUNCTION DOC] Checks garbage-collector tracking of dict literals: untracked when
        # holding only immutable values, tracked when holding mutable ones. CPython only.
        # Test GC-optimization of dict literals
        x, y, z, w = 1.5, "a", (1, None), []

        self._not_tracked({})
        self._not_tracked({x:(), y:x, z:1})
        self._not_tracked({1: "a", "b": 2})
        self._not_tracked({1: 2, (None, True, False, ()): int})
        self._not_tracked({1: object()})

        # Dicts with mutable elements are always tracked, even if those
        # elements are not tracked right now.
        self._tracked({1: []})
        self._tracked({1: ([],)})
        self._tracked({1: {}})
        self._tracked({1: set()})

    @support.cpython_only
    def test_track_dynamic(self):
        # [FUNCTION DOC] Checks tracking of dynamically built dicts through setitem, copy,
        # fromkeys, update, constructors and nested dicts. CPython only.
        # Test GC-optimization of dynamically-created dicts

        class MyObject(object):
            # [CLASS DOC] HELPER: empty class for creating a trackable object.
            pass
        x, y, z, w, o = 1.5, "a", (1, object()), [], MyObject()

        d = dict()
        self._not_tracked(d)
        d[1] = "a"
        self._not_tracked(d)
        d[y] = 2
        self._not_tracked(d)
        d[z] = 3
        self._not_tracked(d)
        self._not_tracked(d.copy())
        d[4] = w
        self._tracked(d)
        self._tracked(d.copy())
        d[4] = None
        self._not_tracked(d)
        self._not_tracked(d.copy())

        # dd isn't tracked right now, but it may mutate and therefore d
        # which contains it must be tracked.
        d = dict()
        dd = dict()
        d[1] = dd
        self._not_tracked(dd)
        self._tracked(d)
        dd[1] = d
        self._tracked(dd)

        d = dict.fromkeys([x, y, z])
        self._not_tracked(d)
        dd = dict()
        dd.update(d)
        self._not_tracked(dd)
        d = dict.fromkeys([x, y, z, o])
        self._tracked(d)
        dd = dict()
        dd.update(d)
        self._tracked(dd)

        d = dict(x=x, y=y, z=z)
        self._not_tracked(d)
        d = dict(x=x, y=y, z=z, w=w)
        self._tracked(d)
        d = dict()
        d.update(x=x, y=y, z=z)
        self._not_tracked(d)
        d.update(w=w)
        self._tracked(d)

        d = dict([(x, y), (z, 1)])
        self._not_tracked(d)
        d = dict([(x, y), (z, w)])
        self._tracked(d)
        d = dict()
        d.update([(x, y), (z, 1)])
        self._not_tracked(d)
        d.update([(x, y), (z, w)])
        self._tracked(d)

    @support.cpython_only
    def test_track_subtypes(self):
        # [FUNCTION DOC] Checks dict subclass instances are always tracked. CPython only.
        # Dict subtypes are always tracked
        class MyDict(dict):
            # [CLASS DOC] HELPER: empty dict subclass.
            pass
        self._tracked(MyDict())

    def make_shared_key_dict(self, n):
        # [FUNCTION DOC] HELPER: returns a list of n instance dicts that share keys, each with
        # attributes x, y, z set to 1, 2, 3.
        class C:
            # [CLASS DOC] HELPER: empty class whose instance dicts share keys.
            pass

        dicts = []
        for i in range(n):
            a = C()
            a.x, a.y, a.z = 1, 2, 3
            dicts.append(a.__dict__)

        return dicts

    @support.cpython_only
    def test_splittable_setdefault(self):
        # [FUNCTION DOC] Checks that setdefault on a shared-key dict keeps the correct per-dict
        # insertion order. CPython only.
        """split table must keep correct insertion
        order when attributes are adding using setdefault()"""
        a, b = self.make_shared_key_dict(2)

        a['a'] = 1
        size_a = sys.getsizeof(a)
        a['b'] = 2
        b.setdefault('b', 2)
        size_b = sys.getsizeof(b)
        b['a'] = 1

        self.assertEqual(list(a), ['x', 'y', 'z', 'a', 'b'])
        self.assertEqual(list(b), ['x', 'y', 'z', 'b', 'a'])

    @support.cpython_only
    def test_splittable_del(self):
        # [FUNCTION DOC] Checks deleting a key from a shared-key dict combines the table without
        # affecting the sibling dict. CPython only.
        """split table must be combined when del d[k]"""
        a, b = self.make_shared_key_dict(2)

        orig_size = sys.getsizeof(a)

        del a['y']  # split table is combined
        with self.assertRaises(KeyError):
            del a['y']

        self.assertEqual(list(a), ['x', 'z'])
        self.assertEqual(list(b), ['x', 'y', 'z'])

        # Two dicts have different insertion order.
        a['y'] = 42
        self.assertEqual(list(a), ['x', 'z', 'y'])
        self.assertEqual(list(b), ['x', 'y', 'z'])

    @support.cpython_only
    def test_splittable_pop(self):
        # [FUNCTION DOC] Checks pop on a shared-key dict, KeyError for a repeated pop, and
        # independent insertion orders. CPython only.
        a, b = self.make_shared_key_dict(2)

        a.pop('y')
        with self.assertRaises(KeyError):
            a.pop('y')

        self.assertEqual(list(a), ['x', 'z'])
        self.assertEqual(list(b), ['x', 'y', 'z'])

        # Two dicts have different insertion order.
        a['y'] = 42
        self.assertEqual(list(a), ['x', 'z', 'y'])
        self.assertEqual(list(b), ['x', 'y', 'z'])

    @support.cpython_only
    def test_splittable_pop_pending(self):
        # [FUNCTION DOC] Checks popping a key that is pending (added only to another dict)
        # raises KeyError without crashing. CPython only.
        """pop a pending key in a split table should not crash"""
        a, b = self.make_shared_key_dict(2)

        a['a'] = 4
        with self.assertRaises(KeyError):
            b.pop('a')

    @support.cpython_only
    def test_splittable_popitem(self):
        # [FUNCTION DOC] Checks popitem on a shared-key dict returns the last item and combines
        # the table. CPython only.
        """split table must be combined when d.popitem()"""
        a, b = self.make_shared_key_dict(2)

        orig_size = sys.getsizeof(a)

        item = a.popitem()  # split table is combined
        self.assertEqual(item, ('z', 3))
        with self.assertRaises(KeyError):
            del a['z']

        self.assertGreater(sys.getsizeof(a), orig_size)
        self.assertEqual(list(a), ['x', 'y'])
        self.assertEqual(list(b), ['x', 'y', 'z'])

    @support.cpython_only
    def test_splittable_update(self):
        # [FUNCTION DOC] Checks update copies an instance dict preserving its (reversed) key
        # order. CPython only.
        """dict.update(other) must preserve order in other."""
        class C:
            # [CLASS DOC] HELPER: class whose constructor sets attributes a, b, c in forward
            # or reverse order.
            def __init__(self, order):
                # [FUNCTION DOC] HELPER: sets three attributes in a given order depending
                # on the flag.
                if order:
                    self.a, self.b, self.c = 1, 2, 3
                else:
                    self.c, self.b, self.a = 1, 2, 3
        o = C(True)
        o = C(False)  # o.__dict__ has reversed order.
        self.assertEqual(list(o.__dict__), ["c", "b", "a"])

        d = {}
        d.update(o.__dict__)
        self.assertEqual(list(d), ["c", "b", "a"])

    @support.cpython_only
    def test_splittable_to_generic_combinedtable(self):
        # [FUNCTION DOC] Checks a shared-key dict is converted to a generic table when given a
        # non-string key. CPython only.
        """split table must be correctly resized and converted to generic combined table"""
        class C:
            # [CLASS DOC] HELPER: empty class whose instance dict is used.
            pass

        a = C()
        a.x = 1
        d = a.__dict__
        d[2] = 2 # split table is resized to a generic combined table

        self.assertEqual(list(d), ['x', 2])

    def test_iterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a dict key iterator for all protocols, including after
        # advancing and removing the consumed key.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            it = iter(data)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), list(data))

            it = pickle.loads(d)
            try:
                drop = next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            del data[drop]
            self.assertEqual(list(it), list(data))

    def test_itemiterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a dict item iterator for all protocols,
        # including after advancing.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            # dictviews aren't picklable, only their iterators
            itorg = iter(data.items())
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # note that the type of the unpickled iterator
            # is not necessarily the same as the original.  It is
            # merely an object supporting the iterator protocol, yielding
            # the same objects as the original one.
            # self.assertEqual(type(itorg), type(it))
            self.assertIsInstance(it, collections.abc.Iterator)
            self.assertEqual(dict(it), data)

            it = pickle.loads(d)
            drop = next(it)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            del data[drop[0]]
            self.assertEqual(dict(it), data)

    def test_valuesiterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a dict value iterator for all protocols,
        # including after advancing.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            # data.values() isn't picklable, only its iterator
            it = iter(data.values())
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), list(data.values()))

            it = pickle.loads(d)
            drop = next(it)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            values = list(it) + [drop]
            self.assertEqual(sorted(values), sorted(list(data.values())))

    def test_reverseiterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a reversed key iterator for all protocols,
        # including after advancing.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            it = reversed(data)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), list(reversed(data)))

            it = pickle.loads(d)
            try:
                drop = next(it)
            except StopIteration:
                continue
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            del data[drop]
            self.assertEqual(list(it), list(reversed(data)))

    def test_reverseitemiterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a reversed item iterator for all protocols,
        # including after advancing.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            # dictviews aren't picklable, only their iterators
            itorg = reversed(data.items())
            d = pickle.dumps(itorg, proto)
            it = pickle.loads(d)
            # note that the type of the unpickled iterator
            # is not necessarily the same as the original.  It is
            # merely an object supporting the iterator protocol, yielding
            # the same objects as the original one.
            # self.assertEqual(type(itorg), type(it))
            self.assertIsInstance(it, collections.abc.Iterator)
            self.assertEqual(dict(it), data)

            it = pickle.loads(d)
            drop = next(it)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            del data[drop[0]]
            self.assertEqual(dict(it), data)

    def test_reversevaluesiterator_pickling(self):
        # [FUNCTION DOC] Checks pickling a reversed value iterator for all protocols,
        # including after advancing.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            data = {1:"a", 2:"b", 3:"c"}
            # data.values() isn't picklable, only its iterator
            it = reversed(data.values())
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            self.assertEqual(list(it), list(reversed(data.values())))

            it = pickle.loads(d)
            drop = next(it)
            d = pickle.dumps(it, proto)
            it = pickle.loads(d)
            values = list(it) + [drop]
            self.assertEqual(sorted(values), sorted(data.values()))

    def test_instance_dict_getattr_str_subclass(self):
        # [FUNCTION DOC] Checks attribute lookup and instance dict access work with a
        # str subclass key.
        class Foo:
            # [CLASS DOC] HELPER: class storing a message attribute.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: stores the message.
                self.msg = msg
        f = Foo('123')
        class _str(str):
            # [CLASS DOC] HELPER: empty str subclass.
            pass
        self.assertEqual(f.msg, getattr(f, _str('msg')))
        self.assertEqual(f.msg, f.__dict__[_str('msg')])

    def test_object_set_item_single_instance_non_str_key(self):
        # [FUNCTION DOC] Checks setting a non-string key in an instance dict followed by an
        # attribute set works.
        class Foo: pass
        f = Foo()
        f.__dict__[1] = 1
        f.a = 'a'
        self.assertEqual(f.__dict__, {1:1, 'a':'a'})

    def check_reentrant_insertion(self, mutate):
        # [FUNCTION DOC] HELPER: builds a dict of objects whose destructors call the given
        # mutate function on the dict, then replaces each value, triggering reentrant mutation.
        # Relies on reference counting.
        # This object will trigger mutation of the dict when replaced
        # by another value.  Note this relies on refcounting: the test
        # won't achieve its purpose on fully-GCed Python implementations.
        class Mutating:
            # [CLASS DOC] HELPER: object whose destructor calls the enclosing mutate function on
            # the dict d.
            def __del__(self):
                # [FUNCTION DOC] HELPER: calls mutate on d when destroyed.
                mutate(d)

        d = {k: Mutating() for k in 'abcdefghijklmnopqr'}
        for k in list(d):
            d[k] = k

    def test_reentrant_insertion(self):
        # [FUNCTION DOC] Checks reentrant insertion, update plus clear, and popitem loops during
        # value replacement do not crash.
        # Reentrant insertion shouldn't crash (see issue #22653)
        def mutate(d):
            # [FUNCTION DOC] HELPER: inserts key 'b' into the dict.
            d['b'] = 5
        self.check_reentrant_insertion(mutate)

        def mutate(d):
            # [FUNCTION DOC] HELPER: updates the dict with the test's instance dict,
            # then clears it.
            d.update(self.__dict__)
            d.clear()
        self.check_reentrant_insertion(mutate)

        def mutate(d):
            # [FUNCTION DOC] HELPER: pops items until the dict is empty.
            while d:
                d.popitem()
        self.check_reentrant_insertion(mutate)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_merge_and_mutate(self):
        # [FUNCTION DOC] Checks that update raises RuntimeError when equality of a key clears
        # the source dict mid-update. Skipped under Dynamo.
        class X:
            # [CLASS DOC] HELPER: key with hash 0 whose equality clears the
            # enclosing dict other.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 0.
                return 0

            def __eq__(self, o):
                # [FUNCTION DOC] HELPER: clears other and returns False.
                other.clear()
                return False

        l = [(i,0) for i in range(1, 1337)]
        other = dict(l)
        other[X()] = 0
        d = {X(): 0, 1: 1}
        self.assertRaises(RuntimeError, d.update, other)

    def test_free_after_iterating(self):
        # [FUNCTION DOC] Checks, via the support helper, that dict iterators release their
        # dict after exhaustion.
        support.check_free_after_iterating(self, iter, dict)
        support.check_free_after_iterating(self, lambda d: iter(d.keys()), dict)
        support.check_free_after_iterating(self, lambda d: iter(d.values()), dict)
        support.check_free_after_iterating(self, lambda d: iter(d.items()), dict)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_equal_operator_modifying_operand(self):
        # [FUNCTION DOC] Checks dict equality does not crash when equality or destruction of
        # elements clears an operand (bpo-27945, bpo-38588). Skipped under Dynamo.
        # test fix for seg fault reported in bpo-27945 part 3.
        class X():
            # [CLASS DOC] HELPER: key whose destructor clears dict_b and whose
            # equality clears dict_a.
            def __del__(self):
                # [FUNCTION DOC] HELPER: clears dict_b.
                dict_b.clear()

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears dict_a and returns True.
                dict_a.clear()
                return True

            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 13.
                return 13

        dict_a = {X(): 0}
        dict_b = {X(): X()}
        self.assertTrue(dict_a == dict_b)

        # test fix for seg fault reported in bpo-38588 part 1.
        class Y:
            # [CLASS DOC] HELPER: value whose equality clears dict_d and returns True.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears dict_d and returns True.
                dict_d.clear()
                return True

        dict_c = {0: Y()}
        dict_d = {0: set()}
        self.assertTrue(dict_c == dict_d)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_fromkeys_operator_modifying_dict_operand(self):
        # [FUNCTION DOC] Checks fromkeys on a dict whose int-subclass keys clear it during
        # equality does not crash. Skipped under Dynamo.
        # test fix for seg fault reported in issue 27945 part 4a.
        class X(int):
            # [CLASS DOC] HELPER: int subclass with hash 13 whose equality clears the
            # enclosing dict d.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 13.
                return 13

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears d when it has more than one
                # entry; returns False.
                if len(d) > 1:
                    d.clear()
                return False

        d = {}  # this is required to exist so that d can be constructed!
        d = {X(1): 1, X(2): 2}
        try:
            dict.fromkeys(d)  # shouldn't crash
        except RuntimeError:  # implementation defined
            pass

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_fromkeys_operator_modifying_set_operand(self):
        # [FUNCTION DOC] Checks fromkeys on a set whose int-subclass elements clear the
        # enclosing container during equality does not crash. Skipped under Dynamo.
        # test fix for seg fault reported in issue 27945 part 4b.
        class X(int):
            # [CLASS DOC] HELPER: int subclass with hash 13 whose equality clears
            # the enclosing d.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 13.
                return 13

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears d when it has more than one
                # entry; returns False.
                if len(d) > 1:
                    d.clear()
                return False

        d = {}  # this is required to exist so that d can be constructed!
        d = {X(1), X(2)}
        try:
            dict.fromkeys(d)  # shouldn't crash
        except RuntimeError:  # implementation defined
            pass

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_dictitems_contains_use_after_free(self):
        # [FUNCTION DOC] Checks items view containment is safe when equality clears the dict.
        # Skipped under Dynamo.
        class X:
            # [CLASS DOC] HELPER: value whose equality clears the enclosing dict d
            # and returns NotImplemented.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears d and returns NotImplemented.
                d.clear()
                return NotImplemented

        d = {0: set()}
        (0, X()) in d.items()

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_dict_contain_use_after_free(self):
        # [FUNCTION DOC] Checks dict containment is safe when a str subclass key's equality
        # clears the dict (bpo-40489). Skipped under Dynamo.
        # bpo-40489
        class S(str):
            # [CLASS DOC] HELPER: str subclass whose equality clears the enclosing d and whose
            # hash equals that of 'test'.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: clears d and returns NotImplemented.
                d.clear()
                return NotImplemented

            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns the hash of 'test'.
                return hash('test')

        d = {S(): 'value'}
        self.assertFalse('test' in d)

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_init_use_after_free(self):
        # [FUNCTION DOC] Checks dict construction from a pair whose key hash empties the pair
        # does not crash. Skipped under Dynamo.
        class X:
            # [CLASS DOC] HELPER: key whose hash empties the enclosing list pair and returns 13.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: clears the contents of pair and returns 13.
                pair[:] = []
                return 13

        pair = [X(), 123]
        dict([pair])

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_oob_indexing_dictiter_iternextitem(self):
        # [FUNCTION DOC] Checks the item iterator raises RuntimeError rather
        # than indexing out of bounds when the dict is cleared by a destructor mid-iteration.
        # Skipped under Dynamo.
        class X(int):
            # [CLASS DOC] HELPER: int subclass whose destructor clears the enclosing dict d.
            def __del__(self):
                # [FUNCTION DOC] HELPER: clears d.
                d.clear()

        d = {i: X(i) for i in range(8)}

        def iter_and_mutate():
            # [FUNCTION DOC] HELPER: iterates d items and overwrites the entry for key 2 to
            # free its value.
            for result in d.items():
                if result[0] == 2:
                    d[2] = None # free d[2] --> X(2).__del__ was called

        self.assertRaises(RuntimeError, iter_and_mutate)

    def test_reversed(self):
        # [FUNCTION DOC] Checks a reversed key iterator over a dict with a deleted entry yields
        # keys in reverse order then raises StopIteration.
        d = {"a": 1, "b": 2, "foo": 0, "c": 3, "d": 4}
        del d["foo"]
        r = reversed(d)
        self.assertEqual(list(r), list('dcba'))
        self.assertRaises(StopIteration, next, r)

    def test_reverse_iterator_for_empty_dict(self):
        # [FUNCTION DOC] Checks reversed iteration over empty dicts and their
        # views yields nothing.
        # bpo-38525: reversed iterator should work properly

        # empty dict is directly used for reference count test
        self.assertEqual(list(reversed({})), [])
        self.assertEqual(list(reversed({}.items())), [])
        self.assertEqual(list(reversed({}.values())), [])
        self.assertEqual(list(reversed({}.keys())), [])

        # dict() and {} don't trigger the same code path
        self.assertEqual(list(reversed(dict())), [])
        self.assertEqual(list(reversed(dict().items())), [])
        self.assertEqual(list(reversed(dict().values())), [])
        self.assertEqual(list(reversed(dict().keys())), [])

    def test_reverse_iterator_for_shared_shared_dicts(self):
        # [FUNCTION DOC] Checks reversed iteration over shared-key instance dicts
        # with optional attributes.
        class A:
            # [CLASS DOC] HELPER: class that sets attributes x and y only when
            # given truthy values.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: sets attribute x if x is truthy and y if y is truthy.
                if x: self.x = x
                if y: self.y = y

        self.assertEqual(list(reversed(A(1, 2).__dict__)), ['y', 'x'])
        self.assertEqual(list(reversed(A(1, 0).__dict__)), ['x'])
        self.assertEqual(list(reversed(A(0, 1).__dict__)), ['y'])

    def test_dict_copy_order(self):
        # [FUNCTION DOC] Checks dict() copying preserves order from an OrderedDict, a plain dict
        # subclass, and a subclass that overrides iteration and keys in reverse (bpo-34320).
        # bpo-34320
        od = collections.OrderedDict([('a', 1), ('b', 2)])
        od.move_to_end('a')
        expected = list(od.items())

        copy = dict(od)
        self.assertEqual(list(copy.items()), expected)

        # dict subclass doesn't override __iter__
        class CustomDict(dict):
            # [CLASS DOC] HELPER: empty dict subclass.
            pass

        pairs = [('a', 1), ('b', 2), ('c', 3)]

        d = CustomDict(pairs)
        self.assertEqual(pairs, list(dict(d).items()))

        class CustomReversedDict(dict):
            # [CLASS DOC] HELPER: dict subclass whose keys and items iterate in reverse order.
            def keys(self):
                # [FUNCTION DOC] HELPER: returns an iterator over keys in reverse.
                return reversed(list(dict.keys(self)))

            __iter__ = keys

            def items(self):
                # [FUNCTION DOC] HELPER: returns an iterator over items in reverse.
                return reversed(dict.items(self))

        d = CustomReversedDict(pairs)
        self.assertEqual(pairs[::-1], list(dict(d).items()))

    @support.cpython_only
    def test_dict_items_result_gc(self):
        # [FUNCTION DOC] Checks the result tuple reused by the item iterator is re-tracked by
        # the garbage collector (bpo-42536). CPython only.
        # bpo-42536: dict.items's tuple-reuse speed trick breaks the GC's
        # assumptions about what can be untracked. Make sure we re-track result
        # tuples whenever we reuse them.
        it = iter({None: []}.items())
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None, None). Make sure it's re-tracked
        # when it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    @support.cpython_only
    def test_dict_items_result_gc_reversed(self):
        # [FUNCTION DOC] Same check as the forward item iterator test but for the reversed item
        # iterator. CPython only.
        # Same as test_dict_items_result_gc above, but reversed.
        it = reversed({None: []}.items())
        gc.collect()
        self.assertTrue(gc.is_tracked(next(it)))

    def test_store_evilattr(self):
        # [FUNCTION DOC] Checks storing attributes whose destructor deletes from the instance
        # dict and runs collection does not crash.
        class EvilAttr:
            # [CLASS DOC] HELPER: object whose destructor deletes the 'attr' entry from a dict
            # and runs garbage collection.
            def __init__(self, d):
                # [FUNCTION DOC] HELPER: stores the dict.
                self.d = d

            def __del__(self):
                # [FUNCTION DOC] HELPER: deletes 'attr' from the stored dict if present and
                # runs garbage collection.
                if 'attr' in self.d:
                    del self.d['attr']
                gc.collect()

        class Obj:
            # [CLASS DOC] HELPER: empty class with a replaceable instance dict.
            pass

        obj = Obj()
        obj.__dict__ = {}
        for _ in range(10):
            obj.attr = EvilAttr(obj.__dict__)

    def test_str_nonstr(self):
        # [FUNCTION DOC] Checks dicts with non-str keys (str subclass or unrelated
        # key type) built in many ways still look up str keys correctly and call key equality
        # for unrelated keys.
        # cpython uses a different lookup function if the dict only contains
        # `str` keys. Make sure the unoptimized path is used when a non-`str`
        # key appears.

        class StrSub(str):
            # [CLASS DOC] HELPER: empty str subclass.
            pass

        eq_count = 0
        # This class compares equal to the string 'key3'
        class Key3:
            # [CLASS DOC] HELPER: key hashing like the string 'key3' and equal to itself and to
            # that string, counting equality calls in the enclosing eq_count.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns the hash of 'key3'.
                return hash('key3')

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: returns True and increments the enclosing counter for
                # Key3 or the string 'key3'; else False.
                nonlocal eq_count
                if isinstance(other, Key3) or isinstance(other, str) and other == 'key3':
                    eq_count += 1
                    return True
                return False

        key3_1 = StrSub('key3')
        key3_2 = Key3()
        key3_3 = Key3()

        dicts = []

        # Create dicts of the form `{'key1': 42, 'key2': 43, key3: 44}` in a
        # bunch of different ways. In all cases, `key3` is not of type `str`.
        # `key3_1` is a `str` subclass and `key3_2` is a completely unrelated
        # type.
        for key3 in (key3_1, key3_2):
            # A literal
            dicts.append({'key1': 42, 'key2': 43, key3: 44})

            # key3 inserted via `dict.__setitem__`
            d = {'key1': 42, 'key2': 43}
            d[key3] = 44
            dicts.append(d)

            # key3 inserted via `dict.setdefault`
            d = {'key1': 42, 'key2': 43}
            self.assertEqual(d.setdefault(key3, 44), 44)
            dicts.append(d)

            # key3 inserted via `dict.update`
            d = {'key1': 42, 'key2': 43}
            d.update({key3: 44})
            dicts.append(d)

            # key3 inserted via `dict.__ior__`
            d = {'key1': 42, 'key2': 43}
            d |= {key3: 44}
            dicts.append(d)

            # `dict(iterable)`
            def make_pairs():
                # [FUNCTION DOC] HELPER: generator yielding three key-value pairs, the
                # last with key3.
                yield ('key1', 42)
                yield ('key2', 43)
                yield (key3, 44)
            d = dict(make_pairs())
            dicts.append(d)

            # `dict.copy`
            d = d.copy()
            dicts.append(d)

            # dict comprehension
            d = {key: 42 + i for i,key in enumerate(['key1', 'key2', key3])}
            dicts.append(d)

        for d in dicts:
            with self.subTest(d=d):
                self.assertEqual(d.get('key1'), 42)

                # Try to make an object that is of type `str` and is equal to
                # `'key1'`, but (at least on cpython) is a different object.
                noninterned_key1 = 'ke'
                noninterned_key1 += 'y1'
                if support.check_impl_detail(cpython=True):
                    # suppress a SyntaxWarning
                    interned_key1 = 'key1'
                    self.assertFalse(noninterned_key1 is interned_key1)
                self.assertEqual(d.get(noninterned_key1), 42)

                self.assertEqual(d.get('key3'), 44)
                self.assertEqual(d.get(key3_1), 44)
                self.assertEqual(d.get(key3_2), 44)

                # `key3_3` itself is definitely not a dict key, so make sure
                # that `__eq__` gets called.
                #
                # Note that this might not hold for `key3_1` and `key3_2`
                # because they might be the same object as one of the dict keys,
                # in which case implementations are allowed to skip the call to
                # `__eq__`.
                eq_count = 0
                self.assertEqual(d.get(key3_3), 44)
                self.assertGreaterEqual(eq_count, 1)


class CAPITest(__TestCase):
    # [CLASS DOC] Test case for C-API level dict helpers; needs the CPython
    # internal test module.

    # Test _PyDict_GetItem_KnownHash()
    @support.cpython_only
    def test_getitem_knownhash(self):
        # [FUNCTION DOC] Checks the internal known-hash lookup: returns values, raises
        # SystemError for non-dicts, KeyError for missing keys, and propagates equality
        # exceptions. CPython only.
        _testinternalcapi = import_helper.import_module('_testinternalcapi')
        dict_getitem_knownhash = _testinternalcapi.dict_getitem_knownhash

        d = {'x': 1, 'y': 2, 'z': 3}
        self.assertEqual(dict_getitem_knownhash(d, 'x', hash('x')), 1)
        self.assertEqual(dict_getitem_knownhash(d, 'y', hash('y')), 2)
        self.assertEqual(dict_getitem_knownhash(d, 'z', hash('z')), 3)

        # not a dict
        self.assertRaises(SystemError, dict_getitem_knownhash, [], 1, hash(1))
        # key does not exist
        self.assertRaises(KeyError, dict_getitem_knownhash, {}, 1, hash(1))

        class Exc(Exception): pass
        class BadEq:
            # [CLASS DOC] HELPER: key with hash 7 whose equality raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: always raises Exc.
                raise Exc
            def __hash__(self):
                # [FUNCTION DOC] HELPER: returns 7.
                return 7

        k1, k2 = BadEq(), BadEq()
        d = {k1: 1}
        self.assertEqual(dict_getitem_knownhash(d, k1, hash(k1)), 1)
        self.assertRaises(Exc, dict_getitem_knownhash, d, k2, hash(k2))


from test import mapping_tests

class GeneralMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol tests against the plain dict type.
    type2test = dict

class Dict(dict):
    # [CLASS DOC] HELPER: empty dict subclass used as the type under test.
    pass

class SubclassMappingTests(mapping_tests.BasicTestMappingProtocol):
    # [CLASS DOC] Runs the generic mapping protocol tests against a dict subclass.
    type2test = Dict


if __name__ == "__main__":
    run_tests()
