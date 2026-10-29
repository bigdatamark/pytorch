# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Reusable mapping-protocol test base classes copied from CPython 3.13
# Lib/test/mapping_tests.py, patched so the base test case is CPythonTestCase (kept in
# the private name __TestCase). Defines BasicTestMappingProtocol, TestMappingProtocol and
# TestHashMappingProtocol; concrete test modules subclass them after setting type2test.
# Import-time side effect: inserts a RedirectImportFinder at the front of sys.meta_path
# so imports of the listed test.* modules resolve to the same-named top-level modules.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/mapping_tests.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

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
    # [CLASS DOC] Meta path finder that redirects selected "test.X" imports (see redirect_imports)
    # to the top-level module "X". Stateless; installed once at import time into sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] Finder hook. If fullname is in redirect_imports, imports the module
        # named without the "test." prefix, registers it in sys.modules under fullname and
        # returns that module's spec; returns None (defer to other finders) if the name is
        # not listed or the import fails with ImportError. Side effect: mutates sys.modules.
        # KNOWN ISSUE: uses importlib.util without importing it explicitly (only
        # importlib.abc is imported), relying on it being loaded elsewhere.
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

# tests common to dict and UserDict
import unittest
import collections
from test.support import get_c_recursion_limit


class BasicTestMappingProtocol(__TestCase):
    # [CLASS DOC] Base test case checking that an object conforms to the mapping
    # protocol (dict-like read, write, update, get, pop, popitem). Subclasses set
    # type2test to the class under test and may override _reference, _empty_mapping and
    # _full_mapping. Per-instance state set in __init__: reference (invariant key/value
    # dict), other (one pair not in the mapping) and inmapping (one pair that is in it).
    # This base class can be used to check that an object conforms to the
    # mapping protocol

    # Functions that can be useful to override to adapt to dictionary
    # semantics
    type2test = None # which class is being tested (overwrite in subclasses)

    def _reference(self):
        # [FUNCTION DOC] HELPER: Returns the reference dict of key/value pairs that survive
        # storage unchanged; subclasses override it to adapt to other mapping semantics.
        """Return a dictionary of values which are invariant by storage
        in the object under test."""
        return {"1": "2", "key1":"value1", "key2":(1,2,3)}
    def _empty_mapping(self):
        # [FUNCTION DOC] HELPER: Returns a new empty instance of type2test.
        """Return an empty mapping object"""
        return self.type2test()
    def _full_mapping(self, data):
        # [FUNCTION DOC] HELPER: Builds an empty mapping and stores every key/value
        # pair of the given data dict into it by item assignment; returns the mapping.
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    def __init__(self, *args, **kw):
        # [FUNCTION DOC] Constructs the test case via unittest.TestCase.__init__ with
        # the given args (bypassing the patched base), then prepares fixtures: a copy
        # of the reference dict, the last pair popped into other, the next pair
        # popped into inmapping and put back so the reference still contains it.
        unittest.TestCase.__init__(self, *args, **kw)
        self.reference = self._reference().copy()

        # A (key, value) pair not in the mapping
        key, value = self.reference.popitem()
        self.other = {key:value}

        # A (key, value) pair in the mapping
        key, value = self.reference.popitem()
        self.inmapping = {key:value}
        self.reference[key] = value

    def test_read(self):
        # [FUNCTION DOC] Checks read-only operations on a full mapping: item lookup, KeyError
        # for a missing key, length, membership, equality, truthiness, key/value/item iterators
        # matching the reference, and get with defaults. Raises AssertionError on any mismatch.
        # Test for read only operations on mapping
        p = self._empty_mapping()
        p1 = dict(p) #workaround for singleton objects
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        #Indexing
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = list(self.other.keys())[0]
        self.assertRaises(KeyError, lambda:d[knownkey])
        #len
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        #__contains__
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        #cmp
        self.assertEqual(p, p)
        self.assertEqual(d, d)
        self.assertNotEqual(p, d)
        self.assertNotEqual(d, p)
        #bool
        if p: self.fail("Empty mapping must compare to False")
        if not d: self.fail("Full mapping must compare to True")
        # keys(), items(), iterkeys() ...
        def check_iterandlist(iter, lst, ref):
            # [FUNCTION DOC] HELPER: Asserts that iter is an iterator (has __next__ and __iter__)
            # and that the sets of its items, the list lst and the reference ref are all equal.
            self.assertTrue(hasattr(iter, '__next__'))
            self.assertTrue(hasattr(iter, '__iter__'))
            x = list(iter)
            self.assertTrue(set(x)==set(lst)==set(ref))
        check_iterandlist(iter(d.keys()), list(d.keys()),
                          self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()),
                          self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()),
                          self.reference.items())
        #get
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        # [FUNCTION DOC] Checks write operations on a mapping: item assignment and
        # deletion, update from a mapping and from an item list, setdefault, pop with
        # and without default, and popitem including KeyError on an empty mapping.
        # Test for write operations on mapping
        p = self._empty_mapping()
        #Indexing
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference.keys():
            del p[key]
            self.assertRaises(KeyError, lambda:p[key])
        p = self._empty_mapping()
        #update
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        #setdefault
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        #pop
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        #popitem
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p=self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        # [FUNCTION DOC] Checks that two freshly created empty mappings compare equal.
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        # [FUNCTION DOC] Checks truthiness: an empty mapping
        # is false and the non-empty reference dict is true.
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        # [FUNCTION DOC] Checks keys(): empty list for an empty mapping, membership
        # of known/unknown keys, and TypeError when called with an extra argument.
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(list(self.inmapping.keys())[0], d.keys())
        self.assertNotIn(list(self.other.keys())[0], d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        # [FUNCTION DOC] Checks values(): empty for an empty
        # mapping and TypeError when called with an extra argument.
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        # [FUNCTION DOC] Checks items(): empty for an empty mapping
        # and TypeError when called with an extra argument.
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        # [FUNCTION DOC] Checks that an empty mapping has length zero.
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        # [FUNCTION DOC] Checks lookup of a known key on the reference
        # dict and TypeError when __getitem__ is called without arguments.
        d = self.reference
        self.assertEqual(d[list(self.inmapping.keys())[0]],
                         list(self.inmapping.values())[0])

        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        # [FUNCTION DOC] Checks update(): from a mapping, no argument, an item iterable,
        # rejection of a non-iterable with TypeError or AttributeError, a minimal object with
        # keys and __getitem__, propagation of exceptions raised by keys(), by the key iterator,
        # by __getitem__ and by a failing iterator, and ValueError for a three-element item.
        # mapping argument
        d = self._empty_mapping()
        d.update(self.other)
        self.assertEqual(list(d.items()), list(self.other.items()))

        # No argument
        d = self._empty_mapping()
        d.update()
        self.assertEqual(d, self._empty_mapping())

        # item sequence
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # Iterator
        d = self._empty_mapping()
        d.update(self.other.items())
        self.assertEqual(list(d.items()), list(self.other.items()))

        # FIXME: Doesn't work with UserDict
        # self.assertRaises((TypeError, AttributeError), d.update, None)
        self.assertRaises((TypeError, AttributeError), d.update, 42)

        outerself = self
        class SimpleUserDict:
            # [CLASS DOC] HELPER: Minimal mapping-like object exposing keys
            # and __getitem__ over the enclosing test's reference dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the enclosing
                # test's reference dict as attribute d.
                self.d = outerself.reference
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the keys of the wrapped dict.
                return self.d.keys()
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the value for key i from the wrapped dict.
                return self.d[i]
        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted(d.items())
        i2 = sorted(self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception): pass

        d = self._empty_mapping()
        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose
            # keys method raises the local exception Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())

        d.clear()

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys method
            # returns an iterator that yields one key and then raises Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns a BogonIter instance.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator yielding
                    # the single key 'a' and then raising Exc.
                    def __init__(self):
                        # [FUNCTION DOC] HELPER: Sets the remaining-yield flag i to 1.
                        self.i = 1
                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Returns self (iterator protocol).
                        return self
                    def __next__(self):
                        # [FUNCTION DOC] HELPER: Returns 'a' on the first
                        # call (clearing the flag), then raises Exc.
                        if self.i:
                            self.i = 0
                            return 'a'
                        raise Exc
                return BogonIter()
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the key itself as its value.
                return key
        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys method
            # iterates the letters a to z and whose __getitem__ raises Exc.
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns a BogonIter instance.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator yielding the characters
                    # from 'a' through 'z', then raising StopIteration.
                    def __init__(self):
                        # [FUNCTION DOC] HELPER: Initializes the
                        # counter i to the code point of 'a'.
                        self.i = ord('a')
                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Returns self (iterator protocol).
                        return self
                    def __next__(self):
                        # [FUNCTION DOC] HELPER: Returns the next letter and
                        # increments the counter, or raises StopIteration after 'z'.
                        if self.i <= ord('z'):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration
                return BogonIter()
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc
        self.assertRaises(Exc, d.update, FailingUserDict())

        d = self._empty_mapping()
        class badseq(object):
            # [CLASS DOC] HELPER: Iterable whose iterator raises Exc on the first step.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()

        self.assertRaises(Exc, d.update, badseq())

        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    # no test_fromkeys or test_copy as both os.environ and selves don't support it

    def test_get(self):
        # [FUNCTION DOC] Checks get(): default None and explicit default
        # for missing keys on empty and reference mappings, correct value
        # for present keys, and TypeError for wrong argument counts.
        d = self._empty_mapping()
        self.assertTrue(d.get(list(self.other.keys())[0]) is None)
        self.assertEqual(d.get(list(self.other.keys())[0], 3), 3)
        d = self.reference
        self.assertTrue(d.get(list(self.other.keys())[0]) is None)
        self.assertEqual(d.get(list(self.other.keys())[0], 3), 3)
        self.assertEqual(d.get(list(self.inmapping.keys())[0]),
                         list(self.inmapping.values())[0])
        self.assertEqual(d.get(list(self.inmapping.keys())[0], 3),
                         list(self.inmapping.values())[0])
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        # [FUNCTION DOC] Checks that setdefault called with no arguments raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        # [FUNCTION DOC] Checks that popitem on an empty mapping raises
        # KeyError and that passing an argument raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        # [FUNCTION DOC] Checks pop() on a one-item mapping: KeyError for a missing key,
        # returns the value and empties the mapping for the stored key, then KeyError again.
        d = self._empty_mapping()
        k, v = list(self.inmapping.items())[0]
        d[k] = v
        self.assertRaises(KeyError, d.pop, list(self.other.keys())[0])

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)


class TestMappingProtocol(BasicTestMappingProtocol):
    # [CLASS DOC] Extends the basic mapping-protocol tests with fuller dict-like
    # checks (clear, copy, fromkeys, contains, update variants, popitem over
    # many sizes). Each overridden test first runs the base-class version.
    def test_constructor(self):
        # [FUNCTION DOC] Runs the base check, then verifies that two empty mappings are
        # distinct objects and that keyword construction yields the expected contents.
        BasicTestMappingProtocol.test_constructor(self)
        self.assertTrue(self._empty_mapping() is not self._empty_mapping())
        self.assertEqual(self.type2test(x=1, y=2), {"x": 1, "y": 2})

    def test_bool(self):
        # [FUNCTION DOC] Runs the base truthiness check, then verifies
        # empty mappings are false and non-empty ones are true.
        BasicTestMappingProtocol.test_bool(self)
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self._full_mapping({"x": "y"}))
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self._full_mapping({"x": "y"})) is True)

    def test_keys(self):
        # [FUNCTION DOC] Runs the base check, then verifies keys()
        # of a two-item mapping reports membership correctly.
        BasicTestMappingProtocol.test_keys(self)
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self._full_mapping({'a': 1, 'b': 2})
        k = d.keys()
        self.assertIn('a', k)
        self.assertIn('b', k)
        self.assertNotIn('c', k)

    def test_values(self):
        # [FUNCTION DOC] Runs the base check, then verifies values() of a one-item mapping.
        BasicTestMappingProtocol.test_values(self)
        d = self._full_mapping({1:2})
        self.assertEqual(list(d.values()), [2])

    def test_items(self):
        # [FUNCTION DOC] Runs the base check, then verifies items() of a one-item mapping.
        BasicTestMappingProtocol.test_items(self)

        d = self._full_mapping({1:2})
        self.assertEqual(list(d.items()), [(1, 2)])

    def test_contains(self):
        # [FUNCTION DOC] Checks membership on empty and full mappings
        # and TypeError when __contains__ is called without arguments.
        d = self._empty_mapping()
        self.assertNotIn('a', d)
        self.assertTrue(not ('a' in d))
        self.assertTrue('a' not in d)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertIn('a', d)
        self.assertIn('b', d)
        self.assertNotIn('c', d)

        self.assertRaises(TypeError, d.__contains__)

    def test_len(self):
        # [FUNCTION DOC] Runs the base length check, then
        # verifies a two-item mapping has length two.
        BasicTestMappingProtocol.test_len(self)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertEqual(len(d), 2)

    def test_getitem(self):
        # [FUNCTION DOC] Runs the base check, then verifies item reads, overwrite,
        # new-key assignment, deletion and TypeError for a missing argument.
        BasicTestMappingProtocol.test_getitem(self)
        d = self._full_mapping({'a': 1, 'b': 2})
        self.assertEqual(d['a'], 1)
        self.assertEqual(d['b'], 2)
        d['c'] = 3
        d['a'] = 4
        self.assertEqual(d['c'], 3)
        self.assertEqual(d['a'], 4)
        del d['b']
        self.assertEqual(d, {'a': 4, 'c': 3})

        self.assertRaises(TypeError, d.__getitem__)

    def test_clear(self):
        # [FUNCTION DOC] Checks that clear empties a mapping
        # and that passing an argument raises TypeError.
        d = self._full_mapping({1:1, 2:2, 3:3})
        d.clear()
        self.assertEqual(d, {})

        self.assertRaises(TypeError, d.clear, None)

    def test_update(self):
        # [FUNCTION DOC] Runs the base check, then verifies update with mappings,
        # no argument, keyword arguments, item sequences, both together (keywords
        # win), an items iterator, and a minimal keys/__getitem__ object.
        BasicTestMappingProtocol.test_update(self)
        # mapping argument
        d = self._empty_mapping()
        d.update({1:100})
        d.update({2:20})
        d.update({1:1, 2:2, 3:3})
        self.assertEqual(d, {1:1, 2:2, 3:3})

        # no argument
        d.update()
        self.assertEqual(d, {1:1, 2:2, 3:3})

        # keyword arguments
        d = self._empty_mapping()
        d.update(x=100)
        d.update(y=20)
        d.update(x=1, y=2, z=3)
        self.assertEqual(d, {"x":1, "y":2, "z":3})

        # item sequence
        d = self._empty_mapping()
        d.update([("x", 100), ("y", 20)])
        self.assertEqual(d, {"x":100, "y":20})

        # Both item sequence and keyword arguments
        d = self._empty_mapping()
        d.update([("x", 100), ("y", 20)], x=1, y=2)
        self.assertEqual(d, {"x":1, "y":2})

        # iterator
        d = self._full_mapping({1:3, 2:4})
        d.update(self._full_mapping({1:2, 3:4, 5:6}).items())
        self.assertEqual(d, {1:2, 2:4, 3:4, 5:6})

        class SimpleUserDict:
            # [CLASS DOC] HELPER: Minimal mapping-like object exposing
            # keys and __getitem__ over a fixed three-item dict.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the wrapped dict mapping 1, 2, 3 to themselves.
                self.d = {1:1, 2:2, 3:3}
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the keys of the wrapped dict.
                return self.d.keys()
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the value for key i from the wrapped dict.
                return self.d[i]
        d.clear()
        d.update(SimpleUserDict())
        self.assertEqual(d, {1:1, 2:2, 3:3})

    def test_fromkeys(self):
        # [FUNCTION DOC] Checks fromkeys: from strings, tuples with a value, empty
        # and generator inputs, TypeError for non-iterables, preservation of subclass
        # type, a subclass whose __new__ returns a UserDict, and propagation of
        # exceptions from a failing __init__, iterator or __setitem__.
        self.assertEqual(self.type2test.fromkeys('abc'), {'a':None, 'b':None, 'c':None})
        d = self._empty_mapping()
        self.assertTrue(not(d.fromkeys('abc') is d))
        self.assertEqual(d.fromkeys('abc'), {'a':None, 'b':None, 'c':None})
        self.assertEqual(d.fromkeys((4,5),0), {4:0, 5:0})
        self.assertEqual(d.fromkeys([]), {})
        def g():
            # [FUNCTION DOC] HELPER: Generator yielding the single value 1.
            yield 1
        self.assertEqual(d.fromkeys(g()), {1:None})
        self.assertRaises(TypeError, {}.fromkeys, 3)
        class dictlike(self.type2test): pass
        self.assertEqual(dictlike.fromkeys('a'), {'a':None})
        self.assertEqual(dictlike().fromkeys('a'), {'a':None})
        self.assertTrue(dictlike.fromkeys('a').__class__ is dictlike)
        self.assertTrue(dictlike().fromkeys('a').__class__ is dictlike)
        self.assertTrue(type(dictlike.fromkeys('a')) is dictlike)
        class mydict(self.type2test):
            # [CLASS DOC] HELPER: type2test subclass whose __new__
            # returns a collections.UserDict instead of itself.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Ignores the class and returns a new UserDict.
                return collections.UserDict()
        ud = mydict.fromkeys('ab')
        self.assertEqual(ud, {'a':None, 'b':None})
        self.assertIsInstance(ud, collections.UserDict)
        self.assertRaises(TypeError, dict.fromkeys)

        class Exc(Exception): pass

        class baddict1(self.type2test):
            # [CLASS DOC] HELPER: type2test subclass whose constructor always raises Exc.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()

        self.assertRaises(Exc, baddict1.fromkeys, [1])

        class BadSeq(object):
            # [CLASS DOC] HELPER: Iterable whose iterator raises Exc on the first step.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()

        self.assertRaises(Exc, self.type2test.fromkeys, BadSeq())

        class baddict2(self.type2test):
            # [CLASS DOC] HELPER: type2test subclass whose item assignment always raises Exc.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()

        self.assertRaises(Exc, baddict2.fromkeys, [1])

    def test_copy(self):
        # [FUNCTION DOC] Checks copy: equal contents, same class
        # as the original, and TypeError for an extra argument.
        d = self._full_mapping({1:1, 2:2, 3:3})
        self.assertEqual(d.copy(), {1:1, 2:2, 3:3})
        d = self._empty_mapping()
        self.assertEqual(d.copy(), d)
        self.assertIsInstance(d.copy(), d.__class__)
        self.assertRaises(TypeError, d.copy, None)

    def test_get(self):
        # [FUNCTION DOC] Runs the base get check, then verifies
        # defaults and present keys on empty and two-item mappings.
        BasicTestMappingProtocol.test_get(self)
        d = self._empty_mapping()
        self.assertTrue(d.get('c') is None)
        self.assertEqual(d.get('c', 3), 3)
        d = self._full_mapping({'a' : 1, 'b' : 2})
        self.assertTrue(d.get('c') is None)
        self.assertEqual(d.get('c', 3), 3)
        self.assertEqual(d.get('a'), 1)
        self.assertEqual(d.get('a', 3), 1)

    def test_setdefault(self):
        # [FUNCTION DOC] Runs the base check, then verifies setdefault inserts None or
        # the given default once and returns the stored (mutable) value on later calls.
        BasicTestMappingProtocol.test_setdefault(self)
        d = self._empty_mapping()
        self.assertTrue(d.setdefault('key0') is None)
        d.setdefault('key0', [])
        self.assertTrue(d.setdefault('key0') is None)
        d.setdefault('key', []).append(3)
        self.assertEqual(d['key'][0], 3)
        d.setdefault('key', []).append(4)
        self.assertEqual(len(d['key']), 2)

    def test_popitem(self):
        # [FUNCTION DOC] Runs the base check, then for sizes 1 to 2048 and for both a
        # same-structure mapping and a copy, pops every item and verifies key/value consistency,
        # ending empty.
        BasicTestMappingProtocol.test_popitem(self)
        for copymode in -1, +1:
            # -1: b has same structure as a
            # +1: b is a.copy()
            for log2size in range(12):
                size = 2**log2size
                a = self._empty_mapping()
                b = self._empty_mapping()
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
                    self.assertTrue(not(copymode < 0 and ta != tb))
                self.assertTrue(not a)
                self.assertTrue(not b)

    def test_pop(self):
        # [FUNCTION DOC] Runs the base check, then verifies
        # pop with a default for a missing and a present key.
        BasicTestMappingProtocol.test_pop(self)

        # Tests for pop with specified key
        d = self._empty_mapping()
        k, v = 'abc', 'def'

        self.assertEqual(d.pop(k, v), v)
        d[k] = v
        self.assertEqual(d.pop(k, 1), v)


class TestHashMappingProtocol(TestMappingProtocol):
    # [CLASS DOC] Mapping-protocol tests for hash-based mappings; adds checks on how bad __hash__,
    # __eq__ and __repr__ behaviours propagate, plus iteration mutation and deep repr recursion.

    def test_getitem(self):
        # [FUNCTION DOC] Runs the parent check, then verifies lookup with a
        # key whose equality raises is a KeyError for a different hash-equal
        # key, and that an exception raised by __hash__ propagates.
        TestMappingProtocol.test_getitem(self)
        class Exc(Exception): pass

        class BadEq(object):
            # [CLASS DOC] HELPER: Key type with constant
            # hash 24 whose equality comparison raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns constant hash 24.
                return 24

        d = self._empty_mapping()
        d[BadEq()] = 42
        self.assertRaises(KeyError, d.__getitem__, 23)

        class BadHash(object):
            # [CLASS DOC] HELPER: Key type with hash 42 that
            # raises Exc from __hash__ once its fail flag is set.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Raises Exc if fail is set, otherwise returns 42.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.__getitem__, x)

    def test_fromkeys(self):
        # [FUNCTION DOC] Runs the parent check, then re-verifies
        # fromkeys on a subclass whose __new__ returns a UserDict.
        TestMappingProtocol.test_fromkeys(self)
        class mydict(self.type2test):
            # [CLASS DOC] HELPER: type2test subclass whose __new__ returns a collections.UserDict.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: Ignores the class and returns a new UserDict.
                return collections.UserDict()
        ud = mydict.fromkeys('ab')
        self.assertEqual(ud, {'a':None, 'b':None})
        self.assertIsInstance(ud, collections.UserDict)

    def test_pop(self):
        # [FUNCTION DOC] Runs the parent check, then verifies
        # pop propagates an exception raised by __hash__.
        TestMappingProtocol.test_pop(self)

        class Exc(Exception): pass

        class BadHash(object):
            # [CLASS DOC] HELPER: Key type with hash 42 that
            # raises Exc from __hash__ once its fail flag is set.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Raises Exc if fail is set, otherwise returns 42.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.pop, x)

    def test_mutatingiteration(self):
        # [FUNCTION DOC] Verifies that adding keys to a mapping while iterating it raises
        # RuntimeError (fails the test if iteration proceeds past the first step).
        d = self._empty_mapping()
        d[1] = 1
        try:
            count = 0
            for i in d:
                d[i+1] = 1
                if count >= 1:
                    self.fail("changing dict size during iteration doesn't raise Error")
                count += 1
        except RuntimeError:
            pass

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of empty, one-item and self-referencing
        # mappings and that an exception raised by an element's repr propagates.
        d = self._empty_mapping()
        self.assertEqual(repr(d), '{}')
        d[1] = 2
        self.assertEqual(repr(d), '{1: 2}')
        d = self._empty_mapping()
        d[1] = d
        self.assertEqual(repr(d), '{1: {...}}')

        class Exc(Exception): pass

        class BadRepr(object):
            # [CLASS DOC] HELPER: Value type whose repr raises Exc.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()

        d = self._full_mapping({1: BadRepr()})
        self.assertRaises(Exc, repr, d)

    def test_repr_deep(self):
        # [FUNCTION DOC] Builds a mapping nested deeper than the C
        # recursion limit and checks that repr raises RecursionError.
        d = self._empty_mapping()
        for i in range(get_c_recursion_limit() + 1):
            d0 = d
            d = self._empty_mapping()
            d[1] = d0
        self.assertRaises(RecursionError, repr, d)

    def test_eq(self):
        # [FUNCTION DOC] Checks equality of empty and one-item mappings and that
        # an exception from a key's __eq__ propagates through mapping comparison.
        self.assertEqual(self._empty_mapping(), self._empty_mapping())
        self.assertEqual(self._full_mapping({1: 2}),
                         self._full_mapping({1: 2}))

        class Exc(Exception): pass

        class BadCmp(object):
            # [CLASS DOC] HELPER: Key type with hash 1 whose equality comparison raises Exc.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always raises Exc.
                raise Exc()
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns constant hash 1.
                return 1

        d1 = self._full_mapping({BadCmp(): 1})
        d2 = self._full_mapping({1: 1})
        self.assertRaises(Exc, lambda: BadCmp()==1)
        self.assertRaises(Exc, lambda: d1==d2)

    def test_setdefault(self):
        # [FUNCTION DOC] Runs the parent check, then verifies
        # setdefault propagates an exception raised by __hash__.
        TestMappingProtocol.test_setdefault(self)

        class Exc(Exception): pass

        class BadHash(object):
            # [CLASS DOC] HELPER: Key type with hash 42 that
            # raises Exc from __hash__ once its fail flag is set.
            fail = False
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Raises Exc if fail is set, otherwise returns 42.
                if self.fail:
                    raise Exc()
                else:
                    return 42

        d = self._empty_mapping()
        x = BadHash()
        d[x] = 42
        x.fail = True
        self.assertRaises(Exc, d.setdefault, x, [])
