# Owner(s): ["module: meta tensors"]
# [MODULE DOC] Tests for torch.utils.weak.WeakIdKeyDictionary, a weak-key mapping keyed by
# object identity, ported from the CPython weakref and mapping test suites. Contains WeakTest
# (dict behaviors, union operators, threaded copy), and two mapping-protocol suites: one keyed
# by tensors and one keyed by TorchScript custom-class objects via the hash-based weak
# reference. Importing has no side effects; running as a script calls run_tests.

import copy
import gc
import random
import threading
import unittest

import torch
from torch.testing._internal.common_utils import IS_MACOS, run_tests, TestCase
from torch.testing._internal.torchbind_impls import load_torchbind_test_lib
from torch.utils.weak import _WeakHashRef, WeakIdKeyDictionary


def C():
    # [FUNCTION DOC] HELPER: Return a fresh one-element random float tensor; used as a distinct
    # weak-referenceable key for the dictionary tests.
    return torch.randn(1)


# These tests are ported from cpython/Lib/test/test_weakref.py,
# but adapted to use tensor rather than object
class WeakTest(TestCase):
    # [CLASS DOC] Test case exercising WeakIdKeyDictionary with tensor keys: construction from
    # dict and from another weak dictionary, popitem, setdefault, update, deletion, union
    # operators, repr, and copy/deepcopy while keys are garbage collected on another thread.
    # COUNT is unused.
    COUNT = 10

    def test_make_weak_keyed_dict_from_dict(self):
        # [FUNCTION DOC] Build a weak dictionary from a plain dict keyed by a tensor and check
        # that the stored value can be looked up by that tensor.
        o = torch.randn(2)
        dict = WeakIdKeyDictionary({o: 364})
        self.assertEqual(dict[o], 364)

    def test_make_weak_keyed_dict_from_weak_keyed_dict(self):
        # [FUNCTION DOC] Build a weak dictionary from a plain dict, then build a second weak
        # dictionary from the first, and check the value is retrievable from both by the
        # tensor key.
        o = torch.randn(3)
        dict = WeakIdKeyDictionary({o: 364})
        self.assertEqual(dict[o], 364)
        dict2 = WeakIdKeyDictionary(dict)
        self.assertEqual(dict2[o], 364)

    def check_popitem(self, klass, key1, value1, key2, value2):
        # [FUNCTION DOC] HELPER: Shared popitem check. Fill a new mapping of the given class
        # with two key/value pairs, then pop twice; each pop must shrink the length by one and
        # return a pair whose value matches the key that was popped (identity comparison). Pops
        # are checked down to an empty mapping.
        weakdict = klass()
        weakdict[key1] = value1
        weakdict[key2] = value2
        self.assertEqual(len(weakdict), 2)
        k, v = weakdict.popitem()
        self.assertEqual(len(weakdict), 1)
        if k is key1:
            self.assertIs(v, value1)
        else:
            self.assertIs(v, value2)
        k, v = weakdict.popitem()
        self.assertEqual(len(weakdict), 0)
        if k is key1:
            self.assertIs(v, value1)
        else:
            self.assertIs(v, value2)

    def test_weak_keyed_dict_popitem(self):
        # [FUNCTION DOC] Run the popitem check on WeakIdKeyDictionary with two tensor keys.
        self.check_popitem(WeakIdKeyDictionary, C(), "value 1", C(), "value 2")

    def check_setdefault(self, klass, key, value1, value2):
        # [FUNCTION DOC] HELPER: Shared setdefault check. Requires the two values to be distinct
        # objects (fails otherwise). Setting a default for a new key stores and returns the
        # first value; a second call with a different default returns the existing first value
        # and leaves it unchanged. Membership, get and indexing are verified after each call.
        self.assertIsNot(
            value1,
            value2,
            "invalid test -- value parameters must be distinct objects",
        )
        weakdict = klass()
        o = weakdict.setdefault(key, value1)
        self.assertIs(o, value1)
        self.assertIn(key, weakdict)
        self.assertIs(weakdict.get(key), value1)
        self.assertIs(weakdict[key], value1)

        o = weakdict.setdefault(key, value2)
        self.assertIs(o, value1)
        self.assertIn(key, weakdict)
        self.assertIs(weakdict.get(key), value1)
        self.assertIs(weakdict[key], value1)

    def test_weak_keyed_dict_setdefault(self):
        # [FUNCTION DOC] Run the setdefault check on WeakIdKeyDictionary with one tensor key and
        # two distinct string values.
        self.check_setdefault(WeakIdKeyDictionary, C(), "value 1", "value 2")

    def check_update(self, klass, dict):
        # [FUNCTION DOC] HELPER: Shared update check. Updates a new mapping of the given class
        # from the supplied dict and verifies equal length, that every key in the weak mapping
        # is in the source and every source key is in the weak mapping, and that get and
        # indexing return the identical value objects.
        #
        #  This exercises d.update(), len(d), d.keys(), k in d,
        #  d.get(), d[].
        #
        weakdict = klass()
        weakdict.update(dict)
        self.assertEqual(len(weakdict), len(dict))
        for k in weakdict:
            self.assertIn(k, dict, "mysterious new key appeared in weak dict")
            v = dict.get(k)
            self.assertIs(v, weakdict[k])
            self.assertIs(v, weakdict.get(k))
        for k in dict:
            self.assertIn(k, weakdict, "original key disappeared in weak dict")
            v = dict[k]
            self.assertIs(v, weakdict[k])
            self.assertIs(v, weakdict.get(k))

    def test_weak_keyed_dict_update(self):
        # [FUNCTION DOC] Run the update check with a dict keyed by three distinct tensors.
        self.check_update(WeakIdKeyDictionary, {C(): 1, C(): 2, C(): 3})

    def test_weak_keyed_delitem(self):
        # [FUNCTION DOC] Delete one of two keys from a weak dictionary and check that length
        # drops to one and the remaining key list contains only the other tensor.
        d = WeakIdKeyDictionary()
        o1 = torch.randn(1)
        o2 = torch.randn(2)
        d[o1] = "something"
        d[o2] = "something"
        self.assertEqual(len(d), 2)
        del d[o1]
        self.assertEqual(len(d), 1)
        self.assertEqual(list(d.keys()), [o2])

    def test_weak_keyed_union_operators(self):
        # [FUNCTION DOC] Check the dict union operators on weak dictionaries: union of two weak
        # dictionaries, of a weak dictionary and a plain mapping, in-place union with an
        # iterable of pairs, and the reflected union with a plain dict on the left. Each result
        # must equal the plain dict union and have type WeakIdKeyDictionary. After the first
        # tensor is deleted, values associated with it must disappear from the results. The test
        # is skipped when the interpreter does not support the dict union operator.
        try:
            {} | {}
        except TypeError:
            self.skipTest("dict union not supported in this Python")

        o1 = C()
        o2 = C()
        o3 = C()
        wkd1 = WeakIdKeyDictionary({o1: 1, o2: 2})
        wkd2 = WeakIdKeyDictionary({o3: 3, o1: 4})
        wkd3 = wkd1.copy()
        d1 = {o2: "5", o3: "6"}
        pairs = [(o2, 7), (o3, 8)]

        tmp1 = wkd1 | wkd2  # Between two WeakKeyDictionaries
        self.assertEqual(dict(tmp1), dict(wkd1) | dict(wkd2))
        self.assertIs(type(tmp1), WeakIdKeyDictionary)
        wkd1 |= wkd2
        self.assertEqual(wkd1, tmp1)

        tmp2 = wkd2 | d1  # Between WeakKeyDictionary and mapping
        self.assertEqual(dict(tmp2), dict(wkd2) | d1)
        self.assertIs(type(tmp2), WeakIdKeyDictionary)
        wkd2 |= d1
        self.assertEqual(wkd2, tmp2)

        tmp3 = wkd3.copy()  # Between WeakKeyDictionary and iterable key, value
        tmp3 |= pairs
        self.assertEqual(dict(tmp3), dict(wkd3) | dict(pairs))
        self.assertIs(type(tmp3), WeakIdKeyDictionary)

        tmp4 = d1 | wkd3  # Testing .__ror__
        self.assertEqual(dict(tmp4), d1 | dict(wkd3))
        self.assertIs(type(tmp4), WeakIdKeyDictionary)

        del o1
        self.assertNotIn(4, tmp1.values())
        self.assertNotIn(4, tmp2.values())
        self.assertNotIn(1, tmp3.values())
        self.assertNotIn(1, tmp4.values())

    def test_weak_keyed_bad_delitem(self):
        # [FUNCTION DOC] Check error behavior: deleting or reading a missing tensor key raises
        # KeyError, while deleting, reading or setting with an integer key (not weak-
        # referenceable) raises TypeError.
        d = WeakIdKeyDictionary()
        o = torch.randn(1)
        # An attempt to delete an object that isn't there should raise
        # KeyError.  It didn't before 2.3.
        self.assertRaises(KeyError, d.__delitem__, o)
        self.assertRaises(KeyError, d.__getitem__, o)

        # If a key isn't of a weakly referenceable type, __getitem__ and
        # __setitem__ raise TypeError.  __delitem__ should too.
        self.assertRaises(TypeError, d.__delitem__, 13)
        self.assertRaises(TypeError, d.__getitem__, 13)
        self.assertRaises(TypeError, d.__setitem__, 13, 13)

    def test_make_weak_keyed_dict_repr(self):
        # [FUNCTION DOC] Check that repr of an empty weak dictionary matches a pattern
        # containing the class name and a hex address.
        dict = WeakIdKeyDictionary()
        self.assertRegex(repr(dict), "<WeakIdKeyDictionary at 0x.*>")

    def check_threaded_weak_dict_copy(self, type_, deepcopy):
        # [FUNCTION DOC] HELPER: Stress copying under concurrent garbage collection. Builds a
        # mapping of the given type with 70000 dummy key/value objects, then runs one thread
        # that copies it (copy or deepcopy depending on the flag) while another randomly pops
        # the strong key references so the keys die. Any exception raised in the copy thread is
        # collected and re-raised in the main thread. Guards against crashes when entries are
        # collected during copying.
        # `deepcopy` should be either True or False.
        exc = []

        # Cannot give these slots as weakrefs weren't supported
        # on these objects until later versions of Python
        class DummyKey:
            # [CLASS DOC] HELPER: Plain weak-referenceable key object holding a counter value.
            def __init__(self, ctr):
                # [FUNCTION DOC] HELPER: Store the counter on the instance.
                self.ctr = ctr

        class DummyValue:
            # [CLASS DOC] HELPER: Plain value object holding a counter value.
            def __init__(self, ctr):
                # [FUNCTION DOC] HELPER: Store the counter on the instance.
                self.ctr = ctr

        def dict_copy(d, exc):
            # [FUNCTION DOC] HELPER: Thread body. Copies the dictionary (deep copy if the
            # enclosing flag is true, otherwise the copy method), discarding the result, and
            # appends any raised exception to the shared list instead of propagating it.
            try:
                if deepcopy is True:
                    _ = copy.deepcopy(d)
                else:
                    _ = d.copy()
            except Exception as ex:
                exc.append(ex)

        def pop_and_collect(lst):
            # [FUNCTION DOC] HELPER: Thread body. Repeatedly removes a randomly chosen element
            # from the given list until it is empty, which drops the strong references to the
            # keys; runs a full garbage collection after every 10000 removals.
            gc_ctr = 0
            while lst:
                i = random.randint(0, len(lst) - 1)
                gc_ctr += 1
                lst.pop(i)
                if gc_ctr % 10000 == 0:
                    gc.collect()  # just in case

        d = type_()
        keys = []
        values = []
        # Initialize d with many entries
        for i in range(70000):
            k, v = DummyKey(i), DummyValue(i)
            keys.append(k)
            values.append(v)
            d[k] = v
            del k
            del v

        t_copy = threading.Thread(target=dict_copy, args=(d, exc))
        t_collect = threading.Thread(target=pop_and_collect, args=(keys,))

        t_copy.start()
        t_collect.start()

        t_copy.join()
        t_collect.join()

        # Test exceptions
        if exc:
            raise exc[0]

    def test_threaded_weak_key_dict_copy(self):
        # [FUNCTION DOC] Run the threaded copy stress test with plain copy (see the
        # CPython issue noted below about weak entries collected during copying).
        # Issue #35615: Weakref keys or values getting GC'ed during dict
        # copying should not result in a crash.
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, False)

    def test_threaded_weak_key_dict_deepcopy(self):
        # [FUNCTION DOC] Run the threaded copy stress test with deepcopy.
        # Issue #35615: Weakref keys or values getting GC'ed during dict
        # copying should not result in a crash.
        self.check_threaded_weak_dict_copy(WeakIdKeyDictionary, True)


# Adapted from cpython/Lib/test/mapping_tests.py
class WeakKeyDictionaryTestCase(TestCase):
    # [CLASS DOC] Mapping-protocol test suite adapted from CPython, run on WeakIdKeyDictionary
    # with tensor keys. A class attribute holds a reference dict of three tensor keys; the
    # constructor derives self.reference (all but the held-out pair), self.other (a pair that is
    # not in the mapping) and self.inmapping (a pair that is in the mapping). Tests cover read,
    # write, constructor, bool, keys, values, items, len, getitem, update, get, setdefault,
    # popitem and pop.
    __ref = {torch.randn(1): 1, torch.randn(2): 2, torch.randn(3): 3}
    type2test = WeakIdKeyDictionary

    def _reference(self):
        # [FUNCTION DOC] HELPER: Return a shallow copy of the class-level reference dict.
        return self.__ref.copy()

    def _empty_mapping(self):
        # [FUNCTION DOC] HELPER: Return a new empty mapping of the type under test.
        """Return an empty mapping object"""
        return self.type2test()

    def _full_mapping(self, data):
        # [FUNCTION DOC] HELPER: Return a new mapping of the type under test filled with the
        # items of the given dict.
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    def __init__(self, *args, **kw):
        # [FUNCTION DOC] HELPER: Initialize the test case, then build the fixture dicts. Pops
        # one pair into self.other (not part of the mapping), pops a second into self.inmapping
        # and puts it back into self.reference, so the reference holds two entries.
        unittest.TestCase.__init__(self, *args, **kw)
        self.reference = self._reference().copy()

        # A (key, value) pair not in the mapping
        key, value = self.reference.popitem()
        self.other = {key: value}

        # A (key, value) pair in the mapping
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value

    def test_read(self):
        # [FUNCTION DOC] Check read-only mapping behavior of a mapping filled from the
        # reference: indexing, KeyError for an absent key, len, membership, equality and
        # inequality against an empty mapping, truthiness, iteration over keys/values/items, and
        # get with defaults.
        # Test for read only operations on mapping
        p = self._empty_mapping()
        p1 = dict(p)  # workaround for singleton objects
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # Indexing
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # cmp
        self.assertTrue(
            p == p
        )  # NB: don't use assertEqual, that doesn't actually use ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # bool
        if p:
            self.fail("Empty mapping must compare to False")
        if not d:
            self.fail("Full mapping must compare to True")

        # keys(), items(), iterkeys() ...
        def check_iterandlist(iter, lst, ref):
            # [FUNCTION DOC] HELPER: Check that the given iterator has iteration protocol
            # methods and that, once listed, its elements form the same set as the supplied list
            # and the reference view.
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        # [FUNCTION DOC] Check mutation behavior: setting and deleting items, update from a
        # mapping and from a list of pairs, setdefault for present and absent keys, pop with and
        # without a default (KeyError when missing and no default), and popitem, including
        # KeyError on an empty mapping.
        # Test for write operations on mapping
        p = self._empty_mapping()
        # Indexing
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference:
            del p[key]
            self.assertRaises(KeyError, lambda: p[key])
        p = self._empty_mapping()
        # update
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        # setdefault
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        # pop
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        # popitem
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        # [FUNCTION DOC] Check that two empty mappings compare equal.
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        # [FUNCTION DOC] Check that an empty mapping is falsy and a non-empty one is truthy.
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        # [FUNCTION DOC] Check keys on an empty mapping gives an empty list, that membership of
        # the in-mapping and other keys is reported correctly on the reference dict, and that
        # passing an argument to keys raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        self.assertNotIn(next(iter(self.other.keys())), d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        # [FUNCTION DOC] Check values on an empty mapping is empty and that passing an argument
        # raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        # [FUNCTION DOC] Check items on an empty mapping is empty and that passing an argument
        # raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        # [FUNCTION DOC] Check that an empty mapping has length zero.
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        # [FUNCTION DOC] Check indexing returns the stored value for the in-mapping key on the
        # reference dict, and that calling the getitem method with no key raises TypeError.
        d = self.reference
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        # [FUNCTION DOC] Check update with a mapping, with no argument, with item pairs and with
        # an iterator; a non-iterable integer raises TypeError or AttributeError. Also checks
        # update with a minimal object providing keys and getitem, and that exceptions raised by
        # the keys method, by iteration, by getitem or by a failing iterator propagate
        # unchanged, and that a triple instead of a pair raises ValueError.
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
            # [CLASS DOC] HELPER: Minimal mapping-like object exposing keys and getitem over the
            # reference dict, used to test the update protocol for non-dict objects.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Capture the reference dict of the enclosing test case.
                self.d = outerself.reference

            def keys(self):
                # [FUNCTION DOC] HELPER: Return the keys view of the captured dict.
                return self.d.keys()

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the value stored for the given key in the
                # captured dict.
                return self.d[i]

        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted((id(k), v) for k, v in d.items())
        i2 = sorted((id(k), v) for k, v in self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception):
            # [CLASS DOC] HELPER: Marker exception used to confirm that errors from user
            # objects propagate.
            pass

        d = self._empty_mapping()

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys method always raises the
            # marker exception.
            def keys(self):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d.clear()

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys iterator yields one key and
            # then raises the marker exception; getitem echoes the key.
            def keys(self):
                # [FUNCTION DOC] HELPER: Return an iterator that yields a single string and then
                # raises the marker exception.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator yielding one item then raising the
                    # marker exception.
                    def __init__(self) -> None:
                        # [FUNCTION DOC] HELPER: Set a counter so exactly one item is yielded.
                        self.i = 1

                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Return self as the iterator.
                        return self

                    def __next__(self):
                        # [FUNCTION DOC] HELPER: On the first call return a single string and
                        # clear the counter; on later calls raise the marker exception.
                        if self.i:
                            self.i = 0
                            return "a"
                        raise Exc

                return BogonIter()

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the key unchanged.
                return key

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys iterate over letters a to z but
            # whose getitem always raises the marker exception.
            def keys(self):
                # [FUNCTION DOC] HELPER: Return an iterator over the lowercase letters.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator over the lowercase letters a through z.
                    def __init__(self) -> None:
                        # [FUNCTION DOC] HELPER: Start at the code point of the letter a.
                        self.i = ord("a")

                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Return self as the iterator.
                        return self

                    def __next__(self):
                        # [FUNCTION DOC] HELPER: Return the next letter and advance, raising
                        # StopIteration after z.
                        if self.i <= ord("z"):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration

                return BogonIter()

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d = self._empty_mapping()

        class badseq:
            # [CLASS DOC] HELPER: Iterable whose next step always raises the marker exception.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self as the iterator.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, badseq())

        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    # no test_fromkeys or test_copy as both os.environ and selves don't support it

    def test_get(self):
        # [FUNCTION DOC] Check get returns None or the supplied default for an absent key on
        # both an empty mapping and the reference dict, returns the stored value for a present
        # key, and raises TypeError when called with no key or too many arguments.
        d = self._empty_mapping()
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        d = self.reference
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        # [FUNCTION DOC] Check that setdefault with no arguments raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError, and with an extra
        # argument raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        # [FUNCTION DOC] Check pop of an absent key raises KeyError, pop of a present key
        # returns its value and empties the mapping, and a second pop of it raises KeyError.
        d = self._empty_mapping()
        k, v = next(iter(self.inmapping.items()))
        d[k] = v
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)


# Adapted from cpython/Lib/test/mapping_tests.py
class WeakKeyDictionaryScriptObjectTestCase(TestCase):
    # [CLASS DOC] Same mapping-protocol suite as above, but keys are TorchScript custom class
    # (torchbind _Foo) objects and the dictionary uses the hash-based weak reference type.
    # Skipped on macOS; the constructor skips fixture setup if the torchbind test library cannot
    # be loaded.
    def _reference(self):
        # [FUNCTION DOC] HELPER: Create three new torchbind Foo objects as keys of a reference
        # dict, keep the dict on the instance, and return a shallow copy of it.
        self.__ref = {
            torch.classes._TorchScriptTesting._Foo(1, 2): 1,
            torch.classes._TorchScriptTesting._Foo(2, 3): 2,
            torch.classes._TorchScriptTesting._Foo(3, 4): 3,
        }
        return self.__ref.copy()

    def _empty_mapping(self):
        # [FUNCTION DOC] HELPER: Return an empty weak dictionary using the hash-based weak
        # reference type.
        """Return an empty mapping object"""
        return WeakIdKeyDictionary(ref_type=_WeakHashRef)

    def _full_mapping(self, data):
        # [FUNCTION DOC] HELPER: Return a weak dictionary filled with the items of the
        # given dict.
        """Return a mapping object with the value contained in data
        dictionary"""
        x = self._empty_mapping()
        for key, value in data.items():
            x[key] = value
        return x

    def setUp(self):
        # [FUNCTION DOC] HELPER: Run base setup, then skip the test on macOS because the test
        # library loading call is not portable there.
        super().setUp()
        if IS_MACOS:
            raise unittest.SkipTest("non-portable load_library call used in test")

    def __init__(self, *args, **kw):
        # [FUNCTION DOC] HELPER: Initialize the test case, then load the torchbind test library
        # (returns early without fixtures if loading raises a skip) and build the same
        # reference, other and inmapping fixtures as the tensor suite.
        unittest.TestCase.__init__(self, *args, **kw)
        try:
            load_torchbind_test_lib()
        except unittest.SkipTest:
            return  # Skip in setup

        self.reference = self._reference().copy()

        # A (key, value) pair not in the mapping
        key, value = self.reference.popitem()
        self.other = {key: value}

        # A (key, value) pair in the mapping
        key, value = self.reference.popitem()
        self.inmapping = {key: value}
        self.reference[key] = value

    def test_read(self):
        # [FUNCTION DOC] Check read-only mapping behavior of a mapping filled from the
        # reference: indexing, KeyError for an absent key, len, membership, equality and
        # inequality against an empty mapping, truthiness, iteration over keys/values/items, and
        # get with defaults.
        # Test for read only operations on mapping
        p = self._empty_mapping()
        p1 = dict(p)  # workaround for singleton objects
        d = self._full_mapping(self.reference)
        if d is p:
            p = p1
        # Indexing
        for key, value in self.reference.items():
            self.assertEqual(d[key], value)
        knownkey = next(iter(self.other.keys()))
        self.assertRaises(KeyError, lambda: d[knownkey])
        # len
        self.assertEqual(len(p), 0)
        self.assertEqual(len(d), len(self.reference))
        # __contains__
        for k in self.reference:
            self.assertIn(k, d)
        for k in self.other:
            self.assertNotIn(k, d)
        # cmp
        self.assertTrue(
            p == p
        )  # NB: don't use assertEqual, that doesn't actually use ==
        self.assertTrue(d == d)
        self.assertTrue(p != d)
        self.assertTrue(d != p)
        # bool
        if p:
            self.fail("Empty mapping must compare to False")
        if not d:
            self.fail("Full mapping must compare to True")

        # keys(), items(), iterkeys() ...
        def check_iterandlist(iter, lst, ref):
            # [FUNCTION DOC] HELPER: Check that the given iterator has iteration protocol
            # methods and that, once listed, its elements form the same set as the supplied list
            # and the reference view.
            self.assertTrue(hasattr(iter, "__next__"))
            self.assertTrue(hasattr(iter, "__iter__"))
            x = list(iter)
            self.assertTrue(set(x) == set(lst) == set(ref))

        check_iterandlist(iter(d.keys()), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d), list(d.keys()), self.reference.keys())
        check_iterandlist(iter(d.values()), list(d.values()), self.reference.values())
        check_iterandlist(iter(d.items()), list(d.items()), self.reference.items())
        # get
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.get(key, knownvalue), value)
        self.assertEqual(d.get(knownkey, knownvalue), knownvalue)
        self.assertNotIn(knownkey, d)

    def test_write(self):
        # [FUNCTION DOC] Check mutation behavior: setting and deleting items, update from a
        # mapping and from a list of pairs, setdefault for present and absent keys, pop with and
        # without a default (KeyError when missing and no default), and popitem, including
        # KeyError on an empty mapping.
        # Test for write operations on mapping
        p = self._empty_mapping()
        # Indexing
        for key, value in self.reference.items():
            p[key] = value
            self.assertEqual(p[key], value)
        for key in self.reference:
            del p[key]
            self.assertRaises(KeyError, lambda: p[key])
        p = self._empty_mapping()
        # update
        p.update(self.reference)
        self.assertEqual(dict(p), self.reference)
        items = list(p.items())
        p = self._empty_mapping()
        p.update(items)
        self.assertEqual(dict(p), self.reference)
        d = self._full_mapping(self.reference)
        # setdefault
        key, value = next(iter(d.items()))
        knownkey, knownvalue = next(iter(self.other.items()))
        self.assertEqual(d.setdefault(key, knownvalue), value)
        self.assertEqual(d[key], value)
        self.assertEqual(d.setdefault(knownkey, knownvalue), knownvalue)
        self.assertEqual(d[knownkey], knownvalue)
        # pop
        self.assertEqual(d.pop(knownkey), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertRaises(KeyError, d.pop, knownkey)
        default = 909
        d[knownkey] = knownvalue
        self.assertEqual(d.pop(knownkey, default), knownvalue)
        self.assertNotIn(knownkey, d)
        self.assertEqual(d.pop(knownkey, default), default)
        # popitem
        key, value = d.popitem()
        self.assertNotIn(key, d)
        self.assertEqual(value, self.reference[key])
        p = self._empty_mapping()
        self.assertRaises(KeyError, p.popitem)

    def test_constructor(self):
        # [FUNCTION DOC] Check that two empty mappings compare equal.
        self.assertEqual(self._empty_mapping(), self._empty_mapping())

    def test_bool(self):
        # [FUNCTION DOC] Check that an empty mapping is falsy and a non-empty one is truthy.
        self.assertTrue(not self._empty_mapping())
        self.assertTrue(self.reference)
        self.assertTrue(bool(self._empty_mapping()) is False)
        self.assertTrue(bool(self.reference) is True)

    def test_keys(self):
        # [FUNCTION DOC] Check keys on an empty mapping gives an empty list, that membership of
        # the in-mapping and other keys is reported correctly on the reference dict, and that
        # passing an argument to keys raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.keys()), [])
        d = self.reference
        self.assertIn(next(iter(self.inmapping.keys())), d.keys())
        self.assertNotIn(next(iter(self.other.keys())), d.keys())
        self.assertRaises(TypeError, d.keys, None)

    def test_values(self):
        # [FUNCTION DOC] Check values on an empty mapping is empty and that passing an argument
        # raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.values()), [])

        self.assertRaises(TypeError, d.values, None)

    def test_items(self):
        # [FUNCTION DOC] Check items on an empty mapping is empty and that passing an argument
        # raises TypeError.
        d = self._empty_mapping()
        self.assertEqual(list(d.items()), [])

        self.assertRaises(TypeError, d.items, None)

    def test_len(self):
        # [FUNCTION DOC] Check that an empty mapping has length zero.
        d = self._empty_mapping()
        self.assertEqual(len(d), 0)

    def test_getitem(self):
        # [FUNCTION DOC] Check indexing returns the stored value for the in-mapping key on the
        # reference dict, and that calling the getitem method with no key raises TypeError.
        d = self.reference
        self.assertEqual(
            d[next(iter(self.inmapping.keys()))], next(iter(self.inmapping.values()))
        )

        self.assertRaises(TypeError, d.__getitem__)

    def test_update(self):
        # [FUNCTION DOC] Check update with a mapping, with no argument, with item pairs and with
        # an iterator; a non-iterable integer raises TypeError or AttributeError. Also checks
        # update with a minimal object providing keys and getitem, and that exceptions raised by
        # the keys method, by iteration, by getitem or by a failing iterator propagate
        # unchanged, and that a triple instead of a pair raises ValueError.
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
            # [CLASS DOC] HELPER: Minimal mapping-like object exposing keys and getitem over the
            # reference dict, used to test the update protocol for non-dict objects.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Capture the reference dict of the enclosing test case.
                self.d = outerself.reference

            def keys(self):
                # [FUNCTION DOC] HELPER: Return the keys view of the captured dict.
                return self.d.keys()

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the value stored for the given key in the
                # captured dict.
                return self.d[i]

        d.clear()
        d.update(SimpleUserDict())
        i1 = sorted((id(k), v) for k, v in d.items())
        i2 = sorted((id(k), v) for k, v in self.reference.items())
        self.assertEqual(i1, i2)

        class Exc(Exception):
            # [CLASS DOC] HELPER: Marker exception used to confirm that errors from user
            # objects propagate.
            pass

        d = self._empty_mapping()

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys method always raises the
            # marker exception.
            def keys(self):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d.clear()

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys iterator yields one key and
            # then raises the marker exception; getitem echoes the key.
            def keys(self):
                # [FUNCTION DOC] HELPER: Return an iterator that yields a single string and then
                # raises the marker exception.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator yielding one item then raising the
                    # marker exception.
                    def __init__(self) -> None:
                        # [FUNCTION DOC] HELPER: Set a counter so exactly one item is yielded.
                        self.i = 1

                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Return self as the iterator.
                        return self

                    def __next__(self):
                        # [FUNCTION DOC] HELPER: On the first call return a single string and
                        # clear the counter; on later calls raise the marker exception.
                        if self.i:
                            self.i = 0
                            return "a"
                        raise Exc

                return BogonIter()

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the key unchanged.
                return key

        self.assertRaises(Exc, d.update, FailingUserDict())

        class FailingUserDict:
            # [CLASS DOC] HELPER: Mapping-like object whose keys iterate over letters a to z but
            # whose getitem always raises the marker exception.
            def keys(self):
                # [FUNCTION DOC] HELPER: Return an iterator over the lowercase letters.
                class BogonIter:
                    # [CLASS DOC] HELPER: Iterator over the lowercase letters a through z.
                    def __init__(self) -> None:
                        # [FUNCTION DOC] HELPER: Start at the code point of the letter a.
                        self.i = ord("a")

                    def __iter__(self):
                        # [FUNCTION DOC] HELPER: Return self as the iterator.
                        return self

                    def __next__(self):
                        # [FUNCTION DOC] HELPER: Return the next letter and advance, raising
                        # StopIteration after z.
                        if self.i <= ord("z"):
                            rtn = chr(self.i)
                            self.i += 1
                            return rtn
                        raise StopIteration

                return BogonIter()

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, FailingUserDict())

        d = self._empty_mapping()

        class badseq:
            # [CLASS DOC] HELPER: Iterable whose next step always raises the marker exception.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return self as the iterator.
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Always raise the marker exception.
                raise Exc

        self.assertRaises(Exc, d.update, badseq())

        self.assertRaises(ValueError, d.update, [(1, 2, 3)])

    # no test_fromkeys or test_copy as both os.environ and selves don't support it

    def test_get(self):
        # [FUNCTION DOC] Check get returns None or the supplied default for an absent key on
        # both an empty mapping and the reference dict, returns the stored value for a present
        # key, and raises TypeError when called with no key or too many arguments.
        d = self._empty_mapping()
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        d = self.reference
        self.assertTrue(d.get(next(iter(self.other.keys()))) is None)
        self.assertEqual(d.get(next(iter(self.other.keys())), 3), 3)
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys()))),
            next(iter(self.inmapping.values())),
        )
        self.assertEqual(
            d.get(next(iter(self.inmapping.keys())), 3),
            next(iter(self.inmapping.values())),
        )
        self.assertRaises(TypeError, d.get)
        self.assertRaises(TypeError, d.get, None, None, None)

    def test_setdefault(self):
        # [FUNCTION DOC] Check that setdefault with no arguments raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(TypeError, d.setdefault)

    def test_popitem(self):
        # [FUNCTION DOC] Check popitem on an empty mapping raises KeyError, and with an extra
        # argument raises TypeError.
        d = self._empty_mapping()
        self.assertRaises(KeyError, d.popitem)
        self.assertRaises(TypeError, d.popitem, 42)

    def test_pop(self):
        # [FUNCTION DOC] Check pop of an absent key raises KeyError, pop of a present key
        # returns its value and empties the mapping, and a second pop of it raises KeyError.
        d = self._empty_mapping()
        k, v = next(iter(self.inmapping.items()))
        d[k] = v
        self.assertRaises(KeyError, d.pop, next(iter(self.other.keys())))

        self.assertEqual(d.pop(k), v)
        self.assertEqual(len(d), 0)

        self.assertRaises(KeyError, d.pop, k)


if __name__ == "__main__":
    run_tests()
