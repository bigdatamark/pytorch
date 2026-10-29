# Owner(s): ["module: dynamo"]

# [MODULE DOC] Dynamo tests for mapping item assignment and deletion. Defines helper mapping
# types (a user-defined mapping class, a dict subclass, a defaultdict factory) and a list of
# (label, factory) pairs used to parametrize tests, then one test class checking setitem/delitem
# on builtin mappings, user mappings and dict subclasses, plus mutation of outer dicts from a
# compiled function. Calls instantiate_parametrized_tests on the class at import time.
"""Tests for mapping protocol operations (mp_*) in PyTorch Dynamo.

Covers mp_ass_subscript (__setitem__) on builtin mappings, user-defined
mappings, and dict subclasses.
"""

import collections

import torch
import torch._dynamo.test_case
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


# ---------------------------------------------------------------------------
# Helper classes
# ---------------------------------------------------------------------------


class WithSetitemViaMapping:
    # [CLASS DOC] HELPER: Minimal user-defined (non-dict) mapping. Wraps a plain dict in the data
    # attribute, defaulting to a new empty dict when none (or an empty one) is given, and
    # forwards item get/set, membership and keys/values/items to it. It deliberately defines no
    # item deletion.
    """Custom mapping with __setitem__ via mp_ass_subscript."""

    def __init__(self, data=None):
        # [FUNCTION DOC] HELPER: Store the given mapping as data, or a fresh empty dict when the
        # argument is empty or None.
        self.data = data or {}

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Return the value stored for the key in the wrapped dict; a
        # missing key raises KeyError.
        return self.data[key]

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Store the value under the key in the wrapped dict, replacing any
        # previous value.
        self.data[key] = value

    def __contains__(self, key):
        # [FUNCTION DOC] HELPER: Return whether the key is present in the wrapped dict.
        return key in self.data

    def keys(self):
        # [FUNCTION DOC] HELPER: Return the key view of the wrapped dict.
        return self.data.keys()

    def values(self):
        # [FUNCTION DOC] HELPER: Return the value view of the wrapped dict.
        return self.data.values()

    def items(self):
        # [FUNCTION DOC] HELPER: Return the item view of the wrapped dict.
        return self.data.items()


class _DictSubclassParam(dict):
    # [CLASS DOC] HELPER: Trivial dict subclass with no overrides, used as a parametrization case
    # for subclass behavior.
    pass


def _make_defaultdict(data):
    # [FUNCTION DOC] HELPER: Factory building a defaultdict whose default factory is the int type
    # from the given data.
    return collections.defaultdict(int, data)


# (label, factory) — factory builds an instance from a dict.
_MAPPING_TYPES = [
    ("dict", dict),
    ("OrderedDict", collections.OrderedDict),
    ("Counter", collections.Counter),
    ("defaultdict", _make_defaultdict),
    ("DictSubclass", _DictSubclassParam),
    ("WithSetitemViaMapping", WithSetitemViaMapping),
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMpAssSubscript(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for item assignment and deletion on mappings under Dynamo. The setUp
    # and tearDown methods enable and then restore two Dynamo trace config flags. Most tests use
    # make_dynamo_test (eager run then compiled run); parametrized tests are expanded over
    # mapping types by instantiate_parametrized_tests.
    """All mapping __setitem__ tests in one class."""

    def setUp(self):
        # [FUNCTION DOC] Save the current values of the enable_trace_unittest and
        # enable_trace_load_build_class Dynamo configs, then turn both on for the test.
        super().setUp()
        self._u_prev = torch._dynamo.config.enable_trace_unittest
        self._b_prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_unittest = True
        torch._dynamo.config.enable_trace_load_build_class = True

    def tearDown(self):
        # [FUNCTION DOC] Run base teardown, then restore the two Dynamo config flags saved in
        # setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._u_prev
        torch._dynamo.config.enable_trace_load_build_class = self._b_prev

    # -- parameterized over mapping container type --

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_basic(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each mapping type, build it from three entries, assign a new value to an existing
        # key and check the new value is read back.
        c = factory({"a": 1, "b": 2, "c": 3})
        c["b"] = 100
        self.assertEqual(c["b"], 100)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_multiple(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each mapping type, assign new values to two existing keys and check both are read
        # back.
        c = factory({"a": 1, "b": 2, "c": 3})
        c["a"] = 100
        c["b"] = 200
        self.assertEqual(c["a"], 100)
        self.assertEqual(c["b"], 200)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_replaces_value(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each mapping type, replace the value of an existing key with another and check it.
        # Nearly identical to the basic setitem test.
        c = factory({"a": 1, "b": 2, "c": 3})
        c["b"] = 999
        self.assertEqual(c["b"], 999)

    @parametrize("label,factory", _MAPPING_TYPES)
    @make_dynamo_test
    def test_setitem_new_key(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each mapping type, build an empty instance, assign a new key and check it reads
        # back.
        c = factory({})
        c["new"] = 42
        self.assertEqual(c["new"], 42)

    # -- dict subclass --

    @make_dynamo_test
    def test_subclass_dict_inherited_setitem(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass without overrides supports assigning and reading two keys.
        class D(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

    @make_dynamo_test
    def test_subclass_dict_with_missing(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass defining a missing-key hook returning minus one: an assigned key reads
        # back its value, while an absent key reads as minus one.
        class D(dict):
            # [CLASS DOC] HELPER: Dict subclass whose missing-key hook returns minus one for
            # absent keys.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Return minus one for any missing key without inserting
                # it.
                return -1

        d = D()
        d["a"] = 5
        self.assertEqual(d["a"], 5)
        self.assertEqual(d["unknown"], -1)

    @make_dynamo_test
    def test_subclass_dict_track_writes(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass that records every item assignment (key, value) in a writes list while
        # delegating to the base. Checks the recorded writes and values after two assignments,
        # then that a second instance built from initial data records only the later overwrite
        # (not the constructor data).
        class D(dict):
            # [CLASS DOC] HELPER: Dict subclass keeping a writes list on the instance, appended
            # to by each item assignment before delegating to the base implementation.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Initialize the base dict with the given arguments and
                # create an empty writes list.
                super().__init__(*args, **kwargs)
                self.writes = []

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Append the (key, value) pair to the writes list, then
                # store it in the base dict.
                self.writes.append((key, value))
                super().__setitem__(key, value)

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d.writes, [("a", 1), ("b", 2)])
        self.assertEqual(d["a"], 1)
        self.assertEqual(d["b"], 2)

        # also exercise replace-existing
        d2 = D({"a": 1})
        d2["a"] = 99
        self.assertEqual(d2["a"], 99)
        self.assertEqual(d2.writes, [("a", 99)])

    @make_dynamo_test
    def test_subclass_dict_overriding(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass whose item assignment adds one thousand to the value before storing;
        # assigning five reads back as 1005.
        class D(dict):
            # [CLASS DOC] HELPER: Dict subclass whose item assignment stores the value increased
            # by one thousand.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store the key with the value plus one thousand using the
                # base dict implementation.
                super().__setitem__(key, value + 1000)

        d = D()
        d["a"] = 5
        self.assertEqual(d["a"], 1005)

    @make_dynamo_test
    def test_subclass_ordereddict(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # an OrderedDict subclass without overrides keeps insertion order of assigned keys and
        # reads values back.
        class D(collections.OrderedDict):
            # [CLASS DOC] HELPER: Empty OrderedDict subclass.
            pass

        d = D()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(list(d.keys()), ["a", "b"])
        self.assertEqual(d["a"], 1)

    # -- key / value variation --

    @make_dynamo_test
    def test_dict_various_key_types(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict accepts a string key and an integer key on assignment and returns both values.
        d = {}
        d["key1"] = "value1"
        d[42] = "value42"
        self.assertEqual(d["key1"], "value1")
        self.assertEqual(d[42], "value42")

    @make_dynamo_test
    def test_dict_various_value_types(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict accepts integer, string, list and nested-dict values on assignment and returns
        # equal values.
        d = {}
        d["int"] = 42
        d["str"] = "hello"
        d["list"] = [1, 2, 3]
        d["dict"] = {"nested": True}
        self.assertEqual(d["int"], 42)
        self.assertEqual(d["str"], "hello")
        self.assertEqual(d["list"], [1, 2, 3])
        self.assertEqual(d["dict"], {"nested": True})

    @make_dynamo_test
    def test_symnode_dict_value(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # tensor's first dimension size is stored as a dict value and reads back as five.
        t = torch.randn(5)
        d = {}
        d["len"] = t.shape[0]
        self.assertEqual(d["len"], 5)

    # -- mutation visibility --

    def test_mutation_outer_dict_persists(self):
        # [FUNCTION DOC] Compile (eager backend, full graph) a function that assigns an existing
        # and a new key in a dict captured from the enclosing scope, call it once, and check the
        # outer dict now holds both updates.
        outer = {"a": 1}

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function: mutate the captured outer dict (overwrite
            # one key, add another) and return the input plus one.
            outer["a"] = 99
            outer["b"] = 100
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, {"a": 99, "b": 100})

    @make_dynamo_test
    def test_mutation_nested_dict(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # mutating a list stored inside a dict through an alias is visible both via the alias and
        # via the dict.
        d = {"list": [1, 2, 3]}
        lst = d["list"]
        lst[0] = 99
        self.assertEqual(lst[0], 99)
        self.assertEqual(d["list"][0], 99)

    # -- errors --

    @make_dynamo_test
    def test_error_dict_unhashable_key(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # assigning with an unhashable (list) key raises TypeError.
        d = {}
        with self.assertRaises(TypeError):
            d[[1, 2]] = 1  # type: ignore[index]

    # -- __delitem__ (NULL value via mp_ass_subscript) --
    # WithSetitemViaMapping has no __delitem__; restrict to dict-like types.

    @parametrize(
        "label,factory",
        [
            ("dict", dict),
            ("OrderedDict", collections.OrderedDict),
            ("Counter", collections.Counter),
            ("defaultdict", _make_defaultdict),
            ("DictSubclass", _DictSubclassParam),
        ],
    )
    @make_dynamo_test
    def test_delitem_basic(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each dict-like mapping type (not the user mapping, which lacks deletion), delete
        # one key and check it is gone while the others remain.
        c = factory({"a": 1, "b": 2, "c": 3})
        del c["b"]
        self.assertNotIn("b", c)
        self.assertEqual(c["a"], 1)
        self.assertEqual(c["c"], 3)

    @parametrize(
        "label,factory",
        [
            ("dict", dict),
            ("OrderedDict", collections.OrderedDict),
            ("Counter", collections.Counter),
            ("defaultdict", _make_defaultdict),
            ("DictSubclass", _DictSubclassParam),
        ],
    )
    @make_dynamo_test
    def test_delitem_then_set(self, label, factory):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # for each dict-like mapping type, delete a key then assign it again and check the new
        # value.
        c = factory({"a": 1, "b": 2})
        del c["a"]
        c["a"] = 99
        self.assertEqual(c["a"], 99)

    @make_dynamo_test
    def test_delitem_dict_missing_key(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # deleting a missing key from a dict raises KeyError.
        d = {"a": 1}
        with self.assertRaises(KeyError):
            del d["missing"]

    @make_dynamo_test
    def test_delitem_dict_all_keys(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # deleting every key of a dict leaves it empty.
        d = {"a": 1, "b": 2, "c": 3}
        del d["a"]
        del d["b"]
        del d["c"]
        self.assertEqual(len(d), 0)

    @make_dynamo_test
    def test_delitem_dict_unhashable_key(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed):
        # deleting with an unhashable (list) key raises TypeError.
        d = {"a": 1}
        with self.assertRaises(TypeError):
            del d[[1, 2]]  # type: ignore[index]

    @make_dynamo_test
    def test_delitem_subclass_dict_inherited(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass without overrides supports deleting a key, leaving the remaining entries.
        class D(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        d = D({"a": 1, "b": 2})
        del d["a"]
        self.assertEqual(dict(d), {"b": 2})

    @make_dynamo_test
    def test_delitem_subclass_dict_override(self):
        # [FUNCTION DOC] Dynamo test (runs eagerly, then compiled with no graph break allowed): a
        # dict subclass overriding item deletion to record deleted keys: after two deletions the
        # remaining content and the ordered deleted-key list are checked.
        class D(dict):
            # [CLASS DOC] HELPER: Dict subclass keeping a deleted list on the instance, appended
            # to by each item deletion before delegating to the base implementation.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Initialize the base dict with the given arguments and
                # create an empty deleted list.
                super().__init__(*args, **kwargs)
                self.deleted = []

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Append the key to the deleted list, then delete it from
                # the base dict.
                self.deleted.append(key)
                super().__delitem__(key)

        d = D({"a": 1, "b": 2, "c": 3})
        del d["a"]
        del d["c"]
        self.assertEqual(dict(d), {"b": 2})
        self.assertEqual(d.deleted, ["a", "c"])

    def test_delitem_outer_dict_persists(self):
        # [FUNCTION DOC] Compile (eager backend, full graph) a function that deletes a key from a
        # dict captured from the enclosing scope, call it once, and check the outer dict lost
        # that key.
        outer = {"a": 1, "b": 2}

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function: delete one key from the captured outer
            # dict and return the input plus one.
            del outer["a"]
            return x + 1

        f(torch.zeros(1))
        self.assertEqual(outer, {"b": 2})


instantiate_parametrized_tests(TestMpAssSubscript)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
