# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo's unified hash protocol (hash builtin and the hash dunder method)
# across many variable types: constants, tuples, frozensets, ranges, slices, unhashable containers,
# dict views, tensors, symbolic ints, functions, user classes with default or custom or missing
# hash, partials, dataclasses, weakrefs, bound methods, builtin and C-level hash types, typing
# forms, torch ops, distributed members and opaque custom classes. Uses the eager backend with full
# graph so any graph break fails the test. No import-time side effects.
"""Tests for hash_impl: unified hash() / __hash__ protocol in Dynamo."""

import sys

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TpHashTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for hash behavior under torch.compile. Provides two shared assertion
    # helpers (compare with eager hash value, and check the TypeError message for unhashable
    # values) and one test per variable type or scenario, most comparing Dynamo output with CPython
    # or using the value as a dictionary key.
    def _assert_hash_equals(self, value):
        # [FUNCTION DOC] HELPER: Computes the hash of the given value in plain Python, compiles a
        # function (eager backend, full graph) that returns the hash of the same value, runs it on
        # a dummy tensor, and asserts both hashes are equal.
        """Assert Dynamo's hash(value) matches eager Python's hash(value)."""
        expected = hash(value)

        def fn(_):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the hash of the
            # captured value.
            return hash(value)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def _assert_hash_raises(self, make_value, *, expect_msg):
        # [FUNCTION DOC] HELPER: Takes a factory for a value and a required expected message
        # substring. Compiles (eager backend, full graph) a function that hashes the factory
        # result, catching TypeError and returning its text, then asserts the expected message
        # appears in the returned text.
        """Assert hash(make_value()) raises TypeError with expect_msg under Dynamo."""

        def fn(_):
            # [FUNCTION DOC] HELPER: Ignores its argument, hashes the value produced by the factory
            # and returns the message of a TypeError if one is raised.
            try:
                return hash(make_value())
            except TypeError as e:
                return str(e)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertIn(expect_msg, result)

    # --- Constants (ConstantVariable) ---

    def test_hash_int(self):
        # [FUNCTION DOC] Hash of the integer 42 under Dynamo equals the eager hash.
        self._assert_hash_equals(42)

    def test_hash_str(self):
        # [FUNCTION DOC] Hash of a string constant equals the eager hash.
        self._assert_hash_equals("hello")

    def test_hash_bool(self):
        # [FUNCTION DOC] Hash of the boolean True equals the eager hash.
        self._assert_hash_equals(True)

    def test_hash_none(self):
        # [FUNCTION DOC] Hash of None equals the eager hash.
        self._assert_hash_equals(None)

    def test_hash_float(self):
        # [FUNCTION DOC] Hash of a float constant equals the eager hash.
        self._assert_hash_equals(3.14)

    # --- Tuples (TupleVariable) ---

    def test_hash_tuple(self):
        # [FUNCTION DOC] Hash of a tuple of three ints equals the eager hash.
        self._assert_hash_equals((1, 2, 3))

    def test_hash_tuple_strings(self):
        # [FUNCTION DOC] Hash of a tuple of two strings equals the eager hash.
        self._assert_hash_equals(("hello", "world"))

    def test_hash_tuple_mixed(self):
        # [FUNCTION DOC] Hash of a tuple mixing int, string, float, None and bool equals the
        # eager hash.
        self._assert_hash_equals((1, "hello", 3.14, None, True))

    def test_hash_empty_tuple(self):
        # [FUNCTION DOC] Hash of the empty tuple equals the eager hash.
        self._assert_hash_equals(())

    def test_hash_nested_tuple(self):
        # [FUNCTION DOC] Hash of a nested tuple of tuples equals the eager hash.
        self._assert_hash_equals(((1, 2), (3, 4)))

    def test_hash_tuple_with_tensor(self):
        # [FUNCTION DOC] A tuple containing a captured tensor can be hashed under Dynamo; two
        # hashes of the same tuple within the compiled function must be equal (this takes the
        # non-constant fallback path).
        """Tuple containing a tensor exercises the _RawHash fallback path
        (not the constant fast-path), and the hash is consistent."""
        t = torch.tensor([1.0])

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a tuple of an int, a string and the captured tensor and
            # returns its hash twice.
            tup = (1, "hello", t)
            return hash(tup), hash(tup)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    def test_hash_tuple_with_tensor_as_dict_key(self):
        # [FUNCTION DOC] A tuple containing a tensor can be used as a dictionary key under Dynamo;
        # lookup returns 99.
        """Tuple containing a tensor can be used as a dict key."""
        t = torch.tensor([1.0])

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a tuple with an int, a string and the captured tensor,
            # stores it as a dictionary key and looks it up.
            tup = (1, "hello", t)
            d = {tup: 99}
            return d[tup]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 99)

    def test_hash_torch_size(self):
        # [FUNCTION DOC] Hash of a torch Size of three entries equals the eager hash.
        self._assert_hash_equals(torch.Size([1, 2, 3]))

    def test_hash_torch_size_from_shape(self):
        # [FUNCTION DOC] A tensor's shape (symbolic-capable sizes, not plain constants) can be used
        # as a dictionary key; lookup returns 42.
        """hash(x.shape) where shape items are symbolic, not plain constants."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the shape of the input, inserts it as a dictionary key
            # with value 42 and reads it back.
            s = x.shape
            d = {}
            d[s] = 42
            return d[s]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3, 4))
        self.assertEqual(result, 42)

    # --- Frozenset (FrozensetVariable) ---

    def test_hash_frozenset(self):
        # [FUNCTION DOC] Hash of a frozenset of three ints equals the eager hash.
        self._assert_hash_equals(frozenset({1, 2, 3}))

    def test_hash_empty_frozenset(self):
        # [FUNCTION DOC] Hash of the empty frozenset equals the eager hash.
        self._assert_hash_equals(frozenset())

    def test_hash_frozenset_with_tensor(self):
        # [FUNCTION DOC] A frozenset containing a tensor can be hashed; two hashes of the same set
        # are equal inside the compiled function.
        """Frozenset containing a tensor exercises the _RawHash fallback."""
        t = torch.tensor(1.0)

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a frozenset of two ints and the captured tensor and
            # returns its hash twice.
            fs = frozenset({1, 2, t})
            return hash(fs), hash(fs)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    # --- Range (RangeVariable) ---

    def test_hash_range(self):
        # [FUNCTION DOC] Hash of range of ten equals the eager hash.
        self._assert_hash_equals(range(10))

    def test_hash_range_with_step(self):
        # [FUNCTION DOC] Hash of a range with a step equals the eager hash.
        self._assert_hash_equals(range(0, 10, 2))

    def test_hash_empty_range(self):
        # [FUNCTION DOC] Hash of an empty range equals the eager hash.
        self._assert_hash_equals(range(0))

    def test_hash_single_element_range(self):
        # [FUNCTION DOC] Hash of a one-element range equals the eager hash.
        self._assert_hash_equals(range(1))

    # --- Slice (SliceVariable) ---

    def test_hash_slice(self):
        # [FUNCTION DOC] Slices: on Python 3.12 and later the hash equals eager; on older versions
        # hashing raises TypeError with the message that slice is unhashable.
        """SliceVariable: CPython slicehash (hashable in 3.12+)."""
        if sys.version_info >= (3, 12):
            self._assert_hash_equals(slice(1, 10, 2))
        else:
            self._assert_hash_raises(
                lambda: slice(1, 10, 2), expect_msg="unhashable type: 'slice'"
            )

    # --- Unhashable types raise TypeError ---

    def test_hash_list_raises(self):
        # [FUNCTION DOC] Hashing a list raises TypeError mentioning that list is unhashable.
        self._assert_hash_raises(lambda: [1, 2], expect_msg="unhashable type: 'list'")

    def test_hash_dict_raises(self):
        # [FUNCTION DOC] Hashing a dict raises TypeError mentioning that dict is unhashable.
        self._assert_hash_raises(lambda: {1: 2}, expect_msg="unhashable type: 'dict'")

    def test_hash_set_raises(self):
        # [FUNCTION DOC] Hashing a set raises TypeError mentioning that set is unhashable.
        self._assert_hash_raises(lambda: {1, 2}, expect_msg="unhashable type: 'set'")

    def test_hash_deque_raises(self):
        # [FUNCTION DOC] Hashing a deque raises TypeError mentioning that deque is unhashable.
        from collections import deque

        self._assert_hash_raises(
            lambda: deque([1, 2]), expect_msg="unhashable type: 'deque'"
        )

    def test_hash_mappingproxy_raises(self):
        # [FUNCTION DOC] Hashing a read-only mapping proxy of a dict raises TypeError whose message
        # names the dict type.
        import types

        self._assert_hash_raises(
            lambda: types.MappingProxyType({1: 2}),
            expect_msg="unhashable type: 'dict'",
        )

    def test_hash_tuple_with_list_raises(self):
        # [FUNCTION DOC] Hashing a tuple that contains a list raises TypeError mentioning list.
        self._assert_hash_raises(
            lambda: ([1, 2], 3), expect_msg="unhashable type: 'list'"
        )

    def test_hash_tuple_with_dict_raises(self):
        # [FUNCTION DOC] Hashing a tuple that contains a dict raises TypeError mentioning dict.
        self._assert_hash_raises(
            lambda: (1, {2: 3}), expect_msg="unhashable type: 'dict'"
        )

    # --- Dict views ---

    def test_hash_dict_keys_raises(self):
        # [FUNCTION DOC] Hashing the keys view of a dict raises TypeError mentioning dict_keys
        # is unhashable.
        self._assert_hash_raises(
            lambda: {1: 2}.keys(), expect_msg="unhashable type: 'dict_keys'"
        )

    def test_hash_dict_items_raises(self):
        # [FUNCTION DOC] Hashing the items view of a dict raises TypeError mentioning dict_items
        # is unhashable.
        self._assert_hash_raises(
            lambda: {1: 2}.items(), expect_msg="unhashable type: 'dict_items'"
        )

    def test_hash_dict_values(self):
        # [FUNCTION DOC] The values view of a dict is hashable by identity (it defines no
        # equality), so its Dynamo hash equals the eager hash.
        """dict_values IS hashable (identity hash) — unlike dict_keys/dict_items,
        it does not define __eq__, so CPython does not set __hash__ = None.

        https://github.com/python/cpython/blob/e76aa128fe/Objects/dictobject.c#L6630
        """
        d = {1: 2, 3: 4}
        self._assert_hash_equals(d.values())

    # --- Tensor (TensorVariable) ---

    def test_hash_tensor_as_dict_key(self):
        # [FUNCTION DOC] A captured tensor can be used as a dictionary key under Dynamo; lookup
        # returns 42.
        """hash(tensor) works for dict key usage."""
        t = torch.tensor([1.0, 2.0])

        def fn(x):
            # [FUNCTION DOC] HELPER: Inserts the captured tensor as a key in a dictionary and looks
            # it up.
            d = {t: 42}
            return d[t]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_tensor_consistent(self):
        # [FUNCTION DOC] Hash of a captured tensor is the same on repeated calls inside one
        # compiled function.
        """hash(tensor) is consistent across calls within a compiled function."""
        t = torch.tensor([1.0, 2.0])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash of the captured tensor twice.
            return hash(t), hash(t)

        h1, h2 = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(h1, h2)

    def test_hash_tensor_return_graph_breaks(self):
        # [FUNCTION DOC] Returning the hash of a tensor from a full-graph compiled function raises
        # Unsupported mentioning FakeIdVariable, because the tensor hash is a placeholder value.
        """Returning hash(tensor) graph-breaks — tensor hash is fake."""
        from torch._dynamo.exc import Unsupported

        t = torch.tensor([1.0, 2.0])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash of the captured tensor.
            return hash(t)

        with self.assertRaisesRegex(Unsupported, "FakeIdVariable"):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))

    # --- SymNodeVariable ---

    def test_hash_symbolic_int(self):
        # [FUNCTION DOC] The hash of a symbolic integer specializes its value; using the first
        # dimension as a dictionary key returns 99.
        """hash() on a symbolic int specializes the value."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the first dimension of the input, uses it as a
            # dictionary key and reads it back.
            s = x.shape[0]
            d = {s: 99}
            return d[s]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 99)

    # --- Functions ---

    def test_hash_function(self):
        # [FUNCTION DOC] A user-defined function can be a dictionary key under Dynamo; lookup
        # returns 42.
        def my_func():
            # [FUNCTION DOC] HELPER: Empty function used as a dictionary key.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses the captured user function as a dictionary key and reads
            # it back.
            d = {my_func: 42}
            return d[my_func]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_lambda(self):
        # [FUNCTION DOC] A lambda can be a dictionary key under Dynamo; lookup returns 42.
        my_lambda = lambda: None  # noqa: E731

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses the captured lambda as a dictionary key and reads
            # it back.
            d = {my_lambda: 42}
            return d[my_lambda]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    # --- User-defined classes ---

    def test_hash_user_object_default(self):
        # [FUNCTION DOC] An object of a class without a custom hash uses the identity hash,
        # matching eager.
        """Class without __hash__ override uses identity hash."""

        class MyClass:
            # [CLASS DOC] HELPER: Empty class with the default hash.
            pass

        self._assert_hash_equals(MyClass())

    def test_hash_user_class_custom_hash(self):
        # [FUNCTION DOC] A class with a custom hash method is traced into with no graph break; the
        # Dynamo hash equals the eager hash (the stored value times 31).
        """Class with custom __hash__ is traced into, no graph break."""

        class MyObj:
            # [CLASS DOC] HELPER: Class holding a value and defining a hash equal to the value
            # times 31.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the given value.
                self.x = x

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value times 31.
                return self.x * 31

        self._assert_hash_equals(MyObj(7))

    def test_hash_user_class_custom_hash_as_dict_key(self):
        # [FUNCTION DOC] A class defining hash and equality can be a dictionary key under Dynamo;
        # lookup with the same key returns 99.
        """Custom __hash__ works for dict key usage."""

        class MyKey:
            # [CLASS DOC] HELPER: Key class that hashes and compares by its stored value.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Stores the given value.
                self.v = v

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the hash of the stored value.
                return hash(self.v)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns true if the other object is the same key class
                # with an equal stored value.
                return isinstance(other, MyKey) and self.v == other.v

        key = MyKey(42)

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses the captured key as a dictionary key and reads it back.
            d = {key: 99}
            return d[key]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 99)

    def test_hash_user_class_unhashable(self):
        # [FUNCTION DOC] A class whose hash attribute is set to None raises TypeError saying
        # Unhashable is unhashable.
        """Class with __hash__ = None raises TypeError."""

        class Unhashable:
            # [CLASS DOC] HELPER: Class that disables hashing by setting its hash attribute
            # to None.
            __hash__ = None

        obj = Unhashable()
        self._assert_hash_raises(
            lambda: obj, expect_msg="unhashable type: 'Unhashable'"
        )

    def test_hash_user_class_eq_without_hash(self):
        # [FUNCTION DOC] A class that defines equality but no hash is unhashable (CPython clears
        # the hash); raises TypeError naming EqOnly.
        """Class with __eq__ but no __hash__ is unhashable (CPython sets __hash__=None)."""

        class EqOnly:
            # [CLASS DOC] HELPER: Class defining only an equality method that always returns true.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality that always returns true.
                return True

        obj = EqOnly()
        self._assert_hash_raises(lambda: obj, expect_msg="unhashable type: 'EqOnly'")

    def test_hash_object_created_in_compile_region(self):
        # [FUNCTION DOC] An object created inside the compiled region can be a dictionary key;
        # hashing uses variable identity; lookup returns 42.
        """hash() on an object created inside torch.compile uses VT identity."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty class instantiated inside the compiled function.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an instance, uses it as a dictionary key and reads
            # it back.
            obj = MyObj()
            d = {obj: 42}
            return d[obj]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_custom_hash_returns_fake_id(self):
        # [FUNCTION DOC] A custom hash that returns the id of a sourceless inner list does not
        # break the graph; two distinct objects as keys yield values summing to 3.
        """Custom __hash__ returning id() of sourceless object: no graph break."""

        class HashByInnerID:
            # [CLASS DOC] HELPER: Class holding an empty list whose hash is the id of that list.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates an empty list in the data attribute.
                self.data = []

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the identity number of the data list.
                return id(self.data)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates two instances, uses both as dictionary keys and
            # returns the sum of their values.
            a = HashByInnerID()
            b = HashByInnerID()
            d = {a: 1, b: 2}
            return d[a] + d[b]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 3)

    @torch._dynamo.config.patch(assume_static_by_default=False)
    def test_hash_custom_hash_returns_symnode(self):
        # [FUNCTION DOC] With dynamic shapes by default, a custom hash that returns a symbolic
        # first dimension specializes with no graph break; dictionary lookup returns 42.
        """Custom __hash__ returning a dynamic shape: specializes, no graph break."""

        class ShapeHash:
            def __init__(self, t):
                # [CLASS DOC] HELPER: Class wrapping a tensor whose hash is the tensor's
                # first dimension.
                self.t = t

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Stores the given tensor.
                return self.t.shape[0]

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the first dimension of the stored tensor.
            obj = ShapeHash(x)
            d = {obj: 42}
            return d[obj]
        # [FUNCTION DOC] HELPER: Wraps the input in an instance, uses it as a dictionary key and
        # reads it back.

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(5))
        self.assertEqual(result, 42)

    def test_hash_custom_hash_non_int_raises(self):
        # [FUNCTION DOC] A custom hash returning a non-integer raises TypeError saying the hash
        # method should return an integer.
        """Custom __hash__ returning a non-int raises TypeError."""

        class BadHash:
            # [CLASS DOC] HELPER: Class whose hash method returns a string.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns a string instead of an integer; lint rule about
                # the wrong return type is suppressed.
                return "not an int"  # noqa: PLE0309

        obj = BadHash()
        self._assert_hash_raises(
            lambda: obj,
            expect_msg="__hash__ method should return an integer",
        )

    def test_hash_custom_hash_c_tp_hash(self):
        # [FUNCTION DOC] A user subclass of datetime that inherits the C-level hash is traceable
        # without a graph break; hash equals eager.
        """User class inheriting C tp_hash (datetime_hash): no graph break."""
        import datetime

        class DateHash(datetime.datetime):
            # [CLASS DOC] HELPER: Empty subclass of the datetime class.
            pass

        dt = DateHash(2024, 1, 1)
        self._assert_hash_equals(dt)

    def test_hash_allow_c_slot_api(self):
        # [FUNCTION DOC] Using an sqlite Row object: hashing first raises Unsupported mentioning an
        # untraceable C hash slot; after resetting Dynamo and registering the Row type via the
        # allow_c_slot API, hash equals eager. Side effect: leaves the Row type registered as
        # allowed for the rest of the process. Creates an in-memory database for the row.
        """allow_c_slot registers a C extension type's hash as safe."""
        import sqlite3

        from torch._dynamo.exc import Unsupported

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (a, b)")
        conn.execute("INSERT INTO t VALUES (1, 2)")
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM t").fetchone()

        # Before registration: graph-breaks
        with self.assertRaisesRegex(Unsupported, "Untraceable C tp_hash"):
            torch.compile(lambda _: hash(row), backend="eager", fullgraph=True)(
                torch.tensor(0)
            )

        torch._dynamo.reset()

        # After registration: no graph break
        torch._dynamo.allow_c_slot(sqlite3.Row)
        self._assert_hash_equals(row)

    def test_hash_non_constant_getattr_graph_breaks(self):
        # [FUNCTION DOC] Regression test: hashing a non-constant attribute-lookup variable
        # must graph break rather than crash. Compiled without full graph; result is the
        # unchanged input.
        """Hashing a non-constant GetAttrVariable graph-breaks instead of crashing.

        Regression test for test_public_api_surface: v.__name__ in seen where
        the GetAttrVariable wraps a dict with non-constant values (ModuleSpec).
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds the fx module name to a set and then checks the names of
            # attributes in the fx module dictionary against that set.
            seen = set()
            seen.add(torch.fx.__name__)
            for v in torch.fx.__dict__.values():
                if hasattr(v, "__name__") and v.__name__ in seen:
                    pass
            return x

        result = torch.compile(fn, backend="eager")(torch.tensor(0))
        self.assertEqual(result, torch.tensor(0))

    # --- functools.partial ---

    def test_hash_partial_as_dict_key(self):
        # [FUNCTION DOC] A partial created inside the compiled function can be a dictionary key;
        # lookup returns 42.
        """Same partial used as dict key works."""
        from functools import partial

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Adds two values.
            return x + y

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a partial of the adder, uses it as a dictionary key
            # and reads it back.
            p = partial(gn, x=1)
            d = {p: 42}
            return d[p]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_partial_preexisting_as_dict_key(self):
        # [FUNCTION DOC] A partial created before compilation can be a dictionary key; lookup
        # returns 42.
        """Pre-existing partial works as dict key."""
        from functools import partial

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Adds two values.
            return x + y

        p = partial(gn, x=1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses the captured partial as a dictionary key and reads
            # it back.
            d = {p: 42}
            return d[p]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_partial_identity_based(self):
        # [FUNCTION DOC] Two equal-looking partials are different keys (identity-based hash, as in
        # CPython): lookup with the second one raises KeyError inside, and the function returns
        # minus one.
        """Two equivalent partials are different keys, matching CPython."""
        from functools import partial

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Adds two values.
            return x + y

        def fn(x):
            # [FUNCTION DOC] HELPER: Stores one partial as a key and looks up another equivalent
            # partial, returning minus one on KeyError.
            p1 = partial(gn, x=1)
            p2 = partial(gn, x=1)
            d = {p1: 42}
            try:
                return d[p2]
            except KeyError:
                return -1

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, -1)

    # --- Frozen dataclass ---

    def test_hash_frozen_dataclass(self):
        # [FUNCTION DOC] Hash of a frozen dataclass instance equals the eager hash (field-based).
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            # [CLASS DOC] HELPER: Frozen dataclass with two integer fields.
            x: int
            y: int

        p = Point(1, 2)
        self._assert_hash_equals(p)

    def test_hash_frozen_dataclass_custom_hash(self):
        # [FUNCTION DOC] A frozen dataclass with its own hash method (using only the key field) is
        # traced, rather than hashing all fields, so an unhashable payload field does not cause
        # failure; hash equals eager.
        """A frozen dataclass with a custom __hash__ is traced, not replaced by
        the generated field-tuple hash. The unhashable ``payload`` field would
        make the field hash raise, so this also covers specs (e.g. the cxx
        pytree TreeSpec) that hold unhashable metadata."""
        from dataclasses import dataclass

        @dataclass(frozen=True, eq=False)
        class Custom:
            # [CLASS DOC] HELPER: Frozen dataclass with equality generation off, an integer key and
            # an unhashable list payload.
            key: int
            payload: list  # unhashable; excluded by __hash__

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the hash of the key field only.
                return hash(self.key)

        self._assert_hash_equals(Custom(7, [1, 2, 3]))

    # --- Weakref ---

    def test_hash_weakref(self):
        # [FUNCTION DOC] Hash of a weak reference to an object equals the eager hash.
        import weakref

        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instance is weakly referenced.
            pass

        obj = MyObj()
        self._assert_hash_equals(weakref.ref(obj))

    # --- Bound method / user method ---

    def test_hash_bound_method(self):
        # [FUNCTION DOC] Hash of a bound builtin method (upper of a string) under Dynamo equals the
        # eager hash.
        """GetAttrVariable wrapping a bound method — has wrapper_hash/meth_hash."""
        s = "hello"
        expected = hash(s.upper)

        def fn(_):
            # [FUNCTION DOC] HELPER: Returns the hash of the bound upper method of the
            # captured string.
            return hash(s.upper)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_user_method(self):
        # [FUNCTION DOC] Hash of a bound user method equals the eager hash.
        """UserMethodVariable: CPython method_hash."""

        class MyClass:
            # [CLASS DOC] HELPER: Class with one empty method.
            def my_method(self):
                # [FUNCTION DOC] HELPER: Empty method whose bound version is hashed.
                pass

        obj = MyClass()
        expected = hash(obj.my_method)

        def fn(_):
            # [FUNCTION DOC] HELPER: Returns the hash of the bound method of the captured object.
            return hash(obj.my_method)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    # --- Builtin types ---

    def test_hash_type(self):
        # [FUNCTION DOC] Hash of the int type equals the eager hash.
        self._assert_hash_equals(int)

    def test_hash_builtin(self):
        # [FUNCTION DOC] Hash of the len builtin function equals the eager hash.
        self._assert_hash_equals(len)

    def test_hash_list_type(self):
        # [FUNCTION DOC] Hash of the list type equals the eager hash, exercising the fallback
        # through the Python constant.
        """hash(list) exercises the as_python_constant fallback in base hash_impl."""
        self._assert_hash_equals(list)

    # --- C tp_hash types ---

    def test_hash_datetime(self):
        # [FUNCTION DOC] Hash of a datetime instance equals the eager hash.
        import datetime

        self._assert_hash_equals(datetime.datetime(2024, 1, 1))

    def test_hash_date(self):
        # [FUNCTION DOC] Hash of a date instance equals the eager hash.
        import datetime

        self._assert_hash_equals(datetime.date(2024, 1, 1))

    def test_hash_timedelta(self):
        # [FUNCTION DOC] Hash of a timedelta of seven days equals the eager hash.
        import datetime

        self._assert_hash_equals(datetime.timedelta(days=7))

    def test_hash_timezone(self):
        # [FUNCTION DOC] Hash of the UTC timezone object equals the eager hash.
        import datetime

        self._assert_hash_equals(datetime.timezone.utc)

    def test_hash_decimal(self):
        # [FUNCTION DOC] Hash of a Decimal value equals the eager hash.
        import decimal

        self._assert_hash_equals(decimal.Decimal("3.14"))

    def test_hash_re_pattern(self):
        # [FUNCTION DOC] Hash of a compiled regular expression pattern equals the eager hash.
        import re

        self._assert_hash_equals(re.compile(r"\d+"))

    def test_hash_method_wrapper(self):
        # [FUNCTION DOC] Hash of a method-wrapper (the hash method of a string) equals the
        # eager hash.
        """method-wrapper has wrapper_hash (C tp_hash), handled via as_python_constant."""
        mw = "hello".__hash__
        self._assert_hash_equals(mw)

    # --- TypingVariable ---

    def test_hash_typing_union(self):
        # [FUNCTION DOC] Hash of a typing union of int and str equals the eager hash.
        """TypingVariable: custom hash on typing special forms."""
        import typing

        u = typing.Union[int, str]  # noqa: UP007
        self._assert_hash_equals(u)

    # --- BaseTorchVariable ---

    def test_hash_torch_function(self):
        # [FUNCTION DOC] Hash of the torch sine function equals the eager hash.
        """BaseTorchVariable: hash of torch ops."""
        self._assert_hash_equals(torch.sin)

    # --- DistributedVariable ---

    def test_hash_distributed_group_member(self):
        # [FUNCTION DOC] Skipped if torch.distributed is unavailable; otherwise hash of the group
        # member class object equals the eager hash.
        """DistributedVariable (WorldMetaClassVariable): hash of GroupMember."""
        if not torch.distributed.is_available():
            self.skipTest("torch.distributed not available")
        import torch.distributed as dist

        self._assert_hash_equals(dist.GroupMember)

    # --- CustomClassVariable ---

    def test_hash_opaque_value_type(self):
        # [FUNCTION DOC] Registers a small class as a constant-type opaque custom class and checks
        # the hash of an instance equals the eager hash. Side effect: registers the class with the
        # opaque object library.
        """CustomClassVariable: hash of registered opaque value type."""
        from torch._library.opaque_object import register_custom_class

        class _HashTestOpaque:
            # [CLASS DOC] HELPER: Opaque test class that stores a value, compares and hashes by it
            # and provides the representation hook needed for registration.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the given value.
                self.x = x

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Returns true if the other object is the same class with an
                # equal value.
                return isinstance(other, _HashTestOpaque) and self.x == other.x

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Returns the hash of the stored value.
                return hash(self.x)

            def __fx_repr__(self):
                # [FUNCTION DOC] HELPER: Returns a pair of a text form with the value and an empty
                # mapping, used as the graph representation.
                return (f"_HashTestOpaque({self.x})", {})

        register_custom_class(_HashTestOpaque, typ="constant")
        self._assert_hash_equals(_HashTestOpaque(42))

    # --- Dunder __hash__ and consistency ---

    def test_dunder_hash_int(self):
        # [FUNCTION DOC] Calling the hash dunder method of an int constant inside compiled code
        # gives the same value as eager.
        expected = (42).__hash__()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash method result of the integer 42.
            return (42).__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_dunder_hash_str(self):
        # [FUNCTION DOC] Calling the hash dunder method of a string constant gives the same value
        # as eager.
        expected = "hello".__hash__()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash method result of the string hello.
            return "hello".__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_consistency(self):
        # [FUNCTION DOC] The dunder hash call on an int constant agrees with the eager
        # hash builtin.
        expected = hash(42)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash method result of the integer 42.
            return (42).__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    def test_hash_consistency_tuple(self):
        # [FUNCTION DOC] The dunder hash call on a captured tuple agrees with the eager
        # hash builtin.
        t = (1, 2, 3)
        expected = hash(t)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the hash method result of the captured tuple.
            return t.__hash__()

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, expected)

    # --- FakeIdVariable propagation through containers ---

    def test_hash_tuple_with_sourceless_object_graph_breaks_on_return(self):
        # [FUNCTION DOC] Hashing a tuple that includes an object created in the compiled function
        # cannot return a real number: the frame is not compiled (counter frame count zero) but the
        # call still returns an int.
        """hash(tuple) containing a sourceless object can't be returned."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty class instantiated inside the function.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an instance and returns the hash of a pair holding one
            # and the instance.
            obj = MyObj()
            return hash((1, obj))

        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnt)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertEqual(cnt.frame_count, 0)

    def test_hash_tuple_with_sourceless_object_works_as_dict_key(self):
        # [FUNCTION DOC] A tuple containing a sourceless object works as a dictionary key under
        # full graph; lookup returns 42.
        """hash(tuple) containing a sourceless object works for dict keys."""

        class MyObj:
            # [CLASS DOC] HELPER: Empty class instantiated inside the function.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates an instance, builds a tuple with it, uses it as a
            # dictionary key and reads it back.
            obj = MyObj()
            tup = (1, obj)
            d = {tup: 42}
            return d[tup]

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, 42)

    def test_hash_tuple_with_tensor_graph_breaks_on_return(self):
        # [FUNCTION DOC] Hashing a tuple that includes a tensor created in the compiled function
        # cannot be returned from a compiled graph: frame count stays zero but the call returns
        # an int.
        """hash(tuple) containing a sourceless tensor can't be returned."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a scalar tensor and returns the hash of a tuple
            # holding an int, a string and that tensor.
            t = torch.tensor(1.0)
            return hash((1, "hello", t))

        cnt = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnt)(torch.tensor(0))
        self.assertIsInstance(result, int)
        self.assertEqual(cnt.frame_count, 0)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
