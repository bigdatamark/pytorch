# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for dictionary support: dict, OrderedDict, defaultdict, UserDict and
# dict subclasses, mappingproxy, dict views, and __dict__ access under torch.compile. Most tests
# compile a function with the eager backend and compare against eager results, and check
# guard/recompilation behavior using CompileCounter or the error_on_recompile config. Also defines
# small helper classes used by the tests and the DictTests case plus other test classes below. No
# import-time side effects beyond imports.


import enum
import itertools
import operator
import types
import unittest
import weakref
from collections import defaultdict, namedtuple, OrderedDict, UserDict
from collections.abc import Callable
from functools import partial
from typing import Any, NamedTuple

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._functorch.config
import torch.fx as fx
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import same
from torch._dynamo.utils import dict_items
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    munge_exc,
    parametrize,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


class _BadCmpExc(Exception):
    # [CLASS DOC] HELPER: Exception raised by _BadCmp when it is compared for equality.
    pass


class _BadCmp:
    # [CLASS DOC] HELPER: Hashable value whose equality comparison always raises _BadCmpExc, used
    # to check that comparison errors propagate out of dict operations.
    # Hashable (so it can be a dict value compared via PyObject_RichCompareBool)
    # but its __eq__ raises.
    def __eq__(self, other: object) -> bool:
        # [FUNCTION DOC] HELPER: Always raises _BadCmpExc regardless of the other operand.
        raise _BadCmpExc

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Returns the constant hash 1 so instances can be dict members.
        return 1


class SimpleDict(dict):
    # [CLASS DOC] HELPER: Plain dict subclass with no overrides; used to test dict
    # subclass handling.
    pass


class DummyUserDict(UserDict):
    # [CLASS DOC] HELPER: Plain UserDict subclass with no overrides.
    pass


class FakeMapping:
    # [CLASS DOC] HELPER: Mapping-like object (not a dict) that has a keys callable returning a,
    # b, c and a getitem that returns the stored value for any key.
    def __init__(self, value: Any) -> None:
        # [FUNCTION DOC] HELPER: Stores value and sets an instance attribute keys to a lambda
        # returning the list a, b, c (an attribute, not a method).
        self._value = value
        self.keys = lambda: ["a", "b", "c"]  # not required to be a method

    def __getitem__(self, key: str) -> Any:
        # [FUNCTION DOC] HELPER: Returns the stored value regardless of the key.
        return self._value


class DictTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case covering dict behavior under torch.compile: subclass handling,
    # guards on keys and key order, side-effect replay of mutations, mapping proxies, dict views
    # and set operations, supported key types, __dict__ access, and canonical graph node ordering.
    def test_dict_subclass_instantiation(self):
        # [FUNCTION DOC] Compiles a function that builds a SimpleDict via keyword args and indexes
        # it; fullgraph compile must match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a SimpleDict and returns its entry times the input.
            sd = SimpleDict(x=5)
            return sd["x"] * x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_local_mutation(self):
        # [FUNCTION DOC] Checks a locally created dict subclass instance can be mutated during
        # tracing with fullgraph compile and still match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a SimpleDict, reads, overwrites an entry, and
            # combines both.
            sd = SimpleDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return z * sd["x"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_contains_enum(self):
        # [FUNCTION DOC] Checks that comparing a str-based Enum member with plain strings in both
        # operand orders inside a compiled module traces in fullgraph mode.
        class TensorDim(str, enum.Enum):
            # [CLASS DOC] HELPER: str-mixin Enum with members ddp, fsdp, cp, tp.
            DDP = "ddp"
            FSDP = "fsdp"
            CP = "cp"
            TP = "tp"

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward compares an enum member to a string
            # both ways.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent Module initializer only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns sin of the input plus cos of it for each
                # true comparison.
                val = x.sin()
                if TensorDim.DDP == "ddp":
                    val += x.cos()
                if "ddp" == TensorDim.DDP:
                    val += x.cos()
                return val

        inp = torch.randn(4, 4)
        mod = Foo()
        opt_f = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(inp), opt_f(inp))

    def test_dict_subclass_local_with_non_dict_method(self):
        # [FUNCTION DOC] Checks a user-defined (non-dict) method on a dict subclass is inlined and
        # works on a locally created instance.
        # Checks that add_1 method is inlined
        class MethodDict(dict):
            # [CLASS DOC] HELPER: dict subclass adding a method that adds one to its argument.
            def add_1(self, x):
                # [FUNCTION DOC] HELPER: Returns the argument plus one.
                return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a MethodDict, mutates it, and calls the extra method.
            sd = MethodDict(x=5)
            z = sd["x"] * x
            sd["x"] = 10
            return sd.add_1(z * sd["x"])

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_contains(self):
        # [FUNCTION DOC] Checks guards for the contains operator on a global dict: adding a
        # queried key recompiles, while changing a value of an unrelated key does not (verified
        # with error_on_recompile patched to True).
        sd = dict()
        sd[2] = 5
        sd[4] = 10

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies by 2 if key 1 is in the global dict, else by 3.
            if 1 in sd:
                x = x * 2
            else:
                x = x * 3
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[1] = 15
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure not recompilation because the traced program remains same here.
        sd[2] = 10
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_torch_size_dynamic_key(self):
        # [FUNCTION DOC] Checks dicts keyed by torch.Size, looked up with a size built from tensor
        # values, compile under the eager backend for several feature widths; each case resets
        # dynamo and compares outputs with eager using the same helper.
        class DynamicShapeModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding dicts of layers and activations keyed
            # by torch.Size.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the two size-keyed dicts of Linear layers
                # and activations.
                super().__init__()
                self.layer_configs = {
                    torch.Size([32, 64]): torch.nn.Linear(32, 64),
                    torch.Size([64, 128]): torch.nn.Linear(64, 128),
                    torch.Size([128, 256]): torch.nn.Linear(128, 256),
                }
                self.activation_functions = {
                    torch.Size([32]): torch.nn.ReLU(),
                    torch.Size([64]): torch.nn.Tanh(),
                    torch.Size([128]): torch.nn.Sigmoid(),
                }

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Looks up a layer by a size derived from the input tensor
                # and an activation by the output feature size, applying each if present.
                current_shape = torch.tensor(x.shape[1:])
                shape_key = torch.Size([current_shape[0], 64])
                if shape_key in self.layer_configs:
                    x = self.layer_configs[shape_key](x)

                activation_shape = torch.Size([x.shape[1]])
                if activation_shape in self.activation_functions:
                    x = self.activation_functions[activation_shape](x)
                return x

        model = DynamicShapeModel().eval()

        for features in (32, 64, 128, 16):
            x = torch.randn(4, features)
            with torch.no_grad():
                eager_out = model(x)

            torch._dynamo.reset()
            compiled_model = torch.compile(model, backend="eager")
            with torch.no_grad():
                compiled_out = compiled_model(x)

            self.assertTrue(same(eager_out, compiled_out))

    def test_dict_subclass_methods_fallback_readonly(self):
        # [FUNCTION DOC] Checks read-only dict methods (values, iteration, items, contains, get,
        # len, plus a custom attribute) on a global SimpleDict compile fullgraph; adding a key
        # afterwards must still give matching results.
        sd = SimpleDict()
        sd[2] = 5
        sd[4] = 10
        # check that regular attr accesses work well
        sd.attr = 4

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by values, keys, items, and lookups of
            # the global SimpleDict.
            for value in sd.values():
                x = x * value
            for key in sd:
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            x = x * sd.attr
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

        # Ensure a recompilation
        sd[6] = 15
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_subclass_instantiation_return(self):
        # [FUNCTION DOC] Checks a SimpleDict created and mutated inside a compiled function can be
        # returned with preserved type and contents.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a SimpleDict with a tensor value, adds a key,
            # returns it.
            sd = SimpleDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_dict_subclass_methods_fallback_mutation(self):
        # [FUNCTION DOC] Checks that a function mutating a passed-in dict subclass produces the
        # same result and same final dict content as eager when run on separate equal copies.
        def fn(sd, x):
            # [FUNCTION DOC] HELPER: Reads values, inserts key 6, then iterates, queries and
            # measures the dict.
            for value in sd.values():
                x = x * value
            sd[6] = 14
            for key in sd:
                x = x * key
            for k, v in sd.items():
                x = x * k
                x = x * v
            # for k in sd:
            #     x = x * k

            if 1 in sd:
                x = x * 2
            else:
                x = x * 3

            x = x * sd.get(2, 0)
            x = x * sd.get(3, 4)
            x = len(sd) * x
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        sd1 = SimpleDict()
        sd1[2] = 5
        sd1[4] = 10

        sd2 = SimpleDict()
        sd2[2] = 5
        sd2[4] = 10
        self.assertTrue(sd1 == sd2)

        self.assertEqual(fn(sd1, x), opt_fn(sd2, x))
        self.assertTrue(sd1 == sd2)

    def test_dict_subclass_setitem(self):
        # [FUNCTION DOC] Checks that a dict subclass overriding item assignment has its override
        # honored (values stored incremented) when used inside a compiled function.
        class SetItemDict(dict):
            # [CLASS DOC] HELPER: dict subclass whose item assignment stores the value plus one.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Stores the value incremented by one via the
                # parent implementation.
                super().__setitem__(key, value + 1)

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates and mutates a SetItemDict and returns it.
            sd = SetItemDict(x=5 * x)
            sd["y"] = 10
            return sd

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(type(ref), type(res))
        self.assertEqual(ref["x"], res["x"])
        self.assertEqual(ref["y"], res["y"])

    def test_custom_iter_dict(self):
        # [FUNCTION DOC] Checks a dict subclass with a custom iteration order plus attribute and
        # item side effects: second call recompiles because the dict changed, third does not.
        class ReversedDict(dict):
            # [CLASS DOC] HELPER: dict subclass whose iterator yields keys in reversed order.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns an iterator over the keys in reverse order.
                return reversed(list(self.keys()))

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Sets an attribute and a new key on the dict, then multiplies.
            # Forces side effects attribute reapplication logic
            d.sample = 1
            d["baz"] = 4
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        # This is intentional because the dict is mutated, so we will have a recompilation.
        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_custom_keys_iter_dict(self):
        # [FUNCTION DOC] Checks a dict subclass with an overridden keys method compiles and does
        # not recompile on a second call.
        class ReversedDict(dict):
            # [CLASS DOC] HELPER: dict subclass whose keys method returns a fixed list bar, foo.
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the list bar, foo.
                return ["bar", "foo"]

        d = {
            "foo": 1,
            "bar": 2,
        }

        d = ReversedDict(d)

        @torch.compile(backend="eager")
        def fn(x, d):
            # [FUNCTION DOC] HELPER: Multiplies the input by two dict entries.
            return x * d["foo"] * d["bar"]

        fn(torch.randn(4), d)
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(4), d)

    def test_dict_guard_on_keys_order(self):
        # [FUNCTION DOC] Checks that moving a key to the end of a dict (iterated through items)
        # triggers a recompilation, via CompileCounter frame counts, and results match eager.
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Accumulates key times input plus value over the dict items.
            for key, value in d.items():
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_dict_guard_on_keys_order2(self):
        # [FUNCTION DOC] Same as the key-order guard test but iterates the dict directly and looks
        # values up by key; a reordered dict must recompile once.
        d = {
            2: 4,
            3: 5,
        }

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Iterates keys, looks up each value, accumulates.
            for key in d:
                value = d[key]
                x = x * key + value
            return x

        opt_fn = torch.compile(fn, backend=cnts)
        opt_fn(torch.randn(4), d)
        opt_fn(torch.randn(4), d)
        # No recompilation
        self.assertEqual(cnts.frame_count, 1)

        # move 2 to the end
        d[2] = d.pop(2)

        x = torch.randn(4)
        res = opt_fn(x, d)
        # Check recompilation
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(res, fn(x, d))

    def test_ordered_dict_reordered_keys(self):
        # [FUNCTION DOC] Checks an OrderedDict that had a key moved to the end is traversed in its
        # new order by a compiled function and matches eager.
        d = OrderedDict()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Sums sin of the first value product and cos of later ones.
            y = 0
            for idx, value in enumerate(d.values()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_ordered_dict_subclass_reordered_keys(self):
        # [FUNCTION DOC] Same as the OrderedDict reorder test, but with an OrderedDict subclass
        # overriding keys.
        class ODSubclass(OrderedDict):
            # [CLASS DOC] HELPER: OrderedDict subclass whose keys method delegates to the parent.
            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the parent keys view.
                return super().keys()

        d = ODSubclass()
        d[2] = 4
        d[3] = 5
        d.move_to_end(2)

        cnts = torch._dynamo.testing.CompileCounter()

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Sums sin of the first value product and cos of later ones.
            y = 0
            for idx, value in enumerate(d.values()):
                if idx == 0:
                    y += torch.sin(x * value)
                else:
                    y += torch.cos(x * value)
            return y

        opt_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, d), fn(x, d))

    def test_lazy_key_guarding(self):
        # [FUNCTION DOC] Checks guards on a dict are installed lazily per accessed key: removing
        # and adding keys that were never read must not recompile (error_on_recompile enabled).
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads one key of the global dict.
            return x * d["a"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop("c")
        d["d"] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_lazy_key_non_const_guarding(self):
        # [FUNCTION DOC] Same lazy-guard test, but with non-constant keys (type objects); changing
        # unrelated keys must not recompile.
        d = {
            list: 2,
            dict: 3,
            OrderedDict: 5,
            namedtuple: 7,
        }

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads the entry keyed by the list type from the global dict.
            return x * d[list]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # Since key c was not used, it should not lead to a recompilation
        d.pop(dict)
        d[defaultdict] = 10

        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

    def test_dict_mutation_side_effect(self):
        # [FUNCTION DOC] Checks that mutations of an input dict inside a compiled function are
        # applied to the caller's dict (same identity), results match eager, and exactly one frame
        # and one op are produced.
        def fn(d):
            # [FUNCTION DOC] HELPER: Stores the sum of one entry and a popped entry under a
            # new key.
            d["c"] = d["a"] + d.pop("b")
            return d

        args1 = {"a": torch.randn(10), "b": torch.randn(10)}
        args2 = dict(args1)
        if fn(args1) is not args1:
            raise AssertionError("Expected fn(args1) to be args1")
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertIs(opt_fn(args2), args2)
        self.assertTrue(same(args1, args2))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_dict_copy_alias(self):
        # [FUNCTION DOC] Checks that copying an input dict then mutating the copy leaves the
        # original untouched and returns the copy.
        @torch.compile(backend="eager", fullgraph=True)
        def run(x, d0):
            # [FUNCTION DOC] HELPER: Copies the dict, adds an entry to the copy, returns tensor
            # and copy.
            d1 = d0.copy()
            d1[0] = 1
            return x + 1, d1

        d0 = {}
        res, d1 = run(torch.zeros(1), d0)
        self.assertTrue(same(res, torch.ones(1)))
        self.assertEqual(d0, {})
        self.assertEqual(d1, {0: 1})

    def test_dict_subclass_get_method(self):
        # [FUNCTION DOC] Checks that calling get on a dict subclass that aliases attribute access
        # to dict operations works under fullgraph compile.
        class dotdict(dict):
            # [CLASS DOC] HELPER: dict subclass mapping attribute get, set and delete onto
            # dict ops.
            """dot.notation access to dictionary attributes"""

            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        config = dotdict({"a": 1, "b": 2})

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses get with a default on the global dotdict.
            x2 = x * 2  # noqa: F841
            x3 = x * config.get("a", 3)
            return x3

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_order_keys(self):
        # [FUNCTION DOC] Checks key order matters: sum over values compiles once, a dict with the
        # same entries in different insertion order triggers one recompilation.
        def fn(d):
            # [FUNCTION DOC] HELPER: Sums the values of the dict.
            c = 0
            for v in d.values():
                c += v
            return c

        args1 = {}
        args1["a"] = torch.rand(10)
        args1["b"] = torch.rand(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # A different order of keys recompiles
        args2 = {}
        args2["b"] = args1["b"]
        args2["a"] = args1["a"]
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)
        # Extra calls don't recompile
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_namedtuple(self):
        # [FUNCTION DOC] Checks a dict containing the namedtuple factory as a key: guard fails and
        # recompiles when that key is absent.
        def fn(d):
            # [FUNCTION DOC] HELPER: Branches on whether the namedtuple factory is a key.
            if namedtuple in d:
                return d[3] * 2
            else:
                return d[3] * 3

        args1 = {namedtuple: None, 3: torch.randn(3)}
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(fn(args1), opt_fn(args1))
        self.assertEqual(cnts.frame_count, 1)
        # Test a failing namedtuple guard
        args2 = {2: None, 3: torch.randn(3)}
        self.assertEqual(fn(args2), opt_fn(args2))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_tensors(self):
        # [FUNCTION DOC] Checks a dict keyed by tensors: same order no recompile, different
        # insertion order recompiles (frame counts verified).
        def fn(d, x):
            # [FUNCTION DOC] HELPER: Looks up the tensor key and adds three.
            return d[x] + 3

        args1 = {}
        x = torch.randn(10)
        y = torch.randn(10)
        z = torch.randn(10)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_dict_order_keys_modules(self):
        # [FUNCTION DOC] Same as the tensor-key order test but keys are nn.Module instances that
        # are then called.
        def fn(d, x):
            # [FUNCTION DOC] HELPER: Calls the module found under the given key.
            return d[x](torch.ones(2, 2))

        args1 = {}
        x = torch.nn.Linear(2, 2)
        y = torch.nn.Linear(2, 2)
        z = torch.nn.Linear(2, 2)
        args1[x] = y
        args1[3] = z

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        self.assertEqual(fn(args1, x), opt_fn(args1, x))
        self.assertEqual(cnts.frame_count, 1)

        # Calling again doesn't recompile (same id and key order)
        opt_fn(args1, x)
        self.assertEqual(cnts.frame_count, 1)
        args2 = {}
        args2[3] = z
        args2[x] = y

        # Different order recompiles
        self.assertEqual(fn(args2, x), opt_fn(args2, x))
        self.assertEqual(cnts.frame_count, 2)

    def test_contains_dunder_dict(self):
        # [FUNCTION DOC] Checks the contains operator and get on an object's __dict__ (including
        # an attribute added during tracing) in fullgraph mode.
        class UserDefined:
            # [CLASS DOC] HELPER: Plain object with attributes a and b whose run method queries
            # its own __dict__.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attributes a and b.
                self.a = 3
                self.b = 5

            def run(self, x):
                # [FUNCTION DOC] HELPER: Multiplies by attributes present in __dict__, adds
                # attribute c, then uses get with and without defaults.
                if "a" in self.__dict__:
                    x = x * self.a
                if "b" in self.__dict__:
                    x = x * self.b
                self.c = 7
                if "c" in self.__dict__:
                    x = x * self.c
                return x * self.__dict__.get("a") * self.__dict__.get("z", 2)

        obj = UserDefined()

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls run on the global object.
            return obj.run(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_len_dunder_dict_after_delete(self):
        # [FUNCTION DOC] Regression test: after deleting an attribute during tracing, the length
        # of the instance __dict__ must reflect the deletion.
        # After deleting an attribute during tracing, len(obj.__dict__) must
        # reflect the deletion. The bug: SideEffectsProxyDict.__len__ returned
        # len(item_dict) + len(side_effects_table), double-counting deleted
        # entries and keys mutated from item_dict.
        class UserDefined:
            # [CLASS DOC] HELPER: Object with attributes a and b whose run deletes a and uses the
            # remaining __dict__ length.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attributes a and b.
                self.a = 3
                self.b = 5

            def run(self, x):
                # [FUNCTION DOC] HELPER: Deletes a, then scales the input by the __dict__ length.
                del self.a
                # len should be 1 (only b remains), not 3 (item_dict=2 + mutations=1)
                return x * len(self.__dict__)

        obj1 = UserDefined()
        obj2 = UserDefined()

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Calls run on the given object.
            return obj.run(x)

        x = torch.ones(2)
        ref = fn(x, obj1)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x, obj2)
        self.assertEqual(ref, res)

    def test_contains_module_dunder_dict(self):
        # [FUNCTION DOC] Checks contains on a module's __dict__ in a compiled forward
        # matches eager.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with plain attributes foo, bar, baz.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent initializer and sets three
                # int attributes.
                super().__init__()
                self.foo = 1
                self.bar = 2
                self.baz = 3

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Branches on whether foo is in __dict__.
                if "foo" in self.__dict__:
                    return x * self.bar
                return x * self.baz

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_update_dunder_dict(self):
        # [FUNCTION DOC] Checks item assignment on an object's __dict__ is visible as an attribute
        # and persisted on the object exactly like eager.
        class UserDefined:
            # [CLASS DOC] HELPER: Object whose run writes attribute a through its __dict__.
            def run(self, x):
                # [FUNCTION DOC] HELPER: Sets a via __dict__ then reads it as attribute and
                # by key.
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        obj1 = UserDefined()
        obj2 = UserDefined()

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Calls run on the given object.
            return obj.run(x)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, obj1)
        res = opt_fn(x, obj2)
        self.assertEqual(ref, res)
        # Make sure only `a` is updated.
        self.assertEqual(obj1.__dict__, obj2.__dict__)

    def test_update_module_dunder_dict(self):
        # [FUNCTION DOC] Same as the __dict__ update test but for an nn.Module inside forward.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward writes attribute a through __dict__.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent initializer only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sets a via __dict__ then reads it as attribute and
                # by key.
                self.__dict__["a"] = 10
                return x * self.a + self.__dict__["a"]

        mod = MyModule()
        x = torch.randn(10)
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        self.assertEqual(mod(x), opt_mod(x))

    def test_dict_reconstruct_keeps_original_order(self):
        # [FUNCTION DOC] Checks that when an OrderedDict and a ModuleDict are updated inside the
        # compiled function and returned, the reconstructed objects preserve insertion order and
        # the same module objects.
        def fn():
            # [FUNCTION DOC] HELPER: Builds an OrderedDict and ModuleDict, updates both with
            # extra modules.
            modules = OrderedDict([("act", torch.nn.ReLU())])
            module_dict = torch.nn.ModuleDict(modules)

            next_modules = {"fc4": torch.nn.Linear(5, 6), "act3": torch.nn.Sigmoid()}
            modules.update(next_modules.items())
            module_dict.update(next_modules)
            return modules, module_dict

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        modules, module_dict = opt_fn()

        self.assertEqual(len(module_dict), len(modules))
        for k1, m2 in zip(modules, module_dict.children()):
            self.assertTrue(modules[k1] is m2)

    # FIXME: see comment in torch/_dynamo/polyfills/__init__.py:mutable_mapping_update
    @unittest.expectedFailure
    def test_dict_construct_from_mapping_like(self):
        # [FUNCTION DOC] Expected-failure test: constructing a dict from a non-dict mapping-like
        # object plus keyword args; see the polyfill FIXME above.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict from a FakeMapping and a keyword argument.
            fm = FakeMapping(x)
            d = dict(fm, x=x)
            return d

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_no_args(self):
        # [FUNCTION DOC] Checks calling update with no arguments leaves the dict unchanged and
        # returns None.
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls update with nothing and reports the result.
            d = {"a": x}
            result = d.update()
            return d["a"], result is None, len(d)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_from_mapping_like(self):
        # [FUNCTION DOC] Checks update from a user-defined mapping-like object with keys and
        # getitem merges its entries and returns None.
        class MappingLike:
            # [CLASS DOC] HELPER: Mapping-like object exposing keys and getitem over an
            # inner dict.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Builds an inner dict of the tensor and the tensor
                # plus one.
                self.d = {"a": x, "b": x + 1}

            def keys(self):
                # [FUNCTION DOC] HELPER: Returns the inner dict keys.
                return self.d.keys()

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns the inner dict value for the key.
                return self.d[key]

        def fn(x):
            # [FUNCTION DOC] HELPER: Updates a dict from a MappingLike and returns two entries and
            # whether update returned None.
            d = {"a": x - 1}
            result = d.update(MappingLike(x))
            return d["a"], d["b"], result is None

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_from_mapping_proxy(self):
        # [FUNCTION DOC] Checks update from a mappingproxy view of a dict.
        def fn(x):
            # [FUNCTION DOC] HELPER: Updates a dict from a mappingproxy and returns two entries.
            source = {"a": x, "b": x + 1}
            d = {"a": x - 1}
            d.update(types.MappingProxyType(source))
            return d["a"], d["b"]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_dict_update_rejects_bad_sequence_element_length(self):
        # [FUNCTION DOC] Checks update with a sequence element of the wrong length raises
        # ValueError with the message about the required length, same as eager.
        def fn():
            # [FUNCTION DOC] HELPER: Catches the error and reports whether the message
            # mentions length.
            try:
                {}.update([(1, 2, 3)])
            except ValueError as exc:
                return "length 3; 2 is required" in str(exc)
            return False

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), opt_fn())

    def test_dict_update_rejects_too_many_args(self):
        # [FUNCTION DOC] Checks update with two positional arguments raises TypeError both in
        # eager and when compiled.
        def fn():
            # [FUNCTION DOC] HELPER: Returns whether TypeError was raised.
            try:
                {}.update({}, {})
            except TypeError:
                return True
            return False

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(), opt_fn())

    def test_dict_subclass_initialization_in_graph(self):
        # [FUNCTION DOC] Checks, for OrderedDict and dict as base, that a subclass overriding new
        # and init can be instantiated and used inside a fullgraph compiled function.
        for super_class in (
            OrderedDict,
            dict,
        ):

            class CustomDict(super_class):
                # [CLASS DOC] HELPER: Subclass of the loop's base class that overrides new and
                # init by delegating to the parent.
                def __new__(cls, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Delegates to the parent new.
                    return super().__new__(cls, *args, **kwargs)

                def __init__(self, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Delegates to the parent init.
                    super().__init__(*args, **kwargs)

            def fn(x):
                # [FUNCTION DOC] HELPER: Creates CustomDict, sets and checks a key, returns value
                # plus one.
                c = CustomDict()
                c["key"] = x
                assert "key" in c  # noqa: S101
                return c["key"] + 1

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

            x = torch.rand(4)
            self.assertEqual(fn(x), opt_fn(x))

    def test_dict_list_values(self):
        # [FUNCTION DOC] Checks that a dict whose value is a list of tensors, passed to a helper
        # through zip with a counter, compiles with different list lengths.
        def inner_fn(args):
            # [FUNCTION DOC] HELPER: Returns the shapes of the second element of each pair.
            return [x[1].shape for x in args]

        @torch.compile(backend="eager")
        def fn(tensors):
            # [FUNCTION DOC] HELPER: Passes zipped counter and tensors to the inner helper.
            return inner_fn(zip(itertools.count(), tensors["args"]))

        fn({"args": [torch.ones(5, 5), torch.ones(5, 6), torch.ones(5, 7)]})
        fn({"args": [torch.ones(5, 5)]})

    def test_dict_iter(self):
        # [FUNCTION DOC] Checks iterating over a dict literal built inside a module forward yields
        # the sum of values (10) under fullgraph compile.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward sums values of a local dict by
            # iterating keys.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the values of a local dict keyed by words.
                z = {"my": 1, "const": 2, "dict": 3, "variable": 4}
                tot = 0
                for key in z:
                    tot += z[key]

                return tot

        x = torch.tensor([0])
        model = MyMod()
        opt_model = torch.compile(model, backend="eager", fullgraph=True)
        y = opt_model(x)

        self.assertEqual(y, 10)

    def test_dict_subclass_contains(self):
        # [FUNCTION DOC] Checks contains on an OrderedDict subclass input (huggingface-style
        # pattern) and that different key sets give different results.
        # pattern from huggingface
        class ClassInstantier(OrderedDict):
            # [CLASS DOC] HELPER: Empty OrderedDict subclass.
            pass

        @torch.compile(fullgraph=True, backend="eager")
        def f(x, d):
            # [FUNCTION DOC] HELPER: Adds constants depending on which keys are present.
            if "key1" in d:
                x = x + 2
            if "key2" in d:
                x = x + 4
            x = x + 8
            return x

        result = f(torch.ones(8), ClassInstantier({"key1": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 11.0)))

        result = f(torch.ones(8), ClassInstantier({"key2": torch.ones(8)}))
        self.assertTrue(same(result, torch.full([8], 13.0)))

    def test_dict_tag_guard(self):
        # [FUNCTION DOC] Checks guards on dict values and on the attribute of an object stored in
        # the dict: changing a value or the attribute must still match eager.
        class Foo:
            # [CLASS DOC] HELPER: Object with a scalar attribute of 10.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets scalar to 10.
                self.scalar = 10

        def fn(d, x):
            # [FUNCTION DOC] HELPER: Multiplies three dict entries, an attribute, and the input.
            return d["a"] * d["b"] * d["c"].scalar * x

        foo = Foo()

        d = {"a": 2, "b": 3, "c": foo}

        opt_fn = torch.compile(fn, backend="eager")
        inp = torch.randn(3, 3)
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        d["a"] = 4
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

        # Check that recompilation happens
        foo.scalar = 12
        self.assertEqual(fn(d, inp), opt_fn(d, inp))

    def test_empty_dict_recompilation(self):
        # [FUNCTION DOC] Checks truthiness of dict inputs: empty and non-empty dicts take
        # different branches and both match eager.
        def fn(d, x):
            # [FUNCTION DOC] HELPER: Returns cos when the dict is truthy, otherwise sin.
            if d:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn({}, x), opt_fn({}, x))
        self.assertEqual(fn({"a": 1}, x), opt_fn({"a": 1}, x))

    def test_udf_dict_reconstruction(self):
        # [FUNCTION DOC] Checks creating a dict via the raw dict new on a class (subclass or dict)
        # inside compiled code, setting item and attribute, and returning with correct type.
        class MyDict(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        def fn(x, klass):
            # [FUNCTION DOC] HELPER: Creates an empty instance of the given class via the dict
            # new, sets an item, and sets an attribute only for the subclass.
            x = x * 2
            sc_dict = dict.__new__(klass)
            sc_dict["x"] = x
            if isinstance(sc_dict, MyDict):
                sc_dict.attr = 3
            return sc_dict

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x, MyDict)
        res = opt_fn(x, MyDict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, MyDict))
        self.assertEqual(ref.attr, res.attr)

        ref = fn(x, dict)
        res = opt_fn(x, dict)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, dict))

    def test_weakref_dict(self):
        # [FUNCTION DOC] Checks contains on a WeakKeyDictionary with module keys inside a
        # compiled function.
        states = weakref.WeakKeyDictionary()

        mod1 = torch.nn.Module()
        mod2 = torch.nn.Module()

        states[mod1] = 2
        states[mod2] = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies sin or cos depending on module membership.
            if mod1 in states:
                x = torch.sin(x)
            if mod2 in states:
                x = torch.cos(x)
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_construct_user_dict_and_return(self):
        # [FUNCTION DOC] Checks that a UserDict subclass created inside compiled code can be
        # returned and read.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a DummyUserDict holding the input plus one.
            return DummyUserDict({"a": x + 1})

        x = torch.randn(4)
        res = fn(x)
        self.assertEqual(res["a"], x + 1)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(res["a"], opt_fn(x)["a"])

    def test_mapping_proxy_for_local(self):
        # [FUNCTION DOC] Checks a mappingproxy built from a local dict in compiled code can be
        # indexed, iterated and returned, and is returned as a mappingproxy type.
        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps a local dict in a mappingproxy, reads and iterates,
            # returns it.
            d = {"a": 2, "b": 3, "c": 5 * x}
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

    def test_mapping_proxy_for_nonlocal(self):
        # [FUNCTION DOC] Checks a mappingproxy over a global dict that is mutated during tracing
        # is returned as a mappingproxy that still reflects later mutations of the dict.
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x):
            # [FUNCTION DOC] HELPER: Wraps the global dict, reads and iterates it, adds a key,
            # returns proxy.
            mp = types.MappingProxyType(d)
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            d["d"] = 4
            return mp

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertTrue(type(res) is types.MappingProxyType)

        # check update to d is reflected in res
        d["e"] = 5
        self.assertEqual(d["e"], res["e"])

    def test_mapping_proxy_existing(self):
        # [FUNCTION DOC] Checks a mappingproxy passed as input is readable and recognized by
        # an isinstance check; changing a value and popping a key of the backing dict must
        # still match.
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x, mp):
            # [FUNCTION DOC] HELPER: Indexes and iterates the proxy, doubling if it is
            # a mappingproxy.
            y = torch.sin(x * mp["a"])
            for v in mp.values():
                y += torch.cos(x * v)
            if isinstance(mp, types.MappingProxyType):
                y *= 2
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        mp = types.MappingProxyType(d)
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

        d["a"] = 3
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

        d.pop("b")
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

    def test_dict_construction_from_mapping_proxy(self):
        # [FUNCTION DOC] Checks constructing a dict from a mappingproxy input inside
        # compiled code.
        d = {"a": 2, "b": 3, "c": 5}

        def fn(x, mp):
            # [FUNCTION DOC] HELPER: Copies the proxy into a dict and uses one entry.
            d = dict(mp)
            y = torch.sin(x * d["a"])
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        mp = types.MappingProxyType(d)
        ref = fn(x, mp)
        res = opt_fn(x, mp)
        self.assertEqual(ref, res)

    def test_mapping_proxy_existing_mutation(self):
        # [FUNCTION DOC] Checks that mutating a global dict inside compiled code is visible
        # through a pre-existing mappingproxy, both during tracing and after the call.
        d = {"a": 2, "b": 3, "c": 5}

        mp = types.MappingProxyType(d)

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds a key to the backing dict and reads it through
            # the proxy.
            d["d"] = 4
            y = torch.sin(x * mp["d"])
            return y

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        ref = torch.sin(x * 4)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(d.keys(), mp.keys())

    def test_mapping_proxy_existing_local_mutation(self):
        # [FUNCTION DOC] Checks mutating a fresh local dict does not cause a graph break when a
        # pre-existing mappingproxy exists (fullgraph compile succeeds).
        d = {"a": 2, "b": 3, "c": 5}

        mp = types.MappingProxyType(d)

        def fn(x):
            # [FUNCTION DOC] HELPER: Mutates an unrelated local dict and reads through the proxy.
            # Dynamo should not cause a graph break here because it knows that
            # the existing proxy can't point to this new dict
            other_dict = {}
            other_dict["d"] = 4
            y = torch.sin(x * mp["c"])
            return y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = torch.sin(x * mp["c"])
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(d.keys(), mp.keys())

    def test_dict_view_mapping(self):
        # [FUNCTION DOC] Checks the mapping attribute of dict keys, values and items views gives a
        # read-only mappingproxy equal to the dict.
        # dict_keys/values/items expose a read-only mappingproxy via .mapping
        mappingproxy = type(type.__dict__)

        def fn(x):
            # [FUNCTION DOC] HELPER: Gets mappings from the three views and checks them against
            # the dict.
            d = {"a": 1, "b": 2}
            m_keys = d.keys().mapping
            m_values = d.values().mapping
            m_items = d.items().mapping
            y = torch.sin(x)
            for m in (m_keys, m_values, m_items):
                if isinstance(m, mappingproxy) and m == d:
                    y = y + 1
            return y, m_keys, m_values, m_items

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        for m in res[1:]:
            self.assertTrue(type(m) is mappingproxy)
            self.assertEqual(m, {"a": 1, "b": 2})

    def test_dict_view_mapping_reflects_mutation(self):
        # [FUNCTION DOC] Checks the mappingproxy from a view mapping attribute reflects later
        # mutation of the underlying dict.
        # The mappingproxy returned by .mapping proxies the live dict, so a
        # later mutation must be visible through it.
        mappingproxy = type(type.__dict__)

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the view mapping of an empty dict, adds a key, copies
            # the proxy.
            d = {}
            m = d.keys().mapping
            d["foo"] = "bar"
            return torch.sin(x), isinstance(m, mappingproxy), dict(m)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertTrue(res[1])
        self.assertEqual(res[2], {"foo": "bar"})

    def test_move_to_end(self):
        # [FUNCTION DOC] Checks OrderedDict move_to_end changes key order in the compiled result.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds an OrderedDict and moves key a to the end.
            d = OrderedDict({"a": torch.cos(x), "b": 3, "c": 5})
            d.move_to_end("a")
            return d

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(["b", "c", "a"], list(opt_fn(x).keys()))
        self.assertEqual(fn(x), opt_fn(x))

    def test_mapping_proxy_ban_muation_on_dict_realization(self):
        # [FUNCTION DOC] Checks that copying a class __dict__ into a dict and then assigning a new
        # class attribute (which requires a graph break) still gives eager results and the same
        # class attribute.
        def fn(x):
            # [FUNCTION DOC] HELPER: Defines a local class, copies its namespace, then sets a new
            # class attribute and reads it from the namespace proxy.
            class Foo:
                # [CLASS DOC] HELPER: Local class with a class attribute b of 4.
                b = 4

            d = dict(Foo.__dict__)
            y = torch.sin(x) * d["b"]
            # This should cause a graph break, because otherwise the
            # Foo.__dict__ will not be updated.
            Foo.bar = 3
            return Foo, y * Foo.__dict__["bar"]

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        foo1, ref = fn(x)
        foo2, res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(foo1.bar, foo2.bar)

    def test_overridden_get_item(self):
        # [FUNCTION DOC] Checks that an overridden getitem on a dict subclass is called the same
        # number of times in compiled code as in eager and its result (value plus one) is used.
        class MyDict(dict):
            # [CLASS DOC] HELPER: dict subclass counting getitem calls in calls and returning
            # value plus one.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Initializes the dict and sets the call counter to 0.
                super().__init__(*args, **kwargs)
                self.calls = 0

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Increments the counter and returns the stored value
                # plus one.
                self.calls += 1
                return super().__getitem__(key) + 1

        def fn(x, d):
            # [FUNCTION DOC] HELPER: Adds a key, then sums the input times one entry with others.
            d["d"] = 4
            return x * d["a"] + d["b"] + d["c"] + d["d"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        d1 = MyDict({"a": 2, "b": 3, "c": 5})
        ref = fn(x, d1)

        d2 = MyDict({"a": 2, "b": 3, "c": 5})
        res = opt_fn(x, d2)
        self.assertEqual(ref, res)
        self.assertEqual(d1.calls, d2.calls)

    def test_items_type(self):
        # [FUNCTION DOC] Checks the items view returned from compiled code has the dict_items type
        # and equals the eager result.
        def fn():
            # [FUNCTION DOC] HELPER: Builds a dict with mixed value types and returns its items.
            d = dict({"a": 1, "b": "2", "c": torch.tensor(3)})  # noqa: C418
            return d.items()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn()
        res = opt_fn()
        self.assertEqual(ref, res)
        self.assertEqual(type(res), dict_items)

    def test_builtin_or_with_invalid_types(self):
        # [FUNCTION DOC] Checks the or operator of a dict with non-dict operands (int, float, str,
        # tuple, list) is unsupported: each call raises Unsupported.
        args = (
            1,  # int
            1.0,  # float
            "a",  # str
            (1, 2),  # tuple
            [1, 2],  # list
        )

        @torch.compile(backend="eager", fullgraph=True)
        def fn(b: Any):
            # [FUNCTION DOC] HELPER: Returns the union of a literal dict with the argument.
            a = {"one": torch.ones(1)}
            return a | b

        for arg in args:
            with self.assertRaises(Unsupported):
                _ = fn(arg)

    def test_builtin_or_with_diff_keys(self):
        # [FUNCTION DOC] Checks dict union for dicts with disjoint keys, via operator and dunder
        # method calls in both orders.
        def f():
            # [FUNCTION DOC] HELPER: Returns two dicts and their unions in all forms.
            a = {"one": torch.ones(1)}
            b = {"two": torch.ones(2)}
            return a, b, a | b, b | a, a.__or__(b), b.__or__(a)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_builtin_or_with_same_keys(self):
        # [FUNCTION DOC] Checks dict union with overlapping keys, in both orders and via
        # dunder calls.
        def f():
            # [FUNCTION DOC] HELPER: Returns two dicts and their unions in all forms.
            a = {"one": torch.ones(1), "two": torch.ones(2)}
            b = {"one": torch.ones(1), "three": torch.ones(3)}
            return a, b, a | b, b | a, a.__or__(b), b.__or__(a)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_builtin_ior_(self):
        # [FUNCTION DOC] Checks in-place dict union updates the left dict inside compiled code.
        def f():
            # [FUNCTION DOC] HELPER: Updates one dict in place with another and returns both.
            a = {"one": torch.ones(1)}
            b = {"two": torch.ones(2)}
            a |= b
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_defaultdict_inplace_union_preserves_factory(self):
        # [FUNCTION DOC] Checks in-place union of a defaultdict with a list of pairs and a dict
        # keeps the same object, the factory and the type, and still creates missing keys.
        def f():
            # [FUNCTION DOC] HELPER: Applies in-place union twice and returns identity, factory
            # and type flags, contents, key list and a missing-key lookup.
            d = defaultdict(int, {1: 1, 2: 2})
            d |= [(0, "zero"), (1, "one")]
            result = d.__ior__({3: "three"})
            return (
                result is d,
                d.default_factory is int,
                type(d) is defaultdict,
                dict(d),
                list(d),
                d[4],
            )

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_defaultdict_shallow_copy_preserves_factory(self):
        # [FUNCTION DOC] Checks the three shallow copy forms of a defaultdict preserve the
        # factory, contents and equality; copies stay inside the graph.
        # defaultdict.copy(), dd.__copy__(), and copy.copy(dd) all produce a
        # new defaultdict with the same default_factory and a shallow copy of
        # the contents. The copies are consumed in-graph because reconstructing
        # an escaping sourceless defaultdict is a separate unsupported case.
        import copy

        def f():
            # [FUNCTION DOC] HELPER: Copies a defaultdict three ways and reports factory,
            # contents, equality and a lookup of a missing key.
            d = defaultdict(list, {1: 1, 2: 2})
            c1 = d.copy()
            c2 = d.__copy__()
            c3 = copy.copy(d)
            return (
                c1.default_factory is list,
                c2.default_factory is list,
                c3.default_factory is list,
                dict(c1),
                dict(c2),
                dict(c3),
                c1 == d,
                c3 == d,
                c1[5],
            )

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_newly_constructed_default_dict(self):
        # [FUNCTION DOC] Checks a defaultdict created and filled in compiled code and returned
        # matches eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Builds a defaultdict of lists, sets one key, returns it with
            # a tensor.
            d = defaultdict(list)
            d[0] = [
                42,
            ]
            return x + 1, d

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_newly_constructed_default_dict_no_default_factory(self):
        # [FUNCTION DOC] Checks a defaultdict created with no factory (explicit none or omitted)
        # raises KeyError on missing keys in compiled code just like eager, in three variants.
        def f1(x):
            # [FUNCTION DOC] HELPER: Empty defaultdict with no factory; falls back to assigning
            # on KeyError.
            d = defaultdict()
            try:
                d[1] += 42
            except KeyError:
                d[1] = 1
            return x + 1, d

        x = torch.ones(2)
        ref = f1(x)
        res = torch.compile(f1, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

        def f2(x):
            # [FUNCTION DOC] HELPER: Same as the first variant with an explicit none factory.
            d = defaultdict(None)
            try:
                d[1] += 42
            except KeyError:
                d[1] = 1
            return x + 1, d

        ref = f2(x)
        res = torch.compile(f2, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

        def f3(x):
            # [FUNCTION DOC] HELPER: No-factory defaultdict with an initial entry; one update
            # succeeds and a missing key falls back on KeyError.
            d = defaultdict(None, {1: 10})
            d[1] += 42
            try:
                d[2] += 24
            except KeyError:
                d[2] = 1
            return x + 1, d

        ref = f3(x)
        res = torch.compile(f3, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_newly_constructed_default_dict_with_dict(self):
        # [FUNCTION DOC] Checks constructing a defaultdict from a factory, an existing dict and
        # keyword arguments, then appending to a missing key.
        def f(x):
            # [FUNCTION DOC] HELPER: Builds a dict and a defaultdict from it and appends to a
            # new key.
            d = dict([("a", 1), ("b", 2)], c=3)
            dd = defaultdict(list, d, d=4, e=5)
            dd["x"].append(42)
            return x + 1, d, dd

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_iter_default_dict(self):
        # [FUNCTION DOC] Checks iterating a defaultdict while updating its existing values.
        def f(x):
            # [FUNCTION DOC] HELPER: Sets one key, then increments each value found
            # while iterating.
            d = defaultdict(list)
            d[0] = 42
            for k in d:
                d[k] += 1
            return x + 1, d

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)

        self.assertEqual(ref, res)

    def test_dict_new_ignores_extra_args(self):
        # [FUNCTION DOC] Checks that creating an OrderedDict with keyword args works (the dict new
        # ignores extra args) in compiled code.
        """dict.__new__ ignores extra args (CPython behavior).

        This matters for instantiate_user_defined_class_object which calls
        cls.__new__(cls, *args, **kwargs) — dict.__new__ should not fail
        on the extra args.
        """

        def f(x):
            # [FUNCTION DOC] HELPER: Builds an OrderedDict with keyword args and adds one entry
            # to input.
            od = OrderedDict(a=1, b=2)
            return x + od["a"]

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_ordered_dict_repr(self):
        # [FUNCTION DOC] Checks calling repr on an OrderedDict does not graph break
        # under fullgraph.
        """repr() on OrderedDictVariable should not graph break."""

        def f(x):
            # [FUNCTION DOC] HELPER: Takes repr of an OrderedDict and branches on it being
            # a string.
            od = OrderedDict(a=1, b=2)
            r = repr(od)
            # Just verify repr doesn't graph break and returns a string
            return x + (1 if isinstance(r, str) else 0)

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_c_new_init_args_ignored_for_dict(self):
        # [FUNCTION DOC] Checks that an OrderedDict built from a generator compiles under
        # fullgraph (generator is not kept in the reconstruction arguments).
        """C-level __new__ for dict/set passes init_args=[] since they
        ignore extra args. This ensures generators passed to OrderedDict()
        don't end up in reconstruction."""

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding two key and tensor pairs.
            yield 1, t + 1
            yield 2, t + 2

        def f(t):
            # [FUNCTION DOC] HELPER: Builds an OrderedDict from the generator.
            return OrderedDict(whoo(t))

        t = torch.randn(2)
        ref = f(t)
        res = torch.compile(f, backend="eager", fullgraph=True)(t)
        self.assertEqual(ref, res)

    def test_ordered_dict_as_python_constant_preserves_type(self):
        # [FUNCTION DOC] Checks the python constant form of an OrderedDict keeps its type, by
        # using it in the enum functional API.
        """as_python_constant should return OrderedDict, not plain dict."""

        def f(x):
            # [FUNCTION DOC] HELPER: Builds an enum from an OrderedDict and adds a member value
            # to input.
            od = OrderedDict(a=1, b=2)
            # Enum functional API calls as_python_constant on the OrderedDict
            import enum

            E = enum.Enum("E", od)
            return x + E.a.value

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    def test_default_dict_as_python_constant_preserves_type(self):
        # [FUNCTION DOC] Checks the python constant form of a defaultdict keeps its type, via an
        # isinstance check in compiled code.
        """as_python_constant should return defaultdict, not plain dict."""

        def f(x):
            # [FUNCTION DOC] HELPER: Asserts a defaultdict is a defaultdict, adds an entry
            # to input.
            dd = defaultdict(int, a=1, b=2)
            # isinstance triggers as_python_constant internally for
            # constant folding the type check
            assert isinstance(dd, defaultdict)  # noqa: S101
            return x + dd["a"]

        x = torch.ones(2)
        ref = f(x)
        res = torch.compile(f, backend="eager", fullgraph=True)(x)
        self.assertEqual(ref, res)

    @parametrize("op", ["or_", "and_", "xor", "sub"])
    def test_dict_keys_binop(self, op):
        # [FUNCTION DOC] Parametrized over or, and, xor, sub: checks set operators between dict
        # keys views of two dicts in both operand orders.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator to the keys views of two dicts
            # both ways.
            a = {"one": torch.ones(1), "two": torch.ones(2)}
            b = {"one": torch.ones(1), "three": torch.ones(3)}
            return op(a.keys(), b.keys()), op(b.keys(), a.keys())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["ior", "iand", "ixor", "isub"])
    def test_dict_keys_inplace_binop(self, op):
        # [FUNCTION DOC] Parametrized over in-place set operators: checks them on dict keys views
        # including a rebinding of the variable.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the in-place operator twice on keys views and
            # returns both.
            a = {"one": torch.ones(1), "two": torch.ones(2)}.keys()
            b = {"one": torch.ones(1), "three": torch.ones(3)}.keys()
            c = {"one": torch.ones(1), "two": torch.ones(2)}.keys()
            a = op(a, b)
            b = op(b, c)
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub"])
    def test_dict_items_binop(self, op):
        # [FUNCTION DOC] Parametrized over or, and, xor, sub: checks set operators on dict
        # items views.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator to the items views of two dicts
            # both ways.
            a = {"one": 1, "two": 2}
            b = {"one": 1, "three": 3}
            return op(a.items(), b.items()), op(b.items(), a.items())

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["ior", "iand", "ixor", "isub"])
    def test_dict_items_inplace_binop(self, op):
        # [FUNCTION DOC] Parametrized over in-place set operators: checks them on dict
        # items views.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the in-place operator twice on items views and
            # returns both.
            a = {"one": 1, "two": 2}.items()
            b = {"one": 1, "three": 3}.items()
            c = {"one": 1, "two": 2}.items()
            a = op(a, b)
            b = op(b, c)
            return a, b

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_set(self, op):
        # [FUNCTION DOC] Parametrized over arithmetic and in-place set operators: checks mixing a
        # dict keys view with a set in both orders.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between a keys view and a set both ways.
            keys = {"one": 1, "two": 2, "three": 3}.keys()
            s = {"one", "four"}
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_items_vs_set(self, op):
        # [FUNCTION DOC] Parametrized over set operators: checks mixing a dict items view with
        # a set.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between an items view and a set of pairs
            # both ways.
            items = {"one": 1, "two": 2}.items()
            s = {("one", 1), ("three", 3)}
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_dict_items(self, op):
        # [FUNCTION DOC] Parametrized over set operators: checks mixing a keys view with an
        # items view.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between keys and items views both ways.
            keys = {"one": 1, "two": 2}.keys()
            items = {"three": 3, "four": 4}.items()
            return op(keys, items), op(items, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_keys_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over set operators: checks mixing a keys view with a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between a keys view and a MySet
            # both ways.
            keys = {"one": 1, "two": 2, "three": 3}.keys()
            s = MySet({"one", "four"})
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_dict_items_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over set operators: checks mixing an items view with a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between an items view and a MySet
            # both ways.
            items = {"one": 1, "two": 2, "three": 3}.items()
            s = MySet({"one", "four"})
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["or_", "and_", "xor", "sub", "iand", "ior", "ixor", "isub"])
    def test_cross_type_set_binop_set_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over set operators: checks mixing a plain set with a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the operator between a set and a MySet both ways.
            s = {"one", "two", "three"}
            u = MySet({"one", "four"})
            return op(s, u), op(u, s)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_set(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares dict keys view and a set
        # both ways.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between a keys view and a set
            # both ways.
            keys = {"one": 1, "two": 2}.keys()
            s = {"one", "two", "three"}
            return op(keys, s), op(s, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_items_vs_set(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares dict items view and
        # a set.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between an items view and a set
            # both ways.
            items = {"one": 1, "two": 2}.items()
            s = {("one", 1), ("two", 2), ("three", 3)}
            return op(items, s), op(s, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_dict_items(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares keys view and
        # items view.
        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between keys and items views
            # both ways.
            keys = {"one": 1, "two": 2}.keys()
            items = {"one": 1, "two": 2}.items()
            return op(keys, items), op(items, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_set_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares a set and a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between a set and a MySet both ways.
            s = {"one", "two"}
            u = MySet({"one", "two", "three"})
            return op(s, u), op(u, s)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_keys_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares keys view and a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between a keys view and a MySet
            # both ways.
            keys = {"one": 1, "two": 2}.keys()
            u = MySet({"one", "two", "three"})
            return op(keys, u), op(u, keys)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    @parametrize("op", ["lt", "le", "gt", "ge", "eq", "ne"])
    def test_cross_type_cmp_dict_items_vs_user_defined_set(self, op):
        # [FUNCTION DOC] Parametrized over comparison operators: compares items view and a
        # set subclass.
        class MySet(set):
            # [CLASS DOC] HELPER: Empty subclass of set.
            pass

        op = getattr(operator, op)

        def f():
            # [FUNCTION DOC] HELPER: Applies the comparison between an items view and a MySet
            # both ways.
            items = {"one": 1, "two": 2}.items()
            u = MySet({("one", 1), ("two", 2), ("three", 3)})
            return op(items, u), op(u, items)

        opt_f = torch.compile(f, backend="eager", fullgraph=True)
        self.assertEqual(f(), opt_f())

    def test_dict_view_iand_rebinds_variable(self):
        # [FUNCTION DOC] Checks that in-place intersection on a variable holding a keys or items
        # view rebinds the variable to a set, leaves the dict unchanged, and matches eager.
        def f_keys():
            # [FUNCTION DOC] HELPER: Intersects a keys view with a set in place; returns result
            # and dict.
            d = {"one": 1, "two": 2, "three": 3}
            tmp = d.keys()
            tmp &= {"one", "four"}
            return tmp, d

        opt_f_keys = torch.compile(f_keys, backend="eager", fullgraph=True)
        eager_tmp, eager_d = f_keys()
        compiled_tmp, compiled_d = opt_f_keys()
        self.assertIsInstance(eager_tmp, set)
        self.assertEqual(eager_tmp, compiled_tmp)
        self.assertEqual(eager_d, compiled_d)

        def f_items():
            # [FUNCTION DOC] HELPER: Intersects an items view with a set of pairs in place;
            # returns result and dict.
            d = {"one": 1, "two": 2}
            tmp = d.items()
            tmp &= {("one", 1), ("three", 3)}
            return tmp, d

        opt_f_items = torch.compile(f_items, backend="eager", fullgraph=True)
        eager_tmp, eager_d = f_items()
        compiled_tmp, compiled_d = opt_f_items()
        self.assertIsInstance(eager_tmp, set)
        self.assertEqual(eager_tmp, compiled_tmp)
        self.assertEqual(eager_d, compiled_d)

    def test_dict_union_result_type(self):
        # [FUNCTION DOC] Eager-only check of the result type of dict union across defaultdict,
        # OrderedDict and dict (the left operand type wins, except plain dict left with a subclass
        # right operand gives the subclass); no compilation involved.
        def_dict = defaultdict(int, {1: 1, 2: 2})
        ord_dict = OrderedDict({3: 3, 4: 4})
        base_dict = {5: 5, 6: 6}

        exp_def = def_dict | ord_dict
        self.assertIsInstance(exp_def, defaultdict)

        exp_def_2 = def_dict | base_dict
        self.assertIsInstance(exp_def_2, defaultdict)

        exp_def_3 = base_dict | def_dict
        self.assertIsInstance(exp_def_3, defaultdict)

        exp_ord = ord_dict | def_dict
        self.assertIsInstance(exp_ord, OrderedDict)

        exp_ord_2 = base_dict | ord_dict
        self.assertIsInstance(exp_ord_2, OrderedDict)

        exp_base = base_dict | base_dict
        self.assertIsInstance(exp_base, dict)

    def test_range_as_dict_key(self):
        # [FUNCTION DOC] Checks range objects as dict keys, looked up with equal ranges.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by ranges and looks up with equal ranges.
            d = {range(5): x * 2, range(10, 15): x * 3}
            return d[range(0, 5, 1)] + d[range(10, 15)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_tuple_as_dict_key(self):
        # [FUNCTION DOC] Checks tuples as dict keys.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by tuples and looks both up.
            d = {(1, 2): x * 2, (3, 4, 5): x * 3}
            return d[(1, 2)] + d[(3, 4, 5)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_enum_as_dict_key(self):
        # [FUNCTION DOC] Checks Enum members as dict keys.
        class Color(enum.Enum):
            # [CLASS DOC] HELPER: Enum with members RED, GREEN, BLUE.
            RED = 1
            GREEN = 2
            BLUE = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by Color members and sums lookups.
            d = {Color.RED: x * 2, Color.GREEN: x * 3, Color.BLUE: x * 4}
            return d[Color.RED] + d[Color.GREEN] + d[Color.BLUE]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_intenum_as_dict_key(self):
        # [FUNCTION DOC] Checks IntEnum members as dict keys.
        class Priority(enum.IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members LOW, MEDIUM, HIGH.
            LOW = 1
            MEDIUM = 2
            HIGH = 3

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by Priority members and sums lookups.
            d = {Priority.LOW: x * 2, Priority.MEDIUM: x * 3, Priority.HIGH: x * 4}
            return d[Priority.LOW] + d[Priority.MEDIUM] + d[Priority.HIGH]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_frozenset_as_dict_key(self):
        # [FUNCTION DOC] Checks frozensets as dict keys.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by frozensets and looks up
            # equal frozensets.
            d = {frozenset([1, 2]): x * 2, frozenset([3, 4, 5]): x * 3}
            return d[frozenset([1, 2])] + d[frozenset([3, 4, 5])]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_typing_union_as_dict_key(self):
        # [FUNCTION DOC] Checks typing Union types as dict keys.
        from typing import Union

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by two Union types and looks them up.
            d = {Union[int, str]: x * 2, Union[float, bool]: x * 3}  # noqa: UP007
            return d[Union[int, str]] + d[Union[float, bool]]  # noqa: UP007

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_numpy_dtype_as_dict_key(self):
        # [FUNCTION DOC] Checks numpy scalar types as dict keys.
        import numpy as np

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by three numpy types and sums lookups.
            d = {np.float32: x * 2, np.int64: x * 3, np.bool_: x * 4}
            return d[np.float32] + d[np.int64] + d[np.bool_]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_method_wrapper_as_dict_key(self):
        # [FUNCTION DOC] Checks built-in method wrapper objects (list add and mul) as dict keys.
        add_method = list.__add__
        mul_method = list.__mul__

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by the captured method wrappers and
            # sums lookups.
            # Method wrappers are the type of bound methods on built-in types
            d = {add_method: x * 2, mul_method: x * 3}
            return d[add_method] + d[mul_method]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_torch_builtin_function_as_dict_key(self):
        # [FUNCTION DOC] Checks torch builtin functions as dict keys.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by torch.add, torch.mul, torch.sub and
            # sums lookups.
            # Using torch built-in functions as dictionary keys
            d = {torch.add: x * 2, torch.mul: y * 3, torch.sub: x + y}
            return d[torch.add] + d[torch.mul] + d[torch.sub]

        x = torch.randn(4)
        y = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, y), opt_fn(x, y))

    def test_frozen_dataclass_as_dict_key(self):
        # [FUNCTION DOC] Checks instances of a frozen dataclass as dict keys, looked up by equal
        # newly constructed instances.
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class Point:
            # [CLASS DOC] HELPER: Frozen dataclass with int fields x and y (hashable by value).
            x: int
            y: int

        def fn(tensor):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by two Point instances and looks up
            # equal ones.
            p1 = Point(1, 2)
            p2 = Point(3, 4)
            d = {p1: tensor * 2, p2: tensor * 3}
            return d[Point(1, 2)] + d[Point(3, 4)]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_list_as_dict_key_raises_typeerror(self):
        # [FUNCTION DOC] Checks that using a list as a dict key raises TypeError in eager,
        # and that the compiled version raises Unsupported whose message mentions an
        # observed exception.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict with a list key (unhashable) and looks it up.
            d = {[1, 2, 3]: x * 2}
            return d[[1, 2, 3]]

        x = torch.randn(4)

        # First check that eager execution raises TypeError
        with self.assertRaises(TypeError):
            fn(x)

        # Also check that compiled version raises TypeError
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(Unsupported, "Observed exception"):
            opt_fn(x)

    def test_property_backed_by_dunder_dict(self):
        # [FUNCTION DOC] Checks a property that reads the instance __dict__ works in
        # compiled code.
        class Obj:
            # [CLASS DOC] HELPER: Object storing x only in __dict__ and exposing it via
            # a property.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the instance __dict__.
                self.__dict__["x"] = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property returning the x entry of __dict__.
                return self.__dict__["x"]

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the x attribute of the object.
            return obj.x

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_property_backed_by_dunder_dict_mutated(self):
        # [FUNCTION DOC] Checks reading then assigning through a property backed by __dict__ in
        # compiled code matches eager on a fresh object.
        class Obj:
            # [CLASS DOC] HELPER: Object with property x backed by the x entry of __dict__,
            # with setter.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the instance __dict__.
                self.__dict__["x"] = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the x entry of __dict__.
                return self.__dict__["x"]

            @x.setter
            def x(self, value):
                # [FUNCTION DOC] HELPER: Property setter storing the value in the x entry
                # of __dict__.
                self.__dict__["x"] = value

        def fn(obj):
            # [FUNCTION DOC] HELPER: Reads x, assigns x plus one, returns x.
            a = obj.x
            obj.x = a + 1
            return obj.x

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(Obj(3)), opt_fn(obj))

    def test_property_backed_by_dunder_dict_pending_side_effects(self):
        # [FUNCTION DOC] Checks direct reads and writes on the object __dict__ when a property of
        # the same name exists and side effects are pending.
        class Obj:
            # [CLASS DOC] HELPER: Object with property x backed by the x entry of __dict__.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the instance __dict__.
                self.__dict__["x"] = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the x entry of __dict__.
                return self.__dict__["x"]

        def fn(obj):
            # [FUNCTION DOC] HELPER: Reads and writes the x entry of __dict__ and returns it.
            a = obj.__dict__["x"]
            obj.__dict__["x"] = a + 1
            return obj.__dict__["x"]

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(Obj(3)), opt_fn(obj))

    def test_dunder_dict_getitem_guard_recompiles(self):
        # [FUNCTION DOC] Checks that reading __dict__ directly (not via the shadowing property)
        # gives the stored value and that changing the entry recompiles (frame count goes 1 to 2).
        class Obj:
            # [CLASS DOC] HELPER: Object whose x entry in __dict__ is shadowed by a property
            # returning 999.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores 3 under x in the instance __dict__.
                self.__dict__["x"] = 3

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property returning the constant 999.
                return 999

        def fn(obj, t):
            # [FUNCTION DOC] HELPER: Adds the __dict__ x entry to the tensor.
            return t + obj.__dict__["x"]

        obj = Obj()
        t = torch.tensor(0.0)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        self.assertEqual(opt_fn(obj, t), torch.tensor(3.0))
        self.assertEqual(cnt.frame_count, 1)

        obj.__dict__["x"] = 5
        self.assertEqual(opt_fn(obj, t), torch.tensor(5.0))
        self.assertEqual(cnt.frame_count, 2)

    def test_dunder_dict_get_with_property(self):
        # [FUNCTION DOC] Checks get on __dict__ with a property of the same name returns the
        # stored value.
        class Obj:
            # [CLASS DOC] HELPER: Object with property x backed by the x entry of __dict__.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the instance __dict__.
                self.__dict__["x"] = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the x entry of __dict__.
                return self.__dict__["x"]

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the get result for key x on the object __dict__.
            return obj.__dict__.get("x")

        obj = Obj(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_dunder_dict_getitem_tensor(self):
        # [FUNCTION DOC] Checks a tensor stored in __dict__ and read via a property in
        # compiled code.
        class Obj:
            # [CLASS DOC] HELPER: Object with property x backed by the x entry of __dict__.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x in the instance __dict__.
                self.__dict__["x"] = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property getter returning the x entry of __dict__.
                return self.__dict__["x"]

        def fn(obj):
            # [FUNCTION DOC] HELPER: Returns the property value plus one.
            return obj.x + 1

        obj = Obj(torch.tensor(3.0))
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(obj), opt_fn(obj))

    def test_get_default_nowrap_functions_as_dict_key(self):
        # [FUNCTION DOC] Checks the default nowrap function set can be used (as a frozenset) as a
        # dict key and tested for membership of the tensor grad getter.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict keyed by the frozen set of nowrap functions and
            # returns its entry plus input when the grad getter is a member.
            # Get the set of default nowrap functions
            nowrap_funcs = torch.overrides.get_default_nowrap_functions()
            # Use the set as a dict key and search for Tensor.grad.__get__ in it
            d = {frozenset(nowrap_funcs): x * 2}
            # Check if Tensor.grad.__get__ is in the set
            if torch.Tensor.grad.__get__ in nowrap_funcs:
                return d[frozenset(nowrap_funcs)] + x
            return x

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_custom_bool(self):
        # [FUNCTION DOC] Checks that boolean conversion of a user object defining a boolean hook
        # (not a real dict) honors the hook, with and without negation, for both True and False.
        class CustomBoolDict:
            # [CLASS DOC] HELPER: Object whose truth value is the stored bool_result.
            def __init__(self, bool_result: bool):
                # [FUNCTION DOC] HELPER: Stores the requested truth value.
                super().__init__()
                self._bool_result = bool_result

            def __bool__(self):
                # [FUNCTION DOC] HELPER: Returns the stored truth value.
                return self._bool_result

        def fn(t: torch.Tensor, d, apply_not: bool):
            # [FUNCTION DOC] HELPER: Returns sin of the tensor and the bool (optionally negated)
            # of d.
            if apply_not:
                return t.sin(), bool(not d)
            else:
                return t.sin(), bool(d)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        t = CustomBoolDict(True)
        self.assertTrue(
            opt_fn(x, t, False)[1],
            "CustomBoolDict(True) should evaluate to True in boolean context",
        )
        self.assertFalse(
            opt_fn(x, t, True)[1],
            "not CustomBoolDict(True) should evaluate to False in boolean context",
        )

        f = CustomBoolDict(False)
        self.assertFalse(
            opt_fn(x, f, False)[1],
            "CustomBoolDict(False) should evaluate to False in boolean context",
        )
        self.assertTrue(
            opt_fn(x, f, True)[1],
            "not CustomBoolDict(False) should evaluate to True in boolean context",
        )

    def _get_fw_graphs_for_dict_orders(self):
        # [FUNCTION DOC] HELPER: Compiles a small model twice under the AOT eager recording
        # backend with inputs whose dict key order differs (the second compile after a dynamo
        # reset and zero grad), runs backward each time, and returns the two forward FX graphs.
        # The model has a deep path for key a and a shallow path for other keys, so order changes
        # op order.
        """Compile a model with two different dict key orders, return both
        forward graphs from AOTAutograd (the level where has_same_nodes operates).

        Uses asymmetric paths (deep vs shallow) so that different dict iteration
        orders produce different op orderings in the graph.
        """

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with deep (Linear, ReLU, Linear) and shallow (Linear)
            # branches applied per dict item depending on the key.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the deep and shallow submodules.
                super().__init__()
                self.deep = torch.nn.Sequential(
                    torch.nn.Linear(8, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                )
                self.shallow = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Applies deep for key a, shallow otherwise, concatenates
                # and sums.
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.deep(val))
                    else:
                        results.append(self.shallow(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        backend1 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        torch.compile(model, backend=backend1)(d1).backward()
        torch._dynamo.reset()
        model.zero_grad()
        backend2 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        torch.compile(model, backend=backend2)(d2).backward()

        return backend1.fw_graphs[0].graph, backend2.fw_graphs[0].graph

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=False)
    def test_name_based_hash_diverges_on_dict_order(self):
        # [FUNCTION DOC] Run with canonicalization of output graph node order disabled: shows that
        # a hash of node names differs between the graphs produced with different dict key orders.
        # Demonstrates the original bug: the name-based hash used in
        # has_same_nodes diverges when different ranks trace with different
        # dict iteration orders, producing different node orderings.
        import hashlib

        graph1, graph2 = self._get_fw_graphs_for_dict_orders()

        def name_based_hash(graph):
            # [FUNCTION DOC] HELPER: Returns the SHA-256 hex digest of the node names joined
            # by slashes.
            node_str = "/".join(x.name for x in graph.nodes)
            return hashlib.sha256(node_str.encode("utf-8")).hexdigest()

        self.assertNotEqual(name_based_hash(graph1), name_based_hash(graph2))

    def test_canonical_names_invariant_to_dict_order(self):
        # [FUNCTION DOC] Checks the canonical node naming helper from the partitioners module
        # gives identical name-to-structure mappings (op and target, targets of placeholders
        # excluded) for graphs traced with different dict orders.
        # The canonical naming produces identical mappings for structurally
        # equivalent graphs traced with different dict iteration orders.
        from torch._functorch.partitioners import _canonical_node_names

        graph1, graph2 = self._get_fw_graphs_for_dict_orders()

        canonical1 = _canonical_node_names(graph1)
        canonical2 = _canonical_node_names(graph2)

        # Build {canonical_name: (op, target)} for each graph.
        # For placeholders, exclude target (it's the rank-local name like
        # primals_N which differs across ranks).
        def structure(graph, canonical):
            # [FUNCTION DOC] HELPER: Maps each node's canonical name to its op and target string
            # (empty for placeholders).
            return {
                canonical[n]: (n.op, str(n.target) if n.op != "placeholder" else "")
                for n in graph.nodes
            }

        self.assertEqual(structure(graph1, canonical1), structure(graph2, canonical2))

    def test_canonical_names_different_models(self):
        # [FUNCTION DOC] Checks the canonical naming helper yields different structures for graphs
        # of two different models (second applies relu after the linear layer).
        from torch._functorch.partitioners import _canonical_node_names

        class ModelA(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a single Linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        class ModelB(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying a Linear layer followed by relu.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer then relu.
                return self.linear(x).relu()

        inp = torch.randn(4, 8)
        backend1 = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(ModelA(), backend=backend1)(inp)
        torch._dynamo.reset()
        backend2 = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(ModelB(), backend=backend2)(inp)

        c1 = _canonical_node_names(backend1.graphs[0].graph)
        c2 = _canonical_node_names(backend2.graphs[0].graph)
        structure1 = {
            c1[n]: (n.op, str(n.target)) for n in backend1.graphs[0].graph.nodes
        }
        structure2 = {
            c2[n]: (n.op, str(n.target)) for n in backend2.graphs[0].graph.nodes
        }
        self.assertNotEqual(structure1, structure2)

    def test_stable_target_str(self):
        # [FUNCTION DOC] Checks the stable target string helper: plain Python functions and op
        # overloads give qualified names with no memory address, non-callable targets fall back to
        # plain string conversion, and callables without a resolvable qualified name also fall
        # back to string conversion.
        # A call_function whose target is a plain Python function (e.g.
        # torch.sym_not) must stringify to its qualified name, not its repr,
        # which bakes in the per-process memory address and so differs across
        # ranks (poisoning the cross-rank node_str hash).
        from torch._functorch.partitioners import _stable_target_str

        # Plain Python function: qualified name, no memory address.
        self.assertEqual(_stable_target_str(torch.sym_not), "torch.sym_not")
        self.assertNotIn("0x", _stable_target_str(torch.sym_not))
        self.assertIn("0x", str(torch.sym_not))  # the behavior being fixed

        # OpOverload targets: stable qualified name (matches the FX printer),
        # never an address.
        from torch.fx.node import _get_qualified_name

        add = torch.ops.aten.add.Tensor
        self.assertEqual(_stable_target_str(add), _get_qualified_name(add))
        self.assertNotIn("0x", _stable_target_str(add))

        # Non-callable targets (e.g. get_attr names) fall through to str().
        self.assertEqual(_stable_target_str("_param_constant0"), "_param_constant0")

        # Callables _get_qualified_name cannot resolve fall back to str().
        class NoName:
            # [CLASS DOC] HELPER: Callable object without a qualified name.
            def __call__(self):
                # [FUNCTION DOC] HELPER: Returns none.
                return None

        obj = NoName()
        self.assertEqual(_stable_target_str(obj), str(obj))

    def test_canonical_node_str_invariant_to_function_target_address(self):
        # [FUNCTION DOC] Regression test: two distinct functions with the same qualified name but
        # different reprs (simulating two ranks) must hash differently with plain string
        # conversion of the target, but identically when using the stable target string, in a node
        # hash built from canonical names.
        # Regression test: two ranks tracing the same graph reference the same
        # torch.sym_not function, but the function object lives at a different
        # memory address in each process. The old code stringified the target
        # with str(), baking that address into the cross-rank hash, so
        # structurally identical graphs looked different and the sync was
        # skipped. Simulate two ranks with two distinct function objects that
        # share a qualified name but differ in repr (address).
        import hashlib

        from torch._functorch.partitioners import (
            _canonical_node_names,
            _stable_target_str,
        )

        def _make_sym_not():
            # [FUNCTION DOC] HELPER: Returns a fresh inner function named like the symbolic
            # not function.
            def sym_not(x):
                # [FUNCTION DOC] HELPER: Returns logical not of its argument.
                return not x

            return sym_not

        fn_rank0 = _make_sym_not()
        fn_rank1 = _make_sym_not()
        # Same qualified name, different repr (mimics differing addresses).
        self.assertNotEqual(str(fn_rank0), str(fn_rank1))
        self.assertEqual(_stable_target_str(fn_rank0), _stable_target_str(fn_rank1))

        def build_graph(fn):
            # [FUNCTION DOC] HELPER: Builds an FX graph with one placeholder, one call_function
            # node calling the given function, and an output, with meta values set.
            g = torch.fx.Graph()
            p = g.placeholder("x")
            p.meta["val"] = torch.tensor(True)
            n = g.create_node("call_function", fn, (p,))
            n.meta["val"] = torch.tensor(False)
            g.output((n,))
            return g

        graph0 = build_graph(fn_rank0)
        graph1 = build_graph(fn_rank1)

        # Reconstruct the node_str hash the way _sync_decision_cross_ranks does.
        def node_str(graph, stringify):
            # [FUNCTION DOC] HELPER: Returns a SHA-256 digest of a sorted-by-canonical-name list
            # of node descriptions using the supplied function to turn call targets into strings.
            canonical = _canonical_node_names(graph)

            def hash_str(n):
                # [FUNCTION DOC] HELPER: Describes one node as canonical name and op (plus target
                # string for non-placeholders).
                if n.op == "placeholder":
                    return f"{canonical[n]}:{n.op}"
                return f"{canonical[n]}:{n.op}:{stringify(n.target)}"

            joined = "/".join(
                hash_str(n) for n in sorted(graph.nodes, key=lambda n: canonical[n])
            )
            return hashlib.sha256(joined.encode("utf-8")).hexdigest()

        # Old behavior: str(target) bakes in the address -> hashes diverge.
        self.assertNotEqual(node_str(graph0, str), node_str(graph1, str))
        # Fixed behavior: _stable_target_str -> identical hashes.
        self.assertEqual(
            node_str(graph0, _stable_target_str),
            node_str(graph1, _stable_target_str),
        )

    def _get_graph_node_names(self, model, inp):
        # [FUNCTION DOC] HELPER: Compiles the model on the input with the recording eager backend
        # and returns the names of the nodes of the first recorded graph.
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(model, backend=backend)(inp)
        return [n.name for n in backend.graphs[0].graph.nodes]

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph(self):
        # [FUNCTION DOC] With canonicalization enabled, checks that node names of the recorded
        # graph are identical for inputs with different dict key orders.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with deep and shallow branches applied per dict item
            # by key.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the deep and shallow submodules.
                super().__init__()
                self.deep = torch.nn.Sequential(
                    torch.nn.Linear(8, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 16),
                )
                self.shallow = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Applies deep for key a, shallow otherwise, concatenates
                # and sums.
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.deep(val))
                    else:
                        results.append(self.shallow(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": torch.randn(4, 8), "a": torch.randn(4, 8)}

        names1 = self._get_graph_node_names(model, d1)
        torch._dynamo.reset()
        names2 = self._get_graph_node_names(model, d2)
        self.assertEqual(names1, names2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_correctness(self):
        # [FUNCTION DOC] With canonicalization enabled, checks compiled results match eager for
        # two dict key orders.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers applied to dict entries a and b.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the two linear layers.
                super().__init__()
                self.linear_a = torch.nn.Linear(8, 16)
                self.linear_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Returns the sum of the two layers applied to
                # their entries.
                return self.linear_a(d["a"]) + self.linear_b(d["b"])

        model = Model()

        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        eager_result = model(d1)
        compiled_result1 = torch.compile(model, backend="eager")(d1)
        torch._dynamo.reset()
        compiled_result2 = torch.compile(model, backend="eager")(d2)

        self.assertEqual(eager_result, compiled_result1)
        self.assertEqual(eager_result, compiled_result2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_aot_eager(self):
        # [FUNCTION DOC] With canonicalization enabled, checks that under AOT eager both forward
        # and backward graph node names, and the loss, are identical for two dict key orders.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two projections applied per dict item by key.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the two projection layers.
                super().__init__()
                self.proj_a = torch.nn.Linear(8, 16)
                self.proj_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Applies projection a for key a else b, concatenates
                # and sums.
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.proj_a(val))
                    else:
                        results.append(self.proj_b(val))
                return torch.cat(results, dim=-1).sum()

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": d1["b"], "a": d1["a"]}

        backend1 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        loss1 = torch.compile(model, backend=backend1)(d1)
        loss1.backward()

        torch._dynamo.reset()
        model.zero_grad()

        backend2 = torch._dynamo.testing.AotEagerAndRecordGraphs()
        loss2 = torch.compile(model, backend=backend2)(d2)
        loss2.backward()

        self.assertEqual(loss1, loss2)

        fw_names1 = [n.name for n in backend1.fw_graphs[0].graph.nodes]
        fw_names2 = [n.name for n in backend2.fw_graphs[0].graph.nodes]
        self.assertEqual(fw_names1, fw_names2)
        bw_names1 = [n.name for n in backend1.bw_graphs[0].graph.nodes]
        bw_names2 = [n.name for n in backend2.bw_graphs[0].graph.nodes]
        self.assertEqual(bw_names1, bw_names2)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_dict_order_canonical_graph_idempotent(self):
        # [FUNCTION DOC] Checks that running the graph canonicalization helper a second time on an
        # already canonicalized recorded graph does not change the node names.
        from torch._dynamo.output_graph import _canonicalize_graph

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying one linear layer to two dict entries and adding.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Returns the sum of the layer applied to entries a and b.
                return self.linear(d["a"]) + self.linear(d["b"])

        model = Model()
        d = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        torch.compile(model, backend=backend)(d)
        graph = backend.graphs[0].graph

        names_once = [n.name for n in graph.nodes]
        _canonicalize_graph(graph)
        names_twice = [n.name for n in graph.nodes]
        self.assertEqual(names_once, names_twice)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_overlapping_unsqueeze_with_mutation(self):
        # [FUNCTION DOC] With canonicalization enabled, checks that two in-place updates on inputs
        # that are views of one base (dynamic shapes, aot_eager) give the eager result.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds one in place to both inputs and returns the first.
            x.add_(1)
            y.add_(1)
            return x

        base = torch.ones(10)
        inputs = [base.unsqueeze(0), base.unsqueeze(0)]
        out_eager = f(*inputs)

        optf = torch.compile(backend="aot_eager", dynamic=True)(f)
        base = torch.ones(10)
        inputs = [base.unsqueeze(0), base.unsqueeze(0)]
        out_compiled = optf(*inputs)

        self.assertEqual(out_eager, out_compiled)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_barrier_preserves_order(self):
        # [FUNCTION DOC] Checks that canonicalizing a hand-built FX graph keeps an operation
        # placed after an in-place method call (a barrier) after that call.
        from torch._dynamo.output_graph import _canonicalize_graph

        graph = fx.Graph()
        x = graph.placeholder("x")
        # Pure ops that could be reordered
        a = graph.call_function(torch.relu, (x,))
        b = graph.call_function(torch.sigmoid, (x,))
        # In-place call_method acts as barrier
        barrier = graph.call_method("add_", (x, a))
        # Pure op after barrier
        c = graph.call_function(torch.neg, (x,))
        graph.output((a, b, barrier, c))

        _canonicalize_graph(graph)
        ops = [n.name for n in graph.nodes if n.op in ("call_function", "call_method")]
        barrier_idx = next(i for i, name in enumerate(ops) if "add_" in name)
        neg_idx = next(i for i, name in enumerate(ops) if "neg" in name)
        # neg must come after the barrier even though it only depends on x
        self.assertGreater(neg_idx, barrier_idx)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_in_place_ops_are_barriers(self):
        # [FUNCTION DOC] Checks that in a compiled graph with an in-place add, multiplications
        # before and after it stay on their respective sides (at least one each).
        def f(x, y):
            # [FUNCTION DOC] HELPER: Multiplies y twice with an in-place add on x between them.
            a = y * 2
            x.add_(1)
            b = y * 3
            return a + b + x

        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        out = torch.compile(f, backend=backend)(torch.randn(4), torch.randn(4))
        out_eager = f(torch.randn(4), torch.randn(4))
        self.assertEqual(out.shape, out_eager.shape)

        graph = backend.graphs[0].graph
        ops = [
            (n.name, n.op)
            for n in graph.nodes
            if n.op in ("call_function", "call_method")
        ]
        barrier_idx = next(
            i
            for i, (name, op) in enumerate(ops)
            if op == "call_method" and "add_" in name
        )
        mul_before = [
            i
            for i, (name, _) in enumerate(ops)
            if name.startswith("mul") and i < barrier_idx
        ]
        mul_after = [
            i
            for i, (name, _) in enumerate(ops)
            if name.startswith("mul") and i > barrier_idx
        ]
        self.assertTrue(len(mul_before) >= 1)
        self.assertTrue(len(mul_after) >= 1)

    def test_canonical_graph_config_gating(self):
        # [FUNCTION DOC] Checks the config gating: with canonicalization off, graphs for different
        # dict key orders have different node names, and with it on the names are equal.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two linear layers applied per dict item by key.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Builds the two linear layers.
                super().__init__()
                self.linear_a = torch.nn.Linear(8, 16)
                self.linear_b = torch.nn.Linear(8, 16)

            def forward(self, d):
                # [FUNCTION DOC] HELPER: Applies layer a for key a else b and concatenates
                # the outputs.
                results = []
                for key, val in d.items():
                    if key == "a":
                        results.append(self.linear_a(val))
                    else:
                        results.append(self.linear_b(val))
                return torch.cat(results, dim=-1)

        model = Model()
        d1 = {"a": torch.randn(4, 8), "b": torch.randn(4, 8)}
        d2 = {"b": torch.randn(4, 8), "a": torch.randn(4, 8)}

        with torch._dynamo.config.patch(canonicalize_output_graph_node_order=False):
            names1 = self._get_graph_node_names(model, d1)
            torch._dynamo.reset()
            names2 = self._get_graph_node_names(model, d2)
        self.assertNotEqual(names1, names2)

        torch._dynamo.reset()
        with torch._dynamo.config.patch(canonicalize_output_graph_node_order=True):
            names3 = self._get_graph_node_names(model, d1)
            torch._dynamo.reset()
            names4 = self._get_graph_node_names(model, d2)
        self.assertEqual(names3, names4)

    @torch._dynamo.config.patch(canonicalize_output_graph_node_order=True)
    def test_canonical_graph_is_safe_to_reorder(self):
        # [FUNCTION DOC] Checks the reorder-safety predicate on hand-built FX nodes: pure calls
        # and pure methods are safe; in-place methods, in-place operators, calls with an out
        # argument, calls with no node inputs, and the batch dimension add/remove helpers are not
        # safe; pure operators whose names start with i are safe.
        import operator

        from torch.fx.passes.canonicalize import _is_safe_to_reorder

        graph = fx.Graph()
        x = graph.placeholder("x")

        pure_call = graph.call_function(torch.relu, (x,))
        self.assertTrue(_is_safe_to_reorder(pure_call))

        inplace_method = graph.call_method("add_", (x, x))
        self.assertFalse(_is_safe_to_reorder(inplace_method))

        safe_method = graph.call_method("add", (x, x))
        self.assertTrue(_is_safe_to_reorder(safe_method))

        iadd_node = graph.call_function(operator.iadd, (x, x))
        self.assertFalse(_is_safe_to_reorder(iadd_node))

        # operator.invert is pure despite starting with "i"
        invert_node = graph.call_function(operator.invert, (x,))
        self.assertTrue(_is_safe_to_reorder(invert_node))

        index_node = graph.call_function(operator.index, (x,))
        self.assertTrue(_is_safe_to_reorder(index_node))

        # out= kwarg makes a node unsafe
        out_node = graph.call_function(torch.add, (x, x), {"out": x})
        self.assertFalse(_is_safe_to_reorder(out_node))

        # Functions with no FX Node arguments are treated as barriers
        def _fake_state_fn():
            # [FUNCTION DOC] HELPER: Does nothing; used as a call target without graph inputs.
            pass

        no_input_node = graph.call_function(_fake_state_fn, ())
        self.assertFalse(_is_safe_to_reorder(no_input_node))

        # _add_batch_dim / _remove_batch_dim are barriers
        add_batch = graph.call_function(torch._add_batch_dim, (x, x, x))
        self.assertFalse(_is_safe_to_reorder(add_batch))

        remove_batch = graph.call_function(torch._remove_batch_dim, (x, x, x, x))
        self.assertFalse(_is_safe_to_reorder(remove_batch))


instantiate_parametrized_tests(DictTests)


class DictGuardTests(LoggingTestCase):
    # [CLASS DOC] Logging-based tests that check the recompile reason messages produced by dict
    # guards for popitem, comparison and union operations. The dict type under test is the class
    # attribute thetype (dict here), which subclasses can override.
    thetype = dict

    @make_logging_test(recompiles=True)
    def test_popitem(self, records):
        # [FUNCTION DOC] Checks popitem on a global dict: a first call produces no recompile log,
        # changing the value of the popped key triggers one recompile whose log mentions a guard
        # on the popped key value.
        d = self.thetype()
        d[1] = 2
        d[3] = 4

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Pops an item and returns sin if it is the pair 3 and 4,
            # else cos.
            k, v = d.popitem()
            if k == 3 and v == 4:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        y = fn(x)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        d[3] = 5
        y = fn(x)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d")
        self.assertIn(
            """d[3] == 4""",
            munge_exc(record),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_eq(self, records):
        # [FUNCTION DOC] Checks equality of two dict inputs: different dicts then identical dicts
        # causes a recompile whose message mentions the dict-contains guard on the second dict.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            # [FUNCTION DOC] HELPER: Returns sin if the dicts compare equal else cos.
            if d1 == d2:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.cos())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.sin())
        record = self.getRecord(records, "d2")
        self.assertIn(
            "___dict_contains",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_ne(self, records):
        # [FUNCTION DOC] Same flow as the equality test (the compiled function also tests equality
        # rather than inequality); checks the recompile message mentions the dict-contains guard.
        # KNOWN ISSUE: name says not-equal but the code compares for equality, duplicating the
        # eq test.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            # [FUNCTION DOC] HELPER: Returns sin if the dicts compare equal else cos.
            if d1 == d2:
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.cos())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.sin())
        record = self.getRecord(records, "d2")
        self.assertIn(
            "___dict_contains",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_or(self, records):
        # [FUNCTION DOC] Checks union of two input dicts followed by a lookup: a recompile occurs
        # when the second dict lacks key 5, and its log reports a KeyError on that dict.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            # [FUNCTION DOC] HELPER: Merges the dicts and returns sin if key 5 has a truthy value,
            # else cos.
            d = d1 | d2
            if d.get(5, False):
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        y = fn(x, d1, d1)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d2")
        self.assertIn(
            """KeyError on d2[5]""",
            munge_exc(record.getMessage()),
        )

    @make_logging_test(recompiles=True)
    def test_cmp_ior(self, records):
        # [FUNCTION DOC] Checks in-place union into an input dict: calling with different
        # arguments causes a recompile whose log reports a KeyError on the first dict for key 3.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, d1, d2):
            # [FUNCTION DOC] HELPER: Updates d2 in place with d1, returns sin if key 3 is truthy
            # else cos.
            d2 |= d1
            if d2.get(3, False):
                return x.sin()
            return x.cos()

        x = torch.tensor(1.0)
        d1 = self.thetype({1: 2, 3: 4})
        d2 = self.thetype({1: 2, 5: 6})
        d3, d4 = d2.copy(), d2.copy()
        y = fn(x, d1, d2)
        # sanity check
        self.assertEqual(len(records), 0)
        self.assertEqual(y, x.sin())

        y = fn(x, d3, d4)
        self.assertEqual(len(records), 1)
        self.assertEqual(y, x.cos())
        record = self.getRecord(records, "d1")
        self.assertIn(
            """KeyError on d1[3]""",
            munge_exc(record.getMessage()),
        )


class DictMethodsTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests dict methods and operators (comparison, union, and more below) written as
    # traced-by-dynamo test bodies using the make_dynamo_test decorator. The dict class under test
    # is thetype (dict), overridable by subclasses. setUp enables tracing of unittest and tearDown
    # restores it; assertEqual and assertNotEqual are overridden to compare by the equality
    # operator with a lazily built message.
    thetype = dict

    # Methods:
    # + bool
    # + clear
    # + copy
    # + fromkeys
    # + get
    # + items
    # + keys
    # + pop
    # + popitem
    # + setdefault
    # + update
    # + values
    # BinOps:
    # ==, !=, |

    def setUp(self):
        # [FUNCTION DOC] Saves the current value of the enable_trace_unittest dynamo config in an
        # attribute, turns the config on, then calls the parent setUp.
        self._prev_trace_unittest = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved enable_trace_unittest config value and returns the
        # result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self._prev_trace_unittest
        return super().tearDown()

    def assertEqual(self, x, y):
        # [FUNCTION DOC] Override: asserts the two values are equal using the equality operator
        # (not the tensor-aware base class check); the failure message gets both values appended.
        self.assertTrue(x == y, lambda msg: f"{msg}\nExpected {x} to be equal to {y}")

    def assertNotEqual(self, x, y):
        # [FUNCTION DOC] Override: asserts the two values are not equal using the equality
        # operator; the failure message gets both values appended.
        self.assertFalse(
            x == y, lambda msg: f"{msg}\nExpected {x} to not be equal to {y}"
        )

    @make_dynamo_test
    def test_dict_items_cmp_value_eq_raises(self):
        # [FUNCTION DOC] Checks that when a stored dict value's equality hook raises, comparing
        # item views with equal, not-equal, less-or-equal and greater-or-equal propagates the
        # exception, as do strict less-than and greater-than for a proper subset view.
        # dictitems_contains calls PyObject_RichCompareBool(found, value, Py_EQ)
        # on the stored value, so a value whose __eq__ raises must propagate that
        # exception. Equal-length views push the comparison past
        # dictview_richcompare's length short-circuit for eq/ne/le/ge.
        d1 = self.thetype({1: _BadCmp()})
        d2 = self.thetype({1: _BadCmp()})
        for op in (operator.eq, operator.ne, operator.le, operator.ge):
            with self.assertRaises(_BadCmpExc):
                op(d1.items(), d2.items())
        # lt/gt reach the value comparison only on a proper-subset length match.
        d3 = self.thetype({1: _BadCmp(), 2: _BadCmp()})
        with self.assertRaises(_BadCmpExc):
            d1.items() < d3.items()  # noqa: B015
        with self.assertRaises(_BadCmpExc):
            d3.items() > d1.items()  # noqa: B015

    @make_dynamo_test
    def test_dict_items_cmp_value_present_absent(self):
        # [FUNCTION DOC] Checks item view comparisons (equal, subset, proper subset, superset)
        # give the expected truth values for ordinary values.
        # Normal value comparison: matching/non-matching stored values resolve
        # membership without raising, matching eager dict_items semantics.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2, "c": 3})
        d3 = self.thetype({"a": 1, "b": 99})
        self.assertEqual(d1.items() == d2.items(), False)
        self.assertEqual(d1.items() <= d2.items(), True)
        self.assertEqual(d1.items() < d2.items(), True)
        self.assertEqual(d1.items() == d3.items(), False)
        self.assertEqual(d1.items() <= d3.items(), False)
        self.assertEqual(d2.items() >= d1.items(), True)

    @make_dynamo_test
    def test_cmp_eq_key_raises(self):
        # [FUNCTION DOC] Checks that comparing dicts raises the key equality exception when a
        # key's equality hook raises.
        # A key whose __eq__ raises must propagate that exception during dict
        # comparison, mirroring CPython's PyObject_RichCompareBool lookup.
        d1 = self.thetype({_BadCmp(): 1})
        d2 = self.thetype({1: 1})
        with self.assertRaises(_BadCmpExc):
            d1 == d2  # noqa: B015

    @make_dynamo_test
    def test_cmp_eq(self):
        # [FUNCTION DOC] Checks dict equality through the operator, the dunder method, and the
        # unbound class method, for equal and unequal dicts.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2})
        d3 = self.thetype({"a": 1, "b": 3})
        self.assertEqual(d1, d2)
        self.assertNotEqual(d1, d3)

        # Test the == operator
        self.assertEqual(d1 == d2, True)
        self.assertEqual(d1 == d3, False)

        # Test the __eq__ method
        self.assertEqual(d1.__eq__(d2), True)
        self.assertEqual(d1.__eq__(d3), False)

        # Test Dict.__eq__
        self.assertEqual(dict.__eq__(d1, d2), True)
        self.assertEqual(self.thetype.__eq__(d1, d3), False)

    @make_dynamo_test
    def test_cmp_ne(self):
        # [FUNCTION DOC] Checks dict inequality through the operator, the dunder method, and the
        # unbound class method, for equal and unequal dicts.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"a": 1, "b": 2})
        d3 = self.thetype({"a": 1, "b": 3})
        self.assertNotEqual(d1, d3)
        self.assertEqual(d1, d2)

        # Test the != operator
        self.assertEqual(d1 != d3, True)
        self.assertEqual(d1 != d2, False)

        # Test the __ne__ method
        self.assertEqual(d1.__ne__(d3), True)
        self.assertEqual(d1.__ne__(d2), False)

        # Test Dict.__ne__
        self.assertEqual(dict.__ne__(d1, d3), True)
        self.assertEqual(self.thetype.__ne__(d1, d2), False)

    @make_dynamo_test
    def test_binop_or(self):
        # [FUNCTION DOC] Checks the dict union through the operator, dunder method and unbound
        # class method; right operand values win for overlapping keys.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test the | operator
        self.assertEqual(d1 | d2, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d2 | d1, {"a": 1, "b": 2, "c": 4})

        # Test the __or__ method
        self.assertEqual(d1.__or__(d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d2.__or__(d1), {"a": 1, "b": 2, "c": 4})

        # Test Dict.__or__
        self.assertEqual(dict.__or__(d1, d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(self.thetype.__or__(d2, d1), {"a": 1, "b": 2, "c": 4})

        # Test with non-dict types
        self.assertRaises(TypeError, lambda: d1 | 1)

    @make_dynamo_test
    def test_binop_ior(self):
        # [FUNCTION DOC] Checks in-place dict union through the operator, the dunder method and
        # the unbound class method on copies, including the return value, and that a non-dict
        # operand raises TypeError.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test the |= operator
        d3, d4 = d1.copy(), d2.copy()
        d3 |= d2
        d4 |= d1
        self.assertEqual(d3, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4, {"a": 1, "b": 2, "c": 4})

        # Test the __ior__ method
        d3, d4 = d1.copy(), d2.copy()
        d3.__ior__(d2)
        d4.__ior__(d1)
        self.assertEqual(d3, {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4, {"a": 1, "b": 2, "c": 4})

        # Test Dict.__or__
        d3, d4 = d1.copy(), d2.copy()
        self.assertEqual(dict.__ior__(d3, d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(self.thetype.__ior__(d4, d1), {"a": 1, "b": 2, "c": 4})

        # Test return value
        d3, d4 = d1.copy(), d2.copy()
        self.assertEqual(d3.__ior__(d2), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(dict.__ior__(d4, d1), {"a": 1, "b": 2, "c": 4})

        # Test with non-dict types
        self.assertRaises(TypeError, lambda: dict.__ior__(d1, 1))

    @make_dynamo_test
    def test_binop_ior_iterable(self):
        # [FUNCTION DOC] Checks in-place union accepts an items view and a generator of pairs.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})
        d3, d4 = d1.copy(), d2.copy()

        def fn(d):
            # [FUNCTION DOC] HELPER: Generator yielding the items of the given dict.
            yield from d.items()

        self.assertEqual(d3.__ior__(d2.items()), {"a": 1, "b": 3, "c": 4})
        self.assertEqual(d4.__ior__(fn(d1)), {"a": 1, "b": 2, "c": 4})

    @make_dynamo_test
    def test_clear(self):
        # [FUNCTION DOC] Checks clear empties the dict (method, class method), returns none, and
        # raises TypeError when given an extra argument.
        d = self.thetype({"a": 1, "b": 2})
        d.clear()
        self.assertEqual(d, {})

        # Test that clear returns None
        d = self.thetype({"a": 1, "b": 2})
        self.assertIsNone(d.clear())

        # Test Dict.clear
        d = self.thetype({"a": 1, "b": 2})
        dict.clear(d)
        self.assertEqual(d, {})

        d = self.thetype({"a": 1, "b": 2})
        self.thetype.clear(d)
        self.assertEqual(d, {})

        # Test invalid usage
        self.assertRaises(TypeError, d.clear, 1)

    @make_dynamo_test
    def test_copy(self):
        # [FUNCTION DOC] Checks copy returns an equal but distinct object (method and class
        # method) and rejects extra arguments with TypeError.
        d = self.thetype({"a": 1, "b": 2})
        d2 = d.copy()
        self.assertEqual(d, d2)

        # Test that copy returns a new instance
        self.assertIsNot(d, d2)

        # Test Dict.copy
        self.assertEqual(dict.copy(d), d2)
        self.assertEqual(self.thetype.copy(d), d2)

        # Test invalid usage
        self.assertRaises(TypeError, d.copy, 1)

    @make_dynamo_test
    def test_fromkeys(self):
        # [FUNCTION DOC] Checks the fromkeys class method with values 1 and none, and TypeError
        # for missing or non-iterable arguments.
        d = self.thetype.fromkeys(["a", "b"], 1)
        self.assertEqual(d, {"a": 1, "b": 1})
        p = self.thetype.fromkeys(["a", "b"], None)
        self.assertEqual(p, {"a": None, "b": None})

        # Test Dict.fromkeys
        d2 = self.thetype.fromkeys(["c", "d"], 2)
        self.assertEqual(d2, {"c": 2, "d": 2})

        # Test invalid usage
        self.assertRaises(TypeError, self.thetype.fromkeys)
        self.assertRaises(TypeError, self.thetype.fromkeys, 1, 2)

    @make_dynamo_test
    def test_get(self):
        # [FUNCTION DOC] Checks get with and without a default (method and class method) and that
        # calling it with no arguments raises TypeError.
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.get("a"), 1)
        self.assertEqual(d.get("c", 3), 3)
        self.assertIsNone(d.get("c"))

        # Test Dict.get
        self.assertEqual(dict.get(d, "b"), 2)
        self.assertEqual(self.thetype.get(d, "b"), 2)

        # Test invalid usage
        self.assertRaises(TypeError, d.get)

    @make_dynamo_test
    def test_items(self):
        # [FUNCTION DOC] Checks the items view contents (method and class method) and TypeError on
        # an extra argument.
        d = self.thetype({"a": 1, "b": 2})
        items = d.items()
        self.assertEqual(set(items), {("a", 1), ("b", 2)})

        # Test Dict.items
        self.assertEqual(set(dict.items(d)), {("a", 1), ("b", 2)})
        self.assertEqual(set(self.thetype.items(d)), {("a", 1), ("b", 2)})

        # Test invalid usage
        self.assertRaises(TypeError, d.items, 1)

    @make_dynamo_test
    def test_keys(self):
        # [FUNCTION DOC] Checks the keys view contents (method and class method) and TypeError on
        # an extra argument.
        d = self.thetype({"a": 1, "b": 2})
        keys = d.keys()
        self.assertEqual(set(keys), {"a", "b"})

        # Test Dict.keys
        self.assertEqual(set(dict.keys(d)), {"a", "b"})
        self.assertEqual(set(self.thetype.keys(d)), {"a", "b"})

        # Test invalid usage
        self.assertRaises(TypeError, d.keys, 1)

    @make_dynamo_test
    def test_pop(self):
        # [FUNCTION DOC] Checks pop removing keys, the default argument, class method forms,
        # KeyError for a missing key without default, and TypeError with no arguments.
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.pop("a"), 1)
        self.assertEqual(d, {"b": 2})
        self.assertIsNone(d.pop("c", None))

        # Test Dict.pop
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(dict.pop(d, "b"), 2)
        self.assertEqual(self.thetype.pop(d, "a"), 1)

        # Test invalid usage
        self.assertRaises(KeyError, d.pop, "c")
        self.assertRaises(TypeError, d.pop)

    @make_dynamo_test
    def test_popitem(self):
        # [FUNCTION DOC] Checks popitem returns the last inserted pair, empties the dict (method
        # and class forms), and rejects a positional argument except for OrderedDict.
        d = self.thetype({"a": 1})
        key, value = d.popitem()
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)
        self.assertEqual(len(d), 0)
        # check LIFO
        d = self.thetype()
        d["a"] = 1
        d["b"] = 2
        self.assertEqual(d.popitem(), ("b", 2))

        # Test Dict.popitem
        d = self.thetype({"a": 1})
        key, value = dict.popitem(d)
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)

        d = self.thetype({"a": 1})
        key, value = self.thetype.popitem(d)
        self.assertEqual(key, "a")
        self.assertEqual(value, 1)

        # Test invalid usage
        if self.thetype is not OrderedDict:
            # OrderedDict accepts a keyword arg
            self.assertRaises(TypeError, d.popitem, 1)

    @make_dynamo_test
    def test_setdefault(self):
        # [FUNCTION DOC] Checks setdefault returns existing or inserted values, inserts none for a
        # missing default, supports class method forms, and raises TypeError for no arguments or
        # an unhashable key.
        d = self.thetype({"a": 1, "b": 2})
        self.assertEqual(d.setdefault("a", 3), 1)
        self.assertEqual(d.setdefault("c", 3), 3)
        self.assertIsNone(d.setdefault("d"), None)
        self.assertEqual(d, {"a": 1, "b": 2, "c": 3, "d": None})

        # Test Dict.setdefault
        self.assertEqual(dict.setdefault(d, "f", 5), 5)
        self.assertEqual(self.thetype.setdefault(d, "e", 5), 5)

        # Test invalid usage
        self.assertRaises(TypeError, d.setdefault)
        self.assertRaises(TypeError, d.setdefault, [[]])

    @make_dynamo_test
    def test_update(self):
        # [FUNCTION DOC] Checks update from a dict, from another dict instance, via the class
        # method, with keyword arguments, and that a non-iterable argument raises TypeError.
        d = self.thetype({"a": 1, "b": 2})
        d.update({"b": 3, "c": 4})
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4})

        # Test with another dict
        d2 = self.thetype({"d": 5})
        d.update(d2)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5})

        # Test Dict.update
        d3 = self.thetype({"e": 6})
        dict.update(d, d3)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6})
        d4 = self.thetype({"f": 7})
        self.thetype.update(d, d4)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7})

        # Test with keyword arguments
        d.update(f=7, g=8)
        self.assertEqual(d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8})

        # Test Dict.update with keyword arguments
        self.thetype.update(d, h=9, i=10)
        self.assertEqual(
            d, {"a": 1, "b": 3, "c": 4, "d": 5, "e": 6, "f": 7, "g": 8, "h": 9, "i": 10}
        )

        # Test invalid usage
        self.assertRaises(TypeError, d.update, 1)

    @make_dynamo_test
    def test_values(self):
        # [FUNCTION DOC] Checks the values view contents (method and class method) and TypeError
        # on an extra argument.
        d = self.thetype({"a": 1, "b": 2})
        values = d.values()
        self.assertEqual(set(values), {1, 2})

        # Test Dict.values
        self.assertEqual(set(dict.values(d)), {1, 2})
        self.assertEqual(set(self.thetype.values(d)), {1, 2})

        # Test invalid usage
        self.assertRaises(TypeError, d.values, 1)

    @make_dynamo_test
    def test_keys_contains(self):
        # [FUNCTION DOC] Checks membership tests on keys views obtained via method and
        # class method.
        d = self.thetype({"a": 1, "b": 2})
        keys = d.keys()
        self.assertTrue("a" in keys)
        self.assertTrue("b" in keys)
        self.assertFalse("c" in keys)

        # Test Dict.keys contains
        self.assertTrue("a" in dict.keys(d))
        self.assertFalse("c" in dict.keys(d))
        self.assertTrue("b" in self.thetype.keys(d))

    @make_dynamo_test
    def test_items_contains(self):
        # [FUNCTION DOC] Checks membership tests of pairs on items views via method and
        # class method.
        d = self.thetype({"a": 1, "b": 2})
        items = d.items()
        self.assertTrue(("a", 1) in items)
        self.assertTrue(("b", 2) in items)
        self.assertFalse(("a", 2) in items)
        self.assertFalse(("c", 1) in items)

        # Test Dict.items contains
        self.assertTrue(("a", 1) in dict.items(d))
        self.assertFalse(("c", 1) in dict.items(d))
        self.assertTrue(("b", 2) in self.thetype.items(d))

    @make_dynamo_test
    def test_values_contains(self):
        # [FUNCTION DOC] Checks membership tests on values views via method and class method.
        d = self.thetype({"a": 1, "b": 2})
        values = d.values()
        self.assertTrue(1 in values)
        self.assertTrue(2 in values)
        self.assertFalse(3 in values)

        # Test Dict.values contains
        self.assertTrue(1 in dict.values(d))
        self.assertFalse(3 in dict.values(d))
        self.assertTrue(2 in self.thetype.values(d))

    @make_dynamo_test
    def test_type(self):
        # [FUNCTION DOC] Checks a new instance is exactly of the tested dict class.
        d = self.thetype({"a": 1, "b": 2})
        self.assertIsInstance(d, self.thetype)
        self.assertIs(type(d), self.thetype)

    @make_dynamo_test
    def test_dict_type_comparison(self):
        # [FUNCTION DOC] Checks the tested dict class is equal and identical to itself and differs
        # from the other dict types in the list (dict, OrderedDict, defaultdict).
        types = (dict, OrderedDict, defaultdict)
        self.assertEqual(self.thetype, self.thetype)
        self.assertTrue(self.thetype is self.thetype)
        for other in types:
            if self.thetype == other:
                continue
            self.assertNotEqual(self.thetype, other)
            self.assertTrue(
                self.thetype is not other,
                lambda msg: f"{msg}\n{self.thetype=}, {other=}",
            )

    @make_dynamo_test
    def test_dict___iter__(self):
        # [FUNCTION DOC] Checks the iterator of a one-key dict yields that key.
        d = self.thetype({1: 2})
        it = d.__iter__()
        self.assertEqual(next(it), 1)

    def test_functools_partial_key(self):
        # [FUNCTION DOC] Checks a functools partial object as a dict key in compiled code.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return x + y

        def fn(x):
            # [FUNCTION DOC] HELPER: Stores a value under a partial key and reads it back to
            # scale input.
            new_dict = {}
            new_gn1 = partial(gn, x=1)
            new_dict[new_gn1] = 5
            return x * new_dict[new_gn1]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_namedtuple_functools(self):
        # [FUNCTION DOC] Checks a named tuple holding a partial object and an int as a dict key:
        # an equal second tuple finds the same entry.
        class Container(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with a callable field and an int field.
            partial_fn: Callable
            const: int

        def gn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return x + y

        def fn(x):
            # [FUNCTION DOC] HELPER: Stores a value under one container key and reads it with an
            # equal one.
            new_dict = {}

            new_gn = partial(gn, x=1)
            key = Container(new_gn, 4)
            new_dict[key] = 5
            # Make another key that should hash to the same value
            key1 = Container(new_gn, 4)
            return x * new_dict[key1]

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        ref = fn(x)
        res = opt_fn(x)
        self.assertTrue(same(ref, res))

    def test_custom_object_as_dict_key(self):
        # [FUNCTION DOC] Checks that two distinct instances of a user class with default hashing
        # are separate keys; the docstring talks of equal hash and equality, but the class defines
        # neither, so the second assignment does not overwrite the first.
        # KNOWN ISSUE: docstring claims overridden hash and equality, but CustomKey
        # defines neither.
        """Test that custom objects with __hash__ as dict keys are properly handled.

        This test verifies that when using custom objects with overridden __hash__
        and __eq__ as dictionary keys, two instances with the same hash and equality
        should be recognized as the same key.
        """

        class CustomKey:
            # [CLASS DOC] HELPER: Plain object with value and name attributes; no hash or
            # eq overrides.
            def __init__(self, value, name):
                # [FUNCTION DOC] HELPER: Stores value and name.
                self.value = value
                self.name = name

        def fn(x):
            # [FUNCTION DOC] HELPER: Uses two instances as keys and multiplies the two lookups.
            d = {}
            # Create first instance
            key1 = CustomKey(42, "test")
            d[key1] = x * 2

            # Create second instance with same values - should hash to same value
            key2 = CustomKey(42, "test")
            d[key2] = x * 3  # This should overwrite the first value

            return d[key1] * d[key2]

        x = torch.randn(4)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertTrue(same(opt_fn(x), fn(x)))

    def test_user_defined_object(self):
        # [FUNCTION DOC] Checks a user object registering itself in an outer dict at construction
        # time and a compiled function writing into that registry entry.
        class A:
            # [CLASS DOC] HELPER: Object that adds itself as a key of the closure dict REF with an
            # empty dict value and has an empty dict attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets attribute x and registers self in REF.
                self.x = {}
                REF[self] = {}

        REF = {}

        def f(a, x):
            # [FUNCTION DOC] HELPER: Stores the tensor into the registry entry of the object and
            # adds one.
            REF[a]["foo"] = x
            return x + 1

        opt_f = torch.compile(f, backend="eager", fullgraph=True)

        x = torch.randn(4)
        self.assertTrue(same(f(A(), x), opt_f(A(), x)))

    @make_dynamo_test
    def test_bool(self):
        # [FUNCTION DOC] Checks the truthiness of an empty and non-empty dict of the tested type.
        p = self.thetype()
        q = self.thetype({"a": 1, "b": 2})
        if p:
            self.fail("empty mapping must compare to False")
        if not q:
            self.fail("non-empty mapping must compare to True")
        if bool(p):
            self.fail("empty mapping must compare to False")
        if not bool(q):
            self.fail("non-empty mapping must compare to True")


class DictSubclassMethodsTests(DictMethodsTests):
    # [CLASS DOC] Re-runs all the dict method tests with a plain dict subclass (SimpleDict).
    thetype = SimpleDict


class OrderedDictMethodsTests(DictMethodsTests):
    # [CLASS DOC] Re-runs the dict method tests with OrderedDict and adds OrderedDict specific
    # tests: move_to_end, order-sensitive equality, union and in-place union return types, and
    # popitem with the last argument.
    thetype = OrderedDict

    # Methods:
    # - popitem - Inherited from DictMethodsTest
    # + move_to_end

    @make_dynamo_test
    def test_move_to_end(self):
        # [FUNCTION DOC] Checks move_to_end (default, as unbound method, with last false) reorders
        # the keys, and a missing key raises KeyError.
        d = self.thetype.fromkeys("abcde")
        self.assertEqual("".join(d), "abcde")
        d.move_to_end("b")
        self.assertEqual("".join(d), "acdeb")

        # Test OrderedDict.move_to_end
        self.thetype.move_to_end(d, "a")
        self.assertEqual("".join(d), "cdeba")

        # Test last=False
        self.thetype.move_to_end(d, "a", last=False)
        self.assertEqual("".join(d), "acdeb")

        # Test KeyError
        self.assertRaises(KeyError, d.move_to_end, "f")

    def test_cmp_eq_order(self):
        # [FUNCTION DOC] Eager-only check (not traced) that OrderedDicts with the same keys in
        # different order compare unequal.
        a = self.thetype.fromkeys("abc")
        b = self.thetype.fromkeys("bca")
        self.assertFalse(a == b)

    @make_dynamo_test
    def test_binop_or_return_type(self):
        # [FUNCTION DOC] Checks the union of OrderedDicts with each other or with a plain dict
        # yields an OrderedDict.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test return type
        self.assertIs(type(d1 | d2), OrderedDict)
        self.assertIs(type(dict(d1) | d2), OrderedDict)
        self.assertIs(type(d1 | dict(d2)), OrderedDict)

    @make_dynamo_test
    def test_binop_ior_return_type(self):
        # [FUNCTION DOC] Checks the type returned by in-place union: it follows the left operand
        # (OrderedDict stays OrderedDict, plain dict stays dict), even through the unbound method.
        d1 = self.thetype({"a": 1, "b": 2})
        d2 = self.thetype({"b": 3, "c": 4})

        # Test return type
        d3, d4 = d1.copy(), d2.copy()
        self.assertIs(type(d3.__ior__(d2)), OrderedDict)
        self.assertIs(type(dict.__ior__(d4, d2)), OrderedDict)
        self.assertIs(type(self.thetype.__ior__(d4, d2)), OrderedDict)

        d3, d4 = d1.copy(), d2.copy()
        self.assertIs(type(dict.__ior__(d3, dict(d2))), OrderedDict)
        self.assertIs(type(dict.__ior__(dict(d3), d2)), dict)
        self.assertIs(type(dict(d4).__ior__(d2)), dict)

    @make_dynamo_test
    def test_popitem_kwarg(self):
        # [FUNCTION DOC] Checks popitem with last as keyword or positional argument pops from the
        # end or the front as requested.
        d = self.thetype.fromkeys("abcdf")
        self.assertEqual(d.popitem(last=True), ("f", None))
        self.assertEqual(list(d), list("abcd"))
        self.assertEqual(d.popitem(last=False), ("a", None))
        self.assertEqual(list(d), list("bcd"))
        self.assertEqual(d.popitem(False), ("b", None))
        self.assertEqual(list(d), list("cd"))
        self.assertEqual(d.popitem(True), ("d", None))
        self.assertEqual(list(d), list("c"))


class OrderedDictSubclassOverload(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for an OrderedDict subclass overriding get and move_to_end. Like
    # DictMethodsTests it enables unittest tracing in setUp, restores it in tearDown, and
    # overrides the equality assertions. The class attribute thetype is the subclass.
    def setUp(self):
        # [FUNCTION DOC] Saves the current enable_trace_unittest dynamo config, turns it on, then
        # calls the parent setUp.
        self._prev_trace_unittest = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] Restores the saved enable_trace_unittest config value and returns the
        # result of the parent tearDown.
        torch._dynamo.config.enable_trace_unittest = self._prev_trace_unittest
        return super().tearDown()

    def assertEqual(self, x, y):
        # [FUNCTION DOC] Override: asserts equality using the equality operator with a
        # lazy message.
        self.assertTrue(x == y, lambda msg: f"{msg}\nExpected {x} to be equal to {y}")

    def assertNotEqual(self, x, y):
        # [FUNCTION DOC] Override: asserts inequality using the equality operator with a
        # lazy message.
        self.assertFalse(
            x == y, lambda msg: f"{msg}\nExpected {x} to not be equal to {y}"
        )

    class OrderedDictSubclass(OrderedDict):
        # [CLASS DOC] HELPER: OrderedDict subclass whose get always returns the default and whose
        # move_to_end pops the key instead of moving it.
        def get(self, key, default=None, /):
            # [FUNCTION DOC] HELPER: Ignores the dict contents and returns the default.
            return default

        def move_to_end(self, key, last=True, /):
            # [FUNCTION DOC] HELPER: Removes the key from the dict instead of reordering it.
            # change the behavior to something else
            self.pop(key)

    thetype = OrderedDictSubclass

    @make_dynamo_test
    def test_move_to_end(self):
        # [FUNCTION DOC] Checks the overridden move_to_end of the subclass is the one called (key
        # a is removed rather than moved).
        p = self.thetype({"a": 1, "b": 2, "c": 3})
        p.move_to_end("a")
        self.assertEqual(list(p.keys()), list("bc"))


class DunderDictVariableTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for tracing of an object's __dict__ view in compiled code: iteration and
    # views including attribute mutations made during tracing, copy semantics, key order after pop
    # and reinsert, KeyError and default behavior of pop, and non-string keys.
    """Tests for DunderDictVariable (object.__dict__ handling in Dynamo)"""

    def test_dunder_dict_items_includes_mutations(self):
        # [FUNCTION DOC] Checks the items of the object __dict__ after a new attribute is set
        # during tracing include original and new attributes.
        """Test that __dict__.items() includes both original and mutated keys"""

        class MyClass:
            # [CLASS DOC] HELPER: Object with attributes a=1 and b=2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a and b.
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Adds attribute c and returns a dict of the __dict__ items.
            obj.c = 3
            items = dict(obj.__dict__.items())
            return items

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, {"a": 1, "b": 2, "c": 3})

    def test_dunder_dict_keys_includes_mutations(self):
        # [FUNCTION DOC] Checks the keys of __dict__ include attributes added during tracing.
        """Test that __dict__.keys() includes both original and mutated keys"""

        class MyClass:
            # [CLASS DOC] HELPER: Object with attributes a=1 and b=2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a and b.
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Adds attributes c and d and returns the sorted keys.
            obj.c = 3
            obj.d = 4
            keys = sorted(obj.__dict__.keys())
            return keys

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, ["a", "b", "c", "d"])

    def test_dunder_dict_values_includes_mutations(self):
        # [FUNCTION DOC] Checks the values of __dict__ include values of attributes added
        # during tracing.
        """Test that __dict__.values() includes both original and mutated values"""

        class MyClass:
            # [CLASS DOC] HELPER: Object with attributes a=1 and b=2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a and b.
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Adds attribute c and returns the sorted values.
            obj.c = 3
            values = sorted(obj.__dict__.values())
            return values

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, [1, 2, 3])

    def test_dunder_dict_copy_does_not_alias_instance_dict(self):
        # [FUNCTION DOC] Checks that a copy of the __dict__ is independent: mutating the copy does
        # not add the key to the object.
        class MyClass:
            # [CLASS DOC] HELPER: Object with attribute y=1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets y.
                self.y = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Sets a key in a copy of __dict__ and returns whether the
            # object has it plus a dict of the live items.
            obj.__dict__.copy()["x"] = 4
            return "x" in obj.__dict__, dict(obj.__dict__.items())

        obj = MyClass()
        has_x, live_dict = fn(obj)
        self.assertFalse(has_x)
        self.assertEqual(live_dict, {"y": 1})
        self.assertNotIn("x", obj.__dict__)

    def test_dunder_dict_comprehension_with_mutations(self):
        # [FUNCTION DOC] Checks building a dict from the __dict__ items inside a method of an
        # object modified during tracing (scheduler-style state_dict).
        """Test dict comprehension over __dict__ with mutations (scheduler use case)"""

        class SimpleScheduler:
            # [CLASS DOC] HELPER: Scheduler-like object with base_lrs and last_epoch attributes.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets base_lrs to a list with 0.1 and last_epoch to 0.
                self.base_lrs = [0.1]
                self.last_epoch = 0

            def state_dict(self):
                # [FUNCTION DOC] HELPER: Returns a dict copy of the __dict__ items.
                state = dict(self.__dict__.items())
                return state

        @torch.compile(backend="eager", fullgraph=True)
        def fn(scheduler):
            # [FUNCTION DOC] HELPER: Modifies one attribute, adds another, and returns the
            # state dict.
            scheduler.last_epoch = 5
            scheduler.new_attr = "test"
            return scheduler.state_dict()

        scheduler = SimpleScheduler()
        result = fn(scheduler)
        self.assertEqual(result["base_lrs"], [0.1])
        self.assertEqual(result["last_epoch"], 5)
        self.assertEqual(result["new_attr"], "test")

    def test_dunder_dict_multiple_mutations(self):
        # [FUNCTION DOC] Checks repeated attribute assignments during tracing show the last values
        # in __dict__ items and keys.
        """Test __dict__ operations with multiple mutations"""

        class MyClass:
            # [CLASS DOC] HELPER: Object with attribute x=1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets x.
                self.x = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Sets y twice and z, returns the items dict and key set
            # of __dict__.
            obj.y = 2
            obj.z = 3
            obj.y = 20
            items = dict(obj.__dict__.items())
            keys = set(obj.__dict__.keys())
            return items, keys

        obj = MyClass()
        items, keys = fn(obj)
        self.assertEqual(items, {"x": 1, "y": 20, "z": 3})
        self.assertEqual(keys, {"x", "y", "z"})

    def test_dunder_dict_pop_reinsert_order(self):
        # [FUNCTION DOC] Checks that popping and re-adding a key of an object's __dict__ moves it
        # to the end of the key order, as in CPython.
        # CPython appends a re-inserted key at the end of the dict rather than
        # reusing its old slot. obj.__dict__ ordering is observable, so popping
        # then re-adding a key must move it to the end (mirrors
        # test_dict.DictTest.test_splittable_pop).
        class MyClass:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Creates an object with x, y, z, pops y, reinserts it, and
            # returns the key list and value.
            obj = MyClass()
            obj.x, obj.y, obj.z = 1, 2, 3
            d = obj.__dict__
            d.pop("y")
            d["y"] = 42
            return list(d), d["y"]

        keys, value = fn()
        self.assertEqual(keys, ["x", "z", "y"])
        self.assertEqual(value, 42)

    def test_dunder_dict_pop_reinsert_order_with_other_mutations(self):
        # [FUNCTION DOC] Checks key order after pop, update of an existing key, new key and
        # reinsertion: updates keep their place, new keys append, the reinserted key goes last.
        # Mutating other keys after a pop must not disturb their relative order;
        # only the re-inserted popped key moves to the end. Updating an existing
        # key keeps its slot, a brand-new key appends, and the reinserted key
        # lands last.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Applies a sequence of pop, update, add and reinsert
            # operations on the object __dict__ and returns key list and dict.
            obj = MyClass()
            obj.a, obj.b, obj.c = 1, 2, 3
            d = obj.__dict__
            d.pop("b")  # a, c
            d["c"] = 30  # update existing -> stays in place: a, c
            d["e"] = 5  # new key -> appended: a, c, e
            d["b"] = 20  # reinsert popped -> end: a, c, e, b
            return list(d), dict(d)

        keys, value = fn()
        self.assertEqual(keys, ["a", "c", "e", "b"])
        self.assertEqual(value, {"a": 1, "c": 30, "e": 5, "b": 20})

    def test_dunder_dict_pop_missing_raises_keyerror(self):
        # [FUNCTION DOC] Checks popping a missing key from an object __dict__ raises KeyError
        # inside compiled code.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Pops the only key twice, catching KeyError the second time,
            # and returns the remaining keys and a raised flag.
            obj = MyClass()
            obj.x = 1
            d = obj.__dict__
            d.pop("x")
            try:
                d.pop("x")
                raised = False
            except KeyError:
                raised = True
            return list(d), raised

        self.assertEqual(fn(), ([], True))

    def test_dunder_dict_pop_default(self):
        # [FUNCTION DOC] Checks pop on an object __dict__ with a default for a missing key and
        # without default for an existing key.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Pops a missing key with a default and an existing key,
            # returns both plus the remaining keys.
            obj = MyClass()
            obj.x = 1
            d = obj.__dict__
            return d.pop("missing", "fallback"), d.pop("x"), list(d)

        self.assertEqual(fn(), ("fallback", 1, []))

    def test_dunder_dict_items_iteration(self):
        # [FUNCTION DOC] Checks a loop over the __dict__ items sees attributes added
        # during tracing.
        """Test iterating over __dict__.items() with mutations"""

        class MyClass:
            # [CLASS DOC] HELPER: Object with attributes a=1 and b=2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets a and b.
                self.a = 1
                self.b = 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Adds attribute c and returns a sorted list of key and
            # value pairs.
            obj.c = 3
            result = []
            for k, v in obj.__dict__.items():
                result.append((k, v))
            return sorted(result)

        obj = MyClass()
        result = fn(obj)
        self.assertEqual(result, [("a", 1), ("b", 2), ("c", 3)])

    def test_method_dict_direct_fullgraph(self):
        # [FUNCTION DOC] Regression test: accessing __dict__ of a bound method in a fullgraph
        # compiled function must not raise Unsupported.
        # KNOWN ISSUE: the docstring says it should fail, while the test comment says it should
        # not raise.
        """
        Regression test: Accessing __dict__ directly on UserMethodVariable.
        This should fail with: unsupported variable type for __dict__ access
        """

        class Foo:
            # [CLASS DOC] HELPER: Class with a method bar returning 42.
            def bar(self):
                # [FUNCTION DOC] HELPER: Returns 42.
                return 42

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Accesses __dict__ of a bound method and returns it.
            obj = Foo()
            # Access __dict__ on bound method - creates UserMethodVariable
            method = obj.bar
            d = method.__dict__
            return d

        # This should not raise Unsupported
        fn()

    def test_dunder_dict_non_str_key_setitem(self):
        # [FUNCTION DOC] Checks non-string keys can be set in an object __dict__ through the
        # mapping interface while attribute assignment still works, preserving insertion order.
        # CPython's instance __dict__ accepts arbitrary hashable keys when set
        # via the mapping API; only attribute access via setattr requires str.
        # Mirrors test_dict.DictTest.test_object_set_item_single_instance_non_str_key.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Sets key 1 via __dict__ and attribute a, returns dict and
            # key list.
            f = Foo()
            f.__dict__[1] = 1
            f.a = "a"
            return dict(f.__dict__), list(f.__dict__)

        d, keys = fn()
        self.assertEqual(d, {1: 1, "a": "a"})
        self.assertEqual(keys, [1, "a"])

    def test_dunder_dict_non_str_key_roundtrip(self):
        # [FUNCTION DOC] Checks non-string keys in an object __dict__ can be read back
        # and deleted.
        # Non-str instance-dict keys must read back and delete correctly.
        class Foo:
            # [CLASS DOC] HELPER: Empty class.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Sets int, float and tuple keys, reads them, deletes one, and
            # returns the values and remaining dict.
            f = Foo()
            d = f.__dict__
            d[1] = 10
            d[2.5] = 20
            d[(3, 4)] = 30
            got = (d[1], d[2.5], d[(3, 4)])
            del d[2.5]
            return got, dict(d)

        got, d = fn()
        self.assertEqual(got, (10, 20, 30))
        self.assertEqual(d, {1: 10, (3, 4): 30})


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
