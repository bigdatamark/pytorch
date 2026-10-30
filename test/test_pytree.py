# Owner(s): ["module: pytree"]
# [MODULE DOC] Tests for torch.utils._pytree (pure Python) and torch.utils._cxx_pytree (optree
# backed, skipped in fbcode). Defines generic tests run against both implementations, plus
# Python-only and C++-only test cases covering flatten/unflatten, tree_map variants, namedtuple and
# structseq detection, treespec serialization, dataclass and constant registration, and key paths.
# Import-time side effects: builds the pytree_modules mapping, registers a module-level custom type
# on the C++ pytree, and instantiates the parametrized test classes at the bottom of the file.

import copy
import enum
import inspect
import os
import pickle
import re
import subprocess
import sys
import time
import unittest
import warnings
from collections import defaultdict, deque, namedtuple, OrderedDict, UserDict
from dataclasses import dataclass, field
from enum import auto
from typing import Any, NamedTuple

import torch
import torch.utils._pytree as python_pytree
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    parametrize,
    run_tests,
    skipIfTorchDynamo,
    subtest,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


pytree_modules = {
    "python": python_pytree,
}
if not IS_FBCODE:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
else:
    # optree is not yet enabled in fbcode, so just re-test the python implementation
    cxx_pytree = python_pytree


parametrize_pytree_module = parametrize(
    "pytree",
    [subtest(module, name=name) for name, module in pytree_modules.items()],
)


GlobalPoint = namedtuple("GlobalPoint", ["x", "y"])


class GlobalDummyType:
    # [CLASS DOC] HELPER: Plain test fixture class with two attributes, x and y, and value
    # equality and hashing based on the pair. It is registered at import time as a C++ pytree
    # node (serialized name GlobalDummyType) so serialization tests can use a module-level
    # custom type.
    def __init__(self, x, y):
        # [FUNCTION DOC] HELPER: Stores the two constructor arguments as attributes x and y.
        self.x = x
        self.y = y

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality by value. Returns NotImplemented for non-
        # GlobalDummyType operands, otherwise True when both x and y match.
        if not isinstance(other, GlobalDummyType):
            return NotImplemented
        return self.x == other.x and self.y == other.y

    def __hash__(self):
        # [FUNCTION DOC] HELPER: Hash derived from the pair of x and y, consistent with the
        # equality method.
        return hash((self.x, self.y))


cxx_pytree.register_pytree_node(
    GlobalDummyType,
    lambda dummy: ([dummy.x, dummy.y], None),
    lambda xs, _: GlobalDummyType(*xs),
    serialized_type_name="GlobalDummyType",
)


class TestEnum(enum.Enum):
    # [CLASS DOC] HELPER: Single-member enum (member A) used as a dict key or treespec context
    # in serialization tests. Named Test-prefixed but is not a test case.
    A = auto()


class TestGenericPytree(TestCase):
    # [CLASS DOC] Test case holding pytree behavior tests that run against every implementation
    # in pytree_modules (Python and, outside fbcode, C++ via optree). Tests receive the
    # implementation module through the parametrize decorator and are expanded by
    # instantiate_parametrized_tests at the bottom of the file.
    def test_aligned_public_apis(self):
        # [FUNCTION DOC] Checks that the Python and C++ pytree modules expose the same public
        # API. Compares the all lists, then for each name compares class-ness and function-ness,
        # parameter names, positional parameter names, kinds, defaults and annotations.
        # Annotations mentioning TreeSpec are compared after stripping module prefixes from the
        # TreeSpec name. Fails the test on any mismatch; no side effects.
        public_apis = python_pytree.__all__

        self.assertEqual(public_apis, cxx_pytree.__all__)

        for name in public_apis:
            cxx_api = getattr(cxx_pytree, name)
            python_api = getattr(python_pytree, name)

            self.assertEqual(inspect.isclass(cxx_api), inspect.isclass(python_api))
            self.assertEqual(
                inspect.isfunction(cxx_api),
                inspect.isfunction(python_api),
            )
            if inspect.isfunction(cxx_api):
                cxx_signature = inspect.signature(cxx_api)
                python_signature = inspect.signature(python_api)

                # Check the parameter names are the same.
                cxx_param_names = list(cxx_signature.parameters)
                python_param_names = list(python_signature.parameters)
                self.assertEqual(cxx_param_names, python_param_names)

                # Check the positional parameters are the same.
                cxx_positional_param_names = [
                    n
                    for n, p in cxx_signature.parameters.items()
                    if (
                        p.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        }
                    )
                ]
                python_positional_param_names = [
                    n
                    for n, p in python_signature.parameters.items()
                    if (
                        p.kind
                        in {
                            inspect.Parameter.POSITIONAL_ONLY,
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        }
                    )
                ]
                self.assertEqual(
                    cxx_positional_param_names,
                    python_positional_param_names,
                )

                for python_name, python_param in python_signature.parameters.items():
                    self.assertIn(python_name, cxx_signature.parameters)
                    cxx_param = cxx_signature.parameters[python_name]

                    # Check parameter kinds and default values are the same.
                    self.assertEqual(cxx_param.kind, python_param.kind)
                    self.assertEqual(cxx_param.default, python_param.default)

                    # Check parameter annotations are the same.
                    if "TreeSpec" in str(cxx_param.annotation):
                        self.assertIn("TreeSpec", str(python_param.annotation))
                        self.assertEqual(
                            re.sub(
                                r"(?:\b)([\w\.]*)TreeSpec(?:\b)",
                                "TreeSpec",
                                str(cxx_param.annotation),
                            ),
                            re.sub(
                                r"(?:\b)([\w\.]*)TreeSpec(?:\b)",
                                "TreeSpec",
                                str(python_param.annotation),
                            ),
                            msg=(
                                lambda msg: f"{msg}\nC++ parameter {cxx_param} "
                                f"does not match Python parameter {python_param} "
                                f"for API `{name}`"
                            ),
                        )
                    else:
                        self.assertEqual(
                            cxx_param.annotation,
                            python_param.annotation,
                            msg=(
                                lambda msg: f"{msg}\nC++ parameter {cxx_param} "
                                f"does not match Python parameter {python_param} "
                                f"for API `{name}`"
                            ),
                        )

    @parametrize_pytree_module
    def test_register_pytree_node(self, pytree):
        # [FUNCTION DOC] Verifies register_pytree_node. A UserDict subclass is first flattened
        # as a single leaf (spec is a leaf), then registered with flatten and unflatten
        # functions based on its values and keys, after which it flattens into its values and
        # round-trips. Registering the same type a second time must raise ValueError mentioning
        # already registered. Note the registration persists in the global registry for the
        # process.
        class MyDict(UserDict):
            # [CLASS DOC] HELPER: Empty UserDict subclass used as the custom container type
            # registered in this test.
            pass

        d = MyDict(a=1, b=2, c=3)

        # Custom types are leaf nodes by default
        values, spec = pytree.tree_flatten(d)
        self.assertEqual(values, [d])
        self.assertIs(values[0], d)
        self.assertEqual(d, pytree.tree_unflatten(values, spec))
        self.assertTrue(spec.is_leaf())

        # Register MyDict as a pytree node
        pytree.register_pytree_node(
            MyDict,
            lambda d: (list(d.values()), list(d.keys())),
            lambda values, keys: MyDict(zip(keys, values)),
        )

        values, spec = pytree.tree_flatten(d)
        self.assertEqual(values, [1, 2, 3])
        self.assertEqual(d, pytree.tree_unflatten(values, spec))

        # Do not allow registering the same type twice
        with self.assertRaisesRegex(ValueError, "already registered"):
            pytree.register_pytree_node(
                MyDict,
                lambda d: (list(d.values()), list(d.keys())),
                lambda values, keys: MyDict(zip(keys, values)),
            )

    @parametrize_pytree_module
    def test_flatten_unflatten_leaf(self, pytree):
        # [FUNCTION DOC] Checks that scalars, None, a type object and a tensor are each
        # flattened to a single-leaf list with the leaf treespec and unflattened back to the
        # same value.
        def run_test_with_leaf(leaf):
            # [FUNCTION DOC] HELPER: Flattens one value, asserts the result is a one-element
            # list holding it and that the spec equals the leaf spec, then unflattens and
            # compares to the original.
            values, treespec = pytree.tree_flatten(leaf)
            self.assertEqual(values, [leaf])
            self.assertEqual(treespec, pytree.treespec_leaf())

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, leaf)

        run_test_with_leaf(1)
        run_test_with_leaf(1.0)
        run_test_with_leaf(None)
        run_test_with_leaf(bool)
        run_test_with_leaf(torch.randn(3, 3))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda tup: python_pytree.TreeSpec(
                        tuple, None, [python_pytree.treespec_leaf() for _ in tup]
                    ),
                ),
                name="python",
            ),
            subtest(
                (cxx_pytree, lambda tup: cxx_pytree.tree_structure((0,) * len(tup))),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_tuple(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for tuples, parametrized over implementation and a
        # factory producing the expected treespec (a tuple node with leaf children for Python;
        # the structure of a zero-filled tuple for C++). Covers empty, single, mixed and tensor-
        # containing tuples.
        def run_test(tup):
            # [FUNCTION DOC] HELPER: Flattens a tuple, checks values is a list equal to the
            # tuple contents, the spec equals the expected one, and the unflattened result
            # equals the input and is a tuple.
            expected_spec = gen_expected_fn(tup)
            values, treespec = pytree.tree_flatten(tup)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertIsInstance(unflattened, tuple)

        run_test(())
        run_test((1.0,))
        run_test((1.0, 2))
        run_test((torch.tensor([1.0, 2]), 2, 10, 9, 11))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda lst: python_pytree.TreeSpec(
                        list, None, [python_pytree.treespec_leaf() for _ in lst]
                    ),
                ),
                name="python",
            ),
            subtest(
                (cxx_pytree, lambda lst: cxx_pytree.tree_structure([0] * len(lst))),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_list(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for lists, parametrized over implementation and an
        # expected-treespec factory. Covers empty, mixed and tensor-containing lists.
        def run_test(lst):
            # [FUNCTION DOC] HELPER: Flattens a list, checks leaf values, spec equality with the
            # expected spec, and that unflatten returns an equal list of type list.
            expected_spec = gen_expected_fn(lst)
            values, treespec = pytree.tree_flatten(lst)
            self.assertIsInstance(values, list)
            self.assertEqual(values, lst)
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, lst)
            self.assertIsInstance(unflattened, list)

        run_test([])
        run_test([1.0, 2])
        run_test([torch.tensor([1.0, 2]), 2, 10, 9, 11])

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda dct: python_pytree.TreeSpec(
                        dict,
                        list(dct.keys()),
                        [python_pytree.treespec_leaf() for _ in dct.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda dct: cxx_pytree.tree_structure(dict.fromkeys(dct, 0)),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_dict(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for plain dicts, parametrized over implementation and
        # expected-treespec factory (the Python spec stores the key list as context). Covers
        # empty dicts and string or integer keys with scalar and tensor values.
        def run_test(dct):
            # [FUNCTION DOC] HELPER: Flattens a dict, checks values equal the dict values in
            # order, spec equality, and that unflatten returns an equal dict of type dict.
            expected_spec = gen_expected_fn(dct)
            values, treespec = pytree.tree_flatten(dct)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(dct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, dct)
            self.assertIsInstance(unflattened, dict)

        run_test({})
        run_test({"a": 1})
        run_test({"abcdefg": torch.randn(2, 3)})
        run_test({1: torch.randn(2, 3)})
        run_test({"a": 1, "b": 2, "c": torch.randn(2, 3)})

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda odict: python_pytree.TreeSpec(
                        OrderedDict,
                        list(odict.keys()),
                        [python_pytree.treespec_leaf() for _ in odict.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda odict: cxx_pytree.tree_structure(
                        OrderedDict.fromkeys(odict, 0)
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_ordereddict(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for OrderedDict, parametrized over implementation and
        # expected-treespec factory. Runs an empty instance and then one with two entries
        # inserted in non-sorted key order, one being a tensor.
        def run_test(odict):
            # [FUNCTION DOC] HELPER: Flattens an OrderedDict, checks values follow insertion
            # order, spec equality, and that unflatten returns an equal OrderedDict.
            expected_spec = gen_expected_fn(odict)
            values, treespec = pytree.tree_flatten(odict)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(odict.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, odict)
            self.assertIsInstance(unflattened, OrderedDict)

        od = OrderedDict()
        run_test(od)

        od["b"] = 1
        od["a"] = torch.tensor(3.14)
        run_test(od)

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda ddct: python_pytree.TreeSpec(
                        defaultdict,
                        [ddct.default_factory, list(ddct.keys())],
                        [python_pytree.treespec_leaf() for _ in ddct.values()],
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda ddct: cxx_pytree.tree_structure(
                        defaultdict(ddct.default_factory, dict.fromkeys(ddct, 0))
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_defaultdict(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for defaultdict, parametrized over implementation and
        # expected-treespec factory (the Python spec context holds the default factory and the
        # key list). Covers empty and populated instances with list or int factories.
        def run_test(ddct):
            # [FUNCTION DOC] HELPER: Flattens a defaultdict, checks values, spec equality, and
            # that the unflattened result is an equal defaultdict that preserves the same
            # default_factory.
            expected_spec = gen_expected_fn(ddct)
            values, treespec = pytree.tree_flatten(ddct)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(ddct.values()))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, ddct)
            self.assertEqual(unflattened.default_factory, ddct.default_factory)
            self.assertIsInstance(unflattened, defaultdict)

        run_test(defaultdict(list, {}))
        run_test(defaultdict(int, {"a": 1}))
        run_test(defaultdict(int, {"abcdefg": torch.randn(2, 3)}))
        run_test(defaultdict(int, {1: torch.randn(2, 3)}))
        run_test(defaultdict(int, {"a": 1, "b": 2, "c": torch.randn(2, 3)}))

    @parametrize(
        "pytree,gen_expected_fn",
        [
            subtest(
                (
                    python_pytree,
                    lambda deq: python_pytree.TreeSpec(
                        deque, deq.maxlen, [python_pytree.treespec_leaf() for _ in deq]
                    ),
                ),
                name="python",
            ),
            subtest(
                (
                    cxx_pytree,
                    lambda deq: cxx_pytree.tree_structure(
                        deque(deq, maxlen=deq.maxlen)
                    ),
                ),
                name="cxx",
            ),
        ],
    )
    def test_flatten_unflatten_deque(self, pytree, gen_expected_fn):
        # [FUNCTION DOC] Round-trip test for deque, parametrized over implementation and
        # expected-treespec factory (the Python spec context is the maxlen). Covers empty, small
        # and tensor-containing deques, one with maxlen set to 8.
        def run_test(deq):
            # [FUNCTION DOC] HELPER: Flattens a deque, checks values, spec equality, and that
            # unflatten returns an equal deque with the same maxlen.
            expected_spec = gen_expected_fn(deq)
            values, treespec = pytree.tree_flatten(deq)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(deq))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, deq)
            self.assertEqual(unflattened.maxlen, deq.maxlen)
            self.assertIsInstance(unflattened, deque)

        run_test(deque([]))
        run_test(deque([1.0, 2]))
        run_test(deque([torch.tensor([1.0, 2]), 2, 10, 9, 11], maxlen=8))

    @parametrize_pytree_module
    def test_flatten_unflatten_namedtuple(self, pytree):
        # [FUNCTION DOC] Round-trip test for a locally defined namedtuple type named Point. The
        # expected spec is built per implementation (explicit namedtuple TreeSpec for Python,
        # structure of a sample Point for C++). Covers float and tensor fields.
        Point = namedtuple("Point", ["x", "y"])

        def run_test(tup):
            # [FUNCTION DOC] HELPER: Flattens a namedtuple instance, checks values, spec
            # equality, and that unflatten returns an equal instance of the Point class.
            if pytree is python_pytree:
                expected_spec = python_pytree.TreeSpec(
                    namedtuple, Point, [python_pytree.treespec_leaf() for _ in tup]
                )
            else:
                expected_spec = cxx_pytree.tree_structure(Point(0, 1))
            values, treespec = pytree.tree_flatten(tup)
            self.assertIsInstance(values, list)
            self.assertEqual(values, list(tup))
            self.assertEqual(treespec, expected_spec)

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tup)
            self.assertIsInstance(unflattened, Point)

        run_test(Point(1.0, 2))
        run_test(Point(torch.tensor(1.0), 2))

    @parametrize(
        "op",
        [
            subtest(torch.max, name="max"),
            subtest(torch.min, name="min"),
        ],
    )
    @parametrize_pytree_module
    def test_flatten_unflatten_return_types(self, pytree, op):
        # [FUNCTION DOC] Checks that torch.max and torch.min results with a dim argument (named-
        # tuple-like return types) flatten into a list of tensors only and unflatten to the same
        # return type with equal contents. Parametrized over the operator and the pytree
        # implementation.
        x = torch.randn(3, 3)
        expected = op(x, dim=0)

        values, spec = pytree.tree_flatten(expected)
        # Check that values is actually List[Tensor] and not (ReturnType(...),)
        for value in values:
            self.assertIsInstance(value, torch.Tensor)
        result = pytree.tree_unflatten(values, spec)

        self.assertEqual(type(result), type(expected))
        self.assertEqual(result, expected)

    @parametrize_pytree_module
    def test_flatten_unflatten_nested(self, pytree):
        # [FUNCTION DOC] Round-trip test over several nested trees (empty containers, dicts,
        # lists, tuples, tensors): the number of flattened values must equal treespec.num_leaves
        # and unflatten must reproduce the original tree.
        def run_test(tree):
            # [FUNCTION DOC] HELPER: Flattens one tree, asserts values is a list sized
            # num_leaves, and that unflatten returns an equal tree.
            values, treespec = pytree.tree_flatten(tree)
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), treespec.num_leaves)

            # NB: python basic data structures (dict list tuple) all have
            # contents equality defined on them, so the following works for them.
            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tree)

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 0, "b": [{"c": 1}]},
            {"a": 0, "b": [1, {"c": 2}, torch.randn(3)], "c": (torch.randn(2, 3), 1)},
        ]
        for case in cases:
            run_test(case)

    @parametrize_pytree_module
    def test_flatten_with_is_leaf(self, pytree):
        # [FUNCTION DOC] Tests tree_flatten with an is_leaf predicate that treats every node
        # except the root as a leaf, so only one level is flattened. Verifies the number of
        # values, leaves, children and nodes minus the root, the expected one-level leaf list,
        # that the spec equals the structure rebuilt from it, and that unflatten restores the
        # tree.
        def run_test(tree, one_level_leaves):
            # [FUNCTION DOC] HELPER: Flattens a tree with the one-level is_leaf predicate and
            # checks counts, values against one_level_leaves, spec equality and the unflatten
            # round trip.
            values, treespec = pytree.tree_flatten(
                tree, is_leaf=lambda x: x is not tree
            )
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), treespec.num_nodes - 1)
            self.assertEqual(len(values), treespec.num_leaves)
            self.assertEqual(len(values), treespec.num_children)
            self.assertEqual(values, one_level_leaves)

            self.assertEqual(
                treespec,
                pytree.tree_structure(
                    pytree.tree_unflatten([0] * treespec.num_leaves, treespec)
                ),
            )

            unflattened = pytree.tree_unflatten(values, treespec)
            self.assertEqual(unflattened, tree)

        cases = [
            ([()], [()]),
            (([],), [[]]),
            ({"a": ()}, [()]),
            ({"a": 0, "b": [{"c": 1}]}, [0, [{"c": 1}]]),
            (
                {
                    "a": 0,
                    "b": [1, {"c": 2}, torch.ones(3)],
                    "c": (torch.zeros(2, 3), 1),
                },
                [0, [1, {"c": 2}, torch.ones(3)], (torch.zeros(2, 3), 1)],
            ),
        ]
        for case in cases:
            run_test(*case)

    @parametrize_pytree_module
    def test_tree_map(self, pytree):
        # [FUNCTION DOC] Tests tree_map: mapping a multiply-by-three function preserves the sum
        # of leaves relative to mapping over the leaves list, and mapping the integer-divide-by-
        # three inverse restores the original tree. Runs over several nested container cases.
        def run_test(tree):
            # [FUNCTION DOC] HELPER: Applies the multiply and divide mapping checks to a single
            # tree.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns its argument multiplied by three.
                return x * 3

            sm1 = sum(map(f, pytree.tree_leaves(tree)))
            sm2 = sum(pytree.tree_leaves(pytree.tree_map(f, tree)))
            self.assertEqual(sm1, sm2)

            def invf(x):
                # [FUNCTION DOC] HELPER: Returns its argument floor-divided by three; inverse of
                # the multiply helper for multiples of three.
                return x // 3

            self.assertEqual(
                pytree.tree_map(invf, pytree.tree_map(f, tree)),
                tree,
            )

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        for case in cases:
            run_test(case)

    @parametrize_pytree_module
    def test_tree_map_multi_inputs(self, pytree):
        # [FUNCTION DOC] Tests tree_map with several trees. Builds a second tree with one-
        # element tuples and a third with dicts from the first, and checks mapping a three-
        # argument function across the three trees matches mapping a single-argument function
        # that builds the same extra arguments per leaf.
        def run_test(tree):
            # [FUNCTION DOC] HELPER: Builds the companion trees for one input tree and compares
            # multi-tree tree_map to the equivalent single-tree map.
            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Returns a nested structure built from its three
                # arguments: the first, then a list holding the second and a tuple of the third
                # and a zero. Used to check multi-tree mapping with structured outputs.
                return x, [y, (z, 0)]

            tree_x = tree
            tree_y = pytree.tree_map(lambda x: (x + 1,), tree)
            tree_z = pytree.tree_map(lambda x: {"a": x * 2, "b": 2}, tree)

            self.assertEqual(
                pytree.tree_map(f, tree_x, tree_y, tree_z),
                pytree.tree_map(lambda x: f(x, (x + 1,), {"a": x * 2, "b": 2}), tree),
            )

        cases = [
            [()],
            ([],),
            {"a": ()},
            {"a": 1, "b": [{"c": 2}]},
            {"a": 0, "b": [2, {"c": 3}, 4], "c": (5, 6)},
        ]
        for case in cases:
            run_test(case)

    @parametrize_pytree_module
    def test_tree_map_dict_order(self, pytree):
        # [FUNCTION DOC] Checks that tree_map preserves key order for dict, OrderedDict and
        # defaultdict whose keys are inserted in non-sorted order: the keys and values lists of
        # the result must match the input order.
        d = {"b": 2, "a": 1, "c": 3}
        od = OrderedDict([("b", 2), ("a", 1), ("c", 3)])
        dd = defaultdict(int, {"b": 2, "a": 1, "c": 3})
        for tree in (d, od, dd):
            result = pytree.tree_map(lambda x: x, tree)
            self.assertEqual(
                list(result.keys()),
                list(tree.keys()),
                msg=lambda msg: f"{msg}\nDictionary keys order changed in tree_map: {tree!r} vs. {result!r}",
            )
            self.assertEqual(
                list(result.values()),
                list(tree.values()),
                msg=lambda msg: f"{msg}\nDictionary keys order changed in tree_map: {tree!r} vs. {result!r}",
            )

    @parametrize_pytree_module
    def test_tree_map_only(self, pytree):
        # [FUNCTION DOC] Checks tree_map_only applies the function only to leaves of the given
        # type (int), leaving the string leaf unchanged.
        self.assertEqual(pytree.tree_map_only(int, lambda x: x + 2, [0, "a"]), [2, "a"])

    @parametrize_pytree_module
    def test_tree_map_only_predicate_fn(self, pytree):
        # [FUNCTION DOC] Checks tree_map_only accepts a predicate function in place of a type:
        # only leaves equal to zero are mapped.
        self.assertEqual(
            pytree.tree_map_only(lambda x: x == 0, lambda x: x + 2, [0, 1]), [2, 1]
        )

    @parametrize_pytree_module
    def test_tree_all_any(self, pytree):
        # [FUNCTION DOC] Checks tree_all, tree_any, tree_all_only and tree_any_only with an odd-
        # number predicate, for true and false outcomes, including that the _only variants
        # ignore leaves of other types.
        self.assertTrue(pytree.tree_all(lambda x: x % 2, [1, 3]))
        self.assertFalse(pytree.tree_all(lambda x: x % 2, [0, 1]))
        self.assertTrue(pytree.tree_any(lambda x: x % 2, [0, 1]))
        self.assertFalse(pytree.tree_any(lambda x: x % 2, [0, 2]))
        self.assertTrue(pytree.tree_all_only(int, lambda x: x % 2, [1, 3, "a"]))
        self.assertFalse(pytree.tree_all_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertTrue(pytree.tree_any_only(int, lambda x: x % 2, [0, 1, "a"]))
        self.assertFalse(pytree.tree_any_only(int, lambda x: x % 2, [0, 2, "a"]))

    @parametrize_pytree_module
    def test_broadcast_to_and_flatten(self, pytree):
        # [FUNCTION DOC] Table-driven test of the private _broadcast_to_and_flatten helper. For
        # each (tree, target tree, expected) case it flattens the target to get a spec and
        # asserts the result equals the expected flat list, or None when structures mismatch.
        # Covers equal flat and nested structures, mismatches in type, length and dict keys, and
        # broadcasting a single leaf or multiple subtrees.
        cases = [
            (1, (), []),
            # Same (flat) structures
            ((1,), (0,), [1]),
            ([1], [0], [1]),
            ((1, 2, 3), (0, 0, 0), [1, 2, 3]),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0}, [1, 2]),
            # Mismatched (flat) structures
            ([1], (0,), None),
            ([1], (0,), None),
            ((1,), [0], None),
            ((1, 2, 3), (0, 0), None),
            ({"a": 1, "b": 2}, {"a": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "c": 0}, None),
            ({"a": 1, "b": 2}, {"a": 0, "b": 0, "c": 0}, None),
            # Same (nested) structures
            ((1, [2, 3]), (0, [0, 0]), [1, 2, 3]),
            ((1, [(2, 3), 4]), (0, [(0, 0), 0]), [1, 2, 3, 4]),
            # Mismatched (nested) structures
            ((1, [2, 3]), (0, (0, 0)), None),
            ((1, [2, 3]), (0, [0, 0, 0]), None),
            # Broadcasting single value
            (1, (0, 0, 0), [1, 1, 1]),
            (1, [0, 0, 0], [1, 1, 1]),
            (1, {"a": 0, "b": 0}, [1, 1]),
            (1, (0, [0, [0]], 0), [1, 1, 1, 1]),
            (1, (0, [0, [0, [], [[[0]]]]], 0), [1, 1, 1, 1, 1]),
            # Broadcast multiple things
            ((1, 2), ([0, 0, 0], [0, 0]), [1, 1, 1, 2, 2]),
            ((1, 2), ([0, [0, 0], 0], [0, 0]), [1, 1, 1, 1, 2, 2]),
            (([1, 2, 3], 4), ([0, [0, 0], 0], [0, 0]), [1, 2, 2, 3, 4, 4]),
        ]
        for tree, to_tree, expected in cases:
            _, to_spec = pytree.tree_flatten(to_tree)
            result = pytree._broadcast_to_and_flatten(tree, to_spec)
            self.assertEqual(result, expected, msg=str([tree, to_spec, expected]))

    @parametrize_pytree_module
    def test_pytree_serialize_bad_input(self, pytree):
        # [FUNCTION DOC] Checks that treespec_dumps raises TypeError when given a non-treespec
        # value (a string).
        with self.assertRaises(TypeError):
            pytree.treespec_dumps("random_blurb")

    @parametrize_pytree_module
    def test_is_namedtuple(self, pytree):
        # [FUNCTION DOC] Tests is_namedtuple and is_namedtuple_class on direct and indirect
        # namedtuple and NamedTuple types and instances, and on negatives (struct_time, tuple,
        # list, dict, set, int). Instances satisfy the instance check; classes satisfy the class
        # check as well as the instance-or-class check.
        DirectNamedTuple1 = namedtuple("DirectNamedTuple1", ["x", "y"])

        class DirectNamedTuple2(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple-style class with integer fields x and y.
            x: int
            y: int

        class IndirectNamedTuple1(DirectNamedTuple1):
            # [CLASS DOC] HELPER: Subclass of the collections namedtuple type that adds no
            # fields.
            pass

        class IndirectNamedTuple2(DirectNamedTuple2):
            # [CLASS DOC] HELPER: Subclass of the typing NamedTuple class that adds no fields.
            pass

        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple1(0, 1)))
        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple2(0, 1)))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple1(0, 1)))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple2(0, 1)))
        self.assertFalse(pytree.is_namedtuple(time.gmtime()))
        self.assertFalse(pytree.is_namedtuple((0, 1)))
        self.assertFalse(pytree.is_namedtuple([0, 1]))
        self.assertFalse(pytree.is_namedtuple({0: 1, 1: 2}))
        self.assertFalse(pytree.is_namedtuple({0, 1}))
        self.assertFalse(pytree.is_namedtuple(1))

        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple(DirectNamedTuple2))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple(IndirectNamedTuple2))
        self.assertFalse(pytree.is_namedtuple(time.struct_time))
        self.assertFalse(pytree.is_namedtuple(tuple))
        self.assertFalse(pytree.is_namedtuple(list))

        self.assertTrue(pytree.is_namedtuple_class(DirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple_class(DirectNamedTuple2))
        self.assertTrue(pytree.is_namedtuple_class(IndirectNamedTuple1))
        self.assertTrue(pytree.is_namedtuple_class(IndirectNamedTuple2))
        self.assertFalse(pytree.is_namedtuple_class(time.struct_time))
        self.assertFalse(pytree.is_namedtuple_class(tuple))
        self.assertFalse(pytree.is_namedtuple_class(list))

    @parametrize_pytree_module
    def test_is_structseq(self, pytree):
        # [FUNCTION DOC] Tests is_structseq and is_structseq_class. A hand-written tuple
        # subclass imitating struct sequence attributes, namedtuples, builtin containers and
        # ints are not structseqs, while time.struct_time and every torch.return_types class
        # are. For return types it also checks instances, and that none is a namedtuple. Non-
        # tuple module attributes must be rejected by all four predicates.
        class FakeStructSeq(tuple):
            # [CLASS DOC] HELPER: Fake struct-sequence lookalike: tuple subclass with n_fields
            # attributes and properties x and y but not a real PyStructSequence type; should not
            # be detected as a structseq.
            n_fields = 2
            n_sequence_fields = 2
            n_unnamed_fields = 0

            __slots__ = ()
            __match_args__ = ("x", "y")

            def __new__(cls, sequence):
                # [FUNCTION DOC] HELPER: Constructs the tuple from the given sequence by
                # delegating to the tuple constructor.
                return super().__new__(cls, sequence)

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property returning the first element.
                return self[0]

            @property
            def y(self):
                # [FUNCTION DOC] HELPER: Property returning the second element.
                return self[1]

        DirectNamedTuple1 = namedtuple("DirectNamedTuple1", ["x", "y"])

        class DirectNamedTuple2(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple-style class with integer fields x and y.
            x: int
            y: int

        self.assertFalse(pytree.is_structseq(FakeStructSeq((0, 1))))
        self.assertTrue(pytree.is_structseq(time.gmtime()))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple1(0, 1)))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple2(0, 1)))
        self.assertFalse(pytree.is_structseq((0, 1)))
        self.assertFalse(pytree.is_structseq([0, 1]))
        self.assertFalse(pytree.is_structseq({0: 1, 1: 2}))
        self.assertFalse(pytree.is_structseq({0, 1}))
        self.assertFalse(pytree.is_structseq(1))

        self.assertFalse(pytree.is_structseq(FakeStructSeq))
        self.assertTrue(pytree.is_structseq(time.struct_time))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple1))
        self.assertFalse(pytree.is_structseq(DirectNamedTuple2))
        self.assertFalse(pytree.is_structseq(tuple))
        self.assertFalse(pytree.is_structseq(list))

        self.assertFalse(pytree.is_structseq_class(FakeStructSeq))
        self.assertTrue(
            pytree.is_structseq_class(time.struct_time),
        )
        self.assertFalse(pytree.is_structseq_class(DirectNamedTuple1))
        self.assertFalse(pytree.is_structseq_class(DirectNamedTuple2))
        self.assertFalse(pytree.is_structseq_class(tuple))
        self.assertFalse(pytree.is_structseq_class(list))

        # torch.return_types.* are all PyStructSequence types
        for cls in vars(torch.return_types).values():
            if isinstance(cls, type) and issubclass(cls, tuple):
                self.assertTrue(pytree.is_structseq(cls))
                self.assertTrue(pytree.is_structseq_class(cls))
                self.assertFalse(pytree.is_namedtuple(cls))
                self.assertFalse(pytree.is_namedtuple_class(cls))

                inst = cls(range(cls.n_sequence_fields))
                self.assertTrue(pytree.is_structseq(inst))
                self.assertTrue(pytree.is_structseq(type(inst)))
                self.assertFalse(pytree.is_structseq_class(inst))
                self.assertTrue(pytree.is_structseq_class(type(inst)))
                self.assertFalse(pytree.is_namedtuple(inst))
                self.assertFalse(pytree.is_namedtuple_class(inst))
            else:
                self.assertFalse(pytree.is_structseq(cls))
                self.assertFalse(pytree.is_structseq_class(cls))
                self.assertFalse(pytree.is_namedtuple(cls))
                self.assertFalse(pytree.is_namedtuple_class(cls))

    @parametrize_pytree_module
    def test_enum_treespec_roundtrip(self, pytree):
        # [FUNCTION DOC] Round-trips the treespec of a dict keyed by an enum member through
        # treespec_dumps and treespec_loads, and checks the loaded spec equals the original.
        data = {TestEnum.A: 5}
        spec = pytree.tree_structure(data)

        serialized = pytree.treespec_dumps(spec)
        deserialized_spec = pytree.treespec_loads(serialized)
        self.assertEqual(spec, deserialized_spec)

    @parametrize_pytree_module
    def test_treespec_deepcopy_roundtrip(self, pytree):
        # [FUNCTION DOC] Checks that copy.deepcopy of treespecs for a leaf, tuple, list, dicts
        # and nested containers yields an equal treespec.
        cases = [
            1,
            (1, 2),
            [1, 2, 3],
            {"a": 1, "b": 2},
            (1, [2, {"a": 3}]),
            {"a": [1, 2], "b": (3, 4)},
        ]

        for tree in cases:
            treespec = pytree.tree_structure(tree)
            reconstructed = copy.deepcopy(treespec)
            self.assertEqual(treespec, reconstructed)

    @parametrize_pytree_module
    def test_treespec_pickle_roundtrip(self, pytree):
        # [FUNCTION DOC] Checks that a pickle dump and load round trip of treespecs for several
        # trees yields an equal treespec.
        cases = [
            1,
            (1, 2),
            [1, 2, 3],
            {"a": 1, "b": 2},
            (1, [2, {"a": 3}]),
            {"a": [1, 2], "b": (3, 4)},
        ]

        for tree in cases:
            treespec = pytree.tree_structure(tree)
            reconstructed = pickle.loads(pickle.dumps(treespec))
            self.assertEqual(treespec, reconstructed)


class TestPythonPytree(TestCase):
    # [CLASS DOC] Test case for behavior specific to the pure Python pytree module: leaf spec
    # copy and pickle, deprecated registration, import hygiene, treespec equality and repr, JSON
    # serialization with protocols and custom types, dataclass and constant registration, and
    # key-path APIs. Several tests register types into the global registry and may leave them
    # registered.
    def test_leafspec_copy_pickle_no_deprecation_warning(self):
        # [FUNCTION DOC] Checks that deep-copying and unpickling the leaf spec emit no
        # FutureWarning (warnings escalated to errors) and return the identical shared singleton
        # object.
        # LeafSpec is @deprecated, so reconstructing it via copy/pickle used to
        # re-invoke the constructor and leak a FutureWarning to users who never
        # wrote an isinstance check. __reduce__ rebuilds via the factory, which
        # both silences the warning and reuses the shared singleton.
        leaf = python_pytree.treespec_leaf()
        for reconstruct in (
            lambda: copy.deepcopy(leaf),
            lambda: pickle.loads(pickle.dumps(leaf)),
        ):
            with warnings.catch_warnings():
                warnings.simplefilter("error", FutureWarning)
                reconstructed = reconstruct()
            self.assertIs(reconstructed, leaf)

    def test_deprecated_register_pytree_node(self):
        # [FUNCTION DOC] Checks the deprecated _register_pytree_node wrapper emits a
        # FutureWarning naming the new API, and that registering the same type again emits a
        # UserWarning mentioning already registered.
        class DummyType:
            # [CLASS DOC] HELPER: Plain class holding two attributes, registered as a pytree
            # node in this test.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        with self.assertWarnsRegex(
            FutureWarning, "torch.utils._pytree._register_pytree_node"
        ):
            python_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

        with self.assertWarnsRegex(UserWarning, "already registered"):
            python_pytree._register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
            )

    def test_import_pytree_doesnt_import_optree(self):
        # [FUNCTION DOC] Runs a child Python process that imports torch and torch.utils._pytree
        # and raises if the cxx pytree module or optree ended up in sys.modules. Passes if the
        # subprocess exits cleanly; on a nonzero exit the test fails with the decoded subprocess
        # output. The working directory is set to this script's directory.
        # importing torch.utils._pytree shouldn't import optree.
        # only importing torch.utils._cxx_pytree should.
        script = """
import sys
import torch
import torch.utils._pytree
assert "torch.utils._pytree" in sys.modules
if "torch.utils._cxx_pytree" in sys.modules:
    raise RuntimeError("importing torch.utils._pytree should not import torch.utils._cxx_pytree")
if "optree" in sys.modules:
    raise RuntimeError("importing torch.utils._pytree should not import optree")
"""
        try:
            subprocess.check_output(
                [sys.executable, "-c", script],
                stderr=subprocess.STDOUT,
                # On Windows, opening the subprocess with the default CWD makes `import torch`
                # fail, so just set CWD to this script's directory
                cwd=os.path.dirname(os.path.realpath(__file__)),
            )
        except subprocess.CalledProcessError as e:
            self.fail(
                msg=(
                    "Subprocess exception while attempting to run test: "
                    + e.output.decode("utf-8")
                )
            )

    def test_treespec_equality(self):
        # [FUNCTION DOC] Checks treespec equality semantics: identical leaf specs and identical
        # list specs (empty or with a leaf child) compare equal, while tuple and list specs with
        # the same children compare unequal using both equal and not-equal operators.
        self.assertEqual(
            python_pytree.treespec_leaf(),
            python_pytree.treespec_leaf(),
        )
        self.assertEqual(
            python_pytree.TreeSpec(list, None, []),
            python_pytree.TreeSpec(list, None, []),
        )
        self.assertEqual(
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
        )
        self.assertFalse(
            python_pytree.TreeSpec(tuple, None, [])
            == python_pytree.TreeSpec(list, None, []),
        )
        self.assertTrue(
            python_pytree.TreeSpec(tuple, None, [])
            != python_pytree.TreeSpec(list, None, []),
        )

    def test_treespec_repr(self):
        # [FUNCTION DOC] Checks the exact repr string of the treespec of a nested tuple and
        # list, with leaves printed as stars and nested specs indented on new lines.
        # Check that it looks sane
        tree = (0, [0, 0, [0]])
        spec = python_pytree.tree_structure(tree)
        self.assertEqual(
            repr(spec),
            (
                "TreeSpec(tuple, None, [*,\n"
                "  TreeSpec(list, None, [*,\n"
                "    *,\n"
                "    TreeSpec(list, None, [*])])])"
            ),
        )

    @parametrize(
        "spec",
        [
            # python_pytree.tree_structure([])
            python_pytree.TreeSpec(list, None, []),
            # python_pytree.tree_structure(())
            python_pytree.TreeSpec(tuple, None, []),
            # python_pytree.tree_structure({})
            python_pytree.TreeSpec(dict, [], []),
            # python_pytree.tree_structure([0])
            python_pytree.TreeSpec(list, None, [python_pytree.treespec_leaf()]),
            # python_pytree.tree_structure([0, 1])
            python_pytree.TreeSpec(
                list,
                None,
                [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
            ),
            # python_pytree.tree_structure((0, 1, 2))
            python_pytree.TreeSpec(
                tuple,
                None,
                [
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                ],
            ),
            # python_pytree.tree_structure({"a": 0, "b": 1, "c": 2})
            python_pytree.TreeSpec(
                dict,
                ["a", "b", "c"],
                [
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                    python_pytree.treespec_leaf(),
                ],
            ),
            # python_pytree.tree_structure(OrderedDict([("a", (0, 1)), ("b", 2), ("c", {"a": 3, "b": 4, "c": 5})])
            python_pytree.TreeSpec(
                OrderedDict,
                ["a", "b", "c"],
                [
                    python_pytree.TreeSpec(
                        tuple,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.treespec_leaf(),
                    python_pytree.TreeSpec(
                        dict,
                        ["a", "b", "c"],
                        [
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                        ],
                    ),
                ],
            ),
            # python_pytree.tree_structure([(0, 1, [2, 3])])
            python_pytree.TreeSpec(
                list,
                None,
                [
                    python_pytree.TreeSpec(
                        tuple,
                        None,
                        [
                            python_pytree.treespec_leaf(),
                            python_pytree.treespec_leaf(),
                            python_pytree.TreeSpec(
                                list,
                                None,
                                [
                                    python_pytree.treespec_leaf(),
                                    python_pytree.treespec_leaf(),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            # python_pytree.tree_structure(defaultdict(list, {"a": [0, 1], "b": [1, 2], "c": {}}))
            python_pytree.TreeSpec(
                defaultdict,
                [list, ["a", "b", "c"]],
                [
                    python_pytree.TreeSpec(
                        list,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.TreeSpec(
                        list,
                        None,
                        [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                    ),
                    python_pytree.TreeSpec(dict, [], []),
                ],
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        # [FUNCTION DOC] Parametrized over many Python treespecs (empty containers, flat and
        # nested lists, tuples, dicts, OrderedDict and defaultdict). Verifies each spec equals
        # the structure of its own unflattened tree, then that treespec_dumps returns a string
        # and treespec_loads restores an equal spec.
        # Ensure that the spec is valid
        self.assertEqual(
            spec,
            python_pytree.tree_structure(
                python_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)
        self.assertEqual(spec, python_pytree.treespec_loads(serialized_spec))

    def test_pytree_serialize_defaultdict_enum(self):
        # [FUNCTION DOC] Checks treespec_dumps succeeds and returns a string for a defaultdict
        # spec whose key list contains an enum member.
        spec = python_pytree.TreeSpec(
            defaultdict,
            [list, [TestEnum.A]],
            [
                python_pytree.TreeSpec(
                    list,
                    None,
                    [
                        python_pytree.treespec_leaf(),
                    ],
                ),
            ],
        )
        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

    def test_pytree_serialize_enum(self):
        # [FUNCTION DOC] Checks treespec_dumps succeeds and returns a string for a dict spec
        # whose context is an enum member.
        spec = python_pytree.TreeSpec(dict, TestEnum.A, [python_pytree.treespec_leaf()])

        serialized_spec = python_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

    def test_pytree_serialize_namedtuple(self):
        # [FUNCTION DOC] Registers three namedtuple types (a collections namedtuple, a typing
        # NamedTuple subclass, and a subclass of that) with serialized names via
        # _register_namedtuple, then for each checks the spec type is the namedtuple marker and
        # the dump and load round trip gives an equal spec. Side effect: persistent registry
        # entries.
        Point1 = namedtuple("Point1", ["x", "y"])
        python_pytree._register_namedtuple(
            Point1,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point1",
        )

        spec = python_pytree.tree_structure(Point1(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

        class Point2(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple-style class with integer fields x and y.
            x: int
            y: int

        python_pytree._register_namedtuple(
            Point2,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point2",
        )

        spec = python_pytree.tree_structure(Point2(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

        class Point3(Point2):
            # [CLASS DOC] HELPER: Subclass of Point2 adding nothing.
            pass

        python_pytree._register_namedtuple(
            Point3,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.Point3",
        )

        spec = python_pytree.tree_structure(Point3(1, 2))
        self.assertIs(spec.type, namedtuple)
        roundtrip_spec = python_pytree.treespec_loads(
            python_pytree.treespec_dumps(spec)
        )
        self.assertEqual(spec, roundtrip_spec)

    def test_pytree_serialize_namedtuple_bad(self):
        # [FUNCTION DOC] Checks treespec_dumps of an unregistered namedtuple spec raises
        # NotImplementedError telling the caller to register with _register_namedtuple.
        DummyType = namedtuple("DummyType", ["x", "y"])

        spec = python_pytree.tree_structure(DummyType(1, 2))

        with self.assertRaisesRegex(
            NotImplementedError, "Please register using `_register_namedtuple`"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_custom_type_serialize_bad(self):
        # [FUNCTION DOC] Checks that a custom pytree node registered without a serialized type
        # name makes treespec_dumps raise NotImplementedError about no registered serialization
        # name. Side effect: leaves the type registered.
        class DummyType:
            # [CLASS DOC] HELPER: Plain class holding two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
        )

        spec = python_pytree.tree_structure(DummyType(1, 2))
        with self.assertRaisesRegex(
            NotImplementedError, "No registered serialization name"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_custom_type_serialize(self):
        # [FUNCTION DOC] Registers a custom node with a serialized name and to/from dumpable
        # context functions, dumps its spec with protocol 1, checks the output contains the
        # dumped context text and that loading gives an equal spec. Side effect: leaves the type
        # registered.
        class DummyType:
            # [CLASS DOC] HELPER: Plain class holding two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            serialized_type_name="test_pytree_custom_type_serialize.DummyType",
            to_dumpable_context=lambda context: "moo",
            from_dumpable_context=lambda dumpable_context: None,
        )
        spec = python_pytree.tree_structure(DummyType(1, 2))
        serialized_spec = python_pytree.treespec_dumps(spec, 1)
        self.assertIn("moo", serialized_spec)
        roundtrip_spec = python_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_register_bad(self):
        # [FUNCTION DOC] Checks register_pytree_node raises ValueError when only one of
        # to_dumpable_context and from_dumpable_context is supplied.
        class DummyType:
            # [CLASS DOC] HELPER: Plain class holding two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        with self.assertRaisesRegex(
            ValueError, "Both to_dumpable_context and from_dumpable_context"
        ):
            python_pytree.register_pytree_node(
                DummyType,
                lambda dummy: ([dummy.x, dummy.y], None),
                lambda xs, _: DummyType(*xs),
                serialized_type_name="test_pytree_serialize_register_bad.DummyType",
                to_dumpable_context=lambda context: "moo",
            )

    def test_pytree_context_serialize_bad(self):
        # [FUNCTION DOC] Registers a custom node whose dumpable context function returns a non-
        # JSON-serializable type object, then checks treespec_dumps raises TypeError about the
        # type not being JSON serializable. Side effect: leaves the type registered.
        class DummyType:
            # [CLASS DOC] HELPER: Plain class holding two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        python_pytree.register_pytree_node(
            DummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: DummyType(*xs),
            serialized_type_name="test_pytree_serialize_serialize_bad.DummyType",
            to_dumpable_context=lambda context: DummyType,
            from_dumpable_context=lambda dumpable_context: None,
        )

        spec = python_pytree.tree_structure(DummyType(1, 2))

        with self.assertRaisesRegex(
            TypeError, "Object of type type is not JSON serializable"
        ):
            python_pytree.treespec_dumps(spec)

    def test_pytree_serialize_bad_protocol(self):
        # [FUNCTION DOC] Checks an unknown serialization protocol is rejected: dumping with
        # protocol minus one raises ValueError, and loading a JSON payload whose protocol field
        # was replaced by minus one also raises ValueError mentioning Unknown protocol.
        # Registers a namedtuple name as a side effect.
        import json

        Point = namedtuple("Point", ["x", "y"])
        spec = python_pytree.tree_structure(Point(1, 2))
        python_pytree._register_namedtuple(
            Point,
            serialized_type_name="test_pytree.test_pytree_serialize_bad_protocol.Point",
        )

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            python_pytree.treespec_dumps(spec, -1)

        serialized_spec = python_pytree.treespec_dumps(spec)
        _, data = json.loads(serialized_spec)
        bad_protocol_serialized_spec = json.dumps((-1, data))

        with self.assertRaisesRegex(ValueError, "Unknown protocol"):
            python_pytree.treespec_loads(bad_protocol_serialized_spec)

    def test_saved_serialized(self):
        # [FUNCTION DOC] Compares the dumped JSON of a hand-built nested OrderedDict and tuple
        # and dict spec against a stored literal string (format regression check), then checks
        # loading the saved string reproduces the spec. Also validates the spec first via
        # unflatten and structure.
        # python_pytree.tree_structure(OrderedDict([(1, (0, 1)), (2, 2), (3, {4: 3, 5: 4, 6: 5})]))
        complicated_spec = python_pytree.TreeSpec(
            OrderedDict,
            [1, 2, 3],
            [
                python_pytree.TreeSpec(
                    tuple,
                    None,
                    [python_pytree.treespec_leaf(), python_pytree.treespec_leaf()],
                ),
                python_pytree.treespec_leaf(),
                python_pytree.TreeSpec(
                    dict,
                    [4, 5, 6],
                    [
                        python_pytree.treespec_leaf(),
                        python_pytree.treespec_leaf(),
                        python_pytree.treespec_leaf(),
                    ],
                ),
            ],
        )
        # Ensure that the spec is valid
        self.assertEqual(
            complicated_spec,
            python_pytree.tree_structure(
                python_pytree.tree_unflatten(
                    [0] * complicated_spec.num_leaves, complicated_spec
                )
            ),
        )

        serialized_spec = python_pytree.treespec_dumps(complicated_spec)
        saved_spec = (
            '[1, {"type": "collections.OrderedDict", "context": "[1, 2, 3]", '
            '"children_spec": [{"type": "builtins.tuple", "context": "null", '
            '"children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, '
            '"children_spec": []}]}, {"type": null, "context": null, '
            '"children_spec": []}, {"type": "builtins.dict", "context": '
            '"[4, 5, 6]", "children_spec": [{"type": null, "context": null, '
            '"children_spec": []}, {"type": null, "context": null, "children_spec": '
            '[]}, {"type": null, "context": null, "children_spec": []}]}]}]'
        )
        self.assertEqual(serialized_spec, saved_spec)
        self.assertEqual(complicated_spec, python_pytree.treespec_loads(saved_spec))

    def test_tree_map_with_path(self):
        # [FUNCTION DOC] Checks tree_map_with_path on a list holding a dict: the path key
        # entries (list index and dict key) are used so the function returns zero for every
        # value.
        tree = [{i: i for i in range(10)}]
        all_zeros = python_pytree.tree_map_with_path(
            lambda kp, val: val - kp[1].key + kp[0].idx, tree
        )
        self.assertEqual(all_zeros, [dict.fromkeys(range(10), 0)])

    def test_dataclass(self):
        # [FUNCTION DOC] Tests register_dataclass on a dataclass with a non-init field. Default
        # registration maps the tree and drops the non-init field value (it resets to its
        # default). Wrong field_names raise ValueError (Missing fields, Unexpected fields, also
        # for the non-init field). With field_names and drop_field_names the dropped fields come
        # back as defaults. Deregisters the type between steps.
        @dataclass
        class Data:
            # [CLASS DOC] HELPER: Dataclass with tensor field a, defaulted string fields b and c
            # (c optional), and non-init string field d.
            a: torch.Tensor
            b: str = "moo"
            c: str | None = None
            d: str = field(init=False, default="")

        python_pytree.register_dataclass(Data)
        old_data = Data(torch.tensor(3), "b", "c")
        old_data.d = "d"
        new_data = python_pytree.tree_map(lambda x: x, old_data)
        self.assertEqual(new_data.a, torch.tensor(3))
        self.assertEqual(new_data.b, "b")
        self.assertEqual(new_data.c, "c")
        self.assertEqual(new_data.d, "")
        python_pytree._deregister_pytree_node(Data)

        with self.assertRaisesRegex(ValueError, "Missing fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b"])

        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b", "e"])

        with self.assertRaisesRegex(ValueError, "Unexpected fields"):
            python_pytree.register_dataclass(Data, field_names=["a", "b", "c", "d"])

        python_pytree.register_dataclass(
            Data, field_names=["a"], drop_field_names=["b", "c"]
        )
        old_data = Data(torch.tensor(3), "b", "c")
        new_data = python_pytree.tree_map(lambda x: x, old_data)
        self.assertEqual(new_data.a, torch.tensor(3))
        self.assertEqual(new_data.b, "moo")
        self.assertEqual(new_data.c, None)
        python_pytree._deregister_pytree_node(Data)

    def test_register_dataclass_class(self):
        # [FUNCTION DOC] Checks register_dataclass on a plain non-dataclass class: it raises
        # ValueError when field_names is missing, and works when field_names is given so
        # tree_map maps its tensor attributes. Always deregisters the class in a finally block.
        class CustomClass:
            # [CLASS DOC] HELPER: Plain class holding two attributes.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

        with self.assertRaisesRegex(ValueError, "field_names must be specified"):
            python_pytree.register_dataclass(CustomClass)

        python_pytree.register_dataclass(CustomClass, field_names=["x", "y"])
        try:
            c = CustomClass(torch.tensor(0), torch.tensor(1))
            mapped = python_pytree.tree_map(lambda x: x + 1, c)
            self.assertEqual(mapped.x, torch.tensor(1))
            self.assertEqual(mapped.y, torch.tensor(2))
        finally:
            python_pytree._deregister_pytree_node(CustomClass)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/182645")
    def test_constant(self):
        # [FUNCTION DOC] Checks register_constant on a hashable dataclass: flattening it yields
        # no leaves and the spec context value equals the original object. Skipped under Dynamo
        # due to a tracked issue.
        # Either use `frozen=True` or `unsafe_hash=True` so we have a
        # non-default `__hash__`.
        @dataclass(unsafe_hash=True)
        class Config:
            # [CLASS DOC] HELPER: Dataclass with a string field norm, with a generated hash from
            # the unsafe_hash option.
            norm: str

        python_pytree.register_constant(Config)

        config = Config("l1")
        elements, spec = python_pytree.tree_flatten(config)
        self.assertEqual(elements, [])
        self.assertEqual(spec.context.value, config)

    def test_constant_default_eq_error(self):
        # [FUNCTION DOC] Checks register_constant raises TypeError when the class has the
        # default equality. The test body uses a try and except form: the assertion that fails
        # if no error is raised is placed in the try, and the message text is checked in the
        # handler.
        # KNOWN ISSUE: the sentinel failing assertion inside the try raises AssertionError,
        #   which is not caught by the TypeError handler, so it surfaces correctly; but the
        #   check relies on that implicit behavior.
        class Config:
            # [CLASS DOC] HELPER: Class with a norm attribute and only default equality and
            # hash.
            def __init__(self, norm: str):
                # [FUNCTION DOC] HELPER: Stores the norm argument as attribute norm.
                self.norm = norm

        try:
            python_pytree.register_constant(Config)
            self.assertFalse(True)  # must raise error before this
        except TypeError as e:
            msg = "register_constant(cls) expects `cls` to have a non-default `__eq__` implementation."
            self.assertIn(msg, str(e))

    def test_constant_default_hash_error(self):
        # [FUNCTION DOC] Checks register_constant raises TypeError when the class defines
        # equality but keeps the default hash, with the error message naming the hash
        # requirement.
        # KNOWN ISSUE: same try and except sentinel pattern as the default-equality test; a
        #   missing error surfaces only as an AssertionError.
        class Config:
            # [CLASS DOC] HELPER: Class with a custom equality comparing norm but no custom
            # hash.
            def __init__(self, norm: str):
                # [FUNCTION DOC] HELPER: Stores the norm argument as attribute norm.
                self.norm = norm

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equality comparing the norm attribute of both objects;
                # does not guard against non-matching types.
                return self.norm == other.norm

        try:
            python_pytree.register_constant(Config)
            self.assertFalse(True)  # must raise error before this
        except TypeError as e:
            msg = "register_constant(cls) expects `cls` to have a non-default `__hash__` implementation."
            self.assertIn(msg, str(e))

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/184507")
    def test_tree_map_with_path_multiple_trees(self):
        # [FUNCTION DOC] Registers a custom dataclass node with a flatten-with-keys function,
        # then checks tree_map_with_path over two trees with a two-argument function gives the
        # same result as tree_map adding the constant, treating the second tree as all twos.
        # Skipped under Dynamo due to a tracked issue.
        @dataclass
        class ACustomPytree:
            # [CLASS DOC] HELPER: Dataclass with three untyped fields x, y and z; z is used as
            # the node context.
            x: Any
            y: Any
            z: Any

        tree1 = [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5]
        tree2 = [ACustomPytree(x=2, y={"cin": [2, 2, 2], "bar": 2}, z="leaf"), 2]

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )
        from_two_trees = python_pytree.tree_map_with_path(
            lambda kp, a, b: a + b, tree1, tree2
        )
        from_one_tree = python_pytree.tree_map(lambda a: a + 2, tree1)
        self.assertEqual(from_two_trees, from_one_tree)

    def test_tree_flatten_with_path_is_leaf(self):
        # [FUNCTION DOC] Checks tree_flatten_with_path honors is_leaf: a dict inside nested
        # containers treated as a leaf is returned as the same object in the last key-path
        # entry.
        leaf_dict = {"foo": [(3)]}
        tree = (["hello", [1, 2], leaf_dict],)
        key_leaves, _ = python_pytree.tree_flatten_with_path(
            tree, is_leaf=lambda x: isinstance(x, dict)
        )
        self.assertTrue(key_leaves[-1][1] is leaf_dict)

    def test_tree_flatten_with_path_roundtrip(self):
        # [FUNCTION DOC] Round trip test: flatten with paths several trees (containing None,
        # nested containers, a namedtuple with a tensor, and a registered custom dataclass node)
        # and unflatten the leaf values with the returned spec; the result must equal the input.
        # Registers a custom node as side effect.
        class ANamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with tensor x, int y and string z.
            x: torch.Tensor
            y: int
            z: str

        @dataclass
        class ACustomPytree:
            # [CLASS DOC] HELPER: Dataclass with three untyped fields x, y, z; registered as a
            # pytree node with z as context.
            x: Any
            y: Any
            z: Any

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )

        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]
        for tree in SOME_PYTREES:
            key_leaves, spec = python_pytree.tree_flatten_with_path(tree)
            actual = python_pytree.tree_unflatten(
                [leaf for _, leaf in key_leaves], spec
            )
            self.assertEqual(actual, tree)

    def test_tree_leaves_with_path(self):
        # [FUNCTION DOC] Checks tree_leaves_with_path returns the same list as the first output
        # of tree_flatten_with_path for the same example trees as the round trip test, including
        # a registered custom node.
        class ANamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with tensor x, int y and string z.
            x: torch.Tensor
            y: int
            z: str

        @dataclass
        class ACustomPytree:
            # [CLASS DOC] HELPER: Dataclass with three untyped fields x, y, z; registered as a
            # pytree node with z as context.
            x: Any
            y: Any
            z: Any

        python_pytree.register_pytree_node(
            ACustomPytree,
            flatten_fn=lambda f: ([f.x, f.y], f.z),
            unflatten_fn=lambda xy, z: ACustomPytree(xy[0], xy[1], z),
            flatten_with_keys_fn=lambda f: ((("x", f.x), ("y", f.y)), f.z),
        )

        SOME_PYTREES = [
            (None,),
            ["hello", [1, 2], {"foo": [(3)]}],
            [ANamedTuple(x=torch.rand(2, 3), y=1, z="foo")],
            [ACustomPytree(x=12, y={"cin": [1, 4, 10], "bar": 18}, z="leaf"), 5],
        ]
        for tree in SOME_PYTREES:
            flat_out, _ = python_pytree.tree_flatten_with_path(tree)
            leaves_out = python_pytree.tree_leaves_with_path(tree)
            self.assertEqual(flat_out, leaves_out)

    def test_key_str(self):
        # [FUNCTION DOC] Checks keystr formatting of key paths from a nested tuple, list, dict
        # and namedtuple tree: index entries print in brackets, string keys print quoted in
        # brackets, namedtuple fields print after a dot.
        class ANamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with string x and int y.
            x: str
            y: int

        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        flat, _ = python_pytree.tree_flatten_with_path(tree)
        paths = [f"{python_pytree.keystr(kp)}: {val}" for kp, val in flat]
        self.assertEqual(
            paths,
            [
                "[0][0]: hello",
                "[0][1][0]: 1",
                "[0][1][1]: 2",
                "[0][2]['foo'][0]: 3",
                "[0][2]['bar'][0].x: baz",
                "[0][2]['bar'][0].y: 10",
            ],
        )

    def test_flatten_flatten_with_key_consistency(self):
        # [FUNCTION DOC] For each supported node type with an example in a local table, checks
        # that flattening with paths and plain flattening produce the same leaf values and equal
        # specs. Types without an example are skipped.
        """Check that flatten and flatten_with_key produces consistent leaves/context."""
        reg = python_pytree.SUPPORTED_NODES

        EXAMPLE_TREE = {
            list: [1, 2, 3],
            tuple: (1, 2, 3),
            dict: {"foo": 1, "bar": 2},
            namedtuple: namedtuple("ANamedTuple", ["x", "y"])(1, 2),
            OrderedDict: OrderedDict([("foo", 1), ("bar", 2)]),
            defaultdict: defaultdict(int, {"foo": 1, "bar": 2}),
            deque: deque([1, 2, 3]),
            torch.Size: torch.Size([1, 2, 3]),
            immutable_dict: immutable_dict({"foo": 1, "bar": 2}),
            immutable_list: immutable_list([1, 2, 3]),
        }

        for typ in reg:
            example = EXAMPLE_TREE.get(typ)
            if example is None:
                continue
            flat_with_path, spec1 = python_pytree.tree_flatten_with_path(example)
            flat, spec2 = python_pytree.tree_flatten(example)

            self.assertEqual(flat, [x[1] for x in flat_with_path])
            self.assertEqual(spec1, spec2)

    def test_key_access(self):
        # [FUNCTION DOC] Checks key_get returns, for every key path from tree_flatten_with_path,
        # the same leaf value found in the tree.
        class ANamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with string x and int y.
            x: str
            y: int

        tree = (["hello", [1, 2], {"foo": [(3)], "bar": [ANamedTuple(x="baz", y=10)]}],)
        flat, _ = python_pytree.tree_flatten_with_path(tree)
        for kp, val in flat:
            self.assertEqual(python_pytree.key_get(tree, kp), val)


class TestCxxPytree(TestCase):
    # [CLASS DOC] Test case for the C++ (optree-backed) pytree module. setUp skips everything in
    # fbcode. Provides a custom assertEqual that handles treespec comparison under Dynamo.
    # Registers a namedtuple through the Python module helper and custom types on the C++
    # module.
    def setUp(self):
        # [FUNCTION DOC] Skips the test with SkipTest when running under fbcode; otherwise
        # defers to the parent setUp.
        super().setUp()
        if IS_FBCODE:
            raise unittest.SkipTest("C++ pytree tests are not supported in fbcode")

    def assertEqual(self, x, y, *args, **kwargs):
        # [FUNCTION DOC] Overrides equality assertion. For non-treespec values it asserts plain
        # equality. It then also compares the type names, and for treespecs under Dynamo with
        # differing classes compares the trees obtained by unflattening a range of leaves.
        # Passes extra arguments through to the parent.
        # KNOWN ISSUE: no early return after the non-treespec branch, so plain values fall
        #   through and are compared a second time (harmless redundancy).
        x_typename, y_typename = type(x).__name__, type(y).__name__
        if not ("treespec" in x_typename.lower() or "treespec" in y_typename.lower()):
            super().assertEqual(x, y, *args, **kwargs)

        # The Dynamo polyfill returns a polyfilled Python class for C++ PyTreeSpec instead of the
        # C++ class. So we compare the type names and reprs instead because the types themselves
        # won't be equal.
        super().assertEqual(x_typename, y_typename, *args, **kwargs)
        if not TEST_WITH_TORCHDYNAMO or type(x) is type(y):
            super().assertEqual(x, y, *args, **kwargs)
        else:
            super().assertEqual(
                x.unflatten(range(x.num_leaves)),
                y.unflatten(range(y.num_leaves)),
                *args,
                **kwargs,
            )

    def test_treespec_equality(self):
        # [FUNCTION DOC] Checks two C++ leaf specs compare equal.
        self.assertEqual(cxx_pytree.treespec_leaf(), cxx_pytree.treespec_leaf())

    def test_treespec_repr(self):
        # [FUNCTION DOC] Checks the repr of a nested C++ treespec equals the expected PyTreeSpec
        # string including the none-is-leaf marker and the torch namespace.
        # Check that it looks sane
        tree = (0, [0, 0, [0]])
        spec = cxx_pytree.tree_structure(tree)
        self.assertEqual(
            repr(spec), "PyTreeSpec((*, [*, *, [*]]), NoneIsLeaf, namespace='torch')"
        )

    @parametrize(
        "spec",
        [
            cxx_pytree.tree_structure([]),
            cxx_pytree.tree_structure(()),
            cxx_pytree.tree_structure({}),
            cxx_pytree.tree_structure([0]),
            cxx_pytree.tree_structure([0, 1]),
            cxx_pytree.tree_structure((0, 1, 2)),
            cxx_pytree.tree_structure({"a": 0, "b": 1, "c": 2}),
            cxx_pytree.tree_structure(
                OrderedDict([("a", (0, 1)), ("b", 2), ("c", {"a": 3, "b": 4, "c": 5})])
            ),
            cxx_pytree.tree_structure([(0, 1, [2, 3])]),
            cxx_pytree.tree_structure(
                defaultdict(list, {"a": [0, 1], "b": [1, 2], "c": {}})
            ),
        ],
    )
    def test_pytree_serialize(self, spec):
        # [FUNCTION DOC] Parametrized over C++ structures (empty and populated lists, tuples,
        # dicts, OrderedDict and defaultdict). Verifies each spec equals the structure of its
        # unflattened tree, then that dumps returns a string and loads gives an equal spec.
        self.assertEqual(
            spec,
            cxx_pytree.tree_structure(
                cxx_pytree.tree_unflatten([0] * spec.num_leaves, spec)
            ),
        )

        serialized_spec = cxx_pytree.treespec_dumps(spec)
        self.assertIsInstance(serialized_spec, str)

        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

    def test_pytree_serialize_namedtuple(self):
        # [FUNCTION DOC] Registers a module-level and a function-local namedtuple through the
        # Python helper with serialized names, then checks that dumping and loading the C++
        # treespec of each keeps the namedtuple field names. Side effect: persistent registry
        # entries.
        python_pytree._register_namedtuple(
            GlobalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.GlobalPoint",
        )
        spec = cxx_pytree.tree_structure(GlobalPoint(0, 1))

        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

        LocalPoint = namedtuple("LocalPoint", ["x", "y"])
        python_pytree._register_namedtuple(
            LocalPoint,
            serialized_type_name="test_pytree.test_pytree_serialize_namedtuple.LocalPoint",
        )
        spec = cxx_pytree.tree_structure(LocalPoint(0, 1))

        roundtrip_spec = cxx_pytree.treespec_loads(cxx_pytree.treespec_dumps(spec))
        self.assertEqual(roundtrip_spec.type._fields, spec.type._fields)

    def test_pytree_custom_type_serialize(self):
        # [FUNCTION DOC] Checks C++ treespec dump and load round trips for the module-level
        # custom type, registered at import time, and for a locally defined custom type
        # registered inside the test with a serialized name. Both must reload to an equal spec.
        spec = cxx_pytree.tree_structure(GlobalDummyType(0, 1))
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)

        class LocalDummyType:
            # [CLASS DOC] HELPER: Local class with attributes x and y, value equality and hash;
            # registered as a C++ pytree node in this test.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores the arguments as attributes x and y.
                self.x = x
                self.y = y

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Value equality of x and y; returns NotImplemented for
                # other types.
                if not isinstance(other, LocalDummyType):
                    return NotImplemented
                return self.x == other.x and self.y == other.y

            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hash of the pair of x and y.
                return hash((self.x, self.y))

        cxx_pytree.register_pytree_node(
            LocalDummyType,
            lambda dummy: ([dummy.x, dummy.y], None),
            lambda xs, _: LocalDummyType(*xs),
            serialized_type_name="LocalDummyType",
        )
        spec = cxx_pytree.tree_structure(LocalDummyType(0, 1))
        serialized_spec = cxx_pytree.treespec_dumps(spec)
        roundtrip_spec = cxx_pytree.treespec_loads(serialized_spec)
        self.assertEqual(roundtrip_spec, spec)


instantiate_parametrized_tests(TestGenericPytree)
instantiate_parametrized_tests(TestPythonPytree)
instantiate_parametrized_tests(TestCxxPytree)


if __name__ == "__main__":
    run_tests()
