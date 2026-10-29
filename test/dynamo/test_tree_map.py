# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that torch.compile (eager backend, full graph) traces pytree
# tree_map, tree_map_only and tree_map_with_path correctly for the python pytree module
# and, when available, the C++ pytree wrapper and native optree. Covers multiple trees,
# is_leaf predicates, none_is_leaf and namespace kwargs, namedtuples, user-defined
# leaves, custom registered node types, and treespec equality/hashing. Import-time side
# effects: builds the pytree_modules mapping and optional cxx_pytree and optree globals
# based on python_pytree._cxx_pytree_dynamo_traceable.

from collections import namedtuple
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch._dynamo
import torch.utils._pytree as python_pytree
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)


def _tensor_leaf(*values):
    # [FUNCTION DOC] HELPER: Leaf combiner for tensors. Clones the first value and adds every
    # other positional value to it, returning a new tensor without mutating the inputs.
    first = values[0].clone()
    for other in values[1:]:
        first = first + other
    return first


def _combine_leaves(*values):
    # [FUNCTION DOC] HELPER: Leaf combiner used by the multi-tree tests. Tensors are summed
    # via _tensor_leaf, None stays None, tuples (when they are treated as leaves) return the
    # first tuple unchanged so specs stay aligned, and other values are added together.
    first = values[0]
    if isinstance(first, torch.Tensor):
        return _tensor_leaf(*values)
    if first is None:
        return None
    if isinstance(first, tuple):
        # When tuples are marked as leaves, keep the structure from
        # the leading tree so that specs remain aligned.
        return first
    total = first
    for other in values[1:]:
        total = total + other
    return total


def _tuple_is_leaf(node):
    # [FUNCTION DOC] HELPER: is_leaf predicate that returns True only for tuple nodes.
    return isinstance(node, tuple)


pytree_modules = {
    "python": python_pytree,
}
if python_pytree._cxx_pytree_dynamo_traceable:
    import torch.utils._cxx_pytree as cxx_pytree

    pytree_modules["cxx"] = cxx_pytree
    pytree_modules["native_optree"] = cxx_pytree.optree
else:
    cxx_pytree = None

optree = cxx_pytree.optree if cxx_pytree is not None else None


def _require_optree(test_case):
    # [FUNCTION DOC] HELPER: Skips the given test case via skipTest when the optree
    # package is unavailable (module global optree is None); otherwise does nothing.
    if optree is None:
        test_case.skipTest("optree is unavailable")


parametrize_pytree_module = parametrize(
    "pytree_name,pytree",
    [subtest((name, module), name=name) for name, module in pytree_modules.items()],
)

_PYTREE_MODULES_WITH_PATH = {"python"}

parametrize_pytree_module_with_path = parametrize(
    "pytree_name,pytree",
    [
        subtest((name, module), name=name)
        for name, module in pytree_modules.items()
        if name in _PYTREE_MODULES_WITH_PATH
    ],
)


KWARG_CASES = [
    ("default", {}, None),
    ("none_is_leaf", {"none_is_leaf": True}, {"native_optree"}),
    ("is_leaf", {"is_leaf": _tuple_is_leaf}, None),
    ("namespace", {"namespace": "torch"}, {"native_optree"}),
    (
        "namespace_and_none_is_leaf",
        {"namespace": "torch", "none_is_leaf": True},
        {"native_optree"},
    ),
    (
        "namespace_none_is_leaf_predicate",
        {"namespace": "torch", "none_is_leaf": True, "is_leaf": _tuple_is_leaf},
        {"native_optree"},
    ),
]


_NONE_IS_LEAF_UNSET = object()


def _build_tree(offset: int) -> dict[str, object]:
    # [FUNCTION DOC] HELPER: Builds a nested dict test tree of float tensors
    # shifted by offset. Keys: tensor, list (tensor plus a dict holding a
    # tensor and None), tuple (an int and a nested tuple of a tensor and None),
    # const_dict (dict with one tensor). Returns the dict.
    base = torch.arange(4, dtype=torch.float32).reshape(2, 2) + offset
    nested = base + 5
    return {
        "tensor": base,
        "list": [
            base + 1,
            {
                "inner": base + 2,
                "none": None,
            },
        ],
        "tuple": (3 + offset, (nested, None)),
        "const_dict": {"leaf": base + 3},
    }


def _assert_trees_allclose(test_case: TestCase, ref, res) -> None:
    # [FUNCTION DOC] HELPER: Asserts two pytrees match. Flattens both with the python pytree,
    # requires equal treespecs, then compares leaves pairwise: tensors via allclose, other
    # values via assertEqual. Fails the given test_case on any mismatch; returns None.
    ref_flat, ref_spec = python_pytree.tree_flatten(ref)
    res_flat, res_spec = python_pytree.tree_flatten(res)
    test_case.assertEqual(ref_spec, res_spec)
    for expected, actual in zip(ref_flat, res_flat):
        if isinstance(expected, torch.Tensor):
            test_case.assertTrue(torch.allclose(expected, actual))
        else:
            test_case.assertEqual(expected, actual)


@instantiate_parametrized_tests
class TreeMapCompileTests(TestCase):
    # [CLASS DOC] Compile tests for tree_map-style APIs. Each test runs the
    # function eagerly and under torch.compile (eager backend, fullgraph) and
    # compares results. Parametrized over the available pytree implementations.
    # setUp resets dynamo so tests do not share caches.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, then resets torch._dynamo state.
        super().setUp()
        torch._dynamo.reset()

    def _run_tree_map(self, pytree, kwargs):
        # [FUNCTION DOC] HELPER: Builds two trees (offsets 0 and 7), defines a
        # function applying the given pytree module's tree_map with
        # _combine_leaves and extra kwargs, compiles it with fullgraph, and
        # asserts compiled output equals eager output via _assert_trees_allclose.
        lhs = _build_tree(0)
        rhs = _build_tree(7)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Applies tree_map with the
            # combiner to two trees, forwarding kwargs.
            return pytree.tree_map(_combine_leaves, a, b, **kwargs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(lhs, rhs)
        result = compiled(lhs, rhs)
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module
    @parametrize("kwargs_name,kwargs,allowed_impls", KWARG_CASES)
    def test_tree_map_variants(
        self,
        pytree_name: str,
        pytree,
        kwargs_name: str,
        kwargs: dict,
        allowed_impls,
    ) -> None:
        # [FUNCTION DOC] Parametrized over pytree implementation and kwargs case
        # (default, none_is_leaf, is_leaf, namespace and combinations). Skips
        # when the implementation is not in the case's allowed set; otherwise
        # runs _run_tree_map to compare eager and compiled.
        if allowed_impls is not None and pytree_name not in allowed_impls:
            self.skipTest("kwargs unsupported for implementation")
        self._run_tree_map(pytree, kwargs)

    def test_tree_map_rejects_mismatched_container_types(self) -> None:
        # [FUNCTION DOC] Mapping over a list and a tuple together must raise
        # ValueError in eager mode; the compiled version must raise either
        # ValueError or dynamo Unsupported mentioning a node type mismatch.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: tree_map adding two trees leafwise.
            return python_pytree.tree_map(lambda u, v: u + v, a, b)

        lhs = [torch.ones(2), torch.ones(2)]
        rhs = (torch.ones(2), torch.ones(2))

        with self.assertRaises(ValueError):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (ValueError, torch._dynamo.exc.Unsupported),
            "Node type mismatch",
        ):
            compiled(lhs, rhs)

    def test_tree_map_is_leaf_handles_tensor_nodes(self) -> None:
        # [FUNCTION DOC] With an is_leaf predicate matching tuples, a list of one tuple of
        # tensors is mapped by stacking and summing the tuple; compiled result must equal
        # eager.
        def fn(tree):
            # [FUNCTION DOC] HELPER: tree_map that stacks and
            # sums each tuple treated as a leaf.
            return python_pytree.tree_map(
                lambda pair: torch.stack(pair).sum(dim=0),
                tree,
                is_leaf=lambda node: isinstance(node, tuple),
            )

        tree = [(torch.ones(2), torch.ones(2) * 4)]
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_cxx_treespec_namespace_wildcard_eq_hash(self) -> None:
        # [FUNCTION DOC] Checks C++ treespec handling of the optree empty-namespace wildcard
        # under compile: a leaf spec from tree_structure and one from treespec_leaf compare
        # equal, hash equal, and are interchangeable as mapping keys. Skipped without optree.
        # optree treats an empty namespace as a wildcard: tree_structure uses
        # namespace 'torch' while treespec_leaf uses '', yet the leaf specs
        # compare equal.  With namespace excluded from the hash they also hash
        # equal, so a spec looked up across namespaces is found.
        _require_optree(self)

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a tuple of equality, hash
            # equality, and a mapping lookup result for the two leaf specs.
            a = cxx_pytree.tree_structure(x)  # leaf, namespace='torch'
            b = cxx_pytree.treespec_leaf()  # leaf, namespace=''
            return a == b, hash(a) == hash(b), {a: 1}.get(b)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.randn(3)), (True, True, 1))

    def test_cxx_treespec_dict_is_hashable(self) -> None:
        # [FUNCTION DOC] Checks that a dict treespec (whose metadata is
        # unhashable) can be hashed under compile and used as a mapping key, with
        # lookup by an equal spec returning 1. Skipped without optree.
        # A dict treespec carries optree's unhashable DictMetaData; hashing it
        # under compile must trace the spec's custom __hash__ (which hashes the
        # keys), not the frozen-dataclass field hash that would choke on it.
        _require_optree(self)

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dict spec as a
            # key, then looks it up with an equal spec.
            spec = cxx_pytree.tree_structure({"a": x, "b": x})
            return {spec: 1}[cxx_pytree.tree_structure({"a": x, "b": x})]

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.randn(3)), 1)

    def test_cxx_treespec_ordereddict_defaultdict_hashable(self) -> None:
        # [FUNCTION DOC] Checks OrderedDict and defaultdict treespecs hash and compare
        # correctly under compile: lookups succeed, defaultdicts with different factories
        # are separate keys, and dict-family specs of different types are unequal. Expects
        # the result tuple 1, 2, 3, 2, False, False. Skipped without optree.
        # OrderedDict and defaultdict specs also carry unhashable optree metadata
        # (defaultdict's includes the default factory), so they must hash through
        # the spec's custom __hash__ too.  defaultdicts with different factories
        # are distinct keys (metadata differs) even though they hash alike;
        # dict-family types stay unequal.
        _require_optree(self)
        from collections import defaultdict, OrderedDict

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds several dict-family specs and returns lookup
            # results, the number of distinct keys, and two inequality checks.
            od = cxx_pytree.tree_structure(OrderedDict(a=x, b=x))
            dd_int = cxx_pytree.tree_structure(defaultdict(int, {"a": x, "b": x}))
            dd_list = cxx_pytree.tree_structure(defaultdict(list, {"a": x, "b": x}))
            factories = {dd_int: 2, dd_list: 3}
            return (
                {od: 1}[cxx_pytree.tree_structure(OrderedDict(a=x, b=x))],
                factories[
                    cxx_pytree.tree_structure(defaultdict(int, {"a": x, "b": x}))
                ],
                factories[
                    cxx_pytree.tree_structure(defaultdict(list, {"a": x, "b": x}))
                ],
                len(factories),  # int/list factories -> distinct keys despite same keys
                dd_int == dd_list,
                od == cxx_pytree.tree_structure({"a": x, "b": x}),
            )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(torch.randn(3)), (1, 2, 3, 2, False, False))

    def test_tree_map_only_applies_to_tensor_nodes(self) -> None:
        # [FUNCTION DOC] tree_map_only restricted to tensors must apply
        # the mapper only to tensor leaves (the mapper raises
        # AssertionError otherwise); compiled output must equal eager.
        tree = {"tensor": torch.ones(2), "int": 3}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Adds 2 to a tensor; raises
            # AssertionError if given a non-tensor.
            if not isinstance(node, torch.Tensor):
                raise AssertionError("mapper should only see tensors")
            return node + 2

        def fn(arg):
            # [FUNCTION DOC] HELPER: Runs tree_map_only for tensors on the argument.
            return python_pytree.tree_map_only(torch.Tensor, mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_only_multiple_trees_falls_back(self) -> None:
        # [FUNCTION DOC] tree_map_only given two trees is invalid: eager
        # raises TypeError mentioning callable; compiled must raise
        # TypeError or dynamo Unsupported with a matching message.
        lhs = {"a": torch.ones(2), "b": torch.ones(2) * 2}
        rhs = {"a": torch.ones(2) * 3, "b": torch.ones(2) * 4}

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls tree_map_only with
            # two trees, which is not supported.
            return python_pytree.tree_map_only(torch.Tensor, lambda x, y: x + y, a, b)

        with self.assertRaisesRegex(TypeError, "callable"):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (TypeError, torch._dynamo.exc.Unsupported),
            r"(callable|Unsupported function call)",
        ):
            compiled(lhs, rhs)

    def test_tree_map_only_handles_multiple_types(self) -> None:
        # [FUNCTION DOC] tree_map_only with a tuple of the types int and
        # tuple applies the mapper to ints and tuples only, leaving
        # tensors alone; compiled result must equal eager.
        tree = {"int": 7, "tuple": (1, 2), "tensor": torch.ones(2)}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Adds 1 to ints, adds 10 to each element of
            # tuples, and raises AssertionError for any other node type.
            if isinstance(node, int):
                return node + 1
            if isinstance(node, tuple):
                return tuple(val + 10 for val in node)
            raise AssertionError("unexpected node passed to mapper")

        def fn(arg):
            # [FUNCTION DOC] HELPER: Runs tree_map_only with the int and tuple type selector.
            return python_pytree.tree_map_only((int, tuple), mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_is_leaf_non_constant_fallback(self) -> None:
        # [FUNCTION DOC] An is_leaf predicate depending on a runtime tensor value (not
        # constant foldable) must still compile with fullgraph and match eager output.
        tree = {"a": torch.arange(2.0), "b": torch.arange(2.0) + 1}

        def is_leaf(node):
            # [FUNCTION DOC] HELPER: For tensors returns whether the sum exceeds 1 as a
            # Python value read from the tensor; for other nodes returns False.
            if isinstance(node, torch.Tensor):
                # Depends on runtime tensor value; cannot be folded to a constant.
                return (node.sum() > 1).item()
            return False

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles tensors; returns other nodes unchanged.
            return node * 2 if isinstance(node, torch.Tensor) else node

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map with the mapper and the data-dependent is_leaf.
            return python_pytree.tree_map(mapper, arg, is_leaf=is_leaf)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_only_predicate_selector_skips_fastpath(self) -> None:
        # [FUNCTION DOC] tree_map_only with a predicate function as selector (rather
        # than a type) must work under compile and match eager output.
        tree = {"keep": torch.ones(2), "other": (1, 2)}

        def selector(node):
            # [FUNCTION DOC] HELPER: Selects tensors whose
            # shape is a single dimension of size 2.
            return isinstance(node, torch.Tensor) and node.shape == (2,)

        def mapper(node):
            # [FUNCTION DOC] HELPER: Adds 5 to tensors; returns other nodes unchanged.
            return node + 5 if isinstance(node, torch.Tensor) else node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Runs tree_map_only with the predicate selector.
            return python_pytree.tree_map_only(selector, mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    def test_tree_map_none_nodes_reject_mismatched_siblings(self) -> None:
        # [FUNCTION DOC] With optree, mapping two trees where one has a None node
        # and the other a tensor at the same key must raise ValueError mentioning
        # None in eager mode; compiled must raise ValueError or dynamo Unsupported
        # with a similar message. Skipped without optree.
        _require_optree(self)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: optree tree_map pairing leaves of two trees into tuples.
            return optree.tree_map(lambda u, v: (u, v), a, b)

        lhs = {"k": None}
        rhs = {"k": torch.ones(2)}

        with self.assertRaisesRegex(ValueError, "Expected None"):
            fn(lhs, rhs)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaisesRegex(
            (ValueError, torch._dynamo.exc.Unsupported),
            r"(Expected None|expected <class 'NoneType'>)",
        ):
            compiled(lhs, rhs)

    @parametrize_pytree_module
        # [FUNCTION DOC] Parametrized over pytree implementations. For python and cxx pytree,
        # None is a leaf by default so mapping two trees holding None yields a pair of None;
        # compiled must equal eager. Skipped for native optree, which treats None as a node.
    def test_tree_map_none_nodes_default_behavior(
        self, pytree_name: str, pytree
    ) -> None:
        if pytree_name == "native_optree":
            self.skipTest("optree treats None as an internal node by default")

        def fn(a, b):
            # [FUNCTION DOC] HELPER: tree_map pairing leaves of two trees into tuples.
            return pytree.tree_map(lambda u, v: (u, v), a, b)

        tree = {"k": None}
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree, tree)
        result = compiled(tree, tree)

        self.assertEqual(result["k"], (None, None))
        self.assertEqual(result, expected)

    def test_constantvariable_handles_none_is_leaf_kwarg(self) -> None:
        # [FUNCTION DOC] Checks the none_is_leaf kwarg is handled as a constant under compile:
        # unset and False leave None unvisited, True makes the mapper see None and return the
        # string visited. Compares compiled to eager in each case. Skipped without optree.
        _require_optree(self)

        tree = {"none": None}

        def run_case(none_is_leaf_flag):
            # [FUNCTION DOC] HELPER: Compiles and runs an optree tree_map on a None-containing
            # tree, passing none_is_leaf only if the flag is not the unset sentinel; asserts
            # compiled equals eager and returns the mapped value under the none key.
            def fn(arg):
                # [FUNCTION DOC] HELPER: Maps the tree with optional none_is_leaf kwarg.
                def mapper(node):
                    # [FUNCTION DOC] HELPER: Returns the string
                    # visited for None; other nodes unchanged.
                    if node is None:
                        return "visited"
                    return node

                kwargs = {}
                if none_is_leaf_flag is not _NONE_IS_LEAF_UNSET:
                    kwargs["none_is_leaf"] = none_is_leaf_flag
                return optree.tree_map(mapper, arg, **kwargs)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            expected = fn(tree)
            result = compiled(tree)
            self.assertEqual(result, expected)
            return result["none"]

        self.assertEqual(run_case(_NONE_IS_LEAF_UNSET), None)
        self.assertEqual(run_case(False), None)
        self.assertEqual(run_case(True), "visited")

    def test_constantvariable_handles_python_and_dtype_leaves(self) -> None:
        # [FUNCTION DOC] Checks optree tree_map under compile maps int, string
        # and torch dtype leaves: int plus one, string uppercased, dtype replaced
        # by float64. Compares to eager. Skipped without optree.
        _require_optree(self)

        tree = {
            "int": 7,
            "nested": {"string": "foo", "dtype": torch.float32},
        }

        def fn(arg):
            # [FUNCTION DOC] HELPER: Runs optree tree_map with the mapper on the argument.
            def mapper(node):
                # [FUNCTION DOC] HELPER: Increments ints,
                # uppercases strings, maps dtypes to float64.
                if isinstance(node, int):
                    return node + 1
                if isinstance(node, str):
                    return node.upper()
                if isinstance(node, torch.dtype):
                    return torch.float64
                return node

            return optree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        self.assertEqual(result["int"], 8)
        self.assertEqual(result["nested"]["string"], "FOO")
        self.assertIs(result["nested"]["dtype"], torch.float64)
        self.assertEqual(result, expected)

    @parametrize_pytree_module
    def test_user_defined_object_treated_as_leaf(
        self, pytree_name: str, pytree
    ) -> None:
        # [FUNCTION DOC] Instances of an unregistered user-defined class must be treated as
        # leaves by each pytree implementation: the mapper sees the object, and the compiled
        # result holds a new instance with tripled tensor value; tensors elsewhere get one
        # added.
        """User-defined objects (not registered in pytree) should be treated as leaves."""

        class MyCustomClass:
            # [CLASS DOC] HELPER: Plain class holding a single
            # attribute value; not pytree registered.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores value on the instance.
                self.value = value

        obj1 = MyCustomClass(torch.ones(2))
        tree = {"custom": obj1, "tensor": torch.zeros(2)}

        visited_types = []

        def mapper(node):
            # [FUNCTION DOC] HELPER: Records the node type, triples the value of custom
            # objects, adds one to tensors, returns anything else unchanged.
            visited_types.append(type(node))
            if isinstance(node, MyCustomClass):
                return MyCustomClass(node.value * 3)
            if isinstance(node, torch.Tensor):
                return node + 1
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Runs tree_map with the mapper on the argument.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        # Run eager first to establish expected behavior
        visited_types.clear()
        fn(tree)
        eager_types = visited_types.copy()

        # Run compiled
        visited_types.clear()
        result = compiled(tree)

        # Verify the custom object was visited as a leaf
        self.assertIn(MyCustomClass, eager_types)

        # Verify results match
        self.assertIsInstance(result["custom"], MyCustomClass)
        self.assertTrue(torch.allclose(result["custom"].value, obj1.value * 3))
        self.assertTrue(torch.allclose(result["tensor"], torch.ones(2)))

    @parametrize_pytree_module
    def test_user_defined_object_multiple_trees(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] Mapping two trees that contain unregistered Point objects: the mapper
        # sums coordinates of Point pairs and adds other leaves. Checks the compiled result.
        """User-defined objects should work correctly with multiple input trees."""

        class Point:
            # [CLASS DOC] HELPER: Plain two-coordinate class with attributes x and y.
            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Stores x and y.
                self.x = x
                self.y = y

        tree1 = {"point": Point(1, 2), "val": 10}
        tree2 = {"point": Point(3, 4), "val": 20}

        def mapper(a, b):
            # [FUNCTION DOC] HELPER: Adds coordinates of two Points, else adds the two values.
            if isinstance(a, Point) and isinstance(b, Point):
                return Point(a.x + b.x, a.y + b.y)
            return a + b

        def fn(t1, t2):
            # [FUNCTION DOC] HELPER: Maps two trees with the mapper.
            return pytree.tree_map(mapper, t1, t2)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree1, tree2)

        self.assertIsInstance(result["point"], Point)
        self.assertEqual(result["point"].x, 4)
        self.assertEqual(result["point"].y, 6)
        self.assertEqual(result["val"], 30)

    @parametrize_pytree_module
    def test_dict_subclass_treated_as_leaf(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] A dict subclass not registered with pytree must
        # be treated as a leaf by every implementation: the mapper doubles
        # its values, a plain dict is still traversed.
        """Dict subclasses (not registered in pytree) should be treated as leaves."""

        class MyDict(dict):
            # [CLASS DOC] HELPER: Dict subclass with an extra method summing values.
            def custom_method(self):
                # [FUNCTION DOC] HELPER: Returns the sum of the dict's values.
                return sum(self.values())

        my_dict = MyDict({"a": 1, "b": 2})
        tree = {"custom_dict": my_dict, "regular": {"x": 3}}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Returns a new subclass
            # instance with doubled values; else unchanged.
            if isinstance(node, MyDict):
                result = MyDict()
                for k, v in node.items():
                    result[k] = v * 2
                return result
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # The MyDict should be treated as a leaf, not traversed into
        self.assertIsInstance(result["custom_dict"], MyDict)
        self.assertEqual(result["custom_dict"]["a"], 2)
        self.assertEqual(result["custom_dict"]["b"], 4)
        # Regular dict should still be traversed
        self.assertEqual(result["regular"]["x"], 3)

    @parametrize_pytree_module
    def test_list_subclass_treated_as_leaf(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] A list subclass not registered with pytree must be a leaf: the mapper
        # doubles its elements, while a plain list is traversed and ints get ten added.
        """List subclasses (not registered in pytree) should be treated as leaves."""

        class MyList(list):
            # [CLASS DOC] HELPER: List subclass with an extra method summing elements.
            def custom_sum(self):
                # [FUNCTION DOC] HELPER: Returns the sum of the elements.
                return sum(self)

        my_list = MyList([1, 2, 3])
        tree = {"custom_list": my_list, "regular": [4, 5, 6]}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles elements of the
            # subclass; adds ten to ints; else unchanged.
            if isinstance(node, MyList):
                return MyList([x * 2 for x in node])
            if isinstance(node, int):
                return node + 10
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # MyList should be treated as a leaf
        self.assertIsInstance(result["custom_list"], MyList)
        self.assertEqual(list(result["custom_list"]), [2, 4, 6])
        # Regular list should be traversed
        self.assertEqual(result["regular"], [14, 15, 16])

    @parametrize_pytree_module
    def test_tuple_subclass_treated_as_leaf(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] A tuple subclass not registered with pytree must be a leaf: the
        # mapper doubles its elements, while a plain tuple is traversed and ints get ten
        # added.
        """Tuple subclasses (not registered in pytree) should be treated as leaves."""

        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Empty tuple subclass used as an unregistered leaf type.
            pass

        my_tuple = MyTuple((1, 2, 3))
        tree = {"custom_tuple": my_tuple, "regular": (4, 5, 6)}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles elements of the
            # subclass; adds ten to ints; else unchanged.
            if isinstance(node, MyTuple):
                return MyTuple(tuple(x * 2 for x in node))
            if isinstance(node, int):
                return node + 10
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        # MyTuple should be treated as a leaf
        self.assertIsInstance(result["custom_tuple"], MyTuple)
        self.assertEqual(tuple(result["custom_tuple"]), (2, 4, 6))
        # Regular tuple should be traversed
        self.assertEqual(result["regular"], (14, 15, 16))

    @parametrize_pytree_module
    def test_user_defined_object_nested_in_containers(
        self, pytree_name: str, pytree
    ) -> None:
        # [FUNCTION DOC] Unregistered objects nested inside lists and dicts remain leaves
        # and are replaced by new objects whose value is multiplied by ten.
        """User-defined objects nested inside containers should be leaves."""

        class Wrapper:
            # [CLASS DOC] HELPER: Plain class holding one attribute value.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores value.
                self.value = value

        tree = {
            "list_of_wrappers": [Wrapper(1), Wrapper(2)],
            "nested": {"wrapper": Wrapper(3)},
        }

        def mapper(node):
            # [FUNCTION DOC] HELPER: Returns a new Wrapper
            # with value times ten; else unchanged.
            if isinstance(node, Wrapper):
                return Wrapper(node.value * 10)
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertEqual(result["list_of_wrappers"][0].value, 10)
        self.assertEqual(result["list_of_wrappers"][1].value, 20)
        self.assertEqual(result["nested"]["wrapper"].value, 30)

    @parametrize_pytree_module
    def test_user_defined_object_with_is_leaf_predicate(
        self, pytree_name: str, pytree
    ) -> None:
        # [FUNCTION DOC] An is_leaf predicate marking custom containers and
        # lists as leaves: the mapper doubles the container items and adds
        # ten to list items. Checks the compiled result.
        """Test that is_leaf predicate interacts correctly with user-defined objects."""

        class Container:
            # [CLASS DOC] HELPER: Plain class holding a list in attribute items.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores items.
                self.items = items

        container = Container([1, 2, 3])
        tree = {"container": container, "list": [4, 5]}

        def is_leaf_fn(node):
            # [FUNCTION DOC] HELPER: Returns whether the node is a Container or a list.
            # Make lists be treated as leaves too
            return isinstance(node, (Container, list))

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles container items;
            # adds ten to list items; else unchanged.
            if isinstance(node, Container):
                return Container([x * 2 for x in node.items])
            if isinstance(node, list):
                return [x + 10 for x in node]
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps with the mapper and the is_leaf predicate.
            return pytree.tree_map(mapper, arg, is_leaf=is_leaf_fn)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertEqual(result["container"].items, [2, 4, 6])
        self.assertEqual(result["list"], [14, 15])

    def test_registered_custom_type_falls_back_pytree(self) -> None:
        # [FUNCTION DOC] A class registered with the python pytree is traversed (not a
        # leaf) under compile: tensors inside are incremented and the result is the same
        # registered class. Registration is removed in a finally block.
        """Custom types registered with pytree should fall back to tracing."""

        class RegisteredContainer:
            # [CLASS DOC] HELPER: Plain container class whose items are copied into a list.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores a list copy of items.
                self.items = list(items)

        # Register with pytree
        python_pytree.register_pytree_node(
            RegisteredContainer,
            lambda x: (x.items, None),
            lambda items, _: RegisteredContainer(items),
        )

        try:
            tree = RegisteredContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
            # [FUNCTION DOC] HELPER: Adds one to tensors; else unchanged.
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
            # [FUNCTION DOC] HELPER: Python pytree tree_map with the mapper.
                return python_pytree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since it's registered, the container should be traversed
            self.assertIsInstance(result, RegisteredContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            # Clean up registration
            python_pytree._deregister_pytree_node(RegisteredContainer)

    def test_registered_custom_type_multiple_trees_with_eq(self) -> None:
        # [FUNCTION DOC] A dataclass registered with the python pytree is mapped over two
        # trees under compile, comparing results to eager. Exercises class comparison
        # through the metaclass. Registration is removed in a finally block.
        """Dataclass registrations compare class objects through the metaclass."""

        @dataclass
        class RegisteredContainer:
            # [CLASS DOC] HELPER: Dataclass with two tensor fields, x and y.
            x: torch.Tensor
            y: torch.Tensor

        python_pytree.register_pytree_node(
            RegisteredContainer,
            lambda c: ([c.x, c.y], None),
            lambda children, _: RegisteredContainer(*children),
        )

        try:
            lhs = RegisteredContainer(torch.zeros(2, 3), torch.ones(2, 3))
            rhs = RegisteredContainer(torch.ones(2, 3), torch.zeros(2, 3))

            def fn(a, b):
                # [FUNCTION DOC] HELPER: tree_map adding two trees leafwise.
                return python_pytree.tree_map(lambda x, y: x + y, a, b)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            expected = fn(lhs, rhs)
            result = compiled(lhs, rhs)

            self.assertIsInstance(result, RegisteredContainer)
            self.assertEqual(result.x, expected.x)
            self.assertEqual(result.y, expected.y)
        finally:
            python_pytree._deregister_pytree_node(RegisteredContainer)

    def test_registered_custom_type_falls_back_optree(self) -> None:
        # [FUNCTION DOC] A class registered with optree in its global namespace is
        # traversed under compile. The test looks up the optree registry global
        # namespace constant by several name-mangling fallbacks and skips if not
        # found. Tensors are incremented; unregisters at end.
        """Custom types registered with optree should fall back to tracing."""
        _require_optree(self)
        import optree.registry as optree_registry

        global_namespace = getattr(
            optree_registry, "_TreeMapCompileTests__GLOBAL_NAMESPACE", None
        )
        if global_namespace is None:
            # Handle Python name mangling - the attribute has double underscore prefix
            global_namespace = getattr(optree_registry, "__GLOBAL_NAMESPACE", None)
        if global_namespace is None:
            # Fallback: access via module internals
            for name in dir(optree_registry):
                if "GLOBAL_NAMESPACE" in name:
                    global_namespace = getattr(optree_registry, name)
                    break

        if global_namespace is None:
            self.skipTest("Could not find optree global namespace")

        class OptreeRegisteredContainer:
            # [CLASS DOC] HELPER: Plain container class whose items are copied into a list.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores a list copy of items.
                self.items = list(items)

        # Register with optree in global namespace
        # Note: optree unflatten_func signature is (metadata, children)
        optree.register_pytree_node(
            OptreeRegisteredContainer,
            lambda x: (x.items, None),
            lambda _, children: OptreeRegisteredContainer(children),
            namespace=global_namespace,
        )

        try:
            tree = OptreeRegisteredContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
            # [FUNCTION DOC] HELPER: Adds one to tensors; else unchanged.
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
            # [FUNCTION DOC] HELPER: optree tree_map with the mapper.
                return optree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since it's registered, the container should be traversed
            self.assertIsInstance(result, OptreeRegisteredContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            # Clean up registration
            optree.unregister_pytree_node(
                OptreeRegisteredContainer, namespace=global_namespace
            )

    def test_optree_namespaced_registration_with_namespace_arg(self) -> None:
        # [FUNCTION DOC] A type registered with optree under a named namespace is
        # traversed when tree_map is called with that same namespace: tensors inside
        # are incremented. Unregisters at the end. Skipped without optree.
        """Types registered with a namespace should be traversed when namespace is provided."""
        _require_optree(self)

        class NamespacedContainer:
            # [CLASS DOC] HELPER: Plain container class registered under the test namespace.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores a list copy of items.
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="test_namespace",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            def mapper(node):
            # [FUNCTION DOC] HELPER: Adds one to tensors; else unchanged.
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
            # [FUNCTION DOC] HELPER: optree tree_map using the matching namespace.
                # Call with matching namespace - should traverse
                return optree.tree_map(mapper, arg, namespace="test_namespace")

            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Since namespace matches, the container should be traversed
            self.assertIsInstance(result, NamespacedContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) + 1))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) + 1))
        finally:
            optree.unregister_pytree_node(
                NamespacedContainer, namespace="test_namespace"
            )

    def test_optree_namespaced_registration_without_namespace_arg(self) -> None:
        # [FUNCTION DOC] A type registered under a named namespace is a leaf when tree_map
        # is called without a namespace: the mapper sees the container itself and doubles
        # its tensors. Unregisters at the end. Skipped without optree.
        """Types registered with a namespace should be leaves when no namespace is provided."""
        _require_optree(self)

        class NamespacedContainer:
            # [CLASS DOC] HELPER: Plain container class registered under the test namespace.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores a list copy of items.
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="test_namespace",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            mapper_calls = []

            def mapper(node):
            # [FUNCTION DOC] HELPER: Records node types; doubles
            # container items; adds one to tensors.
                mapper_calls.append(type(node))
                if isinstance(node, NamespacedContainer):
                    # Should be called as a leaf
                    return NamespacedContainer([t * 2 for t in node.items])
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
            # [FUNCTION DOC] HELPER: optree tree_map without a namespace.
                # Call WITHOUT namespace - container should be treated as leaf
                return optree.tree_map(mapper, arg)

            compiled = torch.compile(fn, backend="eager", fullgraph=True)

            # Clear and run
            mapper_calls.clear()
            result = compiled(tree)

            # Container should be treated as a leaf (mapper sees NamespacedContainer)
            self.assertIn(NamespacedContainer, mapper_calls)
            self.assertIsInstance(result, NamespacedContainer)
            # The container was mapped directly, so items are doubled tensors
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) * 2))
            self.assertTrue(torch.allclose(result.items[1], torch.zeros(2) * 2))
        finally:
            optree.unregister_pytree_node(
                NamespacedContainer, namespace="test_namespace"
            )

    def test_optree_namespace_mismatch(self) -> None:
        # [FUNCTION DOC] A type registered under one namespace is a leaf when tree_map is
        # called with a different namespace: the mapper sees the container and doubles
        # its tensors. Unregisters at the end. Skipped without optree.
        """Types registered with one namespace should be leaves when different namespace is used."""
        _require_optree(self)

        class NamespacedContainer:
            # [CLASS DOC] HELPER: Plain container class registered under namespace_a.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Stores a list copy of items.
                self.items = list(items)

        # Register with a specific namespace
        optree.register_pytree_node(
            NamespacedContainer,
            lambda x: (x.items, None),
            lambda _, children: NamespacedContainer(children),
            namespace="namespace_a",
        )

        try:
            tree = NamespacedContainer([torch.ones(2), torch.zeros(2)])

            mapper_calls = []

            def mapper(node):
            # [FUNCTION DOC] HELPER: Records node types; doubles
            # container items; adds one to tensors.
                mapper_calls.append(type(node))
                if isinstance(node, NamespacedContainer):
                    return NamespacedContainer([t * 2 for t in node.items])
                if isinstance(node, torch.Tensor):
                    return node + 1
                return node

            def fn(arg):
            # [FUNCTION DOC] HELPER: optree tree_map with a
            # different namespace than registered.
                # Call with DIFFERENT namespace - container should be leaf
                return optree.tree_map(mapper, arg, namespace="namespace_b")

            compiled = torch.compile(fn, backend="eager", fullgraph=True)

            mapper_calls.clear()
            result = compiled(tree)

            # Container should be treated as a leaf
            self.assertIn(NamespacedContainer, mapper_calls)
            self.assertIsInstance(result, NamespacedContainer)
            self.assertTrue(torch.allclose(result.items[0], torch.ones(2) * 2))
        finally:
            optree.unregister_pytree_node(NamespacedContainer, namespace="namespace_a")

    @parametrize_pytree_module
    def test_dataclass_treated_as_leaf(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] An unregistered dataclass is a leaf for all implementations: the
        # mapper doubles its fields and adds ten to the other leaf. Checks compiled output.
        """Dataclasses should be treated as leaves (not registered by default)."""
        import dataclasses

        @dataclasses.dataclass
        class DataPoint:
            # [CLASS DOC] HELPER: Dataclass with integer fields x and y.
            x: int
            y: int

        tree = {"point": DataPoint(1, 2), "val": 3}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles dataclass fields; adds ten to anything else.
            if isinstance(node, DataPoint):
                return DataPoint(node.x * 2, node.y * 2)
            return node + 10

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertIsInstance(result["point"], DataPoint)
        self.assertEqual(result["point"].x, 2)
        self.assertEqual(result["point"].y, 4)
        self.assertEqual(result["val"], 13)

    @parametrize_pytree_module
    def test_user_defined_object_with_tensor_attribute(
        self, pytree_name: str, pytree
    ) -> None:
        # [FUNCTION DOC] An unregistered object holding a tensor attribute is a leaf: the
        # mapper doubles the held tensor, and a direct tensor leaf gets one added.
        """User-defined objects containing tensors should be treated as leaves."""

        class TensorWrapper:
            # [CLASS DOC] HELPER: Plain class holding a tensor in attribute tensor.
            def __init__(self, tensor):
                # [FUNCTION DOC] HELPER: Stores the tensor.
                self.tensor = tensor

        wrapper = TensorWrapper(torch.ones(2, 2))
        tree = {"wrapper": wrapper, "direct_tensor": torch.zeros(2)}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles the held tensor,
            # adds one to tensors, else unchanged.
            if isinstance(node, TensorWrapper):
                return TensorWrapper(node.tensor * 2)
            if isinstance(node, torch.Tensor):
                return node + 1
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertIsInstance(result["wrapper"], TensorWrapper)
        self.assertTrue(torch.allclose(result["wrapper"].tensor, torch.ones(2, 2) * 2))
        self.assertTrue(torch.allclose(result["direct_tensor"], torch.ones(2)))

    @parametrize_pytree_module
    def test_user_defined_object_no_fallback(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] Verifies user-defined objects take the tree_map fast path: attaches a
        # temporary log handler to the dynamo variables.base logger at debug level, compiles,
        # checks results, and asserts no recorded message reports a fastpath fallback for a
        # UserDefinedObjectVariable. Handler and level are restored in a finally block.
        """Verify user-defined objects use fastpath without triggering fallback."""
        import logging

        class SimpleObj:
            # [CLASS DOC] HELPER: Plain class holding a single attribute x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x

        tree = {"obj": SimpleObj(42), "val": 10}

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles x of SimpleObj; adds one to anything else.
            if isinstance(node, SimpleObj):
                return SimpleObj(node.x * 2)
            return node + 1

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        # Capture debug logs to ensure no fallback is triggered
        log_records = []
        handler = logging.Handler()
        handler.emit = lambda record: log_records.append(record)
        logger = logging.getLogger("torch._dynamo.variables.base")
        old_level = logger.level
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

        try:
            compiled = torch.compile(fn, backend="eager", fullgraph=True)
            result = compiled(tree)

            # Verify results
            self.assertIsInstance(result["obj"], SimpleObj)
            self.assertEqual(result["obj"].x, 84)
            self.assertEqual(result["val"], 11)

            # Verify no fallback was triggered for UserDefinedObjectVariable
            fallback_messages = [
                r.getMessage()
                for r in log_records
                if "tree_map fastpath fallback triggered" in r.getMessage()
                and "UserDefinedObjectVariable" in r.getMessage()
            ]
            self.assertEqual(
                len(fallback_messages),
                0,
                lambda msg: f"{msg}\nFallback was triggered unexpectedly: {fallback_messages}",
            )
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

    @parametrize_pytree_module
    def test_namedtuple_tree_map(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] tree_map over a tree with namedtuples: tensors
        # inside are incremented and the namedtuple type is preserved,
        # for each implementation. Compared against eager.
        """Test tree_map with namedtuple uses fast path."""
        Point = namedtuple("Point", ["x", "y"])

        tree = {
            "point": Point(torch.ones(2), torch.zeros(2)),
            "nested": [Point(torch.ones(3), torch.ones(3) * 2)],
        }

        def mapper(node):
            # [FUNCTION DOC] HELPER: Adds one to tensors; else unchanged.
            if isinstance(node, torch.Tensor):
                return node + 1
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps the argument with the mapper.
            return pytree.tree_map(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)

        self.assertIsInstance(result["point"], Point)
        self.assertTrue(torch.allclose(result["point"].x, torch.ones(2) + 1))
        self.assertTrue(torch.allclose(result["point"].y, torch.zeros(2) + 1))
        self.assertIsInstance(result["nested"][0], Point)
        self.assertTrue(torch.allclose(result["nested"][0].x, torch.ones(3) + 1))
        self.assertTrue(torch.allclose(result["nested"][0].y, torch.ones(3) * 2 + 1))
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module
    def test_namedtuple_tree_map_multiple_trees(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] tree_map over two trees of namedtuples: tensor leaves are summed and
        # the namedtuple type is preserved, for each implementation. Compared against eager.
        """Test tree_map with multiple namedtuple trees."""
        Point = namedtuple("Point", ["x", "y"])

        tree1 = {"point": Point(torch.ones(2), torch.zeros(2))}
        tree2 = {"point": Point(torch.ones(2) * 2, torch.ones(2) * 3)}

        def mapper(a, b):
            # [FUNCTION DOC] HELPER: Sums tensor pairs; otherwise returns the first value.
            if isinstance(a, torch.Tensor):
                return a + b
            return a

        def fn(t1, t2):
            # [FUNCTION DOC] HELPER: Maps two trees with the mapper.
            return pytree.tree_map(mapper, t1, t2)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree1, tree2)
        result = compiled(tree1, tree2)

        self.assertIsInstance(result["point"], Point)
        self.assertTrue(torch.allclose(result["point"].x, torch.ones(2) * 3))
        self.assertTrue(torch.allclose(result["point"].y, torch.ones(2) * 3))
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module
    def test_namedtuple_with_is_leaf(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] tree_map with an is_leaf predicate that marks a NamedTuple as a leaf:
        # the mapper doubles both fields of the tuple and adds ten to other tensors.
        """Test tree_map with namedtuple and is_leaf predicate."""

        class Point(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with fields x and
            # y (annotated as ints, holds tensors).
            x: int
            y: int

        tree = {"point": Point(torch.ones(2), torch.zeros(2)), "val": torch.ones(3)}

        def is_leaf_fn(node):
            # [FUNCTION DOC] HELPER: Returns whether the node is a Point.
            # Treat Point as a leaf
            return isinstance(node, Point)

        def mapper(node):
            # [FUNCTION DOC] HELPER: Doubles Point fields;
            # adds ten to tensors; else unchanged.
            if isinstance(node, Point):
                return Point(node.x * 2, node.y * 2)
            if isinstance(node, torch.Tensor):
                return node + 10
            return node

        def fn(arg):
            # [FUNCTION DOC] HELPER: Maps with the mapper and is_leaf predicate.
            return pytree.tree_map(mapper, arg, is_leaf=is_leaf_fn)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)

        self.assertIsInstance(result["point"], Point)
        self.assertTrue(torch.allclose(result["point"].x, torch.ones(2) * 2))
        self.assertTrue(torch.allclose(result["point"].y, torch.zeros(2) * 2))
        self.assertTrue(torch.allclose(result["val"], torch.ones(3) + 10))
        _assert_trees_allclose(self, expected, result)


@instantiate_parametrized_tests
class TreeMapWithPathCompileTests(TestCase):
    # [CLASS DOC] Compile tests for tree_map_with_path on implementations supporting key
    # paths (only the python pytree module). Each test compares collected key paths and
    # results between eager and compiled runs. setUp resets dynamo.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setUp, then resets torch._dynamo state.
        super().setUp()
        torch._dynamo.reset()

    @parametrize_pytree_module_with_path
    def test_basic_nested_tree(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] Maps a nested dict of tensors, a list and a tuple with key paths: the
        # collected key paths and mapped results must match between eager and compiled runs.
        """Keypaths are correctly constructed for nested dicts, lists, and tuples."""
        tree = {
            "tensor": torch.ones(2),
            "list": [torch.zeros(2), torch.ones(3)],
            "tuple": (torch.ones(4),),
        }

        collected = []

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Records the key path, then adds one to tensors.
            collected.append(kp)
            return x + 1 if isinstance(x, torch.Tensor) else x

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path with the mapper.
            return pytree.tree_map_with_path(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        # Eager reference
        collected.clear()
        expected = fn(tree)
        eager_keypaths = list(collected)

        # Compiled
        collected.clear()
        result = compiled(tree)
        compiled_keypaths = list(collected)

        _assert_trees_allclose(self, expected, result)
        self.assertEqual(eager_keypaths, compiled_keypaths)

    @parametrize_pytree_module_with_path
    def test_multiple_trees(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] tree_map_with_path over two trees: the mapper sums
        # corresponding leaves. Compiled result must equal eager.
        """tree_map_with_path with multiple input trees."""
        tree1 = {"a": torch.ones(2), "b": [torch.zeros(3)]}
        tree2 = {"a": torch.ones(2) * 2, "b": [torch.ones(3) * 3]}

        def mapper(kp, x, y):
            # [FUNCTION DOC] HELPER: Sums two leaves, ignoring the key path.
            return x + y

        def fn(t1, t2):
            # [FUNCTION DOC] HELPER: tree_map_with_path over two trees.
            return pytree.tree_map_with_path(mapper, t1, t2)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree1, tree2)
        result = compiled(tree1, tree2)
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module_with_path
    def test_is_leaf_predicate(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] With an is_leaf predicate marking lists as leaves,
        # the list is passed whole to the mapper with a single key path;
        # asserts one key path in eager and equal to compiled.
        """is_leaf stops traversal and passes the subtree as a leaf."""
        tree = {"a": [torch.ones(2), torch.zeros(2)]}

        def is_leaf_fn(node):
            # [FUNCTION DOC] HELPER: Returns whether the node is a list.
            return isinstance(node, list)

        collected_keypaths = []

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Records the key path;
            # doubles each element of a list leaf.
            collected_keypaths.append(kp)
            if isinstance(x, list):
                return [t * 2 for t in x]
            return x

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path using the is_leaf predicate.
            return pytree.tree_map_with_path(mapper, arg, is_leaf=is_leaf_fn)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        collected_keypaths.clear()
        fn(tree)
        eager_keypaths = list(collected_keypaths)

        collected_keypaths.clear()
        compiled(tree)
        compiled_keypaths = list(collected_keypaths)

        # The list should be treated as a single leaf with keypath (MappingKey("a"),)
        self.assertEqual(len(eager_keypaths), 1)
        self.assertEqual(eager_keypaths, compiled_keypaths)

    @parametrize_pytree_module_with_path
    def test_namedtuple_uses_getattr_key(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] Namedtuple fields are reported with attribute-style keys in key
        # paths: collected paths and results from compiled run must match eager.
        """Namedtuple fields produce GetAttrKey in keypaths."""
        Point = namedtuple("Point", ["x", "y"])
        tree = {"point": Point(torch.ones(2), torch.zeros(2))}

        collected = []

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Records the key path; adds one to tensors.
            collected.append(kp)
            if isinstance(x, torch.Tensor):
                return x + 1
            return x

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path with the mapper.
            return pytree.tree_map_with_path(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        collected.clear()
        expected = fn(tree)
        eager_keypaths = list(collected)

        collected.clear()
        result = compiled(tree)
        compiled_keypaths = list(collected)

        _assert_trees_allclose(self, expected, result)
        self.assertEqual(eager_keypaths, compiled_keypaths)

    @parametrize_pytree_module_with_path
    def test_deeply_nested_keypaths(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] A tree of dict, dict and list nesting gives one leaf whose key
        # path has three entries; compiled key paths and results must match eager.
        """Deeply nested structures produce correct multi-level keypaths."""
        tree = {"outer": {"inner": [torch.ones(2)]}}

        collected = []

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Records the key path; doubles tensors.
            collected.append(kp)
            return x * 2 if isinstance(x, torch.Tensor) else x

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path with the mapper.
            return pytree.tree_map_with_path(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)

        collected.clear()
        expected = fn(tree)
        eager_keypaths = list(collected)

        collected.clear()
        result = compiled(tree)
        compiled_keypaths = list(collected)

        self.assertEqual(len(eager_keypaths), 1)
        self.assertEqual(len(eager_keypaths[0]), 3)  # outer -> inner -> [0]
        self.assertEqual(eager_keypaths, compiled_keypaths)
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module_with_path
    def test_keypath_values_used_in_computation(self, pytree_name: str, pytree) -> None:
        # [FUNCTION DOC] The mapper inspects the last key of the key path (a mapping key) and
        # scales by three for key scale3 else by two. Compiled result must equal eager.
        """The map function can use keypath values to influence the result."""
        from torch.utils._pytree import MappingKey

        tree = {"scale2": torch.ones(2), "scale3": torch.ones(2)}

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Multiplies by three if the last key
            # is the scale3 mapping key, otherwise by two.
            # Use the dict key name from the keypath to determine scaling
            key = kp[-1]
            if isinstance(key, MappingKey) and key.key == "scale3":
                return x * 3
            return x * 2

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path with the mapper.
            return pytree.tree_map_with_path(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        expected = fn(tree)
        result = compiled(tree)
        _assert_trees_allclose(self, expected, result)

    @parametrize_pytree_module_with_path
    def test_user_defined_object_treated_as_leaf(
        self, pytree_name: str, pytree
    ) -> None:
        # [FUNCTION DOC] Unregistered user-defined objects are leaves for
        # tree_map_with_path: the mapper doubles the object's value and
        # adds one to tensors. Checks compiled output.
        """Unregistered user-defined objects are leaves in tree_map_with_path."""

        class MyObj:
            # [CLASS DOC] HELPER: Plain class holding attribute val.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Stores val.
                self.val = val

        tree = {"obj": MyObj(42), "tensor": torch.ones(2)}

        def mapper(kp, x):
            # [FUNCTION DOC] HELPER: Doubles val of MyObj; adds one to anything else.
            if isinstance(x, MyObj):
                return MyObj(x.val * 2)
            return x + 1

        def fn(arg):
            # [FUNCTION DOC] HELPER: tree_map_with_path with the mapper.
            return pytree.tree_map_with_path(mapper, arg)

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        result = compiled(tree)

        self.assertIsInstance(result["obj"], MyObj)
        self.assertEqual(result["obj"].val, 84)
        self.assertTrue(torch.allclose(result["tensor"], torch.ones(2) + 1))


if __name__ == "__main__":  # pragma: no cover
    run_tests()
