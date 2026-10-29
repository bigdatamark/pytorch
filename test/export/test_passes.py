# [MODULE DOC] Tests for export passes and graph utilities: runtime assertion insertion,
# functionalization, effect token removal, auto-functionalized removal, set-grad and autocast
# replacement with higher-order operators, sequential splitting and inlining, constant and
# script-object handling, device movement, view-to-view-copy replacement, and partitioner support.
# Top of file also has a docstring asking pytest not to rewrite assertions. Defines helper
# operator-support classes, model fixtures with torchbind attributes, and factories producing
# pre-dispatch modules for context-manager tests.
"""
PYTEST_DONT_REWRITE (prevents pytest from rewriting assertions, which interferes
with test_functionalization_with_native_python_assertion)
"""

# Owner(s): ["oncall: export"]
import math
import operator
import unittest
from re import escape

import torch
from functorch.experimental.control_flow import cond
from torch._dynamo.eval_frame import is_dynamo_supported
from torch._export import config
from torch._export.non_strict_utils import (
    _fakify_script_objects,
    _gather_constant_attrs,
)
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._export.passes.replace_set_grad_with_hop_pass import (
    _is_set_grad_enabled_node,
    _is_set_grad_enabled_sub_mod,
)
from torch._export.passes.replace_view_ops_with_view_copy_ops_pass import (
    get_view_copy_of_view_op,
    is_view_op,
    ReplaceViewOpsWithViewCopyOpsPass,
)
from torch._export.utils import (
    node_inline_,
    nodes_count,
    nodes_filter,
    nodes_map,
    sequential_split,
)
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.export import export
from torch.export._remove_auto_functionalized_pass import (
    unsafe_remove_auto_functionalized_pass,
)
from torch.export._remove_effect_tokens_pass import _remove_effect_tokens
from torch.export.passes import move_to_device_pass
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.fx.passes.infra.partitioner import Partition
from torch.fx.passes.operator_support import OperatorSupport
from torch.library import _scoped_library, impl
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfTorchDynamo,
    TestCase,
)
from torch.testing._internal.torchbind_impls import init_torchbind_implementations
from torch.utils import _pytree as pytree


def count_call_function(graph: torch.fx.Graph, target: torch.ops.OpOverload) -> int:
    # [FUNCTION DOC] HELPER: Counts the call_function nodes in a graph whose target equals the
    # given operator overload and returns the integer count.
    count = 0
    for node in graph.nodes:
        if node.op == "call_function" and node.target == target:
            count += 1
    return count


class _AddOperatorSupport(OperatorSupport):
    # [CLASS DOC] HELPER: Operator support policy that accepts only call_function nodes whose
    # target is the python add operator.
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Returns true only for call_function nodes calling the python add
        # operator; submodules are ignored.
        return node.op == "call_function" and node.target == operator.add


class _AtenAddOperatorSupport(OperatorSupport):
    # [CLASS DOC] HELPER: Operator support policy that accepts only call_function nodes targeting
    # the aten add Tensor overload.
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Returns true only for call_function nodes targeting aten add
        # Tensor; submodules are ignored.
        return node.op == "call_function" and node.target == torch.ops.aten.add.Tensor


def _to_partition_names(partitions: list[Partition]) -> list[set[str]]:
    # [FUNCTION DOC] HELPER: Converts a list of partitions to a list of sets holding the node names
    # of each partition, so tests can compare partitions independent of order inside a partition.
    return [{n.name for n in p.nodes} for p in partitions]


def _get_output_names(gm: torch.fx.GraphModule) -> list[str]:
    # [FUNCTION DOC] HELPER: Finds the output node of a graph module, flattens its args with pytree
    # and returns the string form of each leaf, in order.
    output_node = next(n for n in gm.graph.nodes if n.op == "output")
    args = pytree.tree_leaves(output_node.args)
    # if isinstance(args, tuple) and len(args) == 1:
    #     args = args[0]
    return [str(arg) for arg in args]


class ModelsWithScriptObjectAttr:
    # [CLASS DOC] HELPER: Namespace holding nested module fixtures that own torchbind script
    # objects as attributes, directly, in containers, and in submodules. Not instantiated itself.
    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a single script object attribute.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the script object attribute with two integer fields.
            super().__init__()
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

    class SimpleWithAttrInContainer(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a script object attribute plus a list attribute holding
        # script objects, one in a set and one in a dict.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the single script object attribute and the list of
            # script objects containing a set and a mapping.
            super().__init__()
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),
                {
                    torch.classes._TorchScriptTesting._Foo(3, 4),
                },
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},
            ]

    class NestedWithAttrInContainer(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a script object attribute, a container attribute of
        # script objects, and two submodules: a simple one and one with container attributes.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the script object attribute, the container attribute
            # and both submodules.
            super().__init__()
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),
                {
                    torch.classes._TorchScriptTesting._Foo(3, 4),
                },
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},
            ]
            self.sub_mod = ModelsWithScriptObjectAttr.Simple()
            self.sub_mod2 = ModelsWithScriptObjectAttr.SimpleWithAttrInContainer()

    class MoreNestedWithAttrInContainer(torch.nn.Module):
        # [CLASS DOC] HELPER: Like the nested fixture but one level deeper: its second submodule is
        # itself the nested fixture.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the script object attribute, the container attribute,
            # a simple submodule and a nested-fixture submodule.
            super().__init__()
            self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)
            self.pytree_attr2 = [
                torch.classes._TorchScriptTesting._Foo(1, 2),
                {
                    torch.classes._TorchScriptTesting._Foo(3, 4),
                },
                {"foo": torch.classes._TorchScriptTesting._Foo(5, 6)},
            ]
            self.sub_mod = ModelsWithScriptObjectAttr.Simple()
            self.sub_mod2 = ModelsWithScriptObjectAttr.NestedWithAttrInContainer()


def _set_grad_enabled_tests():
    # [FUNCTION DOC] HELPER: Builds a dictionary of cases for set-grad tests. Each value is a
    # triple of an eager module, its pre-dispatch exported module, and example inputs. Cases cover
    # a raw grad-mode setter op, context managers, a multi-dependency variant, each also traced
    # under an ambient no-grad setting. Exports run at call time, with the new experimental tracer
    # config enabled.
    from torch.export._trace import _export

    class SetGradOp(torch.nn.Module):
        # [CLASS DOC] HELPER: Module that flips grad mode by calling the low-level setter between
        # computations.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Adds one, turns grad on and sums a sine, turns grad off and
            # adds one, turns grad on again and subtracts one; returns the last two results.
            x = x + 1
            torch._C._set_grad_enabled(True)
            c = x.sin().sum()
            torch._C._set_grad_enabled(False)
            d = c + 1
            torch._C._set_grad_enabled(True)
            e = d - 1
            return d, e

    class SetGradCtxManager(torch.nn.Module):
        # [CLASS DOC] HELPER: Module using grad-enabling and no-grad context managers around three
        # stages.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Same stages as the setter module but with context managers;
            # returns two results.
            x = x + 1
            with torch.enable_grad():
                c = x.sin().sum()
            with torch.no_grad():
                d = c + 1
            with torch.enable_grad():
                e = d - 1
            return d, e

    class SetGradCtxManagerMultiDep(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with context-managed stages carrying two independent
        # dependency chains.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Computes sine and cosine sums under enable-grad, adds one to
            # each under no-grad, subtracts one under enable-grad; returns four values.
            x = x + 1
            with torch.enable_grad():
                c1 = x.sin().sum()
                c2 = x.cos().sum()
            with torch.no_grad():
                d1 = c1 + 1
                d2 = c2 + 1
            with torch.enable_grad():
                e1 = d1 - 1
                e2 = d2 - 1
            return d1, d2, e1, e2

    x = torch.randn(2, 2)

    def _get_predispatch_module(mod, args, ambient_grad_enabled=True):
        # [FUNCTION DOC] HELPER: Exports with pre-dispatch tracing under the given ambient grad
        # mode (default enabled) using the experimental tracer config, and returns the unlifted
        # module.
        with torch.set_grad_enabled(ambient_grad_enabled):
            with config.patch(use_new_tracer_experimental=True):
                return _export(mod, args, pre_dispatch=True).module()

    return {
        "ctx_manager": (
            SetGradCtxManager(),
            _get_predispatch_module(SetGradCtxManager(), (x,)),
            (x,),
        ),
        "ctx_manager_under_no_grad": (
            SetGradCtxManager(),
            _get_predispatch_module(SetGradCtxManager(), (x,), False),
            (x,),
        ),
        "ctx_manager_multi_dep": (
            SetGradCtxManagerMultiDep(),
            _get_predispatch_module(SetGradCtxManagerMultiDep(), (x,)),
            (x,),
        ),
        "ctx_manager_multi_dep_no_grad": (
            SetGradCtxManagerMultiDep(),
            _get_predispatch_module(SetGradCtxManagerMultiDep(), (x,), False),
            (x,),
        ),
        "op": (SetGradOp(), _get_predispatch_module(SetGradOp(), (x,)), (x,)),
        "op_under_no_grad": (
            SetGradOp(),
            _get_predispatch_module(SetGradOp(), (x,), False),
            (x,),
        ),
    }


def _with_autocast_tests():
    # [FUNCTION DOC] HELPER: Builds a dictionary of autocast cases (plain, multi-dependency, split,
    # nested), each a triple of eager module, pre-dispatch exported module and example inputs.
    from torch.export._trace import _export

    class WithAutocastOp(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with alternating enabled, disabled and enabled CPU autocast
        # regions.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Adds one, then sine sum under autocast, adds one with autocast
            # disabled, subtracts one under autocast; returns two values.
            x = x + 1
            with torch.autocast(device_type="cpu", enabled=True):
                c = x.sin().sum()
            with torch.autocast(device_type="cpu", enabled=False):
                d = c + 1
            with torch.autocast(device_type="cpu", enabled=True):
                e = d - 1
            return d, e

    class WithAutocastOpMultiDep(torch.nn.Module):
        # [CLASS DOC] HELPER: Autocast module with two independent dependency chains in each
        # region.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Same region pattern as the single-chain module with sine and
            # cosine sums; returns four values.
            x = x + 1
            with torch.autocast(device_type="cpu", enabled=True):
                c1 = x.sin().sum()
                c2 = x.cos().sum()
            with torch.autocast(device_type="cpu", enabled=False):
                d1 = c1 + 1
                d2 = c2 + 1
            with torch.autocast(device_type="cpu", enabled=True):
                e1 = d1 - 1
                e2 = d2 - 1
            return d1, d2, e1, e2

    class SplitAutocastOp(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with two separate autocast regions with an unguarded operation
        # between them.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Sine sum under autocast, adds one outside, subtracts one under
            # autocast; returns two values.
            x = x + 1
            with torch.autocast(device_type="cpu", enabled=True):
                c = x.sin().sum()
            d = c + 1
            with torch.autocast(device_type="cpu", enabled=True):
                e = d - 1
            return d, e

    class NestedAutocastOp(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with an autocast-disabled region nested inside an enabled one.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Sine sum in an enabled region, adds one in a nested disabled
            # region, subtracts one in a later enabled region; returns two values.
            x = x + 1
            with torch.autocast(device_type="cpu", enabled=True):
                c = x.sin().sum()
                with torch.autocast(device_type="cpu", enabled=False):
                    d = c + 1
            with torch.autocast(device_type="cpu", enabled=True):
                e = d - 1
            return d, e

    x = torch.randn(2, 2)

    def _get_predispatch_module(mod, args):
        # [FUNCTION DOC] HELPER: Exports with pre-dispatch tracing and returns the unlifted module.
        return _export(mod, args, pre_dispatch=True).module()

    return {
        "ctx_manager": (
            WithAutocastOp(),
            _get_predispatch_module(WithAutocastOp(), (x,)),
            (x,),
        ),
        "ctx_manager_multi_dep": (
            WithAutocastOpMultiDep(),
            _get_predispatch_module(WithAutocastOpMultiDep(), (x,)),
            (x,),
        ),
        "ctx_manager_split": (
            SplitAutocastOp(),
            _get_predispatch_module(SplitAutocastOp(), (x,)),
            (x,),
        ),
        "ctx_manager_nested": (
            NestedAutocastOp(),
            _get_predispatch_module(NestedAutocastOp(), (x,)),
            (x,),
        ),
    }


def _with_mixed_autocast_set_grad_tests():
    # [FUNCTION DOC] HELPER: Builds a dictionary with one case mixing grad-mode setter calls and an
    # autocast region, as a triple of eager module, pre-dispatch module and inputs.
    from torch.export._trace import _export

    class WithAutocastSetGradOp(torch.nn.Module):
        # [CLASS DOC] HELPER: Module that turns grad on and off by setter calls and uses a disabled
        # autocast region.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Adds one, sine with grad on, cosine with grad off, adds one in
            # a disabled autocast region, subtracts one; returns two values.
            x = x + 1
            torch._C._set_grad_enabled(True)
            c = x.sin()
            torch._C._set_grad_enabled(False)
            c = c.cos()
            with torch.autocast(device_type="cpu", enabled=False):
                d = c + 1
            e = d - 1
            return d, e

    x = torch.randn(2, 2)

    def _get_predispatch_module(mod, args):
        # [FUNCTION DOC] HELPER: Exports with pre-dispatch tracing under the experimental tracer
        # config and returns the unlifted module.
        with torch._export.config.patch(use_new_tracer_experimental=True):
            ep = _export(mod, args, pre_dispatch=True).module()
            return ep

    return {
        "multi_ctx_manager": (
            WithAutocastSetGradOp(),
            _get_predispatch_module(WithAutocastSetGradOp(), (x,)),
            (x,),
        ),
    }


def _sequential_split_inline_tests():
    # [FUNCTION DOC] HELPER: Builds a dictionary of cases for sequential split and inline tests,
    # each a pair of a pre-dispatch module with grad-setter delimiter nodes inserted at every
    # second or third (or every) call, and example inputs. Uses a simple and a multi-dependency
    # module.
    from torch.export._trace import _export

    class Simple(torch.nn.Module):
        # [CLASS DOC] HELPER: Straight-line module computing a sine sum and two arithmetic steps.
        def forward(self, x):
            # [FUNCTION DOC] HELPER: Adds one, takes a sine sum, adds one, subtracts one; returns
            # two values.
            x = x + 1
            c = x.sin().sum()
            d = c + 1
            e = d - 1
            return d, e

    class MultiDep(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with two independent chains over two inputs.
        def forward(self, x1, x2):
            # [FUNCTION DOC] HELPER: Applies add, sine or cosine, add and subtract on two inputs
            # separately; returns four values.
            x1 = x1 + 1
            x2 = x2 + 1
            c1 = x1.sin()
            c2 = x2.cos()
            d1 = c1 + 1
            d2 = c2 + 1
            e1 = d1 - 1
            e2 = d2 - 1
            return d1, d2, e1, e2

    def _get_predispatch_module(mod, args):
        # [FUNCTION DOC] HELPER: Exports with pre-dispatch tracing and returns the unlifted module.
        return _export(mod, args, pre_dispatch=True).module()

    def _insert_dilimiter_nodes(gm: torch.fx.GraphModule, step: int = 1):
        # [FUNCTION DOC] HELPER: Inserts a grad-mode setter call before every step-th call_function
        # node of the module, alternating enabled and disabled values by insertion index. Mutates
        # and returns the module.
        insert_locs = []
        for i, node in enumerate(
            nodes_filter(gm.graph.nodes, lambda n: n.op == "call_function")
        ):
            if i % step == 0:
                insert_locs.append(node)

        for i, node in enumerate(insert_locs):
            with gm.graph.inserting_before(node):
                gm.graph.call_function(torch._C._set_grad_enabled, (i % 2 == 0,), {})
        return gm

    x = torch.randn(2, 2)
    simple = _get_predispatch_module(Simple(), (x,))
    simple1 = _get_predispatch_module(Simple(), (x,))
    multi_dep = _get_predispatch_module(MultiDep(), (x, x.sin()))
    multi_dep1 = _get_predispatch_module(MultiDep(), (x, x.sin()))
    return {
        "simple_step1": (_insert_dilimiter_nodes(simple1, 1), (x,)),
        "simple_step2": (_insert_dilimiter_nodes(simple, 2), (x,)),
        "multi_dep_step2": (_insert_dilimiter_nodes(multi_dep, 2), (x, x.sin())),
        "multi_dep_step3": (_insert_dilimiter_nodes(multi_dep1, 3), (x, x.sin())),
    }


@skipIfTorchDynamo("recursively running dynamo on export is unlikely")
@unittest.skipIf(not is_dynamo_supported(), "Dynamo not supported")
class TestPasses(TestCase):
    # [CLASS DOC] Tests for export passes. The setup builds the case dictionaries for set-grad,
    # autocast, mixed and sequential-split tests and initializes torchbind implementations;
    # teardown clears them. Skipped under Dynamo or when unsupported.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the base setup, then builds the four case dictionaries and
        # initializes torchbind implementations.
        super().setUp()
        self.MIXED_AUTOCAST_SET_GRAD_TESTS = _with_mixed_autocast_set_grad_tests()
        self.SEQUENTIAL_SPLIT_INLINE_TESTS = _sequential_split_inline_tests()
        self.SET_GRAD_ENABLED_TESTS = _set_grad_enabled_tests()
        self.WITH_AUTOCAST_TESTS = _with_autocast_tests()
        init_torchbind_implementations()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Clears the four case dictionaries and then calls the base
        # teardown.
        self.SEQUENTIAL_SPLIT_INLINE_TESTS.clear()
        self.SET_GRAD_ENABLED_TESTS.clear()
        self.WITH_AUTOCAST_TESTS.clear()
        self.MIXED_AUTOCAST_SET_GRAD_TESTS.clear()
        super().tearDown()

    def _check_node_users_in_the_same_graph(self, gm):
        # [FUNCTION DOC] HELPER: Asserts that every user of every node in the graph module belongs
        # to the same graph.
        for node in gm.graph.nodes:
            for user in node.users:
                self.assertTrue(user.graph is gm.graph)

    def test_runtime_assert_one_dim(self) -> None:
        # [FUNCTION DOC] Exports a cosine module with a bounded dynamic dimension (strict) and
        # checks the resulting module raises AssertionError with a guard-failed message for an
        # out-of-range size and returns the right values for an in-range size.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning cosine of its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns cosine of the input.
                return x.cos()

        x = torch.zeros(2, 2, 3)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        ep = torch.export.export(
            M(), (x,), dynamic_shapes={"x": {1: dim1_x}}, strict=True
        )

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[1] <= 6"),
        ):
            # expected <= 6, but got 7
            ep.module()(torch.zeros(2, 7, 3))

        self.assertEqual(
            ep.module()(torch.ones(2, 4, 3)), M().forward(torch.ones(2, 4, 3))
        )

    def test_runtime_assert_multiple_dims(self) -> None:
        # [FUNCTION DOC] Exports (strict) a module of two inputs with a bounded dynamic dimension
        # on the first input, an unbounded-above dynamic leading dimension on both, and checks that
        # running with an out-of-range size fails with a guard-failed AssertionError message for
        # the upper bound on the first input and for the lower bound on the second input.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of cosine of the first input and sine of
            # the second input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of cosine of the first input plus the sine
                # sum of the second.
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x, dim0_y = torch.export.dims("dim0_x", "dim0_y", min=3)

        ep = torch.export.export(
            M(),
            (x, y),
            dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": {0: dim0_y}},
            strict=True,
        )

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[1] <= 6"),
        ):
            # expected <= 6, but got 7
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y.size()[0] >= 3"),
        ):
            # expected >= 3, but got 2
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

    def test_runtime_assert_some_dims_not_specified(self) -> None:
        # [FUNCTION DOC] Same module as the multi-dim runtime assert test but the second input has
        # no dynamic shapes. Checks the bounded dimension still raises a guard-failed error, that
        # the unspecified input is specialized to its example size (an error when the size
        # differs), and that a size of one on a dimension without a recorded lower-bound guard
        # still matches eager output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the sum of cosine of one input and sine of the
            # other.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the cosine sum of the first input plus the sine
                # sum of the second.
                return x.cos().sum() + y.sin().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        dim1_x = torch.export.Dim("dim1_x", min=2, max=6)
        dim0_x = torch.export.Dim("dim0_x", min=3)

        ep = torch.export.export(
            M(),
            (x, y),
            dynamic_shapes={"x": {0: dim0_x, 1: dim1_x}, "y": None},
            strict=True,
        )

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[1] <= 6"),
        ):
            # expected <= 6, but got 7
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y.size()[0] == 5"),
        ):
            # expected 5, but got 2
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep.module()(torch.ones(3, 1, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.ones(3, 1, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_runtime_assert_some_inps_not_used(self) -> None:
        # [FUNCTION DOC] Module that ignores its first input. Exports with the unused input static
        # and the second input dynamic in one dimension. Checks that the static unused input still
        # gets size guards (an error on mismatch), that the second input's first dimension is
        # specialized, and that a matching call equals eager output. The comment mentions a size of
        # one but the final call uses the original shapes.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning the cosine sum of only its second input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the cosine sum of the second input; the first is
                # unused.
                return y.cos().sum()

        x = torch.zeros(4, 2, 3)
        y = torch.zeros(5, 5, 5)

        dim1_y = torch.export.Dim("dim1_y", min=3, max=6)
        ep = torch.export.export(
            M(), (x, y), dynamic_shapes={"x": None, "y": {1: dim1_y}}, strict=True
        )

        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: x.size()[1] == 2"),
        ):
            # expected 2, but got 7
            ep.module()(torch.zeros(4, 7, 3), torch.ones(5, 5, 5))

        # y is specialized to 5
        with self.assertRaisesRegex(
            AssertionError,
            escape("Guard failed: y.size()[0] == 5"),
        ):
            # expected 5, but got 2
            ep.module()(torch.zeros(4, 2, 3), torch.ones(2, 5, 5))

        # Since we didn't insert the constraint for x[1] >= 2, it should work for case where x[1] == 1
        gm_result_for_1_size = ep.module()(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))
        eager_result_for_1_size = M().forward(torch.zeros(4, 2, 3), torch.ones(5, 5, 5))

        self.assertEqual(gm_result_for_1_size, eager_result_for_1_size)

    def test_view_to_view_copy(self) -> None:
        # [FUNCTION DOC] Exports a module that reshapes with a view and checks the graph has
        # exactly one aten view before the view-to-view-copy replacement pass and none after
        # applying it.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module taking a same-shape view of its input and returning the
            # cosine sum.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Views the input with its own shape and returns the cosine
                # sum.
                z = x.view(x.shape)
                return z.cos().sum()

        x = torch.zeros(4, 2, 3)

        ep = export(M(), (x,), strict=True)
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 1)

        ep = ep._transform_do_not_use(ReplaceViewOpsWithViewCopyOpsPass())
        self.assertEqual(count_call_function(ep.graph, torch.ops.aten.view.default), 0)

    def test_functionalization_with_view_copy(self) -> None:
        # [FUNCTION DOC] Exports a module with an in-place add followed by a view, applies the
        # view-copy replacement pass, and checks there are no view nodes left and at least one
        # view-copy node.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an in-place add on a temporary, a view and cosines of
            # input and view.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds four, adds four in place, views, returns the sum of
                # cosines of the input and the view.
                y = x + 4
                y.add_(4)
                z = y.view(y.shape)
                return x.cos() + z.cos()

        x = torch.zeros(4, 2, 3)
        foo = Module()
        ep = export(foo, (x,), strict=True)._transform_do_not_use(
            ReplaceViewOpsWithViewCopyOpsPass()
        )
        # After this pass, there shouldn't be any view nodes in the graph
        self.assertTrue(count_call_function(ep.graph, torch.ops.aten.view.default) == 0)
        self.assertTrue(
            count_call_function(ep.graph, torch.ops.aten.view_copy.default) > 0
        )

    def test_views_op_having_view_copy(self) -> None:
        # [FUNCTION DOC] Walks every registered aten operator schema, splitting names into operator
        # and overload (default when none; more than two parts raises AssertionError). For
        # operators tagged core that are view ops, asserts a view-copy counterpart can be found for
        # the schema.
        schemas = torch._C._dispatch_get_registrations_for_dispatch_key("")
        aten_schemas = [s[6:] for s in schemas if s.startswith("aten::")]

        for aten_schema in aten_schemas:
            val = aten_schema.split(".")
            if len(val) > 2:
                raise AssertionError(f"Expected at most 2 parts, got {len(val)}: {val}")
            name = ""
            overload = ""
            if len(val) == 1:
                name = val[0]
                overload = "default"
            else:
                name, overload = val[0], val[1]

            op_overload = getattr(getattr(torch.ops.aten, name), overload)
            if torch.Tag.core in op_overload.tags and is_view_op(op_overload._schema):
                self.assertIsNotNone(get_view_copy_of_view_op(op_overload._schema))

    def test_custom_obj_tuple_out(self):
        # [FUNCTION DOC] Exports (non-strict) a module that calls two torchbind custom ops, one
        # returning a tuple, strips effect tokens from the program, verifies it and checks that
        # eager, exported and token-free outputs are all close.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module owning a torchbind object and chaining a tuple-returning
            # custom op with a second custom op.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the torchbind object attribute.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the tuple-returning custom op, adds the two outputs,
                # passes the sum to the second custom op and returns that.
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(self.attr, x)
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(self.attr, y)
                return b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        ep = torch.export.export(m, inputs, strict=False)

        inp = torch.randn(2, 3)
        orig_res = m(inp)
        ep_res = ep.module()(inp)

        without_token_ep = _remove_effect_tokens(ep)
        without_token_ep.verifier().check(without_token_ep)
        without_token_res = without_token_ep.module()(inp)

        self.assertTrue(torch.allclose(orig_res, ep_res))
        self.assertTrue(torch.allclose(orig_res, without_token_res))

    def test_remove_effect_token_kwargs(self):
        # [FUNCTION DOC] Exports (non-strict), decomposes, and removes effect tokens from a program
        # where custom ops are called with keyword arguments, then compares the printed graph code
        # with an expected string showing the keyword calls and no tokens.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module like the tuple-out module but calling the custom ops with
            # keyword arguments.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the torchbind object attribute.
                super().__init__()
                self.attr = torch.classes._TorchScriptTesting._Foo(10, 20)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the tuple-returning op with keyword arguments, adds
                # the outputs and calls the second op with keywords.
                a = torch.ops._TorchScriptTesting.takes_foo_tuple_return(
                    foo=self.attr, x=x
                )
                y = a[0] + a[1]
                b = torch.ops._TorchScriptTesting.takes_foo(foo=self.attr, x=y)
                return b

        m = MyModule()
        inputs = (torch.ones(2, 3),)
        ep = export(m, inputs, strict=False).run_decompositions({})
        without_token_ep = _remove_effect_tokens(ep)
        self.assertExpectedInline(
            without_token_ep.graph_module.code.strip(),
            """\
def forward(self, obj_attr, x):
    takes_foo_tuple_return_default = torch.ops._TorchScriptTesting.takes_foo_tuple_return.default(foo = obj_attr, x = x);  x = None
    getitem_1 = takes_foo_tuple_return_default[0]
    getitem_2 = takes_foo_tuple_return_default[1];  takes_foo_tuple_return_default = None
    add = torch.ops.aten.add.Tensor(getitem_1, getitem_2);  getitem_1 = getitem_2 = None
    takes_foo_default = torch.ops._TorchScriptTesting.takes_foo.default(foo = obj_attr, x = add);  obj_attr = add = None
    return (takes_foo_default,)""",
        )

    def test_fakify_script_objects(self):
        # [FUNCTION DOC] For each script-object fixture module, gathers constant attribute names,
        # enters the fakify-script-objects context under a fake mode, and checks the fake attribute
        # mapping has the same size and maps each fake back to the real object with the same name.
        for m in [
            ModelsWithScriptObjectAttr.Simple(),
            ModelsWithScriptObjectAttr.SimpleWithAttrInContainer(),
            ModelsWithScriptObjectAttr.NestedWithAttrInContainer(),
            ModelsWithScriptObjectAttr.MoreNestedWithAttrInContainer(),
        ]:
            constant_attrs = _gather_constant_attrs(m)
            fake_mode = FakeTensorMode(
                shape_env=ShapeEnv(tracked_fakes=[]),
                allow_non_fake_inputs=True,
            )
            with _fakify_script_objects(m, (), {}, fake_mode) as (
                _,
                _,
                _,
                fake_constant_attrs,
                fake_to_real,
            ):
                self.assertEqual(len(fake_constant_attrs), len(constant_attrs))
                for fake_obj, fqn in fake_constant_attrs.items():
                    self.assertEqual(constant_attrs[fake_to_real[fake_obj]], fqn)

    # TODO: _gather_constants doesn't recursively look into the pytree containers.
    @unittest.expectedFailure
    def test_fakify_script_objects_properly_handle_containers(self):
        # [FUNCTION DOC] Expected failure: with the container fixture, the fakified constant
        # attributes should include both the direct and the container attribute, but gathering does
        # not look inside containers. Marked as expected failure.
        m = ModelsWithScriptObjectAttr.SimpleWithAttrInContainer()
        fake_mode = FakeTensorMode(
            shape_env=ShapeEnv(tracked_fakes=[]),
            allow_non_fake_inputs=True,
        )
        with _fakify_script_objects(m, (), {}, fake_mode) as (
            _,
            _,
            _,
            fake_constant_attrs,
            _,
        ):
            self.assertTrue("attr" in fake_constant_attrs.values())
            self.assertTrue("pytree_attr2" in fake_constant_attrs.values())

    def test_runtime_assert_inline_constraints_for_item(self) -> None:
        # [FUNCTION DOC] Exports (strict) a module that reads a scalar and constrains it between
        # two and five. Checks an out-of-range value gives a RuntimeError about a failed runtime
        # assertion on the upper bound and an in-range value equals eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module reading a scalar from its input with lower and upper bound
            # checks.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads a scalar from the input, checks it lies between two
                # and five, returns it.
                b = x.item()
                torch._check(b >= 2)
                torch._check(b <= 5)
                return b

        x = torch.tensor([2])
        mod = M()
        ep = export(mod, (x,), strict=True)

        with self.assertRaisesRegex(
            RuntimeError, r"Runtime assertion failed for expression u[\d+] \<\= 5"
        ):
            ep.module()(torch.tensor([6]))

        new_inp = torch.tensor([5])
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    def test_runtime_assert_inline_constraints_for_nonzero(self) -> None:
        # [FUNCTION DOC] Exports (strict) a module that applies nonzero and constrains the result
        # length between three and five, with a dynamic input dimension. Checks the graph has two
        # scalar assert nodes and no symbolic range constraint nodes, that too few or too many
        # nonzero entries give RuntimeError messages on the failed bound, and that an in-range
        # input equals eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning nonzero indices with bound checks on their
            # count.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Computes nonzero, checks the count lies between three and
                # five, returns it.
                b = x.nonzero()
                torch._check(b.shape[0] >= 3)
                torch._check(b.shape[0] <= 5)
                return b

        x = torch.tensor([2, 1, 2, 3, 5, 0])

        mod = M()
        dim0_x = torch.export.Dim("dim0_x")
        ep = torch.export.export(
            mod, (x,), dynamic_shapes={"x": {0: dim0_x}}, strict=True
        )

        num_assert = count_call_function(
            ep.graph, torch.ops.aten._assert_scalar.default
        )
        self.assertEqual(num_assert, 2)
        num_constrain_range = count_call_function(
            ep.graph, torch.ops.aten.sym_constrain_range.default
        )
        self.assertEqual(num_constrain_range, 0)

        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression u[\d+] \>\= 3",
        ):
            ep.module()(torch.tensor([1, 1, 0, 0, 0]))

        with self.assertRaisesRegex(
            RuntimeError,
            r"Runtime assertion failed for expression u[\d+] \<\= 5",
        ):
            ep.module()(torch.ones(6))

        new_inp = torch.tensor([1, 1, 1, 1])
        self.assertEqual(mod(new_inp), ep.module()(new_inp))

    @unittest.skipIf(IS_WINDOWS, "Windows not supported")
    @unittest.expectedFailure
    # TODO(pianpwk): add back runtime asserts to subgraphs
    def test_runtime_assert_inline_constraints_for_cond(self) -> None:
        # [FUNCTION DOC] Expected failure (runtime asserts are not yet carried into subgraphs;
        # skipped on Windows). Exports (strict) a conditional module and expects an out-of-range
        # value in a branch to raise a RuntimeError mentioning the inline constraint.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward runs a conditional on a predicate with two
            # branches that each read a scalar, bound it between two and five, and subtract it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Only calls the base constructor.
                super().__init__()

            def forward(self, pred, x, y):
                # [FUNCTION DOC] HELPER: Defines both branch functions and returns the conditional
                # result.
                def true_fn(x, y):
                    # [FUNCTION DOC] HELPER: True branch: reads a scalar from the first input,
                    # checks it is within two to five, returns the first input minus it.
                    b = x.item()
                    torch._check(b >= 2)
                    torch._check(b <= 5)
                    return x - b

                def false_fn(x, y):
                    # [FUNCTION DOC] HELPER: False branch: reads a scalar from the second input,
                    # checks it is within two to five, returns the second input minus it.
                    c = y.item()
                    torch._check(c >= 2)
                    torch._check(c <= 5)
                    return y - c

                ret = cond(pred, true_fn, false_fn, [x, y])
                return ret

        x = torch.tensor([2])
        y = torch.tensor([5])
        mod = M()
        ep = export(mod, (torch.tensor(True), x, y), strict=True)

        with self.assertRaisesRegex(
            RuntimeError, "is outside of inline constraint \\[2, 5\\]."
        ):
            ep.module()(torch.tensor(False), torch.tensor([6]), torch.tensor([6]))

    def test_math_ops(self):
        # [FUNCTION DOC] Exports (strict) a module converting ceiling and floor of a scalar input
        # into tensors, then runs the deprecated pass base over the exported graph module; passes
        # if no error is raised.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning one-element tensors of the ceiling and the floor
            # of the input's scalar value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a pair of tensors holding ceiling and floor of the
                # scalar from the input.
                return (
                    torch.tensor([math.ceil(x.item())]),
                    torch.tensor([math.floor(x.item())]),
                )

        func = Module()
        x = torch.randn(1, dtype=torch.float32)
        ep = torch.export.export(func, args=(x,), strict=True)
        _ExportPassBaseDeprecatedDoNotUse()(ep.graph_module)

    def test_predispatch_set_grad(self):
        # [FUNCTION DOC] Compares pre-dispatch exported modules against eager and against expected
        # generated code for set-grad cases (raw setter op and context manager, each with and
        # without ambient no-grad, plus the multi-dependency variants). Each case first checks that
        # all node users are in the same graph and that eager and exported outputs agree. The
        # expected code shows grad-mode regions replaced by calls to the set-grad higher-order
        # operator with a submodule.
        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS["op"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_4 = self.submod_2
    add_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_4, sum_1);  submod_4 = sum_1 = None
    getitem = add_1[0];  add_1 = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1)
    return pytree.tree_unflatten((getitem, sub), self._out_spec)
    """,
        )

        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS["op_under_no_grad"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_4 = self.submod_1
    sum_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_4, add);  submod_4 = add = None
    getitem = sum_1[0];  sum_1 = None
    add_1 = torch.ops.aten.add.Tensor(getitem, 1);  getitem = None
    submod_5 = self.submod_3
    sub = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_5, add_1);  submod_5 = None
    getitem_1 = sub[0];  sub = None
    return pytree.tree_unflatten((add_1, getitem_1), self._out_spec)
    """,
        )

        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_3 = self.submod_1
    add_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_3, sum_1);  submod_3 = sum_1 = None
    getitem = add_1[0];  add_1 = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1)
    return pytree.tree_unflatten((getitem, sub), self._out_spec)
    """,
        )

        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager_under_no_grad"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_5 = self.submod_1
    sum_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_5, add);  submod_5 = add = None
    getitem = sum_1[0];  sum_1 = None
    add_1 = torch.ops.aten.add.Tensor(getitem, 1);  getitem = None
    submod_6 = self.submod_3
    sub = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_6, add_1);  submod_6 = None
    getitem_1 = sub[0];  sub = None
    return pytree.tree_unflatten((add_1, getitem_1), self._out_spec)
    """,
        )

        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS["ctx_manager_multi_dep"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    sin = torch.ops.aten.sin.default(add)
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    cos = torch.ops.aten.cos.default(add);  add = None
    sum_2 = torch.ops.aten.sum.default(cos);  cos = None
    submod_3 = self.submod_1
    wrap_with_set_grad_enabled = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_3, sum_1, sum_2);  submod_3 = sum_1 = sum_2 = None
    add_1 = wrap_with_set_grad_enabled[0]
    add_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    sub = torch.ops.aten.sub.Tensor(add_1, 1)
    sub_1 = torch.ops.aten.sub.Tensor(add_2, 1)
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    """,
        )

        mod_orig, mod, args = self.SET_GRAD_ENABLED_TESTS[
            "ctx_manager_multi_dep_no_grad"
        ]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_5 = self.submod_1
    wrap_with_set_grad_enabled = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_5, add);  submod_5 = add = None
    sum_1 = wrap_with_set_grad_enabled[0]
    sum_2 = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    add_2 = torch.ops.aten.add.Tensor(sum_2, 1);  sum_2 = None
    submod_6 = self.submod_3
    wrap_with_set_grad_enabled_1 = torch.ops.higher_order.wrap_with_set_grad_enabled(True, submod_6, add_1, add_2);  submod_6 = None
    sub = wrap_with_set_grad_enabled_1[0]
    sub_1 = wrap_with_set_grad_enabled_1[1];  wrap_with_set_grad_enabled_1 = None
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    """,
        )

    def test_sequential_split(self):
        # [FUNCTION DOC] For each sequential-split case, counts set-grad delimiter nodes before
        # splitting, splits the module at those nodes, counts set-grad submodules in the result,
        # and checks the counts are equal and the outputs of both modules match.
        for gm, args in self.SEQUENTIAL_SPLIT_INLINE_TESTS.values():
            set_grad_counts = nodes_count(gm.graph.nodes, _is_set_grad_enabled_node)
            new_gm = sequential_split(gm, _is_set_grad_enabled_node)
            new_set_grad_counts = nodes_count(
                new_gm.graph.nodes, _is_set_grad_enabled_sub_mod
            )
            self.assertEqual(set_grad_counts, new_set_grad_counts)
            self.assertEqual(gm(*args), new_gm(*args))

    def test_sequential_split_graph(self):
        # [FUNCTION DOC] Splits the multi-dependency step-two case at its set-grad delimiter nodes,
        # checks output equality and compares the expected code of the split module and three of
        # its submodules, which contain the setter call and the operations until the next
        # delimiter.
        gm, args = self.SEQUENTIAL_SPLIT_INLINE_TESTS["multi_dep_step2"]

        new_gm = sequential_split(gm, _is_set_grad_enabled_node)
        self.assertEqual(gm(*args), new_gm(*args))
        self.assertExpectedInline(
            new_gm.code.strip("\n"),
            """\
def forward(self, x1, x2):
    x1, x2, = fx_pytree.tree_flatten_spec(([x1, x2], {}), self._in_spec)
    submod_0 = self.submod_0(x1, x2);  submod_0 = None
    submod_1 = self.submod_1(x1, x2);  x1 = x2 = None
    getitem = submod_1[0]
    getitem_1 = submod_1[1];  submod_1 = None
    submod_2 = self.submod_2(getitem, getitem_1);  getitem = getitem_1 = None
    getitem_2 = submod_2[0]
    getitem_3 = submod_2[1];  submod_2 = None
    submod_3 = self.submod_3(getitem_2, getitem_3);  getitem_2 = getitem_3 = None
    getitem_4 = submod_3[0]
    getitem_5 = submod_3[1];  submod_3 = None
    submod_4 = self.submod_4(getitem_4, getitem_5)
    getitem_6 = submod_4[0]
    getitem_7 = submod_4[1];  submod_4 = None
    return pytree.tree_unflatten((getitem_4, getitem_5, getitem_6, getitem_7), self._out_spec)
    """,
        )
        self.assertExpectedInline(
            new_gm.submod_1.code.strip("\n"),
            """\
def forward(self, x1, x2):
    _set_grad_enabled = torch._C._set_grad_enabled(True);  _set_grad_enabled = None
    add = torch.ops.aten.add.Tensor(x1, 1);  x1 = None
    add_1 = torch.ops.aten.add.Tensor(x2, 1);  x2 = None
    return (add, add_1)
    """,
        )
        self.assertExpectedInline(
            new_gm.submod_2.code.strip("\n"),
            """\
def forward(self, add, add_1):
    _set_grad_enabled_1 = torch._C._set_grad_enabled(False);  _set_grad_enabled_1 = None
    sin = torch.ops.aten.sin.default(add);  add = None
    cos = torch.ops.aten.cos.default(add_1);  add_1 = None
    return (sin, cos)
    """,
        )
        self.assertExpectedInline(
            new_gm.submod_3.code.strip("\n"),
            """\
def forward(self, sin, cos):
    _set_grad_enabled_2 = torch._C._set_grad_enabled(True);  _set_grad_enabled_2 = None
    add_2 = torch.ops.aten.add.Tensor(sin, 1);  sin = None
    add_3 = torch.ops.aten.add.Tensor(cos, 1);  cos = None
    return (add_2, add_3)
    """,
        )

    def test_predispatch_autocast_and_set_grad(self):
        # [FUNCTION DOC] For the mixed autocast and set-grad case, checks node users, eager
        # equality and expected code of the module and two nested submodules: the set-grad region
        # wraps a submodule containing a cosine, a wrapped autocast-disabled submodule and a
        # subtraction.
        mod_orig, mod, args = self.MIXED_AUTOCAST_SET_GRAD_TESTS["multi_ctx_manager"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    sin = torch.ops.aten.sin.default(add);  add = None
    submod_3 = self.submod_2
    wrap_with_set_grad_enabled = torch.ops.higher_order.wrap_with_set_grad_enabled(False, submod_3, sin);  submod_3 = sin = None
    add_1 = wrap_with_set_grad_enabled[0]
    sub = wrap_with_set_grad_enabled[1];  wrap_with_set_grad_enabled = None
    return pytree.tree_unflatten((add_1, sub), self._out_spec)
    """,
        )
        self.assertExpectedInline(
            mod.submod_2.code.strip("\n"),
            """\
def forward(self, sin):
    cos = torch.ops.aten.cos.default(sin);  sin = None
    submod_3 = self.submod_1
    add_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, False, None, submod_3, cos);  submod_3 = cos = None
    getitem = add_1[0];  add_1 = None
    sub = torch.ops.aten.sub.Tensor(getitem, 1)
    return (getitem, sub)
    """,
        )
        self.assertExpectedInline(
            mod.submod_2.submod_1.code.strip("\n"),
            """\
def forward(self, cos):
    add_1 = torch.ops.aten.add.Tensor(cos, 1);  cos = None
    return (add_1,)
    """,
        )

    def test_predispatch_autocast(self):
        # [FUNCTION DOC] For the nested, plain, and multi-dependency autocast cases, checks node
        # users are in the same graph, eager equality, and expected code for the module and each
        # submodule: autocast regions become calls to the autocast higher-order operator on
        # submodules with the matching enabled flag.
        mod_orig, mod, args = self.WITH_AUTOCAST_TESTS["ctx_manager_nested"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_3 = self.submod_1
    add_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_3, add);  submod_3 = add = None
    getitem = add_1[0];  add_1 = None
    submod_4 = self.submod_2
    sub = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_4, getitem);  submod_4 = None
    getitem_1 = sub[0];  sub = None
    return pytree.tree_unflatten((getitem, getitem_1), self._out_spec)
    """,
        )

        self.assertExpectedInline(
            mod.submod_1.code.strip("\n"),
            """\
def forward(self, add):
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    submod_2 = self.submod_1
    add_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, False, None, submod_2, sum_1);  submod_2 = sum_1 = None
    getitem = add_1[0];  add_1 = None
    return (getitem,)
    """,
        )

        mod_orig, mod, args = self.WITH_AUTOCAST_TESTS["ctx_manager"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_4 = self.submod_1
    sum_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_4, add);  submod_4 = add = None
    getitem = sum_1[0];  sum_1 = None
    submod_5 = self.submod_2
    add_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, False, None, submod_5, getitem);  submod_5 = getitem = None
    getitem_1 = add_1[0];  add_1 = None
    submod_6 = self.submod_3
    sub = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_6, getitem_1);  submod_6 = None
    getitem_2 = sub[0];  sub = None
    return pytree.tree_unflatten((getitem_1, getitem_2), self._out_spec)
    """,
        )

        self.assertExpectedInline(
            mod.submod_1.code.strip("\n"),
            """\
def forward(self, add):
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    return (sum_1,)
    """,
        )

        self.assertExpectedInline(
            mod.submod_2.code.strip("\n"),
            """\
def forward(self, sum_1):
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    return (add_1,)
    """,
        )

        self.assertExpectedInline(
            mod.submod_3.code.strip("\n"),
            """\
def forward(self, add_1):
    sub = torch.ops.aten.sub.Tensor(add_1, 1);  add_1 = None
    return (sub,)
    """,
        )

        mod_orig, mod, args = self.WITH_AUTOCAST_TESTS["ctx_manager_multi_dep"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_4 = self.submod_1
    wrap_with_autocast = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_4, add);  submod_4 = add = None
    sum_1 = wrap_with_autocast[0]
    sum_2 = wrap_with_autocast[1];  wrap_with_autocast = None
    submod_5 = self.submod_2
    wrap_with_autocast_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, False, None, submod_5, sum_1, sum_2);  submod_5 = sum_1 = sum_2 = None
    add_1 = wrap_with_autocast_1[0]
    add_2 = wrap_with_autocast_1[1];  wrap_with_autocast_1 = None
    submod_6 = self.submod_3
    wrap_with_autocast_2 = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_6, add_1, add_2);  submod_6 = None
    sub = wrap_with_autocast_2[0]
    sub_1 = wrap_with_autocast_2[1];  wrap_with_autocast_2 = None
    return pytree.tree_unflatten((add_1, add_2, sub, sub_1), self._out_spec)
    """,
        )

        self.assertExpectedInline(
            mod.submod_1.code.strip("\n"),
            """\
def forward(self, add):
    sin = torch.ops.aten.sin.default(add)
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    cos = torch.ops.aten.cos.default(add);  add = None
    sum_2 = torch.ops.aten.sum.default(cos);  cos = None
    return (sum_1, sum_2)
    """,
        )

        self.assertExpectedInline(
            mod.submod_2.code.strip("\n"),
            """\
def forward(self, sum_1, sum_2):
    add_1 = torch.ops.aten.add.Tensor(sum_1, 1);  sum_1 = None
    add_2 = torch.ops.aten.add.Tensor(sum_2, 1);  sum_2 = None
    return (add_1, add_2)
    """,
        )

        self.assertExpectedInline(
            mod.submod_3.code.strip("\n"),
            """\
def forward(self, add_1, add_2):
    sub = torch.ops.aten.sub.Tensor(add_1, 1);  add_1 = None
    sub_1 = torch.ops.aten.sub.Tensor(add_2, 1);  add_2 = None
    return (sub, sub_1)
    """,
        )

        mod_orig, mod, args = self.WITH_AUTOCAST_TESTS["ctx_manager_split"]
        self._check_node_users_in_the_same_graph(mod)
        self.assertEqual(mod_orig(*args), mod(*args))
        self.assertExpectedInline(
            mod.code.strip("\n"),
            """\
def forward(self, x):
    x, = fx_pytree.tree_flatten_spec(([x], {}), self._in_spec)
    _guards_fn = self._guards_fn(x);  _guards_fn = None
    add = torch.ops.aten.add.Tensor(x, 1);  x = None
    submod_4 = self.submod_1
    sum_1 = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_4, add);  submod_4 = add = None
    getitem = sum_1[0];  sum_1 = None
    add_1 = torch.ops.aten.add.Tensor(getitem, 1);  getitem = None
    submod_5 = self.submod_3
    sub = torch.ops.higher_order.wrap_with_autocast('cpu', None, True, None, submod_5, add_1);  submod_5 = None
    getitem_1 = sub[0];  sub = None
    return pytree.tree_unflatten((add_1, getitem_1), self._out_spec)
    """,
        )

        self.assertExpectedInline(
            mod.submod_1.code.strip("\n"),
            """\
def forward(self, add):
    sin = torch.ops.aten.sin.default(add);  add = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    return (sum_1,)
    """,
        )

        self.assertExpectedInline(
            mod.submod_3.code.strip("\n"),
            """\
def forward(self, add_1):
    sub = torch.ops.aten.sub.Tensor(add_1, 1);  add_1 = None
    return (sub,)
    """,
        )

    def test_inline_(self):
        # [FUNCTION DOC] For each sequential-split case, splits the module at delimiters, inlines
        # every call_module node back into the graph through a node map, and checks the readable
        # source is identical to the original one. Then copies the guards function onto the inlined
        # module and checks outputs match.
        for gm, args in self.SEQUENTIAL_SPLIT_INLINE_TESTS.values():
            before_str = gm.print_readable(print_output=False)
            new_gm = sequential_split(gm, _is_set_grad_enabled_node)
            nodes_map(
                new_gm.graph.nodes,
                lambda node: node_inline_(node) if node.op == "call_module" else node,
            )
            after_inline_str = new_gm.print_readable(print_output=False)
            self.assertEqual(before_str, after_inline_str)
            new_gm._guards_fn = gm._guards_fn
            self.assertEqual(gm(*args), new_gm(*args))

    def test_remove_auto_functionalized_pass(self) -> None:
        # [FUNCTION DOC] Defines in a scoped test library a custom mutating operator with meta and
        # explicit-autograd implementations, exports a module that mutates a buffer through it
        # (strict), removes auto-functionalized calls with the unsafe pass, and checks no node
        # targets the auto-functionalized operator or item access, and no output spec arg name
        # contains getitem.
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:
            lib.define("custom_mutator(Tensor x, Tensor(a!) y) -> Tensor")

            @impl(lib, "custom_mutator", "Meta")
            def custom_mutator_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Meta kernel returning an empty tensor like the first
                # argument.
                return torch.empty_like(x)

            @impl(lib, "custom_mutator", "CompositeExplicitAutograd")
            def custom_mutator(
                x: torch.Tensor,
                y: torch.Tensor,
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Real kernel: adds one to the second argument in place and
                # returns the first argument plus it.
                return x + y.add_(1)

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a zero buffer that passes the input and buffer to
                # the custom mutator.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Registers a one-element zero buffer.
                    super().__init__()
                    self.state = torch.nn.Buffer(torch.zeros(1))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Calls the custom mutating operator with the input and
                    # the buffer and returns its result.
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator(x, self.state)

            mod = M()
            x = torch.randn([3, 3])
            ep = export(mod, (x,), strict=True)
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)
            nodes = inplace_ep.graph.nodes
            for node in nodes:
                if node.op == "call_function":
                    self.assertFalse(node.target is auto_functionalized)
                    self.assertFalse(node.target is operator.getitem)

            for spec in inplace_ep.graph_signature.output_specs:
                self.assertFalse("getitem" in spec.arg.name)

    def test_remove_auto_functionalized_pass_tuple(self) -> None:
        # [FUNCTION DOC] Like the single-output test but the custom operator returns a tuple of two
        # tensors. Exports (strict), decomposes, applies the unsafe removal pass and compares the
        # graph's text with an expected string with the mutated buffer as a returned output and two
        # item accesses.
        with _scoped_library("DO_NOT_USE_TEST_ONLY", "DEF") as lib:
            lib.define(
                "custom_mutator_tuple(Tensor x, Tensor(a!) y) -> (Tensor, Tensor)"
            )

            @impl(lib, "custom_mutator_tuple", "Meta")
            def custom_mutator_tuple_meta(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Meta kernel returning two empty tensors like the first
                # argument.
                return (torch.empty_like(x), torch.empty_like(x))

            @impl(lib, "custom_mutator_tuple", "CompositeExplicitAutograd")
            def custom_mutator_tuple(
                x: torch.Tensor,
                y: torch.Tensor,
            ):
                # [FUNCTION DOC] HELPER: Real kernel: returns the first argument and the first
                # argument plus the second after adding one to it in place.
                return (x, x + y.add_(1))

            class M(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a zero buffer that passes the input and buffer to
                # the tuple-returning custom mutator.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Registers a one-element zero buffer.
                    super().__init__()
                    self.state = torch.nn.Buffer(torch.zeros(1))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Calls the tuple-returning custom mutating operator
                    # with the input and the buffer and returns its result.
                    return torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator_tuple(
                        x, self.state
                    )

            mod = M()
            x = torch.randn([3, 3])
            ep = export(mod, (x,), strict=True).run_decompositions({})
            inplace_ep = unsafe_remove_auto_functionalized_pass(ep)
            graph_text = str(inplace_ep.graph)
            self.assertExpectedInline(
                graph_text,
                """\
graph():
    %b_state : [num_users=2] = placeholder[target=b_state]
    %x : [num_users=1] = placeholder[target=x]
    %custom_mutator_tuple_default : [num_users=2] = call_function[target=torch.ops.DO_NOT_USE_TEST_ONLY.custom_mutator_tuple.\
default](args = (%x, %b_state), kwargs = {})
    %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%custom_mutator_tuple_default, 0), kwargs = {})
    %getitem_4 : [num_users=1] = call_function[target=operator.getitem](args = (%custom_mutator_tuple_default, 1), kwargs = {})
    return (b_state, getitem_3, getitem_4)""",
            )

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_move_device_to(self):
        # [FUNCTION DOC] Requires CUDA. Exports a module calling device-to with an explicit cuda
        # device, moves the program to cuda via the device pass, recompiles and compares generated
        # code with expected output where the metadata assertion, the to call and the add use cuda.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that moves its input to a cuda device with float dtype and
            # doubles it by self-addition.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the aten to-device operator and returns the sum of
                # the result with itself.
                x = torch.ops.aten.to.device(x, device="cuda:0", dtype=torch.float32)
                return x + x

        ep = torch.export.export(M(), (torch.ones(3),))
        ep = move_to_device_pass(ep, "cuda")
        ep.graph_module.recompile()
        self.assertExpectedInline(
            ep.graph_module.code.strip("\n"),
            """\
def forward(self, x):
    _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(x, dtype = torch.float32, device = 'cuda', layout = torch.strided);  _assert_tensor_metadata_default = None
    to = torch.ops.aten.to.device(x, 'cuda', torch.float32);  x = None
    add = torch.ops.aten.add.Tensor(to, to);  to = None
    return (add,)
    """,
        )

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_move_device_submod(self):
        # [FUNCTION DOC] Requires CUDA. Exports a module with an autocast region containing a
        # device move, applies the device pass, recompiles the autocast submodule and compares its
        # code with expected output using a cuda device.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a CUDA bfloat16 autocast region that moves the input
            # to cuda and doubles it.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Inside autocast, moves the input to cuda and returns the
                # sum of the result with itself.
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    x = x.to(device="cuda:0")
                    return x + x

        ep = torch.export.export(M(), (torch.ones(3),))
        ep = move_to_device_pass(ep, "cuda")
        ep.graph_module.submod_1.recompile()
        self.assertExpectedInline(
            ep.graph_module.submod_1.code.strip("\n"),
            """\
def forward(self, arg0_1):
    _assert_tensor_metadata_default = torch.ops.aten._assert_tensor_metadata.default(arg0_1, dtype = torch.float32, device = 'cuda', layout = torch.strided);  _assert_tensor_metadata_default = None
    to = torch.ops.aten.to.dtype_layout(arg0_1, dtype = torch.float32, layout = torch.strided, device = 'cuda');  arg0_1 = None
    add = torch.ops.aten.add.Tensor(to, to);  to = None
    return (add,)
    """,
        )

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_move_to_device_pass(self):
        # [FUNCTION DOC] Requires CUDA. Exports a GRU model on CPU (strict), moves it to cuda by
        # device object, checks outputs live on cuda, moves it back by string to CPU, then to cuda
        # again using a mapping from device names to target names, checking output devices each
        # time.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a batch-first GRU returning its final states.
            def __init__(self, size=4, h_dim=10):
                # [FUNCTION DOC] HELPER: Creates the GRU layer with the given input and hidden
                # sizes.
                super().__init__()
                self.rnn = torch.nn.GRU(size, h_dim, batch_first=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the GRU on the input and returns only the final
                # hidden states.
                _, states = self.rnn(x)
                return states

        # move the exported program from cpu to cuda:0
        mod = Model()
        example_inputs = (torch.rand(1, 10, 4),)
        ep = export(mod, example_inputs, strict=True)
        location = torch.device("cuda:0")
        ep = move_to_device_pass(ep, location=location)
        gm = ep.module()
        test_inputs = (torch.rand(1, 10, 4).to("cuda:0"),)
        outputs = gm(*test_inputs)
        self.assertEqual(outputs.device, torch.device("cuda:0"))
        # move it back to cpu
        location = "cpu"
        ep = move_to_device_pass(ep, location=location)
        gm = ep.module()
        test_inputs = (torch.rand(1, 10, 4).to("cpu"),)
        outputs = gm(*test_inputs)
        self.assertEqual(outputs.device, torch.device("cpu"))
        # move it to cuda:0 again
        location = {"cpu": "cuda:0"}
        ep = move_to_device_pass(ep, location=location)
        gm = ep.module()
        test_inputs = (torch.rand(1, 10, 4).to("cuda:0"),)
        outputs = gm(*test_inputs)
        self.assertEqual(outputs.device, torch.device("cuda:0"))

    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_move_device_example_inputs(self):
        # [FUNCTION DOC] Requires CUDA. Exports a linear model with positional and keyword example
        # inputs, checks the stored example inputs start on CPU, applies the device pass to a cuda
        # device, and checks the positional and keyword example inputs of the new program are on
        # cuda.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer added to two extra inputs.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer of four features.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the first input and adds the
                # other two inputs.
                return self.linear(x) + y + z

        # Create model with example inputs on CPU
        mod = Model()
        example_args = (torch.rand(4, 4), torch.rand(4, 4))
        example_kwargs = {"z": torch.tensor([1.0, 2.0, 3.0, 4.0])}

        # Export with example inputs
        ep = export(mod, example_args, example_kwargs)

        # Verify initial state - all tensors should be on CPU
        self.assertEqual(ep.example_inputs[0][0].device, torch.device("cpu"))
        self.assertEqual(ep.example_inputs[0][1].device, torch.device("cpu"))
        self.assertEqual(ep.example_inputs[1]["z"].device, torch.device("cpu"))

        # Move to CUDA
        location = torch.device("cuda:0")
        ep_cuda = move_to_device_pass(ep, location=location)

        # Verify example_inputs moved to CUDA
        self.assertEqual(ep_cuda.example_inputs[0][0].device, torch.device("cuda:0"))
        self.assertEqual(ep_cuda.example_inputs[0][1].device, torch.device("cuda:0"))
        self.assertEqual(ep_cuda.example_inputs[1]["z"].device, torch.device("cuda:0"))


if __name__ == "__main__":
    run_tests()
