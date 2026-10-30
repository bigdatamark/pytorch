# Owner(s): ["module: fx.passes"]
# ruff: noqa: F841
# [MODULE DOC] Tests for FX graph passes: the capability based partitioner and its fusion step
# (with a mock operator support that only allows add, getitem, view, permute, std_mean and a
# wrapped tuple producer), the fuse-by-partitions utility, topological sort stability, and the
# subgraph matcher. Defines sample functions and modules used as traced inputs, pattern/forward
# model classes with expected match counts, and sets up root logging at warning level at import
# time.

from dataclasses import dataclass
import operator
import logging
import sys

import torch
from torch.fx._symbolic_trace import symbolic_trace

from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.utils.fuser_utils import fuse_by_partitions, topo_sort
from torch.fx.passes.utils.matcher_utils import SubgraphMatcher

from torch.testing._internal.common_utils import run_tests, parametrize, instantiate_parametrized_tests
from torch.testing._internal.jit_utils import JitTestCase

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class TestModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with two linear layers and a parameter used as a traced
    # input for fuser and topo sort tests. Its forward builds a chain of adds named add through
    # add_6 plus two linear calls and a relu, and returns three values.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates two 4 by 4 linear layers and a random 4 by 4 parameter.
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.linear2 = torch.nn.Linear(4, 4)
        self.param = torch.nn.Parameter(torch.rand(4, 4))

    def forward(self, a, b, c):
        # [FUNCTION DOC] HELPER: Computes a sum of the first two inputs, applies the first linear
        # layer, mixes in the third input, the parameter, and second linear layer through a series
        # of adds, ends with a relu. Returns the fourth add, the sixth add, and the relu result.
        add = a + b

        linear_1 = self.linear(add)

        add_1 = add + c
        add_2 = add_1 + self.param
        add_3 = add_1 + linear_1
        add_4 = add_2 + add_3

        linear_2 = self.linear2(add_4)

        add_5 = linear_2 + add_4
        add_6 = add_5 + a
        relu = add_6.relu()

        return add_4, add_6, relu

class TestDeepModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with one linear layer whose forward creates a very long chain of
    # subtractions, to ensure passes do not use recursion that would overflow the interpreter
    # stack.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a single 4 by 4 linear layer (unused in forward).
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, a, b, c):
        # [FUNCTION DOC] HELPER: Adds the first two inputs, adds one, then subtracts the third
        # input once more than the interpreter recursion limit in a loop. Returns the result.
        o = a + b
        o = o + 1.0

        # testing to avoid DFS uses in passes. Since Python has max recursion depth.
        for _ in range(sys.getrecursionlimit() + 1):
            o = o - c

        return o


class TestPartitionFunctions:
    # [CLASS DOC] HELPER: Namespace of static sample functions forward1 through forward21 used as
    # partitioner inputs. Each is a small add/relu graph shaped to hit a specific partitioning
    # case; the test parameter lists give expected node groupings.
    @staticmethod
    def forward1(a, b, c):
        # [FUNCTION DOC] HELPER: Chain of adds with one relu in the middle; returns two of the add
        # results. Used to check vertical grouping around an unsupported op.
        add = a + b
        add_1 = add + b
        add_2 = add_1 + c
        relu_1 = add_2.relu()
        add_3 = add_1 + add_2
        add_4 = add_1 + relu_1 + add_3
        relu_2 = add_4.relu()
        add_5 = relu_2 + add_4
        add_6 = add_5 + add_4
        return add_4, add_6

    @staticmethod
    def forward2(a, b, _):
        # [FUNCTION DOC] HELPER: Adds on two inputs with a relu in the middle that blocks fusion;
        # returns the last add and an earlier add. Third argument is unused.
        add = a + b
        add_1 = add + b
        relu_1 = add_1.relu()  # blocked by this
        add_3 = add_1 + relu_1
        add_4 = add_1 + add_3
        return add_4, add_1

    @staticmethod
    def forward3(a, b, c):
        # [FUNCTION DOC] HELPER: Three independent adds over pairs of inputs, all returned; a
        # horizontal fusion case with common producers.
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return add, add_1, add_2

    @staticmethod
    def forward4(a, b, c):
        # [FUNCTION DOC] HELPER: Three independent adds feeding a where call on the first add,
        # picking between the other two. Horizontal fusion case.
        add = a + b
        add_1 = a + c
        add_2 = b + c
        return torch.where(add > 0, add_1, add_2)

    @staticmethod
    def forward5(a, b, c):
        # [FUNCTION DOC] HELPER: An add feeding a relu (unsupported) branch and an add branch; the
        # add should join the supported right branch. Returns both branches.
        # add should be fused right branch, as left branch is not supported
        add = a + 1
        # left branch
        relu = add.relu()
        # right branch
        add_1 = add + 2
        return relu, add_1

    @staticmethod
    def forward6(a, b, c):
        # [FUNCTION DOC] HELPER: An add feeding two relu branches, neither supported, so the add
        # stands in its own partition. Returns both relu results.
        # add should have its own partition, as neither branches are supported
        add = a + 1
        # left branch
        relu = add.relu()
        # right branch
        relu_1 = add.relu()
        return relu, relu_1

    @staticmethod
    def forward7(a, b, c):
        # [FUNCTION DOC] HELPER: An add feeding two supported branches of different lengths;
        # everything should fuse together. Returns the longer and shorter branch outputs.
        # both branches are supported, all adds should be fused together
        add = a + 1
        # left branch
        add_1 = add + 2
        # right branch is larger
        add_2 = add + 1
        add_3 = add_2 + 1
        return add_3, add_1

    @staticmethod
    def forward8(a, b, c):
        # [FUNCTION DOC] HELPER: An add feeding two add branches that merge again by an add; all in
        # one partition. Returns the merged value.
        # both branches are in the same partition, add should join the same partition
        add = a + 1
        # left branch
        add_1 = add + 2
        # right branch
        add_2 = add + 1
        # left and right branch merges
        add_3 = add_2 + add_1

        return add_3

    @staticmethod
    def forward9(a, b, c):
        # [FUNCTION DOC] HELPER: One add feeding three independent add branches that are stacked
        # into a single output tensor.
        add = a + 1
        # branch 1
        add_1 = add + 1
        # branch 2
        add_2 = add + 1
        # branch_3
        add_3 = add + 1
        out = torch.stack([add_1, add_2, add_3])
        return out

    @staticmethod
    def forward10(a, b, c):
        # [FUNCTION DOC] HELPER: Like the three branch case but the third branch also depends on
        # the second branch; outputs are stacked.
        add = a + 1
        # branch 1
        add_1 = add + 1
        # branch 2
        add_2 = add + 1
        # branch 3: depends on branch 2
        add_3 = add + add_2
        out = torch.stack([add_1, add_2, add_3])
        return out

    @staticmethod
    def forward11(a, b, c):
        # [FUNCTION DOC] HELPER: An add feeding a relu branch, a branch that depends on that relu,
        # and another relu branch; outputs are stacked. Unsupported relus split partitions.
        add = a + 1
        # branch 1
        add_1 = add.relu()
        # branch 2 depends on branch 1
        add_2 = add + add_1
        # branch 3
        add_3 = add.relu()
        out = torch.stack([add_1, add_2, add_3])
        return out

    @staticmethod
    def forward12(a, b, c):
        # [FUNCTION DOC] HELPER: Builds two adds from the first input and relus of each, merged
        # with cross dependencies so merging a group could introduce a cycle; checks the
        # partitioner avoids cycles. Returns two values.
        b0 = a + 1.0
        c0 = a + 1.5
        x0 = b0.relu()
        x1 = c0.relu()
        b1 = b0 + x1
        c1 = c0 + 1.2
        # c2 has dependency on x0 & b0, when we merge {c0, c1, c2}
        # this dependency should be updated to the fusion group and reflected
        # on the decision to not fuse b0 & b1, which forms a cyclic dependency in
        # the new graph
        c2 = x0 + c0
        return b1, c2

    @staticmethod
    def forward13(a, b, c):
        # [FUNCTION DOC] HELPER: Splits the first input into four pieces along dimension zero, adds
        # two of them to the other inputs and returns their sum; exercises a getitem special case.
        a0, a1, a2, a3 = a.split(1, 0)
        b1 = a0 + b
        c1 = a1 + c
        return b1 + c1

    @staticmethod
    def forward14(a, b, c):
        # [FUNCTION DOC] HELPER: Calls the aten std_mean operator on the first input, then adds one
        # to the first result. Exercises tuple outputs with getitem.
        a0, a1 = torch.ops.aten.std_mean(a)
        out = a0 + 1.0
        return out

    @staticmethod
    def forward15(a, b, c):
        # [FUNCTION DOC] HELPER: A view to 2 by 2 followed by alternating permutes and adds, ending
        # with two permutes in a row; used with and without the bookend non-compute pass.
        a0 = torch.ops.aten.view(a, [2, 2])
        a1 = torch.ops.aten.permute(a0, [1, 0])
        a2 = a1 + 1.0
        a3 = torch.ops.aten.permute(a2, [1, 0])
        a4 = a3 + 1.0
        a5 = torch.ops.aten.permute(a4, [1, 0])
        return torch.ops.aten.permute(a5, [1, 0])

    @staticmethod
    def forward16(a, b, c):
        # [FUNCTION DOC] HELPER: Subtracts one, then view, permute, add, permute, add, two
        # permutes, and a final subtract. The unsupported subtracts bound a supported middle
        # section for the bookend pass.
        a0 = a - 1.0
        a1 = torch.ops.aten.view(a0, [2, 2])
        a2 = torch.ops.aten.permute(a1, [1, 0])
        a3 = a2 + 1.0
        a4 = torch.ops.aten.permute(a3, [1, 0])
        a5 = a4 + 1.0
        a6 = torch.ops.aten.permute(a5, [1, 0])
        a7 = torch.ops.aten.permute(a6, [1, 0])
        return a7 - 1.0

    @staticmethod
    def forward17(a, b, c, d, e, f):
        # [FUNCTION DOC] HELPER: Three independent adds over three input pairs, returned as a
        # tuple; used for independent-output and horizontal fusion tests.
        a0 = a + b
        a1 = c + d
        a2 = e + f
        return a0, a1, a2

    @staticmethod
    def forward18(a, b, c):
        # [FUNCTION DOC] HELPER: Calls the aten var_mean operator (not supported by the mock) and
        # returns only the first result, giving an empty partition expectation.
        a0, a1 = torch.ops.aten.var_mean(a)
        return a0

    @staticmethod
    def forward19(a, b, c):
        # [FUNCTION DOC] HELPER: A data dependent chain of adds ending with an add of a constant;
        # returns the last result.
        add = a + b
        add_1 = add + c
        add_2 = add_1 + 1
        return add_2

    @staticmethod
    def forward20(a, b, c):
        # [FUNCTION DOC] HELPER: A relu (unsupported) of the first input feeding two adds with the
        # other inputs; returns both.
        relu = a.relu()
        add = relu + b
        add_1 = relu + c
        return add, add_1

    @staticmethod
    def forward21(a, b, c):
        # [FUNCTION DOC] HELPER: Two adds, a relu of one, and combinations of the relu result and
        # the adds; returns two values. Used to check that vertical partitions avoid cross-
        # partition cycles.
        add = a + b
        add_1 = c + c
        relu = add_1.relu()
        add_2 = add + relu
        add_3 = add + add_1
        return add_2, add_3

@torch.fx.wrap
def _nested_tuple_producer(x):
    # [FUNCTION DOC] HELPER: Opaque (wrapped for tracing) function returning a nested tuple of the
    # input plus one and a pair with input plus two and three. Registered as a leaf call during
    # symbolic tracing.
    """Returns a nested tuple structure: (x+1, (x+2, x+3))"""
    return (x + 1, (x + 2, x + 3))


class TestNestedGetitemFunctions:
    # [CLASS DOC] HELPER: Namespace holding the sample function for the nested getitem chain
    # partitioner test.
    """Test functions for nested getitem chain handling in partitioner."""

    @staticmethod
    def forward_nested_getitem_cross_partition(a, b, c):
        # [FUNCTION DOC] HELPER: Calls the wrapped nested tuple producer, takes the first element
        # and an element of the inner tuple, applies relu (unsupported) to the first element and
        # returns inner element plus relu result plus the second input. Used to check getitem
        # chains stay with the producer.
        result = _nested_tuple_producer(a)
        outer_0 = result[0]
        inner_tuple = result[1]
        inner_0 = inner_tuple[0]
        # UNSUPPORTED operation (relu) - this creates a partition boundary
        # the getitem chain should still be reassigned to producer's partition
        blocked = outer_0.relu()
        # downstream use that crosses partition boundary
        return inner_0 + blocked + b


# A mock OperatorSupport class, where only operator.add is supported
class MockOperatorSupport(OperatorSupport):
    # [CLASS DOC] HELPER: Mock operator support. A node is supported only if it is a function call
    # whose target is addition, getitem, aten view, permute, std_mean or the wrapped tuple
    # producer.
    def is_node_supported(self, submodules, node: torch.fx.Node) -> bool:
        # [FUNCTION DOC] HELPER: Returns True when the node is a call_function node with a target
        # in the supported set listed in the class; submodules are ignored.
        return (node.op == "call_function" and
                node.target in {operator.add, operator.getitem,
                                torch.ops.aten.view,
                                torch.ops.aten.permute,
                                torch.ops.aten.std_mean,
                                _nested_tuple_producer})

@instantiate_parametrized_tests
class TestFXGraphPasses(JitTestCase):
    # [CLASS DOC] Parameterized tests for the capability based partitioner and the fuse-by-
    # partitions utility. Stateless; each test traces a sample function or module symbolically.

    @parametrize("fn, expected_partition, bookend_non_compute_pass", [
        (TestPartitionFunctions.forward1, [["add_7", "add_6"], ["add_5", "add_4", "add_3"], ["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward2, [["add_3", "add_2"], ["add_1", "add"]], False),

        # 1 horizontal fusion with common producer
        (TestPartitionFunctions.forward3, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward4, [["add_2", "add_1", "add"]], False),

        # 2 branches cases
        (TestPartitionFunctions.forward5, [["add_1", "add"]], False),
        (TestPartitionFunctions.forward6, [["add"]], False),
        (TestPartitionFunctions.forward7, [["add_3", "add_2", "add", "add_1"]], False),
        (TestPartitionFunctions.forward8, [["add_3", "add_2", "add", "add_1"]], False),

        # 3 branch cases
        (TestPartitionFunctions.forward9, [['add_3', 'add_2', 'add_1', 'add']], False),
        (TestPartitionFunctions.forward10, [['add_3', 'add_2', 'add', 'add_1']], False),
        (TestPartitionFunctions.forward11, [['add_1'], ['add']], False),

        # 4 not necessarily the only partition, just to verify that there's no cyclic dependency after partition
        (TestPartitionFunctions.forward12, [["add_2", "add_3", "add_4"], ["add", "add_1"]], False),

        # 5 getitem special case
        (TestPartitionFunctions.forward13, [["add_2", "add_1", "add"]], False),
        (TestPartitionFunctions.forward14, [["add", "std_mean", "getitem", "getitem_1"]], False),

        # 6 bookend non_compute pass
        (TestPartitionFunctions.forward15, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward15, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        (TestPartitionFunctions.forward16, [["permute_1", "add_1", "add"]], True),
        (TestPartitionFunctions.forward16, [['add_1', 'add', 'permute_1', 'view', 'permute_2', 'permute_3', 'permute']], False),
        # should be empty partition, not a partition with empty nodes
        (TestPartitionFunctions.forward18, [], False),
    ])
    def test_partitioner(self, fn, expected_partition, bookend_non_compute_pass):
        # [FUNCTION DOC] Symbolically traces the function, optionally lists view and permute as
        # non- compute ops, proposes partitions (single node allowed), applies the bookend removal
        # when requested, and checks the number of partitions and per-partition node name sets (in
        # order) equal the expectation, raising AssertionError otherwise. Fuses the partitions and
        # asserts the fused graph gives the same output as the original on random inputs.
        traced = symbolic_trace(fn)

        non_compute_ops = []
        if bookend_non_compute_pass:
            non_compute_ops = ["torch.ops.aten.view", "torch.ops.aten.permute"]

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True,
                                                 non_compute_ops=non_compute_ops)
        partitions = partitioner.propose_partitions()
        if bookend_non_compute_pass:
            partitioner.remove_bookend_non_compute_ops(partitions)

        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        if len(partitions_name) != len(expected_partition):
            raise AssertionError(f"partition count mismatch: {len(partitions_name)} != {len(expected_partition)}")
        for i in range(len(partitions_name)):
            if set(partitions_name[i]) != set(expected_partition[i]):
                raise AssertionError(f"partition {i} mismatch: {set(partitions_name[i])} != {set(expected_partition[i])}")

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)

    @parametrize("fn, expected_partition", [
        (TestPartitionFunctions.forward17, [['add', 'add_1', 'add_2']]),
    ])
    def test_partitioner_independent_output(self, fn, expected_partition):
        # [FUNCTION DOC] Traces the function, proposes partitions with single node partitions
        # allowed, and checks partition count and name sets against the expectation, raising
        # AssertionError on mismatch. Then fuses and compares the fused graph output with the
        # original on six random inputs.
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()
        partitions_name = [[node.name for node in partition.nodes] for partition in partitions]
        if len(partitions_name) != len(expected_partition):
            raise AssertionError(f"partition count mismatch: {len(partitions_name)} != {len(expected_partition)}")
        for i in range(len(partitions_name)):
            if set(partitions_name[i]) != set(expected_partition[i]):
                raise AssertionError(f"partition {i} mismatch: {set(partitions_name[i])} != {set(expected_partition[i])}")

        fused_graph = partitioner.fuse_partitions(partitions)

        a, b, c, d, e, f = torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4), torch.rand(4)

        expected = fn(a, b, c, d, e, f)
        result = fused_graph(a, b, c, d, e, f)
        torch.testing.assert_close(expected, result)

    @parametrize("fn, expected_partition", [
        # Independent supported outputs are horizontal fusion and stay separate.
        (TestPartitionFunctions.forward17, [["add"], ["add_1"], ["add_2"]]),
        # Data-dependent chains still fuse vertically.
        (TestPartitionFunctions.forward19, [["add", "add_1", "add_2"]]),
        # Consumers of an unsupported producer are not fused horizontally.
        (TestPartitionFunctions.forward20, [["add"], ["add_1"]]),
        # Reachability through unsupported nodes still prevents cyclic partitions.
        (
            TestPartitionFunctions.forward12,
            [["add"], ["add_1", "add_3", "add_4"], ["add_2"]],
        ),
        # Do not leave two vertical partitions with cross-partition cycles.
        (TestPartitionFunctions.forward21, [["add_2"], ["add", "add_1", "add_3"]]),
        # Tuple producer/getitem chains still fuse with their data-dependent user.
        (TestPartitionFunctions.forward14, [["std_mean", "getitem", "getitem_1", "add"]]),
    ])
    def test_partitioner_skip_horizontal_fusion(self, fn, expected_partition):
        # [FUNCTION DOC] Proposes partitions with horizontal fusion skipped and compares the set of
        # node name sets to the expected sets (order agnostic). Fuses and compares output with the
        # original on random inputs, with as many inputs as the function takes positional
        # arguments.
        traced = symbolic_trace(fn)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(
            traced,
            supported_ops,
            allows_single_node_partition=True,
            skip_horizontal_fusion=True,
        )
        partitions = partitioner.propose_partitions()
        partitions_name = [
            frozenset(node.name for node in partition.nodes)
            for partition in partitions
        ]
        expected = [frozenset(partition) for partition in expected_partition]
        self.assertEqual(set(partitions_name), set(expected))

        fused_graph = partitioner.fuse_partitions(partitions)

        args = [torch.rand(4) for _ in range(fn.__code__.co_argcount)]
        expected_result = fn(*args)
        result = fused_graph(*args)
        torch.testing.assert_close(expected_result, result)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_5', 'add_6']],
        [['add', 'add_1', 'add_2']],  # vertical fusion
        [['add_2', 'add_3']],         # horizontal fusion
        [['add_3', 'add_4']],
        [['add_6', 'add_5']],     # arbitrary node order
        [['add_4', 'add_1', 'add_3', 'add_2']],           # arbitrary node order
        [['add_5', 'add_6'], ['add_1', 'add_2', 'add_3', 'add_4']],  # arbitrary partition order
        [['add_5', 'linear2']],   # includes call_function + call_module node
        [['add_6', 'relu']],   # includes call_function + call_module node
        [['param', 'add_2']],   # includes get_attr + call_module nodes
        [['param', 'add_1', 'linear']],   # includes get_attr + call_function + call_module nodes
        [["add", "linear", "add_1", "param", "add_2", "add_3", "add_4", "linear2", "add_5", "add_6", "relu"]],  # full graph
    ])
    def test_fuser_util(self, partition):
        # [FUNCTION DOC] Traces the test module, builds the partition list from the named nodes
        # (each partition an ordered mapping with keys only), fuses via the utility, and checks the
        # fused module output equals the original on random inputs. Cases cover vertical,
        # horizontal, arbitrary order and full graph fusion.
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        partitions = []
        for node_names in partition:
            partitions.append(dict.fromkeys([nodes_by_name[name] for name in node_names]))

        fused_graph = fuse_by_partitions(gm, partitions)

        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)

        expected = m(a, b, c)
        result = fused_graph(a, b, c)

        torch.testing.assert_close(expected, result)

    @parametrize("partition", [
        [['add', 'add_1'], ['add_1', 'add_5', 'add_6']],  # add_1 exists in multiple partitions
        [['add', 'add_1', 'add_3']],    # invalid partition: circular dependency
        [['add_4', 'add_5']],    # invalid partition: circular dependency
        [['relu', 'add_5']],    # invalid partition: circular dependency
    ])
    def test_fuser_util_xfail(self, partition):
        # [FUNCTION DOC] Builds partitions from named nodes as in the success test, but with
        # invalid inputs (duplicate node in two partitions or a partition that creates a cycle) and
        # asserts that the fuse utility raises an exception.
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name : node for node in gm.graph.nodes}

        partitions = []
        for node_names in partition:
            partitions.append(dict.fromkeys([nodes_by_name[name] for name in node_names]))

        with self.assertRaises(Exception):
            fuse_by_partitions(gm, partitions)

    def test_fuser_pass_deep_model(self):
        # [FUNCTION DOC] Traces the deep module and proposes partitions with a single node
        # partition allowed. Passes if no exception such as recursion error occurs; there are no
        # other assertions.
        m = TestDeepModule()
        traced = symbolic_trace(m)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(traced,
                                                 supported_ops,
                                                 allows_single_node_partition=True)
        partitions = partitioner.propose_partitions()

    def test_topo_sort_stability(self):
        # [FUNCTION DOC] Calls the topological sort helper on node lists in two input orders for
        # independent nodes, and asserts dependencies come first, independent nodes keep the order
        # they were given, and the dependent node is last.
        """Test that topo_sort preserves relative order of independent nodes."""
        m = TestModule()
        gm = symbolic_trace(m)

        nodes_by_name = {node.name: node for node in gm.graph.nodes}

        # add_2 and linear depend on add_1 but are independent of each other.
        # add_3 depends on both add_2 and linear (via add_1 -> linear -> add_3
        # and add_1 -> add_2 -> add_3 paths in TestModule).
        # Provide them in a specific order and verify it is preserved.
        node_list = [
            nodes_by_name["add_1"],
            nodes_by_name["add_2"],
            nodes_by_name["linear"],
            nodes_by_name["add_3"],
        ]
        sorted_nodes = topo_sort(node_list)
        sorted_names = [n.name for n in sorted_nodes]

        # add_1 must come first (dependency of add_2, linear, add_3)
        self.assertEqual(sorted_names[0], "add_1")
        # add_2 and linear are independent; stable sort preserves input order
        self.assertEqual(sorted_names[1], "add_2")
        self.assertEqual(sorted_names[2], "linear")
        # add_3 depends on both, so it comes last
        self.assertEqual(sorted_names[3], "add_3")

        # Now reverse the order of the independent nodes and verify stability
        # is with respect to the new input order.
        node_list_reversed = [
            nodes_by_name["add_1"],
            nodes_by_name["linear"],
            nodes_by_name["add_2"],
            nodes_by_name["add_3"],
        ]
        sorted_nodes_reversed = topo_sort(node_list_reversed)
        sorted_names_reversed = [n.name for n in sorted_nodes_reversed]

        self.assertEqual(sorted_names_reversed[0], "add_1")
        # Now linear should come before add_2 since that's the input order
        self.assertEqual(sorted_names_reversed[1], "linear")
        self.assertEqual(sorted_names_reversed[2], "add_2")
        self.assertEqual(sorted_names_reversed[3], "add_3")

    def test_fuse_preserves_original_placeholder_input_order(self):
        # [FUNCTION DOC] Fuses all non placeholder, non output nodes of a two input module into one
        # partition and checks the fused submodule placeholders and the call node arguments keep
        # the original input order. Also compares numeric output with the original module.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding one to the attention mask, doubling the input ids,
            # and returning their sum.
            def forward(self, input_ids, attention_mask):
                # [FUNCTION DOC] HELPER: Returns mask plus one summed with ids times two.
                mask = attention_mask + 1
                ids = input_ids * 2
                return mask + ids

        gm = symbolic_trace(M())
        partition = {
            node: None
            for node in gm.graph.nodes
            if node.op not in ("placeholder", "output")
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["input_ids", "attention_mask"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        input_ids, attention_mask = torch.rand(4), torch.rand(4)
        torch.testing.assert_close(
            M()(input_ids, attention_mask),
            fused_graph(input_ids, attention_mask),
        )

    def test_fuse_preserves_original_placeholder_input_order_with_internal_deps(self):
        # [FUNCTION DOC] Same as the placeholder order test but for a three input module where an
        # internal value depends on two inputs; verifies the fused submodule inputs follow the
        # original placeholder order and numerics match.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module computing the product of the second input and the sum of
            # the first and third.
            def forward(self, x, y, z):
                # [FUNCTION DOC] HELPER: Returns the sum of x and z multiplied by y.
                a = x + z
                return a * y

        gm = symbolic_trace(M())
        partition = {
            node: None
            for node in gm.graph.nodes
            if node.op not in ("placeholder", "output")
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["x", "y", "z"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        x, y, z = torch.rand(4), torch.rand(4), torch.rand(4)
        torch.testing.assert_close(M()(x, y, z), fused_graph(x, y, z))

    def test_fuse_preserves_intermediate_input_encounter_order(self):
        # [FUNCTION DOC] Fuses only a multiply and the add that follows it; the fused inputs are
        # intermediate values and must appear in the order they are first encountered (add_1 then
        # add). Checks call args and numerics.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding one to each input, multiplying the results and
            # adding one.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the product of y plus one and x plus one, plus
                # one.
                a = x + 1
                b = y + 1
                c = b * a
                return c + 1

        gm = symbolic_trace(M())
        nodes_by_name = {node.name: node for node in gm.graph.nodes}
        partition = {
            nodes_by_name["mul"]: None,
            nodes_by_name["add_2"]: None,
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["add_1", "add"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        x, y = torch.rand(4), torch.rand(4)
        torch.testing.assert_close(M()(x, y), fused_graph(x, y))

    def test_fuse_preserves_mixed_input_encounter_order(self):
        # [FUNCTION DOC] Fuses a second matrix multiply and an add. The fused inputs mix a computed
        # value and graph inputs and must follow encounter order. Checks the submodule placeholder
        # names, the call args, and numerical equality on 4 by 4 inputs.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two matrix multiplies, an add and a subtract reusing
            # the input names.
            def forward(self, a, x, b):
                # [FUNCTION DOC] HELPER: Computes mm of a and x, adds b, subtracts a, multiplies by
                # x again and adds b. Returns the last add.
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                return y + b

        gm = symbolic_trace(M())
        nodes_by_name = {node.name: node for node in gm.graph.nodes}
        partition = {
            nodes_by_name["mm_1"]: None,
            nodes_by_name["add_1"]: None,
        }

        fused_graph = fuse_by_partitions(gm, [partition])
        fused_submodule = fused_graph.get_submodule("fused_0")
        submodule_inputs = [
            node.name for node in fused_submodule.graph.nodes if node.op == "placeholder"
        ]
        self.assertEqual(submodule_inputs, ["sub", "x", "b"])

        fused_node = next(
            node for node in fused_graph.graph.nodes if node.op == "call_module"
        )
        self.assertEqual([node.name for node in fused_node.args], submodule_inputs)

        a, x, b = torch.rand(4, 4), torch.rand(4, 4), torch.rand(4, 4)
        torch.testing.assert_close(M()(a, x, b), fused_graph(a, x, b))

    def test_partitioner_nested_getitem_chains(self):
        # [FUNCTION DOC] Traces the nested getitem sample, proposes partitions, finds the producer
        # node and all getitem nodes, asserts at least three getitems exist, and for getitems that
        # trace back to the producer through getitem chains asserts they are in the same partition
        # as the producer (when the producer is in a partition). Finally fuses and compares output
        # numerically.
        """Test that nested getitem chains are properly reassigned to producer's partition.

        this tests the fix for handling patterns like:
            producer_node = call(...)  # returns ((a, b), c)
            getitem_1 = producer_node[0]  # gets (a, b)
            getitem_2 = getitem_1[0]  # gets a

        without the iterative fix, getitem_2 would be assigned to getitem_1's original
        partition instead of the producer's partition.
        """
        traced = symbolic_trace(TestNestedGetitemFunctions.forward_nested_getitem_cross_partition)

        supported_ops = MockOperatorSupport()
        partitioner = CapabilityBasedPartitioner(
            traced,
            supported_ops,
            allows_single_node_partition=True
        )
        partitions = partitioner.propose_partitions()

        node_to_partition: dict[torch.fx.Node, int] = {}
        for idx, partition in enumerate(partitions):
            for node in partition.nodes:
                node_to_partition[node] = idx

        producer_node = None
        getitem_nodes = []
        for node in traced.graph.nodes:
            if node.op == "call_function":
                if node.target == _nested_tuple_producer:
                    producer_node = node
                elif node.target == operator.getitem:
                    getitem_nodes.append(node)

        self.assertIsNotNone(producer_node, "Should find the nested tuple producer node")
        self.assertGreaterEqual(len(getitem_nodes), 3, "Should find at least 3 getitem nodes")

        # all getitems that derive from the producer (including nested ones)
        # should be in the same partition as the producer
        if producer_node in node_to_partition:
            producer_partition = node_to_partition[producer_node]

            for getitem_node in getitem_nodes:
                current = getitem_node.args[0]
                derives_from_producer = False
                while hasattr(current, 'op'):
                    if current == producer_node:
                        derives_from_producer = True
                        break
                    if current.op == "call_function" and current.target == operator.getitem:
                        current = current.args[0]
                    else:
                        break

                if derives_from_producer:
                    getitem_partition = node_to_partition.get(getitem_node)
                    self.assertEqual(
                        getitem_partition,
                        producer_partition,
                        lambda msg: f"{msg}\nGetitem node '{getitem_node.name}' (nested in chain from producer) "
                        f"should be in same partition as producer. "
                        f"Got partition {getitem_partition}, expected {producer_partition}"
                    )

        fused_graph = partitioner.fuse_partitions(partitions)
        a, b, c = torch.rand(4), torch.rand(4), torch.rand(4)
        expected = TestNestedGetitemFunctions.forward_nested_getitem_cross_partition(a, b, c)
        result = fused_graph(a, b, c)
        torch.testing.assert_close(expected, result)

@dataclass
class TestCase:
    # [CLASS DOC] HELPER: Dataclass describing a matcher expectation: whether the output and
    # placeholders are matched, the expected number of matches, and whether overlapping matches are
    # removed (default true). Not a unittest case despite its name.
    match_output: bool
    match_placeholder: bool
    num_matches: int
    remove_overlapping_matches: bool = True

class SingleNodePattern:
    # [CLASS DOC] HELPER: Matcher model: forward negates and adds the result to itself, and the
    # pattern is the negate; test_cases lists the expected match counts per flag combination.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Negates the input and returns the sum of the result with itself.
        val = torch.neg(x)
        return torch.add(val, val)

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: negate of the input.
        return torch.neg(a)

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]
class SimplePattern:
    # [CLASS DOC] HELPER: Matcher model with three concatenate-and-sum operations; the pattern is
    # concatenate then sum. Expected counts are in test_cases.
    @staticmethod
    def forward(x, w1, w2):
        # [FUNCTION DOC] HELPER: Builds two concat sums, concat sum of those, and returns the input
        # plus maxima of the first two plus the third.
        m1 = torch.cat([w1, w2]).sum()
        m2 = torch.cat([w2, w1]).sum()
        m3 = torch.cat([m1, m2]).sum()
        return x + torch.max(m1) + torch.max(m2) + m3

    @staticmethod
    def pattern(a, b):
        # [FUNCTION DOC] HELPER: Pattern: concatenate two inputs then sum.
        return torch.cat([a, b]).sum()

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3),
        TestCase(True, False, 0),
        TestCase(False, True, 2),
        TestCase(True, True, 0)
    ]

class SimpleFullGraphMatching:
    # [CLASS DOC] HELPER: Matcher model where the pattern equals the whole forward graph (negate
    # then add to itself), so exactly one match is expected for every flag choice.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Negates the input and returns the sum of the result with itself.
        a = torch.neg(x)
        return torch.add(a, a)

    @staticmethod
    def pattern(x):
        # [FUNCTION DOC] HELPER: Same computation as the model forward, used as the pattern.
        a = torch.neg(x)
        return torch.add(a, a)

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 1),
        TestCase(True, True, 1)
    ]

class DiamondShapePatternTestCase:
    # [CLASS DOC] HELPER: Matcher model with a diamond shaped subgraph (relu, then sigmoid and relu
    # branches that are added), preceded by a negate.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Negates input, applies relu, then returns sigmoid plus relu of
        # that value.
        a = torch.neg(x)

        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right

        return out

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu, then add of a sigmoid branch and a relu branch.
        a = a.relu()
        left = a.sigmoid()
        right = a.relu()
        out = left + right
        return out

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class NonFullyContainedMatches:
    # [CLASS DOC] HELPER: Matcher model where one occurrence of the pattern is fully contained and
    # another leaks an intermediate value to outside users.
    @staticmethod
    def forward(x, w1, w2, b1, b2):
        # [FUNCTION DOC] HELPER: Builds two concatenations and an addmm, summing the result (not
        # leaking); repeats with the first concatenation also summed and thus leaking. Returns two
        # sums.
        # fully contained matched subgraph
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        t0 = torch.addmm(b1, m1, m2.t())
        t0_sum = torch.sum(t0)   # use of t0 is not leaking

        # leaking matched subgraph, m3 is leaked
        m3 = torch.cat([w1, w2])
        m4 = torch.cat([x, b2])
        t1 = torch.addmm(b1, m3, m4.t())
        m3_sum = torch.sum(m3)

        return t0_sum, m3_sum

    @staticmethod
    def pattern(x, w1, w2, b1, b2):
        # [FUNCTION DOC] HELPER: Pattern: two concatenations then addmm with the transposed second.
        m1 = torch.cat([w1, w2])
        m2 = torch.cat([x, b2])
        return torch.addmm(b1, m1, m2.t())

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),

        TestCase(True, False, 0),

        TestCase(False, True, 1),     # leaked used of placeholder is not leaking
    ]

class ChainRepeatedPattern:
    # [CLASS DOC] HELPER: Matcher model with four chained sigmoids; the pattern is two chained
    # sigmoids, testing overlapping match handling.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Applies sigmoid four times in sequence and returns the result.
        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        x = torch.sigmoid(x)
        return torch.sigmoid(x)

    @staticmethod
    def pattern(x):
        # [FUNCTION DOC] HELPER: Pattern: two sigmoids in sequence.
        return torch.sigmoid(torch.sigmoid(x))

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 3, remove_overlapping_matches=False),
        TestCase(False, False, 2, remove_overlapping_matches=True),
        TestCase(True, False, 1),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]

class QuantizationModel:
    # [CLASS DOC] HELPER: Matcher model with an in-place add followed by dequantize, sigmoid and
    # conversion to half precision; the pattern is the last three.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds three in place, dequantizes, applies sigmoid, converts to
        # float16 and returns.
        x += 3
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    @staticmethod
    def pattern(x):
        # [FUNCTION DOC] HELPER: Pattern: dequantize, sigmoid, convert to float16.
        x = x.dequantize()
        x = torch.sigmoid(x)
        x = x.to(torch.float16)
        return x

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultipleOutputsWithDependency:
    # [CLASS DOC] HELPER: Matcher model whose pattern has two outputs, one depending on the other.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Relu then sigmoid; returns the sigmoid result and the relu result.
        y = x.relu()
        z = y.sigmoid()
        return z, y

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu then sigmoid, returning both with a data dependency.
        b = a.relu()
        c = b.sigmoid()
        return b, c     # outputs have data dependency

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 1),
        TestCase(True, True, 0)
    ]

class MultipleOutputsWithoutDependency:
    # [CLASS DOC] HELPER: Matcher model whose pattern has two outputs that do not depend on each
    # other.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one, then relu, sum and sigmoid of the relu; returns sigmoid
        # of the sigmoid plus the sum of the sum.
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        out = y.sigmoid() + z.sum()
        return out

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu, then sigmoid and sum; returns the sigmoid and the
        # sum.
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultipleOutputsMultipleOverlappingMatches:
    # [CLASS DOC] HELPER: Matcher model with multiple overlapping matches for a three output
    # pattern, tested with and without overlap removal.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one, relu, two sums and two sigmoids of the relu, and returns
        # their sum.
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        z1 = x.sum()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return z + z1 + y + y1

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu, sigmoid, sum; returns the relu, sigmoid and sum.
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return a, b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 4, remove_overlapping_matches=False),
        TestCase(False, False, 1, remove_overlapping_matches=True),
    ]

class MultipleOutputsMultipleNonOverlappingMatches:
    # [CLASS DOC] HELPER: Matcher model with two disjoint occurrences of a two output pattern.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one, then two successive relu, sum and sigmoid groups;
        # returns the sum of all four results.
        x = x + 1

        # target subgraph to match
        x = x.relu()
        z = x.sum()
        y = x.sigmoid()

        x = x.relu()
        z1 = x.sum()
        y1 = x.sigmoid()

        return z + z1 + y + y1

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu, then sigmoid and sum; returns sigmoid and sum.
        a = a.relu()
        b = a.sigmoid()
        c = a.sum()
        return b, c

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
    ]

class MultipleOutputsIdenticalAnchor:
    # [CLASS DOC] HELPER: Matcher model whose pattern has two identical sigmoid outputs (identical
    # anchors). A commented-out case notes a known miscount.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one, relu, two sigmoids of the relu; returns both.
        x = x + 1

        # target subgraph to match
        x = x.relu()
        y = x.sigmoid()
        y1 = x.sigmoid()

        return y, y1

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu then two sigmoids; returns both.
        a = a.relu()
        b = a.sigmoid()
        b1 = a.sigmoid()
        return b, b1

    test_cases = [
        # match_output, match_placeholder, num_matches
        # (False, False, 2),  # FIXME: currently still matches to 2, should fix to 1
        TestCase(True, False, 1),
        TestCase(False, True, 0),
    ]


class MultipleOutputsHorizontalPattern:
    # [CLASS DOC] HELPER: Matcher model with a horizontal two output pattern of relu and sigmoid on
    # the same input.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one and returns relu and sigmoid of the result.
        x = x + 1

        # target subgraph to match
        y1 = x.relu()
        y2 = x.sigmoid()

        return y1, y2

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu and sigmoid of the same input, both returned.
        b1 = a.relu()
        b2 = a.sigmoid()

        return b1, b2

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
        TestCase(True, False, 1),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

class MultiOutputWithWithInvalidMatches:
    # [CLASS DOC] HELPER: Matcher model where a multi output pattern finds only invalid matches,
    # expecting zero.
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Linear with a random weight, sigmoid, multiply with the linear
        # output, then sum along dimension one.
        res0 = torch.nn.functional.linear(x, torch.rand(3, 3))
        res1 = torch.sigmoid(res0)
        res2 = res0 * res1
        res3 = torch.sum(res2, dim=1)
        return res3

    @staticmethod
    def pattern(a, b, c):
        # [FUNCTION DOC] HELPER: Pattern: linear, then multiply the linear result by a third input;
        # returns linear and product.
        lin_res = torch.nn.functional.linear(a, b)
        mul_res = lin_res * c
        return lin_res, mul_res

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 0),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
    ]

class QuantizationFp8Pattern:
    # [CLASS DOC] HELPER: Matcher model using custom fp8 quantize and dequantize ops registered in
    # a temporary library. The setup and tearDown class methods create and destroy that library
    # (kept in a class attribute).
    @classmethod
    def setup(cls):
        # [FUNCTION DOC] HELPER: Defines a library for the fp8 quantization namespace with schemas
        # for the quantize and dequantize per tensor ops, storing it on the class.
        cls.quantization = torch.library.Library("fp8_quantization", "DEF")  # noqa: SCOPED_LIBRARY
        cls.quantization.define("quantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")
        cls.quantization.define("dequantize_per_tensor_affine_fp8(Tensor self, int dtype, float scale) -> Tensor")

    @classmethod
    def tearDown(cls):
        # [FUNCTION DOC] HELPER: Destroys the library created in setup.
        cls.quantization._destroy()

    @staticmethod
    def forward(self, arg0_1, arg1_1):
        # [FUNCTION DOC] HELPER: Despite being declared static, takes a first argument named self
        # that tracing turns into a placeholder; builds three quantize/dequantize pairs around two
        # adds with scale read from an attribute of that argument. Returns the last dequantized
        # value.
        qt = torch.ops.fp8_quantization
        _scale_0 = self._scale_0
        quantize_per_tensor_affine_fp8 = qt.quantize_per_tensor_affine_fp8(arg0_1, 0, _scale_0)
        dequantize_per_tensor_affine_fp8 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8, 0, _scale_0)
        _scale_1 = self._scale_0
        quantize_per_tensor_affine_fp8_1 = qt.quantize_per_tensor_affine_fp8(arg1_1, 0, _scale_1)
        dequantize_per_tensor_affine_fp8_1 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_1, 0, _scale_1)
        add = torch.ops.aten.add.Tensor(dequantize_per_tensor_affine_fp8, dequantize_per_tensor_affine_fp8_1)
        _scale_2 = self._scale_0
        quantize_per_tensor_affine_fp8_2 = qt.quantize_per_tensor_affine_fp8(add, 0, _scale_2)
        dequantize_per_tensor_affine_fp8_2 = qt.dequantize_per_tensor_affine_fp8(quantize_per_tensor_affine_fp8_2, 0, _scale_2)
        return dequantize_per_tensor_affine_fp8_2

    @staticmethod
    def pattern(a, a_dtype, a_scale, b, b_dtype, b_scale, out_scale):
        # [FUNCTION DOC] HELPER: Pattern: dequantize two inputs, add, then quantize the sum.
        # Contains a stray statement that only references the dequantize op and has no effect.
        qt = torch.ops.fp8_quantization
        a = qt.dequantize_per_tensor_affine_fp8(a, a_dtype, a_scale)
        b = qt.dequantize_per_tensor_affine_fp8(b, b_dtype, b_scale)
        output = torch.ops.aten.add.Tensor(a, b)

        qt.dequantize_per_tensor_affine_fp8

        output = qt.quantize_per_tensor_affine_fp8(output, a_dtype, out_scale)
        return output

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 1),
    ]

class NoAnchorFound:
    # [CLASS DOC] HELPER: Matcher model where the pattern has no anchor in the target, so no
    # matches are expected.
    # This test case is for pattern where no matching anchor is found in the target graph
    # `anchor` is the starting point of the pattern matching, it's usually the boundary returning nodes
    @staticmethod
    def forward(x):
        # [FUNCTION DOC] HELPER: Adds one to the input and returns it.
        x = x + 1
        return x

    @staticmethod
    def pattern(a):
        # [FUNCTION DOC] HELPER: Pattern: relu of the input.
        b1 = a.relu()
        return b1

    test_cases = [
        # match_output, match_placeholder, num_matches
        TestCase(False, False, 0),
        TestCase(True, False, 0),
        TestCase(False, True, 0),
        TestCase(True, True, 0)
    ]

@instantiate_parametrized_tests
class TestFXMatcherUtils(JitTestCase):
    # [CLASS DOC] Parameterized tests for the subgraph matcher over the model classes defined
    # above. Stateless.

    @parametrize("test_model", [
        SingleNodePattern,
        SimplePattern,
        SimpleFullGraphMatching,
        DiamondShapePatternTestCase,
        NonFullyContainedMatches,
        ChainRepeatedPattern,
        QuantizationModel,
        MultipleOutputsWithDependency,
        MultipleOutputsWithoutDependency,
        MultipleOutputsMultipleOverlappingMatches,
        MultipleOutputsMultipleNonOverlappingMatches,
        MultipleOutputsIdenticalAnchor,
        MultipleOutputsHorizontalPattern,
        MultiOutputWithWithInvalidMatches,
        QuantizationFp8Pattern,
        NoAnchorFound,
    ])
    def test_subgraph_matcher(self, test_model):
        # [FUNCTION DOC] Runs the optional class setup, traces the model forward and pattern, then
        # for each expectation builds a matcher with the given flags, matches against the traced
        # graph and raises AssertionError if the count differs or if any pattern node (skipping
        # placeholders or outputs per flags) is missing from a match. Finally calls the class
        # tearDown.
        # KNOWN ISSUE: the tearDown step is guarded by a check on setup being callable rather than
        # tearDown, so it would fail for a model with setup but no tearDown.

        setup = getattr(test_model, "setup", None)
        if callable(setup):
            setup()

        traced = symbolic_trace(test_model.forward)
        pattern_traced = symbolic_trace(test_model.pattern)

        for test_case in test_model.test_cases:

            matcher = SubgraphMatcher(pattern_traced.graph,
                                      match_output=test_case.match_output,
                                      match_placeholder=test_case.match_placeholder,
                                      remove_overlapping_matches=test_case.remove_overlapping_matches)
            matches = matcher.match(traced.graph)

            if len(matches) != test_case.num_matches:
                raise AssertionError(f"match count mismatch: {len(matches)} != {test_case.num_matches}")

            for match in matches:
                for node in pattern_traced.graph.nodes:
                    if not test_case.match_placeholder and node.op == "placeholder":
                        continue
                    if not test_case.match_output and node.op == "output":
                        continue
                    if node not in match.nodes_map:
                        raise AssertionError(f"node {node} not in match.nodes_map")

        tearDown = getattr(test_model, "tearDown", None)
        if callable(setup):
            tearDown()


if __name__ == "__main__":
    run_tests()
