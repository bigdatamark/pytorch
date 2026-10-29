# Owner(s): ["module: fx"]
# [MODULE DOC] Tests that CapabilityBasedPartitioner keeps the original FX graph node order
# inside a proposed partition, both for one run and across repeated runs. Defines a dummy
# operator support that accepts every node and a partitioner subclass using it.
# Running the file directly raises a RuntimeError since it is not used by test discovery.

from collections.abc import Mapping

import torch
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.testing._internal.common_utils import TestCase


class DummyDevOperatorSupport(OperatorSupport):
    # [CLASS DOC] HELPER: OperatorSupport implementation that reports every node as supported,
    # so the whole graph forms one partition.
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # [FUNCTION DOC] HELPER: Ignores the submodules mapping and the node and
        # always returns true.
        return True


class DummyPartitioner(CapabilityBasedPartitioner):
    # [CLASS DOC] HELPER: CapabilityBasedPartitioner configured with the always-supported
    # operator support and with single node partitions allowed.
    def __init__(self, graph_module: torch.fx.GraphModule):
        # [FUNCTION DOC] HELPER: Passes the graph module, a DummyDevOperatorSupport instance and
        # allows_single_node_partition set to true to the base class constructor.
        super().__init__(
            graph_module,
            DummyDevOperatorSupport(),
            allows_single_node_partition=True,
        )


# original graph node order is: ['x', 'add', 'add_1', 'output']
class AddModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module computing the sum of the input with itself and then adds the
    # input again; its traced graph has nodes x, add, add_1 and output.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the input to itself, adds the input again to that result
        # and returns it.
        y = torch.add(x, x)
        z = torch.add(y, x)
        return z


class TestPartitionerOrder(TestCase):
    # [CLASS DOC] Test case checking partition node ordering stability for
    # the dummy partitioner.
    # partitioner test to check graph node order remains the same with the original graph after partitioning
    def test_partitioner_graph_node_order(self):
        # [FUNCTION DOC] Traces AddModule, proposes partitions, and asserts the node names of
        # the first partition appear in the same order as the nodes of the traced graph.
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        origin_node_order = [n.name for n in traced_m.graph.nodes]
        partitions = DummyPartitioner(traced_m).propose_partitions()
        partition_nodes = [list(partition.nodes) for partition in partitions]
        partition_node_order = [n.name for n in partition_nodes[0]]
        self.assertTrue(partition_node_order == origin_node_order)

    # partitioner test to check graph node order remains the same during multiple runs
    def test_partitioner_multiple_runs_order(self):
        # [FUNCTION DOC] Proposes partitions for a freshly traced AddModule, then repeats the
        # trace and partitioning ten times and asserts each time that the first partition's node
        # name order equals the first run's order.
        m = AddModule()
        traced_m = torch.fx.symbolic_trace(m)
        partitions = DummyPartitioner(traced_m).propose_partitions()
        partition_nodes = [list(partition.nodes) for partition in partitions]
        node_order = [n.name for n in partition_nodes[0]]
        for _ in range(10):
            traced_m = torch.fx.symbolic_trace(m)
            new_partition = DummyPartitioner(traced_m).propose_partitions()
            new_partition_nodes = [list(partition.nodes) for partition in new_partition]
            new_node_order = [n.name for n in new_partition_nodes[0]]
            self.assertTrue(node_order == new_node_order)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
