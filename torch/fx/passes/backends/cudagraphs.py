# [MODULE DOC] Support for partitioning an FX graph into sub-graph modules that can run under CUDA
# graphs: an operator-support policy (CudaGraphsSupport) and the partition_cudagraphs entry point.
# No import-time side effects beyond imports.
import operator
from collections.abc import Mapping, Sequence
from typing import Any

import torch


__all__ = ["CudaGraphsSupport", "partition_cudagraphs"]
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.infra.partitioner import CapabilityBasedPartitioner
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from torch.utils import _pytree as pytree


class CudaGraphsSupport(OperatorSupport):
    # [CLASS DOC] Operator-support policy deciding which nodes may be placed in a CUDA-graph partition.
    # Stateless beyond what OperatorSupport holds; the only override is is_node_supported.
    # TODO: why is submodules passed here
    def is_node_supported(
        self, submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node
    ) -> bool:
        # [FUNCTION DOC] Returns True if the node can run under CUDA graphs. Non-callable nodes
        # (placeholder, output, get_attr) and the embedding dense backward op are rejected; getitem
        # nodes are always accepted; any other node is accepted only if every tensor in the fake values
        # of its inputs and of its own result lives on a CUDA device. Reads the value from the
        # val metadata key, falling back to fake_result. Raises KeyError if neither metadata key exists
        # (fake tensor propagation must have run). The submodules argument is unused. No side effects.
        if node.op not in CALLABLE_NODE_OPS:
            return False

        if node.target is torch.ops.aten.embedding_dense_backward.default:
            return False

        if node.target is operator.getitem:
            return True

        found_not_cuda = False

        def meta_fk(meta: dict[str, Any]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Returns the fake value stored in the metadata mapping, preferring the
            # val key and falling back to fake_result; raises KeyError if both are missing.
            return meta["val"] if "val" in meta else meta["fake_result"]

        def find_not_cuda(t: object) -> None:
            # [FUNCTION DOC] HELPER: Tree-map callback; sets the enclosing found_not_cuda flag when the
            # leaf is a tensor whose device type is not cuda. Non-tensor leaves are ignored.
            nonlocal found_not_cuda
            if isinstance(t, torch.Tensor) and t.device.type != "cuda":
                found_not_cuda = True

        for n in node.all_input_nodes:
            pytree.tree_map_(find_not_cuda, meta_fk(n.meta))

        pytree.tree_map_(find_not_cuda, meta_fk(node.meta))

        # NB: factory function is accounted for because the result would be
        # cpu or cuda

        return not found_not_cuda


def partition_cudagraphs(
    gm: torch.fx.GraphModule, inputs: Sequence[object]
) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Runs fake tensor propagation on the module with the given example inputs (which
    # annotates nodes with fake values, mutating node metadata), proposes partitions of CUDA-only nodes
    # using CudaGraphsSupport (single-node partitions allowed) and fuses them into submodules.
    # Returns the fused graph module. Errors from propagation or partitioning propagate to the caller.
    """
    Partition an FX graph into sub-GraphModules that can be validly run under
    CUDA graphs.  For a subgraph to be runnable under CUDA, all of the operations
    must involve CUDA tensors only.
    """

    FakeTensorProp(gm).propagate(*inputs)
    supported_ops = CudaGraphsSupport()
    # TODO: single node partition may be wrong due to the pessimization
    # from copying in and out the data.  Check in benchmarks, perhaps
    partitioner = CapabilityBasedPartitioner(
        gm, supported_ops, allows_single_node_partition=True
    )
    partitions = partitioner.propose_partitions()
    fused_graph = partitioner.fuse_partitions(partitions)
    return fused_graph
