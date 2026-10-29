# [MODULE DOC] Wrappers over the C++ lazy computation APIs: inspect device-data nodes of an IR graph,
# get a graph hash, and run a cached graph. Device-data and run-cached helpers are TorchScript-backend specific.
from collections.abc import Sequence

import torch
import torch._C._lazy
import torch._C._lazy_ts_backend


def get_tensors_ts_device_data_node(
    tensors: Sequence[torch.Tensor],
) -> tuple[list[int], list[object]]:
    # [FUNCTION DOC] Returns two parallel lists (tensor ids, eager tensors) for the DeviceData nodes in the IR of the
    # given lazy tensors. TorchScript backend only.
    """Return tensor ids and eager tensors for DeviceData nodes in the
    IR for the passed in lazy tensors.

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(list(tensors))


def get_graph_hash(tensors: Sequence[torch.Tensor]) -> str:
    # [FUNCTION DOC] Returns the string hash of the IR graph that produces the given lazy tensors.
    """Return the graph hash for the passed in lazy tensors"""
    return torch._C._lazy._get_graph_hash(list(tensors))


def run_cached_graph(
    hash_str: str, graph_inputs: Sequence[object]
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Executes the cached computation identified by hash_str with the given inputs and returns the
    # resulting tensors. TorchScript backend only.
    """Running the cached computation graph with the given inputs

    TODO: This API is currently ts backend specific. We are working on
    generalizing it to all backends including XLA.
    """
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, list(graph_inputs))
