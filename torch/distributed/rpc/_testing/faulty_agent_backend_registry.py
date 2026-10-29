#!/usr/bin/env python3
# mypy: allow-untyped-defs
# [MODULE DOC] Registers the FAULTY_TENSORPIPE RPC backend (a TensorPipe agent that can
# fail or delay chosen messages, used for tests). Import-time side effect: calls
# register_backend on the rpc backend registry with the two handlers defined here.

import torch.distributed as dist
import torch.distributed.rpc as rpc


def _faulty_tensorpipe_construct_rpc_backend_options_handler(
    rpc_timeout,
    init_method,
    num_worker_threads,
    messages_to_fail,
    messages_to_delay,
    num_fail_sends,
    **kwargs,
):
    # [FUNCTION DOC] HELPER: Backend-registry callback that builds a
    # FaultyTensorPipeRpcBackendOptions from the given timeout, init method, thread count and
    # fault settings (messages to fail/delay, number of failed sends). Extra kwargs are ignored.
    from . import FaultyTensorPipeRpcBackendOptions

    return FaultyTensorPipeRpcBackendOptions(
        num_worker_threads=num_worker_threads,
        rpc_timeout=rpc_timeout,
        init_method=init_method,
        messages_to_fail=messages_to_fail,
        messages_to_delay=messages_to_delay,
        num_fail_sends=num_fail_sends,
    )


def _faulty_tensorpipe_init_backend_handler(
    store, name, rank, world_size, rpc_backend_options
):
    # [FUNCTION DOC] HELPER: Backend-registry callback that creates a FaultyTensorPipeAgent
    # with empty reverse device map and device list, and installs it as the process RPC state
    # via api._init_rpc_states. Returns the agent. Raises TypeError if store is not a
    # c10d Store or the options are not FaultyTensorPipeRpcBackendOptions.
    from torch.distributed.rpc import api

    from . import FaultyTensorPipeAgent, FaultyTensorPipeRpcBackendOptions

    if not isinstance(store, dist.Store):
        raise TypeError(f"`store` must be a c10d::Store. {store}")

    if not isinstance(rpc_backend_options, FaultyTensorPipeRpcBackendOptions):
        raise TypeError(
            f"`rpc_backend_options` must be a `FaultyTensorPipeRpcBackendOptions`. {rpc_backend_options}"
        )

    agent = FaultyTensorPipeAgent(
        store,
        name,
        rank,
        world_size,
        rpc_backend_options,
        {},  # reverse_device_map
        [],  # devices
    )
    api._init_rpc_states(agent)

    return agent


rpc.backend_registry.register_backend(
    "FAULTY_TENSORPIPE",
    _faulty_tensorpipe_construct_rpc_backend_options_handler,
    _faulty_tensorpipe_init_backend_handler,
)
