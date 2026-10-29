# [PACKAGE DOC] Testing-only RPC support package. When the C++ faulty agent bindings are
# built in, it initializes them (raising RuntimeError on failure), re-exports
# FaultyTensorPipeAgent and FaultyTensorPipeRpcBackendOptions, and imports the registry
# module that registers the FAULTY_TENSORPIPE backend. Exports is_available.
import torch


def is_available() -> bool:
    # [FUNCTION DOC] Returns True if the torch C extension was built with the faulty agent
    # init hook, i.e. fault-injection RPC testing support is present.
    return hasattr(torch._C, "_faulty_agent_init")


if is_available() and not torch._C._faulty_agent_init():
    raise RuntimeError("Failed to initialize torch.distributed.rpc._testing")

if is_available():
    # Registers FAULTY_TENSORPIPE RPC backend.
    from torch._C._distributed_rpc_testing import (
        FaultyTensorPipeAgent,
        FaultyTensorPipeRpcBackendOptions,
    )

    from . import faulty_agent_backend_registry
