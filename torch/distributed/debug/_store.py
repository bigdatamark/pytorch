# [MODULE DOC] HELPER: Environment-variable based helpers (rank, world size, TCPStore client)
# used by the distributed debug tooling.
import os

import torch.distributed as dist


def get_rank() -> int:
    # [FUNCTION DOC] Returns the RANK environment variable as int; raises KeyError if unset.
    return int(os.environ["RANK"])


def get_world_size() -> int:
    # [FUNCTION DOC] Returns the WORLD_SIZE environment variable as int; raises KeyError if unset.
    return int(os.environ["WORLD_SIZE"])


def tcpstore_client(prefix: str = "debug_server") -> dist.Store:
    # [FUNCTION DOC] Creates a non-master TCPStore client connecting to MASTER_ADDR and
    # MASTER_PORT from the environment (KeyError if unset), wrapped in a PrefixStore when the
    # prefix is non-empty. Returns the store.
    MASTER_ADDR = os.environ["MASTER_ADDR"]
    MASTER_PORT = int(os.environ["MASTER_PORT"])

    store = dist.TCPStore(
        host_name=MASTER_ADDR,
        port=MASTER_PORT,
        is_master=False,
    )
    if prefix:
        store = dist.PrefixStore(prefix, store)
    return store
