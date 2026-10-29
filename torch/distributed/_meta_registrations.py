# [MODULE DOC] Registers Meta-device implementations for the c10d collective ops (broadcast, allreduce, allgather, reduce_scatter,
# barrier, send/recv, ...) so they can run under fake/meta tracing. Each implementation returns a FakeWork handle
# (plus the first argument for in-place style ops). Import-time side effect: creates a c10d IMPL library and registers
# every entry of _META_FUNCTIONS for the Meta dispatch key.
import random

import torch
from torch._C._distributed_c10d import FakeWork


used_ids: set[int] = set()


def generate_unique_id() -> int:
    # [FUNCTION DOC] HELPER: Return a random integer in 1..10**9 not previously issued, and record it in the module-level used_ids set (mutates global state).
    # Loops until an unused value is found.
    while True:
        new_id = random.randint(1, 10**9)
        if new_id not in used_ids:
            used_ids.add(new_id)
            return new_id


# Function to create and return FakeWork object
def create_fakework(args, return_first_arg=True):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] HELPER: Create a FakeWork, assign it a fresh unique seq_id, and box it into a script object.
    # Returns a pair (first element of args, boxed work) when return_first_arg is true, otherwise only the boxed work.
    work = FakeWork()
    work.seq_id = generate_unique_id()
    fakework_script_obj = work.boxed()
    return (args[0], fakework_script_obj) if return_first_arg else fakework_script_obj


# Dictionary mapping collective operations to their meta functions
# All 20 ops from torch.csrc.distributed.c10d.Ops.cpp are included
# _DEPRECATED_META_FUNCTIONS = {
#     "allreduce_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "allgather_into_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
#     "reduce_scatter_tensor_coalesced_": lambda *args: create_fakework(args, return_first_arg=False),
# }
_META_FUNCTIONS = {
    "broadcast_": lambda *args: create_fakework(args),
    "allreduce_": lambda *args: create_fakework(args),
    "allgather_": lambda *args: create_fakework(args),
    "_allgather_base_": lambda *args: create_fakework(args),
    "reduce_scatter_": lambda *args: create_fakework(args),
    "_reduce_scatter_base_": lambda *args: create_fakework(args),
    "reduce_": lambda *args: create_fakework(args, return_first_arg=False),
    "gather_": lambda *args: create_fakework(args, return_first_arg=False),
    "scatter_": lambda *args: create_fakework(args),
    "alltoall_": lambda *args: create_fakework(args),
    "alltoall_base_": lambda *args: create_fakework(args, return_first_arg=False),
    "barrier": lambda *args: create_fakework(args, return_first_arg=False),
    "monitored_barrier_": lambda *args: None,
    "send": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_": lambda *args: create_fakework(args, return_first_arg=False),
    "recv_any_source_": lambda *args: create_fakework(args, return_first_arg=False),
}

lib_impl = torch.library.Library("c10d", "IMPL")
for op, meta_func in _META_FUNCTIONS.items():
    lib_impl.impl(op, meta_func, "Meta")
