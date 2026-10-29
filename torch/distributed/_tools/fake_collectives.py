# [MODULE DOC] Classification tables for c10d collective operators, used by tools that
# trace collectives under fake process groups. Defines at import time (resolving
# torch.ops namespaces) the sets non_functional_collectives, functional_collectives
# (including DTensor all-to-all and p2p batch ops), sync_ops (barriers and
# wait_tensor) and their union collective_ops, plus the CollectiveOp helper class
# that, given an op overload, finds its process group, communicated byte size and work
# handle.
from typing import Any

import torch
from torch._C._distributed_c10d import (
    _resolve_process_group,
    FakeWork,
    ProcessGroup,
    Work,
)
from torch.utils._pytree import tree_map_only


c10d = torch.ops.c10d
_c10d_functional = torch.ops._c10d_functional
_c10d_functional_autograd = torch.ops._c10d_functional_autograd
_dtensor = torch.ops._dtensor

# List of collective operation functions including functional collectives
# Note: The following collectives might be deprecated soon hence not adding them
# depcreated_non_functional_collectives = [
#     c10d.allreduce_coalesced_.default,
#     c10d.reduce_scatter_tensor_coalesced_.default,
#     c10d.allgather_into_tensor_coalesced_.default,
#     c10d.allgather_coalesced_.default,
# ]
non_functional_collectives: set[torch._ops.OpOverload] = {
    c10d.broadcast_.default,
    c10d.allreduce_.default,
    c10d.reduce_.default,
    c10d.send.default,
    c10d.recv_.default,
    c10d.recv_any_source_.default,
    c10d.allgather_.default,
    c10d.reduce_scatter_.default,
    c10d._reduce_scatter_base_.default,
    c10d._allgather_base_.default,
    c10d.gather_.default,
    c10d.scatter_.default,
    c10d.alltoall_.default,
    c10d.alltoall_base_.default,
    c10d.barrier.default,
    c10d.monitored_barrier_.default,
}
functional_collectives: set[torch._ops.OpOverload] = {
    _c10d_functional.broadcast.default,
    _c10d_functional.all_reduce.default,
    _c10d_functional.all_gather_into_tensor.default,
    _c10d_functional.reduce_scatter_tensor.default,
    _c10d_functional.reduce_scatter_tensor_out.default,
    _c10d_functional.all_to_all_single.default,
    _c10d_functional_autograd.all_to_all_single.default,
    _c10d_functional.wait_tensor.default,
    _c10d_functional.all_reduce_.default,
    _c10d_functional.all_reduce_coalesced.default,
    _c10d_functional.all_reduce_coalesced_.default,
    _c10d_functional.all_gather_into_tensor_out.default,
    _c10d_functional.all_gather_into_tensor_coalesced.default,
    _c10d_functional_autograd.all_gather_into_tensor.default,
    _c10d_functional.reduce_scatter_tensor_coalesced.default,
    _c10d_functional_autograd.reduce_scatter_tensor.default,
    _c10d_functional.broadcast_.default,
    _c10d_functional.isend.default,
    _c10d_functional.irecv.default,
    _c10d_functional.batch_p2p_ops.default,
    _dtensor.shard_dim_alltoall.default,
}

sync_ops: set[torch._ops.OpOverload] = {
    c10d.barrier.default,
    c10d.monitored_barrier_.default,
    _c10d_functional.wait_tensor.default,
}

collective_ops = set.union(functional_collectives, non_functional_collectives)


class CollectiveOp:
    # [CLASS DOC] Namespace of static lookup sets and static methods describing collective
    # op overloads. The PG_ARG_n sets say which positional argument index (1 to 5 as
    # numbered by the set suffix mapping to argument positions 1, 2, 2, 3, 4) holds the
    # process group; WK and WK_ARG_1 say where the work handle is in the result;
    # COMM_TENSOR_* sets say how communicated tensor size is derived (from an argument,
    # the result, a single storage, or a sum). Methods: sum_tensors, get_process_group,
    # get_comm_tensor_size, get_work. No instance state.
    # Static sets for performance optimization
    PG_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.reduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.barrier.default,
        # c10d.allreduce_coalesced_.default
    }

    PG_ARG_2 = {
        c10d.allgather_.default,
        c10d._allgather_base_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d.gather_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
        c10d.alltoall_base_.default,
        # c10d.allgather_coalesced_.default,
        # c10d.allgather_into_tensor_coalesced_.default
        # c10d.reduce_scatter_tensor_coalesced_.default
    }

    PG_ARG_3 = {
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_out.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
        _c10d_functional.all_gather_into_tensor_coalesced.default,
    }

    PG_ARG_4 = {
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _c10d_functional.isend.default,
        _c10d_functional.irecv.default,
        _dtensor.shard_dim_alltoall.default,
    }

    PG_ARG_5 = {
        _c10d_functional.batch_p2p_ops.default,
    }

    WK_ARG_1 = {
        c10d.broadcast_.default,
        c10d.allreduce_.default,
        c10d.allgather_.default,
        c10d.reduce_scatter_.default,
        c10d._reduce_scatter_base_.default,
        c10d._allgather_base_.default,
        c10d.scatter_.default,
        c10d.alltoall_.default,
        _c10d_functional.isend.default,
        _c10d_functional.irecv.default,
    }

    WK = {
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.reduce_.default,
        c10d.gather_.default,
        c10d.alltoall_base_.default,
        c10d.barrier.default,
    }

    COMM_TENSOR_ARG_0 = {
        c10d.allreduce_.default,
        c10d.send.default,
        c10d.recv_.default,
        c10d.recv_any_source_.default,
        c10d.allgather_.default,
        c10d.gather_.default,
        c10d.reduce_.default,
        c10d.broadcast_.default,
        _c10d_functional.all_reduce_coalesced.default,
        _c10d_functional.all_reduce_coalesced_.default,
        # c10d.allreduce_coalesced_.default
        # c10d.allgather_coalesced_.default
        # c10d.allgather_into_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_1 = {
        c10d.reduce_scatter_.default,
        c10d.scatter_.default,
        # c10d.reduce_scatter_tensor_coalesced_.default,
    }

    COMM_TENSOR_ARG_RES = {
        _c10d_functional.all_gather_into_tensor.default,
        _c10d_functional_autograd.all_gather_into_tensor.default,
    }

    COMM_TENSOR_SINGLE_UNTYPED_STORAGE = {
        c10d._allgather_base_.default,
        _c10d_functional.broadcast.default,
        _c10d_functional.broadcast_.default,
        _c10d_functional.all_reduce.default,
        _c10d_functional.all_reduce_.default,
        _c10d_functional.reduce_scatter_tensor.default,
        _c10d_functional_autograd.reduce_scatter_tensor.default,
    }

    COMM_TENSOR_ARG_0_AND_RES = {
        _c10d_functional.all_to_all_single.default,
        _c10d_functional_autograd.all_to_all_single.default,
        _dtensor.shard_dim_alltoall.default,
    }

    COMM_TENSOR_RES_SUM = {
        _c10d_functional.all_gather_into_tensor_coalesced.default,
        _c10d_functional.reduce_scatter_tensor_coalesced.default,
    }

    @staticmethod
    def sum_tensors(arg: Any) -> int:
        # [FUNCTION DOC] Returns total bytes of the underlying untyped storages of every
        # tensor found in a (possibly nested) argument structure. Note shared storages seen in
        # several tensors are counted once per tensor, not deduplicated.
        """Calculate total memory consumed by the tensors in the argument."""
        total_memory = 0

        def sum_bytes(t: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Adds the storage byte size of one tensor to the enclosing
            # running total.
            nonlocal total_memory
            total_memory += t.untyped_storage().nbytes()

        tree_map_only(torch.Tensor, sum_bytes, arg)
        return total_memory

    @staticmethod
    def get_process_group(func, args) -> ProcessGroup:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Returns the ProcessGroup used by a collective op overload given its
        # positional args, by looking up which argument index holds it: unboxing a script
        # object for the c10d ops (index 1 or 2) or resolving a group name for functional ops
        # (index 2, 3 or 4). Not for wait_tensor. Raises TypeError if the op is not a known
        # collective.
        """Retrieve the process group for collective operations, except `wait_tensor`."""
        if func in CollectiveOp.PG_ARG_1:
            return ProcessGroup.unbox(args[1])
        if func in CollectiveOp.PG_ARG_2:
            return ProcessGroup.unbox(args[2])
        if func in CollectiveOp.PG_ARG_3:
            return _resolve_process_group(args[2])
        if func in CollectiveOp.PG_ARG_4:
            return _resolve_process_group(args[3])
        if func in CollectiveOp.PG_ARG_5:
            return _resolve_process_group(args[4])
        raise TypeError(f"Func {func} not found in {collective_ops}")

    @staticmethod
    def get_comm_tensor_size(func, res, args, kwargs) -> int:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Returns the number of bytes communicated by a collective call, given
        # op overload, result, args and kwargs (kwargs unused). Depends on the op: sum over a
        # list argument, the result storage, one argument storage, the larger of input and
        # output for alltoall variants, input plus output for DTensor all-to-all, and so on.
        # Not valid for wait_tensor or barriers. Raises TypeError for an unknown op. Some all-
        # to-all size rules are marked TODO as unconfirmed.
        """Compute the communication tensor size, except for `wait_tensor`, `barrier`, and `monitored_barrier`."""
        if func in CollectiveOp.COMM_TENSOR_ARG_0:
            return CollectiveOp.sum_tensors(args[0])
        if func in CollectiveOp.COMM_TENSOR_ARG_1:
            return CollectiveOp.sum_tensors(args[1])
        if func in CollectiveOp.COMM_TENSOR_ARG_RES:
            return res.untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_SINGLE_UNTYPED_STORAGE:
            return args[0].untyped_storage().nbytes()
        if func is c10d._reduce_scatter_base_.default:
            return args[1].untyped_storage().nbytes()
        if func is c10d.alltoall_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                CollectiveOp.sum_tensors(args[0]), CollectiveOp.sum_tensors(args[1])
            )
        if func is c10d.alltoall_base_.default:
            # TODO(@sanketpurandare) - Confirm size computation
            return max(
                args[0].untyped_storage().nbytes(), args[1].untyped_storage().nbytes()
            )
        if func == _c10d_functional.all_gather_into_tensor_out.default:
            return args[-1].untyped_storage().nbytes()
        if func in CollectiveOp.COMM_TENSOR_RES_SUM:
            return CollectiveOp.sum_tensors(res)
        if func in CollectiveOp.COMM_TENSOR_ARG_0_AND_RES:
            # TODO(@sanketpurandare) - Confirm size computation
            return args[0].untyped_storage().nbytes() + res.untyped_storage().nbytes()
        if func is _c10d_functional.batch_p2p_ops.default:
            return CollectiveOp.sum_tensors(args[3])
        raise TypeError(f"Unknown function: {func} in {collective_ops}")

    @staticmethod
    def get_work(func, res) -> Work:  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Extracts the Work handle (a FakeWork) from an op result: the result
        # itself for ops in the WK set, or its second element for ops in WK_ARG_1. Raises
        # TypeError for any other op.
        if func in CollectiveOp.WK:
            return FakeWork.unbox(res)
        elif func in CollectiveOp.WK_ARG_1:
            return FakeWork.unbox(res[1])
        raise TypeError(f"Func {func} not found in {collective_ops}")
