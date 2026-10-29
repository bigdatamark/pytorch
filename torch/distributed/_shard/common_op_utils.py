# mypy: allow-untyped-defs
# [MODULE DOC] Shared helpers for ShardedTensor op handlers: input validation (_basic_validation) and registration of the default pass-through handler (_register_default_op).

import torch
from torch.utils import _pytree as pytree


def _basic_validation(op, args=(), kwargs=None):
    # [FUNCTION DOC] HELPER: Validate the call of op: raises ValueError if there are no arguments at all, TypeError if no ShardedTensor appears anywhere in args or kwargs
    # (searched recursively), and RuntimeError if the ShardedTensors do not all share the same process group.
    """
    Common validation across all ops go in here.
    """
    from torch.distributed._shard.sharded_tensor import ShardedTensor

    if len(args) == 0 and (kwargs is None or len(kwargs) == 0):
        raise ValueError(f" No input for '{op.__name__}'!")

    # Validate types
    has_distributed_tensor = False

    def is_distributed_tensor(e):
    # [FUNCTION DOC] HELPER: Tree-map callback that sets the enclosing flag when e is a ShardedTensor.
        nonlocal has_distributed_tensor
        if isinstance(e, ShardedTensor):
            has_distributed_tensor = True

    pytree.tree_map_(is_distributed_tensor, args)
    pytree.tree_map_(is_distributed_tensor, kwargs)

    if not has_distributed_tensor:
        raise TypeError(
            f"torch function '{op.__name__}', with args: {args} and "
            f"kwargs: {kwargs} are called without any distributed tensor!"
        )

    # Validate all distributed tensors use the same PG.
    cur_pg: torch.distributed.ProcessGroup | None = None

    def validate_pg(e):
    # [FUNCTION DOC] HELPER: Tree-map callback that tracks the first seen ShardedTensor's process group and raises RuntimeError when a later one differs.
        nonlocal cur_pg
        if isinstance(e, ShardedTensor):
            if cur_pg is not None and e._process_group is not cur_pg:
                raise RuntimeError(
                    "All distributed tensors should use the "
                    "same ProcessGroup if used together in an op."
                )
            cur_pg = e._process_group

    pytree.tree_map_(validate_pg, args)
    pytree.tree_map_(validate_pg, kwargs)


def _register_default_op(op, decorator):
    # [FUNCTION DOC] HELPER: Use the given registration decorator to register a handler for op that simply calls the real op with torch-function subclass dispatch disabled (to avoid recursion).
    @decorator(op)
    def tensor_default_op(types, args=(), kwargs=None, pg=None):
        # [FUNCTION DOC] HELPER: Handler body: default kwargs to empty, then run the op on args under the disabled subclass dispatch context; the types and pg parameters are unused.
        """
        Handles ``__torch_function__`` dispatch for the default tensor ops that
        behave the same as ``torch.Tensor`` such as ``torch.Tensor.shape`` or
        ``torch.Tensor.dtype``. We simply lower to the real op call with
        DisableTorchFunctionSubclass context like ``torch.Tensor.__torch_function__``
        to avoid recursions.
        """
        if kwargs is None:
            kwargs = {}

        with torch._C.DisableTorchFunctionSubclass():
            return op(*args, **kwargs)
