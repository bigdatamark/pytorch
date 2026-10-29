# [PACKAGE DOC]
# Experimental DTensor APIs. Exports context_parallel, local_map,
# register_sharding and the implicit_replication context manager, and rewrites
# their __module__ so they appear under this public namespace.
# Copyright (c) Meta Platforms, Inc. and affiliates
from collections.abc import Iterator
from contextlib import contextmanager

from torch.distributed.tensor._api import DTensor
from torch.distributed.tensor.experimental._attention import context_parallel
from torch.distributed.tensor.experimental._func_map import local_map
from torch.distributed.tensor.experimental._register_sharding import register_sharding


__all__ = ["context_parallel", "implicit_replication", "local_map", "register_sharding"]


@contextmanager
def implicit_replication() -> Iterator[None]:
    # [FUNCTION DOC]
    # Context manager that turns on implicit replication in the DTensor operator
    # dispatcher: plain tensors mixed with DTensors are treated as replicated
    # DTensors. Takes no arguments, yields nothing. Side effect: sets a global
    # flag on the dispatcher, and always resets it to False on exit (even on error),
    # so nesting does not restore a previously enabled state.
    """
    This context manager allows :class:`DTensor` to implicitly treat all non-DTensors (``torch.Tensor``)
    in the program be replicate :class:`DTensor` s during the operator computation.

    .. warning:: This might possible lead to incorrect results if ``torch.Tensor`` s are not replicated
        in practice, please use it at your discretion.
    """
    try:
        DTensor._op_dispatcher._allow_implicit_replication = True
        yield
    finally:
        DTensor._op_dispatcher._allow_implicit_replication = False


# Set namespace for exposed private names
context_parallel.__module__ = "torch.distributed.tensor.experimental"
implicit_replication.__module__ = "torch.distributed.tensor.experimental"
local_map.__module__ = "torch.distributed.tensor.experimental"
register_sharding.__module__ = "torch.distributed.tensor.experimental"
