# mypy: allow-untyped-defs
# [MODULE DOC] Traceable "functional" collectives (all_reduce, all_gather, reduce_scatter,
# all_to_all, broadcast, coalesced variants, deprecated *_tensor aliases) that return new
# tensors instead of mutating inputs. Under compilation they trace to native
# _c10d_functional ops; in eager mode results are wrapped in AsyncCollectiveTensor, which
# waits on the pending collective at first use (matched via a data-pointer to Work registry).
# Also holds group resolution helpers (ranks/ProcessGroup/DeviceMesh to group name), meta
# and autograd registrations, and legacy c10d_functional op registrations and py-kernel
# fallbacks. Import-time side effects: may warn if dynamo utilities are missing; registers
# library implementations and autograd formulas for the ops it defines.
import contextlib
import math
import sys
import warnings
from collections.abc import Callable, Sequence
from typing import Any, cast, TYPE_CHECKING

import torch
import torch.compiler.config
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from torch._utils import _maybe_view_chunk_cat
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from . import _functional_collectives_impl as fun_col_impl


try:
    from torch.utils._cxx_pytree import tree_map_only
except ImportError:
    from torch.utils._pytree import tree_map_only  # type: ignore[no-redef]


try:
    from torch.compiler import is_dynamo_compiling as is_torchdynamo_compiling
except Exception:
    warnings.warn(
        "Unable to import torchdynamo util `is_torchdynamo_compiling`, so won't support torchdynamo correctly",
        stacklevel=2,
    )

    def is_torchdynamo_compiling():  # type: ignore[misc]
        # [FUNCTION DOC] HELPER: Fallback used when the dynamo utility cannot be imported;
        # always reports not compiling.
        return False


"""
New traceable, functional collectives.
RFC: https://github.com/pytorch/pytorch/issues/93173

  compiler: trace these ops with plain-old-data schemas, then choose how to lower them.
  eager: execute these 'functional' ops which in eager return AsyncCollectiveTensor subclasses,
         automatically calling .wait() on underlying/hidden async 'work' obj only when fed to
         a downstream op.

Issues:
* Where should these ops live? Couldn't `import torch` if putting these ops in existing torch.distributed files
* Proper support for eager requires inplace ops. We should explore having it as an option for the API.
"""

"""
Functional collectives are asynchronous only and we perform implicit stream synchronization
on behalf of the user.

We use AsyncCollectiveTensor to wrap the result tensor of a collective and it lets us witness
first usage of the tensor and insert cross stream sync at the right place.

The above are the easy bits, the hard one is how we match the Work object returned by
c10d and the tensor AsyncCollectiveTensor wraps. We alloc the tensor inside the collective
op implementation (see ``clone()`` call in ``_all_reduce``) and then it's handled by the
dispatcher which might call other implementations that are allowed to change the returned
tensor - even return a tensor with a different shape (see ``torch.vmap``).

This means the caller of our ops receives a Tensor that is not guaranteed to be the same
allocated by our implementations and that makes pairing The AsyncTensor to the original
tensor a lot harder. This pairing is needed so we can lookup the Work object to use.

Originally, we tried WeakKeyDictionary to map from Tensor to Work, but because Tensor's
identity is not stable across dispatch, the op caller would end up with a different Tensor
instance that would not match any in the dictionary.

With Tensor identity out of the question, we decided use the tensor data pointer, which
should be stable across all the Tensor changes done during dispatch.

We have a dictionary of tensor::data_ptr -> Work that we insert right after we call into c10d.

We use this dictionary when AsyncCollectiveTensor is used to invoke Work::wait()

Finally, we setup a finalizer against the tensor wrapper to observe it getting collected so we
can clean up stale entries in the dictionary.

To eliminate the possibility of races we have a global version counter that is used by the finalizer.

As a wise man said once: Don't cross the streams (https://www.youtube.com/watch?v=wyKQe_i9yyo)

"""

"""
Functional collectives can accept any of these types to describe the ranks participating in collectives.

The different types will be desugared to a canonical format
"""
RANK_TYPES = (
    list[int]
    | list[list[int]]
    | dist.ProcessGroup
    | DeviceMesh
    | tuple["dist.tensor.DeviceMesh", int]
    | c10d.GroupName
)


from torch._utils import _chunk_or_narrow_cat


"""
User facing APIs for functional collectives
-------------------------------------------

These apis are called by user code and expected to work both in eager execution and compilation,
but there are significant differences to how the two modes are implemented underneath.

Eager execution is 'optimized' using a tensor subclass that schedules the synchronization (via wait_tensor() op)
just before the tensor is first used.  Compiled tracing currently relies on the compiler to perform this optimization,
and cannot yet correctly trace the AsyncTensor wrapper class.  In the future, these paths may be unified
if sufficient subclass support is added in dynamo.

Example: all_reduce is an entrypoint API, and other collectives follow a similar pattern.

Here's how it works under torch.compile/dynamo:
all_reduce(...)
  |--> _expand_group(...)               - desugars processgroup into canonical/traceable format
  |--> c10d_functional.all_reduce(...)  - dynamo captures this op call, doesn't trace deeper
  |--> _maybe_wrap_tensor(...)          - wait_tensor() op is immediately called, no AsyncTensor subclass needed

And under eager execution:
all_reduce(...)
  |--> _expand_group(...)               - same as above, but less critical for eager
  |--> c10d_functional.all_reduce(...)  - dispatches to real kernel OR records op in trace
  |--> _maybe_wrap_tensor(...)          - AsyncTensor wrapper applied to returned tensor,
                                          which issues wait_tensor() at the time of first use
"""


def wait_tensor(tensor):
    # [FUNCTION DOC] Calls the native wait_tensor op so the collective that produced the
    # tensor completes (blocks on CPU, syncs streams on CUDA); returns the tensor.
    """
    Wait on a tensor returned by the collectives ops.

    Waiting follows device semantics, which means blocking on CPU and synchronizing streams on CUDA.
    """
    return torch.ops._c10d_functional.wait_tensor(tensor)  # type: ignore[attr-defined]


def broadcast(self: torch.Tensor, src: int, group: RANK_TYPES, tag: str = ""):
    # [FUNCTION DOC] Resolves the group, calls native broadcast from rank src, and returns the
    # result wrapped for lazy waiting (AsyncCollectiveTensor in eager, waited under compile).
    """
    Broadcasts the tensor to all processes in the given process group.

    Args:
        src (int): Source rank
        group (ProcessGroup or List[int]): The process group to work on.
        tag (str, optional): A unique identifier for the collective. Default: empty string
    """
    group = _resolve_group(group, tag)
    tensor = torch.ops._c10d_functional.broadcast(
        self, src, _group_or_group_name(group)
    )
    return _maybe_wrap_tensor(tensor)


def all_reduce(self: torch.Tensor, reduceOp: str, group: RANK_TYPES, tag: str = ""):
    # [FUNCTION DOC] Reduces the tensor across the group with the lower-cased reduceOp name and
    # returns a new tensor (input untouched), wrapped for lazy waiting.
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result.

    The input tensor is left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    tensor = torch.ops._c10d_functional.all_reduce(
        self, reduceOp.lower(), _group_or_group_name(group)
    )
    return _maybe_wrap_tensor(tensor)


def all_gather_single(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] All-gathers the tensor along the first dimension via the native op; if
    # gather_dim is not 0, rearranges the result by chunking and concatenating along
    # gather_dim (a view when cheap, otherwise it first waits on the async result). Returns
    # the gathered tensor, possibly an AsyncCollectiveTensor. Input is not modified.
    # KNOWN ISSUE: the docstring claims only gather_dim = 0 is supported, but other values work.
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    tensor = torch.ops._c10d_functional.all_gather_into_tensor(
        self, group_size, _group_or_group_name(group)
    )
    res = _maybe_wrap_tensor(tensor)
    if gather_dim != 0:
        # Check if _maybe_view_chunk_cat can use the view optimization.
        # If not, it will use torch.cat which needs the data anyway, so
        # wait early to avoid AsyncCollectiveTensor dispatch overhead.
        if isinstance(res, AsyncCollectiveTensor):
            shape = list(res.shape)
            numel_between = math.prod(shape[1:gather_dim]) if gather_dim > 1 else 1
            can_use_view = shape[0] == group_size and numel_between == 1
            if not can_use_view:
                res = res.wait()
        res = _maybe_view_chunk_cat(res, group_size, gather_dim)
    return res


def all_gather_single_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Thin alias of all_gather_single, which already supports autograd.
    """
    Gather tensor data across from all machines and concatenate over ``gather_dim``.

    Note that it currently only supports gather_dim = 0.

    This function is the same as all_gather_single but will propagate the
    backwards gradient across workers.

    See all_gather_single for more details on usage.
    """
    # The base all_gather_single now has autograd support, so we can just call it
    return all_gather_single(self, gather_dim, group, tag)


def reduce_scatter_single(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Reduce-scatters the tensor with the lower-cased reduceOp. Raises
    # AssertionError unless the size along scatter_dim is divisible by the group size. For a
    # nonzero scatter_dim, first rearranges the data so the scatter dimension becomes leading.
    # Returns this rank's shard, wrapped for lazy waiting. Input is not modified.
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.


    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh
    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)

    if self.size(scatter_dim) % group_size != 0:
        raise AssertionError(
            f"input dimension 0 ({self.size(0)} must be a multiple of group_size {group_size})"
        )
    if scatter_dim != 0:
        self = _chunk_or_narrow_cat(self, group_size, narrow_dim=scatter_dim, cat_dim=0)

    tensor = torch.ops._c10d_functional.reduce_scatter_tensor(
        self,
        reduceOp.lower(),
        group_size,
        _group_or_group_name(group),
    )
    res = _maybe_wrap_tensor(tensor)
    return res


def reduce_scatter_single_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Thin alias of reduce_scatter_single, which already supports autograd.
    """
    Reduces the tensor data across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    This function is the same as reduce_scatter_single but will propagate the
    backwards gradient across workers.

    Currently only the "sum" reduceOp is supported.

    See reduce_scatter_single for more details on usage.
    """
    # The base reduce_scatter_single now has autograd support, so we can just call it
    return reduce_scatter_single(self, reduceOp, scatter_dim, group, tag)


# The deprecated *_tensor aliases warn only outside compile: the
# typing_extensions @deprecated wrapper lives in a Dynamo-skipped module, and a
# raw warnings.warn is an untraceable builtin, either of which breaks
# fullgraph=True tracing of these still-supported entry points. Guarding on
# is_compiling() keeps the eager-time deprecation while letting Dynamo trace
# straight through to the _single implementation.
def all_gather_tensor(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Deprecated alias of all_gather_single; emits a FutureWarning only outside
    # compilation (so tracing is unaffected), then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_tensor` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single(self, gather_dim, group, tag)


def all_gather_tensor_autograd(
    self: torch.Tensor,
    gather_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Deprecated alias of all_gather_single_autograd; warns (FutureWarning)
    # only outside compilation, then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_tensor_autograd` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single_autograd` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single_autograd(self, gather_dim, group, tag)


def reduce_scatter_tensor(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Deprecated alias of reduce_scatter_single; warns (FutureWarning) only
    # outside compilation, then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single(self, reduceOp, scatter_dim, group, tag)


def reduce_scatter_tensor_autograd(
    self: torch.Tensor,
    reduceOp: str,
    scatter_dim: int,
    group: RANK_TYPES,
    tag: str = "",
):
    # [FUNCTION DOC] Deprecated alias of reduce_scatter_single_autograd; warns (FutureWarning)
    # only outside compilation, then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor_autograd` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single_autograd` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single_autograd(self, reduceOp, scatter_dim, group, tag)


def all_reduce_coalesced(
    self: list[torch.Tensor], reduceOp: str, group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Reduces each tensor in the list across the group in one coalesced native
    # call; returns a new list of wrapped results. Inputs are not modified.
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result.

    The all tensors in the input list are left unmodified.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    tensor_list = torch.ops._c10d_functional.all_reduce_coalesced(  # type: ignore[attr-defined]
        self,
        reduceOp.lower(),
        _group_or_group_name(group),
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def all_gather_single_coalesced(
    self: list[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    # [FUNCTION DOC] All-gathers each tensor of the list along the first dimension in one
    # coalesced native call; returns a list of wrapped gathered tensors.
    """
    Gather a list of tensors across from all machines.

    Note that it currently only supports gather_dim = 0.

    The input tensor is left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    tensor_list = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(  # type: ignore[attr-defined]
        self,
        group_size,
        _group_or_group_name(group),
    )
    return list(map(_maybe_wrap_tensor, tensor_list))


def reduce_scatter_single_coalesced(
    inputs: list[torch.Tensor],
    reduceOp: str,
    scatter_dim: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Reduce-scatters each tensor along its own entry in scatter_dim in one
    # coalesced native call. Raises AssertionError if scatter_dim length differs from inputs or a
    # scattered dimension is not divisible by the group size. For nonzero dims the corresponding
    # entries of the inputs list are replaced in place by rearranged tensors (mutates the
    # caller's list). Returns a list of wrapped shards.
    """
    Reduces a list of tensors across all machines in such a way that all get
    the final result, then scatter the results to corresponding ranks.

    The input tensors are left unmodified.
    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)

    if len(scatter_dim) != len(inputs):
        raise AssertionError(
            f"Length of scatter_dim ({len(scatter_dim)}) must equal length of inputs ({len(inputs)})"
        )
    for idx, (dim, tensor) in enumerate(zip(scatter_dim, inputs)):
        if tensor.size(dim) % group_size != 0:
            raise AssertionError(
                f"input dimension {dim} ({tensor.size(dim)} must be a multiple of group_size {group_size} for tensor at index {idx}"
            )
        if dim != 0:
            tensor_list = torch.chunk(tensor, group_size, dim=dim)
            inputs[idx] = torch.cat(tensor_list)

    tensor_list = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(  # type: ignore[attr-defined]
        inputs,
        reduceOp.lower(),
        group_size,
        _group_or_group_name(group),
    )

    return list(map(_maybe_wrap_tensor, tensor_list))


# Guarded warning rather than the @deprecated wrapper so Dynamo can trace
# through to the _single_coalesced implementation; see the non-coalesced
# aliases above for the rationale.
def all_gather_into_tensor_coalesced(
    self: list[torch.Tensor], group: RANK_TYPES, tag: str = ""
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Deprecated alias of all_gather_single_coalesced; warns (FutureWarning)
    # only outside compilation, then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.all_gather_into_tensor_coalesced` is deprecated. "
            "Please use `torch.distributed._functional_collectives.all_gather_single_coalesced` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return all_gather_single_coalesced(self, group, tag)


def reduce_scatter_tensor_coalesced(
    inputs: list[torch.Tensor],
    reduceOp: str,
    scatter_dim: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> list[torch.Tensor]:
    # [FUNCTION DOC] Deprecated alias of reduce_scatter_single_coalesced; warns (FutureWarning)
    # only outside compilation, then delegates.
    if not torch.compiler.is_compiling():
        warnings.warn(
            "`torch.distributed._functional_collectives.reduce_scatter_tensor_coalesced` is deprecated. "
            "Please use `torch.distributed._functional_collectives.reduce_scatter_single_coalesced` instead.",
            FutureWarning,
            stacklevel=2,
        )
    return reduce_scatter_single_coalesced(inputs, reduceOp, scatter_dim, group, tag)


# This is a bit unsafe: it checks if the first argument in the schema reports as a non-mutable alias.
# Today, this maps 1:1 with "aten ops that are views".
def _is_view_op(tgt):
    # [FUNCTION DOC] HELPER: True if the op overload's first argument is a non-writing alias
    # (i.e. the op is a view) and it has no CompositeImplicitAutograd kernel; False for such
    # composite ops; returns None if the schema has no arguments. Raises AssertionError if tgt
    # is not an OpOverload.
    if not isinstance(tgt, torch._ops.OpOverload):
        raise AssertionError(f"Expected torch._ops.OpOverload, got {type(tgt)}")
    # Don't apply the view optimization to any `CompositeImplicitAutograd` ops.
    # See issue: https://github.com/pytorch/pytorch/issues/133421
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        tgt.name(), torch.DispatchKey.CompositeImplicitAutograd
    ):
        return False
    schema = tgt._schema
    if len(schema.arguments) > 0:
        first_arg = schema.arguments[0]
        # check if op is a view
        return first_arg.alias_info is not None and not first_arg.alias_info.is_write


def all_to_all_single(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] All-to-all over the first dimension. Split size lists must be ints or
    # symbolic ints (AssertionError otherwise) and both given or both None; if None, an even split
    # by group size is used. Returns the concatenated received data, wrapped for lazy waiting.
    """
    Each process splits input tensor and then scatters the split list
    to all processes in a group. Then concatenate the received tensors from all
    the processes in the group and return single output tensor.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one dimension of the DeviceMesh

    :: N.B. If you pass a PG or a 1D list to perform a MPMD collective, the compiler won't be able to recover
    that information and perform collective algebraic optimization. Use other forms of input for that.
    """
    if output_split_sizes is not None:
        if not all(
            isinstance(size, (int, torch.SymInt)) for size in output_split_sizes
        ):
            raise AssertionError(
                f"All output_split_sizes must be int or SymInt, got {output_split_sizes}"
            )
    if input_split_sizes is not None:
        if not all(isinstance(size, (int, torch.SymInt)) for size in input_split_sizes):
            raise AssertionError(
                f"All input_split_sizes must be int or SymInt, got {input_split_sizes}"
            )
    group = _resolve_group(group, tag)
    group_size = c10d._get_group_size_by_name(group)
    if output_split_sizes is None or input_split_sizes is None:
        if not (output_split_sizes is None and input_split_sizes is None):
            raise AssertionError(
                "output_split_sizes and input_split_sizes must either be "
                "specified together or both set to None"
            )
        output_split_sizes = [self.shape[0] // group_size] * group_size
        input_split_sizes = output_split_sizes
    tensor = torch.ops._c10d_functional.all_to_all_single(  # type: ignore[attr-defined]
        self,
        output_split_sizes,
        input_split_sizes,
        _group_or_group_name(group),
    )
    return _maybe_wrap_tensor(tensor)


def all_to_all_single_autograd(
    self: torch.Tensor,
    output_split_sizes: list[int] | None,
    input_split_sizes: list[int] | None,
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Thin alias of all_to_all_single, which already supports autograd.
    """
    Same as all_to_all_single but supports autograd.
    """
    # The base all_to_all_single now has autograd support, so we can just call it
    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)


# ============================================================================
# Collective Autograd Functions / Custom Ops
# ============================================================================


def wait_tensor_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] Autograd backward for wait_tensor: returns grad_output unchanged.
    """
    Backward for wait_tensor: identity (no-op).
    Wait is just a synchronization primitive, so gradient flows through unchanged.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def wait_tensor_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Autograd setup for wait_tensor; saves nothing and returns None.
    """
    Setup context for wait_tensor backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor,)
        output: Output from forward pass
    """
    return


torch.library.register_autograd(
    "_c10d_functional::wait_tensor",
    wait_tensor_backward,
    setup_context=wait_tensor_setup_context,
)


def all_reduce_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] Backward of native all_reduce: all-reduces the contiguous gradient with the
    # same op and waits on it. Raises RuntimeError unless the saved op is sum or avg. Returns
    # (grad_input, None, None).
    """
    Backward for all_reduce: all_reduce with same reduce_op.
    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_name, grad_reduce_op)
        grad_group_name and grad_reduce_op are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    # Only linear reductions have a well-defined all_reduce backward: for both
    # 'sum' and 'avg' the gradient is an all_reduce of grad_output with the same
    # op (grad wrt each rank's input is the same reduction of the per-rank
    # grad_outputs). Nonlinear ops (min/max/product/premul_sum) do not.
    if reduce_op not in ("sum", "avg"):
        raise RuntimeError(
            f"all_reduce backward only supports 'sum' and 'avg' reductions, got '{reduce_op}'"
        )

    # Backward does all_reduce with the same reduce_op
    output = torch.ops._c10d_functional.all_reduce(
        grad_output.contiguous(), reduce_op, group_name
    )
    return wait_tensor(output), None, None


def all_reduce_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves the group name and lower-cased reduce op on ctx for backward.
    """
    Setup context for all_reduce backward.
    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce",
    all_reduce_backward,
    setup_context=all_reduce_setup_context,
)


def all_gather_into_tensor_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] Backward of native all-gather: sum reduce-scatters the contiguous gradient
    # over the saved group and waits. Returns (grad_input, None, None).
    """
    Backward for all_gather_into_tensor: reduce_scatter with sum.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward is reduce_scatter with sum
    output = torch.ops._c10d_functional.reduce_scatter_tensor(
        grad_output.contiguous(),
        "sum",
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None


def all_gather_into_tensor_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name and group size on ctx for backward.
    """
    Setup context for all_gather_into_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, group_size, group_name)
        output: Output from forward pass
    """
    input, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)


def reduce_scatter_tensor_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] Backward of native reduce-scatter: all-gathers the contiguous gradient and
    # waits. Raises RuntimeError unless the saved op is sum. Returns (grad_input, None, None, None).
    """
    Backward for reduce_scatter_tensor: all_gather.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward is all_gather
    output = torch.ops._c10d_functional.all_gather_into_tensor(
        grad_output.contiguous(),
        group_size,
        group_name,
    )
    return wait_tensor(output), None, None, None


def reduce_scatter_tensor_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name, group size and lower-cased reduce op on ctx.
    """
    Setup context for reduce_scatter_tensor backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    input, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)


def all_to_all_single_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] Backward of native all-to-all: runs all-to-all on the contiguous gradient
    # with input and output split sizes swapped, waits, and returns (grad_input, None, None, None).
    """
    Backward for all_to_all_single: all_to_all with reversed split sizes.

    Forward does all-to-all with specified split sizes, backward reverses them.

    Args:
        ctx: Context object with group_name, output_split_sizes, and input_split_sizes
        grad_output: Gradient from downstream operations

    Returns:
        Tuple of (grad_input, grad_output_split_sizes, grad_input_split_sizes, grad_group_name)
        All except grad_input are None (not differentiable)
    """
    group_name = ctx.group_name
    output_split_sizes = ctx.output_split_sizes
    input_split_sizes = ctx.input_split_sizes

    # Backward is all_to_all with reversed split sizes
    output = torch.ops._c10d_functional.all_to_all_single(
        grad_output.contiguous(),
        input_split_sizes,  # Reversed
        output_split_sizes,  # Reversed
        group_name,
    )
    return wait_tensor(output), None, None, None


def all_to_all_single_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name and the output and input split sizes on ctx.
    """
    Setup context for all_to_all_single backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (input, output_split_sizes, input_split_sizes, group_name)
        output: Output from forward pass
    """
    input, output_split_sizes, input_split_sizes, group_name = inputs
    ctx.group_name = group_name
    ctx.output_split_sizes = output_split_sizes
    ctx.input_split_sizes = input_split_sizes


torch.library.register_autograd(
    "_c10d_functional::all_to_all_single",
    all_to_all_single_backward,
    setup_context=all_to_all_single_setup_context,
)

# Backward compatibility: register autograd for _c10d_functional_autograd ops
# These ops redirect to _c10d_functional but need their own autograd registration
torch.library.register_autograd(
    "_c10d_functional_autograd::all_gather_into_tensor",
    all_gather_into_tensor_backward,
    setup_context=all_gather_into_tensor_setup_context,
)

torch.library.register_autograd(
    "_c10d_functional_autograd::reduce_scatter_tensor",
    reduce_scatter_tensor_backward,
    setup_context=reduce_scatter_tensor_setup_context,
)

torch.library.register_autograd(
    "_c10d_functional_autograd::all_to_all_single",
    all_to_all_single_backward,
    setup_context=all_to_all_single_setup_context,
)


def all_reduce_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    # [FUNCTION DOC] Backward of coalesced all-reduce: all-reduces the list of contiguous
    # gradients and waits on each. Raises RuntimeError unless the saved op is sum. Returns
    # (list of grads, None, None).
    # KNOWN ISSUE: unlike the non-coalesced all_reduce backward, avg is rejected here.
    """
    Backward for all_reduce_coalesced: all_reduce each gradient.

    Forward aggregates tensors, backward aggregates gradients.

    Args:
        ctx: Context object with group_name and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_name)
        grad_reduce_op and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    reduce_op = ctx.reduce_op

    if reduce_op != "sum":
        raise RuntimeError(
            f"all_reduce_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_reduce on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_reduce_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        reduce_op,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_reduce_coalesced_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name and lower-cased reduce op on ctx for backward.
    """
    Setup context for all_reduce_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_name = inputs
    ctx.group_name = group_name
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::all_reduce_coalesced",
    all_reduce_coalesced_backward,
    setup_context=all_reduce_coalesced_setup_context,
)


def all_gather_into_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    # [FUNCTION DOC] Backward of coalesced all-gather: sum reduce-scatters the list of contiguous
    # gradients and waits on each. Returns (list of grads, None, None).
    """
    Backward for all_gather_into_tensor_coalesced: reduce_scatter each gradient.

    Forward gathers tensors from all ranks, backward scatters gradients back
    with sum reduction.

    Args:
        ctx: Context object with group_name and group_size
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_group_size, grad_group_name)
        grad_group_size and grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size

    # Backward does reduce_scatter on list of gradients
    grad_inputs = torch.ops._c10d_functional.reduce_scatter_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        "sum",
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None)


def all_gather_into_tensor_coalesced_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name and group size on ctx for backward.
    """
    Setup context for all_gather_into_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size


torch.library.register_autograd(
    "_c10d_functional::all_gather_into_tensor_coalesced",
    all_gather_into_tensor_coalesced_backward,
    setup_context=all_gather_into_tensor_coalesced_setup_context,
)


def reduce_scatter_tensor_coalesced_backward(ctx, grad_outputs: list[torch.Tensor]):
    # [FUNCTION DOC] Backward of coalesced reduce-scatter: all-gathers the list of contiguous
    # gradients and waits on each. Raises RuntimeError unless the saved op is sum. Returns
    # (list of grads, None, None, None).
    """
    Backward for reduce_scatter_tensor_coalesced: all_gather each gradient.

    Forward reduces and scatters tensors to ranks, backward gathers gradients
    from all ranks.

    Args:
        ctx: Context object with group_name, group_size, and reduce_op
        grad_outputs: Gradients from downstream operations (one per input tensor)

    Returns:
        Tuple of (grad_inputs..., grad_reduce_op, grad_group_size, grad_group_name)
        grad_reduce_op, grad_group_size, grad_group_name are None (not differentiable)
    """
    group_name = ctx.group_name
    group_size = ctx.group_size
    reduce_op = ctx.reduce_op

    # Lazy validation: check reduce_op only when backward is called
    if reduce_op != "sum":
        raise RuntimeError(
            f"reduce_scatter_tensor_coalesced backward only supports 'sum' reduction, got '{reduce_op}'"
        )

    # Backward does all_gather on list of gradients
    grad_inputs = torch.ops._c10d_functional.all_gather_into_tensor_coalesced(
        [grad_output.contiguous() for grad_output in grad_outputs],
        group_size,
        group_name,
    )
    return (list(map(wait_tensor, grad_inputs)), None, None, None)


def reduce_scatter_tensor_coalesced_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] Saves group name, group size and lower-cased reduce op on ctx.
    """
    Setup context for reduce_scatter_tensor_coalesced backward.

    Args:
        ctx: Context object to save state for backward
        inputs: Tuple of (tensor_list, reduce_op, group_size, group_name)
        output: Output from forward pass
    """
    tensor_list, reduce_op, group_size, group_name = inputs
    ctx.group_name = group_name
    ctx.group_size = group_size
    ctx.reduce_op = reduce_op.lower()


torch.library.register_autograd(
    "_c10d_functional::reduce_scatter_tensor_coalesced",
    reduce_scatter_tensor_coalesced_backward,
    setup_context=reduce_scatter_tensor_coalesced_setup_context,
)


def permute_tensor(
    self: torch.Tensor,
    src_dst: list[int],
    group: RANK_TYPES,
    tag: str = "",
) -> torch.Tensor:
    # [FUNCTION DOC] Sends this rank's whole tensor to the destination given by src_dst (entry m
    # is the destination of rank m) using an all-to-all whose split sizes are the element count for
    # matching source/destination pairs and zero elsewhere. Returns the all-to-all result.
    # Needs the group's ranks via the group expansion; may create a process group by ranks.
    """
    Permutes the elements of the tensor according to the given source/destination pairs. `src_dst` should
    be defined such that src_dst[m] == n means m sends to n.

    Group can be one of:
        List[int]: ranks participating in the collective.
        List[List[int]]: 2D mesh of ranks taking part of this collective in MPMD.
        ProcessGroup: Will perform a collective using the ranks and tag of the PG.
        DeviceMesh: Do a SPMD collective over all ranks of the mesh
        (DeviceMesh, int): Do a MPMD collective over one
    """
    t, rankset, group_size = _expand_group(group, tag)
    local_pg = c10d._find_or_create_pg_by_ranks_and_tag(t, rankset, group_size)

    output_split_sizes = [0] * group_size
    input_split_sizes = [0] * group_size
    for src, dst in enumerate(src_dst):
        if src == dist.get_rank(local_pg):
            input_split_sizes[dst] = self.numel()
        if dst == dist.get_rank(local_pg):
            output_split_sizes[src] = self.numel()

    return all_to_all_single(self, output_split_sizes, input_split_sizes, group, tag)


class AsyncCollectiveTensor(torch.Tensor):
    # [CLASS DOC] Tensor wrapper subclass around the output of an async collective. Attributes:
    # elem (the real tensor) and completed (whether wait was already issued). Any non-view op
    # applied to it first waits on elem and runs on the unwrapped data; view ops stay wrapped
    # without waiting. Key methods: trigger_wait (wait once, cached), wait (always wait).
    r"""
    A Tensor wrapper subclass that is used to trigger a call to wait
    prior to first use of the underlying tensor.
    Use it inside functional collective pytorch wrappers like the following:
    def functional_collective(self, group, tag):
        tag, rankset, group_size = _expand_group(group, tag)
        tensor = torch.ops.c10d_functional.{collective}(self, tag, rankset, group_size)
        return _maybe_wrap_tensor(tensor)
    """

    elem: torch.Tensor
    completed: bool

    __slots__ = ["elem", "completed"]

    @staticmethod
    def __new__(cls, elem: torch.Tensor):
        # [FUNCTION DOC] Builds a wrapper subclass instance with the same size, strides, offset,
        # dtype, layout, device and requires_grad as elem; stores elem and marks it not completed.
        r = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
        )
        r.elem = elem
        r.completed = False
        return r

    def __tensor_flatten__(self):
        # [FUNCTION DOC] Tensor-subclass protocol: reports the inner attribute names (just elem)
        # and no extra metadata.
        return ["elem"], None

    def tolist(self):
        # [FUNCTION DOC] Waits for the collective, then returns the nested list of values.
        return self.trigger_wait().tolist()

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        # [FUNCTION DOC] Rebuilds a wrapper from the mapping of inner tensors; raises
        # AssertionError if metadata is not None. Size and stride arguments are unused.
        if meta is not None:
            raise AssertionError(
                "meta must be None for AsyncCollectiveTensor unflatten"
            )
        elem = inner_tensors["elem"]
        return AsyncCollectiveTensor(elem)

    def __coerce_same_metadata_as_tangent__(
        self, expected_metadata: Any, expected_type: type | None = None
    ):
        # [FUNCTION DOC] Hook used when matching gradients to expected tangent type: returns None
        # unless the expected type is a plain Tensor, in which case it waits and returns the
        # unwrapped tensor.
        if expected_type is not torch.Tensor:
            return None

        return self.trigger_wait()

    def __repr__(self) -> str:  # type: ignore[override]
        # [FUNCTION DOC] Waits for the collective, then shows the wrapped value's repr.
        return f"AsyncCollectiveTensor({self.trigger_wait()})"

    def trigger_wait(self):
        # [FUNCTION DOC] On first call waits on elem, marks completed and returns the waited
        # tensor; later calls return elem directly without waiting again.
        if not self.completed:
            out = wait_tensor(self.elem)
            self.completed = True
            return out
        else:
            return self.elem

    def wait(self) -> torch.Tensor:
        # [FUNCTION DOC] Issues the wait op on elem every time (idempotent) and returns the
        # result; does not update the completed flag.
        return wait_tensor(self.elem)

    def _get_acs_underlying_tensor(self):
        # [FUNCTION DOC] HELPER: Returns the wrapped tensor; also acts as a marker that lets the
        # impl module recognize an AsyncCollectiveTensor.
        """This method enables  _functional_collectives_impl to test if a tensor is an ACS"""
        return self.elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):  # type: ignore[override]
        # [FUNCTION DOC] Dispatch handler. A view fast path re-wraps the viewed inner tensor
        # without waiting. Otherwise unwraps every wrapper argument (waiting first unless the op is
        # a view), runs the op, and re-wraps all tensor outputs only for view ops. Raises
        # AssertionError if an output to wrap is already a wrapper.
        if func is torch.ops.aten.view.default:
            # Fast handle aten.view as a lot of view related op goes to aten.view
            # eventually, this avoids pytree slowdown

            res = func(args[0].elem, args[1])
            wrapper_res = AsyncCollectiveTensor(res)
            return wrapper_res

        is_view_op = _is_view_op(func)

        def unwrap(e: AsyncCollectiveTensor):
            # [FUNCTION DOC] HELPER: Returns the inner tensor, waiting first unless the op is a view.
            # wait_tensor is idepotent and will do stream sync only once
            if not is_view_op:
                return e.trigger_wait()
            return e.elem

        def wrap(e: torch.Tensor):
            # [FUNCTION DOC] HELPER: Wraps a tensor in a new wrapper; raises AssertionError if it
            # is already a wrapper.
            # wait_tensor is idepotent and will do stream sync only once
            if isinstance(e, AsyncCollectiveTensor):
                raise AssertionError(
                    "Cannot wrap an AsyncCollectiveTensor inside another AsyncCollectiveTensor"
                )
            res = AsyncCollectiveTensor(e)
            return res

        unwrapped_args = tree_map_only(AsyncCollectiveTensor, unwrap, args)
        unwrapped_kwargs = tree_map_only(AsyncCollectiveTensor, unwrap, kwargs)

        # we don't wrap the result as it doesn't need to be waited on.
        out = func(*unwrapped_args, **unwrapped_kwargs)

        # View ops don't require a sync, so we should re-wrap the outputs.
        if is_view_op:
            out = tree_map_only(torch.Tensor, wrap, out)

        return out

    def numpy(self):  # type: ignore[override]
        # [FUNCTION DOC] Waits for the collective and returns the data as a NumPy array.
        return self.wait().numpy()


"""
Utils and infrastructure for tracing support
"""


def _expand_group(group: RANK_TYPES, tag: str = "") -> tuple[str, list[int], int]:
    # [FUNCTION DOC] HELPER: Desugars a group description (rank list, nested rank lists with equal
    # sizes, ProcessGroup, 1D DeviceMesh, or (DeviceMesh, dim) pair) into (tag, flat rank list,
    # group size). Tag defaults to the process group's tag when not given. Raises ValueError for
    # unequal nested sizes or invalid types/tuples, AssertionError for a mesh over one dimension.
    # Defines runtime no-op cast helpers so dynamo can trace it.
    """
    _expand_group desugars the different RANK_TYPES types into a canonical format that is traceable.

    By having this be part of the explicit eager codepath, we avoid having to specialize behavior inside
    torchdynamo and can still interoperate with processgroup objects or other untraceable forms.
    """
    # had to define this hack _inside_ expand_group to avoid
    # graph_break [('torch.* op returned non-Tensor int
    # caused by 'cast_*` functions being treated as 'torch.*' ops (iiuc)
    if TYPE_CHECKING:

        def cast_listlistint(x):
            return cast(list[list[int]], x)

        def cast_listint(x):
            return cast(list[int], x)

    else:
        # fake cast op for use at runtime since dynamo doesn't support real cast
        # also, dynamo didn't like encountering 'typing' objects ()
        # NotImplementedError: argument of type: <class 'typing._GenericAlias'>
        def cast_listlistint(x):
            return x

        def cast_listint(x):
            return x

    rankset: list[int]
    if isinstance(group, list):
        if isinstance(group[0], list):
            nested_list = cast_listlistint(group)
            rankset = []
            group_size = -1
            for rs in nested_list:
                rankset.extend(rs)
                if group_size != -1 and group_size != len(rs):
                    raise ValueError(
                        f"group sizes must be identical found {group_size} and {len(rs)}"
                    )
                group_size = len(rs)
        else:
            rankset = cast_listint(group)
            group_size = len(rankset)
    elif isinstance(group, dist.ProcessGroup):
        rankset = dist.get_process_group_ranks(group)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(group)
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        pg = group.get_group()
        rankset = dist.get_process_group_ranks(pg)
        group_size = len(rankset)
        tag = tag or c10d._get_group_tag(pg)
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            pg = dmesh.get_group(dim)
            rankset = dist.get_process_group_ranks(pg)
            group_size = len(rankset)
            tag = tag or c10d._get_group_tag(pg)
        else:
            raise ValueError("Invalid tuple for group must be (DeviceMesh, int)")
    else:
        raise ValueError(
            "Invalid type for group, must be one of List, Processgroup, DeviceMesh or (DeviceMesh, int)."
        )

    return (tag, rankset, group_size)


def _resolve_group(
    group: RANK_TYPES, tag: str = ""
) -> dist.ProcessGroup | c10d.GroupName:
    # [FUNCTION DOC] HELPER: Normalizes a group description to a ProcessGroup or a group name.
    # ProcessGroup and name strings pass through; a 1D DeviceMesh or (DeviceMesh, dim) pair gives
    # the mesh's group name (or, under compile_on_one_rank, a traced process-group op result); a
    # rank list is resolved by ranks and tag with a FutureWarning (suppressed when compiling).
    # Raises AssertionError for a multi-dimensional mesh, ValueError for bad tuples or types.
    """
    Given group in RANK_TYPES, return a ProcessGroup or group name.
    """
    # `tag` will be deprecated. See details in:
    # https://github.com/pytorch/pytorch/issues/93173#issuecomment-1907095208
    if isinstance(group, dist.ProcessGroup):
        return group
    elif isinstance(group, str):
        # In some cases Dynamo doesn't like tracing through NewType constructors
        # - so use a cast instead (the actual newtype representation is
        # literally the underlying type so this is fine). I haven't been able to
        # reproduce it in isolation (see T247631668).
        # pyrefly: ignore [redundant-cast]
        group_name = cast(c10d.GroupName, group)  # c10d.GroupName(group)
        return group_name
    elif isinstance(group, DeviceMesh):
        if group.ndim != 1:
            raise AssertionError(
                "Only 1D mesh is supported, pass in (DeviceMesh, int) together if mesh > 1D"
            )
        if torch.compiler.config.compile_on_one_rank:
            return torch.ops._dtensor.mesh_get_process_group(group, 0)
        return group._dim_group_names[0]
    elif isinstance(group, tuple):
        if (
            len(group) == 2
            and isinstance(group[0], DeviceMesh)
            and isinstance(group[1], int)
        ):
            dmesh = group[0]
            dim = group[1]
            if torch.compiler.config.compile_on_one_rank:
                return torch.ops._dtensor.mesh_get_process_group(dmesh, dim)
            return dmesh._dim_group_names[dim]
        else:
            raise ValueError(
                f"Invalid tuple for group must be (DeviceMesh, int). Instead got {(type(group[0]), type(group[1]))}"
            )
    elif isinstance(group, list):
        if not is_torchdynamo_compiling():
            warnings.warn(
                "The combination of ranks + tag as process group "
                "identifier has been deprecated. Please switch to "
                "using ProcessGroup, DeviceMesh, or group name instead.",
                FutureWarning,
                stacklevel=3,
            )
        return c10d._resolve_group_name_by_ranks_and_tag(
            # pyrefly: ignore [redundant-cast]
            cast(list[int], group),
            tag,
        )
    else:
        raise ValueError(f"Unsupported group type: {type(group)}, {group}")


def _resolve_group_name(group: RANK_TYPES, tag: str = "") -> c10d.GroupName:
    # [FUNCTION DOC] HELPER: Resolves the group and returns its name (wrapping a string or
    # reading the group_name of a ProcessGroup).
    """
    Given group in RANK_TYPES, return the group name.
    """
    group = _resolve_group(group, tag)
    if isinstance(group, str):
        return c10d.GroupName(group)
    else:
        return group.group_name


@torch.library.custom_op(
    "_c10d_functional::_wrap_tensor_autograd",
    mutates_args=(),
    schema="(Tensor input) -> Tensor",
)
def _wrap_tensor_autograd(input: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Custom op (no mutation) that wraps the input in an
    # AsyncCollectiveTensor while keeping autograd connected; returns the wrapper.
    """
    Custom op that allows autograd to propagate
    from a normal Tensor to an AsyncCollectiveTensor.

    This is the low-level implementation. Users should call _maybe_wrap_tensor directly.

    Args:
        input: Input tensor to wrap in AsyncCollectiveTensor

    Returns:
        AsyncCollectiveTensor wrapping the input (or wait_tensor result if tracing)
    """
    return AsyncCollectiveTensor(input)


@_wrap_tensor_autograd.register_fake
def _(input: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Fake/meta kernel: returns an uninitialized tensor like input.
    """
    Meta kernel for _wrap_tensor_autograd.
    """
    return torch.empty_like(input)


def _wrap_tensor_autograd_backward(ctx, grad_output: torch.Tensor):
    # [FUNCTION DOC] HELPER: Identity backward; returns the gradient unchanged.
    """
    Backward for _wrap_tensor_autograd: identity (no-op).

    The wrapping is just for async optimization, gradients flow through unchanged.

    Args:
        ctx: Context object (unused)
        grad_output: Gradient from downstream operations

    Returns:
        Gradient unchanged (identity)
    """
    return grad_output


def _wrap_tensor_autograd_setup_context(ctx, inputs, output):
    # [FUNCTION DOC] HELPER: Autograd setup that saves nothing; returns None.
    """
    Setup context for _wrap_tensor_autograd backward.

    Args:
        ctx: Context object to save state for backward (nothing to save)
        inputs: Tuple of (input,)
        output: Output from forward pass
    """
    return


_wrap_tensor_autograd.register_autograd(
    _wrap_tensor_autograd_backward,
    setup_context=_wrap_tensor_autograd_setup_context,
)


def _are_we_tracing() -> bool:
    # [FUNCTION DOC] HELPER: True if running under dynamo compilation, fake tensor mode, the
    # Python dispatcher, or a proxy (make_fx) mode; otherwise False.
    if is_torchdynamo_compiling():
        return True
    # If fake mode is turned on, we are almost definitely compiling/tracing.
    if torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is not None:
        return True
    # See Note [enable_python_dispatcher in dynamo]
    if torch._C._dispatch_tls_is_dispatch_key_included(
        torch._C.DispatchKey.PythonDispatcher
    ):
        return True
    return get_proxy_mode() is not None


def _maybe_wrap_tensor(self) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: When tracing, immediately waits on the tensor; in eager mode wraps
    # it in an AsyncCollectiveTensor via the autograd-aware custom op. Returns the result.
    if _are_we_tracing():
        return wait_tensor(self)
    return _wrap_tensor_autograd(self)


@contextlib.contextmanager
def allow_inflight_collective_as_graph_input_ctx(value: bool = True):
    # [FUNCTION DOC] Context manager that sets the global flag allowing in-flight eager
    # collectives to be graph inputs of compiled regions (registering their Work objects) to the
    # given value, and restores the previous flag on exit even on error.
    """
    Context manager to temporarily set whether inflight collectives are allowed as torch.compile graph inputs.
    Common use case is when the collective is issued in eager (with `async_op=True`) but waited in compiled region:
    ```
    def all_reduce_eager(x):
        y = x * x
        req = dist.all_reduce(y, op=dist.ReduceOp.SUM, async_op=True)
        return y


    @torch.compile(fullgraph=True)
    def all_reduce_wait_compiled(y):
        torch.ops.c10d_functional.wait_tensor(y)
        return y * y


    x = torch.ones(1280, 1280, device="cuda") + self.rank
    # the context manager ensures that `wait_tensor(y)` will wait on the correct work object
    with allow_inflight_collective_as_graph_input_ctx():
        y = all_reduce_eager(x)
        z = all_reduce_wait_compiled(y)
    ```
    With this context manager, when a collective is called, under the hood the work object of the collective
    will be registered in the work registry, and the wait_tensor() in compiled region called on
    the output tensor of the collective will wait on the correct work object.
    """
    previous = torch._C._distributed_c10d._allow_inflight_collective_as_graph_input()

    try:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(value)
        yield
    finally:
        torch._C._distributed_c10d._set_allow_inflight_collective_as_graph_input(
            previous
        )


def _make_all_gather_out_tensor(input, group_size):
    # [FUNCTION DOC] HELPER: Allocates an uninitialized output for an all-gather: the first
    # dimension multiplied by group_size, or a one-element dimension of group_size for scalars.
    out_size = list(input.size())
    if len(out_size) == 0:
        out_size.append(group_size)
    else:
        out_size[0] *= group_size
    out_tensor = input.new_empty(out_size)
    return out_tensor


def _all_gather_into_tensor_coalesced_meta(self, tag, rankset, group_size):
    # [FUNCTION DOC] HELPER: Meta kernel: one all-gather output allocation per input tensor.
    return [_make_all_gather_out_tensor(t, group_size) for t in self]


# We now register meta kernels to deal with tracing
def _broadcast_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for broadcast: empty tensor like the input.
    return torch.empty_like(self)


def _all_reduce_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for all_reduce: contiguous empty tensor like the input.
    return torch.empty_like(self, memory_format=torch.contiguous_format)


def _wait_tensor_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for wait_tensor: empty tensor like the input.
    return torch.empty_like(self)


def _isend_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for isend: zero-element tensor of the input's dtype and device.
    return torch.empty(0, dtype=self.dtype, device=self.device)


def _irecv_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for irecv: empty tensor like the input.
    return torch.empty_like(self)


def _batch_p2p_ops_meta(op_list, peer_list, tag_list, tensors, group_name):
    # [FUNCTION DOC] HELPER: Meta kernel for batched point-to-point ops: for each op, the tensor
    # itself for receives and a zero-element tensor for sends.
    return [
        t if op == "irecv" else torch.empty(0, dtype=t.dtype, device=t.device)
        for op, t in zip(op_list, tensors)
    ]


def _all_gather_into_tensor_meta(shard, tag, rankset, group_size):
    # [FUNCTION DOC] HELPER: Meta kernel (unused by registrations here): all-gather output shape.
    return _make_all_gather_out_tensor(shard, group_size)


def _reduce_scatter_tensor_meta(input, reduce_op, tag, rankset, group_size):
    # [FUNCTION DOC] HELPER: Meta kernel (unused by registrations here): output with the first
    # dimension divided by group_size.
    out_size = list(input.size())
    out_size[0] //= group_size
    return input.new_empty(out_size)


def _all_reduce_coalesced_meta(self, *args):
    # [FUNCTION DOC] HELPER: Meta kernel: an empty tensor like each input in the list.
    return [torch.empty_like(t) for t in self]


def _all_reduce__meta(inp, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for in-place all_reduce: returns the input itself.
    return inp


def _broadcast__meta(inp, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for in-place broadcast: returns the input itself.
    return inp


def _all_reduce_coalesced__meta(inputs, *args):
    # [FUNCTION DOC] HELPER: Meta kernel for in-place coalesced all_reduce: returns the inputs list.
    return inputs


def _reduce_scatter_tensor_coalesced_meta(inputs, reduceOp, tag, rankset, group_size):
    # [FUNCTION DOC] HELPER: Meta kernel (unused by registrations here): per-input outputs with
    # the first dimension divided by group_size.
    def mk_out_tensor(input):
        # [FUNCTION DOC] HELPER: Allocates the reduced-shard shape for one input.
        out_size = list(input.size())
        out_size[0] //= group_size
        out_tensor = input.new_empty(out_size)
        return out_tensor

    return [mk_out_tensor(t) for t in inputs]


# NB: We often say all_to_all has dynamic output size, but this is not
# technically true: instead, what typically happens is you manually
# communicate the output_split_sizes ahead of time (which is dynamic),
# but then you pass those sizes explicitly, and the all to all itself
# isn't dynamic, it just follows the specified output splits
def _all_to_all_single_meta(
    input, output_split_sizes, input_split_sizes, *args, **kwargs
):
    # [FUNCTION DOC] HELPER: Meta kernel for all-to-all: same-shape output if output splits are
    # None, else first dimension equal to the sum of output splits (each checked non-negative).
    if output_split_sizes is None:
        return input.new_empty(input.size())
    else:
        for s in output_split_sizes:
            torch._check(s >= 0)
        out_size = list(input.size())
        out_size[0] = sum(output_split_sizes)
        return input.new_empty(out_size)


def _all_gather_into_tensor_out_native_meta(input, group_size, group_name, *, out):
    # [FUNCTION DOC] HELPER: Meta kernel for the out-variant all-gather; returns a new tensor of
    # gathered shape (the out argument is not used).
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_native_meta(input, group_size, group_name):
    # [FUNCTION DOC] HELPER: Meta kernel for native all-gather: gathered-shape empty tensor.
    return _make_all_gather_out_tensor(input, group_size)


def _all_gather_into_tensor_coalesced_native_meta(inputs, group_size, group_name):
    # [FUNCTION DOC] HELPER: Meta kernel: gathered-shape empty tensor for each input.
    return [
        _all_gather_into_tensor_native_meta(input, group_size, group_name)
        for input in inputs
    ]


def _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name):
    # [FUNCTION DOC] HELPER: Meta kernel for native reduce-scatter: first dimension divided by group_size.
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_out_native_meta(
    inp, reduce_op, group_size, group_name, *, out
):
    # [FUNCTION DOC] HELPER: Meta kernel for the out-variant reduce-scatter; returns a new tensor
    # with first dimension divided by group_size (the out argument is not used).
    shape = list(inp.size())
    shape[0] //= group_size
    return inp.new_empty(shape)


def _reduce_scatter_tensor_coalesced_native_meta(
    inputs, reduce_op, group_size, group_name
):
    # [FUNCTION DOC] HELPER: Meta kernel: reduce-scatter output shape for each input.
    return [
        _reduce_scatter_tensor_native_meta(inp, reduce_op, group_size, group_name)
        for inp in inputs
    ]


# Library MUST be defined at module scope or it doesn't work
lib_impl = torch.library.Library("_c10d_functional", "IMPL")
lib_impl.impl("all_reduce", _all_reduce_meta, "Meta")
lib_impl.impl("all_reduce_", _all_reduce__meta, "Meta")
lib_impl.impl("all_reduce_coalesced", _all_reduce_coalesced_meta, "Meta")
lib_impl.impl("all_reduce_coalesced_", _all_reduce_coalesced__meta, "Meta")
lib_impl.impl("wait_tensor", _wait_tensor_meta, "Meta")
lib_impl.impl("isend", _isend_meta, "Meta")
lib_impl.impl("irecv", _irecv_meta, "Meta")
lib_impl.impl("batch_p2p_ops", _batch_p2p_ops_meta, "Meta")

lib_impl.impl(
    "all_gather_into_tensor_out", _all_gather_into_tensor_out_native_meta, "Meta"
)
lib_impl.impl("all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta")
lib_impl.impl(
    "all_gather_into_tensor_coalesced",
    _all_gather_into_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta")
lib_impl.impl(
    "reduce_scatter_tensor_out", _reduce_scatter_tensor_out_native_meta, "Meta"
)
lib_impl.impl(
    "reduce_scatter_tensor_coalesced",
    _reduce_scatter_tensor_coalesced_native_meta,
    "Meta",
)
lib_impl.impl("all_to_all_single", _all_to_all_single_meta, "Meta")
lib_impl.impl("broadcast", _broadcast_meta, "Meta")
lib_impl.impl("broadcast_", _broadcast__meta, "Meta")

# Backward compatibility: register Meta impls for _c10d_functional_autograd ops
# These ops redirect to _c10d_functional but need their own Meta registration so
# torch.compile / FakeTensor mode can dispatch them.
lib_impl_autograd = torch.library.Library("_c10d_functional_autograd", "IMPL")
lib_impl_autograd.impl(
    "all_gather_into_tensor", _all_gather_into_tensor_native_meta, "Meta"
)
lib_impl_autograd.impl(
    "reduce_scatter_tensor", _reduce_scatter_tensor_native_meta, "Meta"
)
lib_impl_autograd.impl("all_to_all_single", _all_to_all_single_meta, "Meta")

# Mark these ops as side effectful so that DCE does not remove communication
# whose result tensors are ignored by user code.
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.wait_tensor)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.isend.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.isend)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.irecv.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.irecv)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.batch_p2p_ops.default)  # type: ignore[has-type]
torch.fx.node.has_side_effect(torch.ops._c10d_functional.batch_p2p_ops)  # type: ignore[has-type]


# Register legacy ops for backward compatibility
# TODO(yifu): remove these in functional collective beta release
legacy_lib = torch.library.Library("c10d_functional", "DEF")
legacy_lib_impl = torch.library.Library("c10d_functional", "IMPL")
ops_defs = [
    "broadcast(Tensor self, int src, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce(Tensor self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "all_reduce_coalesced(Tensor[] self, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "wait_tensor(Tensor self) -> Tensor",
    "isend(Tensor self, int dst, int tag, str group_name) -> Tensor",
    "irecv(Tensor self, int src, int tag, str group_name) -> Tensor",
    "batch_p2p_ops(str[] op_list, int[] peer_list, int[] tag_list, Tensor[] tensors, str group_name) -> Tensor[]",
    "all_gather_into_tensor(Tensor shard, str tag, int[] ranks, int group_size) -> Tensor",
    "all_gather_into_tensor_coalesced(Tensor[] input, str tag, int[] ranks, int group_size) -> Tensor[]",
    "reduce_scatter_tensor(Tensor input, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor",
    "reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduceOp, str tag, int[] ranks, int group_size) -> Tensor[]",
    "all_to_all_single(Tensor input, SymInt[]? output_split_sizes, SymInt[]? input_split_sizes, str tag, int[] ranks, int group_size) -> Tensor",
]

my_module = sys.modules[__name__]
for op_def in ops_defs:
    op_name = op_def[0 : op_def.index("(")]
    backend_impl = getattr(fun_col_impl, f"_{op_name}")
    legacy_lib.define(op_def, tags=torch.Tag.pt2_compliant_tag)
    legacy_lib_impl.impl(op_name, backend_impl, "CompositeImplicitAutograd")


"""
Dynamo Remappings allow seamless translation from non-functional collectives of supportable form into
functional collective calls followed by inplace copy ops, allowing them to be traced into a functional graph.

We implement this by writing a decomposition and teaching dynamo how to associate it to a corresponding op via
the mapping dict below.

These schemas intentionally match torch.distributed.distributed_c10d.* ops that we are trying to remap from
"""


def all_gather_tensor_inplace(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group=None,  # TODO add a type,
    async_op: bool = False,
    tag: str = "",
    gather_dim: int = 0,
):
    # [FUNCTION DOC] Traceable remap of the in-place all-gather: raises AssertionError for
    # async_op or a missing group (default world group), otherwise copies the functional
    # all_gather_single result into output_tensor and returns it.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output_tensor.copy_(all_gather_single(input_tensor, gather_dim, group, tag))


def reduce_scatter_tensor_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    op: str = "sum",  # TODO type is actually c10d ReduceOp. is this ok?
    group=None,  # TODO add a type
    async_op: bool = False,
    scatter_dim: int = 0,
    tag: str = "",
):
    # [FUNCTION DOC] Traceable remap of the in-place reduce-scatter-tensor: raises AssertionError
    # for async_op or no group, otherwise copies the functional reduce_scatter_single result
    # (op given as a string) into output and returns it.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(reduce_scatter_single(input, op, scatter_dim, group, tag))


REDUCE_OP_TO_STR = {
    dist.ReduceOp.SUM: "sum",
    dist.ReduceOp.AVG: "avg",
    dist.ReduceOp.PRODUCT: "product",
    dist.ReduceOp.MIN: "min",
    dist.ReduceOp.MAX: "max",
    dist.ReduceOp.BAND: "band",
    dist.ReduceOp.BOR: "bor",
    dist.ReduceOp.BXOR: "bxor",
}


def all_reduce_inplace(
    tensor: torch.Tensor,
    op: str = "sum",
    group=None,
    async_op: bool = False,
    tag: str = "",
):
    # [FUNCTION DOC] Traceable remap of in-place all_reduce: raises AssertionError for async_op or
    # no group, otherwise copies the functional all_reduce result into tensor and returns it.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return tensor.copy_(all_reduce(tensor, op, group, tag))


def all_to_all_inplace(
    output: torch.Tensor,
    input: torch.Tensor,
    output_split_sizes=None,
    input_split_sizes=None,
    group=None,
    async_op=False,
    tag: str = "",
):
    # [FUNCTION DOC] Traceable remap of in-place all_to_all_single: raises AssertionError for
    # async_op or no group, otherwise copies the functional all-to-all result into output.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    return output.copy_(
        all_to_all_single(
            input,
            output_split_sizes,
            input_split_sizes,
            group,
            tag,
        )
    )


def all_gather_inplace(
    tensor_list: list[torch.Tensor],
    tensor: torch.Tensor,
    group=None,
    async_op=False,
    tag: str = "",
):
    # [FUNCTION DOC] Traceable remap of in-place list all_gather. Raises AssertionError for
    # async_op, for list entries with differing first-dimension sizes (non-scalar input), or no
    # group. All-gathers along dim 0, slices the result per list entry (scalars take one
    # element) and copies into each tensor in tensor_list; returns tensor_list.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )
    if tensor.dim() != 0 and not all(t.size(0) == tensor.size(0) for t in tensor_list):
        raise AssertionError("Remapping variable size all_gather is not yet supported")

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    output = all_gather_single(tensor, 0, group, tag)

    # Use aten.slice instead of aten.split because the latter causes
    # tensor.shape(0) to be unnecessarily baked in when it's a SymInt.
    output_splits = []
    offset = 0
    for t in tensor_list:
        is_scalar = t.dim() == 0
        t_offset = 1 if is_scalar else t.size(0)

        out = output[offset] if is_scalar else output[offset : offset + t_offset]
        output_splits.append(out)

        offset += t_offset
    for dst, src in zip(tensor_list, output_splits):
        dst.copy_(src)
    return tensor_list


def reduce_scatter_inplace(
    output: torch.Tensor,
    input_list: list[torch.Tensor],
    op: str = "sum",
    group: dist.ProcessGroup | None = None,
    async_op: bool = False,
    tag: str = "",
):
    # [FUNCTION DOC] Traceable remap of in-place list reduce_scatter. Raises AssertionError for
    # async_op, input entries not matching the output size, or no group. Concatenates (or, for
    # scalar output, stacks) the inputs, reduce-scatters along dim 0, copies into output and
    # returns it.
    if async_op:
        raise AssertionError(
            "Can't remap async version of inplace op to functional collective"
        )
    if not all(t.size() == output.size() for t in input_list):
        raise AssertionError(
            "reduce_scatter requires every input_list element to have the same size as output"
        )

    group = group or dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")

    if output.dim() == 0:
        # scalars have no dim to scatter along; stack into 1-D then reshape back
        result = reduce_scatter_single(torch.stack(input_list), op, 0, group, tag)
        return output.copy_(result.reshape(output.shape))
    return output.copy_(reduce_scatter_single(torch.cat(input_list), op, 0, group, tag))


def isend_inplace(
    tensor: torch.Tensor,
    dst: int | None = None,
    tag: int = 0,
    group: dist.ProcessGroup | None = None,
    group_dst: int = -1,
):
    # [FUNCTION DOC] Traceable remap of isend. Uses the default group if none; the destination is
    # group_dst or the group-local rank of global dst (ValueError if both or neither given). Calls
    # native isend and returns the tensor, wrapped for lazy waiting unless tracing.
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    if group_dst != -1:
        if dst is not None:
            raise ValueError(
                "Cannot specify both 'dst' and 'group_dst' args as per eager impl"
            )
        local_dst = group_dst
    elif dst is not None:
        local_dst = c10d.get_group_rank(group, dst)
    else:
        raise ValueError("Must specify either 'dst' or 'group_dst'")

    group_name = _resolve_group_name(group)
    tensor = torch.ops._c10d_functional.isend(tensor, local_dst, tag, group_name)
    if _are_we_tracing():
        return tensor
    return _maybe_wrap_tensor(tensor)


def irecv_inplace(
    tensor: torch.Tensor,
    src: int | None = None,
    tag: int = 0,
    group: dist.ProcessGroup | None = None,
    group_src: int = -1,
):
    # [FUNCTION DOC] Traceable remap of irecv. Uses the default group if none; the source is
    # group_src or the group-local rank of global src (ValueError if both or neither given). Calls
    # native irecv and returns the received tensor wrapped for lazy waiting.
    if group is None:
        group = dist.group.WORLD
    if group is None:
        raise AssertionError("group cannot be None")
    if group_src != -1:
        if src is not None:
            raise ValueError(
                "Cannot specify both 'src' and 'group_src' args as per eager impl"
            )
        local_src = group_src
    elif src is not None:
        local_src = c10d.get_group_rank(group, src)
    else:
        raise ValueError("Must specify either 'src' or 'group_src'")
    group_name = _resolve_group_name(group)
    tensor = torch.ops._c10d_functional.irecv(tensor, local_src, tag, group_name)
    return _maybe_wrap_tensor(tensor)


def batch_p2p_ops_inplace(
    op_list: list[str],
    peer_list: list[int],
    tag_list: list[int],
    tensors: list[torch.Tensor],
    group_name: RANK_TYPES,
):
    # [FUNCTION DOC] Traceable remap of batched point-to-point ops. Raises AssertionError if
    # distributed is uninitialized; an empty group uses the default group. Calls native
    # batch_p2p_ops with parallel lists (op names, peers, tags, tensors) and returns the
    # resulting tensors; receive outputs are wrapped (only receives when tracing).
    if not dist.is_initialized():
        raise AssertionError("torch.distributed must be initialized")
    if group_name is None or group_name == "":
        group_name = c10d._get_default_group()
    resolved = _resolve_group(group_name)
    group_name = resolved if isinstance(resolved, str) else resolved.group_name
    tensors = torch.ops._c10d_functional.batch_p2p_ops(
        op_list, peer_list, tag_list, tensors, group_name
    )
    if _are_we_tracing():
        return [
            _maybe_wrap_tensor(t) if op == "irecv" else t
            for op, t in zip(op_list, tensors)
        ]
    return list(map(_maybe_wrap_tensor, tensors))


def _group_or_group_name(
    group: dist.ProcessGroup | c10d.GroupName,
) -> dist.ProcessGroup | c10d.GroupName:
    # [FUNCTION DOC] HELPER: Returns a name string unchanged; under compile_on_one_rank returns the
    # process group itself; otherwise returns the group's name.
    if isinstance(group, str):
        return group
    elif torch.compiler.config.compile_on_one_rank:
        return group
    else:
        return group.group_name


from torch.distributed.distributed_c10d import (  # pyrefly: ignore  # deprecated; pyrefly: ignore [deprecated]
    _all_gather_base as legacy_all_gather_base,
    _reduce_scatter_base as legacy_reduce_scatter_base,
    all_gather as legacy_all_gather,
    all_gather_into_tensor as legacy_allgather,
    all_gather_single as legacy_allgather_single,
    all_reduce as legacy_allreduce,
    all_to_all_single as legacy_all_to_all_single,
    batch_isend_irecv as legacy_batch_p2p_ops,
    irecv as legacy_irecv,
    isend as legacy_isend,
    reduce_scatter as legacy_reduce_scatter,
    reduce_scatter_single as legacy_reducescatter_single,
    reduce_scatter_tensor as legacy_reducescatter,
)


# Dynamo remaps dist.* collectives to these wrappers via traceable_collective_remaps.
# Each wrapper calls the in-place functional collective and returns None,
# matching the return type of the original dist.* APIs when async_op=False.
# async_op=True already graph-breaks in CollectiveFunctionRewriteVariable.
# These must be module-level def statements (not closures from a decorator factory)
# because _traceable_collectives_source resolves Dynamo guard sources by looking up
# fn.__name__ as a module attribute — a def's __name__ matches its variable name
# automatically, whereas a closure's would not.
def _remapped_allgather(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Dynamo remap target for legacy all-gather variants; raises
    # AssertionError unless tracing, then runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError("_remapped_allgather should only be called during tracing")
    all_gather_tensor_inplace(*args, **kwargs)


def _remapped_reducescatter(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy reduce-scatter-tensor variants; AssertionError
    # unless tracing; runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_reducescatter should only be called during tracing"
        )
    reduce_scatter_tensor_inplace(*args, **kwargs)


def _remapped_allreduce(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy all_reduce; AssertionError unless tracing;
    # runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError("_remapped_allreduce should only be called during tracing")
    all_reduce_inplace(*args, **kwargs)


def _remapped_all_to_all_single(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy all_to_all_single; AssertionError unless
    # tracing; runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_all_to_all_single should only be called during tracing"
        )
    all_to_all_inplace(*args, **kwargs)


def _remapped_all_gather(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy list all_gather; AssertionError unless
    # tracing; runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_all_gather should only be called during tracing"
        )
    all_gather_inplace(*args, **kwargs)


def _remapped_reduce_scatter(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy list reduce_scatter; AssertionError unless
    # tracing; runs the in-place functional version; returns None.
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_reduce_scatter should only be called during tracing"
        )
    reduce_scatter_inplace(*args, **kwargs)


def _remapped_isend(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy isend; AssertionError unless tracing;
    # returns the result of the functional isend.
    if not _are_we_tracing():
        raise AssertionError("_remapped_isend should only be called during tracing")
    return isend_inplace(*args, **kwargs)


def _remapped_irecv(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy irecv; AssertionError unless tracing;
    # returns the result of the functional irecv.
    if not _are_we_tracing():
        raise AssertionError("_remapped_irecv should only be called during tracing")
    return irecv_inplace(*args, **kwargs)


def _remapped_batch_p2p_ops(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Remap target for legacy batch_isend_irecv; AssertionError unless
    # tracing; returns the result of the functional batched point-to-point call.
    if not _are_we_tracing():
        raise AssertionError(
            "_remapped_batch_p2p_ops should only be called during tracing"
        )
    return batch_p2p_ops_inplace(*args, **kwargs)


# This dict should contain sets of functions that dynamo is allowed to remap.
# Functions in this set should accept the same args/kwargs 1:1 as their mapping.
traceable_collective_remaps = {
    legacy_allgather: _remapped_allgather,
    legacy_allgather_single: _remapped_allgather,
    legacy_reducescatter: _remapped_reducescatter,
    legacy_reducescatter_single: _remapped_reducescatter,
    legacy_allreduce: _remapped_allreduce,
    legacy_all_to_all_single: _remapped_all_to_all_single,
    legacy_all_gather: _remapped_all_gather,
    legacy_reduce_scatter: _remapped_reduce_scatter,
    legacy_reduce_scatter_base: _remapped_reducescatter,
    legacy_all_gather_base: _remapped_allgather,
    legacy_isend: _remapped_isend,
    legacy_irecv: _remapped_irecv,
    legacy_batch_p2p_ops: _remapped_batch_p2p_ops,
}


def _remap_traceable_collective(
    func: Callable[..., Any],
    args: Sequence[Any],
    kwargs: dict[str, Any] | None,
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None:
    # [FUNCTION DOC] HELPER: If func is a key of the remap table, binds args/kwargs to func's
    # signature, converts a ReduceOp value to its lowercase string for reduction collectives, and
    # returns (replacement function, empty positional tuple, bound keyword mapping); else None.
    """Redirect a legacy ``torch.distributed`` collective to its functional
    equivalent, following Dynamo's CollectiveFunctionRewriteVariable.

    Returns ``(mapped_func, (), kwargs)`` if ``func`` is a remappable collective,
    else None. Shared by the make_fx compile_on_one_rank mode below and
    non-strict export's _NonStrictTorchFunctionHandler.
    """
    if func not in traceable_collective_remaps:
        return None
    import inspect

    mapped_func = traceable_collective_remaps[func]
    bound = dict(inspect.signature(func).bind(*args, **(kwargs or {})).arguments)
    if func in (
        torch.distributed.all_reduce,
        torch.distributed.reduce_scatter,
        torch.distributed.reduce_scatter_single,
        torch.distributed.reduce_scatter_tensor,
        torch.distributed._reduce_scatter_base,
    ):
        if "op" in bound:
            bound["op"] = REDUCE_OP_TO_STR[bound["op"]]
    return mapped_func, (), bound


class _LegacyToFunctionalCollectiveMode(torch.overrides.TorchFunctionMode):
    # [CLASS DOC] HELPER: Torch-function mode that redirects legacy distributed collectives to
    # their functional equivalents during tracing; other calls pass through unchanged.
    """Redirect legacy ``torch.distributed`` collectives to functional collectives
    while tracing, mirroring Dynamo and non-strict export.

    Used by ``make_fx`` under ``compile_on_one_rank``: the in-place ``c10d.*`` ops
    bind the ProcessGroup as a torchbind constant that cannot be serialized,
    whereas the functional collectives take the group as an op argument, so the
    group (e.g. a ``DeviceMesh.get_group()`` -> ``mesh_get_process_group`` output)
    flows into the graph instead of being baked in.
    """

    def __torch_function__(
        self,
        func: Callable[..., Any],
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # [FUNCTION DOC] HELPER: Calls the remapped functional collective if func is remappable,
        # otherwise calls func with the original args and kwargs; returns its result.
        remapped = _remap_traceable_collective(func, args, kwargs)
        if remapped is not None:
            mapped_func, mapped_args, mapped_kwargs = remapped
            return mapped_func(*mapped_args, **mapped_kwargs)
        return func(*args, **(kwargs or {}))
