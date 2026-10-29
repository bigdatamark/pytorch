# mypy: allow-untyped-defs
# [MODULE DOC] Differentiable (autograd-aware) wrappers around torch.distributed
# collectives: broadcast, gather, scatter, reduce, reduce_scatter,
# all_gather, all_to_all, all_to_all_single and all_reduce, plus a
# private all-gather-into-tensor variant. Each public function is backed
# by a private autograd Function whose backward applies the matching
# collective on gradients. Most are deprecated in favor of functional
# collectives and raise under torch.compile. Imports group and ReduceOp
# from torch.distributed at import time.
import warnings

import torch
import torch.distributed as dist
from torch.autograd import Function

# The two imports below are not always available depending on the
# USE_DISTRIBUTED compile flag. Make sure they raise import error
# if we're trying to use them.
from torch.distributed import group, ReduceOp


def _not_supported_under_compile(name, *, suggestion=None):
    # [FUNCTION DOC] HELPER: Always raises RuntimeError saying the named function is
    # not supported under torch.compile, with an optional suggestion of
    # an alternative to use.
    msg = (
        f"torch.distributed.nn.functional.{name} is not supported under torch.compile."
    )
    if suggestion:
        msg += f" Use {suggestion} instead."
    raise RuntimeError(msg)


def _deprecated(name, suggestion):
    # [FUNCTION DOC] HELPER: Emits a FutureWarning that the named function is
    # deprecated and the suggested alternative should be used.
    warnings.warn(
        f"torch.distributed.nn.functional.{name} is deprecated, "
        f"use {suggestion} instead.",
        category=FutureWarning,
        stacklevel=3,
    )


def broadcast(tensor, src, group=group.WORLD):
    # [FUNCTION DOC] Broadcasts the tensor from rank src to all ranks of the group and
    # returns the received tensor (a clone, not in place), with
    # autograd support. Raises RuntimeError under torch.compile; emits
    # a deprecation warning.
    """
    Broadcasts the tensor to the whole group.

    ``tensor`` must have the same number of elements in all processes
    participating in the collective.

    Arguments:
        tensor (Tensor): Data to be sent if ``src`` is the rank of current
            process.
        src (int): Source rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Received tensor from the broadcast op.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "broadcast",
            suggestion="torch.distributed._functional_collectives.broadcast",
        )
    _deprecated("broadcast", "torch.distributed._functional_collectives.broadcast")
    return _Broadcast.apply(src, group, tensor)


def gather(tensor, dst=0, group=group.WORLD):
    # [FUNCTION DOC] Gathers the tensor from every rank onto rank dst and returns a
    # tuple of tensors (one per rank), with autograd support. Raises
    # RuntimeError under torch.compile.
    """
    Gathers a list of tensors in a single process.

    Arguments:
        tensor (Tensor): Input tensor.
        dst (int, optional): Destination rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple[Tensor]: List of appropriately-sized tensors with the gathered data.
    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("gather")
    return _Gather.apply(dst, group, tensor)


def scatter(tensors, src=0, group=group.WORLD):
    # [FUNCTION DOC] Scatters the given list of tensors from rank src so each rank
    # receives one; returns this rank's tensor, with autograd support.
    # Non-source ranks still pass a tensor list of the same length in
    # order to size the output. Raises RuntimeError under
    # torch.compile.
    """
    Scatters a list of tensors to all processes in a group.

    Each process will receive exactly one tensor and store its data in the
    ``tensor`` argument.

    Arguments:
        tensors (list[Tensor]): List of tensors to scatter on the source rank.
            Receivers must pass ``None`.
        src (int, optional): Source rank (default is 0).
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output tensor from the scatter operation.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("scatter")
    return _Scatter.apply(src, group, *tensors)


def reduce(tensor, dst, op=ReduceOp.SUM, group=group.WORLD):
    # [FUNCTION DOC] Reduces the tensor with the given reduction op onto rank dst and
    # returns the result (a clone), with autograd support; only dst
    # holds the final value. Raises RuntimeError under torch.compile.
    """
    Reduces the tensor data across all machines.

    Only the process with rank ``dst`` is going to receive the final result.

    Arguments:
        tensor (Tensor): Input of the collective.
        dst (int): Destination rank.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("reduce")
    return _Reduce.apply(dst, op, group, tensor)


def reduce_scatter(output, input_list, op=ReduceOp.SUM, group=group.WORLD):
    # [FUNCTION DOC] Reduces the list of input tensors across ranks and scatters the
    # results so each rank gets one piece, written into the output
    # tensor which is also returned; with autograd support. Raises
    # RuntimeError under torch.compile; emits a deprecation warning.
    """
    Reduces, then scatters a list of tensors to all processes in a group.

    Arguments:
        output (Tensor): Output tensor.
        input_list (list[Tensor]): List of tensors to reduce and scatter.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "reduce_scatter",
            suggestion="torch.distributed._functional_collectives.reduce_scatter_single",
        )
    _deprecated(
        "reduce_scatter",
        "torch.distributed._functional_collectives.reduce_scatter_single",
    )
    return _Reduce_Scatter.apply(op, group, output, *input_list)


def all_gather(tensor, group=group.WORLD):
    # [FUNCTION DOC] Gathers the tensor from all ranks and returns a tuple of tensors
    # (one per rank), with autograd support. Raises RuntimeError under
    # torch.compile; emits a deprecation warning.
    """
    Gathers tensors from the whole group in a list.

    Arguments:
        tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_gather",
            suggestion="torch.distributed._functional_collectives.all_gather_single",
        )
    _deprecated(
        "all_gather", "torch.distributed._functional_collectives.all_gather_single"
    )
    return _AllGather.apply(group, tensor)


def _all_gather_base(output_tensor, input_tensor, group=group.WORLD):
    # [FUNCTION DOC] HELPER: Experimental single-tensor all-gather: gathers inputs
    # from all ranks into the pre-sized output tensor and returns it,
    # with autograd support (backward only works on NCCL-like
    # backends). Raises RuntimeError under torch.compile.
    """
    Single tensor all gather. Gathers a single tensor from all ranks, and puts them in a single output tensor.

    Args:
        output_tensor (Tensor): Output tensor. It should contain
            correctly-sized tensors to be used for output of the collective.
        input_tensor (Tensor): Tensor to be broadcast from current process.
        group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.

    Examples:
        >>> # All tensors below are of torch.int64 dtype.
        >>> # We have 2 process groups, 2 ranks.
        >>> # xdoctest: +SKIP("incorrect want text")
        >>> output_tensor = torch.zeros(2, dtype=torch.int64)
        >>> output_tensor
        [tensor([0, 0])] # Rank 0 and 1
        >>> tensor = torch.arange(1, dtype=torch.int64) + 1 + rank
        >>> tensor
        tensor([1]) # Rank 0
        tensor([2]) # Rank 1
        >>> dist.all_gather_base(output_tensor, tensor)
        >>> output_tensor
        tensor([1,2]) # Rank 0
        tensor([1,2]) # Rank 1

    .. warning::
        `_all_gather_base` is experimental and subject to change.
        It is the caller's responsibility to ensure the output_tensor
        is correctly sized.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("_all_gather_base")
    return _AllGatherBase.apply(output_tensor, input_tensor, group)


def all_to_all(output_tensor_list, input_tensor_list, group=group.WORLD):
    # [FUNCTION DOC] Every rank splits its input list of tensors across ranks and
    # receives one piece from each rank into the output list; returns a
    # tuple of the output tensors, with autograd support. Raises
    # RuntimeError under torch.compile.
    """
    Each process scatters list of input tensors to all processes in a group and return gathered list of tensors in output list.

    Arguments:
        output_tensor_list (list[Tensor]): list of tensors to gather one per rank.
        input_tensor_list (list[Tensor]): List of tensors to scatter one per rank.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        tuple([Tensor]): Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile("all_to_all")
    return _AlltoAll.apply(group, output_tensor_list, *input_tensor_list)


def all_to_all_single(
    output,
    input,
    output_split_sizes=None,
    input_split_sizes=None,
    group=group.WORLD,
):
    # [FUNCTION DOC] Splits the input tensor along dim 0 (optionally with given split
    # sizes), exchanges pieces among ranks and writes the concatenation
    # into output, which is returned; with autograd support. Raises
    # RuntimeError under torch.compile; emits a deprecation warning.
    """
    Each process splits input tensor and then scatters the split list to all processes in a group.

    Then concatenate the received tensors from all the processes in the group and return single output tensor.

    Arguments:
        output (Tensor): Gathered concatenated output tensor.
        input (Tensor): Input tensor to scatter.
        output_split_sizes: (list[Int], optional): Output split sizes for dim 0
            if specified None or empty, dim 0 of ``output`` tensor must divide
            equally by ``world_size``.
        input_split_sizes: (list[Int], optional): Input split sizes for dim 0
            if specified None or empty, dim 0 of ``input`` tensor must divide
            equally by ``world_size``.

    Returns:
        Tensor: Output of the collective.

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_to_all_single",
            suggestion="torch.distributed._functional_collectives.all_to_all_single",
        )
    _deprecated(
        "all_to_all_single",
        "torch.distributed._functional_collectives.all_to_all_single",
    )
    return _AlltoAllSingle.apply(
        group, output, output_split_sizes, input_split_sizes, input
    )


def all_reduce(tensor, op=ReduceOp.SUM, group=group.WORLD):
    # [FUNCTION DOC] Reduces the tensor across all ranks with the given op so every
    # rank gets the same result (a clone), with autograd support.
    # Raises RuntimeError under torch.compile; emits a deprecation
    # warning.
    """
    Reduces the tensor data across all machines in such a way that all get the final result.

    After the call the returned tensor is going to be bitwise
    identical in all processes.

    Arguments:
        tensor (Tensor): Input of the collective.
        op (optional): One of the values from
            ``torch.distributed.ReduceOp``
            enum.  Specifies an operation used for element-wise reductions.
        group (ProcessGroup, optional): The process group to work on.

    Returns:
        Tensor: Output of the collective

    """
    if torch.compiler.is_compiling():
        _not_supported_under_compile(
            "all_reduce",
            suggestion="torch.distributed._functional_collectives.all_reduce",
        )
    _deprecated("all_reduce", "torch.distributed._functional_collectives.all_reduce")
    return _AllReduce.apply(op, group, tensor)


class _Broadcast(Function):
    # [CLASS DOC] HELPER: Autograd Function for broadcast. Forward clones the tensor
    # and broadcasts it from src; backward sums the gradients onto src
    # with a reduce and zeroes the result on non-source ranks.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, group, tensor):
        # [FUNCTION DOC] HELPER: Saves src, group and the global rank on ctx, clones
        # the tensor, broadcasts it in the group and returns it.
        ctx.src = src
        ctx.group = group
        ctx.global_rank = dist.get_rank()
        # torch.distributed makes all the calls in place
        # we allocate new tensors to avoid this
        tensor = tensor.clone()
        dist.broadcast(tensor, src, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Reduces the gradient (sum) to the source rank; non-
        # source ranks get zeros. Returns no gradient for src and
        # group.
        gx = _Reduce.apply(ctx.src, ReduceOp.SUM, ctx.group, grad_output)
        if ctx.src != ctx.global_rank:
            gx.zero_()
        return (None, None, gx)


class _Gather(Function):
    # [CLASS DOC] HELPER: Autograd Function for gather. Forward gathers tensors onto
    # dst; backward scatters the gradients from dst.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, dst, group, tensor):
        # [FUNCTION DOC] HELPER: Saves dst and group; allocates one zero tensor per
        # rank, gathers the contiguous input (on dst receives into the
        # list, elsewhere sends only) and returns the tuple.
        ctx.dst = dst
        ctx.group = group
        # Need to create a list of tensors here to do the
        # aggregation, get it from the group size
        # tensor should be correctly sized for the method
        # gathering
        tensor_list = [
            torch.zeros_like(tensor) for i in range(dist.get_world_size(group=group))
        ]

        tensor = tensor.contiguous()
        if dist.get_rank(group=group) == dst:
            dist.gather(tensor, tensor_list, dst, group=group)
        else:
            dist.gather(tensor, None, dst, group=group)
        return tuple(tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # [FUNCTION DOC] HELPER: Scatters the incoming gradients from dst back to all
        # ranks; no gradient for dst and group.
        return (None, None) + (_Scatter.apply(ctx.dst, ctx.group, *grad_outputs),)


class _Scatter(Function):
    # [CLASS DOC] HELPER: Autograd Function for scatter. Forward scatters tensors from
    # src; backward gathers the gradients onto src.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, group, *tensors):
        # [FUNCTION DOC] HELPER: Saves src and group, checks that all input tensors
        # have the same size (AssertionError otherwise), scatters them
        # from src into a zero tensor shaped like the first and returns
        # this rank's piece.
        ctx.src = src
        ctx.group = group
        if not all(t.size() == tensors[0].size() for t in tensors):
            raise AssertionError
        output = torch.zeros_like(tensors[0])
        if dist.get_rank(group=group) == src:
            dist.scatter(output, list(tensors), src, group=group)
        else:
            dist.scatter(output, None, src, group=group)
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Gathers the gradient onto src and returns the
        # gathered tuple placed after the empty gradients for src and
        # group.
        return (None, None) + _Gather.apply(ctx.src, ctx.group, grad_output)


class _Reduce(Function):
    # [CLASS DOC] HELPER: Autograd Function for reduce. Forward reduces to src with
    # the given op; backward broadcasts the gradient from src.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, src, op, group, tensor):
        # [FUNCTION DOC] HELPER: Saves src and group, clones the tensor, reduces it
        # onto src and returns it.
        ctx.src = src
        ctx.group = group
        tensor = tensor.clone()
        dist.reduce(tensor, src, op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Broadcasts the gradient from the destination rank; no
        # gradient for src, op and group.
        return (None, None, None) + (_Broadcast.apply(ctx.src, ctx.group, grad_output),)


class _Reduce_Scatter(Function):
    # [CLASS DOC] HELPER: Autograd Function for reduce_scatter. Forward reduce-
    # scatters inputs; backward all-gathers the gradient.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, op, group, tensor, *input_tensor_list):
        # [FUNCTION DOC] HELPER: Saves group, makes tensors contiguous, reduce-
        # scatters the input list into the output tensor and returns
        # it.
        ctx.group = group
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()
        input_tensor_list = tuple(t.contiguous() for t in input_tensor_list)
        dist.reduce_scatter(tensor, list(input_tensor_list), op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: All-gathers the gradient over the group; returns no
        # gradient for op, group and the output tensor argument slots.
        return (None, None, None) + _AllGather.apply(ctx.group, grad_output)


class _AllGather(Function):
    # [CLASS DOC] HELPER: Autograd Function for all_gather. Forward gathers tensors
    # from all ranks; backward sums gradients over ranks and returns this
    # rank's piece.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, tensor):
        # [FUNCTION DOC] HELPER: Saves group, gathers the contiguous tensor from all
        # ranks into freshly allocated tensors and returns them as a
        # tuple.
        # Need contiguous tensors for collectives.
        tensor = tensor.contiguous()

        ctx.group = group
        out_tensor_list = [
            torch.empty_like(tensor) for _ in range(dist.get_world_size(group=group))
        ]

        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # [FUNCTION DOC] HELPER: Computes the input gradient: for NCCL and XCCL
        # backends via a reduce-scatter of the gradients, otherwise via
        # all-to-all then summing, since other backends lack reduce-
        # scatter. Returns no gradient for the group.
        if dist.get_backend(group=ctx.group) in (dist.Backend.NCCL, dist.Backend.XCCL):
            rank = dist.get_rank(group=ctx.group)
            gx = torch.empty_like(grad_outputs[rank])
            gx = _Reduce_Scatter.apply(ReduceOp.SUM, ctx.group, gx, *grad_outputs)
        else:
            # As many backends doesn't support ReduceScatter, we use AlltoAll with .sum()
            # to emulate the ReduceScatter behavior
            tensor_list = [torch.empty_like(tensor) for tensor in grad_outputs]
            gxs = _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)
            gx = torch.sum(torch.stack(gxs), dim=0)
        return (None, gx)


class _AllGatherBase(Function):
    # [CLASS DOC] HELPER: Autograd Function for the single-tensor all-gather. Forward
    # gathers into a preallocated output; backward reduce-scatters the
    # gradient.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, output_tensor, input_tensor, group):
        # [FUNCTION DOC] HELPER: Saves group, gathers the contiguous input from all
        # ranks into the given output tensor and returns it.
        ctx.group = group
        dist._all_gather_base(output_tensor, input_tensor.contiguous(), group=group)
        return output_tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Only supported on NCCL and XCCL: reduce-scatters
        # (sum) the gradient into a tensor whose first dimension is
        # divided by the world size. Raises RuntimeError if that
        # dimension is not divisible or the backend is unsupported.
        if dist.get_backend(group=ctx.group) in (dist.Backend.NCCL, dist.Backend.XCCL):
            world_size = dist.get_world_size(group=ctx.group)
            out_size = list(grad_output.size())
            if out_size[0] % world_size != 0:
                raise RuntimeError(
                    f"Tensor with dimensions: {out_size} does "
                    f"not have first dimension divisible by world_size: {world_size}"
                )
            out_size[0] = out_size[0] // dist.get_world_size(group=ctx.group)
            gx = torch.empty(
                out_size, device=grad_output.device, dtype=grad_output.dtype
            )
            dist._reduce_scatter_base(gx, grad_output, ReduceOp.SUM, ctx.group)
        else:
            raise RuntimeError("Backend not supported!")
        return (None, gx, None)


class _AlltoAll(Function):
    # [CLASS DOC] HELPER: Autograd Function for all_to_all. Forward exchanges tensors
    # between ranks; backward performs the reverse exchange.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, out_tensor_list, *tensors):
        # [FUNCTION DOC] HELPER: Saves group and the input tensor sizes; for Gloo
        # emulates all-to-all with one scatter per rank, otherwise uses
        # all-to-all; returns the output tensors as a tuple.
        ctx.group = group
        ctx.input_tensor_size_list = [
            tensors[i].size() for i in range(dist.get_world_size(group=group))
        ]
        my_rank = dist.get_rank(group=group)
        tensors = tuple(t.contiguous() for t in tensors)
        # Implement it on means of scatter/gather, send/recv async operations have issues
        if dist.get_backend(group=group) is dist.Backend.GLOO:
            for i in range(dist.get_world_size(group=group)):
                to_send = None
                if i == my_rank:
                    to_send = list(tensors)
                dist.scatter(out_tensor_list[i], to_send, i, group=group)
        else:
            dist.all_to_all(
                out_tensor_list,
                list(tensors),
                group=group,
            )
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        # [FUNCTION DOC] HELPER: Allocates tensors with the original input sizes and
        # runs all-to-all on the gradients to route them back; no
        # gradient for group and output list.
        tensor_list = [
            torch.empty(
                size, device=grad_outputs[0].device, dtype=grad_outputs[0].dtype
            )
            for size in ctx.input_tensor_size_list
        ]
        return (None, None) + _AlltoAll.apply(ctx.group, tensor_list, *grad_outputs)


class _AlltoAllSingle(Function):
    # [CLASS DOC] HELPER: Autograd Function for all_to_all_single. Forward exchanges
    # splits of one tensor; backward does the reverse exchange with split
    # sizes swapped.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, group, output, output_split_sizes, input_split_sizes, input):
        # [FUNCTION DOC] HELPER: Saves group, input size and swapped split sizes, runs
        # all-to-all-single into output and returns output.
        ctx.group = group
        ctx.input_size = input.size()
        ctx.output_split_sizes = input_split_sizes
        ctx.input_split_sizes = output_split_sizes
        dist.all_to_all_single(
            output,
            input,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )
        return output

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Allocates a tensor of the original input size and
        # runs all-to-all-single on the contiguous gradient with
        # swapped split sizes; no gradients for the other arguments.
        tensor = torch.empty(
            ctx.input_size, device=grad_output.device, dtype=grad_output.dtype
        )
        return (None, None, None, None) + (
            _AlltoAllSingle.apply(
                ctx.group,
                tensor,
                ctx.output_split_sizes,
                ctx.input_split_sizes,
                grad_output.contiguous(),
            ),
        )


class _AllReduce(Function):
    # [CLASS DOC] HELPER: Autograd Function for all_reduce. Forward all-reduces a
    # clone of the tensor; backward all-reduces the gradient with the same
    # op.
    @staticmethod
    # pyrefly: ignore [bad-override]
    def forward(ctx, op, group, tensor):
        # [FUNCTION DOC] HELPER: Saves group and op, clones the tensor, all-reduces it
        # and returns it.
        ctx.group = group
        ctx.op = op
        tensor = tensor.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(tensor, op=op, group=group)
        return tensor

    @staticmethod
    # pyrefly: ignore [bad-override]
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: All-reduces the incoming gradient with the same op
        # and group; no gradient for op and group.
        return (None, None) + (_AllReduce.apply(ctx.op, ctx.group, grad_output),)
