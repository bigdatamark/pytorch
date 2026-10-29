# mypy: allow-untyped-defs
# [MODULE DOC] Default FSDP-style gradient communication hooks: state holders for the
# pre/post-division factors, all-reduce and reduce-scatter hooks, and fp16/bf16 compression hooks.
# No import-time side effects beyond imports.
import functools

import torch
import torch.distributed as dist


class DefaultState:
# [CLASS DOC] Holds the process group, world size, and the pre- and post-division factors used to
# average gradients by world size in two steps (avoiding overflow/underflow). Uses slots; the
# product of the two factors equals the world size.
    r"""
    Stores state needed to perform the default communication algorithm within a communication hook.

    Args:
        process_group (ProcessGroup): The process group to be used.
    """

    __slots__ = [
        "process_group",
        "world_size",
        "gradient_predivide_factor",
        "gradient_postdivide_factor",
    ]

    def __init__(self, process_group: dist.ProcessGroup):
        # [FUNCTION DOC] Store the process group and compute world size and the two division factors.
        # Raises ValueError if process_group is None.
        if process_group is None:
            raise ValueError(f"Expected to pass in an explicit ProcessGroup to {self}.")
        self.process_group = process_group
        self.world_size = dist.get_world_size(process_group)
        # Setting two factors `self.gradient_predivide_factor`
        # and `self.gradient_postdivide_factor` to avoid underflow and overflow
        self.gradient_predivide_factor = self._get_gradient_predivide_factor(
            self.world_size
        )
        self.gradient_postdivide_factor = (
            self.world_size / self.gradient_predivide_factor
        )

    @staticmethod
    def _get_gradient_predivide_factor(world_size: int) -> float:
        # [FUNCTION DOC] HELPER: Returns the largest power of two (as float) that divides world_size
        # and does not exceed its square root-ish bound (loop doubles while divisible and quotient > factor).
        factor: int = 1
        while world_size % factor == 0 and world_size / factor > factor:
            factor *= 2
        return float(factor)


class LowPrecisionState(DefaultState):
# [CLASS DOC] DefaultState extended with parameter_type, the dtype that compressed gradients are
# cast back to after communication (default float32).
    r"""
    Stores state needed to perform gradient communication in a lower precision within a communication hook.

    Communication hook will cast gradients back to the original
    parameter precision specified by ``parameter_type`` (default: torch.float32).
    Builds on top of the :class:`DefaultState`.

    Args:
        parameter_type (torch.dtype): The precision of model's parameters.
        Required for a hook to cast gradients back to a parameter's precision.
    """

    __slots__ = [
        "parameter_type",
    ]

    def __init__(
        self,
        process_group,
        parameter_type=torch.float32,
    ):
        # [FUNCTION DOC] Initialize the base state and record parameter_type.
        super().__init__(process_group)
        self.parameter_type = parameter_type


def _decompress(state: LowPrecisionState, grad: torch.Tensor):
    # [FUNCTION DOC] HELPER: Cast grad.data back to state.parameter_type and keep the original (low
    # precision) storage alive until the current stream of the grad's device backend finishes, via
    # record_stream. Mutates grad in place. Raises AttributeError if no torch backend module exists
    # for the grad's device type (privateuse1 resolves to its registered backend name).
    """
    Casts gradients back to full parameter precision so that further computation happens in full precision.
    """
    orig_grad_data = grad.data
    grad.data = grad.data.to(state.parameter_type)
    device_type = ""
    try:
        if grad.device.type == "privateuse1":
            device_type = torch._C._get_privateuse1_backend_name()
        else:
            device_type = grad.device.type
        backend = getattr(torch, device_type)
    except AttributeError as e:
        raise AttributeError(
            f"Device {grad.device}  does not have a \
                corresponding backend registered as 'torch.device_type'."
        ) from e

    # Don't let this memory get reused until after the transfer.
    orig_grad_data.record_stream(backend.current_stream())  # type: ignore[arg-type]


def allreduce_hook(state: DefaultState, grad: torch.Tensor):
    # [FUNCTION DOC] All-reduce hook: divide grad in place by the pre-division factor (if > 1),
    # all_reduce across state.process_group, then divide by the post-division factor (if > 1),
    # yielding an average over world size. Mutates grad; returns None.
    r"""
    Implement the  FSDP communication hook for ``all_reduce`` algorithm and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks.
    """
    # Average grad by pre-division factor. Together pre- and post-division factors
    # lead to an overall averaging by world_size, required for consistency with PyTorch DDP.
    # This is a two-step process to avoid potential underflow and overflow.
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.all_reduce(grad, group=state.process_group)
    # Average grad by post-division factor.
    if state.gradient_postdivide_factor > 1:
        grad.div_(state.gradient_postdivide_factor)


def reduce_scatter_hook(state: DefaultState, grad: torch.Tensor, output: torch.Tensor):
    # [FUNCTION DOC] Reduce-scatter hook: pre-divide grad in place, reduce_scatter_single into
    # output (this rank's shard), then post-divide output in place. Mutates both; returns None.
    r"""
    Implement the  FSDP communication hook for ``reduce_scatter`` algorithm.

    For sharded FSDP strategies and a necessary pre- and post-division of gradients.

    Args:
        state (DefaultState): State information, configures pre- and post-division factors.
        grad (torch.Tensor): An unsharded gradient for the local batch that needs to be
        communicated across ranks.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    # Average grad by pre-division factor.
    if state.gradient_predivide_factor > 1:
        grad.div_(state.gradient_predivide_factor)
    dist.reduce_scatter_single(output, grad, group=state.process_group)
    # Average grad's shard by post-division factor.
    if state.gradient_postdivide_factor > 1:
        output.div_(state.gradient_postdivide_factor)


def _low_precision_hook(
    prec: torch.dtype,
    state: LowPrecisionState,
    grad: torch.Tensor,
    output: torch.Tensor | None,
):
    # [FUNCTION DOC] HELPER: Shared implementation of the compress hooks. Casts grad (and output if
    # given) to dtype prec, then runs reduce_scatter_hook (when output is given) or allreduce_hook,
    # and casts the result back to the parameter dtype via _decompress. Mutates tensors in place.
    if grad.dtype != prec:
        grad.data = grad.data.to(prec)
    if output is not None:
        if output.dtype != prec:
            output.data = output.data.to(prec)
        reduce_scatter_hook(state, grad, output)
        _decompress(state, output)
    else:
        allreduce_hook(state, grad)
        _decompress(state, grad)


def fp16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None
):
    # [FUNCTION DOC] FSDP hook that communicates gradients in float16; delegates to the shared
    # low-precision helper. output selects reduce-scatter (sharded) vs all-reduce when None.
    r"""
    Implement FSDP communication hook for a simple gradient compression approach.
    Casts ``grad`` to half-precision floating-point format (``torch.float16``).

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    fp16_hook = functools.partial(_low_precision_hook, torch.float16)
    return fp16_hook(state, grad, output)


def bf16_compress_hook(
    state: LowPrecisionState, grad: torch.Tensor, output: torch.Tensor | None = None
):
    # [FUNCTION DOC] FSDP hook that communicates gradients in bfloat16 (docstring says "half
    # precision" generically); delegates to the shared low-precision helper. Same args as fp16 hook.
    r"""
    Implement FSDP communication hook for a simple gradient compression approach .
    Casts ``grad`` to half-precision floating-point format.

    It also averages gradients by ``world_size`` in two steps: first it pre-divides gradients by a
    ``state.gradient_predivide_factor``, and after a communication step (``all_reduce`` or ``reduce_scatter``)
    gradients are averaged by a ``state.gradient_postdivide_factor``.
    Once post-division is done, compressed gradients are casted back to parameters' precision.

    Args:
        state (LowPrecisionState): State information, configures pre- and post-division factors, parameters' precision.
        grad (torch.Tensor): A gradient for the local batch that needs to be communicated across ranks in a lower precision.
        output (torch.Tensor): Stores a single shard of the gradient after ``reduce_scatter``.
    """
    bf16_hook = functools.partial(_low_precision_hook, torch.bfloat16)
    return bf16_hook(state, grad, output)
