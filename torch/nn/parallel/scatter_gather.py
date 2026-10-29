# mypy: allow-untyped-defs
# [MODULE DOC] Helpers that split inputs across devices (scatter, scatter_kwargs) and combine
# per-device outputs (gather) for DataParallel, recursing through tensors, tuples, lists, dicts
# and namedtuples via the autograd Scatter/Gather functions. No import-time side effects.
from collections.abc import Sequence
from typing import Any, overload, TypeVar
from typing_extensions import deprecated

import torch
from torch.nn.parallel._functions import Gather, Scatter


__all__ = ["scatter", "scatter_kwargs", "gather"]


@deprecated(
    "`is_namedtuple` is deprecated, please use the python checks instead",
    category=FutureWarning,
)
def is_namedtuple(obj: Any) -> bool:
    # [FUNCTION DOC] HELPER: Deprecated public alias (emits a FutureWarning) that returns whether
    # obj is a namedtuple instance by delegating to the private check.
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return _is_namedtuple(obj)


def _is_namedtuple(obj: Any) -> bool:
    # [FUNCTION DOC] HELPER: Return True if obj is a tuple that also has the namedtuple
    # attributes for field names and dict conversion; otherwise False.
    # Check if type was created from collections.namedtuple or a typing.NamedTuple.
    return (
        isinstance(obj, tuple) and hasattr(obj, "_asdict") and hasattr(obj, "_fields")
    )


T = TypeVar("T", dict, list, tuple)


# For some reason, 'scatter' returns a tuple when given a single Tensor input but a list otherwise.
@overload
def scatter(
    inputs: torch.Tensor,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> tuple[torch.Tensor, ...]: ...


@overload
def scatter(
    inputs: T,
    target_gpus: Sequence[int | torch.device],
    dim: int = ...,
) -> list[T]: ...


def scatter(inputs, target_gpus, dim=0):
    # [FUNCTION DOC] Recursively split inputs into one piece per target device. Tensors are
    # chunked along dim by the Scatter autograd function; tuples, lists, dicts and namedtuples
    # are scattered element-wise and regrouped per device; any other object (and empty
    # containers) is repeated by reference for each device. Returns a list with one entry per
    # device (a tuple when the input is a single tensor). The recursive helper is cleared in a
    # finally block to avoid a reference cycle.
    r"""Slice tensors into approximately equal chunks and distributes them across given GPUs.

    Duplicates references to objects that are not tensors.
    """

    def scatter_map(obj):
        # [FUNCTION DOC] HELPER: Recursive worker for scatter; maps obj to a per-device list
        # using target_gpus and dim from the enclosing scope. Tensors go through Scatter, known
        # containers are scattered per element and zipped back, others are repeated per device.
        if isinstance(obj, torch.Tensor):
            return Scatter.apply(target_gpus, None, dim, obj)
        if _is_namedtuple(obj):
            return [
                type(obj)(*args)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                for args in zip(*map(scatter_map, obj), strict=False)
            ]
        if isinstance(obj, tuple) and len(obj) > 0:
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            return list(zip(*map(scatter_map, obj), strict=False))
        if isinstance(obj, list) and len(obj) > 0:
            # pyrefly: ignore [bad-argument-type, no-matching-overload]
            return [list(i) for i in zip(*map(scatter_map, obj), strict=False)]
        if isinstance(obj, dict) and len(obj) > 0:
            return [
                type(obj)(i)
                # pyrefly: ignore [bad-argument-type, no-matching-overload]
                for i in zip(*map(scatter_map, obj.items()), strict=False)
            ]
        return [obj for _ in target_gpus]

    # After scatter_map is called, a scatter_map cell will exist. This cell
    # has a reference to the actual function scatter_map, which has references
    # to a closure that has a reference to the scatter_map cell (because the
    # fn is recursive). To avoid this reference cycle, we set the function to
    # None, clearing the cell
    try:
        res = scatter_map(inputs)
    finally:
        scatter_map = None  # type: ignore[assignment]
    return res


def scatter_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_gpus: Sequence[int | torch.device],
    dim: int = 0,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    # [FUNCTION DOC] Scatter positional inputs and keyword arguments across target_gpus along
    # dim, then pad the shorter of the two scattered lists (with empty tuples or empty dicts) so
    # both have one entry per replica. Returns a pair of tuples (inputs per device, kwargs per
    # device).
    # KNOWN ISSUE: the elif compares the scattered kwargs length against len(inputs) (the
    # unscattered input count) rather than the scattered inputs length, which looks like a bug.
    r"""Scatter with support for kwargs dictionary."""
    scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
    scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
    if len(scattered_inputs) < len(scattered_kwargs):
        scattered_inputs.extend(
            () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
        )
    elif len(scattered_kwargs) < len(inputs):
        scattered_kwargs.extend(
            {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
        )
    return tuple(scattered_inputs), tuple(scattered_kwargs)


def gather(outputs: Any, target_device: int | torch.device, dim: int = 0) -> Any:
    # [FUNCTION DOC] Combine a sequence of per-device outputs into one object on target_device.
    # Tensors are concatenated along dim via the Gather autograd function, None passes through,
    # dicts are gathered per key, and namedtuples, tuples and lists are gathered per position.
    # Raises ValueError if dicts differ in key count. The recursive helper is cleared in a
    # finally block to avoid a reference cycle.
    r"""Gather tensors from different GPUs on a specified device.

    This function is useful for gathering the results of a distributed computation.
    It takes a sequence of objects, one for each GPU, and returns a single object
    on the specified device.

    Args:
        outputs (Any): A sequence of objects (potentially tensors) to gather.
        target_device (Union[int, torch.device]): The device to gather the tensors to.
            Use 'cpu' for CPU to avoid a deprecation warning.
        dim (int, optional): The dimension along which to gather. Default: 0.

    Returns:
        Any: A gathered object (potentially tensor) on the specified device.
    """

    def gather_map(outputs):
        # [FUNCTION DOC] HELPER: Recursive worker for gather; dispatches on the type of the first
        # device's output and rebuilds the same container type from the gathered elements.
        out = outputs[0]
        if isinstance(out, torch.Tensor):
            return Gather.apply(target_device, dim, *outputs)
        if out is None:
            return None
        if isinstance(out, dict):
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")
            # pyrefly: ignore [not-callable]
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)
        if _is_namedtuple(out):
            # pyrefly: ignore [bad-argument-type]
            return type(out)._make(map(gather_map, zip(*outputs, strict=True)))
        # pyrefly: ignore [bad-argument-type]
        return type(out)(map(gather_map, zip(*outputs, strict=True)))

    # Recursive function calls like this create reference cycles.
    # Setting the function to None clears the refcycle.
    try:
        res = gather_map(outputs)
    finally:
        gather_map = None  # type: ignore[assignment]
    return res
