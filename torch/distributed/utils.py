# mypy: allow-untyped-defs
# [MODULE DOC]
# Private helpers shared by distributed wrappers (DDP, FSDP and similar): packing
# and unpacking keyword arguments, casting and moving nested inputs between
# dtypes/devices, storage allocation and freeing, module state synchronization
# across ranks, state dict key renaming, and root module discovery. Exports
# nothing publicly (empty __all__). No import-time side effects.
import dataclasses
import traceback
from collections import OrderedDict
from collections.abc import Callable, Container
from typing import Any, Optional, overload, TypeVar

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.utils.rnn import PackedSequence


__all__ = []  # type: ignore[var-annotated]


def _pack_kwargs(*args: Any, **kwargs: Any) -> tuple[tuple[Any, ...], tuple[str, ...]]:
    """
    Turn argument list into separate key list and value list (unpack_kwargs does the opposite).

    Inspiration: https://github.com/facebookresearch/fairscale/blob/eeb6684/fairscale/internal/containers.py#L70
    Usage::

        kwarg_keys, flat_args = pack_kwargs(1, 2, a=3, b=4)
        assert kwarg_keys == ("a", "b")
        assert flat_args == (1, 2, 3, 4)
        args, kwargs = unpack_kwargs(kwarg_keys, flat_args)
        assert args == (1, 2)
        assert kwargs == {"a": 3, "b": 4}
    Returns:
        Tuple[Tuple[Any, ...], Tuple[str, ...]]: The first tuple element gives
        gives both positional args and kwarg values, where the positional args
        proceed kwarg values and kwarg values are ordered consistently with the
        kwarg keys. The second tuple element gives the kwarg keys.
        The second tuple element's length is at most the first tuple element's length.
    """
    # [FUNCTION DOC] HELPER: flattens positional and keyword arguments into one
    # tuple (positionals first, then keyword values in insertion order) and a
    # tuple of the keyword names. Returns (flat_args, kwarg_keys). Inverse of
    # _unpack_kwargs. No side effects.
    kwarg_keys: list[str] = []
    flat_args: list[Any] = list(args)
    for k, v in kwargs.items():
        kwarg_keys.append(k)
        flat_args.append(v)

    return tuple(flat_args), tuple(kwarg_keys)


def _cast_forward_inputs(
    dtype: torch.dtype | None,
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, Any]:
    """
    Cast floating point tensors in ``args`` and ``kwargs`` to ``input_dtype``.

    This respects the existing ``requires_grad`` on the tensors.
    """
    # [FUNCTION DOC] HELPER: converts every floating point tensor found (recursively
    # through containers) in args and kwargs to dtype; non-floating tensors and
    # other objects are untouched. If dtype is None, returns the inputs unchanged.
    # Returns a pair (new_args, new_kwargs). Does not mutate inputs.
    # KNOWN ISSUE: the docstring mentions an input_dtype name; the parameter is dtype.
    if dtype is None:
        return args, kwargs

    def cast_fn(x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: returns x as-is if it is not floating point or
        # already has the target dtype; otherwise returns a converted copy.
        if not torch.is_floating_point(x) or x.dtype == dtype:
            return x

        return x.to(dtype)

    return (_apply_to_tensors(cast_fn, args), _apply_to_tensors(cast_fn, kwargs))


def _unpack_kwargs(
    flat_args: tuple[Any, ...], kwarg_keys: tuple[str, ...]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """See _pack_kwargs."""
    # [FUNCTION DOC] HELPER: splits flat_args into positional args (all but the
    # last len(kwarg_keys) entries) and a mapping from each key to its value.
    # Returns (args, kwargs). Raises AssertionError if there are more keys than
    # values.
    if len(kwarg_keys) > len(flat_args):
        raise AssertionError(f"too many keys {len(kwarg_keys)} vs. {len(flat_args)}")
    if len(kwarg_keys) == 0:
        return flat_args, {}
    args = flat_args[: -len(kwarg_keys)]
    kwargs = dict(zip(kwarg_keys, flat_args[-len(kwarg_keys) :]))
    return args, kwargs


S = TypeVar("S", dict, list, tuple)
T = TypeVar("T", torch.Tensor, PackedSequence)


@overload
def _recursive_to(
    inputs: S, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> list[S]: ...


@overload
def _recursive_to(
    inputs: T, target_device: torch.device, use_side_stream_for_tensor_copies: bool
) -> tuple[T]: ...


def _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies):
    r"""Recursively moves input to the target_device."""
    # [FUNCTION DOC] HELPER: recursively moves tensors and packed sequences found
    # in inputs (including inside tuples, namedtuples, lists and dicts) to
    # target_device. Returns a list of the moved structure (a one-element list for
    # a single device); containers whose elements all stayed identical are
    # returned as the original object. When use_side_stream_for_tensor_copies is
    # true, non-CPU tensors are copied on a side stream, the current stream waits
    # on it, and record_stream is called on the result (CPU tensors are copied
    # normally). Breaks the closure reference cycle before returning.
    # KNOWN ISSUE: the docstring calls the argument input and claims no list.

    def to_map(obj):
        # [FUNCTION DOC] HELPER: maps one object to a one-element tuple holding
        # the object moved to the target device (tensors/packed sequences) or
        # a list of rebuilt containers; recurses into containers.
        if isinstance(obj, (torch.Tensor, PackedSequence)):
            device = obj.data.device if isinstance(obj, PackedSequence) else obj.device
            if device == target_device:
                return (obj,)
            if not use_side_stream_for_tensor_copies:
                return (obj.to(target_device),)
            else:
                # If the custom module is not registered to torch, stream is not used for acceleration
                if device.type == "cpu":
                    return (obj.to(target_device),)

                from torch.nn.parallel._functions import _get_stream

                # Perform CPU -> target_device copies in a background stream. This code is
                # motivated from similar logic in torch/nn/parallel/_functions.py
                stream = _get_stream(target_device)
                with stream:
                    output = obj.to(target_device)
                # synchronize with the copy stream
                with torch.accelerator.device_index(target_device.index):
                    current_stream = torch.accelerator.current_stream()
                    # Sync the current stream with the copy stream
                    current_stream.wait_stream(stream)
                    # Ensure tensor memory is not reused until work on
                    # main stream is complete
                    if isinstance(obj, PackedSequence):
                        output.data.record_stream(current_stream)  # type: ignore[arg-type]
                    else:
                        if not isinstance(output, torch.Tensor):
                            raise AssertionError("output must be a torch.Tensor")
                        output.record_stream(current_stream)  # type: ignore[arg-type]
                return (output,)

        from torch.nn.parallel.scatter_gather import _is_namedtuple

        def _handle_container(obj, elements, make_container):
            """Handle container types with object identity preservation."""
            # [FUNCTION DOC] HELPER: applies the mapper to each element and
            # rebuilds the container; returns the original object (in a list) if
            # nothing changed, a single rebuilt container, or one container per
            # device when elements expanded to several results.
            # pyrefly: ignore [bad-argument-type]
            mapped = list(map(to_map, elements))
            # Preserve object identity when all elements are unchanged (single-device case)
            if all(len(m) == 1 for m in mapped):
                transformed = [m[0] for m in mapped]
                if all(t is o for t, o in zip(transformed, elements)):
                    return [obj]
                # pyrefly: ignore [no-matching-overload]
                return [make_container(transformed)]
            # pyrefly: ignore [no-matching-overload]
            return [make_container(args) for args in zip(*mapped)]

        if _is_namedtuple(obj):
            return _handle_container(obj, obj, lambda x: type(obj)(*x))
        if isinstance(obj, tuple) and len(obj) > 0:
            return _handle_container(obj, obj, tuple)
        if isinstance(obj, list) and len(obj) > 0:
            return _handle_container(obj, obj, list)
        if isinstance(obj, dict) and len(obj) > 0:
            keys = list(obj.keys())
            return _handle_container(
                obj, obj.values(), lambda v: type(obj)(zip(keys, v))
            )
        return [obj]

    # Avoid reference cycle
    try:
        res = to_map(inputs)
    finally:
        to_map = None  # type: ignore[assignment]
    return res


def _p_assert(cond: Any, s: str, raise_assertion_error: bool = True) -> None:
    """Alternate to ``assert`` when in the backward context to print the error message ``s`` since otherwise, it is swallowed."""
    # [FUNCTION DOC] HELPER: if cond is falsy, prints the message s and the current
    # stack to standard output, then raises AssertionError(s) unless
    # raise_assertion_error is false. Returns None.
    if not cond:
        print(s)
        traceback.print_stack()
        if raise_assertion_error:
            raise AssertionError(s)


def _alloc_storage(tensor: torch.Tensor, size: torch.Size) -> None:
    """
    Allocate storage for ``tensor`` with the given size.

    Returns:
        bool: ``True`` if this method allocated storage and ``False`` if the
        storage was already allocated.
    """
    # [FUNCTION DOC] HELPER: under no-grad and when not compiling with Dynamo,
    # resizes the tensor's typed storage to the element count of size, if it is
    # not already that size; asserts first that the storage was empty.
    # KNOWN ISSUE: docstring says it returns a bool, but the function returns None.
    # KNOWN ISSUE: the assertion message contains a literal placeholder word that
    # is never filled in with the actual size.
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_allocated = tensor._typed_storage()._size() == size.numel()
            if not already_allocated:
                tensor_storage_size = tensor._typed_storage()._size()
                _p_assert(
                    tensor_storage_size == 0,
                    "Tensor storage should have been resized to be 0 but got PLACEHOLDER",
                )
                tensor._typed_storage()._resize_(size.numel())


def _free_storage(tensor: torch.Tensor):
    """
    Frees the underlying storage of ``tensor``.

    Returns:
        bool: ``True`` if the method freed the storage and ``False`` if the
        storage was already freed.
    """
    # [FUNCTION DOC] HELPER: under no-grad and when not compiling with Dynamo,
    # resizes the tensor's typed storage to zero if it is non-empty, first
    # asserting the tensor has storage offset zero (is sole occupant).
    # KNOWN ISSUE: docstring says it returns a bool, but the function returns None.
    with torch.no_grad():
        if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
            already_freed = tensor._typed_storage()._size() == 0
            if not already_freed:
                _p_assert(
                    tensor.storage_offset() == 0,
                    "Freeing a tensor's storage is unsafe when it is not the sole occupant\n"
                    f"storage offset: {tensor.storage_offset()}\n"
                    f"storage size: {tensor._typed_storage()._size()}\n"
                    f"tensor shape: {tensor.shape}",
                )
                tensor._typed_storage()._resize_(0)


Q = TypeVar("Q")
R = TypeVar("R", dict, list, tuple, set, OrderedDict, PackedSequence, Any)


@overload
def _apply_to_tensors(
    fn: Callable[[torch.Tensor], Q], container: torch.Tensor
) -> Q: ...


@overload
def _apply_to_tensors(fn: Callable[[torch.Tensor], Any], container: R) -> R: ...


def _apply_to_tensors(fn, container):
    """Recursively apply to all tensor in different kinds of container types."""
    # [FUNCTION DOC] HELPER: returns a copy of container in which fn has been
    # applied to every tensor, traversing dataclasses, ordered dicts, dicts,
    # namedtuples, lists, tuples and sets; other objects are returned unchanged.
    # A bare tensor returns fn's result directly.
    # KNOWN ISSUE: for a packed sequence, fn is applied to its data but the result
    # is discarded and the original object is returned, so only side effects of
    # fn take effect.

    def apply(x):
        # [FUNCTION DOC] HELPER: recursive worker that dispatches on the type of x
        # and rebuilds it with fn applied to nested tensors.
        from torch.nn.parallel.scatter_gather import _is_namedtuple

        if isinstance(x, torch.Tensor):
            return fn(x)
        elif hasattr(x, "__dataclass_fields__"):
            dc = dataclasses.replace(x)
            changes = {
                f.name: apply(getattr(dc, f.name)) for f in dataclasses.fields(dc)
            }
            return dataclasses.replace(dc, **changes)
        elif isinstance(x, OrderedDict):
            od = x.__class__()
            for key, value in x.items():
                od[key] = apply(value)
            return od
        elif isinstance(x, PackedSequence):
            apply(x.data)
            return x
        elif isinstance(x, dict):
            return {key: apply(value) for key, value in x.items()}
        elif _is_namedtuple(x):
            res = (apply(el) for el in x)
            return type(x)(*res)
        elif isinstance(x, (list, tuple, set)):
            return type(x)(apply(el) for el in x)
        else:
            return x

    return apply(container)


def _to_kwargs(
    inputs: tuple[Any, ...],
    kwargs: dict[str, Any] | None,
    target_device: torch.device,
    use_side_stream_for_tensor_copies: bool,
) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
    # [FUNCTION DOC] HELPER: moves positional inputs and keyword arguments to
    # target_device via _recursive_to, then pads the shorter result list with
    # empty tuples or empty dicts so both have equal length. Returns
    # (inputs_tuple, kwargs_tuple).
    # KNOWN ISSUE: the padding for inputs uses the length of the original inputs
    # rather than of moved_inputs, which gives a wrong count if they differ.
    moved_inputs = (
        _recursive_to(inputs, target_device, use_side_stream_for_tensor_copies)
        if inputs
        else []
    )
    moved_kwargs = (
        _recursive_to(kwargs, target_device, use_side_stream_for_tensor_copies)
        if kwargs
        else []
    )
    if len(moved_inputs) < len(moved_kwargs):
        moved_inputs.extend([() for _ in range(len(moved_kwargs) - len(inputs))])
    elif len(moved_kwargs) < len(moved_inputs):
        moved_kwargs.extend([{} for _ in range(len(moved_inputs) - len(moved_kwargs))])
    return tuple(moved_inputs), tuple(moved_kwargs)


def _verify_param_shape_across_processes(
    process_group: dist.ProcessGroup,
    tensors: list[torch.Tensor],
    logger: Optional["dist.Logger"] = None,
):
    # [FUNCTION DOC] HELPER: thin wrapper that asks the distributed backend to
    # verify that tensor shapes match across all processes of process_group
    # (optionally logging via logger); errors come from the backend.
    return dist._verify_params_across_processes(process_group, tensors, logger)


def _sync_module_states(
    module: nn.Module,
    process_group: dist.ProcessGroup,
    broadcast_bucket_size: int,
    src: int,
    params_and_buffers_to_ignore: Container[str],
    broadcast_buffers: bool = True,
) -> None:
    """
    Sync ``module``'s parameters and buffers state.

    Syncs ``module``'s parameters and buffers state so that all ranks contain
    the same module state across all ranks. Note that this API assumes that all
    parameter shapes are consistent before running the synchronization. This can
    be checked with ``_verify_param_shape_across_processes``.
    """
    # [FUNCTION DOC] HELPER: collects detached parameters (and buffers, when
    # broadcast_buffers is true) of module whose names are not in
    # params_and_buffers_to_ignore, then broadcasts them in place from rank src of
    # process_group in buckets of broadcast_bucket_size. Mutates module state on
    # non-source ranks; collective call. Returns None.
    module_states: list[torch.Tensor] = []
    for name, param in module.named_parameters():
        if name not in params_and_buffers_to_ignore:
            module_states.append(param.detach())

    if broadcast_buffers:
        for name, buffer in module.named_buffers():
            if name not in params_and_buffers_to_ignore:
                module_states.append(buffer.detach())

    _sync_params_and_buffers(process_group, module_states, broadcast_bucket_size, src)


def _sync_params_and_buffers(
    process_group: dist.ProcessGroup,
    module_states: list[torch.Tensor],
    broadcast_bucket_size: int,
    src: int,
) -> None:
    """Synchronize ``module_states`` (list of tensors) across all processes by broadcasting them from rank 0."""
    # [FUNCTION DOC] HELPER: if module_states is non-empty, broadcasts the tensors
    # in place with coalescing from rank src of process_group using buckets of
    # broadcast_bucket_size. Collective call. Returns None.
    # KNOWN ISSUE: docstring says the source is rank 0, but the src argument is used.
    if len(module_states) > 0:
        dist._broadcast_coalesced(
            process_group, module_states, broadcast_bucket_size, src
        )


def _replace_by_prefix(
    state_dict: dict[str, Any],
    old_prefix: str,
    new_prefix: str,
) -> None:
    """
    Replace all keys that match a given old_prefix with a new_prefix (in-place).

    Usage::

        state_dict = {"layer.xyz": torch.tensor(1)}
        replace_by_prefix_(state_dict, "layer.", "module.layer.")
        assert state_dict == {"module.layer.xyz": torch.tensor(1)}
    """
    # [FUNCTION DOC] HELPER: renames in place every state_dict key beginning with
    # old_prefix so that it begins with new_prefix instead, preserving values.
    # Raises ValueError if the prefixes are equal. Mutates state_dict (new keys
    # are added at the end of the ordering). Returns None.
    if old_prefix == new_prefix:
        raise ValueError("old_prefix and new_prefix must be distinct")
    for key in list(state_dict.keys()):
        if not key.startswith(old_prefix):
            continue
        new_key = new_prefix + key[len(old_prefix) :]
        state_dict[new_key] = state_dict[key]
        del state_dict[key]


def _data_ptr_allocated(tensor: torch.Tensor) -> bool:
    # [FUNCTION DOC] HELPER: returns True if the tensor's untyped storage has a
    # nonzero data pointer (i.e. memory is currently allocated).
    return tensor.untyped_storage().data_ptr() > 0


def _get_root_modules(modules: list[nn.Module]) -> list[nn.Module]:
    """
    Returns the modules in ``modules`` that are root modules (i.e.
    parent-less) with respect to the set ``modules``. In other words, these
    are the modules in ``modules`` that are the not child of any other
    module in ``modules``.
    """
    # [FUNCTION DOC] HELPER: returns, in input order, the modules that are not a
    # submodule (by identity-based set membership over each module's full module
    # tree) of any other module in the list. Pure function.
    root_modules: list[nn.Module] = []
    module_to_modules: dict[nn.Module, set[nn.Module]] = {
        module: set(module.modules()) for module in modules
    }
    for candidate_module in modules:
        is_root_module = True
        for module, _modules in module_to_modules.items():
            is_child_module = (
                candidate_module is not module and candidate_module in _modules
            )
            if is_child_module:
                is_root_module = False
                break
        if is_root_module:
            root_modules.append(candidate_module)
    return root_modules
