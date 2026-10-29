# mypy: allow-untyped-defs
# [PACKAGE DOC] ShardedTensor package (deprecated in favor of DTensor). Re-exports from api: Shard, ShardedTensor, ShardedTensorBase, ShardedTensorMetadata, TensorProperties, plus ShardMetadata.
# Defines creation functions (empty, ones, zeros, full, rand, randn, init_from_local_shards), state-dict hooks that carry ShardedTensor attributes, and decorators for registering sharded op handlers
# (custom_sharded_op_impl for user ops, _sharded_op_impl for builtin ones). Import-time side effect: the final star-import of the _ops subpackage registers all builtin op handlers.
import functools
from typing import TYPE_CHECKING

import torch
from torch.distributed._shard.op_registry_utils import _decorator_func

from .api import (
    _CUSTOM_SHARDED_OPS,
    _SHARDED_OPS,
    Shard,
    ShardedTensor,
    ShardedTensorBase,
    ShardedTensorMetadata,
    TensorProperties,
)
from .metadata import ShardMetadata


if TYPE_CHECKING:
    from torch.distributed._shard.sharding_spec import ShardingSpec
else:
    ShardingSpec = "ShardingSpec"


def empty(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor with uninitialized data using sharding_spec and the given global size (varargs), forwarding dtype, layout, requires_grad, pin_memory,
    # memory_format, process_group and init_rrefs to the constructor. Collective: call on all ranks.
    """
    Returns a :class:`ShardedTensor` filled with uninitialized data.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        memory_format (:class:`torch.memory_format`, optional): the desired memory format of
            returned Tensor. Default: ``torch.contiguous_format``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def ones(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor filled with ones by delegating to full with fill value 1; other arguments are forwarded. Collective: call on all ranks.
    """
    Returns a :class:`ShardedTensor` with the scalar value 1.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return full(
        sharding_spec,
        size,
        fill_value=1,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def zeros(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor filled with zeros by delegating to full with fill value 0; other arguments are forwarded. Collective: call on all ranks.
    """
    Returns a :class:`ShardedTensor` filled with the scalar value 0.
        Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...): a sequence of integers defining the shape of the output
            tensor. Can be a variable number of arguments or a collection like a list or tuple.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    return full(
        sharding_spec,
        size,
        fill_value=0,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )


def full(
    sharding_spec: ShardingSpec,
    size,
    fill_value,
    *,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor of the given global size and fill every local shard with fill_value via the constant init handler.
    # Keyword arguments (dtype, layout, requires_grad, pin_memory, memory_format, process_group, init_rrefs) are forwarded to the constructor. Collective: call on all ranks.
    """
    Creates a :class:`ShardedTensor` filled with fill_value. The tensor's dtype
        is inferred from fill_value. If dtype is specified, it will override the
        inferred type from fill_value. Needs to be called on all ranks in an SPMD fashion.
    Args:
        sharding_spec (:class:`torch.distributed._sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.
        fill_value (Scalar) - the value to fill the output tensor with.
    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.
    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.constant_(sharded_tensor, fill_value)  # type: ignore[arg-type]
    return sharded_tensor


def rand(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor and fill local shards with samples from the uniform distribution on zero to one using the uniform init handler. Collective: call on all ranks.
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        on the interval :math:`[0, 1)`. The shape of the tensor is defined by the
        variable argument `size`. Needs to be called on all ranks in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.uniform_(sharded_tensor, 0, 1)  # type: ignore[arg-type]
    return sharded_tensor


def randn(
    sharding_spec: ShardingSpec,
    *size,
    dtype=None,
    layout=torch.strided,
    requires_grad=False,
    pin_memory=False,
    memory_format=torch.contiguous_format,
    process_group=None,
    init_rrefs=False,
) -> ShardedTensor:
    # [FUNCTION DOC] Create a ShardedTensor and fill local shards with samples from the standard normal distribution (mean 0, std 1) via the normal init handler. Collective: call on all ranks.
    # KNOWN ISSUE: the docstring says uniform distribution but the code samples a normal distribution.
    """
    Creates a :class:`ShardedTensor` filled with random numbers from a uniform distribution
        with mean `0` and variance `1` (also called standard normal distribution). The shape
        of the tensor is defined by the variable argument `size`. Needs to be called on all ranks
        in an SPMD fashion.

    Args:
        sharding_spec (:class:`torch.distributed._shard.sharding_spec.ShardingSpec`): The specification
            describing how to shard the Tensor.
        size (int...):  a list, tuple, or `torch.Size` of integers defining the shape of the
            output tensor.

    Keyword args:
        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.
            Default: if ``None``, uses a global default (see :func:`torch.set_default_dtype`).
        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.
            Default: ``torch.strided``.
        requires_grad (bool, optional): If autograd should record operations on the
            returned tensor. Default: ``False``.
        pin_memory (bool, optional): If set, returned tensor would be allocated in
            the pinned memory. Works only for CPU tensors. Default: ``False``.
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object on each rank
    """
    sharded_tensor = ShardedTensor(
        sharding_spec,
        *size,
        dtype=dtype,
        layout=layout,
        requires_grad=requires_grad,
        pin_memory=pin_memory,
        memory_format=memory_format,
        process_group=process_group,
        init_rrefs=init_rrefs,
    )
    torch.nn.init.normal_(sharded_tensor, 0, 1)  # type: ignore[arg-type]
    return sharded_tensor


def init_from_local_shards(
    local_shards: list[Shard], *global_size, process_group=None, init_rrefs=False
) -> ShardedTensor:
    # [FUNCTION DOC] Build a ShardedTensor from this rank's list of local Shard objects and the global size (varargs), via the class-level construction helper
    # (which gathers global metadata). Collective: call on all ranks.
    # KNOWN ISSUE: the docstring examples construct ShardMetadata with a shard_lengths keyword while the class field is shard_sizes.
    """
    Creates an :class:`ShardedTensor` from local shards and the global metadata.
    Needs to be called on all ranks in an SPMD fashion.

    Args:
        local_shards (List[:class `torch.distributed._shard.sharded_tensor.Shard`]): A list
            of shards that represent the local shards on this rank.
        global_size (int...):  a list, tuple, or `torch.Size` of integers defining the
            shape of the overall sharded tensor.

    Keyword args:
        process_group (ProcessGroup, optional): The process group to work on. If None,
            the default process group will be used.
        init_rrefs (bool, optional): Whether or not to initialize
            :class:`torch.distributed.rpc.RRef`s pointing to remote shards.
            Need to initialize the RPC Framework if specified as ``True``.
            Default: ``False``.

    Returns:
        A :class:`ShardedTensor` object handle on this rank


    Examples:
        Suppose we want construct a sharded tensor on two ranks, global size = (10, 5),
        each shard have a (5, 5) local tensor, we can do it like below:

        on rank 0:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[0, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:0/cuda:0"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])

        on rank 1:
        >>> # xdoctest: +SKIP("not distributed")
        >>> local_shard_metadata = ShardMetadata(
        >>>     shard_offsets=[5, 0],
        >>>     shard_lengths=[5, 5],
        >>>     placement="rank:1/cuda:1"
        >>> )
        >>> local_shards = [Shard(torch.randn(5, 5), local_shard_metadata)]
        >>> sharded_tensor = init_from_local_shards(local_shards, [10, 5])
    """
    return ShardedTensor._init_from_local_shards(
        local_shards, *global_size, process_group=process_group, init_rrefs=init_rrefs
    )


def state_dict_hook(module, destination, prefix, local_metadata):
    # [FUNCTION DOC] State dict hook: for the module and each submodule, every instance attribute that is a ShardedTensor is stored into destination under the dotted
    # prefix plus submodule path plus attribute name. Mutates destination; returns nothing.
    """
    Hook to add ShardedTensor to Module's ``state_dict``. Needs to be
    registered to the Module using
    :meth:`torch.nn.Module._register_state_dict_hook`.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name, attr in submodule.__dict__.items():
            if isinstance(attr, ShardedTensor):
                mod_prefix = prefix + submodule_name
                key = mod_prefix + ("." if mod_prefix else "") + attr_name
                destination[key] = attr


def pre_load_state_dict_hook(
    module,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # [FUNCTION DOC] Pre-load hook: for every submodule attribute name whose prefixed dotted key is in state_dict and holds a ShardedTensor, set that attribute on the submodule.
    # The strict, missing/unexpected keys and error message arguments are unused.
    """
    Pre-load state dict hook to add ShardedTensor to the module.
    """
    for submodule_name, submodule in module.named_modules():
        for attr_name in submodule.__dict__:
            mod_prefix = prefix + submodule_name
            key = mod_prefix + ("." if mod_prefix else "") + attr_name
            if key in state_dict:
                if isinstance(state_dict[key], ShardedTensor):
                    setattr(submodule, attr_name, state_dict[key])


def custom_sharded_op_impl(func):
    # [FUNCTION DOC] Return a decorator (a partial application of the registration helper) that registers a user function as the handler for op func in the custom sharded op table
    # (takes priority for overrides of builtin handling). The handler gets (types, args, kwargs, process_group).
    """
    Provides a way for users to write their own custom sharded operator. This
    can be used to override existing ShardedTensor operators or write a new
    one not supported by ShardedTensor. If the operator in question is covered
    by ``__torch_function__`` dispatch and has a ShardedTensor as any of its
    parameters, the function provided will be invoked for that operator.

    Example::
        >>> # xdoctest: +SKIP
        >>> @custom_sharded_op_impl(torch.nn.functional.linear)
        >>> def my_custom_sharded_linear(types, args, kwargs, process_group):
        >>>     ...
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> input = torch.rand(10, 32)
        >>> weight = sharded_tensor.rand(32, 16)
        >>> bias = torch.rand(16)
        >>> # This will call 'my_custom_sharded_linear'
        >>> torch.nn.functional.linear(input, weight, bias)

    The types, args and kwargs parameters are the same parameters that are
    passed to ``__torch_function__`` dispatch API
    (https://pytorch.org/docs/stable/notes/extending.html#extending-torch).
    There is an additional ``process_group`` parameter which is the
    process_group used for the ShardedTensor and can be used by
    implementations for communications within a sharded implementation.

    Args:
        func(Callable): Torch function for which we want to provide a sharded
            implementation (ex: torch.nn.functional.linear)
    """
    return functools.partial(_decorator_func, op=func, op_table=_CUSTOM_SHARDED_OPS)


def _sharded_op_impl(func):
    # [FUNCTION DOC] HELPER: Return a decorator that registers a function as the builtin sharded handler for op func in the default sharded op table.
    """
    Decorator to register a default sharded op.
    """
    return functools.partial(_decorator_func, op=func, op_table=_SHARDED_OPS)


# Import all builtin sharded ops
from ._ops import *  # noqa: F403
