# mypy: allow-untyped-defs
# [MODULE DOC] ShardedTensor handlers for torch.nn.init ops (uniform_, normal_, kaiming_uniform_, constant_), which apply the init to each local shard tensor,
# and for tensor-like creation ops (full_like, empty_like, zeros_like, ones_like, rand_like, randn_like) mapped to the sharded creation functions via tensor_like_creation_op_map.
# Registers all handlers at import time.
import torch
import torch.distributed._shard.sharded_tensor as sharded_tensor
from torch.distributed._shard.sharded_tensor import _sharded_op_impl


def validate_param(param, param_name):
    # [FUNCTION DOC] HELPER: Raise ValueError if param is None, naming param_name in the message.
    if param is None:
        raise ValueError(f"param: {param_name} shouldn't be None!")


@_sharded_op_impl(torch.nn.init.uniform_)
def uniform_(types, args=(), kwargs=None, pg=None):
    # [FUNCTION DOC] Sharded handler for uniform init: reads tensor, a and b from kwargs (each must be present and not None, else ValueError or KeyError),
    # fills every local shard tensor in place, and returns the same ShardedTensor.
    r"""
    Fills the Tensor in tensor.local_shards with values drawn from the uniform
    distribution :math:`\mathcal{U}(a, b)`.
    Args:
        tensor: tensor sharded across devices
        a: the lower bound of the uniform distribution
        b: the upper bound of the uniform distribution
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    a = kwargs["a"]
    validate_param(a, "a")
    # pyrefly: ignore [unsupported-operation]
    b = kwargs["b"]
    validate_param(b, "b")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.uniform_(shard.tensor, a=a, b=b)
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.normal_)
def normal_(types, args=(), kwargs=None, pg=None):
    # [FUNCTION DOC] Sharded handler for normal init: reads tensor, mean and std from kwargs (validated as not None), applies the init to every local shard in place and returns the ShardedTensor.
    r"""
    Fills the Tensors in tensor.local_shards with values drawn from the normal
    distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`.
    Args:
        tensor: tensor sharded across devices
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    mean = kwargs["mean"]
    validate_param(mean, "mean")
    # pyrefly: ignore [unsupported-operation]
    std = kwargs["std"]
    validate_param(std, "std")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.normal_(shard.tensor, mean=mean, std=std)
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.kaiming_uniform_)
def kaiming_uniform_(types, args=(), kwargs=None, pg=None):
    # [FUNCTION DOC] Sharded handler for Kaiming uniform init: reads tensor, a, mode and nonlinearity from kwargs (validated as not None), initializes every local shard in place and returns the ShardedTensor.
    # Note fan computation is per local shard, not global.
    r"""
    Fills the Tensors in tensor.local_shards with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where
    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}
    Also known as He initialization.
    Args:
        tensor: tensor sharded across devices
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    a = kwargs["a"]
    validate_param(a, "a")
    # pyrefly: ignore [unsupported-operation]
    mode = kwargs["mode"]
    validate_param(mode, "mode")
    # pyrefly: ignore [unsupported-operation]
    nonlinearity = kwargs["nonlinearity"]
    validate_param(nonlinearity, "nonlinearity")

    for shard in sharded_tensor.local_shards():
        torch.nn.init.kaiming_uniform_(
            shard.tensor, a=a, mode=mode, nonlinearity=nonlinearity
        )
    return sharded_tensor


@_sharded_op_impl(torch.nn.init.constant_)
def constant_(types, args=(), kwargs=None, pg=None):
    # [FUNCTION DOC] Sharded handler for constant fill: reads tensor and val from kwargs (validated as not None), fills every local shard in place and returns the ShardedTensor.
    r"""
    Fills the input ShardedTensor with the value \text{val}val.
    Args:
        tensor: tensor sharded across devices
        val: the value to fill the tensor with
    """
    validate_param(kwargs, "kwargs")
    # pyrefly: ignore [unsupported-operation]
    sharded_tensor = kwargs["tensor"]
    validate_param(sharded_tensor, "tensor")
    # pyrefly: ignore [unsupported-operation]
    val = kwargs["val"]
    validate_param(val, "val")
    for shard in sharded_tensor.local_shards():
        torch.nn.init.constant_(shard.tensor, val=val)
    return sharded_tensor


tensor_like_creation_op_map = {
    torch.full_like: sharded_tensor.full,
    torch.empty_like: sharded_tensor.empty,
    torch.zeros_like: sharded_tensor.zeros,
    torch.ones_like: sharded_tensor.ones,
    torch.rand_like: sharded_tensor.rand,
    torch.randn_like: sharded_tensor.randn,
}


# tensor ops that behave the same as the default tensor
def register_tensor_creation_op(op):
    # [FUNCTION DOC] HELPER: Register a handler for a tensor-like creation op that builds a new ShardedTensor using the matching sharded creation function, the input's sharding spec and size, plus remaining positional and keyword arguments.
    @_sharded_op_impl(op)
    def tensor_creation_op(types, args=(), kwargs=None, pg=None):
        # [FUNCTION DOC] HELPER: Handler body: looks up the creation function for the op (RuntimeError if unsupported), defaults kwargs, and returns the newly created ShardedTensor.
        """
        Handles ``__torch_function__`` dispatch for tensor creation ops that
        takes a ShardedTensor as argument, such as ``torch.zeros_like`` or
        ``torch.full_like``.
        """
        creation_op = tensor_like_creation_op_map.get(op)
        if creation_op is None:
            raise RuntimeError(f"Tensor creation {op} not supported!")
        if kwargs is None:
            kwargs = {}

        # pyrefly: ignore [bad-index]
        st = args[0]

        new_st = creation_op(st.sharding_spec(), st.size(), *args[1:], **kwargs)  # type: ignore[operator]
        return new_st


register_tensor_creation_op(torch.full_like)
register_tensor_creation_op(torch.empty_like)
register_tensor_creation_op(torch.zeros_like)
register_tensor_creation_op(torch.ones_like)
register_tensor_creation_op(torch.rand_like)
register_tensor_creation_op(torch.randn_like)
