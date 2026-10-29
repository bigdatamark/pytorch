# [PACKAGE DOC] Sharded optimizer support: exports ShardedOptimizer (from api, via import) and defines named_params_with_sharded_tensor, a helper that yields
# parameters including ShardedTensor members so they can be handed to ShardedOptimizer.
from collections.abc import Iterator
from typing import Union

import torch.nn as nn
from torch.distributed._shard.sharded_tensor import ShardedTensor

from .api import ShardedOptimizer


def named_params_with_sharded_tensor(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
) -> Iterator[tuple[str, nn.Parameter | ShardedTensor]]:
    # [FUNCTION DOC] Generator over (name, parameter) pairs for a module. First yields every ShardedTensor attribute found in the instance dictionaries of the module
    # (and its submodules when recurse is true, using prefix for names), each only once; then yields all regular parameters from named_parameters.
    # Note the prefix and recurse arguments only affect the first phase. Names of sharded tensors are joined with a dot.
    r"""Returns an iterator over module parameters (together with the
    ShardedTensor parameters), yielding both the name of the parameter
    as well as the parameter itself. This is typically passed to a
    :class:torch.distributed._shard.sharded_optim.ShardedOptimizer

    Args:
        prefix (str): prefix to prepend to all parameter names.
        recurse (bool): if True, then yields parameters of this module
            and all submodules. Otherwise, yields only parameters that
            are direct members of this module.

    Yields:
        (str, Union[Tensor, ShardedTensor]): Tuple containing
            the name and parameter (or ShardedTensor parameter)

    Example::

        >>> # xdoctest: +SKIP
        >>> model = torch.nn.Linear(*linear_size)
        >>> shard_parameter(model, "weight", spec)
        >>> for name, param in named_params_with_sharded_tensor(model):
        >>>    if name in ['weight']:
        >>>        print(param.size())

    """
    modules = module.named_modules(prefix=prefix) if recurse else [(prefix, module)]

    memo = set()
    for mod_prefix, mod in modules:
        # find all sharded tensor params
        for name, val in vars(mod).items():
            if isinstance(val, ShardedTensor) and val not in memo:
                memo.add(val)
                name = mod_prefix + ("." if mod_prefix else "") + name
                yield name, val

    # find all nn.Parameters
    for name, val in module.named_parameters():
        yield name, val
