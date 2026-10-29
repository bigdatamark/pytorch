# mypy: allow-untyped-defs
# [PACKAGE DOC]
# Top-level neural network package. Re-exports the parameter/buffer classes, every
# layer class from torch.nn.modules (star import), and exposes the subpackages
# attention, functional, init, modules, parallel, parameter and utils. Also defines
# the helper factory_kwargs used by modules to validate factory keyword arguments.
# Importing this package eagerly imports all those subpackages (no other side effects).
from torch.nn.parameter import (  # usort: skip
    Buffer as Buffer,
    Parameter as Parameter,
    UninitializedBuffer as UninitializedBuffer,
    UninitializedParameter as UninitializedParameter,
)
from torch.nn.modules import *  # usort: skip # noqa: F403
from torch.nn import (
    attention as attention,
    functional as functional,
    init as init,
    modules as modules,
    parallel as parallel,
    parameter as parameter,
    utils as utils,
)
from torch.nn.parallel import DataParallel as DataParallel


def factory_kwargs(kwargs):
    # [FUNCTION DOC]
    # Validate and canonicalize module constructor kwargs into a new mapping usable by
    # tensor factory functions. Input: a mapping (or None, which gives an empty mapping)
    # whose keys may only be device, dtype, memory_format, or factory_kwargs (a nested
    # mapping of explicit factory arguments). Returns a fresh mapping; the input is not
    # mutated. Raises TypeError for unrecognized keys, or when a key appears both at the
    # top level and inside the nested factory_kwargs mapping.
    r"""Return a canonicalized dict of factory kwargs.

    Given kwargs, returns a canonicalized dict of factory kwargs that can be directly passed
    to factory functions like torch.empty, or errors if unrecognized kwargs are present.

    This function makes it simple to write code like this::

        class MyModule(nn.Module):
            def __init__(self, **kwargs):
                factory_kwargs = torch.nn.factory_kwargs(kwargs)
                self.weight = Parameter(torch.empty(10, **factory_kwargs))

    Why should you use this function instead of just passing `kwargs` along directly?

    1. This function does error validation, so if there are unexpected kwargs we will
    immediately report an error, instead of deferring it to the factory call
    2. This function supports a special `factory_kwargs` argument, which can be used to
    explicitly specify a kwarg to be used for factory functions, in the event one of the
    factory kwargs conflicts with an already existing argument in the signature (e.g.
    in the signature ``def f(dtype, **kwargs)``, you can specify ``dtype`` for factory
    functions, as distinct from the dtype argument, by saying
    ``f(dtype1, factory_kwargs={"dtype": dtype2})``)
    """
    if kwargs is None:
        return {}
    simple_keys = {"device", "dtype", "memory_format"}
    expected_keys = simple_keys | {"factory_kwargs"}
    if not kwargs.keys() <= expected_keys:
        raise TypeError(f"unexpected kwargs {kwargs.keys() - expected_keys}")

    # guarantee no input kwargs is untouched
    r = dict(kwargs.get("factory_kwargs", {}))
    for k in simple_keys:
        if k in kwargs:
            if k in r:
                raise TypeError(
                    f"{k} specified twice, in **kwargs and in factory_kwargs"
                )
            r[k] = kwargs[k]

    return r
