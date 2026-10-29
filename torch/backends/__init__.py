# mypy: allow-untyped-defs
# [PACKAGE DOC] torch.backends: per-backend configuration flags (cpu, cuda, cudnn, mkl, mkldnn, mps, etc.)
# plus a generic fp32_precision property. Defines a global switch that can forbid bare assignment of
# flags (disable_global_flags), the flags() context manager, and helper classes for property-style
# flags. Import-time side effect: replaces this module in sys.modules with a GenericModule wrapper,
# then imports all backend submodules.
import sys
import types
from contextlib import contextmanager

import torch


# The idea for this parameter is that we forbid bare assignment
# to torch.backends.<cudnn|mkldnn>.enabled and friends when running our
# test suite, where it's very easy to forget to undo the change
# later.
__allow_nonbracketed_mutation_flag = True


def disable_global_flags():
    # [FUNCTION DOC] Turns off bare assignment to backend flags for the rest of the process (mutates a module global).
    global __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = False


def flags_frozen():
    # [FUNCTION DOC] Returns True if bare assignment to backend flags is currently forbidden.
    return not __allow_nonbracketed_mutation_flag


@contextmanager
def __allow_nonbracketed_mutation():
    # [FUNCTION DOC] HELPER: Context manager that temporarily permits flag assignment, restoring the prior setting on exit (even on error).
    global __allow_nonbracketed_mutation_flag
    old = __allow_nonbracketed_mutation_flag
    __allow_nonbracketed_mutation_flag = True
    try:
        yield
    finally:
        __allow_nonbracketed_mutation_flag = old


class ContextProp:
    # [CLASS DOC] Descriptor binding a flag to a getter and setter callable; reads call the getter, writes call the setter unless flags are frozen.
    def __init__(self, getter, setter):
        # [FUNCTION DOC] Stores the getter and setter callables.
        self.getter = getter
        self.setter = setter

    def __get__(self, obj, objtype):
        # [FUNCTION DOC] Returns the current value by calling the getter; ignores the instance and type.
        return self.getter()

    def __set__(self, obj, val):
        # [FUNCTION DOC] Calls the setter with val unless flags are frozen, in which case raises RuntimeError advising the flags() context manager.
        if not flags_frozen():
            self.setter(val)
        else:
            raise RuntimeError(
                f"not allowed to set {obj.__name__} flags "
                "after disable_global_flags; please use flags() context manager instead"
            )


class PropModule(types.ModuleType):
    # [CLASS DOC] Module subclass wrapping an original module so descriptor properties can live on the subclass; unknown attributes are forwarded to the wrapped module.
    def __init__(self, m, name):
        # [FUNCTION DOC] Creates a module with the given name and keeps the wrapped module m.
        super().__init__(name)
        self.m = m

    def __getattr__(self, attr):
        # [FUNCTION DOC] Fallback lookup: returns the attribute from the wrapped module.
        return self.m.__getattribute__(attr)


class _FP32Precision:
    # [CLASS DOC] HELPER: Attribute proxy for one (backend, op) pair exposing a single fp32_precision attribute backed by C++ getter/setter.
    def __init__(self, backend, op):
        # [FUNCTION DOC] Records the backend and op names.
        self.backend = backend
        self.op = op

    def __setattr__(self, name, value):
        # [FUNCTION DOC] Setting fp32_precision forwards to the C++ setter; backend and op are stored normally; any other name raises AttributeError.
        if name == "fp32_precision":
            torch._C._set_fp32_precision_setter(self.backend, self.op, value)
        elif name in ("backend", "op"):
            super().__setattr__(name, value)
        else:
            raise AttributeError("Unknown attribute " + name)

    def __getattr__(self, name):
        # [FUNCTION DOC] Getting fp32_precision queries the C++ getter; any other name raises AttributeError.
        if name == "fp32_precision":
            return torch._C._get_fp32_precision_getter(self.backend, self.op)
        else:
            raise AttributeError("Unknown attribute " + name)


def set_flags(_fp32_precision="none"):
    # [FUNCTION DOC] Sets the generic fp32 precision (if not None) and returns a one-element tuple holding the previous value.
    orig_flags = (torch._C._get_fp32_precision_getter("generic", "all"),)
    if _fp32_precision is not None:
        torch._C._set_fp32_precision_setter("generic", "all", _fp32_precision)
    return orig_flags


@contextmanager
def flags(fp32_precision="none"):
    # [FUNCTION DOC] Context manager that sets the generic fp32 precision for the block and restores the previous value afterward, bypassing the frozen-flags check.
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(fp32_precision)
    try:
        yield
    finally:
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


def _get_fp32_precision_getter(backend, op):
    # [FUNCTION DOC] HELPER: Returns a zero-argument callable reading the fp32 precision for backend and op.
    def inner():
        # [FUNCTION DOC] HELPER: Reads the precision from C++.
        return torch._C._get_fp32_precision_getter(backend, op)

    return inner


def _set_fp32_precision_setter(backend, op):
    # [FUNCTION DOC] HELPER: Returns a one-argument callable that sets the fp32 precision for backend and op.
    def inner(precision):
        # [FUNCTION DOC] HELPER: Writes the precision to C++.
        return torch._C._set_fp32_precision_setter(backend, op, precision)

    return inner


class GenericModule(PropModule):
    # [CLASS DOC] Replacement for the torch.backends module; adds the generic fp32_precision property.
    fp32_precision = ContextProp(
        _get_fp32_precision_getter("generic", "all"),
        _set_fp32_precision_setter("generic", "all"),
    )


sys.modules[__name__] = GenericModule(sys.modules[__name__], __name__)

from torch.backends import (
    cpu as cpu,
    cuda as cuda,
    cudnn as cudnn,
    cusparselt as cusparselt,
    kleidiai as kleidiai,
    mha as mha,
    miopen as miopen,
    mkl as mkl,
    mkldnn as mkldnn,
    mps as mps,
    nnpack as nnpack,
    openmp as openmp,
    opt_einsum as opt_einsum,
    python_native as python_native,
    quantized as quantized,
)
