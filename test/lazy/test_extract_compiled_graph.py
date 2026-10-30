# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for extracting a reusable compiled graph from an FX-traced module on the lazy
# TorchScript backend. Import-time side effect: initializes the TS lazy backend before the
# remaining imports. Defines small modules (scale, subtract, addcmul, multi-return, duplicate
# return, in-place update), context managers, a verification routine and generated test methods.
# The file is not run by default; running it directly raises an error saying it must be enabled in
# the test discovery list first.

import unittest

from torch._lazy.ts_backend import init as init_ts_backend


init_ts_backend()
import copy
import dis
import inspect
import re
from contextlib import contextmanager

import torch
from torch import fx, nn
from torch._lazy import config
from torch._lazy.extract_compiled_graph import extract_compiled_graph


class ModuleConstScale(nn.Module):
    # [CLASS DOC] HELPER: Module whose forward multiplies its input by the constant 2.
    def forward(self, a):
        # [FUNCTION DOC] HELPER: Returns the input times 2.
        return a * 2


class ModuleSub(nn.Module):
    # [CLASS DOC] HELPER: Module whose forward subtracts the second input from the first.
    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: Returns the first input minus the second.
        return a - b


class ModuleAddcmul(nn.Module):
    # [CLASS DOC] HELPER: Module using addcmul with a scalar value argument, which produces a
    # special lazy data holding a scalar rather than a tensor.
    """
    addcmul function takes a at::Scalar which results in a special TSData containing a Scalar rather than a Tensor.
    """

    def forward(self, a, b, c):
        # [FUNCTION DOC] HELPER: Returns addcmul of the three inputs with value 5.
        return torch.addcmul(a, b, c, value=5)


class ModuleReturnMulti(nn.Module):
    # [CLASS DOC] HELPER: Module returning two outputs in swapped order, each modified.
    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: Returns a tuple of the second input plus 1 and the first input
        # minus 1.
        return (b + 1, a - 1)


# The default fx tracer will convert torch.randn to a constant.. We may need
# a custom tracer.
# class ModuleEagerTensor(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self, a):
#         b = torch.randn(2, 3, device="cpu") # eager device
#         return a + b

# The module was planned to cover the case that a Fx graph return an eager
# tensor on the default device. It's harder than ModuleEagerTensor because
# we can not just override the device argument to Lazy since there is no
# explicit device argument.
#
# Unfortunately, the default fx tracer convert the return value of the forward
# method to a constant.. Comment out for now
# class ModuleReturnEagerTensorOnDefaultDevice(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#
#     def forward(self):
#         return torch.tensor((2, 3), dtype=torch.float32)


class ModuleReturnDupTensor(nn.Module):
    # [CLASS DOC] HELPER: Module whose forward returns a tuple in which the same computed tensor
    # appears twice, covering duplicated outputs.
    """
    Handle the corner case that the same tensor appears multiple times in the
    returned tuple. torchbench like drq will hit this corner case when running
    thru torchdynamo..
    """

    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: Returns a tuple of difference, sum, first input plus 1, and the
        # sum again.
        c = a + b
        return a - b, c, a + 1, c


class ModuleInplaceUpdate(nn.Module):
    # [CLASS DOC] HELPER: Module that updates its first argument in place and returns two tensors.
    def forward(self, a, b):
        # [FUNCTION DOC] HELPER: Subtracts the second input from the first in place, then returns a
        # tuple of the second input minus 1 and plus 1.
        a.sub_(b)
        return b - 1, b + 1


@contextmanager
def force_fallback_ctx_mgr(fallback_op):
    # [FUNCTION DOC] HELPER: Context manager that sets the lazy force-fallback config to the given
    # operator name for the duration of the block and restores the previous value on exit, even if
    # the block raises. Yields nothing useful.
    oldconfig = config.get_force_fallback()
    config.set_force_fallback(fallback_op)
    try:
        yield None
    finally:
        config.set_force_fallback(oldconfig)


@contextmanager
def nop_ctx_mgr():
    # [FUNCTION DOC] HELPER: Context manager that does nothing; used when no fallback is forced.
    try:
        yield None
    finally:
        pass


def gen_rand_args(mod):
    # [FUNCTION DOC] HELPER: Returns a list with one random 2 by 3 tensor for every parameter of
    # the module's forward method, as found by signature inspection.
    args = []
    for _ in range(len(inspect.signature(mod.forward).parameters)):
        args.append(torch.randn(2, 3))
    return args


def allclose(expected, actual):
    # [FUNCTION DOC] HELPER: Compares two results. A list or tuple of length one is first replaced
    # by its element. Two tensors are compared with allclose; two lists or tuples are equal if they
    # have the same length and all pairs are close. Returns a bool; raises RuntimeError for any
    # other combination of types.
    def unwrap(cont):
        # [FUNCTION DOC] HELPER: Returns the sole element of a one-element list or tuple, otherwise
        # the value unchanged.
        if isinstance(cont, (list, tuple)) and len(cont) == 1:
            return cont[0]
        return cont

    expected = unwrap(expected)
    actual = unwrap(actual)

    if isinstance(expected, torch.Tensor) and isinstance(actual, torch.Tensor):
        return torch.allclose(expected, actual)
    elif isinstance(expected, (tuple, list)) and isinstance(actual, (tuple, list)):
        return len(expected) == len(actual) and all(
            torch.allclose(a, b) for a, b in zip(expected, actual)
        )
    else:
        raise RuntimeError("Unexpected types")


def verify_reusing_compiled_graph(mod, exception_msg_pattern, ncase=10):
    # [FUNCTION DOC] HELPER: Core check. Generates random args for the module, runs it eagerly,
    # prints its bytecode, then traces it with the FX tracer and extracts a compiled graph. If
    # extraction raises a runtime error: re-raise when no pattern is expected, return quietly if
    # the message matches the pattern, else raise a runtime error about the mismatch. If extraction
    # succeeds but an error was expected, raise. Otherwise print the optimized output and compare
    # it against eager results on ncase random inputs, also comparing the (possibly in-place
    # updated) arguments; raises RuntimeError reporting how many cases failed.
    args = gen_rand_args(mod)
    mod(*args)

    dis.dis(mod.forward)

    try:
        optimized_mod = extract_compiled_graph(fx.symbolic_trace(mod), args)
    except RuntimeError as e:
        if exception_msg_pattern is None:
            raise e  # reraise the exception
        exception_message = str(e)
        if not re.search(exception_msg_pattern, exception_message):
            raise RuntimeError(
                f"Exception message does not match the required pattern: {exception_message}"
            ) from e
        else:
            # We are done for the test case that expects an exception
            return

    if exception_msg_pattern is not None:
        raise RuntimeError(
            f"Expect an exception matching pattern {exception_msg_pattern}"
        )
    print("return value of optimized_mod", optimized_mod(*args))

    # check correctness
    failed_index = []
    for i in range(ncase):
        rand_args = gen_rand_args(mod)
        rand_args_copy = copy.deepcopy(rand_args)
        expected = mod(*rand_args)
        actual = optimized_mod(*rand_args_copy)

        if not allclose(expected, actual):
            print(f"Incorrect results. expected {expected}, actual {actual}")
            failed_index.append(i)
            continue

        # make sure arguments match after calling the model forward method to handle inplace
        # updates.
        if not allclose(rand_args, rand_args_copy):
            print(
                f"Incorrect updated arguments. expected {rand_args}, actual {rand_args_copy}"
            )
            failed_index.append(i)
            continue

    if len(failed_index) > 0:
        raise RuntimeError(f"Failed {len(failed_index)}/{ncase} cases")


def maketest(module_cls, exception_msg_pattern=None, ctxmgr=None):
    # [FUNCTION DOC] HELPER: Test factory. Returns a test method that runs the verification for a
    # fresh instance of the given module class with the optional expected-error pattern, inside the
    # given context manager (a no-op one if none given).
    def wrapper(self):
        # [FUNCTION DOC] HELPER: Test method body. On first call, replaces a missing context
        # manager in the enclosing scope with a no-op one (the choice is kept for later calls),
        # then verifies the module under that context manager.
        nonlocal ctxmgr
        if not ctxmgr:
            ctxmgr = nop_ctx_mgr()
        with ctxmgr:
            verify_reusing_compiled_graph(module_cls(), exception_msg_pattern)

    return wrapper


class OptimizeTest(unittest.TestCase):
    # [CLASS DOC] Test case whose test methods are generated with the factory: plain
    # subtraction, subtraction with the sub operator forced to fall back (expects a fallback
    # error mentioning that operator), constant scale, addcmul, multiple returns, duplicated
    # returns, and in-place update.
    test_sub = maketest(ModuleSub)
    # Same as test_sub but force aten::sub to fallback
    # We expect an exception caught because of LTC fallback.
    test_ltc_fallback = maketest(
        ModuleSub,
        exception_msg_pattern="fallback.*aten::sub",
        ctxmgr=force_fallback_ctx_mgr("aten::sub"),
    )
    test_const_scale = maketest(ModuleConstScale)
    test_addcmul = maketest(ModuleAddcmul)
    test_return_multi = maketest(ModuleReturnMulti)
    test_return_dup_tensor = maketest(ModuleReturnDupTensor)
    test_inplace_update = maketest(ModuleInplaceUpdate)


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
