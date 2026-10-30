# Owner(s): ["module: nn"]
# [MODULE DOC] Tests for nn.Module hook machinery: per-module forward, forward-pre, full
# backward and backward-pre hooks (ordering, prepend, kwargs variants, always_call, removal),
# state dict pre/post hooks and load_state_dict pre/post hooks (including backward
# compatibility with older pickled modules), module-global hooks, the legacy backward hook, and
# buffer/submodule/parameter registration hooks. Defines small toy models and hook factories
# used with partial binding. Parametrized test classes are instantiated at import time.
import gc
import math
import pickle
import unittest
import warnings
import weakref
from collections import namedtuple, OrderedDict
from copy import deepcopy
from functools import partial
from tempfile import NamedTemporaryFile
from typing import Any

import torch
import torch.nn as nn
from torch.testing._internal.common_nn import _create_basic_net, NNTestCase
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    IS_WINDOWS,
    parametrize as parametrize_test,
    run_tests,
    skipIfTorchDynamo,
    swap,
    TestCase,
)


class Net(nn.Module):
    # [CLASS DOC] HELPER: Small model with two Sequential blocks, each of two 10 by 10 linear
    # layers, chained in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two Sequential blocks seq1 and seq2.
        super().__init__()
        self.seq1 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])
        self.seq2 = nn.Sequential(*[nn.Linear(10, 10) for _ in range(2)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Applies seq1 then seq2 to the input and returns the result.
        return self.seq2(self.seq1(x))


ToyNamedTuple = namedtuple("ToyNamedTuple", "content")


class ToyModel(nn.Module):
    # [CLASS DOC] HELPER: Model of two Net instances in series. Forward returns a one element
    # tuple, or a ToyNamedTuple wrapper if constructed with the named tuple flag, so hooks are
    # exercised with both output container types.
    def __init__(self, with_named_tuple=False) -> None:
        # [FUNCTION DOC] HELPER: Creates net1 and net2 and records whether the output should be
        # a named tuple.
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()
        self.with_named_tuple = with_named_tuple

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Runs net1 then net2 and returns the result wrapped in a named
        # tuple or a plain tuple depending on the stored flag.
        res = self.net2(self.net1(x))
        if self.with_named_tuple:
            return ToyNamedTuple(res)
        else:
            return (res,)


def forward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: tuple[torch.Tensor],
    out: torch.Tensor,
) -> None:
    # [FUNCTION DOC] HELPER: Forward hook body used with partial binding. Records its hook id
    # in the fired list, asserts the module is the expected module and that exactly one input
    # was passed. Returns None.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def forward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    inp: tuple[torch.Tensor],
) -> None:
    # [FUNCTION DOC] HELPER: Forward pre-hook body used with partial binding. Records its hook
    # id, asserts the module is the expected module and that exactly one input was passed.
    # Returns None, so inputs are not changed.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(inp), 1)


def full_backward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: tuple[torch.Tensor],
    grad_output: tuple[torch.Tensor],
) -> None:
    # [FUNCTION DOC] HELPER: Full backward hook body used with partial binding. Records its
    # hook id, asserts the module is the expected one and that grad_input and grad_output each
    # have exactly one element. Returns None.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)
    self.assertEqual(len(grad_output), 1)


def full_backward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    grad_input: tuple[torch.Tensor],
) -> None:
    # [FUNCTION DOC] HELPER: Full backward pre-hook body used with partial binding. Records its
    # hook id, asserts the module is the expected one and that the gradient tuple has exactly
    # one element. Returns None.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(grad_input), 1)


class KwargModel(nn.Module):
    # [CLASS DOC] HELPER: Model with a keyword argument bias used to test hooks registered with
    # the kwargs option. Also has a method usable as a bound forward hook.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates two Net children (unused by forward).
        super().__init__()
        self.net1 = Net()
        self.net2 = Net()

    def forward(self, x: torch.Tensor, bias: torch.Tensor = None) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the input, plus the bias if one was given.
        if bias is not None:
            x = x + bias
        return x

    def internal_forward_hook(
        self,
        module: nn.Module,
        args: tuple[torch.Tensor],
        kwargs: dict[str, Any],
        out: torch.Tensor,
    ):
        # [FUNCTION DOC] HELPER: Bound-method forward hook taking args, kwargs and output;
        # returns the output plus the bias keyword argument.
        return out + kwargs["bias"]


class FailsInForwardModel(nn.Module):
    # [CLASS DOC] HELPER: Model whose forward raises a RuntimeError by default, to test hooks
    # that must still run on failure.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates a Net child.
        super().__init__()
        self.net1 = Net()

    def forward(self, x: torch.Tensor, fail: bool = True) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Raises RuntimeError with the message failing in forward when
        # the fail flag is true (the default); otherwise returns the child net applied to the
        # input.
        if fail:
            raise RuntimeError("failing in forward")
        return self.net1(x)


def kwarg_forward_pre_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: tuple[torch.Tensor],
    kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    # [FUNCTION DOC] HELPER: Forward pre-hook with kwargs support. Records its hook id, asserts
    # the module is the expected one and that one positional argument was passed, doubles the
    # bias keyword value in place, and returns the args and kwargs pair.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)
    kwargs["bias"] = 2 * kwargs["bias"]
    return args, kwargs


def kwarg_forward_hook(
    self: TestCase,
    fired_hooks: list[int],
    expected_module: nn.Module,
    hook_id: int,
    module: nn.Module,
    args: tuple[torch.Tensor],
    kwargs: dict[str, Any],
    out: torch.Tensor,
) -> Any:
    # [FUNCTION DOC] HELPER: Forward hook with kwargs support. Records its hook id, asserts the
    # module is the expected one and one positional argument, and returns the output plus the
    # bias keyword argument.
    fired_hooks.append(hook_id)
    self.assertEqual(id(module), id(expected_module))
    self.assertEqual(len(args), 1)

    out = out + kwargs["bias"]
    return out


class DummyContextManager:
    # [CLASS DOC] HELPER: Trivial context manager recording events into a list: entering
    # appends 2 and exiting appends -1 (the list given at construction).
    def __init__(self, inp):
        # [FUNCTION DOC] HELPER: Stores the list to which events are appended.
        self.input = inp

    def __enter__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Appends the value 2 to the stored list; returns None.
        self.input.append(2)

    def __exit__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Appends the value -1 to the stored list; returns None and does
        # not suppress exceptions.
        self.input.append(-1)


class TestModuleHooks(TestCase):
    # [CLASS DOC] Generic-hardware test case for per-instance module hooks (forward, forward-
    # pre, full backward, backward-pre, kwargs variants and always_call). Parametrized tests
    # are instantiated at import time.
    hw_classification = HardwareClassification.GENERIC

    @parametrize_test("named_tuple", (True, False))
    def test_forward_hooks(self, named_tuple):
        # [FUNCTION DOC] Parametrized by output container type. Registers five forward hooks
        # with some prepended and checks they fire in prepend-aware order, only on forward (not
        # on backward), and fire again on a second forward; output is a named tuple or a tuple
        # as requested.
        fired_hooks: list[int] = []
        model = ToyModel(named_tuple)
        x = torch.randn(10, 10)
        hook = partial(forward_hook, self, fired_hooks, model.net1.seq2)
        model.net1.seq2.register_forward_hook(partial(hook, 0))
        model.net1.seq2.register_forward_hook(partial(hook, 1), prepend=True)
        model.net1.seq2.register_forward_hook(partial(hook, 2))
        model.net1.seq2.register_forward_hook(partial(hook, 3))
        model.net1.seq2.register_forward_hook(partial(hook, 4), prepend=True)
        expected = [4, 1, 0, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        out[0].sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x)[0].sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @parametrize_test("named_tuple", (True, False))
    def test_forward_pre_hooks(self, named_tuple):
        # [FUNCTION DOC] Parametrized by output container type. Registers five forward pre-
        # hooks with some prepended and checks they fire in prepend-aware order on forward only
        # and repeat on the second forward.
        fired_hooks: list[int] = []
        model = ToyModel(named_tuple)
        x = torch.randn(10, 10)
        hook = partial(forward_pre_hook, self, fired_hooks, model.net2.seq1)
        model.net2.seq1.register_forward_pre_hook(partial(hook, 0), prepend=True)
        model.net2.seq1.register_forward_pre_hook(partial(hook, 1))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 2))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 3))
        model.net2.seq1.register_forward_pre_hook(partial(hook, 4), prepend=True)
        expected = [4, 0, 1, 2, 3]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, expected)
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        out[0].sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x)[0].sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @parametrize_test("named_tuple", (True, False))
    def test_full_backward_hooks(self, named_tuple):
        # [FUNCTION DOC] Parametrized by output container type. Registers five full backward
        # hooks with some prepended; none fires during forward, and all fire in prepend-aware
        # order during backward and again on a second backward.
        fired_hooks: list[int] = []
        model = ToyModel(named_tuple)
        x = torch.randn(10, 10)
        hook = partial(full_backward_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_hook(partial(hook, 0))
        model.net1.register_full_backward_hook(partial(hook, 1))
        model.net1.register_full_backward_hook(partial(hook, 2))
        model.net1.register_full_backward_hook(partial(hook, 3), prepend=True)
        model.net1.register_full_backward_hook(partial(hook, 4), prepend=True)
        expected = [4, 3, 0, 1, 2]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        out[0].sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x)[0].sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

    @parametrize_test("named_tuple", (True, False))
    def test_full_backward_pre_hooks(self, named_tuple):
        # [FUNCTION DOC] Parametrized by output container type. Checks full backward pre-hook
        # ordering with prepend (none fire at forward, all fire at backward and repeat), then
        # verifies that a pre-hook returning zeroed gradients zeroes the weight gradient and
        # the input gradient (input gradient is none when the input does not require grad).
        fired_hooks: list[int] = []
        model = ToyModel(named_tuple)
        x = torch.randn(10, 10)
        hook = partial(full_backward_pre_hook, self, fired_hooks, model.net1)
        model.net1.register_full_backward_pre_hook(partial(hook, 0), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 1), prepend=True)
        model.net1.register_full_backward_pre_hook(partial(hook, 2))
        model.net1.register_full_backward_pre_hook(partial(hook, 3))
        model.net1.register_full_backward_pre_hook(partial(hook, 4))
        expected = [1, 0, 2, 3, 4]

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [])
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        out[0].sum().backward()
        self.assertEqual(fired_hooks, expected)
        model(x)[0].sum().backward()
        self.assertEqual(fired_hooks, expected + expected)

        # Backward pre hook can affect subsequent gradient computation
        for rg in [True, False]:
            a = torch.ones(2, requires_grad=rg)
            model = nn.Linear(2, 2)

            def fn(_unused_module, grad_output):
                # [FUNCTION DOC] HELPER: Backward pre-hook returning a one element tuple with
                # the first output gradient multiplied by zero.
                return (grad_output[0] * 0,)

            model.register_full_backward_pre_hook(fn)

            out = model(a)
            out.sum().backward()
            self.assertEqual(model.weight.grad, torch.zeros(2, 2))
            if rg:
                self.assertEqual(a.grad, torch.zeros_like(a))
            else:
                self.assertIsNone(a.grad)

    @parametrize_test("named_tuple", (True, False))
    def test_mixed_hooks(self, named_tuple):
        # [FUNCTION DOC] Parametrized by output container type. Registers one forward pre-hook,
        # forward hook, backward pre-hook and full backward hook on the whole model and checks
        # they fire as 0 and 1 on forward, then 2 and 3 on backward, repeating on the next
        # pass.
        fired_hooks: list[int] = []
        model = ToyModel(named_tuple)
        x = torch.randn(10, 10)
        model.register_forward_pre_hook(
            partial(forward_pre_hook, self, fired_hooks, model, 0)
        )
        model.register_forward_hook(partial(forward_hook, self, fired_hooks, model, 1))
        model.register_full_backward_pre_hook(
            partial(full_backward_pre_hook, self, fired_hooks, model, 2)
        )
        model.register_full_backward_hook(
            partial(full_backward_hook, self, fired_hooks, model, 3)
        )

        self.assertEqual(fired_hooks, [])
        out = model(x)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertIsInstance(out, ToyNamedTuple if named_tuple else tuple)
        out[0].sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3])
        model(x)[0].sum().backward()
        self.assertEqual(fired_hooks, [0, 1, 2, 3, 0, 1, 2, 3])

    def test_kwarg_hooks(self):
        # [FUNCTION DOC] Tests hooks registered with kwargs on a model with a bias keyword: (1)
        # a pre-hook that doubles bias gives input plus twice the bias; (2) pre-hook plus
        # forward hook gives input plus four times the bias with firing order pre then post;
        # (3) a bound method as forward hook adds bias again, giving twice the bias.
        # 1. test forward pre hook
        fired_hooks: list[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # So, out = x + bias * 2
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0])
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)

        # 2. test forward pre and forward hooks
        fired_hooks: list[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )
        model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # forward-post: out = out + bias'
        # So, out = x + bias * 4
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)

        # 3. test nn.Module member method as forward-post hook
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        model.register_forward_hook(model.internal_forward_hook, with_kwargs=True)

        # forward: out = x + bias
        # forward-post: out = out + bias
        # So, out = x + bias * 2
        out = model(x, bias=bias)
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)

    def test_remove_kwarg_hooks(self):
        # [FUNCTION DOC] Registers a kwargs forward hook and pre-hook, checks the output is
        # input plus four times bias, removes the forward hook handle (output becomes input
        # plus twice bias and its id disappears from the kwargs hook table), then removes the
        # pre-hook handle (output is input plus bias and its table entry is gone).
        # test forward pre and forward hooks
        fired_hooks: list[int] = []
        x: torch.Tensor = torch.ones(10, 10)
        bias: torch.Tensor = torch.ones(10, 10)
        model = KwargModel()
        forward_hook_handle = model.register_forward_hook(
            partial(kwarg_forward_hook, self, fired_hooks, model, 1),
            with_kwargs=True,
        )
        forward_pre_hook_handle = model.register_forward_pre_hook(
            partial(kwarg_forward_pre_hook, self, fired_hooks, model, 0),
            with_kwargs=True,
        )

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # forward-post: out = out + bias'
        # So, out = x + bias * 4
        self.assertEqual(fired_hooks, [])
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1])
        self.assertEqual(out, x + 4 * bias, rtol=0, atol=1e-5)

        # forward-pre: bias' = bias * 2
        # forward: out = x + bias'
        # So, out = x + bias * 2
        forward_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)
        self.assertFalse(forward_hook_handle.id in model._forward_hooks_with_kwargs)

        # forward: out = x + bias
        # So, out = x + bias
        forward_pre_hook_handle.remove()
        out = model(x, bias=bias)
        self.assertEqual(fired_hooks, [0, 1, 0])
        self.assertEqual(out, x + bias, rtol=0, atol=1e-5)
        self.assertFalse(
            forward_pre_hook_handle.id in model._forward_pre_hooks_with_kwargs
        )

    def test_always_called_forward_hooks(self):
        # [FUNCTION DOC] Tests always_call forward hooks on a model whose forward can fail,
        # using a recording context manager. Checks the hook runs when forward raises, runs
        # once when forward succeeds, runs when a pre-hook raises, still runs and does not mask
        # the original error when another always_call hook raises, is removed cleanly from the
        # internal always-called table, is not run twice if it fails itself, and that a global
        # always_call forward hook behaves the same and can be removed. Final recorded stack
        # assertions encode the expected enter and exit events.
        x: torch.Tensor = torch.ones(10, 10)
        model = FailsInForwardModel()
        stack = []
        ctx = None

        def setup_context():
            # [FUNCTION DOC] HELPER: Creates a fresh recording context manager on the shared
            # stack and stores it in the enclosing variable.
            nonlocal ctx
            ctx = DummyContextManager(stack)

        def ctx_setup_hook(m, i):
            # [FUNCTION DOC] HELPER: Forward pre-hook that sets up the context and enters it
            # (appends 2 to the stack).
            setup_context()
            ctx.__enter__()

        def ctx_setup_failure_hook(m, i):
            # [FUNCTION DOC] HELPER: Forward pre-hook that enters the context then raises
            # RuntimeError with message failing in ctx setup.
            setup_context()
            ctx.__enter__()
            raise RuntimeError("failing in ctx setup")

        def ctx_shutdown_hook(m, i, o):
            # [FUNCTION DOC] HELPER: Forward hook that exits the context (appends -1 to the
            # stack).
            ctx.__exit__()

        def ctx_shutdown_failure_hook(m, i, o):
            # [FUNCTION DOC] HELPER: Forward hook that exits the context then raises
            # RuntimeError with message failing in ctx shutdown.
            ctx.__exit__()
            raise RuntimeError("failing in ctx shutdown")

        def throw_hook(m, i, o):
            # [FUNCTION DOC] HELPER: Forward hook that always raises RuntimeError with message
            # failing in throw.
            raise RuntimeError("failing in throw")

        forward_pre_hook_handle = model.register_forward_pre_hook(ctx_setup_hook)
        forward_hook_handle = model.register_forward_hook(
            ctx_shutdown_hook, always_call=True
        )
        self.assertTrue(len(model._forward_hooks_always_called) == 1)

        # make sure always_called forward hook runs when model.forward raises RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failing in forward"):
            model(x)
        self.assertEqual(stack, [2, -1])

        # make sure that always_called forward hook does not run twice if there is no error
        model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1])

        # make sure always_called forward hook runs when forward pre hook raises RuntimeError
        forward_pre_hook_handle.remove()
        model.register_forward_pre_hook(ctx_setup_failure_hook)

        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1])

        # make sure always_called hook runs when another always_called forward hook raises an error
        forward_hook_handle2 = model.register_forward_hook(
            throw_hook, prepend=True, always_call=True
        )

        # error raised should not be error of the forced hook
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1])

        # make sure that always called forward hooks are properly removed
        forward_hook_handle.remove()
        forward_hook_handle2.remove()
        self.assertTrue(len(model._forward_hooks_always_called) == 0)

        # make sure that always called forward hook is not run twice if it fails while running
        forward_hook_handle3 = model.register_forward_hook(
            ctx_shutdown_failure_hook, always_call=True
        )
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1])

        forward_hook_handle3.remove()

        global_forward_hook_handle = nn.modules.module.register_module_forward_hook(
            ctx_shutdown_hook, always_call=True
        )
        self.assertTrue(len(nn.modules.module._global_forward_hooks_always_called) == 1)
        # make sure global forward hook runs when forward pre hook raises RuntimeError
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x, fail=False)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1])

        # make sure forced global forward hook is properly removed
        global_forward_hook_handle.remove()
        self.assertTrue(len(nn.modules.module._global_forward_hooks_always_called) == 0)
        with self.assertRaisesRegex(RuntimeError, "failing in ctx setup"):
            model(x)
        self.assertEqual(stack, [2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2, -1, 2])

    def test_bw_hook_warning_for_non_tensor_or_tuple(self):
        # [FUNCTION DOC] Registers forward pre-hook, forward hook, backward pre-hook and full
        # backward hook on a module whose output is a dict, not a tensor or tuple. Checks both
        # forward hooks ran, the backward hooks never ran, and exactly one warning was recorded
        # saying the result should be a tensor or tuple of tensors.
        # Test to verify that backward hook raises warning
        # if result is not a Tensor or tuple of Tensors.
        counter = {"forward": 0, "backward": 0}

        def fw_pre_hook(module: nn.Module, _inputs):
            # [FUNCTION DOC] HELPER: Forward pre-hook counting a forward call in the shared
            # counter.
            counter["forward"] += 1

        def fw_hook(module: nn.Module, _inputs, _outputs):
            # [FUNCTION DOC] HELPER: Forward hook counting a forward call in the shared
            # counter.
            counter["forward"] += 1

        def bw_hook(module: nn.Module, _inputs, _outputs):
            # [FUNCTION DOC] HELPER: Backward hook counting a backward call in the shared
            # counter.
            counter["backward"] += 1

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes a dict, applies softmax over
            # dimension 0 to the entry x, and returns a dict (a non tensor output).
            def forward(self, dict):
                # [FUNCTION DOC] HELPER: Returns a dict with softmax of the x entry of the
                # input dict.
                inp = dict["x"]
                x = torch.nn.functional.softmax(inp, dim=0)
                return {"x": x}

        x = torch.ones(2, requires_grad=True)
        model = TestModule()
        model.register_forward_pre_hook(fw_pre_hook)
        model.register_forward_hook(fw_hook)
        model.register_full_backward_pre_hook(bw_hook)
        model.register_full_backward_hook(bw_hook)

        with warnings.catch_warnings(record=True) as w:
            y = model({"x": x})["x"]
            loss = y.sum()
            loss.backward()

        self.assertEqual(counter["forward"], 2)
        self.assertEqual(counter["backward"], 0)
        self.assertEqual(len(w), 1)
        self.assertTrue("should be a Tensor or a tuple of Tensors" in str(w[0].message))


def _hook_to_pickle(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Module level no-op load-state-dict hook (picklable by reference);
    # accepts any arguments and returns None.
    pass


class TestStateDictHooks(TestCase):
    # [CLASS DOC] Generic-hardware test case for state dict hooks: load_state_dict pre and post
    # hooks, state_dict pre and post hooks, reference and pickling behavior, and backward
    # compatibility with modules saved without the hook tables.
    hw_classification = HardwareClassification.GENERIC

    @swap([True, False])
    def test_load_state_dict_pre_hook(self):
        # [FUNCTION DOC] Checks load_state_dict pre-hooks: the private registration (without
        # module argument), the public registration (with module argument) and the private
        # registration with module flag true each fire; the hook count accumulates 1, then 2,
        # then 3 across successive loads as hooks accumulate.
        m = nn.Linear(10, 10)
        m_state_dict = m.state_dict()

        m_load = nn.Linear(10, 10)

        hook_called = 0

        def hook_without_module(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # [FUNCTION DOC] HELPER: Pre-hook without module parameter; asserts the state dict
            # equals the saved one and increments the shared counter.
            self.assertEqual(m_state_dict, state_dict)
            nonlocal hook_called
            hook_called += 1

        def hook_with_module(
            module,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            # [FUNCTION DOC] HELPER: Pre-hook with module parameter; asserts the state dict
            # equals the saved one, the module is the target module, and increments the shared
            # counter.
            self.assertEqual(m_state_dict, state_dict)
            self.assertTrue(m_load is module)
            nonlocal hook_called
            hook_called += 1

        hook_called = 0
        # Test private API since this sets with_module=False which diverges from public API
        m_load._register_load_state_dict_pre_hook(hook_without_module)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(1, hook_called)

        hook_called = 0
        m_load.register_load_state_dict_pre_hook(hook_with_module)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(2, hook_called)

        # Test private API with with_module=True
        hook_called = 0
        m_load._register_load_state_dict_pre_hook(hook_with_module, True)
        m_load.load_state_dict(m_state_dict)
        self.assertEqual(3, hook_called)

    def test_no_extra_ref_to_module(self):
        # [FUNCTION DOC] Registers a load_state_dict pre-hook on a module, deletes the module
        # with garbage collection disabled, and checks a weak reference is dead, i.e.
        # registration holds no strong reference cycle. Re-enables garbage collection in a
        # finally block.
        try:
            gc.disable()
            m = nn.Linear(10, 10)

            m.register_load_state_dict_pre_hook(_hook_to_pickle)
            weak_m = weakref.ref(m)
            del m

            self.assertEqual(weak_m(), None)
        finally:
            gc.enable()

    def test_pickled_hook(self):
        # [FUNCTION DOC] Registers a module-level function as load_state_dict pre-hook and
        # checks the module can be pickled and unpickled without error.
        m = nn.Linear(10, 10)
        m.register_load_state_dict_pre_hook(_hook_to_pickle)
        pickle.loads(pickle.dumps(m))

    @swap([True, False])
    def test_load_state_dict_module_pre_hook(self):
        # [FUNCTION DOC] Checks load_state_dict pre-hooks that are bound methods of the module,
        # run in both swap modes, for the module itself and nested in a container. The first
        # hook (private registration) is called once; adding the module-taking hook gives two
        # calls. Both hooks assert empty error, missing and unexpected lists and strict being
        # true.
        hook_called = 0

        # Test with module instance method as hook
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter and two bound-method load pre-hooks
            # that assert error lists are empty and strict is set, and count calls through the
            # shared counter; the second also asserts the module argument is self.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 10 element random parameter foo.
                super().__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_pre_load_hook(
                self,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # [FUNCTION DOC] HELPER: Load pre-hook without module argument; raises
                # AssertionError if error messages, unexpected keys or missing keys are non-
                # empty or if strict is false; otherwise increments the shared counter.
                if error_msgs != []:
                    raise AssertionError(f"Expected empty error_msgs, got {error_msgs}")
                if unexpected_keys != []:
                    raise AssertionError(
                        f"Expected empty unexpected_keys, got {unexpected_keys}"
                    )
                if missing_keys != []:
                    raise AssertionError(
                        f"Expected empty missing_keys, got {missing_keys}"
                    )
                if not strict:
                    raise AssertionError("Expected strict to be True")
                nonlocal hook_called
                hook_called += 1

            def my_pre_load_hook_with_module(
                self,
                module,
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                # [FUNCTION DOC] HELPER: Load pre-hook with module argument; same assertions as
                # the module-less one plus raising AssertionError when the module argument is
                # not self; increments the shared counter.
                if error_msgs != []:
                    raise AssertionError(f"Expected empty error_msgs, got {error_msgs}")
                if unexpected_keys != []:
                    raise AssertionError(
                        f"Expected empty unexpected_keys, got {unexpected_keys}"
                    )
                if missing_keys != []:
                    raise AssertionError(
                        f"Expected empty missing_keys, got {missing_keys}"
                    )
                if not strict:
                    raise AssertionError("Expected strict to be True")
                if self is not module:
                    raise AssertionError("Expected self is module")
                nonlocal hook_called
                hook_called += 1

        # Test that hooks registered on a submodule are also called
        # appropriately, i.e. with the submodule as module argument in
        # my_pre_load_hook_with_module.
        class MyModuleContainer(nn.Module):
            # [CLASS DOC] HELPER: Wrapper module holding the given module as the child mod,
            # used to check hooks registered on submodules.
            def __init__(self, mod):
                # [FUNCTION DOC] HELPER: Stores the given module as the child mod.
                super().__init__()
                self.mod = mod

        for ctor in [MyModuleContainer, lambda x: x]:
            m = ctor(MyModule())
            state_dict = m.state_dict()
            if isinstance(m, MyModuleContainer):
                mod = m.mod
            else:
                mod = m

            hook_called = 0
            # Test private API since this sets with_module=False which diverges from public API
            mod._register_load_state_dict_pre_hook(mod.my_pre_load_hook)
            m.load_state_dict(state_dict)
            self.assertEqual(1, hook_called)

            hook_called = 0
            mod.register_load_state_dict_pre_hook(mod.my_pre_load_hook_with_module)
            m.load_state_dict(state_dict)
            self.assertEqual(2, hook_called)

    @swap([True, False])
    def test_load_state_dict_post_hook(self):
        # [FUNCTION DOC] Checks load_state_dict post-hooks on a module nested in a ModuleList:
        # the hook is called even when wrapped, its additions to missing and unexpected keys
        # appear in the non-strict result and in the strict error message, handle removal stops
        # it from firing, and a second hook that clears both key lists makes a strict load with
        # an extra key succeed.
        hook_called = 0

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with one parameter and a post-load hook that checks it
            # received itself, appends fake missing key foo and unexpected key bar, and counts
            # calls.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a 10 element random parameter foo.
                super().__init__()
                self.foo = torch.nn.Parameter(torch.rand(10))

            def my_post_load_hook(self, module, incompatible_keys):
                # [FUNCTION DOC] HELPER: Post-load hook that raises AssertionError if the
                # module argument is not self, appends a missing key and an unexpected key to
                # the incompatible keys, and increments the shared counter.
                if module is not self:
                    raise AssertionError("Expected module is self")
                nonlocal hook_called
                incompatible_keys.missing_keys.append("foo")
                incompatible_keys.unexpected_keys.append("bar")
                hook_called += 1

        nested = MyModule()
        wrapped = nn.ModuleList([nested])
        handle = nested.register_load_state_dict_post_hook(
            nested.my_post_load_hook,
        )
        # Hook must be called even if it is wrapped
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(hook_called, 1)
        # Ensure that the hook modified missing_keys and unexpected_keys
        missing = ret.missing_keys
        unexpected = ret.unexpected_keys
        self.assertEqual(missing, ["foo"])
        self.assertEqual(unexpected, ["bar"])
        # When called with strict=True, the error raised should mention the
        # missing and unexpected keys the hook added.
        with self.assertRaisesRegex(RuntimeError, "foo.*\n.*bar"):
            wrapped.load_state_dict(wrapped.state_dict(), strict=True)
        self.assertEqual(hook_called, 2)
        # Removing the hook via handle.remove() should cause it not to
        # fire anymore.
        handle.remove()
        # Hook did not run so it should not have added any keys
        ret = wrapped.load_state_dict(wrapped.state_dict(), strict=False)
        self.assertEqual(ret.missing_keys, [])
        self.assertEqual(ret.unexpected_keys, [])
        # hook_called should not have been incremented
        self.assertEqual(hook_called, 2)

        def load_hook_clear_incompatible(module, incompatible_keys):
            # [FUNCTION DOC] HELPER: Post-load hook that empties both the missing and
            # unexpected key lists.
            incompatible_keys.missing_keys.clear()
            incompatible_keys.unexpected_keys.clear()

        nested.register_load_state_dict_post_hook(load_hook_clear_incompatible)
        state_dict = wrapped.state_dict()
        state_dict["extra"] = torch.ones(1)
        # load state_dict with strict=True should not throw.
        ret = wrapped.load_state_dict(state_dict, strict=True)
        # explicitly ensure that the post hook clearned out incompatible_keys
        self.assertEqual([], ret.missing_keys)
        self.assertEqual([], ret.unexpected_keys)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    @swap([True, False])
    def test_load_state_dict_post_hook_backward_compatibility(self):
        # [FUNCTION DOC] Backward compatibility check for load post-hooks (skipped on Windows,
        # run in both swap modes): for Softmin, Softmax and LogSoftmax, deletes the post-hook
        # table to simulate an old module, saves and reloads it through a temporary file,
        # checks load_state_dict still works without calling hooks, then registers a hook and
        # checks it fires.
        def my_post_load_hook(mod, _):
            # [FUNCTION DOC] HELPER: Post-load hook that sets the enclosing called flag to
            # true.
            nonlocal called
            called = True

        for m in [nn.Softmin(10), nn.Softmax(10), nn.LogSoftmax(10)]:
            called = False
            sd = deepcopy(m.state_dict())
            self.assertTrue(hasattr(m, "_load_state_dict_post_hooks"))
            # Simulate an older model that did not have this attr
            delattr(m, "_load_state_dict_post_hooks")
            # Save and load, and ensure that load_state_dict works (without proper
            # BC we would run into errors because this attribute would be expected).
            # In particular, Softmax runs into the issue described here:
            # https://github.com/pytorch/pytorch/issues/77280
            with NamedTemporaryFile() as f:
                # Note that torch.save / torch.load is not recommended to save/load
                # modules.
                torch.save(m, f.name)
                # weights_only=False as this is legacy code that saves the model
                m = torch.load(f.name, weights_only=False)
                m.load_state_dict(sd)
                self.assertFalse(called)

            # Ensure hooks can be registered and called.
            m.register_load_state_dict_post_hook(my_post_load_hook)
            m.load_state_dict(sd)
            self.assertTrue(called)

    def _test_register_state_dict_pre_hook(self, model, submodule):
        # [FUNCTION DOC] HELPER: Shared check for state_dict pre-hooks. Registers a pre-hook on
        # the model and on a submodule, then calls state_dict with prefix foo. and keep_vars
        # both true and false, asserting each call runs the hook twice, receives the keep_vars
        # setting and a prefix starting with foo.; repeats after a forward pass.
        _state_dict_prefix = "foo."
        state_dict_pre_hook_count = 0
        keep_var_setting = False

        def my_state_dict_pre_hook(module, prefix, keep_vars):
            # [FUNCTION DOC] HELPER: State dict pre-hook that asserts keep_vars equals the
            # current setting and the prefix starts with the shared prefix, and increments the
            # call counter.
            self.assertEqual(keep_vars, keep_var_setting)
            nonlocal state_dict_pre_hook_count
            state_dict_pre_hook_count += 1
            self.assertTrue(prefix.startswith(_state_dict_prefix))

        model.register_state_dict_pre_hook(my_state_dict_pre_hook)
        # Test to ensure submodules run the hook as well.
        submodule.register_state_dict_pre_hook(my_state_dict_pre_hook)

        def check_results(model):
            # [FUNCTION DOC] HELPER: Runs state_dict for both keep_vars settings and asserts
            # the hook ran twice per call, resetting the counter each time. The argument is the
            # model.
            nonlocal state_dict_pre_hook_count, keep_var_setting
            for keep_var_setting in [True, False]:
                _ = model.state_dict(
                    prefix=_state_dict_prefix, keep_vars=keep_var_setting
                )
                self.assertEqual(2, state_dict_pre_hook_count)
                state_dict_pre_hook_count = 0

        # Test state dict works as expected after model construction
        check_results(model)
        # Test state dict works as expected after forward
        model(torch.ones(10, 3))
        check_results(model)

    def test_register_state_dict_pre_hook(self):
        # [FUNCTION DOC] Builds a model with a three layer Sequential and runs the shared
        # state_dict pre-hook check with the Sequential as the submodule.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model wrapping a three layer Sequential of 3 by 3 linear
            # layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the Sequential a.
                super().__init__()
                self.a = nn.Sequential(
                    nn.Linear(3, 3), nn.Linear(3, 3), nn.Linear(3, 3)
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the Sequential to the input.
                return self.a(x)

        mod = MyModule()
        self._test_register_state_dict_pre_hook(mod, mod.a)

    def test_register_state_dict_pre_hook_lazy_module(self):
        # [FUNCTION DOC] Runs the shared state_dict pre-hook check on a model with two lazy
        # linear layers, using the first as submodule (the check does a forward pass that
        # materializes them).
        class MyLazyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with two lazy linear layers of output sizes 8 and 5.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates layer1 and layer2.
                super().__init__()
                self.layer1 = nn.LazyLinear(8)
                self.layer2 = nn.LazyLinear(5)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies layer1 then layer2.
                return self.layer2(self.layer1(x))

        mod = MyLazyModule()
        self._test_register_state_dict_pre_hook(mod, mod.layer1)

    @unittest.skipIf(IS_WINDOWS, "Tempfile permission issue on windows")
    def test_register_state_dict_pre_hook_backward_compat(self):
        # [FUNCTION DOC] Backward compatibility (skipped on Windows): deletes the state dict
        # pre-hook table from a Linear to mimic an old module, saves and reloads via a
        # temporary file, checks state_dict works without calling hooks, then checks a newly
        # registered pre-hook is called.
        called = False

        def my_state_dict_pre_hook(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Pre-hook accepting any arguments that sets the enclosing
            # called flag.
            nonlocal called
            called = True

        m = nn.Linear(1, 1)
        self.assertTrue(hasattr(m, "_state_dict_pre_hooks"))
        delattr(m, "_state_dict_pre_hooks")
        # Save and load, ensure we can still call state_dict
        # without running into issues.
        with NamedTemporaryFile() as f:
            # Note that torch.save / torch.load is not recommended
            # to save / load modules.
            torch.save(m, f.name)
            # weights_only=False as this is legacy code that saves the model
            m = torch.load(f.name, weights_only=False)

        # Ensure we can run state_dict without issues
        _ = m.state_dict()
        self.assertFalse(called)
        m.register_state_dict_pre_hook(my_state_dict_pre_hook)
        _ = m.state_dict()
        self.assertTrue(called)

    @parametrize_test("private", [True, False])
    def test_register_state_dict_post_hook(self, private):
        # [FUNCTION DOC] Parametrized by private versus public registration. Registers a post-
        # hook that wraps entries in Parameters on every linear layer of a transformer and
        # checks only linear and attention output projection entries become Parameters. Then
        # registers on the root a hook returning an empty dict: with the private API the state
        # dict becomes empty; with the public API the hook is marked as public, state_dict
        # raises a RuntimeError that post-hooks must return None, and registering the same
        # function via the private API raises a RuntimeError.
        m = nn.Transformer(
            d_model=4, nhead=2, num_encoder_layers=2, num_decoder_layers=2
        )

        def linear_state_dict_post_hook(module, state_dict, prefix, local_metadata):
            # [FUNCTION DOC] HELPER: State dict post-hook that replaces each own parameter
            # entry (non recursive) under the prefix with a Parameter wrapping it; modifies the
            # dict in place.
            for name, _param in module.named_parameters(recurse=False):
                state_dict[prefix + name] = torch.nn.Parameter(
                    state_dict[prefix + name]
                )

        def register_linear_hook(module):
            # [FUNCTION DOC] HELPER: If the module is a Linear, registers the linear post-hook
            # using the private or public registration function according to the outer flag.
            if isinstance(module, nn.Linear):
                hook_registration_fn = (
                    module._register_state_dict_hook
                    if private
                    else module.register_state_dict_post_hook
                )
                hook_registration_fn(linear_state_dict_post_hook)

        def _check_sd(state_dict):
            # [FUNCTION DOC] HELPER: Asserts that entries of the model's state dict with linear
            # or out_proj in the key are Parameters and others are not.
            # KNOWN ISSUE: the state_dict argument is ignored; the check re-reads the state dict
            #    from the enclosing model.
            for k, v in m.state_dict().items():
                if "linear" in k or "out_proj" in k:
                    self.assertTrue(isinstance(v, torch.nn.Parameter))
                else:
                    self.assertFalse(isinstance(v, torch.nn.Parameter))

        # verify that return type of hook registered on child submodules has no effect
        # regardless of whether using public or private API
        m.apply(register_linear_hook)
        _check_sd(m.state_dict())

        # verify that return type of hook registered root module has no effect
        # for public API but has effect for private API
        hook_registration_fn = (
            m._register_state_dict_hook if private else m.register_state_dict_post_hook
        )

        def fn(m, s, p, l):
            # [FUNCTION DOC] HELPER: Post-hook that ignores its arguments and returns an empty
            # ordered dict (the invalid or replacing return used by the test).
            return OrderedDict()

        hook_registration_fn(fn)
        if private:
            self.assertFalse(hasattr(fn, "_from_public_api"))
            self.assertTrue(len(m.state_dict()) == 0)
        else:
            self.assertTrue(hasattr(fn, "_from_public_api"))
            with self.assertRaisesRegex(
                RuntimeError, "state_dict post-hook must return None"
            ):
                m.state_dict()
            with self.assertRaisesRegex(
                RuntimeError, "previously registered via register_state_dict_post_hook"
            ):
                m._register_state_dict_hook(fn)


class TestModuleGlobalHooks(TestCase):
    # [CLASS DOC] Generic-hardware test case for hooks registered globally across all modules
    # (forward, forward-pre and backward), including ordering relative to local hooks and
    # kwargs. tearDown resets the global hook tables.
    hw_classification = HardwareClassification.GENERIC

    def tearDown(self):
        # [FUNCTION DOC] Resets the three global hook tables (backward, forward, forward-pre)
        # to empty ordered dicts so tests do not leak global hooks.
        nn.modules.module._global_backward_hooks = OrderedDict()
        nn.modules.module._global_forward_hooks = OrderedDict()
        nn.modules.module._global_forward_pre_hooks = OrderedDict()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_hooks(self):
        # [FUNCTION DOC] Registers global forward and backward hooks and counts calls over
        # three Sigmoid modules: forward hooks fire per forward, backward hooks per backward,
        # multiple registered hooks add their increments, and removing handles stops them.
        # Checks arguments (tuples, tensor output, module type, expected values) inside hooks.
        # Skipped under dynamo.
        module = nn.Sigmoid

        module_1 = module()
        module_2 = module()
        module_3 = module()

        input = torch.ones(5, 5, requires_grad=True)

        counter = {"forwards": 0, "backwards": 0}

        def fw_hook(inc, h_module, input, output):
            # [FUNCTION DOC] HELPER: Global forward hook asserting input is a tuple, output a
            # tensor, module a Sigmoid, and input and output values; adds the given increment
            # to the forward counter.
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter["forwards"] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Global backward hook asserting grads are tuples, module is
            # a Sigmoid and output gradient is twice ones; adds the given increment to the
            # backward counter.
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(isinstance(h_module, module))
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter["backwards"] += inc

        test_fwd = nn.modules.module.register_module_forward_hook(
            lambda *args: fw_hook(1, *args)
        )

        module_1(input)
        module_2(input)
        module_3(input)
        self.assertEqual(counter["forwards"], 3)
        self.assertEqual(counter["backwards"], 0)

        test_bwd = nn.modules.module.register_module_backward_hook(
            lambda *args: bw_hook(1, *args)
        )

        output_1 = module_1(input)
        output_2 = module_2(input)
        output_3 = module_3(input)
        self.assertEqual(counter["forwards"], 6)
        self.assertEqual(counter["backwards"], 0)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        output_2.backward(torch.ones(5, 5) * 2, retain_graph=False)
        output_3.backward(torch.ones(5, 5) * 2, retain_graph=False)
        self.assertEqual(counter["forwards"], 6)
        self.assertEqual(counter["backwards"], 3)

        output_1.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter["forwards"], 6)
        self.assertEqual(counter["backwards"], 4)

        test2_fwd = nn.modules.module.register_module_forward_hook(
            lambda *args: fw_hook(2, *args)
        )

        module_1(input)
        module_2(input)
        module_3(input)
        self.assertEqual(counter["forwards"], 15)
        self.assertEqual(counter["backwards"], 4)

        test2_bwd = nn.modules.module.register_module_backward_hook(
            lambda *args: bw_hook(2, *args)
        )

        module_1(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 18)
        self.assertEqual(counter["backwards"], 7)

        test2_bwd.remove()

        module_2(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 21)
        self.assertEqual(counter["backwards"], 8)

        test2_fwd.remove()

        module_3(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 22)
        self.assertEqual(counter["backwards"], 9)

        test_fwd.remove()
        test_bwd.remove()

    def test_module_global_hook_invalid_outputs(self):
        # [FUNCTION DOC] Checks that global backward hooks returning a grad_input of the wrong
        # length (too short or too long) raise a RuntimeError reporting got 0 or got 2 but
        # expected 1 on backward. Hooks are registered as context managers so they are removed
        # afterwards.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook returning grad_input without its last
            # element.
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook returning grad_input with an extra random
            # tensor appended.
            return grad_input + (torch.randn(2, 2),)

        with nn.modules.module.register_module_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                module(input).sum().backward()

        with nn.modules.module.register_module_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                module(input).sum().backward()

    def test_module_backward_global_hook_writeable(self):
        # [FUNCTION DOC] Registers a global backward hook that doubles grad_input and checks
        # the input gradient equals twice the sigmoid derivative. The hook is not removed here;
        # tearDown clears it.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook asserting all gradients are tensors and
            # returning grad_input scaled by two.
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        nn.modules.module.register_module_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_forward_preforward_hook_writeable(self):
        # [FUNCTION DOC] Registers global forward-pre and forward hooks (relu on the input,
        # negation of output) and checks the output and the input gradient equal the expected
        # values including the relu mask. Skipped under dynamo.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.sigmoid(input)

        def forward_pre_hook(m, input):
            # [FUNCTION DOC] HELPER: Forward pre-hook returning relu of the first input.
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Forward hook returning the negated output.
            return -output

        nn.modules.module.register_module_forward_pre_hook(forward_pre_hook)
        nn.modules.module.register_module_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = input > 0
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    def test_module_forward_preforward_hook_removable(self):
        # [FUNCTION DOC] Registers two forward pre-hooks that each remove their own handle
        # during the call and checks both were registered before, forward output is correct,
        # and the hooks dictionary is empty afterwards (removal during iteration is safe).
        """
        This test is to test when multiple pre-forward hook functions can be
        registered successfully and used correctly, if the handle can be removable
        during the pre-forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input):
            # [FUNCTION DOC] HELPER: Pre-hook that removes the first handle and returns the
            # input.
            nonlocal handle
            handle.remove()
            return input

        def removable_hook_2(m, input):
            # [FUNCTION DOC] HELPER: Pre-hook that removes the second handle and returns the
            # input.
            nonlocal handle_2
            handle_2.remove()
            return input

        handle = module.register_forward_pre_hook(removable_hook)
        handle_2 = module.register_forward_pre_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    def test_module_forward_forward_hook_removable(self):
        # [FUNCTION DOC] Same as the pre-hook removal test but with forward hooks: two hooks
        # remove their own handles while running, forward output is correct and the hooks
        # dictionary ends empty.
        """
        This test is to test when multiple forward hook functions can be registered
        successfully and used correctly, if the handle can be removable during the
        forward hook function call.
        """
        module = nn.Sigmoid()

        def removable_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Forward hook that removes the first handle and returns the
            # output.
            nonlocal handle
            handle.remove()
            return output

        def removable_hook_2(m, input, output):
            # [FUNCTION DOC] HELPER: Forward hook that removes the second handle and returns
            # the output.
            nonlocal handle_2
            handle_2.remove()
            return output

        handle = module.register_forward_hook(removable_hook)
        handle_2 = module.register_forward_hook(removable_hook_2)

        # make sure hook register is successful
        self.assertEqual(len(handle.hooks_dict_ref()), 2)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 2)

        input = torch.randn(2, 2)
        output = module(input)
        self.assertEqual(torch.sigmoid(input), output)

        # make sure hook removal is successful
        self.assertFalse(handle.id in handle.hooks_dict_ref())
        self.assertFalse(handle_2.id in handle.hooks_dict_ref())
        self.assertEqual(len(handle.hooks_dict_ref()), 0)
        self.assertEqual(len(handle_2.hooks_dict_ref()), 0)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_global_and_local_hooks_order(self):
        # [FUNCTION DOC] Registers global and local forward-pre, forward and backward hooks and
        # asserts, through flags, that each global hook runs before its local counterpart, and
        # that all were called after forward and backward. Uses the legacy backward hook API.
        # Skipped under dynamo.
        module = nn.Sigmoid()

        global_forward_pre_called = False
        local_forward_pre_called = False
        global_forward_called = False
        local_forward_called = False
        global_backward_called = False
        local_backward_called = False

        def global_forward_pre_hook(m, input):
            # [FUNCTION DOC] HELPER: Global pre-hook asserting the local pre-hook has not run
            # yet; sets its flag and returns the input.
            nonlocal global_forward_pre_called
            self.assertTrue(not local_forward_pre_called)
            global_forward_pre_called = True
            return input

        def local_forward_pre_hook(m, input):
            # [FUNCTION DOC] HELPER: Local pre-hook asserting the global pre-hook already ran;
            # sets its flag and returns the input.
            nonlocal local_forward_pre_called
            self.assertTrue(global_forward_pre_called)
            local_forward_pre_called = True
            return input

        def global_forward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Global forward hook asserting the local forward hook has
            # not run yet; sets its flag and returns the output.
            nonlocal global_forward_called
            self.assertTrue(not local_forward_called)
            global_forward_called = True
            return output

        def local_forward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Local forward hook asserting the global forward hook
            # already ran; sets its flag and returns the output.
            nonlocal local_forward_called
            self.assertTrue(global_forward_called)
            local_forward_called = True
            return output

        def global_backward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Global backward hook asserting the local one has not run
            # yet; sets its flag and returns the second argument (the gradient inputs).
            nonlocal global_backward_called
            self.assertTrue(not local_backward_called)
            global_backward_called = True
            return input

        def local_backward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Local backward hook asserting the global one already ran;
            # sets its flag and returns the second argument (the gradient inputs).
            nonlocal local_backward_called
            self.assertTrue(global_backward_called)
            local_backward_called = True
            return input

        input = torch.randn(5, 5, requires_grad=True)
        nn.modules.module.register_module_forward_pre_hook(global_forward_pre_hook)
        module.register_forward_pre_hook(local_forward_pre_hook)
        nn.modules.module.register_module_forward_hook(global_forward_hook)
        module.register_forward_hook(local_forward_hook)
        nn.modules.module.register_module_backward_hook(global_backward_hook)
        module.register_backward_hook(local_backward_hook)

        output = module(input)
        self.assertTrue(
            local_forward_called
            and local_forward_pre_called
            and global_forward_called
            and global_forward_pre_called
        )

        output.backward(torch.ones(5, 5), retain_graph=True)
        self.assertTrue(local_backward_called and global_backward_called)

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_module_global_hooks_with_kwargs(self):
        # [FUNCTION DOC] Registers a global forward hook with kwargs that adds the bias keyword
        # to the output, on a model whose forward already adds the bias, and checks the result
        # is input plus twice the bias. Skipped under dynamo.
        def kwarg_global_forward_hook(
            module: nn.Module,
            args: tuple[torch.Tensor],
            kwargs: dict[str, Any],
            out: torch.Tensor,
        ) -> Any:
            # [FUNCTION DOC] HELPER: Global forward hook with kwargs returning the output plus
            # the bias keyword argument.
            out = out + kwargs["bias"]
            return out

        model = KwargModel()
        nn.modules.module.register_module_forward_hook(
            kwarg_global_forward_hook,
            with_kwargs=True,
        )
        x: torch.Tensor = torch.randn(10, 20)
        bias: torch.Tensor = torch.randn(10, 20)
        out = model(x, bias=bias)
        self.assertEqual(out, x + 2 * bias, rtol=0, atol=1e-5)


class TestModuleHookNN(NNTestCase):
    # [CLASS DOC] Generic-hardware NNTestCase (CUDA leak and non-default stream checks enabled)
    # for per-module backward hooks (legacy, full and pre), invalid hook outputs, requires-grad
    # edge cases, in-place errors, warnings for legacy hooks, and registration hooks for
    # buffers, submodules and parameters.
    hw_classification = HardwareClassification.GENERIC
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def _test_hooks(self, backward_register_fn):
        # [FUNCTION DOC] HELPER: Shared driver for hook counting tests. Takes the name of the
        # backward hook registration method. Registers forward and backward (or backward-pre)
        # hooks with increments of 1 then 2 on a Sigmoid, verifies forward and backward counts
        # at each step, and removes handles to check hooks stop firing.
        module = nn.Sigmoid()
        input = torch.ones(5, 5, requires_grad=True)

        counter = {"forwards": 0, "backwards": 0}

        def fw_hook(inc, h_module, input, output):
            # [FUNCTION DOC] HELPER: Forward hook asserting tuple input, tensor output, same
            # module and expected values; adds the increment to the forward counter.
            self.assertIsInstance(input, tuple)
            self.assertTrue(isinstance(output, torch.Tensor))
            self.assertTrue(h_module is module)
            self.assertEqual(input[0], torch.ones(5, 5))
            self.assertEqual(output, torch.empty(5, 5).fill_(1 / (1 + 1 / math.e)))
            counter["forwards"] += inc

        def bw_hook(inc, h_module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook asserting tuple gradients, same module and
            # output gradient equal to twice ones; adds the increment to the backward counter.
            self.assertIsInstance(grad_input, tuple)
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter["backwards"] += inc

        # backward_pre_hook expects callback with only `module` and `grad_output`
        # as arguments.
        def bw_pre_hook(inc, h_module, grad_output):
            # [FUNCTION DOC] HELPER: Backward pre-hook (module and grad_output only) asserting
            # tuple gradient, same module and value twice ones; adds the increment to the
            # backward counter.
            self.assertIsInstance(grad_output, tuple)
            self.assertTrue(h_module is module)
            self.assertEqual(grad_output[0], torch.ones(5, 5) * 2)
            counter["backwards"] += inc

        test_fwd = module.register_forward_hook(lambda *args: fw_hook(1, *args))

        module(input)
        module(input)
        self.assertEqual(counter["forwards"], 2)
        self.assertEqual(counter["backwards"], 0)

        bw_hook_fn = (
            bw_pre_hook
            if backward_register_fn == "register_full_backward_pre_hook"
            else bw_hook
        )
        test_bwd = getattr(module, backward_register_fn)(
            lambda *args: bw_hook_fn(1, *args)
        )

        output = module(input)
        self.assertEqual(counter["forwards"], 3)
        self.assertEqual(counter["backwards"], 0)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter["forwards"], 3)
        self.assertEqual(counter["backwards"], 1)

        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        self.assertEqual(counter["forwards"], 3)
        self.assertEqual(counter["backwards"], 2)

        test2_fwd = module.register_forward_hook(lambda *args: fw_hook(2, *args))

        output = module(input)
        self.assertEqual(counter["forwards"], 6)
        self.assertEqual(counter["backwards"], 2)

        test2_bwd = getattr(module, backward_register_fn)(
            lambda *args: bw_hook_fn(2, *args)
        )

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 9)
        self.assertEqual(counter["backwards"], 5)

        test2_bwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 12)
        self.assertEqual(counter["backwards"], 6)

        test2_fwd.remove()

        module(input).backward(torch.ones(5, 5) * 2)
        self.assertEqual(counter["forwards"], 13)
        self.assertEqual(counter["backwards"], 7)

        test_fwd.remove()
        test_bwd.remove()

    def test_hooks(self):
        # [FUNCTION DOC] Runs the shared hook counting driver with the legacy backward hook,
        # the full backward hook and the full backward pre-hook registration methods.
        self._test_hooks("register_backward_hook")
        self._test_hooks("register_full_backward_hook")
        self._test_hooks("register_full_backward_pre_hook")

    def test_hook_cpp(self):
        # [FUNCTION DOC] Registers a full backward hook on a BatchNorm1d (a native backward
        # node) and checks the hook receives single element gradient tuples and the module
        # itself during backward.
        bn = nn.BatchNorm1d(5)

        def hook(module, grad_inputs, grad_outputs):
            # [FUNCTION DOC] HELPER: Backward hook asserting single element grad_inputs and
            # grad_outputs and that the module is the batch norm.
            self.assertEqual(len(grad_inputs), 1)
            self.assertEqual(len(grad_outputs), 1)
            self.assertEqual(module, bn)

        bn.register_full_backward_hook(hook)
        output = bn(torch.randn(5, 5, requires_grad=True))
        output.sum().backward()

    def test_backward_hooks_interaction(self):
        # [FUNCTION DOC] Checks that gradients modified by a full backward pre-hook (scaled by
        # half) are what the full backward hook sees as grad_output; both hooks must run,
        # giving a count of 2.
        # Test to make sure that the grad_outputs
        # updated by full_backward_pre_hook are received by
        # the full_backward_hook
        module = torch.nn.Sigmoid()

        cnt = {"backward_cnt": 0}

        def bw_pre_hook(m, grad_output):
            # [FUNCTION DOC] HELPER: Backward pre-hook that counts the call and returns
            # grad_output scaled by one half.
            cnt["backward_cnt"] += 1
            return (grad_output[0] * 0.5,)

        def bw_hook(m, grad_in, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook that asserts grad_output equals one half,
            # counts the call and returns grad_output.
            self.assertEqual(torch.full_like(grad_output[0], 0.5), grad_output[0])
            cnt["backward_cnt"] += 1
            return grad_output

        module.register_full_backward_pre_hook(bw_pre_hook)
        module.register_full_backward_hook(bw_hook)

        t = torch.ones(1, 2, requires_grad=True)
        module(t).sum().backward()
        self.assertEqual(cnt["backward_cnt"], 2)

    def test_hook_invalid_outputs(self):
        # [FUNCTION DOC] Checks that legacy backward hooks and full backward pre-hooks
        # returning a gradient tuple of the wrong length raise a RuntimeError saying got N but
        # expected 1 on backward. Hooks are registered as context managers.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)

        def bw_fail1(self, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Legacy backward hook returning grad_input without the last
            # element.
            return grad_input[:-1]

        def bw_fail2(self, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Legacy backward hook returning grad_input plus an extra
            # random tensor.
            return grad_input + (torch.randn(2, 2),)

        with module.register_backward_hook(bw_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                module(input).sum().backward()

        with module.register_backward_hook(bw_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                module(input).sum().backward()

        def bw_pre_fail1(self, grad_output):
            # [FUNCTION DOC] HELPER: Backward pre-hook returning an empty tuple.
            return ()

        def bw_pre_fail2(self, grad_output):
            # [FUNCTION DOC] HELPER: Backward pre-hook returning grad_output plus an extra
            # random tensor.
            return grad_output + (torch.randn(2, 2),)

        with module.register_full_backward_pre_hook(bw_pre_fail1):
            with self.assertRaisesRegex(RuntimeError, "got 0, but expected 1"):
                module(input).sum().backward()

        with module.register_full_backward_pre_hook(bw_pre_fail2):
            with self.assertRaisesRegex(RuntimeError, "got 2, but expected 1"):
                module(input).sum().backward()

    def test_hook_requires_grad(self):
        # [FUNCTION DOC] Checks that registering a full backward hook keeps requires_grad flags
        # of the module inputs: forward asserts the first and third inputs require grad and the
        # second (detached) does not, both without and with the hook.
        test_self = self

        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module whose forward asserts requires_grad on its three
            # arguments (true, false, true) via the outer test case and returns the sum of
            # their sums.
            def forward(self, arg1, arg2, arg3):
                # [FUNCTION DOC] HELPER: Asserts requires_grad flags of the three arguments and
                # returns the sum of their sums.
                test_self.assertTrue(arg1.requires_grad)
                test_self.assertFalse(arg2.requires_grad)
                test_self.assertTrue(arg3.requires_grad)
                return arg1.sum() + arg2.sum() + arg3.sum()

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        mod(inp, inp.detach(), inp)
        # Ensure that requires grad is properly propagated
        mod.register_full_backward_hook(lambda mod, gI, gO: None)
        mod(inp, inp.detach(), inp)

    def test_hook_no_requires_grad(self):
        # [FUNCTION DOC] Registers a full backward hook on a Linear whose input does not
        # require grad. The hook sees none for grad_input and output gradients of size 1 by 3.
        # The first backward emits a UserWarning that the hook fires from outputs since no
        # inputs require grad; returning grad_input is fine; returning a tensor raises a
        # RuntimeError saying no input requires gradient.
        mod = nn.Linear(2, 3)

        inp = torch.rand(1, 2)

        return_val = "None"
        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook counting calls, asserting grad_input entries
            # are none and output gradients have size 1 by 3, and returning a value chosen by
            # the enclosing return_val setting (grad_input, an invalid tensor, none) or raising
            # RuntimeError for an unknown setting.
            hook_called[0] += 1
            for gI in grad_input:
                self.assertIsNone(gI)
            for gO in grad_output:
                self.assertEqual(gO.size(), (1, 3))

            if return_val == "grad_input":
                return grad_input
            elif return_val == "invalid":
                # If the inputs were requiring gradients, this would be
                # a valid return
                return inp
            elif return_val == "None":
                return None
            else:
                raise RuntimeError("Invalid return_val string")

        mod.register_full_backward_hook(hook)

        # This should run and trigger the hook properly
        with self.assertWarnsRegex(
            UserWarning,
            (
                "Full backward hook is firing when gradients are computed with "
                "respect to module outputs since no inputs require gradients"
            ),
        ):
            mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 1)

        return_val = "grad_input"

        mod(inp).sum().backward()
        self.assertEqual(hook_called[0], 2)

        return_val = "invalid"
        with self.assertRaisesRegex(RuntimeError, "where no input requires gradient"):
            mod(inp).sum().backward()

    @skipIfTorchDynamo("TorchDynamo does not work well with hooks")
    def test_pre_hook_no_requires_grad_no_warning(self):
        # [FUNCTION DOC] Checks that a full backward pre-hook on a Linear whose input has no
        # grad requirement does not emit the full-backward-hook warning. Skipped under dynamo.
        # A full backward pre-hook only receives grad_output, so firing from the
        # output side when no input requires grad is expected and must not emit
        # the full-backward-hook warning. See
        # https://github.com/pytorch/pytorch/issues/189093
        mod = nn.Linear(2, 3)
        inp = torch.rand(1, 2)
        mod.register_full_backward_pre_hook(lambda mod, gO: None)
        self.assertNotWarn(lambda: mod(inp).sum().backward())

    def test_hook_last_arg_requires_grad(self):
        # [FUNCTION DOC] Checks that registering a full backward hook on L1Loss and calling it
        # with a first input that does not require grad and a last one that does requires no
        # exception.
        mod = nn.L1Loss()
        inp = torch.rand(1, requires_grad=True)
        mod.register_full_backward_hook(lambda m, gI, gO: None)

        try:
            mod(inp.detach(), inp)
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

    def test_hook_extra_input(self):
        # [FUNCTION DOC] Checks full backward hook gradient tuples with a non-tensor first
        # input: grad_input has none for it and a tensor for the second, grad_output has a
        # tensor for the first output and none for the non-tensor output.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning a clone of the tensor argument and the non-
            # tensor argument unchanged.
            def forward(self, non_tensor, tensor):
                # [FUNCTION DOC] HELPER: Returns a pair of the cloned tensor and the non-tensor
                # argument.
                return tensor.clone(), non_tensor

        inp = torch.rand(2, requires_grad=True)
        mod = MyModule()

        def hook(mod, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook asserting grad_input is none then tensor,
            # and grad_output is tensor then none.
            self.assertIsNone(grad_input[0])
            self.assertIsInstance(grad_input[1], torch.Tensor)

            self.assertIsInstance(grad_output[0], torch.Tensor)
            self.assertIsNone(grad_output[1])

        mod.register_full_backward_hook(hook)
        out, _ = mod(True, inp)
        out.sum().backward()

    def test_hook_inplace(self):
        # [FUNCTION DOC] For both a full backward hook and a full backward pre-hook, checks in-
        # place behavior: no in-place gives one hook call; modifying the input in place raises
        # a RuntimeError about a view being modified in place; reusing a view after its base
        # was modified raises a RuntimeError; modifying the output in place also raises.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module that stores its input, optionally increments it in
            # place, and returns a clone.
            def forward(self, inp, do_inplace):
                # [FUNCTION DOC] HELPER: Stores the input on the module, adds one to it in
                # place when requested, and returns a clone.
                self.inp = inp
                if do_inplace:
                    inp += 1
                return inp.clone()

        hook_called = [0]

        def hook(mod, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook that counts a call.
            hook_called[0] += 1

        def hook_pre(mod, grad_output):
            # [FUNCTION DOC] HELPER: Backward pre-hook that counts a call.
            hook_called[0] += 1

        inp = torch.rand(10, requires_grad=True)
        mod = MyModule()
        for hook_fn, register_fn in [
            (hook, mod.register_full_backward_hook),
            (hook_pre, mod.register_full_backward_pre_hook),
        ]:
            hook_called[0] = 0
            with register_fn(hook_fn):
                # No inplace should work
                mod(inp, False).sum().backward()
                self.assertEqual(hook_called[0], 1)

                # Input inplace error should throw an error
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Output 0 of BackwardHookFunction is "
                    "a view and is being modified inplace.",
                ):
                    mod(inp.clone(), True)

                # Input inplace error should throw an error if we try to reuse the view after they have
                # been modified
                local_inp = inp.clone()
                out = mod(local_inp, False)
                local_inp[0] *= 1
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Output 0 of BackwardHookFunction is "
                    "a view and its base or another view",
                ):
                    # Any operation involving the view will fail here
                    mod.inp + 2

                # Output inplace error should throw an error
                out = mod(inp, False)
                with self.assertRaisesRegex(
                    RuntimeError,
                    "BackwardHookFunction is a view and is being modified inplace.",
                ):
                    out += 1

    def test_hook_non_full_warning(self):
        # [FUNCTION DOC] Checks that registering a legacy backward hook on modules with
        # unsupported structure emits FutureWarning on forward: input in a list container,
        # output in a list, outputs from different autograd nodes, and a forward with multiple
        # autograd nodes.
        def noop(*args):
            # [FUNCTION DOC] HELPER: No-op hook accepting any arguments.
            pass

        a = torch.rand(2, requires_grad=True)
        b = torch.rand(2, requires_grad=True)

        # Check invalid input container
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module taking a list and returning clones of its first two
            # elements as a tuple.
            def forward(self, l):
                # [FUNCTION DOC] HELPER: Returns clones of the first two entries of the list as
                # a tuple.
                return l[0].clone(), l[1].clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning,
            "does not take as input a single Tensor or a tuple of Tensors",
        ):
            m([a, b])

        # Check invalid output container
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning clones of its two arguments in a list.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a list of the two cloned arguments.
                return [a.clone(), b.clone()]

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "does not return a single Tensor or a tuple of Tensors"
        ):
            m(a, b)

        # Check invalid output from different Nodes
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning clones of its two arguments as a tuple
            # (separate autograd nodes).
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: Returns a tuple of clones of the two arguments.
                return a.clone(), b.clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "outputs are generated by different autograd Nodes"
        ):
            m(a, b)

        # Check invalid forward with multiple Nodes
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning a double clone of its argument (two autograd
            # nodes in forward).
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns a clone of a clone of the argument.
                return a.clone().clone()

        m = MyModule()
        m.register_backward_hook(noop)

        with self.assertWarnsRegex(
            FutureWarning, "the forward contains multiple autograd Nodes"
        ):
            m(a)

    def test_hook_backward_size(self):
        # [FUNCTION DOC] Uses a module with several ops, two inputs of different sizes and an
        # eight element output, and checks a full backward hook sees grad_input sizes of the
        # inputs (5 by 5 and 10 by 10) and one grad_output of size 8.
        # Make module with multiple operations in forward
        # And different size for input and outputs
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Module with multiple operations producing an eight element
            # output from two inputs of different sizes.
            def forward(self, arg1, arg2):
                # [FUNCTION DOC] HELPER: Combines sums of the two inputs, reduces to one
                # element, expands to eight elements and returns a contiguous tensor.
                tmp = arg1.sum() * arg2
                tmp = tmp + arg2.sum() * arg1.sum()
                tmp = tmp.sum().view(1)
                tmp = tmp.expand(8).contiguous()
                return tmp

        module = MyModule()
        inp1 = torch.randn(5, 5, requires_grad=True)
        inp2 = torch.randn(10, 10, requires_grad=True)

        def bw_hook(module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook asserting the shapes of grad_input (two
            # entries) and grad_output (one entry of size 8).
            self.assertEqual(len(grad_input), 2)
            self.assertEqual(grad_input[0].size(), torch.Size([5, 5]))
            self.assertEqual(grad_input[1].size(), torch.Size([10, 10]))
            self.assertEqual(len(grad_output), 1)
            self.assertEqual(grad_output[0].size(), torch.Size([8]))

        with module.register_full_backward_hook(bw_hook):
            module(inp1, inp2).sum().backward()

    def test_hook_backward_writeable(self):
        # [FUNCTION DOC] Registers a legacy backward hook doubling grad_input on Sigmoid and
        # checks the input gradient equals twice the sigmoid derivative.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def bw_hook(module, grad_input, grad_output):
            # [FUNCTION DOC] HELPER: Backward hook asserting all gradients are tensors and
            # returning grad_input scaled by two.
            for grad in grad_input:
                self.assertTrue(isinstance(grad, torch.Tensor))
            for grad in grad_output:
                self.assertTrue(isinstance(grad, torch.Tensor))
            return tuple(gi * 2 for gi in grad_input)

        module.register_backward_hook(bw_hook)
        module(input).backward(torch.ones(5, 5))
        expected_grad = sig_x * (1 - sig_x) * 2
        self.assertEqual(input.grad, expected_grad)

    def test_hook_forward_preforward_writable(self):
        # [FUNCTION DOC] Registers per-module forward-pre (relu of input) and forward
        # (negation) hooks on Sigmoid and checks the output, and the input gradient including
        # the positive mask, match expected values.
        module = nn.Sigmoid()
        input = torch.randn(5, 5, requires_grad=True)
        sig_x = torch.nn.functional.sigmoid(input)

        def forward_pre_hook(m, input):
            # [FUNCTION DOC] HELPER: Forward pre-hook returning relu of the first input.
            return torch.nn.functional.relu(input[0])

        def forward_hook(m, input, output):
            # [FUNCTION DOC] HELPER: Forward hook returning the negated output.
            return -output

        module.register_forward_pre_hook(forward_pre_hook)
        module.register_forward_hook(forward_hook)
        output = module(input)
        expected_res = -torch.nn.functional.sigmoid(torch.nn.functional.relu(input))
        self.assertEqual(output, expected_res)
        output.backward(torch.ones(5, 5) * 2, retain_graph=True)
        mask = input > 0
        expected_grad = -sig_x * (1 - sig_x) * 2 * mask
        self.assertEqual(input.grad, expected_grad)

    def test_hook_buffer_registration(self):
        # [FUNCTION DOC] With the buffer registration hook returning the buffer or none,
        # registers the global hook, builds a basic net from the test helper and checks every
        # buffer of the net is marked registered; the handle is removed in a finally block.
        for return_buffer in (True, False):

            def buffer_registration_hook(module, name, buffer):
                # [FUNCTION DOC] HELPER: Marks the buffer as registered and returns it if the
                # outer flag asks for a replacement; otherwise returns none.
                buffer.registered = True
                if return_buffer:
                    return buffer

            handle = torch.nn.modules.module.register_module_buffer_registration_hook(
                buffer_registration_hook
            )
            try:
                l, n, s = _create_basic_net()
                for b in s.buffers():
                    self.assertTrue(getattr(b, "registered", False))
            finally:
                handle.remove()

    def test_hook_submodule_registration(self):
        # [FUNCTION DOC] Same as the buffer test for the global module registration hook: every
        # module in the basic net must be marked registered; hook returns the submodule or none
        # per iteration; handle removed in a finally block.
        for return_submodule in (True, False):

            def module_registration_hook(module, name, submodule):
                # [FUNCTION DOC] HELPER: Marks both the parent module and the submodule as
                # registered; returns the submodule if the outer flag asks for it.
                module.registered = True
                submodule.registered = True
                if return_submodule:
                    return submodule

            handle = torch.nn.modules.module.register_module_module_registration_hook(
                module_registration_hook
            )
            try:
                l, n, s = _create_basic_net()
                for m in s.modules():
                    self.assertTrue(getattr(m, "registered", False))
            finally:
                handle.remove()

    def test_hook_parameter_registration(self):
        # [FUNCTION DOC] Same as the buffer test for the global parameter registration hook:
        # every parameter of the basic net must be marked registered; handle removed in a
        # finally block.
        for return_parameter in (True, False):

            def parameter_registration_hook(module, name, parameter):
                # [FUNCTION DOC] HELPER: Marks the parameter as registered and returns it if
                # the outer flag asks for it, otherwise returns none.
                parameter.registered = True
                if return_parameter:
                    return parameter

            handle = (
                torch.nn.modules.module.register_module_parameter_registration_hook(
                    parameter_registration_hook
                )
            )
            try:
                l, n, s = _create_basic_net()
                for p in s.parameters():
                    self.assertTrue(getattr(p, "registered", False))
            finally:
                handle.remove()


instantiate_parametrized_tests(TestModuleHooks)
instantiate_parametrized_tests(TestStateDictHooks)

if __name__ == "__main__":
    run_tests()
