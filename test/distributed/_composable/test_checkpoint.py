# Owner(s): ["oncall: distributed"]

# [MODULE DOC] Unit tests for the composable activation checkpoint API
# (torch.distributed._composable.checkpoint). Defines a memory-measuring context manager,
# several small toy models, and one TestCheckpoint case that compares gradients and memory
# between checkpointed and plain modules, and exercises error handling and keyword options.
# Import-time side effect: computes the module-level device_type string from the current
# accelerator (falls back to "cpu").

import unittest
from collections import deque, OrderedDict
from contextlib import ContextDecorator, contextmanager, nullcontext
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
from torch.distributed._composable import checkpoint
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import run_tests, TEST_XPU, TestCase
from torch.utils.checkpoint import CheckpointError


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


class MemoryDelta(ContextDecorator):
    # [CLASS DOC] HELPER: Context manager/decorator that records the accelerator's currently
    # active allocated bytes on entry and on exit, so delta() gives the net memory change
    # across the block. Only cuda and xpu devices are measured; other devices always report 0.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] HELPER: Stores the device to measure and zero-initializes the
        # enter/exit byte counters.
        self.device: torch.device = device
        self.active_memory_enter: int = 0
        self.active_memory_exit: int = 0

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Records active allocator bytes on entry (0 for non-cuda/xpu
        # devices) and returns self so the caller can query delta() afterwards.
        self.active_memory_enter = (
            torch.accelerator.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda" or self.device.type == "xpu"
            else 0
        )
        return self

    def __exit__(self, *exc):
        # [FUNCTION DOC] HELPER: Records active allocator bytes on exit (0 for non-cuda/xpu
        # devices). Returns nothing, so exceptions from the block are not suppressed.
        self.active_memory_exit = (
            torch.accelerator.memory_stats()["active_bytes.all.current"]
            if self.device.type == "cuda" or self.device.type == "xpu"
            else 0
        )

    def delta(self) -> int:
        # [FUNCTION DOC] HELPER: Returns exit minus enter active bytes, i.e. memory retained
        # by the measured block (activations kept for backward). Valid after the block ends.
        return self.active_memory_exit - self.active_memory_enter


class ToyModel(nn.Module):
    # [CLASS DOC] HELPER: Small model with a linear layer l1 followed by a sequential block
    # seq (ReLU, Linear, ReLU), all 100 wide. The tests checkpoint the seq submodule.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the l1 linear layer and the seq block.
        super().__init__()
        self.l1 = nn.Linear(100, 100)
        self.seq = nn.Sequential(
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies l1 then seq to the input and returns the result.
        return self.seq(self.l1(x))


class RandomModel(nn.Module):
    # [CLASS DOC] HELPER: Model whose forward consumes random numbers, used to verify that
    # checkpoint recomputation reproduces the same random state. Holds one 100x100 parameter p.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the randomly initialized parameter p.
        super().__init__()
        self.p = nn.Parameter(torch.randn(100, 100))

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Multiplies p by a freshly drawn random matrix (on p's
        # device) and then multiplies the input by that product. Non-deterministic across
        # calls unless the RNG state is restored.
        y = torch.matmul(self.p, torch.randn(100, 100, device=self.p.device))
        return torch.matmul(x, y)


class MultiOutputModel(nn.Module):
    # [CLASS DOC] HELPER: Model with two 100x100 weights that returns a pair of tensors, so
    # checkpointing it yields multiple backward inputs.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] HELPER: Creates random parameters w1 and w2 on the given device.
        super().__init__()
        self.w1 = nn.Parameter(torch.randn((100, 100), device=device))
        self.w2 = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Computes matmul with w1, ReLU, matmul with w2, then returns
        # the elementwise sine and cosine of the result as a two-tensor tuple.
        z = x @ self.w1
        z = nn.functional.relu(z)
        z = z @ self.w2
        return z.sin(), z.cos()


class MultiInputModel(nn.Module):
    # [CLASS DOC] HELPER: Model that takes a two-tensor tuple (the output of
    # MultiOutputModel) and reduces it to a single tensor using one 100x100 weight w.
    def __init__(self, device: torch.device):
        # [FUNCTION DOC] HELPER: Creates the random parameter w on the given device.
        super().__init__()
        self.w = nn.Parameter(torch.randn((100, 100), device=device))

    def forward(self, xs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Takes a tuple of exactly two tensors, adds them, multiplies
        # by w and applies ReLU. Raises AssertionError if the tuple length is not 2.
        if len(xs) != 2:
            raise AssertionError(f"Expects 2 args but got {len(xs)}")
        x, y = xs
        z = x + y
        z = z @ self.w
        return nn.functional.relu(z)


class TestCheckpoint(TestCase):
    # [CLASS DOC] Tests for the composable checkpoint API: gradient equality versus
    # non-checkpointed models, memory savings on accelerators, RNG handling, multi-output
    # modules, state cleanup after errors, and the supported keyword arguments.
    def _get_graph_size(self, out: torch.Tensor) -> int:
        # [FUNCTION DOC] HELPER: Counts autograd graph nodes reachable from the output's
        # grad_fn by traversing next_functions (nodes reached via several paths are counted
        # once per path). Not used by the tests in this file.
        q = deque([out.grad_fn])
        num_functions = 0
        while len(q):
            fn = q.pop()
            num_functions += 1
            for next_fn, _ in fn.next_functions:
                if next_fn:
                    q.append(next_fn)

        return num_functions

    def _test_tensor_only(
        self,
        net: nn.Module,
        x: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Runs the net twice on clones of x requiring grad: once
        # plain, and once on a deep copy with its seq submodule wrapped by composable
        # checkpoint, measuring memory retained by each forward. Asserts all parameter
        # gradients match; on cuda/xpu also asserts the checkpointed forward retains less
        # memory. Mutates the copy by applying checkpoint to it.
        x1 = x.clone()
        x2 = x.clone()
        x1.requires_grad = True
        x2.requires_grad = True

        net1 = net
        net2 = deepcopy(net)

        # no checkpoint
        with MemoryDelta(x.device) as mem1:
            loss1 = net1(x1).sum()
        loss1.backward()

        # with checkpoint
        checkpoint(net2.seq)
        with MemoryDelta(x.device) as mem2:
            loss2 = net2(x2).sum()
        loss2.backward()

        if x.is_cuda or x.is_xpu:
            self.assertTrue(mem2.delta() < mem1.delta())

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_tensor_only_cpu(self):
        # [FUNCTION DOC] Runs the tensor-only checkpoint comparison on CPU with a ToyModel.
        x = torch.randn(20, 100)
        net = ToyModel()
        self._test_tensor_only(net, x)

    @unittest.skipIf(not TEST_CUDA and not TEST_XPU, "no cuda/xpu")
    def test_tensor_only_gpu(self):
        # [FUNCTION DOC] Same comparison as the CPU test but on the first accelerator;
        # skipped when neither CUDA nor XPU is available. Also checks memory savings.
        x = torch.randn(20, 100, device=f"{device_type}:0")
        net = ToyModel().to(f"{device_type}:0")
        self._test_tensor_only(net, x)

    def test_random_cpu(self):
        # [FUNCTION DOC] Verifies checkpointing a random-consuming module reproduces the
        # same gradients: runs a plain copy, resets the CPU RNG state to the saved value,
        # then runs the checkpointed copy and compares parameter gradients.
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()

        net1 = RandomModel()
        net2 = deepcopy(net1)

        cpu_rng_state = torch.get_rng_state()
        net1(x1).sum().backward()
        torch.set_rng_state(cpu_rng_state)
        checkpoint(net2)(x2).sum().backward()

        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_multi_args(self):
        # [FUNCTION DOC] Checkpoints the first and third modules of a four-module
        # sequential (multi-output then multi-input, twice) and checks the parameter
        # gradients equal those of the unchecked copy.
        """
        Tests checkpoint for modules with multiple output args and hence
        multiple backward function input args.
        """
        device = torch.device("cpu")
        net1 = nn.Sequential(
            MultiOutputModel(device),
            MultiInputModel(device),
            MultiOutputModel(device),
            MultiInputModel(device),
        )
        net2 = deepcopy(net1)
        checkpoint(net2[0])
        checkpoint(net2[2])
        x1 = torch.randn(20, 100, requires_grad=True)
        x2 = x1.clone()
        net1(x1).sum().backward()
        net2(x2).sum().backward()
        for p1, p2 in zip(net1.parameters(), net2.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    def test_clears_state_on_error_in_forward(self):
        # [FUNCTION DOC] Checks the checkpoint state's internal generator is reset to None
        # when the wrapped module raises, both during recomputation in backward and during
        # the initial forward.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear wrapper that raises RuntimeError "foo" either on
            # its second call (recomputation) or on its first call, depending on a flag.
            def __init__(self, raise_in_recomp):
                # [FUNCTION DOC] HELPER: Stores the flag, a forward call counter, and a
                # 2x2 linear layer.
                super().__init__()
                self.fwd_count = 0
                self.raise_in_recomp = raise_in_recomp
                self.a = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: If raising in recomputation, the first call
                # succeeds (counter incremented) and the second raises RuntimeError. If not,
                # the first call raises immediately. Otherwise applies the linear layer.
                if self.raise_in_recomp and self.fwd_count == 1:
                    raise RuntimeError("foo")
                else:
                    if not self.raise_in_recomp:
                        # raise in the first forward
                        raise RuntimeError("foo")
                    self.fwd_count += 1
                    return self.a(x)

        m = MyModel(raise_in_recomp=True)
        m_seq = torch.nn.Sequential(OrderedDict({"m": m}))
        checkpoint(m_seq.m)
        inp = torch.randn(1, 2)
        out = m_seq(inp).sum()
        # Should raise in forward recomputation
        with self.assertRaisesRegex(RuntimeError, "foo"):
            out.backward()

        # Check that _ac_generator is cleared out
        self.assertEqual(None, checkpoint.state(m)._ac_generator)

        m = MyModel(raise_in_recomp=False)
        checkpoint(m)
        inp = torch.randn(1, 2)
        # Should raise in first forward
        with self.assertRaises(RuntimeError):
            m(inp)

        self.assertEqual(None, checkpoint.state(m)._ac_generator)

    def test_checkpoint_kwargs(self):
        # [FUNCTION DOC] Exercises checkpoint keyword arguments: use_reentrant=True raises
        # NotImplementedError, an unknown keyword raises ValueError, custom context_fn
        # context managers wrap forward and/or recomputation, determinism_check "none"
        # versus "default" (CheckpointError on mismatch), and preserve_rng_state on/off
        # affecting whether gradients match a non-checkpointed copy.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear wrapper with a call counter that can raise on the
            # first call ("foo") and second call ("bar"), or mutate its input in place on
            # the second call so recomputation differs from the original forward.
            def __init__(self, raise_exp: bool, change_shape_in_recomp: bool):
                # [FUNCTION DOC] HELPER: Stores the two behavior flags, a call counter and
                # a 2x2 linear layer.
                super().__init__()
                self.fwd_count = 0
                self.raise_exp = raise_exp
                self.change_shape_in_recomp = change_shape_in_recomp
                self.a = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Raises RuntimeError "foo" on call 0 and "bar" on
                # call 1 when raise_exp is set; on call 1 optionally applies in-place ReLU
                # to the input. Then adds random noise, applies the linear layer,
                # increments the counter and returns the result.
                if self.raise_exp and self.fwd_count == 0:
                    raise RuntimeError("foo")
                if self.raise_exp and self.fwd_count == 1:
                    raise RuntimeError("bar")
                if self.change_shape_in_recomp and self.fwd_count == 1:
                    x.relu_()
                random_tensor = torch.randn(1, 2)
                x = self.a(x + random_tensor)
                self.fwd_count += 1
                return x

        m = MyModel(True, False)
        m0, m1, m2, m3 = (deepcopy(m) for _ in range(4))

        # composable checkpoint does not support use_reentrant=True
        with self.assertRaisesRegex(
            NotImplementedError,
            "use_reentrant=True is not supported in composable checkpoint. "
            "Please use torch.utils.checkpoint.checkpoint instead.",
        ):
            checkpoint(m, use_reentrant=True)

        # check giving an unsupported kwarg
        with self.assertRaisesRegex(ValueError, "Unexpected keyword arguments: foo"):
            checkpoint(m0, foo="bar")

        handled_fwd_exp = False
        handled_recomp_exp = False

        @contextmanager
        def fwd_ctx(mod: MyModel):
            # [FUNCTION DOC] HELPER: Context manager for the original forward: disables the
            # module's raise_exp flag inside, and on exit sets handled_fwd_exp and restores
            # raise_exp to True.
            try:
                mod.raise_exp = False
                yield
            finally:
                nonlocal handled_fwd_exp
                handled_fwd_exp = True
                mod.raise_exp = True

        @contextmanager
        def recomp_ctx(mod: MyModel):
            # [FUNCTION DOC] HELPER: Context manager for recomputation: disables the
            # module's raise_exp flag inside, and on exit sets handled_recomp_exp and
            # restores raise_exp to True.
            try:
                mod.raise_exp = False
                yield
            finally:
                nonlocal handled_recomp_exp
                handled_recomp_exp = True
                mod.raise_exp = True

        # Test different context functions
        x = torch.randn(1, 2, requires_grad=True)
        checkpoint(
            m1, context_fn=lambda: (partial(fwd_ctx, m1)(), partial(recomp_ctx, m1)())
        )
        m1(x.clone()).sum().backward()
        self.assertEqual((handled_fwd_exp, handled_recomp_exp), (True, True))

        checkpoint(m2, context_fn=lambda: (nullcontext(), partial(recomp_ctx, m2)()))
        with self.assertRaisesRegex(RuntimeError, "foo"):
            m2(x.clone())

        handled_fwd_exp = False  # Reset flag
        checkpoint(m3, context_fn=lambda: (partial(fwd_ctx, m3)(), nullcontext()))
        with self.assertRaisesRegex(RuntimeError, "bar"):
            m3(x.clone()).sum().backward()
        self.assertEqual(handled_fwd_exp, True)

        # Test determinism check failure
        m4 = MyModel(False, True)
        m5 = deepcopy(m4)
        # Determinism check should not throw an error,
        # but autograd should throw a RuntimeError
        checkpoint(m4, determinism_check="none")
        with self.assertRaises(RuntimeError):
            m4(x.clone()).sum().backward()

        # Determinism check should throw a CheckpointError
        checkpoint(m5, determinism_check="default")
        with self.assertRaises(CheckpointError):
            m5(x.clone()).sum().backward()

        # Test preserving random state
        m6 = MyModel(False, False)
        m7, m8 = (deepcopy(m6) for _ in range(2))
        checkpoint(m7, preserve_rng_state=False)
        checkpoint(m8, preserve_rng_state=True)

        for mi in (m6, m7, m8):
            torch.manual_seed(42)
            loss = mi(x.clone()).sum()
            torch.manual_seed(41)
            loss.backward()
        # check that m6 and m7 have at least one different grad
        self.assertNotEqual(
            (p1.grad for p1 in m6.parameters()), (p2.grad for p2 in m7.parameters())
        )
        # check that m6 and m8 have identical grads
        for p1, p2 in zip(m6.parameters(), m8.parameters()):
            self.assertEqual(p1.grad, p2.grad)


if __name__ == "__main__":
    run_tests()
