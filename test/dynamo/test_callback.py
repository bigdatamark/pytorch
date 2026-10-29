# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._dynamo.callback: the compile start/end callback handler, its
# duplicate-prevention and pending-counter bookkeeping, and the callback triggers emitted during
# a real max-autotune compile (GPU only). Computes a module-level device_type string from the
# current accelerator, defaulting to cpu.

import unittest
from unittest.mock import Mock

import torch
from torch._dynamo.callback import callback_handler, CallbackArgs, CallbackTrigger
from torch._dynamo.test_case import run_tests, TestCase
from torch._guards import CompileId
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.triton_utils import HAS_CUDA_AND_TRITON


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class CallbackTests(TestCase):
    # [CLASS DOC] Test case that registers two Mock callbacks (start and end) on the global
    # callback_handler in setUp and clears the handler in tearDown.
    def setUp(self) -> None:
        # [FUNCTION DOC] Run the base setUp, create Mock objects for the start and end callbacks
        # and register them with the global callback_handler.
        super().setUp()
        self._on_compile_start = Mock()
        self._on_compile_end = Mock()
        callback_handler.register_start_callback(self._on_compile_start)
        callback_handler.register_end_callback(self._on_compile_end)

    def tearDown(self) -> None:
        # [FUNCTION DOC] Clear all callbacks on the global callback_handler (which raises if
        # compiles are still pending), then run the base tearDown and return its result.
        callback_handler.clear()
        return super().tearDown()

    def test_callbacks_with_duplicate_prevention(self) -> None:
        # [FUNCTION DOC] Enter install_callbacks twice nested with the same trigger and compile id;
        # assert the start callback ran exactly once inside and the end callback exactly once after
        # exit.
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(frame_id=0, frame_compile_id=0)
        with (
            callback_handler.install_callbacks(trigger, compile_id),
            callback_handler.install_callbacks(trigger, compile_id),
        ):
            self._on_compile_start.assert_called_once()
        self._on_compile_end.assert_called_once()

    def test_counter(self) -> None:
        # [FUNCTION DOC] Assert the handler's private pending-callbacks counter is 1 inside
        # install_callbacks and back to 0 after exit.
        trigger = CallbackTrigger.DYNAMO
        compile_id = CompileId(frame_id=0, frame_compile_id=0)
        with callback_handler.install_callbacks(trigger, compile_id):
            self.assertEqual(
                callback_handler._CompilationCallbackHandler__pending_callbacks_counter,
                1,
            )
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    def test_counter_assertion(self) -> None:
        # [FUNCTION DOC] Decrement the private pending counter to -1, then enter install_callbacks
        # (counter returns to 0 on entry); assert exit raises AssertionError saying the counter
        # cannot become negative, and that the counter is 0 afterwards.
        # Note: the compile id is passed as a string here although the signature takes CompileId
        # elsewhere; harmless since callbacks only store it.
        callback_handler._CompilationCallbackHandler__pending_callbacks_counter -= 1
        with self.assertRaisesRegex(
            AssertionError, "Pending callbacks counter cannot become negative."
        ):
            trigger = CallbackTrigger.DYNAMO
            compile_id = CompileId(frame_id=0, frame_compile_id=0)
            with callback_handler.install_callbacks(trigger, str(compile_id)):
                pass
        self.assertEqual(
            callback_handler._CompilationCallbackHandler__pending_callbacks_counter, 0
        )

    @unittest.skipIf(not HAS_GPU, "requires GPU and Triton")
    @torch._inductor.config.patch(force_disable_caches=True)
    def test_triggers(self) -> None:
        # [FUNCTION DOC] Skipped without GPU and Triton; caches disabled. Register recording
        # callbacks, compile a small model containing a graph break with mode max-autotune, run
        # forward and backward and assert the exact sequence of trigger events (DYNAMO for frames
        # 0/0 and 1/0, then LAZY_BACKWARD). On CUDA with Triton, run again and assert
        # CUDAGRAPH_RECORDING events; a third run must emit no events.
        # KNOWN ISSUE: both recording callbacks are registered with on_compile_start, so the 'end='
        # lines come from a second start callback, not from real end callbacks.
        torch._dynamo.reset()
        order = []

        def on_start(args: CallbackArgs):
            # [FUNCTION DOC] HELPER: Append a string describing the start event arguments to the
            # shared order list.
            nonlocal order
            order.append(f"start={args}")

        def on_end(args: CallbackArgs):
            # [FUNCTION DOC] HELPER: Append a string describing the arguments to the shared order
            # list (registered as a start callback; see test_triggers note).
            nonlocal order
            order.append(f"end={args}")

        torch._dynamo.callback.on_compile_start(on_start)
        torch._dynamo.callback.on_compile_start(on_end)

        class TinyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module with linear, ReLU, linear layers and a graph break
            # between the ReLU and the second linear.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the two Linear(10, 10) layers and the ReLU.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Apply fc1 and relu, force a dynamo graph break, then apply
                # fc2 and return the result.
                temp = self.fc1(x)
                temp = self.relu(temp)
                torch._dynamo.graph_break()
                return self.fc2(temp)

        model = TinyModel().to(device_type)
        compiled_model = torch.compile(model, mode="max-autotune")  # noqa: UNSPECIFIED_BACKEND
        x = torch.randn(10, 10, device=device_type)

        loss = compiled_model(x).sum()
        loss.backward()
        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.DYNAMO: 1>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.LAZY_BACKWARD: 2>, compile_id='0/0')""",
        )
        order.clear()

        if not HAS_CUDA_AND_TRITON:
            return

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()

        self.assertExpectedInline(
            "\n".join(order),
            """\
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='1/0')
start=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')
end=CallbackArgs(callback_trigger=<CallbackTrigger.CUDAGRAPH_RECORDING: 4>, compile_id='0/0')""",
        )
        order.clear()

        compiled_model.zero_grad()
        loss = compiled_model(x).sum()
        loss.backward()
        self.assertEqual(len(order), 0)


if __name__ == "__main__":
    run_tests()
