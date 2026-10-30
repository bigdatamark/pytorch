# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT tests for the script profiler (torch.jit._ScriptProfile). Defines a small
# LSTM-based Sequence module and a test class checking that profiles collect statistics only
# while enabled, that several profiles can run concurrently, and that an unused one is empty.
# Importing appends the test directory to sys.path; nothing runs on import. Direct execution
# is refused and redirects to test/test_jit.py.

import os
import sys

import torch
from torch import nn


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


class Sequence(nn.Module):
    # [CLASS DOC] HELPER: Test model with two LSTM cells and a linear layer, run over a
    # sequence split along dimension 1 to produce one prediction per time step.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates two LSTMCell layers (1->51 and 51->51) and a
        # Linear layer (51->1) as submodules.
        super().__init__()
        self.lstm1 = nn.LSTMCell(1, 51)
        self.lstm2 = nn.LSTMCell(51, 51)
        self.linear = nn.Linear(51, 1)

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Runs both LSTM cells over each column of the input, with
        # zero initial states sized by the batch, applies the linear layer to the second
        # cell's hidden state, and returns all per-step outputs concatenated along dim 1.
        outputs = []
        h_t = torch.zeros(input.size(0), 51)
        c_t = torch.zeros(input.size(0), 51)
        h_t2 = torch.zeros(input.size(0), 51)
        c_t2 = torch.zeros(input.size(0), 51)

        for input_t in input.split(1, dim=1):
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            output = self.linear(h_t2)
            outputs += [output]
        outputs = torch.cat(outputs, dim=1)
        return outputs


class TestScriptProfile(JitTestCase):
    # [CLASS DOC] Tests for the script profiler: enabling/disabling collection and the text
    # returned by dump_string, using scripted modules and scripted functions that call them.
    def test_basic(self):
        # [FUNCTION DOC] Profiles a scripted Sequence call between enable and disable and
        # asserts the dumped profile text is not empty.
        seq = torch.jit.script(Sequence())
        p = torch.jit._ScriptProfile()
        p.enable()
        seq(torch.rand((10, 100)))
        p.disable()
        self.assertNotEqual(p.dump_string(), "")

    def test_script(self):
        # [FUNCTION DOC] Profiles a scripted function (nested fn) that calls an eager
        # Sequence instance and asserts the profile text is not empty.
        seq = Sequence()

        p = torch.jit._ScriptProfile()
        p.enable()

        @torch.jit.script
        def fn():
            # [FUNCTION DOC] HELPER: Scripted function that runs the captured Sequence on
            # a random 10 by 100 input and discards the result.
            _ = seq(torch.rand((10, 100)))

        fn()
        p.disable()

        self.assertNotEqual(p.dump_string(), "")

    def test_multi(self):
        # [FUNCTION DOC] Enables five profiles on one scripted model, then repeatedly runs
        # the model, pops and disables one profile, and asserts each profile's text is
        # non-empty and differs from the previously popped one (each later one covers more runs).
        seq = torch.jit.script(Sequence())
        profiles = [torch.jit._ScriptProfile() for _ in range(5)]
        for p in profiles:
            p.enable()

        last = None
        while len(profiles) > 0:
            seq(torch.rand((10, 10)))
            p = profiles.pop()
            p.disable()
            stats = p.dump_string()
            self.assertNotEqual(stats, "")
            if last:
                self.assertNotEqual(stats, last)
            last = stats

    def test_section(self):
        # [FUNCTION DOC] Checks that only runs between enable and disable are counted: the
        # first profile covers a 100-wide run; a later run while disabled does not change
        # the text (s0 equals s1); re-enabling and running a 10-wide input changes it (s2).
        seq = Sequence()

        @torch.jit.script
        def fn(max: int):
            # [FUNCTION DOC] HELPER: Scripted function taking an int width that runs the
            # captured Sequence on a random input with that many columns.
            _ = seq(torch.rand((10, max)))

        p = torch.jit._ScriptProfile()
        p.enable()
        fn(100)
        p.disable()
        s0 = p.dump_string()

        fn(10)
        p.disable()
        s1 = p.dump_string()

        p.enable()
        fn(10)
        p.disable()
        s2 = p.dump_string()

        self.assertEqual(s0, s1)
        self.assertNotEqual(s1, s2)

    def test_empty(self):
        # [FUNCTION DOC] Asserts a profile that is enabled then disabled with nothing run
        # dumps an empty string.
        p = torch.jit._ScriptProfile()
        p.enable()
        p.disable()
        self.assertEqual(p.dump_string(), "")


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
