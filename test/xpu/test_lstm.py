# Owner(s): ["module: intel"]
# [MODULE DOC] XPU-only LSTM smoke tests. A single test class builds small and large torch.nn.LSTM
# modules on the XPU device across dtypes, directions, layer counts and batch_first settings, and
# checks output shapes and absence of NaNs, plus a determinism check. At import it instantiates
# the device-generic class for XPU only; run as a script it invokes the test runner.

import torch
from torch.testing._internal.common_device_type import (
    dtypes,
    dtypesIfXPU,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_nn import NNTestCase
from torch.testing._internal.common_utils import run_tests


class TestLstmXpu(NNTestCase):
    # [CLASS DOC] Device-generic test case (derived from NNTestCase) holding the XPU LSTM tests;
    # the shared helper builds and checks a module, and each test method calls it with one
    # configuration. Instantiated for the xpu device only by the call at the bottom of the module.
    def _test_lstm(
        self,
        device,
        dtype,
        input_size,
        hidden_size,
        num_layers,
        bidirectional,
        batch_first,
    ):
        # [FUNCTION DOC] HELPER: Builds an LSTM on the given device and dtype in eval mode with
        # the given sizes, layers, direction and batch_first options, runs it without gradients on
        # random input (batch 2, sequence length 10) and random initial hidden and cell states,
        # then asserts that the output shape is as expected for the layout, that final hidden and
        # cell states have layers times directions by batch by hidden shape, and that none of the
        # outputs contain NaN. Fails via assertions.
        batch = 2
        seq_len = 10
        num_directions = 2 if bidirectional else 1

        lstm = (
            torch.nn.LSTM(
                input_size,
                hidden_size,
                num_layers=num_layers,
                bidirectional=bidirectional,
                batch_first=batch_first,
            )
            .to(device, dtype)
            .eval()
        )

        if batch_first:
            x = torch.randn(batch, seq_len, input_size, device=device, dtype=dtype)
        else:
            x = torch.randn(seq_len, batch, input_size, device=device, dtype=dtype)

        h0 = torch.randn(
            num_layers * num_directions,
            batch,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        c0 = torch.randn(
            num_layers * num_directions,
            batch,
            hidden_size,
            device=device,
            dtype=dtype,
        )

        with torch.no_grad():
            out, (hn, cn) = lstm(x, (h0, c0))

        expected_out_dim = (
            (batch, seq_len, hidden_size * num_directions)
            if batch_first
            else (seq_len, batch, hidden_size * num_directions)
        )
        self.assertEqual(out.shape, expected_out_dim)
        self.assertEqual(hn.shape, (num_layers * num_directions, batch, hidden_size))
        self.assertEqual(cn.shape, (num_layers * num_directions, batch, hidden_size))

        self.assertFalse(out.isnan().any())
        self.assertFalse(hn.isnan().any())
        self.assertFalse(cn.isnan().any())

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_basic(self, device, dtype):
        # [FUNCTION DOC] Checks a single-layer unidirectional batch-first LSTM (input 64, hidden
        # 32) for each tested dtype (float32 and bfloat16, plus float16 on XPU).
        self._test_lstm(device, dtype, 64, 32, 1, False, True)

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_bidirectional(self, device, dtype):
        # [FUNCTION DOC] Checks a single-layer bidirectional batch-first LSTM (input 64, hidden
        # 32) for each tested dtype.
        self._test_lstm(device, dtype, 64, 32, 1, True, True)

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_multilayer(self, device, dtype):
        # [FUNCTION DOC] Checks a two-layer unidirectional batch-first LSTM (input 64, hidden 32)
        # for each tested dtype.
        self._test_lstm(device, dtype, 64, 32, 2, False, True)

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_multilayer_bidirectional(self, device, dtype):
        # [FUNCTION DOC] Checks a two-layer bidirectional batch-first LSTM (input 64, hidden 32)
        # for each tested dtype.
        self._test_lstm(device, dtype, 64, 32, 2, True, True)

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_batch_first_false(self, device, dtype):
        # [FUNCTION DOC] Checks a single-layer unidirectional LSTM with sequence-first layout
        # (input 64, hidden 32) for each tested dtype.
        self._test_lstm(device, dtype, 64, 32, 1, False, False)

    @dtypesIfXPU(torch.float32, torch.float16, torch.bfloat16)
    @dtypes(torch.float32, torch.bfloat16)
    def test_lstm_large_hidden(self, device, dtype):
        # [FUNCTION DOC] Checks a large two-layer bidirectional sequence-first LSTM with input and
        # hidden size 2048 for each tested dtype.
        self._test_lstm(device, dtype, 2048, 2048, 2, True, False)

    def test_lstm_deterministic(self, device):
        # [FUNCTION DOC] Runs a bidirectional float32 LSTM (input 64, hidden 32) twice on the same
        # random input, once with default settings and once with deterministic algorithms enabled,
        # and asserts the outputs are equal. Side effect: the deterministic algorithms flag is
        # turned off afterwards in a cleanup block regardless of its prior value.
        lstm = torch.nn.LSTM(64, 32, 1, bidirectional=True).to(device).eval()
        x = torch.randn(10, 2, 64, device=device)

        with torch.no_grad():
            out1, _ = lstm(x)

        torch.use_deterministic_algorithms(True)
        try:
            with torch.no_grad():
                out2, _ = lstm(x)
            self.assertEqual(out1, out2)
        finally:
            torch.use_deterministic_algorithms(False)


instantiate_device_type_tests(TestLstmXpu, globals(), only_for="xpu", allow_xpu=True)

if __name__ == "__main__":
    run_tests()
