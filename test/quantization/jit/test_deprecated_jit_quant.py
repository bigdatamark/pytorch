# Owner(s): ["oncall: quantization"]
# ruff: noqa: F841
# [MODULE DOC] Tests that the deprecated TorchScript quantized RNN and linear conversion helpers
# now fail. Each test builds float recurrent or linear modules, assigns fixed weights, and
# asserts that the torch.jit.quantized conversion function raises a RuntimeError saying it is no
# longer supported. Also checks erasing tensor shape information on a scripted class. Running
# the file directly raises a RuntimeError pointing to the quantization test entry.

import torch
from torch.testing._internal.common_quantization import skipIfNoFBGEMM
from torch.testing._internal.jit_utils import JitTestCase


class TestDeprecatedJitQuantized(JitTestCase):
    # [CLASS DOC] Test case (JitTestCase) for deprecated JIT quantization APIs. Holds no state;
    # its tests expect errors from removed conversion functions and exercise one shape-erasure
    # pass.
    @skipIfNoFBGEMM
    def test_rnn_cell_quantized(self):
        # [FUNCTION DOC] For each of LSTM cell, GRU cell and RNN cell (input and hidden size 2),
        # replaces the input and hidden weights with fixed non-trainable values (truncated to
        # hidden size times the gate count) and asserts that quantize_rnn_cell_modules raises
        # RuntimeError with a no longer supported message. Skipped without FBGEMM.
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTMCell(d_in, d_hid).float(),
            torch.nn.GRUCell(d_in, d_hid).float(),
            torch.nn.RNNCell(d_in, d_hid).float(),
        ]:
            if isinstance(cell, torch.nn.LSTMCell):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRUCell):
                num_chunks = 3
            elif isinstance(cell, torch.nn.RNNCell):
                num_chunks = 1

            # Replace parameter values s.t. the range of values is exactly
            # 255, thus we will have 0 quantization error in the quantized
            # GEMM call. This i s for testing purposes.
            #
            # Note that the current implementation does not support
            # accumulation values outside of the range representable by a
            # 16 bit integer, instead resulting in a saturated value. We
            # must take care that in our test we do not end up with a dot
            # product that overflows the int16 range, e.g.
            # (255*127+255*127) = 64770. So, we hardcode the test values
            # here and ensure a mix of signedness.
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            vals = vals[: d_hid * num_chunks]
            cell.weight_ih = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            with self.assertRaisesRegex(
                RuntimeError,
                "quantize_rnn_cell_modules function is no longer supported",
            ):
                cell = torch.jit.quantized.quantize_rnn_cell_modules(cell)

    @skipIfNoFBGEMM
    def test_rnn_quantized(self):
        # [FUNCTION DOC] For LSTM and GRU modules (sizes 2), overwrites layer-0 weights with
        # fixed values and asserts that quantize_rnn_modules raises RuntimeError saying it is no
        # longer supported, both for int8 and float16 dtype. Skipped without FBGEMM.
        d_in, d_hid = 2, 2

        for cell in [
            torch.nn.LSTM(d_in, d_hid).float(),
            torch.nn.GRU(d_in, d_hid).float(),
        ]:
            # Replace parameter values s.t. the range of values is exactly
            # 255, thus we will have 0 quantization error in the quantized
            # GEMM call. This i s for testing purposes.
            #
            # Note that the current implementation does not support
            # accumulation values outside of the range representable by a
            # 16 bit integer, instead resulting in a saturated value. We
            # must take care that in our test we do not end up with a dot
            # product that overflows the int16 range, e.g.
            # (255*127+255*127) = 64770. So, we hardcode the test values
            # here and ensure a mix of signedness.
            vals = [
                [100, -155],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
                [-155, 100],
                [-155, 100],
                [100, -155],
            ]
            if isinstance(cell, torch.nn.LSTM):
                num_chunks = 4
            elif isinstance(cell, torch.nn.GRU):
                num_chunks = 3
            vals = vals[: d_hid * num_chunks]
            cell.weight_ih_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )
            cell.weight_hh_l0 = torch.nn.Parameter(
                torch.tensor(vals, dtype=torch.float), requires_grad=False
            )

            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_int8 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.int8
                )

            with self.assertRaisesRegex(
                RuntimeError, "quantize_rnn_modules function is no longer supported"
            ):
                cell_fp16 = torch.jit.quantized.quantize_rnn_modules(
                    cell, dtype=torch.float16
                )

    if "fbgemm" in torch.backends.quantized.supported_engines:

        def test_quantization_modules(self):
            # [FUNCTION DOC] Defined only when fbgemm is a supported quantized engine. Builds a
            # one-linear model with fixed weights and zero bias, computes a float reference,
            # then asserts that quantize_linear_modules raises RuntimeError (no longer
            # supported) with default and float16 arguments.
            K1, N1 = 2, 2

            class FooBar(torch.nn.Module):
                # [CLASS DOC] HELPER: Single-layer model holding one float linear layer of sizes
                # K1 by N1.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer attribute.
                    super().__init__()
                    self.linear1 = torch.nn.Linear(K1, N1).float()

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x and returns the
                    # result.
                    x = self.linear1(x)
                    return x

            fb = FooBar()
            fb.linear1.weight = torch.nn.Parameter(
                torch.tensor([[-150, 100], [100, -150]], dtype=torch.float),
                requires_grad=False,
            )
            fb.linear1.bias = torch.nn.Parameter(
                torch.zeros_like(fb.linear1.bias), requires_grad=False
            )

            x = (torch.rand(1, K1).float() - 0.5) / 10.0
            value = torch.tensor([[100, -150]], dtype=torch.float)

            y_ref = fb(value)

            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_int8 = torch.jit.quantized.quantize_linear_modules(fb)

            with self.assertRaisesRegex(
                RuntimeError, "quantize_linear_modules function is no longer supported"
            ):
                fb_fp16 = torch.jit.quantized.quantize_linear_modules(fb, torch.float16)

    @skipIfNoFBGEMM
    def test_erase_class_tensor_shapes(self):
        # [FUNCTION DOC] Scripts a module that stores a packed quantized linear weight, with get
        # and set state and a weight property, with emit hooks disabled, then runs the erase-
        # shape-information pass on its graph. Passes if nothing raises. Skipped without FBGEMM.
        class Linear(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a prepacked quantized int8 linear weight, with
            # state saving, forward returning the packed parameter, and a weight property.
            def __init__(self, in_features, out_features):
                # [FUNCTION DOC] HELPER: Creates an empty int8 affine-quantized weight of the
                # given shape (scale 1, zero point 0), prepacks it with the quantized linear
                # prepack op and stores it.
                super().__init__()
                qweight = torch._empty_affine_quantized(
                    [out_features, in_features],
                    scale=1,
                    zero_point=0,
                    dtype=torch.qint8,
                )
                self._packed_weight = torch.ops.quantized.linear_prepack(qweight)

            @torch.jit.export
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: Exported method used when scripting: returns a tuple of
                # the unpacked weight tensor and the training flag.
                return (
                    torch.ops.quantized.linear_unpack(self._packed_weight)[0],
                    self.training,
                )

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the packed weight parameter.
                return self._packed_weight

            @torch.jit.export
            def __setstate__(self, state):
                # [FUNCTION DOC] HELPER: Exported method: restores the packed weight by
                # prepacking the first state entry and sets the training flag from the second
                # entry.
                self._packed_weight = torch.ops.quantized.linear_prepack(state[0])
                self.training = state[1]

            @property
            def weight(self):
                # [FUNCTION DOC] HELPER: Property getter: returns the unpacked weight tensor of
                # the packed parameter.
                return torch.ops.quantized.linear_unpack(self._packed_weight)[0]

            @weight.setter
            def weight(self, w):
                # [FUNCTION DOC] HELPER: Property setter: prepacks the given tensor w and stores
                # it as the packed weight.
                self._packed_weight = torch.ops.quantized.linear_prepack(w)

        with torch._jit_internal._disable_emit_hooks():
            x = torch.jit.script(Linear(10, 10))
            torch._C._jit_pass_erase_shape_information(x.graph)


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
