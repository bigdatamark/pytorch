# Owner(s): ["oncall: quantization"]
# [MODULE DOC] Eager-mode numerics tests that compare float models against post-training
# quantized, fake-quantized (QAT) and partially quantized versions using
# signal-to-quantization-noise ratio thresholds. Uses the multi-op toy models from
# common_quantization. No import-time side effects; running the file directly raises an error
# pointing to the quantization test driver.

import torch
from torch.testing._internal.common_quantization import (
    ModelMultipleOps,
    ModelMultipleOpsNoAvgPool,
    QuantizationTestCase,
)
from torch.testing._internal.common_quantized import (
    override_quantized_engine,
    supported_qengines,
)


class TestModelNumericsEager(QuantizationTestCase):
    # [CLASS DOC] Numerics test case for eager-mode quantization flows. Each test seeds the random
    # generator, builds a toy model, and asserts that the SQNR in decibels between reference and
    # quantized outputs exceeds a threshold.
    def test_float_quant_compare_per_tensor(self):
        # [FUNCTION DOC] For every supported quantization engine, builds a float multi-op model in
        # eval mode, wraps it in a quantization wrapper with the default per-tensor qconfig, fuses
        # conv, bn and relu, prepares, calibrates on random data, converts, and asserts the SQNR
        # against the float output on separate eval data is above 30 dB.
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(42)
                my_model = ModelMultipleOps().to(torch.float32)
                my_model.eval()
                calib_data = torch.rand(1024, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(1, 3, 15, 15, dtype=torch.float32)
                out_ref = my_model(eval_data)
                qModel = torch.ao.quantization.QuantWrapper(my_model)
                qModel.eval()
                qModel.qconfig = torch.ao.quantization.default_qconfig
                torch.ao.quantization.fuse_modules(
                    qModel.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                torch.ao.quantization.prepare(qModel, inplace=True)
                qModel(calib_data)
                torch.ao.quantization.convert(qModel, inplace=True)
                out_q = qModel(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_ref) / torch.norm(out_ref - out_q)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 30 dB so that relative error is 1e-3 below the desired
                # output
                self.assertGreater(
                    SQNRdB,
                    30,
                    msg="Quantized model numerics diverge from float, expect SQNR > 30 dB",
                )

    def test_float_quant_compare_per_channel(self):
        # [FUNCTION DOC] Same flow as the per-tensor test (single default engine, no engine
        # override) but uses the default per-channel qconfig with different seed and data sizes,
        # and requires SQNR above 35 dB against the float output.
        # Test for per-channel Quant
        torch.manual_seed(67)
        my_model = ModelMultipleOps().to(torch.float32)
        my_model.eval()
        calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
        eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
        out_ref = my_model(eval_data)
        q_model = torch.ao.quantization.QuantWrapper(my_model)
        q_model.eval()
        q_model.qconfig = torch.ao.quantization.default_per_channel_qconfig
        torch.ao.quantization.fuse_modules(
            q_model.module, [["conv1", "bn1", "relu1"]], inplace=True
        )
        torch.ao.quantization.prepare(q_model)
        q_model(calib_data)
        torch.ao.quantization.convert(q_model)
        out_q = q_model(eval_data)
        SQNRdB = 20 * torch.log10(torch.norm(out_ref) / torch.norm(out_ref - out_q))
        # Quantized model output should be close to floating point model output numerically
        # Setting target SQNR to be 35 dB
        self.assertGreater(
            SQNRdB,
            35,
            msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
        )

    def test_fake_quant_true_quant_compare(self):
        # [FUNCTION DOC] For every supported engine, builds a QAT-prepared model of the
        # no-average-pool variant: fuses with the QAT fuser, prepares for QAT, then calibrates
        # with fake quant disabled and batch norm statistics frozen, enables fake quant and
        # disables observers, and requires SQNR above 35 dB versus the float model. Then converts
        # to a true quantized model and requires fake-quant versus true-quant outputs to agree
        # above 60 dB SQNR.
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                my_model.eval()
                out_ref = my_model(eval_data)
                fq_model = torch.ao.quantization.QuantWrapper(my_model)
                fq_model.train()
                fq_model.qconfig = torch.ao.quantization.default_qat_qconfig
                torch.ao.quantization.fuse_modules_qat(
                    fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                )
                torch.ao.quantization.prepare_qat(fq_model)
                fq_model.eval()
                fq_model.apply(torch.ao.quantization.disable_fake_quant)
                fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                fq_model(calib_data)
                fq_model.apply(torch.ao.quantization.enable_fake_quant)
                fq_model.apply(torch.ao.quantization.disable_observer)
                out_fq = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_ref) / torch.norm(out_ref - out_fq)
                )
                # Quantized model output should be close to floating point model output numerically
                # Setting target SQNR to be 35 dB
                self.assertGreater(
                    SQNRdB,
                    35,
                    msg="Quantized model numerics diverge from float, expect SQNR > 35 dB",
                )
                torch.ao.quantization.convert(fq_model)
                out_q = fq_model(eval_data)
                SQNRdB = 20 * torch.log10(
                    torch.norm(out_fq) / (torch.norm(out_fq - out_q) + 1e-10)
                )
                self.assertGreater(
                    SQNRdB,
                    60,
                    msg="Fake quant and true quant numerics diverge, expect SQNR > 60 dB",
                )

    # Test to compare weight only quantized model numerics and
    # activation only quantized model numerics with float
    def test_weight_only_activation_only_fakequant(self):
        # [FUNCTION DOC] For every supported engine, runs the fake-quant calibration flow twice,
        # once with the weight-only qconfig and once with the activation-only qconfig, and checks
        # SQNR against the float model exceeds the per-config target (35 and 45 dB paired by
        # iteration index).
        # KNOWN ISSUE: the qconfigs are held in a set, so the iteration order, and thus which
        # target is paired with which qconfig, is not defined by the code; the test only works if
        # the pairing happens to be right.
        for qengine in supported_qengines:
            with override_quantized_engine(qengine):
                torch.manual_seed(67)
                calib_data = torch.rand(2048, 3, 15, 15, dtype=torch.float32)
                eval_data = torch.rand(10, 3, 15, 15, dtype=torch.float32)
                qconfigset = {
                    torch.ao.quantization.default_weight_only_qconfig,
                    torch.ao.quantization.default_activation_only_qconfig,
                }
                SQNRTarget = [35, 45]
                for idx, qconfig in enumerate(qconfigset):
                    my_model = ModelMultipleOpsNoAvgPool().to(torch.float32)
                    my_model.eval()
                    out_ref = my_model(eval_data)
                    fq_model = torch.ao.quantization.QuantWrapper(my_model)
                    fq_model.train()
                    fq_model.qconfig = qconfig
                    torch.ao.quantization.fuse_modules_qat(
                        fq_model.module, [["conv1", "bn1", "relu1"]], inplace=True
                    )
                    torch.ao.quantization.prepare_qat(fq_model)
                    fq_model.eval()
                    fq_model.apply(torch.ao.quantization.disable_fake_quant)
                    fq_model.apply(torch.ao.nn.intrinsic.qat.freeze_bn_stats)
                    fq_model(calib_data)
                    fq_model.apply(torch.ao.quantization.enable_fake_quant)
                    fq_model.apply(torch.ao.quantization.disable_observer)
                    out_fq = fq_model(eval_data)
                    SQNRdB = 20 * torch.log10(
                        torch.norm(out_ref) / torch.norm(out_ref - out_fq)
                    )
                    self.assertGreater(
                        SQNRdB,
                        SQNRTarget[idx],
                        msg="Quantized model numerics diverge from float",
                    )


if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_quantization.py TESTNAME\n\n"
        "instead."
    )
