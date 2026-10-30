# Owner(s): ["module: onnx"]
# [MODULE DOC] Legacy export tests that trace well-known model definitions (local model_defs
# networks and torchvision models) to ONNX and verify numerics via the verify helper using the
# caffe2 ONNX backend. Defines a device-moving helper toC, batch size constant, and one test class
# with a shared exportTest method; many tests are skipped or gated on opset version. Imports
# torchvision, model_defs and verify at import time. Runs the common test runner when executed as
# a script.

import unittest

import pytorch_test_common
from model_defs.dcgan import _netD, _netG, bsz, imgsz, nz, weights_init
from model_defs.emb_seq import EmbeddingNetwork1, EmbeddingNetwork2
from model_defs.mnist import MNIST
from model_defs.op_test import ConcatNet, DummyNet, FakeQuantNet, PermuteNet, PReluNet
from model_defs.squeezenet import SqueezeNet
from model_defs.srresnet import SRResNet
from model_defs.super_resolution import SuperResolutionNet
from pytorch_test_common import skipIfUnsupportedMinOpsetVersion, skipScriptTest
from torchvision.models import shufflenet_v2_x1_0
from torchvision.models.alexnet import alexnet
from torchvision.models.densenet import densenet121
from torchvision.models.googlenet import googlenet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet1_0
from torchvision.models.mobilenet import mobilenet_v2
from torchvision.models.resnet import resnet50
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from torchvision.models.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from torchvision.models.video import mc3_18, r2plus1d_18, r3d_18
from verify import verify

import torch
from torch.ao import quantization
from torch.autograd import Variable
from torch.onnx import OperatorExportTypes
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack


if torch.cuda.is_available():

    def toC(x):
        # [FUNCTION DOC] HELPER: CUDA variant, defined when CUDA is available at import time: moves
        # the argument to the GPU and returns it.
        return x.cuda()

else:

    def toC(x):
        # [FUNCTION DOC] HELPER: CPU variant, defined when CUDA is unavailable: returns the
        # argument unchanged.
        return x


BATCH_SIZE = 2


class TestModels(pytorch_test_common.ExportTestCase):
    # [CLASS DOC] Export-and-verify tests for many models. Class attributes: opset 9 (because
    # caffe2 cannot handle the default) and initializers not kept as graph inputs. All tests
    # funnel through exportTest.
    opset_version = 9  # Caffe2 doesn't support the default.
    keep_initializers_as_inputs = False

    def exportTest(self, model, inputs, rtol=1e-2, atol=1e-7, **kwargs):
        # [FUNCTION DOC] Shared checker. Imports the caffe2 ONNX backend lazily, puts the model in
        # eval export mode, traces it to an ONNX-type graph with the internal trace utility, lints
        # the graph, and calls verify with the backend, tolerances rtol and atol, and the class
        # opset version. Extra keyword arguments are accepted but ignored.
        # KNOWN ISSUE: depends on the caffe2 backend module, which fails to import where caffe2 is
        #   unavailable, so every test using it would error there.
        import caffe2.python.onnx.backend as backend

        with torch.onnx.select_model_mode_for_export(
            model, torch.onnx.TrainingMode.EVAL
        ):
            graph = torch.onnx.utils._trace(model, inputs, OperatorExportTypes.ONNX)
            torch._C._jit_pass_lint(graph)
            verify(
                model,
                inputs,
                backend,
                rtol=rtol,
                atol=atol,
                opset_version=self.opset_version,
            )

    def test_ops(self):
        # [FUNCTION DOC] Exports the dummy op-test network on a batch of ones of shape batch by 3
        # by 224 by 224.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(DummyNet()), toC(x))

    def test_prelu(self):
        # [FUNCTION DOC] Exports the PReLU network on a batch of ones of shape batch by 3 by 224
        # by 224.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(PReluNet(), x)

    @skipScriptTest()
    def test_concat(self):
        # [FUNCTION DOC] Exports the concat network with a tuple of two random batch-by-3 inputs;
        # skipped in scripting mode.
        input_a = Variable(torch.randn(BATCH_SIZE, 3))
        input_b = Variable(torch.randn(BATCH_SIZE, 3))
        inputs = ((toC(input_a), toC(input_b)),)
        self.exportTest(toC(ConcatNet()), inputs)

    def test_permute(self):
        # [FUNCTION DOC] Exports the permute network on a random batch by 3 by 10 by 12 input.
        x = Variable(torch.randn(BATCH_SIZE, 3, 10, 12))
        self.exportTest(PermuteNet(), x)

    @skipScriptTest()
    def test_embedding_sequential_1(self):
        # [FUNCTION DOC] Exports the first embedding network on random integer indices below 10 of
        # shape batch by 3; skipped in scripting mode.
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        self.exportTest(EmbeddingNetwork1(), x)

    @skipScriptTest()
    def test_embedding_sequential_2(self):
        # [FUNCTION DOC] Exports the second embedding network on random integer indices below 10
        # of shape batch by 3; skipped in scripting mode.
        x = Variable(torch.randint(0, 10, (BATCH_SIZE, 3)))
        self.exportTest(EmbeddingNetwork2(), x)

    @unittest.skip("This model takes too much memory")
    def test_srresnet(self):
        # [FUNCTION DOC] Exports the super-resolution-residual network (rescale factor 4, 64
        # filters, 8 blocks) on a ones image; unconditionally skipped because it uses too much
        # memory.
        x = Variable(torch.randn(1, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(SRResNet(rescale_factor=4, n_filters=64, n_blocks=8)), toC(x)
        )

    @skipIfNoLapack
    def test_super_resolution(self):
        # [FUNCTION DOC] Exports the super resolution network (upscale factor 3) on a
        # single-channel ones image with absolute tolerance 1e-6; requires LAPACK.
        x = Variable(torch.randn(BATCH_SIZE, 1, 224, 224).fill_(1.0))
        self.exportTest(toC(SuperResolutionNet(upscale_factor=3)), toC(x), atol=1e-6)

    def test_alexnet(self):
        # [FUNCTION DOC] Exports the AlexNet model on a ones batch of 3-channel 224 by 224 images.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(alexnet()), toC(x))

    def test_mnist(self):
        # [FUNCTION DOC] Exports the MNIST network on a ones batch of single-channel 28 by 28
        # images.
        x = Variable(torch.randn(BATCH_SIZE, 1, 28, 28).fill_(1.0))
        self.exportTest(toC(MNIST()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg16(self):
        # [FUNCTION DOC] Exports the 16-layer VGG model on a ones batch; unconditionally skipped
        # for memory.
        # VGG 16-layer model (configuration "D")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg16_bn(self):
        # [FUNCTION DOC] Exports the 16-layer VGG model with batch norm on a ones batch;
        # unconditionally skipped for memory.
        # VGG 16-layer model (configuration "D") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg16_bn()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg19(self):
        # [FUNCTION DOC] Exports the 19-layer VGG model on a ones batch; unconditionally skipped
        # for memory.
        # VGG 19-layer model (configuration "E")
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19()), toC(x))

    @unittest.skip("This model takes too much memory")
    def test_vgg19_bn(self):
        # [FUNCTION DOC] Exports the 19-layer VGG model with batch norm on a ones batch;
        # unconditionally skipped for memory.
        # VGG 19-layer model (configuration "E") with batch normalization
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(vgg19_bn()), toC(x))

    def test_resnet(self):
        # [FUNCTION DOC] Exports ResNet50 on a ones batch with absolute tolerance 1e-6.
        # ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(resnet50()), toC(x), atol=1e-6)

    # This test is numerically unstable. Sporadic single element mismatch occurs occasionally.
    def test_inception(self):
        # [FUNCTION DOC] Exports Inception v3 on a random batch of 299 by 299 images, passing an
        # acceptable error percentage of 0.01.
        # KNOWN ISSUE: exportTest accepts but ignores extra keyword arguments, so that error
        #   allowance has no effect; the existing comment says the test is numerically unstable.
        x = Variable(torch.randn(BATCH_SIZE, 3, 299, 299))
        self.exportTest(toC(inception_v3()), toC(x), acceptable_error_percentage=0.01)

    def test_squeezenet(self):
        # [FUNCTION DOC] Exports SqueezeNet twice on ones batches.
        # KNOWN ISSUE: both models are built with version 1.1 although the first variable and
        #   comment refer to version 1.0, so the same version is tested twice.
        # SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
        # <0.5MB model size
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_0 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_0), toC(x))

        # SqueezeNet 1.1 has 2.4x less computation and slightly fewer params
        # than SqueezeNet 1.0, without sacrificing accuracy.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        sqnet_v1_1 = SqueezeNet(version=1.1)
        self.exportTest(toC(sqnet_v1_1), toC(x))

    def test_densenet(self):
        # [FUNCTION DOC] Exports DenseNet-121 on a ones batch with rtol 1e-2 and atol 1e-5.
        # Densenet-121 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(densenet121()), toC(x), rtol=1e-2, atol=1e-5)

    @skipScriptTest()
    def test_dcgan_netD(self):
        # [FUNCTION DOC] Builds the DCGAN discriminator, initializes its weights with the
        # weights_init function, and exports it on a normal-random image batch; skipped in
        # scripting mode.
        netD = _netD(1)
        netD.apply(weights_init)
        input = Variable(torch.empty(bsz, 3, imgsz, imgsz).normal_(0, 1))
        self.exportTest(toC(netD), toC(input))

    @skipScriptTest()
    def test_dcgan_netG(self):
        # [FUNCTION DOC] Builds the DCGAN generator, initializes its weights with weights_init,
        # and exports it on a normal-random latent batch of shape batch by latent size by 1 by 1;
        # skipped in scripting mode.
        netG = _netG(1)
        netG.apply(weights_init)
        input = Variable(torch.empty(bsz, nz, 1, 1).normal_(0, 1))
        self.exportTest(toC(netG), toC(input))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quant(self):
        # [FUNCTION DOC] Exports the fake-quantization network on a ones batch; requires opset 10
        # or newer.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(FakeQuantNet()), toC(x))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_resnet_pertensor(self):
        # [FUNCTION DOC] Prepares ResNet50 for quantization-aware training with per-tensor fake
        # quantization for activation and weights, enables observers and fake quant, runs one
        # forward pass, recomputes qparams for each fake-quant module, disables observers, then
        # exports it. Requires opset 10 or newer.
        # Quantize ResNet50 model
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        qat_resnet50 = resnet50()

        # Use per tensor for weight. Per channel support will come with opset 13
        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_fake_quant,
        )
        quantization.prepare_qat(qat_resnet50, inplace=True)
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        _ = qat_resnet50(x)
        for module in qat_resnet50.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        self.exportTest(toC(qat_resnet50), toC(x))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_resnet_per_channel(self):
        # [FUNCTION DOC] Same as the per-tensor QAT test but the weight fake quantization is
        # per-channel; the input is a plain ones tensor. Requires opset 13 or newer.
        # Quantize ResNet50 model
        x = torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0)
        qat_resnet50 = resnet50()

        qat_resnet50.qconfig = quantization.QConfig(
            activation=quantization.default_fake_quant,
            weight=quantization.default_per_channel_weight_fake_quant,
        )
        quantization.prepare_qat(qat_resnet50, inplace=True)
        qat_resnet50.apply(torch.ao.quantization.enable_observer)
        qat_resnet50.apply(torch.ao.quantization.enable_fake_quant)

        _ = qat_resnet50(x)
        for module in qat_resnet50.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        qat_resnet50.apply(torch.ao.quantization.disable_observer)

        self.exportTest(toC(qat_resnet50), toC(x))

    @skipScriptTest(skip_before_opset_version=15, reason="None type in outputs")
    def test_googlenet(self):
        # [FUNCTION DOC] Exports GoogLeNet on a ones batch with rtol 1e-3, atol 1e-5; script-mode
        # runs are skipped before opset 15 because of None outputs.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(googlenet()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mnasnet(self):
        # [FUNCTION DOC] Exports MNASNet 1.0 on a ones batch with rtol 1e-3, atol 1e-5.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(mnasnet1_0()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mobilenet(self):
        # [FUNCTION DOC] Exports MobileNet v2 on a ones batch with rtol 1e-3, atol 1e-5.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(mobilenet_v2()), toC(x), rtol=1e-3, atol=1e-5)

    @skipScriptTest()  # prim_data
    def test_shufflenet(self):
        # [FUNCTION DOC] Exports ShuffleNet v2 (1.0x) on a ones batch with rtol 1e-3, atol 1e-5;
        # skipped in scripting mode.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(toC(shufflenet_v2_x1_0()), toC(x), rtol=1e-3, atol=1e-5)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_fcn(self):
        # [FUNCTION DOC] Exports the FCN ResNet101 segmentation model without pretrained weights
        # on a ones batch with rtol 1e-3, atol 1e-5; requires opset 11 or newer.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(fcn_resnet101(weights=None, weights_backbone=None)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_deeplab(self):
        # [FUNCTION DOC] Exports the DeepLabV3 ResNet101 model without pretrained weights on a
        # ones batch with rtol 1e-3, atol 1e-5; requires opset 11 or newer.
        x = Variable(torch.randn(BATCH_SIZE, 3, 224, 224).fill_(1.0))
        self.exportTest(
            toC(deeplabv3_resnet101(weights=None, weights_backbone=None)),
            toC(x),
            rtol=1e-3,
            atol=1e-5,
        )

    def test_r3d_18_video(self):
        # [FUNCTION DOC] Exports the R3D-18 video model on a ones clip of shape 1 by 3 by 4 by 112
        # by 112 with rtol 1e-3, atol 1e-5.
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(r3d_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_mc3_18_video(self):
        # [FUNCTION DOC] Exports the MC3-18 video model on a ones clip of shape 1 by 3 by 4 by 112
        # by 112 with rtol 1e-3, atol 1e-5.
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(mc3_18()), toC(x), rtol=1e-3, atol=1e-5)

    def test_r2plus1d_18_video(self):
        # [FUNCTION DOC] Exports the R(2+1)D-18 video model on a ones clip of shape 1 by 3 by 4 by
        # 112 by 112 with rtol 1e-3, atol 1e-5.
        x = Variable(torch.randn(1, 3, 4, 112, 112).fill_(1.0))
        self.exportTest(toC(r2plus1d_18()), toC(x), rtol=1e-3, atol=1e-5)


if __name__ == "__main__":
    common_utils.run_tests()
