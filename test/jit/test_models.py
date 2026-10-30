# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests that trace (and in one case script) well-known model architectures and
# check that traced results, gradients and export/import round trips match eager behavior:
# DCGAN, neural style transfer, MNIST, a reinforcement-learning policy, SNLI classifier,
# super resolution, a time-sequence LSTM, VAE, torchvision ResNet18 and AlexNet. Each
# architecture has a static helper parameterized by device, with separate CPU and CUDA test
# methods. Import time: appends the test directory to sys.path, then tries to import
# torchvision and sets HAS_TORCHVISION (false on ImportError or RuntimeError); defines the
# skipIfNoTorchVision decorator.

import os
import sys
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_cuda import tf32_on_and_off
from torch.testing._internal.common_utils import (
    enable_profiling_mode_for_profiling_tests,
    GRAPH_EXECUTOR,
    ProfilingMode,
    raise_on_run_directly,
    set_default_dtype,
    slowTest,
    suppress_warnings,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA


try:
    import torchvision

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")


class MnistNet(nn.Module):
    # [CLASS DOC] MNIST-style convolutional classifier (two convolutions with max pooling, a
    # spatial dropout, two linear layers) used as a tracing fixture. Input is a batch of
    # single-channel 28x28 images; output is per-class log probabilities over 10 classes.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the two convolutions, the 2D dropout, and the 320-to-50 and
        # 50-to-10 linear layers.
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # [FUNCTION DOC] Applies conv, max pool, relu twice (the second with dropout), flattens to
        # 320 features, applies a relu linear layer, functional dropout following the module
        # training flag, the final linear layer, and returns log softmax over the class dimension.
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.reshape(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class TestModels(JitTestCase):
    # [CLASS DOC] JitTestCase that traces large reference models. Static helpers named with a
    # leading underscore take the test case as their first argument and a device, and are
    # called by CPU and CUDA test methods. CUDA variants disable export/import checking (known
    # CUDA limitation).
    @staticmethod
    def _test_dcgan_models(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Builds a DCGAN generator and discriminator and traces each with
        # checkTrace. The generator input is random noise of shape batch by latent size by one
        # by one; the discriminator input is the generator output. Export/import is checked only
        # if requested.
        class DCGANGenerator(nn.Module):
            # [CLASS DOC] HELPER: DCGAN generator, a stack of transposed convolutions, batch
            # norms and relus ending in tanh that upsamples a latent vector to a 64x64 image.
            def __init__(self, nz, ngf, nc):
                # [FUNCTION DOC] HELPER: Builds the sequential generator from latent size,
                # feature map size and channel count.
                super().__init__()
                self.main = nn.Sequential(
                    # input is Z, going into a convolution
                    nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(ngf * 8),
                    nn.ReLU(True),
                    # state size. (ngf*8) x 4 x 4
                    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 4),
                    nn.ReLU(True),
                    # state size. (ngf*4) x 8 x 8
                    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf * 2),
                    nn.ReLU(True),
                    # state size. (ngf*2) x 16 x 16
                    nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ngf),
                    nn.ReLU(True),
                    # state size. (ngf) x 32 x 32
                    nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                    nn.Tanh(),
                    # state size. (nc) x 64 x 64
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Runs the sequential stack on the input and returns
                # the image.
                return self.main(input)

        class DCGANDiscriminator(nn.Module):
            # [CLASS DOC] HELPER: DCGAN discriminator, strided convolutions with batch norm and
            # leaky relu, ending in a sigmoid that yields one probability per image.
            def __init__(self, nc, ndf):
                # [FUNCTION DOC] HELPER: Builds the sequential discriminator from channel count
                # and feature map size.
                super().__init__()
                self.main = nn.Sequential(
                    # input is (nc) x 64 x 64
                    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf) x 32 x 32
                    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 2),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*2) x 16 x 16
                    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 4),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*4) x 8 x 8
                    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(ndf * 8),
                    nn.LeakyReLU(0.2, inplace=True),
                    # state size. (ndf*8) x 4 x 4
                    nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                    nn.Sigmoid(),
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Runs the stack, reshapes to a column and drops the
                # singleton dimension, returning one score per batch element.
                return self.main(input).view(-1, 1).squeeze(1)

        bs, nz, ngf, nc, ndf = 5, 6, 9, 3, 10
        self.checkTrace(
            DCGANGenerator(nz, ngf, nc).to(device),
            (torch.rand(bs, nz, 1, 1, device=device),),
            export_import=check_export_import,
        )
        example_input = DCGANGenerator(nz, ngf, nc).to(device)(
            torch.rand(bs, nz, 1, 1, device=device)
        )
        self.checkTrace(
            DCGANDiscriminator(nc, ndf).to(device),
            (example_input,),
            export_import=check_export_import,
        )

    def test_dcgan_models(self):
        # [FUNCTION DOC] Runs the DCGAN trace helper on CPU with the default dtype set to double
        # (float can fail on precision) and export/import checking enabled.
        # Note: Can sometimes fail with low precision if run with float dtype
        with set_default_dtype(torch.double):
            self._test_dcgan_models(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_dcgan_models_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the DCGAN trace helper on CUDA in
        # double precision with export/import checking disabled because it does not work
        # for CUDA modules.
        # Note: Can sometimes fail with low precision if run with float dtype
        with set_default_dtype(torch.double):
            # XXX: export_import on CUDA modules doesn't work (#11480)
            self._test_dcgan_models(self, device="cuda", check_export_import=False)

    @staticmethod
    def _test_neural_style(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Defines the fast neural style transformer network and its
        # building blocks, then traces it with checkTrace on a random batch of five 3-channel
        # 16x16 images. The device argument is accepted but the model and input stay on the CPU.
        # Export/import is optional.
        class TransformerNet(torch.nn.Module):
            # [CLASS DOC] HELPER: Style transfer network: three convolutions with instance norm,
            # five residual blocks, two upsampling convolutions and a final convolution.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the conv layers, instance norms, residual
                # blocks, upsampling layers and relu used by forward.
                super().__init__()
                # Initial convolution layers
                self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
                self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
                self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
                self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
                self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
                # Residual layers
                self.res1 = ResidualBlock(128)
                self.res2 = ResidualBlock(128)
                self.res3 = ResidualBlock(128)
                self.res4 = ResidualBlock(128)
                self.res5 = ResidualBlock(128)
                # Upsampling Layers
                self.deconv1 = UpsampleConvLayer(
                    128, 64, kernel_size=3, stride=1, upsample=2
                )
                self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
                self.deconv2 = UpsampleConvLayer(
                    64, 32, kernel_size=3, stride=1, upsample=2
                )
                self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
                self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
                # Non-linearities
                self.relu = torch.nn.ReLU()

            def forward(self, X):
                # [FUNCTION DOC] HELPER: Applies conv, instance norm and relu three times, the
                # residual blocks, two upsampling stages with relu, then the final convolution;
                # returns the result.
                y = self.relu(self.in1(self.conv1(X)))
                y = self.relu(self.in2(self.conv2(y)))
                y = self.relu(self.in3(self.conv3(y)))
                y = self.res1(y)
                y = self.res2(y)
                y = self.res3(y)
                y = self.res4(y)
                y = self.res5(y)
                y = self.relu(self.in4(self.deconv1(y)))
                y = self.relu(self.in5(self.deconv2(y)))
                y = self.deconv3(y)
                return y

        class ConvLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Reflection padding followed by a convolution; padding is half
            # the kernel size.
            def __init__(self, in_channels, out_channels, kernel_size, stride):
                # [FUNCTION DOC] HELPER: Creates the reflection pad (kernel size integer-halved)
                # and the convolution.
                super().__init__()
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Pads the input then convolves it.
                out = self.reflection_pad(x)
                out = self.conv2d(out)
                return out

        class ResidualBlock(torch.nn.Module):
            # [CLASS DOC] HELPER: Residual block of two padded convolutions with instance norm,
            # adding the input back to the output.
            """ResidualBlock
            introduced in: https://arxiv.org/abs/1512.03385
            recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
            """

            def __init__(self, channels):
                # [FUNCTION DOC] HELPER: Creates two same-width conv layers, their instance
                # norms and relu.
                super().__init__()
                self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
                self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Conv, norm and relu, then conv and norm, and adds the
                # original input.
                residual = x
                out = self.relu(self.in1(self.conv1(x)))
                out = self.in2(self.conv2(out))
                out = out + residual
                return out

        class UpsampleConvLayer(torch.nn.Module):
            # [CLASS DOC] HELPER: Optionally upsamples the input by nearest-neighbor scaling,
            # then applies reflection padding and a convolution.
            """UpsampleConvLayer
            Upsamples the input and then does a convolution. This method gives better results
            compared to ConvTranspose2d.
            ref: http://distill.pub/2016/deconv-checkerboard/
            """

            def __init__(
                self, in_channels, out_channels, kernel_size, stride, upsample=None
            ):
                # [FUNCTION DOC] HELPER: Stores the upsample factor; creates an upsample layer
                # only if the factor is truthy; creates the reflection pad and convolution.
                super().__init__()
                self.upsample = upsample
                if upsample:
                    self.upsample_layer = torch.nn.Upsample(
                        mode="nearest", scale_factor=upsample
                    )
                reflection_padding = kernel_size // 2
                self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
                self.conv2d = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Upsamples if configured, then pads and convolves.
                x_in = x
                if self.upsample:
                    x_in = self.upsample_layer(x_in)
                out = self.reflection_pad(x_in)
                out = self.conv2d(out)
                return out

        self.checkTrace(
            TransformerNet(),
            (torch.rand(5, 3, 16, 16),),
            export_import=check_export_import,
        )

    @slowTest
    def test_neural_style(self):
        # [FUNCTION DOC] Slow test. Runs the neural style trace helper on CPU.
        self._test_neural_style(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_neural_style_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the neural style helper requesting CUDA with
        # export/import checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_neural_style(self, device="cuda", check_export_import=False)

    @unittest.skipIf(
        GRAPH_EXECUTOR == ProfilingMode.LEGACY, "Bug found in deprecated executor"
    )
    @staticmethod
    def _test_mnist(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Skipped under the legacy graph executor. Under profiling mode,
        # traces the MNIST net in eval mode (dropout would make results nondeterministic) on a
        # random batch of five 28x28 images on the device, with optional export/import checking.
        # eval() is present because dropout makes this nondeterministic
        with enable_profiling_mode_for_profiling_tests():
            self.checkTrace(
                MnistNet().to(device).eval(),
                (torch.rand(5, 1, 28, 28, device=device),),
                export_import=check_export_import,
            )

    def test_mnist(self):
        # [FUNCTION DOC] Runs the MNIST trace helper on CPU.
        self._test_mnist(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the MNIST trace helper on CUDA with
        # export/import checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_mnist(self, device="cuda", check_export_import=False)

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_mnist_training_leaks_no_memory_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Traces the MNIST net on CUDA without trace
        # checking (dropout makes it nondeterministic), runs one training warm-up so
        # parameters have gradient buffers, then asserts that five more training iterations
        # leak no CUDA tensors.
        net = MnistNet().cuda()
        # MnistNet uses dropout, don't check its trace
        traced_net = torch.jit.trace(
            net, [torch.randn(5, 1, 28, 28, device="cuda")], check_trace=False
        )

        def train(iters):
            # [FUNCTION DOC] HELPER: Runs the given number of iterations; each runs the
            # traced net on random CUDA input, backpropagates the summed output, and zeroes
            # the gradients.
            for _ in range(iters):
                # Get some fake data
                inp = torch.randn(5, 1, 28, 28, device="cuda")
                out = traced_net(inp)

                # Here's some fake loss
                out.sum().backward()

                # Zero out grads
                traced_net.zero_grad()

        # Set it up so the params have .grad fields so they are not reported as leaks
        train(1)

        with self.assertLeaksNoCudaTensors():
            train(5)

    @staticmethod
    def _test_reinforcement_learning(self, device, test_export_import=True):
        # [FUNCTION DOC] HELPER: Traces a small policy network under profiling mode on a random
        # 1x4 input on the device, with optional export/import checking.
        class Policy(nn.Module):
            # [CLASS DOC] HELPER: Policy network with two linear layers (4 to 128 to 2)
            # returning action probabilities.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two linear layers.
                super().__init__()
                self.affine1 = nn.Linear(4, 128)
                self.affine2 = nn.Linear(128, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies relu to the first layer, the second layer, and
                # softmax over the action dimension.
                x = F.relu(self.affine1(x))
                action_scores = self.affine2(x)
                return F.softmax(action_scores, dim=1)

        with enable_profiling_mode_for_profiling_tests():
            self.checkTrace(
                Policy().to(device),
                (torch.rand(1, 4, device=device),),
                export_import=test_export_import,
            )

    def test_reinforcement_learning(self):
        # [FUNCTION DOC] Runs the reinforcement learning trace helper on CPU.
        self._test_reinforcement_learning(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_reinforcement_learning_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the policy trace helper on CUDA with
        # export/import checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_reinforcement_learning(self, device="cuda", test_export_import=False)

    @staticmethod
    def _test_snli(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Defines an SNLI sentence-pair classifier and traces it on
        # random integer token matrices (premise 48x64, hypothesis 24x64) on the device, without
        # requiring input gradients, with optional export/import checking.
        class Bottle(nn.Module):
            # [CLASS DOC] HELPER: Mixin that applies the next class in the method order to
            # inputs with more than two dimensions by flattening the first two, then
            # restoring them.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: For inputs of at most two dimensions, delegates to the
                # parent forward; otherwise flattens the first two dimensions, delegates, and
                # reshapes the result back.
                if len(input.size()) <= 2:
                    return super().forward(input)
                size = input.size()[:2]
                out = super().forward(input.view(size[0] * size[1], -1))
                return out.view(size[0], size[1], -1)

        class Linear(Bottle, nn.Linear):
            # [CLASS DOC] HELPER: Linear layer combined with the flattening mixin.
            pass

        class Encoder(nn.Module):
            # [CLASS DOC] HELPER: LSTM sentence encoder built from a config object.
            def __init__(self, config):
                # [FUNCTION DOC] HELPER: Stores the config and builds the LSTM; input size
                # depends on whether projection is used; dropout is zero for a single layer.
                super().__init__()
                self.config = config
                input_size = config.d_proj if config.projection else config.d_embed
                dropout = 0 if config.n_layers == 1 else config.dp_ratio
                self.rnn = nn.LSTM(
                    input_size=input_size,
                    hidden_size=config.d_hidden,
                    num_layers=config.n_layers,
                    dropout=dropout,
                    bidirectional=config.birnn,
                )

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Runs the LSTM from zero initial states and returns the
                # last hidden state, or for a bidirectional encoder the last two directions
                # concatenated per batch element.
                batch_size = inputs.size()[1]
                state_shape = self.config.n_cells, batch_size, self.config.d_hidden
                h0 = c0 = inputs.new_zeros(state_shape)
                _, (ht, _) = self.rnn(inputs, (h0, c0))
                return (
                    ht[-1]
                    if not self.config.birnn
                    else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
                )

        class SNLIClassifier(nn.Module):
            # [CLASS DOC] HELPER: Sentence-pair classifier: embedding, optional projection,
            # shared encoder, and a feed-forward head over the concatenated encodings.
            def __init__(self, config):
                # [FUNCTION DOC] HELPER: Creates the embedding, projection, encoder, dropout and
                # relu, and a head of three hidden linear layers then an output layer; input
                # width doubles when bidirectional.
                super().__init__()
                self.config = config
                self.embed = nn.Embedding(config.n_embed, config.d_embed)
                self.projection = Linear(config.d_embed, config.d_proj)
                self.encoder = Encoder(config)
                self.dropout = nn.Dropout(p=config.dp_ratio)
                self.relu = nn.ReLU()
                seq_in_size = 2 * config.d_hidden
                if self.config.birnn:
                    seq_in_size *= 2
                lin_config = [seq_in_size] * 2
                self.out = nn.Sequential(
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(*lin_config),
                    self.relu,
                    self.dropout,
                    Linear(seq_in_size, config.d_out),
                )

            def forward(self, premise, hypothesis):
                # [FUNCTION DOC] HELPER: Embeds both sentences, optionally detaches the
                # embeddings and applies the projection with relu, encodes each, and returns
                # head scores of the concatenation.
                prem_embed = self.embed(premise)
                hypo_embed = self.embed(hypothesis)
                if self.config.fix_emb:
                    prem_embed = prem_embed.detach()
                    hypo_embed = hypo_embed.detach()
                if self.config.projection:
                    prem_embed = self.relu(self.projection(prem_embed))
                    hypo_embed = self.relu(self.projection(hypo_embed))
                premise = self.encoder(prem_embed)
                hypothesis = self.encoder(hypo_embed)
                scores = self.out(torch.cat([premise, hypothesis], 1))
                return scores

        class Config:
            # [CLASS DOC] HELPER: Plain constants holder for model hyperparameters
            # (vocab, embedding and hidden sizes, layer counts, flags). Dropout ratio is
            # zero for determinism.
            n_embed = 100
            d_embed = 100
            d_proj = 300
            dp_ratio = 0.0  # For deterministic testing TODO: change by fixing seed in checkTrace?
            d_hidden = 30
            birnn = True
            d_out = 300
            fix_emb = True
            projection = True
            n_layers = 2
            n_cells = 4  # 2 * n_layers because birnn = True

        premise = torch.LongTensor(48, 64).random_(0, 100).to(device)
        hypothesis = torch.LongTensor(24, 64).random_(0, 100).to(device)

        self.checkTrace(
            SNLIClassifier(Config()).to(device),
            (premise, hypothesis),
            inputs_require_grads=False,
            export_import=check_export_import,
        )

    @slowTest
    def test_snli(self):
        # [FUNCTION DOC] Slow test. Runs the SNLI trace helper on CPU.
        self._test_snli(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_snli_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the SNLI trace helper on CUDA with
        # export/import checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_snli(self, device="cuda", check_export_import=False)

    @staticmethod
    def _test_super_resolution(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Traces a super-resolution network with upscale factor four
        # on a random batch of five 1x32x32 inputs on the device, with optional
        # export/import checking.
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Four convolutions with relu followed by pixel shuffle
            # for upscaling.
            def __init__(self, upscale_factor):
                # [FUNCTION DOC] HELPER: Creates relu, four convolutions (the last outputs the
                # squared upscale factor channels) and the pixel shuffle.
                super().__init__()

                self.relu = nn.ReLU()
                self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
                self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
                self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
                self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
                self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies three conv+relu stages, the last conv, and
                # pixel shuffle.
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.pixel_shuffle(self.conv4(x))
                return x

        net = Net(upscale_factor=4).to(device)
        self.checkTrace(
            net,
            (torch.rand(5, 1, 32, 32, device=device),),
            export_import=check_export_import,
        )

    @slowTest
    def test_super_resolution(self):
        # [FUNCTION DOC] Slow test. Runs the super resolution trace helper on CPU.
        self._test_super_resolution(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    @tf32_on_and_off(0.02)
    def test_super_resolution_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA; decorated to run with tf32 on and off at
        # 0.02 tolerance. Runs the super resolution helper on CUDA with export/import
        # checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_super_resolution(self, device="cuda", check_export_import=False)

    @suppress_warnings
    def test_time_sequence_prediction(self):
        # [FUNCTION DOC] Traces a ScriptModule wrapper of a two-layer LSTM-cell sequence model
        # on a random 3x4 input, with warnings suppressed and emit hooks disabled, export/import
        # turned off.
        class Sequence(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule with two LSTM cells and a linear layer that
            # predicts the next value for each of four time steps and two further future steps.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two LSTM cells (1 to 51 and 51 to 51) and
                # the output linear.
                super().__init__()
                self.lstm1 = nn.LSTMCell(1, 51)
                self.lstm2 = nn.LSTMCell(51, 51)
                self.linear = nn.Linear(51, 1)

            @torch.jit.script_method
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Script method. Splits the input into four columns
                # (hard-coded count), steps both cells over each and appends the linear output,
                # then continues for two future steps feeding back the last output; returns all
                # outputs concatenated.
                # TODO: add future as input with default val
                # see https://github.com/pytorch/pytorch/issues/8724
                outputs = torch.empty((3, 0))
                h_t = torch.zeros((3, 51))
                c_t = torch.zeros((3, 51))
                h_t2 = torch.zeros((3, 51))
                c_t2 = torch.zeros((3, 51))

                output = torch.zeros([3, 51])
                future = 2

                # TODO: chunk call should appear as the for loop iterable
                # We hard-code it to 4 for now.
                a, b, c, d = input.chunk(input.size(1), dim=1)
                for input_t in (a, b, c, d):
                    h_t, c_t = self.lstm1(input_t, (h_t, c_t))
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    output = self.linear(h_t2)
                    outputs = torch.cat((outputs, output), 1)
                for _ in range(future):  # if we should predict the future
                    h_t, c_t = self.lstm1(output, (h_t, c_t))
                    h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
                    output = self.linear(h_t2)
                    outputs = torch.cat((outputs, output), 1)
                return outputs

        class Traced(nn.Module):
            # [CLASS DOC] HELPER: Eager wrapper that holds the script module so tracing goes
            # through it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the sequence script module.
                super().__init__()
                self.seq = Sequence()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the sequence module forward and returns
                # its result.
                return self.seq.forward(input)

        # disabled due to a jitter issues that will be fixed by using load/store in the compiler
        with torch._jit_internal._disable_emit_hooks():
            # TODO: toggle export_import once above issues are fixed
            self.checkTrace(Traced(), (torch.rand(3, 4),), export_import=False)

    @staticmethod
    def _test_vae(self, device, check_export_import=True):
        # [FUNCTION DOC] HELPER: Traces a VAE in eval mode (random sampling would be
        # nondeterministic) on a random 128x1x28x28 input under profiling mode, with optional
        # export/import checking.
        class VAE(nn.Module):
            # [CLASS DOC] HELPER: Variational autoencoder with a 784-400-20 encoder and
            # 20-400-784 decoder.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the encoder layer, mean and log-variance heads,
                # and decoder layers.
                super().__init__()

                self.fc1 = nn.Linear(784, 400)
                self.fc21 = nn.Linear(400, 20)
                self.fc22 = nn.Linear(400, 20)
                self.fc3 = nn.Linear(20, 400)
                self.fc4 = nn.Linear(400, 784)

            def encode(self, x):
                # [FUNCTION DOC] HELPER: Returns the mean and log variance heads from a relu
                # hidden layer.
                h1 = F.relu(self.fc1(x))
                return self.fc21(h1), self.fc22(h1)

            def reparameterize(self, mu, logvar):
                # [FUNCTION DOC] HELPER: In training mode returns a sample using noise scaled by
                # the standard deviation plus the mean; in eval mode returns the mean.
                if self.training:
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    return eps.mul(std).add_(mu)
                else:
                    return mu

            def decode(self, z):
                # [FUNCTION DOC] HELPER: Maps a latent vector through a relu layer and a sigmoid
                # output layer.
                h3 = F.relu(self.fc3(z))
                return torch.sigmoid(self.fc4(h3))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Flattens the input to 784 features, encodes, samples,
                # decodes, and returns the reconstruction with the mean and log variance.
                mu, logvar = self.encode(x.view(-1, 784))
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        with enable_profiling_mode_for_profiling_tests():
            # eval() is present because randn_like makes this nondeterministic
            self.checkTrace(
                VAE().to(device).eval(),
                (torch.rand(128, 1, 28, 28, device=device),),
                export_import=check_export_import,
            )

    def test_vae(self):
        # [FUNCTION DOC] Runs the VAE trace helper on CPU.
        self._test_vae(self, device="cpu")

    @unittest.skipIf(not RUN_CUDA, "no CUDA")
    def test_vae_cuda(self):
        # [FUNCTION DOC] Skipped without CUDA. Runs the VAE trace helper on CUDA with
        # export/import checking disabled.
        # XXX: export_import on CUDA modules doesn't work (#11480)
        self._test_vae(self, device="cuda", check_export_import=False)

    @slowTest
    @skipIfNoTorchVision
    def test_script_module_trace_resnet18(self):
        # [FUNCTION DOC] Slow test, skipped without torchvision. Traces ResNet18, makes an
        # export/import copy, then checks that outputs and input gradients from both match.
        m_orig = torch.jit.trace(
            torchvision.models.resnet18(), torch.ones(1, 3, 224, 224)
        )
        m_import = self.getExportImportCopy(m_orig)

        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        output_orig = m_orig(input)
        output_orig.sum().backward()
        grad_orig = input.grad.clone()
        input.grad.zero_()

        output_import = m_import(input)
        output_import.sum().backward()
        grad_import = input.grad.clone()

        self.assertEqual(output_orig, output_import)
        self.assertEqual(grad_orig, grad_import)

    @slowTest
    @skipIfNoTorchVision
    def test_script_module_script_resnet(self):
        # [FUNCTION DOC] Slow test, skipped without torchvision. Defines a ScriptModule ResNet18
        # from script-method blocks, makes an export/import copy, and checks outputs and input
        # gradients match.
        def conv1x1(in_planes, out_planes, stride=1):
            # [FUNCTION DOC] HELPER: Returns a bias-free one-by-one convolution with the given
            # planes and stride.
            """1x1 convolution"""
            return nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )

        def conv3x3(in_planes, out_planes, stride=1):
            # [FUNCTION DOC] HELPER: Returns a bias-free three-by-three convolution with
            # padding one.
            """3x3 convolution with padding"""
            return nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )

        class BasicBlock(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ResNet basic block as a ScriptModule: two 3x3 convs with batch
            # norm, an optional downsample of the shortcut and a residual add. Declares
            # downsample as a constant.
            expansion = 1
            __constants__ = ["downsample"]

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                # [FUNCTION DOC] HELPER: Builds the convs, norms, relu, stores the optional
                # downsample and stride.
                super().__init__()
                self.conv1 = conv3x3(inplanes, planes, stride)
                self.bn1 = nn.BatchNorm2d(planes)
                self.relu = nn.ReLU(inplace=True)
                self.conv2 = conv3x3(planes, planes)
                self.bn2 = nn.BatchNorm2d(planes)
                self.downsample = downsample
                self.stride = stride

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method. Conv, norm, relu, conv, norm, adds the
                # (optionally downsampled) input and applies relu.
                residual = x

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    residual = self.downsample(x)

                out += residual
                out = self.relu(out)

                return out

        class ResNet(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ResNet as a ScriptModule: stem, four residual layer groups,
            # global pooling and a linear classifier. Initializes convs with Kaiming normal and
            # norms to weight one, bias zero.
            __constants__ = ["layer1", "layer2", "layer3", "layer4"]

            def __init__(self, block, layers, num_classes=1000):
                # [FUNCTION DOC] HELPER: Builds the stem and layers from the block type
                # and per-stage block counts, then applies the weight initialization over
                # all submodules.
                super().__init__()
                self.inplanes = 64
                self.conv1 = nn.Conv2d(
                    3, 64, kernel_size=7, stride=2, padding=3, bias=False
                )
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.layer1 = self._make_layer(block, 64, layers[0])
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
                self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
                self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(512 * block.expansion, num_classes)

                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.kaiming_normal_(
                            m.weight, mode="fan_out", nonlinearity="relu"
                        )
                    elif isinstance(m, nn.BatchNorm2d):
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)

            def _make_layer(self, block, planes, blocks, stride=1):
                # [FUNCTION DOC] HELPER: Builds a sequential stage of blocks, using a one-by-one
                # conv downsample for the first block when stride or width changes. Mutates the
                # running input plane count.
                downsample = None
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        nn.BatchNorm2d(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes))

                return nn.Sequential(*layers)

            @torch.jit.script_method
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Script method. Runs stem, four stages, global average
                # pooling, flatten and the classifier.
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)

                return x

        resnet18 = ResNet(BasicBlock, [2, 2, 2, 2])

        resnet18_imported = self.getExportImportCopy(resnet18)

        input = torch.randn(1, 3, 224, 224, requires_grad=True)
        output_orig = resnet18(input)
        output_orig.sum().backward()
        grad_orig = input.grad.clone()
        input.grad.zero_()
        output_import = resnet18_imported(input)
        output_import.sum().backward()
        grad_import = input.grad.clone()

        self.assertEqual(output_orig, output_import)
        self.assertEqual(grad_orig, grad_import)

    @skipIfNoTorchVision
    def test_alexnet(self):
        # [FUNCTION DOC] Skipped without torchvision. Gets the traced graph of AlexNet and its
        # outputs with the random state forked, applies the common subexpression elimination
        # pass, builds a function from the graph and checks it reproduces the outputs.
        x = torch.ones(1, 3, 224, 224)
        model = torchvision.models.AlexNet()
        with torch.random.fork_rng(devices=[]):
            g, outputs, inputs = torch.jit._get_trace_graph(
                model, x, return_inputs=True
            )
        self.run_pass("cse", g)
        m = self.createFunctionFromGraph(g)
        with torch.random.fork_rng(devices=[]):
            self.assertEqual(outputs, m(*inputs))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
