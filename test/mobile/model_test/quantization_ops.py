# [MODULE DOC] Mobile model test fixtures for quantized operators. Provides a module calling
# quantized functional, embedding and transposed conv ops, plus factory classes producing
# dynamically quantized, statically quantized and fused-then-quantized models. Used by the
# mobile test model generator. No import-time side effects.
import torch
import torch.nn as nn


class GeneralQuantModule(torch.nn.Module):
    # [CLASS DOC] Module holding quantized Embedding, QFunctional and ConvTranspose 1d/2d/3d
    # layers, built with fixed sizes. forward exercises them on freshly quantized inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates a quantized embedding (10 entries, dim 12) with a fixed index
        # tensor, a QFunctional, and quantized transposed convs (16 to 33 channels, kernel 3,
        # stride 2) in 1d, 2d and 3d.
        super().__init__()
        self.embedding = torch.ao.nn.quantized.Embedding(
            num_embeddings=10, embedding_dim=12
        )
        self.embedding_input = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8])
        self.func = torch.ao.nn.quantized.QFunctional()
        self.conv1 = torch.ao.nn.quantized.ConvTranspose1d(16, 33, 3, stride=2)
        self.conv2 = torch.ao.nn.quantized.ConvTranspose2d(16, 33, 3, stride=2)
        self.conv3 = torch.ao.nn.quantized.ConvTranspose3d(16, 33, 3, stride=2)

    def forward(self):
        # [FUNCTION DOC] Quantizes small constant and random tensors per tensor, then calls the
        # QFunctional ops (add, cat, mul, add_relu, add_scalar, mul_scalar), the embedding and the
        # 1d and 2d transposed convs on quantized random inputs; the 3d conv call is commented out
        # because it failed on iOS. All results are passed to the len builtin.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists only so the ops appear in a scripted graph.
        a = torch.quantize_per_tensor(torch.tensor([3.0]), 1.0, 0, torch.qint32)
        b = torch.quantize_per_tensor(torch.tensor(4.0), 1.0, 0, torch.qint32)
        c = torch.quantize_per_tensor(
            torch.tensor([3.0]), torch.tensor(1.0), torch.tensor(0), torch.qint32
        )
        input1 = torch.randn(1, 16, 4)
        input2 = torch.randn(1, 16, 4, 4)
        return len(
            self.func.add(a, b),
            self.func.cat((a, a), 0),
            self.func.mul(a, b),
            self.func.add_relu(a, b),
            self.func.add_scalar(a, b),
            self.func.mul_scalar(a, b),
            self.embedding(self.embedding_input),
            self.conv1(
                torch.quantize_per_tensor(
                    input1, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            self.conv2(
                torch.quantize_per_tensor(
                    input2, scale=1.0, zero_point=0, dtype=torch.quint8
                )
            ),
            c,
            # self.conv3(torch.quantize_per_tensor(input3, scale=1.0, zero_point=0, dtype=torch.quint8)), # failed on iOS
        )


class DynamicQuantModule:
    # [CLASS DOC] Factory for a dynamically quantized model; the module to quantize is the
    # nested class M, instantiated and stored in the constructor.
    def __init__(self) -> None:
        # [FUNCTION DOC] Calls the parent initializer and stores a new M instance as module.
        super().__init__()
        self.module = self.M()

    def getModule(self):
        # [FUNCTION DOC] Returns the stored module converted with dynamic quantization to int8.
        return torch.ao.quantization.quantize_dynamic(self.module, dtype=torch.qint8)

    class M(torch.nn.Module):
        # [CLASS DOC] Float module combining RNN, GRU, LSTM (and their cells), identity, linear,
        # bilinear and transformer layers, used as input for dynamic quantization.
        def __init__(self) -> None:
            # [FUNCTION DOC] Creates the recurrent layers and cells, a list of linear style
            # layers and a list of three transformer variants, all with small fixed sizes.
            super(DynamicQuantModule.M, self).__init__()
            self.rnn = nn.RNN(4, 8, 2)
            self.rnncell = nn.RNNCell(4, 8)
            self.gru = nn.GRU(4, 8, 2)
            self.grucell = nn.GRUCell(4, 8)
            self.lstm = nn.LSTM(4, 8, 2)
            self.lstmcell = nn.LSTMCell(4, 8)
            self.linears = nn.ModuleList(
                [
                    nn.Identity(54),
                    nn.Linear(20, 20),
                    nn.Bilinear(20, 20, 40),
                ]
            )
            self.transformers = nn.ModuleList(
                [
                    nn.Transformer(
                        d_model=2, nhead=2, num_encoder_layers=1, num_decoder_layers=1
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=2, nhead=2), num_layers=1
                    ),
                    nn.TransformerDecoder(
                        nn.TransformerDecoderLayer(d_model=2, nhead=2), num_layers=1
                    ),
                ]
            )
            # self.a = torch.nn.utils.rnn.pad_sequence([torch.tensor([1,2,3]), torch.tensor([3,4])], batch_first=True)

        def forward(self):
            # [FUNCTION DOC] Takes no inputs. Creates random inputs and hidden states, runs each
            # recurrent layer and cell, the three transformer modules and three linear layers,
            # and returns the length of the tuple of all outputs (twelve).
            input = torch.randn(5, 3, 4)
            h = torch.randn(2, 3, 8)
            c = torch.randn(2, 3, 8)
            linear_input = torch.randn(32, 20)
            trans_input = torch.randn(1, 16, 2)
            tgt = torch.rand(1, 16, 2)

            return len(
                (
                    self.rnn(input, h),
                    self.rnncell(input[0], h[0]),
                    self.gru(input, h),
                    self.grucell(input[0], h[0]),
                    self.lstm(input, (h, c)),
                    # self.lstm(torch.nn.utils.rnn.pack_padded_sequence(self.a, lengths=torch.tensor([3,2,1])), (h, c)),
                    self.lstmcell(input[0], (h[0], c[0])),
                    self.transformers[0](trans_input, tgt),
                    self.transformers[1](trans_input),
                    self.transformers[2](trans_input, tgt),
                    self.linears[0](linear_input),
                    self.linears[1](linear_input),
                    self.linears[2](linear_input, linear_input),
                )
            )


class StaticQuantModule:
    # [CLASS DOC] Factory for a statically quantized model built from the nested class M. The
    # model is prepared and converted without any calibration pass.
    def getModule(self):
        # [FUNCTION DOC] Instantiates M, puts it in eval mode, sets the qnnpack default qconfig,
        # prepares and converts it, and returns the resulting int8 model.
        model_fp32 = self.M()
        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        return model_int8

    class M(torch.nn.Module):
        # [CLASS DOC] Float module with quant and dequant stubs, random 1d, 2d and 3d inputs
        # stored as attributes, and conv/norm/activation stacks for each dimensionality.
        def __init__(self) -> None:
            # [FUNCTION DOC] Creates the stubs, fixed random input tensors and four sequential
            # blocks: 1d conv with instance norm and hardswish, 2d conv with batch norm,
            # instance norm and leaky relu, 3d conv with norms and relu, and a linear layer.
            super(StaticQuantModule.M, self).__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.input1d = torch.randn(4, 2, 2)
            self.input2d = torch.randn((4, 2, 4, 4))
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            self.linear_input = torch.randn(32, 20)

            self.layer1 = nn.Sequential(
                nn.Conv1d(2, 2, 1), nn.InstanceNorm1d(1), nn.Hardswish()
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(2, 2, 1),
                nn.BatchNorm2d(2),
                nn.InstanceNorm2d(1),
                nn.LeakyReLU(),
            )
            self.layer3 = nn.Sequential(
                nn.Conv3d(2, 2, 1), nn.BatchNorm3d(2), nn.InstanceNorm3d(1), nn.ReLU()
            )
            self.layer4 = nn.Sequential(nn.Linear(4, 3))
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self):
            # [FUNCTION DOC] Takes no inputs. Passes the stored 1d input through layer one, the 2d
            # input through layers two and four, and the 3d input through layer three, each
            # wrapped by quantize and dequantize stubs. Returns the three outputs as a tuple.
            x = self.quant(self.input1d)
            x = self.layer1(x)
            x = self.dequant(x)

            y = self.input2d
            y = self.quant(y)
            y = self.layer2(y)
            y = self.layer4(y)
            y = self.dequant(y)

            z = self.quant(self.input3d)
            z = self.layer3(z)
            z = self.dequant(z)

            return (x, y, z)


class FusedQuantModule:
    # [CLASS DOC] Factory for a model with conv/relu and linear/relu pairs fused before static
    # quantization; the module is the nested class M.
    def getModule(self):
        # [FUNCTION DOC] Instantiates M in eval mode with the qnnpack default qconfig, fuses the
        # four conv or linear and relu pairs, prepares and converts it, and returns the int8
        # model (no calibration).
        model_fp32 = self.M()
        model_fp32.eval()
        model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")
        model_fp32_fused = torch.ao.quantization.fuse_modules(
            model_fp32,
            [
                ["conv1d", "relu1"],
                ["conv2d", "relu2"],
                ["conv3d", "relu3"],
                ["linear", "relu4"],
            ],
        )
        model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
        return model_int8

    class M(torch.nn.Module):
        # [CLASS DOC] Float module with quant and dequant stubs, random 1d, 2d and 3d inputs,
        # conv layers for each dimension, a linear layer and four relu layers to be fused.
        def __init__(self) -> None:
            # [FUNCTION DOC] Creates the stubs, three random inputs, conv1d, conv2d, conv3d,
            # a linear layer and four relu modules with fixed small sizes.
            super(FusedQuantModule.M, self).__init__()
            self.quant = torch.ao.quantization.QuantStub()
            self.input1d = torch.randn(4, 2, 2)
            self.input2d = torch.randn((4, 2, 4, 4))
            self.input3d = torch.randn(4, 2, 2, 4, 4)
            self.conv1d = nn.Conv1d(2, 2, 1)
            self.conv2d = nn.Conv2d(2, 2, 1)
            self.conv3d = nn.Conv3d(2, 2, 1)
            self.linear = nn.Linear(4, 2)
            self.relu1 = nn.ReLU()
            self.relu2 = nn.ReLU()
            self.relu3 = nn.ReLU()
            self.relu4 = nn.ReLU()
            self.dequant = torch.ao.quantization.DeQuantStub()

        def forward(self):
            # [FUNCTION DOC] Takes no inputs. Quantizes each stored input, applies conv and relu
            # (and for the 3d input also linear and a second relu), dequantizes, and returns
            # the three results as a tuple.
            x = self.input1d
            y = self.input2d
            z = self.input3d

            x = self.quant(x)
            x = self.conv1d(x)
            x = self.relu1(x)
            x = self.dequant(x)

            y = self.quant(y)
            y = self.conv2d(y)
            y = self.relu2(y)
            y = self.dequant(y)

            z = self.quant(z)
            z = self.conv3d(z)
            z = self.relu3(z)
            z = self.linear(z)
            z = self.relu4(z)
            z = self.dequant(z)

            return (x, y, z)
