# [MODULE DOC] Mobile model test fixtures for torch.nn modules and functional ops, one nn.Module
# per documentation family (convolution, pooling, padding, normalization, activation, recurrent,
# transformer, linear, dropout, sparse, distance, loss, vision, shuffle, utils). Each forward
# runs the family on random inputs so its ops appear in generated test models. No import-time
# side effects besides imports.
import torch
import torch.nn as nn
import torch.nn.functional as F


# https://pytorch.org/docs/stable/nn.html
class NNConvolutionModule(torch.nn.Module):
    # [CLASS DOC] Holds 1d, 2d and 3d random inputs and lists of convolution style layers
    # (conv, transposed conv, fold, unfold) matching each input rank.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates random inputs of rank three, four and five and the three module
        # lists with fixed channel counts and kernel sizes.
        super().__init__()
        self.input1d = torch.randn(1, 4, 36)
        self.input2d = torch.randn(1, 4, 30, 10)
        self.input3d = torch.randn(1, 4, 10, 4, 4)
        self.module1d = nn.ModuleList(
            [
                nn.Conv1d(4, 33, 3),
                nn.ConvTranspose1d(4, 33, 3),
                nn.Fold(output_size=(5, 10), kernel_size=(2, 2)),
            ]
        )
        self.module2d = nn.ModuleList(
            [
                nn.Conv2d(4, 33, 3),
                nn.ConvTranspose2d(4, 33, 3),
                nn.Unfold(kernel_size=3),
            ]
        )
        self.module3d = nn.ModuleList(
            [
                nn.Conv3d(4, 33, 2),
                nn.ConvTranspose3d(4, 33, 3),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Applies every layer in each list to the matching input
        # and returns the length of the tuple of the three result lists, which is three.
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNPoolingModule(torch.nn.Module):
    # [CLASS DOC] Holds random 1d, 2d and 3d inputs and lists of max, average, power-average,
    # adaptive and fractional pooling layers for each rank (max unpool is a TODO).
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates three random inputs with 16 channels and the pooling module
        # lists for each rank with fixed kernel sizes and strides.
        super().__init__()
        self.input1d = torch.randn(1, 16, 50)
        self.module1d = nn.ModuleList(
            [
                nn.MaxPool1d(3, stride=2),
                nn.AvgPool1d(3, stride=2),
                nn.LPPool1d(2, 3, stride=2),
                nn.AdaptiveMaxPool1d(3),
                nn.AdaptiveAvgPool1d(3),
            ]
        )

        self.input2d = torch.randn(1, 16, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.MaxPool2d((3, 2), stride=(2, 1)),
                nn.AvgPool2d((3, 2), stride=(2, 1)),
                nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5)),
                nn.LPPool2d(2, 3, stride=(2, 1)),
                nn.AdaptiveMaxPool2d((5, 7)),
                nn.AdaptiveAvgPool2d(7),
            ]
        )

        self.input3d = torch.randn(1, 16, 20, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.MaxPool3d(2),
                nn.AvgPool3d(2),
                nn.FractionalMaxPool3d(2, output_ratio=(0.5, 0.5, 0.5)),
                nn.AdaptiveMaxPool3d((5, 7, 9)),
                nn.AdaptiveAvgPool3d((5, 7, 9)),
            ]
        )
        # TODO max_unpool

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs each pooling layer on the matching input and
        # returns the length of the tuple of the three result lists, which is three.
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNPaddingModule(torch.nn.Module):
    # [CLASS DOC] Holds random 1d, 2d and 3d inputs and lists of reflection, replication, zero
    # and constant padding layers matching each input rank.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates three random inputs with four channels and the padding module
        # lists for each rank with fixed pad sizes and a constant value of 3.5.
        super().__init__()
        self.input1d = torch.randn(1, 4, 50)
        self.module1d = nn.ModuleList(
            [
                nn.ReflectionPad1d(2),
                nn.ReplicationPad1d(2),
                nn.ConstantPad1d(2, 3.5),
            ]
        )

        self.input2d = torch.randn(1, 4, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.ReflectionPad2d(2),
                nn.ReplicationPad2d(2),
                nn.ZeroPad2d(2),
                nn.ConstantPad2d(2, 3.5),
            ]
        )

        self.input3d = torch.randn(1, 4, 10, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.ReflectionPad3d(1),
                nn.ReplicationPad3d(3),
                nn.ConstantPad3d(3, 3.5),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs each padding layer on the matching input and
        # returns the length of the tuple of the three result lists, which is three.
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNNormalizationModule(torch.nn.Module):
    # [CLASS DOC] Holds random inputs and lists of batch, instance, group, layer and local
    # response norm layers plus a channel shuffle layer, grouped by input rank.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates three random inputs with four channels and the normalization
        # module lists for each rank. The layer norm shape matches the 2d input's last dims.
        super().__init__()
        self.input1d = torch.randn(1, 4, 50)
        self.module1d = nn.ModuleList(
            [
                nn.BatchNorm1d(4),
                nn.InstanceNorm1d(4),
            ]
        )

        self.input2d = torch.randn(1, 4, 30, 10)
        self.module2d = nn.ModuleList(
            [
                nn.BatchNorm2d(4),
                nn.GroupNorm(4, 4),
                nn.InstanceNorm2d(4),
                nn.LayerNorm([4, 30, 10]),
                nn.LocalResponseNorm(2),
            ]
        )

        self.input3d = torch.randn(1, 4, 10, 4, 4)
        self.module3d = nn.ModuleList(
            [
                nn.BatchNorm3d(4),
                nn.InstanceNorm3d(4),
                nn.ChannelShuffle(2),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs each normalization layer on the matching input
        # and returns the length of the tuple of the three result lists, which is three.
        return len(
            (
                [module(self.input1d) for i, module in enumerate(self.module1d)],
                [module(self.input2d) for i, module in enumerate(self.module2d)],
                [module(self.input3d) for i, module in enumerate(self.module3d)],
            )
        )


class NNActivationModule(torch.nn.Module):
    # [CLASS DOC] Holds a list of about two dozen activation layers (relu family, sigmoid
    # family, softmax family, glu and others); some are commented out as unsupported.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the list of default-constructed activation layers.
        super().__init__()
        self.activations = nn.ModuleList(
            [
                nn.ELU(),
                nn.Hardshrink(),
                nn.Hardsigmoid(),
                nn.Hardtanh(),
                nn.Hardswish(),
                nn.LeakyReLU(),
                nn.LogSigmoid(),
                # nn.MultiheadAttention(),
                nn.PReLU(),
                nn.ReLU(),
                nn.ReLU6(),
                nn.RReLU(),
                nn.SELU(),
                nn.CELU(),
                nn.GELU(),
                nn.Sigmoid(),
                nn.SiLU(),
                nn.Mish(),
                nn.Softplus(),
                nn.Softshrink(),
                nn.Softsign(),
                nn.Tanh(),
                nn.Tanhshrink(),
                # nn.Threshold(0.1, 20),
                nn.GLU(),
                nn.Softmin(),
                nn.Softmax(),
                nn.Softmax2d(),
                nn.LogSoftmax(),
                # nn.AdaptiveLogSoftmaxWithLoss(),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Creates a random 2x3x4 input, applies every activation to it and
        # returns the length of a one element tuple containing the result list, i.e. one.
        input = torch.randn(2, 3, 4)
        return len(([module(input) for i, module in enumerate(self.activations)],))


class NNRecurrentModule(torch.nn.Module):
    # [CLASS DOC] Holds lists of RNN, GRU and LSTM layers each paired with the matching cell.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the three lists; layers use input size four, hidden size eight
        # and two layers, cells use sizes four and eight.
        super().__init__()
        self.rnn = nn.ModuleList(
            [
                nn.RNN(4, 8, 2),
                nn.RNNCell(4, 8),
            ]
        )
        self.gru = nn.ModuleList([nn.GRU(4, 8, 2), nn.GRUCell(4, 8)])
        self.lstm = nn.ModuleList(
            [
                nn.LSTM(4, 8, 2),
                nn.LSTMCell(4, 8),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs each layer on a random sequence and each cell on
        # its first step, keeping only the last result (the LSTM cell output), and returns the
        # length of that result.
        input = torch.randn(5, 3, 4)
        h = torch.randn(2, 3, 8)
        c = torch.randn(2, 3, 8)
        r = self.rnn[0](input, h)
        r = self.rnn[1](input[0], h[0])
        r = self.gru[0](input, h)
        r = self.gru[1](input[0], h[0])
        r = self.lstm[0](input, (h, c))
        r = self.lstm[1](input[0], (h[0], c[0]))
        return len(r)


class NNTransformerModule(torch.nn.Module):
    # [CLASS DOC] Holds a list of a full transformer, a transformer encoder and a transformer
    # decoder, all with model size two, two heads and one layer.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the three transformer modules with small fixed sizes.
        super().__init__()
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

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs the three transformers on random inputs, keeping
        # only the last result (the decoder output), and returns its length.
        input = torch.rand(1, 16, 2)
        tgt = torch.rand((1, 16, 2))
        r = self.transformers[0](input, tgt)
        r = self.transformers[1](input)
        r = self.transformers[2](input, tgt)
        return len(r)


class NNLinearModule(torch.nn.Module):
    # [CLASS DOC] Holds a list of identity, linear and bilinear layers.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the identity, 20 to 20 linear and 20 by 20 to 40 bilinear layers.
        super().__init__()
        self.linears = nn.ModuleList(
            [
                nn.Identity(54),
                nn.Linear(20, 20),
                nn.Bilinear(20, 20, 40),
                # nn.LazyLinear(20, 30),
            ]
        )

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs the three layers on a random 32x20 input, keeping
        # only the last result (the bilinear output), and returns its length.
        input = torch.randn(32, 20)
        r = self.linears[0](input)
        r = self.linears[1](input)
        r = self.linears[2](input, input)
        return len(r)


class NNDropoutModule(torch.nn.Module):
    # [CLASS DOC] Stateless module calling the functional dropout variants.
    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Creates random tensors of rank two, four and five and
        # applies dropout, 2d and 3d dropout, alpha dropout and feature alpha dropout, passing
        # all results to the len builtin.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        a = torch.randn(8, 4)
        b = torch.randn(8, 4, 4, 4)
        c = torch.randn(8, 4, 4, 4, 4)
        return len(
            F.dropout(a),
            F.dropout2d(b),
            F.dropout3d(c),
            F.alpha_dropout(a),
            F.feature_alpha_dropout(c),
        )


class NNSparseModule(torch.nn.Module):
    # [CLASS DOC] Stateless module calling the functional embedding and one-hot ops.
    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Builds index tensors, a random embedding matrix and
        # offsets, then calls embedding, embedding bag and one hot (five classes), passing the
        # results to the len builtin.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        input = torch.tensor([[1, 2, 4, 5], [4, 3, 2, 9]])
        input2 = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9])
        embedding_matrix = torch.rand(10, 3)
        offsets = torch.tensor([0, 4])
        return len(
            F.embedding(input, embedding_matrix),
            F.embedding_bag(input2, embedding_matrix, offsets),
            F.one_hot(torch.arange(0, 5) % 3, num_classes=5),
        )


class NNDistanceModule(torch.nn.Module):
    # [CLASS DOC] Stateless module calling the functional distance ops.
    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Creates two random 8x4 tensors and calls pairwise
        # distance, cosine similarity and pdist, passing the results to the len builtin.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        a = torch.randn(8, 4)
        b = torch.randn(8, 4)
        return len(
            F.pairwise_distance(a, b),
            F.cosine_similarity(a, b),
            F.pdist(a),
        )


class NNLossFunctionModule(torch.nn.Module):
    # [CLASS DOC] Stores a float tensor x and an integer tensor y used as multilabel loss
    # inputs, and runs the functional loss ops in forward.
    def __init__(self) -> None:
        # [FUNCTION DOC] Stores the fixed float and long tensors x and y used by forward.
        super().__init__()
        self.x = torch.FloatTensor([[0.1, 0.2, 0.4, 0.8]])
        self.y = torch.LongTensor([[3, 0, -1, 1]])

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Creates random predictions and targets plus CTC inputs
        # (log probabilities, targets and lengths) and calls about twenty functional losses
        # (cross entropy, binary, ctc, margin, nll, huber, smooth l1, triplet and others),
        # passing all to the len builtin. Gaussian nll and triplet with distance are
        # commented out as unsupported on mobile.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        a = torch.randn(3, 2)
        b = torch.rand(3, 2)
        c = torch.rand(3)
        log_probs = torch.randn(50, 16, 20).log_softmax(2).detach()
        targets = torch.randint(1, 20, (16, 30), dtype=torch.long)
        input_lengths = torch.full((16,), 50, dtype=torch.long)
        target_lengths = torch.randint(10, 30, (16,), dtype=torch.long)
        return len(
            F.binary_cross_entropy(torch.sigmoid(a), b),
            F.binary_cross_entropy_with_logits(torch.sigmoid(a), b),
            F.poisson_nll_loss(a, b),
            F.cosine_embedding_loss(a, b, c),
            F.cross_entropy(a, b),
            F.ctc_loss(log_probs, targets, input_lengths, target_lengths),
            # F.gaussian_nll_loss(a, b, torch.ones(5, 1)), # ENTER is not supported in mobile module
            F.hinge_embedding_loss(a, b),
            F.kl_div(a, b),
            F.l1_loss(a, b),
            F.mse_loss(a, b),
            F.margin_ranking_loss(c, c, c),
            F.multilabel_margin_loss(self.x, self.y),
            F.multilabel_soft_margin_loss(self.x, self.y),
            F.multi_margin_loss(self.x, torch.tensor([3])),
            F.nll_loss(a, torch.tensor([1, 0, 1])),
            F.huber_loss(a, b),
            F.smooth_l1_loss(a, b),
            F.soft_margin_loss(a, b),
            F.triplet_margin_loss(a, b, -b),
            # F.triplet_margin_with_distance_loss(a, b, -b), # can't take variable number of arguments
        )


class NNVisionModule(torch.nn.Module):
    # [CLASS DOC] Holds a random input, a list of pixel shuffle and upsampling layers, and
    # separate linear and trilinear upsample layers.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates a random 1x4x9x9 input, the list of pixel shuffle, pixel
        # unshuffle and nearest, bilinear and bicubic upsampling layers, and the linear and
        # trilinear upsample layers.
        super().__init__()
        self.input = torch.randn(1, 4, 9, 9)
        self.vision_modules = nn.ModuleList(
            [
                nn.PixelShuffle(2),
                nn.PixelUnshuffle(3),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Upsample(scale_factor=2, mode="bilinear"),
                nn.Upsample(scale_factor=2, mode="bicubic"),
                nn.UpsamplingNearest2d(scale_factor=2),
                nn.UpsamplingBilinear2d(scale_factor=2),
            ]
        )
        self.linear_sample = nn.Upsample(scale_factor=2, mode="linear")
        self.trilinear_sample = nn.Upsample(scale_factor=2, mode="trilinear")

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Runs every layer in the list on the stored input (only
        # the last output is kept), then calls the linear and trilinear upsamples and a grid
        # sample on fresh random tensors, passing these results to the len builtin.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        input = torch.randn(1, 3, 16, 16)
        for module in self.vision_modules:
            r = module(self.input)
        return len(
            r,
            self.linear_sample(torch.randn(4, 9, 9)),
            self.trilinear_sample(torch.randn(1, 3, 4, 9, 9)),
            F.grid_sample(input, torch.ones(1, 4, 4, 2)),
        )


class NNShuffleModule(torch.nn.Module):
    # [CLASS DOC] Holds a single channel shuffle layer with two groups.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the channel shuffle layer with two groups.
        super().__init__()
        self.shuffle = nn.ChannelShuffle(2)

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Applies the channel shuffle to a random 1x4x2x2 tensor
        # and returns the length of that result (its first dimension size).
        return len(
            self.shuffle(torch.randn(1, 4, 2, 2)),
        )


class NNUtilsModule(torch.nn.Module):
    # [CLASS DOC] Holds a sequential of a 50 to 50 linear layer and an unflatten layer that
    # reshapes the second dimension to 2x5x5.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the sequential container described by the class.
        super().__init__()
        self.flatten = nn.Sequential(nn.Linear(50, 50), nn.Unflatten(1, (2, 5, 5)))

    def forward(self):
        # [FUNCTION DOC] Takes no inputs. Pads a list of two integer tensors into a batch
        # (batch first), runs the stored sequential on a random 2x50 input and passes both
        # results to the len builtin. The pack_padded_sequence call is commented out.
        # KNOWN ISSUE: len is given many positional arguments, which plain Python rejects
        # with a type error; the body exists so the listed ops appear in a scripted graph.
        a = [torch.tensor([1, 2, 3]), torch.tensor([3, 4])]
        b = nn.utils.rnn.pad_sequence(a, batch_first=True)
        # c = nn.utils.rnn.pack_padded_sequence(b, batch_first=True, lengths=torch.tensor([3, 2]))
        input = torch.randn(2, 50)
        return len(
            self.flatten(input),
            b,
        )
