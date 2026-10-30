# The model is from here:
#   https://github.com/pytorch/examples/blob/master/word_language_model/model.py
# [MODULE DOC] Word language model (RNN encoder/decoder) from the PyTorch examples
# repository, with variants for scripting where the hidden state is a single tensor or a
# tuple, used as ONNX test models.


import torch
import torch.nn as nn
from torch import Tensor


class RNNModel(nn.Module):
    # [CLASS DOC] Language model with a dropout layer, an embedding encoder, an LSTM, GRU or
    # RNN core, and a linear decoder. Keeps rnn_type, nhid, nlayers and the last hidden
    # state in hidden.
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(
        self,
        rnn_type,
        ntoken,
        ninp,
        nhid,
        nlayers,
        dropout=0.5,
        tie_weights=False,
        batchsize=2,
    ):
        # [FUNCTION DOC] Builds the layers. rnn_type picks LSTM or GRU directly, or an RNN
        # with tanh or relu nonlinearity; any other value raises ValueError. When
        # tie_weights is set, requires nhid equal to ninp (else ValueError) and shares the
        # encoder weight with the decoder. Initializes weights and the hidden state for
        # the given batchsize.
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ["LSTM", "GRU"]:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {"RNN_TANH": "tanh", "RNN_RELU": "relu"}[rnn_type]
            except KeyError:
                raise ValueError(
                    """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']"""
                ) from None
            self.rnn = nn.RNN(
                ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout
            )
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError(
                    "When using the tied flag, nhid must be equal to emsize"
                )
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.hidden = self.init_hidden(batchsize)

    @staticmethod
    def repackage_hidden(h):
        # [FUNCTION DOC] Static helper that returns a detached copy of a tensor hidden
        # state, or a tuple built recursively for nested hidden states.
        """Detach hidden states from their history."""
        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple([RNNModel.repackage_hidden(v) for v in h])

    def init_weights(self):
        # [FUNCTION DOC] Fills encoder and decoder weights with uniform noise in the range
        # minus to plus 0.1 and zeroes the decoder bias, in place.
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        # [FUNCTION DOC] Embeds the input, applies dropout, runs the recurrent core with the
        # given hidden state, applies dropout, and decodes every time step through the
        # linear layer. Stores the detached new hidden state on the instance and returns
        # logits shaped sequence length, batch, ntoken.
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = RNNModel.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))

    def init_hidden(self, bsz):
        # [FUNCTION DOC] Returns zero-filled hidden state for batch size bsz, allocated like
        # the first parameter: a pair of tensors for LSTM, a single tensor otherwise, each
        # shaped layers, batch, nhid.
        weight = next(self.parameters()).data
        if self.rnn_type == "LSTM":
            return (
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
                weight.new(self.nlayers, bsz, self.nhid).zero_(),
            )
        else:
            return weight.new(self.nlayers, bsz, self.nhid).zero_()


class RNNModelWithTensorHidden(RNNModel):
    # [CLASS DOC] RNNModel variant for GRU-style models with a single tensor hidden
    # state; for scripting.
    """Supports GRU scripting."""

    @staticmethod
    def repackage_hidden(h):
        # [FUNCTION DOC] Static override that detaches a single tensor hidden state.
        """Detach hidden states from their history."""
        return h.detach()

    def forward(self, input: Tensor, hidden: Tensor):
        # [FUNCTION DOC] Same computation as the base forward but typed for a tensor hidden
        # state, and stores the detached hidden state on the instance. Returns logits shaped
        # sequence length, batch, ntoken.
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = RNNModelWithTensorHidden.repackage_hidden(hidden)
        return decoded.view(output.size(0), output.size(1), decoded.size(1))


class RNNModelWithTupleHidden(RNNModel):
    # [CLASS DOC] RNNModel variant for LSTM models with a pair of tensors as hidden
    # state; for scripting.
    """Supports LSTM scripting."""

    @staticmethod
    def repackage_hidden(h: tuple[Tensor, Tensor]):
        # [FUNCTION DOC] Static override that detaches both tensors of a hidden state pair
        # and returns them as a tuple.
        """Detach hidden states from their history."""
        return (h[0].detach(), h[1].detach())

    def forward(self, input: Tensor, hidden: tuple[Tensor, Tensor] | None = None):
        # [FUNCTION DOC] Same computation as the base forward but typed for an optional pair
        # hidden state (default none, so the core uses zeros), and stores the detached pair
        # on the instance. Returns logits shaped sequence length, batch, ntoken.
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        self.hidden = self.repackage_hidden(tuple(hidden))
        return decoded.view(output.size(0), output.size(1), decoded.size(1))
