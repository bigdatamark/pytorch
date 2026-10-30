# [MODULE DOC] LSTM wrappers for ONNX tests whose forward flattens the (output, (hidden,
# cell)) result of an LSTM into a plain three-element tuple (output, hidden, cell).
from torch import nn
from torch.nn.utils.rnn import PackedSequence


class LstmFlatteningResult(nn.LSTM):
    # [CLASS DOC] LSTM subclass that returns hidden and cell state as
    # separate tuple entries.
    def forward(self, input, *fargs, **fkwargs):
        # [FUNCTION DOC] Calls the base LSTM forward with the given input and extra
        # arguments, then returns the tuple (output, hidden, cell) instead of the
        # nested state tuple.
        output, (hidden, cell) = nn.LSTM.forward(self, input, *fargs, **fkwargs)
        return output, hidden, cell


class LstmFlatteningResultWithSeqLength(nn.Module):
    # [CLASS DOC] Module holding an LSTM in inner_model; forward accepts a PackedSequence
    # input and returns flattened (output, hidden, cell). Also records batch_first.
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        # [FUNCTION DOC] Stores batch_first and builds inner_model, an LSTM configured
        # from the given input size, hidden size, layer count, bidirectional flag,
        # dropout and batch_first.
        super().__init__()

        self.batch_first = batch_first
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, input: PackedSequence, hx=None):
        # [FUNCTION DOC] Runs inner_model on the packed input with optional initial state hx
        # and returns the flattened tuple (output, hidden, cell).
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        return output, hidden, cell


class LstmFlatteningResultWithoutSeqLength(nn.Module):
    # [CLASS DOC] Same as the packed-sequence variant but intended for ordinary padded
    # tensor input (no sequence length information); wraps an LSTM and flattens its result.
    def __init__(self, input_size, hidden_size, layers, bidirect, dropout, batch_first):
        # [FUNCTION DOC] Stores batch_first and builds inner_model, an LSTM configured
        # from the given input size, hidden size, layer count, bidirectional flag,
        # dropout and batch_first.
        super().__init__()

        self.batch_first = batch_first
        self.inner_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=layers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(self, input, hx=None):
        # [FUNCTION DOC] Runs inner_model on the tensor input with optional initial state hx
        # and returns the flattened tuple (output, hidden, cell).
        output, (hidden, cell) = self.inner_model.forward(input, hx)
        return output, hidden, cell
