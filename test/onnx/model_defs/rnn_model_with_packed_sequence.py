# [MODULE DOC] Wrapper modules for ONNX tests that run an RNN on padded input by packing it
# with sequence lengths first and unpacking the output afterwards. Three variants differ in
# how the extra arguments are passed and what container is returned.
from torch import nn
from torch.nn.utils import rnn as rnn_utils


class RnnModelWithPackedSequence(nn.Module):
    # [CLASS DOC] Wraps an arbitrary recurrent model and a batch_first flag. Its forward
    # takes the padded input plus extra positional arguments whose last item is the sequence
    # lengths; the remaining ones are forwarded to the wrapped model.
    def __init__(self, model, batch_first):
        # [FUNCTION DOC] Stores the wrapped model and the batch_first flag.
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, *args):
        # [FUNCTION DOC] Splits the trailing sequence lengths off the extra arguments, packs
        # the padded input with them, calls the model with the packed input and remaining
        # arguments, pads the first returned item back, and returns a tuple of the padded
        # output followed by the other results.
        args, seq_lengths = args[:-1], args[-1]
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, *args)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return tuple([ret] + list(rets))


class RnnModelWithPackedSequenceWithoutState(nn.Module):
    # [CLASS DOC] Wrapper for recurrent models called with only a packed input (no initial
    # state); holds the model and batch_first.
    def __init__(self, model, batch_first):
        # [FUNCTION DOC] Stores the wrapped model and the batch_first flag.
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, seq_lengths):
        # [FUNCTION DOC] Packs the input using the lengths, calls the model with only the
        # packed input, unpacks the first result back to padded form, and returns a list
        # holding it followed by the remaining results.
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return list([ret] + list(rets))


class RnnModelWithPackedSequenceWithState(nn.Module):
    # [CLASS DOC] Wrapper for recurrent models called with a packed input and an initial
    # state; holds the model and batch_first.
    def __init__(self, model, batch_first):
        # [FUNCTION DOC] Stores the wrapped model and the batch_first flag.
        super().__init__()
        self.model = model
        self.batch_first = batch_first

    def forward(self, input, hx, seq_lengths):
        # [FUNCTION DOC] Packs the input using the lengths, calls the model with the packed
        # input and the initial state hx, unpacks the first result back to padded form, and
        # returns a list holding it followed by the remaining results.
        input = rnn_utils.pack_padded_sequence(input, seq_lengths, self.batch_first)
        rets = self.model(input, hx)
        ret, rets = rets[0], rets[1:]
        ret, _ = rnn_utils.pad_packed_sequence(ret, self.batch_first)
        return list([ret] + list(rets))
