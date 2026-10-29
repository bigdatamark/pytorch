# [PACKAGE DOC] Deprecated alias package re-exporting the quantizable LSTM, LSTMCell and
# MultiheadAttention modules from torch.ao.nn.quantizable.modules (see __all__).
from torch.ao.nn.quantizable.modules.activation import MultiheadAttention
from torch.ao.nn.quantizable.modules.rnn import LSTM, LSTMCell


__all__ = [
    "LSTM",
    "LSTMCell",
    "MultiheadAttention",
]
