# [PACKAGE DOC] Reference quantized modules. Imports and re-exports Linear,
# Conv1d/2d/3d, ConvTranspose1d/2d/3d, RNNCell, LSTMCell, GRUCell, LSTM, GRU,
# Embedding and EmbeddingBag; these hold quantized weights but run in float by
# dequantizing, then quantize outputs as needed.
from .conv import (
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
)
from .linear import Linear
from .rnn import GRU, GRUCell, LSTM, LSTMCell, RNNCell
from .sparse import Embedding, EmbeddingBag


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "LSTM",
    "GRU",
    "Embedding",
    "EmbeddingBag",
]
