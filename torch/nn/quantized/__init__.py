# [PACKAGE DOC] Deprecated alias package for torch.ao.nn.quantized (quantized inference modules).
# Imports the dynamic, functional and modules subpackages and star-imports the module classes
# (conv, linear, norm, activation, embedding, LSTM, functional wrappers) listed in __all__.
# MaxPool2d is imported for convenience but is not listed in __all__.
from torch.nn.quantized import dynamic, functional, modules
from torch.nn.quantized.modules import *  # noqa: F403
from torch.nn.quantized.modules import MaxPool2d


__all__ = [
    "BatchNorm2d",
    "BatchNorm3d",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "DeQuantize",
    "Dropout",
    "ELU",
    "Embedding",
    "EmbeddingBag",
    "GroupNorm",
    "Hardswish",
    "InstanceNorm1d",
    "InstanceNorm2d",
    "InstanceNorm3d",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "LSTM",
    "MultiheadAttention",
    "PReLU",
    "Quantize",
    "ReLU6",
    "Sigmoid",
    "Softmax",
    # Wrapper modules
    "FloatFunctional",
    "FXFloatFunctional",
    "QFunctional",
]
