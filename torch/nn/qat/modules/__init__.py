# [PACKAGE DOC] Deprecated alias package for torch.ao.nn.qat.modules.
# Re-exports Conv1d/2d/3d, Embedding, EmbeddingBag and Linear from the torch.ao implementation
# and imports the conv, embedding_ops and linear submodules so direct submodule imports work.
r"""QAT Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.modules` instead.
"""

from torch.ao.nn.qat.modules.conv import Conv1d, Conv2d, Conv3d
from torch.ao.nn.qat.modules.embedding_ops import Embedding, EmbeddingBag
from torch.ao.nn.qat.modules.linear import Linear
from torch.nn.qat.modules import conv, embedding_ops, linear


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
