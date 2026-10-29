# [PACKAGE DOC] Deprecated alias package for torch.ao.nn.qat (quantization-aware training modules).
# Imports the dynamic and modules subpackages and star-imports torch.nn.qat.modules, so
# Linear, Conv1d/2d/3d, Embedding and EmbeddingBag are exported (see __all__).
# KNOWN ISSUE: the docstring says 'QAT Dynamic Modules' but this package is the non-dynamic QAT
# one; the dynamic variant lives in the dynamic subpackage.
r"""QAT Dynamic Modules.

This package is in the process of being deprecated.
Please, use `torch.ao.nn.qat.dynamic` instead.
"""

from torch.nn.qat import dynamic, modules
from torch.nn.qat.modules import *  # noqa: F403


__all__ = [
    "Linear",
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Embedding",
    "EmbeddingBag",
]
