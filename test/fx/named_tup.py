# [MODULE DOC] HELPER: Test fixture defining a small NamedTuple type used by the FX tests.
from typing import NamedTuple

import torch


class MyNamedTup(NamedTuple):
    # [CLASS DOC] HELPER: Named tuple with two fields, i and f, both annotated as tensors.
    # Plain data container with no methods; it exists so FX tracing can be tested on named
    # tuples.
    i: torch.Tensor
    f: torch.Tensor
