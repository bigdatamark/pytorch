# mypy: allow-untyped-defs
# [MODULE DOC] Defines amp_definitely_not_available, a check used to decide whether mixed-precision features can be used.
from importlib.util import find_spec

import torch


__all__ = ["amp_definitely_not_available"]


def amp_definitely_not_available():
    # [FUNCTION DOC] Returns True when neither CUDA is available nor the torch_xla package can be found, else False.
    return not (torch.cuda.is_available() or find_spec("torch_xla"))
