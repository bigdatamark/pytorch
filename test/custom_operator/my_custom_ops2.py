# [MODULE DOC] HELPER: loads the custom op library at import and registers the fake implementation
# for custom::sin. Imported by name through torch.ops.import_module.
from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


@torch.library.register_fake("custom::sin")
def sin_abstract(x):
    # [FUNCTION DOC] HELPER: fake implementation of custom::sin; returns an empty tensor with the
    # same shape and dtype as x.
    return torch.empty_like(x)
