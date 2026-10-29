# [MODULE DOC] HELPER: loads the custom op library at import and registers fake implementations
# for custom::cos and custom::tan; the cos stub is deliberately declared to live in another module.
from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


# NB: The impl_abstract_pystub for cos actually
# specifies it should live in the my_custom_ops2 module.
@torch.library.register_fake("custom::cos")
def cos_abstract(x):
    # [FUNCTION DOC] HELPER: fake implementation of custom::cos; returns an empty tensor like x.
    return torch.empty_like(x)


# NB: There is no impl_abstract_pystub for tan
@torch.library.register_fake("custom::tan")
def tan_abstract(x):
    # [FUNCTION DOC] HELPER: fake implementation of custom::tan; returns an empty tensor like x.
    return torch.empty_like(x)
