# [MODULE DOC] HELPER: loads the custom op library at import and registers a fake (shape-only)
# implementation for custom::nonzero. Imported by name through torch.ops.import_module.
from model import get_custom_op_library_path

import torch


torch.ops.load_library(get_custom_op_library_path())


@torch.library.register_fake("custom::nonzero")
def nonzero_abstract(x):
    # [FUNCTION DOC] HELPER: fake implementation of custom::nonzero; allocates an unbacked
    # symbolic size from the library context and returns an empty long tensor of shape
    # (that size, number of dims of x).
    n = x.dim()
    ctx = torch.library.get_ctx()
    nnz = ctx.create_unbacked_symint()
    shape = [nnz, n]
    return x.new_empty(shape, dtype=torch.long)
