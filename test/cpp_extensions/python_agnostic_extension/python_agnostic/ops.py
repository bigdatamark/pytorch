# [MODULE DOC] Python-facing wrapper for the python_agnostic custom operators. At import
# time creates a scoped FRAGMENT library in the "python_agnostic" namespace and defines the
# ultra_norm schema (list of tensors in, tensor out); the kernel comes from the loaded _C
# library. Exposes the ultra_norm function.
import torch
from torch import Tensor


lib = torch.library._scoped_library("python_agnostic", "FRAGMENT")
lib.define("ultra_norm(Tensor[] inputs) -> Tensor")


def ultra_norm(inputs: list[Tensor]) -> Tensor:
    # [FUNCTION DOC] Computes the L2 norm of the per-tensor norms of a list of tensors by
    # dispatching to the registered python_agnostic operator. Input list must be non-empty
    # and share one device and dtype; returns a scalar tensor. Errors come from the operator.
    """
    Computes the ultra-L2-norm of a list of tensors via computing the norm of norms.

    Assumes:
    - inputs should not be empty
    - all tensors in inputs should be on the same device and have the same dtype

    Args:
        inputs: list of torch.tensors

    Returns:
        Scalar torch.tensor of shape ()

    """
    return torch.ops.python_agnostic.ultra_norm.default(inputs)
