# [PACKAGE DOC]
# Public torch.masked namespace (prototype) for masked reductions and normalizations
# plus the MaskedTensor type. Re-exports, with no logic of its own: the masked
# operations from torch.masked._ops (amax, amin, argmax, argmin, cumprod, cumsum,
# log_softmax, logaddexp, logsumexp, mean, median, norm, normalize, prod, softmax,
# softmin, std, sum, var), the MaskedTensor class and is_masked_tensor predicate from
# maskedtensor.core, and the as_masked_tensor / masked_tensor constructors from
# maskedtensor.creation. Private helpers from _ops are also imported (not in __all__).
# __all__ lists the public names. No import-time side effects beyond submodule imports.
from torch.masked._ops import (
    _canonical_dim,
    _combine_input_and_mask,
    _generate_docstring,
    _input_mask,
    _output_mask,
    _reduction_identity,
    _where,
    amax,
    amin,
    argmax,
    argmin,
    cumprod,
    cumsum,
    log_softmax,
    logaddexp,
    logsumexp,
    mean,
    median,
    norm,
    normalize,
    prod,
    softmax,
    softmin,
    std,
    sum,
    var,
)
from torch.masked.maskedtensor.core import is_masked_tensor, MaskedTensor
from torch.masked.maskedtensor.creation import as_masked_tensor, masked_tensor


__all__ = [
    "amax",
    "amin",
    "argmax",
    "argmin",
    "as_masked_tensor",
    "cumprod",
    "cumsum",
    "is_masked_tensor",
    "log_softmax",
    "logaddexp",
    "logsumexp",
    "masked_tensor",
    "MaskedTensor",
    "mean",
    "median",
    "norm",
    "normalize",
    "prod",
    "softmax",
    "softmin",
    "std",
    "sum",
    "var",
]
