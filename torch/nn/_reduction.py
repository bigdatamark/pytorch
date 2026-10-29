# [MODULE DOC]
# Helpers that convert reduction specifications (the strings none/mean/sum, and the
# legacy boolean size_average/reduce pair) into the integer enum used by the C++ side.
# No import-time side effects.
import warnings


# NB: Keep this file in sync with enums in aten/src/ATen/core/Reduction.h


def get_enum(reduction: str) -> int:
    # [FUNCTION DOC]
    # Map a reduction string to its integer code: none gives 0, mean gives 1, sum gives 2.
    # The deprecated spelling elementwise_mean also gives 1 and emits a warning.
    # Raises ValueError for any other string. Must stay in sync with the C++ Reduction enum.
    if reduction == "none":
        ret = 0
    elif reduction == "mean":
        ret = 1
    elif reduction == "elementwise_mean":
        warnings.warn(
            "reduction='elementwise_mean' is deprecated. "
            "Please use reduction='mean' instead.",
            stacklevel=2,
        )
        ret = 1
    elif reduction == "sum":
        ret = 2
    else:
        ret = -1  # TODO: remove once JIT exceptions support control flow
        raise ValueError(f"{reduction} is not a valid value for reduction")
    return ret


# In order to support previous versions, accept boolean size_average and reduce
# and convert them into the new constants for now


# We use these functions in torch/legacy as well, in which case we'll silence the warning
def legacy_get_string(
    size_average: bool | None,
    reduce: bool | None,
    emit_warning: bool = True,
) -> str:
    # [FUNCTION DOC]
    # Convert the legacy size_average/reduce flags into a reduction string. None counts
    # as true for either flag. Both true gives mean; reduce true only gives sum; reduce
    # false gives none (size_average is then ignored). Warns about the deprecation
    # unless emit_warning is false. Returns the string.
    warning = "size_average and reduce args will be deprecated, please use reduction='{}' instead."

    if size_average is None:
        size_average = True
    if reduce is None:
        reduce = True

    if size_average and reduce:
        ret = "mean"
    elif reduce:
        ret = "sum"
    else:
        ret = "none"
    if emit_warning:
        warnings.warn(warning.format(ret), stacklevel=2)
    return ret


def legacy_get_enum(
    size_average: bool | None,
    reduce: bool | None,
    emit_warning: bool = True,
) -> int:
    # [FUNCTION DOC]
    # Convert the legacy size_average/reduce flags straight to the integer reduction
    # code by composing legacy_get_string and get_enum. Warning behavior follows emit_warning.
    return get_enum(legacy_get_string(size_average, reduce, emit_warning))
