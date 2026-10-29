# mypy: allow-untyped-defs
# [MODULE DOC] MaskedTensor support for unary ATen ops (abs, exp, sqrt, etc.). Main contents:
# UNARY_NAMES, INPLACE_UNARY_NAMES, UNARY_NAMES_UNSUPPORTED (documentation of ops known to be
# unsupported), the shared _unary_helper, wrapper factories, NATIVE_*_MAP dispatch tables and
# _is_native_unary / _apply_native_unary. Exports nothing publicly.
# Import-time side effect: builds the dispatch maps by resolving torch.ops.aten overloads.
# Copyright (c) Meta Platforms, Inc. and affiliates

import torch

from .core import _map_mt_args_kwargs, _wrap_result


__all__ = []  # type: ignore[var-annotated]


UNARY_NAMES = [
    "abs",
    "absolute",
    "acos",
    "arccos",
    "acosh",
    "arccosh",
    "angle",
    "asin",
    "arcsin",
    "asinh",
    "arcsinh",
    "atan",
    "arctan",
    "atanh",
    "arctanh",
    "bitwise_not",
    "ceil",
    "clamp",
    "clip",
    "conj_physical",
    "cos",
    "cosh",
    "deg2rad",
    "digamma",
    "erf",
    "erfc",
    "erfinv",
    "exp",
    "exp2",
    "expm1",
    "fix",
    "floor",
    "frac",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "logit",
    "i0",
    "isnan",
    "nan_to_num",
    "neg",
    "negative",
    "positive",
    "pow",
    "rad2deg",
    "reciprocal",
    "round",
    "rsqrt",
    "sigmoid",
    "sign",
    "sgn",
    "signbit",
    "sin",
    "sinc",
    "sinh",
    "sqrt",
    "square",
    "tan",
    "tanh",
    "trunc",
]

INPLACE_UNARY_NAMES = [
    n + "_"
    for n in (list(set(UNARY_NAMES) - {"angle", "positive", "signbit", "isnan"}))
]

# Explicitly tracking functions we know are currently not supported
# This might be due to missing code gen or because of complex semantics
UNARY_NAMES_UNSUPPORTED = [
    "atan2",
    "arctan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "copysign",
    "float_power",
    "fmod",
    "frexp",
    "gradient",
    "imag",
    "ldexp",
    "lerp",
    "logical_not",
    "hypot",
    "igamma",
    "igammac",
    "mvlgamma",
    "nextafter",
    "polygamma",
    "real",
    "remainder",
    "true_divide",
    "xlogy",
]


def _unary_helper(fn, args, kwargs, inplace):
    # [FUNCTION DOC] HELPER: Applies a unary op to the data of a MaskedTensor, keeping the mask.
    # Takes the op, positional args (the tensor plus scalar extras), kwargs (must be empty) and
    # an inplace flag. Raises ValueError if kwargs are given, TypeError if an extra argument is
    # a Tensor. For sparse COO (coalesced first) and CSR only the values are transformed and the
    # sparse tensor is rebuilt from the original indices. Inplace: updates data and mask of
    # args[0] and returns it; otherwise returns a new MaskedTensor.
    # KNOWN ISSUE: the CSR branch rebuilds the tensor without passing the original size, unlike
    # the COO branch, so trailing empty rows/columns may be lost when size is inferred.
    if len(kwargs) != 0:
        raise ValueError(
            "MaskedTensor unary ops require that len(kwargs) == 0. "
            "If you need support for this, please open an issue on Github."
        )
    for a in args[1:]:
        if torch.is_tensor(a):
            raise TypeError(
                "MaskedTensor unary ops do not support additional Tensor arguments"
            )

    mask_args, _mask_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_mask
    )
    data_args, _data_kwargs = _map_mt_args_kwargs(
        args, kwargs, lambda x: x._masked_data
    )

    if args[0].layout == torch.sparse_coo:
        data_args[0] = data_args[0].coalesce()
        s = data_args[0].size()
        i = data_args[0].indices()
        data_args[0] = data_args[0].coalesce().values()
        v = fn(*data_args)
        result_data = torch.sparse_coo_tensor(i, v, size=s)

    elif args[0].layout == torch.sparse_csr:
        crow = data_args[0].crow_indices()
        col = data_args[0].col_indices()
        data_args[0] = data_args[0].values()
        v = fn(*data_args)
        result_data = torch.sparse_csr_tensor(crow, col, v)

    else:
        result_data = fn(*data_args)

    if inplace:
        args[0]._set_data_mask(result_data, mask_args[0])
        return args[0]
    else:
        return _wrap_result(result_data, mask_args[0])


def _torch_unary(fn_name):
    # [FUNCTION DOC] HELPER: Looks up the aten op by name and returns a wrapper applying it to
    # a MaskedTensor through _unary_helper in out-of-place mode.
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Out-of-place masked unary op; forwards to _unary_helper.
        return _unary_helper(fn, args, kwargs, inplace=False)

    return unary_fn


def _torch_inplace_unary(fn_name):
    # [FUNCTION DOC] HELPER: Like _torch_unary but the wrapper mutates the MaskedTensor in place.
    fn = getattr(torch.ops.aten, fn_name)

    def unary_fn(*args, **kwargs):
        # [FUNCTION DOC] HELPER: In-place masked unary op; forwards to _unary_helper.
        return _unary_helper(fn, args, kwargs, inplace=True)

    return unary_fn


NATIVE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_unary(name) for name in UNARY_NAMES
}
NATIVE_INPLACE_UNARY_MAP = {
    getattr(torch.ops.aten, name): _torch_inplace_unary(name)
    for name in INPLACE_UNARY_NAMES
}

NATIVE_UNARY_FNS = list(NATIVE_UNARY_MAP.keys())
NATIVE_INPLACE_UNARY_FNS = list(NATIVE_INPLACE_UNARY_MAP.keys())


def _is_native_unary(fn):
    # [FUNCTION DOC] HELPER: True if fn is a supported unary op (regular or in-place variant).
    return fn in NATIVE_UNARY_FNS or fn in NATIVE_INPLACE_UNARY_FNS


def _apply_native_unary(fn, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Dispatches fn to its masked wrapper (regular map first, then
    # in-place map); returns NotImplemented if fn is in neither map.
    if fn in NATIVE_UNARY_FNS:
        return NATIVE_UNARY_MAP[fn](*args, **kwargs)
    if fn in NATIVE_INPLACE_UNARY_FNS:
        return NATIVE_INPLACE_UNARY_MAP[fn](*args, **kwargs)
    return NotImplemented
