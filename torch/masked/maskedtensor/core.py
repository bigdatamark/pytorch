# mypy: allow-untyped-defs
# [MODULE DOC] Defines MaskedTensor, a torch.Tensor wrapper subclass pairing a data tensor with
# a boolean mask of the same shape (strided, sparse COO or sparse CSR layout), plus the helper
# is_masked_tensor and internal helpers for comparing tensors/masks, mapping over arguments,
# wrapping results and pretty printing. Exports MaskedTensor and is_masked_tensor.
# Import-time side effects: none beyond imports; creating a MaskedTensor emits a prototype warning.
# Copyright (c) Meta Platforms, Inc. and affiliates

import warnings
from typing import Any
from typing_extensions import TypeIs

import torch
from torch.overrides import get_default_nowrap_functions


__all__ = [
    "MaskedTensor",
    "is_masked_tensor",
]


def is_masked_tensor(obj: Any, /) -> TypeIs["MaskedTensor"]:
    # [FUNCTION DOC] Returns True if obj is an instance of MaskedTensor, else False. No errors.
    # KNOWN ISSUE: the docstring documents an argument named a, but the actual parameter is obj
    # (positional-only).
    r"""Returns True if the input is a MaskedTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.masked import MaskedTensor
        >>> data = torch.arange(6).reshape(2, 3)
        >>> mask = torch.tensor([[True, False, False], [True, True, False]])
        >>> mt = MaskedTensor(data, mask)
        >>> is_masked_tensor(mt)
        True
    """
    return isinstance(obj, MaskedTensor)


def _tensors_match(a, b, exact=True, rtol=1e-05, atol=1e-08):
    # [FUNCTION DOC] HELPER: Compares two plain tensors for equality. Raises ValueError if either
    # is a MaskedTensor or the layouts differ. Casts b to a's dtype if needed. Sparse COO compares
    # values and indices, sparse CSR compares crow indices, col indices and values (recursively).
    # Strided: requires equal number of dims and then exact equality, or closeness within rtol/atol
    # when exact is false. Returns a bool.
    # KNOWN ISSUE: the strided check compares only the number of dims before comparing values, so
    # tensors of different shapes may raise a broadcasting error rather than return False.
    if is_masked_tensor(a) or is_masked_tensor(b):
        raise ValueError("Neither `a` nor `b` can be a MaskedTensor.")
    if a.layout != b.layout:
        raise ValueError(
            f"`a` and `b` must have the same layout. Got {a.layout} and {b.layout}"
        )

    if a.dtype != b.dtype:
        b = b.type(a.dtype)
    if a.layout == b.layout == torch.sparse_coo:
        return _tensors_match(a.values(), b.values(), exact) and _tensors_match(
            a.indices(), b.indices(), exact
        )
    elif a.layout == b.layout == torch.sparse_csr:
        return (
            _tensors_match(a.crow_indices(), b.crow_indices(), exact)
            and _tensors_match(a.col_indices(), b.col_indices(), exact)
            and _tensors_match(a.values(), b.values(), exact)
        )
    if exact:
        return (a.dim() == b.dim()) and torch.eq(a, b).all().item()
    return (a.dim() == b.dim()) and torch.allclose(a, b, rtol=rtol, atol=atol)


def _masks_match(a, b):
    # [FUNCTION DOC] HELPER: If both arguments are MaskedTensors, returns whether their masks are
    # exactly equal; if either is not a MaskedTensor, returns True (nothing to compare).
    if is_masked_tensor(a) and is_masked_tensor(b):
        mask_a = a.get_mask()
        mask_b = b.get_mask()
        return _tensors_match(mask_a, mask_b, exact=True)
    return True


def _map_mt_args_kwargs(args, kwargs, map_fn):
    # [FUNCTION DOC] HELPER: Applies map_fn to every MaskedTensor found in args and kwargs,
    # descending into lists and tuples; plain tensors and other values pass through unchanged.
    # Returns a new list of args and a new mapping of kwargs. None kwargs is treated as empty.
    # KNOWN ISSUE: the kwargs loop passes the stale loop variable a (the last positional arg)
    # to the helper instead of the kwarg value, so every kwarg gets a mapped copy of the last
    # positional argument (and raises NameError if there are no positional args).
    def _helper(a, map_fn):
        # [FUNCTION DOC] HELPER: Maps a single value: MaskedTensor -> map_fn(value); Tensor ->
        # itself; list/tuple -> recursively mapped container of the same kind; else unchanged.
        if is_masked_tensor(a):
            return map_fn(a)
        elif torch.is_tensor(a):
            return a
        elif isinstance(a, list):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return a_impl
        elif isinstance(a, tuple):
            a_impl, _ = _map_mt_args_kwargs(a, {}, map_fn)
            return tuple(a_impl)
        else:
            return a

    if kwargs is None:
        kwargs = {}
    impl_args = []
    for a in args:
        impl_args.append(_helper(a, map_fn))
    impl_kwargs = {}
    for k in kwargs:
        impl_kwargs[k] = _helper(a, map_fn)
    return impl_args, impl_kwargs


def _wrap_result(result_data, result_mask):
    # [FUNCTION DOC] HELPER: Pairs result data with result mask into MaskedTensor(s). Lists and
    # tuples are handled elementwise (zipped with the mask sequence); a tensor becomes a
    # MaskedTensor; any other kind of value yields NotImplemented.
    if isinstance(result_data, list):
        return [_wrap_result(r, m) for (r, m) in zip(result_data, result_mask)]
    if isinstance(result_data, tuple):
        return tuple(_wrap_result(r, m) for (r, m) in zip(result_data, result_mask))
    if torch.is_tensor(result_data):
        return MaskedTensor(result_data, result_mask)
    # Expect result_data and result_mask to be Tensors only
    return NotImplemented


def _masked_tensor_str(data, mask, formatter):
    # [FUNCTION DOC] HELPER: Recursively renders data and mask as a nested bracketed string for
    # repr. Sparse inputs are densified first. 1-d: elements are formatted (floats via formatter),
    # masked-out entries show as dashes, all right-aligned to a common width. Higher dims: each
    # slice rendered recursively, indented and joined with newlines. Returns the string.
    if data.layout in {torch.sparse_coo, torch.sparse_csr}:
        data = data.to_dense()
        mask = mask.to_dense()
    if data.dim() == 1:
        formatted_elements = [
            formatter.format(d.item()) if isinstance(d.item(), float) else str(d.item())
            for d in data
        ]
        max_len = max(8 if x[1] else len(x[0]) for x in zip(formatted_elements, ~mask))
        return (
            "["
            + ", ".join(
                [
                    "--".rjust(max_len) if m else e
                    for (e, m) in zip(formatted_elements, ~mask)
                ]
            )
            + "]"
        )
    sub_strings = [_masked_tensor_str(d, m, formatter) for (d, m) in zip(data, mask)]
    sub_strings = ["\n".join(["  " + si for si in s.split("\n")]) for s in sub_strings]
    return "[\n" + ",\n".join(sub_strings) + "\n]"


def _get_data(a):
    # [FUNCTION DOC] HELPER: Returns the raw stored data of a MaskedTensor (not autograd
    # wrapped), or the argument itself if it is not a MaskedTensor.
    if is_masked_tensor(a):
        return a._masked_data
    return a


def _maybe_get_mask(a):
    # [FUNCTION DOC] HELPER: Returns the mask of a MaskedTensor, or None for anything else.
    if is_masked_tensor(a):
        return a.get_mask()
    return None


class MaskedTensor(torch.Tensor):
    # [CLASS DOC] Tensor wrapper subclass that holds a data tensor and a boolean mask (True means
    # the element is specified/valid). State: _masked_data and _masked_mask, cloned copies set in
    # init. Invariants (checked by _validate_members): same type, layout and shape for data and
    # mask, layout in strided/sparse COO/sparse CSR, bool mask, supported data dtype (float16/32/64,
    # bool, int8/16/32/64), matching sparse indices. Operator behavior is routed through
    # __torch_function__ and __torch_dispatch__ to tables in _ops_refs. Key methods: get_data,
    # get_mask, to_tensor, is_sparse*, _from_values (differentiable constructor).
    @staticmethod
    def __new__(cls, data, mask, requires_grad=False):
        # [FUNCTION DOC] Creates the wrapper tensor with data's size, device, dtype and layout.
        # Raises TypeError if data or mask is a MaskedTensor or not a Tensor. Emits a UserWarning
        # that the API is a prototype, and another if data requires grad. Returns the new wrapper.
        if is_masked_tensor(data) or not torch.is_tensor(data):
            raise TypeError("data must be a Tensor")
        if is_masked_tensor(mask) or not torch.is_tensor(mask):
            raise TypeError("mask must be a Tensor")
        # Use a Tensor that of the give size for the wrapper.
        kwargs = {
            "device": data.device,
            "dtype": data.dtype,
            "layout": data.layout,
            "requires_grad": requires_grad,
            "dispatch_sizes_strides_policy": "strides",
            "dispatch_layout": True,
        }
        warnings.warn(
            (
                "The PyTorch API of MaskedTensors is in prototype stage "
                "and will change in the near future. Please open a Github issue "
                "for feature requests and see our documentation on the torch.masked "
                "module for further information about the project."
            ),
            UserWarning,
            stacklevel=2,
        )
        if data.requires_grad:
            warnings.warn(
                "It is not recommended to create a MaskedTensor with a tensor that requires_grad. "
                "To avoid this, you can use data.detach().clone()",
                UserWarning,
                stacklevel=2,
            )
        # pyrefly: ignore [bad-argument-type]
        return torch.Tensor._make_wrapper_subclass(cls, data.size(), **kwargs)

    def _preprocess_data(self, data, mask):
        # [FUNCTION DOC] HELPER: Normalizes and stores data and mask. Raises TypeError if layouts
        # differ. For sparse COO both are coalesced; if nonzero counts differ, data is filtered
        # by the mask via the sparse where helpers (same for CSR). Stores clones of both in the
        # _masked_data and _masked_mask attributes.
        from .._ops import _sparse_coo_where, _sparse_csr_where

        if data.layout != mask.layout:
            raise TypeError("data and mask must have the same layout.")
        if data.layout == torch.sparse_coo:
            data = data.coalesce()
            mask = mask.coalesce()
            if data._nnz() != mask._nnz():
                data = _sparse_coo_where(mask, data, torch.tensor(0))
        elif data.layout == torch.sparse_csr:
            if data._nnz() != mask._nnz():
                data = _sparse_csr_where(mask, data, torch.tensor(0))

        # Have to pick awkward names to not conflict with existing fields such as data
        self._masked_data = data.clone()
        self._masked_mask = mask.clone()

    def _validate_members(self):
        # [FUNCTION DOC] HELPER: Checks invariants of the stored data and mask. Raises TypeError
        # for differing tensor types, unsupported layout, non-bool mask or unsupported dtype;
        # raises ValueError for differing sparse indices, number of dims or sizes.
        data = self._masked_data
        mask = self.get_mask()
        if type(data) is not type(mask):
            raise TypeError(
                f"data and mask must have the same type. Got {type(data)} and {type(mask)}"
            )
        if data.layout not in {torch.strided, torch.sparse_coo, torch.sparse_csr}:
            raise TypeError(f"data layout of {data.layout} is not supported.")
        if data.layout == torch.sparse_coo:
            if not _tensors_match(data.indices(), mask.indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse COO tensors but do not have the same indices."
                )
        elif data.layout == torch.sparse_csr:
            if not _tensors_match(
                data.crow_indices(), mask.crow_indices(), exact=True
            ) or not _tensors_match(data.col_indices(), mask.col_indices(), exact=True):
                raise ValueError(
                    "data and mask are both sparse CSR tensors but do not share either crow or col indices."
                )
        if mask.dtype != torch.bool:
            raise TypeError("mask must have dtype bool.")
        if not (
            data.dtype == torch.float16
            or data.dtype == torch.float32
            or data.dtype == torch.float64
            or data.dtype == torch.bool
            or data.dtype == torch.int8
            or data.dtype == torch.int16
            or data.dtype == torch.int32
            or data.dtype == torch.int64
        ):
            raise TypeError(f"{data.dtype} is not supported in MaskedTensor.")
        if data.dim() != mask.dim():
            raise ValueError("data.dim() must equal mask.dim()")
        if data.size() != mask.size():
            raise ValueError("data.size() must equal mask.size()")

    def __init__(self, data, mask, requires_grad=False):
        # [FUNCTION DOC] Stores a normalized copy of data and mask, then validates them (errors
        # as in _validate_members). requires_grad is consumed in __new__ and unused here.
        self._preprocess_data(data, mask)
        self._validate_members()

    @staticmethod
    def _from_values(data, mask):
        # [FUNCTION DOC] HELPER: Differentiable constructor: applies an autograd function that
        # builds a MaskedTensor from data and mask, passing the gradient straight to data (none
        # for mask). Returns the MaskedTensor.
        """Differentiable constructor for MaskedTensor"""

        class Constructor(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function wrapping MaskedTensor construction; forward
            # builds the MaskedTensor and backward returns the incoming gradient for data and
            # None for mask.
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, data, mask):
                # [FUNCTION DOC] HELPER: Returns a new MaskedTensor made from data and mask.
                return MaskedTensor(data, mask)

            @staticmethod
            # pyrefly: ignore [bad-override]
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Gradient is the output gradient for data, None for mask.
                return grad_output, None

        result = Constructor.apply(data, mask)
        return result

    def _set_data_mask(self, data, mask):
        # [FUNCTION DOC] HELPER: Replaces the stored data and mask directly (no cloning or
        # coalescing) and re-validates; raises as _validate_members does. Mutates self.
        self._masked_data = data
        self._masked_mask = mask
        self._validate_members()

    def __repr__(self):  # type: ignore[override]
        # [FUNCTION DOC] String form: for 0-dim shows the value (or dashes if masked out) and the
        # mask value; otherwise a nested multi-line rendering of data with masked-out entries as
        # dashes. Returns a string.
        formatter = "{0:8.4f}"
        if self.dim() == 0:
            scalar_data = self.get_data().item()
            data_formatted = (
                formatter.format(scalar_data)
                if isinstance(scalar_data, float)
                else str(scalar_data)
            )
            if not self.get_mask().item():
                data_formatted = "--"
            return (
                "MaskedTensor("
                + data_formatted
                + ", "
                + str(self.get_mask().item())
                + ")"
            )
        s = _masked_tensor_str(self.get_data(), self.get_mask(), formatter)
        s = "\n".join("  " + si for si in s.split("\n"))
        return "MaskedTensor(\n" + s + "\n)"

    # Seems like this needs to be defined before torch_dispatch to work
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Torch-function hook. If func has an entry in the function table of the
        # _ops_refs module, calls that handler. Otherwise returns NotImplemented if any of the
        # types is not a subclass relation of this class, else runs func with torch-function
        # handling disabled and converts the result back to this class unless func is a
        # default no-wrap function.
        kwargs = kwargs or {}

        from ._ops_refs import _MASKEDTENSOR_FUNCTION_TABLE

        if func in _MASKEDTENSOR_FUNCTION_TABLE:
            return _MASKEDTENSOR_FUNCTION_TABLE[func](*args, **kwargs)

        if not all(issubclass(cls, t) for t in types):
            return NotImplemented
        with torch._C.DisableTorchFunctionSubclass():
            ret = func(*args, **kwargs)
            if func in get_default_nowrap_functions():
                return ret
            else:
                return torch._tensor._convert(ret, cls)

    @classmethod
    def unary(cls, fn, data, mask):
        # [FUNCTION DOC] Builds a MaskedTensor from fn applied to data, reusing the given mask.
        return MaskedTensor(fn(data), mask)

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):  # type: ignore[override]
        # [FUNCTION DOC] Dispatch hook. Looks up the overload packet of func in the dispatch table
        # of the _ops_refs module and calls the handler. If missing, emits a warning asking for a
        # feature request and returns NotImplemented.
        func = func.overloadpacket

        from ._ops_refs import _MASKEDTENSOR_DISPATCH_TABLE

        if func in _MASKEDTENSOR_DISPATCH_TABLE:
            return _MASKEDTENSOR_DISPATCH_TABLE[func](*args, **kwargs)

        msg = (
            f"{func.__name__} is not implemented in __torch_dispatch__ for MaskedTensor.\n"
            "If you would like this operator to be supported, please file an issue for a feature request at "
            "https://github.com/pytorch/maskedtensor/issues with a minimal reproducible code snippet.\n"
            "In the case that the semantics for the operator are not trivial, it would be appreciated "
            "to also include a proposal for the semantics."
        )
        warnings.warn(msg, stacklevel=2)
        return NotImplemented

    def __lt__(self, other):
        # [FUNCTION DOC] Elementwise less-than against a MaskedTensor or a plain value; the result
        # is a MaskedTensor with the comparison of the data and this tensor's mask. Does not check
        # that the masks match.
        if is_masked_tensor(other):
            return MaskedTensor(self.get_data() < _get_data(other), self.get_mask())
        return MaskedTensor(self.get_data() < other, self.get_mask())

    def to_tensor(self, value):
        # [FUNCTION DOC] Returns a plain tensor: the data with masked-out positions filled with value.
        return self.get_data().masked_fill(~self.get_mask(), value)

    def get_data(self):
        # [FUNCTION DOC] Returns the data tensor through an autograd function, so gradients flow
        # back to this MaskedTensor as MaskedTensor gradients using this tensor's mask.
        class GetData(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function exposing the stored data (detached) in forward
            # and wrapping the incoming gradient with the mask in backward.
            @staticmethod
            # pyrefly: ignore [bad-override]
            def forward(ctx, self):
                # [FUNCTION DOC] HELPER: Returns the stored data detached from autograd.
                return self._masked_data.detach()

            @staticmethod
            # pyrefly: ignore [bad-override]
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns grad_output if already a MaskedTensor, else
                # wraps it with the mask of the enclosing MaskedTensor.
                if is_masked_tensor(grad_output):
                    return grad_output
                return MaskedTensor(grad_output, self.get_mask())

        return GetData.apply(self)

    def get_mask(self):
        # [FUNCTION DOC] Returns the stored boolean mask tensor (not a copy).
        return self._masked_mask

    def is_sparse_coo(self):
        # [FUNCTION DOC] True if the layout is sparse COO.
        return self.layout == torch.sparse_coo

    def is_sparse_csr(self):  # type: ignore[override]
        # [FUNCTION DOC] True if the layout is sparse CSR.
        return self.layout == torch.sparse_csr

    # Update later to support more sparse layouts
    @property
    def is_sparse(self):  # type: ignore[override]
        # [FUNCTION DOC] Property: True if the layout is sparse COO or sparse CSR.
        return self.is_sparse_coo() or self.is_sparse_csr()
