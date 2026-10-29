# mypy: allow-untyped-defs
# [MODULE DOC] Implements the text repr of tensors: global print options (PRINT_OPTS, set/get/context-manager APIs),
# element formatting (_Formatter), recursive nested-list rendering with summarization, and the entry points
# _str / _str_intern used by Tensor repr. Import time creates the global PRINT_OPTS instance only.
import contextlib
import dataclasses
import math
import textwrap
from typing import Any

import torch
from torch import inf


@dataclasses.dataclass
class __PrinterOptions:
    # [CLASS DOC] HELPER: Dataclass holding the global print settings: precision, threshold (element count that triggers
    # summarization), edgeitems, linewidth and sci_mode (None means auto-select).
    precision: int = 4
    threshold: float = 1000
    edgeitems: int = 3
    linewidth: int = 80
    sci_mode: bool | None = None


PRINT_OPTS = __PrinterOptions()


# We could use **kwargs, but this will give better docs
def set_printoptions(
    precision=None,
    threshold=None,
    edgeitems=None,
    linewidth=None,
    profile=None,
    sci_mode=None,
):
    # [FUNCTION DOC] Updates the global PRINT_OPTS. A profile (default, short, full) first resets precision, threshold,
    # edgeitems and linewidth to preset values; unknown profile names are silently ignored. Then each non-None argument overrides
    # its field. sci_mode is always assigned, so omitting it resets it to automatic.
    # Side effect: mutates global state; returns nothing.
    r"""Set options for printing. Items shamelessly taken from NumPy

    Args:
        precision: Number of digits of precision for floating point output
            (default = 4).
        threshold: Total number of array elements which trigger summarization
            rather than full `repr` (default = 1000).
        edgeitems: Number of array items in summary at beginning and end of
            each dimension (default = 3).
        linewidth: The number of characters per line for the purpose of
            inserting line breaks (default = 80). Thresholded matrices will
            ignore this parameter.
        profile: Sane defaults for pretty printing. Can override with any of
            the above options. (any one of `default`, `short`, `full`)
        sci_mode: Enable (True) or disable (False) scientific notation. If
            None (default) is specified, the value is defined by
            `torch._tensor_str._Formatter`. This value is automatically chosen
            by the framework.

    Example::

        >>> # Limit the precision of elements
        >>> torch.set_printoptions(precision=2)
        >>> torch.tensor([1.12345])
        tensor([1.12])
        >>> # Limit the number of elements shown
        >>> torch.set_printoptions(threshold=5)
        >>> torch.arange(10)
        tensor([0, 1, 2, ..., 7, 8, 9])
        >>> # Restore defaults
        >>> torch.set_printoptions(profile='default')
        >>> torch.tensor([1.12345])
        tensor([1.1235])
        >>> torch.arange(10)
        tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    if profile is not None:
        if profile == "default":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80
        elif profile == "short":
            PRINT_OPTS.precision = 2
            PRINT_OPTS.threshold = 1000
            PRINT_OPTS.edgeitems = 2
            PRINT_OPTS.linewidth = 80
        elif profile == "full":
            PRINT_OPTS.precision = 4
            PRINT_OPTS.threshold = inf
            PRINT_OPTS.edgeitems = 3
            PRINT_OPTS.linewidth = 80

    if precision is not None:
        PRINT_OPTS.precision = precision
    if threshold is not None:
        PRINT_OPTS.threshold = threshold
    if edgeitems is not None:
        PRINT_OPTS.edgeitems = edgeitems
    if linewidth is not None:
        PRINT_OPTS.linewidth = linewidth
    PRINT_OPTS.sci_mode = sci_mode


def get_printoptions() -> dict[str, Any]:
    # [FUNCTION DOC] Returns the current print options as a plain dictionary suitable for passing back to set_printoptions.
    r"""Gets the current options for printing, as a dictionary that
    can be passed as ``**kwargs`` to set_printoptions().
    """
    return dataclasses.asdict(PRINT_OPTS)


@contextlib.contextmanager
def printoptions(**kwargs):
    # [FUNCTION DOC] Context manager: saves current options, applies the given keyword options via set_printoptions, yields,
    # and restores the saved options on exit even if the body raises. Note sci_mode is restored too.
    r"""Context manager that temporarily changes the print options.  Accepted
    arguments are same as :func:`set_printoptions`."""
    old_kwargs = get_printoptions()
    set_printoptions(**kwargs)
    try:
        yield
    finally:
        set_printoptions(**old_kwargs)


def tensor_totype(t):
    # [FUNCTION DOC] HELPER: Converts a tensor to double for formatting math, or to float on devices lacking double
    # (MPS, MAIA, XPU without fp64). Returns the converted tensor.
    dtype = (
        torch.float
        if (
            t.is_mps
            or (t.is_xpu and not torch.xpu.get_device_properties(t.device).has_fp64)
            or t.is_maia
        )
        else torch.double
    )
    return t.to(dtype=dtype)


class _Formatter:
    # [CLASS DOC] HELPER: Computes how the elements of one tensor should be formatted: whether it is floating, whether
    # all values are integers (int_mode), whether scientific notation applies (sci_mode) and the maximum rendered width.
    # Filled in at construction; format and width are then used per element.
    def __init__(self, tensor):
        # [FUNCTION DOC] HELPER: Scans the tensor (flattened) to choose int_mode, sci_mode and max_width. Non-floating dtypes use the
        # widest string form. For floats only non-zero finite values are considered; with none, defaults are kept and it returns early.
        # Sci mode is chosen when the max/min ratio exceeds 1000, max exceeds 1e8 or min is below 1e-4, unless PRINT_OPTS.sci_mode
        # forces it. Special-cases float4 (viewed as uint8) and float8_e8m0fnu (converted to float).
        self.floating_dtype = tensor.dtype.is_floating_point
        self.int_mode = True
        self.sci_mode = False
        self.max_width = 1

        with torch.no_grad():
            tensor_view = tensor.reshape(-1)

        if not self.floating_dtype:
            for value in tensor_view:
                value_str = f"{value}"
                self.max_width = max(self.max_width, len(value_str))

        else:
            if tensor.dtype == torch.float4_e2m1fn_x2:  # type: ignore[attr-defined]
                # torch.float4_e2m1fn_x2 is special and does not support the casts necessary
                # to print it, we choose to display the uint8 representation here for
                # convenience of being able to print a tensor.
                # TODO(#146647): extend this to other dtypes without casts defined, such
                # as the bits, uint1..7 and int1..7 dtypes.
                tensor_view = tensor_view.view(torch.uint8)

            nonzero_finite_vals = torch.masked_select(
                tensor_view, torch.isfinite(tensor_view) & tensor_view.ne(0)
            )

            if nonzero_finite_vals.numel() == 0:
                # no valid number, do nothing
                return

            if tensor.dtype == torch.float8_e8m0fnu:  # type: ignore[attr-defined]
                # float8_e8m0fnu is special and does not define arithmetic ops,
                # and printing code further in this file assumes the existence
                # of various arithmetic ops to figure out what to print. We hack
                # and convert to float here to make printing work correctly.
                # TODO(#113663): also add the other float8 dtypes here after arithmetic
                # support for them is removed
                nonzero_finite_vals = nonzero_finite_vals.float()

            # Convert to double (or float) for easy calculation. HalfTensor overflows with 1e8, and there's no div() on CPU.
            nonzero_finite_abs = tensor_totype(nonzero_finite_vals.abs())
            nonzero_finite_min = tensor_totype(nonzero_finite_abs.min())
            nonzero_finite_max = tensor_totype(nonzero_finite_abs.max())

            for value in nonzero_finite_vals:
                if value != torch.ceil(value):
                    self.int_mode = False
                    break

            self.sci_mode = (
                nonzero_finite_max / nonzero_finite_min > 1000.0
                or nonzero_finite_max > 1.0e8
                or nonzero_finite_min < 1.0e-4
                if PRINT_OPTS.sci_mode is None
                else PRINT_OPTS.sci_mode
            )

            if self.int_mode:
                # in int_mode for floats, all numbers are integers, and we append a decimal to nonfinites
                # to indicate that the tensor is of floating type. add 1 to the len to account for this.
                if self.sci_mode:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f"{value:.0f}"
                        self.max_width = max(self.max_width, len(value_str) + 1)
            else:
                # Check if scientific representation should be used.
                if self.sci_mode:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}e}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))
                else:
                    for value in nonzero_finite_vals:
                        value_str = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
                        self.max_width = max(self.max_width, len(value_str))

    def width(self):
        # [FUNCTION DOC] HELPER: Returns the computed maximum element width.
        return self.max_width

    def format(self, value):
        # [FUNCTION DOC] HELPER: Formats one value as a string left-padded to max_width: scientific, integer-with-trailing-dot (not for
        # inf or nan), fixed-point with configured precision, or plain string for non-floating dtypes.
        if self.floating_dtype:
            if self.sci_mode:
                ret = f"{{:{self.max_width}.{PRINT_OPTS.precision}e}}".format(value)
            elif self.int_mode:
                ret = f"{value:.0f}"
                if not (math.isinf(value) or math.isnan(value)):
                    ret += "."
            else:
                ret = f"{{:.{PRINT_OPTS.precision}f}}".format(value)
        else:
            ret = f"{value}"
        return (self.max_width - len(ret)) * " " + ret


def _scalar_str(self, formatter1, formatter2=None):
# [FUNCTION DOC] HELPER: Renders a 0-dim tensor. With a second formatter (complex) combines real and imaginary parts with a sign
# and a j suffix; otherwise formats the single item with the first formatter.
    if formatter2 is not None:
        real_str = _scalar_str(self.real, formatter1)
        imag_str = (_scalar_str(self.imag, formatter2) + "j").lstrip()
        # handles negative numbers, +0.0, -0.0
        if imag_str[0] == "+" or imag_str[0] == "-":
            return real_str + imag_str
        else:
            return real_str + "+" + imag_str
    else:
        return formatter1.format(self.item())


def _vector_str(self, indent, summarize, formatter1, formatter2=None):
    # length includes spaces and comma between elements
# [FUNCTION DOC] HELPER: Renders a 1-dim tensor as a bracketed, comma separated string wrapped to the line width using the
# given indent. When summarizing, shows only edgeitems at each end with an ellipsis (or just an ellipsis if edgeitems is 0).
# A float4 tensor is viewed as uint8 first. Complex values use both formatters.
    element_length = formatter1.width() + 2
    if formatter2 is not None:
        # width for imag_formatter + an extra j for complex
        element_length += formatter2.width() + 1

    elements_per_line = max(
        1, math.floor((PRINT_OPTS.linewidth - indent) / (element_length))
    )

    def _val_formatter(val, formatter1=formatter1, formatter2=formatter2):
    # [FUNCTION DOC] HELPER: Formats one element; for complex values formats real and imaginary parts and joins them with a sign.
        if formatter2 is not None:
            real_str = formatter1.format(val.real)
            imag_str = (formatter2.format(val.imag) + "j").lstrip()
            # handles negative numbers, +0.0, -0.0
            if imag_str[0] == "+" or imag_str[0] == "-":
                return real_str + imag_str
            else:
                return real_str + "+" + imag_str
        else:
            return formatter1.format(val)

    if self.dtype == torch.float4_e2m1fn_x2:  # type: ignore[attr-defined]
        # torch.float4_e2m1fn_x2 is special and does not support the casts necessary
        # to print it, we choose to display the uint8 representation here for
        # convenience of being able to print a tensor.
        # TODO(#146647): extend this to other dtypes without casts defined, such
        # as the bits, uint1..7 and int1..7 dtypes.
        self = self.view(torch.uint8)

    if summarize and not PRINT_OPTS.edgeitems:
        # Deal with edge case that negative zero is zero
        data = ["..."]
    elif summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        data = (
            [_val_formatter(val) for val in self[: PRINT_OPTS.edgeitems].tolist()]
            + [" ..."]
            + [_val_formatter(val) for val in self[-PRINT_OPTS.edgeitems :].tolist()]
        )
    else:
        data = [_val_formatter(val) for val in self.tolist()]

    data_lines = [
        data[i : i + elements_per_line] for i in range(0, len(data), elements_per_line)
    ]
    lines = [", ".join(line) for line in data_lines]
    return "[" + ("," + "\n" + " " * (indent + 1)).join(lines) + "]"


# formatter2 is only used for printing complex tensors.
# For complex tensors, formatter1 and formatter2 are the formatters for tensor.real
# and tensor.imag respectively
def _tensor_str_with_formatter(self, indent, summarize, formatter1, formatter2=None):
# [FUNCTION DOC] HELPER: Recursively renders a tensor of any rank: 0-dim and 1-dim delegate to scalar/vector renderers; higher
# ranks render each slice along dim 0 (summarized with an ellipsis when long) joined by commas and blank lines per rank.
    dim = self.dim()

    if dim == 0:
        return _scalar_str(self, formatter1, formatter2)

    if dim == 1:
        return _vector_str(self, indent, summarize, formatter1, formatter2)

    if summarize and self.size(0) > 2 * PRINT_OPTS.edgeitems:
        slices = (
            [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(PRINT_OPTS.edgeitems)
            ]
            + ["..."]
            + [
                _tensor_str_with_formatter(
                    self[i], indent + 1, summarize, formatter1, formatter2
                )
                for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))
            ]
        )
    else:
        slices = [
            _tensor_str_with_formatter(
                self[i], indent + 1, summarize, formatter1, formatter2
            )
            for i in range(self.size(0))
        ]

    tensor_str = ("," + "\n" * (dim - 1) + " " * (indent + 1)).join(slices)
    return "[" + tensor_str + "]"


def _tensor_str(self, indent):
# [FUNCTION DOC] HELPER: Builds the content string for a tensor: empty gives brackets, large tensors are summarized, zero/neg/conj bits
# are resolved, float8 types are cast to half, and formatters are created (separate real and imaginary for complex)
# before delegating to the recursive renderer.
    if self.numel() == 0:
        return "[]"

    summarize = self.numel() > PRINT_OPTS.threshold

    if self._is_zerotensor():
        self = self.clone()

    # handle the negative bit
    if self.is_neg():
        self = self.resolve_neg()

    # TODO: Remove me when `masked_select` is implemented for FP8
    if self.dtype in [
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ]:
        self = self.half()

    if self.dtype.is_complex:
        # handle the conjugate bit
        self = self.resolve_conj()
        real_formatter = _Formatter(
            get_summarized_data(self.real) if summarize else self.real
        )
        imag_formatter = _Formatter(
            get_summarized_data(self.imag) if summarize else self.imag
        )
        return _tensor_str_with_formatter(
            self, indent, summarize, real_formatter, imag_formatter
        )
    else:
        formatter = _Formatter(get_summarized_data(self) if summarize else self)
        return _tensor_str_with_formatter(self, indent, summarize, formatter)


def _add_suffixes(tensor_str, suffixes, indent, force_newline):
# [FUNCTION DOC] HELPER: Appends suffix strings (device, dtype, etc.) and the closing parenthesis to the tensor text, putting each
# suffix on a new indented line when forced or when it would exceed the line width. Returns the joined string.
# KNOWN ISSUE: last_line_len is computed with a plus 1 offset, which looks like an approximation of the current line length.
    tensor_strs = [tensor_str]
    last_line_len = len(tensor_str) - tensor_str.rfind("\n") + 1
    for suffix in suffixes:
        suffix_len = len(suffix)
        if force_newline or last_line_len + suffix_len + 2 > PRINT_OPTS.linewidth:
            tensor_strs.append(",\n" + " " * indent + suffix)
            last_line_len = indent + suffix_len
            force_newline = False
        else:
            tensor_strs.append(", " + suffix)
            last_line_len += suffix_len + 2
    tensor_strs.append(")")
    return "".join(tensor_strs)


def get_summarized_data(self):
# [FUNCTION DOC] Returns a reduced tensor containing only the edge items of each dimension (recursively) so formatters see just
# the values that will be printed. Returns the tensor itself when small, an empty tensor if edgeitems is 0 for rank 2+.
    dim = self.dim()
    if dim == 0:
        return self
    if dim == 1:
        if self.size(0) > 2 * PRINT_OPTS.edgeitems:
            return torch.cat(
                (self[: PRINT_OPTS.edgeitems], self[-PRINT_OPTS.edgeitems :])
            )
        else:
            return self
    if not PRINT_OPTS.edgeitems:
        return self.new_empty([0] * self.dim())
    elif self.size(0) > 2 * PRINT_OPTS.edgeitems:
        start = [self[i] for i in range(PRINT_OPTS.edgeitems)]
        end = [self[i] for i in range(len(self) - PRINT_OPTS.edgeitems, len(self))]
        return torch.stack([get_summarized_data(x) for x in (start + end)])
    else:
        return torch.stack([get_summarized_data(x) for x in self])


def _str_intern(inp, *, tensor_contents=None):
# [FUNCTION DOC] HELPER: Builds the full repr string of a tensor. Handles functorch wrappers, picks prefix (tensor, nested_tensor,
# subclass name), collects suffixes (device, size, nnz, dtype, layout, grad_fn or requires_grad, tangent), and renders contents
# for sparse COO, sparse compressed, quantized, nested, functional, meta/fake and strided tensors. tensor_contents, if given,
# replaces the generated contents. Non-plain Parameter subclasses are wrapped as Parameter. Moves xla, lazy, ipu, mtia tensors
# to cpu first. Forward-AD primal is used for printing.
    if torch._C._functorch.is_functorch_wrapped_tensor(inp):
        return _functorch_wrapper_str_intern(inp, tensor_contents=tensor_contents)
    is_plain_tensor = type(inp) is torch.Tensor or type(inp) is torch.nn.Parameter
    if inp.is_nested:
        prefix = "nested_tensor("
    elif is_plain_tensor:
        prefix = "tensor("
    else:
        prefix = f"{type(inp).__name__}("
    indent = len(prefix)
    suffixes = []
    custom_contents_provided = tensor_contents is not None
    if custom_contents_provided:
        tensor_str = tensor_contents

    # This is used to extract the primal value and thus disable the forward AD
    # within this function.
    # TODO(albanD) This needs to be updated when more than one level is supported
    self, tangent = torch.autograd.forward_ad.unpack_dual(inp)

    # Note [Print tensor device]:
    # A general logic here is we only print device when it doesn't match
    # the device specified in default tensor type.
    # Currently torch.set_default_tensor_type() only supports CPU/CUDA, thus
    # torch._C._get_default_device() only returns either cpu or cuda.
    # In other cases, we don't have a way to set them as default yet,
    # and we should always print out device for them.
    if (
        self.device.type != torch._C._get_default_device()
        or (
            self.device.type == "cuda"
            and torch.cuda.current_device() != self.device.index
        )
        or (self.device.type == "mps")
    ):
        suffixes.append("device='" + str(self.device) + "'")

    # Tensor printing performs tensor operations like slice, indexing, etc to make it in a
    # representable format. These operations on ipu/xla/lazy/mtia tensor results in compilations. Hence,
    # to avoid compilations, copying the tensor to cpu before printing.
    if self.device.type in ["xla", "lazy", "ipu", "mtia"]:
        self = self.to("cpu")

    # TODO: add an API to map real -> complex dtypes
    _default_complex_dtype = (
        torch.cdouble if torch.get_default_dtype() == torch.double else torch.cfloat
    )
    has_default_dtype = self.dtype in (
        torch.get_default_dtype(),
        _default_complex_dtype,
        torch.int64,
        torch.bool,
    )
    if self.is_sparse:
        suffixes.append("size=" + str(tuple(self.shape)))
        from torch._subclasses.fake_tensor import is_fake_tensor

        is_meta = self.is_meta or is_fake_tensor(self)
        if not is_meta:
            suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            indices_prefix = "indices=tensor("
            indices = self._indices().detach()
            if is_meta:
                indices_str = "..."
            else:
                indices_str = _tensor_str(indices, indent + len(indices_prefix))
            if is_meta or indices.numel() == 0:
                indices_str += ", size=" + str(tuple(indices.shape))
            values_prefix = "values=tensor("
            values = self._values().detach()
            if is_meta:
                values_str = "..."
            else:
                values_str = _tensor_str(values, indent + len(values_prefix))
            if is_meta or values.numel() == 0:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                indices_prefix
                + indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        from torch._subclasses.fake_tensor import is_fake_tensor

        suffixes.append("size=" + str(tuple(self.shape)))
        is_meta = self.is_meta or is_fake_tensor(self)
        if not is_meta:
            suffixes.append("nnz=" + str(self._nnz()))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        if not custom_contents_provided:
            compressed_indices_method, plain_indices_method = {
                torch.sparse_csr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_csc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
                torch.sparse_bsr: (torch.Tensor.crow_indices, torch.Tensor.col_indices),
                torch.sparse_bsc: (torch.Tensor.ccol_indices, torch.Tensor.row_indices),
            }[self.layout]
            if self.layout in {torch.sparse_csr, torch.sparse_bsr}:
                cdimname, pdimname = "row", "column"
            else:
                cdimname, pdimname = "column", "row"
            compressed_indices_prefix = f"c{cdimname[:3]}_indices=tensor("
            compressed_indices = compressed_indices_method(self).detach()
            if is_meta:
                compressed_indices_str = "..."
            else:
                compressed_indices_str = _tensor_str(
                    compressed_indices, indent + len(compressed_indices_prefix)
                )
            if compressed_indices.numel() == 0 or is_meta:
                compressed_indices_str += ", size=" + str(
                    tuple(compressed_indices.shape)
                )
            plain_indices_prefix = f"{pdimname[:3]}_indices=tensor("
            plain_indices = plain_indices_method(self).detach()
            if is_meta:
                plain_indices_str = "..."
            else:
                plain_indices_str = _tensor_str(
                    plain_indices, indent + len(plain_indices_prefix)
                )
            if plain_indices.numel() == 0 or is_meta:
                plain_indices_str += ", size=" + str(tuple(plain_indices.shape))
            values_prefix = "values=tensor("
            values = self.values().detach()
            if is_meta:
                values_str = "..."
            else:
                values_str = _tensor_str(values, indent + len(values_prefix))
            if values.numel() == 0 or is_meta:
                values_str += ", size=" + str(tuple(values.shape))
            tensor_str = (
                compressed_indices_prefix
                + compressed_indices_str
                + "),\n"
                + " " * indent
                + plain_indices_prefix
                + plain_indices_str
                + "),\n"
                + " " * indent
                + values_prefix
                + values_str
                + ")"
            )
    elif self.is_quantized:
        suffixes.append("size=" + str(tuple(self.shape)))
        if not has_default_dtype:
            suffixes.append("dtype=" + str(self.dtype))
        suffixes.append("quantization_scheme=" + str(self.qscheme()))
        if (
            self.qscheme() == torch.per_tensor_affine
            or self.qscheme() == torch.per_tensor_symmetric
        ):
            suffixes.append("scale=" + str(self.q_scale()))
            suffixes.append("zero_point=" + str(self.q_zero_point()))
        elif (
            self.qscheme() == torch.per_channel_affine
            or self.qscheme() == torch.per_channel_symmetric
            or self.qscheme() == torch.per_channel_affine_float_qparams
        ):
            suffixes.append("scale=" + str(self.q_per_channel_scales()))
            suffixes.append("zero_point=" + str(self.q_per_channel_zero_points()))
            suffixes.append("axis=" + str(self.q_per_channel_axis()))
        if not custom_contents_provided:
            tensor_str = _tensor_str(self.dequantize(), indent)
    elif self.is_nested:
        if not custom_contents_provided:

            def indented_str(s, indent):
                # [FUNCTION DOC] HELPER: Indents every line of a string by two spaces.
                # KNOWN ISSUE: the indent argument is ignored; a fixed two-space indent is always used.
                return "\n".join(f"  {line}" for line in s.split("\n"))

            strs = ",\n".join(
                indented_str(str(t), indent + 1)
                for t in torch.ops.aten.unbind.int(self, 0)
            )
            tensor_str = f"[\n{strs}\n]"
    elif torch._is_functional_tensor(self):
        prefix = "_to_functional_tensor("
        tensor_str = repr(torch._from_functional_tensor(self))
    else:
        # Circular import problem, so we import it here
        from torch._subclasses.fake_tensor import is_fake_tensor

        if self.is_meta or is_fake_tensor(self):
            suffixes.append("size=" + str(tuple(self.shape)))
            if self.dtype != torch.get_default_dtype():
                suffixes.append("dtype=" + str(self.dtype))
            # TODO: This implies that ellipses is valid syntax for allocating
            # a meta tensor or FakeTensor, which it could be, but it isn't right now
            if not custom_contents_provided:
                tensor_str = "..."
        else:
            if self.numel() == 0 and not self.is_sparse:
                # Explicitly print the shape if it is not (0,), to match NumPy behavior
                if self.dim() != 1:
                    suffixes.append("size=" + str(tuple(self.shape)))

                # In an empty tensor, there are no elements to infer if the dtype
                # should be int64, so it must be shown explicitly.
                if self.dtype != torch.get_default_dtype():
                    suffixes.append("dtype=" + str(self.dtype))
                if not custom_contents_provided:
                    tensor_str = "[]"
            else:
                if not PRINT_OPTS.edgeitems:
                    suffixes.append("size=" + str(tuple(self.shape)))

                if not has_default_dtype:
                    suffixes.append("dtype=" + str(self.dtype))

                if not custom_contents_provided:
                    if self.layout != torch.strided:
                        tensor_str = _tensor_str(self.to_dense(), indent)
                    else:
                        tensor_str = _tensor_str(self, indent)

    if self.layout != torch.strided:
        suffixes.append("layout=" + str(self.layout))

    # Use inp here to get the original grad_fn and not the one generated by the forward grad
    # unpacking.
    grad_fn_name = None
    try:
        grad_fn = inp.grad_fn
    except RuntimeError:
        # Accessing the grad_fn calls rebasing logic which would cause an error
        # if that tensor is a view created in no-grad mode modified in-place in
        # no-grad mode. See: https://github.com/pytorch/pytorch/issues/99968
        grad_fn_name = "Invalid"

    if grad_fn_name is None and grad_fn is not None:  # type: ignore[possibly-undefined]
        # pyrefly: ignore [unbound-name]
        grad_fn_name = type(grad_fn).__name__
        if grad_fn_name == "CppFunction":
            # pyrefly: ignore [unbound-name]
            grad_fn_name = grad_fn.name().rsplit("::", 1)[-1]

    if grad_fn_name is not None:
        suffixes.append(f"grad_fn=<{grad_fn_name}>")
    elif inp.requires_grad:
        suffixes.append("requires_grad=True")

    if tangent is not None:
        suffixes.append(f"tangent={tangent}")

    string_repr = _add_suffixes(
        prefix + tensor_str,  # type: ignore[possibly-undefined]
        suffixes,
        indent,
        force_newline=self.is_sparse,
    )

    # Check if this instance is flagged as a parameter and change the repr accordingly.
    # Unfortunately, this function has to be aware of this detail.
    # NB: This is currently skipped for plain tensor parameters to maintain BC. In the future,
    # this should be done for those as well to produce a valid repr.
    if isinstance(self, torch.nn.Parameter) and not is_plain_tensor:
        string_repr = f"Parameter({string_repr})"

    return string_repr


def _functorch_wrapper_str_intern(tensor, *, tensor_contents=None):
# [FUNCTION DOC] HELPER: Renders functorch wrapper tensors (BatchedTensor, GradTrackingTensor, FunctionalTensor) showing level
# and the indented repr of the wrapped value. Syncs functional tensors first. Raises AssertionError for invalid level or bdim,
# and ValueError for unknown wrapper kinds.
    level = torch._C._functorch.maybe_get_level(tensor)
    if level == -1:
        raise AssertionError("expected functorch level to be >= 0, got -1")

    if torch._C._functorch.is_functionaltensor(tensor):
        # Since we're unwrapping the FunctionalTensorWrapper, we need to make sure
        # that it's up to date first
        torch._sync(tensor)

    value = torch._C._functorch.get_unwrapped(tensor)
    value_repr = repr(value)

    indented_value_repr = textwrap.indent(value_repr, " " * 4)
    if torch._C._functorch.is_batchedtensor(tensor):
        bdim = torch._C._functorch.maybe_get_bdim(tensor)
        if bdim == -1:
            raise AssertionError("expected batch dimension to be >= 0, got -1")
        return (
            f"BatchedTensor(lvl={level}, bdim={bdim}, value=\n{indented_value_repr}\n)"
        )
    if torch._C._functorch.is_gradtrackingtensor(tensor):
        return f"GradTrackingTensor(lvl={level}, value=\n{indented_value_repr}\n)"
    if torch._C._functorch.is_functionaltensor(tensor):
        return f"FunctionalTensor(lvl={level}, value=\\\n{value_repr})"

    raise ValueError("We don't know how to print this, please file us an issue")


def _str(self, *, tensor_contents=None):
# [FUNCTION DOC] HELPER: Entry point for tensor repr: runs _str_intern with grad mode and Python dispatch modes disabled
# (and functorch disabled) so printing does not trace or record operations. Returns the string.
    with torch.no_grad(), torch.utils._python_dispatch._disable_current_modes():
        guard = torch._C._DisableFuncTorch()  # noqa: F841
        return _str_intern(self, tensor_contents=tensor_contents)
