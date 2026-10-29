# mypy: allow-untyped-defs
# [MODULE DOC] Implementation of torch.testing.assert_close and its comparison machinery. Contains
# default dtype tolerances, mismatch message builders, ErrorMeta (an internal exception carrying
# the real error type, message and item id), and a family of Pair classes (None, bool, number,
# tensor-like, plain object) that each know how to compare one actual/expected couple. Inputs are
# walked recursively (sequences, mappings, dataclasses) to originate pairs, which are then
# compared. Import-time side effects: tries to import numpy (HAS_NUMPY), builds the table of dtype
# precisions (extended with the quantized dtypes) and the tuple of types skipped for dataclass
# detection, and records whether distributed support is available.

import abc
import cmath
import collections.abc
import contextlib
import dataclasses
import types
from collections.abc import Callable, Collection, Sequence
from typing import Any, NoReturn
from typing_extensions import deprecated

import torch


try:
    import numpy as np

    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# Types for which Dynamo can fault if we call dataclasses.is_dataclass on them.
_IS_DATACLASS_SKIP_TYPES: tuple[type, ...] = (
    type,
    torch.Tensor,
    types.UnionType,
    types.GenericAlias,
)
if HAS_NUMPY:
    # pyrefly: ignore [missing-attribute]
    _IS_DATACLASS_SKIP_TYPES = (*_IS_DATACLASS_SKIP_TYPES, np.ndarray, np.generic)

_HAS_DTENSOR = torch.distributed.is_available()


def _unwrap_dtensor_for_comparison(actual, expected):
    # [FUNCTION DOC] HELPER: Prepares inputs for comparison when DTensors are involved. If
    # distributed support is missing, returns the inputs unchanged. If both are DTensors, checks
    # that placements and device meshes match and returns their local tensors; raises
    # AssertionError on a mismatch. If exactly one is a DTensor, raises TypeError because the
    # comparison is ambiguous. Otherwise returns the inputs unchanged.
    """Handle DTensor inputs for assertEqual/assert_close."""
    if not _HAS_DTENSOR:
        return actual, expected
    from torch.distributed.tensor import DTensor

    actual_dt = isinstance(actual, DTensor)
    expected_dt = isinstance(expected, DTensor)
    if actual_dt and expected_dt:
        if actual.placements != expected.placements:
            raise AssertionError(
                f"DTensor placements do not match: "
                f"{actual.placements} != {expected.placements}"
            )
        if actual.device_mesh != expected.device_mesh:
            raise AssertionError(
                f"DTensor device meshes do not match: "
                f"{actual.device_mesh} != {expected.device_mesh}"
            )
        return actual.to_local(), expected.to_local()
    elif actual_dt != expected_dt:
        raise TypeError(
            "Comparing a DTensor to a non-DTensor is ambiguous. "
            "Call .full_tensor() to compare the full logical tensor "
            "or .to_local() to compare the local shard."
        )
    return actual, expected


class ErrorMeta(Exception):
    # [CLASS DOC] HELPER: Internal exception that carries the information needed to produce a
    # user-facing error: the target exception type, the message, and an id tuple locating the
    # failing item in nested inputs. Its own message tells users to file an issue if they see it.
    # Key method to_error builds the real exception.
    """Internal testing exception that makes that carries error metadata."""

    def __init__(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ()
    ) -> None:
        # [FUNCTION DOC] HELPER: Stores the exception type, message and id (default empty), and
        # initializes the base exception with a fixed message asking users to report the issue.
        super().__init__(
            "If you are a user and see this message during normal operation "
            "please file an issue at https://github.com/pytorch/pytorch/issues. "
            "If you are a developer and working on the comparison functions, please `raise ErrorMeta.to_error()` "
            "for user facing errors."
        )
        self.type = type
        self.msg = msg
        self.id = id

    def to_error(self, msg: str | Callable[[str], str] | None = None) -> Exception:
        # [FUNCTION DOC] HELPER: Builds the user-facing exception. If msg is not a string, starts
        # from the stored message, appends the item location from the id when present, and, when
        # msg is a callable, passes the text through it. Returns an instance of the stored
        # exception type; does not raise.
        if not isinstance(msg, str):
            generated_msg = self.msg
            if self.id:
                generated_msg += f"\n\nThe failure occurred for item {''.join(str([item]) for item in self.id)}"

            msg = msg(generated_msg) if callable(msg) else generated_msg

        return self.type(msg)


# Some analysis of tolerance by logging tests from test_torch.py can be found in
# https://github.com/pytorch/pytorch/pull/32538.
# {dtype: (rtol, atol)}
_DTYPE_PRECISIONS = {
    torch.float16: (0.001, 1e-5),
    torch.bfloat16: (0.016, 1e-5),
    torch.float32: (1.3e-6, 1e-5),
    torch.float64: (1e-7, 1e-7),
    torch.complex32: (0.001, 1e-5),
    torch.complex64: (1.3e-6, 1e-5),
    torch.complex128: (1e-7, 1e-7),
}
# The default tolerances of torch.float32 are used for quantized dtypes, because quantized tensors are compared in
# their dequantized and floating point representation. For more details see `TensorLikePair._compare_quantized_values`
_DTYPE_PRECISIONS.update(
    dict.fromkeys(
        (torch.quint8, torch.quint2x4, torch.quint4x2, torch.qint8, torch.qint32),
        _DTYPE_PRECISIONS[torch.float32],
    )
)


def default_tolerances(
    *inputs: torch.Tensor | torch.dtype,
    dtype_precisions: dict[torch.dtype, tuple[float, float]] | None = None,
) -> tuple[float, float]:
    # [FUNCTION DOC] Returns the loosest (maximum) relative and absolute tolerance over the dtypes
    # of the given tensors or dtypes, looking each up in the precision table (an optional custom
    # table can be passed); unknown dtypes count as zero tolerance. Raises TypeError if an input
    # is neither a tensor nor a dtype.
    """Returns the default absolute and relative testing tolerances for a set of inputs based on the dtype.

    See :func:`assert_close` for a table of the default tolerance for each dtype.

    Returns:
        (Tuple[float, float]): Loosest tolerances of all input dtypes.
    """
    dtypes = []
    for input in inputs:
        if isinstance(input, torch.Tensor):
            dtypes.append(input.dtype)
        elif isinstance(input, torch.dtype):
            dtypes.append(input)
        else:
            raise TypeError(
                f"Expected a torch.Tensor or a torch.dtype, but got {type(input)} instead."
            )
    dtype_precisions = dtype_precisions or _DTYPE_PRECISIONS
    rtols, atols = zip(
        *[dtype_precisions.get(dtype, (0.0, 0.0)) for dtype in dtypes], strict=True
    )
    return max(rtols), max(atols)


def get_tolerances(
    *inputs: torch.Tensor | torch.dtype,
    rtol: float | None,
    atol: float | None,
    id: tuple[Any, ...] = (),
) -> tuple[float, float]:
    # [FUNCTION DOC] Returns the (rtol, atol) pair to use. If both are given they are returned
    # unchanged; if neither is given, default tolerances for the inputs are returned. Raises
    # ErrorMeta carrying a ValueError if exactly one is given.
    """Gets absolute and relative to be used for numeric comparisons.

    If both ``rtol`` and ``atol`` are specified, this is a no-op. If both are not specified, the return value of
    :func:`default_tolerances` is used.

    Raises:
        ErrorMeta: With :class:`ValueError`, if only ``rtol`` or ``atol`` is specified.

    Returns:
        (Tuple[float, float]): Valid absolute and relative tolerances.
    """
    if (rtol is None) ^ (atol is None):
        # We require both tolerance to be omitted or specified, because specifying only one might lead to surprising
        # results. Imagine setting atol=0.0 and the tensors still match because rtol>0.0.
        raise ErrorMeta(
            ValueError,
            f"Both 'rtol' and 'atol' must be either specified or omitted, "
            f"but got no {'rtol' if rtol is None else 'atol'}.",
            id=id,
        )
    elif rtol is not None and atol is not None:
        return rtol, atol
    else:
        return default_tolerances(*inputs)


def _make_bitwise_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    first_mismatch_idx: tuple[int, ...] | None = None,
):
    # [FUNCTION DOC] HELPER: Builds the error text for a bitwise mismatch: a header with the
    # identifier (default or derived through a callable) followed by optional extra text and the
    # index of the first mismatched element. Returns the stripped string.
    """Makes a mismatch error message for bitwise values.

    Args:
        default_identifier (str): Default description of the compared values, e.g. "Tensor-likes".
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier that overrides
            ``default_identifier``. Can be passed as callable in which case it will be called with
            ``default_identifier`` to create the description at runtime.
        extra (Optional[str]): Extra information to be placed after the message header and the mismatch statistics.
        first_mismatch_idx (Optional[tuple[int, ...]]): the index of the first mismatch, for each dimension.
    """
    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not 'equal'!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"
    if first_mismatch_idx is not None:
        msg += f"The first mismatched element is at index {first_mismatch_idx}.\n"
    return msg.strip()


def _make_mismatch_msg(
    *,
    default_identifier: str,
    identifier: str | Callable[[str], str] | None = None,
    extra: str | None = None,
    abs_diff: float,
    abs_diff_idx: int | tuple[int, ...] | None = None,
    atol: float,
    rel_diff: float,
    rel_diff_idx: int | tuple[int, ...] | None = None,
    rtol: float,
) -> str:
    # [FUNCTION DOC] HELPER: Builds the error text for a numeric mismatch: header says equal or
    # close depending on whether both tolerances are zero, then optional extra text, then the
    # absolute and relative difference lines (with optional index and allowed tolerance). Returns
    # the stripped string.
    """Makes a mismatch error message for numeric values.

    Args:
        default_identifier (str): Default description of the compared values, e.g. "Tensor-likes".
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional identifier that overrides
            ``default_identifier``. Can be passed as callable in which case it will be called with
            ``default_identifier`` to create the description at runtime.
        extra (Optional[str]): Extra information to be placed after the message header and the mismatch statistics.
        abs_diff (float): Absolute difference.
        abs_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the absolute difference.
        atol (float): Allowed absolute tolerance. Will only be added to mismatch statistics if it or ``rtol`` are
            ``> 0``.
        rel_diff (float): Relative difference.
        rel_diff_idx (Optional[Union[int, Tuple[int, ...]]]): Optional index of the relative difference.
        rtol (float): Allowed relative tolerance. Will only be added to mismatch statistics if it or ``atol`` are
            ``> 0``.
    """
    equality = rtol == 0 and atol == 0

    def make_diff_msg(
        *,
        type: str,
        diff: float,
        idx: int | tuple[int, ...] | None,
        tol: float,
    ) -> str:
        # [FUNCTION DOC] HELPER: Formats one difference line for either absolute or relative
        # difference, including the index when known and the allowed tolerance when not doing an
        # equality check. Returns a line ending with a newline.
        if idx is None:
            msg = f"{type.title()} difference: {diff}"
        else:
            msg = f"Greatest {type} difference: {diff} at index {idx}"
        if not equality:
            msg += f" (up to {tol} allowed)"
        return msg + "\n"

    if identifier is None:
        identifier = default_identifier
    elif callable(identifier):
        identifier = identifier(default_identifier)

    msg = f"{identifier} are not {'equal' if equality else 'close'}!\n\n"

    if extra:
        msg += f"{extra.strip()}\n"

    msg += make_diff_msg(type="absolute", diff=abs_diff, idx=abs_diff_idx, tol=atol)
    msg += make_diff_msg(type="relative", diff=rel_diff, idx=rel_diff_idx, tol=rtol)

    return msg.strip()


def make_scalar_mismatch_msg(
    actual: bool | int | float | complex,
    expected: bool | int | float | complex,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
) -> str:
    # [FUNCTION DOC] Builds the mismatch error text for two Python scalars. Computes the absolute
    # difference and the relative difference (infinite if the expected value is zero) and formats
    # them with the shared message builder. Returns the string.
    """Makes a mismatch error message for scalars.

    Args:
        actual (Union[bool, int, float, complex]): Actual scalar.
        expected (Union[bool, int, float, complex]): Expected scalar.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional description for the scalars. Can be passed
            as callable in which case it will be called by the default value to create the description at runtime.
            Defaults to "Scalars".
    """
    abs_diff = abs(actual - expected)
    # pyrefly: ignore [bad-argument-type]
    rel_diff = float("inf") if expected == 0 else abs_diff / abs(expected)
    return _make_mismatch_msg(
        default_identifier="Scalars",
        identifier=identifier,
        extra=f"Expected {expected} but got {actual}.",
        abs_diff=abs_diff,
        atol=atol,
        rel_diff=rel_diff,
        rtol=rtol,
    )


def make_tensor_mismatch_msg(
    actual: torch.Tensor,
    expected: torch.Tensor,
    matches: torch.Tensor,
    *,
    rtol: float,
    atol: float,
    identifier: str | Callable[[str], str] | None = None,
):
    # [FUNCTION DOC] Builds the mismatch error text for two tensors given the boolean match mask.
    # Reports how many elements mismatch; for one-byte floating types it only gives the first
    # mismatch index (bitwise report). Otherwise it computes the greatest absolute and relative
    # difference over mismatches only (integer types are promoted to int64 first) and the
    # positions of those maxima. Returns the string.
    # KNOWN ISSUE: the relative difference divides by the expected value, which gives infinite or
    # NaN when expected is zero, and the one-byte float branch assumes at least one mismatch.
    """Makes a mismatch error message for tensors.

    Args:
        actual (torch.Tensor): Actual tensor.
        expected (torch.Tensor): Expected tensor.
        matches (torch.Tensor): Boolean mask of the same shape as ``actual`` and ``expected`` that indicates the
            location of matches.
        rtol (float): Relative tolerance.
        atol (float): Absolute tolerance.
        identifier (Optional[Union[str, Callable[[str], str]]]): Optional description for the tensors. Can be passed
            as callable in which case it will be called by the default value to create the description at runtime.
            Defaults to "Tensor-likes".
    """

    def unravel_flat_index(flat_index: int) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Converts a flat element position into a per-dimension index tuple
        # using the shape of the match mask; returns an empty tuple for zero-dimensional tensors.
        if not matches.shape:
            return ()

        inverse_index = []
        for size in matches.shape[::-1]:
            div, mod = divmod(flat_index, size)
            flat_index = div
            inverse_index.append(mod)

        return tuple(inverse_index[::-1])

    number_of_elements = matches.numel()
    total_mismatches = number_of_elements - int(torch.sum(matches))
    extra = (
        f"Mismatched elements: {total_mismatches} / {number_of_elements} "
        f"({total_mismatches / number_of_elements:.1%})"
    )
    if actual.dtype.is_floating_point and actual.dtype.itemsize == 1:
        # skip checking for max_abs_diff and max_rel_diff for float8-like values
        first_mismatch_idx = tuple(torch.nonzero(~matches, as_tuple=False)[0].tolist())
        return _make_bitwise_mismatch_msg(
            default_identifier="Tensor-likes",
            identifier=identifier,
            extra=extra,
            first_mismatch_idx=first_mismatch_idx,
        )

    actual_flat = actual.flatten()
    expected_flat = expected.flatten()
    matches_flat = matches.flatten()

    if not actual.dtype.is_floating_point and not actual.dtype.is_complex:
        # TODO: Instead of always upcasting to int64, it would be sufficient to cast to the next higher dtype to avoid
        #  overflow
        actual_flat = actual_flat.to(torch.int64)
        expected_flat = expected_flat.to(torch.int64)

    abs_diff = torch.abs(actual_flat - expected_flat)
    # Ensure that only mismatches are used for the max_abs_diff computation
    abs_diff[matches_flat] = 0
    max_abs_diff, max_abs_diff_flat_idx = torch.max(abs_diff, 0)

    rel_diff = abs_diff / torch.abs(expected_flat)
    # Ensure that only mismatches are used for the max_rel_diff computation
    rel_diff[matches_flat] = 0
    max_rel_diff, max_rel_diff_flat_idx = torch.max(rel_diff, 0)
    return _make_mismatch_msg(
        default_identifier="Tensor-likes",
        identifier=identifier,
        extra=extra,
        abs_diff=max_abs_diff.item(),
        abs_diff_idx=unravel_flat_index(int(max_abs_diff_flat_idx)),
        atol=atol,
        rel_diff=max_rel_diff.item(),
        rel_diff_idx=unravel_flat_index(int(max_rel_diff_flat_idx)),
        rtol=rtol,
    )


class UnsupportedInputs(Exception):
    # [CLASS DOC] Exception raised from a Pair constructor to say the pair type cannot handle the
    # given inputs, so the next pair type should be tried.
    """Exception to be raised during the construction of a :class:`Pair` in case it doesn't support the inputs."""


class Pair(abc.ABC):
    # [CLASS DOC] Abstract base class of all comparison pairs. A pair holds an actual value, an
    # expected value and the id locating them in nested inputs; the subclass compare method
    # performs the check and reports failures as ErrorMeta. Constructors receive all options and
    # keep unknown ones; a constructor raises UnsupportedInputs to decline. Subclasses add
    # extra_repr to show their options.
    """ABC for all comparison pairs to be used in conjunction with :func:`assert_equal`.

    Each subclass needs to overwrite :meth:`Pair.compare` that performs the actual comparison.

    Each pair receives **all** options, so select the ones applicable for the subclass and forward the rest to the
    super class. Raising an :class:`UnsupportedInputs` during constructions indicates that the pair is not able to
    handle the inputs and the next pair type will be tried.

    All other errors should be raised as :class:`ErrorMeta`. After the instantiation, :meth:`Pair._make_error_meta` can
    be used to automatically handle overwriting the message with a user supplied one and id handling.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        **unknown_parameters: Any,
    ) -> None:
        # [FUNCTION DOC] Stores actual, expected, the id and any unrecognized options.
        self.actual = actual
        self.expected = expected
        self.id = id
        self._unknown_parameters = unknown_parameters

    @staticmethod
    def _inputs_not_supported() -> NoReturn:
        # [FUNCTION DOC] HELPER: Raises UnsupportedInputs. Never returns.
        raise UnsupportedInputs

    @staticmethod
    def _check_inputs_isinstance(*inputs: Any, cls: type | tuple[type, ...]):
        # [FUNCTION DOC] HELPER: Raises UnsupportedInputs unless every input is an instance of the
        # given class or tuple of classes.
        """Checks if all inputs are instances of a given class and raise :class:`UnsupportedInputs` otherwise."""
        if not all(isinstance(input, cls) for input in inputs):
            Pair._inputs_not_supported()

    def _fail(
        self, type: type[Exception], msg: str, *, id: tuple[Any, ...] = ()
    ) -> NoReturn:
        # [FUNCTION DOC] HELPER: Raises an ErrorMeta with the given exception type and message.
        # The id is the explicit one, or the stored id of the pair if none is given and one
        # exists. Never returns.
        """Raises an :class:`ErrorMeta` from a given exception type and message and the stored id.

        .. warning::

            If you use this before the ``super().__init__(...)`` call in the constructor, you have to pass the ``id``
            explicitly.
        """
        raise ErrorMeta(type, msg, id=self.id if not id and hasattr(self, "id") else id)

    @abc.abstractmethod
    def compare(self) -> None:
        # [FUNCTION DOC] Abstract: compares actual with expected and raises ErrorMeta on mismatch;
        # returns nothing.
        """Compares the inputs and raises an :class`ErrorMeta` in case they mismatch."""

    def extra_repr(self) -> Sequence[str | tuple[str, Any]]:
        # [FUNCTION DOC] Returns the extra options to show in the representation, as names of
        # attributes or name and value pairs; empty here.
        """Returns extra information that will be included in the representation.

        Should be overwritten by all subclasses that use additional options. The representation of the object will only
        be surfaced in case we encounter an unexpected error and thus should help debug the issue. Can be a sequence of
        key-value-pairs or attribute names.
        """
        return []

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Returns a multi-line text with the class name, id, actual,
        # expected and every extra representation item, one per line.
        head = f"{type(self).__name__}("
        tail = ")"
        body = [
            f"    {name}={value!s},"
            for name, value in [
                ("id", self.id),
                ("actual", self.actual),
                ("expected", self.expected),
                *[
                    (extra, getattr(self, extra)) if isinstance(extra, str) else extra
                    for extra in self.extra_repr()
                ],
            ]
        ]
        return "\n".join((head, *body, *tail))


class ObjectPair(Pair):
    # [CLASS DOC] Fallback pair for any inputs: compares with the equality operator. It accepts
    # every input, so it must be last in the list of pair types.
    """Pair for any type of inputs that will be compared with the `==` operator.

    .. note::

        Since this will instantiate for any kind of inputs, it should only be used as fallback after all other pairs
        couldn't handle the inputs.

    """

    def compare(self) -> None:
        # [FUNCTION DOC] Compares with equality. Raises ErrorMeta (ValueError) if the equality
        # operation itself raises, chaining the original error, and ErrorMeta (AssertionError) if
        # the values are not equal.
        try:
            equal = self.actual == self.expected
        except Exception as error:
            # We are not using `self._raise_error_meta` here since we need the exception chaining
            raise ErrorMeta(
                ValueError,
                f"{self.actual} == {self.expected} failed with:\n{error}.",
                id=self.id,
            ) from error

        if not equal:
            self._fail(AssertionError, f"{self.actual} != {self.expected}")


class NonePair(Pair):
    # [CLASS DOC] Pair for None: handles inputs where either side is None, and requires that both
    # are None.
    """Pair for ``None`` inputs."""

    def __init__(self, actual: Any, expected: Any, **other_parameters: Any) -> None:
        # [FUNCTION DOC] Declines the inputs (UnsupportedInputs) unless at least one is None, then
        # initializes the base pair.
        if not (actual is None or expected is None):
            self._inputs_not_supported()

        super().__init__(actual, expected, **other_parameters)

    def compare(self) -> None:
        # [FUNCTION DOC] Raises ErrorMeta (AssertionError) unless both values are None.
        if not (self.actual is None and self.expected is None):
            self._fail(
                AssertionError, f"None mismatch: {self.actual} is not {self.expected}"
            )


class BooleanPair(Pair):
    # [CLASS DOC] Pair for booleans, including numpy bool values when numpy is available. Inputs
    # are converted to Python bool at construction; compare requires identical truth values.
    """Pair for :class:`bool` inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.bool_` inputs.

    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...],
        **other_parameters: Any,
    ) -> None:
        # [FUNCTION DOC] Converts both inputs to Python bool (declining unsupported types) and
        # initializes the base pair. The id is a required keyword.
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, **other_parameters)

    @property
    def _supported_types(self) -> tuple[type, ...]:
        # [FUNCTION DOC] HELPER: Property returning the tuple of accepted types: bool, plus the
        # numpy bool type when numpy is available.
        cls: list[type] = [bool]
        if HAS_NUMPY:
            # pyrefly: ignore [missing-attribute]
            cls.append(np.bool_)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[bool, bool]:
        # [FUNCTION DOC] HELPER: Declines inputs that are not supported booleans, then converts
        # both to Python bool. Returns the converted pair.
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        actual, expected = (
            self._to_bool(bool_like, id=id) for bool_like in (actual, expected)
        )
        return actual, expected

    def _to_bool(self, bool_like: Any, *, id: tuple[Any, ...]) -> bool:
        # [FUNCTION DOC] HELPER: Converts a bool or numpy bool to a Python bool. Raises ErrorMeta
        # (TypeError) for any other type.
        if isinstance(bool_like, bool):
            return bool_like
        # pyrefly: ignore [missing-attribute]
        elif isinstance(bool_like, np.bool_):
            return bool_like.item()
        else:
            raise ErrorMeta(
                TypeError, f"Unknown boolean type {type(bool_like)}.", id=id
            )

    def compare(self) -> None:
        # [FUNCTION DOC] Raises ErrorMeta (AssertionError) if the two booleans differ.
        if self.actual is not self.expected:
            self._fail(
                AssertionError,
                f"Booleans mismatch: {self.actual} is not {self.expected}",
            )


class NumberPair(Pair):
    # [CLASS DOC] Pair for Python numbers (int, float, complex) and numpy numbers. Stores rtol and
    # atol (defaulting from the number types treated as int64, float64 and complex128), equal_nan
    # and check_dtype. compare uses an absolute-plus-relative tolerance test.
    """Pair for Python number (:class:`int`, :class:`float`, and :class:`complex`) inputs.

    .. note::

        If ``numpy`` is available, also handles :class:`numpy.number` inputs.

    Kwargs:
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the type are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the type are selected with the below table.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_dtype (bool): If ``True``, the type of the inputs will be checked for equality. Defaults to ``False``.

    The following table displays correspondence between Python number type and the ``torch.dtype``'s. See
    :func:`assert_close` for the corresponding tolerances.

    +------------------+-------------------------------+
    | ``type``         | corresponding ``torch.dtype`` |
    +==================+===============================+
    | :class:`int`     | :attr:`~torch.int64`          |
    +------------------+-------------------------------+
    | :class:`float`   | :attr:`~torch.float64`        |
    +------------------+-------------------------------+
    | :class:`complex` | :attr:`~torch.complex64`      |
    +------------------+-------------------------------+
    """

    _TYPE_TO_DTYPE = {
        int: torch.int64,
        float: torch.float64,
        complex: torch.complex128,
    }
    _NUMBER_TYPES = tuple(_TYPE_TO_DTYPE.keys())

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_dtype: bool = False,
        **other_parameters: Any,
    ) -> None:
        # [FUNCTION DOC] Converts both inputs to Python numbers (declining other types),
        # initializes the base pair, resolves tolerances from the corresponding dtypes
        # (raising ErrorMeta ValueError if only one of rtol and atol is given), and stores
        # equal_nan and check_dtype.
        # KNOWN ISSUE: the class docstring table maps complex to complex64, but the code table
        # uses complex128.
        actual, expected = self._process_inputs(actual, expected, id=id)
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            *[self._TYPE_TO_DTYPE[type(input)] for input in (actual, expected)],
            rtol=rtol,
            atol=atol,
            id=id,
        )
        self.equal_nan = equal_nan
        self.check_dtype = check_dtype

    @property
    def _supported_types(self) -> tuple[type, ...]:
        # [FUNCTION DOC] HELPER: Property returning the accepted types: int, float, complex, plus
        # the numpy number type when numpy is available.
        cls = list(self._NUMBER_TYPES)
        if HAS_NUMPY:
            # pyrefly: ignore [missing-attribute]
            cls.append(np.number)
        return tuple(cls)

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...]
    ) -> tuple[int | float | complex, int | float | complex]:
        # [FUNCTION DOC] HELPER: Declines unsupported inputs, then converts both to Python
        # numbers. Returns the converted pair.
        self._check_inputs_isinstance(actual, expected, cls=self._supported_types)
        actual, expected = (
            self._to_number(number_like, id=id) for number_like in (actual, expected)
        )
        return actual, expected

    def _to_number(
        self, number_like: Any, *, id: tuple[Any, ...]
    ) -> int | float | complex:
        # [FUNCTION DOC] HELPER: Converts numpy numbers through their item method and passes
        # Python numbers through. Raises ErrorMeta (TypeError) for any other type.
        # pyrefly: ignore [missing-attribute]
        if HAS_NUMPY and isinstance(number_like, np.number):
            return number_like.item()
        elif isinstance(number_like, self._NUMBER_TYPES):
            return number_like  # type: ignore[return-value]
        else:
            raise ErrorMeta(
                TypeError, f"Unknown number type {type(number_like)}.", id=id
            )

    def compare(self) -> None:
        # [FUNCTION DOC] Compares the numbers. If check_dtype is set, raises ErrorMeta
        # (AssertionError) when the Python types differ. Equal values pass; two NaNs pass if
        # equal_nan is set; otherwise passes when the absolute difference is finite and within
        # atol plus rtol times the expected magnitude. Else raises ErrorMeta (AssertionError) with
        # a scalar mismatch message.
        if self.check_dtype and type(self.actual) is not type(self.expected):
            self._fail(
                AssertionError,
                f"The (d)types do not match: {type(self.actual)} != {type(self.expected)}.",
            )

        if self.actual == self.expected:
            return

        if self.equal_nan and cmath.isnan(self.actual) and cmath.isnan(self.expected):
            return

        abs_diff = abs(self.actual - self.expected)
        tolerance = self.atol + self.rtol * abs(self.expected)

        if cmath.isfinite(abs_diff) and abs_diff <= tolerance:
            return

        self._fail(
            AssertionError,
            make_scalar_mismatch_msg(
                self.actual, self.expected, rtol=self.rtol, atol=self.atol
            ),
        )

    def extra_repr(self) -> Sequence[str]:
        # [FUNCTION DOC] Returns the names of the options to show in the representation: rtol,
        # atol, equal_nan and check_dtype.
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_dtype",
        )


class TensorLikePair(Pair):
    # [CLASS DOC] Pair for tensors and objects convertible to tensors. Stores tolerances and the
    # flags check_device, check_dtype, check_layout, check_stride and equal_nan. compare first
    # checks attributes (shape, quantization, layout, stride, device, dtype), skips value checks
    # for meta tensors, then equalizes device, dtype and layout and compares values with a
    # layout-specific strategy (quantized, sparse COO, sparse compressed, jagged, one-byte float,
    # or regular).
    # KNOWN ISSUE: the class docstring leaves allow_subclasses undescribed.
    """Pair for :class:`torch.Tensor`-like inputs.

    Kwargs:
        allow_subclasses (bool):
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the type are selected. See :func:`assert_close` for details.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the type are selected. See :func:`assert_close` for details.
        equal_nan (bool): If ``True``, two ``NaN`` values are considered equal. Defaults to ``False``.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same ``layout``. If this
            check is disabled, tensors with different ``layout``'s are converted to strided tensors before being
            compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
    """

    def __init__(
        self,
        actual: Any,
        expected: Any,
        *,
        id: tuple[Any, ...] = (),
        allow_subclasses: bool = True,
        rtol: float | None = None,
        atol: float | None = None,
        equal_nan: bool = False,
        check_device: bool = True,
        check_dtype: bool = True,
        check_layout: bool = True,
        check_stride: bool = False,
        **other_parameters: Any,
    ):
        # [FUNCTION DOC] Validates and converts inputs to tensors (declining non-related types, or
        # differing types when subclasses are disallowed), initializes the base pair, resolves
        # tolerances (raising ErrorMeta ValueError if only one of rtol and atol is given), and
        # stores the check flags.
        actual, expected = self._process_inputs(
            actual, expected, id=id, allow_subclasses=allow_subclasses
        )
        super().__init__(actual, expected, id=id, **other_parameters)

        self.rtol, self.atol = get_tolerances(
            actual, expected, rtol=rtol, atol=atol, id=self.id
        )
        self.equal_nan = equal_nan
        self.check_device = check_device
        self.check_dtype = check_dtype
        self.check_layout = check_layout
        self.check_stride = check_stride

    def _process_inputs(
        self, actual: Any, expected: Any, *, id: tuple[Any, ...], allow_subclasses: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Requires the inputs to be directly related by subclassing (and of
        # identical type if subclasses are not allowed), otherwise declines. Converts both to
        # tensors and checks that the layout is supported. Returns the two tensors.
        directly_related = isinstance(actual, type(expected)) or isinstance(
            expected, type(actual)
        )
        if not directly_related:
            self._inputs_not_supported()

        if not allow_subclasses and type(actual) is not type(expected):
            self._inputs_not_supported()

        actual, expected = (self._to_tensor(input) for input in (actual, expected))
        for tensor in (actual, expected):
            self._check_supported(tensor, id=id)
        return actual, expected

    def _to_tensor(self, tensor_like: Any) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Returns the tensor itself, or converts the object with the
        # as_tensor function. Declines the inputs if conversion fails.
        if isinstance(tensor_like, torch.Tensor):
            return tensor_like

        try:
            return torch.as_tensor(tensor_like)
        except Exception:
            self._inputs_not_supported()

    def _check_supported(self, tensor: torch.Tensor, *, id: tuple[Any, ...]) -> None:
        # [FUNCTION DOC] HELPER: Raises ErrorMeta (ValueError) if the tensor layout is not
        # strided, jagged or one of the sparse layouts.
        if tensor.layout not in {
            torch.strided,
            torch.jagged,
            torch.sparse_coo,
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            raise ErrorMeta(
                ValueError, f"Unsupported tensor layout {tensor.layout}", id=id
            )

    def compare(self) -> None:
        # [FUNCTION DOC] Checks the attributes first; returns early if either tensor is on the
        # meta device; otherwise equalizes attributes and compares values. Raises ErrorMeta on
        # any mismatch.
        actual, expected = self.actual, self.expected

        self._compare_attributes(actual, expected)
        if any(input.device.type == "meta" for input in (actual, expected)):
            return

        actual, expected = self._equalize_attributes(actual, expected)
        self._compare_values(actual, expected)

    def _compare_attributes(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
    ) -> None:
        # [FUNCTION DOC] HELPER: Checks shape, whether both are quantized and the quantization
        # scheme, then optionally layout (or stride for strided tensors), device and dtype
        # according to the check flags. Raises ErrorMeta (AssertionError) via a local
        # reporting helper.
        """Checks if the attributes of two tensors match.

        Always checks

        - the :attr:`~torch.Tensor.shape`,
        - whether both inputs are quantized or not,
        - and if they use the same quantization scheme.

        Checks for

        - :attr:`~torch.Tensor.layout`,
        - :meth:`~torch.Tensor.stride`,
        - :attr:`~torch.Tensor.device`, and
        - :attr:`~torch.Tensor.dtype`

        are optional and can be disabled through the corresponding ``check_*`` flag during construction of the pair.
        """

        def raise_mismatch_error(
            attribute_name: str, actual_value: Any, expected_value: Any
        ) -> NoReturn:
            # [FUNCTION DOC] HELPER: Raises an ErrorMeta (AssertionError) saying the named
            # attribute differs between actual and expected.
            self._fail(
                AssertionError,
                f"The values for attribute '{attribute_name}' do not match: {actual_value} != {expected_value}.",
            )

        if actual.shape != expected.shape:
            raise_mismatch_error("shape", actual.shape, expected.shape)

        if actual.is_quantized != expected.is_quantized:
            raise_mismatch_error(
                "is_quantized", actual.is_quantized, expected.is_quantized
            )
        elif actual.is_quantized and actual.qscheme() != expected.qscheme():
            raise_mismatch_error("qscheme()", actual.qscheme(), expected.qscheme())

        if actual.layout != expected.layout:
            if self.check_layout:
                raise_mismatch_error("layout", actual.layout, expected.layout)
        elif (
            actual.layout == torch.strided
            and self.check_stride
            and actual.stride() != expected.stride()
        ):
            raise_mismatch_error("stride()", actual.stride(), expected.stride())

        if self.check_device and actual.device != expected.device:
            raise_mismatch_error("device", actual.device, expected.device)

        if self.check_dtype and actual.dtype != expected.dtype:
            raise_mismatch_error("dtype", actual.dtype, expected.dtype)

    def _equalize_attributes(
        self, actual: torch.Tensor, expected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Brings the tensors to a comparable form: MPS tensors and tensors
        # on different devices move to CPU; differing dtypes are promoted to a common dtype
        # (unsigned 16, 32 and 64-bit types are first treated as int64); differing layouts are
        # converted to dense. Returns the pair of tensors.
        """Equalizes some attributes of two tensors for value comparison.

        If ``actual`` and ``expected`` are ...

        - ... not on the same :attr:`~torch.Tensor.device`, they are moved CPU memory.
        - ... not of the same ``dtype``, they are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`).
        - ... not of the same ``layout``, they are converted to strided tensors.

        Args:
            actual (Tensor): Actual tensor.
            expected (Tensor): Expected tensor.

        Returns:
            (Tuple[Tensor, Tensor]): Equalized tensors.
        """
        # The comparison logic uses operators currently not supported by the MPS backends.
        #  See https://github.com/pytorch/pytorch/issues/77144 for details.
        # TODO: Remove this conversion as soon as all operations are supported natively by the MPS backend
        if actual.is_mps or expected.is_mps:  # type: ignore[attr-defined]
            actual = actual.cpu()
            expected = expected.cpu()

        if actual.device != expected.device:
            actual = actual.cpu()
            expected = expected.cpu()

        if actual.dtype != expected.dtype:
            actual_dtype = actual.dtype
            expected_dtype = expected.dtype
            # For uint64, this is not sound in general, which is why promote_types doesn't
            # allow it, but for easy testing, we're unlikely to get confused
            # by large uint64 overflowing into negative int64
            if actual_dtype in [torch.uint64, torch.uint32, torch.uint16]:
                actual_dtype = torch.int64
            if expected_dtype in [torch.uint64, torch.uint32, torch.uint16]:
                expected_dtype = torch.int64
            dtype = torch.promote_types(actual_dtype, expected_dtype)
            actual = actual.to(dtype)
            expected = expected.to(dtype)

        if actual.layout != expected.layout:
            # These checks are needed, since Tensor.to_dense() fails on tensors that are already strided
            actual = actual.to_dense() if actual.layout != torch.strided else actual
            expected = (
                expected.to_dense() if expected.layout != torch.strided else expected
            )

        return actual, expected

    def _compare_values(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        # [FUNCTION DOC] HELPER: Chooses the comparison strategy from the tensor kind: quantized,
        # sparse COO, sparse compressed, jagged (compares the packed values), one-byte floating
        # point (bitwise equality only) or regular, and calls it with the stored tolerances and
        # equal_nan. Raises ErrorMeta on mismatch.
        if actual.is_quantized:
            compare_fn = self._compare_quantized_values
        elif actual.is_sparse:
            compare_fn = self._compare_sparse_coo_values
        elif actual.layout in {
            torch.sparse_csr,
            torch.sparse_csc,
            torch.sparse_bsr,
            torch.sparse_bsc,
        }:
            compare_fn = self._compare_sparse_compressed_values
        elif actual.layout == torch.jagged:
            actual, expected = actual.values(), expected.values()
            compare_fn = self._compare_regular_values_close
        elif actual.dtype.is_floating_point and actual.dtype.itemsize == 1:

            def bitwise_comp(
                actual: torch.Tensor,
                expected: torch.Tensor,
                *,
                rtol: float,
                atol: float,
                equal_nan: bool,
                identifier: str | Callable[[str], str] | None = None,
            ) -> None:
                # [FUNCTION DOC] HELPER: Bitwise comparison for one-byte floating types: raises
                # ErrorMeta (AssertionError) if a nonzero tolerance is requested, then does an
                # exact value comparison.
                if rtol != 0.0 or atol != 0.0:
                    raise ErrorMeta(
                        AssertionError,
                        f"Rtol={rtol} and atol={atol} are not supported for bitwise comparison of low"
                        " dimensional floats. Please use rtol=0.0 and atol=0.0.",
                    )

                return self._compare_regular_values_close(
                    actual,
                    expected,
                    rtol=rtol,
                    atol=atol,
                    equal_nan=equal_nan,
                    identifier=identifier,
                )

            compare_fn = bitwise_comp
        else:
            compare_fn = self._compare_regular_values_close

        compare_fn(
            actual, expected, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        )

    def _compare_quantized_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Compares quantized tensors by dequantizing both and comparing as
        # regular values with the given tolerances. Raises ErrorMeta on mismatch.
        """Compares quantized tensors by comparing the :meth:`~torch.Tensor.dequantize`'d variants for closeness.

        .. note::

            A detailed discussion about why only the dequantized variant is checked for closeness rather than checking
            the individual quantization parameters for closeness and the integer representation for equality can be
            found in https://github.com/pytorch/pytorch/issues/68548.
        """
        return self._compare_regular_values_close(
            actual.dequantize(),
            expected.dequantize(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier=lambda default_identifier: f"Quantized {default_identifier.lower()}",
        )

    def _compare_sparse_coo_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Compares sparse COO tensors: sparse dimension counts and stored
        # value counts must match (ErrorMeta AssertionError otherwise), indices must be exactly
        # equal and values close.
        """Compares sparse COO tensors by comparing

        - the number of sparse dimensions,
        - the number of non-zero elements (nnz) for equality,
        - the indices for equality, and
        - the values for closeness.
        """
        if actual.sparse_dim() != expected.sparse_dim():
            self._fail(
                AssertionError,
                (
                    f"The number of sparse dimensions in sparse COO tensors does not match: "
                    f"{actual.sparse_dim()} != {expected.sparse_dim()}"
                ),
            )

        if actual._nnz() != expected._nnz():
            self._fail(
                AssertionError,
                (
                    f"The number of specified values in sparse COO tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        self._compare_regular_values_equal(
            actual._indices(),
            expected._indices(),
            identifier="Sparse COO indices",
        )
        self._compare_regular_values_close(
            actual._values(),
            expected._values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier="Sparse COO values",
        )

    def _compare_sparse_compressed_values(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
    ) -> None:
        # [FUNCTION DOC] HELPER: Compares sparse compressed (CSR, CSC, BSR, BSC) tensors: stored
        # value counts must match, compressed and plain indices must be equal after promotion to a
        # common integer dtype, and values must be close. Raises ErrorMeta on mismatch.
        """Compares sparse compressed tensors by comparing

        - the number of non-zero elements (nnz) for equality,
        - the plain indices for equality,
        - the compressed indices for equality, and
        - the values for closeness.
        """
        format_name, compressed_indices_method, plain_indices_method = {
            torch.sparse_csr: (
                "CSR",
                torch.Tensor.crow_indices,
                torch.Tensor.col_indices,
            ),
            torch.sparse_csc: (
                "CSC",
                torch.Tensor.ccol_indices,
                torch.Tensor.row_indices,
            ),
            torch.sparse_bsr: (
                "BSR",
                torch.Tensor.crow_indices,
                torch.Tensor.col_indices,
            ),
            torch.sparse_bsc: (
                "BSC",
                torch.Tensor.ccol_indices,
                torch.Tensor.row_indices,
            ),
        }[actual.layout]

        if actual._nnz() != expected._nnz():
            self._fail(
                AssertionError,
                (
                    f"The number of specified values in sparse {format_name} tensors does not match: "
                    f"{actual._nnz()} != {expected._nnz()}"
                ),
            )

        # Compressed and plain indices in the CSR / CSC / BSR / BSC sparse formats can be `torch.int32` _or_
        # `torch.int64`. While the same dtype is enforced for the compressed and plain indices of a single tensor, it
        # can be different between two tensors. Thus, we need to convert them to the same dtype, or the comparison will
        # fail.
        actual_compressed_indices = compressed_indices_method(actual)
        expected_compressed_indices = compressed_indices_method(expected)
        indices_dtype = torch.promote_types(
            actual_compressed_indices.dtype, expected_compressed_indices.dtype
        )

        self._compare_regular_values_equal(
            actual_compressed_indices.to(indices_dtype),
            expected_compressed_indices.to(indices_dtype),
            identifier=f"Sparse {format_name} {compressed_indices_method.__name__}",
        )
        self._compare_regular_values_equal(
            plain_indices_method(actual).to(indices_dtype),
            plain_indices_method(expected).to(indices_dtype),
            identifier=f"Sparse {format_name} {plain_indices_method.__name__}",
        )
        self._compare_regular_values_close(
            actual.values(),
            expected.values(),
            rtol=rtol,
            atol=atol,
            equal_nan=equal_nan,
            identifier=f"Sparse {format_name} values",
        )

    def _compare_regular_values_equal(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        equal_nan: bool = False,
        identifier: str | Callable[[str], str] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Checks exact equality of values by calling the closeness check
        # with zero tolerances. Raises ErrorMeta on mismatch.
        """Checks if the values of two tensors are equal."""
        self._compare_regular_values_close(
            actual, expected, rtol=0, atol=0, equal_nan=equal_nan, identifier=identifier
        )

    def _compare_regular_values_close(
        self,
        actual: torch.Tensor,
        expected: torch.Tensor,
        *,
        rtol: float,
        atol: float,
        equal_nan: bool,
        identifier: str | Callable[[str], str] | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Checks element closeness with an elementwise closeness test. On
        # failure builds a scalar message for zero-dimensional tensors or a tensor message
        # otherwise, and raises ErrorMeta (AssertionError).
        """Checks if the values of two tensors are close up to a desired tolerance."""
        matches = torch.isclose(
            actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan
        )
        if torch.all(matches):
            return

        if actual.shape == torch.Size([]):
            msg = make_scalar_mismatch_msg(
                actual.item(),
                expected.item(),
                rtol=rtol,
                atol=atol,
                identifier=identifier,
            )
        else:
            msg = make_tensor_mismatch_msg(
                actual, expected, matches, rtol=rtol, atol=atol, identifier=identifier
            )
        self._fail(AssertionError, msg)

    def extra_repr(self) -> Sequence[str]:
        # [FUNCTION DOC] Returns the names of the options to show in the representation:
        # tolerances, equal_nan and the check flags.
        return (
            "rtol",
            "atol",
            "equal_nan",
            "check_device",
            "check_dtype",
            "check_layout",
            "check_stride",
        )


def originate_pairs(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]],
    sequence_types: tuple[type, ...] = (collections.abc.Sequence,),
    mapping_types: tuple[type, ...] = (collections.abc.Mapping,),
    id: tuple[Any, ...] = (),
    **options: Any,
    # pyrefly: ignore [bad-return]
) -> list[Pair]:
    # [FUNCTION DOC] Walks the inputs and returns the list of pairs to compare. Sequences (not
    # strings) of equal length and mappings with equal key sets are recursed element by element,
    # extending the id with the index or key; dataclasses are compared as a whole object if their
    # equality gives a real bool, else field by field (compare-enabled fields only). Otherwise
    # the first pair type that accepts the inputs is used. Raises ErrorMeta for mismatched
    # lengths or keys or when no pair type accepts, and RuntimeError for unexpected errors from a
    # pair constructor.
    # KNOWN ISSUE: the docstring says dataclass handling raises nothing special, but a
    # RuntimeError from equality that is not about tensor truth values is re-raised.
    """Originates pairs from the individual inputs.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s,
    :class:`~collections.abc.Mapping`'s, or dataclass instances. In this case the pairs are
    originated by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of pair types that will be tried to construct with the inputs.
            First successful pair will be used.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        id (Tuple[Any, ...]): Optional id of a pair that will be included in an error message.
        **options (Any): Options passed to each pair during construction.

    Raises:
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Sequence`'s, but their
            length does not match.
        ErrorMeta: With :class`AssertionError`, if the inputs are :class:`~collections.abc.Mapping`'s, but their set of
            keys do not match.
        ErrorMeta: With :class`TypeError`, if no pair is able to handle the inputs.
        ErrorMeta: With any expected exception that happens during the construction of a pair.

    Returns:
        (List[Pair]): Originated pairs.
    """
    # We explicitly exclude str's here since they are self-referential and would cause an infinite recursion loop:
    # "a" == "a"[0][0]...
    if (
        isinstance(actual, sequence_types)
        and not isinstance(actual, str)
        and isinstance(expected, sequence_types)
        and not isinstance(expected, str)
    ):
        actual_len = len(actual)  # type: ignore[arg-type]
        expected_len = len(expected)  # type: ignore[arg-type]
        if actual_len != expected_len:
            raise ErrorMeta(
                AssertionError,
                f"The length of the sequences mismatch: {actual_len} != {expected_len}",
                id=id,
            )

        pairs = []
        for idx in range(actual_len):
            pairs.extend(
                originate_pairs(
                    actual[idx],  # type: ignore[index]
                    expected[idx],  # type: ignore[index]
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, idx),
                    **options,
                )
            )
        return pairs

    elif isinstance(actual, mapping_types) and isinstance(expected, mapping_types):
        actual_keys = set(actual.keys())  # type: ignore[attr-defined]
        expected_keys = set(expected.keys())  # type: ignore[attr-defined]
        if actual_keys != expected_keys:
            missing_keys = expected_keys - actual_keys
            additional_keys = actual_keys - expected_keys
            raise ErrorMeta(
                AssertionError,
                (
                    f"The keys of the mappings do not match:\n"
                    f"Missing keys in the actual mapping: {sorted(missing_keys)}\n"
                    f"Additional keys in the actual mapping: {sorted(additional_keys)}"
                ),
                id=id,
            )

        keys: Collection = actual_keys
        # Since the origination aborts after the first failure, we try to be deterministic
        with contextlib.suppress(Exception):
            keys = sorted(keys)

        pairs = []
        for key in keys:
            pairs.extend(
                originate_pairs(
                    actual[key],  # type: ignore[index]
                    expected[key],  # type: ignore[index]
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, key),
                    **options,
                )
            )
        return pairs

    # Dataclass instances normally compare via ``==`` / ObjectPair so custom ``__eq__``
    # (including ``eq=False`` classes) is respected. Generated dataclass ``__eq__``
    # treats each field comparison as a Python bool, so multi-element tensor fields
    # raise on Python 3.13+ (no same-object shortcut), or return a Tensor when that
    # field is the sole compare=True field. Only then recurse field-by-field so
    # tensors use the normal tensor comparison path.
    #
    # Skip is_dataclass for types Dynamo mishandles (Tensor, ndarray, UnionType, ...).
    elif (
        not isinstance(actual, _IS_DATACLASS_SKIP_TYPES)
        and not isinstance(expected, _IS_DATACLASS_SKIP_TYPES)
        and dataclasses.is_dataclass(actual)
        and dataclasses.is_dataclass(expected)
        and type(actual) is type(expected)
    ):
        try:
            equal = actual == expected
        except RuntimeError as error:
            if "Boolean value of Tensor" not in str(error):
                raise
        else:
            # Single tensor-field dataclasses can return a Tensor from __eq__
            # without raising; only a real bool means == is usable as-is.
            if type(equal) is bool:
                return [ObjectPair(actual, expected, id=id, **options)]

        pairs = []
        # pyrefly: ignore [bad-argument-type]  # narrowed by is_dataclass above
        for field in dataclasses.fields(actual):
            if not field.compare:
                continue
            pairs.extend(
                originate_pairs(
                    getattr(actual, field.name),
                    getattr(expected, field.name),
                    pair_types=pair_types,
                    sequence_types=sequence_types,
                    mapping_types=mapping_types,
                    id=(*id, field.name),
                    **options,
                )
            )
        return pairs

    else:
        for pair_type in pair_types:
            try:
                # pyrefly: ignore [bad-instantiation]
                return [pair_type(actual, expected, id=id, **options)]
            # Raising an `UnsupportedInputs` during origination indicates that the pair type is not able to handle the
            # inputs. Thus, we try the next pair type.
            except UnsupportedInputs:
                continue
            # Raising an `ErrorMeta` during origination is the orderly way to abort and so we simply re-raise it. This
            # is only in a separate branch, because the one below would also except it.
            except ErrorMeta:
                raise
            # Raising any other exception during origination is unexpected and will give some extra information about
            # what happened. If applicable, the exception should be expected in the future.
            except Exception as error:
                raise RuntimeError(
                    f"Originating a {pair_type.__name__}() at item {''.join(str([item]) for item in id)} with\n\n"
                    f"{type(actual).__name__}(): {actual}\n\n"
                    f"and\n\n"
                    f"{type(expected).__name__}(): {expected}\n\n"
                    f"resulted in the unexpected exception above. "
                    f"If you are a user and see this message during normal operation "
                    "please file an issue at https://github.com/pytorch/pytorch/issues. "
                    "If you are a developer and working on the comparison functions, "
                    "please except the previous error and raise an expressive `ErrorMeta` instead."
                ) from error
        else:
            raise ErrorMeta(
                TypeError,
                f"No comparison pair was able to handle inputs of type {type(actual)} and {type(expected)}.",
                id=id,
            )


def not_close_error_metas(
    actual: Any,
    expected: Any,
    *,
    pair_types: Sequence[type[Pair]] = (ObjectPair,),
    sequence_types: tuple[type, ...] = (collections.abc.Sequence,),
    mapping_types: tuple[type, ...] = (collections.abc.Mapping,),
    **options: Any,
) -> list[ErrorMeta]:
    # [FUNCTION DOC] Builds the pairs and compares each, returning the list of ErrorMeta for
    # failing pairs (empty means everything matched). Origination errors are converted to real
    # exceptions and raised; unexpected exceptions from a comparison are wrapped in RuntimeError.
    # Hides itself from pytest tracebacks. The returned list is built in a way that breaks a
    # reference cycle between frames and errors.
    # KNOWN ISSUE: the docstring says this asserts that inputs are equal, but it only returns
    # error metadata and raises only for origination failures.
    """Asserts that inputs are equal.

    ``actual`` and ``expected`` can be possibly nested :class:`~collections.abc.Sequence`'s or
    :class:`~collections.abc.Mapping`'s. In this case the comparison happens elementwise by recursing through them.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        pair_types (Sequence[Type[Pair]]): Sequence of :class:`Pair` types that will be tried to construct with the
            inputs. First successful pair will be used. Defaults to only using :class:`ObjectPair`.
        sequence_types (Tuple[Type, ...]): Optional types treated as sequences that will be checked elementwise.
        mapping_types (Tuple[Type, ...]): Optional types treated as mappings that will be checked elementwise.
        **options (Any): Options passed to each pair during construction.
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    try:
        pairs = originate_pairs(
            actual,
            expected,
            pair_types=pair_types,
            sequence_types=sequence_types,
            mapping_types=mapping_types,
            **options,
        )
    except ErrorMeta as error_meta:
        # Explicitly raising from None to hide the internal traceback
        raise error_meta.to_error() from None

    error_metas: list[ErrorMeta] = []
    for pair in pairs:
        try:
            pair.compare()
        except ErrorMeta as error_meta:
            error_metas.append(error_meta)
        # Raising any exception besides `ErrorMeta` while comparing is unexpected and will give some extra information
        # about what happened. If applicable, the exception should be expected in the future.
        except Exception as error:
            raise RuntimeError(
                f"Comparing\n\n"
                f"{pair}\n\n"
                f"resulted in the unexpected exception above. "
                f"If you are a user and see this message during normal operation "
                "please file an issue at https://github.com/pytorch/pytorch/issues. "
                "If you are a developer and working on the comparison functions, "
                "please except the previous error and raise an expressive `ErrorMeta` instead."
            ) from error

    # [ErrorMeta Cycles]
    # ErrorMeta objects in this list capture
    # tracebacks that refer to the frame of this function.
    # The local variable `error_metas` refers to the error meta
    # objects, creating a reference cycle. Frames in the traceback
    # would not get freed until cycle collection, leaking cuda memory in tests.
    # We break the cycle by removing the reference to the error_meta objects
    # from this frame as it returns.
    # pyrefly: ignore [bad-assignment]
    error_metas = [error_metas]
    # pyrefly: ignore [bad-return]
    return error_metas.pop()


def assert_close(
    actual: Any,
    expected: Any,
    *,
    allow_subclasses: bool = True,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = False,
    check_device: bool = True,
    check_dtype: bool = True,
    check_layout: bool = True,
    check_stride: bool = False,
    msg: str | Callable[[str], str] | None = None,
):
    # [FUNCTION DOC] Public entry point: asserts that actual and expected are close. Unwraps
    # DTensors, runs the pair comparison with the none, bool, number and tensor-like pairs and the
    # given options, and raises the first collected error converted through the optional message
    # (string, or callable receiving the generated text). Returns nothing. Hides itself from
    # pytest tracebacks.
    # KNOWN ISSUE: the docstring types equal_nan as bool or str but only bool is used; it lists
    # ValueError for inputs that cannot be turned into tensors and AssertionError for unrelated
    # types, but the code raises TypeError when no pair accepts the inputs. Only the first error
    # is reported (see the TODO in the code).
    r"""Asserts that ``actual`` and ``expected`` are close.

    If ``actual`` and ``expected`` are strided, non-quantized, real-valued, and finite, they are considered close if

    .. math::

        \lvert \text{actual} - \text{expected} \rvert \le \texttt{atol} + \texttt{rtol} \cdot \lvert \text{expected} \rvert

    Non-finite values (``-inf`` and ``inf``) are only considered close if and only if they are equal. ``NaN``'s are
    only considered equal to each other if ``equal_nan`` is ``True``.

    In addition, they are only considered close if they have the same

    - :attr:`~torch.Tensor.device` (if ``check_device`` is ``True``),
    - ``dtype`` (if ``check_dtype`` is ``True``),
    - ``layout`` (if ``check_layout`` is ``True``), and
    - stride (if ``check_stride`` is ``True``).

    If either ``actual`` or ``expected`` is a meta tensor, only the attribute checks will be performed.

    If ``actual`` and ``expected`` are sparse (either having COO, CSR, CSC, BSR, or BSC layout), their strided members are
    checked individually. Indices, namely ``indices`` for COO, ``crow_indices`` and ``col_indices`` for CSR and BSR,
    or ``ccol_indices``  and ``row_indices`` for CSC and BSC layouts, respectively,
    are always checked for equality whereas the values are checked for closeness according to the definition above.

    If ``actual`` and ``expected`` are quantized, they are considered close if they have the same
    :meth:`~torch.Tensor.qscheme` and the result of :meth:`~torch.Tensor.dequantize` is close according to the
    definition above.

    ``actual`` and ``expected`` can be :class:`~torch.Tensor`'s or any tensor-or-scalar-likes from which
    :class:`torch.Tensor`'s can be constructed with :func:`torch.as_tensor`. Except for Python scalars the input types
    have to be directly related. In addition, ``actual`` and ``expected`` can be :class:`~collections.abc.Sequence`'s
    or :class:`~collections.abc.Mapping`'s in which case they are considered close if their structure matches and all
    their elements are considered close according to the above definition.

    .. note::

        Python scalars are an exception to the type relation requirement, because their :func:`type`, i.e.
        :class:`int`, :class:`float`, and :class:`complex`, is equivalent to the ``dtype`` of a tensor-like. Thus,
        Python scalars of different types can be checked, but require ``check_dtype=False``.

    Args:
        actual (Any): Actual input.
        expected (Any): Expected input.
        allow_subclasses (bool): If ``True`` (default) and except for Python scalars, inputs of directly related types
            are allowed. Otherwise type equality is required.
        rtol (Optional[float]): Relative tolerance. If specified ``atol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        atol (Optional[float]): Absolute tolerance. If specified ``rtol`` must also be specified. If omitted, default
            values based on the :attr:`~torch.Tensor.dtype` are selected with the below table.
        equal_nan (Union[bool, str]): If ``True``, two ``NaN`` values will be considered equal.
        check_device (bool): If ``True`` (default), asserts that corresponding tensors are on the same
            :attr:`~torch.Tensor.device`. If this check is disabled, tensors on different
            :attr:`~torch.Tensor.device`'s are moved to the CPU before being compared.
        check_dtype (bool): If ``True`` (default), asserts that corresponding tensors have the same ``dtype``. If this
            check is disabled, tensors with different ``dtype``'s are promoted  to a common ``dtype`` (according to
            :func:`torch.promote_types`) before being compared.
        check_layout (bool): If ``True`` (default), asserts that corresponding tensors have the same ``layout``. If this
            check is disabled, tensors with different ``layout``'s are converted to strided tensors before being
            compared.
        check_stride (bool): If ``True`` and corresponding tensors are strided, asserts that they have the same stride.
        msg (Optional[Union[str, Callable[[str], str]]]): Optional error message to use in case a failure occurs during
            the comparison. Can also passed as callable in which case it will be called with the generated message and
            should return the new message.

    Raises:
        ValueError: If no :class:`torch.Tensor` can be constructed from an input.
        ValueError: If only ``rtol`` or ``atol`` is specified.
        AssertionError: If corresponding inputs are not Python scalars and are not directly related.
        AssertionError: If ``allow_subclasses`` is ``False``, but corresponding inputs are not Python scalars and have
            different types.
        AssertionError: If the inputs are :class:`~collections.abc.Sequence`'s, but their length does not match.
        AssertionError: If the inputs are :class:`~collections.abc.Mapping`'s, but their set of keys do not match.
        AssertionError: If corresponding tensors do not have the same :attr:`~torch.Tensor.shape`.
        AssertionError: If ``check_layout`` is ``True``, but corresponding tensors do not have the same
            :attr:`~torch.Tensor.layout`.
        AssertionError: If only one of corresponding tensors is quantized.
        AssertionError: If corresponding tensors are quantized, but have different :meth:`~torch.Tensor.qscheme`'s.
        AssertionError: If ``check_device`` is ``True``, but corresponding tensors are not on the same
            :attr:`~torch.Tensor.device`.
        AssertionError: If ``check_dtype`` is ``True``, but corresponding tensors do not have the same ``dtype``.
        AssertionError: If ``check_stride`` is ``True``, but corresponding strided tensors do not have the same stride.
        AssertionError: If the values of corresponding tensors are not close according to the definition above.

    The following table displays the default ``rtol`` and ``atol`` for different ``dtype``'s. In case of mismatching
    ``dtype``'s, the maximum of both tolerances is used.

    +---------------------------+------------+----------+
    | ``dtype``                 | ``rtol``   | ``atol`` |
    +===========================+============+==========+
    | :attr:`~torch.float16`    | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.bfloat16`   | ``1.6e-2`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float32`    | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.float64`    | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex32`  | ``1e-3``   | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex64`  | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.complex128` | ``1e-7``   | ``1e-7`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint8`     | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint2x4`   | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.quint4x2`   | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.qint8`      | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | :attr:`~torch.qint32`     | ``1.3e-6`` | ``1e-5`` |
    +---------------------------+------------+----------+
    | other                     | ``0.0``    | ``0.0``  |
    +---------------------------+------------+----------+

    .. note::

        :func:`~torch.testing.assert_close` is highly configurable with strict default settings. Users are encouraged
        to :func:`~functools.partial` it to fit their use case. For example, if an equality check is needed, one might
        define an ``assert_equal`` that uses zero tolerances for every ``dtype`` by default:

        >>> import functools
        >>> assert_equal = functools.partial(torch.testing.assert_close, rtol=0, atol=0)
        >>> assert_equal(1e-9, 1e-10)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not equal!
        <BLANKLINE>
        Expected 1e-10 but got 1e-09.
        Absolute difference: 9.000000000000001e-10
        Relative difference: 9.0

    Examples:
        >>> # tensor to tensor comparison
        >>> expected = torch.tensor([1e0, 1e-1, 1e-2])
        >>> actual = torch.acos(torch.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # scalar to scalar comparison
        >>> import math
        >>> expected = math.sqrt(2.0)
        >>> actual = 2.0 / math.sqrt(2.0)
        >>> torch.testing.assert_close(actual, expected)

        >>> # numpy array to numpy array comparison
        >>> import numpy as np
        >>> expected = np.array([1e0, 1e-1, 1e-2])
        >>> actual = np.arccos(np.cos(expected))
        >>> torch.testing.assert_close(actual, expected)

        >>> # sequence to sequence comparison
        >>> import numpy as np
        >>> # The types of the sequences do not have to match. They only have to have the same
        >>> # length and their elements have to match.
        >>> expected = [torch.tensor([1.0]), 2.0, np.array(3.0)]
        >>> actual = tuple(expected)
        >>> torch.testing.assert_close(actual, expected)

        >>> # mapping to mapping comparison
        >>> from collections import OrderedDict
        >>> import numpy as np
        >>> foo = torch.tensor(1.0)
        >>> bar = 2.0
        >>> baz = np.array(3.0)
        >>> # The types and a possible ordering of mappings do not have to match. They only
        >>> # have to have the same set of keys and their elements have to match.
        >>> expected = OrderedDict([("foo", foo), ("bar", bar), ("baz", baz)])
        >>> actual = {"baz": baz, "bar": bar, "foo": foo}
        >>> torch.testing.assert_close(actual, expected)

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = expected.clone()
        >>> # By default, directly related instances can be compared
        >>> torch.testing.assert_close(torch.nn.Parameter(actual), expected)
        >>> # This check can be made more strict with allow_subclasses=False
        >>> torch.testing.assert_close(
        ...     torch.nn.Parameter(actual), expected, allow_subclasses=False
        ... )
        Traceback (most recent call last):
        ...
        TypeError: No comparison pair was able to handle inputs of type
        <class 'torch.nn.parameter.Parameter'> and <class 'torch.Tensor'>.
        >>> # If the inputs are not directly related, they are never considered close
        >>> torch.testing.assert_close(actual.numpy(), expected)
        Traceback (most recent call last):
        ...
        TypeError: No comparison pair was able to handle inputs of type <class 'numpy.ndarray'>
        and <class 'torch.Tensor'>.
        >>> # Exceptions to these rules are Python scalars. They can be checked regardless of
        >>> # their type if check_dtype=False.
        >>> torch.testing.assert_close(1.0, 1, check_dtype=False)

        >>> # NaN != NaN by default.
        >>> expected = torch.tensor(float("Nan"))
        >>> actual = expected.clone()
        >>> torch.testing.assert_close(actual, expected)
        Traceback (most recent call last):
        ...
        AssertionError: Scalars are not close!
        <BLANKLINE>
        Expected nan but got nan.
        Absolute difference: nan (up to 1e-05 allowed)
        Relative difference: nan (up to 1.3e-06 allowed)
        >>> torch.testing.assert_close(actual, expected, equal_nan=True)

        >>> expected = torch.tensor([1.0, 2.0, 3.0])
        >>> actual = torch.tensor([1.0, 4.0, 5.0])
        >>> # The default error message can be overwritten.
        >>> torch.testing.assert_close(
        ...     actual, expected, msg="Argh, the tensors are not close!"
        ... )
        Traceback (most recent call last):
        ...
        AssertionError: Argh, the tensors are not close!
        >>> # If msg is a callable, it can be used to augment the generated message with
        >>> # extra information
        >>> torch.testing.assert_close(
        ...     actual, expected, msg=lambda msg: f"Header\n\n{msg}\n\nFooter"
        ... )
        Traceback (most recent call last):
        ...
        AssertionError: Header
        <BLANKLINE>
        Tensor-likes are not close!
        <BLANKLINE>
        Mismatched elements: 2 / 3 (66.7%)
        Greatest absolute difference: 2.0 at index (1,) (up to 1e-05 allowed)
        Greatest relative difference: 1.0 at index (1,) (up to 1.3e-06 allowed)
        <BLANKLINE>
        Footer
    """
    # Hide this function from `pytest`'s traceback
    __tracebackhide__ = True

    actual, expected = _unwrap_dtensor_for_comparison(actual, expected)

    error_metas = not_close_error_metas(
        actual,
        expected,
        pair_types=(
            NonePair,
            BooleanPair,
            NumberPair,
            TensorLikePair,
        ),
        allow_subclasses=allow_subclasses,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=check_device,
        check_dtype=check_dtype,
        check_layout=check_layout,
        check_stride=check_stride,
        msg=msg,
    )

    if error_metas:
        # TODO: compose all metas into one AssertionError
        raise error_metas[0].to_error(msg)


@deprecated(
    "`torch.testing.assert_allclose()` is deprecated since 1.12 and will be removed in a future release. "
    "Please use `torch.testing.assert_close()` instead. "
    "You can find detailed upgrade instructions in https://github.com/pytorch/pytorch/issues/61844.",
    category=FutureWarning,
)
def assert_allclose(
    actual: Any,
    expected: Any,
    rtol: float | None = None,
    atol: float | None = None,
    equal_nan: bool = True,
    msg: str = "",
) -> None:
    # [FUNCTION DOC] Deprecated wrapper (emits a FutureWarning). Converts inputs to tensors
    # (expected takes the dtype of actual), chooses legacy default tolerances (float16, float32
    # and float64 table) when neither tolerance is given, and calls assert_close with check_dtype
    # off, equal_nan default True and the message if non-empty. Raises what assert_close raises.
    # KNOWN ISSUE: if only one of rtol and atol is given the default tolerances are not applied
    # and assert_close raises ValueError.
    """
    .. warning::

       :func:`torch.testing.assert_allclose` is deprecated since ``1.12`` and will be removed in a future release.
       Please use :func:`torch.testing.assert_close` instead. You can find detailed upgrade instructions
       `here <https://github.com/pytorch/pytorch/issues/61844>`_.
    """
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual)
    if not isinstance(expected, torch.Tensor):
        expected = torch.tensor(expected, dtype=actual.dtype)

    if rtol is None and atol is None:
        rtol, atol = default_tolerances(
            actual,
            expected,
            dtype_precisions={
                torch.float16: (1e-3, 1e-3),
                torch.float32: (1e-4, 1e-5),
                torch.float64: (1e-5, 1e-8),
            },
        )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
        check_device=True,
        check_dtype=False,
        check_stride=False,
        msg=msg or None,
    )
