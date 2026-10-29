# [PACKAGE DOC] functorch.dim: first-class dimensions. Lets users create named dimension objects
# (Dim, DimList via dims() and dimlists()), index tensors with them, and have operations
# automatically batch over the bound dimensions using functorch vmap layers. Contents: Dim and
# DimList, tensor wrappers (_Tensor, Tensor) with a torch-function hook, dimension-aware stack,
# split, cat, dot and index helpers, and error types DimensionBindError and
# DimensionMismatchError. Import-time side effects: registers wrapped torch functions and Tensor
# methods onto the classes at the end of the module (see _def and the code below it); module
# globals POINTWISE_OPTIMIZE and DOT_OPTIMIZED toggle fast paths.
from __future__ import annotations

import dis
import inspect
import sys
from typing import Any, Optional, TYPE_CHECKING, Union


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

import torch
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from ._dim_entry import _match_levels, DimEntry, ndim_of_levels
from ._enable_all_layers import EnableAllLayers
from ._py_inst_decoder import _PyInstDecoder
from ._tensor_info import TensorInfo


POINTWISE_OPTIMIZE = True
DOT_OPTIMIZED = True

# Global dimension level counter
_n_dims_created = 0


def _relevant_op(opcode: str | None) -> bool:
    # [FUNCTION DOC] HELPER: True if opcode is a non-empty name starting with STORE_ (a variable
    # store instruction).
    """Check if opcode is relevant for variable assignment."""
    return bool(opcode and opcode.startswith("STORE_"))


def handle_from_tensor(tensor: torch.Tensor) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Identity function; returns the tensor unchanged (a hook point).
    """Handle tensor conversion for torch function integration."""
    return tensor


def _create_dim(name: str, size: int | None = None) -> Dim:
    # [FUNCTION DOC] HELPER: Build a Dim with the given name; a size of None becomes -1 (unbound).
    """Create a new Dim object."""
    return Dim(name, size if size is not None else -1)


def dims(
    n: int | None = None, sizes: list[int | None] | None = None
) -> Dim | tuple[Dim, ...]:
    # [FUNCTION DOC] Create n new Dim objects (n from argument n, else the length of sizes, else
    # inferred by inspecting the caller's bytecode for the assignment target count). Names are
    # taken from the assigned variable names when the call is directly assigned; otherwise they
    # are d0, d1, and so on. sizes gives per-dim sizes (None leaves unbound). Returns one Dim if
    # exactly one is requested, else a tuple. Raises SyntaxError if the count cannot be
    # determined, ValueError if sizes length mismatches, RuntimeError if no caller frame.
    # Reads the caller's frame; no other side effects.
    """
    Create and return one or more Dim objects.

    Uses bytecode inspection to determine variable names when possible.

    Args:
        n (int, optional): The number of dimensions to create. Can be omitted if sizes is specified.
        sizes (List[Optional[int]], optional): A list the same size as the number of dimensions to be
          created, specifying each dimensions size, or None to leave the size unset.

    Returns:
        Union[Dim, Tuple[Dim, ...]]: Single Dim if n=1, tuple of Dims otherwise.

    Examples:
        >>> batch, channel, width, height = dims(4)
        >>> batch, channel, width, height = dims(sizes=[None, 3, 224, 224])
        >>> single_dim = dims(1)
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    # Use bytecode inspection
    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dims() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        def genobject(i: int) -> Dim:
            # [FUNCTION DOC] HELPER: Create the i-th Dim, naming it from the decoder's current
            # store target when names were found (and advancing the decoder), else d followed
            # by i (and disabling further name lookup). Uses sizes[i] if sizes was given.
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            size = sizes[i] if sizes is not None else None
            return _create_dim(name, size)

        # Validate sizes parameter
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        if specified_ndims == 1:
            return genobject(0)

        result = []
        for i in range(specified_ndims):
            result.append(genobject(i))

        return tuple(result)

    finally:
        del frame


class DimList:
    # [CLASS DOC] A sequence of first-class Dims whose length may be unknown. State: _name (used
    # to name generated Dims), _dims (list of Dim), _bound (whether the length is fixed). Once
    # bound the length cannot change (rebinding to a different length raises
    # DimensionBindError). Used in indexing to stand for several dimensions. Key methods:
    # bind_len, bind, size, indexing by int or slice.
    """
    A list of first-class dimensions that can be bound to tensor dimensions.

    A DimList can be in one of two states:
    1. Unbound: Created with just a name, no specific dimensions yet
    2. Bound: Either created with specific dimensions/sizes, or bound later via bind() or bind_len()
    """

    _name: str | None
    _dims: list[Dim]
    _bound: bool

    def __init__(
        self,
        len_or_dims: int | Sequence | None = None,
        name: str | None = None,
    ):
        # [FUNCTION DOC] Create a DimList. An int len_or_dims binds it to that many fresh Dims;
        # a sequence creates one Dim per item (an int item is a size, with a generated name;
        # any other item is passed as the Dim name) and marks it bound; None leaves it unbound.
        """
        Initialize a new DimList object.

        Args:
            len_or_dims: Optional length (int) or sequence of dimensions/sizes
            name: Optional name for the dimension list
        """
        # Initialize attributes
        self._name = name
        self._dims: list = []
        self._bound = False

        if isinstance(len_or_dims, int):
            self.bind_len(len_or_dims)
        elif len_or_dims is not None:
            dims = []
            for i, item in enumerate(len_or_dims):
                if isinstance(item, int):
                    dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                    dims.append(Dim(dim_name, item))
                else:
                    dims.append(Dim(item))
            self._set_dims(dims)

    def _set_dims(self, dims: list) -> None:
        # [FUNCTION DOC] HELPER: Store the given Dim list and mark this DimList bound.
        """Set the dimensions and mark as bound."""
        self._bound = True
        self._dims = dims

    def bind_len(self, size: int) -> None:
        # [FUNCTION DOC] Fix the length. If unbound, mark bound and create size fresh unsized
        # Dims named from _name plus index. If already bound, only check the length matches and
        # raise DimensionBindError otherwise.
        """
        Bind this DimList to a specific length.

        Args:
            size: Number of dimensions to bind to

        Raises:
            DimensionBindError: If already bound to a different size
        """
        if self._bound:
            if len(self._dims) != size:
                raise DimensionBindError(
                    f"Dimlist has size {len(self._dims)} but it is being bound to size {size}"
                )
        else:
            self._bound = True
            self._dims = []
            for i in range(size):
                dim_name = f"{self._name}{i}" if self._name else f"dim{i}"
                self._dims.append(Dim(dim_name))

    def bind(self, sizes: Sequence[int]) -> None:
        # [FUNCTION DOC] Bind the length to len(sizes) via bind_len, then set each Dim's size to
        # the corresponding value (converted to int; may raise from the Dim size setter if a
        # different size was already set). Raises ValueError if sizes is not a sized sequence.
        """
        Bind this DimList to specific sizes.

        Args:
            sizes: Sequence of sizes for each dimension

        Raises:
            ValueError: If sizes is not a sequence
        """
        if not hasattr(sizes, "__len__") or not hasattr(sizes, "__getitem__"):
            raise ValueError("expected a sequence")

        size = len(sizes)
        self.bind_len(size)

        for i, dim_size in enumerate(sizes):
            self._dims[i].size = int(dim_size)

    def _size(self) -> int:
        # [FUNCTION DOC] HELPER: Number of Dims; raises DimensionBindError if unbound.
        if not self._bound:
            raise DimensionBindError("DimList not bound")
        return len(self._dims)

    def size(self) -> int:
        # [FUNCTION DOC] Number of Dims in the list; raises DimensionBindError if unbound.
        """Return the size (number of dimensions) of this DimList."""
        return self._size()

    def _set_bound(self, b: bool) -> None:
        # [FUNCTION DOC] HELPER: Directly set the bound flag without touching the Dims.
        """Set the bound status (for internal use)."""
        self._bound = b

    @property
    def is_bound(self) -> bool:
        # [FUNCTION DOC] Property: True once the length of the list is fixed.
        """Property to check if DimList is bound."""
        return self._bound

    def __len__(self) -> int:
        # [FUNCTION DOC] Length of the list; raises DimensionBindError if unbound.
        """Return the length of the DimList."""
        return self.size()

    def __getitem__(self, key: int | slice) -> Dim | tuple[Dim, ...]:
        # [FUNCTION DOC] Return the Dim at an int index, or a tuple of Dims for a slice. Raises
        # DimensionBindError if unbound, IndexError for an out-of-range or negative int index
        # (negative indices are not supported, unlike ordinary lists), ValueError for other key
        # types.
        if not self._bound:
            raise DimensionBindError("DimList not bound")

        if isinstance(key, int):
            if key < 0 or key >= len(self._dims):
                raise IndexError("index out of bounds")
            return self._dims[key]
        elif isinstance(key, slice):
            start, stop, step = key.indices(len(self._dims))
            result = []
            for i in range(start, stop, step):
                result.append(self._dims[i])
            return tuple(result)
        else:
            raise ValueError("expected an int or a slice")

    def __repr__(self) -> str:
        # [FUNCTION DOC] Bound: parenthesized comma-separated reprs of the Dims; unbound with a
        # name: a star followed by the name; otherwise the text <unbound_dimlist>.
        """Return string representation of the DimList."""
        if self._bound:
            # Show as tuple representation
            return f"({', '.join(repr(dim) for dim in self._dims)})"
        elif self._name is not None:
            # Show as *name for unbound with name
            return f"*{self._name}"
        else:
            # Show as <unbound_dimlist> for unbound without name
            return "<unbound_dimlist>"

    def __str__(self) -> str:
        # [FUNCTION DOC] Same text as __repr__.
        """Return string representation of the DimList."""
        return self.__repr__()

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # [FUNCTION DOC] Torch-function hook for DimList: forwards the call unchanged to the
        # _Tensor hook and returns its result.
        return _Tensor.__torch_function__(func, types, args, kwargs)


def _create_dimlist(name: str, size: int | list[int | None] | None = None) -> DimList:
    # [FUNCTION DOC] HELPER: Make a named DimList. An int size binds it to that length; a list
    # binds it to the list length and sets each non-None entry as that Dim's size; None leaves
    # it unbound.
    """Create a DimList object with the given name and optional size."""
    dimlist = DimList(name=name)
    if size is not None:
        if isinstance(size, int):
            dimlist.bind_len(size)
        else:
            # size is a list of optional ints
            dimlist.bind_len(len(size))
            for i, s in enumerate(size):
                if s is not None:
                    dimlist._dims[i].size = s
    return dimlist


def dimlists(
    n: int | None = None, sizes: list[int | None] | None = None
) -> DimList | tuple[DimList, ...]:
    # [FUNCTION DOC] Same protocol as dims() but creates DimList objects: count comes from n,
    # sizes length, or the caller's assignment target; names come from the assigned variable
    # names, else d0, d1, and so on. A single result is returned bare, several as a tuple.
    # Raises SyntaxError, ValueError or RuntimeError in the same situations as dims().
    """
    Create and return one or more DimList objects.

    Similar to dims() but creates DimList objects instead.
    """
    specified_ndims = -1
    found_ndims = 0

    # Parse arguments
    if sizes is not None:
        specified_ndims = len(sizes)
    if n is not None:
        specified_ndims = n

    frame = inspect.currentframe()
    if frame is None:
        raise RuntimeError("Unable to get current frame")
    frame = frame.f_back
    try:
        if frame is None:
            raise RuntimeError("Unable to get caller frame")
        code = frame.f_code
        lasti = frame.f_lasti

        decoder = _PyInstDecoder(code, lasti)

        if sys.version_info >= (3, 11):
            if decoder.opcode() == "PRECALL":
                decoder.next()

        # Move to next instruction after the call
        decoder.next()

        # Determine number of dimensions from bytecode
        if _relevant_op(decoder.opcode()):
            found_ndims = 1
        elif decoder.opcode() == "UNPACK_SEQUENCE":
            found_ndims = decoder.oparg()
            decoder.next()  # Move past UNPACK_SEQUENCE

        if specified_ndims == -1:
            if found_ndims == 0:
                raise SyntaxError(
                    "dimlists() must be assigned to a sequence of variable names or have argument n specified"
                )
            specified_ndims = found_ndims

        if found_ndims != specified_ndims:
            found_ndims = 0

        # Generator function for dimlist names
        def genobject(i: int) -> str:
            # [FUNCTION DOC] HELPER: Return the name for the i-th DimList: the decoder's current
            # store target (advancing it) when names were found, else d followed by i (which
            # also disables further name lookup).
            nonlocal found_ndims
            name = None
            if i < found_ndims:
                name = decoder.name()

            if not name:
                name = f"d{i}"
                found_ndims = 0
            else:
                decoder.next()  # Move to next STORE instruction

            return name

        # Validate sizes
        if sizes is not None and len(sizes) != specified_ndims:
            raise ValueError(f"expected {specified_ndims} sizes but found {len(sizes)}")

        # Create dimlists
        if specified_ndims == 1:
            name = genobject(0)
            return _create_dimlist(name, sizes[0] if sizes is not None else None)

        result = []
        for i in range(specified_ndims):
            name = genobject(i)
            size = sizes[i] if sizes is not None else None
            result.append(_create_dimlist(name, size))

        return tuple(result)

    finally:
        del frame


class DimensionMismatchError(Exception):
    # [CLASS DOC] Error raised when dimensions of operands do not match as required.
    pass


class DimensionBindError(Exception):
    # [CLASS DOC] Error raised when a Dim or DimList cannot be bound to a size or length (for
    # example conflicting sizes, or an unbound DimList used where a length is needed).
    pass


from . import op_properties


def _safe_print(*args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] HELPER: Print args to standard error, replacing any argument whose type name
    # mentions torch with a placeholder of its class name, to avoid triggering torch-function
    # dispatch while printing. Extra keyword arguments go to print.
    # KNOWN ISSUE: passing a file keyword argument would conflict with the fixed file argument.
    """Safe print that avoids recursive torch function dispatches."""
    import sys

    # Convert any torch objects to basic representations
    safe_args = []
    for arg in args:
        if hasattr(arg, "__class__") and "torch" in str(type(arg)):
            safe_args.append(f"<{type(arg).__name__}>")
        else:
            safe_args.append(str(arg))

    print(*safe_args, **kwargs, file=sys.stderr)


class _Tensor:
    # [CLASS DOC] Base class of first-class-dimension values (Dim and Tensor). Subclasses supply
    # levels (_get_levels), the underlying tensor (_get_tensor) and ndim. This class provides
    # the torch-function hook that intercepts torch operations: it special-cases multiply of
    # scalars, item get/set, len, softmax, stack and split, and otherwise runs the operation
    # either pointwise (aligning operands by level) or under functorch vmap layers, then wraps
    # results as Tensor objects. Also defines expand, index, order-related helpers, and repr.
    def _get_levels(self) -> list[Any]:
        # [FUNCTION DOC] Abstract: subclasses return their list of levels; raises
        # NotImplementedError here.
        raise NotImplementedError("_get_levels must be implemented by subclass")

    def _get_tensor(self) -> torch.Tensor | None:
        # [FUNCTION DOC] Abstract: subclasses return the underlying tensor; raises
        # NotImplementedError here.
        raise NotImplementedError("_get_tensor must be implemented by subclass")

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Abstract property: number of positional dimensions; raises
        # NotImplementedError here.
        raise NotImplementedError("ndim must be implemented by subclass")

    @property
    def dims(self) -> tuple[Any, ...]:
        # [FUNCTION DOC] Property: tuple of the first-class Dim objects among the levels, in
        # level order (positional levels excluded).
        return tuple(l.dim() for l in self._get_levels() if not l.is_positional())

    def dim(self) -> int:
        # [FUNCTION DOC] Number of positional dimensions (same as ndim), mirroring the tensor
        # method of the same name.
        return self.ndim

    @classmethod
    def __torch_function__(
        cls,
        func: Callable,
        types: tuple,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # [FUNCTION DOC] Intercept a torch function call involving first-class dims. Special
        # cases: multiply of two zero-dim floating tensors becomes a delayed Tensor (when
        # DOT_OPTIMIZED); item get and set are routed to the indexing module (set returns None,
        # raises ValueError unless given exactly three args); length returns the first size;
        # softmax, stack and split use the dim-aware implementations. Everything else goes to
        # _torch_function_fallback. Returns the result of the chosen path.
        if kwargs is None:
            kwargs = {}

        if DOT_OPTIMIZED and func is torch.Tensor.__mul__:
            # Check conditions: 2 args, both are tensor-like, both 0-dimensional
            if (
                len(args) == 2
                and not kwargs
                and isinstance(args[0], (_Tensor, torch.Tensor))
                and isinstance(args[1], (_Tensor, torch.Tensor))
            ):
                # Get tensor info for both operands
                lhs_info = TensorInfo.create(
                    args[0], ensure_batched=False, ensure_present=False
                )
                rhs_info = TensorInfo.create(
                    args[1], ensure_batched=False, ensure_present=False
                )

                if (
                    lhs_info
                    and rhs_info
                    and lhs_info.tensor is not None
                    and rhs_info.tensor is not None
                    and lhs_info.tensor.dim() == 0
                    and rhs_info.tensor.dim() == 0
                ):
                    if (
                        lhs_info.tensor.is_floating_point()
                        and rhs_info.tensor.is_floating_point()
                    ):
                        # Collect all unique levels and has_device
                        has_device = lhs_info.has_device or rhs_info.has_device
                        levels = []

                        for level in lhs_info.levels:
                            if level not in levels:
                                levels.append(level)
                        for level in rhs_info.levels:
                            if level not in levels:
                                levels.append(level)

                        # Debug print
                        # print(f"DEBUG: Creating delayed mul, levels: {levels}, has_device: {has_device}")

                        # Create delayed tensor
                        return Tensor.create_delayed(func, args, levels, has_device)

        if func is torch.Tensor.__getitem__:
            from functorch.dim._getsetitem import getitem

            return getitem(cls, func, types, args, kwargs)

        if func is torch.Tensor.__setitem__:
            from functorch.dim._getsetitem import setitem

            # args should be (tensor, index, value)
            if len(args) == 3:
                setitem(args[0], args[1], args[2])
                return None
            else:
                raise ValueError(f"Expected 3 args for __setitem__, got {len(args)}")

        # Fast-path for len; mostly to avoid infinite loop in TestMinFunctorchOnly.test_softmax_split
        if func is torch.Tensor.__len__:
            return args[0].size(0)

        # Special handling for torch.softmax - use the pre-wrapped version
        if func is torch.softmax:
            return softmax(*args, **kwargs)

        # Special handling for torch.stack - use the custom stack function
        if func is torch.stack:
            return stack(*args, **kwargs)

        if (
            func is torch.Tensor.split
            or func is torch._VF.split  # type: ignore[attr-defined]
            or func is torch._VF.split_with_sizes  # type: ignore[attr-defined]
            or func is torch.split
        ):
            return split(*args, **kwargs)

        return _Tensor._torch_function_fallback(func, types, args, kwargs)

    @staticmethod
    def _torch_function_fallback(
        func: Callable, types: tuple, args: tuple, kwargs: dict
    ) -> Any:
        # [FUNCTION DOC] HELPER: Run func on arguments that may carry first-class dims. Flattens
        # args/kwargs, gathers each tensor-like argument's levels and a device-holding tensor.
        # For pointwise ops (when POINTWISE_OPTIMIZE) aligns all operands to the union of levels
        # and calls func directly, wrapping tensor results with those levels. Otherwise enters
        # EnableAllLayers, passes batched tensors with updated levels, and converts results
        # back from batched form. Returns results in the same structure as func's output.
        """Fallback torch function implementation for non-special-cased functions."""
        is_pointwise = POINTWISE_OPTIMIZE and func in op_properties.pointwise
        # TODO: optimize pytree here
        flat_args, spec = tree_flatten((args, kwargs))
        device_holding_tensor = None

        infos: list[TensorInfo] = []
        result_levels: list[DimEntry] = []

        for f in flat_args:
            info = TensorInfo.create(f, not is_pointwise, False)
            infos.append(info)
            if info:
                if not (is_pointwise or info.batchedtensor is not None):
                    raise AssertionError(
                        "Expected pointwise or batchedtensor to be set"
                    )
                if device_holding_tensor is None and info.has_device:
                    device_holding_tensor = info.tensor
                # Collect all unique levels
                for level in info.levels:
                    if not isinstance(level, DimEntry):
                        raise AssertionError(f"Expected DimEntry, got {type(level)}")
                    if level not in result_levels:
                        result_levels.append(level)

        if is_pointwise:
            # Pointwise operation: match all tensors to common levels
            for i, info in enumerate(infos):
                if info and info.tensor is not None:
                    tensor = info.tensor
                    if device_holding_tensor is not None and not info.has_device:
                        tensor = tensor.to(device_holding_tensor.device)
                    ml = _match_levels(tensor, info.levels, result_levels)
                    flat_args[i] = handle_from_tensor(ml)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Wrap tensor results
            def wrap_tensor(obj: Any) -> Any:
                # [FUNCTION DOC] HELPER: Wrap a plain tensor as a first-class-dim Tensor with the
                # common result levels; other objects are returned unchanged.
                if isinstance(obj, torch.Tensor):
                    return Tensor.from_positional(
                        obj, result_levels, device_holding_tensor is not None
                    )
                return obj

            # Small fastpath
            if isinstance(result, torch.Tensor):
                return wrap_tensor(result)
            else:
                return tree_map(wrap_tensor, result)

        # Non-pointwise operation: use functorch vmap layers
        with EnableAllLayers(result_levels) as guard:
            # Update arguments with batched tensors
            for i, info in enumerate(infos):
                if info and info.batchedtensor is not None:
                    batched = info.batchedtensor
                    if device_holding_tensor is not None and not info.has_device:
                        batched = batched.to(device_holding_tensor.device)
                    guard.inplace_update_layers(batched, info.levels)
                    flat_args[i] = handle_from_tensor(batched)

            unflat_args, unflat_kwargs = tree_unflatten(flat_args, spec)
            result = func(*unflat_args, **unflat_kwargs)

            # Unwrap results from functorch layers
            def unwrap_tensor(obj: Any) -> Any:
                # [FUNCTION DOC] HELPER: Convert a batched tensor result back to a Tensor via the
                # layer guard; other objects are returned unchanged.
                if isinstance(obj, torch.Tensor):
                    return guard.from_batched(obj, device_holding_tensor is not None)
                return obj

            if isinstance(result, torch.Tensor):
                return unwrap_tensor(result)
            else:
                return tree_map(unwrap_tensor, result)

    def __setitem__(self, index: Any, value: Any) -> None:
        # [FUNCTION DOC] Assign value into self at index using the first-class-dim setitem logic
        # (in-place mutation of the underlying tensor). Raises DimensionBindError if value
        # carries a Dim missing from the target.
        """Set values in tensor using first-class dimensions."""
        from functorch.dim._getsetitem import setitem

        return setitem(self, index, value)

    # expand and index are OK to be methods because they don't have torch.*
    # versions, but if they did they need the stack/cat treatment

    def expand(self, *args: Dim) -> _Tensor:
        # [FUNCTION DOC] If every argument is a Dim, return a new Tensor with those Dims prepended
        # as zero-stride levels sized by each Dim (a view, no copy). Raises DimensionBindError if
        # a Dim is already present or repeated, or ValueError if an unbound Dim size is read.
        # If any argument is not a Dim, falls back to ordinary tensor expand via the torch
        # function path, as it also does when no tensor info is available.
        """
        Expand tensor by adding new dimensions or expanding existing dimensions.

        If all arguments are Dim objects, adds new named dimensions.
        Otherwise, falls back to regular tensor expansion behavior.

        Args:
            args: Either Dim objects for new dimensions or sizes for regular expansion

        Returns:
            New tensor with expanded dimensions

        Example:
            >>> i, j = dims()
            >>> t = torch.randn(3, 4)
            >>> expanded = t[i].expand(j, k)  # Add j, k dimensions
            >>> expanded2 = t[i].expand(2, 4)  # Regular expand with sizes
        """
        info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

        for arg in args:
            if not isinstance(arg, Dim):
                # Not all args are Dims, fallback to regular expand
                if isinstance(self, torch.Tensor) and not isinstance(self, _Tensor):
                    return torch.Tensor.expand(self, *args)
                else:
                    return self.__torch_function__(
                        torch.Tensor.expand, (type(self),), (self,) + args
                    )

        # All args are Dim objects - proceed with first-class dimension expansion
        if not info:
            # No tensor info available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        # First-class dimension expansion - all args are Dim objects
        data = info.tensor
        if data is None:
            # No tensor data available, fallback
            return self.__torch_function__(
                torch.Tensor.expand, (type(self),), (self,) + args
            )

        levels = info.levels

        new_levels: list[DimEntry] = []
        new_sizes = []
        new_strides = []

        for d in args:
            # Check if dimension already exists in current levels or new_levels
            for level in levels:
                if not level.is_positional() and level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )
            for new_level in new_levels:
                if not new_level.is_positional() and new_level.dim() is d:
                    raise DimensionBindError(
                        f"expanding dimension {d} already exists in tensor with dims"
                    )

            new_levels.append(DimEntry(d))
            new_sizes.append(d.size)
            new_strides.append(0)

        # Add existing levels
        new_levels.extend(levels)

        # Add existing sizes and strides
        orig_sizes = list(data.size())
        orig_strides = list(data.stride())
        new_sizes.extend(orig_sizes)
        new_strides.extend(orig_strides)

        # Create expanded tensor using as_strided
        expanded_data = data.as_strided(new_sizes, new_strides, data.storage_offset())

        # Return new tensor with expanded dimensions
        result = Tensor.from_positional(expanded_data, new_levels, info.has_device)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable

    def index(
        self,
        dims: int | Dim | tuple[int | Dim, ...] | list[int | Dim],
        indices: int
        | slice
        | torch.Tensor
        | tuple[int | slice | torch.Tensor, ...]
        | list[int | slice | torch.Tensor],
    ) -> _Tensor:
        # [FUNCTION DOC] Index the given dims of self with the matching indices. dims and indices
        # may each be a single item or equal-length sequences. A dim may be a dim pack (sequence):
        # the packed dims are flattened into the first one before indexing. Returns the indexed
        # result as a Tensor. Raises TypeError for a bad dim spec, a dim missing from the tensor,
        # or mismatched lengths between dims and indices.
        """
        Index tensor using first-class dimensions.
        """
        from ._dim_entry import _match_levels
        from ._getsetitem import getsetitem_flat, invoke_getitem
        from ._wrap import _wrap_dim

        # Helper to check if obj is a dimpack (tuple/list) and extract items
        def maybe_dimpack(obj: Any, check_first: bool = False) -> tuple[Any, bool]:
            # [FUNCTION DOC] HELPER: If obj is a tuple or list, return (list of its items, True);
            # else (None, False). The check_first argument is unused.
            if isinstance(obj, (tuple, list)):
                return list(obj), True
            return None, False

        def parse_dim_entry(s: Any) -> Any:
            # [FUNCTION DOC] HELPER: Convert a dim specification s (Dim or int position) into a
            # DimEntry relative to self's ndim; raises TypeError if it is not a valid dimension.
            d = _wrap_dim(s, self.ndim, False)
            if d.is_none():
                raise TypeError(f"expected a dimension specifyer but found {repr(s)}")
            return d

        # Helper for dimension not present errors
        def dim_not_present(d: Any) -> None:
            # [FUNCTION DOC] HELPER: Always raises TypeError describing the missing dimension d
            # (position out of range or Dim not in the tensor); never returns normally.
            if d.is_positional():
                raise TypeError(
                    f"dimension {d.position() + self.ndim} not in tensor of {self.ndim} dimensions"
                )
            else:
                raise TypeError(f"dimension {repr(d.dim())} not in tensor")

        dims_list: list[int | Dim] = []
        indices_list: list[int | slice | torch.Tensor] = []

        lhs_list = isinstance(dims, (tuple, list))
        rhs_list = isinstance(indices, (tuple, list))

        if lhs_list and rhs_list:
            # Type narrowing: we know dims and indices are sequences here
            dims_seq = dims  # type: ignore[assignment]
            indices_seq = indices  # type: ignore[assignment]
            if len(dims_seq) != len(indices_seq):  # type: ignore[arg-type]
                raise TypeError(
                    f"dims ({len(dims_seq)}) and indices ({len(indices_seq)}) must have the same length"  # type: ignore[arg-type]
                )
            dims_list.extend(dims_seq)  # type: ignore[arg-type]
            indices_list.extend(indices_seq)  # type: ignore[arg-type]
        else:
            dims_list.append(dims)  # type: ignore[arg-type]
            indices_list.append(indices)  # type: ignore[arg-type]

        # Create tensor info
        self_info = TensorInfo.create(self, False, False)

        new_levels: list[Any] = []
        to_flatten: list[Any] = []
        dims_list_flat = []

        # Process each dim specification
        for i in range(len(dims_list)):
            m, is_dimpack = maybe_dimpack(dims_list[i], check_first=False)
            if is_dimpack:
                if len(m) == 0:
                    dims_list_flat.append(DimEntry())  # Empty dimpack
                    continue

                first = parse_dim_entry(m[0])
                dims_list_flat.append(first)

                if len(m) == 1:
                    continue

                # Multi-element dimpack requires flattening
                if len(to_flatten) == 0:
                    new_levels.extend(self_info.levels)

                rest = []
                for j in range(1, len(m)):
                    d = parse_dim_entry(m[j])
                    removed = False
                    for k in range(len(new_levels)):
                        if new_levels[k] == d:
                            new_levels.pop(k)
                            removed = True
                            break
                    if not removed:
                        dim_not_present(d)
                    rest.append(d)

                # Find first in new_levels
                first_idx = None
                for k in range(len(new_levels)):
                    if new_levels[k] == first:
                        first_idx = k
                        break
                if first_idx is None:
                    dim_not_present(first)
                    continue  # Skip this iteration if dimension not found

                for j, r in enumerate(rest):
                    new_levels.insert(first_idx + 1 + j, r)
                to_flatten.extend(rest)
            else:
                dims_list_flat.append(parse_dim_entry(dims_list[i]))

        # Handle dimension flattening if needed
        if len(to_flatten) > 0:
            if self_info.tensor is None:
                raise AssertionError(
                    "Cannot perform dimension flattening on None tensor"
                )
            rearranged = _match_levels(self_info.tensor, self_info.levels, new_levels)
            sizes = rearranged.size()
            new_sizes: list[Any] = []
            reshape_levels = []

            for i in range(len(new_levels)):
                if new_levels[i] in to_flatten:
                    if len(new_sizes) == 0:
                        new_sizes.append(sizes[i])
                    else:
                        new_sizes[-1] *= sizes[i]
                else:
                    new_sizes.append(sizes[i])
                    reshape_levels.append(new_levels[i])

            self_info.tensor = rearranged.reshape(new_sizes)
            self_info.levels = reshape_levels

        # Check for dimpacks in indices
        has_dimpacks = False
        for idx in indices_list:
            if isinstance(idx, (tuple, list)):
                has_dimpacks = True
                break

        # Call getsetitem_flat with correct parameters
        info = getsetitem_flat(
            self_info,
            [],  # empty input_list
            dims_list_flat,  # keys
            indices_list,  # values
            has_dimpacks,
        )

        return invoke_getitem(info)

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form: the underlying tensor, then a line listing dims (positional
        # levels shown as non-negative positions, first-class levels as their Dims) and the
        # tensor sizes.
        tensor, levels, ndim = self._get_tensor(), self._get_levels(), self.ndim
        dims_repr = []
        for l in levels:
            if hasattr(l, "is_positional") and l.is_positional():
                # Convert negative positional to positive: -1 -> ndim-1, -2 -> ndim-2, etc.
                dims_repr.append(l.position() + ndim)
            elif hasattr(l, "dim"):
                dims_repr.append(l.dim())
            elif hasattr(l, "data"):
                dims_repr.append(l.data)
            else:
                dims_repr.append(l)
        return f"{tensor}\nwith dims={tuple(dims_repr)} sizes={tuple(tensor.size())}"  # type: ignore[union-attr]


TensorLike = (_Tensor, torch.Tensor)


class Dim(_Tensor):
    # [CLASS DOC] A first-class dimension: a named, optionally sized axis usable as an index and
    # as an implicit vmap batch dimension. State: _name, _size (-1 means unbound), _level (unique
    # creation-order id from the module counter, used to order layers), and lazily cached _range
    # and _batchtensor. The size may be set once; setting a different size raises
    # DimensionBindError. A Dim behaves as a one-dimensional tensor of its range.
    _level: int
    _name: str
    _size: int
    _range: torch.Tensor | None
    _batchtensor: torch.Tensor | None

    def __init__(self, name: str, s: int = -1) -> None:
        # [FUNCTION DOC] Create a Dim with the given name and size (-1 = unbound). Assigns the
        # next value of the global creation counter as its level and increments that counter
        # (global side effect).
        global _n_dims_created
        self._name = name
        self._size = s
        self._level = _n_dims_created
        _n_dims_created += 1
        self._range = None
        self._batchtensor = None

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Property: always 1.
        return 1

    @classmethod
    def check_exact(cls, obj: Any) -> bool:
        # [FUNCTION DOC] True only if obj's type is exactly this class (subclasses excluded).
        return type(obj) is cls

    @property
    def size(self) -> int:
        # [FUNCTION DOC] Property: the bound size; raises ValueError if the Dim is unbound.
        if self._size == -1:
            raise ValueError(f"dimension {self._name} is unbound")
        return self._size

    @size.setter
    def size(self, v: int) -> None:
        # [FUNCTION DOC] Setter: bind the size to v if unbound; if already bound to a different
        # value raises DimensionBindError; the same value is accepted (no-op).
        if self._size == -1:
            self._size = v
        elif self._size != v:
            raise DimensionBindError(
                f"Dim '{repr(self)}' previously bound to a dimension of size {self._size} "
                f"cannot bind to a dimension of size {v}"
            )

    @property
    def is_bound(self) -> bool:
        # [FUNCTION DOC] Property: True once a size has been set.
        """Return True if this dimension is bound to a size."""
        return self._size != -1

    def _get_range(self) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return (creating and caching on first use) the 1-D integer range
        # tensor of this Dim's size. Raises ValueError if the Dim is unbound.
        """
        Get a tensor representing the range [0, size) for this dimension.

        Returns:
            A 1D tensor with values [0, 1, 2, ..., size-1]
        """
        if self._range is None:
            self._range = torch.arange(self.size)
        return self._range

    def _get_batchtensor(self) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Return (creating and caching) the range tensor wrapped as a
        # functorch batched tensor at batch dimension 0 and this Dim's level.
        """
        Get a batched tensor representation of this dimension.

        Returns:
            A batched tensor created from the range tensor
        """
        if self._batchtensor is None:
            self._batchtensor = torch._C._functorch._add_batch_dim(
                self._get_range(), 0, self._level
            )
        return self._batchtensor

    def __repr__(self) -> str:
        # [FUNCTION DOC] Text form is just the Dim's name.
        """String representation of a Dim object."""
        return self._name

    # note that Dim comes before tensor because we want the Dim API for things like size to take precedence.
    # Tensor defines format, but we want to print Dims with special formatting
    __format__ = object.__format__


# Somewhat confusingly, an FCD tensor is also called Tensor.  This confusion
# is somewhat intentional, as FCD tensors are intended to be substitutable
# with regular Tensor (just with some positional dims hidden).
class Tensor(_Tensor):
    # [CLASS DOC] A tensor with first-class dimensions. State: _tensor (underlying data, or None
    # while a delayed computation is pending), _levels (list of DimEntry, one per underlying
    # dimension, mixing positional and Dim levels), _has_device (whether it is on a non-CPU
    # device), _batchtensor (lazily built functorch batched form) and delayed-evaluation fields
    # (_delayed, _delayed_orig, _delayed_args). Instances are made through from_positional or
    # create_delayed, not by direct construction. Key methods: ndim, order, tensor access.
    _tensor: torch.Tensor | None
    _batchtensor: torch.Tensor | None
    _levels: list[DimEntry]
    _has_device: bool
    _delayed: Callable[[], torch.Tensor] | None
    _delayed_orig: Callable | None
    _delayed_args: tuple | None

    @property
    def ndim(self) -> int:
        # [FUNCTION DOC] Property: number of positional levels.
        return sum(1 if l.is_positional() else 0 for l in self._levels)

    @classmethod
    def check_exact(cls, other: Any) -> bool:
        # [FUNCTION DOC] True only if other's type is exactly Tensor (subclasses excluded).
        return type(other) is cls

    @classmethod
    def from_positional(
        cls, tensor: torch.Tensor, levels: list[DimEntry], has_device: bool
    ) -> _Tensor | torch.Tensor:
        # [FUNCTION DOC] Wrap tensor with levels (one per tensor dimension). Validates that
        # positional levels are consecutive and end at -1 (raises AssertionError otherwise, and
        # if the level count differs from tensor.dim()). If no level is a Dim, returns the plain
        # tensor unchanged; else returns a new Tensor holding tensor, levels and has_device.
        """
        Create a functorch Tensor from a regular PyTorch tensor with specified dimension levels.

        This is the primary way to create Tensor objects with first-class dimensions.

        Args:
            tensor: The underlying PyTorch tensor
            levels: List of DimEntry objects specifying the dimension structure
            has_device: Whether the tensor is on a device (not CPU)

        Returns:
            A new Tensor instance with the specified dimensions, or a regular torch.Tensor
            if there are no named dimensions
        """
        seen_dims = 0
        last = 0

        for l in levels:
            if l.is_positional():
                # Validate consecutive positional dimensions
                if not (last == 0 or last + 1 == l.position()):
                    raise AssertionError(
                        f"Positional dimensions must be consecutive, got {last} then {l.position()}"
                    )
                last = l.position()
            else:
                # This is a named dimension
                seen_dims += 1

        # Validate final positional dimension
        if not (last == 0 or last == -1):
            raise AssertionError(
                f"Final positional dimension must be 0 or -1, got {last}"
            )

        if not seen_dims:
            return tensor

        # Create Tensor object with proper level management
        result = cls()
        result._tensor = tensor
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None  # Will be created lazily if needed
        result._delayed = None
        result._delayed_orig = None
        result._delayed_args = None

        # Validate tensor dimensionality matches levels
        if tensor.dim() != len(levels):
            raise AssertionError(
                f"Tensor has {tensor.dim()} dimensions but {len(levels)} levels provided"
            )

        return result

    @classmethod
    def create_delayed(
        cls, orig: Callable, args: tuple, levels: list[DimEntry], has_device: bool
    ) -> _Tensor:
        # [FUNCTION DOC] Build a Tensor with no data yet that records orig applied to args; the
        # operation runs when the tensor is first read (see _get_tensor), after unwrapping any
        # tensor-like args. Stores levels and has_device as given.
        """
        Create a delayed tensor that defers the operation until later.
        """
        result = cls()
        result._tensor = None  # Will be computed when needed
        result._levels = levels
        result._has_device = has_device
        result._batchtensor = None
        result._delayed_orig = orig
        result._delayed_args = args

        # Create delayed evaluation function that unwraps Tensor objects
        def evaluate_delayed() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Replace each argument that has a tensor getter with its
            # underlying tensor, call the recorded operation and return its result.
            unwrapped_args = []
            for arg in args:
                if hasattr(arg, "_get_tensor"):
                    unwrapped_args.append(arg._get_tensor())
                else:
                    unwrapped_args.append(arg)
            return orig(*unwrapped_args)

        result._delayed = evaluate_delayed

        return result

    def _get_tensor(self) -> torch.Tensor | None:
        # [FUNCTION DOC] Return the underlying tensor, first running and caching the pending
        # delayed computation if there is one (then clearing the delayed fields).
        """Get the underlying tensor, handling delayed operations if needed."""
        if (
            hasattr(self, "_delayed")
            and self._delayed is not None
            and self._tensor is None
        ):
            # Execute the delayed operation
            self._tensor = self._delayed()
            # Clear delayed operation to avoid re-execution
            self._delayed = None
            self._delayed_orig = None
            self._delayed_args = None
        return self._tensor

    def _get_levels(self) -> list[Any]:
        # [FUNCTION DOC] Return the stored list of levels (not a copy).
        """Get the dimension levels."""
        return self._levels

    def _get_has_device(self) -> bool:
        # [FUNCTION DOC] Return the stored has-device flag.
        """Get whether this tensor has device information."""
        return self._has_device

    def _get_batchtensor(self) -> torch.Tensor | None:
        # [FUNCTION DOC] Return the functorch batched form of the data (one batch dimension per
        # Dim level), building and caching it on first use via _add_batch_dims.
        """Get the batched tensor representation, creating it lazily if needed."""
        if self._batchtensor is None:
            self._batchtensor = self._add_batch_dims(
                self._get_tensor(), self._get_levels()
            )
        return self._batchtensor

    def _add_batch_dims(
        self, t: torch.Tensor | None, levels_: list[Any]
    ) -> torch.Tensor | None:
        # [FUNCTION DOC] HELPER: Wrap t as a functorch batched tensor, adding one batch dimension
        # per Dim level, repeatedly picking the Dim with the smallest creation level first and
        # replacing its entry with a "none" entry. Returns t when no Dim levels remain. Raises
        # AssertionError if t is None while Dim levels exist.
        levels = list(levels_)

        while True:
            min_real_index = -1
            min_index = -1
            min_value = float("inf")  # INT_MAX equivalent
            i = 0
            r = 0

            for r, l in enumerate(levels):
                if not l.is_none():
                    if not l.is_positional() and l.dim()._level < min_value:
                        min_value = l.dim()._level
                        min_index = i
                        min_real_index = r
                    i += 1

            if min_index == -1:
                return t

            if t is None:
                raise AssertionError("Expected t to be non-None")
            t = torch._C._functorch._add_batch_dim(t, min_index, int(min_value))

            levels[min_real_index] = DimEntry()
        return None

    def order(self, *dims: Any) -> _Tensor:
        # [FUNCTION DOC] Convert the given Dims of this tensor into positional dimensions in the
        # given order, delegating to the order module; returns the resulting tensor.
        """Reorder the dimensions of this tensor."""
        from ._order import order

        result = order(self, *dims)
        return result  # type: ignore[return-value]  # Tensor and torch.Tensor are interchangeable


def stack(tensors: Any, new_dim: Any, dim: int = 0) -> _Tensor:
    # [FUNCTION DOC] Stack tensors (which may carry first-class dims) along new_dim. If new_dim is
    # not a Dim, falls back to ordinary stack at position dim. Otherwise aligns all inputs to the
    # union of their levels, binds new_dim's size to the count of tensors, stacks at the
    # resolved position (dim given as an int, position or Dim, relative to the result's
    # positional dims) and inserts new_dim as a level there. Raises ValueError for an empty
    # input, TypeError if the dim is not found. Returns a Tensor.
    """
    Stack tensors along a new dimension.

    Args:
        tensors: Sequence of tensors to stack
        new_dim: The new Dim to create for stacking
        dim: The dimension position to insert the new dimension (default: 0)

    Returns:
        Stacked tensor with the new dimension
    """
    if not tensors:
        raise ValueError("stack expects a non-empty sequence of tensors")

    # Check if new_dim is a Dim object
    if not isinstance(new_dim, Dim):
        # Fall back to regular torch.stack
        result = torch.stack(tensors, dim=dim)
        return result  # type: ignore[return-value]

    # Collect all result_levels from input tensors
    result_levels = []
    infos = []

    for t in tensors:
        info = TensorInfo.create(t, ensure_batched=False, ensure_present=False)
        infos.append(info)
        for level in info.levels:
            if level not in result_levels:
                result_levels.append(level)

    # Set the new_dim size to match number of tensors
    new_dim.size = len(tensors)

    # Match all tensors to the common level structure using _match_levels
    inputs = []
    for info in infos:
        if info.tensor is None:
            raise AssertionError("Cannot stack tensors with None tensor data")
        matched_tensor = _match_levels(info.tensor, info.levels, result_levels)
        inputs.append(matched_tensor)

    # Calculate ndim and resolve the dim parameter
    ndim = ndim_of_levels(result_levels)
    rawdim = 0
    if dim is not None and not (isinstance(dim, int) and dim == 0):
        from ._wrap import _wrap_dim

        d = _wrap_dim(dim, ndim, False)
        try:
            idx = result_levels.index(d)
        except ValueError:
            raise TypeError(f"Dimension {dim} does not exist in inputs") from None
        rawdim = idx

    # Stack tensors at the resolved dimension
    result = torch.stack(inputs, rawdim)

    # Insert new dimension entry at the correct position
    result_levels.insert(rawdim, DimEntry(new_dim))

    # Return as a first-class tensor
    tensor_result = Tensor.from_positional(
        result, result_levels, infos[0].has_device if infos else True
    )
    return tensor_result  # type: ignore[return-value]


def split(tensor: Any, split_size_or_sections: Any, dim: Any = None) -> tuple:
    # [FUNCTION DOC] Split tensor along dim. Integer sizes (single int or all-int sequence)
    # use the ordinary split path (raising TypeError if dim is a Dim). If the sizes are all
    # Dims, splits the chosen dimension into pieces whose sizes bind those Dims; unbound Dims
    # share the remaining length evenly (rounded up chunks), bound ones keep their size; the
    # sizes must add up to the source size. Each piece is a Tensor with the split level
    # replaced by the corresponding Dim. Raises TypeError for mixed sizes, missing dimension,
    # or size mismatch. Returns a tuple of pieces.
    """
    Split tensor along a dimension.

    Can handle both regular integer sizes and Dim objects for split sizes.
    When Dim objects are used, they get bound to the resulting tensor dimensions.
    """
    from ._wrap import _wrap_dim

    # Check if dim is a Dim object
    dim_is_object = isinstance(dim, Dim)

    # Parse split_size_or_sections
    if isinstance(split_size_or_sections, int):
        # Single integer - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    # Check if it's a sequence
    sizes = []
    all_dims = True
    all_ints = True

    for item in split_size_or_sections:
        sizes.append(item)
        if isinstance(item, Dim):
            all_ints = False
        else:
            all_dims = False

    if all_ints:
        # All integers - use regular split
        if dim_is_object:
            raise TypeError(
                "when dim is specified as a Dim object, split sizes must also be dimensions."
            )
        return _Tensor._torch_function_fallback(
            torch.Tensor.split,
            (type(tensor),),
            (tensor, split_size_or_sections),
            {"dim": dim},
        )

    if not all_dims:
        raise TypeError("split list must be ints or dims but got a mix")

    # All are Dim objects - handle first-class dimension split
    self_info = TensorInfo.create(tensor, ensure_batched=False, ensure_present=False)
    ndim = self_info.ndim()

    if not dim_is_object and ndim == 0:
        raise TypeError("split expects at least a 1-dimension tensor")

    # Wrap the dimension
    dim_l = _wrap_dim(dim, ndim, False) if dim is not None else DimEntry(-ndim)

    # Find the index of the dimension in levels
    idx = None
    for i, level in enumerate(self_info.levels):
        if level == dim_l:
            idx = i
            break

    if idx is None:
        if dim is None:
            dim = 0
        raise TypeError(f"tensor does not contain dimension {dim}")

    # Calculate split indices
    indices = []
    total_size = 0
    unbound = []

    for i, size_dim in enumerate(sizes):
        if size_dim.is_bound:
            indices.append(size_dim.size)
            total_size += indices[-1]
        else:
            indices.append(0)
            unbound.append(i)

    if self_info.tensor is None:
        raise AssertionError("Cannot get tensor size on None tensor")
    tensor_size = self_info.tensor.size(idx)

    # Handle unbound dimensions
    if unbound:
        if total_size > tensor_size:
            raise TypeError(
                f"sizes of target dimensions add up to more ({total_size}) than source dim ({tensor_size})"
            )
        remaining_size = tensor_size - total_size
        chunk_size = (remaining_size + len(unbound) - 1) // len(unbound)
        for u in unbound:
            sz = min(chunk_size, remaining_size)
            sizes[u].size = sz
            indices[u] = sz
            remaining_size -= sz
    elif tensor_size != total_size:
        raise TypeError(
            f"sum of sizes of target dimensions ({total_size}) do not match the source dim ({tensor_size})"
        )

    # Perform the split
    result_tensors = self_info.tensor.split_with_sizes(indices, idx)

    # Create result with new levels
    result = []
    new_levels = list(self_info.levels)

    for i, (result_tensor, size_dim) in enumerate(zip(result_tensors, sizes)):
        new_levels[idx] = DimEntry(size_dim)
        result.append(
            Tensor.from_positional(
                result_tensor, list(new_levels), self_info.has_device
            )
        )

    return tuple(result)


def cat(tensors: Any, dim: Any, new_dim: Any) -> _Tensor:
    # [FUNCTION DOC] Concatenate tensors along dim by stacking them along a fresh temporary Dim
    # and then indexing the pair (temporary Dim, dim) with new_dim, which flattens them into
    # new_dim. Returns the resulting Tensor.
    n = dims(1)  # Get single Dim instead of tuple
    return stack(tensors, n, dim).index([n, dim], new_dim)  # type: ignore[list-item]


class DotPart:
    # [CLASS DOC] HELPER: Accumulator for one group of dimension entries in a dot product. State:
    # dims (list of DimEntry) and total_size (product of the sizes of the first-class Dims
    # appended; positional entries do not contribute).
    """
    Helper class for organizing dimensions in dot products.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Start with no entries and a total size of one.
        self.dims: list[DimEntry] = []
        self.total_size = 1

    def append(self, dim_entry: Any) -> None:
        # [FUNCTION DOC] HELPER: Add dim_entry to dims and, if it is a first-class Dim, multiply
        # total_size by its size (raises ValueError if that Dim is unbound).
        """Add a dimension entry to this part."""
        self.dims.append(dim_entry)
        if not dim_entry.is_positional():
            self.total_size *= dim_entry.dim().size


def dot_prepare(parts: list[DotPart], tensor_info: TensorInfo) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Rearrange the tensor so its dimensions follow the concatenated
    # dims of the parts, then, if any part has other than exactly one dim, reshape to one
    # dimension per part using each part's total size. Raises RuntimeError if the tensor is None.
    """
    Prepare tensor for dot product by matching levels and reshaping.
    """
    new_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        new_levels.extend(part.dims)

    if tensor_info.tensor is None:
        raise RuntimeError("Cannot perform dot product on None tensor")
    result = _match_levels(tensor_info.tensor, tensor_info.levels, new_levels)

    if not needs_reshape:
        return result

    # Reshape for matrix operations
    view = [part.total_size for part in parts]
    return result.reshape(view)


def dot_finish(parts: list[DotPart], result_tensor: torch.Tensor) -> Tensor:
    # [FUNCTION DOC] HELPER: Turn a matrix-multiply result back into a Tensor whose levels are the
    # concatenated dims of the parts, reshaping to the individual Dim sizes if any part had
    # other than one dim. The has-device flag is always set to true.
    """
    Finish dot product by reshaping result and creating Tensor.
    """
    result_levels = []
    needs_reshape = False

    for part in parts:
        if len(part.dims) != 1:
            needs_reshape = True
        result_levels.extend(part.dims)

    if needs_reshape:
        new_size = []
        for level in result_levels:
            new_size.append(level.dim().size)
        result_tensor = result_tensor.reshape(new_size)

    tensor_result = Tensor.from_positional(result_tensor, result_levels, True)
    return tensor_result  # type: ignore[return-value]


def dot(lhs: Any, rhs: Any, sum_dims: Any) -> _Tensor | torch.Tensor:
    # [FUNCTION DOC] Contract lhs and rhs over sum_dims using a matrix multiply. Classifies each
    # level as batch (both or neither have stride), left-only, right-only or contracted (in
    # sum_dims), reshapes both operands accordingly and calls a batched multiply when there are
    # batch dims, else a plain matrix multiply. If either operand has no tensor info, falls back
    # to ordinary matmul. Raises ValueError if a summed dimension exists in neither operand.
    # Returns a Tensor with levels batch, left-only, right-only.
    """
    Perform dot product between two tensors along specified dimensions.

    Args:
        lhs: Left-hand side tensor
        rhs: Right-hand side tensor
        sum_dims: Dimensions to sum over (contract)

    Returns:
        Result of dot product
    """
    # Get tensor info
    lhs_info = TensorInfo.create(lhs, ensure_batched=False, ensure_present=False)
    rhs_info = TensorInfo.create(rhs, ensure_batched=False, ensure_present=False)

    if not (lhs_info and rhs_info):
        # Fall back to regular operations
        return torch.matmul(lhs, rhs)

    if lhs_info.tensor is None or rhs_info.tensor is None:
        raise AssertionError("Cannot perform dot product on None tensors")

    lhs_strides = lhs_info.tensor.stride()
    rhs_strides = rhs_info.tensor.stride()

    # Create dot parts for different dimension categories
    lro_dims = DotPart()  # Left-right-output (batch dims)
    lo_dims = DotPart()  # Left-output only
    ro_dims = DotPart()  # Right-output only
    lr_dims = DotPart()  # Left-right (contracted dims)

    def insert_dim(d: Any, lhs_idx: Any, rhs_idx: Any) -> None:
        # [FUNCTION DOC] HELPER: File level d into the contracted part if it is in sum_dims, else
        # by the strides of its position in lhs/rhs (None index counts as stride zero): both or
        # neither nonzero goes to batch, lhs only to left-only, otherwise right-only.
        """Insert dimension into appropriate part based on stride pattern."""
        reduced = d in sum_dims
        lhs_stride = lhs_strides[lhs_idx] if lhs_idx is not None else 0
        rhs_stride = rhs_strides[rhs_idx] if rhs_idx is not None else 0

        if reduced:
            lr_dims.append(d)
        else:
            if (lhs_stride == 0) == (rhs_stride == 0):
                lro_dims.append(d)  # Both have or both lack this dim
            elif lhs_stride != 0:
                lo_dims.append(d)  # Only lhs has this dim
            else:
                ro_dims.append(d)  # Only rhs has this dim

    # Track which rhs dimensions we've seen
    rhs_seen = [False] * len(rhs_info.levels)

    # Process lhs dimensions
    for i, lhs_level in enumerate(lhs_info.levels):
        rhs_idx = None
        for j, rhs_level in enumerate(rhs_info.levels):
            if lhs_level == rhs_level:
                rhs_idx = j
                rhs_seen[j] = True
                break

        insert_dim(lhs_level, i, rhs_idx)

    # Process remaining rhs dimensions
    for i, rhs_level in enumerate(rhs_info.levels):
        if not rhs_seen[i]:
            insert_dim(rhs_level, None, i)

    # Validate sum dimensions exist
    if len(lr_dims.dims) != len(sum_dims):
        for d in sum_dims:
            if d not in lhs_info.levels and d not in rhs_info.levels:
                raise ValueError(f"summing over non-existent dimension {d}")

    # Prepare tensors and perform matrix multiplication
    if len(lro_dims.dims) != 0:
        # Batched matrix multiply
        lhs_tensor = dot_prepare([lro_dims, lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lro_dims, lr_dims, ro_dims], rhs_info)
        result = torch.bmm(lhs_tensor, rhs_tensor)
        return dot_finish([lro_dims, lo_dims, ro_dims], result)
    else:
        # Regular matrix multiply
        lhs_tensor = dot_prepare([lo_dims, lr_dims], lhs_info)
        rhs_tensor = dot_prepare([lr_dims, ro_dims], rhs_info)
        result = torch.mm(lhs_tensor, rhs_tensor)
        return dot_finish([lo_dims, ro_dims], result)


from functorch.dim._wrap import _wrap
from functorch.dim.wrap_type import wrap_type


wrap_type(_Tensor, torch.Tensor, _Tensor.__torch_function__)
del _Tensor.ndim


def index(self: Any, positions: Any, dims: Any) -> _Tensor:
    # [FUNCTION DOC] Module-level index function usable on ordinary tensors: if self is already
    # a first-class tensor, calls its index method; otherwise wraps the plain tensor as a
    # first-class Tensor and then indexes positions with dims. Returns the result.
    """
    Index a regular tensor by binding specified positions to dims.

    This converts a regular tensor to a first-class tensor by binding
    the specified positional dimensions to Dim objects.

    Args:
        positions: Tuple of dimension positions to bind
        dims: Dim objects or tuple of Dim objects to bind to

    Returns:
        First-class tensor with specified dimensions bound
    """
    # If this is already a first-class tensor (_Tensor), call its index method directly
    if isinstance(self, _Tensor):
        return _Tensor.index(self, positions, dims)

    # Convert regular tensor to first-class tensor
    info = TensorInfo.create(self, ensure_batched=False, ensure_present=False)

    # Create the first-class tensor
    if info.tensor is None:
        raise AssertionError("Cannot index None tensor")
    result = Tensor.from_positional(info.tensor, info.levels, info.has_device)

    # Now call the index method on the first-class tensor
    # Cast result to _Tensor for the method call
    return _Tensor.index(result, positions, dims)  # type: ignore[arg-type]


def _def(name: str, *args: Any, **kwargs: Any) -> None:
    # [FUNCTION DOC] HELPER: Look up the torch.Tensor method called name, wrap it with _wrap
    # (forwarding the extra arguments such as reduce, single_dim, dim_offset) and install the
    # wrapper as an attribute of _Tensor. Global side effect at import time.
    orig = getattr(torch.Tensor, name)
    setattr(_Tensor, name, _wrap(orig, *args, **kwargs))


_def("mean")
_def("sum")
_def("all")
_def("amax")
_def("amin")
_def("aminmax")
_def("any")
_def("count_nonzero")
_def("logsumexp")
_def("nanmean")
_def("nansum")
_def("prod")
_def("std", keepdim_offset=2)
_def("var", keepdim_offset=2)
_def("max", single_dim=True)
_def("min", single_dim=True)
_def("argmax", single_dim=True)
_def("argmin", single_dim=True)
_def("kthvalue", single_dim=True)
_def("median", single_dim=True)
_def("nanmedian", single_dim=True)
_def("mode", single_dim=True)
_def("sort", reduce=False)
_def("argsort", reduce=False)
_def("unbind", single_dim=True)
_def("chunk", dim_offset=1, reduce=False)
_def("cummax", single_dim=True, reduce=False)
_def("cummin", single_dim=True, reduce=False)
_def("cumprod", single_dim=True, reduce=False)
_def("cumprod_", single_dim=True, reduce=False)
_def("cumsum", single_dim=True, reduce=False)
_def("cumsum_", single_dim=True, reduce=False)
_def("logcumsumexp", single_dim=True, reduce=False)
_def("renorm", dim_offset=1, single_dim=True, reduce=False)
_def("softmax", single_dim=True, reduce=False)
softmax = _wrap(torch.nn.functional.softmax, single_dim=True, reduce=False)

# stuff to handle in the future, because they require special
# binding logic for dims
# cross
# diag_embed
# diagonal
# diagonal_scatter
# diff
# nanquantile
# quantile
# roll
# rot90
# topk (new dimes on output)
# should these all be subsumed by inplace indexing?
# index_add_
# index_add
# index_copy
# index_copy_
# index_fill
# index_fill_
# index_select
# scatter
# scatter_
# scatter_add
# scatter_add_
# scatter_reduce
