# [MODULE DOC] Shared infrastructure for ComplexTensor ops: the op-to-implementation table, dtype maps, registration
# decorators (generic, error-raising, binary nonlinear, simple per-part), lookup with decomposition fallback,
# splitting/promotion helpers, and the dispatch modes WrapComplexMode and ComplexTensorMode.
# Import-time side effect: computes the table of aten decompositions for all aten ops.
from collections.abc import Callable
from typing import Any, overload, TypeAlias
from typing_extensions import Never, ParamSpec, TypeIs, TypeVar

import torch
from torch import Tensor
from torch._decomp import get_decompositions
from torch._ops import OpOverload, OpOverloadPacket
from torch._refs import is_complex as _is_complex
from torch.types import Number
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

from .._core import ComplexTensor


_P = ParamSpec("_P")
_R = TypeVar("_R")

OpType: TypeAlias = OpOverloadPacket | OpOverload

# pyrefly: ignore [implicit-any]
TableType: TypeAlias = dict[OpType, Callable]

# Mapping from ops to implementations
COMPLEX_OPS_TABLE: TableType = {}

# Ops that expand complex multiply into four real ops (different accumulation order
# than native complex BLAS).
BINARY_NONLINEAR_OPS: set[OpType] = set()

COMPLEX_TO_REAL = {
    torch.complex128: torch.float64,
    torch.complex64: torch.float32,
    torch.complex32: torch.float16,
}

REAL_TO_COMPLEX = {v: k for k, v in COMPLEX_TO_REAL.items()}

# Used to promote dtypes in `promote_real_cpu_tensors`
PROMOTE_TYPES = {
    torch.float16: torch.float32,
    torch.bfloat16: torch.float32,
    torch.complex32: torch.complex64,
}


def is_complex_tensor(obj: Any, /) -> TypeIs[ComplexTensor]:
    # [FUNCTION DOC] Return True if the object is a ComplexTensor instance.
    # KNOWN ISSUE: the docstring example imports from a module path that does not match this package.
    r"""Returns True if the input is a ComplexTensor, else False

    Args:
        a: any input

    Examples:

        >>> # xdoctest: +SKIP
        >>> from torch.complex import ComplexTensor
        >>> data = torch.zeros((3, 2), dtype=torch.complex64)
        >>> ct = ComplexTensor.from_interleaved(data)
        >>> is_complex_tensor(ct)
        True
    """
    return isinstance(obj, ComplexTensor)


@overload
def promote_tensors(
    *tensors: ComplexTensor,
) -> tuple[torch.dtype, tuple[ComplexTensor, ...]]: ...


@overload
def promote_tensors(
    *tensors: Tensor,
) -> tuple[torch.dtype, tuple[Tensor, ...]]: ...


def promote_tensors(
    *tensors: Tensor | ComplexTensor,
) -> tuple[torch.dtype, tuple[Tensor | ComplexTensor, ...]]:
    # [FUNCTION DOC] Promote all tensor arguments to a common dtype (at least float32 for half types and complex32 up to complex64).
    # Non-tensor entries are turned into tensors of that dtype. Returns the original-promoted dtype and the converted tuple.
    # Raises StopIteration if no argument is a tensor.
    """
    Promotes all tensors to a common dtype.
    Additionally promotes CPU tensors to at least `float32`.
    """
    tensor = next(t for t in tensors if isinstance(t, Tensor))
    out_dt = tensor.dtype
    for t in tensors:
        if isinstance(t, Tensor):
            out_dt = torch.promote_types(out_dt, t.dtype)

    prom_dt = PROMOTE_TYPES.get(out_dt, out_dt)
    return out_dt, tuple(
        t.to(prom_dt) if isinstance(t, Tensor) else torch.asarray(t, dtype=prom_dt)
        for t in tensors
    )


def register_complex(
    op: OpType,
    func_impl: Callable[..., Any] | None = None,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    # [FUNCTION DOC] Decorator factory registering an implementation for an op in the table. With func_impl given, registers it immediately.
    # Raises RuntimeError if a different function is already registered for the op.
    """Decorator to register an implementation for some ops in some dispatch tables"""

    def inner(func: Callable[_P, _R]) -> Callable[_P, _R]:
        # [FUNCTION DOC] HELPER: Register func for the op, erroring on duplicates, and return func unchanged.
        if COMPLEX_OPS_TABLE.get(op, func) is not func:
            raise RuntimeError(f"Attempted to register multiple functions for {op}")
        COMPLEX_OPS_TABLE[op] = func
        return func

    if func_impl is None:
        return inner

    return inner(func_impl)


FORCE_TEST_LIST: list[OpType] = []


def register_force_test(
    op: OpType, func_impl: Callable[..., Any] | None = None
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    # [FUNCTION DOC] Register as register_complex does and also add the op to the list of ops tested even if they fail on normal inputs.
    """Will attempt to test these ops even if they err on "normal" inputs"""
    FORCE_TEST_LIST.append(op)
    return register_complex(op, func_impl)


DECOMPOSITIONS = get_decompositions(list(torch.ops.aten))  # type: ignore[no-matching-overload]


def lookup_complex(
    func: OpOverload, *args: Any, **kwargs: Any
) -> Callable[..., Any] | None:
    # [FUNCTION DOC] Find the implementation for an op: the overload, then its packet, then decompositions for each.
    # Returns None if nothing is found. Extra arguments are ignored.
    """
    Lookup an impl from the table.

    Try the particular overload first, then the overload packet.

    If nothing is found, try the decompositions with both.
    """
    return COMPLEX_OPS_TABLE.get(
        func,
        COMPLEX_OPS_TABLE.get(
            func.overloadpacket,
            DECOMPOSITIONS.get(func, DECOMPOSITIONS.get(func.overloadpacket)),
        ),
    )


def is_complex(x: Any, /) -> bool:
    # [FUNCTION DOC] True if x is a tensor with complex dtype or a Python complex number.
    """Utility to detect if a given object is (known) to be complex."""
    return (isinstance(x, Tensor) and _is_complex(x)) or isinstance(x, complex)


@overload
def split_complex_arg(
    arg: Tensor | ComplexTensor,
) -> tuple[Tensor, Tensor]: ...


@overload
def split_complex_arg(
    arg: complex | Number,
) -> tuple[Number, Number]: ...


def split_complex_arg(
    arg: Tensor | ComplexTensor | complex | Number,
) -> tuple[Tensor, Tensor] | tuple[Number, Number]:
    # [FUNCTION DOC] Split a tensor, ComplexTensor or number into real and imaginary parts. Real tensors/numbers get zeros for imaginary.
    # Raises TypeError for unsupported types.
    """
    Split a complex argument into a real/imaginary component.

    If real, use zero for the imaginary part.
    """
    if isinstance(arg, ComplexTensor):
        return split_complex_tensor(arg)
    if isinstance(arg, Tensor):
        if is_complex(arg):
            return arg.real, arg.imag
        return arg, torch.zeros_like(arg)
    # TODO (hameerabbasi): Should there be a `torch.SymComplex`?
    if isinstance(arg, complex):
        return arg.real, arg.imag
    if isinstance(arg, float | torch.SymFloat):
        return arg, 0.0
    if isinstance(arg, int | torch.SymInt):
        return arg, 0
    if isinstance(arg, bool | torch.SymBool):
        return arg, False
    raise TypeError(f"Expected tensor or number got, {type(arg)}")


def split_complex_tensor(complex_tensor: ComplexTensor) -> tuple[Tensor, Tensor]:
    # [FUNCTION DOC] Return the real and imaginary tensors of a ComplexTensor with any lazy negation resolved.
    """Split a ComplexTensor into its real and imaginary parts."""
    return torch.resolve_neg(complex_tensor.re), torch.resolve_neg(complex_tensor.im)


def complex_to_real_dtype(dtype: torch.dtype) -> torch.dtype:
    # [FUNCTION DOC] Map a complex dtype to the matching real dtype; other dtypes pass through.
    """Convert a complex dtype to the dtype of its real part. Return other dtypes as-is."""
    return COMPLEX_TO_REAL.get(dtype, dtype)


def _get_op_name(op: OpType) -> str:
    # [FUNCTION DOC] HELPER: Return the op's name without the namespace prefix (using its packet for overloads).
    """Get the op name from the op."""
    if isinstance(op, OpOverload):
        op = op.overloadpacket
    return str(op).split(".", 1)[1]


def is_binary_nonlinear_op(op: OpType) -> bool:
    # [FUNCTION DOC] Return True if the op (or its packet) was registered as a binary nonlinear op.
    if isinstance(op, OpOverload):
        op = op.overloadpacket
    return op in BINARY_NONLINEAR_OPS


def _get_func_name(op: OpType) -> str:
    # [FUNCTION DOC] HELPER: Name of the implementation function for an op, formed from the op name plus an impl suffix.
    """Get the name of the implementation function from the op."""
    return f"{_get_op_name(op)}_impl"


def register_error(
    op: OpType, exc_type: type[Exception] = NotImplementedError
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    # [FUNCTION DOC] Register an implementation for the op that always raises the given exception type (default NotImplementedError)
    # with a message naming the op; registered as force-tested. Returns the registered function.
    msg = f"`aten.{_get_op_name(op)}` not implemented for `{ComplexTensor.__name__}`."

    def ordered_impl(*args: Any, **kwargs: Any) -> Never:
        # [FUNCTION DOC] HELPER: Always raises the configured exception with the not-implemented message.
        raise exc_type(msg)

    func_name = _get_func_name(op)
    ordered_impl.__name__ = func_name
    ordered_impl.__qualname__ = func_name

    return register_force_test(op, ordered_impl)


def register_binary_nonlinear(
    op: OpType,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    # [FUNCTION DOC] Register a multiplication-style op (mul, mm, ...) by expanding it into four real op calls: real = a_r*b_r - a_i*b_i,
    # imag = a_r*b_i + a_i*b_r. Also records the op in the binary nonlinear set.
    """Register a "multiplication-style" op, e.g. aten.mul, aten.mm, ..."""
    BINARY_NONLINEAR_OPS.add(op)

    def impl(
        a: ComplexTensor, b: ComplexTensor, *args: Any, **kwargs: Any
    ) -> ComplexTensor:
        # [FUNCTION DOC] HELPER: Promote operands, split both into parts, apply the op four ways, combine, and cast back to the promoted dtype.
        out_dt, (a, b) = promote_tensors(a, b)
        a_r, a_i = split_complex_arg(a)
        b_r, b_i = split_complex_arg(b)
        real = op(a_r, b_r, *args, **kwargs) - op(a_i, b_i, *args, **kwargs)
        imag = op(a_r, b_i, *args, **kwargs) + op(a_i, b_r, *args, **kwargs)
        return ComplexTensor(real, imag).to(out_dt)  # type: ignore[bad-return]

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def register_simple(
    op: OpType,
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]] | Callable[..., Any]:
    # [FUNCTION DOC] Register an op that applies independently to real and imaginary parts (supports nested outputs via flatten).
    """Register an op which can be applied independently to the real and complex parts to get the result."""

    def impl(
        self: ComplexTensor, *args: Any, dtype: torch.dtype | None = None, **kwargs: Any
    ) -> ComplexTensor:
        # [FUNCTION DOC] HELPER: Apply the op to each part and wrap pairs of results in ComplexTensor, preserving output structure.
        # A dtype keyword must be a complex dtype (mapped to its real dtype) else raises RuntimeError.
        x, y = split_complex_tensor(self)
        if dtype is not None and dtype not in COMPLEX_TO_REAL:
            raise RuntimeError(
                "Non-complex `dtype` specified, please write custom impl."
            )

        if dtype in COMPLEX_TO_REAL:
            if dtype is None:
                raise AssertionError("dtype must not be None when in COMPLEX_TO_REAL")
            kwargs["dtype"] = COMPLEX_TO_REAL[dtype]

        u = op(x, *args, **kwargs)
        v = op(y, *args, **kwargs)

        u_flat, u_spec = tree_flatten(u)
        v_flat, v_spec = tree_flatten(v)
        if u_spec != v_spec:
            raise AssertionError(f"Tree specs must match: {u_spec} != {v_spec}")
        out_flat = [
            ComplexTensor(ui, vi) for ui, vi in zip(u_flat, v_flat, strict=False)
        ]
        return tree_unflatten(out_flat, u_spec)

    func_name = _get_func_name(op)
    impl.__name__ = func_name
    impl.__qualname__ = func_name

    return register_complex(op, impl)


def _as_complex_tensor(arg: Tensor | Any) -> Tensor | ComplexTensor | Any:
    # [FUNCTION DOC] HELPER: Convert complex-dtype ordinary tensors to ComplexTensor; return other values unchanged.
    """Convert a Tensor with complex dtypes to a ComplexTensor. Pass along other args as-is."""
    if (
        not isinstance(arg, ComplexTensor)
        and isinstance(arg, Tensor)
        and arg.dtype in COMPLEX_TO_REAL
    ):
        return ComplexTensor.from_interleaved(arg)
    return arg


def _as_interleaved(arg: ComplexTensor | Any) -> Tensor | Any:
    # [FUNCTION DOC] HELPER: Convert a ComplexTensor to an ordinary complex tensor; return other values unchanged.
    """Convert a ComplexTensor to a Tensor with a complex dtype. Pass other arguments as-is."""
    if isinstance(arg, ComplexTensor):
        return arg.as_interleaved()
    return arg


class WrapComplexMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that converts complex tensor inputs to ComplexTensor, runs the op, and converts complex-dtype
    # results to ComplexTensor (outputs stay wrapped).
    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # [FUNCTION DOC] Convert complex tensors in args and kwargs to ComplexTensor, call the op, and wrap complex results the same way.
        if kwargs is None:
            kwargs = {}
        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)
        return tree_map(_as_complex_tensor, func(*args, **kwargs))


class ComplexTensorMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that computes with ComplexTensor internally but returns ordinary complex tensors. Attribute _compile
    # selects compiling each op before running.
    # KNOWN ISSUE: the descriptive string after the attribute annotation is not a real class docstring.
    _compile: bool

    """ A TorchDispatchMode to replace any Tensor that has a complex dtype with a ComplexTensor for the computation. """

    def __init__(self, _dispatch_key: Any = None, *, _compile: bool = False) -> None:
        # [FUNCTION DOC] Initialize the mode, passing the dispatch key to the base class and storing the compile flag.
        """Initialize a ComplexTensorMode.

        Args:
            _dispatch_key: passed on to TorchDispatchMode
            _compile: Compile the op before the computation
        """
        super().__init__(_dispatch_key)
        self._compile = _compile

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # [FUNCTION DOC] Convert complex inputs to ComplexTensor, optionally compile the op, run it, and convert results back to ordinary complex tensors.
        if kwargs is None:
            kwargs = {}

        # TODO (hameerabbasi): Test perf with `_compile` set to `True`
        if self._compile:
            func = torch.compile(func)  # type: ignore[bad-assignment]

        args = tree_map(_as_complex_tensor, args)
        kwargs = tree_map(_as_complex_tensor, kwargs)

        return tree_map(_as_interleaved, func(*args, **kwargs))
