# [MODULE DOC] Dtype propagation for Inductor ops. Provides type promotion helpers and
# DtypePropagationOpsHandler, an ops handler that computes the output dtype of each op
# from its argument dtypes. Rules are built at first instantiation from registries in utils and
# codegen.common. Defines a cached promotion function and module constants only.
# mypy: allow-untyped-defs
import functools
from collections.abc import Callable, Sequence
from typing import Any, Optional, Protocol, TYPE_CHECKING, TypeVar

import sympy

import torch
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    is_integer_dtype,
    type_to_dtype,
)
from torch.utils._ordered_set import OrderedSet

from .ops_handler import OP_NAMES, OpsHandler
from .utils import upcast_compute_type
from .virtualized import OpsValue, V


T = TypeVar("T")
_MISSING_SHAPE = object()
_UNSIGNED_INT_DTYPES: frozenset[torch.dtype] = frozenset(
    (
        torch.uint8,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    )
)


class DTypeVar(Protocol):
    # [CLASS DOC] Protocol for objects exposing a dtype property (such as CSE variables).
    @property
    def dtype(self) -> torch.dtype: ...


DTypeArg = DTypeVar | torch.types.Number | str | OpsValue


# Inputs need to be cacheable (e.g., not a CSEVar) in order for the cache to be effective
# So first decompose CSEVars -> tuple before calling this


@functools.cache
def get_promoted_dtype(
    *args: Sequence[tuple[torch.dtype, bool]],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None = None,
):
    # [FUNCTION DOC] Cached helper: given (dtype, is_scalar) pairs, build empty tensors (zero-dim for scalars)
    # and return the result dtype of elementwise type promotion for the given promotion kind
    # (default kind if None). Inputs must be hashable, so decompose CSE variables first.
    def construct_input(inp):
        # [FUNCTION DOC] HELPER: Make an empty zero-dim tensor if flagged scalar, else a one-element tensor.
        if inp[1]:
            return torch.empty([], dtype=inp[0])
        else:
            return torch.empty([1], dtype=inp[0])

    inps = [construct_input(arg) for arg in args]
    _, dtype = torch._prims_common.elementwise_dtypes(
        *inps,
        type_promotion_kind=(
            type_promotion_kind
            if type_promotion_kind
            else ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
        ),
    )
    return dtype


def promote_types(
    args: Sequence[DTypeArg],
    type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND | None = None,
):
    # [FUNCTION DOC] Compute the promoted dtype of args. Numbers use the dtype of their Python type and count as
    # scalars; OpsValue is unwrapped; other args need a dtype and optional is_scalar flag.
    # Raises an error for string args or unwrapped values with neither Number nor dtype.
    dtype_prop_candidates = []

    for arg in args:
        if isinstance(arg, str):
            raise AssertionError(f"expected non-str arg, got {type(arg)}")
        if isinstance(arg, OpsValue):
            arg = arg.value
            if not (
                isinstance(arg, torch._prims_common.Number) or hasattr(arg, "dtype")
            ):
                raise AssertionError("expected a Number or an object with a dtype")

        if isinstance(arg, torch._prims_common.Number):
            dtype_prop_candidates.append((type_to_dtype(type(arg)), True))
            continue

        # pyrefly: ignore [missing-attribute]
        dtype_prop_candidates.append((arg.dtype, getattr(arg, "is_scalar", False)))

    dtype = get_promoted_dtype(
        *dtype_prop_candidates,
        type_promotion_kind=type_promotion_kind,
    )

    return dtype


def _unwrap_dtype_arg(arg: DTypeArg) -> DTypeVar | torch.types.Number | str:
    # [FUNCTION DOC] HELPER: Return the wrapped value if given an OpsValue, otherwise the argument.
    if isinstance(arg, OpsValue):
        return arg.value
    return arg


def _is_scalar_dtype_arg(arg: DTypeArg) -> bool:
    # [FUNCTION DOC] HELPER: Decide whether an op argument is a scalar: a Number, an is_scalar flag (property or
    # callable) that is true, or a shape that is None or empty. False if no shape info.
    arg = _unwrap_dtype_arg(arg)
    if isinstance(arg, torch._prims_common.Number):
        return True

    is_scalar = getattr(arg, "is_scalar", False)
    if callable(is_scalar):
        if is_scalar():
            return True
    elif is_scalar:
        return True

    shape = getattr(arg, "shape", _MISSING_SHAPE)
    if shape is _MISSING_SHAPE:
        return False

    if shape is None:
        return True
    if not isinstance(shape, Sequence):
        return False

    return len(shape) == 0


def _has_known_nonnegative_scalar_int_value(arg: DTypeArg) -> bool:
    # [FUNCTION DOC] HELPER: True if the argument is known to be a nonnegative integer: a bool, a nonnegative
    # int, or an integer-dtype value that is unsigned or has a known nonnegative lower bound.
    # False when unknown.
    arg = _unwrap_dtype_arg(arg)

    if isinstance(arg, bool):
        return True
    if isinstance(arg, int):
        return arg >= 0

    dtype = getattr(arg, "dtype", None)
    if dtype is None or not is_integer_dtype(dtype):
        return False
    if dtype in _UNSIGNED_INT_DTYPES:
        return True

    lower = getattr(getattr(arg, "bounds", None), "lower", None)
    if lower is None:
        return False
    if isinstance(lower, sympy.Expr):
        return lower.is_nonnegative is True
    return lower >= 0


class DtypePropagationOpsHandler:
    # [CLASS DOC] Ops handler mapping each op to a function returning its output dtype. Singleton
    # (via __new__) because rules are attached dynamically in __init__ from the dtype propagation
    # rule registry, pointwise override data, and boolean ops. Explicit static methods below
    # override or specialize individual ops.
    """
    Propagate dtype from args to output
    """

    # Singleton DtypePropagationOpsHandler, because we meta program over a number of op rules.
    # Those are only defined after other inductor state has run.

    _instance: Optional["DtypePropagationOpsHandler"] = None

    def __new__(cls):
        # [FUNCTION DOC] Return the single shared instance, creating it on first call.
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        # [FUNCTION DOC] Attach per-op rules as attributes: registry rules (fixed dtype or promotion kind), pointwise
        # override rules, and boolean ops returning bool, without overriding existing methods.
        # Raises an error (via torch._check) if any op in OP_NAMES remains unimplemented.
        for op, rule in torch._inductor.utils.op_dtype_propagation_rules.items():
            fn = (
                functools.partial(self.return_dtype, dtype=rule.override_return_dtype)
                if rule.override_return_dtype
                else functools.partial(
                    self.op_dtype_rule, type_promotion_kind=rule.type_promotion_kind
                )
            )
            setattr(self, op, fn)

        # Set pointwise operation rules
        for op in torch._inductor.codegen.common.pointwise_overrides_data.values():
            if not hasattr(self, op.name):
                setattr(
                    self,
                    op.name,
                    functools.partial(
                        self.op_dtype_rule, type_promotion_kind=op.type_promotion_kind
                    ),
                )

        # Set boolean operation rules
        for op in torch._inductor.utils.boolean_ops():
            if not hasattr(self, op):
                setattr(
                    self, op, functools.partial(self.return_dtype, dtype=torch.bool)
                )

        unimplemented_ops = OP_NAMES - OrderedSet(dir(self))
        torch._check(
            len(unimplemented_ops) == 0,
            lambda: f"Unimplemented dtype rule for ops: {unimplemented_ops}",
        )

    # metaprogrammed in __init__

    @staticmethod
    def op_dtype_rule(
        *args: DTypeArg, type_promotion_kind: ELEMENTWISE_TYPE_PROMOTION_KIND
    ) -> torch.dtype:
        # [FUNCTION DOC] Generic rule: promote the argument dtypes using the given promotion kind.
        return promote_types(args, type_promotion_kind=type_promotion_kind)

    @staticmethod
    def return_dtype(*args: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype
        # [FUNCTION DOC] Generic rule: ignore args and return the fixed dtype.

    # op rules

    @staticmethod
    def constant(value: torch.types.Number, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for constant: the compute dtype (upcast for low precision types) of the given dtype.
        return upcast_compute_type(dtype)

    @staticmethod
    def load_seed(name: str, offset: int) -> torch.dtype:
        return upcast_compute_type(V.graph.get_dtype(name))
        # [FUNCTION DOC] Rule for load_seed: compute dtype of the named buffer's dtype in the graph.

    @staticmethod
    def randint64(seed: int, offset: int, low: int, high: int) -> torch.dtype:
        return torch.int64
        # [FUNCTION DOC] Rule for randint64: always int64.

    @staticmethod
    def masked(
        mask: DTypeArg, body: Callable[[], DTypeArg], other: DTypeArg
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for masked: if the body block has at most one load, promote the other value; otherwise
        # the compute dtype of the last load's buffer. Raises an error if body is not a LoopBodyBlock.
        from .loop_body import LoopBodyBlock

        if not isinstance(body, LoopBodyBlock):
            raise AssertionError("body must be a LoopBodyBlock")
        # TODO - we avoid calling this in codegen, needs work for non codegen use cases
        loads = body.graph.find_nodes(op="call_method", target="load")
        if len(loads) <= 1:
            return promote_types([other])

        return upcast_compute_type(V.graph.get_dtype(loads[-1].args[1]))

    @staticmethod
    def where(a: DTypeArg, b: DTypeArg, c: DTypeArg) -> torch.dtype:
        return promote_types([b, c])
        # [FUNCTION DOC] Rule for where: promotion of the two value branches (condition ignored).

    @staticmethod
    def index_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for index_expr: for int32/int64 requests on a kernel with an index dtype, use the
        # kernel's index dtype; otherwise the compute dtype of the request.
        # `index_expr` is for indexing-style uses: the kernel's chosen
        # indexing dtype wins over the requested int dtype. For uses that
        # must honor `dtype` (e.g. user-typed `arange(int64)` whose result
        # participates in tensor computation), `value_expr` is the right op
        # - see `convert_index_expr_to_value_expr` which rewrites accordingly.
        if dtype not in (torch.int32, torch.int64) or not hasattr(
            V.kernel, "index_dtype"
        ):
            return upcast_compute_type(dtype)

        return V.kernel.get_index_dtype_as_torch_dtype()

    @staticmethod
    def value_expr(expr: sympy.Expr, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for value_expr: always the requested dtype, with no compute-type upcast.
        # Unlike `index_expr`, `value_expr` always honors the requested
        # dtype because the result feeds tensor-value computation. The
        # sympy expression is emitted literally, so there is no
        # bf16/fp16 -> fp32 compute-type promotion here.
        return dtype

    @staticmethod
    def to_dtype(
        x: DTypeArg,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types=True,
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for to_dtype: the target dtype, upcast to compute dtype if use_compute_types.
        return upcast_compute_type(dtype) if use_compute_types else dtype

    @staticmethod
    def to_dtype_bitcast(
        x: DTypeArg, dtype: torch.dtype, src_dtype: torch.dtype
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for to_dtype_bitcast: compute dtype of the target dtype.
        return upcast_compute_type(dtype)

    @staticmethod
    def gelu(x: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for gelu: promoted dtype of the input.
        return promote_types([x])

    @staticmethod
    def mul(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for mul: promotion of both operands.
        return promote_types([a, b])

    @staticmethod
    def truediv(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for truediv: promotion of both operands (no int-to-float kind is used here).
        return promote_types([a, b])

    @staticmethod
    def div_rn(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for div_rn: promotion of both operands.
        return promote_types([a, b])

    @staticmethod
    def pow(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for pow: promotion of both operands, except float64 when the promoted dtype is an
        # integer, both operands are scalars, and the exponent is not known nonnegative (Python-style
        # negative exponent gives a float).
        dtype = promote_types([a, b])
        if (
            is_integer_dtype(dtype)
            and _is_scalar_dtype_arg(a)
            and _is_scalar_dtype_arg(b)
            and not _has_known_nonnegative_scalar_int_value(b)
        ):
            # Scalar integer pow follows Python semantics: negative exponents
            # produce a floating result, even though tensor integer pow stays
            # integral or errors out in separate lowering paths.
            return torch.float64
        return dtype

    @staticmethod
    def mod(a: DTypeArg, b: DTypeArg) -> torch.dtype:
        return promote_types([a, b])
        # [FUNCTION DOC] Rule for mod: promotion of both operands.

    @staticmethod
    def indirect_indexing(
        x: DTypeArg, size: int, check: bool = True, wrap_neg: bool = True
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for indirect_indexing: always int64.
        return torch.int64

    @staticmethod
    def randn(seed: int, offset: int) -> torch.dtype:
        return torch.float
        # [FUNCTION DOC] Rule for randn: float32.

    @staticmethod
    def rand(seed: int, offset: int) -> torch.dtype:
        return torch.float
        # [FUNCTION DOC] Rule for rand: float32.

    @staticmethod
    def rand_eager(seed, offset, threads_per_round, tid, vec) -> torch.dtype:
        return torch.float
        # [FUNCTION DOC] Rule for rand_eager: float32.

    @staticmethod
    def store_reduction(name: str, index, value: DTypeArg) -> None:
        return None
        # [FUNCTION DOC] Rule for store_reduction: produces no value, returns None.

    @staticmethod
    def reduction(
        dtype: torch.dtype, src_dtype: torch.dtype, reduction_type: str, value: DTypeArg
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for reduction: the requested output dtype.
        return dtype

    @staticmethod
    def store(name: str, index, value: DTypeArg, mode: str | None = None) -> None:
        return None
        # [FUNCTION DOC] Rule for store: produces no value, returns None.

    @staticmethod
    def partial_accumulate(
        name: str,
        reduction_type: str,
        value: DTypeArg,
        extra_meta: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] Rule for partial_accumulate: produces no value, returns None.
        return None

    @staticmethod
    def load(name: str, index) -> torch.dtype:
        # [FUNCTION DOC] Rule for load: compute dtype of the named buffer's dtype.
        return upcast_compute_type(V.graph.get_dtype(name))

    @staticmethod
    def floor(x: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for floor: promotion with integer-to-float kind.
        return promote_types(
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def ceil_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for ceil_to_int: the requested dtype.
        return dtype

    @staticmethod
    def int_truediv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for int_truediv: promotion of both operands with integer-to-float kind.
        return promote_types(
            [x, y], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def scan(
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]],
        values: tuple[T, ...],
    ) -> tuple[torch.dtype, ...]:
        # [FUNCTION DOC] Rule for scan: output dtypes equal the given per-output dtypes.
        return dtypes

    @staticmethod
    def fmod(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for fmod: promotion of both operands.
        return promote_types([x, y])

    @staticmethod
    def round_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for round_to_int: the requested dtype.
        return dtype

    @staticmethod
    def identity(x: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for identity: promoted dtype of the input.
        return promote_types([x])

    @staticmethod
    def frexp(x: DTypeArg) -> tuple[torch.dtype, torch.dtype]:
        # [FUNCTION DOC] Rule for frexp: tuple of (promoted input dtype, int32).
        # KNOWN ISSUE: marked TODO in code; multiple outputs are only partly handled.
        # TODO - need to handle multiple outputs
        return (promote_types([x]), torch.int32)

    @staticmethod
    def sort(
        dtypes: tuple[torch.dtype, ...],
        values: tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[torch.dtype, ...]:
        # [FUNCTION DOC] Rule for sort: output dtypes equal the given per-output dtypes.
        return dtypes

    @staticmethod
    def trunc(x: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for trunc: promoted dtype of the input.
        return promote_types([x])

    @staticmethod
    def bucketize(
        values: DTypeArg,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: DTypeArg,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> torch.dtype:
        # [FUNCTION DOC] Rule for bucketize: the requested indexing dtype.
        return indexing_dtype

    @staticmethod
    def rshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])
        # [FUNCTION DOC] Rule for rshift: promoted dtype of the left operand only (shift amount ignored).

    @staticmethod
    def round(x: DTypeArg) -> torch.dtype:
        return promote_types(
        # [FUNCTION DOC] Rule for round: promotion with integer-to-float kind.
            [x], type_promotion_kind=ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
        )

    @staticmethod
    def trunc_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        return dtype
        # [FUNCTION DOC] Rule for trunc_to_int: the requested dtype.

    @staticmethod
    def floor_to_int(x: DTypeArg, dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] Rule for floor_to_int: the requested dtype.
        return dtype

    @staticmethod
    def truncdiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for truncdiv: promotion of both operands.
        return promote_types([x, y])

    @staticmethod
    def floordiv(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for floordiv: promotion of both operands.
        return promote_types([x, y])

    @staticmethod
    def halide_clamp(value, size, check):
        # [FUNCTION DOC] Rule for halide_clamp: always int32 (backend rule registration is a TODO).
        # TODO - way of registering dtype for op in backend
        return torch.int32

    @staticmethod
    def dot(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        # [FUNCTION DOC] Rule for dot: float32 (matches the Triton default output dtype).
        # triton tl.dot out_dtype is tl.float32 by default.
        return torch.float32

    @staticmethod
    def inline_asm_elementwise(
        *inputs,
        asm,
        constraints=None,
        dtype=torch.float32,
        is_pure=True,
        pack=1,
        input_dtypes=None,
    ):
        # [FUNCTION DOC] Rule for inline_asm_elementwise: the requested output dtype.
        return dtype

    @staticmethod
    def lshift(x: DTypeArg, y: DTypeArg) -> torch.dtype:
        return promote_types([x])
        # [FUNCTION DOC] Rule for lshift: promoted dtype of the left operand only.

    @staticmethod
    def check_bounds(
        expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Rule for check_bounds: produces no value, returns None.
        return None

    def output(self, *args: DTypeArg) -> None:
        # [FUNCTION DOC] Always raises an error: output ops must not reach dtype propagation.
        raise AssertionError(
            f"{type(self).__name__}: ops.output should not appear here"
        )

    def placeholder(self, index: int) -> torch.dtype:
        # [FUNCTION DOC] Always raises an error: placeholder ops must not reach dtype propagation.
        raise AssertionError(
            f"{type(self).__name__}: ops.placeholder should not appear here"
        )

    @staticmethod
    def device_assert_async(cond, msg: str) -> None:
        # [FUNCTION DOC] Rule for device_assert_async: produces no value, returns None.
        return None


if TYPE_CHECKING:
    # pyrefly: ignore [inconsistent-inheritance]
    class _typecheck_DtypePropagation(DtypePropagationOpsHandler, OpsHandler[Any]):
        # [CLASS DOC] HELPER: Type-check-only class combining the handler with OpsHandler so signature
        # mismatches are reported by the type checker.
        pass  # mypy will error if we got any of the signatures wrong
