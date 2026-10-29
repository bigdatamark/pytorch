# [MODULE DOC] Defines the Inductor ops handler protocol: OpsHandler lists all scalar ops on virtualized ops, with OP_NAMES, plus
# type aliases (AtomicMode, StoreMode, ReductionType). Also provides generic handlers: DefaultHandler (single hook),
# NoopHandler, MockHandler (string rendering), KernelFormatterHandler, WrapperHandler, AddParenHandler, OpCounterCSE,
# ExtractConstantsHandler and SimpleCSEHandler. Import-time side effect: DefaultHandler._init_cls generates and installs methods.
# mypy: allow-untyped-defs
from __future__ import annotations

import inspect
import itertools
import re
import warnings
from io import StringIO
from typing import Any, Generic, Literal, NamedTuple, TYPE_CHECKING, TypeVar
from unittest.mock import patch

import sympy

import torch
import torch.utils._pytree as pytree

from ..utils._ordered_set import OrderedSet
from .utils import IndentedBuffer, reduction_num_outputs, sympy_index_symbol, sympy_str


if TYPE_CHECKING:
    from collections.abc import Callable


T = TypeVar("T")
AtomicMode = Literal[
    "atomic_add",
    "atomic_max",
    "atomic_min",
    "atomic_and",
    "atomic_or",
    "atomic_xor",
    "atomic_cas",
    "atomic_xchg",
]
StoreMode = AtomicMode | Literal["tma"] | None
ReductionType = Literal[
    "argmax",
    "argmin",
    "argmax_value",
    "argmin_value",
    "argmax_with_value",
    "argmin_with_value",
    "welford_reduce",
    "welford_combine",
    "any",
    "fmax",
    "max",
    "min",
    "prod",
    "sum",
    "dot",
    "xor_sum",
    "online_softmax_reduce",
]


def _arg_str(a: object) -> str:
    # [FUNCTION DOC] HELPER: Format an op argument as a string: sympy expressions via sympy_str, everything else via str.
    if isinstance(a, sympy.Expr):
        return sympy_str(a)
    return str(a)


# See OpDecompositions for superclass that desugars operations like reciprocal/square.
class OpsHandler(Generic[T]):
    # [CLASS DOC] Protocol class listing every op available on virtualized ops, parameterized by the value domain T.
    # Methods are abstract and raise NotImplementedError; concrete handlers (codegen, analysis, mock) override them.
    # Ops do not promote dtypes. Covers constants, random, conversion, memory, reduction, math, comparison and special functions.
    """
    Protocol describing the set of valid operations on ``torch._inductor.virtualized.ops``,
    as well as the contract for op handlers.  The type T signifies the domain
    of the abstract analysis AKA what all the functions return / take as arguments
    anywhere compute occurs.

    While these operators are typically dtype polymorphic (e.g., you can use mul
    on both integers and floats), they do NOT do promotion and usually return the
    same dtype as the input.  You are expected to have handled type promotion
    during ATen decompositions.  Most operators correspond exactly to pointwise
    operations as defined by torch, so when in doubt about semantics, check the
    corresponding torch documentation.  These are all scalar operations (so they
    are defined to operate on a single element at a time.)

    For convenience, many operators take a src_dtype which indicates what the dtype
    of the input argument is.  Although in principle this can be derived by an
    analysis, providing this for ops where it is useful helps avoid having to repeatedly
    recompute dtype in code generation.

    Note that this often describes a class of static methods, for stateless
    ops handlers.

    Handlers are often defined using metaprogramming (e.g. _initialize_pointwise_overrides),
    which means you will not get type errors for those methods.  We have tests in
    test/inductor/test_op_completeness.py which check that all operators are implemented after
    all the metaprogramming has run.
    """

    def constant(self, value: bool | float | int, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Produce a scalar constant of the given value and dtype.
        """Produces a scalar constant of type dtype."""
        raise NotImplementedError

    def load_seed(self, name: str, offset: T) -> T:
        # [FUNCTION DOC] Abstract op: look up the random seed buffer named name at an offset (inductor_prims.lookup_seed).
        """Computes inductor_prims.lookup_seed."""
        raise NotImplementedError

    def rand(self, seed: T, offset: T) -> T:
        # [FUNCTION DOC] Abstract op: uniform random float from a seed and int32 offset (mode rand).
        """Computes inductor_prims.random with mode="rand".  offset has dtype int32."""
        raise NotImplementedError

    def rand_eager(
        self, seed: T, base_offset: T, threads_per_round: T, tid: T, vec: T
    ) -> T:
        # [FUNCTION DOC] Abstract op: eager-compatible uniform random generation from seed, base offset, threads-per-round, thread id and vector width.
        """Computes inductor_prims.random with mode="rand_eager".  offset has dtype int32."""
        raise NotImplementedError

    def randn(self, seed: T, offset: T) -> T:
        # [FUNCTION DOC] Abstract op: normal random float from a seed and int32 offset.
        """Computes inductor_prims.random with mode="randn".  offset has dtype int32."""
        raise NotImplementedError

    def randint64(self, seed: T, offset: T, low: T, high: T) -> T:
        # [FUNCTION DOC] Abstract op: random int64 in [low, high) from a seed and int32 offset.
        """Computes inductor_prims.randint.  offset has dtype int32."""
        raise NotImplementedError

    def masked(self, mask: T, body: Callable[[], T], other: T) -> T:
        # [FUNCTION DOC] Abstract op: evaluate the body callable only where mask is true (guarding loads/stores);
        # where false the result is the constant other. Unlike where, the body is not computed unconditionally.
        """
        Computes body, but only perform loads/stores if the boolean mask
        evaluates to true.  For example, you would use this if you needed to
        perform an indirect load that may not be valid on some elements;
        without masking, invalid accesses can cause IMAs.  When mask is true,
        the result is the result of body; otherwise it is other. Here, `other`
        needs to be a constant.

        Contrast this with ops.where, which can multiplex between two values
        that have been unconditionally computed.
        """
        raise NotImplementedError

    def where(self, condition: T, input: T, other: T) -> T:
        # [FUNCTION DOC] Abstract op: select input where condition is true, otherwise other (torch.where).
        """
        Computes torch.where: when condition is true, return input; otherwise return other.
        """
        raise NotImplementedError

    def index_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: turn a sympy expression into a scalar usable for indexing. The kernel picks the compute dtype;
        # the dtype argument is advisory only. Use value_expr when the dtype must be honored.
        """
        Converts a sympy expression into a scalar suitable for indexing memory.
        The kernel decides the actual computation dtype (typically int32) — the
        ``dtype`` argument is advisory and is not honored when it would conflict
        with the kernel's indexing dtype. For values that must respect the
        requested dtype, use ``value_expr`` instead.
        """
        raise NotImplementedError

    def value_expr(self, expr: sympy.Expr, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: turn a sympy expression into a scalar of exactly the requested dtype for value computation.
        """
        Converts a sympy expression into a scalar of type ``dtype`` that
        participates in tensor value computation. Unlike ``index_expr``, the
        result dtype is respected, so this is the right op when the user
        explicitly requested the dtype (e.g. ``arange(dtype=torch.int64)``
        whose result is added to a tensor rather than used as an index).
        """
        raise NotImplementedError

    def to_dtype(
        self,
        x: T,
        dtype: torch.dtype,
        src_dtype: torch.dtype | None = None,
        use_compute_types: bool = True,
    ) -> T:
        # [FUNCTION DOC] Abstract op: convert x to dtype; optional src_dtype describes the original dtype and use_compute_types controls
        # whether backend compute types are used.
        """
        Convert x to dtype.  src_dtype can be optionally set to specify what the original
        dtype of x was, which can improve code generation (used by torch to(dtype=dtype)).
        """
        raise NotImplementedError

    def trunc_to_int(self, x: T, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: convert x to an integer dtype with truncation toward zero.
        """
        Convert x to dtype with truncation semantics (similar to how the int
        constructor works in Python).  In Inductor codegen, this just decays
        to trunc and then to_dtype, but this composite operation helps
        roundtrips for Sympy evaluation.

        dtype is taken as an explicit parameter because the desired output
        dtype is typically the index dtype, which may vary between int32 and
        int64 depending on if we've shown that all the indexing operations can
        be done in int32.
        """
        raise NotImplementedError

    def ceil_to_int(self, x: T, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: convert x to an integer dtype with ceiling semantics.
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def floor_to_int(self, x: T, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: convert x to an integer dtype with floor semantics.
        # KNOWN ISSUE: the docstring says ceiling semantics (copy-paste from ceil_to_int); the name indicates floor.
        """
        Convert x to dtype with ceiling semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def round_to_int(self, x: T, dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: convert x to an integer dtype with round-half-to-even semantics.
        """
        Convert x to dtype with round-to-even semantics.  See also trunc_to_int.
        """
        raise NotImplementedError

    def to_dtype_bitcast(self, x: T, dtype: torch.dtype, src_dtype: torch.dtype) -> T:
        # [FUNCTION DOC] Abstract op: reinterpret the bits of x (of type src_dtype) as dtype.
        """
        Reinterpret cast x to dtype (reinterpreting the bits in memory as another dtype.)
        src_dtype must be the original type of x.
        """
        raise NotImplementedError

    def identity(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op: return x unchanged; used to force a CSE boundary.
        """
        Returns x as is.  This is used to trigger CSE.
        """
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operations are only available in a "kernel" context.  Check
    # torch._inductor.codegen.common.CSEProxy for their typical implementation
    # in op handler (routing to their respective implementations in the kernel
    # handler)
    #
    # Importantly, inside a kernel, indexing and mask variables are available
    # in scope, which are typically used by sympy.Expr indexing.

    def indirect_indexing(
        self, x: T, size: sympy.Expr, check: bool = True, wrap_neg=True
    ) -> sympy.Expr:
        # [FUNCTION DOC] Abstract op: convert an integer value x into a sympy expression usable in indexing; size is the upper bound of valid
        # indices, check requests a bounds check and wrap_neg requests negative index wrapping.
        # Must return a valid sympy expression in any implementation.
        """
        Convert an integral x into a sympy.Expr that can be subsequently used in
        indexing computation.  'size' represents an upper bound on what valid
        indexes can be; when 'check' is True, we check that the x is in bounds.

        NB: This is typically mandatory to implement for any analysis, because you
        MUST return a valid sympy.Expr of some sort (even if it's a meaningless symbol).
        """
        raise NotImplementedError

    def load(self, name: str, index: sympy.Expr) -> T:
        # [FUNCTION DOC] Abstract op: load from buffer name at the offset given by a sympy index expression.
        """
        Load from the memory location 'name', offset by some indexing expression 'index'.
        """
        raise NotImplementedError

    def store(
        self,
        name: str,
        index: sympy.Expr,
        value: T,
        mode: StoreMode = None,
    ) -> None:
        # [FUNCTION DOC] Abstract op: store value to buffer name at the given index; mode may request an atomic or tma store.
        """
        Store 'value' to the memory location 'name' offset by 'expr'.  If
        specified, 'mode' can require the store to be an atomic addition.
        """
        raise NotImplementedError

    # TODO: Better explain how the "collective" semantics of these ops;
    # remember that the input value is a scalar, you can't reduce on it in the
    # traditional sense!
    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: T,
    ) -> T | tuple[T, ...]:
        # [FUNCTION DOC] Abstract op: reduce value (of src_dtype) with reduction_type, accumulating in dtype. Returns an intermediate that
        # must be written with store_reduction; welford types return several outputs.
        """
        Perform a 'reduction_type' reduction on 'value' of dtype 'src_dtype',
        using 'dtype' as the accumulation dtype for the reduction.  The result
        is an intermediate computation which should be stored to the final
        location using 'ops.store_reduction'.

        Valid reduction types are .  For Welford reduction types, this
        function returns multiple outputs; consult reduction_num_outputs to
        determine the amount in metaprogramming applications.
        """
        raise NotImplementedError

    # TODO: in practice, this seems to actually return None, but not returning
    # a T makes common __getattr__ idioms not type correctly.  Figure out if
    # this should be returning something.
    def store_reduction(self, name: str, index: sympy.Expr, value: T) -> None:
        # [FUNCTION DOC] Abstract op: store the fully accumulated reduction result to buffer name at the index. Returns None in practice.
        """
        Store the fully accumulated result of 'reduction' to the memory
        location 'name' offset by 'expr'.
        """
        raise NotImplementedError

    def scan(
        self,
        dtypes: tuple[torch.dtype, ...],
        combine_fn: Callable[[tuple[T, ...], tuple[T, ...]], tuple[T, ...]],
        values: tuple[T, ...],
    ) -> tuple[T, ...]:
        # [FUNCTION DOC] Abstract op: associative scan over values using combine_fn; returns a tuple of results, one per value.
        """
        Perform an associative scan on 'value'.
        """
        # TODO: Improve the description with some pseudocode
        raise NotImplementedError

    def sort(
        self,
        dtypes: tuple[torch.dtype, ...],
        values: tuple[T, ...],
        stable: bool,
        descending: bool,
    ) -> tuple[T, ...]:
        # [FUNCTION DOC] Abstract op: sort values along the reduction dimension (stable and descending flags); returns a tuple of sorted values.
        """
        Sort values along the reduction dimension.
        """
        raise NotImplementedError

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> T:
        # [FUNCTION DOC] Abstract op: find bucket indices of values within sorted boundaries (optionally via a sorter), right selects the
        # right-closed variant; result has indexing_dtype.
        # See [Note: Inductor bucketize op]
        raise NotImplementedError

    def partial_accumulate(
        self,
        name: str,
        reduction_type: ReductionType,
        value: T,
        extra_meta: dict[str, Any],
    ) -> None:
        # [FUNCTION DOC] Abstract op: accumulate value into a partial reduction buffer named name (extra_meta carries backend metadata).
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The following ops have semantics that correspond exactly to the torch
    # operation with the same corresponding name.

    def abs(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op abs: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def exp(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op exp: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def exp2(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op exp2: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def expm1(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op expm1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sqrt(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op sqrt: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def relu(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op relu: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def minimum(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op minimum: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def maximum(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op maximum: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def fmaximum(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op fmaximum: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def cos(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op cos: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sin(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op sin: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def lgamma(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op lgamma: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def erf(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op erf: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def cosh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op cosh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sinh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op sinh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def acos(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op acos: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def acosh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op acosh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def asin(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op asin: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def asinh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op asinh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def atan2(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op atan2: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def atan(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op atan: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def atanh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op atanh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def copysign(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op copysign: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def erfc(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op erfc: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def erfinv(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op erfinv: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def frexp(self, x0: T):
        # [FUNCTION DOC] Abstract op frexp: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def hypot(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op hypot: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def log10(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op log10: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def log2(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op log2: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def ldexp(self, x0: T, n: T) -> T:
        # [FUNCTION DOC] Abstract op ldexp: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def nextafter(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op nextafter: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def logical_and(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op logical_and: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def logical_not(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op logical_not: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def logical_or(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op logical_or: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def logical_xor(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op logical_xor: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_and(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_and: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_not(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_not: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_or(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_or: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_xor(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_xor: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_left_shift(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_left_shift: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bitwise_right_shift(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op bitwise_right_shift: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def rsqrt(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op rsqrt: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def log1p(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op log1p: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def tan(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op tan: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def tanh(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op tanh: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sigmoid(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op sigmoid: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def signbit(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op signbit: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def fmod(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op fmod: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def log(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op log: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def isinf(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op isinf: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def isnan(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op isnan: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    # This rounds half to even to break ties
    def round(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op round: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def floor(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op floor: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sign(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op sign: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def trunc(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op trunc: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def ceil(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op ceil: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def neg(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op neg: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def reciprocal(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op reciprocal: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def eq(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op eq: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def ne(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op ne: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def lt(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op lt: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def gt(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op gt: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def le(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op le: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def ge(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op ge: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def add(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op add: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def sub(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op sub: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def mul(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op mul: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # NB: this returns a float, like the torch operation
    def pow(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op pow: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def and_(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op and_: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def or_(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op or_: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def xor(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op xor: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # These are metaprogrammed by MockHandler._init_cls
    def lshift(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op lshift: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def rshift(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op rshift: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These are "special" operators.  These only exist if the target
    # language actually supports the operator.  Keep this in sync with
    # pointwise_overrides_data.

    def airy_ai(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op airy_ai: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bessel_j0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op bessel_j0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bessel_j1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op bessel_j1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bessel_y0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op bessel_y0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def bessel_y1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op bessel_y1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def digamma(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op digamma: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def erfcx(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op erfcx: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def fma(self, x: T, y: T, z: T) -> T:
        # [FUNCTION DOC] Abstract op fma: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def mul_rn(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op: multiplication with round-to-nearest that prevents fusion with later ops.
        """Multiplication with round-to-nearest, preventing fusion with subsequent ops."""
        raise NotImplementedError

    def igamma(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op igamma: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def igammac(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op igammac: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def gammainc(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op gammainc: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def gammaincc(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op gammaincc: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def i0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op i0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def i0e(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op i0e: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def i1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op i1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def i1e(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op i1e: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def log_ndtr(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op log_ndtr: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def modified_bessel_i0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op modified_bessel_i0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def modified_bessel_i1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op modified_bessel_i1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def modified_bessel_k0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op modified_bessel_k0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def modified_bessel_k1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op modified_bessel_k1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def ndtr(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op ndtr: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def ndtri(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op ndtri: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def polygamma(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op polygamma: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def scaled_modified_bessel_k0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op scaled_modified_bessel_k0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def scaled_modified_bessel_k1(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op scaled_modified_bessel_k1: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def spherical_bessel_j0(self, x: T) -> T:
        # [FUNCTION DOC] Abstract op spherical_bessel_j0: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def zeta(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op zeta: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def chebyshev_polynomial_t(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op chebyshev_polynomial_t: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def chebyshev_polynomial_u(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op chebyshev_polynomial_u: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def chebyshev_polynomial_v(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op chebyshev_polynomial_v: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def chebyshev_polynomial_w(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op chebyshev_polynomial_w: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def legendre_polynomial_p(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op legendre_polynomial_p: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def shifted_chebyshev_polynomial_t(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op shifted_chebyshev_polynomial_t: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def shifted_chebyshev_polynomial_u(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op shifted_chebyshev_polynomial_u: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def shifted_chebyshev_polynomial_v(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op shifted_chebyshev_polynomial_v: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def shifted_chebyshev_polynomial_w(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op shifted_chebyshev_polynomial_w: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def hermite_polynomial_h(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op hermite_polynomial_h: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def hermite_polynomial_he(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op hermite_polynomial_he: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def laguerre_polynomial_l(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op laguerre_polynomial_l: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # These operators are a bit special, because they are conventionally
    # natively supported in both Python and C, but the semantics differ so
    # care must be taken

    def truncdiv(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: C-style integer division rounding toward zero.
        """C-style trunc division between integers only.  Computes the true
        division of two numbers and rounds the result to zero.
        """
        raise NotImplementedError

    def floordiv(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: Python-style integer division rounding toward negative infinity.
        """Python-style floor division between integers only.  Computes the
        true division of two numbers and floors the result.  If you want
        floor division for floats, do regular truediv and floor the result.
        """
        raise NotImplementedError

    def truediv(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: true division of floats; integer inputs are not valid (use int_truediv).
        """True division between floats.  Integer inputs are NOT valid.  To
        do Python-style (int, int) -> float division, use int_truediv"""
        raise NotImplementedError

    def div_rn(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: division with IEEE round-to-nearest, matching eager CUDA semantics.
        """Division with round-to-nearest rounding mode.  Used for matching
        eager CUDA semantics where division uses IEEE round-to-nearest."""
        raise NotImplementedError

    def int_truediv(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: true division of integers computed in higher precision than promote-to-float division.
        """True division between integers.  This is NOT the same as promoting
        to float and doing integer division, there is a bespoke algorithm for
        doing the division in higher precision than the above.
        """
        raise NotImplementedError

    def mod(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: C-style modulus; the result takes the sign of the left operand.
        """C-style modulus, take sign from LHS (x0)."""
        raise NotImplementedError

    def remainder(self, x0: T, x1: T) -> T:
        # [FUNCTION DOC] Abstract op: Python-style modulus; the result takes the sign of the right operand.
        """Python-style modulus, take sign from RHS (x1)."""
        raise NotImplementedError

    def square(self, x0: T) -> T:
        # [FUNCTION DOC] Abstract op square: scalar operation matching the torch operation of the same name (see class doc); base raises NotImplementedError.
        raise NotImplementedError

    def check_bounds(
        self, expr: sympy.Expr, size: sympy.Expr, lower: bool, upper: bool
    ) -> None:
        # [FUNCTION DOC] Abstract op: check that expr lies within [0, size) (lower and upper flag which sides to check); returns nothing.
        raise NotImplementedError

    # halide-only
    def halide_clamp(self, value: T, size: sympy.Expr, check: bool) -> T:
        # [FUNCTION DOC] Abstract op (Halide only): clamp value to the bounds of size, with optional check.
        raise NotImplementedError

    # triton-only
    def dot(self, x: T, y: T) -> T:
        # [FUNCTION DOC] Abstract op (Triton only): dot product of x and y.
        raise NotImplementedError

    # triton-only
    def inline_asm_elementwise(
        self,
        *inputs: T,
        asm: str,
        constraints: str | None = None,
        dtype: torch.dtype = torch.float32,
        is_pure: bool = True,
        pack: int = 1,
        input_dtypes: tuple[torch.dtype, ...] | None = None,
    ) -> T:
        # [FUNCTION DOC] Abstract op (Triton only): elementwise inline assembly over inputs with constraints, dtype, purity, packing and input dtypes.
        raise NotImplementedError

    def output(self, *args: T) -> None:
        # [FUNCTION DOC] Fake op used in analysis, not codegen: marks the outputs of a graph.
        """This is a fake op used in analysis but not codegen"""
        raise NotImplementedError

    def placeholder(self, index: int) -> T:
        # [FUNCTION DOC] Fake op used in analysis, not codegen: reference to graph input number index.
        """This is a fake op used in analysis but not codegen"""
        raise NotImplementedError

    def device_assert_async(self, cond: T, msg: str) -> T:
        # [FUNCTION DOC] Abstract op: asynchronous device-side assertion that cond holds, with message msg.
        raise NotImplementedError


_ignore_op_re = re.compile(r"_.*|paren").fullmatch


def list_ops(cls: type[Any]):
    # [FUNCTION DOC] Return an ordered set of the public op names of a handler class: all attribute names except those
    # starting with an underscore or equal to paren.
    return OrderedSet([x for x in dir(cls) if not _ignore_op_re(x)])


OP_NAMES = list_ops(OpsHandler)


class DefaultHandler(OpsHandler[Any]):
    # [CLASS DOC] OpsHandler whose every op forwards to a single _default hook. _init_cls (run at import time after the class) generates
    # forwarding methods for all ops in OP_NAMES. Unknown ops go through __getattr__ with a warning.
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Hook run for any op without a specific implementation. Receives op name, positional args and keyword args.
        # Base version raises NotImplementedError; subclasses override.
        """
        Default implementation for all ops.  Override in a subclass to
        provide generic op behavior.

        Args:
            name: name of the op, see OpHandler.{name}
            args: positional args passed to the op
            kwargs: keyword args passed to the op

        Returns:
            return value of the op

        """
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        # [FUNCTION DOC] Called for ops not defined on the class. Emits a warning about an undefined op schema and returns a fallback
        # function that forwards the call to _default. Kept for the MTIA backend.
        def fallback(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Forward any call to _default with the captured op name.
            return self._default(name, args, kwargs)

        # would like to remove this function entirely, but it's used in MTIA backend
        warnings.warn(f"undefined OpHandler.{name}, please add missing op schema")
        return fallback

    @staticmethod
    def _call_default(target: str):
        # [FUNCTION DOC] Static helper: build a method for op target that forwards its arguments to _default; used for ops with
        # defaults or variadic arguments. Names the function after the target.
        def call_default(self, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Forward the call to _default with the captured op name.
            return self._default(target, args, kwargs)

        call_default.__name__ = target
        return call_default

    @classmethod
    def _init_cls(cls):
        # [FUNCTION DOC] Class setup: generate source code for a forwarding method per op in OP_NAMES whose signature has only plain
        # positional parameters without defaults (faster than varargs), compile it, and install all as class attributes.
        # Other ops get the slower generic forwarding method. Raises an error if a signature lacks a leading self.
        # Side effect: executes generated code and mutates the class.
        """
        Here we codegen many functions of the form:

            def add(self, a, b):
                return self._default('add', (a, b), {})

        and install them in cls.  This is the same as _call_default above,
        but is about 1.2x faster since CPython varargs parsing is slow.
        """
        code = StringIO()
        for target in OP_NAMES:
            sig = inspect.signature(getattr(OpsHandler, target))
            if all(
                p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and p.default is inspect.Parameter.empty
                for p in sig.parameters.values()
            ):
                self_arg, *args = sig.parameters.keys()
                if self_arg != "self":
                    raise AssertionError(
                        f"expected first parameter 'self', got {self_arg!r}"
                    )
                code.write(
                    f"""
                    def {target}(self, {", ".join(args)}):
                        return self._default({target!r}, ({", ".join(args)}, ), {{}})
                    """.strip()
                )
                code.write("\n\n")
            else:
                # slower fallback for ops with default or variadic arguments
                setattr(cls, target, cls._call_default(target))

        ctx: dict[str, Any] = {}
        exec(code.getvalue(), ctx)
        for target, impl in ctx.items():
            if target in OP_NAMES:
                setattr(cls, target, impl)


DefaultHandler._init_cls()


class NoopHandler(DefaultHandler):
    # [CLASS DOC] DefaultHandler that evaluates nothing: every op returns None (a few return structured placeholders).
    name = "NoopHandler"

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Ignore the op and return None.
        return None

    @staticmethod
    def masked(mask, body, other) -> None:
        # [FUNCTION DOC] Return None without evaluating the body.
        return None

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x) -> tuple[None, None]:
        # [FUNCTION DOC] Return a pair of None values.
        return (None, None)

    @staticmethod
    def scan(dtypes, combine_fn, values) -> tuple[None, ...]:
        # [FUNCTION DOC] Return a tuple of None, one per scanned value.
        return (None,) * len(values)

    @staticmethod
    def sort(dtypes, values, stable, descending) -> tuple[None, ...]:
        # [FUNCTION DOC] Return a tuple of None, one per sorted value.
        return (None,) * len(values)

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(index_var, size, check=True, wrap_neg=True) -> sympy.Symbol:
        # [FUNCTION DOC] Return the sympy zero constant regardless of arguments.
        return sympy.S.Zero


class BasicMathOpsMixin:
    # [CLASS DOC] Mixin of static methods giving string renderings of basic arithmetic, bitwise and comparison ops in infix form.
    @staticmethod
    def add(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation + on the two operand strings.
        return f"{a} + {b}"

    @staticmethod
    def sub(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation - on the two operand strings.
        return f"{a} - {b}"

    @staticmethod
    def mul(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation * on the two operand strings.
        return f"{a} * {b}"

    @staticmethod
    def floordiv(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation // on the two operand strings.
        return f"{a} // {b}"

    @staticmethod
    def truediv(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation / on the two operand strings.
        return f"{a} / {b}"

    @staticmethod
    def mod(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation % on the two operand strings.
        # Target semantics of modulus may vary by backend.
        # careful, depending on target semantics varies
        return f"{a} % {b}"

    @staticmethod
    def pow(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation ** on the two operand strings.
        return f"{a} ** {b}"

    @staticmethod
    def lshift(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation << on the two operand strings.
        return f"{a} << {b}"

    @staticmethod
    def rshift(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation >> on the two operand strings.
        return f"{a} >> {b}"

    @staticmethod
    def and_(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation & on the two operand strings.
        return f"{a} & {b}"

    @staticmethod
    def or_(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation | on the two operand strings.
        return f"{a} | {b}"

    @staticmethod
    def xor(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation ^ on the two operand strings.
        return f"{a} ^ {b}"

    @staticmethod
    def eq(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation == on the two operand strings.
        return f"{a} == {b}"

    @staticmethod
    def ne(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation != on the two operand strings.
        return f"{a} != {b}"

    @staticmethod
    def lt(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation < on the two operand strings.
        return f"{a} < {b}"

    @staticmethod
    def gt(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation > on the two operand strings.
        return f"{a} > {b}"

    @staticmethod
    def le(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation <= on the two operand strings.
        return f"{a} <= {b}"

    @staticmethod
    def ge(a, b):
        # [FUNCTION DOC] Return the string rendering of the infix operation >= on the two operand strings.
        return f"{a} >= {b}"

    @staticmethod
    def neg(a):
        # [FUNCTION DOC] Return the string rendering of the negation of the operand string.
        return f"-{a}"


class MockHandler(BasicMathOpsMixin, DefaultHandler):
    # [CLASS DOC] Handler that renders each op as readable text (infix for basic math, ops.name(...) otherwise); used for keys and debugging.
    name = "MockHandler"

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Render the op call as a string of the form ops.name(arg, ...) with keyword arguments appended as key=value.
        fargs = [*map(_arg_str, args)]
        for k, v in kwargs.items():
            fargs.append(f"{k}={_arg_str(v)}")
        return f"ops.{name}({', '.join(fargs)})"

    @staticmethod
    def masked(mask, body, other) -> str:
        # [FUNCTION DOC] Render a masked op string, calling the body to obtain its text.
        return f"ops.masked({mask}, {body()}, {other})"

    @staticmethod
    # pyrefly: ignore [bad-override]
    def frexp(x):
        # [FUNCTION DOC] Return two strings for the mantissa and exponent parts of an ops.frexp call.
        return (f"ops.frexp({x})[0]", f"ops.frexp({x})[1]")

    @staticmethod
    def scan(dtypes, combine_fn, values):
        # [FUNCTION DOC] Return a tuple of strings indexing the rendered scan call, one per value.
        return tuple(
            f"ops.scan({dtypes}, {combine_fn}, {values})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    def sort(dtypes, values, stable, descending):
        # [FUNCTION DOC] Return a tuple of strings indexing the rendered sort call, one per value.
        return tuple(
            f"ops.sort({dtypes}, {values}, stable={stable}, descending={descending})[{i}]"
            for i in range(len(values))
        )

    @staticmethod
    # pyrefly: ignore [bad-override]
    def indirect_indexing(index_var, size, check=True, wrap_neg=True) -> sympy.Symbol:
        # [FUNCTION DOC] Return a sympy index symbol named after the string form of the index variable.
        return sympy_index_symbol(str(index_var))


class KernelFormatterHandler(DefaultHandler):
    # [CLASS DOC] Handler that formats a computation as source for a function body, assigning each op result to a temp variable.
    # State: parent handler used for rendering, an indented output buffer and a temp counter.
    def __init__(self, parent_handler: OpsHandler[Any]):
        # [FUNCTION DOC] Store the parent handler used to render ops, an indented output buffer (indent level 1) and a temp variable counter.
        self.parent_handler = parent_handler
        self._output = IndentedBuffer(1)
        self.var_counter = itertools.count()

    @staticmethod
    def ir_to_string(ir_fn, index, rindex=None) -> str:
        # [FUNCTION DOC] Static: render an IR inner function as source text. Builds a def line for index (and rindex) arguments, unpacks
        # symbol tuples (integer entries become underscores), then runs ir_fn with the formatter installed as ops handler and
        # FlexibleLayout indexing allowed. Returns the text. Side effects: temporarily sets the virtualized ops handler
        # and patches a class attribute.
        from .ir import FlexibleLayout
        from .virtualized import V

        args = [index, rindex] if rindex is not None else [index]
        names = ["index", "rindex"] if rindex is not None else ["index"]
        formatter = KernelFormatterHandler(MockHandler())

        with formatter._output.indent(-1):
            formatter._output.writeline(f"def inner_fn({', '.join(names)}):")
        for name, arg in zip(names, args):
            if arg:
                lhs = ", ".join(
                    [
                        str("_" if isinstance(v, (int, sympy.Integer)) else v)
                        for v in arg
                    ]
                )
                formatter._output.writeline(f"{lhs} = {name}")

        with (
            V.set_ops_handler(formatter),
            patch.object(FlexibleLayout, "allow_indexing", True),
        ):
            result = ir_fn(*args)
            return formatter.getvalue(result)

    def indirect_indexing(self, *args, **kwargs) -> sympy.Symbol:
        # [FUNCTION DOC] Delegate to the parent handler.
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def _write(self, line):
        # [FUNCTION DOC] HELPER: Emit an assignment of the line to a fresh temp variable (tmpN) and return the variable name.
        # replace line with a new variable name
        varname = f"tmp{next(self.var_counter)}"
        self._output.writeline(f"{varname} = {line}")
        return varname

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Call the op on the parent handler and assign each resulting string to a new temp variable, returning the variable names
        # in the same structure.
        return pytree.tree_map(
            self._write, getattr(self.parent_handler, name)(*args, **kwargs)
        )

    def reduction(
        self,
        dtype: torch.dtype,
        src_dtype: torch.dtype,
        reduction_type: ReductionType,
        value: str | tuple[str, ...],
    ) -> str | tuple[str, ...]:
        # [FUNCTION DOC] Call the parent reduction and emit one assignment binding the result to as many temp variables as the reduction
        # type produces outputs. Returns one name or a tuple of names.
        line = self.parent_handler.reduction(dtype, src_dtype, reduction_type, value)
        num_values = reduction_num_outputs(reduction_type)
        varnames = [f"tmp{next(self.var_counter)}" for _ in range(num_values)]
        self._output.writeline(f"{','.join(varnames)} = {line}")
        return tuple(varnames) if num_values > 1 else varnames[0]

    def getvalue(self, result):
        # [FUNCTION DOC] Emit a return statement for the result and return the accumulated source text.
        self._output.writeline(f"return {result}")
        return self._output.getvalue()


class WrapperHandler(DefaultHandler):
    # [CLASS DOC] DefaultHandler that forwards every op to a wrapped inner handler.
    def __init__(self, inner: OpsHandler[Any]):
        # [FUNCTION DOC] Store the wrapped inner handler.
        self._inner = inner

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Forward the op to the inner handler and return its result.
        return getattr(self._inner, name)(*args, **kwargs)


class AddParenHandler(WrapperHandler):
    # [CLASS DOC] WrapperHandler that parenthesizes string results of ops.
    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Forward the op to the inner handler; wrap string results in parentheses, but pass through falsy values,
        # sympy expressions, tuples and lists unchanged.
        val = getattr(self._inner, name)(*args, **kwargs)
        if not val or isinstance(val, (sympy.Expr, tuple, list)):
            return val
        return f"({val})"


class OpCountResult(NamedTuple):
    # [CLASS DOC] Named tuple of counting results: number of ops, set of used op names, read buffer names, nontrivial read count.
    num_ops: int
    used_ops: OrderedSet[str]
    read_buffers: list[str]
    nontrivial_read_count: int


class OpCounterCSE(DefaultHandler):
    # [CLASS DOC] Shim handler that counts distinct ops (value-numbering results into temp names) and tracks used ops and buffer reads.
    """Shim to count how many ops are used"""

    def __init__(self, inner: OpsHandler[Any]):
        # [FUNCTION DOC] Wrap an inner handler and initialize counters: op count, value-to-name map, used op names, read buffer names and nontrivial read count.
        super().__init__()
        self.parent_handler = inner
        self.op_count = 0
        self.var_names: dict[str, str] = {}
        self._used_ops: OrderedSet[str] = OrderedSet()
        self._read_names: list[str] = []
        self._nontrivial_read_count = 0

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Record the op name as used, call the inner handler, and map result values to temp names (counting new values).
        self._used_ops.add(name)
        return pytree.tree_map(
            self._update_count, getattr(self.parent_handler, name)(*args, **kwargs)
        )

    def _update_count(self, val):
        # [FUNCTION DOC] HELPER: Return the temp name for a value, allocating a new name and incrementing the op count if the value is new.
        varname = self.var_names.get(val)
        if not varname:
            varname = f"tmp{self.op_count}"
            self.op_count += 1
            self.var_names[val] = varname
        return varname

    def indirect_indexing(self, *args, **kwargs):
        # [FUNCTION DOC] Record indirect_indexing as used and delegate to the parent handler.
        self._used_ops.add("indirect_indexing")
        return self.parent_handler.indirect_indexing(*args, **kwargs)

    def load(self, name: str, index: sympy.Expr) -> str:
        # [FUNCTION DOC] Delegate the load; if the value is new, record load as used, record the buffer name, and count a nontrivial read when
        # the index is not an integer constant. Returns the temp name.
        val = self.parent_handler.load(name, index)
        if val not in self.var_names:
            self._used_ops.add("load")
            self._read_names.append(name)
            if not isinstance(index, (sympy.Integer, int)):
                self._nontrivial_read_count += 1
        return self._update_count(val)

    def load_seed(self, name: str, offset: T):
        # [FUNCTION DOC] Delegate the seed load; if the value is new, record load_seed as used and the buffer name. Returns the temp name.
        val = self.parent_handler.load_seed(name, offset)
        if val not in self.var_names:
            self._used_ops.add("load_seed")
            self._read_names.append(name)
        return self._update_count(val)

    def bucketize(
        self,
        values: T,
        boundaries: tuple[str, sympy.Expr, sympy.Expr, sympy.Expr],
        boundary_indices: T,
        indexing_dtype: torch.dtype,
        right: bool,
        sorter: tuple[str, sympy.Expr] | None = None,
        sorter_indices: T | None = None,
    ) -> T:
        # [FUNCTION DOC] Delegate the bucketize; if the value is new, record bucketize as used and the boundaries (and sorter) buffer names
        # as reads. Returns the temp name.
        """
        See [Note: Inductor bucketize op]
        """
        val = self.parent_handler.bucketize(
            values,
            boundaries,
            boundary_indices,
            indexing_dtype,
            right,
            sorter,
            sorter_indices,
        )
        if val not in self.var_names:
            self._used_ops.add("bucketize")
            self._read_names.append(boundaries[0])
            if sorter is not None:
                self._read_names.append(sorter[0])
        return self._update_count(val)

    def getvalue(self):
        # [FUNCTION DOC] Return an OpCountResult with op count, used ops, read buffer names and nontrivial read count.
        return OpCountResult(
            self.op_count, self._used_ops, self._read_names, self._nontrivial_read_count
        )


class ExtractConstantsHandler(NoopHandler):
    # [CLASS DOC] NoopHandler that only builds IR Constant nodes for constant ops; all other ops yield None.
    def __init__(self, device: torch.device | None):
        # [FUNCTION DOC] Store the device used for created constants (may be None).
        self.device = device

    def constant(self, value: Any, dtype: torch.dtype) -> torch._inductor.ir.Constant:
        # [FUNCTION DOC] Return an IR Constant node for the value and dtype on the stored device, or the default device if none.
        from torch._inductor import ir

        return ir.Constant(
            value=value, dtype=dtype, device=self.device or torch.get_default_device()
        )


class SimpleCSEHandler(WrapperHandler):
    # [CLASS DOC] WrapperHandler that adds common subexpression elimination keyed by mock-rendered op calls.
    # Does not support stores (no load cache invalidation).
    """Wraps the underlying handler with a CSE pass

    NOTE: Compared to codegen level CSE this is simplified as it
    doesn't support stores which require load cache invalidation.
    """

    def __init__(self, inner: Any):
        # [FUNCTION DOC] Wrap the inner handler with an empty cache mapping and a MockHandler used to build cache keys.
        super().__init__(inner)
        self.cse_cache: dict[str, Any | tuple[Any, ...]] = {}
        self.mock = MockHandler()

    def indirect_indexing(self, *args, **kwargs) -> sympy.Expr:
        # [FUNCTION DOC] Delegate to the wrapped handler.
        return super().indirect_indexing(*args, **kwargs)  # type: ignore[misc]

    def store(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Not supported: always raises NotImplementedError.
        raise NotImplementedError("store not implemented")

    def store_reduction(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Not supported: always raises NotImplementedError (message says store).
        raise NotImplementedError("store not implemented")

    def _default(self, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        # [FUNCTION DOC] Build a string key for the op call via the mock handler; return the cached result if present, otherwise call the inner
        # handler, cache and return the result. Results that are None are never cache hits.
        key = getattr(self.mock, name)(*args, **kwargs)
        val = self.cse_cache.get(key)
        if val is not None:
            return val

        val = getattr(self._inner, name)(*args, **kwargs)
        self.cse_cache[key] = val
        return val

    def device_assert_async(self, *args, **kwargs) -> None:
        # [FUNCTION DOC] Not supported: always raises NotImplementedError saying CSEProxy should handle it.
        raise NotImplementedError(
            f"{type(self).__name__}: device_assert_async should be handled by CSEProxy"
        )
