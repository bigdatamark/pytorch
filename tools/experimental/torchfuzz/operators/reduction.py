# pyre-strict
# [MODULE DOC] Fuzzer operators for torch reduction and cumulative ops. Defines shape and stride
# helpers and dtype pools, then operator families: cumulative ops with a random dimension
# (cumprod, logcumsumexp, cummax, cummin), full or single-dimension reductions with optional
# keepdim (sum, mean, prod, nansum, var, all, any, amax, amin, argmax, argmin), dimension-based
# value-extracting reductions (max, min, median, mode), and special cases (count_nonzero,
# aminmax, var_mean). Operators stash random choices made while generating input specs and
# consume them in code generation. No import-time side effects besides imports.
"""Reduction Operator subclasses.

Models torch ops in the "Reduction Ops" category. Five operator
families:

  A) Cumulative single-tensor (cumprod) -- same shape as input,
     pick random dim. Mirrors upstream CumsumOperator.
  B) Cumulative tuple .values (cummax, cummin) -- same as A but
     codegen extracts .values.
  C) Full reduction, single tensor (sum, mean, prod, nansum, var,
     all, any, amax, amin, argmax, argmin) -- produces either a 0-D
     scalar (no-dim form) or a keepdim=True-shaped output (one
     reduced dim of size 1).
  D) Dim-required tuple .values (max, min, median, mode) -- always
     uses the 2-arg form with keepdim=True, codegen extracts .values.
  E) Standalone special-case ops:
       - CountNonzeroOperator (0-D int64 output, no keepdim support)
       - AminmaxOperator (0-D output, .min extraction)
       - VarMeanOperator (0-D float output, [0] extraction)

torch.cumsum reuses upstream's CumsumOperator and has no class in
this file.

All operators that stash per-call random choices follow the
defense-in-depth pattern: __init__ initialises stash to None,
fuzz_inputs_specs populates from the global random module,
codegen defensively asserts ``is not None`` (NOT truthiness -- falsy
values are valid stash contents), then snapshots-and-clears the
stash to local variables BEFORE building the output string.
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES, is_float_dtype
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_NUMERIC_INPUT_DTYPES_NO_BOOL: tuple[torch.dtype, ...] = (
    *FLOAT_DTYPES,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.uint16,
    torch.uint32,
    torch.uint64,
)

_BOOL_OR_NUMERIC_INPUT_DTYPES: tuple[torch.dtype, ...] = (
    torch.bool,
    *_NUMERIC_INPUT_DTYPES_NO_BOOL,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return row-major contiguous strides for a size tuple; empty size
    # gives an empty tuple. Pure.
    """Standard row-major contiguous strides for ``size``."""
    if not size:
        return ()
    strides = [1]
    for s in reversed(size[1:]):
        strides.append(strides[-1] * s)
    return tuple(reversed(strides))


def _random_input_shape() -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return a random shape with 1 to 4 dimensions, each of size 1 to 6.
    # Consumes global random state.
    """Pick a non-empty shape for full-reduction scalar inputs.

    ndim in [1, 4]; each dim in [1, 6]. Uses the global random module so
    --seed reproduces.
    """
    ndim = random.randint(1, 4)
    return tuple(random.randint(1, 6) for _ in range(ndim))


def _random_input_shape_min_numel_2() -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return a random shape of 1 to 4 dimensions, sizes 1 to 6, with at
    # least two elements in total; if all dimensions drew 1, one random dimension is redrawn from
    # 2 to 6. Consumes global random state.
    """Pick a non-empty input shape with total numel >= 2.

    Used by Bessel-corrected reductions (``torch.var``, ``torch.var_mean``)
    that divide by ``n - 1`` and return NaN at numel == 1, which would
    pollute numerics comparisons with spurious "NaN != NaN" mismatches.
    Bumps a random dim if every dim drew 1.
    """
    ndim = random.randint(1, 4)
    size = [random.randint(1, 6) for _ in range(ndim)]
    if all(s == 1 for s in size):
        idx = random.randrange(ndim)
        size[idx] = random.randint(2, 6)
    return tuple(size)


# ---------------------------------------------------------------------------
# Family A -- Cumulative single-tensor
# ---------------------------------------------------------------------------


class _CumulativeReductionBase(Operator):
    # [CLASS DOC] HELPER: Base for cumulative ops whose output shape equals the input shape. State:
    # the stashed dimension (None until input specs are generated, reset by code generation).
    # Subclasses provide the torch op name. Accepts non-bool numeric tensors with a dimension.
    """Single-tensor cumulative reduction with same shape as input.

    Picks a random dim at fuzz_inputs_specs time and emits
    ``torch.X(input, dim=k)``.
    """

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Register under the given name and clear the stashed dimension.
        super().__init__(name)
        self._dim: int | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs with at least one dimension and a non-bool
        # numeric dtype (float, signed or unsigned integer).
        return (
            isinstance(output_spec, TensorSpec)
            and len(output_spec.size) >= 1
            and output_spec.dtype in _NUMERIC_INPUT_DTYPES_NO_BOOL
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Pick a random dimension of the output, stash it, and return one input
        # spec identical in size, stride and dtype to the output. Raises AssertionError for a
        # non-tensor spec. Mutates instance state and consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._dim = random.randrange(len(output_spec.size))
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Return source calling the torch op on the first input with the stashed
        # dimension, then clear the stash.
        # KNOWN ISSUE: unlike the module description, no check that the stash is set; a missing
        # stash is emitted as the text None.
        dim = self._dim
        self._dim = None
        return f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim})"


class CumprodOperator(_CumulativeReductionBase):
    # [CLASS DOC] Operator for torch.cumprod over non-bool numeric tensors.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.cumprod.
        super().__init__("torch.cumprod")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class LogcumsumexpOperator(_CumulativeReductionBase):
    # [CLASS DOC] Operator for torch.logcumsumexp; restricted to float dtypes.
    """Float-only cumulative reduction: ``torch.logcumsumexp(input, dim)``."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.logcumsumexp.
        super().__init__("torch.logcumsumexp")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs with at least one dimension and a float
        # dtype.
        return (
            isinstance(output_spec, TensorSpec)
            and len(output_spec.size) >= 1
            and output_spec.dtype in FLOAT_DTYPES
        )


# ---------------------------------------------------------------------------
# Family B -- Cumulative tuple .values
# ---------------------------------------------------------------------------


class _CumulativeTupleReductionBase(_CumulativeReductionBase):
    # [CLASS DOC] HELPER: Cumulative base for ops returning a (values, indices) named tuple; the
    # generated code selects the values field. Otherwise identical to its parent.
    """Same as _CumulativeReductionBase but codegen extracts .values."""

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Return source calling the torch op on the first input with the stashed
        # dimension and taking its values field, then clear the stash.
        dim = self._dim
        self._dim = None
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim}).values"
        )


class CummaxOperator(_CumulativeTupleReductionBase):
    # [CLASS DOC] Operator for torch.cummax, keeping only the values result.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.cummax.
        super().__init__("torch.cummax")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class CumminOperator(_CumulativeTupleReductionBase):
    # [CLASS DOC] Operator for torch.cummin, keeping only the values result.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.cummin.
        super().__init__("torch.cummin")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


# ---------------------------------------------------------------------------
# Family C -- Full reduction, single tensor
# ---------------------------------------------------------------------------


class _FullOrKeepdimReductionBase(Operator):
    # [CLASS DOC] HELPER: Base for reductions producing a 0-D result (reduce everything), or a
    # single-dimension reduction with keepdim true or false. Class attributes configure subclasses
    # (allowed output dtypes, optional input dtype pool, whether to pass a dtype keyword, whether
    # keepdim false is allowed). Instance state: stashed mode, dimension, keepdim flag, input size
    # and input dtype, set when input specs are generated and cleared by code generation.
    """Full reduction producing a 0-D scalar, keepdim=True, or keepdim=False shape.

    Subclasses set:
      - allowed_output_dtypes: tuple of torch.dtype that this op can
        produce as output.
      - input_dtype_pool: tuple of torch.dtype that the input may use
        when the input dtype is decoupled from the output dtype.
        When None (default), the input dtype equals the output dtype.
      - codegen_dtype_kwarg: when True (sum/prod/nansum), append
        ", dtype={output_spec.dtype}" to the emitted call.
      - supports_keepdim_false: when True, ``can_produce`` accepts
        any non-scalar shape (not just those with a size-1 dim).

    Modes:
      "scalar"  : output_spec.size == (); emit ``torch.X(input)``.
      "keepdim" : dim-specific reduction; emit
                  ``torch.X(input, dim=k, keepdim=True)`` or
                  ``torch.X(input, dim=k)`` (keepdim=False).
    """

    allowed_output_dtypes: tuple[torch.dtype, ...] = ()
    input_dtype_pool: tuple[torch.dtype, ...] | None = None
    codegen_dtype_kwarg: bool = False
    supports_keepdim_false: bool = False

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Register under the given name and clear all stashed choices.
        super().__init__(name)
        self._mode: str | None = None
        self._dim: int | None = None
        self._keepdim: bool | None = None
        self._input_size: tuple[int, ...] | None = None
        self._input_dtype: torch.dtype | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs whose dtype is allowed and whose shape is
        # empty, contains a dimension of size 1, or (when keepdim false is supported) anything.
        if not isinstance(output_spec, TensorSpec):
            return False
        if output_spec.dtype not in self.allowed_output_dtypes:
            return False
        if output_spec.size == ():
            return True
        if 1 in output_spec.size:
            return True
        return self.supports_keepdim_false

    def _pick_input_dtype(self, output_dtype: torch.dtype) -> torch.dtype:
        # [FUNCTION DOC] HELPER: Return the output dtype when no input dtype pool is configured,
        # else a random member of the pool. Consumes global random state in the latter case.
        if self.input_dtype_pool is None:
            return output_dtype
        return random.choice(self.input_dtype_pool)

    def _pick_scalar_input_shape(self) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Hook returning a random input shape for the 0-D output case; the
        # default is any shape from the plain random shape helper.
        """Hook for subclasses to override the scalar-mode input-shape draw.

        Default delegates to ``_random_input_shape``. Bessel-corrected
        reductions (e.g., ``torch.var``) override this to delegate to
        ``_random_input_shape_min_numel_2``.
        """
        return _random_input_shape()

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Choose and stash the input dtype, then the reduction mode. For an empty
        # output shape: scalar mode with a random input shape. Otherwise, if the output has a size
        # 1 dimension (and, when keepdim false is supported, with probability one half): keepdim
        # true, reducing a random size-1 dimension whose input size is redrawn from 2 to 6. Else
        # keepdim false: a random insertion position and a new dimension of size 2 to 6 are added
        # to the output shape. Returns one contiguous input spec. Raises AssertionError for a
        # non-tensor spec. Mutates state and consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._input_dtype = self._pick_input_dtype(output_spec.dtype)
        if output_spec.size == ():
            self._mode = "scalar"
            self._keepdim = None
            self._input_size = self._pick_scalar_input_shape()
        elif 1 in output_spec.size and (
            not self.supports_keepdim_false or random.random() < 0.5
        ):
            self._mode = "keepdim"
            self._keepdim = True
            size_one_dims = [i for i, s in enumerate(output_spec.size) if s == 1]
            self._dim = random.choice(size_one_dims)
            new_dim = random.randint(2, 6)
            self._input_size = tuple(
                new_dim if i == self._dim else s for i, s in enumerate(output_spec.size)
            )
        else:
            self._mode = "keepdim"
            self._keepdim = False
            ndim = len(output_spec.size)
            self._dim = random.randint(0, ndim)
            r = random.randint(2, 6)
            input_size = list(output_spec.size)
            input_size.insert(self._dim, r)
            self._input_size = tuple(input_size)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=self._input_dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Snapshot and clear the stashed mode, dimension and keepdim flag, then
        # return source calling the torch op: with only the input in scalar mode, with a dimension
        # and keepdim true, or with a dimension only. When the class flag is set, a dtype keyword
        # naming the output dtype is appended. Raises AssertionError for a non-tensor spec.
        # KNOWN ISSUE: a missing stash (no prior spec generation) falls through to the last branch
        # and emits the text None as the dimension instead of raising an error.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        mode = self._mode
        dim = self._dim
        keepdim = self._keepdim
        self._mode = None
        self._dim = None
        self._keepdim = None
        self._input_size = None
        self._input_dtype = None
        dtype_kwarg = f", dtype={output_spec.dtype}" if self.codegen_dtype_kwarg else ""
        if mode == "scalar":
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}{dtype_kwarg})"
            )
        if keepdim:
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}, "
                f"dim={dim}, keepdim=True{dtype_kwarg})"
            )
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, "
            f"dim={dim}{dtype_kwarg})"
        )


class SumOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.sum over non-bool numeric dtypes; appends a dtype keyword.
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.sum.
        super().__init__("torch.sum")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class MeanOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.mean over float dtypes.
    allowed_output_dtypes = FLOAT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.mean.
        super().__init__("torch.mean")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class ProdOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.prod over non-bool numeric dtypes; appends a dtype keyword.
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.prod.
        super().__init__("torch.prod")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class NansumOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.nansum over float dtypes; appends a dtype keyword.
    allowed_output_dtypes = FLOAT_DTYPES
    codegen_dtype_kwarg = True
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.nansum.
        super().__init__("torch.nansum")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class VarOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.var over float dtypes with a random correction of 0, 1 or 2.
    # Extra state: the stashed correction. Input shapes are forced to at least two elements when
    # the correction is nonzero so results are not NaN.
    allowed_output_dtypes = FLOAT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.var and clear the stashed correction.
        super().__init__("torch.var")
        self._correction: int | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def _pick_scalar_input_shape(self) -> tuple[int, ...]:
        # [FUNCTION DOC] HELPER: Return a random shape; with a nonzero stashed correction (or
        # none) the shape has at least two elements, with correction zero any shape is allowed.
        if self._correction == 0:
            return _random_input_shape()
        return _random_input_shape_min_numel_2()

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Stash a random correction from 0, 1, 2, then defer to the parent to build
        # the input spec. Mutates state and consumes global random state.
        self._correction = random.choice([0, 1, 2])
        return super().fuzz_inputs_specs(output_spec)

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Clear the stashed correction, get the parent's source text, and replace
        # its final closing parenthesis with a correction keyword followed by a closing
        # parenthesis.
        # KNOWN ISSUE: the correction is cleared before calling the parent, whose behavior does
        # not depend on it, but a missing stash would emit the text None as the correction.
        correction = self._correction
        self._correction = None
        base = super().codegen(output_name, input_names, output_spec)
        return base[:-1] + f", correction={correction})"


class AllOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.all producing bool; input dtype drawn from bool or numeric.
    allowed_output_dtypes = (torch.bool,)
    input_dtype_pool = _BOOL_OR_NUMERIC_INPUT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.all.
        super().__init__("torch.all")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class AnyOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.any producing bool; input dtype drawn from bool or numeric.
    allowed_output_dtypes = (torch.bool,)
    input_dtype_pool = _BOOL_OR_NUMERIC_INPUT_DTYPES
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.any.
        super().__init__("torch.any")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class AmaxOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.amax over non-bool numeric dtypes.
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.amax.
        super().__init__("torch.amax")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class AminOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.amin over non-bool numeric dtypes.
    allowed_output_dtypes = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.amin.
        super().__init__("torch.amin")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class ArgmaxOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.argmax producing int64; input dtype drawn from numeric dtypes.
    allowed_output_dtypes = (torch.int64,)
    input_dtype_pool = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.argmax.
        super().__init__("torch.argmax")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class ArgminOperator(_FullOrKeepdimReductionBase):
    # [CLASS DOC] Operator for torch.argmin producing int64; input dtype drawn from numeric dtypes.
    allowed_output_dtypes = (torch.int64,)
    input_dtype_pool = _NUMERIC_INPUT_DTYPES_NO_BOOL
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.argmin.
        super().__init__("torch.argmin")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


# ---------------------------------------------------------------------------
# Family D -- Dim-required tuple .values
# ---------------------------------------------------------------------------


class _DimRequiredTupleReductionBase(Operator):
    # [CLASS DOC] HELPER: Base for reductions over one dimension returning a named tuple whose
    # values field is used (max, min, median, mode). State: stashed dimension, keepdim flag and
    # input size, set when input specs are generated and cleared by code generation. The class
    # flag supports_keepdim_false widens the accepted output shapes.
    """2-arg-with-dim form, .values extraction, keepdim=True or False.

    Used for max, min, median, mode. When ``supports_keepdim_false``
    is False (default), ``can_produce`` requires a size-1 dim in the
    output (keepdim=True only). When True, any non-scalar shape is
    accepted and keepdim=False may be selected.
    """

    supports_keepdim_false: bool = False

    def __init__(self, name: str) -> None:
        # [FUNCTION DOC] Register under the given name and clear all stashed choices.
        super().__init__(name)
        self._dim: int | None = None
        self._keepdim: bool | None = None
        self._input_size: tuple[int, ...] | None = None

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for tensor specs with at least one dimension and a non-bool
        # numeric dtype, provided the shape has a size-1 dimension or keepdim false is supported.
        if not isinstance(output_spec, TensorSpec):
            return False
        if len(output_spec.size) < 1:
            return False
        if output_spec.dtype not in _NUMERIC_INPUT_DTYPES_NO_BOOL:
            return False
        if 1 in output_spec.size:
            return True
        return self.supports_keepdim_false

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Choose and stash keepdim and the reduced dimension. If the output has a
        # size-1 dimension (and, when keepdim false is supported, with probability one half),
        # keepdim is true and a random size-1 dimension is enlarged to a size from 2 to 6. Else
        # keepdim is false and a new dimension of size 2 to 6 is inserted at a random position.
        # Returns one contiguous input spec with the output dtype. Raises AssertionError for a
        # non-tensor spec. Mutates state and consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        if 1 in output_spec.size and (
            not self.supports_keepdim_false or random.random() < 0.5
        ):
            self._keepdim = True
            size_one_dims = [i for i, s in enumerate(output_spec.size) if s == 1]
            self._dim = random.choice(size_one_dims)
            new_dim = random.randint(2, 6)
            self._input_size = tuple(
                new_dim if i == self._dim else s for i, s in enumerate(output_spec.size)
            )
        else:
            self._keepdim = False
            ndim = len(output_spec.size)
            self._dim = random.randint(0, ndim)
            r = random.randint(2, 6)
            input_size = list(output_spec.size)
            input_size.insert(self._dim, r)
            self._input_size = tuple(input_size)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Snapshot and clear the stash, then return source calling the torch op on
        # the first input with the dimension (plus keepdim true when set) and taking the values
        # field of the result.
        dim = self._dim
        keepdim = self._keepdim
        self._dim = None
        self._keepdim = None
        self._input_size = None
        if keepdim:
            return (
                f"{output_name} = {self.torch_op_name}({input_names[0]}, "
                f"dim={dim}, keepdim=True).values"
            )
        return (
            f"{output_name} = {self.torch_op_name}({input_names[0]}, dim={dim}).values"
        )


class MaxOperator(_DimRequiredTupleReductionBase):
    # [CLASS DOC] Operator for the dimension form of torch.max, keeping the values result.
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.max.
        super().__init__("torch.max")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class MinOperator(_DimRequiredTupleReductionBase):
    # [CLASS DOC] Operator for the dimension form of torch.min, keeping the values result.
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.min.
        super().__init__("torch.min")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class MedianOperator(_DimRequiredTupleReductionBase):
    # [CLASS DOC] Operator for the dimension form of torch.median, keeping the values result.
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.median.
        super().__init__("torch.median")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


class ModeOperator(_DimRequiredTupleReductionBase):
    # [CLASS DOC] Operator for the dimension form of torch.mode, keeping the values result.
    supports_keepdim_false = True

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.mode.
        super().__init__("torch.mode")

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name


# ---------------------------------------------------------------------------
# Family E -- Standalone special-case ops
# ---------------------------------------------------------------------------


class CountNonzeroOperator(Operator):
    # [CLASS DOC] Operator for torch.count_nonzero producing a 0-D int64 tensor from an input of
    # random shape and random bool or numeric dtype. State: last drawn input size and dtype
    # (never cleared).
    """torch.count_nonzero -- 0-D int64 scalar output (no dim form)."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.count_nonzero; initialize the stashed
        # input size to none and the input dtype to the last entry of the dtype pool.
        super().__init__("torch.count_nonzero")
        self._input_size: tuple[int, ...] | None = None
        self._input_dtype: torch.dtype = _BOOL_OR_NUMERIC_INPUT_DTYPES[-1]

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for 0-D int64 tensor specs.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and output_spec.dtype == torch.int64
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Stash a random input shape and a random dtype from the bool-or-numeric
        # pool and return one contiguous input spec of that shape and dtype. Raises
        # AssertionError for a non-tensor spec. Consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._input_size = _random_input_shape()
        self._input_dtype = random.choice(_BOOL_OR_NUMERIC_INPUT_DTYPES)
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=self._input_dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Return one line of source calling count_nonzero on the first input.
        return f"{output_name} = torch.count_nonzero({input_names[0]})"


class AminmaxOperator(Operator):
    # [CLASS DOC] Operator for the no-dimension form of torch.aminmax, keeping the min field, so
    # it produces a 0-D tensor of a non-bool numeric dtype from a random-shape input of the same
    # dtype. State: stashed input size, cleared by code generation.
    """torch.aminmax -- 0-D scalar via .min extraction (NamedTuple).

    Note: torch.aminmax DOES support a ``dim``/``keepdim`` form. This
    batch intentionally restricts coverage to the no-dim 0-D scalar form.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.aminmax and clear the stashed input size.
        super().__init__("torch.aminmax")
        self._input_size: tuple[int, ...] | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for 0-D tensor specs with a non-bool numeric dtype.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and output_spec.dtype in _NUMERIC_INPUT_DTYPES_NO_BOOL
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Stash a random input shape and return one contiguous input spec of that
        # shape with the output dtype. Raises AssertionError for a non-tensor spec. Consumes
        # global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._input_size = _random_input_shape()
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Clear the stashed input size and return source taking the min field of
        # aminmax applied to the first input.
        self._input_size = None
        return f"{output_name} = torch.aminmax({input_names[0]}).min"


class VarMeanOperator(Operator):
    # [CLASS DOC] Operator for the no-dimension form of torch.var_mean, keeping the variance
    # (first tuple element), so it produces a 0-D float tensor. State: stashed input size and
    # correction (0, 1 or 2), cleared by code generation.
    """torch.var_mean -- 0-D float scalar via [0] extraction (NamedTuple).

    Note: torch.var_mean DOES support a ``dim``/``keepdim`` form. This
    batch intentionally restricts coverage to the no-dim 0-D scalar form.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.var_mean and clear the stashed choices.
        super().__init__("torch.var_mean")
        self._input_size: tuple[int, ...] | None = None
        self._correction: int | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, which is the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for 0-D tensor specs with a float dtype.
        return (
            isinstance(output_spec, TensorSpec)
            and output_spec.size == ()
            and is_float_dtype(output_spec.dtype)
        )

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Stash a random correction from 0, 1, 2 and a random input shape (with at
        # least two elements unless the correction is zero), then return one contiguous input spec
        # with that shape and the output dtype. Raises AssertionError for a non-tensor spec.
        # Consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise AssertionError(f"expected TensorSpec, got {type(output_spec)}")
        self._correction = random.choice([0, 1, 2])
        if self._correction == 0:
            self._input_size = _random_input_shape()
        else:
            self._input_size = _random_input_shape_min_numel_2()
        return [
            TensorSpec(
                size=self._input_size,
                stride=_contiguous_stride(self._input_size),
                dtype=output_spec.dtype,
            )
        ]

    def codegen(
        self,
        output_name: str,
        input_names: list[str],
        output_spec: Spec,
    ) -> str:
        # [FUNCTION DOC] Snapshot and clear the stash, then return source calling var_mean on the
        # first input with the correction keyword and taking the first tuple element.
        correction = self._correction
        self._input_size = None
        self._correction = None
        # [0] extracts the variance from the (var, mean) tuple;
        # named-tuple .var access breaks under torch.compile tracing.
        return (
            f"{output_name} = torch.var_mean("
            f"{input_names[0]}, correction={correction})[0]"
        )
