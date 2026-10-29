# pyre-strict
# [MODULE DOC] Fuzzer operators for tensor factory functions. Provides fill-value and stride
# helpers, three base classes (factories taking only a size, factories taking a prototype tensor,
# and prototype-method factories named new_X), concrete ones/zeros/full, ones_like/zeros_like/
# full_like, new_ones/new_zeros/new_full operators, and a separate arange operator. Non-contiguous
# outputs are built from an empty strided buffer that is filled or copied into. No import-time
# side effects besides imports.
"""Tensor creation & initialization Operator subclasses.

These operators model torch factory functions (``torch.ones``, ``torch.full``,
``torch.arange``, ``torch.Tensor.new_full``, etc.) for the upstream torchfuzz
framework.  All factories are non-leaf operators: they may take zero or one
prototype tensor input but always produce a fresh ``TensorSpec`` output.

When ``output_spec.stride`` is non-contiguous, factories emit
``torch.empty_strided(...).fill_(...)`` (or a copy from a freshly-allocated
strided buffer for the ``_like`` family) so the requested layout is honored
without aliasing into another tensor's storage (see plan note [H19]).
"""

from __future__ import annotations

import random

import torch

from torchfuzz.operators._dtypes import FLOAT_DTYPES
from torchfuzz.operators.base import Operator
from torchfuzz.tensor_fuzzer import Spec, TensorSpec


_ARANGE_ALLOWED_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    }
)

_NON_BOOL_DTYPES: tuple[torch.dtype, ...] = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.float16,
    torch.float32,
    torch.float64,
    torch.bfloat16,
)


def _contiguous_stride(size: tuple[int, ...]) -> tuple[int, ...]:
    # [FUNCTION DOC] HELPER: Return row-major contiguous strides for a size tuple; empty size
    # gives an empty tuple. Pure.
    """Return the contiguous (row-major) stride for ``size``."""
    if not size:
        return ()
    strides: list[int] = [1]
    for dim in reversed(size[1:]):
        strides.append(strides[-1] * dim)
    return tuple(reversed(strides))


def _default_fill_value(dtype: torch.dtype) -> object:
    # [FUNCTION DOC] HELPER: Return the fixed fill value for a dtype: True for bool, 1.0 for float
    # dtypes, 1 otherwise.
    # KNOWN ISSUE: not called anywhere in this module; the operators use the random fill helper.
    """Canonical fill value for full/full_like/new_full per dtype.

    Centralized so FullOperator, FullLikeOperator, and NewFullOperator cannot
    drift in their fill-value selection (per plan note [H4]).
    """
    if dtype == torch.bool:
        return True
    if dtype in FLOAT_DTYPES:
        return 1.0
    return 1


_UNSIGNED_INT_DTYPES: frozenset[torch.dtype] = frozenset(
    {
        torch.uint8,
    }
)


def _random_fill_value(dtype: torch.dtype) -> object:
    # [FUNCTION DOC] HELPER: Return a random fill value for a dtype: a random bool, a float in
    # the range -10 to 10 rounded to four decimals, an integer from 0 to 100 for unsigned dtypes
    # (only uint8 is listed), else an integer from -100 to 100. Consumes global random state.
    """Random fill value appropriate for ``dtype``.

    Used by Full/FullLike/NewFull operators to exercise a wider range of
    constant values than the fixed ``_default_fill_value``.
    """
    if dtype == torch.bool:
        return random.choice([True, False])
    if dtype in FLOAT_DTYPES:
        return round(random.uniform(-10.0, 10.0), 4)
    if dtype in _UNSIGNED_INT_DTYPES:
        return random.randint(0, 100)
    return random.randint(-100, 100)


def _dtype_str(dt: torch.dtype) -> str:
    # [FUNCTION DOC] HELPER: Return the string form of a dtype, as used in generated source.
    """Mirror the formatting used by upstream operators/constant.py."""
    return str(dt)


def _is_contiguous(output_spec: TensorSpec) -> bool:
    # [FUNCTION DOC] HELPER: Return True if the spec's stride equals the row-major contiguous
    # stride for its size.
    return tuple(output_spec.stride) == _contiguous_stride(tuple(output_spec.size))


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class TensorCreationFromSizeOperatorBase(Operator):
    # [CLASS DOC] Base for factories without input tensors (ones, zeros, full). Accepts any tensor
    # output spec and has no inputs. Subclasses supply the constant used on the strided path and
    # optionally extra factory arguments. Name ends in Base, so the registry skips it.
    """Base for factories that take no input tensor (ones, zeros, full)."""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, used as the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for any tensor spec.
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return an empty list: the factory needs no input specs.
        return []

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Abstract hook returning the constant used to fill a strided
        # buffer; raises NotImplementedError in the base.
        """Subclasses override to provide the constant the factory writes."""
        raise NotImplementedError

    def _factory_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return the text of the arguments after the size in the
        # contiguous call: a dtype keyword for the output dtype.
        """Args to pass after size for the contiguous-path factory call."""
        return f"dtype={_dtype_str(output_spec.dtype)}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source creating the output. For a contiguous layout it calls the
        # torch factory with the size and the factory arguments. Otherwise it allocates an empty
        # strided buffer of the requested size, stride and dtype and fills it with the
        # subclass constant. Raises ValueError for a non-tensor spec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        size = tuple(output_spec.size)
        if _is_contiguous(output_spec):
            return (
                f"{output_name} = {self.torch_op_name}({size}, "
                f"{self._factory_args(output_spec)})"
            )
        stride = tuple(output_spec.stride)
        return (
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}).fill_("
            f"{self._fill_value(output_spec)!r})"
        )


class TensorCreationFromPrototypeOperatorBase(Operator):
    # [CLASS DOC] Base for factories taking one prototype tensor (the like family). Accepts any
    # tensor output spec. Subclasses may add extra positional arguments after the prototype. Name
    # ends in Base, so the registry skips it.
    """Base for factories that take one prototype tensor (ones_like, etc.)."""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, used as the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for any tensor spec.
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return one prototype spec with the output's size and stride. Its dtype is
        # the output dtype, except with probability 0.3 (when the output is not bool) a different
        # non-bool dtype is drawn. Raises ValueError for a non-tensor spec. Consumes global
        # random state.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        proto_dtype = output_spec.dtype
        # 30% chance: use a different prototype dtype to exercise the
        # dtype-override path (codegen passes dtype= explicitly).
        if proto_dtype != torch.bool and random.random() < 0.3:
            candidates = [dt for dt in _NON_BOOL_DTYPES if dt != proto_dtype]
            if candidates:
                proto_dtype = random.choice(candidates)
        return [
            TensorSpec(
                size=output_spec.size,
                stride=output_spec.stride,
                dtype=proto_dtype,
            )
        ]

    def _like_extra_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return text of extra positional arguments after the prototype;
        # empty in the base.
        """Extra positional args after the prototype (e.g. fill value for full_like)."""
        return ""

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source calling the like factory on the prototype with any extra
        # arguments and an explicit output dtype. For a non-contiguous output the result goes to a
        # temporary variable, then an empty strided buffer is allocated and the temporary is
        # copied into it; both steps are on one line separated by semicolons. Raises ValueError for
        # a non-tensor spec or when the input name count is not one.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 input tensor"
            )
        prototype = input_names[0]
        extra = self._like_extra_args(output_spec)
        dtype_kwarg = f", dtype={_dtype_str(output_spec.dtype)}"
        like_call = f"{self.torch_op_name}({prototype}{extra}{dtype_kwarg})"
        if _is_contiguous(output_spec):
            return f"{output_name} = {like_call}"
        size = tuple(output_spec.size)
        stride = tuple(output_spec.stride)
        tmp = f"_tmp_{output_name}"
        return (
            f"{tmp} = {like_call}; "
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}); "
            f"{output_name}.copy_({tmp})"
        )


class TensorMethodNewOperatorBase(Operator):
    # [CLASS DOC] Base for tensor-method factories new_X invoked on a prototype tensor. Class
    # attribute method_name names the method. Accepts any tensor output spec; the prototype is a
    # one-element tensor of the output dtype. Name ends in Base, so the registry skips it.
    """Base for ``Tensor.new_X(size, ...)`` methods (new_ones, new_zeros, ...)."""

    method_name: str = ""

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, used as the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True for any tensor spec.
        return isinstance(output_spec, TensorSpec)

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Return one prototype spec of size (1,), stride (1,) and the output dtype.
        # Raises ValueError for a non-tensor spec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        return [TensorSpec(size=(1,), stride=(1,), dtype=output_spec.dtype)]

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Abstract hook returning the constant for the strided path;
        # raises NotImplementedError in the base.
        raise NotImplementedError

    def _new_extra_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return text of extra positional arguments after the size; empty
        # in the base.
        """Extra positional args after the size for the contiguous-path call."""
        return ""

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source for the output. Contiguous layout: call the new_X method
        # on the prototype with the size, any extra arguments and the output dtype. Otherwise
        # allocate an empty strided buffer and fill it with the subclass constant. Raises
        # ValueError for a non-tensor spec or when the input name count is not one.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        if len(input_names) != 1:
            raise ValueError(
                f"{self.__class__.__name__} requires exactly 1 prototype tensor"
            )
        prototype = input_names[0]
        size = tuple(output_spec.size)
        if _is_contiguous(output_spec):
            extra = self._new_extra_args(output_spec)
            return (
                f"{output_name} = {prototype}.{self.method_name}({size}{extra}, "
                f"dtype={_dtype_str(output_spec.dtype)})"
            )
        stride = tuple(output_spec.stride)
        return (
            f"{output_name} = torch.empty_strided({size}, {stride}, "
            f"dtype={_dtype_str(output_spec.dtype)}).fill_("
            f"{self._fill_value(output_spec)!r})"
        )


# ---------------------------------------------------------------------------
# Concrete factories without a prototype input
# ---------------------------------------------------------------------------


class OnesOperator(TensorCreationFromSizeOperatorBase):
    # [CLASS DOC] Operator for torch.ones.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.ones.
        super().__init__("torch.ones")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the integer 1 as the strided-path fill constant.
        return 1


class ZerosOperator(TensorCreationFromSizeOperatorBase):
    # [CLASS DOC] Operator for torch.zeros.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.zeros.
        super().__init__("torch.zeros")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the integer 0 as the strided-path fill constant.
        return 0


class FullOperator(TensorCreationFromSizeOperatorBase):
    # [CLASS DOC] Operator for torch.full with a random fill value. State: the stashed fill value,
    # drawn when input specs are generated and cleared by code generation.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.full and clear the stashed fill value.
        super().__init__("torch.full")
        self._stashed_fill: object | None = None

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the stashed fill value.
        return self._stashed_fill

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draw and stash a random fill value suited to the output dtype; return an
        # empty list. Raises ValueError for a non-tensor spec. Consumes global random state.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return []

    def _factory_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return the stashed fill value as source text followed by a dtype
        # keyword for the output dtype.
        return f"{self._stashed_fill!r}, dtype={_dtype_str(output_spec.dtype)}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Generate source via the base class, then clear the stashed fill value.
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Concrete factories with a prototype input
# ---------------------------------------------------------------------------


class OnesLikeOperator(TensorCreationFromPrototypeOperatorBase):
    # [CLASS DOC] Operator for torch.ones_like.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.ones_like.
        super().__init__("torch.ones_like")


class ZerosLikeOperator(TensorCreationFromPrototypeOperatorBase):
    # [CLASS DOC] Operator for torch.zeros_like.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.zeros_like.
        super().__init__("torch.zeros_like")


class FullLikeOperator(TensorCreationFromPrototypeOperatorBase):
    # [CLASS DOC] Operator for torch.full_like with a random fill value. State: the stashed fill
    # value, drawn when input specs are generated and cleared by code generation.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.full_like and clear the stashed fill value.
        super().__init__("torch.full_like")
        self._stashed_fill: object | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draw and stash a random fill value for the output dtype, then return the
        # prototype specs from the base class. Raises ValueError for a non-tensor spec. Consumes
        # global random state.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return super().fuzz_inputs_specs(output_spec)

    def _like_extra_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return the stashed fill value as an extra positional argument
        # text, preceded by a comma.
        return f", {self._stashed_fill!r}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Generate source via the base class, then clear the stashed fill value.
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Concrete Tensor.new_X factories
# ---------------------------------------------------------------------------


class NewOnesOperator(TensorMethodNewOperatorBase):
    # [CLASS DOC] Operator for the tensor method new_ones.
    method_name: str = "new_ones"

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.Tensor.new_ones.
        super().__init__("torch.Tensor.new_ones")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the integer 1 as the strided-path fill constant.
        return 1


class NewZerosOperator(TensorMethodNewOperatorBase):
    # [CLASS DOC] Operator for the tensor method new_zeros.
    method_name: str = "new_zeros"

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.Tensor.new_zeros.
        super().__init__("torch.Tensor.new_zeros")

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the integer 0 as the strided-path fill constant.
        return 0


class NewFullOperator(TensorMethodNewOperatorBase):
    # [CLASS DOC] Operator for the tensor method new_full with a random fill value. State: the
    # stashed fill value, drawn when input specs are generated and cleared by code generation.
    method_name: str = "new_full"

    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.Tensor.new_full; clear the stashed fill.
        super().__init__("torch.Tensor.new_full")
        self._stashed_fill: object | None = None

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] Draw and stash a random fill value for the output dtype, then return the
        # prototype spec from the base class. Raises ValueError for a non-tensor spec. Consumes
        # global random state.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError(
                f"{self.__class__.__name__} can only produce TensorSpec outputs"
            )
        self._stashed_fill = _random_fill_value(output_spec.dtype)
        return super().fuzz_inputs_specs(output_spec)

    def _fill_value(self, output_spec: TensorSpec) -> object:
        # [FUNCTION DOC] HELPER: Return the stashed fill value.
        return self._stashed_fill

    def _new_extra_args(self, output_spec: TensorSpec) -> str:
        # [FUNCTION DOC] HELPER: Return the stashed fill value as extra argument text preceded by
        # a comma.
        return f", {self._stashed_fill!r}"

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Generate source via the base class, then clear the stashed fill value.
        result = super().codegen(output_name, input_names, output_spec)
        self._stashed_fill = None
        return result


# ---------------------------------------------------------------------------
# Special-case: torch.arange (always 1-D, contiguous)
# ---------------------------------------------------------------------------


class ArangeOperator(Operator):
    # [CLASS DOC] Operator for torch.arange producing contiguous 1-D tensors of the allowed dtypes
    # (int8 to int64, uint8, float16, float32, float64, bfloat16). State: stashed start and step,
    # either both set or both none, drawn when input specs are generated and cleared by code
    # generation. It has no input tensors.
    def __init__(self) -> None:
        # [FUNCTION DOC] Register under the name torch.arange and clear start and step.
        super().__init__("torch.arange")
        self._start: int | None = None
        self._step: int | None = None

    @property
    def torch_op_name(self) -> str:
        # [FUNCTION DOC] Property returning the operator name, used as the torch op name.
        return self.name

    def can_produce(self, output_spec: Spec) -> bool:
        # [FUNCTION DOC] Return True only for tensor specs with exactly one dimension, stride
        # (1,), and a dtype in the allowed set.
        if not isinstance(output_spec, TensorSpec):
            return False
        if len(output_spec.size) != 1:
            return False
        if tuple(output_spec.stride) != (1,):
            return False
        return output_spec.dtype in _ARANGE_ALLOWED_DTYPES

    def fuzz_inputs_specs(self, output_spec: Spec) -> list[Spec]:
        # [FUNCTION DOC] With probability one half stash a random start from -10 to 10 and a step
        # from 1 to 3; otherwise clear both. Returns an empty list. Mutates state and consumes
        # global random state.
        # KNOWN ISSUE: a negative start is possible for unsigned dtypes, and large lengths can
        # overflow small integer dtypes, so the emitted range may be invalid for them.
        if random.random() < 0.5:
            self._start = random.randint(-10, 10)
            self._step = random.randint(1, 3)
        else:
            self._start = None
            self._step = None
        return []

    def codegen(
        self, output_name: str, input_names: list[str], output_spec: Spec
    ) -> str:
        # [FUNCTION DOC] Return source calling arange with a dtype keyword. With a stashed start
        # and step, the end is start plus step times the output length and all three are passed;
        # otherwise only the length is passed. Clears start and step. Raises ValueError for a
        # non-tensor spec.
        if not isinstance(output_spec, TensorSpec):
            raise ValueError("ArangeOperator can only produce TensorSpec outputs")
        n = output_spec.size[0]
        dtype_arg = f"dtype={_dtype_str(output_spec.dtype)}"
        if self._start is not None and self._step is not None:
            start = self._start
            step = self._step
            end = start + step * n
            result = (
                f"{output_name} = torch.arange({start}, {end}, {step}, {dtype_arg})"
            )
        else:
            result = f"{output_name} = torch.arange({n}, {dtype_arg})"
        self._start = None
        self._step = None
        return result
