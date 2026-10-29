# mypy: allow-untyped-defs
# [MODULE DOC]
# Registry of TorchScript decompositions: Python implementations of ATen ops
# that are scripted, optimized and stored in a table keyed by operator schema
# string. Contents: decomposition_table, the register_decomposition decorator,
# signature-check helpers, and decompositions for var. Import-time side effects:
# scripts and registers the var decompositions (running JIT passes) and adds
# their names to the global used-name set.
import torch
from torch import Tensor


aten = torch.ops.aten
import inspect
import warnings
from collections.abc import Callable
from typing import TypeVar
from typing_extensions import ParamSpec

from torch.types import Number


decomposition_table: dict[str, torch.jit.ScriptFunction] = {}
function_name_set: set[str] = set()

_T = TypeVar("_T")
_P = ParamSpec("_P")


def check_decomposition_has_type_annotations(f) -> None:
    # [FUNCTION DOC] Verifies function f has an annotation on every parameter and
    # on its return. Raises AssertionError naming the first missing one.
    # KNOWN ISSUE: error messages read the name attribute of f, but plain
    # functions only have the dunder name attribute, so an AttributeError would
    # be raised instead of the intended AssertionError.
    inspect_empty = inspect._empty  # type: ignore[attr-defined]
    sig = inspect.signature(f)
    for param in sig.parameters.values():
        if param.annotation == inspect_empty:
            raise AssertionError(
                f"No signature on param {param.name} for function {f.name}"
            )

    if sig.return_annotation == inspect_empty:
        raise AssertionError(f"No return annotation for function {f.name}")


def signatures_match(decomposition_sig, torch_op_sig):
    # [FUNCTION DOC] Compares two inspect signatures. Returns True only if they
    # have the same number of parameters, each pair has equal name and annotation,
    # defaults are equal wherever both are specified, and return annotations are
    # equal. Emits a warning when a decomposition parameter is named self (public
    # API uses input). Kind is not compared. No mutation.
    decomp_params = decomposition_sig.parameters
    op_params = torch_op_sig.parameters

    if len(decomp_params) != len(op_params):
        return False

    for decomp_param, op_param in zip(decomp_params.values(), op_params.values()):
        # can't check full equality yet because not all fields are correctly deduced
        # in the torch_op_sig - like default value
        # can't check 'kind' bc
        # kwarg-only values with defaults not yet supported in TS
        inspect_empty = inspect._empty  # type: ignore[attr-defined]
        for field in ["name", "annotation"]:
            if field == "name" and decomp_param.name == "self":
                warnings.warn(
                    "PyTorch uses 'input' instead of 'self' on public api", stacklevel=2
                )

            if getattr(decomp_param, field) != getattr(op_param, field):
                return False

        decomp_default = decomp_param.default
        op_default = op_param.default
        # default value not always correctly inferred as being present on torch schema,
        # but if specified on both they should be equal
        if decomp_default != inspect_empty and op_default != inspect_empty:
            if decomp_default != op_default:
                return False

    return decomposition_sig.return_annotation == torch_op_sig.return_annotation


def register_decomposition(
    aten_op: torch._ops.OpOverload,
    registry: dict[str, torch.jit.ScriptFunction] | None = None,
) -> Callable[[Callable[_P, _T]], Callable[_P, _T]]:
    # [FUNCTION DOC] Returns a decorator that registers a function as the
    # decomposition of the given ATen overload in registry (default: the global
    # decomposition table). Args: aten_op, an OpOverload; registry, optional
    # mapping from schema string to scripted function.
    def decomposition_decorator(f: Callable[_P, _T]) -> Callable[_P, _T]:
        # [FUNCTION DOC] HELPER: Validates aten_op is an OpOverload and that the
        # function name is unique (AssertionError otherwise), records the name in
        # the global name set, scripts f, inlines it, runs peephole and constant
        # propagation twice, stores it in the registry under the schema string,
        # and returns the original unscripted f. Mutates global state.
        nonlocal registry
        if registry is None:
            registry = decomposition_table

        if not isinstance(aten_op, torch._ops.OpOverload):
            raise AssertionError(
                f"Expected aten_op to be OpOverload, got {type(aten_op)}"
            )

        # Need unique name for jit function serialization
        if f.__name__ in function_name_set:
            raise AssertionError(f"Duplicated function name {f.__name__}")
        function_name_set.add(f.__name__)

        scripted_func = torch.jit.script(f)
        torch._C._jit_pass_inline(scripted_func.graph)

        for _ in range(2):
            torch._C._jit_pass_peephole(scripted_func.graph)
            torch._C._jit_pass_constant_propagation(scripted_func.graph)

        registry[str(aten_op._schema)] = scripted_func
        return f

    return decomposition_decorator


# TODO: replace torch.sigmoid -> aten.sigmoid


@register_decomposition(aten.var.correction)
def var_decomposition(
    input: Tensor,
    dim: list[int] | None = None,
    correction: Number | None = None,
    keepdim: bool = False,
) -> Tensor:
    # [FUNCTION DOC] Decomposition of variance with correction. Treats a missing
    # dim as all dimensions, computes the element count n (total elements, or the
    # product of the sizes of the listed dims), then the sum of squared
    # deviations from the mean divided by n minus correction (default 1). Raises
    # RuntimeError if correction is neither int nor float. Returns a Tensor.
    # KNOWN ISSUE: the divisor is clamped at 0 via the max builtin, so n not
    # exceeding correction divides by zero rather than yielding a clean error.
    if dim is None:
        dim_i: list[int] = []
        dim = dim_i

    if isinstance(dim, (tuple, list)) and len(dim) == 0:
        n = input.numel()
    else:
        n = 1
        for dim_i in dim:  # type: ignore[assignment]
            n *= input.shape[dim_i]  # type: ignore[call-overload]

    mean = aten.mean(input, dim, True)
    sub = input - mean
    sq = sub * sub
    sum = aten.sum(sq, dim, keepdim)

    if correction is None:
        denom = float(n - 1)
    else:
        if isinstance(correction, int):
            denom = float(n - correction)
        elif isinstance(correction, float):
            denom = float(n) - correction
        else:
            raise RuntimeError("correction must be int or float")

    # pyrefly: ignore [no-matching-overload]
    return sum / max(0, denom)


@register_decomposition(aten.var.default)
def var(input: Tensor, unbiased: bool = True) -> Tensor:
    # [FUNCTION DOC] Decomposition of variance with the unbiased flag; delegates
    # to var_decomposition with correction 1 if unbiased else 0 and all dims.
    return var_decomposition(input, correction=(1 if unbiased else 0))
