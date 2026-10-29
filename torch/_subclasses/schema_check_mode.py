# [MODULE DOC]
# Defines SchemaCheckMode, a TorchDispatchMode that validates operator schemas at
# runtime: it records called op names and checks that inputs declared immutable are
# not mutated, that outputs only alias inputs/outputs the schema allows, and that
# non-mutable ops do not directly return their inputs. Also holds the Mutation and
# Aliasing record tuples, short aliases for the C++ schema-info classes, and two
# small tensor-list helpers (kept here to avoid a numpy dependency in common_utils).
# Import-time side effect: binds module-level aliases to torch._C schema classes.
from __future__ import annotations

from copy import deepcopy
from itertools import combinations
from typing import Any, NamedTuple, TYPE_CHECKING

import torch
from torch.fx.operator_schemas import _normalize_function_or_error
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map


if TYPE_CHECKING:
    from collections.abc import Iterable

    from torch._ops import OpOverload


class Mutation(NamedTuple):
    # [CLASS DOC] Record of one observed mutation: operator name and the (standardized) name of the argument that was mutated.
    op_name: str
    arg_name: str


class Aliasing(NamedTuple):
    # [CLASS DOC] Record of one observed alias: operator name, argument name, and a label such as output_0 naming the aliased output.
    op_name: str
    arg_name: str
    output_number: str


# Simplified naming for C++ classes
# pyrefly: ignore [missing-attribute]
SchemaArgument = torch._C._SchemaArgument
# pyrefly: ignore [missing-attribute]
SchemaArgType = torch._C._SchemaArgType
SchemaInfo = torch._C._SchemaInfo

# This TorchDispatchMode Subclass is used to verify op schemas
# This TorchDispatchMode Subclass currently:
#  - Records the called ops
#  - Checks for mutations on all inputs
#  - Checks for aliasing on all inputs


# move these 2 functions here to avoid numpy dependency in testing/_internal/common_utils.py


def is_iterable_of_tensors(iterable: Iterable[Any]) -> bool:
    # [FUNCTION DOC] Returns True only if the argument is a non-empty iterable whose
    # elements are all tensors. A bare tensor, an empty iterable, or a non-iterable
    # (TypeError on len/iteration) yields False. No side effects.
    # Tensor itself is iterable so we check this first
    if isinstance(iterable, torch.Tensor):
        return False
    try:
        # pyrefly: ignore[bad-argument-type]
        if len(iterable) == 0:
            return False
        for t in iter(iterable):
            if not isinstance(t, torch.Tensor):
                return False
    except TypeError:
        return False
    return True


def clone_inputs(args: Iterable[Any]) -> list[Any]:
    # [FUNCTION DOC] Returns a new list where each tensor is replaced by a detached
    # clone, each iterable of tensors by a list of detached clones, and any other
    # value is passed through unchanged.
    inputs: list[Any] = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        else:
            inputs.append(arg)

    return inputs


class SchemaCheckMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that checks every intercepted op against its declared
    # schema. State: ops (names of called ops), mutated (Mutation records for inputs
    # actually mutated and allowed to be), aliasing (Aliasing records for allowed
    # input/output aliasing). Violations raise RuntimeError from __torch_dispatch__.
    # reset_cache clears the records; display_ops prints op names.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes the three empty record lists (ops, mutated, aliasing).
        # Information recorded for testing purposes. For example:
        #  - incorrect schemas
        #  - overly conservative schemas
        self.ops: list[str] = []
        self.mutated: list[Mutation] = []
        self.aliasing: list[Aliasing] = []

    def reset_cache(self) -> None:
        # [FUNCTION DOC] Empties the recorded ops, mutated and aliasing lists in place.
        self.ops.clear()
        self.mutated.clear()
        self.aliasing.clear()

    def display_ops(self) -> None:
        # [FUNCTION DOC] Prints the recorded op names to stdout, comma separated.
        print(*self.ops, sep=",")

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type[Any], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        # [FUNCTION DOC] Runs the op while verifying its schema. Records the op name,
        # normalizes args to keyword form, snapshots clones and stride/storage metadata
        # of inputs, runs the op, then for each non-None input checks: output aliasing
        # is permitted by the schema (except unsafe view/split ops), non-mutable ops do
        # not directly return an input (lift ops exempt), and mutation only occurs on
        # mutable args. Also checks pairs of outputs for unexpected aliasing.
        # Returns the op result. Raises RuntimeError on any violation; appends to
        # self.aliasing and self.mutated for permitted cases.
        def bitwise_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
            # [FUNCTION DOC] HELPER: NaN-tolerant equality; exact equal for quantized tensors, allclose with equal_nan otherwise.
            if lhs.is_quantized:
                # TODO: This is only OK if can't have NaN quantized; idk if
                # this is actually true
                return torch.equal(lhs, rhs)
            else:
                return torch.allclose(lhs, rhs, equal_nan=True)

        def has_mutated(
            before: Any, after: Any, md: tuple[tuple[int, ...], int] | None
        ) -> bool:
            # [FUNCTION DOC] HELPER: True if before/after are plain non-sparse-CSR tensors, metadata md exists, and
            # size, values, stride or storage identity differ from the snapshot. Otherwise False.
            are_tensors = type(before) is torch.Tensor and type(after) is torch.Tensor
            if (
                are_tensors
                and before.layout != torch.sparse_csr
                and after.layout != torch.sparse_csr
            ):
                return md is not None and not (
                    before.size() == after.size()
                    and bitwise_equal(before, after)
                    and md[0] == after.stride()
                    and md[1] == after._typed_storage()._cdata
                )
            return False

        def has_aliased(lhs: Any, rhs: Any) -> bool:
            # [FUNCTION DOC] HELPER: True if the two values overlap in memory; values that cannot be inspected give False; other errors are re-raised.
            try:
                # pyrefly: ignore [missing-attribute]
                return torch._C._overlaps(lhs, rhs)
            except Exception as exception:
                if str(exception).startswith("Cannot inspect value of type "):
                    return False
                else:
                    raise exception

        def standardize_name(name: str) -> str:
            # [FUNCTION DOC] HELPER: maps the schema argument name self to input; others unchanged.
            return name if name != "self" else "input"

        def unwrap(e: Any) -> Any:
            # [FUNCTION DOC] HELPER: for tensor subclass wrappers returns the inner elem tensor (or the value itself if no elem); other values unchanged.
            if isinstance(e, torch.Tensor) and type(e) is not torch.Tensor:
                try:
                    # pyrefly: ignore[missing-attribute]
                    return e.elem
                except AttributeError:
                    return e
            return e

        def parse_metadata(e: Any) -> tuple[tuple[int, ...], int] | None:
            # [FUNCTION DOC] HELPER: for a tensor (or the elem of a subclass wrapper) returns a copy of its stride and the
            # storage identity; returns None for non-tensors, sparse CSR tensors, or wrappers lacking elem.
            if isinstance(e, torch.Tensor):
                if type(e) is not torch.Tensor:
                    try:
                        # pyrefly: ignore[missing-attribute]
                        current = e.elem
                        return (
                            deepcopy(current.stride()),
                            current._typed_storage()._cdata,
                        )
                    except AttributeError:
                        return None
                # Sparse CSR tensors do not have strides or storage
                elif e.layout != torch.sparse_csr:
                    return (deepcopy(e.stride()), e._typed_storage()._cdata)
            return None

        self.ops.append(func._schema.name)

        # Clone and process arguments and outputs
        pre_arguments = _normalize_function_or_error(
            func, args, kwargs, normalize_to_only_use_kwargs=True
        ).kwargs

        c_p_args = dict(zip(pre_arguments.keys(), clone_inputs(pre_arguments.values())))
        cloned_arguments = {
            name: tree_map(unwrap, c_p_args.get(name)) for name in c_p_args
        }
        cloned_metadata = {
            name: [
                parse_metadata(a) for a in pytree.tree_leaves(pre_arguments.get(name))
            ]
            for name in pre_arguments
        }

        out = func(*args, **kwargs)
        arguments = {
            name: tree_map(unwrap, pre_arguments.get(name)) for name in pre_arguments
        }
        tuple_out = out if isinstance(out, tuple) else (out,)
        tuple_out = tree_map(unwrap, tuple_out)

        schema_info = SchemaInfo(func._schema)
        # pyrefly: ignore [missing-attribute]
        schema_info.add_argument_values(pre_arguments)

        # Process arguments with outputs
        for i in range(len(func._schema.arguments)):
            arg = func._schema.arguments[i]
            name = standardize_name(arg.name)
            if arguments.get(name) is not None:
                before = cloned_arguments.get(name)
                md = cloned_metadata.get(name)
                after = arguments.get(name)
                for j in range(len(tuple_out)):
                    # aten::_unsafe_view is intended to have incorrect aliasing notation (hence unsafe)
                    unsafe_ops = ("aten::_unsafe_view", "aten::unsafe_split")
                    if (
                        has_aliased(tuple_out[j], after)
                        and func._schema.name not in unsafe_ops
                    ):
                        # pyrefly: ignore [missing-attribute]
                        if not schema_info.may_contain_alias(
                            SchemaArgument(SchemaArgType.output, j),
                            SchemaArgument(SchemaArgType.input, i),
                        ):
                            raise RuntimeError(
                                f"Argument {name} is not defined to alias output but was aliasing"
                            )
                        else:
                            self.aliasing.append(
                                Aliasing(func._schema.name, name, f"output_{j}")
                            )
                    if after is tuple_out[j] and isinstance(after, torch.Tensor):
                        # Only mutable ops e.g. (add_, add.out) are allowed to directly return inputs.
                        if not schema_info.is_mutable(
                            SchemaArgument(SchemaArgType.input, i)
                        ) and func not in [
                            torch.ops.aten.lift.default,
                            torch.ops.aten.lift_fresh.default,
                        ]:
                            raise RuntimeError(
                                f"""\
Dispatcher operators below autograd are not allowed to directly return inputs.
However, we found that `outputs[{str(j)}] is {name}"""
                            )
                if md is not None and any(
                    has_mutated(a, b, c)
                    for a, b, c in zip(
                        pytree.tree_leaves(before), pytree.tree_leaves(after), md
                    )
                ):
                    if not schema_info.is_mutable(
                        SchemaArgument(SchemaArgType.input, i)
                    ):
                        raise RuntimeError(
                            f"Argument {name} is not defined as mutable but was mutated"
                        )
                    else:
                        self.mutated.append(Mutation(func._schema.name, name))

        # Aliasing between outputs
        for i, j in combinations(range(len(func._schema.returns)), 2):
            if has_aliased(tuple_out[i], tuple_out[j]):
                # pyrefly: ignore [missing-attribute]
                if not schema_info.may_contain_alias(
                    SchemaArgument(SchemaArgType.output, i),
                    SchemaArgument(SchemaArgType.output, j),
                ):
                    raise RuntimeError(f"Outputs {i} and {j} alias unexpectedly")

        return out
