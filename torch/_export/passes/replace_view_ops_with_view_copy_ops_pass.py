# [MODULE DOC] Export pass replacing aliasing view ops with their functional view_copy counterparts, for
# backends needing pure functional graphs. Exports ReplaceViewOpsWithViewCopyOpsPass. Defines a
# table mapping the unsafe-view op to view_copy, plus helpers to detect view ops from a schema and find
# the matching view_copy overload. No import-time side effects beyond building the table.
# mypy: allow-untyped-defs

import torch
from torch._export.error import InternalError
from torch._export.pass_base import _ExportPassBaseDeprecatedDoNotUse
from torch._ops import HigherOrderOperator, OpOverload


__all__ = ["ReplaceViewOpsWithViewCopyOpsPass"]


_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS: dict[OpOverload, OpOverload] = {
    torch.ops.aten._unsafe_view.default: torch.ops.aten.view_copy.default,
}


def is_view_op(schema: torch._C.FunctionSchema) -> bool:
    # [FUNCTION DOC] Returns True if the op schema is a view: it has at least one argument and its first argument
    # carries alias info that is read-only (not a write). Returns False for zero-argument schemas. Pure.
    if len(schema.arguments) == 0:
        return False
    alias_info = schema.arguments[0].alias_info
    return (alias_info is not None) and (not alias_info.is_write)


def get_view_copy_of_view_op(schema: torch._C.FunctionSchema) -> OpOverload | None:
    # [FUNCTION DOC] Given a schema, returns the matching aten view_copy overload when the schema is a view op in the
    # aten namespace; otherwise returns None. The copy op name is the view op name plus a copy suffix,
    # and the overload name defaults to "default" when empty. Raises InternalError if the view_copy
    # packet or the overload does not exist. Reads the torch.ops.aten registry only.
    if is_view_op(schema) and schema.name.startswith("aten::"):
        view_op_name = schema.name.split("::")[1]
        view_op_overload = (
            schema.overload_name if schema.overload_name != "" else "default"
        )
        view_copy_op_name = view_op_name + "_copy"
        if not hasattr(torch.ops.aten, view_copy_op_name):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        view_copy_op_overload_packet = getattr(torch.ops.aten, view_copy_op_name)

        if not hasattr(view_copy_op_overload_packet, view_op_overload):
            raise InternalError(f"{schema.name} is missing a view_copy variant")

        return getattr(view_copy_op_overload_packet, view_op_overload)

    return None


class ReplaceViewOpsWithViewCopyOpsPass(_ExportPassBaseDeprecatedDoNotUse):
    # [CLASS DOC] Export pass (subclass of the deprecated export pass base) that retraces a graph, swapping each view
    # operator for its view_copy variant and the unsafe-view op for view_copy. Higher-order ops and
    # non-view ops pass through unchanged. Stateless; only call_operator is overridden.
    """
    Our backend expects pure functional operators. For efficiency
    purposes, we keep view ops around while functionalizing the exported
    program. This pass replaces view ops with view copy ops for backends that
    need AOT memory planning.
    """

    def call_operator(self, op, args, kwargs, meta):
        # [FUNCTION DOC] Pass hook invoked per operator call. Chooses the replacement op: the mapped functional op for
        # known non-functional ops, the op itself for higher-order ops, the view_copy overload for view
        # ops, else the original op. Forwards op, args, kwargs and meta to the base class and returns its
        # result. May raise InternalError (via the schema lookup) if a view op lacks a view_copy variant.
        if op in _NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS:
            return super().call_operator(
                (_NON_FUNCTIONAL_OPS_TO_FUNCTIONAL_OPS[op]), args, kwargs, meta
            )

        if isinstance(op, HigherOrderOperator):
            return super().call_operator(op, args, kwargs, meta)

        if view_copy_op := get_view_copy_of_view_op(op._schema):
            return super().call_operator(view_copy_op, args, kwargs, meta)

        return super().call_operator(op, args, kwargs, meta)
