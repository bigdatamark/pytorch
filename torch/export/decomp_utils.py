# mypy: allow-untyped-defs
# [MODULE DOC] Provides CustomDecompTable, the decomposition table type used by export's run_decompositions, plus
# PRESERVED_ATEN_CIA_OPS, the aten composite-implicit ops kept undecomposed by default. Exports CustomDecompTable.
# Import-time side effects: resolves several torch.ops.aten operator overloads.
from collections.abc import Callable

import torch
from torch._export.utils import (
    _collect_all_valid_cia_ops,
    _collect_all_valid_cia_ops_for_aten_namespace,
    _get_decomp_for_cia,
    _is_aten_op,
)


__all__ = ["CustomDecompTable"]


"""
Core ATen ops with Composite Implicit Autograd dispatch that should be excluded from decomposition
by default. The decomposition logic should eventually exclude all core-tagged CIA ops, but until all
backends are ready, this list allows opt-in one at a time.
"""
PRESERVED_ATEN_CIA_OPS = {
    torch.ops.aten.upsample_bilinear2d.vec,
    torch.ops.aten.upsample_nearest2d.vec,
    # NB: don't use the C++ decomp, because it is not functional!
    torch.ops.aten.silu_backward.default,
    torch.ops.aten.mish_backward.default,
    torch.ops.aten._fused_rms_norm.default,
}


class CustomDecompTable(dict[torch._ops.OperatorBase, Callable]):
    # [CLASS DOC] Dict subclass used as the decomposition table for export. State: decomp_table (the real dict of
    # op to decomposition, preloaded with core aten decomps), deleted_custom_ops (set of custom ops the user popped
    # that are not yet materialized), has_materialized (flag). Deleting an op means preserving it, so custom ops that
    # may load into the dispatcher later are tracked lazily. Reads materialize first; writes do not.
    # KNOWN ISSUE: update() writes straight to decomp_table without clearing deleted_custom_ops, unlike __setitem__.
    """
    This is a custom dictionary that is specifically used for handling decomp_table in export.
    The reason we need this is because in the new world, you can only *delete* an op from decomp
    table to preserve it. This is problematic for custom ops because we don't know when the custom
    op will actually be loaded to the dispatcher. As a result, we need to record the custom ops operations
    until we really need to materialize it (which is when we run decomposition pass.)

    Invariants we hold are:
     1. All aten decomp is loaded at the init time
     2. We materialize ALL ops when user ever reads from the table to make it more likely
        that dispatcher picks up the custom op.
     3. If it is write operation, we don't necessarily materialize
     4. We load the final time during export, right before calling run_decompositions()

    """

    def __init__(self):
        # [FUNCTION DOC] Builds the table: loads core aten post-autograd decomps, adds decomps for composite-implicit
        # aten ops not in PRESERVED_ATEN_CIA_OPS, and starts with no pending deleted custom ops and has_materialized false.
        super().__init__()
        from torch._decomp import _core_aten_decompositions_post_autograd

        # For aten ops, we load them up in the beginning
        self.decomp_table = _core_aten_decompositions_post_autograd()

        for op in _collect_all_valid_cia_ops_for_aten_namespace():
            if op not in PRESERVED_ATEN_CIA_OPS and op not in self.decomp_table:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        # This is to track the *pending* deleted custom ops that haven't been materialized yet
        self.deleted_custom_ops = set()
        # When this is true, there shouldn't be any pending operations in the table.
        self.has_materialized = False

    def __getitem__(self, key):
        # [FUNCTION DOC] Materializes pending custom ops, then returns the decomposition for key (KeyError if absent).
        self._materialize_if_needed()
        return self.decomp_table.__getitem__(key)

    def __setitem__(self, key, value) -> None:
        # [FUNCTION DOC] Stores a decomposition for key without materializing; removes key from the pending deleted set.
        self.decomp_table.__setitem__(key, value)

        if key in self.deleted_custom_ops:
            self.deleted_custom_ops.remove(key)

    def keys(self):
        # [FUNCTION DOC] Materializes, then returns the keys view of the underlying table.
        self._materialize_if_needed()
        return self.decomp_table.keys()

    def __delitem__(self, key) -> None:
        # [FUNCTION DOC] Deletes key by delegating to pop; the popped value is discarded.
        self.pop(key)

    def update(self, other_dict):  # type: ignore[override]
        # [FUNCTION DOC] Copies every item of other_dict into the underlying table without materializing.
        for k, v in other_dict.items():
            self.decomp_table.__setitem__(k, v)

    def __missing__(self, key) -> bool:
        # [FUNCTION DOC] Returns the negation of membership for key.
        # KNOWN ISSUE: __missing__ should return or raise a value for the missing key; returning a bool is odd and
        # it is never reached because __getitem__ is overridden.
        return not self.__contains__(key)

    def __contains__(self, key) -> bool:
        # [FUNCTION DOC] Materializes, then returns whether key is in the table.
        self._materialize_if_needed()
        return self.decomp_table.__contains__(key)

    def __len__(self) -> int:
        # [FUNCTION DOC] Materializes, then returns the number of entries.
        self._materialize_if_needed()
        return self.decomp_table.__len__()

    def __iter__(self):
        # [FUNCTION DOC] Materializes, then returns an iterator over the keys.
        self._materialize_if_needed()
        return self.decomp_table.__iter__()

    def __reversed__(self):
        # [FUNCTION DOC] Materializes, then returns a reverse iterator over the keys.
        self._materialize_if_needed()
        return self.decomp_table.__reversed__()

    def copy(self) -> "CustomDecompTable":
        # [FUNCTION DOC] Returns a new table (freshly built) whose decomp_table, deleted_custom_ops and has_materialized
        # are copies of this one's.
        new_dict = CustomDecompTable()
        new_dict.decomp_table = self.decomp_table.copy()
        new_dict.deleted_custom_ops = self.deleted_custom_ops.copy()
        new_dict.has_materialized = self.has_materialized
        return new_dict

    def pop(self, *args):
        # [FUNCTION DOC] Removes key (first arg) and returns its decomposition; optional second arg is a default returned
        # on KeyError. Aten ops are popped directly. Custom ops are recorded as deleted so later materialization respects
        # it; popping an op absent from the table pretends it existed and returns its CIA decomposition. Raises KeyError
        # if already deleted. Returns None for any other argument count.
        def _pop_if_can(key):
            # [FUNCTION DOC] HELPER: Performs the pop for a single key as described for the enclosing pop; mutates
            # decomp_table and deleted_custom_ops; raises KeyError if the key was already deleted.
            if _is_aten_op(key):
                return self.decomp_table.pop(key)

            if key in self.decomp_table:
                # Even if we materialized it, we should add it to the deleted
                # custom ops list so that when we materialize next time,
                # we should respect user's intention.
                self.deleted_custom_ops.add(key)
                return self.decomp_table.pop(key)

            if key in self.deleted_custom_ops:
                raise KeyError(f"{key} doesn't exist in the table")

            self.deleted_custom_ops.add(key)
            # We would come here when user pops off something that is
            # not in the table. In this case, we just pretend that it
            # was in the table.
            return _get_decomp_for_cia(key)

        if len(args) == 1:
            return _pop_if_can(args[0])

        if len(args) == 2:
            try:
                return _pop_if_can(args[0])
            except KeyError:
                return args[1]

    def items(self):
        # [FUNCTION DOC] Materializes, then returns the items view of the underlying table.
        self._materialize_if_needed()
        return self.decomp_table.items()

    def materialize(self) -> dict[torch._ops.OperatorBase, Callable]:
        # [FUNCTION DOC] Adds decomps for all valid non-aten composite-implicit ops that are not already present and not
        # pending deletion, sets has_materialized, clears the deleted set, and returns a plain dict copy of the table.
        for op in _collect_all_valid_cia_ops():
            if _is_aten_op(op):
                continue
            elif op in self.decomp_table:
                continue
            elif op not in self.deleted_custom_ops:
                self.decomp_table[op] = _get_decomp_for_cia(op)

        self.has_materialized = True
        self.deleted_custom_ops = set()
        return {**self.decomp_table}

    def _materialize_if_needed(self) -> None:
        # [FUNCTION DOC] HELPER: Calls materialize once, if it has not run yet.
        if not self.has_materialized:
            self.materialize()
