# mypy: allow-untyped-defs
# [MODULE DOC] Graph pass that removes duplicate symbolic-size nodes (SymInt/SymFloat/SymBool values) from an FX graph.
# Provides a hashing wrapper keyed on the underlying sympy expression, a dictionary wrapper using it, and dedupe_symints.
# No import-time side effects.
from dataclasses import dataclass
from typing import Any

import torch
from torch import SymBool, SymFloat, SymInt
from torch.types import py_sym_types
from torch.utils._ordered_set import OrderedSet


@dataclass
class _SymExprHash:
    # [CLASS DOC] HELPER: Dataclass wrapper holding one symbolic value; hashing and equality are based on the sympy expression of the symbolic node,
    # so two distinct symbolic objects with the same expression compare equal and collide in a mapping.
    """
    Hash for a py_sym_types that will use the underlying sympy expression
    """

    sym_obj: SymInt | SymFloat | SymBool

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Return a hash combining the symbolic value's Python type with its sympy expression.
        return hash((type(self.sym_obj), self.sym_obj.node.expr))

    def __eq__(self, value) -> bool:
        # [FUNCTION DOC] HELPER: Equality: True only when the other object is also a wrapper and both sympy expressions are equal; otherwise False.
        if not isinstance(value, _SymExprHash):
            return False
        return self.sym_obj.node.expr == value.sym_obj.node.expr


class _SymHashingDict:
    # [CLASS DOC] HELPER: Dictionary wrapper that transparently wraps symbolic-type keys in the expression-based hash wrapper (other keys are used as-is).
    # State: one inner dict (sym_hash_dict). Supports set, get-by-key, membership and a get with default.
    """
    Wrapper around a dictionary that will convert sym types to hash with _SymExprHash and reuse
    existing sym proxies.

    SymPy hash is not always reliable so optimistically hash sympy expression, and if those fail,
    fallback to symnodes.
    """

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize the empty inner dictionary.
        self.sym_hash_dict = {}

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Store a value under the (possibly wrapped) key.
        self.sym_hash_dict.__setitem__(self._wrap_to_sym_expr_hash(key), value)

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Look up the (possibly wrapped) key; raises KeyError if absent.
        return self.sym_hash_dict[self._wrap_to_sym_expr_hash(key)]

    def __contains__(self, key):
        # [FUNCTION DOC] HELPER: Return whether the (possibly wrapped) key is present.
        return self._wrap_to_sym_expr_hash(key) in self.sym_hash_dict

    def get(self, key, default=None):
        # [FUNCTION DOC] HELPER: Look up the (possibly wrapped) key, returning the supplied default (None) when absent.
        return self.sym_hash_dict.get(self._wrap_to_sym_expr_hash(key), default)

    def _wrap_to_sym_expr_hash(self, key):
        # [FUNCTION DOC] HELPER: Wrap the key in the expression-hash wrapper if it is a symbolic type; otherwise return it unchanged.
        return _SymExprHash(key) if isinstance(key, py_sym_types) else key


def dedupe_symints(graph: torch.fx.Graph):
    # [FUNCTION DOC] Mutate the graph in place so that nodes producing equal symbolic values reuse one node.
    # Walks nodes in order using each node's recorded value; non-symbolic nodes are skipped. Placeholders are always registered as canonical sources.
    # A later node whose symbolic expression matches a registered node has its uses redirected to the existing node and is erased.
    # Otherwise, if all its inputs are themselves canonical (derived from placeholders), it is registered as canonical too.
    # Only deduplicates against input-derived nodes to avoid introducing forward-to-backward dependencies. Returns nothing.
    """
    Dedupes sym ints in the graph to nodes are resolvable to symint graph inputs.

    We only dedupe from graph inputs to avoid adding a potential dependency in the forward
    from the backward.

    """

    sym_dict = _SymHashingDict()
    resolvable_from_input_symints = OrderedSet[Any]()

    for node in graph.nodes:
        val = node.meta.get("val", None)
        if val is None or not isinstance(val, py_sym_types):
            continue

        if node.op == "placeholder":
            resolvable_from_input_symints.add(node)
            sym_dict[val] = node
        elif existing_node := sym_dict.get(val):
            node.replace_all_uses_with(existing_node)
            graph.erase_node(node)
        elif all(n in resolvable_from_input_symints for n in node.all_input_nodes):
            sym_dict[val] = node
            resolvable_from_input_symints.add(node)
