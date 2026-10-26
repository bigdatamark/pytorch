# [MODULE DOC] Small FX graph utility module for the inductor fx_passes package.
# Defines BitsetAncestors, a precomputed transitive-ancestor index over FX nodes.
# No import-time side effects beyond imports.
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet


class BitsetAncestors:
    # [CLASS DOC] Precomputed transitive ancestor sets for a topologically sorted list of FX nodes.
    # Each node gets an index from its list position; its ancestors are stored as one big integer used
    # as a bitset (bit j set means node j is a transitive ancestor). State: per-node bitsets, a node to
    # index mapping, and the original node list. Immutable after construction; queries are read only.
    # Nodes not in the original list are treated as having no relationship to anything.
    """Precomputed transitive ancestor sets using Python arbitrary-precision ints.

    Each node gets an index i (0..N-1) from topological order. A node's
    ancestor set is a single Python ``int`` where bit j is set iff node j
    is a transitive ancestor. Python ``int`` is arbitrary-precision, so for
    N nodes each int is ~N/8 bytes internally (~N/64 machine words).

    The key advantage over ``dict[fx.Node, OrderedSet[fx.Node]]`` is that
    CPython implements ``int |= int`` as a C-level loop over machine words,
    making the transitive closure O(N^2 / 64) instead of O(N^2) in
    Python-level set operations.

    Example -- diamond graph (edges point downward)::

            a          index 0
           / \\
          b   c        index 1, 2
           \\ /
            d          index 3

    Ancestor bitsets::

        a: 0b0000 = 0   (no ancestors)
        b: 0b0001 = 1   (bit 0 -> a)
        c: 0b0001 = 1   (bit 0 -> a)
        d: 0b0111 = 7   (bits 0,1,2 -> a, b, c)

    How d's bitset is built (d has parents b and c)::

        b = 0
        # parent b (idx=1): set bit 1, merge b's ancestors
        b |= (1 << 1) | bits[1]   # b = 0b0010 | 0b0001 = 0b0011
        # parent c (idx=2): set bit 2, merge c's ancestors
        b |= (1 << 2) | bits[2]   # b = 0b0011 | 0b0100 | 0b0001 = 0b0111
        bits[3] = 0b0111

    Querying::

        ancestors = BitsetAncestors(nodes)
        ancestors.is_ancestor(a, d)    # (bits[3] >> 0) & 1 -> True
        ancestors.has_dep(a, d)  # True (a is ancestor of d)
        list(ancestors.iter_ancestors(d))  # [a, b, c] via bit-scan

    Iteration uses the x & -x trick to isolate the lowest set bit::

        bits = 0b0111
        bits & -bits = 0b0001  -> idx 0 -> yield a, clear: 0b0110
        bits & -bits = 0b0010  -> idx 1 -> yield b, clear: 0b0100
        bits & -bits = 0b0100  -> idx 2 -> yield c, clear: 0b0000

    Args:
        nodes: topologically sorted list of FX nodes.
        extra_inputs: optional additional edges beyond the FX graph
            (e.g. hiding-interval deps in the overlap bucketer).
    """

    def __init__(
        self,
        nodes: list[fx.Node],
        extra_inputs: dict[fx.Node, OrderedSet[fx.Node]] | None = None,
    ):
        # [FUNCTION DOC] Builds the bitsets in one pass over nodes (which must be in topological order, so every input
        # precedes its user). For each node, ORs in each input's own bit plus that input's ancestor bits.
        # Inputs come from the node's FX input nodes and from the optional extra_inputs mapping of extra
        # edges; inputs not present in the node list are ignored. Stores results on the instance; no raises.
        n = len(nodes)
        node_to_idx: dict[fx.Node, int] = {nd: i for i, nd in enumerate(nodes)}
        bits = [0] * n
        extra = extra_inputs or {}

        for i, node in enumerate(nodes):
            b = 0
            for inp in node._input_nodes:
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            for inp in extra.get(node, ()):
                j = node_to_idx.get(inp)
                if j is not None:
                    b |= (1 << j) | bits[j]
            bits[i] = b

        self._bits = bits
        self._node_to_idx = node_to_idx
        self._idx_to_node = nodes

    def is_ancestor(self, ancestor: fx.Node, descendant: fx.Node) -> bool:
        # [FUNCTION DOC] Return True if ancestor is a transitive ancestor of descendant. Returns False if either node
        # is unknown to this index. Constant time bit test; does not treat a node as its own ancestor.
        """O(1) test: is ``ancestor`` a transitive ancestor of ``descendant``?"""
        anc_idx = self._node_to_idx.get(ancestor)
        desc_idx = self._node_to_idx.get(descendant)
        if anc_idx is None or desc_idx is None:
            return False
        return bool((self._bits[desc_idx] >> anc_idx) & 1)

    def has_dep(self, n1: fx.Node, n2: fx.Node) -> bool:
        # [FUNCTION DOC] Return True if either node is a transitive ancestor of the other (symmetric dependency test).
        """Check if either node is an ancestor of the other."""
        return self.is_ancestor(n1, n2) or self.is_ancestor(n2, n1)

    def get_ancestor_bits(self, node: fx.Node) -> int:
        # [FUNCTION DOC] Return the raw ancestor bitset integer for node. Raises a key error if the node is unknown.
        """Return the raw ancestor bitset for ``node``."""
        return self._bits[self._node_to_idx[node]]

    def node_bit(self, node: fx.Node) -> int:
        # [FUNCTION DOC] Return an integer with only this node's own index bit set. Raises a key error if unknown.
        """Return the single-bit mask ``1 << idx`` for ``node``."""
        return 1 << self._node_to_idx[node]

    def ancestors_intersect(self, node: fx.Node, mask: int) -> bool:
        # [FUNCTION DOC] Return True if any ancestor of node is also set in the given bit mask. Returns False for an
        # unknown node. Pure query, no mutation.
        """Check if any ancestor of ``node`` is set in ``mask``."""
        idx = self._node_to_idx.get(node)
        if idx is None:
            return False
        return bool(self._bits[idx] & mask)

    def iter_ancestors(self, node: fx.Node):
        # [FUNCTION DOC] Generator yielding every ancestor node of the given node, in ascending index order, by repeatedly
        # extracting the lowest set bit of a local copy of the bitset. Raises a key error if node is unknown.
        """Yield all ancestors of ``node`` via lowest-bit scan."""
        bits = self._bits[self._node_to_idx[node]]
        idx_to_node = self._idx_to_node
        while bits:
            idx = (bits & -bits).bit_length() - 1
            yield idx_to_node[idx]
            bits &= bits - 1
