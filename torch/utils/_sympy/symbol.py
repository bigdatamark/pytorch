# [MODULE DOC] Canonical naming conventions for sympy symbols used by symbolic shapes and
# inductor: an enum of symbol kinds, a mapping from kind to name prefix, and helpers to create
# and classify symbols by prefix. No import-time side effects.
"""
This file contains canonical definitions for our symbol naming conventions,
across torch.fx.experimental.symbolic_shapes and torch._inductor.  The
intention is:

1. To make it easily greppable where all the sites we use a prefix are
2. Make it possible to easily tell if we can introduce a new prefix without
   introducing a conflict

You can occasionally test if prefixes have been hardcoded by renaming prefixes
in this file and seeing what breaks.
"""

from collections.abc import Iterable
from enum import auto, Enum

import sympy


class SymT(Enum):
    # [CLASS DOC] Enum of symbol kinds (sizes, unbacked values, inductor temporaries, index
    # variables, block dimensions, and so on); each member maps to a name prefix in
    # prefix_str.
    SIZE = auto()
    FLOAT = auto()
    UNBACKED_INT = auto()
    UNBACKED_FLOAT = auto()
    # Inductor: The intermediates in inner_fn tmp0, one generated per ops call.
    # If one of these shows up in an indexing expression, that means an
    # indirect load is happening.
    TMP = auto()
    # Inductor: Placeholder variable that is later replaced with TMP
    INDIRECT = auto()
    # Inductor: Some size expressions are replaced with a precomputed size ps0
    # which is computed host side, and then directly reused in the kernel, so
    # we don't repeatedly recompute it on device.
    PRECOMPUTED_SIZE = auto()
    # Inductor: An indexing variable i0 in loops IR which ranges over non-reduced
    # dim in the loop
    INDEX = auto()
    # Inductor: A reduction indexing (r0, r1) variables in loops IR which ranges over
    # reduced dim(s) in the loop
    R0_INDEX = auto()
    R1_INDEX = auto()
    # Inductor: In templated kernels torch._inductor.kernel, we have a hook to
    # store the final output and append epilogue fusions.  To do this, we must
    # know what the indexes the outputs range over.  NB: These will also
    # advertise as INDEX, this is... probably OK?
    TEMPLATE_INDEX = auto()
    # Inductor: iteration domain for blockIdx.x/blockIdx.y
    XBLOCK = auto()
    YBLOCK = auto()
    ZBLOCK = auto()
    # Inductor: this is used solely for dynamic_reshape_indexer
    VIEW = auto()
    # Alternate (non-modular) indexing used in halide kernels
    HALIDE = auto()


# Invariant: there must not be a prefix which is a prefix of another string,
# as this introduces ambiguity
prefix_str = {
    SymT.SIZE: "s",  # integer
    SymT.UNBACKED_INT: "u",  # integer
    # Prefix z here is chosen to avoid false aliasing in symbol_is_type test
    # DO NOT add a "z" type.  You also need to avoid conflicts on these
    # prefixes but this is somewhat easier to manage
    SymT.FLOAT: "zf",
    SymT.UNBACKED_FLOAT: "zuf",
    SymT.TMP: "tmp",
    SymT.PRECOMPUTED_SIZE: "ps",
    SymT.INDEX: "i",
    SymT.R0_INDEX: "r0_",
    SymT.R1_INDEX: "r1_",
    SymT.TEMPLATE_INDEX: "idx",
    SymT.XBLOCK: "x",
    SymT.YBLOCK: "y",
    SymT.ZBLOCK: "z",
    SymT.INDIRECT: "indirect",  # false aliasing?
    SymT.VIEW: "view",
    SymT.HALIDE: "h",
}


def make_symbol(prefix: SymT, idx: int, **kwargs: bool | None) -> sympy.Symbol:
    # [FUNCTION DOC] Create a sympy Symbol named by the prefix for the given kind followed by
    # the integer idx; extra keyword assumptions are passed to the Symbol. Returns the symbol.
    # TODO: maybe put the assumptions here directly
    return sympy.Symbol(f"{prefix_str[prefix]}{idx}", **kwargs)


# This type is a little wider than it should be, because free_symbols says
# that it contains Basic, rather than Symbol
def symbol_is_type(sym: sympy.Basic, prefix: SymT | Iterable[SymT]) -> bool:
    # [FUNCTION DOC] Check whether the lowercased name of symbol sym starts with the prefix of
    # the given kind, or any of several kinds. Raises AssertionError if sym is not a sympy
    # Symbol.
    # KNOWN ISSUE: the stated invariant that no prefix is a prefix of another is violated (for
    #   example the prefix for ZBLOCK is a prefix of the float prefixes, and the INDEX prefix
    #   is a prefix of the indirect and template index prefixes), so matches can alias.
    if not isinstance(sym, sympy.Symbol):
        raise AssertionError("expected sympy.Symbol")
    name_str = sym.name.lower()  # Match capitalized names like XBLOCK, RBLOCK
    if isinstance(prefix, SymT):
        return name_str.startswith(prefix_str[prefix])
    else:
        return name_str.startswith(tuple(prefix_str[p] for p in prefix))


def free_symbol_is_type(e: sympy.Expr, prefix: SymT | Iterable[SymT]) -> bool:
    # [FUNCTION DOC] Return True if any free symbol of expression e matches the given kind or
    # kinds (via symbol_is_type).
    return any(symbol_is_type(v, prefix) for v in e.free_symbols)
