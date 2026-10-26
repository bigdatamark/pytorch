# mypy: allow-untyped-defs
# [MODULE DOC]
# Symbolic analysis of index expressions across vectorized CuteDSL lanes.
# Classifies an expression as lane-uniform, contiguous at an aligned power-of-two
# width, or neither, and decomposes affine expressions into stride and base.
# Uses the current graph's sizevars via V. Exports LaneExprInfo,
# classify_lane_expr, aligned_contiguous_width, lane_group_start and
# decompose_affine_lane_expr. No import-time side effects.

import dataclasses

import sympy

from torch.utils._ordered_set import OrderedSet

from ...virtualized import V


@dataclasses.dataclass(frozen=True, slots=True)
class LaneExprInfo:
    # [CLASS DOC]
    # Immutable result record of lane analysis with three fields: whether all
    # lanes share one value, whether lanes are consecutive and aligned for the
    # requested width, and the largest proven aligned contiguous width (or None).
    """How an index expression changes across vectorized lanes.

    A lane is one element of a vectorized CuteDSL computation. For a 32-lane
    mask/load, lane 0 is the first element in the group and lane 31 is the last.

    Attributes:
        is_uniform: All lanes evaluate to the same value, e.g. for flex-flash``b_idx``.
        is_contiguous: Lanes evaluate to consecutive aligned values for the
            requested width, e.g. ``kv_idx + lane`` when lane 0 is aligned.
        contiguous_width: Largest aligned contiguous power-of-two width proven,
            or None if no vector-width contiguity is proven.
    """

    is_uniform: bool
    is_contiguous: bool
    contiguous_width: int | None


def classify_lane_expr(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    uniform_symbols: OrderedSet[sympy.Symbol] | None = None,
) -> LaneExprInfo:
    # [FUNCTION DOC]
    # Classifies how an index expression varies across the lane variable for a
    # requested vector width. A width of 1 or less, or an expression whose free
    # symbols are all in the optional uniform set, is reported as uniform with
    # no contiguity. Otherwise it simplifies the expression, asks sizevars for
    # lane contiguity, finds the largest aligned contiguous width, and returns
    # uniform/contiguous flags (contiguous requires stride one and the full
    # requested width). Requires an active graph context.
    """Classify how ``expr`` varies across ``lane_var`` vector lanes.

    ``max_width`` is the requested vector width. We start with this as the
    maximum width to check and then decrease in powers of two trying to find a width
    that satisfies contiguity across lanes. We cap since we are trying to get vectorized loads
    which are at 128 and 256(B200) bits so no need to check higher.

    For example a ``lane_var = kv_idx`` and ``max_width = 8``:

    - No dependency on lane var: ``q_idx`` returns
      ``LaneExprInfo(is_uniform=True, is_contiguous=False, contiguous_width=None)``.
    - ``kv_idx`` at an 8-aligned group start returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=True, contiguous_width=8)``.
    - ``kv_idx + 4`` at a 4-aligned but not 8-aligned group start returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=False, contiguous_width=4)``.
    - ``kv_idx * 2`` returns
      ``LaneExprInfo(is_uniform=False, is_contiguous=False, contiguous_width=None)``.
    """
    if max_width <= 1:
        return LaneExprInfo(True, False, None)

    semantic_expr = V.graph.sizevars.simplify(expr)
    if uniform_symbols is not None and semantic_expr.free_symbols <= uniform_symbols:
        return LaneExprInfo(True, False, None)

    lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(semantic_expr, lane_var)
    contiguous_width = aligned_contiguous_width(
        semantic_expr,
        lane_var,
        max_width=max_width,
        lane_contiguity=lane_contiguity,
    )
    return LaneExprInfo(
        is_uniform=lane_contiguity.is_uniform_for(max_width),
        is_contiguous=lane_contiguity.stride == 1 and contiguous_width == max_width,
        contiguous_width=contiguous_width,
    )


def aligned_contiguous_width(
    expr: sympy.Expr,
    lane_var: sympy.Symbol,
    *,
    max_width: int,
    lane_contiguity=None,
) -> int | None:
    # [FUNCTION DOC]
    # Searches widths from max_width halving down to 2 for the first width for
    # which the lanes are contiguous, the lane-0 base is statically a multiple
    # of the width, and the base is statically non-negative. Returns that width
    # or None. Computes lane contiguity if not supplied. Raises AssertionError
    # if max_width is not a power of two of at least 2.
    """Return the largest aligned power-of-two contiguous lane width."""
    if not (max_width >= 2 and max_width.bit_count() == 1):
        raise AssertionError(f"max_width must be a power of two >= 2, got {max_width}")
    if lane_contiguity is None:
        lane_contiguity = V.graph.sizevars.analyze_lane_contiguity(expr, lane_var)
    start_expr = lane_group_start(expr, lane_var)
    width = max_width
    while width >= 2:
        if (
            lane_contiguity.is_contiguous_for(width)
            and V.graph.sizevars.statically_known_multiple_of(start_expr, width)
            and V.graph.sizevars.statically_known_geq(start_expr, 0)
        ):
            return width
        width //= 2
    return None


def lane_group_start(expr: sympy.Expr, lane_var: sympy.Symbol) -> sympy.Expr:
    # [FUNCTION DOC]
    # Returns the simplified value of the expression with the lane variable
    # substituted by zero, i.e. the first lane's value used for alignment checks.
    """Return the lane-group base value by evaluating ``expr`` at lane 0.

    For example, if ``expr = base + lane_var``, this returns ``base``. This is
    useful when checking vector-load alignment: a contiguous lane expression is only
    safe to vectorize when the first lane's address is sufficiently aligned.
    """
    return V.graph.sizevars.simplify(expr.xreplace({lane_var: sympy.Integer(0)}))


def decompose_affine_lane_expr(
    expr: sympy.Expr, lane_var: sympy.Symbol
) -> tuple[sympy.Expr, sympy.Expr] | None:
    # [FUNCTION DOC]
    # Splits an expression into (stride, base) with expression equal to stride
    # times the lane variable plus base. Stride is the lane variable's
    # coefficient and base is the value at lane 0. Returns None if either part
    # still depends on the lane variable or if a nonzero residual remains
    # (non-affine). Pure symbolic computation using sizevars simplification.
    """Return ``(stride, base)`` when ``expr`` is affine in ``lane_var``.

    The result satisfies ``expr == stride * lane_var + base``, where neither
    ``stride`` nor ``base`` depends on ``lane_var``.
    For example: ``stride == 0`` means lane-uniform, ``stride == 1`` means
    consecutive lanes access consecutive indices, and other strides are
    non-contiguous gathers.

    Examples with ``lane_var = i``:
        ``b + i`` -> ``(1, b)``
        ``b + 2 * i`` -> ``(2, b)``
        ``b`` -> ``(0, b)``

    Returns ``None`` for non-affine expressions or when the extracted stride or
    base still depends on ``lane_var``.
    """
    expr = V.graph.sizevars.simplify(expr)
    lane_coeff = expr.coeff(lane_var)
    rest = lane_group_start(expr, lane_var)
    if lane_var in lane_coeff.free_symbols or lane_var in rest.free_symbols:
        return None
    residual = V.graph.sizevars.simplify(expr - (lane_coeff * lane_var + rest))
    if residual != 0:
        return None
    return lane_coeff, rest
