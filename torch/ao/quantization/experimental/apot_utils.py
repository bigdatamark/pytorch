# mypy: allow-untyped-defs
# [MODULE DOC] Scalar helper functions for Additive Powers-of-Two (APoT) nonuniform quantization: mapping a float to an APoT
# index, to the nearest quantization level, and an APoT index back to a float. Note the bare string literals above some
# functions are not docstrings (they are unused expressions).
r"""
This file contains utility functions to convert values
using APoT nonuniform quantization methods.
"""

import math


r"""Converts floating point input into APoT number
    based on quantization levels
"""


def float_to_apot(x, levels, indices, alpha):
    # [FUNCTION DOC] Map a float x to an APoT index. Values below -alpha return -alpha and values above alpha return alpha (clipped float values,
    # not indices). Otherwise returns the index paired with the level closest to x (first one wins on ties; 0 if levels is empty).
    # KNOWN ISSUE: the clipped cases return alpha values while the normal case returns an index, so the return semantics are inconsistent.
    # clip values based on alpha
    if x < -alpha:
        return -alpha
    elif x > alpha:
        return alpha

    levels_lst = list(levels)
    indices_lst = list(indices)

    min_delta = math.inf
    best_idx = 0

    for level, idx in zip(levels_lst, indices_lst):
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_idx = idx

    return best_idx


r"""Converts floating point input into
    reduced precision floating point value
    based on quantization levels
"""


def quant_dequant_util(x, levels, indices):
    # [FUNCTION DOC] Return the level in levels closest to x (first one on ties; 0.0 if levels is empty). The indices argument is unused.
    min_delta = math.inf
    best_fp = 0.0

    for level in levels:
        cur_delta = abs(level - x)
        if cur_delta < min_delta:
            min_delta = cur_delta
            best_fp = level

    return best_fp


r"""Converts APoT input into floating point number
based on quantization levels
"""


def apot_to_float(x_apot, levels, indices):
    # [FUNCTION DOC] Return the level whose position matches the APoT index x_apot within indices. Raises ValueError if x_apot is not in indices.
    idx = list(indices).index(x_apot)
    return levels[idx]
