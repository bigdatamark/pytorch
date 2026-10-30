# [MODULE DOC] Small numeric helpers for generating lists of benchmark sizes. No import-time
# side effects.
import math

import torch


def rounded_linspace(low, high, steps, div):
    # [FUNCTION DOC] Make evenly spaced values from low to high, round each up to a multiple of
    # div, remove duplicates (sorted ascending), and return them as a list of ints. Args are the
    # endpoints, the number of steps, and the rounding divisor.
    ret = torch.linspace(low, high, steps)
    ret = (ret.int() + div - 1) // div * div
    ret = torch.unique(ret)
    return list(map(int, ret))


def powspace(start, stop, pow, step):
    # [FUNCTION DOC] Make roughly logarithmically spaced values from start to stop using the
    # given base. Takes logs of the endpoints in that base, derives the step count from the
    # exponent range divided by step, builds powers of the base at evenly spaced exponents,
    # deduplicates (sorted) and returns a list of ints.
    start = math.log(start, pow)
    stop = math.log(stop, pow)
    steps = int((stop - start + 1) // step)
    ret = torch.pow(pow, torch.linspace(start, stop, steps))
    ret = torch.unique(ret)
    return list(map(int, ret))
