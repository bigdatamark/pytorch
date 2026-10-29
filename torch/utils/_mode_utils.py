# mypy: allow-untyped-defs
# [MODULE DOC] Small mode helpers: a function checking that all modes in a list are equal, and
# the no_dispatch context manager alias of the C++ disable-dispatch guard. Defines a type
# variable T.
from typing import TypeVar

import torch


T = TypeVar("T")


# returns if all are the same mode
def all_same_mode(modes):
    # [FUNCTION DOC] Returns True if every mode in the sequence equals the first one. An empty
    # sequence yields True because the first element is never accessed. Pure.
    return all(tuple(mode == modes[0] for mode in modes))


no_dispatch = torch._C._DisableTorchDispatch
