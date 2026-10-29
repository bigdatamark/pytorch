# mypy: ignore-errors
# [PACKAGE DOC] Experimental NumPy-compatible API implemented on top of torch tensors. Re-exports the fft, linalg and
# random subpackages, dtype classes, array functions, ufuncs, finfo/iinfo, ndarray and related helpers, plus constants
# (pi, e, inf, nan, True_, False_) and the aliases alltrue and sometrue for the builtin all and any (the names all and
# any are rebound to the builtins).

from . import fft, linalg, random
from ._dtypes import *  # noqa: F403
from ._funcs import *  # noqa: F403
from ._getlimits import finfo, iinfo
from ._ndarray import (
    array,
    asarray,
    ascontiguousarray,
    can_cast,
    from_dlpack,
    ndarray,
    newaxis,
    result_type,
)
from ._ufuncs import *  # noqa: F403
from ._util import AxisError, UFuncTypeError


from math import pi, e  # usort: skip


all = all  # noqa: PLW0127
alltrue = all

any = any  # noqa: PLW0127
sometrue = any

inf = float("inf")
nan = float("nan")

False_ = False
True_ = True
