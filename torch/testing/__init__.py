# [PACKAGE DOC] Public testing utilities package. Re-exports FileCheck (the C++ string-pattern
# checker for IR and text output), assert_close and the deprecated assert_allclose (tensor and
# scalar comparison with tolerances, from the private comparison module), and make_tensor (random
# tensor factory, from the private creation module). Also imports the private _utils submodule.
# Import has no other side effects.

from torch._C import FileCheck as FileCheck

from . import _utils

# pyrefly: ignore [deprecated]
from ._comparison import assert_allclose, assert_close as assert_close
from ._creation import make_tensor as make_tensor
