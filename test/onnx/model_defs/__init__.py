# [PACKAGE DOC] Model definitions used by the ONNX tests. Re-exports with star imports every
# public name from the op_test, squeezenet, srresnet and super_resolution submodules so tests
# can import the small nn.Module classes from this package directly.
from .op_test import *  # noqa: F403
from .squeezenet import *  # noqa: F403
from .srresnet import *  # noqa: F403
from .super_resolution import *  # noqa: F403
