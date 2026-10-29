# mypy: ignore-errors
# [PACKAGE DOC] Operator-checking test utilities (opcheck). Re-exports make_fx_check,
# aot_autograd_check, fake_check, autograd_registration_check and the opcheck test
# generation API (generate_opcheck_tests, opcheck, OpCheckError, dontGenerateOpCheckTests,
# is_inside_opcheck_mode). Importing it loads all submodules; no other side effects.

from .make_fx import make_fx_check
from .aot_autograd import aot_autograd_check, _test_aot_autograd_forwards_backwards_helper
from .fake_tensor import fake_check
from .autograd_registration import autograd_registration_check
from .generate_tests import generate_opcheck_tests, opcheck, OpCheckError, dontGenerateOpCheckTests, is_inside_opcheck_mode
