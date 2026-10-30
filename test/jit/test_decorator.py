# Owner(s): ["oncall: jit"]


# [MODULE DOC] JIT test for scripting a decorated function imported from the jit.myfunction_a
# helper module. Not currently run by discovery; running the file directly raises RuntimeError
# saying it must be enabled in discover_tests.py first.
import torch
from jit.myfunction_a import my_function_a
from torch.testing._internal.jit_utils import JitTestCase


class TestDecorator(JitTestCase):
    # [CLASS DOC] Test case checking that torch.jit.script can compile a function wrapped by a
    # user decorator.
    def test_decorator(self):
        # [FUNCTION DOC] Scripts the decorated function my_function_a and checks the scripted
        # function and the original return equal values for the input 1.0. Uses a direct script
        # call because the checkScript helper cannot handle decorators (its source parse fails
        # on the at-sign line).
        # Note: JitTestCase.checkScript() does not work with decorators
        # self.checkScript(my_function_a, (1.0,))
        # Error:
        #   RuntimeError: expected def but found '@' here:
        #   @my_decorator
        #   ~ <--- HERE
        #   def my_function_a(x: float) -> float:
        # Do a simple torch.jit.script() test instead
        fn = my_function_a
        fx = torch.jit.script(fn)
        self.assertEqual(fn(1.0), fx(1.0))


if __name__ == "__main__":
    raise RuntimeError(
        "This test is not currently used and should be "
        "enabled in discover_tests.py if required."
    )
