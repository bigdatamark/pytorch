# Owner(s): ["oncall: jit"]

# [MODULE DOC] Tests that TorchScript constructs still run when the JIT is turned off through
# the PYTORCH_JIT environment variable. Each test runs a small program in a fresh interpreter
# with the JIT enabled and disabled and requires identical stdout. No import-time side effects
# beyond imports; the tests run only on Python older than 3.14.
import sys
import os
import contextlib
import subprocess
from torch.testing._internal.common_utils import TestCase, run_tests, TemporaryFileName


@contextlib.contextmanager
def _jit_disabled():
    # [FUNCTION DOC] HELPER: Context manager that sets the PYTORCH_JIT environment variable to
    # the string zero for the duration of the block, then restores the previous value (default
    # string one if it was unset; it is therefore set, not removed, afterwards). Restoration
    # happens even if the body raises. Affects only subprocesses started inside the block.
    cur_env = os.environ.get("PYTORCH_JIT", "1")
    os.environ["PYTORCH_JIT"] = "0"
    try:
        yield
    finally:
        os.environ["PYTORCH_JIT"] = cur_env


class TestJitDisabled(TestCase):
    # [CLASS DOC] Test case holding subprocess-based checks that scripting APIs behave the same
    # with the JIT disabled as enabled. Stateless; the helper method runs a program source twice.
    """
    These tests are separate from the rest of the JIT tests because we need
    run a new subprocess and `import torch` with the correct environment
    variables set.
    """

    def compare_enabled_disabled(self, src):
        # [FUNCTION DOC] HELPER: Writes the program text src to a temporary file, runs it with
        # the current Python interpreter once with PYTORCH_JIT set to zero and once without that
        # change, and asserts the two captured stdout byte strings are equal. A failing
        # subprocess raises CalledProcessError. The run happens while the temp file is open.
        """
        Runs the script in `src` with PYTORCH_JIT enabled and disabled and
        compares their stdout for equality.
        """
        # Write `src` out to a temporary so our source inspection logic works
        # correctly.
        with TemporaryFileName() as fname:
            with open(fname, 'w') as f:
                f.write(src)
                with _jit_disabled():
                    out_disabled = subprocess.check_output([
                        sys.executable,
                        fname])
                out_enabled = subprocess.check_output([
                    sys.executable,
                    fname])
                self.assertEqual(out_disabled, out_enabled)

    def test_attribute(self):
        # [FUNCTION DOC] Checks that a ScriptModule subclass storing a tensor via
        # torch.jit.Attribute, then printing that attribute, gives the same output JIT on and off.
        _program_string = """
import torch

class Foo(torch.jit.ScriptModule):
    def __init__(self, x):
        super().__init__()
        self.x = torch.jit.Attribute(x, torch.Tensor)

    def forward(self, input):
        return input

s = Foo(torch.ones(2, 3))
print(s.x)
"""
        self.compare_enabled_disabled(_program_string)

    def test_script_module_construction(self):
        # [FUNCTION DOC] Checks that instantiating a ScriptModule with a script_method forward
        # does not raise in either mode; both runs must print the same confirmation line.
        _program_string = """
import torch

class AModule(torch.jit.ScriptModule):
    @torch.jit.script_method
    def forward(self, input):
        pass

AModule()
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

    def test_recursive_script(self):
        # [FUNCTION DOC] Checks that applying torch.jit.script to a plain nn.Module with a
        # trivial forward does not raise in either mode; both runs must print the same line.
        _program_string = """
import torch

class AModule(torch.nn.Module):
    def forward(self, input):
        pass

sm = torch.jit.script(AModule())
print("Didn't throw exception")
"""
        self.compare_enabled_disabled(_program_string)

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
