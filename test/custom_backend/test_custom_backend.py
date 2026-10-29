# Owner(s): ["module: unknown"]

# [MODULE DOC] HELPER: tests that a module lowered to the custom backend runs and survives save and
# load. Needs the prebuilt backend library; defines TestCustomBackend; runs under run_tests.
import tempfile

from backend import get_custom_backend_library_path, Model, to_custom_backend

import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestCustomBackend(TestCase):
    # [CLASS DOC] HELPER: holds the lowered model in self.model, rebuilt for every test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: loads the backend library (path in self.library_path) and lowers a
        # scripted Model into self.model.
        super().setUp()
        # Load the library containing the custom backend.
        self.library_path = get_custom_backend_library_path()
        torch.ops.load_library(self.library_path)
        # Create an instance of the test Module and lower it for
        # the custom backend.
        self.model = to_custom_backend(torch.jit.script(Model()))

    def test_execute(self):
        # [FUNCTION DOC] Runs the lowered model on two random vectors and checks the outputs match
        # the sum and difference computed eagerly.
        """
        Test execution using the custom backend.
        """
        a = torch.randn(4)
        b = torch.randn(4)
        # The custom backend is hardcoded to compute f(a, b) = (a + b, a - b).
        expected = (a + b, a - b)
        out = self.model(a, b)
        self.assertTrue(expected[0].allclose(out[0]))
        self.assertTrue(expected[1].allclose(out[1]))

    def test_save_load(self):
        # [FUNCTION DOC] Runs the model once, saves it to a temporary file, loads it back, replaces
        # self.model with the loaded one and runs the execution check again.
        """
        Test that a lowered module can be executed correctly
        after saving and loading.
        """
        # Test execution before saving and loading to make sure
        # the lowered module works in the first place.
        self.test_execute()

        # Save and load.
        with tempfile.NamedTemporaryFile() as f:
            f.close()
            torch.jit.save(self.model, f.name)
            loaded = torch.jit.load(f.name)
            self.model = loaded

        # Test execution again.
        self.test_execute()


if __name__ == "__main__":
    run_tests()
