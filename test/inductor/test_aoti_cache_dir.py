# Owner(s): ["module: inductor"]
# [MODULE DOC] Regression test that AOTInductor packaging works when the inductor cache directory
# is given as a relative path through the environment. No import-time side effects beyond imports.

import os
import tempfile
from unittest import mock

import torch
import torch._inductor
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches


class TestAOTICacheDir(TestCase):
    # [CLASS DOC] Single-test case class checking cache-directory normalization during packaging.
    def test_aoti_compile_package_with_relative_cache_dir(self):
        # [FUNCTION DOC] Exports a small linear-plus-relu model with a dynamic batch dimension
        # while the current directory is a fresh temporary directory and the cache-dir environment
        # variable is the relative name cache. Compiles and packages it, then asserts the returned
        # path equals the requested package path, the package file exists, and the environment
        # variable was rewritten to the absolute cache path. Side effects: changes the working
        # directory, patches the environment, clears inductor caches before and after; the
        # original working directory is restored in a finally block.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module with one linear layer (10 to 16 features) followed
            # by a relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the single linear submodule.
                super().__init__()
                self.linear = torch.nn.Linear(10, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer to the input and then relu;
                # returns the tensor.
                return torch.relu(self.linear(x))

        with tempfile.TemporaryDirectory() as tmpdir:
            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with mock.patch.dict(os.environ, {"TORCHINDUCTOR_CACHE_DIR": "cache"}):
                    clear_caches()
                    try:
                        model = Model()
                        example_inputs = (torch.randn(8, 10),)
                        batch = torch.export.Dim("batch", min=1, max=1024)
                        exported = torch.export.export(
                            model,
                            example_inputs,
                            dynamic_shapes={"x": {0: batch}},
                        )
                        package_path = os.path.join(tmpdir, "model.pt2")

                        output_path = torch._inductor.aoti_compile_and_package(
                            exported,
                            package_path=package_path,
                        )

                        expected_cache_dir = os.path.abspath("cache")
                        self.assertEqual(output_path, package_path)
                        self.assertTrue(os.path.exists(output_path))
                        self.assertEqual(
                            os.environ["TORCHINDUCTOR_CACHE_DIR"], expected_cache_dir
                        )
                    finally:
                        clear_caches()
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    run_tests()
