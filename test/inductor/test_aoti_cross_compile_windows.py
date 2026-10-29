# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for cross-compiling AOTInductor packages from Linux for Windows GPU targets.
# A framework builds, from model configuration records, a compile test (run on Linux; exports a
# model and writes a package using Windows cross-target options) and a load test (run on Windows;
# loads that package and compares with eager). A class decorator discovers model definition
# methods and attaches both tests under generated names. A second test case checks the helper that
# creates a MinGW import library for the CUDA runtime, using mocks and temporary directories. When
# run as a script it consumes two custom command-line options for the package directory and the
# Windows torch library directory before handing control to the test runner, and runs tests only
# when a GPU is available.
import os
import platform
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import torch
import torch._inductor.config
from torch._environment import is_fbcode
from torch._inductor.cpp_builder import _ensure_mingw_cudart_import_lib
from torch._inductor.test_case import TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_gpu


@dataclass
class ModelTestConfig:
    # [CLASS DOC] HELPER: Dataclass describing one model test case: a name, the model class, a
    # tuple of example input tensors, optional dynamic shape specs, optional extra inductor
    # config overrides, and relative and absolute tolerances (both default to 1e-4) for
    # comparing outputs.
    """Configuration for a model test case."""

    name: str
    model_class: type
    example_inputs: tuple[torch.Tensor, ...]
    dynamic_shapes: dict[str, Any] | None = None
    inductor_configs: dict[str, Any] | None = None
    rtol: float = 1e-4
    atol: float = 1e-4


class WindowsCrossCompilationTestFramework:
    # [CLASS DOC] HELPER: Static-style factory (class-level state only) that produces compile
    # and load test functions for the cross-compilation workflow. State is shared through
    # class attributes: a lazily created base directory for packages and the path to Windows
    # torch libraries.
    """
    Framework for testing cross-compilation from Linux to Windows.

    Provides reusable logic for creating compile and load test methods.
    """

    _base_path: Path | None = None
    _win_torch_libs_path: str | None = None

    @classmethod
    def base_path(cls) -> Path:
        # [FUNCTION DOC] HELPER: Returns the base directory for package files, creating a
        # fresh temporary directory with a cross-compile prefix on first use and caching it on
        # the class.
        """Get or create the base path for package files."""
        if cls._base_path is None:
            cls._base_path = Path(tempfile.mkdtemp(prefix="aoti_cross_compile_"))
        return cls._base_path

    @classmethod
    def set_base_path(cls, path: Path | str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Sets the cached base directory from a path or string; passing
        # nothing (or an empty value) clears it so a new temporary directory is created on
        # next use.
        """Set the base path for package files."""
        cls._base_path = Path(path) if path else None

    @classmethod
    def set_win_torch_libs_path(cls, path: str | None = None) -> None:
        # [FUNCTION DOC] HELPER: Stores the directory holding Windows torch libraries (or clears
        # it with nothing).
        """Set the path for Windows torch libs."""
        cls._win_torch_libs_path = path

    @classmethod
    def get_package_path(cls, model_name: str) -> str:
        # [FUNCTION DOC] HELPER: Ensures the base directory exists and returns, as a string, the
        # path of the package file for the model name, formed by appending the windows suffix and
        # the pt2 extension.
        """Get the path for a model's .pt2 package file."""
        package_dir = cls.base_path()
        package_dir.mkdir(parents=True, exist_ok=True)
        return str(package_dir / f"{model_name}_windows.pt2")

    @classmethod
    def get_win_torch_libs_path(cls) -> str:
        # [FUNCTION DOC] HELPER: Returns the Windows torch libs path as a string; raises a runtime
        # error if it was never set.
        """Get the path for Windows torch libs."""
        if cls._win_torch_libs_path is None:
            raise RuntimeError("Windows torch libs path not set")
        return str(cls._win_torch_libs_path)

    @classmethod
    def create_compile_test(cls, config: ModelTestConfig):
        # [FUNCTION DOC] HELPER: Returns a test function (to be installed as a method) for the
        # given model config. The function itself does the compile step.
        """Create a compile test method for a model configuration."""

        def compile_test(self):
            # [FUNCTION DOC] HELPER: Test body for Linux-side cross compilation. Skips on Windows
            # and in FB code, requires the Windows CUDA home environment variable, then under
            # no-grad builds the model on the GPU device, moves inputs there, exports with the
            # config dynamic shapes, and calls the AOTInductor compile-and-package with
            # cross-target windows options (no precompiled headers, constants stored as a binary
            # blob on disk and not in the shared object, the Windows libs path), merged with any
            # config overrides. Asserts the package file exists at the expected path.
            if platform.system() == "Windows":
                raise unittest.SkipTest(
                    "This test should run on Linux for cross-compilation"
                )

            if is_fbcode():
                raise unittest.SkipTest("requires x86_64-w64-mingw32-gcc")

            self.assertTrue("WINDOWS_CUDA_HOME" in os.environ)

            with torch.no_grad():
                # Windows cross-compilation is only used for GPU.
                # AOTI for CPU should be able to work as native compilation on Windows.
                device = GPU_TYPE
                model = config.model_class().to(device=device)
                example_inputs = config.example_inputs

                # Inputs should already be on GPU_TYPE but ensure they are
                example_inputs = tuple(inp.to(device) for inp in example_inputs)

                # Export the model
                exported = torch.export.export(
                    model, example_inputs, dynamic_shapes=config.dynamic_shapes
                )

                # Prepare inductor configs
                inductor_configs = {
                    "aot_inductor.cross_target_platform": "windows",
                    "aot_inductor.precompile_headers": False,
                    "aot_inductor.package_constants_on_disk_format": "binary_blob",
                    "aot_inductor.package_constants_in_so": False,
                    "aot_inductor.aoti_shim_library_path": cls.get_win_torch_libs_path(),
                }
                if config.inductor_configs:
                    inductor_configs.update(config.inductor_configs)

                # Compile and package directly to the expected location
                package_path = cls.get_package_path(config.name)
                torch._inductor.aoti_compile_and_package(
                    exported,
                    package_path=package_path,
                    inductor_configs=inductor_configs,
                )

                self.assertTrue(
                    os.path.exists(package_path),
                    lambda msg: f"{msg}\nPackage file should exist at {package_path}",
                )

        return compile_test

    @classmethod
    def create_load_test(cls, config: ModelTestConfig):
        # [FUNCTION DOC] HELPER: Returns a test function (to be installed as a method) for the
        # given model config. The function itself does the load and compare step.
        """Create a load test method for a model configuration."""

        def load_test(self):
            # [FUNCTION DOC] HELPER: Test body for Windows-side loading. Skips on non-Windows, in
            # FB code, without a GPU, or when the package file produced by the compile step is
            # missing. Otherwise builds the model on the GPU, loads the package, runs both on the
            # same inputs under no-grad and asserts outputs are close with the config tolerances.
            if platform.system() != "Windows":
                raise unittest.SkipTest("This test should run on Windows")

            if is_fbcode():
                raise unittest.SkipTest("requires x86_64-w64-mingw32-gcc")

            if not HAS_GPU:
                raise unittest.SkipTest("Test requires GPU")

            package_path = cls.get_package_path(config.name)
            if not os.path.exists(package_path):
                raise unittest.SkipTest(
                    f"Package file not found at {package_path}. "
                    f"Run test_{config.name}_compile first."
                )

            with torch.no_grad():
                # Windows cross-compilation is only used for GPU.
                # AOTI for CPU should be able to work as native compilation on Windows.
                device = GPU_TYPE

                # Create original model for comparison
                original_model = config.model_class().to(device=device)
                example_inputs = config.example_inputs

                # Inputs should already be on GPU_TYPE but ensure they are
                example_inputs = tuple(inp.to(device) for inp in example_inputs)

                # Load the compiled package
                loaded_model = torch._inductor.aoti_load_package(package_path)

                # Test with the same inputs
                original_output = original_model(*example_inputs)
                loaded_output = loaded_model(*example_inputs)

                # Compare outputs
                torch.testing.assert_close(
                    original_output, loaded_output, rtol=config.rtol, atol=config.atol
                )

        return load_test


def auto_generate_tests(test_class):
    # [FUNCTION DOC] HELPER: Class decorator. Scans attributes whose names start with the
    # define prefix, calls each on an uninitialized instance, and keeps those returning a
    # config record (any exception is swallowed and that method ignored). For each kept one it
    # installs a compile test and a load test, named by the model name with the compile or load
    # suffix, wrapped with the requires-GPU decorator and given step docstrings. Mutates and
    # returns the class.
    """
    Class decorator to automatically generate compile/load test methods
    from _define_* methods that return ModelTestConfig.
    """
    # Find all _define_* methods that return ModelTestConfig
    define_methods = {}
    for name in dir(test_class):
        if name.startswith("_define_") and callable(getattr(test_class, name)):
            method = getattr(test_class, name)
            # Try to call the method to see if it returns ModelTestConfig
            try:
                # Create a temporary instance to call the method
                temp_instance = test_class.__new__(test_class)
                result = method(temp_instance)
                if isinstance(result, ModelTestConfig):
                    define_methods[name] = result
            except Exception:
                # If method fails, skip it
                pass

    # Generate compile/load methods for each discovered definition
    for define_name, config in define_methods.items():
        model_name = define_name[8:]  # Remove '_define_' prefix

        # Create compile test method
        compile_method_name = f"test_{model_name}_compile"
        compile_method = WindowsCrossCompilationTestFramework.create_compile_test(
            config
        )
        compile_method.__name__ = compile_method_name
        compile_method.__doc__ = f"Step 1: Cross-compile {model_name} model on Linux"
        compile_method = requires_gpu()(compile_method)
        setattr(test_class, compile_method_name, compile_method)

        # Create load test method
        load_method_name = f"test_{model_name}_load"
        load_method = WindowsCrossCompilationTestFramework.create_load_test(config)
        load_method.__name__ = load_method_name
        load_method.__doc__ = f"Step 2: Load and test {model_name} model on Windows"
        load_method = requires_gpu()(load_method)
        setattr(test_class, load_method_name, load_method)

    return test_class


@auto_generate_tests
class TestAOTInductorWindowsCrossCompilation(TestCase):
    # [CLASS DOC] Test case whose compile and load tests are generated by the decorator from the
    # three define methods below (simple MLP, small CNN, small transformer). It defines no test
    # methods directly.
    """
    Test class for AOT Inductor Windows cross-compilation.

    Define test methods that return ModelTestConfig, and the decorator
    will auto-generate compile/load test methods.
    """

    def _define_simple(self):
        # [FUNCTION DOC] HELPER: Defines a small two-layer MLP with sigmoid output and returns its
        # config: name simple, GPU input of batch 8 by 10 features, dynamic batch dimension from 1
        # to 1024, default tolerances.
        """Define the Simple model and its test configuration."""

        class Simple(torch.nn.Module):
            # [CLASS DOC] HELPER: MLP: linear 10 to 16, relu, linear 16 to 1, sigmoid.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two linear layers, relu and sigmoid.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.fc2 = torch.nn.Linear(16, 1)
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers in order and returns the result.
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                x = self.sigmoid(x)
                return x

        return ModelTestConfig(
            name="simple",
            model_class=Simple,
            example_inputs=(torch.randn(8, 10, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        )

    def _define_simple_cnn(self):
        # [FUNCTION DOC] HELPER: Defines a small CNN and returns its config: name simple_cnn,
        # GPU input of shape 2 by 3 by 32 by 32, dynamic batch from 1 to 16, tolerances relaxed
        # to 1e-3.
        """Define the SimpleCNN model and its test configuration."""

        class SimpleCNN(torch.nn.Module):
            # [CLASS DOC] HELPER: CNN: 3 to 16 channel convolution with kernel 3, relu, adaptive
            # average pool to one pixel, flatten, linear 16 to 10.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the convolution, relu, pooling and linear layers.
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, 3)
                self.relu = torch.nn.ReLU()
                self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.fc = torch.nn.Linear(16, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs convolution, relu, pooling, flatten from dimension
                # one and the linear layer; returns the result.
                x = self.conv1(x)
                x = self.relu(x)
                x = self.pool(x)
                x = x.flatten(1)
                x = self.fc(x)
                return x

        return ModelTestConfig(
            name="simple_cnn",
            model_class=SimpleCNN,
            example_inputs=(torch.randn(2, 3, 32, 32, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=16)}},
            rtol=1e-3,
            atol=1e-3,
        )

    def _define_transformer(self):
        # [FUNCTION DOC] HELPER: Defines a small transformer-style model and returns its config:
        # name transformer, GPU input of shape 4 by 16 by 128, dynamic batch from 1 to 32,
        # tolerances 1e-3.
        """Define the SimpleTransformer model and its test configuration."""

        class SimpleTransformer(torch.nn.Module):
            # [CLASS DOC] HELPER: Transformer-like model: embedding linear, multi-head attention,
            # layer norms, feed-forward network, mean pooling over the sequence and an output
            # linear layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the embedding, 8-head batch-first attention, two
                # layer norms, feed-forward block and output layers.
                super().__init__()
                self.embedding = torch.nn.Linear(128, 256)
                self.attention = torch.nn.MultiheadAttention(256, 8, batch_first=True)
                self.norm1 = torch.nn.LayerNorm(256)
                self.ffn = torch.nn.Sequential(
                    torch.nn.Linear(256, 1024),
                    torch.nn.ReLU(),
                    torch.nn.Linear(1024, 256),
                )
                self.norm2 = torch.nn.LayerNorm(256)
                self.output = torch.nn.Linear(256, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Embeds input, applies self-attention with residual and
                # norm, feed-forward with residual and norm, averages over the sequence dimension
                # and applies the output layer.
                # x shape: (batch, seq_len, input_dim)
                x = self.embedding(x)
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                x = x.mean(dim=1)  # Global average pooling
                x = self.output(x)
                return x

        return ModelTestConfig(
            name="transformer",
            model_class=SimpleTransformer,
            example_inputs=(torch.randn(4, 16, 128, device=GPU_TYPE),),
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=32)}},
            rtol=1e-3,
            atol=1e-3,
        )


class TestEnsureMingwCudartImportLib(TestCase):
    # [CLASS DOC] Unit tests for the helper that creates a MinGW-compatible CUDA runtime import
    # library. Each test builds a temporary fake CUDA home and library directory in setup and uses
    # patched environment and subprocess calls.
    """Unit tests for _ensure_mingw_cudart_import_lib."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Creates a temporary directory containing a fake CUDA home with a
        # bin/x64 folder and an empty library directory; stores the paths on the test.
        super().setUp()
        self.tmp_dir = tempfile.mkdtemp(prefix="test_mingw_cudart_")
        self.cuda_home = os.path.join(self.tmp_dir, "cuda")
        self.lib_dir = os.path.join(self.tmp_dir, "lib")
        os.makedirs(os.path.join(self.cuda_home, "bin", "x64"), exist_ok=True)
        os.makedirs(self.lib_dir, exist_ok=True)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Removes the temporary directory tree (ignoring errors) and calls
        # the parent teardown.
        shutil.rmtree(self.tmp_dir, ignore_errors=True)
        super().tearDown()

    def _create_fake_dll(self, version: str = "130") -> str:
        # [FUNCTION DOC] HELPER: Creates an empty cudart DLL file with the given version suffix
        # (default 130) in the fake bin/x64 folder and returns its path.
        dll_path = os.path.join(self.cuda_home, "bin", "x64", f"cudart64_{version}.dll")
        Path(dll_path).touch()
        return dll_path

    def _create_fake_cudart_lib(self) -> str:
        # [FUNCTION DOC] HELPER: Creates an empty cudart.lib file in the fake library directory
        # and returns its path.
        lib_path = os.path.join(self.lib_dir, "cudart.lib")
        Path(lib_path).touch()
        return lib_path

    def test_noop_when_no_windows_cuda_home(self):
        # [FUNCTION DOC] With the Windows CUDA home variable removed, the helper called with the
        # library dir returns an empty list and creates no libcudart.a.
        """Should skip gracefully when WINDOWS_CUDA_HOME is not set."""
        env = os.environ.copy()
        env.pop("WINDOWS_CUDA_HOME", None)
        with patch.dict(os.environ, env, clear=True):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(os.path.join(self.lib_dir, "libcudart.a")))

    def test_noop_when_libcudart_a_already_exists(self):
        # [FUNCTION DOC] When libcudart.a already exists in the library directory, the helper
        # returns an empty list and leaves the file in place.
        """Should skip generation if libcudart.a already exists."""
        existing = os.path.join(self.lib_dir, "libcudart.a")
        Path(existing).touch()
        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])
        self.assertEqual(result, [])
        self.assertTrue(os.path.exists(existing))

    @patch("torch._inductor.cpp_builder._create_msvc_gs_stubs_lib", return_value=None)
    def test_gs_stubs_fallback_when_no_dll_found(self, mock_stubs):
        # [FUNCTION DOC] With a cudart.lib but no cudart DLL, the helper is expected to call
        # the stub-library creation function once with the library directory (mocked to
        # return nothing).
        """Should attempt GS stubs fallback when no cudart64_*.dll is found."""
        self._create_fake_cudart_lib()
        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            _ensure_mingw_cudart_import_lib([self.lib_dir])
        mock_stubs.assert_called_once_with(self.lib_dir)

    @patch(
        "torch._inductor.cpp_builder._create_msvc_gs_stubs_lib",
        return_value="msvc_gs_stubs",
    )
    def test_gs_stubs_returned_when_no_dll(self, mock_stubs):
        # [FUNCTION DOC] With a cudart.lib but no DLL and the stub creation function mocked to
        # return a name, the helper returns a list holding that name.
        """Should return the stubs library name when DLL is unavailable."""
        self._create_fake_cudart_lib()
        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])
        self.assertEqual(result, ["msvc_gs_stubs"])

    def test_noop_when_no_writable_dir_with_cudart_lib(self):
        # [FUNCTION DOC] With a DLL present but no cudart.lib in the library directory, the helper
        # returns an empty list and creates no libcudart.a.
        """Should skip gracefully when no writable directory contains cudart.lib."""
        self._create_fake_dll()
        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])
        self.assertEqual(result, [])
        self.assertFalse(os.path.exists(os.path.join(self.lib_dir, "libcudart.a")))

    @patch("subprocess.run")
    def test_successful_generation(self, mock_run: MagicMock):
        # [FUNCTION DOC] With both DLL and cudart.lib present and subprocess mocked, the helper
        # returns an empty list and runs two commands: first gendef with a dash argument and the
        # DLL path, then the MinGW dlltool with definition, library, and DLL-name options naming
        # the DLL file.
        """Should call gendef and dlltool when all conditions are met."""
        dll_path = self._create_fake_dll()
        self._create_fake_cudart_lib()

        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])

        self.assertEqual(result, [])
        self.assertEqual(mock_run.call_count, 2)

        # Verify gendef call
        gendef_call = mock_run.call_args_list[0]
        gendef_cmd = gendef_call[0][0]
        self.assertEqual(gendef_cmd[0], "gendef")
        self.assertEqual(gendef_cmd[1], "-")
        self.assertEqual(gendef_cmd[2], dll_path)

        # Verify dlltool call
        dlltool_call = mock_run.call_args_list[1]
        dlltool_cmd = dlltool_call[0][0]
        self.assertEqual(dlltool_cmd[0], "x86_64-w64-mingw32-dlltool")
        self.assertIn("-d", dlltool_cmd)
        self.assertIn("-l", dlltool_cmd)
        self.assertIn("-D", dlltool_cmd)
        self.assertIn("cudart64_130.dll", dlltool_cmd)

    @patch(
        "torch._inductor.cpp_builder._create_msvc_gs_stubs_lib",
        return_value="msvc_gs_stubs",
    )
    @patch("subprocess.run", side_effect=FileNotFoundError("gendef not found"))
    def test_gs_stubs_fallback_on_gendef_not_found(self, mock_run, mock_stubs):
        # [FUNCTION DOC] With DLL and lib present but the subprocess call raising file-not-found
        # (gendef missing) and stubs mocked to a name, the helper returns a list with that name
        # and calls the stub creation exactly once.
        """Should fall back to GS stubs when gendef is not installed."""
        self._create_fake_dll()
        self._create_fake_cudart_lib()

        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            result = _ensure_mingw_cudart_import_lib([self.lib_dir])

        self.assertEqual(result, ["msvc_gs_stubs"])
        mock_stubs.assert_called_once()

    @patch("subprocess.run")
    def test_bin_x64_fallback_to_bin(self, mock_run: MagicMock):
        # [FUNCTION DOC] After removing bin/x64 and putting the DLL directly in bin, with
        # subprocess mocked, the helper still finds it: two commands run and the first (gendef)
        # includes the DLL path.
        """Should fall back to bin/ when bin/x64/ does not exist."""
        shutil.rmtree(os.path.join(self.cuda_home, "bin", "x64"))
        bin_dir = os.path.join(self.cuda_home, "bin")
        dll_path = os.path.join(bin_dir, "cudart64_130.dll")
        Path(dll_path).touch()
        self._create_fake_cudart_lib()

        with patch.dict(os.environ, {"WINDOWS_CUDA_HOME": self.cuda_home}):
            _ensure_mingw_cudart_import_lib([self.lib_dir])

        self.assertEqual(mock_run.call_count, 2)
        gendef_cmd = mock_run.call_args_list[0][0][0]
        self.assertIn(dll_path, gendef_cmd)


if __name__ == "__main__":
    import sys

    from torch._inductor.test_case import run_tests

    # Check for --package-dir argument and remove it before unittest sees it
    package_dir = None
    win_torch_lib_dir = None
    filtered_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "--package-dir":
            if i + 1 < len(sys.argv):
                package_dir = sys.argv[i + 1]
                i += 2  # Skip both --package-dir and its value
            else:
                print("Error: --package-dir requires a valid directory path")
                sys.exit(1)
        elif sys.argv[i].startswith("--package-dir="):
            package_dir = sys.argv[i].split("=", 1)[1]
            i += 1
        elif sys.argv[i] == "--win-torch-lib-dir":
            if i + 1 < len(sys.argv):
                win_torch_lib_dir = sys.argv[i + 1]
                i += 2  # Skip both --win-torch-lib-dir and its value
            else:
                print("Error: --win-torch-lib-dir requires a valid directory path")
                sys.exit(1)
        elif sys.argv[i].startswith("--win-torch-lib-dir="):
            win_torch_lib_dir = sys.argv[i].split("=", 1)[1]
            i += 1
        else:
            filtered_argv.append(sys.argv[i])
            i += 1

    # Validate and set the base path for package storage
    if package_dir:
        try:
            package_path = Path(package_dir)
            package_path.mkdir(parents=True, exist_ok=True)
            # Test write access
            test_file = package_path / ".test_write"
            test_file.touch()
            test_file.unlink()
            WindowsCrossCompilationTestFramework.set_base_path(package_path)
        except Exception:
            print("Error: --package-dir requires a valid directory path")
            sys.exit(1)

    # Set Windows torch libs path if provided (only needed for compile tests)
    if win_torch_lib_dir:
        WindowsCrossCompilationTestFramework.set_win_torch_libs_path(win_torch_lib_dir)

    # Update sys.argv to remove our custom arguments
    sys.argv = filtered_argv

    if HAS_GPU:
        run_tests(needs="filelock")
