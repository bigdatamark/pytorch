# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that the kernel-saving step of the Triton autotuner takes kernel name, warp
# count and shared memory size from the launch metadata schema when present, and falls back to
# plain metadata attributes otherwise. A mock-based unit test class always runs; an end-to-end
# AOTInductor class runs only when a CUDA GPU is available. Import time work: probes for CUDA
# and records the result in a module flag.
"""Tests for save_gpu_kernel() Level 0 schema integration (diff 3/N).

Validates that save_gpu_kernel reads metadata from launch_metadata_schema
when available, falling back to hasattr probing for older Triton versions.

Run:
  buck2 test @fbcode//mode/opt --modifier ovr_config//triton:beta \
    fbcode//caffe2/test/inductor:test_triton_launcher_integration
"""

import unittest
from unittest.mock import MagicMock, patch

from torch.testing._internal.common_utils import TestCase


def _make_mock_launcher(
    metadata_name="test_kernel_0d1d2d",
    num_warps=4,
    shared=0,
    launch_metadata_schema=None,
):
    # [FUNCTION DOC] HELPER: Builds a mock launcher whose compiled binary object carries the
    # given kernel name, warp count, shared memory size and launch metadata schema (default
    # none). The launcher also gets fixed argument name lists and no scratch buffers; the binary
    # has dummy cubin and ptx assembly. Returns the mock; no side effects.
    """Create a mock launcher with bin (CompiledKernel)."""
    launcher = MagicMock()
    launcher.config = MagicMock()
    launcher.def_args = ["X", "Y", "N"]
    launcher.call_args = ["X", "Y", "N"]
    launcher.global_scratch = None
    launcher.profile_scratch = None

    bin_mock = MagicMock()
    bin_mock.metadata.name = metadata_name
    bin_mock.metadata.num_warps = num_warps
    bin_mock.metadata.shared = shared
    bin_mock.num_warps = num_warps
    bin_mock.shared = shared
    bin_mock.launch_metadata_schema = launch_metadata_schema
    bin_mock.asm = {"cubin": b"\x00", "ptx": "mock_ptx"}
    launcher.bin = bin_mock

    return launcher


def _make_mock_autotuner(kernel_name="test_kernel"):
    # [FUNCTION DOC] HELPER: Builds a mock autotuner object standing in for the caching
    # autotuner, with inductor metadata holding the given kernel name, a small Triton signature
    # mapping and a device properties mock of type cuda. Returns the mock.
    """Create a mock CachingAutotuner (self) for save_gpu_kernel."""
    autotuner = MagicMock()
    autotuner.inductor_meta = {"kernel_name": kernel_name}
    autotuner.triton_meta = {"signature": {0: "*fp32"}}
    autotuner.device_props = MagicMock()
    autotuner.device_props.type = "cuda"
    return autotuner


class SaveGpuKernelSchemaTest(TestCase):
    # [CLASS DOC] Unit tests for the kernel-saving method of the caching autotuner, checking
    # whether launcher parameters come from the schema (when the compiled binary exposes one) or
    # from legacy metadata attributes. All collaborators are mocks, so no GPU is needed.
    """Unit tests for save_gpu_kernel() reading from Level 0 schema."""

    def _call_save_gpu_kernel(self, launcher, kernel_name="test_kernel"):
        # [FUNCTION DOC] HELPER: Runs the kernel-saving method on a mock autotuner and the given
        # launcher, with the kernel parameter cache setter and the config conversion function
        # patched out (the latter returns a fixed block size mapping). Asserts the cache setter
        # was called exactly once and returns the parameter mapping that was passed as its
        # second positional argument.
        """Call save_gpu_kernel with mocks and return the params passed to cache."""
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner

        autotuner = _make_mock_autotuner(kernel_name)
        with (
            patch(
                "torch._inductor.codecache.CudaKernelParamCache.set"
            ) as mock_cache_set,
            patch(
                "torch._inductor.runtime.triton_heuristics.config_to_dict",
                return_value={"BLOCK_SIZE": 1024},
            ),
        ):
            CachingAutotuner.save_gpu_kernel(autotuner, stream=0, launcher=launcher)
            mock_cache_set.assert_called_once()
            _key, params, _binary, _bin_type, _asm, _asm_type = (
                mock_cache_set.call_args[0]
            )
        return params

    def test_schema_path_reads_entry_name(self):
        # [FUNCTION DOC] With a schema holding entry name, warp count and shared memory that
        # differ from the binary metadata name, checks the saved mangled name equals the schema
        # entry name.
        """When schema exists, mangled_name should come from schema['entry_name']."""
        schema = {
            "abi_version": 1,
            "entry_name": "schema_kernel_name",
            "num_warps": 8,
            "shared_mem": 4096,
        }
        launcher = _make_mock_launcher(
            metadata_name="metadata_kernel_name",
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["mangled_name"], "schema_kernel_name")

    def test_schema_path_reads_num_warps(self):
        # [FUNCTION DOC] With a schema giving sixteen warps while the binary metadata says four,
        # checks the saved warp count is the schema value.
        """When schema exists, num_warps should come from schema."""
        schema = {
            "abi_version": 1,
            "entry_name": "test_kernel",
            "num_warps": 16,
            "shared_mem": 8192,
        }
        launcher = _make_mock_launcher(
            num_warps=4,
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["num_warps"], 16)

    def test_schema_path_reads_shared_mem(self):
        # [FUNCTION DOC] With a schema giving a large shared memory size while the binary
        # reports zero, checks the saved shared memory size is the schema value.
        """When schema exists, shared_mem should come from schema."""
        schema = {
            "abi_version": 1,
            "entry_name": "test_kernel",
            "num_warps": 4,
            "shared_mem": 49152,
        }
        launcher = _make_mock_launcher(
            shared=0,
            launch_metadata_schema=schema,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["shared_mem"], 49152)

    def test_fallback_when_no_schema(self):
        # [FUNCTION DOC] With no schema, checks the saved mangled name, warp count and shared
        # memory size equal the values from the binary metadata, i.e. the fallback path is used.
        """Without schema, should use hasattr probing (metadata.name, etc.)."""
        launcher = _make_mock_launcher(
            metadata_name="fallback_name",
            num_warps=4,
            shared=1024,
            launch_metadata_schema=None,
        )
        params = self._call_save_gpu_kernel(launcher)
        self.assertEqual(params["mangled_name"], "fallback_name")
        self.assertEqual(params["num_warps"], 4)
        self.assertEqual(params["shared_mem"], 1024)


# =========================================================================
# E2E tests: AOTI pipeline → save_gpu_kernel schema integration
# Requires GPU + torch + inductor. Run via buck2 on H100.
# =========================================================================

_HAS_GPU = False
try:
    import torch

    _HAS_GPU = hasattr(torch, "cuda") and torch.cuda.is_available()
except (ImportError, AttributeError):
    pass


@unittest.skipUnless(_HAS_GPU, "requires CUDA GPU")
class SaveGpuKernelAOTIE2ETest(TestCase):
    # [CLASS DOC] End-to-end tests, skipped without a CUDA GPU, that compile small models
    # through the AOTInductor export, compile, package and load pipeline and check results and
    # use of the schema path.
    """E2E tests for 3/N: verify AOTI pipeline uses save_gpu_kernel schema path."""

    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, then works around the CUDA home path being cached
        # as unset at import time: if neither CUDA_HOME nor CUDA_PATH is in the environment,
        # tries to take the SDK home from an fbcode-only Triton build module (ignoring import
        # failure), and if the cpp_extension module-level CUDA home is still unset, sets it from
        # the environment variable. Mutates os.environ and that module attribute.
        super().setUp()
        # Fix a known module-loading-order issue (see D74745573):
        # torch.utils.cpp_extension.CUDA_HOME is evaluated at import time.
        # In CI's RE sandbox, CUDA_HOME env var isn't set yet at that point,
        # so the module-level variable caches None. We fix both the env var
        # AND the already-cached module variable.
        # NOTE: triton.fb is fbcode-only; OSS CI sets CUDA_HOME via env.
        import os

        if "CUDA_HOME" not in os.environ and "CUDA_PATH" not in os.environ:
            try:
                from triton.fb.build import build_paths

                os.environ["CUDA_HOME"] = build_paths.sdk_home
            except ImportError:
                pass

        import torch.utils.cpp_extension

        if torch.utils.cpp_extension.CUDA_HOME is None:
            torch.utils.cpp_extension.CUDA_HOME = os.environ.get("CUDA_HOME")

    def test_aoti_pipeline_correctness(self):
        # [FUNCTION DOC] Exports a simple two-input addition model on CUDA, compiles and
        # packages it with AOTInductor, loads the package, runs it and asserts the result is
        # close to the eager output. Resets dynamo state first.
        """Full AOTI pipeline (export → compile → load → run) should produce correct results."""
        import torch._dynamo
        import torch._inductor
        import torch.export

        torch._dynamo.reset()

        class AddModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module computing the sum of its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two inputs.
                return x + y

        model = AddModel().to("cuda")
        x = torch.randn(1024, device="cuda")
        y = torch.randn(1024, device="cuda")
        expected = model(x, y)

        ep = torch.export.export(model, (x, y))
        pkg_path = torch._inductor.aoti_compile_and_package(ep)
        loaded = torch._inductor.aoti_load_package(pkg_path)
        result = loaded(x, y)
        torch.testing.assert_close(result, expected)

    def test_aoti_schema_populated(self):
        # [FUNCTION DOC] Compiles a multiply-add model through AOTInductor while temporarily
        # replacing the autotuner kernel-saving method with a wrapper that records that it was
        # called and, if the launcher binary has a schema, its key names, then delegates to the
        # original. Asserts the result matches eager, that the method was called, and, only if a
        # schema was seen, that it has entry name, num warps and shared memory keys. The
        # original method is restored in a finally block.
        """AOTI compilation should call save_gpu_kernel with a populated schema."""
        import torch._dynamo
        import torch._inductor
        import torch._inductor.runtime.triton_heuristics as th
        import torch.export

        torch._dynamo.reset()

        original_save = th.CachingAutotuner.save_gpu_kernel
        schema_info = {"called": False, "schema_used": False, "schema_keys": None}

        def patched_save_gpu_kernel(self, stream, launcher):
            # [FUNCTION DOC] HELPER: Replacement for the kernel-saving method: reads the
            # optional schema from the launcher binary, records in the enclosing info mapping
            # whether it was called and the schema keys, and returns the result of the original
            # method called with the same arguments.
            schema = getattr(launcher.bin, "launch_metadata_schema", None)
            schema_info["called"] = True
            if schema is not None:
                schema_info["schema_used"] = True
                schema_info["schema_keys"] = list(schema.keys())
            return original_save(self, stream, launcher)

        th.CachingAutotuner.save_gpu_kernel = patched_save_gpu_kernel
        try:

            class MulAddModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Tiny module computing the product of the first two inputs
                # plus the third.
                def forward(self, x, y, z):
                    # [FUNCTION DOC] HELPER: Returns the elementwise product of the first two
                    # inputs plus the third.
                    return x * y + z

            model = MulAddModel().to("cuda")
            x = torch.randn(512, device="cuda")
            y = torch.randn(512, device="cuda")
            z = torch.randn(512, device="cuda")
            expected = model(x, y, z)

            ep = torch.export.export(model, (x, y, z))
            pkg_path = torch._inductor.aoti_compile_and_package(ep)
            loaded = torch._inductor.aoti_load_package(pkg_path)
            result = loaded(x, y, z)

            torch.testing.assert_close(result, expected)

            self.assertTrue(
                schema_info["called"],
                "save_gpu_kernel was never called during AOTI compilation",
            )
            # launch_metadata_schema is only available on Triton beta (fbcode
            # uses ovr_config//triton:beta via BUCK modifiers).  OSS CI may run
            # with a Triton version that doesn't expose the schema yet, so we
            # conditionally validate the schema contents here.  The Level 0
            # schema reading logic is fully covered by SaveGpuKernelSchemaTest
            # unit tests above (with mocks).
            if schema_info["schema_used"]:
                self.assertIn("entry_name", schema_info["schema_keys"])
                self.assertIn("num_warps", schema_info["schema_keys"])
                self.assertIn("shared_mem", schema_info["schema_keys"])
        finally:
            th.CachingAutotuner.save_gpu_kernel = original_save


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
