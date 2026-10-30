# Owner(s): ["module: dsl-native-ops"]
# [MODULE DOC] Tests for the vendored QuACK kernel library under torch/_vendor/quack and its
# vendoring script. One case (needs cutlass) checks imports resolve to the torch vendor tree and
# that autotune precompile of the RMS norm forward handles tensor keyword arguments; another (needs
# git and patch) re-runs the vendoring script in check mode. Defines paths to the repo root and to
# the vendor script.

import importlib.util
import os
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import run_tests, TestCase


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SCRIPT = REPO_ROOT / "tools" / "vendoring" / "quack" / "vendor.sh"


@unittest.skipIf(
    importlib.util.find_spec("cutlass") is None,
    "vendored QuACK imports require CuTeDSL/CUTLASS",
)
class TestQuackVendor(TestCase):
    # [CLASS DOC] Test case for the vendored QuACK package; skipped unless the cutlass package can
    # be found. Stateless; both tests import lazily.
    def test_vendored_quack_gemm_imports_from_torch_vendor(self):
        # [FUNCTION DOC] Imports the vendored quack package, a gemm function, a config class and
        # the trace context, asserts the package directory is under the torch vendor tree and that
        # the gemm function and config class are callable, then creates a trace context with no
        # destination, records a begin and end event and flushes it.
        import torch._vendor.quack as quack
        from torch._vendor.quack.gemm_act import gemm_act
        from torch._vendor.quack.gemm_config import GemmConfig
        from torch._vendor.quack.trace import TraceContext

        vendor_root = Path(quack.__file__).resolve().parent
        self.assertIn("torch/_vendor/quack", vendor_root.as_posix())
        self.assertTrue(
            all(
                callable(obj)
                for obj in (
                    gemm_act,
                    GemmConfig,
                )
            )
        )
        trace_context = TraceContext.create(None)
        trace_context.b("noop")
        trace_context.e("noop")
        trace_context.flush()

    def test_precompile_serializes_rmsnorm_tuned_tensor_kwargs(self):
        # [FUNCTION DOC] Skipped without CUDA. Builds float16 CUDA tensors for an RMS norm forward
        # with bias, residual, residual output and reciprocal std outputs, plus two identical
        # autotune configs. With two compile workers set in the environment and the clock patched,
        # calls the precompile method of the tuned forward, waits for each config and asserts no
        # failures were recorded, always shutting the handle down.
        import torch
        from torch._vendor.quack.autotuner import AutotuneConfig
        from torch._vendor.quack.rmsnorm import rmsnorm_fwd_tuned
        from torch._vendor.quack.rmsnorm_config import RmsNormFwdConfig

        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        x = torch.randn(2, 128, dtype=torch.float16, device="cuda")
        weight = torch.randn(128, dtype=torch.float16, device="cuda")
        out = torch.empty_like(x)
        bias = torch.randn(128, dtype=torch.float16, device="cuda")
        residual = torch.randn_like(x)
        residual_out = torch.empty_like(x)
        rstd = torch.empty(2, dtype=torch.float32, device="cuda")
        config = RmsNormFwdConfig(
            num_threads=128,
            threads_per_row=16,
            cluster_n=1,
            reload_from=None,
            delay_w_load=False,
        )
        configs = [AutotuneConfig(config=config), AutotuneConfig(config=config)]

        with (
            mock.patch.dict(os.environ, {"QUACK_COMPILE_WORKERS": "2"}),
            mock.patch(
                "torch._vendor.quack.autotuner.time.time",
                side_effect=[0.0, 1.0],
            ),
        ):
            handle = rmsnorm_fwd_tuned._precompile(
                x,
                weight,
                out,
                configs=configs,
                bias=bias,
                residual=residual,
                residual_out=residual_out,
                rstd=rstd,
            )
        try:
            for i in range(len(configs)):
                handle.wait_for(i)
            self.assertEqual(handle.failures, {})
        finally:
            handle.shutdown()


@unittest.skipIf(
    shutil.which("git") is None or shutil.which("patch") is None,
    "re-running the vendoring script requires git and patch",
)
class TestQuackVendorScript(TestCase):
    # [CLASS DOC] Test case for the vendoring script; skipped unless both git and patch are on the
    # path.
    def test_vendor_script_reproduces_committed_tree(self):
        # [FUNCTION DOC] Runs the vendor script in check mode from the repo root to assert the
        # committed vendored tree has no drift. Skipped unless a source checkout path variable is
        # set (passed as the source argument) or the clone-allowed variable is true/1/yes. Fails
        # with a message if the script exit code is nonzero.
        src = os.environ.get("QUACK_VENDOR_SRC")
        allow_clone = os.environ.get("QUACK_VENDOR_ALLOW_CLONE", "").lower() in (
            "1",
            "true",
            "yes",
        )
        if not src and not allow_clone:
            self.skipTest(
                "set QUACK_VENDOR_SRC to a local quack checkout at the pinned SHA, "
                "or QUACK_VENDOR_ALLOW_CLONE=1 to fetch upstream main"
            )

        cmd = ["bash", str(VENDOR_SCRIPT), "--check"]
        if src:
            cmd += ["--src", str(Path(src).expanduser())]
        self.assertEqual(
            subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode,
            0,
            "vendor.sh --check reported drift; edit the FlexGEMM patchset or "
            "PyTorch vendoring patches, not the vendored files",
        )


if __name__ == "__main__":
    run_tests()
