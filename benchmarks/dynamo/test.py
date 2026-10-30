# [MODULE DOC] Unit tests for the dynamo benchmark harness: timm auto-install behavior, the perf
# command line repro preparation, argument parsing defaults, accuracy settings of the TorchBench
# runner, and a CPU smoke test. Also defines a fallback sanitizer check used to skip tests.
# Imports are relative to the dynamo benchmarks package.
import importlib
import importlib.util
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

from .common import parse_args, run
from .torchbench import setup_torchbench_cwd, TorchBenchmarkRunner


try:
    # fbcode only
    from aiplatform.utils.sanitizer_status import is_asan_or_tsan
except ImportError:

    def is_asan_or_tsan():
        # [FUNCTION DOC] HELPER: Fallback used when the internal sanitizer status module is not
        # available; always reports that no address or thread sanitizer is active.
        return False


class TestDynamoBenchmark(unittest.TestCase):
    # [CLASS DOC] Unit test case holding the dynamo benchmark harness tests. Tests do not share
    # state; they patch modules and process-level functions locally.
    def test_timm_auto_install_uses_no_deps(self) -> None:
        # [FUNCTION DOC] Loads the timm models benchmark file as a fresh module under a
        # temporary name, with a fake timm package present in the module table, with module
        # import patched to raise a missing-module error for timm, and with the subprocess call
        # patched. Executes the module and asserts the install command was run exactly once with
        # the interpreter running pip with no-deps on the timm git repository URL. The temporary
        # module entry is always removed.
        module_name = "benchmarks.dynamo._timm_models_install_test"
        module_path = os.path.join(os.path.dirname(__file__), "timm_models.py")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            self.fail("could not load timm_models.py spec")
        if spec.loader is None:
            self.fail("could not load timm_models.py loader")
        module = importlib.util.module_from_spec(spec)

        fake_timm = types.ModuleType("timm")
        fake_timm.__version__ = "1.0.0"
        fake_timm_data = types.ModuleType("timm.data")
        fake_timm_data.resolve_data_config = lambda *args, **kwargs: {}
        fake_timm_models = types.ModuleType("timm.models")
        fake_timm_models.create_model = lambda *args, **kwargs: None
        fake_timm_models.list_models = lambda *args, **kwargs: []
        fake_timm.data = fake_timm_data
        fake_timm.models = fake_timm_models

        original_import_module = importlib.import_module

        def import_module(name, package=None):
            # [FUNCTION DOC] HELPER: Replacement for module import that raises
            # ModuleNotFoundError when asked for timm and otherwise forwards to the original
            # import function.
            if name == "timm":
                raise ModuleNotFoundError("No module named 'timm'")
            return original_import_module(name, package)

        with (
            mock.patch.dict(
                sys.modules,
                {
                    "timm": fake_timm,
                    "timm.data": fake_timm_data,
                    "timm.models": fake_timm_models,
                },
            ),
            mock.patch("importlib.import_module", side_effect=import_module),
            mock.patch("subprocess.check_call") as check_call,
        ):
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            finally:
                sys.modules.pop(module_name, None)

        check_call.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "git+https://github.com/rwightman/pytorch-image-models",
            ]
        )

    def test_prepare_repro_installs_timm_without_deps(self) -> None:
        # [FUNCTION DOC] Runs the perf command line repro preparation for the timm suite with a
        # patched pin reader that returns a fixed commit hash for the timm pin file, capturing
        # stdout. Asserts the printed text contains a pip install command with no-deps pointing
        # at the huggingface timm repository pinned at that commit.
        from . import perf_cli

        def read_pin(name):
            # [FUNCTION DOC] HELPER: Fake pin reader returning a fixed hash for the timm pin
            # file name and a placeholder string for any other name.
            return "abc123" if name == "timm.txt" else "unused"

        output = io.StringIO()
        args = types.SimpleNamespace(suite="timm", no_repro=True)
        with (
            mock.patch.object(perf_cli, "read_pin", side_effect=read_pin),
            redirect_stdout(output),
        ):
            perf_cli.cmd_prepare_repro(args)

        self.assertIn(
            "pip install --no-deps "
            "git+https://github.com/huggingface/pytorch-image-models@abc123",
            output.getvalue(),
        )

    def test_dashboard_performance_uses_warm_peak_memory(self) -> None:
        # [FUNCTION DOC] Parses arguments for a cuda inductor inference performance run. With
        # the dashboard flag the use_warm_peak_memory option must be true; without it must be
        # false.
        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
                "--dashboard",
            ]
        )
        self.assertTrue(args.use_warm_peak_memory)

        args = parse_args(
            [
                "-dcuda",
                "--inductor",
                "--inference",
                "--performance",
            ]
        )
        self.assertFalse(args.use_warm_peak_memory)

    def test_detectron2_maskrcnn_uses_iou_for_bool_masks(self) -> None:
        # [FUNCTION DOC] For the two detectron2 maskrcnn model names, asserts the TorchBench
        # runner says to use intersection-over-union for boolean accuracy checks and that the
        # iou threshold is 0.99.
        runner = TorchBenchmarkRunner()
        for name in (
            "detectron2_maskrcnn_r_101_fpn",
            "detectron2_maskrcnn_r_50_c4",
        ):
            self.assertTrue(runner.use_iou_for_bool_accuracy(name))
            self.assertEqual(runner.get_iou_threshold(name), 0.99)

    @unittest.skipIf(is_asan_or_tsan(), "ASAN/TSAN not supported")
    def test_benchmark_infra_runs(self) -> None:
        # [FUNCTION DOC] Smoke test, skipped under address or thread sanitizers. Switches to the
        # TorchBench working directory, runs a one-iteration cpu inductor training performance
        # run of the BERT_pytorch model with batch size 1 through the runner, and always
        # restores the original working directory. Fails on any exception from the run.
        """
        Basic smoke test that TorchBench runs.

        This test is mainly meant to check that our setup in fbcode
        doesn't break.

        If you see a failure here related to missing CPP headers, then
        you likely need to update the resources list in:
            //caffe2:inductor
        """
        original_dir = setup_torchbench_cwd()
        try:
            args = parse_args(
                [
                    "-dcpu",
                    "--inductor",
                    "--training",
                    "--performance",
                    "--only=BERT_pytorch",
                    "-n1",
                    "--batch-size=1",
                ]
            )
            run(TorchBenchmarkRunner(), args, original_dir)
        finally:
            os.chdir(original_dir)
