# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor AutoHeuristic data collection and usage, mainly for the
# pad_mm and mixed_mm heuristics. Tests compile small matmuls on the GPU device and inspect
# the autoheuristic log files written under the inductor cache directory. Many tests are
# skipped on XPU, and the GPU-specific ones require A100 or H100. When run as a script the
# tests run only if a GPU is available.
import os
import unittest
from unittest.mock import patch

import torch
import torch._inductor.config as inductor_config
from torch._inductor.autoheuristic.autoheuristic import AutoHeuristic, LocalFeedback
from torch._inductor.autoheuristic.autoheuristic_utils import AHContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import get_gpu_shared_memory
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_A100, IS_H100


class AutoHeuristicTest(TestCase):
    # [CLASS DOC] Test case for AutoHeuristic: log-file data collection, the config gates that
    # control collection and use, deterministic-mode behavior and construction without CUDA.
    # Provides helpers to run matmuls, locate the per-device log file and count its lines.
    def count_lines_in_file(self, file_path):
        # [FUNCTION DOC] HELPER: Returns the number of lines in the text file at file_path.
        with open(file_path) as file:
            line_count = sum(1 for line in file)
        return line_count

    def run_mm(self):
        # [FUNCTION DOC] HELPER: Compiles a plain matmul and runs it once on half-precision GPU
        # tensors of shape 2047 by 2048 and 2048 by 2048 (an unaligned shape for pad_mm).
        def f(a, b):
            # [FUNCTION DOC] HELPER: Compiled function; returns the matrix product of a and b.
            return torch.mm(a, b)

        cf = torch.compile(f)
        a = torch.randn(2047, 2048, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2048, 2048, device=GPU_TYPE, dtype=torch.float16)
        cf(a, b)

    def get_path_to_autoheuristic_log(self, name):
        # [FUNCTION DOC] HELPER: Returns the path of the autoheuristic log text file for the given
        # heuristic name, under the inductor cache directory, in a subdirectory named by the
        # AutoHeuristic device identifier.
        device_name = AutoHeuristic.get_device_identifier()
        path = cache_dir() + "/autoheuristic/" + device_name + "/" + name + ".txt"
        return path

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    def test_autoheuristic_pad_mm_default(self):
        # [FUNCTION DOC] Runs a matmul with default config and asserts that no pad_mm autoheuristic
        # log file was created, i.e. no data is collected by default.
        # this test ensures that data is not collected for pad_mm when autoheuristic config is set to its default value
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", False)
    def test_autoheuristic_pad_mm_off(self):
        # [FUNCTION DOC] Runs a matmul with the pad_mm collection flag patched to false and asserts
        # that no pad_mm autoheuristic log file exists.
        # this test ensures that data is not collected for pad_mm when autoheuristic_collect does not contain "pad_mm"
        self.run_mm()
        self.assertFalse(os.path.exists(self.get_path_to_autoheuristic_log("pad_mm")))

    def assert_autoheuristic_collected_data(self):
        # [FUNCTION DOC] HELPER: Runs a matmul, asserts the pad_mm log file exists and that it has
        # exactly four lines (metadata, header and one line for each of the two choices).
        self.run_mm()
        AutoHeuristic.get_device_identifier()
        path = self.get_path_to_autoheuristic_log("pad_mm")
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)

        # 1 line for metadata, 1 line for header, 1 line per choice (orig, padded)
        self.assertEqual(num_lines, 4)

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", True)
    def test_autoheuristic_pad_mm_collect_data(self):
        # [FUNCTION DOC] With the pad_mm collection flag patched to true, asserts that data is
        # collected for pad_mm using the shared assertion helper.
        # this test ensures that data is collected for pad_mm when autoheuristic_collect.pad_mm=True
        self.assert_autoheuristic_collected_data()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_collect.pad_mm", True)
    def test_autoheuristic_pad_mm_collect_data2(self):
        # [FUNCTION DOC] Duplicate of the previous collection test (same patch and same helper);
        # checks again that pad_mm data is collected when its collection flag is true.
        # this test ensures that data is collected for pad_mm when autoheuristic_collect.pad_mm=True
        self.assert_autoheuristic_collected_data()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @patch.dict(os.environ, {"TORCHINDUCTOR_AUTOHEURISTIC_COLLECT": "test"})
    def test_autoheuristic(self):
        # [FUNCTION DOC] Exercises AutoHeuristic directly with the collection environment variable
        # patched to a test value. Builds a heuristic over three choices with a local feedback
        # function (1, 2, 3 for a, b, c; raises RuntimeError for others) and one numerical feature.
        # Asserts get_choice returns the fallback (collect-only mode), the collected feedback
        # values match, and the log file has five lines: a metadata line (feature names,
        # shared memory, device capability, name), the header and one row per choice.
        # test basic functionality of autoheuristic
        def fallback():
            # [FUNCTION DOC] HELPER: Fallback choice provider; returns the string fallback.
            return "fallback"

        choices = ["a", "b", "c"]

        def feedback_fn(choice):
            # [FUNCTION DOC] HELPER: Feedback function mapping choice a to 1, b to 2, c to 3;
            # raises RuntimeError for any other choice.
            if choice == "a":
                return 1
            elif choice == "b":
                return 2
            elif choice == "c":
                return 3
            else:
                raise RuntimeError("unexpected choice")

        feedback = LocalFeedback(feedback_fn)
        context = AHContext()
        context.add_feature("fa", 5)
        name = "test"
        autoheuristic = AutoHeuristic(fallback, choices, feedback, context, name)

        # when autoheuristic is configured to only collect data, we always return fallback
        self.assertEqual(autoheuristic.get_choice(), "fallback")
        self.assertEqual(autoheuristic.get_collected_feedback("a"), 1)
        self.assertEqual(autoheuristic.get_collected_feedback("b"), 2)
        self.assertEqual(autoheuristic.get_collected_feedback("c"), 3)

        path = self.get_path_to_autoheuristic_log(name)
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)
        self.assertEqual(num_lines, 5)

        shared_memory = get_gpu_shared_memory()
        device_capa = list(torch.cuda.get_device_capability())

        with open(path) as file:
            lines = file.readlines()
            self.assertTrue('"numerical_features": ["fa"]' in lines[0])
            self.assertTrue('"categorical_features": []' in lines[0])
            self.assertTrue(f'"shared_memory": {shared_memory}' in lines[0])
            self.assertTrue(f'"device_capa": {device_capa}' in lines[0])
            self.assertTrue('"name": "test"' in lines[0])
            self.assertEqual("fa,choice,feedback", lines[1].rstrip())
            self.assertEqual("5,a,1", lines[2].rstrip())
            self.assertEqual("5,b,2", lines[3].rstrip())
            self.assertEqual("5,c,3", lines[4].rstrip())

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_A100, "heuristic only run on A100")
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_autoheuristic_a100(self):
        # [FUNCTION DOC] A100-only smoke test: with pad_mm heuristic use enabled, runs a matmul to
        # confirm nothing breaks. It does not verify the heuristic was actually applied.
        # Make sure heuristic does not break anything
        # TODO (AlnisM): Find a way to check whether heuristic is used
        self.run_mm()

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_H100, "heuristic only run on H100")
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_autoheuristic_h100(self):
        # [FUNCTION DOC] H100-only smoke test: with pad_mm heuristic use enabled, runs a matmul to
        # confirm nothing breaks. It does not verify the heuristic was actually applied.
        # Make sure heuristic does not break anything
        # TODO (AlnisM): Find a way to check whether heuristic is used
        self.run_mm()

    def run_mixed_mm(self):
        # [FUNCTION DOC] HELPER: Compiles in max-autotune-no-cudagraphs mode a matmul of a half
        # precision 8 by 1024 tensor with a transposed int8 1024 by 1024 tensor cast to the first
        # operand's dtype, and runs it once.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Compiled function; casts b to the dtype of a and multiplies.
            return torch.mm(a, b.to(a.dtype))

        a = torch.randn(8, 1024, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randint(
            -128, 127, (1024, 1024), dtype=torch.int8, device=GPU_TYPE
        ).t()
        torch.compile(fn, mode="max-autotune-no-cudagraphs")(a, b)

    # have to set autoheuristic_use="" because if autoheuristic_use="mixed_mm",
    # autoheuristic creates a precompile key, puts it into the registry, and then
    # a choice made by the heuristic might be added to the list of choices
    # and if select_algorithm now creates a new precompile key, it will be
    # different from the precompile key created by autoheuristic
    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skip(
        "mixed_mm autoheuristic collection is broken after mixed_mm special casing deletion (PR #147151)"
    )
    @inductor_config.patch("autoheuristic_collect.mixed_mm", True)
    @inductor_config.patch("autoheuristic_use.mixed_mm", False)
    @inductor_config.patch(fx_graph_cache=False)
    @inductor_config.patch(fx_graph_remote_cache=False)
    def test_global_feedback(self):
        # [FUNCTION DOC] Skipped test (mixed_mm collection is broken after special-casing removal).
        # It would collect mixed_mm data with use disabled and caches off, run the mixed matmul and
        # assert the log exists with more than four lines.
        self.run_mixed_mm()
        path = self.get_path_to_autoheuristic_log("mixed_mm")
        self.assertTrue(os.path.exists(path))
        num_lines = self.count_lines_in_file(path)

        # 1 line for metadata, 1 line for header
        # 1 line for fallback + at least 1 config
        self.assertTrue(num_lines > 4)

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @inductor_config.patch("autoheuristic_use.mixed_mm", True)
    @unittest.skipIf(not IS_A100, "heuristic only run on A100")
    def test_mixed_mm_a100(self):
        # [FUNCTION DOC] A100-only smoke test: with mixed_mm heuristic use enabled, runs the mixed
        # matmul. No assertion about the heuristic being used.
        self.run_mixed_mm()
        # TODO (AlnisM): Find a way to check whether heuristic is used

    @skipIfXpu(msg="AutoHeuristic doesn't currently work on the XPU stack")
    @unittest.skipIf(not IS_H100 and not IS_A100, "heuristic only run on H100")
    @inductor_config.patch(deterministic=True)
    @inductor_config.patch("autoheuristic_use.pad_mm", True)
    def test_pad_mm_autoheuristic_deterministic_mode(self):
        # [FUNCTION DOC] On A100 or H100 with deterministic mode and pad_mm heuristic use enabled,
        # compiles an unaligned matmul, checks the result is close to eager within loose tolerance,
        # and asserts the pad_mm benchmarking counter stays zero (the decision is made without
        # benchmarking). Clears the dynamo counters first.
        """Test that pad_mm AutoHeuristics works in deterministic mode."""
        from torch._dynamo.utils import counters

        counters.clear()

        def f(a, b):
            # [FUNCTION DOC] HELPER: Compiled function; returns the matrix product of a and b.
            return torch.mm(a, b)

        cf = torch.compile(f)
        # Use shapes that would normally trigger padding but aren't well-aligned
        a = torch.randn(2047, 2048, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(2048, 2049, device=GPU_TYPE, dtype=torch.float16)

        # Run the compiled function
        result = cf(a, b)

        # Verify correctness with tolerance for potential padding differences
        expected = torch.mm(a, b)
        torch.testing.assert_close(result, expected, atol=0.1, rtol=0.05)

        # In deterministic mode with AutoHeuristics enabled,
        # we should not do any benchmarking (pad_mm_bench should stay 0)
        # because AutoHeuristics makes the decision without benchmarking
        self.assertEqual(counters["inductor"]["pad_mm_bench"], 0)

    @inductor_config.patch(deterministic=True)
    def test_use_autoheuristic_pad_mm_enabled_in_deterministic_mode(self):
        # [FUNCTION DOC] In deterministic mode, sets the pad_mm autoheuristic use config to None
        # and asserts use_autoheuristic reports true for pad_mm.
        # KNOWN ISSUE: the description string sits after the assignment so it is not a real
        # docstring; also the config is mutated directly and never restored afterwards.
        inductor_config.autoheuristic_use.pad_mm = None
        """pad_mm set to None; with deterministic=True, use_autoheuristic should return True."""
        self.assertTrue(inductor_config.use_autoheuristic("pad_mm"))

    def test_autoheuristic_init_no_cuda(self):
        # [FUNCTION DOC] Checks AutoHeuristic can be constructed with CUDA reported unavailable and
        # shared memory patched to zero, and that the resulting metadata device capability is the
        # pair of zeros.
        """AutoHeuristic.__init__ must not crash in non-CUDA environments."""

        def fallback():
            # [FUNCTION DOC] HELPER: Fallback choice provider; returns the string fallback.
            return "fallback"

        context = AHContext()
        context.add_feature("x", 1)

        with (
            patch("torch.cuda.is_available", return_value=False),
            patch(
                "torch._inductor.autoheuristic.autoheuristic.get_gpu_shared_memory",
                return_value=0,
            ),
        ):
            ah = AutoHeuristic(fallback, ["a", "b"], None, context, "test_no_cuda")

        self.assertEqual(ah.metadata.device_capa, (0, 0))


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
