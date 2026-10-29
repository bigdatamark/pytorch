# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the ConfigFuzzer in torch._inductor.fuzzer: sampling methods (toggle and
# random), end-to-end fuzzing of the inductor config on CPU/GPU, bisection that isolates the
# config settings that make a test function fail (by exception or by returning false), n-tuple
# fuzzing, and checks that bisect compiles and runs the test once per attempt. Uses a fake config
# module for deterministic cases. Imports MagicMock and patch for call counting.

import unittest
from typing import Literal
from unittest.mock import MagicMock, patch

import torch
from torch._dynamo import config as dynamo_config
from torch._inductor import config as inductor_config
from torch._inductor.fuzzer import ConfigFuzzer, MODULE_DEFAULTS, SamplingMethod, Status
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal import fake_config_module as fake_config
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


def create_simple_test_model_cpu():
    # [FUNCTION DOC] HELPER: Returns a test function that builds a small sequential model (linear,
    # relu, linear) on CPU, runs it on a random batch of 32 by 10 inputs, and returns true.
    def test_fn() -> bool:
        # [FUNCTION DOC] HELPER: Constructs the small model, runs one forward pass, returns true.
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 10), torch.nn.ReLU(), torch.nn.Linear(10, 1)
        )

        x = torch.randn(32, 10)
        model(x)
        return True

    return test_fn


def create_simple_test_model_gpu():
    # [FUNCTION DOC] HELPER: Allocates (at call time, on the GPU device) a random activation of
    # 32 by 50 by 768 and a 768 by 768 weight, and returns a test function that multiplies them and
    # applies a layer norm of width 768, returning true.
    batch_size = 32
    seq_length = 50
    hidden_size = 768

    inp = torch.randn(batch_size, seq_length, hidden_size, device=GPU_TYPE)
    weight = torch.randn(hidden_size, hidden_size, device=GPU_TYPE)

    def test_fn() -> bool:
        # [FUNCTION DOC] HELPER: Runs the matmul and layer norm on the captured GPU tensors and
        # returns true.
        matmul_output = inp @ weight
        torch.nn.LayerNorm(hidden_size, device=GPU_TYPE)(matmul_output)
        return True

    return test_fn


class TestConfigFuzzer(TestCase):
    # [CLASS DOC] Tests for ConfigFuzzer and SamplingMethod. Several tests use the fake config
    # module (default mapping set empty) with boolean and optional entries so failures are
    # deterministic.
    def test_sampling_method_toggle(self):
        # [FUNCTION DOC] Checks the toggle sampling method: booleans flip, a two-value literal
        # switches to the other value, and for list-of-literal types the result contains the
        # option that was not in the input list.
        toggle = SamplingMethod.dispatch(SamplingMethod.TOGGLE)
        self.assertEqual(toggle("", bool, False), True)
        self.assertEqual(toggle("", bool, True), False)
        self.assertEqual(toggle("", Literal["foo", "bar"], "foo"), "bar")
        self.assertEqual(toggle("", Literal["foo", "bar"], "bar"), "foo")
        self.assertTrue("bar" in toggle("", list[Literal["foo", "bar"]], ["foo"]))
        self.assertTrue("foo" in toggle("", list[Literal["foo", "bar"]], ["bar"]))

    def test_sampling_method_random(self):
        # [FUNCTION DOC] Checks the random sampling method over 1000 boolean samples starting
        # from false: not all samples are true, so the value is not constant.
        random = SamplingMethod.dispatch(SamplingMethod.RANDOM)
        samp = [random("", bool, False) for i in range(1000)]
        self.assertTrue(not all(samp))

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_config_fuzzer_inductor_gpu(self):
        # [FUNCTION DOC] GPU-only (skipped without a GPU). Builds a fuzzer on the inductor config
        # with the GPU test model and a fixed seed, checks a default config is recorded, and
        # reproduces a single run with a max_fusion_size setting of 1. Passes if nothing raises.
        fuzzer = ConfigFuzzer(inductor_config, create_simple_test_model_gpu, seed=30)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_inductor_cpu(self):
        # [FUNCTION DOC] CPU variant of the inductor config fuzzer smoke test: checks defaults
        # exist and reproduces one config with max_fusion_size of 1 using the CPU test model.
        fuzzer = ConfigFuzzer(inductor_config, create_simple_test_model_cpu, seed=100)
        self.assertIsNotNone(fuzzer.default)
        fuzzer.reproduce([{"max_fusion_size": 1}])

    def test_config_fuzzer_bisector_exception(self):
        # [FUNCTION DOC] Bisection on the fake config where the test raises a custom exception
        # when the boolean entry is false and the optional entry is none. With probability 1.0
        # and two attempts, bisect must return two results, each exactly those two settings.
        key_1 = {"e_bool": False, "e_optional": None}

        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception raised by the test function to signal the failing
            # config.
            pass

        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the test function used by the fuzzer.
            def myfn():
                # [FUNCTION DOC] HELPER: Raises MyException when the fake config has the boolean
                # entry false and the optional entry none; otherwise returns true.
                if not fake_config.e_bool and fake_config.e_optional is None:
                    raise MyException("hi")
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        results = fuzzer.bisect(num_attempts=2, p=1.0)
        self.assertEqual(len(results), 2)
        for res in results:
            self.assertEqual(res, key_1)

    def test_config_fuzzer_bisector_boolean(self):
        # [FUNCTION DOC] Bisection where failure is signalled by returning false (not raising)
        # under the same two fake config settings. Expects one result per attempt, each equal to
        # exactly those two settings.
        key_1 = {"e_bool": False, "e_optional": None}

        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the test function used by the fuzzer.
            def myfn():
                # [FUNCTION DOC] HELPER: Returns false when the boolean entry is false and the
                # optional entry is none in the fake config; otherwise returns true.
                if not fake_config.e_bool and fake_config.e_optional is None:
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

    def test_config_fuzzer_n_tuple(self):
        # [FUNCTION DOC] Runs n-tuple fuzzing with pairs on the fake config, capped at 100
        # combinations. Expects the result to report exactly 100 runs and the pair of the two
        # failing keys to be recorded with the failed-run-return status.
        key_1 = {"e_bool": False, "e_optional": None}

        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the test function used by the fuzzer.
            def myfn():
                # [FUNCTION DOC] HELPER: Returns false when the boolean entry is false and the
                # optional entry is none in the fake config; otherwise returns true.
                if not fake_config.e_bool and fake_config.e_optional is None:
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(fake_config, create_key_1, seed=100, default={})
        max_combo = 100
        results = fuzzer.fuzz_n_tuple(2, max_combinations=max_combo)
        self.assertEqual(results.num_ran(), max_combo)
        self.assertEqual(results.lookup(tuple(key_1.keys())), Status.FAILED_RUN_RETURN)

    def test_config_fuzzer_inductor_bisect(self):
        # [FUNCTION DOC] Bisection on the real inductor config where the test returns false when
        # split_reductions is off and compute_all_bounds is on. Each of two attempts must return
        # exactly those two settings; reproducing them yields one result whose keys, after
        # removing the module default keys, equal those two settings.
        # these values just chosen randomly, change to different ones if necessary
        key_1 = {"split_reductions": False, "compute_all_bounds": True}

        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the test function used by the fuzzer.
            def myfn():
                # [FUNCTION DOC] HELPER: Returns false when split_reductions is off and
                # compute_all_bounds is on in the inductor config; otherwise returns true.
                if (
                    not inductor_config.split_reductions
                    and inductor_config.compute_all_bounds
                ):
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

        new_results = fuzzer.reproduce(results)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(
            set(key_1.keys()),
            {j for i in new_results.keys() for j in i}  # noqa: SIM118
            - set(MODULE_DEFAULTS["torch._inductor.config"].keys()),
        )

    @unittest.skipIf(not IS_LINUX, "PerfCounters are only supported on Linux")
    @unittest.skip(
        "Need default values for dynamo flags - https://github.com/pytorch/pytorch/issues/164062"
    )
    def test_config_fuzzer_dynamo_bisect(self):
        # [FUNCTION DOC] Bisection on the dynamo config where the test returns false when
        # specialize_int is on; expects the single setting from each attempt and a matching
        # reproduction. Currently skipped unconditionally (default values for dynamo flags are
        # missing, per the issue in the skip reason) and only meant for Linux.
        # these values just chosen randomly, change to different ones if necessary
        key_1 = {"specialize_int": True}

        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the test function used by the fuzzer.
            def myfn():
                # [FUNCTION DOC] HELPER: Returns false when specialize_int is on in the dynamo
                # config; otherwise returns true.
                if dynamo_config.specialize_int:
                    return False
                return True

            return myfn

        fuzzer = ConfigFuzzer(dynamo_config, create_key_1, seed=10)
        num_attempts = 2
        results = fuzzer.bisect(num_attempts=num_attempts, p=1.0)
        self.assertEqual(len(results), num_attempts)
        for res in results:
            self.assertEqual(res, key_1)

        new_results = fuzzer.reproduce(results)
        self.assertEqual(len(new_results), 1)
        self.assertEqual(
            set(key_1.keys()),
            {j for i in new_results for j in i}
            - set(MODULE_DEFAULTS["torch._dynamo.config"].keys()),
        )

    @patch("torch.compile")
    def test_fuzzer_inductor_calling_compile(self, compile):
        # [FUNCTION DOC] With torch.compile patched out by a mock (passed in as compile), runs
        # bisect for 3 attempts with probability 0.5 using an always-true test, and checks the
        # mock was called once per attempt.
        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the always-true test function.
            def myfn():
                # [FUNCTION DOC] HELPER: Always returns true.
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        num_attempts = 3
        fuzzer.bisect(num_attempts=num_attempts, p=0.5)
        self.assertEqual(compile.call_count, num_attempts)

    def test_fuzzer_running_test(self):
        # [FUNCTION DOC] Replaces the fuzzer's per-config test method with a mock that returns the
        # passed status, runs bisect for 20 attempts, and checks the mock was called 20 times.
        def create_key_1():
            # [FUNCTION DOC] HELPER: Factory returning the always-true test function.
            def myfn():
                # [FUNCTION DOC] HELPER: Always returns true.
                return True

            return myfn

        fuzzer = ConfigFuzzer(inductor_config, create_key_1, seed=100)
        fuzzer.test_config = MagicMock(return_value=Status.PASSED)
        num_attempts = 20
        fuzzer.bisect(num_attempts=num_attempts, p=0.5)
        self.assertEqual(fuzzer.test_config.call_count, num_attempts)


if __name__ == "__main__":
    run_tests()
