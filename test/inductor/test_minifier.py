# Owner(s): ["module: inductor"]
# [MODULE DOC] End-to-end tests for the inductor minifier, run in-process-isolated subprocesses
# via the shared minifier test base. Each test injects a deliberate relu bug (compile error,
# accuracy error) through testing-only inductor config for the CPU or Triton backend, runs a
# generated program, then checks the minifier and repro scripts reproduce the error. Covers
# compile and accuracy errors after AOT autograd, constants, strict versus default accuracy,
# disk offload, and AOTInductor packaging with flat and dict plus keyword inputs. The main block
# skips macOS and ASAN.
import unittest
from unittest.mock import patch

import torch._dynamo.config as dynamo_config
import torch._inductor.config as inductor_config
from torch._dynamo.test_minifier_common import MinifierTestBase
from torch._inductor import config
from torch.export import load as export_load
from torch.testing._internal.common_utils import IS_JETSON, IS_MACOS, TEST_WITH_ASAN
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu


class MinifierTests(MinifierTestBase):
    # [CLASS DOC] Minifier test cases built on the shared minifier test base; every test builds
    # program source text, hands it to the full-flow runner without subprocess isolation, and
    # compares errors or minified graphs.
    # Test that compile and accuracy errors after aot can be repro'd (both CPU and CUDA)
    def _test_after_aot(self, device, expected_error):
        # [FUNCTION DOC] HELPER: Runs the after-AOT minifier flow for a tiny compiled function
        # (relu then cos on a 20 by 20 tensor on the given device), expecting the given error
        # name in the output. Not isolated.
        # NB: The program is intentionally quite simple, just enough to
        # trigger one minification step, no more (dedicated minifier tests
        # should exercise minifier only)
        run_code = f"""\
@torch.compile()
def inner(x):
    x = torch.relu(x)
    x = torch.cos(x)
    return x

inner(torch.randn(20, 20).to("{device}"))
"""
        self._run_full_test(run_code, "aot", expected_error, isolate=False)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_cpu_compile_error(self):
        # [FUNCTION DOC] With the CPU relu bug set to produce a compile error, checks the
        # minifier flow reports a C++ compile error. Skipped on Jetson.
        self._test_after_aot("cpu", "CppCompileError")

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_cpu_accuracy_error(self):
        # [FUNCTION DOC] With the CPU relu bug set to produce wrong results, checks the minifier
        # flow reports an accuracy error. Skipped on Jetson.
        self._test_after_aot("cpu", "AccuracyError")

    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "compile_error")
    def test_after_aot_gpu_compile_error(self):
        # [FUNCTION DOC] With the Triton relu bug producing a compile error, checks the flow
        # reports a syntax error on the GPU device. Requires a GPU.
        self._test_after_aot(GPU_TYPE, "SyntaxError")

    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_after_aot_gpu_accuracy_error(self):
        # [FUNCTION DOC] With the Triton relu bug producing wrong results, checks the flow
        # reports an accuracy error on the GPU device. Requires a GPU.
        self._test_after_aot(GPU_TYPE, "AccuracyError")

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_constant_in_graph(self):
        # [FUNCTION DOC] Compiles a function adding a constant tensor to relu of the input with
        # the CPU accuracy bug on, and checks that an accuracy error is still found after AOT.
        run_code = """\
@torch.compile()
def inner(x):
    return torch.tensor(2) + torch.relu(x)

inner(torch.randn(2))
"""
        self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)

    @requires_gpu
    @patch.object(config, "joint_graph_constant_folding", False)
    def test_rmse_improves_over_atol(self):
        # [FUNCTION DOC] Uses a half-precision subtraction on the GPU, with joint graph constant
        # folding disabled, where the compiled result differs slightly from eager. With the fp64
        # comparison disabled it expects an accuracy error (using strict accuracy so the
        # minifier proceeds); with the default fp64 comparison it expects no error at all,
        # showing the root-mean-square check is more tolerant than absolute tolerance.
        # From https://twitter.com/itsclivetime/status/1651135821045719041?s=20
        run_code = """
@torch.compile()
def inner(x):
    return x - torch.tensor(655, dtype=torch.half, device='GPU_TYPE') * 100

inner(torch.tensor(655 * 100, dtype=torch.half, device='GPU_TYPE'))
""".replace("GPU_TYPE", GPU_TYPE)

        # If we disable RMSE against fp64, this triggers accuracy error,
        # as the increased precision from torch.compile changes the result
        # of 655 * 100
        with dynamo_config.patch("same_two_models_use_fp64", False):
            self._run_full_test(
                run_code,
                "aot",
                "AccuracyError",
                isolate=False,
                # NB: need this to avoid refusing to minify when fp64 doesn't work
                # (which it doesn't, due to the config patch above)
                minifier_args=["--strict-accuracy"],
            )

        # But using fp64, we see that the intended semantics is the increased
        # 655 * 100 precision, and so we report no problem
        self._run_full_test(run_code, "aot", None, isolate=False)

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    @inductor_config.patch("cpp.inject_log1p_bug_TESTING_ONLY", "accuracy")
    def test_accuracy_vs_strict_accuracy(self):
        # [FUNCTION DOC] With CPU relu and log1p accuracy bugs on, compiles a function with
        # log1p, a boolean mask doubly negated, relu and a where. Strict accuracy mode must
        # minify to a graph containing only log1p; default mode must minify to a graph
        # containing only relu. Compares the minified repro module text with expected strings.
        run_code = """
@torch.compile()
def inner(x):
    y = torch.log1p(x)
    b = y > 0
    # Need to ensure suffix removal hits a boolean output
    b = torch.logical_not(b)
    b = torch.logical_not(b)
    x = torch.relu(x)
    return torch.where(b, x, x)

inner(torch.randn(20))
"""

        # Strict accuracy gets hung up on the boolean mask difference, which
        # will localize the error to sigmoid, even though it doesn't actually
        # matter to the end result
        res = self._run_full_test(
            run_code,
            "aot",
            "AccuracyError",
            isolate=False,
            minifier_args=["--strict-accuracy"],
        )
        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg0_1):
        log1p = torch.ops.aten.log1p.default(arg0_1);  arg0_1 = None
        return (log1p,)""",
        )

        # FP accuracy will refuse to promote the logical_not on the outputs,
        # and so you'll get to the relu (unless the minifier somehow tries
        # removing entire suffix except the log1p first!)
        res = self._run_full_test(run_code, "aot", "AccuracyError", isolate=False)
        self.assertExpectedInline(
            res.repro_module(),
            """\
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, arg0_1):
        relu = torch.ops.aten.relu.default(arg0_1);  arg0_1 = None
        return (relu,)""",
        )

    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_offload_to_disk(self):
        # [FUNCTION DOC] Smoke test: with the CPU relu accuracy bug on, minifies a chain of sin,
        # sin, cos, relu with the offload-to-disk option. Only checks that an accuracy error is
        # found; does not measure memory.
        # Just a smoketest, this doesn't actually test that memory
        # usage went down.  Test case is carefully constructed to hit
        # delta debugging.
        run_code = """\
@torch.compile()
def inner(x):
    x = torch.sin(x)
    x = torch.sin(x)
    x = torch.cos(x)
    x = torch.relu(x)
    return x

inner(torch.randn(20, 20))
"""
        self._run_full_test(
            run_code,
            "aot",
            "AccuracyError",
            isolate=False,
            minifier_args=["--offload-to-disk"],
        )

    # Test that compile errors in AOTInductor can be repro'd (both CPU and CUDA)
    def _test_aoti(self, device, expected_error):
        # [FUNCTION DOC] HELPER: Runs the AOTInductor minifier flow for a small exported module
        # (linear, relu, sigmoid on 8 by 10 inputs) packaged with aoti_compile_and_package under
        # no-grad mode on the given device, expecting the given error name. Returns the result
        # object from the runner.
        # NB: The program is intentionally quite simple, just enough to
        # trigger one minification step, no more (dedicated minifier tests
        # should exercise minifier only)
        run_code = f"""\
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x
with torch.no_grad():
    model = Model().to("{device}")
    example_inputs = (torch.randn(8, 10).to("{device}"),)
    ep = torch.export.export(
        model, example_inputs
    )
    torch._inductor.aoti_compile_and_package(
        ep
    )
"""
        return self._run_full_test(
            run_code, "aot_inductor", expected_error, isolate=False
        )

    # Test that compile errors in AOTInductor can be repro'd (both CPU and CUDA)
    def _test_aoti_unflattened_inputs(self, device, expected_error):
        # [FUNCTION DOC] HELPER: Like the plain AOTInductor helper but the module takes a
        # dictionary of two tensors positionally plus a keyword-only tensor, exercising
        # unflattened inputs and keyword arguments. Returns the runner result.
        # NB: The program is intentionally quite simple, just enough to
        # trigger one minification step, no more (dedicated minifier tests
        # should exercise minifier only)

        # It tests that the minifier can handle unflattened inputs and kwargs
        run_code = f"""\
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, inp, *, k):
        x = inp["x"]
        y = inp["y"]
        x = self.fc1(x)
        y = self.fc1(y)
        k = self.fc1(k)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x + y + k

with torch.no_grad():
    model = Model().to("{device}")
    val = torch.randn(8, 10).to("{device}")
    example_inputs = ({{"x": val.clone(), "y": val.clone()}},)
    kwargs = {{"k": val.clone()}}
    ep = torch.export.export(
        model, example_inputs, kwargs
    )
    torch._inductor.aoti_compile_and_package(ep)
"""
        return self._run_full_test(
            run_code, "aot_inductor", expected_error, isolate=False
        )

    def _aoti_check_relu_repro(self, res):
        # [FUNCTION DOC] HELPER: Asserts the result exists and has an exported program path,
        # loads that program, and compares its generated forward code with the expected text,
        # which holds only a relu on the linear output.
        if res is None:
            raise AssertionError("res is None")
        ep_file_path = res.get_exported_program_path()
        if ep_file_path is None:
            raise AssertionError("ep_file_path is None")
        gm = export_load(ep_file_path).module(check_guards=False)
        self.assertExpectedInline(
            str(gm.code).strip(),
            """\
def forward(self, linear_default):
    linear_default, = fx_pytree.tree_flatten_spec(([linear_default], {}), self._in_spec)
    relu_default = torch.ops.aten.relu.default(linear_default);  linear_default = None
    return pytree.tree_unflatten((relu_default,), self._out_spec)""",
        )

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch(
        "cpp.inject_relu_bug_TESTING_ONLY",
        "compile_error",
    )
    def test_aoti_cpu_compile_error(self):
        # [FUNCTION DOC] AOTI on CPU with the compile-error bug injected: expects a C++ compile
        # error and a minified graph containing only relu. Skipped on Jetson.
        res = self._test_aoti("cpu", "CppCompileError")
        self._aoti_check_relu_repro(res)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch(
        "cpp.inject_relu_bug_TESTING_ONLY",
        "compile_error",
    )
    def test_aoti_cpu_compile_error_unflatten(self):
        # [FUNCTION DOC] Same as the CPU compile-error AOTI test but with dictionary and keyword
        # inputs. Skipped on Jetson.
        res = self._test_aoti_unflattened_inputs("cpu", "CppCompileError")
        self._aoti_check_relu_repro(res)

    @requires_gpu
    @inductor_config.patch(
        "triton.inject_relu_bug_TESTING_ONLY",
        "compile_error",
    )
    def test_aoti_gpu_compile_error(self):
        # [FUNCTION DOC] AOTI on the GPU with the Triton compile-error bug: expects a syntax
        # error and a relu-only minified graph. Requires a GPU.
        res = self._test_aoti(GPU_TYPE, "SyntaxError")
        self._aoti_check_relu_repro(res)

    @requires_gpu
    @inductor_config.patch(
        "triton.inject_relu_bug_TESTING_ONLY",
        "compile_error",
    )
    def test_aoti_gpu_compile_error_unflatten(self):
        # [FUNCTION DOC] Same as the GPU compile-error AOTI test but with dictionary and keyword
        # inputs. Requires a GPU.
        res = self._test_aoti_unflattened_inputs(GPU_TYPE, "SyntaxError")
        self._aoti_check_relu_repro(res)

    @unittest.skipIf(IS_JETSON, "Fails on Jetson")
    @inductor_config.patch("cpp.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_aoti_cpu_accuracy_error(self):
        # [FUNCTION DOC] AOTI on CPU with the accuracy bug: expects an accuracy error and a
        # relu-only minified graph. Skipped on Jetson.
        res = self._test_aoti("cpu", "AccuracyError")
        self._aoti_check_relu_repro(res)

    @requires_gpu
    @inductor_config.patch("triton.inject_relu_bug_TESTING_ONLY", "accuracy")
    def test_aoti_gpu_accuracy_error(self):
        # [FUNCTION DOC] AOTI on the GPU with the Triton accuracy bug: expects an accuracy error
        # and a relu-only minified graph. Requires a GPU.
        res = self._test_aoti(GPU_TYPE, "AccuracyError")
        self._aoti_check_relu_repro(res)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    # Skip CI tests on mac since CPU inductor does not seem to work due to C++ compile errors,
    # also skip on ASAN due to https://github.com/pytorch/pytorch/issues/98262
    if not IS_MACOS and not TEST_WITH_ASAN:
        run_tests()
