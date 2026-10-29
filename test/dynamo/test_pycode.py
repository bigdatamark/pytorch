# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for dynamo's generated Python code (generate_pycode) output. Each test
# captures a full graph with fullgraph_capture, compares the generated wrapper code against an
# expected inline string with compiled function ids normalized, and runs the generated callable
# to compare with eager results. No import-time side effects besides imports.

import re

import torch
import torch._dynamo.test_case
from torch._dynamo.convert_frame import fullgraph_capture
from torch._dynamo.utils import get_metrics_context
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo


class SimpleLinearModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with one linear layer (3 in, 3 out) used as a capture
    # target.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Builds the module and registers a linear layer of 3 inputs and 3
        # outputs.
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to the input and returns the result.
        return self.linear(x)


@torch._dynamo.config.patch(generate_pycode=True)
@skipIfTorchDynamo("Not suitable for generate_pycode=True")
class TestPycode(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case run with generate_pycode enabled and skipped when running under
    # dynamo itself. Verifies the textual form of the pycode produced for a module, dict outputs,
    # the wrapper source and default arguments, and that the generated callable matches eager.
    def test_pycode_module(self):
        # [FUNCTION DOC] Captures a linear module, joins the pycode fragments, normalizes
        # compiled function ids, and checks the code reads parameters from the module, calls the
        # compiled function and returns its first output. Then builds the forward callable with
        # Python codegen and checks it equals eager output.
        mod = SimpleLinearModule()
        x = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(mod, (x,), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = self._modules['linear']._parameters['bias']
__arg1 = self._modules['linear']._parameters['weight']
__arg2 = x
__graph_out = __compiled_fn_<ID>(__arg0, __arg1, __arg2)
__stack0 = __graph_out[0]
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(mod, x)
        expected = mod(x)
        self.assertEqual(result, expected)

    def test_pycode_dict_output(self):
        # [FUNCTION DOC] Same flow for a function returning a dictionary of sum, difference and
        # product: checks the pycode rebuilds the dictionary from indexed graph outputs, and that
        # the generated callable matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test returning a dictionary with the sum,
            # difference and product of the two inputs.
            return {"sum": x + y, "diff": x - y, "prod": x * y}

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(fn, (x, y), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = x
__arg1 = y
__graph_out = __compiled_fn_<ID>(__arg0, __arg1)
__stack0 = {'sum': __graph_out[0], 'diff': __graph_out[1], 'prod': __graph_out[2]}
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)

    def test_pycode_wrapper_source(self):
        # [FUNCTION DOC] Captures the linear module and checks the full generated wrapper source
        # (outer generator function, inner function taking the module and input, and the final
        # assignment of the generated function) against the expected string, after normalizing
        # compiled function ids.
        mod = SimpleLinearModule()
        x = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(mod, (x,), {})
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        source = runtime_env._build_python_wrapper_source()
        source = re.sub(r"__compiled_fn_\d+_[0-9a-f_]+", "__compiled_fn_<ID>", source)
        self.assertExpectedInline(
            source,
            """\
def __generate_func__():
    def __dynamo_func__(self, x):
        __arg0 = self._modules['linear']._parameters['bias']
        __arg1 = self._modules['linear']._parameters['weight']
        __arg2 = x
        __graph_out = __compiled_fn_<ID>(__arg0, __arg1, __arg2)
        __stack0 = __graph_out[0]
        __ret = __stack0
        return __ret
    return __dynamo_func__
__dynamo_generated_func__ = __generate_func__()""",
        )

    def test_pycode_default_args(self):
        # [FUNCTION DOC] Captures a function with scale and bias default arguments called with
        # only two inputs; checks the pycode only uses the two passed inputs and the single graph
        # output, and that the generated callable matches eager.
        def fn(x, y, scale=1.0, bias=0.0):
            # [FUNCTION DOC] HELPER: Function under test computing the first input times scale,
            # plus the second input, plus bias, with scale and bias defaulting to one and zero.
            return x * scale + y + bias

        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        with get_metrics_context():
            capture_output = fullgraph_capture(fn, (x, y), {})
        pycode = capture_output.graph_capture_output.pycode
        pycode_str = "\n".join("\n".join(p) for p in pycode if p is not None)
        pycode_str = re.sub(
            r"__compiled_fn_\d+_[0-9a-f_]+",
            "__compiled_fn_<ID>",
            pycode_str,
        )
        self.assertExpectedInline(
            pycode_str,
            """\
__arg0 = x
__arg1 = y
__graph_out = __compiled_fn_<ID>(__arg0, __arg1)
__stack0 = __graph_out[0]
__ret = __stack0""",
        )
        runtime_env = capture_output.graph_capture_output.get_runtime_env()
        backend_id = capture_output.backend_input.backend_id
        compiled_fn = runtime_env.forward_callable(
            backend_id,
            capture_output.backend_input.graph_module,
            use_python_codegen=True,
        )
        result = compiled_fn(x, y)
        expected = fn(x, y)
        self.assertEqual(result, expected)


if __name__ == "__main__":
    run_tests()
