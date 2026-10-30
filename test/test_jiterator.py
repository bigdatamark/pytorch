# Owner(s): ["module: cuda"]
# [MODULE DOC] CUDA-only tests for the Python jiterator, which builds elementwise kernels from C++
# template code strings and runs them on tensors. Compares jitted results to eager reference
# functions for dtype combinations, strides, extra keyword arguments, multiple functors, input and
# output counts and invalid code. Import-time side effects: when CUDA is unavailable it prints a
# skip message to stderr and replaces the base test class with a no-op class; it also builds one
# shared jitted function and instantiates the device-type tests for the cuda device only.

import torch
from torch.cuda.jiterator import _create_jit_fn as create_jit_fn
from torch.cuda.jiterator import _create_multi_output_jit_fn as create_multi_output_jit_fn
import sys
from itertools import product
from torch.testing._internal.common_utils import TestCase, parametrize, run_tests, TEST_CUDA, NoTest
from torch.testing._internal.common_dtype import all_types_and_complex_and
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests, dtypes, toleranceOverride, tol)

if not TEST_CUDA:
    print('CUDA not available, skipping tests', file=sys.stderr)
    TestCase = NoTest


code_string = "template <typename T> T my_fused_kernel(T x, T y, T alpha, T beta) { return alpha * x + beta * y; }"
jitted_fn = create_jit_fn(code_string, alpha=1, beta=1)

def ref_fn(x, y, alpha=1, beta=1):
    # [FUNCTION DOC] HELPER: Eager reference for the shared jitted kernel: returns alpha times the
    # first input plus beta times the second, with both scale factors defaulting to one.
    return alpha * x + beta * y

class TestPythonJiterator(TestCase):
    # [CLASS DOC] Test case for the jiterator. Test methods mostly take a device argument supplied
    # by device-type instantiation, and a few use the cuda device directly. Relies on the
    # module-level jitted function and eager reference.
    @parametrize("shape_strides", [
        (([3, 3], [3, 1]), ([3, 3], [3, 1])),  # contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_contiguous(self, device, dtypes, shape_strides):
        # [FUNCTION DOC] For every pair of dtypes, builds two 3x3 contiguous strided views over
        # random buffers cast to those dtypes, runs the jitted add-like kernel and the eager
        # reference, and asserts equal results. Exercises type promotion across all dtype pairs.
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    @parametrize("shape_strides", [
        (([3, 3], [1, 3]), ([3, 1], [1, 3])),  # non-contiguous
    ])
    @dtypes(*product(all_types_and_complex_and(torch.half, torch.bfloat16),
                     all_types_and_complex_and(torch.half, torch.bfloat16)))
    def test_all_dtype_noncontiguous(self, device, dtypes, shape_strides):
        # [FUNCTION DOC] Same as the contiguous dtype test but uses transposed or broadcast-like
        # strides, so the first view is a transposed 3x3 and the second has stride one in the
        # first dim and three in the second over three rows by one column. Asserts the jitted
        # result equals the eager reference.
        a_buffer = torch.rand(9, device=device).mul(10).type(dtypes[0])
        b_buffer = torch.rand(9, device=device).mul(10).type(dtypes[1])

        a = a_buffer.as_strided(*shape_strides[0])
        b = b_buffer.as_strided(*shape_strides[1])

        expected = ref_fn(a, b)
        result = jitted_fn(a, b)

        self.assertEqual(expected, result)

    @dtypes(torch.float, torch.double, torch.float16, torch.bfloat16)
    @parametrize("alpha", [-1, 2.0, None])
    @parametrize("beta", [3, -4.2, None])
    @toleranceOverride({torch.float16 : tol(atol=1e-2, rtol=1e-3)})
    def test_extra_args(self, device, dtype, alpha, beta):
        # [FUNCTION DOC] Runs the jitted kernel and eager reference on random length-3 tensors of
        # the given float dtype, passing the alpha and beta keyword arguments only when they are
        # not None, so defaults are covered. Half precision uses a looser tolerance override.
        # Asserts equal results.
        a = torch.rand(3, device=device).mul(10).type(dtype)
        b = torch.rand(3, device=device).mul(10).type(dtype)

        extra_args = {}
        if alpha is not None:
            extra_args["alpha"] = alpha
        if beta is not None:
            extra_args["beta"] = beta

        expected = ref_fn(a, b, **extra_args)
        result = jitted_fn(a, b, **extra_args)

        self.assertEqual(expected, result)

    @parametrize("is_train", [True, False])
    def test_bool_extra_args(self, device, is_train):
        # [FUNCTION DOC] Builds a jitted kernel with a boolean extra argument defaulting to false,
        # then calls it with the parametrized flag and compares with an eager reference that
        # multiplies by the mask only when the flag is true. Uses local definitions that shadow
        # the module-level ones.
        code_string = "template <typename T> T conditional(T x, T mask, bool is_train) { return is_train ? x * mask : x; }"
        jitted_fn = create_jit_fn(code_string, is_train=False)

        def ref_fn(x, mask, is_train):
            # [FUNCTION DOC] HELPER: Eager reference returning the input times the mask when the
            # flag is true, otherwise the input unchanged.
            return x * mask if is_train else x

        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)

        expected = ref_fn(a, b, is_train=is_train)
        result = jitted_fn(a, b, is_train=is_train)
        self.assertEqual(expected, result)

    def test_multiple_functors(self, device):
        # [FUNCTION DOC] Compiles a code string that defines two functor templates, where the main
        # one calls the helper one, and checks that the jitted result of three random tensors
        # equals multiply-then-add computed eagerly. Shows that the last function in the string is
        # used as the entry point.
        code_string = '''
        template <typename T> T fn(T x, T mask) { return x * mask; }
        template <typename T> T main_fn(T x, T mask, T y) { return fn(x, mask) + y; }
        '''
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(x, mask, y):
            # [FUNCTION DOC] HELPER: Eager reference returning the first input times the mask plus
            # the third input.
            return x * mask + y

        a = torch.rand(3, device=device)
        b = torch.rand(3, device=device)
        c = torch.rand(3, device=device)

        expected = ref_fn(a, b, c)
        result = jitted_fn(a, b, c)
        self.assertEqual(expected, result)

    @parametrize("num_inputs", [1, 5, 8])
    def test_various_num_inputs(self, num_inputs):
        # [FUNCTION DOC] Generates, for the given input count, a code string for a kernel summing
        # all its inputs, jits it, and compares against the eager sum of the stacked random cuda
        # tensors. Covers one, five and eight inputs.
        inputs = []
        for _ in range(num_inputs):
            inputs.append(torch.rand(3, device='cuda').mul(10))

        input_string = ",".join([f"T i{i}" for i in range(num_inputs)])
        function_body = "+".join([f"i{i}" for i in range(num_inputs)])
        code_string = f"template <typename T> T my_kernel({input_string}) {{ return {function_body}; }}"
        jitted_fn = create_jit_fn(code_string)

        def ref_fn(*inputs):
            # [FUNCTION DOC] HELPER: Eager reference that stacks the given tensors and sums over
            # the stacking dimension.
            return torch.sum(torch.stack(inputs), dim=0)

        expected = ref_fn(*inputs)
        result = jitted_fn(*inputs)

        self.assertEqual(expected, result)

    @parametrize("num_outputs", [1, 4, 8])
    def test_various_num_outputs(self, num_outputs):
        # [FUNCTION DOC] Generates a code string for a void kernel with one input and the given
        # number of reference output parameters, each set to the input plus its index, and builds
        # it with the multi-output factory. Compares each returned output with the eager
        # reference. Return type must be void because ROCm otherwise silently fails.
        input = torch.rand(3, device='cuda')

        output_string = ", ".join([f"T& out{i}" for i in range(num_outputs)])
        function_body = ""
        for i in range(num_outputs):
            function_body += f"out{i} = input + {i};\n"
        # NB: return type must be void, otherwise ROCm silently fails
        code_string = f"template <typename T> void my_kernel(T input, {output_string}) {{ {function_body} }}"

        jitted_fn = create_multi_output_jit_fn(code_string, num_outputs)

        def ref_fn(input):
            # [FUNCTION DOC] HELPER: Eager reference returning the input plus index for each
            # output, as a single tensor when there is one output, otherwise a tuple.
            outputs = []
            for i in range(num_outputs):
                outputs.append(input + i)

            if num_outputs == 1:
                return outputs[0]
            return tuple(outputs)

        expected = ref_fn(input)
        result = jitted_fn(input)

        for i in range(num_outputs):
            self.assertEqual(expected[i], result[i])

    @parametrize("code_string", [
        "template <typename T> T my _kernel(T x) { return x; }",
        "template <typename T> Tmy_kernel(T x) { return x; }",
    ])
    def test_invalid_function_name(self, code_string):
        # [FUNCTION DOC] Checks that building a jitted function from a malformed
        # code string (a function name containing a space, or a return type fused
        # to the name) raises an exception.
        with self.assertRaises(Exception):
            create_jit_fn(code_string)


instantiate_device_type_tests(TestPythonJiterator, globals(), only_for="cuda")

if __name__ == '__main__':
    run_tests()
