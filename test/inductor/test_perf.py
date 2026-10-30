# Owner(s): ["module: inductor"]
# [MODULE DOC] Deterministic performance tests for TorchInductor that count memory accesses
# instead of timing. Each test compiles a small function with a backend that runs inductor
# lowering and scheduling (to populate metrics) but executes the graph eagerly, then compares
# the number of fp32-equivalent elements read and written (bytes accessed divided by four)
# against an expected string. Test groups cover pointwise/reduction/cat accounting, fusion,
# scheduler fusion choices, tiling, min-cut partitioning, no-op elimination and in-place
# and mutation handling. Import-time side effects: enables inductor metrics logging, and
# imports triton and its test kernel only when a GPU with triton is available. The device under
# test is the GPU type from the shared test utilities.
import contextlib
import re
import unittest
from unittest.mock import patch

import functorch
import torch
import torch._inductor.config as config
import torch.autograd
from torch._dynamo.device_interface import get_interface_for_device
from torch._inductor import metrics
from torch._inductor.compile_fx import compile_fx, compile_fx_inner
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code

########################
# Explanation of Tests #
########################
# These tests are all testing *memory accesses* of TorchInductor.
# They are intended to be deterministic performance tests.
# The expect tests are all measuring the number of memory bytes read/written by
# the code that Inductor has generated
#
# If the test is failing because the number became smaller, feel free to lower it.
# On the other hand, if the test is failing because the number became larger,
# that means that your change is leading to *more* memory accesses on this test.
#
# That may still be aceeptable, but be aware that you are likely lowering
# performance for that setting.
#
# Defines all the kernels for tests
from torch.testing._internal.common_utils import skipIfXpu, TEST_WITH_ROCM
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton


# set so that metrics appear
torch._logging.set_logs(inductor_metrics=True)

if HAS_GPU_AND_TRITON:
    import triton  # @manual
    import triton.language as tl  # @manual

    from torch.testing._internal.triton_utils import add_kernel

aten = torch.ops.aten


def compile_but_use_eager(gm, example_inputs):
    # [FUNCTION DOC] HELPER: Backend for dynamo: runs the inductor lowering pipeline for the
    # graph with a custom inner compile function that compiles the graph (for its metric side
    # effects) but returns the original graph module, so the result runs eagerly. Returns the
    # compiled callable from the inductor frontend.
    def inner_compile(gm, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner compile hook: invokes the real inner compile on the graph
        # and extra arguments, discards the compiled artifact, and returns the unchanged graph
        # module.
        compile_fx_inner(gm, *args, **kwargs)
        return gm

    return compile_fx(gm, example_inputs, inner_compile=inner_compile)


def count_numel(f, *args):
    # [FUNCTION DOC] HELPER: Resets inductor metrics, compiles the function with the eager-
    # executing backend and calls it with the given arguments, prints the per-node element
    # counts, and returns the number of bytes accessed divided by four as a string (so it can be
    # compared to expected strings). Assumes all inputs are fp32. Side effect: resets and
    # mutates the global metrics.
    """
    Assumes all inputs are fp32
    """
    metrics.reset()
    torch.compile(f, backend=compile_but_use_eager)(*args)
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


def count_numel_train(f, *args):
    # [FUNCTION DOC] HELPER: Like the forward-only counter but for training: compiles the
    # function, calls it, sums the mean of every output, runs backward on that sum, and returns
    # the bytes accessed (divided by four) as a string so forward and backward kernels are both
    # counted. The function must return an iterable of tensors that require grad. Resets global
    # metrics and prints per-node counts.
    """
    Assumes all inputs are fp32
    """
    metrics.reset()

    f = torch.compile(f, backend=compile_but_use_eager)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)


DEVICE = GPU_TYPE


def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    # [FUNCTION DOC] HELPER: Creates a random normal tensor of the given shape with the given
    # dtype, device (default GPU) and requires_grad flag.
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)


def TI(*size, mx=10, dtype=torch.int32, device=DEVICE):
    # [FUNCTION DOC] HELPER: Creates a random integer tensor of the given shape with values in
    # the range from zero up to the exclusive maximum, with the given dtype (default int32) and
    # device.
    return torch.randint(0, mx, size, dtype=dtype, device=device)


class TestCase(InductorTestCase):
    # [CLASS DOC] Base test case for this file: extends the inductor test case and sets the
    # class device attribute to the GPU type used by the helpers.
    device = DEVICE


class NumBytesMetricTests(TestCase):
    # [CLASS DOC] Sanity tests that the bytes-accessed metric is correct for pointwise,
    # reduction, extern matmul, cat and indexing patterns. Stateless.
    """
    Primarily used for sanity testing that the num_bytes_accessed metrics is correct.
    """

    def test_pointwise(self):
        # [FUNCTION DOC] Checks the element-access count for a series of pointwise functions:
        # single unary op, binary op, broadcast, reuse of one input twice, transposed self-add
        # and multiple outputs. Local function names are reused for each case.
        def f(x):
            # [FUNCTION DOC] HELPER: Returns the cosine of its input.
            return x.cos()

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds its two inputs.
            return x + y

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds its two inputs; called with a 2-D and a 1-D tensor so
            # the second is broadcast.
            return x + y

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

        def f(x):
            # [FUNCTION DOC] HELPER: Adds the input to itself.
            return x + x

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """20""")

        def f(x):
            # [FUNCTION DOC] HELPER: Adds the input to its transpose.
            return x + x.t()

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Returns the cosine of the first input and, as a second
            # output, the sum of the sines of the other two.
            return a.cos(), b.sin() + c.sin()

        inp = (T(10), T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    def test_reduction(self):
        # [FUNCTION DOC] Checks the access count of sum reductions over the last and first
        # dimensions of a square matrix.
        def f(x):
            # [FUNCTION DOC] HELPER: Sums over dimension one.
            return x.sum(dim=1)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

        def f(x):
            # [FUNCTION DOC] HELPER: Sums over dimension zero.
            return x.sum(dim=0)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_extern(self):
        # [FUNCTION DOC] Checks the access count when matrix multiplication (an extern kernel)
        # is mixed with pointwise ops before and after it.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by itself as a matrix.
            return torch.mm(x, x)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Matrix product of its two inputs.
            return torch.mm(a, b)

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(x):
            # [FUNCTION DOC] HELPER: Applies cosine, then a matrix product of the result with
            # itself, then cosine again.
            x = x.cos()
            x = torch.mm(x, x)
            x = x.cos()
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        def f(x):
            # [FUNCTION DOC] HELPER: Matrix product of the cosine of the input with the sine of
            # the input.
            a = x.cos()
            b = x.sin()
            x = torch.mm(a, b)
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """600""")

    def test_cat(self):
        # [FUNCTION DOC] Checks the access count for cat in many settings: pointwise inputs,
        # plain inputs, a mix, a shared source, an extern input, three and five offset inputs
        # followed by a pointwise op, and reductions as inputs.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the sines of two inputs.
            return torch.cat([a.sin(), b.sin()])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates two inputs unchanged.
            return torch.cat([a, b])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the cosine of the first input with the second
            # input.
            return torch.cat([a.cos(), b])

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """40""")

        def f(a):
            # [FUNCTION DOC] HELPER: Concatenates the cosine and the sine of the same input.
            return torch.cat([a.cos(), a.sin()])

        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), """30""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates a matrix product of the first input with the
            # sine of the second.
            return torch.cat([torch.mm(a, a), b.sin()])

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Concatenates three inputs each offset by a constant, then
            # adds ten.
            return torch.cat((a + 1, b + 2, c + 3)) + 10

        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        def f(a, b, c, d, e):
            # [FUNCTION DOC] HELPER: Concatenates five inputs each offset by a constant, then
            # adds ten.
            return torch.cat((a + 1, b + 2, c + 3, d + 4, e + 5)) + 10

        inp = [T(10, 10) for _ in range(5)]
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the dimension-zero sums of two 3-D inputs,
            # then adds ten.
            return torch.cat([a.sum(dim=0), b.sum(dim=0)]) + 10

        inp = [T(10, 10, 10), T(10, 10, 10)]
        self.assertExpectedInline(count_numel(f, *inp), """2600""")

    @config.patch("fx_graph_cache", False)
    def test_cat_pointwise(self):
        # [FUNCTION DOC] With the graph cache disabled, checks the access count for cat feeding
        # pointwise ops and for cat with extern inputs (cat is turned into pointwise when some,
        # but not all, inputs are pointwise), plus a padded cat.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the softmax of each input.
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)])

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the softmax of each input, then applies
            # cosine.
            return torch.cat([torch.softmax(a, dim=-1), torch.softmax(b, dim=-1)]).cos()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """680""")

        # Should turn into pointwise even if only some of inputs are pointwise.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates the cosine of the first input with a matrix
            # product of the second, then applies cosine.
            out = torch.cat([a.cos(), torch.mm(b, b)])
            return out.cos()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """600""")

        # Should not turn into pointwise if all inputs are not pointwise
        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates two matrix products, then applies cosine.
            out = torch.cat([torch.mm(a, a), torch.mm(b, b)])
            return out.cos()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """800""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates two inputs, then applies cosine.
            out = torch.cat([a, b])
            return out.cos()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Applies cosine to the second input and concatenates it
            # after the first.
            b = b.cos()
            return torch.cat([a, b])

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the first input with itself, concatenates
            # with the second, and pads the last dimension on both sides with a constant.
            a = a @ a
            return torch.constant_pad_nd(torch.cat([a, b]), [2, 2], 0.5)

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """680""")

    @patch.object(config, "split_cat_fx_passes", False)
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})
    def test_cat_pointwise_many_complex_inputs(self):
        # [FUNCTION DOC] With split-cat passes and batch fusion options turned off, checks the
        # access count of concatenating sixteen gelu results plus a constant.
        def f(*inputs):
            # [FUNCTION DOC] HELPER: Applies gelu to each positional input, concatenates the
            # results, and adds ten.
            input = [torch.nn.functional.gelu(val) for val in inputs]
            return torch.cat(input) + 10

        inp = (T(10, 10) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """6400""")

    @patch.object(config, "split_cat_fx_passes", False)
    @patch.object(
        config,
        "pre_grad_fusion_options",
        {
            "batch_linear": {},
            "batch_linear_lhs": {},
            "batch_layernorm": {},
            "batch_tanh": {},
            "batch_relu": {},
            "batch_sigmoid": {},
        },
    )
    @patch.object(config, "post_grad_fusion_options", {})
    def test_cat_pointwise_many_simple_inputs(self):
        # [FUNCTION DOC] With split-cat passes and batch fusion options turned off, checks the
        # access count of concatenating sixteen relu results plus a constant.
        def f(*inputs):
            # [FUNCTION DOC] HELPER: Applies relu to each positional input, concatenates the
            # results, and adds ten.
            input = [torch.nn.functional.relu(val) for val in inputs]
            return torch.cat(input) + 10

        inp = (T(10, 10) for _ in range(16))
        self.assertExpectedInline(count_numel(f, *inp), """9600""")

    @patch.object(config, "max_pointwise_cat_inputs", 0)
    def test_cat_pointwise_config_option(self):
        # [FUNCTION DOC] With the cap on pointwise cat inputs set to zero, checks the access
        # count for a cat of two offset inputs followed by an add.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Concatenates two offset inputs and adds three.
            return torch.cat([a + 1, b + 2]) + 3

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """400""")

    def test_index(self):
        # [FUNCTION DOC] Checks the access count of indexing a tensor with an integer index
        # tensor.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Indexes the first input with the second.
            return a[b]

        inp = (T(10), TI(10, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """30""")

    @requires_gpu_and_triton
    def test_delay_realize_cheap_outputs_shared_mask(self):
        # [FUNCTION DOC] Compiles a function adding a shared lower-triangular mask to three
        # inputs twice: once with delayed realization of cheap outputs disabled and once with
        # the default. Asserts the deferred version accesses no more bytes than the eager one.
        # Needs a GPU with triton.
        # Shared tril mask across multiple users gets eagerly materialized
        # as an output buffer, inflating downstream read counts. With
        # delay_realize_cheap_outputs, the mask stays inlined as index
        # arithmetic, reducing memory traffic.
        def f(x1, x2, x3):
            # [FUNCTION DOC] HELPER: Builds a lower-triangular ones mask matching the first
            # input device and adds it to each of the three inputs, returning the three sums.
            mask = torch.tril(torch.ones(32, 32, device=x1.device))
            return x1 + mask, x2 + mask, x3 + mask

        inp = (
            T(32, 32, grad=True),
            T(32, 32, grad=True),
            T(32, 32, grad=True),
        )

        # Without deferred realization: mask gets materialized as output buffer
        with patch.object(config, "delay_realize_cheap_outputs", False):
            metrics.reset()
            torch.compile(f, backend=compile_but_use_eager)(*inp)
            eager_bytes = metrics.num_bytes_accessed

        # Default (deferred realization on): mask stays inlined
        metrics.reset()
        torch.compile(f, backend=compile_but_use_eager)(*inp)
        deferred_bytes = metrics.num_bytes_accessed

        self.assertLessEqual(deferred_bytes, eager_bytes)


class FusionTests(TestCase):
    # [CLASS DOC] Tests that operations fuse into few kernels, checked via the access count.
    # Stateless.
    """
    Tests that things can be fused into a single kernel
    """

    def test_horizontal_reduction_pointwise(self):
        # [FUNCTION DOC] Checks a reduction and a pointwise op over the same input fuse
        # horizontally.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns the row sums of the input and its cosine.
            b = a.sum(dim=1)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_horizontal_reduction_reduction(self):
        # [FUNCTION DOC] Checks two reductions over the same input fuse horizontally.
        def f(a):
            # [FUNCTION DOC] HELPER: Returns the row sums and row maxima of the input.
            b = a.sum(dim=1)
            c = a.amax(dim=1)
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    @requires_gpu_and_triton
    @config.patch({"force_disable_caches": True, "triton.multi_kernel": 0})
    def test_aot_autograd_cse_preserves_reduction_fusion(self):
        # [FUNCTION DOC] Compiles a function with three reductions over the same large input,
        # once without and once with autograd, counting generated kernels in each case with
        # caches disabled. Asserts the kernel count with autograd equals the inference count,
        # i.e. common-subexpression elimination keeps the reductions fused. Needs a GPU with
        # triton.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the maximum of the absolute values, the mean of the
            # absolute values and the mean of the squares.
            return x.abs().max(), x.abs().mean(), x.square().mean()

        def count_kernels(requires_grad):
            # [FUNCTION DOC] HELPER: Resets dynamo and metrics, compiles the function on a large
            # random input with or without requires_grad, checks the result matches eager,
            # synchronizes the device and records the generated kernel count after the forward.
            # With grad it also compares gradients of the summed outputs against eager. Returns
            # the forward kernel count.
            torch._dynamo.reset()
            metrics.reset()
            x = torch.rand((1024, 32768), device=DEVICE, requires_grad=requires_grad)
            result = torch.compile(fn, fullgraph=True, dynamic=False)(x)
            expected = fn(x)
            self.assertEqual(result, expected)
            get_interface_for_device(DEVICE).synchronize()
            forward_kernel_count = metrics.generated_kernel_count
            if requires_grad:
                result_sum = result[0] + result[1] + result[2]
                expected_sum = expected[0] + expected[1] + expected[2]
                (result_grad,) = torch.autograd.grad(result_sum, (x,))
                (expected_grad,) = torch.autograd.grad(expected_sum, (x,))
                self.assertEqual(result_grad, expected_grad)
            return forward_kernel_count

        inference_kernel_count = count_kernels(requires_grad=False)
        autograd_kernel_count = count_kernels(requires_grad=True)

        self.assertEqual(autograd_kernel_count, inference_kernel_count)

    def test_horizontal_reduction_pointwise2(self):
        # [FUNCTION DOC] Checks a row-sum plus a pointwise op on a different input fuse
        # together.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Adds the row sums of the first input to the cosine of the
            # second.
            c = a.sum(dim=1)
            b = b.cos()
            return b + c

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_reduction_outer_pointwise(self):
        # [FUNCTION DOC] Checks a column-sum (outer) reduction plus a pointwise op on a
        # different input.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Adds the column sums of the first input to the cosine of
            # the second.
            c = a.sum(dim=0)
            b = b.cos()
            return b + c

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """120""")

    def test_horizontal_sum_pw_broadcast(self):
        # [FUNCTION DOC] Checks a keep-dim row sum multiplied with a broadcast pointwise result.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiplies the keep-dim row sums of the first input by the
            # cosine of the second.
            a = a.sum(dim=1, keepdim=True)
            b = b.cos()
            return a * b

        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    def test_vertical_sum_pw(self):
        # [FUNCTION DOC] Checks a pointwise, reduction, pointwise chain.
        def f(a):
            # [FUNCTION DOC] HELPER: Applies cosine, sums over dimension one, and applies cosine
            # again.
            a = a.cos()
            a = a.sum(dim=1)
            return a.cos()

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """110""")

    def test_norm_chain(self):
        # [FUNCTION DOC] Checks a chain of three normalize-by-row-sum steps.
        def f(a):
            # [FUNCTION DOC] HELPER: Three times: multiplies the tensor by its keep-dim row
            # sums.
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            return a

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_inner(self):
        # [FUNCTION DOC] Checks softmax over the inner dimension.
        def f(a):
            # [FUNCTION DOC] HELPER: Softmax over dimension one.
            return torch.softmax(a, dim=1)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_layer_norm(self):
        # [FUNCTION DOC] Under no-grad, checks the access count of a layer norm module (the TODO
        # notes saved statistics are counted).
        # TODO: Suboptimal! We shouldn't need to save normalization stats.
        mod = torch.nn.LayerNorm(10, device=self.device)

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the layer norm module to the input.
            return mod(x)

        inp = (T(10, 10),)
        with torch.no_grad():
            self.assertExpectedInline(count_numel(f, *inp), """220""")

    def test_double_softmax(self):
        # [FUNCTION DOC] Checks two consecutive softmaxes.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies softmax over dimension one twice.
            x = torch.softmax(x, dim=1)
            x = torch.softmax(x, dim=1)
            return x

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    def test_softmax_backward(self):
        # [FUNCTION DOC] Checks the softmax backward data operator.
        def f(grad_out, out):
            # [FUNCTION DOC] HELPER: Calls the softmax backward data operator on gradient and
            # output with float32.
            return aten._softmax_backward_data(grad_out, out, 1, torch.float32)

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_neighbor(self):
        # [FUNCTION DOC] Checks a pairwise squared-distance style computation with a sum and a
        # maximum.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Squares the difference of broadcast inputs, sums the last
            # dimension, then takes the maximum over dimension one.
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 4), T(1, 10, 4))
        self.assertExpectedInline(count_numel(f, *inp), """90""")

    def test_factory_reduction(self):
        # [FUNCTION DOC] Checks a reduction over a function that creates its own tensors.
        def f():
            # [FUNCTION DOC] HELPER: Builds two tensors of ones with different ranks, adds them
            # (broadcast) and sums the last dimension.
            a = torch.ones(10, device=self.device)
            b = torch.ones(10, 10, device=self.device)
            return (a + b).sum(dim=-1)

        inp = ()
        self.assertExpectedInline(count_numel(f, *inp), """10""")

    def test_index_pointwise(self):
        # [FUNCTION DOC] Checks indexing followed by a pointwise op.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Indexes the first input with the second and applies cosine.
            return a[b].cos()

        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """320""")

    def test_index_reduction(self):
        # [FUNCTION DOC] Checks indexing followed by pointwise and reduction.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Indexes the first input with the second, applies cosine and
            # sums over dimension one.
            return a[b].cos().sum(dim=1)

        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), """140""")

    def test_mutation_fusion(self):
        # [FUNCTION DOC] Checks fusion of in-place copies back into two inputs.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Computes two sums from the inputs and copies them back into
            # the second and first inputs in place; returns nothing.
            a0 = a.add(c)
            b0 = b.add(a0)
            b.copy_(b0)
            a.copy_(a0)

        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """500""")

    @skipIfXpu(msg="copy_(cat()) fusion not supported on XPU")
    @unittest.skipIf(TEST_WITH_ROCM, "copy_(cat()) fusion not supported on ROCm")
    # TODO(ivankobzarev): enable copy_(cat()) fusion for CUDA 13+
    @unittest.skipIf(
        torch.version.cuda
        and tuple(int(x) for x in torch.version.cuda.split(".")) >= (13, 0),
        "copy_(cat()) fusion not supported on CUDA 13+",
    )
    def test_copy_cat_fusion(self):
        # [FUNCTION DOC] Checks that copying a cat result into a destination fuses (no
        # intermediate allocation). The expected count is larger than the real traffic because
        # of an estimate artifact explained in the test comments. Skipped on XPU, ROCm and CUDA
        # 13+.
        """copy_(cat(...)) should fuse: no intermediate allocation for cat."""

        def f(dst, a, b):
            # [FUNCTION DOC] HELPER: Copies the concatenation of two tensors into the
            # destination in place.
            dst.copy_(torch.cat([a, b]))

        dst = T(20)
        inp = (dst, T(10), T(10))
        # The cat is fully fused: the generated kernel reads a and b and writes
        # them straight into dst's slices, with no intermediate allocation --
        # real traffic is 10 (read a) + 10 (read b) + 20 (write dst) = 40.
        # count_numel reports 60 here rather than 40 because its static
        # num_bytes_accessed estimate over-counts the ConcatKernel slices once
        # canonicalization reorders the placeholders (dst is no longer input 0);
        # the emitted kernel is unchanged, so this is an estimate artifact, not
        # a real regression. Without fusion cat would allocate intermediate: 80.
        self.assertExpectedInline(count_numel(f, *inp), """60""")

    def test_reduction_pointwise_multi_level_reduction(self):
        # [FUNCTION DOC] Compiles a layer norm followed by an absolute-value maximum and a
        # scaled sigmoid, for both keep-dim settings, and asserts the access count is within a
        # small relative tolerance of the expected multi-level reduction estimate (a fixed
        # constant when cooperative reductions are enabled).
        hidden_size = 4096
        layer_norm = torch.nn.LayerNorm(hidden_size).to(GPU_TYPE).float()

        @torch.inference_mode()
        def f(x, scale, amax_keep_dim):
            # [FUNCTION DOC] HELPER: Under inference mode: layer-normalizes the float-cast
            # input, takes the maximum of its absolute value (optionally keeping dims),
            # multiplies by the scale, applies sigmoid and returns the sigmoid and the maximum.
            x = layer_norm(x.to(dtype=torch.float))
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)

        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))

        # 2 kernels:
        # kernel 1: (input = X, scale, LN scale, LN bias, output = LN_pointwise(X), first-level amax (split-reduction))
        # kernel 2: (input = first-level amax, output = final amax)
        # scale (1) + X (4*2048*hidden_size) * 2 + LN scale (hidden_size) + LN bias (hidden_size) + amax (4 * 2048 * 2 + 1)
        expected_numel = (
            1 + hidden_size * 2 + 4 * 2048 * hidden_size * 2 + 4 * 2048 * 2 + 1
        )
        if (
            config.triton.cooperative_reductions
            or config.triton.force_cooperative_reductions
        ):
            expected_numel = 134225922

        # Allow ~0.05% tolerance: canonicalization may reorder nodes, causing
        # slightly different fusion decisions and a small numel change.
        actual_keep = int(count_numel(f, *inp, True))
        actual_no_keep = int(count_numel(f, *inp, False))
        self.assertAlmostEqual(actual_keep, expected_numel, delta=expected_numel * 5e-4)
        self.assertAlmostEqual(
            actual_no_keep, expected_numel, delta=expected_numel * 5e-4
        )

    def test_pointwise_multi_level_reduction(self):
        # [FUNCTION DOC] Compiles a pointwise scale, an absolute-value maximum and a scaled
        # sigmoid once, then counts accesses for both keep-dim settings. Asserts the counts are
        # equal and that the count is at least approximately the three-kernel estimate.
        # TODO: this can be optimized by having the first pointwise kernel leveraging block sizes
        # of the first-level reduction kernel.
        hidden_size = 4096

        def f(x, scale, amax_keep_dim):
            # [FUNCTION DOC] HELPER: Scales the input, takes the maximum of its absolute value
            # (optionally keeping dims), multiplies by the scale input, applies sigmoid and
            # returns the sigmoid and the maximum.
            x = x * 1.1
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)

        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))

        compiled_f = torch.compile(f)
        compiled_f(*inp, True)

        # 3 kernels:
        # kernel 1: (input = X, scale, output = pointwise(X))
        # kernel 2: (input = X, output = first-level amax)
        # kernel 3: (input = first-level amax, output = final amax)
        # scale (1) + X (4*2048*hidden_size) * 3 + amax (num_splits * 2 + 1)
        # num_splits depends on SM architectures.
        expected_numel = 1 + 4 * 2048 * hidden_size * 3 + 1
        actual_numel_amax_keep_dim = count_numel(f, *inp, True)
        actual_numel_amax_no_keep_dim = count_numel(f, *inp, False)
        self.assertEqual(actual_numel_amax_keep_dim, actual_numel_amax_no_keep_dim)
        self.assertGreaterAlmostEqual(actual_numel_amax_keep_dim, str(expected_numel))

    def test_create_block_mask(self):
        # [FUNCTION DOC] Builds a three-dimensional neighborhood-attention mask function,
        # compiles block mask creation with it, and asserts the access count is far below a
        # quadratic amount, showing the mask is fused.
        def mk_3d_flex_natten_mask(dims, kernel_size):
            # [FUNCTION DOC] HELPER: Given data dimensions and kernel sizes, returns a mask
            # function implementing neighborhood attention over flattened positions.
            T, H, W = dims
            K_T, K_H, K_W = kernel_size
            spatial = H * W

            def get_x_y_t(idx: int) -> tuple[int, int, int]:
                # [FUNCTION DOC] HELPER: Converts a flat index into spatial x, y and time
                # coordinates using the closure dimensions; returns the three as a tuple.
                t = idx // spatial
                s = idx % spatial
                x = s // W
                y = s % W
                return x, y, t

            def get_mask(b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Mask function taking batch, head, query and key-value
                # indices; clamps the query coordinates so the window fits in bounds and returns
                # true where the key-value coordinates lie within half the kernel size of the
                # window center in all three axes.
                q_x, q_y, q_t = get_x_y_t(q_idx)
                kv_x, kv_y, kv_t = get_x_y_t(kv_idx)
                kernel_x = q_x.clamp(K_W // 2, (W - 1) - K_W // 2)
                kernel_y = q_y.clamp(K_H // 2, (H - 1) - K_H // 2)
                kernel_t = q_t.clamp(K_T // 2, (T - 1) - K_T // 2)
                hori_mask = (kernel_x - kv_x).abs() <= K_W // 2
                vert_mask = (kernel_y - kv_y).abs() <= K_H // 2
                temp_mask = (kernel_t - kv_t).abs() <= K_T // 2
                return hori_mask & vert_mask & temp_mask

            return get_mask

        T = 4
        H = 16
        W = 16
        t = 5
        h = 5
        w = 5
        data_size = (T, H, W)
        kernel_size = (t, h, w)
        S = T * H * W
        from torch.nn.attention.flex_attention import create_block_mask

        mask_mod = mk_3d_flex_natten_mask(data_size, kernel_size)

        torch.compile(create_block_mask)(mask_mod, None, None, S, S)
        numel = int(count_numel(create_block_mask, mask_mod, None, None, S, S))

        # We should be writing way less than a quadratic amount of bytes here
        # With fusion, we should only be writing a linear number of bytes
        self.assertLess(numel * 5, S * S)


class SchedulerFusionTests(TestCase):
    # [CLASS DOC] Tests the scheduler fusion-group heuristic, with inductor rematerialization
    # disabled (opcount threshold zero) for the whole class. State: a class-level exit stack
    # holding the config patch.
    """
    Testing the fusion group creation heuristic (i.e. cases where we can't fuse
    everything into a single kernel)
    Disables inductor rematerialization for easier reasoning of tests.
    """

    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the parent setup, creates an exit stack stored on
        # the class and enters a patch setting the realize opcount threshold to zero.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, "realize_opcount_threshold", 0))

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] Class teardown: closes the exit stack (undoing the config patch), then
        # calls the parent teardown.
        cls._stack.close()
        super().tearDownClass()

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice1(self):
        # [FUNCTION DOC] With the pattern matcher off, checks a case where it does not matter
        # where the fusion group is broken around a matmul.
        # Doesn't matter where we break fusion group here
        def f(a):
            # [FUNCTION DOC] HELPER: Applies cosine, takes a matrix product of the result with
            # itself, adds the cosine of the cosine.
            c = a.cos()
            d = torch.mm(c, c)
            e = c.cos()
            return d + e

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """700""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice2(self):
        # [FUNCTION DOC] With the pattern matcher off, checks that the smaller intermediate (row
        # sums) is materialized around a matmul.
        # We should materialize e (it's smaller!)
        # [c, e]: 210, [f]: 210, [d]: 200
        def f(a):
            # [FUNCTION DOC] HELPER: Computes cosine, a matrix product, row sums, and returns
            # the sum of the matrix product and the row sums.
            c = a.cos()
            d = torch.mm(c, c)
            e = c.sum(dim=1)
            f = d + e
            return f

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """620""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice3(self):
        # [FUNCTION DOC] With the pattern matcher off, checks that a shared intermediate that is
        # also an output is materialized around a matmul.
        # We should materialize e.
        # [c, e]: 300, [f]: 300, [d]: 200
        def f(a):
            # [FUNCTION DOC] HELPER: Computes cosine, a matrix product, adds the cosine to the
            # input, and returns the product plus that sum along with the sum.
            c = a.cos()
            d = torch.mm(c, c)
            e = c + a
            f = d + e
            return f, e

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """800""")

    @patch.object(config, "pattern_matcher", False)
    def test_fusion_choice4_cpu(self):
        # [FUNCTION DOC] With the pattern matcher off, on CPU tensors, checks fusion of nodes
        # with equal element counts and compatible variable ranges: first a multiply and add,
        # then two multiplies combined by an add.
        # Fuse nodes with same number of elements and compatible original var ranges
        # [buf0: {d0: 60, d1: 11}, buf1: {d0: 660}] -> buf0_buf1
        def f(x, w):
            # [FUNCTION DOC] HELPER: Multiplies the input by a weight and adds one.
            o1 = x * w
            output = o1 + 1.0
            return output

        inp = (T(2, 3, 10, 11, device="cpu"), T(11, device="cpu"))
        self.assertExpectedInline(count_numel(f, *inp), """1331""")

        # [buf0_buf1: {d0: 60, d1: 11}, buf2: {d0: 660}] -> buf0_buf1_buf2
        def f(x, w1, w2):
            # [FUNCTION DOC] HELPER: Multiplies the input by two weights and adds the products.
            o1 = x * w1
            o2 = x * w2
            output = o1 + o2
            return output

        inp = (T(2, 3, 10, 11, device="cpu"), T(11, device="cpu"), T(11, device="cpu"))
        self.assertExpectedInline(count_numel(f, *inp), """1342""")


class TilingTests(TestCase):
    # [CLASS DOC] Tests that tiling picks efficient loop orders for transposed and permuted
    # inputs. Stateless.
    def test_tiling_simple(self):
        # [FUNCTION DOC] Checks access counts for adding a tensor to a transposed one, in both
        # operand orders.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Adds the first input to the transpose of the second.
            return a + b.t()

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Adds the transpose of the first input to the second.
            return a.t() + b

        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """300""")

    def test_tiling_three(self):
        # [FUNCTION DOC] Checks the access count for a sum of three 3-D tensors with different
        # permutations.
        def f(a, b, c):
            # [FUNCTION DOC] HELPER: Adds the first input to permuted views of the second and
            # third.
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)

        inp = (T(10, 10, 10), T(10, 10, 10), T(10, 10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """4000""")


class MinCutPartitioningTests(TestCase):
    # [CLASS DOC] Tests min-cut partitioning (what is saved for backward versus recomputed) via
    # training access counts. Stateless.
    def test_partitioning_full_remat(self):
        # [FUNCTION DOC] Checks full recomputation for a chain of cosines.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies cosine three times.
            return x.cos().cos().cos()

        inp = (T(10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """50""")

    def test_partitioning_partial_remat(self):
        # [FUNCTION DOC] Checks partial recomputation for a sum of four inputs followed by two
        # cosines.
        def f(a, b, c, d):
            # [FUNCTION DOC] HELPER: Sums four inputs and applies cosine twice.
            x = a + b + c + d
            return x.cos().cos()

        inp = (T(10, grad=True), T(10, grad=True), T(10, grad=True), T(10, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), """90""")

    def test_partitioning_dtype(self):
        # [FUNCTION DOC] Checks partitioning when a comparison result is converted to a
        # different dtype and multiplied.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by the boolean mask of negative
            # entries.
            return (x < 0) * x

        inp = (T(100, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """450""")

    @config.patch("fx_graph_cache", False)
    @config.patch("autotune_local_cache", False)
    @patch.object(functorch.compile.config, "max_dist_from_bw", 1000)
    def test_partitioning_unremat_bw(self):
        # [FUNCTION DOC] With graph caches off and a large max distance from backward, checks
        # that nodes are not recomputed in backward unnecessarily around a matmul.
        def f(x):
            # [FUNCTION DOC] HELPER: Matrix-multiplies the input with a tensor of ones of the
            # same shape, then applies tanh twice.
            return torch.mm(x, x.new_ones(x.shape)).tanh().tanh()

        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """1300""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_unremat_bw2(self):
        # [FUNCTION DOC] With the pattern matcher off, checks partitioning around two matmuls.
        def f(a):
            # [FUNCTION DOC] HELPER: Squares the input as a matrix, adds one, forms a second
            # tensor by adding two, and returns their matrix product.
            a = torch.mm(a, a)
            a = a + 1
            b = a + 2
            c = torch.mm(a, b)
            return c

        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """2600""")

    def test_partitioning_keops(self):
        # [FUNCTION DOC] Checks partitioning for a keops-style pairwise product reduction.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Multiplies a column by a row, applies cosine and sums over
            # dimension one.
            return (a * b).cos().sum(dim=1)

        inp = (T(20, 1, grad=True), T(1, 20, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), """220""")

    def test_partitioning_cat(self):
        # [FUNCTION DOC] Checks partitioning through a cat.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Applies tanh to the first input and concatenates it with
            # the second.
            a = torch.tanh(a)
            return torch.cat([a, b])

        inp = (T(10, grad=True), T(10, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), """70""")

    def test_partitioning_relu(self):
        # [FUNCTION DOC] Checks partitioning for relu.
        def f(x):
            # [FUNCTION DOC] HELPER: Applies relu to the input.
            return torch.relu(x)

        inp = (T(16, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """72""")

    def test_partitioning_with_view(self):
        # [FUNCTION DOC] Defines a custom autograd function whose forward saves a viewed cosine
        # for backward and checks that the compiled graph does not recompute that chain in
        # backward.
        class Foo(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd function used by the partitioning test;
            # forward computes sine and a viewed cosine, saves both and returns the cosine of
            # the view; backward uses saved values.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes the sine and the cosine of the input, views
                # the cosine as ten by ten, saves the view and the sine, and returns the cosine
                # of the view. Takes the context and input.
                y = x.sin()
                x = x.cos()
                x = x.view(10, 10)
                ctx.save_for_backward(x, y)
                x = x.cos()
                return x

            @staticmethod
            def backward(ctx, gradOut):
                # [FUNCTION DOC] HELPER: Backward: loads saved tensors, matrix-multiplies the
                # incoming gradient with the saved view, flattens to one dimension and
                # multiplies by the saved sine. Returns the input gradient.
                x, y = ctx.saved_tensors
                return torch.mm(gradOut, x).view(100) * y

        def f(a):
            # [FUNCTION DOC] HELPER: Applies the custom function to the input.
            return Foo.apply(a)

        inp = (T(100, grad=True),)
        # We do not want to recompute the x.cos().view() chain, as it's
        # materialized in backwards
        self.assertExpectedInline(count_numel_train(f, *inp), """900""")

    @patch.object(config, "pattern_matcher", False)
    def test_partitioning_long_chain_add(self):
        # [FUNCTION DOC] With the pattern matcher off, checks partitioning for a loop of
        # multiply, matrix product, scale and add run twice.
        def f(x):
            # [FUNCTION DOC] HELPER: Twice: squares the tensor elementwise, matrix-multiplies
            # with itself, doubles and adds the saved original; returns the result.
            orig = x
            for _ in range(2):
                x = x * x
                x = torch.mm(x, x)
                x = x * 2
                x = orig + x
                orig = x
            return x

        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), """3900""")


def unfusible(x):
    # [FUNCTION DOC] HELPER: Calls an ATen operator with neither decomposition nor lowering,
    # forcing inductor to fall back to eager so no-op tests have a barrier. Returns the lazily
    # cloned tensor.
    # For the purpose of noop tests, we want inductor to fall back to
    # eager mode, so, below we must use a aten operator that does not
    # have decomposition nor lowering:
    return aten._lazy_clone(x)


class NoopTests(TestCase):
    # [CLASS DOC] Tests that no-op operations (clones, conversions, trivial cat) do not add
    # memory traffic. Stateless.
    def test_noop_clones(self):
        # [FUNCTION DOC] Checks clone followed by an unfusible op, and a clone that is both
        # consumed and returned.
        def f(a):
            # [FUNCTION DOC] HELPER: Clones the input and passes it through the unfusible op.
            b = a.clone()
            b = unfusible(b)
            return b

        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")

        def f(a):
            # [FUNCTION DOC] HELPER: Clones the input and returns the clone and the unfusible
            # result of the clone.
            b = a.clone()
            c = unfusible(b)
            return b, c

        self.assertExpectedInline(count_numel(f, inp), """40""")

    def test_noop_slice_scatter(self):
        # [FUNCTION DOC] Checks slice_scatter of a tensor into itself is a no-op.
        def f(a):
            # [FUNCTION DOC] HELPER: Slice-scatters the input into itself and passes the result
            # through the unfusible op.
            b = aten.slice_scatter(a, a)
            c = unfusible(b)
            return c

        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_dtype_conversion(self):
        # [FUNCTION DOC] Checks a conversion to the same dtype is a no-op.
        def f(a):
            # [FUNCTION DOC] HELPER: Converts the input to float32 and passes it through the
            # unfusible op.
            b = torch.ops.prims.convert_element_type(a, torch.float32)
            c = unfusible(b)
            return c

        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_device_conversion(self):
        # [FUNCTION DOC] Checks a transfer to the same device is a no-op.
        def f(a):
            # [FUNCTION DOC] HELPER: Moves the input to the test device and passes it through
            # the unfusible op.
            b = torch.ops.prims.device_put(a, DEVICE)
            c = unfusible(b)
            return c

        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), """20""")

    def test_noop_int_ops(self):
        # [FUNCTION DOC] Checks ceil, floor, round and power one on an integer tensor are no-
        # ops, each followed by an unfusible op.
        def f1(a):
            # [FUNCTION DOC] HELPER: Applies ceil, then the unfusible op.
            b = torch.ceil(a)
            c = unfusible(b)
            return c

        def f2(a):
            # [FUNCTION DOC] HELPER: Applies floor, then the unfusible op.
            d = torch.floor(a)
            e = unfusible(d)
            return e

        def f3(a):
            # [FUNCTION DOC] HELPER: Applies round, then the unfusible op.
            f = torch.round(a)
            g = unfusible(f)
            return g

        def f4(a):
            # [FUNCTION DOC] HELPER: Raises to power one, then the unfusible op.
            f = torch.pow(a, 1)
            g = unfusible(f)
            return g

        inp = TI(10)
        self.assertExpectedInline(count_numel(f1, inp), """20""")
        self.assertExpectedInline(count_numel(f2, inp), """20""")
        self.assertExpectedInline(count_numel(f3, inp), """20""")
        self.assertExpectedInline(count_numel(f4, inp), """20""")

    def test_noop_cat(self):
        # [FUNCTION DOC] Checks that concatenating a single tensor is a no-op, with and without
        # an unfusible consumer.
        def f1(a):
            # [FUNCTION DOC] HELPER: Concatenates a one-element list and passes the result
            # through the unfusible op.
            b = torch.cat([a])
            return unfusible(b)

        inp = T(10)
        self.assertExpectedInline(count_numel(f1, inp), """20""")

        def f2(a):
            # [FUNCTION DOC] HELPER: Concatenates a one-element list twice and returns the
            # result.
            b = torch.cat([a])
            c = torch.cat([b])
            return c

        self.assertExpectedInline(count_numel(f2, inp), """20""")


class InplacingTests(TestCase):
    # [CLASS DOC] Tests in-place and re-inplacing behavior (scatter, custom ops, triton kernels
    # with mutated arguments) via access counts. Stateless.
    def test_inplace_scatter(self):
        # [FUNCTION DOC] Checks the access count of index assignment into a cosine result and of
        # functional index_put and unsafe index_put whose result is copied back into the input.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Applies cosine, assigns one at the indexed positions and
            # returns the tensor.
            a = a.cos()
            a[b] = 1
            return a

        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """26""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Calls functional index_put with a scalar tensor value, then
            # copies the result into the input in place and returns it.
            out = aten.index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """6""")

        def f(a, b):
            # [FUNCTION DOC] HELPER: Calls unsafe index_put with a scalar tensor value, then
            # copies the result into the input in place and returns it.
            out = aten._unsafe_index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)

        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """6""")

    def test_inplace_scatter_noop_view(self):
        # [FUNCTION DOC] Checks the access count of assigning one into selected columns of a
        # matrix.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Assigns one into the indexed columns of the input in place
            # and returns it.
            a[:, b] = 1
            return a

        inp = (T(10, 10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), """42""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_training(self):
        # [FUNCTION DOC] Wraps a triton sine kernel writing into a preallocated output in a
        # custom autograd function and checks the training access count. Needs a GPU with
        # triton.
        # [FUNCTION DOC] HELPER: Triton kernel computing the sine of each element of the input
        # and storing it, with a bounds mask, into the output pointer; one block per program id.
        @triton.jit
        def sin_kernel(
            in_ptr0,
            out_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)

        def sin_triton(x, out):
            # [FUNCTION DOC] HELPER: Launches the sine kernel over a grid of one program per
            # element with a block size of four.
            n_elements = x.numel()
            sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

        factory_op = torch.empty_like

        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function whose forward allocates an output, runs the
            # triton sine kernel and saves the output; backward launches the same kernel on the
            # saved tensor.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Allocates an output via the factory op, runs the
                # kernel, saves the output for backward and returns it.
                out = factory_op(x)
                sin_triton(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Allocates a new tensor like the gradient, runs the
                # kernel on the saved tensor into it and returns it.
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin_triton(saved, out)
                return out

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function.
            return MySin.apply(x)

        x = T(3, grad=True)
        self.assertExpectedInline(count_numel_train(f, x), """18""")

    @requires_gpu_and_triton
    def test_triton_kernel_not_fusable_with_users(self):
        # [FUNCTION DOC] Registers a user triton kernel with two outputs as a custom op and uses
        # it in an autograd function. Asserts the cheap pointwise consumer is not computed in
        # forward and saved (the count expects an extra kernel is avoided). Needs a GPU with
        # triton.
        # [FUNCTION DOC] HELPER: Triton kernel computing the sine of each element and storing it
        # into two output pointers with a bounds mask.
        @triton.jit
        def _sin_kernel(
            in_ptr0,
            out_ptr,
            out2_ptr,
            n_elements,
            BLOCK_SIZE: "tl.constexpr",
        ):
            pid = tl.program_id(axis=0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(in_ptr0 + offsets, mask=mask)
            output = tl.sin(x)
            tl.store(out_ptr + offsets, output, mask=mask)
            tl.store(out2_ptr + offsets, output, mask=mask)

        from torch._library import capture_triton, triton_op

        @triton_op("mylib::sin_kernel", mutates_args={})
        def sin_kernel(x: torch.Tensor) -> list[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Custom op body: allocates two outputs like the input,
            # launches the captured triton kernel over a grid of one program per element, and
            # returns the two outputs as a list.
            n_elements = x.numel()
            out = torch.empty_like(x)
            out2 = torch.empty_like(x)
            capture_triton(_sin_kernel)[(n_elements,)](
                x, out, out2, n_elements, BLOCK_SIZE=4
            )
            return [out, out2]

        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function whose forward calls the custom op and saves
            # the input and the second output; backward multiplies the gradient by the sigmoid
            # of the saved output and the input.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Calls the custom op, saves the input and the second
                # output, returns the first output.
                out, saved = tuple(torch.ops.mylib.sin_kernel(x))
                ctx.save_for_backward(x, saved)
                return out

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Returns the gradient times the sigmoid of the saved
                # second output times the saved input.
                (x, saved) = ctx.saved_tensors
                return grad * saved.sigmoid() * x

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function.
            return MySin.apply(x)

        x = T(3, grad=True)
        # Important bit: saved.sigmoid() can be fused into its consumer (mul),
        # but not its producer (user triton kernel).
        # So we should not compute it in the fw and save it for backward
        # (it will cost an extra kernel)
        self.assertExpectedInline(count_numel_train(f, x), """27""")

    @requires_gpu_and_triton
    def test_inplace_custom_op_training_two_mutated_inputs(self):
        # [FUNCTION DOC] Registers a custom op that mutates two output arguments with sine and
        # cosine of the input and checks the access count of a function using it. Needs a GPU
        # with triton.
        @torch.library.custom_op(
            "_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"}
        )
        def sin_cos(
            x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor
        ) -> None:
            # [FUNCTION DOC] HELPER: Custom op body: copies the sine of the input into the first
            # output and the cosine into the second; returns nothing.
            out_sin.copy_(x.sin())
            out_cos.copy_(x.cos())

        def f(x):
            # [FUNCTION DOC] HELPER: Allocates two outputs, calls the custom op to fill them,
            # and returns a clone of the input and the two outputs.
            out0 = torch.empty_like(x)
            out1 = torch.empty_like(x)
            sin_cos(x, out0, out1)
            return x.clone(), out0, out1

        x = T(3, grad=True)
        self.assertExpectedInline(count_numel(f, x), """21""")

    @requires_gpu_and_triton
    def test_inplace_custom_op_training(self):
        # [FUNCTION DOC] Registers a custom op that mutates its result argument and uses it in
        # an autograd function; checks the training access count. Needs a GPU with triton.
        @torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
        def sin(x: torch.Tensor, result: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op body: copies the sine of the input into the
            # result argument; returns nothing.
            result.copy_(x.sin())

        factory_op = torch.empty_like

        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function whose forward and backward each allocate an
            # output and fill it with the mutating sine custom op.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Allocates an output, fills it using the custom op,
                # saves it and returns it.
                out = factory_op(x)
                sin(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Allocates a tensor like the gradient, fills it using
                # the custom op applied to the saved tensor, and returns it.
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin(saved, out)
                return out

        def f(x):
            # [FUNCTION DOC] HELPER: Applies the custom function.
            return MySin.apply(x)

        x = T(3, grad=True)
        self.assertExpectedInline(count_numel_train(f, x), """18""")

    @requires_gpu_and_triton
    def test_inplace_custom_op(self):
        # [FUNCTION DOC] Defines a mutating op in a scoped library fragment and calls it three
        # times on the same output. Asserts the compiled result equals triple sine, that the
        # generated code allocates no new buffers, and the access count. Needs a GPU with
        # triton.
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Op implementation: copies the sine of the first
                # argument into the second in place.
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                # [FUNCTION DOC] HELPER: Calls the op three times, chaining the output, and
                # returns the output tensor.
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3)
            out = T(3)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 0)

            self.assertExpectedInline(count_numel(f, x, out), """21""")

    @requires_gpu_and_triton
    def test_inplace_custom_op_intermediate(self):
        # [FUNCTION DOC] Like the previous test but the function replaces the output argument
        # with a fresh empty tensor first. Asserts the compiled result equals triple sine,
        # exactly one buffer allocation in the generated code, and the access count. Needs a GPU
        # with triton.
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor x, Tensor(a!) out) -> ()")

            def foo(x: torch.Tensor, out: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Op implementation: copies the sine of the first
                # argument into the second in place.
                out.copy_(x.sin())

            m.impl("foo", foo, "CompositeExplicitAutograd")

            def f(x, out):
                # [FUNCTION DOC] HELPER: Allocates a fresh output, calls the op three times
                # chaining it, and returns it.
                out = torch.empty_like(x)
                torch.ops.mylib.foo(x, out)
                torch.ops.mylib.foo(out, out)
                torch.ops.mylib.foo(out, out)
                return out

            x = T(3)
            out = T(3)

            compiled_out, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True), x, out
            )
            self.assertEqual(compiled_out, x.sin().sin().sin())

            # Check that we are allocating the minimum number of intermediate buffers
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 1)

            self.assertExpectedInline(count_numel(f, x, out), """21""")

    @requires_gpu_and_triton
    def test_inplace_custom_op_two_mutated_inputs(self):
        # [FUNCTION DOC] Defines an op that mutates two cache arguments and is called twice.
        # Asserts the generated code has no empty allocations, mentions an in-out buffer, and
        # checks the access count. Needs a GPU with triton.
        with torch.library._scoped_library("mylib", "FRAGMENT") as m:
            m.define("foo(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache) -> Tensor")

            def foo(q, k_cache, v_cache):
                # [FUNCTION DOC] HELPER: Op implementation: adds one in place to both caches and
                # returns the query plus one.
                k_cache.add_(1)
                v_cache.add_(1)
                return q + 1

            m.impl("foo", foo, "CompositeExplicitAutograd")

            q = T(3)
            k_cache = T(3)
            v_cache = torch.rand_like(k_cache)

            def f():
                # [FUNCTION DOC] HELPER: Calls the op twice with captured tensors and
                # accumulates the results.
                x = 0
                for _ in range(2):
                    x = x + torch.ops.mylib.foo(q, k_cache, v_cache)
                return x

            _, (code,) = run_and_get_code(
                torch.compile(f, fullgraph=True),
            )

            # Check that we are not allocate intermediate buffers
            # which can be reused.
            matches = re.findall(r"empty_strided_\w+\(", code)
            self.assertEqual(len(matches), 0)
            self.assertEqual("in_out" in code, True)

            self.assertExpectedInline(count_numel(f), """45""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v1(self):
        # [FUNCTION DOC] Calls an add triton kernel into a zero-initialized output and checks
        # the access count (version one: no extra use).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Creates a zero output like the input, launches the add
            # kernel on both inputs writing to the output, and returns it.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v2(self):
        # [FUNCTION DOC] Triton kernel launch plus an extra independent add; checks the access
        # count (version two).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Like the previous launcher, but also computes the input
            # plus one separately and returns both the output and that tensor.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            tmp = torch.add(x, 1)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output, tmp

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """70""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v3(self):
        # [FUNCTION DOC] Triton kernel launch followed by an in-place add to the first input;
        # checks the access count (version three).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Launches the add kernel into a zero output, then adds one
            # in place to the first input, returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x.add_(1)
            return output

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """80""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v4(self):
        # [FUNCTION DOC] Triton kernel launch plus a multiply on a flattened view of the first
        # input; checks the access count (version four).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Takes a flat view of the first input, launches the add
            # kernel into a zero output, then returns the output and the view times two.
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            output2 = x_view.mul(2)
            return output, output2

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """70""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v5(self):
        # [FUNCTION DOC] Triton kernel launch followed by an in-place multiply through a view of
        # the first input; checks the access count (version five).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Takes a flat view of the first input, launches the add
            # kernel into a zero output, then multiplies the view in place by two and returns
            # the output.
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x_view.mul_(2)
            return output

        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), """80""")

    @requires_gpu_and_triton
    def test_inplace_triton_kernel_v6(self):
        # [FUNCTION DOC] Triton kernel launch where both inputs alias the same storage (one is a
        # view of the other); checks the access count (version six).
        def f(x: torch.Tensor, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Launches the add kernel into a zero output and returns it.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output

        t = T(10)
        inp = (t, t.view(-1))
        self.assertExpectedInline(count_numel(f, *inp), """50""")

    def test_inplace_randperm_scatter(self):
        # [FUNCTION DOC] Checks the access count of an in-place index add with a random
        # permutation index and a scaled source.
        def scaled_index_add(x, y, scale_y):
            # [FUNCTION DOC] HELPER: Builds a random permutation index truncated to the source
            # length, adds the scaled source into the input along dimension zero in place, and
            # returns the result.
            index = torch.randperm(x.shape[0], device=x.device)[: y.shape[0]]
            out = x.index_add_(dim=0, source=y * scale_y, index=index)
            return out

        inp = (T(10, 10), T(5, 10), T(10))
        self.assertExpectedInline(count_numel(scaled_index_add, *inp), """250""")


# Test cases where we don't do the right thing yet.
class WouldBeNiceIfItWorked:
    # [CLASS DOC] Plain (non-test-case) container of scenarios that inductor does not yet handle
    # optimally; its methods are never collected by the test runner. Stateless.
    def test_horizontal(self):
        # [FUNCTION DOC] Horizontal fusion of an outer-dimension sum with a pointwise op (not
        # currently achieved).
        def f(a):
            # [FUNCTION DOC] HELPER: Returns the column sums of the input and its cosine.
            b = a.sum(dim=0)
            c = a.cos()
            return b, c

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """210""")

    # TODO: We aren't fusing outer dim softmaxes
    def test_softmax_outer(self):
        # [FUNCTION DOC] Outer-dimension softmax (a TODO says it is not fused).
        def f(a):
            # [FUNCTION DOC] HELPER: Softmax over dimension zero.
            return torch.softmax(a, dim=0)

        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), """200""")

    # TODO: The greedy fusion strategy results in suboptimal grouping
    @patch.object(config, "realize_opcount_threshold", 0)
    def test_fusion_choice4(self):
        # [FUNCTION DOC] Fusion group choice with greedy fusion giving a suboptimal grouping
        # (see the TODO), with the opcount threshold set to zero.
        def f(a, b, b2):
            # [FUNCTION DOC] HELPER: Adds two inputs, takes a matrix product, forms a second sum
            # with both inputs, and returns a combined result and that sum.
            c = a + b
            d = torch.mm(c, c)
            e = c + b + b2
            f = d + e + b2
            return f, e

        inp = (T(10, 10), T(10, 10, dtype=torch.float16), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), """1000""")

    # TODO: We materialize the intermediate if we don't unroll the reduction
    def test_neighbor(self):
        # [FUNCTION DOC] Squared-distance reduction where the intermediate is materialized
        # unless the reduction is unrolled.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Squares the difference of broadcast inputs, sums the last
            # dimension, takes the maximum over dimension one.
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)

        inp = (T(10, 1, 8), T(1, 10, 8))
        self.assertExpectedInline(count_numel(f, *inp), """170""")


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
