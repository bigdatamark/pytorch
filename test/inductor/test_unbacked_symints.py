# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for graphs containing unbacked symbolic integers, which arise from
# data-dependent operations such as nonzero, item and tolist. Most tests compile a small
# function with fullgraph and compare the compiled result with eager, covering expand, matmul
# autotuning, subclasses, Triton codegen of power and truncation, the optimization hint override
# API, stride-order checks and wrapper input codegen. The test class is instantiated per device
# at import time.
import functools
import unittest
from unittest import mock

import sympy

import torch
from torch._dynamo import config as dynamo_config
from torch._dynamo.exc import InternalTorchDynamoError
from torch._inductor import config as inductor_config, ir
from torch._inductor.codegen.wrapper import PythonWrapperCodegen
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch.testing import make_tensor
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
    skipCUDAIf,
    skipGPUIf,
)
from torch.testing._internal.common_utils import parametrize, skipIfXpu
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.utils._ordered_set import OrderedSet
from torch.utils._sympy.functions import FloorDiv
from torch.utils._sympy.printers import PythonPrinter


class TestUnbackedSymints(InductorTestCase):
    # [CLASS DOC] Device-generic test class for compilation with unbacked symbolic sizes. Each
    # test takes a device argument supplied by device-type instantiation, and many are skipped
    # when no GPU with Triton is available.
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that expands a nonzero
        # result to a fixed width and expands another tensor to the unbacked count, then asserts
        # compiled outputs match eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the nonzero indices expanded to width 128 and the
            # second input expanded along its last dimension to the number of nonzero elements.
            nz = torch.nonzero(x)
            # unbacked symint in nz.size
            x_exp = nz.expand([-1, 128])
            # unbacked symint in target sizes
            y_exp = y.expand([-1, nz.size(0)])
            return x_exp, y_exp

        example_inputs = (
            torch.randn((32), device=device),
            torch.randn((32, 1), device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_expand_ok_with_runtime_assert(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that asserts the nonzero
        # count equals 128 and expands to that size, checking compilation succeeds when a
        # runtime check makes the expand valid. No value comparison.
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes nonzero, adds a check that its length is 128, and
            # returns it expanded to three dimensions.
            nz = x.nonzero()
            torch._check(nz.size(0) == 128)
            return nz.expand([128, -1, 2])

        x = make_tensor(32, 4, device=device, dtype=torch.float32, exclude_zero=True)
        torch.compile(fn, fullgraph=True)(x)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_broadcast_tensors(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function multiplying two tensors
        # whose first dimension is the unbacked nonzero count and compares to eager (broadcast
        # of unbacked sizes).
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a zeros tensor and a ones tensor sized by the
            # nonzero count of the input and returns their product.
            nz = x.nonzero()
            a = torch.zeros([nz.size(0), 512])
            b = torch.ones([nz.size(0), 1])
            return a * b

        x = torch.randn(32, 4, device=device)
        actual = torch.compile(fn, fullgraph=True)(x)
        expected = fn(x)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_autotuning(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function doing a matrix multiply
        # whose left operand has an unbacked row count, with max autotune for matmul enabled via
        # config patch, and compares to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Creates a ones tensor of unbacked row count by the second
            # input row count and returns its matrix product with the second input.
            nz = torch.nonzero(x)
            # unbacked symint in the GEMM input shape
            a = x.new_ones([nz.size(0), y.size(0)])
            return a @ y

        example_inputs = (
            torch.randn((64), device=device),
            torch.randn((32, 16), device=device),
        )

        with inductor_config.patch(
            {
                "max_autotune_gemm": True,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_split_with_sizes(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that splits a tensor by
        # sizes obtained from converting a tensor to a list and returns the first chunk sum and
        # the sum of sizes; compares to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Converts the sizes tensor to a list, splits the first
            # input by it, and returns the sum of the first chunk with the total of the
            # first three sizes.
            l = y.tolist()
            s = torch.split(x, l)
            d = l[0] + l[1] + l[2]
            return s[0].sum(), d

        example_inputs = (torch.randn((32), device=device), torch.tensor((7, 16, 9)))

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_view_of_slice(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Tests creating a view over a slice whose size
        # involves an unbacked symbol; compiles the function and compares to eager.
        # Tests View.create(slice, size_with_unbacked_symint)
        def fn(x):
            # [FUNCTION DOC] HELPER: Squares the nonzero indices, slices the last two columns,
            # adds then removes a leading dimension, and returns the result.
            nz = torch.nonzero(x)  # introduce unbacked symint
            squared = nz * nz  # avoid ReinterpretView when lowering Slice
            sliced = torch.ops.aten.slice.Tensor(squared, dim=1, start=-2, end=None)
            view = sliced.unsqueeze(dim=0)
            return view.squeeze(
                dim=0
            )  # make sure no unbacked symint in output's stride

        example_inputs = (torch.randn(1, 1, 1, 1, device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_kernel_grid(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Skips on CPU. Compiles a function that launches
        # the add Triton kernel with a grid dimension equal to an unbacked scalar taken from an
        # input, and compares to eager.
        if device == "cpu":
            raise unittest.SkipTest("Triton kernel requires GPU")

        from torch.testing._internal.triton_utils import add_kernel

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar from the input, takes the larger of it and
            # 512, makes two ones tensors and a zeros tensor of that length, launches the add
            # kernel over a grid using the length and returns the output.
            maxlen = max(x.item(), 512)
            a = torch.ones(maxlen, device=device)
            b = torch.ones(maxlen, device=device)
            out = torch.zeros_like(a)
            # unbacked symint in grid
            add_kernel[(1, 1, maxlen)](a, b, out, maxlen, 32)
            return out

        example_inputs = (torch.randint(high=1024, size=(1,), device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_nonzero_in_inference_mode(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles nonzero under inference mode and
        # compares compiled and eager results computed in that mode.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the nonzero indices of the input.
            return torch.nonzero(x)

        example_inputs = (torch.randint(0, 2, (128,), device=device),)

        with torch.inference_mode():
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
            expected = fn(*example_inputs)

        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_equivalent_backed_unbacked(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Regression test with max autotune on: an
        # unbacked and a backed symbol are unified by concatenation and used in a batched
        # matmul, where size hint lookup on the unbacked symbol used to take the default
        # fallback. Compares to eager.
        # Tests scenario when there are two equivalent backed & unbacked symints,
        # but when we look-up a size hint on the unbacked symint, we ignorantly
        # use the default fallback hint.

        def fn(x, w, a, b):
            # [FUNCTION DOC] HELPER: Expands the first input by an item-derived unbacked count
            # and by the backed size of the fourth input, concatenates them, permutes, expands
            # the weight and returns the batched matmul.
            # Make tensors where 1st dim is unbacked/backed.
            u0, s0 = a.item(), b.size(0)
            unbacked = x.expand(u0, *x.shape)
            backed = x.expand(s0, *x.shape)

            # The cat unifies u0 and s0 -- i.e. u0 == s0.
            cat = torch.cat([backed, unbacked, unbacked], dim=1)  # [s0, 30, 16]
            mat1 = torch.permute(cat, [0, 2, 1])  # [s0, 16, 30]
            mat2 = w.expand(u0, *w.shape)  # [u0, 30, 32]
            bmm = torch.ops.aten.bmm(mat1, mat2)
            return bmm

        example_inputs = (
            torch.randn((10, 16), dtype=torch.float32, device=device),
            torch.randn((30, 32), dtype=torch.float32, device=device),
            torch.tensor(7, device=device),
            backed := torch.randn((7,), device=device),
        )
        torch._dynamo.mark_dynamic(backed, 0)  # create backed symint

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipCPUIf(True, "precision not good enough on CPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_vertical_pointwise_reduction_fusion(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Resets inductor metrics, compiles a
        # function producing a pointwise result and a sum over unbacked size, compares to eager
        # and asserts exactly one kernel was generated, i.e. pointwise and reduction fused.
        # reset in case we run both cpu and cuda tests
        torch._inductor.metrics.reset()

        # Tests fusing a pointwise & reduction op with unbacked numel/rnumel.
        def fn(x, y, repeats):
            # [FUNCTION DOC] HELPER: Expands the second input to an item-derived unbacked count,
            # adds the first input, and returns that pointwise result together with the sum of
            # it plus the first input again.
            u0 = repeats.item()
            unbacked = y.expand(u0, *y.shape)  # [u0, 1, 16]

            # Note: We add x to both pointwise and reduction. Otherwise, the
            # scheduler will refuse to fuse ops whose only common buffer has
            # unbacked symints.
            pointwise = unbacked + x
            reduction = torch.sum(pointwise + x)
            return pointwise, reduction

        example_inputs = (
            torch.randn(32, 16, device=device),
            torch.randn(1, 16, device=device),
            torch.tensor(32, device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(torch._inductor.metrics.generated_kernel_count, 1)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize(
        "torch_fn", [torch.mm, torch.bmm, torch.addmm], name_fn=lambda fn: fn.__name__
    )
    @parametrize("coordinate_descent_tuning", [True, False], name_fn=str)
    def test_mm_and_friends(self, device, torch_fn, coordinate_descent_tuning):
        # [FUNCTION DOC] Skipped without a GPU. Parametrized over matmul, batched matmul and
        # addmm, and coordinate descent tuning on or off. For addmm the function gets a bias
        # bound by partial application. Compiles a function with unbacked operand sizes and
        # compares to eager.
        if torch_fn == torch.addmm:
            torch_fn = functools.partial(torch_fn, torch.ones(1, device=device))

        def fn(x, w, repeats, is_bmm):
            # [FUNCTION DOC] HELPER: Expands the inputs to an item-derived unbacked size, adds a
            # batch dimension when the flag says batched, and returns the selected matmul
            # variant of them.
            u0 = repeats.item()

            x_unbacked = x.expand(u0, 32)
            w_unbacked = w.expand(32, u0)
            if is_bmm:
                # Make sure inputs are batched.
                x_unbacked = x_unbacked.expand(10, *x_unbacked.shape)
                w_unbacked = w_unbacked.expand(10, *w_unbacked.shape)

            return torch_fn(x_unbacked, w_unbacked)

        example_inputs = (
            torch.randn(1, 32, device=device),
            torch.randn(32, 1, device=device),
            torch.tensor(100, device=device),
            torch_fn == torch.bmm,
        )
        with inductor_config.patch(
            {
                # coordinate_descent_tuning has its own path during decomp
                "coordinate_descent_tuning": coordinate_descent_tuning,
            }
        ):
            actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_range_tree_divisor(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles an indexing function whose index tensor
        # has an unbacked length, exercising range tree divisors with unbacked symbols; compares
        # to eager.
        def fn(x, num):
            # [FUNCTION DOC] HELPER: Builds a zero int index tensor of item-derived length and
            # returns a tuple holding the input indexed along the second dimension by it.
            u0 = num.item()
            zeros = torch.zeros(u0, device=device, dtype=torch.int)
            return (torch.ops.aten.index(x, [None, zeros]),)

        example_inputs = (
            torch.randn(16, 16, device=device),
            torch.tensor(3, device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_unbacked_masked_scatter(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles masked scatter where the source length
        # is the unbacked count of true mask entries and compares to eager.
        def fn(value, mask):
            # [FUNCTION DOC] HELPER: Counts nonzero mask elements, builds a ones source of that
            # length and returns masked scatter of it into the value tensor.
            u0 = mask.count_nonzero()
            source = torch.ones(u0, dtype=torch.float32, device=device)
            return torch.masked_scatter(value, mask, source)

        value = make_tensor(10, 10, dtype=torch.float32, device=device)
        mask = make_tensor(10, 10, dtype=torch.bool, device=device)
        example_inputs = (value, mask)

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_unbacked_repeat(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles repeat with item-derived unbacked
        # repeat counts applied twice and compares to eager.
        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Extracts two scalars from the tensors and returns the input
            # repeated first by the first scalar and two, then by two and the second scalar.
            u0, u1 = a.item(), b.item()

            return x.repeat(u0, 2).repeat(2, u1)

        example_inputs = (
            make_tensor(1, 16, dtype=torch.float32, device=device),
            torch.scalar_tensor(2, dtype=torch.int32, device=device),
            torch.scalar_tensor(4, dtype=torch.int32, device=device),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch(
        {"capture_scalar_outputs": True, "capture_dynamic_output_shape_ops": True}
    )
    def test_repeat_interleave_with_unbacked_scalar(self, device):
        # [FUNCTION DOC] Skipped without a GPU. With scalar output capture and dynamic output
        # shape ops enabled, compiles repeat interleave with an unbacked scalar repeat count
        # with dynamic shapes and compares to eager.
        def fn(x, repeats):
            # [FUNCTION DOC] HELPER: Returns the input repeat-interleaved by the scalar value of
            # the repeats tensor.
            return x.repeat_interleave(repeats.item())

        example_inputs = (
            torch.arange(4, device=device),
            torch.scalar_tensor(3, dtype=torch.int64, device=device),
        )

        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    @parametrize("dynamic", [False, True, None])
    def test_unbacked_slice_on_subclass(self, device, dynamic):
        # [FUNCTION DOC] Skipped without a GPU. Parametrized on dynamic mode (off, on, default).
        # Defines a wrapper tensor subclass that forwards operations to an inner tensor but
        # handles slice using data-dependent bounds, which creates unbacked symbols inside the
        # subclass dispatch. Compiles a slice on an instance and compares the inner tensors of
        # compiled and eager results.
        from torch.testing._internal.common_subclass import WrapperTensor
        from torch.utils._pytree import tree_map

        # NB: the error we're testing for only triggers when unbacked SymInts
        # are created within a subclass's torch_dispatch, because they're not seen
        # by Dynamo and thus are considered freshly-created when the subclass instance
        # return value of the torch_dispatch is handled.
        # Subclass forwards everything along to the single underlying dense tensor
        # component, except for slice(), which it handles via data-dependent bounds access
        class CustomSliceSubclass(WrapperTensor):
            # [CLASS DOC] HELPER: Wrapper tensor subclass holding an inner tensor and a
            # tensor of slice bounds. All operations are forwarded to the inner tensor and
            # results are rewrapped, except slice, which reads the bounds as scalars to
            # create unbacked sizes.
            @classmethod
            def get_wrapper_properties(cls, t, slice_bounds=None):
                # [FUNCTION DOC] HELPER: Class method returning the inner tensor and an empty
                # options mapping, used by the wrapper base to build outer tensor properties.
                return t, {}

            def __init__(self, t, slice_bounds=None):
                # [FUNCTION DOC] HELPER: Stores the inner tensor and the slice bounds on the
                # instance.
                self.t = t
                self.slice_bounds = slice_bounds

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Returns a string showing the class name, the inner
                # tensor repr and the slice bounds repr.
                t_repr = repr(self.t)
                slice_bounds_repr = repr(self.slice_bounds)
                return f"CustomSliceSubclass({t_repr}, {slice_bounds_repr})"

            def __tensor_flatten__(self):
                # [FUNCTION DOC] HELPER: Returns the names of the two tensor attributes to
                # flatten and no extra metadata.
                return ["t", "slice_bounds"], None

            @classmethod
            def __tensor_unflatten__(
                cls, inner_tensors, meta, outer_size, outer_stride
            ):
                # [FUNCTION DOC] HELPER: Class method that rebuilds an instance from the
                # flattened inner tensors mapping, ignoring metadata and outer size and stride.
                t = inner_tensors["t"]
                slice_bounds = inner_tensors["slice_bounds"]
                return cls(t, slice_bounds)

            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch handler. For slice, reads the start and length
                # from the bounds as scalars, adds checks that they are within range, and
                # returns a new subclass instance holding the inner tensor sliced from start for
                # that length. Otherwise returns not implemented unless all types are subclasses
                # of this class, unwraps subclass arguments, runs the operation and wraps tensor
                # results back. Raises no errors itself.
                if func is torch.ops.aten.slice.Tensor:
                    inp = args[0]

                    start = inp.slice_bounds[0].item()
                    torch._check(start >= 0)
                    torch._check(start <= inp.size(0))

                    length = (args[0].slice_bounds[1] - args[0].slice_bounds[0]).item()
                    torch._check(length >= 0)
                    torch._check(start + length <= inp.size(0))

                    return CustomSliceSubclass(
                        func(args[0].t, dim=0, start=start, end=(start + length)),
                        slice_bounds=args[0].slice_bounds,
                    )

                if not all(issubclass(cls, t) for t in types):
                    return NotImplemented

                if kwargs is None:
                    kwargs = {}

                def unwrap(e):
                    # [FUNCTION DOC] HELPER: Returns the inner tensor if the value is an
                    # instance of the subclass, otherwise the value unchanged.
                    return e.t if isinstance(e, CustomSliceSubclass) else e

                def wrap(e):
                    # [FUNCTION DOC] HELPER: Wraps a tensor in the subclass, leaving
                    # non-tensors unchanged.
                    return CustomSliceSubclass(e) if isinstance(e, torch.Tensor) else e

                rs = tree_map(
                    wrap,
                    func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs or {})),
                )
                return rs

        def fn(t, start, length):
            # [FUNCTION DOC] HELPER: Slices the given tensor along dimension zero from start for
            # the given length.
            return torch.ops.aten.slice.Tensor(
                t, dim=0, start=start, end=start + length
            )

        t = make_tensor(22, 5, dtype=torch.float32, device=device)
        sub = CustomSliceSubclass(t, slice_bounds=torch.tensor([2, 5], device=t.device))
        start = 2
        length = 3
        example_inputs = (sub, start, length)

        actual = torch.compile(fn, dynamic=dynamic, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual.t, expected.t)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch(capture_dynamic_output_shape_ops=True)
    def test_issue_143498(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Regression test for issue 143498: compiles a
        # module combining boolean-mask style indexing, concatenation and index put with an
        # unbacked size, and asserts compiled and eager outputs are equal.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward does indexing, unsqueeze, concatenation,
            # select and index put.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base module initializer; defines no state.
                super().__init__()

            def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1):
                # [FUNCTION DOC] HELPER: Indexes the second and first inputs with the third,
                # stacks them as two columns, selects column zero and index-puts the fifth
                # input into the sixth input at that selection and the seventh input.
                # Returns the result.
                index = torch.ops.aten.index.Tensor(arg1_1, [arg2_1])
                index_1 = torch.ops.aten.index.Tensor(arg0_1, [arg2_1])
                unsqueeze = torch.ops.aten.unsqueeze.default(index, 1)
                unsqueeze_1 = torch.ops.aten.unsqueeze.default(index_1, 1)
                cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1)
                select = torch.ops.aten.select.int(cat, 1, 0)
                index_put = torch.ops.aten.index_put.default(
                    arg5_1, [select, arg6_1], arg4_1
                )
                return index_put

        example_inputs = (
            torch.tensor(
                [-1, -1, 14, -1, -1, -1, -1, -1, -1, -1, 49, -1],
                device=device,
                dtype=torch.int64,
            ),
            torch.tensor(
                [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                device=device,
                dtype=torch.int64,
            ),
            torch.tensor(
                [
                    False,
                    False,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    True,
                    False,
                ],
                device=device,
                dtype=torch.bool,
            ),
            torch.tensor([2, 10], device=device, dtype=torch.int64),
            torch.tensor([34, 33], device=device, dtype=torch.int64),
            torch.zeros(3, 50, device=device, dtype=torch.int64),
            torch.tensor([14, 49], device=device, dtype=torch.int64),
        )
        model = Model()
        self.assertEqual(torch.compile(model)(*example_inputs), model(*example_inputs))

    @skipGPUIf(not HAS_GPU, "torch.compile for gpu requires triton")
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_einsum(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that repeats two tensors by
        # an item-derived count then runs several einsum forms on them, and compares to eager.
        def fn(q, k, vector, scalar):
            # [FUNCTION DOC] HELPER: Repeats the query and key along dimension one by an
            # unbacked scalar and returns three einsum results: attention scores, an ellipsis
            # form of the same product, and a product with the vector.
            unbacked = scalar.item()
            q = q.repeat(1, unbacked, 1, 1)
            k = k.repeat(1, unbacked, 1, 1)

            qk = torch.einsum("bcxd,bcyd->bcxy", (q, k))
            qk2 = torch.einsum("b...,b...->b...", (q, k))
            qvec = torch.einsum("b...,b->b...", (q, vector))
            return qk, qk2, qvec

        example_inputs = (
            torch.empty_strided(
                (12, 1, 512, 64), (64, 196608, 768, 1), device=device
            ).uniform_(0, 1),
            torch.empty_strided(
                (12, 1, 512, 64), (64, 196608, 768, 1), device=device
            ).uniform_(0, 1),
            torch.randn((12,), device=device),
            torch.scalar_tensor(10, device=device, dtype=torch.int8),
        )
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_softmax(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles softmax and log softmax over an
        # unbacked-length float tensor from nonzero and compares to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts the nonzero indices to float and returns the
            # product of their softmax and log softmax along dimension zero.
            nz = x.nonzero().float()
            soft = torch.softmax(nz, dim=0)
            logsoft = torch.nn.functional.log_softmax(nz, dim=0)
            return soft * logsoft

        example_inputs = (
            torch.randint(low=0, high=2, size=(32,), device=device, dtype=torch.int8),
        )
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @skipCUDAIf(not SM80OrLater, "Requires sm80 or later.")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_sdpfa(self, device):
        # [FUNCTION DOC] Skipped without GPU, on CUDA before sm80, and on CPU. Compiles a
        # function calling flash attention on random half precision tensors whose sequence
        # length is an unbacked nonzero count. Checks only that compilation and execution
        # succeed.
        if device == "cpu":
            raise unittest.SkipTest(
                "scaled_dot_product_flash_attention has no CPU backend"
            )

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates random query, key and value tensors with sequence
            # length equal to the nonzero count and returns the flash attention result with no
            # dropout and no causal mask.
            B, H, d_h = 2, 4, 8
            nz = torch.nonzero(x)
            seq_len = nz.size(0)

            q = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)
            k = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)
            v = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)

            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, dropout_p=0.0, is_causal=False, scale=None
            )
            return result

        x = torch.tensor([1.0, 0.0, 1.0, 0.0], device=device)
        torch.compile(fn, fullgraph=True)(x)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @skipCUDAIf(not SM80OrLater, "Requires sm80 or later.")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_sdfpa_unbacked_strides(self, device):
        # [FUNCTION DOC] Skipped without GPU, on CUDA before sm80, and on CPU. Like the flash
        # attention test but the query, key and value are strided views whose strides contain an
        # unbacked value. Only checks that compilation and execution succeed.
        if device == "cpu":
            raise unittest.SkipTest("scaled_dot_product_attention has no CPU backend")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds query, key and value with unbacked sequence length,
            # re-views them with custom strides that include an unbacked size from the second
            # input, and returns the flash attention result.
            B, H, d_h = 2, 4, 16
            nz = torch.nonzero(x)
            seq_len = nz.size(0)
            y = torch.nonzero(y).size(0)
            strides = (H * seq_len * d_h, seq_len * d_h, d_h, y)

            q = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)
            k = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)
            v = torch.randn(B, H, seq_len, d_h, device=device, dtype=torch.float16)
            q = torch.as_strided(q, size=(B, H, seq_len, d_h), stride=strides)
            k = torch.as_strided(k, size=(B, H, seq_len, d_h), stride=strides)
            v = torch.as_strided(v, size=(B, H, seq_len, d_h), stride=strides)
            result = torch.ops.aten._scaled_dot_product_flash_attention.default(
                q, k, v, dropout_p=0.0, is_causal=False, scale=None
            )
            return result

        x = torch.tensor([1.0, 0.0] * 8, device=device)
        y = torch.tensor([1.0, 0.0], device=device)
        torch.compile(fn, fullgraph=True)(x, y)

    @skipGPUIf(not HAS_GPU, "torch.compile for gpu requires triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_unbacked_linear_layer_norm_input(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a module with a linear layer and two
        # layer norms applied after masked select reshaped to a fixed width, so the row count is
        # unbacked; compares to eager.
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one linear layer from 387 to 128 features and
            # layer norms over 387 and 128 features, all on the test device.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear layer and the two layer norm layers
                # on the test device.
                super().__init__()
                self.linear = torch.nn.Linear(387, 128, bias=True, device=device)
                self.layer_norm1 = torch.nn.LayerNorm(387, device=device)
                self.layer_norm2 = torch.nn.LayerNorm(128, device=device)

            def forward(self, x, mask):
                # [FUNCTION DOC] HELPER: Selects elements of the first input by the mask,
                # reshapes to rows of 387, applies the linear layer and both layer norms, and
                # returns the linear output and the two normalized outputs.
                masked_select = x.masked_select(mask)
                view = masked_select.view(-1, 387)

                linear = self.linear(view)
                layer_norm1 = self.layer_norm1(view)
                layer_norm2 = self.layer_norm2(linear)
                return linear, layer_norm1, layer_norm2

        model = MyModel()
        inputs = (
            torch.randn((256, 387), dtype=torch.float, device=device),
            torch.randint(
                low=0, high=2, size=(256, 1), dtype=torch.bool, device=device
            ),
        )

        actual = torch.compile(model, fullgraph=True)(*inputs)
        expected = model(*inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "torch.compile for gpu requires triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_to_int_with_unbacked_size(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that builds a transposed
        # ones tensor with an unbacked size from a scalar and casts it to int, comparing to
        # eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar, makes a ones tensor of one quarter that
            # many rows by ten, transposes it to avoid contiguity shortcuts and returns it
            # as int.
            unbacked = x.item()

            # Transpose to avoid contig short-circuit.
            unbacked_size = torch.ones(
                size=(unbacked // 4, 10), device=device
            ).transpose(0, 1)
            return unbacked_size.int()

        example_inputs = (torch.tensor(16, device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    @inductor_config.patch({"combo_kernels": True, "benchmark_combo_kernel": True})
    def test_combo_kernel_size_hint_failure(self, device):
        # [FUNCTION DOC] Skipped without GPU and on CPU. With combo kernels and their
        # benchmarking enabled, compiles a function producing several tensors with sizes
        # derived from an unbacked count, to check that size hint failures are handled.
        # Compares to eager.
        # A size hint failure is "TypeError: Cannot convert symbols to int"
        if device == "cpu":
            raise unittest.SkipTest("Combo kernels must be for GPU.")

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates four tensors with lengths derived from the nonzero
            # count (count, count plus one, twice count, count minus input size), applies a
            # simple arithmetic op to each and returns all four.
            nz = torch.nonzero(x)
            u0 = nz.size(0)
            t1 = torch.ones(u0, device=device)
            t2 = torch.zeros(u0 + 1, device=device)
            t3 = torch.zeros(u0 * 2, device=device)
            t4 = torch.zeros(u0 - x.size(0), device=device)
            out1 = t1 - 1
            out2 = t2 + 2
            out3 = t3 * 3
            out4 = t4 / 4
            return out1, out2, out3, out4

        example_inputs = (torch.randn(32, device=device, dtype=torch.float16),)
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    @inductor_config.patch({"benchmark_kernel": True})
    def test_triton_kernel_with_unbacked_symint_fallback(self, device):
        # [FUNCTION DOC] Skipped without a GPU. With kernel benchmark codegen enabled,
        # compiles a function that selects entries by an index tensor containing an unbacked
        # value, exercising the fallback path where a size argument is an unbacked symbol.
        # Compares to eager.
        # The benchmark_kernel=True config exercises the codegen_kernel_benchmark code path
        # Test isinstance(arg_sig, SizeArg) == True in the fallback path
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds an index tensor from the constant one and the
            # nonzero count minus five, and returns the input index-selected along dimension
            # zero.
            # Create unbacked SymInt
            nz = torch.nonzero(x)
            u0 = nz.size(0)
            # Create indices for index_select operation
            indices = torch.tensor([1, u0 - 5], device=device)
            # Create SizeArg object
            x = torch.index_select(x, 0, indices)
            return x

        example_inputs = (torch.randn(32, device=device, dtype=torch.float16),)
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipIfXpu(
        msg="Invalid SPIR-V module,https://github.com/intel/torch-xpu-ops/issues/2329"
    )
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @inductor_config.patch({"max_autotune": True})
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_autotune_with_unbacked_stride(self, device):
        # [FUNCTION DOC] Skipped on XPU (invalid SPIR-V module) and without a GPU. With max
        # autotune and scalar output capture, compiles a batched matmul whose operand has an
        # unbacked stride and compares to eager.
        def fn(x, y, a):
            # [FUNCTION DOC] HELPER: Reads a scalar, checks it is not one, expands and clones
            # the first input to an unbacked shape, permutes it, expands the second input over a
            # batch and returns the batched matmul.
            u0 = a.item()
            torch._check(u0 != 1)
            unbacked = x.expand(8, u0, *x.shape).clone()
            unbacked = torch.permute(unbacked, [0, 2, 1])
            y = y.expand(8, *y.shape)
            bmm = torch.ops.aten.bmm(unbacked, y)
            return bmm

        example_inputs = (
            torch.randn((32,), dtype=torch.bfloat16, device=device),
            torch.randn((128, 64), dtype=torch.bfloat16, device=device),
            torch.tensor(128, device=device),
        )
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    def test_fmod_with_out_arg(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Compiles a function that applies fmod with the
        # output tensor given as the input itself, over an unbacked-length tensor, and compares
        # to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Converts nonzero indices to float and returns fmod by 2.0
            # written into that same tensor.
            nz = torch.nonzero(x).float()
            return torch.fmod(nz, 2.0, out=nz)

        example_inputs = (torch.randn(32, device=device),)
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_pow_type_mismatch(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Regression test that mixed
        # float32 and float64 arguments to the Triton pow helper work when a square root of
        # a dynamic size is raised to a non-integer power. Compiles with dynamic shapes and
        # compares to eager.
        """
        Test that libdevice.pow handles mixed float32/float64 types correctly.

        Reproducer: sqrt returns float32, but exponent is float64 from sympy.
        This caused: KeyError: (triton.language.float32, triton.language.float64)

        The bug only triggers with a NON-INTEGER float exponent (like 2.5).
        Integer exponents (like 10.0) are printed differently and don't trigger
        the type mismatch.
        """
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Takes the square root of the first dimension size, raises
            # it to the power 2.5 and returns a float64 scalar tensor of the truncated value.
            # sqrt(x.size(0)) creates OpaqueUnaryFn_sqrt -> printed as float32
            # **2.5 creates FloatPow with sympy.Float(2.5) -> printed as float64
            # This causes: KeyError: (triton.language.float32, triton.language.float64)
            r = math.sqrt(x.size(0))
            r = r**2.5  # Non-integer exponent is required to trigger the bug
            return torch.tensor(math.trunc(r), dtype=torch.float64, device=device)

        example_inputs = (torch.randn(4, device=device),)
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_trunc_large_float_scalar_tensor(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Compiles a function raising a square
        # root of a dynamic size to a large power and truncating the result into a float64
        # scalar tensor, checking large values truncate correctly. Compares to eager.
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a float64 scalar tensor of the truncation of the
            # square root of the first dimension size raised to the seventieth power.
            r = math.sqrt(x.size(0))
            r = r**70
            return torch.tensor(math.trunc(r), dtype=torch.float64, device=device)

        example_inputs = (torch.randn(4, device=device),)
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_trunc_float_scalar_tensor_preserves_positive_zero(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Checks truncation of a small negative
        # fractional value keeps a positive zero, by comparing the sign bit of the compiled
        # result with eager using exact equality.
        import math

        def fn(x):
            # [FUNCTION DOC] HELPER: Subtracts 2.5 from the square root of the first dimension
            # size and returns the sign bit of a float64 scalar tensor of the truncated value.
            r = math.sqrt(x.size(0)) - 2.5
            return torch.signbit(
                torch.tensor(math.trunc(r), dtype=torch.float64, device=device)
            )

        example_inputs = (torch.randn(4, device=device),)
        torch._dynamo.mark_dynamic(example_inputs[0], 0)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        self.assertEqual(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_pow_symbolic_int_exponent(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Regression test for issue 177131: a
        # symbolic integer exponent passed as a kernel argument must be cast before the Triton
        # pow call. Compiles a quantization scale function with dynamic shapes and an integer
        # bit count and compares to eager.
        """
        Test that symbolic integer scalar exponents are cast before libdevice.pow.

        Reproducer from https://github.com/pytorch/pytorch/issues/177131:
        capture_scalar_outputs=True can pass a scalar like num_bits as an int64 ks*
        kernel argument, which previously led Triton to reject libdevice.pow(2.0, ks0).
        """

        def calculate_range(num_bits, device):
            # [FUNCTION DOC] HELPER: Computes two to the power of the bit count and returns the
            # minimum and maximum quantization levels as scalar tensors on the given device.
            bit_range = 2**num_bits
            q_max = torch.tensor(bit_range / 2 - 1, device=device)
            q_min = torch.tensor(-bit_range / 2, device=device)
            return q_min, q_max

        def fn(x, num_bits):
            # [FUNCTION DOC] HELPER: Computes a symmetric quantization scale from the largest
            # absolute input value, replaces a zero scale with a tiny value, quantizes and
            # dequantizes the input, and returns the sum of the absolute error raised to 2.4
            # plus the scale twice.
            q_min, q_max = calculate_range(num_bits, x.device)
            bit_range = q_max - q_min
            max_val = torch.max(torch.abs(x))
            scale = max_val / (float(bit_range) / 2)
            scale = torch.where(
                scale == 0,
                torch.tensor(1e-10, device=x.device, dtype=x.dtype),
                scale,
            )
            x_q = torch.clamp(torch.round(x / scale), q_min, q_max)
            x_dq = x_q * scale
            return (x_dq - x).abs().pow(2.4).sum(), scale, scale

        example_inputs = (torch.randn(1, 1, 128, device=device, dtype=torch.float16), 8)
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipCPUIf(True, "Triton codegen bug only affects GPU")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_triton_pow_symbolic_negative_int_exponent(self, device):
        # [FUNCTION DOC] Skipped on CPU and without a GPU. Compiles a function multiplying the
        # input by two raised to an int64 exponent taken from a tensor maximum minus one
        # (possibly negative), with dynamic shapes, and compares to eager.
        def fn(x, exponent_src):
            # [FUNCTION DOC] HELPER: Computes the integer exponent from the max of the second
            # input and returns the first input times two to that power.
            exponent = exponent_src.max().to(torch.int64) - 1
            return x * (2**exponent)

        example_inputs = (
            torch.randn(128, device=device, dtype=torch.float16),
            torch.tensor([0], device=device, dtype=torch.int64),
        )
        actual = torch.compile(fn, fullgraph=True, dynamic=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_slice_unbacked_bindings_with_later_constraint(self, device):
        # [FUNCTION DOC] Skipped without a GPU. Regression test for issue 166460: slicing with
        # an unbacked end index may create a fresh unbacked symbol that later becomes provable;
        # inductor must still define it on the slice view path. Compiles and compares to eager.
        # Regression test for https://github.com/pytorch/pytorch/issues/166460
        # When slicing with an unbacked symint end index (e.g. x[:total] where
        # total = sum of data-dependent values), dynamo may allocate a fresh
        # unbacked symbol for the output size because it can't prove bounds at
        # trace time. Later operations (e.g. new_zeros with padding) may
        # establish constraints that make the bounds provable at inductor time.
        # Inductor must still define the unbacked symbol even when taking the
        # efficient SliceView path.
        def fn(x, sizes):
            # [FUNCTION DOC] HELPER: Converts sizes to a list, slices the input to their sum,
            # splits by the sizes, doubles each piece, concatenates, and pads with a zeros block
            # of the removed row count appended below.
            sizes_list = sizes.tolist()
            total = sum(sizes_list)
            num_padding = x.shape[0] - total
            sliced = x[:total]
            splits = torch.split(sliced, sizes_list, dim=0)
            out = torch.cat([s * 2 for s in splits], dim=0)
            out = torch.vstack((out, out.new_zeros((num_padding, out.shape[-1]))))
            return out

        example_inputs = (
            torch.randn(64, 16, device=device, dtype=torch.bfloat16),
            torch.tensor([8, 8], device=device, dtype=torch.int64),
        )

        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)

    @skipIfXpu(msg="standalone_compile coverage is CUDA-only")
    @skipCPUIf(True, "requires gpu and triton")
    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    def test_standalone_compile_reuses_fallback_unbacked_binding(self, device):
        # [FUNCTION DOC] Skipped on XPU, CPU and without GPU. Traces a function using repeat
        # interleave with symbolic tracing, finds the fake mode from graph node values, and
        # under that tracing context compiles the graph with the standalone compile entry
        # point using dynamic shapes from the tracing context. Asserts the compiled output
        # is close to eager.
        from torch._inductor import standalone_compile
        from torch._subclasses.fake_tensor import FakeTensor
        from torch.fx.experimental.proxy_tensor import make_fx

        def fn(counts, x):
            # [FUNCTION DOC] HELPER: Repeat-interleaves the counts into an index tensor, indexes
            # the second input with it and returns the sine.
            idx = torch.repeat_interleave(counts)
            return x[idx].sin()

        counts = torch.tensor([1, 2, 1, 0], device=device, dtype=torch.int64)
        x = torch.randn(4, 8, device=device)
        torch._dynamo.mark_dynamic(counts, 0, min=1, max=8)
        gm = make_fx(fn, tracing_mode="symbolic")(counts, x)
        fake_mode = next(
            node.meta["val"].fake_mode
            for node in gm.graph.nodes
            if isinstance(node.meta.get("val"), FakeTensor)
        )

        with (
            torch._guards.tracing(torch._guards.TracingContext(fake_mode)),
            fake_mode.shape_env.ignore_fresh_unbacked_symbols(),
        ):
            compiled = standalone_compile(
                gm,
                [counts, x],
                dynamic_shapes="from_tracing_context",
                options={},
            )

        torch.testing.assert_close(compiled(counts, x), fn(counts, x))

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_inplace_multidim_dynamic_slice_tensor_bound(self, device):
        # [FUNCTION DOC] Regression test for issue 183259 with scalar output capture: in-place
        # assignment of a constant to a multi-dimensional slice whose bounds come from tensor
        # values, including out-of-range and negative values. Compiles with dynamic shapes and
        # compares to eager for each span in a fixed set.
        # Regression test for https://github.com/pytorch/pytorch/issues/183259
        def fn(tensor_span):
            # [FUNCTION DOC] HELPER: Allocates a zeros mask of batch by 20 by 20 and, per batch
            # entry, sets the leading square block bounded by the span to one. Returns the mask.
            batch_size = tensor_span.shape[0]
            mask = torch.zeros(
                (batch_size, 20, 20),
                dtype=torch.float32,
                device=tensor_span.device,
            )
            for i in range(batch_size):
                span = tensor_span[i, 0]
                mask[i, :span, :span] = 1.0
            return mask

        compiled_fn = torch.compile(fn, fullgraph=True, dynamic=True)
        for span in (8, 25, -1, -25, 0):
            tensor_span = torch.tensor([[span]], dtype=torch.int64, device=device)
            actual = compiled_fn(tensor_span)
            expected = fn(tensor_span)
            torch.testing.assert_close(actual, expected)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_slice_scatter_dynamic_end_invalid_src_size(self, device):
        # [FUNCTION DOC] With scalar output capture, compiles slice scatter with a
        # data-dependent end and checks that for each of three source lengths a runtime error is
        # raised, whose message mentions the expected source size or an equality constraint.
        def fn(x, src, end):
            # [FUNCTION DOC] HELPER: Returns slice scatter of the source into the input from
            # zero to the scalar end with step one.
            return torch.ops.aten.slice_scatter.default(x, src, 0, 0, end.item(), 1)

        compiled_fn = torch.compile(fn, fullgraph=True, dynamic=True)
        for src_len in (1, 2, 4):
            x = torch.arange(5, dtype=torch.float32, device=device)
            src = torch.ones(src_len, dtype=torch.float32, device=device)
            end = torch.tensor(3, dtype=torch.int64, device=device)
            with self.assertRaisesRegex(
                RuntimeError,
                "expected src to have a size equal to the slice of self|Eq\\(",
            ):
                compiled_fn(x, src, end)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_override_optimization_hint_eager(self, device):
        # [FUNCTION DOC] Eager mode check that the hint override API runs on an unbacked scalar
        # and the function still returns the right value (six).
        """Test that override_optimization_hint updates var_to_hint_override eagerly."""
        t = torch.tensor([5], device=device)
        torch._dynamo.decorators.mark_unbacked(t, 0)

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar, requests an optimization hint override of
            # 42 for it, and returns it plus one.
            u = x.item()
            torch._dynamo.override_optimization_hint(u, 42)
            return u + 1

        result = fn(t)
        self.assertEqual(result, 6)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_override_optimization_hint_compiled(self, device):
        # [FUNCTION DOC] Compiles with fullgraph a function using the hint override on an
        # item-derived value and asserts the result is six.
        """Test override_optimization_hint inside a compiled function with fullgraph=True."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar, overrides its optimization hint to 42 and
            # returns it plus one.
            u = x.item()
            torch._dynamo.override_optimization_hint(u, 42)
            return u + 1

        t = torch.tensor([5], device=device)
        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(t)
        self.assertEqual(result, 6)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_override_optimization_hint_compiled_tolist(self, device):
        # [FUNCTION DOC] Checks the hint override is a harmless no-op on concrete ints from
        # converting a tensor to a list, in a fullgraph compile; asserts the result equals the
        # sum of the input.
        """Test that override_optimization_hint is a no-op on concrete ints from tolist()."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Converts the tensor to a list, calls the hint override with
            # 99 for each element and returns the tensor sum.
            vals = x.tolist()
            for v in vals:
                torch._dynamo.override_optimization_hint(v, 99)
            return x.sum()

        t = torch.tensor([3, 4, 5], device=device)
        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(t)
        self.assertEqual(result, t.sum())

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_override_optimization_hint_multiple_items(self, device):
        # [FUNCTION DOC] Checks hint overrides on two separate unbacked scalars (128 and 256)
        # compile and the returned sum is 30.
        """Test override_optimization_hint on multiple unbacked symbols from separate .item() calls."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Reads two scalars, overrides their hints and returns
            # their sum.
            u = x.item()
            v = y.item()
            torch._dynamo.override_optimization_hint(u, 128)
            torch._dynamo.override_optimization_hint(v, 256)
            return u + v

        tx = torch.tensor([10], device=device)
        ty = torch.tensor([20], device=device)
        compiled_fn = torch.compile(fn, fullgraph=True)
        result = compiled_fn(tx, ty)
        self.assertEqual(result, 30)

    def test_override_optimization_hint_concrete_int_noop(self, device):
        # [FUNCTION DOC] Checks the hint override accepts a plain integer without raising.
        """Test that override_optimization_hint on a plain int is a no-op."""
        torch._dynamo.override_optimization_hint(42, 100)

    def test_override_optimization_hint_rejects_wrong_type(self, device):
        # [FUNCTION DOC] Checks the hint override raises a type error whose message says it
        # expects a symbolic int or int when given a float or a string.
        """Test that override_optimization_hint raises TypeError on non-int/non-SymInt."""
        with self.assertRaisesRegex(TypeError, "expects a torch.SymInt or int"):
            torch._dynamo.override_optimization_hint(3.14, 100)
        with self.assertRaisesRegex(TypeError, "expects a torch.SymInt or int"):
            torch._dynamo.override_optimization_hint("hello", 100)

    def test_override_optimization_hint_rejects_non_int_val(self, device):
        # [FUNCTION DOC] Checks the hint override raises a type error saying the value must be
        # an int when the hint value is a float or a string.
        """Test that override_optimization_hint rejects non-int val."""
        with self.assertRaisesRegex(TypeError, "val to be an int"):
            torch._dynamo.override_optimization_hint(42, 3.14)
        with self.assertRaisesRegex(TypeError, "val to be an int"):
            torch._dynamo.override_optimization_hint(42, "hello")

    def test_override_optimization_hint_rejects_derived_expression(self, device):
        # [FUNCTION DOC] Creates an unbacked symbolic int in a fresh shape environment, derives
        # an expression by adding one, and checks the hint override raises a value error stating
        # it needs a single unbacked symbol.
        """Test that override_optimization_hint rejects derived expressions like u0 + 1."""
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        u = shape_env.create_unbacked_symint()
        v = u + 1  # derived expression: u0 + 1
        with self.assertRaisesRegex(ValueError, "single unbacked symbol"):
            torch._dynamo.override_optimization_hint(v, 42)

    def test_override_optimization_hint_rejects_backed_symbol(self, device):
        # [FUNCTION DOC] Compiles with fullgraph a function applying the hint override to a
        # backed (dynamic size) symbol and asserts the compile raises the internal dynamo error
        # saying an unbacked symbol is expected.
        """Test that override_optimization_hint rejects backed (non-unbacked) symbols."""

        def fn(t):
            # [FUNCTION DOC] HELPER: Takes the size of the input, tries to override its hint and
            # returns the sum.
            s = t.size(0)  # backed symbol inside compile
            torch._dynamo.override_optimization_hint(s, 42)
            return t.sum()

        t = torch.randn(5, device=device)
        torch._dynamo.mark_dynamic(t, 0)
        with self.assertRaisesRegex(
            InternalTorchDynamoError,
            "expects an unbacked symbol",
        ):
            torch.compile(fn, fullgraph=True)(t)

    @dynamo_config.patch({"capture_scalar_outputs": True})
    def test_override_optimization_hint_in_fx_pass(self, device):
        # [FUNCTION DOC] Compiles with a custom backend that walks the graph for item calls,
        # applies the hint override of 512 to their example values, then verifies the shape
        # environment records the override for the symbol, raising an assertion error if missing
        # or different. Asserts the compiled function returns eight.
        """Test using override_optimization_hint in a custom FX pass (backend).

        This shows the intended use case: a custom backend or FX pass walks
        the graph, finds unbacked symbols from .item() calls, and sets
        optimization hints on them before handing the graph to inductor.
        """

        def fx_pass_backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Custom compile backend: sets a hint override of 512 on each
            # symbolic int example value of scalar extraction nodes, then checks the recorded
            # override on the shape environment, raising an assertion error on missing or
            # mismatched value, and returns the graph module unchanged so it runs as is.
            # Walk the graph and set hints on unbacked SymInts
            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.item.default
                ):
                    # The node's example value is a SymInt from .item()
                    sym_val = node.meta.get("example_value", None)
                    if sym_val is not None and isinstance(sym_val, torch.SymInt):
                        torch._dynamo.override_optimization_hint(sym_val, 512)

            # Verify the hint was set on shape_env
            shape_env = None
            for node in gm.graph.nodes:
                if (
                    node.op == "call_function"
                    and node.target is torch.ops.aten.item.default
                ):
                    sym_val = node.meta.get("example_value", None)
                    if sym_val is not None and isinstance(sym_val, torch.SymInt):
                        shape_env = sym_val.node.shape_env
                        expr = sym_val.node.expr
                        if expr not in shape_env.var_to_hint_override:
                            raise AssertionError("hint not set on shape_env")
                        if shape_env.var_to_hint_override[expr] != 512:
                            raise AssertionError("hint value mismatch")

            return gm

        def fn(x):
            # [FUNCTION DOC] HELPER: Reads a scalar from the input and returns it plus one.
            u = x.item()
            return u + 1

        t = torch.tensor([7], device=device)
        compiled_fn = torch.compile(fn, backend=fx_pass_backend, fullgraph=True)
        result = compiled_fn(t)
        self.assertEqual(result, 8)

    def test_stride_order_uses_unbacked_optimization_hint(self, device):
        # [FUNCTION DOC] Creates an unbacked symbol with override hint 16 and checks the stride
        # order helper, for sizes involving a max of one and half the symbol, returns the order
        # two, one, zero, i.e. it uses the override hint.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        seq = shape_env.create_unbacked_symint()
        torch._dynamo.override_optimization_hint(seq, 16)

        stride_order = ir.get_stride_order(
            [256 * sympy.Max(1, seq.node.expr // 2), 256, 1],
            shape_env,
        )
        self.assertEqual(stride_order, [2, 1, 0])

    def test_stride_ordered_uses_symbolic_divisibility(self, device):
        # [FUNCTION DOC] Builds a fixed layout with an unbacked leading size and strides that
        # are multiples, and checks it is reported stride ordered using symbolic divisibility,
        # with the optimization hint method patched to raise so any use fails the test.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        seq = shape_env.create_unbacked_symint().node.expr
        sizevars = SizeVarAllocator(shape_env)
        graph = mock.Mock(sizevars=sizevars)

        layout = ir.FixedLayout(
            torch.device(device),
            torch.float32,
            size=[seq, 8, 16],
            stride=[256 * seq, 256, 1],
        )
        with V.set_graph_handler(graph):
            with mock.patch.object(
                sizevars,
                "optimization_hint",
                side_effect=AssertionError("unexpected optimization hint"),
            ):
                self.assertTrue(layout.is_stride_ordered([2, 1, 0]))

    def test_stride_ordered_rejects_unproved_unbacked_layout(self, device):
        # [FUNCTION DOC] Builds a fixed layout with unbacked size and strides that cannot be
        # proven ordered and checks the stride-ordered query returns false without ever calling
        # the patched optimization hint method.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        seq = shape_env.create_unbacked_symint().node.expr
        sizevars = SizeVarAllocator(shape_env)
        graph = mock.Mock(sizevars=sizevars)

        layout = ir.FixedLayout(
            torch.device(device),
            torch.float32,
            size=[seq, 2],
            stride=[seq, 2],
        )
        with V.set_graph_handler(graph):
            with mock.patch.object(
                sizevars,
                "optimization_hint",
                side_effect=AssertionError("unexpected optimization hint"),
            ):
                self.assertFalse(layout.is_stride_ordered([1, 0]))

    def test_python_wrapper_binds_symbol_from_compound_input_expr(self, device):
        # [FUNCTION DOC] CPU only (skips other devices). Builds mock graph inputs whose sizes
        # are compound expressions of symbols (including an unbacked one) and checks the Python
        # wrapper and the C++ CPU wrapper input codegen bind each symbol from the right input
        # size variable, including a replaced symbol, a floor division size and a sum of two
        # symbols. Examines the generated prefix text and unbacked declarations set. Wrappers
        # are created without running their constructors.
        if device != "cpu":
            self.skipTest("wrapper codegen unit test only needs one device")

        from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        s0 = sympy.Symbol("s0", integer=True)
        s1 = sympy.Symbol("s1", integer=True)
        s2 = sympy.Symbol("s2", integer=True)
        shape_env = ShapeEnv()
        u0 = shape_env.create_unbacked_symint().node.expr
        shape_env.replacements[s0] = s2
        sizevars = SizeVarAllocator(shape_env)
        graph_inputs = {}
        graph = mock.Mock(
            sizevars=sizevars,
            graph_inputs=graph_inputs,
            graph_input_names=list(graph_inputs),
            symbolic_input_sources={},
        )

        value = ir.TensorBox.create(
            ir.InputBuffer(
                name="arg0_1",
                layout=ir.FixedLayout(
                    torch.device(device),
                    torch.float32,
                    size=[s0 + 1, 2 * s1, u0 + 1],
                    stride=[2 * s1, 1, 1],
                ),
            )
        )
        graph_inputs["arg0_1"] = value
        graph.graph_input_names = list(graph_inputs)

        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()

        with V.set_graph_handler(graph):
            wrapper.codegen_inputs()

        code = wrapper.prefix.getvalue()
        self.assertIn("arg0_1_size = arg0_1.size()", code)
        self.assertIn("arg0_1_size_0 = arg0_1_size[0]", code)
        self.assertIn("arg0_1_size_1 = arg0_1_size[1]", code)
        self.assertRegex(code, r"s0 = .*arg0_1_size_0")
        self.assertIn("s2 = s0", code)
        self.assertIn("s1 = arg0_1_size_1 // 2", code)
        self.assertRegex(code, r"u0 = .*arg0_1_size_2")

        cpp_wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        cpp_wrapper.prefix = IndentedBuffer()
        cpp_wrapper.unbacked_symbol_decls = OrderedSet()
        with V.set_graph_handler(graph):
            cpp_wrapper.codegen_inputs()

        cpp_code = cpp_wrapper.prefix.getvalue()
        self.assertIn("auto arg0_1_size = arg0_1.sizes();", cpp_code)
        self.assertIn("int64_t arg0_1_size_0 = arg0_1_size[0];", cpp_code)
        self.assertRegex(cpp_code, r"int64_t s0 = .*arg0_1_size_0")
        self.assertIn("int64_t s2 = s0;", cpp_code)
        self.assertIn("int64_t s1 = c10::div_floor_integer", cpp_code)
        self.assertIn("arg0_1_size_1", cpp_code)
        self.assertRegex(cpp_code, r"int64_t u0 = .*arg0_1_size_2")
        self.assertIn(str(u0), cpp_wrapper.unbacked_symbol_decls)
        cpp_wrapper.declare = "auto "
        self.assertEqual(str(u0), cpp_wrapper.codegen_unbacked_symbol_decl(u0))

        from torch.utils._sympy.functions import FloorDiv

        s3 = sympy.Symbol("s3", integer=True)
        graph_inputs = {
            "arg0_1": ir.TensorBox.create(
                ir.InputBuffer(
                    name="arg0_1",
                    layout=ir.FixedLayout(
                        torch.device(device),
                        torch.float32,
                        size=[FloorDiv(s3, 2)],
                        stride=[1],
                    ),
                )
            ),
            "arg1_1": ir.TensorBox.create(
                ir.InputBuffer(
                    name="arg1_1",
                    layout=ir.FixedLayout(
                        torch.device(device),
                        torch.float32,
                        size=[s3],
                        stride=[1],
                    ),
                )
            ),
        }
        graph = mock.Mock(
            sizevars=SizeVarAllocator(ShapeEnv()),
            graph_inputs=graph_inputs,
            graph_input_names=list(graph_inputs),
            symbolic_input_sources={},
        )

        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()
        with V.set_graph_handler(graph):
            wrapper.codegen_inputs()
        code = wrapper.prefix.getvalue()
        self.assertNotIn("arg0_1_size_0 = arg0_1_size[0]", code)
        self.assertIn("s3 = arg1_1_size[0]", code)

        cpp_wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        cpp_wrapper.prefix = IndentedBuffer()
        cpp_wrapper.unbacked_symbol_decls = OrderedSet()
        with V.set_graph_handler(graph):
            cpp_wrapper.codegen_inputs()
        cpp_code = cpp_wrapper.prefix.getvalue()
        self.assertNotIn("int64_t arg0_1_size_0 = arg0_1_size[0];", cpp_code)
        self.assertIn("int64_t s3 = arg1_1_size[0];", cpp_code)

        s4 = sympy.Symbol("s4", integer=True)
        s5 = sympy.Symbol("s5", integer=True)
        graph_inputs = {
            "arg0_1": ir.TensorBox.create(
                ir.InputBuffer(
                    name="arg0_1",
                    layout=ir.FixedLayout(
                        torch.device(device),
                        torch.float32,
                        size=[s4 + s5, s4],
                        stride=[1, 1],
                    ),
                )
            ),
        }
        graph = mock.Mock(
            sizevars=SizeVarAllocator(ShapeEnv()),
            graph_inputs=graph_inputs,
            graph_input_names=list(graph_inputs),
            symbolic_input_sources={},
        )

        wrapper = PythonWrapperCodegen.__new__(PythonWrapperCodegen)
        wrapper.prefix = IndentedBuffer()
        with V.set_graph_handler(graph):
            wrapper.codegen_inputs()
        code = wrapper.prefix.getvalue()
        self.assertIn("s4 = arg0_1_size[1]", code)
        self.assertRegex(code, r"s5 = .*arg0_1_size_0")

        cpp_wrapper = CppWrapperCpu.__new__(CppWrapperCpu)
        cpp_wrapper.prefix = IndentedBuffer()
        cpp_wrapper.unbacked_symbol_decls = OrderedSet()
        with V.set_graph_handler(graph):
            cpp_wrapper.codegen_inputs()
        cpp_code = cpp_wrapper.prefix.getvalue()
        self.assertIn("int64_t s4 = arg0_1_size[1];", cpp_code)
        self.assertRegex(cpp_code, r"int64_t s5 = .*arg0_1_size_0")

    def test_stride_ordered_handles_reciprocal_in_divisibility_check(self, device):
        # [FUNCTION DOC] Builds an unbacked symbol with range constraints and a layout whose
        # strides contain floor divisions that reduce to reciprocal powers; asserts it is stride
        # ordered without using optimization hints. Then with a mock size variable manager that
        # cannot prove anything, asserts the stride comparison helper returns false, was called
        # once for a multiple-of check, and the guard passed has no negative power terms and
        # prints with a modulo operator.
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        shape_env = ShapeEnv()
        seq = shape_env.create_unbacked_symint().node.expr
        shape_env.constrain_symbol_range(seq, 4, 1024)
        half = FloorDiv(seq, 2)
        reduced = FloorDiv(half**2, half)
        left = 16 * reduced - 30
        right = 16 * (64 * half * reduced - 120 * half - 120 * reduced + 225)
        sizevars = SizeVarAllocator(shape_env)
        graph = mock.Mock(sizevars=sizevars)

        layout = ir.FixedLayout(
            torch.device(device),
            torch.float32,
            size=[seq, 2],
            stride=[left, right],
        )
        with V.set_graph_handler(graph):
            with mock.patch.object(
                sizevars,
                "optimization_hint",
                side_effect=AssertionError("unexpected optimization hint"),
            ):
                self.assertTrue(layout.is_stride_ordered([0, 1]))

        fallback_sizevars = mock.Mock()
        fallback_sizevars.guard_or_false.return_value = False
        fallback_sizevars.statically_known_multiple_of.return_value = False
        graph = mock.Mock(sizevars=fallback_sizevars)
        with V.set_graph_handler(graph):
            self.assertFalse(ir.Layout._stride_expr_ge_or_false(left, right))

        fallback_sizevars.statically_known_multiple_of.assert_called_once_with(
            left, right
        )
        guard = fallback_sizevars.guard_or_false.call_args.args[0]
        negative_powers = [
            power
            for power in guard.atoms(sympy.Pow)
            if power.args[1].is_negative is True
        ]
        self.assertEqual(negative_powers, [])
        self.assertIn("%", PythonPrinter().doprint(guard))

    @skipGPUIf(not HAS_GPU, "requires gpu and triton")
    @dynamo_config.patch({"capture_dynamic_output_shape_ops": True})
    @inductor_config.patch(
        {"max_complex_pointwise_cat_inputs": 1, "max_pointwise_cat_inputs": 1}
    )
    def test_cat_sympy_channels_last_contiguous(self, device):
        # [FUNCTION DOC] Skipped without a GPU. With low pointwise cat input limits, compiles
        # concatenation of two zeros tensors whose sizes involve an unbacked count, checking the
        # channels last contiguity check does not raise a data-dependent guard error. Compares
        # to eager.
        """torch.cat with symbolic shapes in shapes / strides should not raise
        GuardOnDataDependentSymNode when checking channels_last contiguity."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Builds two zeros tensors from the second input with
            # dimension one set to the nonzero count and the count plus one, and returns their
            # concatenation along dimension one.
            # nonzero produces an unbacked symint in dim 0
            nz = torch.nonzero(x)  # (u0, 1)
            u0 = nz.size(0)
            # All inputs are unrealized Pointwise (new_zeros), so
            # any_input_is_storage_and_layout is False, reaching the
            # meta["val"] channels_last check. Lower both pointwise cat
            # thresholds to force ConcatKernel path instead of pointwise_cat.
            inputs = [
                y.new_zeros(y.size(0), u0, y.size(2), y.size(3)),
                y.new_zeros(y.size(0), u0 + 1, y.size(2), y.size(3)),
            ]
            return torch.cat(inputs, dim=1)

        example_inputs = (
            torch.tensor([1.0, 0.0, 1.0, 1.0], device=device),
            torch.randn(2, 3, 4, 10, device=device),
        )
        actual = torch.compile(fn, fullgraph=True)(*example_inputs)
        expected = fn(*example_inputs)
        torch.testing.assert_close(actual, expected)


instantiate_device_type_tests(TestUnbackedSymints, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
