# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Tests that inductor's triton codegen emits (strided, ND) block pointers or
# tensor descriptors for pointwise, broadcast, reduction, welford and similar kernels, by
# compiling small functions, comparing results with eager, and checking the generated code
# (counts of block descriptors and kernels, tile dimensions, expected code
# text). Defines shared helpers and xfail decorators, a base test class, and the device-
# generic CommonTemplate mixin; later parts of the file instantiate it for CPU/GPU with the
# different descriptor modes. At import it skips on Windows CI and requires the filelock
# package to be importable.
import contextlib
import dataclasses
import functools
import importlib
import math
import unittest
from collections.abc import Callable
from typing import Any
from unittest import mock

import torch
import torch.utils._pytree as pytree
from torch._dynamo.debug_utils import InputReader
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.codegen.triton import FixedTritonConfig
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.runtime.runtime_utils import get_max_y_grid, is_power_of_2
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    decorateIf,
    instantiate_parametrized_tests,
    MI200_ARCH,
    NAVI_ARCH,
    parametrize,
    skipIfRocm,
    skipIfRocmArch,
    subtest,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    requires_gpu,
    skip_windows_ci,
    TRITON_HAS_CPU,
)


try:
    from . import test_torchinductor
except ImportError:
    import test_torchinductor


skip_windows_ci(__name__, __file__)

importlib.import_module("filelock")

max_block: int = TRITON_MAX_BLOCK["X"]


def _get_no_split_threshold() -> int:
    # [FUNCTION DOC] HELPER: Returns the reduction size threshold below which inductor does
    # not split an inner reduction. On a machine with CUDA it asks the inductor choices object
    # for the threshold using the current device properties, xnumel of one and the SM count;
    # otherwise it returns the constant 8192.
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        # These tests reduce the whole view to one output, so xnumel is 1.
        return V.choices._inner_reduction_no_split_threshold(
            props, xnumel=1, num_sm=props.multi_processor_count
        )
    return 8192


def tma_xfail(*args, cpu=False, extra_decorators=None):
    # [FUNCTION DOC] HELPER: Builds a parametrization subtest for the given argument values
    # that is marked as expected to fail with TMA tensor descriptors. The decorator list
    # always has the CUDA tensor descriptor xfail, adds the CPU one when the cpu flag is true,
    # and appends any extra decorators. Returns the subtest object.
    """Mark a subtest as expected to fail with TMA tensor descriptors.

    TMA restrictions (see Note: TMA API Restrictions in config.py):
      R1: innermost stride must be 1
      R2: outer strides must be 16-byte aligned
      R3: innermost dimension must load at least 16 bytes
    """
    decorators = [xfail_if_cuda_tensor_descriptor]
    if cpu:
        decorators.append(xfail_if_cpu_tensor_descriptor)
    if extra_decorators:
        decorators += list(extra_decorators)
    return subtest(arg_values=args, decorators=decorators)


# Config shortcuts
tiled_reduction_config = {
    "triton.prefer_nd_tiling": True,
    "triton.tile_reductions": True,
}


def xfail_if_cpu_tensor_descriptor(fn):
    # [FUNCTION DOC] HELPER: Decorator that marks the function with the attribute
    # _expected_failure_cpu_tensor_descriptor set to true and returns the same function.
    fn._expected_failure_cpu_tensor_descriptor = True
    return fn


def xfail_if_cuda_tensor_descriptor(fn):
    # [FUNCTION DOC] HELPER: Decorator that marks the function with the attribute
    # _expected_failure_cuda_tensor_descriptor set to true and returns the same function.
    fn._expected_failure_cuda_tensor_descriptor = True
    return fn


# These xfails are due to restrictions shared by the CPU and CUDA tensor descriptor paths.
def xfail_if_tensor_descriptor(fn):
    # [FUNCTION DOC] HELPER: Decorator for failures common to the CPU and CUDA tensor
    # descriptor paths; sets both expected-failure attributes on the function and returns it.
    fn._expected_failure_cpu_tensor_descriptor = True
    fn._expected_failure_cuda_tensor_descriptor = True
    return fn


class BlockDescriptorTestBase(InductorTestCase):
    # [CLASS DOC] Base test case providing helpers for block descriptor tests: building padded
    # non-contiguous tensors, asserting how many tile dimensions appear in generated code, and
    # compiling a function with inductor while checking numerics and code properties. The
    # class attribute block_descriptor_constructor_str is the text counted in generated code
    # to find block descriptors; subclasses override it for other descriptor styles.
    block_descriptor_constructor_str = "tl.make_block_ptr"

    def _discontiguous_tensor(
        self, view_size: tuple[int, ...], device: torch.device | str
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Makes a random float tensor on the device (a string is
        # converted to a device) whose size is double the requested view size in every
        # dimension, then returns a strided view of the requested size that keeps the full
        # tensor strides, so the view has gaps and is not contiguous.
        """
        Create a padded tensor of the given size.
        The strides correspond to a tensor that is twice as large in each dimension.
        """
        if isinstance(device, str):
            device = torch.device(device)
        full_size = tuple(2 * dim for dim in view_size)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())
        return view

    def _assert_pointwise_ndims(self, code, num_dims: int) -> None:
        # [FUNCTION DOC] HELPER: Asserts the code uses exactly the first num_dims of the
        # pointwise block names XBLOCK, YBLOCK, ZBLOCK, by delegating to the tiling dimension
        # check.
        pointwise_blocks = ["XBLOCK", "YBLOCK", "ZBLOCK"]
        return self._assert_tiling_ndims(code, pointwise_blocks, num_dims)

    def _assert_reduction_ndims(self, code, num_dims: int) -> None:
        # [FUNCTION DOC] HELPER: Asserts the code uses exactly the first num_dims of the
        # reduction block names R0_BLOCK and R1_BLOCK, by delegating to the tiling dimension
        # check.
        reduction_blocks = ["R0_BLOCK", "R1_BLOCK"]
        return self._assert_tiling_ndims(code, reduction_blocks, num_dims)

    def _assert_tiling_ndims(self, code, blocks: list[str], num_dims: int) -> None:
        # [FUNCTION DOC] HELPER: Asserts that each of the first num_dims names in the blocks
        # list appears in the code string and that none of the remaining names appears.
        for expected_block in blocks[:num_dims]:
            self.assertIn(expected_block, code)
        for unexpected_block in blocks[num_dims:]:
            self.assertNotIn(unexpected_block, code)

    def _get_lines_containing_substr(self, code: str, substr: str) -> str:
        # [FUNCTION DOC] HELPER: Returns the lines of the code string that contain the
        # substring, joined by newlines.
        return "\n".join(line for line in code.split("\n") if substr in line)

    def _run_and_compare(
        self: InductorTestCase,
        func: Callable[..., Any],
        *args,
        compile_kwargs: dict | None = None,
        expected_num_block_pointers: int | None = None,
        expected_num_programs: int = 1,
        expected_num_triton_kernels: int = 1,
        config_patches: dict | None = None,
        rtol: float | None = None,
        atol: float | None = None,
    ):
        # [FUNCTION DOC] HELPER: Compiles the function with the inductor backend (optional
        # compile kwargs, under the given config patches), runs it on the arguments and
        # fetches the generated code. Asserts each flattened output is close to eager output
        # (rtol/atol only when given), that the number of generated programs equals
        # expected_num_programs, that the count of jit kernel markers over all programs equals
        # expected_num_triton_kernels, and, if expected_num_block_pointers is not None, that
        # the count of block descriptor constructor text matches it. Returns the compiled
        # result and the list of generated code strings.
        """
        Runs the module through Inductor, comparing to eager reference.
        """
        if compile_kwargs is None:
            compile_kwargs = {}
        if config_patches is None:
            config_patches = {}

        def flatten_tensors(tensors):
            # [FUNCTION DOC] HELPER: Flattens a nested structure of tensors and returns the
            # list of leaves, dropping the tree spec.
            flat, spec = pytree.tree_flatten(tensors)
            return flat

        with config.patch(config_patches):
            compiled = torch.compile(func, backend="inductor", **compile_kwargs)
            result, code = run_and_get_code(compiled, *args)

        # Check numerical accuracy
        ref_tensors = flatten_tensors(func(*args))
        actual_tensors = flatten_tensors(result)
        for ref, actual in zip(ref_tensors, actual_tensors):
            # Don't clobber the default tolerance values
            tol = {
                t: v for t, v in {"rtol": rtol, "atol": atol}.items() if v is not None
            }
            self.assertTrue(torch.allclose(ref, actual, **tol))

        def count_code(substr: str, expected: int | None):
            # [FUNCTION DOC] HELPER: Sums occurrences of the substring over all generated
            # programs and, when expected is not None, asserts the total equals it.
            count = sum(prog.count(substr) for prog in code)
            if expected is not None:
                self.assertEqual(count, expected)

        # Check the code
        self.assertEqual(len(code), expected_num_programs)
        count_code("@triton.jit", expected_num_triton_kernels)
        count_code(self.block_descriptor_constructor_str, expected_num_block_pointers)

        return result, code


@instantiate_parametrized_tests
class CommonTemplate:
    # [CLASS DOC] Device-generic mixin (not a TestCase itself) with the block pointer tests,
    # meant to be combined with a base class that supplies the device attribute and descriptor
    # settings. The class itself is wrapped to expand parametrized tests.
    @parametrize(
        "expected_num_block_pointers,raises",
        [
            (3, False),  # This should pass
            (9, True),  # This should fail
        ],
    )
    def test_expected_num_block_pointers(
        self, expected_num_block_pointers: int, raises: bool
    ):
        # [FUNCTION DOC] Parametrized: passes with the true block pointer count of three and
        # must raise AssertionError when nine is expected. Verifies the harness check on the
        # number of block pointers by compiling an add of two random length-8 tensors.
        """
        Checks that the test harness verifies the number of block pointers correctly.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test; returns the sum of its two
            # arguments.
            return x + y

        device = torch.device(self.device)
        inputs = [torch.randn(8).to(device) for arg_idx in range(2)]

        # Expect failure for bad inputs
        with self.assertRaises(AssertionError) if raises else contextlib.nullcontext():
            # Expect 3 block pointers: 2 inputs 1 output
            self._run_and_compare(
                foo,
                *inputs,
                expected_num_block_pointers=expected_num_block_pointers,
            )

    @parametrize("prefer_nd_tiling", [False, True])
    @parametrize(
        "full_size,view_size,stride,offset,require_block_ptr",
        [
            ((64, 32, 32), (32, 16, 8), None, None, True),
            tma_xfail(
                (16, 8, 8, 8), (8, 8, 4, 2), None, None, True
            ),  # R3: inner dim 2*4B < 16B
            ((8, 8, 8, 8), (4, 4, 4, 4), None, None, True),
            ((8, 8), (4, 4), None, 10, True),  # Storage offset
            tma_xfail(
                (8, 8), (4, 4), (16, 2), None, True, cpu=True
            ),  # R1: inner stride 2 != 1
            ((8, 8), (4, 4), (1, 8), None, True),  # Transposed strides
            tma_xfail(
                (5, 9), (5, 8), None, None, True
            ),  # R2: outer stride 9*4B not 16B-aligned
            (
                (15, 9),
                (15, 3),
                None,
                None,
                False,
            ),  # Non-power-of-2 inner dims: non-block ptr
            ((1, 1, 1), (1, 1, 1), None, None, False),  # Scalar: non-block ptr
            subtest(
                arg_values=((2, 4 * max_block), (2, 3 * max_block), None, None, True),
                decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test"),
                ],
            ),  # Inner dim multiple of max_block
        ],
    )
    def test_pointwise(
        self,
        full_size: tuple[int, ...],
        view_size: tuple[int, ...],
        stride: tuple[int, ...] | None,
        offset: int | None,
        require_block_ptr: bool,
        prefer_nd_tiling: bool,
    ):
        # [FUNCTION DOC] Parametrized over tilings and shapes. Builds two strided views with
        # the given full size, view size, optional stride and storage offset, compiles an add
        # of them with the ND tiling option set, and, when a block pointer is required,
        # expects three block descriptors (two inputs and one output). Odd shapes with no
        # requirement only check numerics/no crash.
        """
        Test generating strided ND block pointers for a pointwise kernel.

        If require_block_ptr is True, the generated code must contain block
        pointers. However, ND block pointers are not supported for all shapes. So
        we also test some odd shapes with require_block_ptr set to False, to ensure that
        block pointer analysis does not break these cases.
        """

        def get_input() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Creates a random tensor of the full size on the test
            # device and returns a strided view of the view size, using the given stride (or
            # the original stride) and storage offset.
            device = torch.device(self.device)
            full = torch.randn(full_size).to(device)

            # Use the original tensor's stride by default
            view_stride = full.stride() if stride is None else stride

            return torch.as_strided(full, view_size, view_stride, storage_offset=offset)

        args = [get_input() for arg_idx in range(2)]

        # Expect 3 block pointers: 2 inputs 1 output
        self._run_and_compare(
            torch.add,
            *args,
            expected_num_block_pointers=3 if require_block_ptr else None,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    @parametrize("prefer_nd_tiling", [False, True])
    @parametrize(
        "x_size,y_size",
        [
            # CUDA R3: y inner dim 1*4B < 16B.
            # CPU: broadcasted y inner stride is 0.
            tma_xfail((8, 8), (8, 1), cpu=True),
            ((8, 8), (1, 8)),
            # CUDA R3: y inner dim 1*4B < 16B.
            # CPU: broadcasted y inner stride is 0.
            tma_xfail((4, 1, 4), (1, 4, 1), cpu=True),
            ((1, 1, 1, 4), (4, 4, 4, 4)),  # Unmatched dims for first operand.
        ],
    )
    def test_broadcast(
        self, x_size: tuple[int, ...], y_size: tuple[int, ...], prefer_nd_tiling: bool
    ):
        # [FUNCTION DOC] Parametrized. Builds two discontiguous inputs of different shapes
        # with at least one size-one dimension, compiles a function that adds the first plus
        # one to the second times two, and expects three block descriptors; asserts the shapes
        # differ and some size is one.
        """
        Test that we can generate strided block pointers when inputs have different
        shapes, and they are broadcast together.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test; returns x plus one plus y times two.
            a = x + 1
            b = y * 2
            return a + b

        x, y = (
            self._discontiguous_tensor(size, self.device) for size in (x_size, y_size)
        )

        # Check that input sizes are not the same
        self.assertNotEqual(x.shape, y.shape)

        # Check that at least one dimension is a singleton
        all_dims = x.shape + y.shape
        self.assertIn(1, all_dims)

        # Expect 3 block pointers: 2 inputs one output
        self._run_and_compare(
            foo,
            x,
            y,
            expected_num_block_pointers=3,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    def test_broadcast_with_singleton_dims(self):
        # [FUNCTION DOC] Regression test for a minified graph from a language model where
        # tensors have both zero strides and singleton dimensions, which used to make the
        # broadcasting dimensions list length disagree with the broadcast shape. Rebuilds the
        # inputs with an input reader and expects four block descriptors; explicit tolerances
        # (atol 1e-7, rtol 1e-5) on XPU/HIP.
        # This tests the case when the input / output contains both zero strides
        # and singleton dimensions. In this case the broadcasting dimensions
        # generated for the descriptor need to ignore dimensions that have zero
        # strides with size 1

        # This is a minified repro based on HuggingFaceTB/SmolLM2-135M
        # original issue:
        # store index=x2 + 192*y0 + 64*y1
        # matched block params = BlockParameters(
        #     shape=[3, 4, 1, 1, 64],
        #     block_shape=[((YBLOCK + 3)//4), Min(4, YBLOCK), 1, 1, XBLOCK],
        #     strides=[64, 192, 0, 0, 1],
        #     offsets=[(yoffset//4), ModularIndexing(yoffset, 1, 4), 0, 0, xoffset]
        # )
        # broadcasting_dims=[False, False, True, True, False]
        # broadcast_shape=[((YBLOCK + 3)//4), Min(4, YBLOCK), XBLOCK]
        # error, len(broadcasting_dims) != broadcast_shape
        def forward(expand_4, permute_4, mul_7):
            # [FUNCTION DOC] HELPER: Graph-style function under test. Clones the expanded
            # input contiguous, views it as 1x4x64, takes its cosine, scales by 1.0, adds a
            # dimension, multiplies by the permuted input, adds the third input, inserts
            # another dimension and returns a one-tuple.
            clone = torch.ops.aten.clone.default(
                expand_4, memory_format=torch.contiguous_format
            )
            expand_4 = None
            view_4 = torch.ops.aten.view.default(clone, [1, 4, 64])
            clone = None
            cos = torch.ops.aten.cos.default(view_4)
            view_4 = None
            mul = torch.ops.aten.mul.Tensor(cos, 1.0)
            cos = None
            unsqueeze_4 = torch.ops.aten.unsqueeze.default(mul, 1)
            mul = None
            mul_6 = torch.ops.aten.mul.Tensor(permute_4, unsqueeze_4)
            permute_4 = unsqueeze_4 = None
            add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7)
            mul_6 = mul_7 = None
            unsqueeze_6 = torch.ops.aten.unsqueeze.default(add_3, 2)
            add_3 = None
            return (unsqueeze_6,)

        def load_args(reader):
            # [FUNCTION DOC] HELPER: Input loader. Uses the given reader to create three
            # tensors on the test device: an expanded one with a zero stride (1x4x2x32), a
            # permuted one (1x3x4x64) and a contiguous 1x3x4x64 one.
            buf0 = reader.storage(storage_hash=None, nbytes=512, device=self.device)
            reader.tensor(buf0, (1, 4, 2, 32), (128, 1, 0, 4), is_leaf=True)  # expand_4
            buf1 = reader.storage(storage_hash=None, nbytes=3072, device=self.device)
            reader.tensor(
                buf1, (1, 3, 4, 64), (768, 64, 192, 1), is_leaf=True
            )  # permute_4
            buf2 = reader.storage(storage_hash=None, nbytes=3072, device=self.device)
            reader.tensor(buf2, (1, 3, 4, 64), is_leaf=True)  # mul_7

        load_args._version = 0

        input_reader = InputReader()
        load_args(input_reader)
        args = input_reader.args
        if self.device == "xpu" or torch.version.hip is not None:
            atol = 1e-7
            rtol = 1e-5
        else:
            atol = None
            rtol = None

        self._run_and_compare(
            forward,
            *args,
            expected_num_block_pointers=4,
            atol=atol,
            rtol=rtol,
        )

    @parametrize(
        "x_size,y_size",
        [
            ((32, 1), (32, 32)),
            ((1, 8), (8, 8)),
            # ((4, 1, 3), (4, 5, 3)), # TODO: T207754224
            ((4, 1, 3), (4, 4, 3)),
            ((1, 5, 5), (5, 5, 5)),
            ((5, 5, 1), (5, 5, 5)),
            ((5, 1, 1), (5, 5, 5)),
            ((1, 1, 5), (5, 5, 5)),
            ((1, 5, 1), (5, 5, 5)),
            ((7, 1, 1, 4), (7, 3, 4, 4)),
            ((5, 6, 1, 1), (5, 6, 4, 3)),
        ],
    )
    def test_expand_broadcast(self, x_size: tuple[int, ...], y_size: tuple[int, ...]):
        # [FUNCTION DOC] Parametrized over input and expanded sizes. Checks the sizes differ
        # and are a valid broadcast (same rank, equal wherever the input size is not one),
        # then compiles an expand followed by clone and checks numerics and the default single
        # program and kernel.
        """
        When the load and store have different shapes, we should use broadcast.
        """

        def foo(x, y_size):
            # [FUNCTION DOC] HELPER: Function under test; expands x to the given size and
            # clones it.
            return x.expand(y_size).clone()

        def get_input(size: tuple[int, ...]) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Creates a random tensor of the size on the test device
            # and returns a strided view with the same size and strides.
            device = torch.device(self.device)
            full = torch.randn(size).to(device)
            view = torch.as_strided(full, size, full.stride())
            return view

        x = get_input(x_size)
        y = y_size

        # Check that input sizes are not the same
        self.assertNotEqual(x_size, y_size)

        # Check that is valid broadcast
        self.assertEqual(len(x_size), len(y_size))
        for i, j in zip(x_size, y_size):
            if i != 1:
                self.assertEqual(i, j)

        result, (triton_code,) = self._run_and_compare(foo, x, y)

    @xfail_if_tensor_descriptor
    @parametrize("prefer_nd_tiling", [False, True])
    @config.patch("triton.skip_l1_cache", False)
    def test_pointwise_broadcast_nonzero_strides(self, prefer_nd_tiling: bool):
        # [FUNCTION DOC] Compiles an add of an 8x8 tensor and an 8x1 column view, expecting
        # three block descriptors with the l1-cache skip option off, for both tilings. Then
        # compares the generated load and store lines against golden strings: with ND tiling
        # the column is loaded as a one-dimensional block pointer and indexed to a column,
        # without it the load is reshaped and broadcast; the store uses an explicit broadcast
        # rather than zero strides. Marked as expected to fail with tensor descriptors.
        """
        Test that we emit tl.broadcast_to instead of using strides of 0.
        """

        full_shape = (8, 8)
        col_shape = (full_shape[1], 1)
        device = torch.device(self.device)
        full = torch.randn(full_shape).to(device)
        col = torch.as_strided(full, col_shape, full.stride())

        # Expect 3 block pointers: 2 inputs one output
        result, (triton_code,) = self._run_and_compare(
            torch.add,
            full,
            col,
            expected_num_block_pointers=3,
            config_patches={
                "triton.prefer_nd_tiling": prefer_nd_tiling,
            },
        )

        # Check the code for broadcasts.
        # We shouldn't see any strides of 0.
        load_lines, store_lines = tuple(
            self._get_lines_containing_substr(triton_code, substr)
            for substr in ("tl.load", "tl.store")
        )
        if prefer_nd_tiling:
            self.assertExpectedInline(
                load_lines,
                """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[8, 8], strides=[8, 1], block_shape=[YBLOCK, XBLOCK], order=[1, 0], offsets=[yoffset, xoffset]), boundary_check=[0, 1])
    tmp1 = tl.load(tl.make_block_ptr(in_ptr1, shape=[8], strides=[8], block_shape=[YBLOCK], order=[0], offsets=[yoffset]), boundary_check=[0], eviction_policy='evict_last')[:, None]""",
            )
            self.assertExpectedInline(
                store_lines,
                """    tl.store(tl.make_block_ptr(out_ptr0, shape=[8, 8], strides=[8, 1], block_shape=[YBLOCK, XBLOCK], order=[1, 0], offsets=[yoffset, xoffset]), tl.broadcast_to(tmp2, [YBLOCK, XBLOCK]).to(tl.float32), boundary_check=[0, 1])""",
            )
        else:
            self.assertExpectedInline(
                load_lines,
                """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[64], strides=[1], block_shape=[XBLOCK], order=[0], offsets=[xoffset]), boundary_check=[0])
    tmp1 = tl.reshape(tl.broadcast_to(tl.load(tl.make_block_ptr(in_ptr1, shape=[8], strides=[8], block_shape=[(7 + XBLOCK) // 8], order=[0], offsets=[xoffset // 8]), boundary_check=[0], eviction_policy='evict_last')[:, None, None], [(7 + XBLOCK) // 8, ((1) * ((1) <= ((7 + XBLOCK) // 8)) + ((7 + XBLOCK) // 8) * (((7 + XBLOCK) // 8) < (1))), ((8) * ((8) <= (XBLOCK)) + (XBLOCK) * ((XBLOCK) < (8)))]), [XBLOCK])""",
            )
            self.assertExpectedInline(
                store_lines,
                """    tl.store(tl.make_block_ptr(out_ptr0, shape=[64], strides=[1], block_shape=[XBLOCK], order=[0], offsets=[xoffset]), tl.broadcast_to(tmp2, [XBLOCK]).to(tl.float32), boundary_check=[0])""",
            )

    @parametrize("prefer_nd_tiling", [False, True])
    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((4, 4), 1, 1),
            ((4, 4, 4), 1, 1),
            ((8, 8, 8), 1, 1),
            ((15, 15), None, 1),  # Non-power of 2
            tma_xfail(  # R3: inner dim 2*4B < 16B. Multiple of max block, uses loops.
                (3 * max_block, 2),
                3,
                2,
                extra_decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
                ],
            ),
            ((2, 3 * max_block), 2, 2),  # Multiple of max block. Uses loops.
            # 2-kernel split xfails TMA; on SM100+ uses single persistent kernel and passes.
            subtest(
                arg_values=((128, 128), 3, 2),
                decorators=[] if SM100OrLater else [xfail_if_cuda_tensor_descriptor],
            ),
        ],
    )
    def test_reduction(
        self,
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
        prefer_nd_tiling: bool,
    ):
        # [FUNCTION DOC] Parametrized sum reduction over discontiguous views. On HIP some
        # sizes are enlarged. When the expected two-kernel split would fit under the no-split
        # threshold it expects one kernel and two fewer block pointers. Skips one long CPU
        # case (it can crash the process pool). Expects the given counts of block pointers and
        # kernels from the compiled sum.
        """
        Tests a reduction kernel.
        """
        if view_size == (2, 3 * max_block) and torch.version.hip is not None:
            view_size = (4, 6 * max_block)

        if view_size == (128, 128) and torch.version.hip is not None:
            view_size = (256, 256)

        # On SM100+, larger max_block means reductions that previously needed 2
        # kernels now fit in a single persistent kernel with fewer block pointers.
        reduction_numel = math.prod(view_size)
        if num_triton_kernels == 2 and reduction_numel <= _get_no_split_threshold():
            num_triton_kernels = 1
            if num_block_pointers is not None:
                num_block_pointers = max(num_block_pointers - 2, 0)

        if self.device == "cpu" and all(
            # Multiple of max block. Uses loops.
            [
                view_size == (3 * max_block, 2),
                num_block_pointers == 3,
                num_triton_kernels == 2,
                prefer_nd_tiling is False,
            ]
        ):
            raise unittest.SkipTest(
                "Long test and raises BrokenProcessPool Error if triton CPU"
            )

        device = torch.device(self.device)

        view = self._discontiguous_tensor(view_size, self.device)

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = self._run_and_compare(
            torch.sum,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
            config_patches={"triton.prefer_nd_tiling": prefer_nd_tiling},
        )

    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels",
        [
            ((8, 8), 2, 1),  # No loops. Should be supported.
            (
                (128, 128),
                None,
                None,
            ),  # Looped reduction. Block pointers not yet supported.
        ],
    )
    def test_mixed_pointwise_reduction(
        self,
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
    ):
        # [FUNCTION DOC] Parametrized. Compiles a sum of the sum of two discontiguous inputs,
        # expecting the given number of block pointers and kernels (None means unchecked, used
        # for the large looped case that does not yet use block pointers).
        """
        Tests mixing pointwise with reduction ops.
        """

        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function under test; returns the total sum of x plus y.
            return torch.sum(x + y)

        inputs = [
            self._discontiguous_tensor(view_size, self.device) for input_idx in range(2)
        ]

        # Expect 2 block pointers: inputs
        result, (code,) = self._run_and_compare(
            foo,
            *inputs,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
        )

    @xfail_if_cuda_tensor_descriptor
    def test_multiple_max_block_non_power_of_2(self):
        # [FUNCTION DOC] Checks that a dimension of size n times the max block size, with n
        # not a power of two, still gets block descriptors. Uses a view of shape (3 * max
        # block, 2) cut from a tensor with last dim 3, asserts the view has a non power of two
        # dimension and more than one dimension larger than one, compiles subtracting one and
        # expects two block descriptors.
        """
        Check that we support dims of size n * MAX_BLOCK, where n is any positive integer, not
        necessarily a power of 2.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns x minus one.
            return x - 1

        device = torch.device(self.device)
        full_size = (3 * max_block, 3)
        view_size = (3 * max_block, 2)
        full = torch.randn(full_size).to(device)
        view = torch.as_strided(full, view_size, full.stride())

        # Check that we're using dims that aren't all powers of 2
        have_np2_dim = not all(is_power_of_2(dim) for dim in view_size)
        self.assertTrue(have_np2_dim)

        # Check that we need more than one stride to represent the tensor
        nontrivial_dims = [dim for dim in view_size if dim > 1]
        self.assertTrue(len(nontrivial_dims) > 1)

        # Expect 2 block pointers: input and output
        self._run_and_compare(foo, view, expected_num_block_pointers=2)

    @parametrize(
        "nd_tiling,num_block_pointers",
        [
            subtest(
                (True, 2), decorators=[xfail_if_cuda_tensor_descriptor]
            ),  # With tiling, the index is affine.
            (False, 1),  # We can't infer that the load is a power of 2.
        ],
    )
    def test_dynamic_shapes_pointwise(self, nd_tiling: bool, num_block_pointers: int):
        # [FUNCTION DOC] Parametrized by whether ND tiling is on. Compiles a division of a 4x4
        # discontiguous view by itself with dynamic shapes and expects two block pointers with
        # tiling (index is affine) and one without.
        """
        Test a pointwise kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        self._run_and_compare(
            torch.div,
            view,
            view,
            expected_num_block_pointers=num_block_pointers,
            config_patches={"triton.prefer_nd_tiling": nd_tiling},
            compile_kwargs={"dynamic": True},
        )

    @parametrize(
        "with_tiling,num_block_pointers",
        [
            subtest(
                (True, 1), decorators=[xfail_if_cuda_tensor_descriptor]
            ),  # With tiling, the index is affine.
            (False, 0),  # We can't infer that the load is a power of 2.
        ],
    )
    def test_dynamic_shapes_reduction(self, with_tiling: bool, num_block_pointers: int):
        # [FUNCTION DOC] Parametrized by whether ND tiling and tiled reductions are on.
        # Compiles a product reduction of a 4x4 discontiguous view with dynamic shapes;
        # expects one block pointer with tiling and none without.
        """
        Test a reduction kernel with dynamic shapes.
        """

        view_size = (4, 4)
        view = self._discontiguous_tensor(view_size, self.device)

        self._run_and_compare(
            torch.prod,
            view,
            expected_num_block_pointers=num_block_pointers,
            config_patches={
                "triton.prefer_nd_tiling": with_tiling,
                "triton.tile_reductions": with_tiling,
            },
            compile_kwargs={"dynamic": True},
        )

    @unittest.skip(reason="Dynamo tracing error")
    def test_dynamic_shapes_pointwise_multiple_max_block(self):
        # [FUNCTION DOC] Skipped (dynamo tracing error). Intended to check that with dynamic
        # shapes that are known multiples of the max block size, a block pointer can be
        # generated. KNOWN ISSUE: the body calls the comparison helper without passing the
        # function foo, so the test would not work even if unskipped.
        """
        Test dynamic shapes, where we know the shape is a multiple of the max block
        size. We should be able to generate a block pointer for this case.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Tiles x to dynamic multiples of the max block size, takes
            # a strided view that drops part of the last dimension and returns the view added
            # to itself.
            tile_dims = (3 * max_block * x.shape[0], 3 * x.shape[1])
            view_size = (3 * max_block * x.shape[0], 2 * x.shape[1])
            full = x.tile(tile_dims)
            view = torch.as_strided(full, view_size, full.stride())
            return view + view

        device = torch.device(self.device)
        x_size = (1, 1)
        x = torch.randn(x_size).to(device)

        # Expect 2 block pointers: input and output
        self._run_and_compare(
            x, compile_kwargs={"dynamic": True}, expected_num_block_pointers=2
        )

    @decorateIf(
        xfail_if_cuda_tensor_descriptor,
        lambda param_kwargs: not (
            param_kwargs["num_block_pointers"] == 3 and param_kwargs["num_tiles"] == 1
        ),
    )
    @parametrize(
        "full_size,view_size,num_block_pointers,num_tiles",
        [
            (
                (32, 32),
                (16, 32),
                3,
                1,
            ),  # Contiguous 2D tensor. Does not require tiling.
            ((5, 9), (3, 7), 3, 2),  # 2D tensor with 1 discontiguous dim.
            ((11, 13, 7), (9, 13, 5), 3, 2),  # 3D tensor with 1 discontiguous dim (2).
            subtest(
                arg_values=(
                    (3, 11, 13, 7),
                    (2, 9, 13, 7),
                    3,
                    2,
                ),
                decorators=[
                    test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
                ],
            ),  # 4D tensor with 1 discontiguous dim (1).
            (
                (3, 11, 13, 7),
                (2, 11, 9, 7),
                3,
                2,
            ),
            # 4D tensor with 1 discontiguous dim (2).
            (
                (5, 5, 5, 5, 5),
                (3, 3, 5, 3, 5),
                1,
                2,
            ),  # 5D tensor with 2 discontiguous dims (3, 1). Block pointers unexpected.
        ],
    )
    def test_nd_tiling_odd_shapes_pointwise(
        self,
        full_size: tuple[int, ...],
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_tiles: int,
    ):
        # [FUNCTION DOC] Parametrized over shapes with ND tiling enabled. Compiles an add of
        # two strided views and expects the given number of block pointers, then asserts that,
        # in every program, exactly the first num_tiles of XBLOCK, YBLOCK, ZBLOCK appear.
        # Marked expected to fail with CUDA tensor descriptors except the contiguous three
        # block pointer one-tile case.
        """
        Test odd shapes with ND tiling enabled.
        Uses a pointwise op.
        """

        def get_input() -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Creates a random tensor of the full size and returns a
            # strided view of the view size with the original strides.
            device = torch.device(self.device)
            full = torch.randn(full_size).to(device)
            return torch.as_strided(full, view_size, full.stride())

        args = [get_input() for arg_idx in range(2)]

        # Expect up to 3 block pointers: 2 inputs 1 output.
        result, code = self._run_and_compare(
            torch.add,
            *args,
            expected_num_block_pointers=num_block_pointers,
            config_patches={
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check the code for the expected tiling.
        all_tiles = ("XBLOCK", "YBLOCK", "ZBLOCK")
        expected_tiles = set(all_tiles[:num_tiles])
        for tile_name in all_tiles:
            for program in code:
                if tile_name in expected_tiles:
                    self.assertIn(tile_name, program)
                else:
                    self.assertNotIn(tile_name, program)

    @xfail_if_cuda_tensor_descriptor
    @parametrize(
        "view_size,num_block_pointers,num_triton_kernels,reduction_op",
        [
            ((15, 15), 1, 1, torch.sum),  # Non-power-of 2 shapes.
            ((129, 129), 3, 2, torch.sum),  # Large size, with loops.
            ((3, 3), 1, 1, torch.argmax),
            ((129, 129), 1, 1, torch.argmax),
            ((5, 5), 1, 1, torch.var_mean),  # Reduction + pointwise fusion.
        ],
    )
    @skipIfRocmArch(MI200_ARCH + NAVI_ARCH)
    def test_2d_reduction_odd_shapes(
        self,
        view_size: tuple[int, ...],
        num_block_pointers: int,
        num_triton_kernels: int,
        reduction_op: Callable,
    ):
        # [FUNCTION DOC] Parametrized over view sizes and reduction ops (sum, argmax,
        # var_mean) with tiled reductions. Adjusts expectations for HIP sizes and for the
        # single persistent kernel case. On HIP the block pointer count is checked manually to
        # be within one below the expected count and not above. Always asserts the generated
        # code uses two reduction dimensions.
        """
        Tests 2D reduction kernels. These arise from "odd" shapes which are not
        expressible with a 1D block pointer.
        """
        if reduction_op == torch.sum and torch.version.hip is not None:
            view_size = (513, 513) if view_size == (129, 129) else view_size
        view = self._discontiguous_tensor(view_size, self.device)

        # On SM100+, these reductions fit in a single persistent kernel.
        reduction_numel = math.prod(view_size)
        if num_triton_kernels == 2 and reduction_numel <= _get_no_split_threshold():
            num_triton_kernels = 1
            num_block_pointers = 1

        # HIP: Backend scheduling / fusion differences (e.g., Navi vs MI*)
        # may result in off-by-one differences in the number of block pointers.
        # Allow a small tolerance here to avoid backend-specific flakiness.
        # We expect num_block_pointers to decrease by at most 1, and not increase.
        # The range created here is checked below.
        if torch.version.hip:
            min_num_block_pointers = max(
                num_block_pointers - 1, 1
            )  # Expected at least one block descriptor for the input
            max_num_block_pointers = (
                num_block_pointers  # We don't expect num_block_pointers to increase.
            )
            # Disable strict checking in _run_and_compare; we assert bounds manually below.
            num_block_pointers = None

        # Expect at least 1 block pointer for the input.
        # Add 2 more if we generate 2 kernels.
        result, (code,) = self._run_and_compare(
            reduction_op,
            view,
            expected_num_block_pointers=num_block_pointers,
            expected_num_triton_kernels=num_triton_kernels,
            config_patches=tiled_reduction_config,
        )

        # HIP: Check the number of block pointers manually.
        if torch.version.hip:
            block_pointer_count = code.count(self.block_descriptor_constructor_str)
            self.assertGreaterEqual(
                block_pointer_count,
                min_num_block_pointers,
                lambda msg: f"{msg}\nToo few block descriptors emitted: {block_pointer_count}",
            )
            self.assertLessEqual(
                block_pointer_count,
                max_num_block_pointers,
                lambda msg: f"{msg}\nToo many block descriptors emitted: {block_pointer_count}",
            )

        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "size,expected_num_block_pointers,expected_num_triton_kernels,expect_fallback",
        [
            ((8, 8), 1, 1, True),  # Persistent Welford fallback
            subtest(
                ((128, 128), 7, 2, False),
                decorators=[] if SM100OrLater else [xfail_if_cuda_tensor_descriptor],
            ),  # Looped Welford reduction
        ],
    )
    def test_2d_welford_reduction(
        self,
        size: tuple[int, ...],
        expected_num_block_pointers: int,
        expected_num_triton_kernels: int,
        expect_fallback: bool,
    ):
        # [FUNCTION DOC] Parametrized welford test with tiled reductions on var_mean of a
        # discontiguous view. Adjusts size on HIP and for the single persistent kernel case,
        # expects the given block pointer and kernel counts, asserts the word welford appears
        # in the code only when no fallback is expected, and that two reduction dimensions are
        # used.
        """
        Tests a 2D welford reduction.

        NB: the input size should be "nice" in the sense that it's a multiple of the
        number of processors. Otherwise, we will get more complex indexing that
        doesn't generate a block pointer. Since tiling welford reductions depends on
        the block pointer analysis, those cases would fall back to 1D.
        """
        if torch.version.hip is not None and expected_num_triton_kernels == 2:
            size = (256, 256)

        # On SM100+, these reductions fit in a single persistent kernel.
        reduction_numel = math.prod(size)
        if (
            expected_num_triton_kernels == 2
            and reduction_numel <= _get_no_split_threshold()
        ):
            expected_num_triton_kernels = 1
            expected_num_block_pointers = 1

        view = self._discontiguous_tensor(size, self.device)

        # We expect many block pointers for this one.
        result, (code,) = self._run_and_compare(
            torch.var_mean,
            view,
            expected_num_block_pointers=expected_num_block_pointers,
            expected_num_triton_kernels=expected_num_triton_kernels,
            config_patches=tiled_reduction_config,
        )

        # Check for a Welford reduction.
        self.assertEqual("welford" in code, not expect_fallback)

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    @test_torchinductor.skip_if_triton_cpu("Triton CPU: slow test")
    def test_welford_non_block_pointer(
        self,
    ):
        # [FUNCTION DOC] Skipped on Triton CPU as slow. Runs var_mean on a 259x311
        # discontiguous view whose size defeats block pointer analysis, with ND tiling.
        # Expects six block pointers and two kernels, or zero pointers and one kernel when
        # cooperative reductions are configured or the view is small enough for one persistent
        # kernel. Asserts the code mentions welford and uses only one reduction dimension.
        """
        Tests a welford reduction where block pointer analysis fails.
        The main loop will be a 1D reduction, instead of 2D.
        """
        # Use a "bad" size that's not evenly divisible by the launch grid.
        # This won't decompose into a block pointer.
        view = self._discontiguous_tensor((259, 311), self.device)

        # We expect many block pointers for this one.
        cooperative_reductions = (
            config.triton.cooperative_reductions
            or config.triton.force_cooperative_reductions
        )
        expected_num_block_pointers = 0 if cooperative_reductions else 6
        expected_num_triton_kernels = 1 if cooperative_reductions else 2

        # On SM100+, these reductions fit in a single persistent kernel.
        if (
            expected_num_triton_kernels == 2
            and view.numel() <= _get_no_split_threshold()
        ):
            expected_num_triton_kernels = 1
            expected_num_block_pointers = 0

        result, (code,) = self._run_and_compare(
            torch.var_mean,
            view,
            expected_num_block_pointers=expected_num_block_pointers,
            expected_num_triton_kernels=expected_num_triton_kernels,
            config_patches={"triton.prefer_nd_tiling": True},
        )

        # Check for a Welford reduction.
        self.assertIn("welford", code)

        # Check for a single reduction dimension.
        self._assert_reduction_ndims(code, 1)

    def test_reduction_multiple_discontiguous_dims(self):
        # [FUNCTION DOC] Sums a 3x7x11 discontiguous view with tiled reductions; expects zero
        # block pointers, one kernel, and two reduction dimensions in the code, because too
        # many discontiguous dimensions cannot be tiled into block pointers.
        """
        Test reducing a tensor with more than one discontiguous dimension. This case
        won't generate a block pointer, since we don'allow enough tiling dimensions.
        """
        # Use odd shapes to frustrate block pointer analysis.
        view = self._discontiguous_tensor((3, 7, 11), self.device)

        result, (code,) = self._run_and_compare(
            torch.sum,
            view,
            expected_num_block_pointers=0,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    @test_torchinductor.skip_if_triton_cpu  # Illegal instruction  File; cannot xfail because it crashes process
    def test_2d_reduction_multi_kernel(self):
        # [FUNCTION DOC] Skipped on Triton CPU because the process crashes. Compiles a softmax
        # over the flattened trailing dims of a discontiguous 2x4x1024 view with multi kernel
        # mode and tiled reductions on; expects five block pointers and two kernels, that the
        # code mentions multi kernel, and that two reduction dimensions are used.
        """
        Test a 2D reduction in multi kernel mode.
        """
        view = self._discontiguous_tensor((2, 4, 1024), self.device)

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; reshapes x to two dimensions keeping
            # dim zero and returns the softmax over the last dimension.
            """
            Reshape to 2D and take the softmax of all trailing dims.
            """
            x = x.reshape(x.shape[0], -1)
            return torch.softmax(x, -1)

        result, (code,) = self._run_and_compare(
            foo,
            view,
            expected_num_block_pointers=5,
            expected_num_triton_kernels=2,
            config_patches={
                "triton.multi_kernel": True,
                **tiled_reduction_config,
            },
        )

        # Check for multi kernel mode.
        self.assertIn("multi_kernel", code)

        # Check for 2 reduction dimensions.
        self._assert_reduction_ndims(code, 2)

    @xfail_if_cuda_tensor_descriptor
    def test_fused_2d_reduction(
        self,
    ):
        # [FUNCTION DOC] Tiled reductions: compiles the sum of a tensor plus its argmax over a
        # 5x7 discontiguous view and expects one block pointer, one kernel, and two reduction
        # dimensions in the code. Marked as expected to fail with CUDA tensor descriptors.
        """
        Tests fusing multiple reductions on the same input, with 2D tiling.
        """

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; returns the total sum of x plus the
            # argmax.
            return torch.sum(x) + torch.argmax(x)

        view_size = (5, 7)
        view = self._discontiguous_tensor(view_size, self.device)

        # Expect at least 1 block pointer for the input.
        result, (code,) = self._run_and_compare(
            foo,
            view,
            expected_num_block_pointers=1,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @skipIfRocm(msg="https://github.com/pytorch/pytorch/issues/158328")
    @parametrize("reduction_op", [torch.sum, torch.argmax])
    def test_2d_reductions_mixed_indexing(
        self,
        reduction_op: Callable,
    ):
        # [FUNCTION DOC] Parametrized over sum and argmax, skipped on ROCm. Reduces one
        # discontiguous and one contiguous 5x7 input in a single function with tiled
        # reductions; no kernel or pointer counts are checked, only numerics and that two
        # reduction dimensions appear in the code.
        """
        Tests a program with multiple reductions using different strides.
        These might not be fused.
        """

        def foo(*args):
            # [FUNCTION DOC] HELPER: Function under test; applies the reduction op to every
            # argument and returns the python sum of the results.
            return sum(reduction_op(arg) for arg in args)

        view_size = (5, 7)
        arg0 = self._discontiguous_tensor(view_size, self.device)
        arg1 = torch.randn(view_size)

        # No guarantees on the number of kernels or pointers.
        result, (code,) = self._run_and_compare(
            foo,
            arg0,
            arg1,
            config_patches=tiled_reduction_config,
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2)

    @parametrize(
        "tile_reductions",
        [False, subtest(True, decorators=[xfail_if_cuda_tensor_descriptor])],
    )
    def test_enable_tiled_reductions(self, tile_reductions: bool):
        # [FUNCTION DOC] Parametrized over whether reductions are tiled. Compiles a sum of a
        # 9x11 discontiguous view with ND tiling; expects one block pointer when tiled and
        # none otherwise, one kernel, and two reduction dimensions in the code when tiled (one
        # when not).
        """
        Tests enabling and disabling tiled reductions.
        """
        view = self._discontiguous_tensor((9, 11), self.device)

        # If tiled, we expect 1 block pointer for the input.
        result, (code,) = self._run_and_compare(
            torch.sum,
            view,
            expected_num_block_pointers=1 if tile_reductions else 0,
            expected_num_triton_kernels=1,
            config_patches={
                "triton.prefer_nd_tiling": True,
                "triton.tile_reductions": tile_reductions,
            },
        )

        # Check the code for multiple Rn_BLOCK's
        self._assert_reduction_ndims(code, 2 if tile_reductions else 1)

    # FIXME: fails for Triton CPU. Tiling does not contain YBLOCK.
    @test_torchinductor.xfail_if_triton_cpu
    @xfail_if_tensor_descriptor
    def test_reduction_padded_output_tiling(self):
        # [FUNCTION DOC] Regression-style test of a reduction over the last dimension of a
        # 9x11x2 random tensor with padded outputs (alignment 32 bytes, stride threshold
        # zero), unrolling limit one and tiled reductions. Expects two block pointers (input
        # and output), one kernel, two pointwise and one reduction dimension. Expected to fail
        # on Triton CPU and with tensor descriptors.
        """
        Test a [Y, X, R0] reduction with tiled output dimensions.
        The key to elicit this test case is a padded output tensor.
        """
        x = torch.randn((9, 11, 2), device=self.device)

        # We expect block pointers for the input and output.
        result, (code,) = self._run_and_compare(
            functools.partial(torch.amax, dim=-1),
            x,
            expected_num_block_pointers=2,
            expected_num_triton_kernels=1,
            config_patches={
                "pad_outputs": True,
                "padding_alignment_bytes": 32,
                "padding_stride_threshold": 0,
                "unroll_reductions_threshold": 1,
                **tiled_reduction_config,
            },
        )

        # Check the code for multiple output dims.
        self._assert_pointwise_ndims(code, 2)
        self._assert_reduction_ndims(code, 1)

    @xfail_if_cuda_tensor_descriptor
    @parametrize(
        "unroll",
        # CPU tensor descriptor codegen emits fewer descriptors than the unrolled
        # reduction test expects today. Keep the original expectation and xfail.
        [False, subtest(True, decorators=[xfail_if_cpu_tensor_descriptor])],
    )
    def test_reduce_trailing_dims_discontiguous_input(self, unroll: bool):
        # [FUNCTION DOC] Parametrized over reduction unrolling. Reduces the last two dims of a
        # discontiguous 7x5x3x2 view by maximum. With unrolling it expects seven block
        # pointers and no reduction dimension; without it expects two block pointers and two
        # reduction dimensions. Both cases expect one kernel and two pointwise dimensions.
        # Expected to fail with CUDA tensor descriptors, and with CPU ones when unrolled.
        """
        Test a [Y, X, R0, R1] reduction where the input tensor is discontiguous, but we
        only reduce over the last two dimensions.
        """
        view = self._discontiguous_tensor((7, 5, 3, 2), self.device)
        expected_num_block_pointers = 7 if unroll else 2

        # We expect block pointers for the inputs and output.
        # Note there are more inputs if unrolled.
        result, (code,) = self._run_and_compare(
            functools.partial(
                torch.amax,
                dim=(-1, -2),
            ),
            view,
            expected_num_block_pointers=expected_num_block_pointers,
            expected_num_triton_kernels=1,
            config_patches={
                "unroll_reductions_threshold": 1e4 if unroll else 1,
                **tiled_reduction_config,
            },
        )

        # Check the code for multiple pointwise dims.
        self._assert_pointwise_ndims(code, 2)
        self._assert_reduction_ndims(code, 0 if unroll else 2)

    def test_2d_reduction_with_broadcast(self):
        # [FUNCTION DOC] Regression test for a codegen error when a block pointer advances in
        # only one of two reduction dimensions: sums a 64x128 matrix times a column-broadcast
        # vector with tiled reductions and asserts correct results and two reduction
        # dimensions in the code.
        """
        Tests 2D tiled reduction with a broadcasted 1D tensor.

        This is a regression test for a bug where block pointers that only
        advance in one reduction dimension (not both) would cause a KeyError
        during codegen. The bug occurred because:
        1. The broadcasted tensor only varies along the first reduction dimension (R0)
        2. When building pointer_advancements, the block_ptr is only
           added to R0_INDEX (non-zero advancement) but skipped for R1_INDEX
           (zero/identity advancement)
        3. During loop suffix generation, the code assumed if a block_ptr exists
           in the outer loop's advancements, it must exist in the inner loop's too

        The pattern: (x * y[:, None]).sum() where x is 2D and y is 1D.
        """
        # Use sizes that require looped (non-persistent) reductions
        # to trigger 2D tiled reduction with R0 and R1 loops
        M, N = 64, 128

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test; multiplies x by y broadcast along
            # columns and returns the total sum.
            # y is 1D (M,), x is 2D (M, N)
            # y[:, None] broadcasts to (M, N)
            # The y block_ptr only advances with R0, not R1
            return (x * y[:, None]).sum()

        x = torch.randn(M, N, device=self.device)
        y = torch.randn(M, device=self.device)

        # This should compile without KeyError and produce correct results
        result, (code,) = self._run_and_compare(
            fn,
            x,
            y,
            config_patches=tiled_reduction_config,
        )

        # Verify 2D reduction is used (R0_BLOCK and R1_BLOCK present)
        self._assert_reduction_ndims(code, 2)

    def test_complex_reshape_block_ptr(self):
        # [FUNCTION DOC] Compiles a function with reshape, permute, reshape, clone and a
        # second permute and clone on an 8x2048 input added to itself, expecting two kernels
        # and four block pointers, and checks the generated first program does not use a Min
        # expression.
        def func(x, y):
            # [FUNCTION DOC] HELPER: Function under test; adds the inputs, reshapes to
            # 8x16x128, permutes the last two dims, reshapes to 1024x16 and clones it, then
            # returns that clone and a clone of its transpose.
            add_ = x + y
            reshape_0 = add_.reshape([8, 16, 128])
            permute_0 = reshape_0.permute([0, 2, 1])
            reshape_1 = permute_0.reshape([1024, 16])
            clone_0 = reshape_1.clone(memory_format=torch.contiguous_format)
            permute_1 = clone_0.permute([1, 0])
            clone_1 = permute_1.clone(memory_format=torch.contiguous_format)

            return clone_0, clone_1

        inps = (torch.rand((8, 2048), device=self.device, dtype=torch.float32),) * 2
        result, code = self._run_and_compare(
            func,
            *inps,
            expected_num_triton_kernels=2,
            expected_num_block_pointers=4,
        )
        self.assertTrue("Min" not in code[0])

    @xfail_if_tensor_descriptor
    @requires_gpu()  # FIXME this test failed on Triton-CPU
    def test_3d_permute_tiling(self):
        # [FUNCTION DOC] Requires a GPU. Compiles a three-dimensional permute and add function
        # on 51 cubed inputs with up to three tiles and ND tiling; expects one kernel, three
        # block pointers and three pointwise tile dimensions. Expected to fail with tensor
        # descriptors.
        """
        Test 3D tiling with permute.
        """

        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Function under test; adds x with dims one and two swapped
            # to y, and adds to that the swapped sum of z and y.
            dims = [0, 2, 1]
            a = x.permute(dims=dims) + y
            b = (z + y).permute(dims=dims)
            return a + b

        inps = (torch.rand((51, 51, 51), device=self.device, dtype=torch.float32),) * 3
        result, (code,) = self._run_and_compare(
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check for 3D tiling
        self._assert_pointwise_ndims(code, 3)

    @torch._dynamo.config.patch({"capture_scalar_outputs": True})
    @parametrize("num_tile_candidates", (1, 2))
    def test_unbacked_size_on_non_contig_dim(self, num_tile_candidates: int):
        # [FUNCTION DOC] Parametrized over the number of tiling candidates (one or two), with
        # scalar output capture enabled. Repeats a tensor by an unbacked size read from a
        # scalar tensor, permutes and takes the cosine; expects one kernel and no block
        # pointers with ND tiling and up to three tiles. The repeat count constant is chosen
        # to select the number of candidate tilings.
        # NUM_REPEAT should determine # of candidate_tilings.
        NUM_REPEAT = 2 if num_tile_candidates == 2 else 8

        def foo(x, length):
            # [FUNCTION DOC] HELPER: Function under test; reads the length scalar as an
            # unbacked integer, repeats x along dims one and two, permutes dims to 1, 2, 0 and
            # returns the cosine.
            unbacked = length.item()

            repeated = x.repeat(1, unbacked, NUM_REPEAT)
            # permute creates split in middle with unbacked symint is the first range
            # ranges: [33*unbacked, NUM_REPEAT, 64]
            permute120 = repeated.permute([1, 2, 0])
            return permute120.cos()

        inps = (
            torch.rand((64, 33, 1), device=self.device, dtype=torch.float32),
            torch.scalar_tensor(16, device=self.device, dtype=torch.int32),
        )

        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            self._run_and_compare(
                foo,
                *inps,
                expected_num_triton_kernels=1,
                expected_num_block_pointers=0,
                config_patches={
                    "triton.max_tiles": 3,
                    "triton.prefer_nd_tiling": True,
                },
            )

    # block_ptr advancements should also be deferrered conditional
    # on the associated buffer not being removed
    # in this case the bernoulli operation is fused with the following sum
    # so an output buffer is not needed to store the immediate result of the
    # bernoulli operation
    # TODO: fails for triton CPU "Failed to convert to LLVM IR"
    @test_torchinductor.xfail_if_triton_cpu
    @xfail_if_cpu_tensor_descriptor
    # Disable split_reductions on this test for now due to the interaction with LOAF
    @config.patch(split_reductions=False)
    def test_removed_buffers(self):
        # [FUNCTION DOC] Checks block pointer advancement is skipped for buffers that were
        # removed by fusion: a bernoulli draw fused with its sum, on a 200x200 tensor of 0.3,
        # expecting one kernel and one block pointer with loose tolerances since the draw is
        # random. Split reductions are off. Expected to fail on Triton CPU and with CPU tensor
        # descriptors.
        from torch.ops import aten

        def fn(a):
            # [FUNCTION DOC] HELPER: Function under test; draws a bernoulli sample from the
            # probabilities and returns its sum divided by the element count of a tensor made
            # from the input size.
            return aten.bernoulli(a).sum() / torch.prod(torch.tensor(a.size()))

        p = 0.3
        result, code = self._run_and_compare(
            fn,
            *[torch.ones(200, 200, device=self.device) * p],
            expected_num_triton_kernels=1,
            expected_num_block_pointers=1,
            atol=p * 0.06,
            rtol=0.06,
        )

    @xfail_if_tensor_descriptor
    def test_pointwise_index_order(self):
        # [FUNCTION DOC] Adds two discontiguous 5x5x5 tensors with up to three tiles and ND
        # tiling and expects one kernel and three block pointers. Compares the load, store and
        # index lines of the generated code with golden text to check that Z is the leading
        # dimension, then Y, then X. Expected to fail with tensor descriptors.
        """
        Test the order of indices in pointwise kernels. Expect Z to be the leading dim,
        then Y, then X.
        """

        inps = [
            self._discontiguous_tensor((5, 5, 5), device=self.device) for _ in range(2)
        ]

        result, (triton_code,) = self._run_and_compare(
            torch.add,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check the load and store for block pointer strides.
        load_lines, store_lines, index_lines = tuple(
            self._get_lines_containing_substr(triton_code, substr)
            for substr in ("tl.load", "tl.store", "index =")
        )
        self.assertExpectedInline(
            load_lines,
            """\
    tmp0 = tl.load(tl.make_block_ptr(in_ptr0, shape=[5, 5, 5], strides=[100, 10, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), boundary_check=[0, 1, 2])
    tmp1 = tl.load(tl.make_block_ptr(in_ptr1, shape=[5, 5, 5], strides=[100, 10, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), boundary_check=[0, 1, 2])""",
        )

        self.assertExpectedInline(
            store_lines,
            """    tl.store(tl.make_block_ptr(out_ptr0, shape=[5, 5, 5], strides=[25, 5, 1], block_shape=[ZBLOCK, YBLOCK, XBLOCK], order=[2, 1, 0], offsets=[zoffset, yoffset, xoffset]), tl.broadcast_to(tmp2, [ZBLOCK, YBLOCK, XBLOCK]).to(tl.float32), boundary_check=[0, 1, 2])""",
        )

        # Check the indices. These are used for non-block pointers.
        self.assertExpectedInline(
            index_lines,
            """\
    zindex = zoffset + tl.arange(0, ZBLOCK)[:, None, None]
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :, None]
    xindex = xoffset + tl.arange(0, XBLOCK)[None, None, :]""",
        )

    def test_expand_clone_broadcast(self):
        # [FUNCTION DOC] Expands a 1x32 input to 32x32 and clones it with up to three tiles
        # and ND tiling; expects one kernel, two block pointers and exactly one explicit
        # broadcast in the code.
        """
        Test expand followed by clone. This uses an explicit Triton broadcast.
        """
        base_size = (1, 32)
        expanded_size = (32, 32)

        def foo(x):
            # [FUNCTION DOC] HELPER: Function under test; expands x to the 32x32 size and
            # clones it.
            return x.expand(*expanded_size).clone()

        inps = [torch.randn(base_size, device=self.device)]
        result, (triton_code,) = self._run_and_compare(
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=2,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # We should only need one broadcast.
        num_broadcasts = triton_code.count("tl.broadcast_to")
        self.assertEqual(num_broadcasts, 1)

    def test_reduction_fused_with_as_strided_scatter_broadcast_error(self):
        # [FUNCTION DOC] Regression test for a fused pointwise and reduction kernel whose side
        # output is made by an as_strided scatter: the store must keep the singleton reduction
        # dimension rather than reducing a two-dimensional temporary to one dimension. Runs
        # the function with persistent reductions off and two tiles, expecting two kernels and
        # atol 3e-5.
        # Regression test for a fused pointwise+reduction kernel where the side
        # output is materialized through as_strided_scatter. The pointwise load
        # is rank-expanded to match the reduction loop, and the immediate store
        # must preserve that singleton reduction dimension instead of lowering a
        # [XBLOCK, 1] temporary to a [XBLOCK] store.
        def fn(a, b, table):
            # [FUNCTION DOC] HELPER: Function under test; builds a zero tensor one element
            # wider in the last dim, scatters a into a strided slice at offset one, gathers
            # embedding rows for a, adds them (broadcast) to b, sums the last dim, and returns
            # the scattered tensor and the sum.
            dst = torch.zeros(
                (a.shape[0], a.shape[1], a.shape[2] + 1),
                dtype=a.dtype,
                device=a.device,
            )
            side = torch.as_strided_scatter(
                dst, a, list(a.shape), list(dst.stride()), 1
            )
            gathered = torch.embedding(table, a)
            reduced = (b + gathered.unsqueeze(0)).sum(-1)
            return side, reduced

        device = torch.device(self.device)
        base = torch.randint(0, 1024, (4, 64, 1), dtype=torch.int64, device=device)
        a = base.expand(4, 64, 32)
        b = torch.randn((4, 64, 32, 128), dtype=torch.float32, device=device)
        table = torch.randn((1024, 128), dtype=torch.float32, device=device)

        self._run_and_compare(
            fn,
            a,
            b,
            table,
            atol=3e-5,
            config_patches={
                "triton.persistent_reductions": False,
                "triton.max_tiles": 2,
            },
            expected_num_triton_kernels=2,
        )

    def test_mul_broadcast_multi_output(self):
        # [FUNCTION DOC] Compiles a function with a vector broadcast multiplied into an
        # 8x11x128 tensor and several derived outputs, with up to three tiles and ND tiling,
        # expecting one kernel and seven block pointers. Asserts only two pointwise dimensions
        # are used since singleton splits are discarded.
        def foo(x, y, z):
            # [FUNCTION DOC] HELPER: Function under test; returns the product a of x and y, a
            # scaled by 128, a times z, and x times z.
            a = x * y
            b = 128.0
            c = a * b
            d = a * z
            e = x * z
            return a, c, d, e

        inps = [
            torch.randn((8, 11, 128), device=self.device),
            torch.randn((128,), device=self.device),
            torch.randn((8, 11, 128), device=self.device),
        ]
        result, (triton_code,) = self._run_and_compare(
            foo,
            *inps,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=7,
            config_patches={
                "triton.max_tiles": 3,
                "triton.prefer_nd_tiling": True,
            },
        )

        # Check that the tiling is 2D, even though we allow up to 3D.
        # Singleton splits should be discarded.
        self._assert_pointwise_ndims(triton_code, 2)

    # Integration test to ensure that matched dims & strides from match_mod_div_expr
    # are unsigned and signed integers respectively. This test case has the following
    # index:=(ModularIndexing(xindex, 4, 4)) + 4*(ModularIndexing(xindex, 32, 2))
    # and the match below is a candidate that is invalid:
    # match={
    #   dim_mod4_: 32, dim_mod3_: 2, stride_mod3_: 4, dim_mod2_: 1/16,
    #   dim_mod1_: 4, stride_mod1_: 1, stride_mod4_: 0, stride_mod2_: 0, stride_mod0_: 0
    # }
    # This is now fixed by ensuring that wild symbols only match integers
    def test_ensure_integral_dims_and_strides(self):
        # [FUNCTION DOC] Integration regression test that dimension and stride matching of
        # modular index expressions only yields integers. Compiles the unfold operation on a
        # float16 zero tensor requiring grad, with full graph capture, expecting one kernel,
        # and asserts that the block descriptor constructor text appears in the generated
        # code.
        def model(data, *args):
            # [FUNCTION DOC] HELPER: Function under test; applies the unfold operation to the
            # data with the remaining arguments (kernel size, dilation, padding, stride).
            return torch.nn.functional.unfold(data, *args)

        data = torch.zeros(
            [2, 3, 5, 5], dtype=torch.float16, requires_grad=True, device=self.device
        )
        args = [2, 1, 0, 1]
        _, code = self._run_and_compare(
            model,
            data,
            *args,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=None,
            compile_kwargs={"fullgraph": True},
        )
        self.assertIn(self.block_descriptor_constructor_str, "\n".join(code))

    # Integration test to test block analysis with index expressions using
    # negative strides.
    # This test case has the following index:
    # index_relative_to_xyr_index = -256*((xindex//64)) - (ModularIndexing(xindex, 1, 8))
    #    - 16*(ModularIndexing(xindex, 8, 8)) + 1911
    # subexpr = -256*((xindex//64)) - (ModularIndexing(xindex, 1, 8)) - 16*(ModularIndexing(xindex, 8, 8))
    # Block analysis should produce the following:
    # BlockParameters(
    #   shape=[8, 8, 8],
    #   block_shape=[((XBLOCK + 63)//64), Min(8, ((XBLOCK + 7)//8)), Min(8, XBLOCK) ],
    #   strides=[-256, -16, -1],
    #   offsets=[(xoffset//64), ModularIndexing(xoffset, 8, 8), ModularIndexing(xoffset, 1, 8)]
    #   )
    # constant_offset = 1911
    @xfail_if_tensor_descriptor
    def test_negative_strides(self):
        # [FUNCTION DOC] Integration test of block analysis for index expressions with
        # negative strides: adds a fully flipped discontiguous 8x8x8 tensor to another one,
        # expecting one kernel and three block pointers. Expected to fail with tensor
        # descriptors.
        def model(x, y):
            # [FUNCTION DOC] HELPER: Function under test; flips x along all three dims and
            # adds y.
            # Slice in reverse order via a negative stride
            return torch.flip(x, [0, 1, 2]) + y

        x, y = (
            self._discontiguous_tensor((8, 8, 8), device=self.device) for _ in range(2)
        )
        self._run_and_compare(
            model,
            x,
            y,
            expected_num_triton_kernels=1,
            expected_num_block_pointers=3,
        )

    @config.patch("triton.prefer_nd_tiling", True)
    @config.patch("triton.max_tiles", 3)
    @parametrize(
        "block_multiple, ynumel_exceed_ygrid_size, include_z",
        [
            # No boundary check in all dimensions
            [True, False, True],
            # No xdim boundary check, ydim is checked since > max_ygrid
            # z dim can be used since its not included
            [True, True, False],
            # Boundary check in all dimensions
            # skip triton_cpu very slow test > 1000s
            subtest(
                [False, False, True], decorators=[test_torchinductor.skip_if_triton_cpu]
            ),
        ],
    )
    @xfail_if_tensor_descriptor
    def test_boundary_check(self, block_multiple, ynumel_exceed_ygrid_size, include_z):
        # [FUNCTION DOC] Parametrized with ND tiling and three tiles. Adds a discontiguous
        # tensor and a second one with last dim one, using block size 8 forced through a
        # choices handler so inputs stay small. Dim sizes are 8 or 9 (not a multiple of the
        # block), the y size can exceed the maximum y grid, and z may be absent. Expects one
        # kernel and three block pointers. If sizes are block multiples: when y exceeds the
        # grid, the offset line uses the extra program id axis and exactly three boundary
        # checks on dim zero appear; else no boundary checks. Otherwise three-dimensional and
        # two-dimensional boundary checks must appear. Expected to fail with tensor
        # descriptors.
        @dataclasses.dataclass
        class InputShape:
            # [CLASS DOC] HELPER: Dataclass describing the test tensor extents: required x and
            # y and an optional z, with to_list giving them in dimension order.
            x: int
            y: int
            z: int | None = None

            def to_list(self):
                # [FUNCTION DOC] HELPER: Returns the sizes as a list ordered y then x, with z
                # first when set.
                out = [self.y, self.x]
                if self.z is not None:
                    out.insert(0, self.z)
                return out

        BLOCK_SIZE = 8
        DIM_SIZE = BLOCK_SIZE if block_multiple else BLOCK_SIZE + 1
        shape = InputShape(DIM_SIZE, DIM_SIZE, DIM_SIZE if include_z else None)
        if ynumel_exceed_ygrid_size:
            shape.y = math.ceil(get_max_y_grid()) * shape.y + shape.y

        # Use fixed block sizes to avoid having to generate very large input tensors
        class FixedBlockSizeChoices(InductorChoices):
            # [CLASS DOC] HELPER: Choices handler forcing triton kernels to use a fixed
            # configuration whose block sizes are all eight, one entry per present dimension.
            def triton_kernel_kwargs(self, kernel_cls, features, groups, kernel_kwargs):
                # [FUNCTION DOC] HELPER: Fills the fixed_config kernel keyword with a fixed
                # config built from a mapping of upper-cased dimension names plus BLOCK to the
                # block size for each non-None extent of the enclosing test shape; returns the
                # updated kernel_kwargs mapping.
                block_sizes = {
                    f"{prefix.upper()}BLOCK": BLOCK_SIZE
                    for prefix, size in dataclasses.asdict(shape).items()
                    if size is not None
                }
                kernel_kwargs["fixed_config"] = FixedTritonConfig(block_sizes)
                return kernel_kwargs

        a = self._discontiguous_tensor(shape.to_list(), device=self.device)
        b_shape = shape.to_list()
        b_shape[-1] = 1
        b = self._discontiguous_tensor(b_shape, device=self.device)

        def func(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function under test; returns the sum a plus b.
            return a + b

        with V.set_choices_handler(FixedBlockSizeChoices()):
            result, code = self._run_and_compare(
                func,
                a,
                b,
                expected_num_triton_kernels=1,
                expected_num_block_pointers=3,
            )

            code = code[0]
            if block_multiple:
                if ynumel_exceed_ygrid_size:
                    self.assertIn(
                        "yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK",
                        code,
                    )
                    # Only the y dimension should be boundary checked
                    # a, b, and output
                    self.assertEqual(code.count("boundary_check=[0]"), 3)
                else:
                    # No boundary checking
                    self.assertNotIn("boundary_check", code)
            else:
                # Loading a
                self.assertTrue("boundary_check=[0, 1, 2]" in code)
                # Loading b
                self.assertTrue("boundary_check=[0, 1]" in code)


@unittest.skipIf(not HAS_GPU, "requires triton GPU backend")
@config.patch("triton.use_block_ptr", True)
class TritonBlockPointerTestGPU(BlockDescriptorTestBase):
    # [CLASS DOC] Block pointer tests on a GPU with the block pointer option enabled. The test
    # methods are copied from CommonTemplate below; skipped when no triton GPU backend exists.
    device = GPU_TYPE


test_torchinductor.copy_tests(CommonTemplate, TritonBlockPointerTestGPU, GPU_TYPE)


@unittest.skipIf(not TRITON_HAS_CPU, "requires triton CPU backend")
@config.patch({"triton.use_tensor_descriptor": True, "cpu_backend": "triton"})
@instantiate_parametrized_tests
class TritonTensorDescriptorTestCPU(BlockDescriptorTestBase):
    # [CLASS DOC] Tensor descriptor tests on CPU with the triton CPU backend. Uses the tensor
    # descriptor constructor text for counting and takes expected failures from the CPU marker
    # attribute. Skipped without a triton CPU backend.
    block_descriptor_constructor_str = "tl.make_tensor_descriptor"
    device = "cpu"


@unittest.skipIf(
    not (HAS_CUDA_AND_TRITON and torch.cuda.get_device_capability()[0] >= 9)
    or torch.version.hip,
    "Requires Triton CUDA backend and CUDA compute capability >= 9.0. Not supported on ROCm",
    # ROCm triton doesn't support/generate "tl.make_tensor_descriptor" which is exactly what this unit test is about
)
@config.patch({"triton.use_tensor_descriptor": True, "assume_aligned_inputs": True})
@instantiate_parametrized_tests
class TritonTensorDescriptorTestCUDA(BlockDescriptorTestBase):
    # [CLASS DOC] Tensor descriptor (TMA) tests on CUDA, needing compute capability 9 or
    # higher and not ROCm. Counts tensor descriptor constructors in generated code and assumes
    # aligned inputs; adds tests for transposes and TMA fallbacks to the shared copied tests.
    block_descriptor_constructor_str = "tl.make_tensor_descriptor"
    device = GPU_TYPE

    @config.patch({"triton.transpose_discontiguous_tensor_descriptor": True})
    @parametrize(
        "view_size,permute_order,num_tensor_descriptors,expect_transpose",
        [
            ((128,), (0,), 3, False),
            ((128, 128), (0, 1), 3, False),
            ((128, 64), (1, 0), 3, True),
            ((256, 32, 16), (2, 0, 1), 3, True),
            ((16, 32, 256), (2, 0, 1), 3, True),
        ],
    )
    def test_match_with_transpose(
        self,
        view_size: tuple[int],
        permute_order: tuple[int],
        num_tensor_descriptors: int,
        expect_transpose: bool,
    ):
        # [FUNCTION DOC] Parametrized with transposed discontiguous tensor descriptors
        # enabled. Multiplies a discontiguous tensor by a permuted one (its input shape is
        # built so the permute yields the view size), with tiled reductions; expects the given
        # number of descriptors, one kernel, and exactly one transpose in the code when a
        # transpose is expected, else none.
        a = self._discontiguous_tensor(view_size, self.device)
        pre_permute_size = [1] * len(view_size)
        for i, value in zip(permute_order, view_size):
            pre_permute_size[i] = value
        b = self._discontiguous_tensor(pre_permute_size, self.device)
        b = b.permute(permute_order)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Function under test; returns the elementwise product of a
            # and b.
            return a * b

        result, (code,) = self._run_and_compare(
            fn,
            a,
            b,
            expected_num_block_pointers=num_tensor_descriptors,
            expected_num_triton_kernels=1,
            config_patches=tiled_reduction_config,
        )

        transpose_count = code.count("tl.trans")
        self.assertEqual(transpose_count, 1 if expect_transpose else 0)

    def test_rms_norm_backward_does_not_crash_with_tma(self):
        # [FUNCTION DOC] Regression test that compiling the backward of an RMS norm over a
        # very wide bfloat16 input (1x1024x40096) does not crash with TMA enabled; checks both
        # gradients exist.
        B, S, D = 1, 1024, 40096
        with torch.device(self.device):
            x = torch.randn(B, S, D, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(D, dtype=torch.bfloat16, requires_grad=True)

        def f(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Function under test; applies RMS norm with weight and a
            # small epsilon then returns the sum of the result scaled by 0.1.
            y = torch.rms_norm(x, (D,), w, 1e-5)
            return (y * 0.1).sum()

        compiled_f = torch.compile(f, backend="inductor", fullgraph=True)
        loss = compiled_f(x, w)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(w.grad)

    @largeTensorTest("1GB", inductor=True)
    def test_large_tensor_pointwise(self):
        # [FUNCTION DOC] Needs about 1 GB. Adds four to an int8 tensor with more than 2**30
        # elements and asserts all results equal four, exercising 64-bit indexing.
        def fn(a):
            # [FUNCTION DOC] HELPER: Function under test; adds four to a.
            return a + 4

        t = torch.zeros(2**30 + 1, dtype=torch.int8, device=GPU_TYPE)
        compiled_fn = torch.compile(fn)
        actual = compiled_fn(t)
        self.assertTrue((actual == 4).all())

    def test_slice_constant_offset_disables_tma(self):
        # [FUNCTION DOC] Checks a slice that starts at element one (a four byte offset, not 16
        # byte aligned) still compiles and matches eager, and that the code contains a plain
        # load.
        """TMA requires 16-byte aligned base; x[1:] with float32 yields 4-byte offset."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test; drops the first element and adds
            # one.
            return x[1:] + 1

        x = torch.randn(1025, device=GPU_TYPE)
        result, (code,) = run_and_get_code(torch.compile(fn), x)
        self.assertEqual(result, fn(x))
        self.assertIn("tl.load", code)

    def test_slice_view_dtype_unaligned_buffer(self):
        # [FUNCTION DOC] Checks that a bfloat16 slice from offset two reinterpreted as float32
        # and incremented gives the same results compiled and eager, covering an unaligned
        # buffer.
        offset = 1

        def f(x):
            # [FUNCTION DOC] HELPER: Function under test; slices from element two, views as
            # float32 and adds one.
            return x[2:].view(dtype=torch.float32) + 1

        x = torch.randn((128 + offset) * 2, dtype=torch.bfloat16, device=GPU_TYPE)
        expected = f(x)
        actual = torch.compile(f)(x)
        self.assertEqual(actual, expected)

    def test_persistent_reduction_store_small_rblock_skips_tma(self):
        # [FUNCTION DOC] Batch norm in training mode on a 2x1 input has a persistent reduction
        # block of two elements (8 bytes, under the 16 byte TMA minimum). Asserts compiled
        # results match eager, the code has a store, and no tensor descriptor is created for
        # it.
        """
        When a persistent reduction has rnumel < 16/element_size, the fixed
        R0_BLOCK cannot satisfy TMA's 16-byte minimum.  The store must fall
        back to scalar indexing instead of emitting an invalid descriptor.
        """

        def fn(x, running_mean, running_var, weight, bias):
            # [FUNCTION DOC] HELPER: Function under test; batch norm in training mode with
            # running stats, weight and bias.
            return torch.nn.functional.batch_norm(
                x, running_mean, running_var, weight, bias, training=True
            )

        # Input (2, 1): reduction over batch dim of size 2.
        # R0_BLOCK = next_power_of_2(2) = 2, so 2 * 4 bytes = 8 < 16.
        x = torch.randn(2, 1, device=GPU_TYPE)
        weight = torch.randn(1, device=GPU_TYPE)
        bias = torch.randn(1, device=GPU_TYPE)
        running_mean = torch.randn(1, device=GPU_TYPE)
        running_var = torch.randn(1, device=GPU_TYPE).abs()

        result, (code,) = run_and_get_code(
            torch.compile(fn), x, running_mean, running_var, weight, bias
        )
        expected = fn(x, running_mean, running_var, weight, bias)
        self.assertEqual(result, expected)
        # The store must fall back to scalar indexing, not TMA.
        self.assertIn("tl.store", code)
        self.assertNotIn("make_tensor_descriptor", code)

    def test_bool_dtype_skips_tma(self):
        # [FUNCTION DOC] Checks a logical-not on a bool tensor of 16 elements compiles with no
        # block pointers or descriptors, since bool maps to a type without a tensor map data
        # type.
        """
        torch.bool buffers map to Triton tl.int1 which has no
        CUtensorMapDataType entry, so they should skip TMA.
        """

        def fn(a):
            # [FUNCTION DOC] HELPER: Function under test; returns the logical not of a.
            return torch.logical_not(a)

        inp = torch.zeros(16, dtype=torch.bool, device=GPU_TYPE)
        self._run_and_compare(fn, inp, expected_num_block_pointers=0)


test_torchinductor.copy_tests(
    CommonTemplate,
    TritonTensorDescriptorTestCPU,
    "cpu",
    xfail_prop="_expected_failure_cpu_tensor_descriptor",
)

test_torchinductor.copy_tests(
    CommonTemplate,
    TritonTensorDescriptorTestCUDA,
    GPU_TYPE,
    xfail_prop="_expected_failure_cuda_tensor_descriptor",
)


class TestTilingExtra(InductorTestCase):
    # [CLASS DOC] Extra tiling regression tests that compile a large model graph on a GPU.
    @requires_gpu()
    def test_tiling_split_valid(self):
        # [FUNCTION DOC] Requires a GPU. Regression test that a speech-encoder-like model
        # (subsampler convolutions, two conformer-style layers with rotary attention,
        # convolution module and feed-forward blocks) compiles with full graph capture and
        # dynamic shapes, with a fixed seed, batch 13 and sequence length 1024 in float16.
        # KNOWN ISSUE: the eager and compiled outputs are both computed but never compared, so
        # only successful compile and run is checked.
        import torch.nn.functional as F

        class GraphModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the traced forward pass of the model
            # described above; has no parameters of its own, all weights and the two symbolic
            # sizes are passed as arguments.
            def forward(
                self,
                sub_dense0_w,  # f16[64, 80]
                sub_dense0_b,  # f16[64]
                s25,  # Sym(s25) - batch size
                s70,  # Sym(s70) - sequence length
                input_feat,  # f16[s25, s70, 80]
                sub_conv0_w,  # f16[64, 64, 5]
                sub_conv0_b,  # f16[64]
                sub_conv1_w,  # f16[32, 64, 5]
                sub_conv1_b,  # f16[32]
                sub_dense1_w,  # f16[64, 32]
                sub_dense1_b,  # f16[64]
                rot_inv_freq,  # f32[8]
                rot_attn_scale,  # f64[] cpu
                l0_norm_ff1_w,  # f16[64]
                l0_ff1_lin1_w,  # f16[256, 64]
                l0_ff1_lin2_w,  # f16[64, 256]
                l0_ff_res0,  # f64[] cpu
                l0_ff_res1,  # f64[] cpu
                l0_norm_attn_w,  # f16[64]
                l0_q_w,  # f16[64, 64]
                l0_k_w,  # f16[64, 64]
                l0_v_w,  # f16[64, 64]
                l0_o_w,  # f16[64, 64]
                l0_norm_conv_w,  # f16[64]
                l0_pw_conv1_w,  # f16[128, 64, 1]
                l0_dw_conv_w,  # f16[64, 1, 8]
                l0_bn_mean,  # f16[64]
                l0_bn_var,  # f16[64]
                l0_bn_w,  # f16[64]
                l0_bn_b,  # f16[64]
                l0_pw_conv2_w,  # f16[64, 64, 1]
                l0_conv_res0,  # f64[] cpu
                l0_conv_res1,  # f64[] cpu
                l0_norm_ff2_w,  # f16[64]
                l0_ff2_lin1_w,  # f16[256, 64]
                l0_ff2_lin2_w,  # f16[64, 256]
                l0_norm_out_w,  # f16[64]
                l1_norm_ff1_w,  # f16[64]
                l1_ff1_lin1_w,  # f16[256, 64]
                l1_ff1_lin2_w,  # f16[64, 256]
                l1_norm_attn_w,  # f16[64]
                l1_q_w,  # f16[64, 64]
                l1_k_w,  # f16[64, 64]
                l1_v_w,  # f16[64, 64]
                l1_o_w,  # f16[64, 64]
                l1_norm_conv_w,  # f16[64]
                l1_pw_conv1_w,  # f16[128, 64, 1]
                l1_dw_conv_w,  # f16[64, 1, 8]
                l1_bn_mean,  # f16[64]
                l1_bn_var,  # f16[64]
                l1_bn_w,  # f16[64]
                l1_bn_b,  # f16[64]
                l1_pw_conv2_w,  # f16[64, 64, 1]
                l1_norm_ff2_w,  # f16[64]
                l1_ff2_lin1_w,  # f16[256, 64]
                l1_ff2_lin2_w,  # f16[64, 256]
                l1_norm_out_w,  # f16[64]
                out_norm_w,  # f16[64]
            ):
                # [FUNCTION DOC] HELPER: Forward pass taking weights as arguments plus batch
                # size and sequence length. Applies a dense layer with relu, two strided
                # convolutions with relu, a dense layer, computes rotary embeddings and a mask
                # for the shortened sequence, then runs two layers of feed forward, rotary
                # self attention with scaled dot product attention, a gated convolution module
                # and a second feed forward, each with layer norm and weighted residuals taken
                # from scalar tensors, and a final layer norm. Returns a one-tuple.
                # Subsampler: dense_0 + relu
                linear = torch._C._nn.linear(input_feat, sub_dense0_w, sub_dense0_b)
                hidden_states = F.relu(linear, inplace=False)

                # Transpose for conv
                hidden_states_1 = hidden_states.transpose(1, 2)

                # Conv_0 with stride=2
                conv1d = torch.conv1d(
                    hidden_states_1, sub_conv0_w, sub_conv0_b, (2,), (0,), (1,), 1
                )
                hidden_states_2 = F.relu(conv1d, inplace=False)

                # Conv_1 with stride=2
                conv1d_1 = torch.conv1d(
                    hidden_states_2, sub_conv1_w, sub_conv1_b, (2,), (0,), (1,), 1
                )
                hidden_states_3 = F.relu(conv1d_1, inplace=False)

                # Transpose back
                hidden_states_4 = hidden_states_3.transpose(1, 2)

                # Dense_1
                hidden_states_5 = torch._C._nn.linear(
                    hidden_states_4, sub_dense1_w, sub_dense1_b
                )

                # Compute output sequence length: ((s70 - 5) // 4) - 1
                # Note: In the dynamo graph, torch.sym_sum is used, but we use direct arithmetic here
                sym_sum = s70 - 5
                floordiv = sym_sum // 4
                sym_sum_1 = floordiv - 1

                # Create position ids
                arange = torch.arange(sym_sum_1, device=GPU_TYPE)
                unsqueeze = arange.unsqueeze(0)

                # Rotary embedding computation
                getitem_3 = rot_inv_freq[(None, slice(None, None, None), None)]
                float_1 = getitem_3.float()
                expand = float_1.expand(1, -1, 1)
                inv_freq_expanded = expand.to(torch.device(GPU_TYPE, index=0))

                getitem_6 = unsqueeze[
                    (slice(None, None, None), None, slice(None, None, None))
                ]
                position_ids_expanded = getitem_6.float()

                # Rotary embedding frequency computation (autocast removed for tracing)
                float_3 = inv_freq_expanded.float()
                float_4 = position_ids_expanded.float()
                matmul = float_3 @ float_4
                freqs = matmul.transpose(1, 2)

                emb = torch.cat((freqs, freqs), dim=-1)

                cos = emb.cos()
                item = rot_attn_scale.item()
                cos_1 = cos * item

                sin = emb.sin()
                sin_1 = sin * item

                cos_2 = cos_1.to(dtype=torch.float16)
                sin_2 = sin_1.to(dtype=torch.float16)

                # Dropout (no-op in eval mode)
                hidden_states_6 = F.dropout(hidden_states_5, p=0.1, training=False)
                cos_3 = F.dropout(cos_2, p=0.0, training=False)
                sin_3 = F.dropout(sin_2, p=0.0, training=False)

                # Create attention mask
                cache_position = torch.arange(
                    sym_sum_1, device=GPU_TYPE, dtype=torch.int64
                )
                arange_4 = torch.arange(sym_sum_1, device=GPU_TYPE)

                q_indices = cache_position[(None, None, slice(None, None, None), None)]
                attention_mask = q_indices >= 0
                attention_mask_1 = attention_mask.expand(s25, -1, sym_sum_1, sym_sum_1)

                # ============ LAYER 0 ============
                # Feed forward 1
                layer_norm = F.layer_norm(
                    hidden_states_6, (64,), l0_norm_ff1_w, None, 1e-06
                )
                linear_2 = torch._C._nn.linear(layer_norm, l0_ff1_lin1_w, None)
                hidden_states_7 = F.silu(linear_2)
                hidden_states_8 = F.dropout(hidden_states_7, p=0.1, training=False)
                hidden_states_9 = torch._C._nn.linear(
                    hidden_states_8, l0_ff1_lin2_w, None
                )

                # Residual connection with weights
                item_5 = l0_ff_res0.item()
                mul_2 = item_5 * hidden_states_6
                item_6 = l0_ff_res1.item()
                mul_3 = item_6 * hidden_states_9
                hidden_states_10 = mul_2 + mul_3

                # Self attention
                normalized_hidden_states = F.layer_norm(
                    hidden_states_10, (64,), l0_norm_attn_w, None, 1e-06
                )

                linear_4 = torch._C._nn.linear(normalized_hidden_states, l0_q_w, None)
                view = linear_4.view((s25, sym_sum_1, -1, 16))
                query_states = view.transpose(1, 2)

                linear_5 = torch._C._nn.linear(normalized_hidden_states, l0_k_w, None)
                view_1 = linear_5.view((s25, sym_sum_1, -1, 16))
                key_states = view_1.transpose(1, 2)

                linear_6 = torch._C._nn.linear(normalized_hidden_states, l0_v_w, None)
                view_2 = linear_6.view((s25, sym_sum_1, -1, 16))
                value_states = view_2.transpose(1, 2)

                # Apply rotary embeddings
                cos_4 = cos_3.unsqueeze(1)
                sin_4 = sin_3.unsqueeze(1)

                mul_4 = query_states * cos_4
                x1 = query_states[(Ellipsis, slice(None, 8, None))]
                x2 = query_states[(Ellipsis, slice(8, None, None))]
                neg = -x2
                cat_1 = torch.cat((neg, x1), dim=-1)
                mul_5 = cat_1 * sin_4
                q_embed = mul_4 + mul_5

                mul_6 = key_states * cos_4
                x1_1 = key_states[(Ellipsis, slice(None, 8, None))]
                x2_1 = key_states[(Ellipsis, slice(8, None, None))]
                neg_1 = -x2_1
                cat_2 = torch.cat((neg_1, x1_1), dim=-1)
                mul_7 = cat_2 * sin_4
                k_embed = mul_6 + mul_7

                # SDPA
                attn_output = torch._C._nn.scaled_dot_product_attention(
                    q_embed,
                    k_embed,
                    value_states,
                    attn_mask=attention_mask_1,
                    dropout_p=0.0,
                    scale=0.25,
                    is_causal=False,
                )

                transpose_6 = attn_output.transpose(1, 2)
                attn_output_1 = transpose_6.contiguous()
                reshape = attn_output_1.reshape(s25, sym_sum_1, -1)
                attn_output_2 = reshape.contiguous()
                attn_output_3 = torch._C._nn.linear(attn_output_2, l0_o_w, None)

                hidden_states_11 = hidden_states_10 + attn_output_3

                # Convolution module
                layer_norm_2 = F.layer_norm(
                    hidden_states_11, (64,), l0_norm_conv_w, None, 1e-06
                )
                hidden_states_12 = layer_norm_2.transpose(1, 2)
                hidden_states_13 = torch.conv1d(
                    hidden_states_12, l0_pw_conv1_w, None, (1,), (0,), (1,), 1
                )
                hidden_states_14 = F.glu(hidden_states_13, dim=1)

                invert = ~attention_mask_1
                all_masked_rows = torch.all(invert, dim=2)
                hidden_states_15 = hidden_states_14.masked_fill(all_masked_rows, 0.0)

                hidden_states_16 = torch.conv1d(
                    hidden_states_15, l0_dw_conv_w, None, (1,), "same", (1,), 64
                )
                hidden_states_17 = F.batch_norm(
                    hidden_states_16,
                    l0_bn_mean,
                    l0_bn_var,
                    l0_bn_w,
                    l0_bn_b,
                    False,
                    0.01,
                    1e-05,
                )
                hidden_states_18 = F.silu(hidden_states_17)
                hidden_states_19 = torch.conv1d(
                    hidden_states_18, l0_pw_conv2_w, None, (1,), (0,), (1,), 1
                )
                conv_output = hidden_states_19.transpose(1, 2)

                # Conv residual
                item_12 = l0_conv_res0.item()
                item_13 = l0_conv_res1.item()
                mul_8 = item_12 * hidden_states_11
                mul_9 = item_13 * conv_output
                hidden_states_20 = mul_8 + mul_9

                # Feed forward 2
                layer_norm_3 = F.layer_norm(
                    hidden_states_20, (64,), l0_norm_ff2_w, None, 1e-06
                )
                linear_8 = torch._C._nn.linear(layer_norm_3, l0_ff2_lin1_w, None)
                hidden_states_21 = F.silu(linear_8)
                hidden_states_22 = F.dropout(hidden_states_21, p=0.1, training=False)
                hidden_states_23 = torch._C._nn.linear(
                    hidden_states_22, l0_ff2_lin2_w, None
                )

                mul_10 = item_5 * hidden_states_20
                mul_11 = item_6 * hidden_states_23
                hidden_states_24 = mul_10 + mul_11

                hidden_states_25 = F.layer_norm(
                    hidden_states_24, (64,), l0_norm_out_w, None, 1e-06
                )

                # ============ LAYER 1 ============
                # Feed forward 1
                layer_norm_5 = F.layer_norm(
                    hidden_states_25, (64,), l1_norm_ff1_w, None, 1e-06
                )
                linear_10 = torch._C._nn.linear(layer_norm_5, l1_ff1_lin1_w, None)
                hidden_states_26 = F.silu(linear_10)
                hidden_states_27 = F.dropout(hidden_states_26, p=0.1, training=False)
                hidden_states_28 = torch._C._nn.linear(
                    hidden_states_27, l1_ff1_lin2_w, None
                )

                mul_12 = item_5 * hidden_states_25
                mul_13 = item_6 * hidden_states_28
                hidden_states_29 = mul_12 + mul_13

                # Self attention
                normalized_hidden_states_1 = F.layer_norm(
                    hidden_states_29, (64,), l1_norm_attn_w, None, 1e-06
                )

                linear_12 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_q_w, None
                )
                view_3 = linear_12.view((s25, sym_sum_1, -1, 16))
                query_states_1 = view_3.transpose(1, 2)

                linear_13 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_k_w, None
                )
                view_4 = linear_13.view((s25, sym_sum_1, -1, 16))
                key_states_1 = view_4.transpose(1, 2)

                linear_14 = torch._C._nn.linear(
                    normalized_hidden_states_1, l1_v_w, None
                )
                view_5 = linear_14.view((s25, sym_sum_1, -1, 16))
                value_states_1 = view_5.transpose(1, 2)

                # Apply rotary embeddings
                cos_5 = cos_3.unsqueeze(1)
                sin_5 = sin_3.unsqueeze(1)

                mul_14 = query_states_1 * cos_5
                x1_2 = query_states_1[(Ellipsis, slice(None, 8, None))]
                x2_2 = query_states_1[(Ellipsis, slice(8, None, None))]
                neg_2 = -x2_2
                cat_3 = torch.cat((neg_2, x1_2), dim=-1)
                mul_15 = cat_3 * sin_5
                q_embed_1 = mul_14 + mul_15

                mul_16 = key_states_1 * cos_5
                x1_3 = key_states_1[(Ellipsis, slice(None, 8, None))]
                x2_3 = key_states_1[(Ellipsis, slice(8, None, None))]
                neg_3 = -x2_3
                cat_4 = torch.cat((neg_3, x1_3), dim=-1)
                mul_17 = cat_4 * sin_5
                k_embed_1 = mul_16 + mul_17

                # SDPA
                attn_output_4 = torch._C._nn.scaled_dot_product_attention(
                    q_embed_1,
                    k_embed_1,
                    value_states_1,
                    attn_mask=attention_mask_1,
                    dropout_p=0.0,
                    scale=0.25,
                    is_causal=False,
                )

                transpose_12 = attn_output_4.transpose(1, 2)
                attn_output_5 = transpose_12.contiguous()
                reshape_1 = attn_output_5.reshape(s25, sym_sum_1, -1)
                attn_output_6 = reshape_1.contiguous()
                attn_output_7 = torch._C._nn.linear(attn_output_6, l1_o_w, None)

                hidden_states_30 = hidden_states_29 + attn_output_7

                # Convolution module
                layer_norm_7 = F.layer_norm(
                    hidden_states_30, (64,), l1_norm_conv_w, None, 1e-06
                )
                hidden_states_31 = layer_norm_7.transpose(1, 2)
                hidden_states_32 = torch.conv1d(
                    hidden_states_31, l1_pw_conv1_w, None, (1,), (0,), (1,), 1
                )
                hidden_states_33 = F.glu(hidden_states_32, dim=1)

                invert_1 = ~attention_mask_1
                all_masked_rows_1 = torch.all(invert_1, dim=2)
                hidden_states_34 = hidden_states_33.masked_fill(all_masked_rows_1, 0.0)

                hidden_states_35 = torch.conv1d(
                    hidden_states_34, l1_dw_conv_w, None, (1,), "same", (1,), 64
                )
                hidden_states_36 = F.batch_norm(
                    hidden_states_35,
                    l1_bn_mean,
                    l1_bn_var,
                    l1_bn_w,
                    l1_bn_b,
                    False,
                    0.01,
                    1e-05,
                )
                hidden_states_37 = F.silu(hidden_states_36)
                hidden_states_38 = torch.conv1d(
                    hidden_states_37, l1_pw_conv2_w, None, (1,), (0,), (1,), 1
                )
                conv_output_1 = hidden_states_38.transpose(1, 2)

                # Conv residual
                mul_18 = item_12 * hidden_states_30
                mul_19 = item_13 * conv_output_1
                hidden_states_39 = mul_18 + mul_19

                # Feed forward 2
                layer_norm_8 = F.layer_norm(
                    hidden_states_39, (64,), l1_norm_ff2_w, None, 1e-06
                )
                linear_16 = torch._C._nn.linear(layer_norm_8, l1_ff2_lin1_w, None)
                hidden_states_40 = F.silu(linear_16)
                hidden_states_41 = F.dropout(hidden_states_40, p=0.1, training=False)
                hidden_states_42 = torch._C._nn.linear(
                    hidden_states_41, l1_ff2_lin2_w, None
                )

                mul_20 = item_5 * hidden_states_39
                mul_21 = item_6 * hidden_states_42
                hidden_states_43 = mul_20 + mul_21

                hidden_states_44 = F.layer_norm(
                    hidden_states_43, (64,), l1_norm_out_w, None, 1e-06
                )

                # Final output norm
                hidden_states_45 = F.layer_norm(
                    hidden_states_44, (64,), out_norm_w, None, 1e-06
                )

                return (hidden_states_45,)

        def create_parameters(device="cuda", dtype=torch.float16):
            # [FUNCTION DOC] HELPER: Builds and returns a mapping from parameter name to a
            # random tensor for every weight the traced module takes, using the given device
            # and dtype (defaults cuda and float16). Rotary frequency is float32; scale and
            # residual weights are float64 scalars on the CPU; batch norm variances are made
            # positive.
            """Create all the parameters needed by the GraphModule."""
            params = {}

            # Subsampler parameters
            params["sub_dense0_w"] = torch.randn(64, 80, device=device, dtype=dtype)
            params["sub_dense0_b"] = torch.randn(64, device=device, dtype=dtype)
            params["sub_conv0_w"] = torch.randn(64, 64, 5, device=device, dtype=dtype)
            params["sub_conv0_b"] = torch.randn(64, device=device, dtype=dtype)
            params["sub_conv1_w"] = torch.randn(32, 64, 5, device=device, dtype=dtype)
            params["sub_conv1_b"] = torch.randn(32, device=device, dtype=dtype)
            params["sub_dense1_w"] = torch.randn(64, 32, device=device, dtype=dtype)
            params["sub_dense1_b"] = torch.randn(64, device=device, dtype=dtype)

            # Rotary embedding
            params["rot_inv_freq"] = torch.randn(8, device=device, dtype=torch.float32)
            params["rot_attn_scale"] = torch.tensor(
                1.0, device="cpu", dtype=torch.float64
            )

            # Layer 0 parameters
            params["l0_norm_ff1_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_ff1_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l0_ff1_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l0_ff_res0"] = torch.tensor(0.5, device="cpu", dtype=torch.float64)
            params["l0_ff_res1"] = torch.tensor(0.5, device="cpu", dtype=torch.float64)
            params["l0_norm_attn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_q_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_k_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_v_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_o_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l0_norm_conv_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_pw_conv1_w"] = torch.randn(
                128, 64, 1, device=device, dtype=dtype
            )
            params["l0_dw_conv_w"] = torch.randn(64, 1, 8, device=device, dtype=dtype)
            params["l0_bn_mean"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_bn_var"] = (
                torch.abs(torch.randn(64, device=device, dtype=dtype)) + 0.1
            )
            params["l0_bn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_bn_b"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_pw_conv2_w"] = torch.randn(64, 64, 1, device=device, dtype=dtype)
            params["l0_conv_res0"] = torch.tensor(
                0.5, device="cpu", dtype=torch.float64
            )
            params["l0_conv_res1"] = torch.tensor(
                0.5, device="cpu", dtype=torch.float64
            )
            params["l0_norm_ff2_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l0_ff2_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l0_ff2_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l0_norm_out_w"] = torch.randn(64, device=device, dtype=dtype)

            # Layer 1 parameters
            params["l1_norm_ff1_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_ff1_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l1_ff1_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l1_norm_attn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_q_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_k_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_v_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_o_w"] = torch.randn(64, 64, device=device, dtype=dtype)
            params["l1_norm_conv_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_pw_conv1_w"] = torch.randn(
                128, 64, 1, device=device, dtype=dtype
            )
            params["l1_dw_conv_w"] = torch.randn(64, 1, 8, device=device, dtype=dtype)
            params["l1_bn_mean"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_bn_var"] = (
                torch.abs(torch.randn(64, device=device, dtype=dtype)) + 0.1
            )
            params["l1_bn_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_bn_b"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_pw_conv2_w"] = torch.randn(64, 64, 1, device=device, dtype=dtype)
            params["l1_norm_ff2_w"] = torch.randn(64, device=device, dtype=dtype)
            params["l1_ff2_lin1_w"] = torch.randn(256, 64, device=device, dtype=dtype)
            params["l1_ff2_lin2_w"] = torch.randn(64, 256, device=device, dtype=dtype)
            params["l1_norm_out_w"] = torch.randn(64, device=device, dtype=dtype)

            # Output norm
            params["out_norm_w"] = torch.randn(64, device=device, dtype=dtype)

            return params

        torch.manual_seed(42)

        device = GPU_TYPE
        dtype = torch.float16

        # Create model and parameters
        model = GraphModule().eval()
        params = create_parameters(device, dtype)

        # Create example input
        batch_size = 13
        seq_len = 1024
        input_features = torch.randn(
            batch_size, seq_len, 80, device=device, dtype=dtype
        )
        compiled_model = torch.compile(model, fullgraph=True, dynamic=True)

        with torch.no_grad():
            eager_output = model(
                s25=batch_size,
                s70=seq_len,
                input_feat=input_features,
                **params,
            )
            compiled_output = compiled_model(
                s25=batch_size,
                s70=seq_len,
                input_feat=input_features,
                **params,
            )


@unittest.skipIf(
    not (HAS_CUDA_AND_TRITON and torch.cuda.get_device_capability()[0] >= 9)
    or torch.version.hip,
    "Requires Triton CUDA backend and CUDA compute capability >= 9.0. Not supported on ROCm",
)
@config.patch(
    {
        "triton.use_tensor_descriptor": True,
        "triton.enable_host_side_tma": True,
        "assume_aligned_inputs": True,
    }
)
@instantiate_parametrized_tests
class TritonHostSideTMATestCUDA(BlockDescriptorTestBase):
    # [CLASS DOC] Runs the full pointwise and reduction suite with host-side TMA, where tensor
    # descriptors are created in the kernel launcher instead of the kernel. Requires compute
    # capability 9 or higher, not ROCm. Test methods are copied from CommonTemplate below,
    # with selected skips and expected failures attached after the copy.
    """Run the full pointwise/reduction suite with host-side TMA.
    Block pointer count is skipped because host-side TMA creates
    descriptors in the launcher, not the kernel."""

    device = GPU_TYPE

    def _run_and_compare(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Overrides the base comparison helper to force the block
        # pointer count check off (sets that keyword to None) and delegates to the parent
        # implementation.
        kwargs["expected_num_block_pointers"] = None
        return super()._run_and_compare(*args, **kwargs)

    def test_host_tma_codegen_markers(self):
        # [FUNCTION DOC] Compiles an add of two bfloat16 1024x1024 tensors and asserts results
        # match eager, that the kernel body has no in-kernel descriptor creation, and that the
        # host descriptor argument metadata appears in the code.
        # Host-side TMA builds descriptors in the launcher, not the kernel: the
        # kernel body must have no in-kernel tl.make_tensor_descriptor, and the
        # inductor metadata must carry host_tma_descriptor_args.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Function under test; returns the sum a plus b.
            return a + b

        a = torch.randn(1024, 1024, device=self.device, dtype=torch.bfloat16)
        b = torch.randn(1024, 1024, device=self.device, dtype=torch.bfloat16)
        result, code_list = run_and_get_code(torch.compile(fn), a, b)
        self.assertTrue(torch.allclose(result, fn(a, b)))
        code = "\n".join(code_list)
        self.assertNotIn("tl.make_tensor_descriptor", code)
        self.assertIn("host_tma_descriptor_args", code)

    def test_misaligned_offset_disables_host_tma(self):
        # [FUNCTION DOC] Compiles an increment of a slice starting at element one of a 1025
        # element tensor (a four byte offset, not 16 byte aligned), asserts results match
        # eager and the code contains a plain load, since that buffer cannot use host-side
        # TMA.
        # A 4-byte (float32) storage offset is not 16-byte aligned, so the
        # misaligned input can't be host-TMA'd and falls back to a plain
        # tl.load. (Disabling is per-buffer: the aligned output may still use a
        # descriptor, so host_tma_descriptor_args can still appear.)
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test; drops the first element and adds
            # one.
            return x[1:] + 1

        x = torch.randn(1025, device=self.device)
        result, code_list = run_and_get_code(torch.compile(fn), x)
        self.assertTrue(torch.allclose(result, fn(x)))
        self.assertIn("tl.load", "\n".join(code_list))

    @config.patch("use_static_triton_launcher", True)
    def test_static_launcher_runs_for_host_tma(self):
        # [FUNCTION DOC] Runs with the static triton launcher enabled. Wraps the feature usage
        # recorder in the triton heuristics module with one that records calls, then compiles
        # silu on a bfloat16 1024x1024 tensor. Asserts host TMA descriptor metadata is
        # present, no in-kernel descriptor creation, the static launcher feature was recorded
        # as used, and results match.
        import torch._inductor.runtime.triton_heuristics as triton_heuristics

        feature_calls = []
        orig_set_feature_use = triton_heuristics.set_feature_use

        def tracking_set_feature_use(feature, usage):
            # [FUNCTION DOC] HELPER: Appends the feature and usage pair to the enclosing list
            # of recorded calls, then forwards to the original recorder and returns its
            # result.
            feature_calls.append((feature, usage))
            return orig_set_feature_use(feature, usage)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test; applies silu to x.
            return torch.nn.functional.silu(x)

        x = torch.randn(1024, 1024, device=self.device, dtype=torch.bfloat16)
        eager_out = fn(x)
        with mock.patch.object(
            triton_heuristics, "set_feature_use", tracking_set_feature_use
        ):
            compiled_out, code_list = run_and_get_code(torch.compile(fn), x)

        code = "\n".join(code_list)
        self.assertIn(
            "host_tma_descriptor_args",
            code,
            "host-side TMA descriptors were not generated for this kernel",
        )
        self.assertNotIn("tl.make_tensor_descriptor", code)
        self.assertIn(
            ("static_triton_launcher", True),
            feature_calls,
            "static Triton launcher path was not taken for host-side TMA kernel",
        )
        self.assertTrue(torch.allclose(compiled_out, eager_out))


test_torchinductor.copy_tests(CommonTemplate, TritonHostSideTMATestCUDA, GPU_TYPE)

# The copy_tests above generates GPU_TYPE-suffixed methods; the skip/xfail
# markers below reference the CUDA variants by name, so only apply them when
# running on CUDA (the class itself is skipped on other backends).
if GPU_TYPE == "cuda":
    # The (9, True) meta-test checks that _run_and_compare raises on wrong block
    # pointer counts. Host-side TMA disables this count check, so skip it.
    TritonHostSideTMATestCUDA.test_expected_num_block_pointers_expected_num_block_pointers_9_raises_True_cuda = unittest.skip(
        "block pointer count check is disabled for host-side TMA"
    )(
        TritonHostSideTMATestCUDA.test_expected_num_block_pointers_expected_num_block_pointers_9_raises_True_cuda
    )

    # Known TMA API limitations: these cases also fail for device-side TMA (they
    # carry @xfail_if_use_tensor_descriptor). For host-side TMA they either produce
    # different (still-correct) codegen that breaks the device-specific code asserts,
    # or hit the same descriptor constraints (e.g. the 16-byte last-dim minimum in
    # test_reduction_padded_output_tiling).
    _HOST_TMA_EXPECTED_FAILURES = [
        "test_boundary_check_block_multiple_False_ynumel_exceed_ygrid_size_False_include_z_True_cuda",
        "test_boundary_check_block_multiple_True_ynumel_exceed_ygrid_size_True_include_z_False_cuda",
        "test_pointwise_broadcast_nonzero_strides_prefer_nd_tiling_False_cuda",
        "test_pointwise_broadcast_nonzero_strides_prefer_nd_tiling_True_cuda",
        "test_pointwise_index_order_cuda",
        "test_reduction_padded_output_tiling_cuda",
    ]
    for _name in _HOST_TMA_EXPECTED_FAILURES:
        setattr(
            TritonHostSideTMATestCUDA,
            _name,
            unittest.expectedFailure(getattr(TritonHostSideTMATestCUDA, _name)),
        )

    # Dynamic shapes are not yet supported for host-side TMA (the launcher cannot
    # resolve symbolic block/shape dims). Tracked as a follow-up.
    TritonHostSideTMATestCUDA.test_dynamic_shapes_pointwise_nd_tiling_False_num_block_pointers_1_cuda = unittest.expectedFailure(
        TritonHostSideTMATestCUDA.test_dynamic_shapes_pointwise_nd_tiling_False_num_block_pointers_1_cuda
    )

    # Unlike the cases above (which also fail device-side), this one passes for
    # device-side TMA and non-TMA. Its im2col output store is emitted as a host-side
    # TMA tensordesc store, so the generated code has no tl.make_block_ptr and the
    # base block_descriptor_constructor_str assert does not hold. Numerics still
    # match (the _run_and_compare check passes before the code-string assert).
    TritonHostSideTMATestCUDA.test_ensure_integral_dims_and_strides_cuda = (
        unittest.expectedFailure(
            TritonHostSideTMATestCUDA.test_ensure_integral_dims_and_strides_cuda
        )
    )


class HostTMAHelperTest(InductorTestCase):
    # [CLASS DOC] Unit tests for host-side TMA helper functions that need no GPU.
    """Device-independent unit tests for host-side TMA helpers."""

    def test_host_tma_aligned_clone_fallback(self):
        # [FUNCTION DOC] Checks the alignment helper in the triton heuristics module: a tensor
        # whose base is 16 byte aligned is returned as the very same object, while a
        # misaligned slice (offset one element of float16) is returned as a new tensor with a
        # 16 byte aligned base and equal values.
        from torch._inductor.runtime.triton_heuristics import _host_tma_aligned

        # 16-byte-aligned base -> returned as-is (no clone).
        aligned = torch.randn(1024)
        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertIs(_host_tma_aligned(aligned, "aligned"), aligned)

        # Misaligned base -> cloned into an aligned buffer, values preserved.
        base = torch.randn(1024 + 8, dtype=torch.float16)
        misaligned = base[1:]
        self.assertNotEqual(misaligned.data_ptr() % 16, 0)
        cloned = _host_tma_aligned(misaligned, "misaligned")
        self.assertIsNot(cloned, misaligned)
        self.assertEqual(cloned.data_ptr() % 16, 0)
        self.assertTrue(torch.equal(cloned, misaligned))


@unittest.skipIf(
    not (HAS_CUDA_AND_TRITON and torch.cuda.get_device_capability()[0] >= 9)
    or torch.version.hip,
    "Requires Triton CUDA backend and CUDA compute capability >= 9.0. Not supported on ROCm",
)
class TritonHostSideTMAConfigTestCUDA(InductorTestCase):
    # [CLASS DOC] Config tests for host-side TMA, needing compute capability 9 or higher and
    # not ROCm.
    @config.patch(
        {
            "triton.enable_host_side_tma": True,
            "triton.use_tensor_descriptor": False,
        }
    )
    def test_enable_host_side_tma_without_prereqs_warns(self):
        # [FUNCTION DOC] With host-side TMA on but tensor descriptors off, compiling a simple
        # add must emit a UserWarning matching the text no effect, still produce correct
        # results, and not emit host descriptor metadata.
        # enable_host_side_tma only selects the descriptor flavor; without
        # use_tensor_descriptor + assume_aligned_inputs it has no effect and
        # should warn rather than silently no-op.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test; returns x plus one.
            return x + 1

        x = torch.randn(1024, device=GPU_TYPE)
        with self.assertWarnsRegex(UserWarning, "no effect"):
            result, code_list = run_and_get_code(torch.compile(fn), x)
        self.assertTrue(torch.allclose(result, fn(x)))
        self.assertNotIn("host_tma_descriptor_args", "\n".join(code_list))


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_GPU or TRITON_HAS_CPU:
        run_tests(needs="filelock")
