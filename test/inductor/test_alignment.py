# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor tests for memory alignment handling of inputs and outputs.
# Defines a device-generic CommonTemplate whose tests are copied into CPU and GPU test classes
# at import time via copy_tests, plus a helper checking aliased outputs of misaligned inputs.
# Skips the whole module (or exits when run as a script) if test_torchinductor raises SkipTest.
import contextlib
import sys
import unittest

import torch
import torch._functorch.config as functorch_config
from torch._inductor import config
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MACOS_VERSION,
    parametrize,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, RUN_CPU, RUN_GPU


try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
except unittest.SkipTest:
    if __name__ == "__main__":
        sys.exit(0)
    raise

TestCase = test_torchinductor.TestCase
check_model = test_torchinductor.check_model
check_model_gpu = test_torchinductor.check_model_gpu
skip_if_cpp_wrapper = test_torchinductor.skip_if_cpp_wrapper
copy_tests = test_torchinductor.copy_tests
define_custom_op_for_test = test_torchinductor.define_custom_op_for_test


def check_alias_of_misaligned_input(self, fn):
    # [FUNCTION DOC] HELPER: Shared check for a regression where, after the runtime alignment check
    # cloned a misaligned input, an aliased output was rebuilt from the wrong storage offset. Args:
    # self is the test case (needs a device attribute); fn takes one tensor and returns a pair
    # whose second element aliases the input. Skips the test (SkipTest) on non-GPU devices.
    # Compiles fn, runs it once on an aligned dynamic-shape tensor, then on a misaligned slice of a
    # bigger tensor, and asserts the aliased output matches eager in storage offset, pointer,
    # and values.
    # https://github.com/pytorch/pytorch/pull/191002: when inductor's runtime
    # alignment check clones a misaligned input, gen_alias_from_base used to
    # apply the clone's storage_offset to the original input, returning the
    # wrong slice of the base tensor.
    if not torch._inductor.utils.is_gpu(self.device):
        raise unittest.SkipTest("runtime alignment cloning is GPU-only")
    fn_c = torch.compile(fn)

    # Compile with an aligned input; the graph then assumes aligned inputs
    # and only checks (and clones) at runtime.
    a = torch.randn(1024, device=self.device)
    torch._dynamo.mark_dynamic(a, 0)
    fn_c(a)

    # Run with a misaligned view of a fresh base: storage_offset is not
    # guarded on, so this hits the same graph but takes the clone path.
    base = torch.randn(1025, device=self.device)
    y = base[1:]
    _, view_out = fn_c(y)

    _, expected = fn(y)
    self.assertEqual(view_out.storage_offset(), expected.storage_offset())
    self.assertEqual(view_out.data_ptr(), expected.data_ptr())
    self.assertEqual(view_out, expected)


@instantiate_parametrized_tests
class CommonTemplate:
    # [CLASS DOC] Device-generic alignment tests; not a TestCase itself. Expects subclasses to
    # provide a device name and a common callable (check_model or check_model_gpu) that compiles
    # and compares a function against eager. Test methods are copied into the CPU and GPU
    # test classes below.
    def test_unaligned_input(self):
        # [FUNCTION DOC] Compiles relu on a 1-D float tensor that is a view with a nonzero storage
        # offset (so it is not 16-byte aligned) and compares against eager. On MPS with macOS older
        # than 15 an AssertionError is expected instead of success.
        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; applies relu to its input.
            return torch.nn.functional.relu(x)

        x = torch.randn(1024 + 16, device=self.device)[1:-15]
        # TODO (malfet): Investigate failures on MacOS-14
        with (
            contextlib.nullcontext()
            if self.device != "mps" or MACOS_VERSION >= 15.0
            else self.assertRaises(AssertionError)
        ):
            self.common(fn, (x,), check_lowp=False)

    def test_unaligned_input_2d(self):
        # [FUNCTION DOC] Same as the 1-D unaligned input test but for a 2-D tensor made by slicing
        # columns off a wider tensor, giving a misaligned storage offset with row padding.
        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; applies relu to its input.
            return torch.nn.functional.relu(x)

        x = torch.randn(1024, 1024 + 16, device=self.device)[:, 1:-15]
        self.common(fn, (x,), check_lowp=False)

    def test_alignment_without_custom_op(self):
        # [FUNCTION DOC] Compiles a function whose intermediate result is sliced inside the graph
        # (relu, scale by 3, slice, cosine) with no custom op, checking inductor handles the
        # unaligned internal view and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; relu, multiply by 3, slice off the ends,
            # then cosine.
            a = torch.nn.functional.relu(x)
            b = (3 * a)[1:-15]
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op(self):
        # [FUNCTION DOC] With implicit fallbacks enabled, registers a custom op that returns a
        # sliced (unaligned) result with a matching meta function, and checks a graph using it
        # compiles and agrees with eager.
        def slice1d(x):
            # [FUNCTION DOC] HELPER: Eager implementation of the custom op; scale by 3 and slice a
            # 1-D tensor.
            return (3 * x)[1:-15]

        def slice1d_meta(x):
            # [FUNCTION DOC] HELPER: Meta implementation of the custom op; returns an empty tensor
            # with the same slice applied, matching the eager result shape and offset.
            return torch.empty_like(x)[1:-15]

        define_custom_op_for_test("slice1d", slice1d, slice1d_meta)

        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; relu, the custom slice op, then cosine.
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice1d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True)
    def test_no_align_for_custom_op_2d(self):
        # [FUNCTION DOC] 2-D variant of the custom op test: registers a custom op that slices the
        # last dim, with a correct meta function, and checks agreement with eager.
        # Implicit fallbacks are enabled.
        def slice2d(x):
            # [FUNCTION DOC] HELPER: Eager implementation; scale by 3 and slice the last dimension.
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            # [FUNCTION DOC] HELPER: Meta implementation; empty tensor, same last-dim
            # slice.
            return torch.empty_like(x)[..., 1:-15]

        define_custom_op_for_test("slice2d", slice2d, slice2d_meta)

        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; relu, the custom slice op, then cosine.
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)
        self.common(fn, (x,), check_lowp=False)

    @config.patch(implicit_fallbacks=True, alignment_asserts=True)
    @skip_if_cpp_wrapper(
        "Inductor does not generate alignment assertion for cpp_wrapper right now"
    )
    def test_incorrect_meta_for_custom_op_2d(self):
        # [FUNCTION DOC] With alignment asserts enabled, registers a custom op whose meta function
        # gives a different slice (offset 0) than the eager one (offset 1). Expects the generated
        # runtime check to raise an AssertionError mentioning 16 byte alignment with
        # storage_offset=1 and itemsize=4. Skipped under cpp_wrapper (no such assertions
        # are generated there).
        def slice2d(x):
            # [FUNCTION DOC] HELPER: Eager implementation; scale by 3 and slice from offset 1.
            return (3 * x)[..., 1:-15]

        def slice2d_meta(x):
            # [FUNCTION DOC] HELPER: Deliberately wrong meta implementation; slices from offset 0
            # so the meta storage offset disagrees with the real result.
            return torch.empty_like(x)[..., 0:-16]

        define_custom_op_for_test("slice2d_incorrect_meta", slice2d, slice2d_meta)

        def fn(x):
            # [FUNCTION DOC] HELPER: Model under test; relu, the mismatched custom op, then cosine.
            a = torch.nn.functional.relu(x)
            b = torch.ops.test.slice2d_incorrect_meta(a)
            c = torch.cos(b)
            return c

        x = torch.randn(1024, 1024 + 16, device=self.device)

        expected_error = "Expect the tensor to be 16 bytes aligned. Fail due to storage_offset=1 itemsize=4"
        with self.assertRaisesRegex(AssertionError, expected_error):
            self.common(fn, (x,), check_lowp=False)

    def test_slice(self):
        # [FUNCTION DOC] Compiles a function that slices off the first element of a 1-D tensor and
        # adds one, comparing with eager.
        def f(x):
            # [FUNCTION DOC] HELPER: Model under test; drop first element then add one.
            return x[1:] + 1

        x = torch.randn(1025, device=self.device)
        self.common(f, (x,))

    def test_alias_of_misaligned_input(self):
        # [FUNCTION DOC] Regression test for aliased outputs of misaligned inputs where the alias
        # is a chunk of a dynamic dimension. Delegates to the shared helper check.
        # chunk on a dynamic dim gives the aliased output a symbolic
        # ViewMeta, so alias regeneration takes the as_strided fallback.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns input plus one and the second chunk of the input (an
            # alias).
            return x + 1, x.chunk(2)[1]

        check_alias_of_misaligned_input(self, fn)

    @functorch_config.patch(view_replay_for_aliased_outputs=False)
    def test_alias_of_misaligned_input_no_view_replay(self):
        # [FUNCTION DOC] Same regression check as above but with view replay for aliased outputs
        # disabled in functorch config, using a plain slice as the aliased output.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns input plus one and a slice of the input (an alias).
            return x + 1, x[1:]

        check_alias_of_misaligned_input(self, fn)

    def test_view_dtype_slice(self):
        # [FUNCTION DOC] Reinterprets a bfloat16 tensor as float32 (half the elements), slices off
        # the first element, adds one, and compares with eager without computing a float reference.
        def f(x):
            # [FUNCTION DOC] HELPER: Model under test; view as float32, slice off the first
            # element, add one.
            return x.view(dtype=torch.float32)[1:] + 1

        x = torch.randn(1025 * 2, dtype=torch.bfloat16, device=self.device)
        self.common(f, (x,), reference_in_float=False)

    @parametrize(
        "size",
        (
            # wrapper for size = 128: https://gist.github.com/shunting314/88f1e72957b9fc5e9826aaa346a0e652
            # ptx: https://gist.github.com/shunting314/eb657ee8821eef9f0685b7b91e2ad5c2
            # the ptx file uses ld.global.b32 to load input buffer
            128,
            # wrapper for size = 1024: https://gist.github.com/shunting314/d7f64e1f52f6b1e2ec25e1a51052ce43
            # ptx: https://gist.github.com/shunting314/a24ff7563bb6b04523d11b119ab0f2b2
            # the ptx file uses ld.global.v2.b32 to load input buffer
            1024,
            # wrapper for size = 1024 * 1024: https://gist.github.com/shunting314/016b95cf0b6e9a75c25f5c9d5ed0a2ba
            # ptx: https://gist.github.com/shunting314/360112a4893c759b114c12fc99958297
            # the ptx file uses ld.global.v4.b32 to load input buffer
            1024 * 1024,
        ),
    )
    def test_slice_view_dtype(self, size):
        # [FUNCTION DOC] Parametrized over input sizes 128, 1024 and 1M (which make the generated
        # kernel use different vector load widths). Slices a bfloat16 tensor by two elements, views
        # it as float32 and adds one, comparing against eager without a float reference.
        offset = 1

        def f(x):
            # [FUNCTION DOC] HELPER: Model under test; drop first two elements, view as float32,
            # add one.
            return x[2:].view(dtype=torch.float32) + 1

        x = torch.randn((size + offset) * 2, dtype=torch.bfloat16, device=self.device)
        self.common(f, (x,), reference_in_float=False)

    def test_Q4_K_dequantization(self):
        # [FUNCTION DOC] Reproduces an alignment problem in a Q4_K block dequantization pipeline
        # (splits, dtype views, bit shifts) on a uint8 tensor and compares compiled output with
        # eager using loose tolerances.
        """
        Test the alignment issue for Q4_K dequantization.
        """

        QK_K = 256
        K_SCALE_SIZE = 12

        def get_scale_min(scales):
            # [FUNCTION DOC] HELPER: Unpacks packed 6-bit scales and mins from the scale bytes of
            # each block. Takes a per-block uint8 scale tensor; returns a pair of tensors, each
            # with 8 values per block.
            n_blocks = scales.shape[0]
            scales = scales.view(torch.uint8)
            scales = scales.reshape((n_blocks, 3, 4))

            d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)

            sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
            min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)

            return (sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8)))

        def split_block_dims(blocks, *args):
            # [FUNCTION DOC] HELPER: Splits a 2-D block tensor along dim 1 into pieces of the given
            # sizes plus a final remainder piece; returns the tuple of pieces.
            n_max = blocks.shape[1]
            dims = list(args) + [n_max - sum(args)]
            return torch.split(blocks, dims, dim=1)

        def dequantize_blocks_Q4_K(blocks, block_size, type_size):
            # [FUNCTION DOC] HELPER: Dequantizes Q4_K blocks. Splits each block into the two half
            # precision scale factors, packed scales and quantized values, expands the 4-bit values
            # and returns a float tensor of shape blocks by QK_K. The block_size and type_size args
            # are unused.
            n_blocks = blocks.shape[0]

            d, dmin, scales, qs = split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
            d = d.view(torch.float16)
            dmin = dmin.view(torch.float16)

            sc, m = get_scale_min(scales)

            d = (d * sc).reshape((n_blocks, -1, 1))
            dm = (dmin * m).reshape((n_blocks, -1, 1))

            qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
                [0, 4], device=d.device, dtype=torch.uint8
            ).reshape((1, 1, 2, 1))
            qs = (qs & 0x0F).reshape((n_blocks, -1, 32))

            return (d * qs - dm).reshape((n_blocks, QK_K))

        data = torch.randint(
            0, 16, (18432, 1728), device=self.device, dtype=torch.uint8
        )

        def dequantize(data):
            # [FUNCTION DOC] HELPER: Model under test; reshapes the byte data into 144-byte blocks,
            # dequantizes them, and reshapes to 18432 by 3072.
            block_size, type_size = 256, 144
            rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
            n_blocks = rows.numel() // type_size
            blocks = rows.reshape((n_blocks, type_size))
            blocks = dequantize_blocks_Q4_K(blocks, block_size, type_size)
            return blocks.reshape(18432, 3072)

        self.common(dequantize, (data,), check_lowp=False, atol=1e-3, rtol=1e-3)


if RUN_CPU:

    class CpuTests(TestCase):
        # [CLASS DOC] CPU instantiation of the alignment tests; uses check_model and device cpu.
        common = check_model
        device = "cpu"

    copy_tests(CommonTemplate, CpuTests, "cpu")

if RUN_GPU:

    class GPUTests(TestCase):
        # [CLASS DOC] GPU instantiation of the alignment tests; uses check_model_gpu and the
        # detected GPU device type.
        common = check_model_gpu
        device = GPU_TYPE

    copy_tests(CommonTemplate, GPUTests, GPU_TYPE)

if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if RUN_CPU or RUN_GPU:
        run_tests()
