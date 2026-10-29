# mypy: ignore-errors

# [MODULE DOC] Shared fixtures for tests of Triton integration: unittest skip decorators
# (requires_cuda_and_triton, requires_xpu_and_triton, requires_gpu_and_triton, requires_gpu)
# built from capability flags, and, only when Triton is importable, a library of small jit
# and autotuned kernels (elementwise add/sub/mul, TMA descriptor variants, block pointers,
# inline assembly, docstring edge cases), a config generator and a descriptor shim.
# Importing evaluates the capability flags and, if Triton exists, imports torch and triton
# and defines (but does not run) all kernels.

import unittest

from torch.testing._internal.inductor_utils import (
    HAS_CUDA_AND_TRITON,
    HAS_GPU,
    HAS_MTIA_AND_TRITON,
    HAS_XPU_AND_TRITON,
)
from torch.utils._triton import has_triton


requires_cuda_and_triton = unittest.skipUnless(
    HAS_CUDA_AND_TRITON, "requires cuda and triton"
)
requires_xpu_and_triton = unittest.skipUnless(
    HAS_XPU_AND_TRITON, "requires xpu and triton"
)
requires_gpu_and_triton = unittest.skipUnless(
    HAS_XPU_AND_TRITON or HAS_CUDA_AND_TRITON or HAS_MTIA_AND_TRITON,
    "requires gpu and triton",
)
requires_gpu = unittest.skipUnless(HAS_GPU, "requires gpu")

if has_triton():
    import triton
    from triton import language as tl

    import torch

    def _get_strange_configs() -> list[triton.Config]:
        # [FUNCTION DOC] HELPER: Returns two autotune configs for the matmul-style kernel
        # with unusual tile sizes (one small, one large); on AMD builds each also carries
        # extra AMD-specific tuning options.
        if torch.version.hip:
            configs = [
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 16,
                        "BLOCK_SIZE_N": 16,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 4,
                        "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 3,
                        "kpack": 2,
                    },
                    num_stages=4,
                    num_warps=4,
                ),
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 64,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 4,
                        "matrix_instr_nonkdim": 16,
                        "waves_per_eu": 3,
                        "kpack": 2,
                    },
                    num_stages=4,
                    num_warps=4,
                ),
            ]
        else:
            configs = [
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 16,
                        "BLOCK_SIZE_N": 16,
                        "BLOCK_SIZE_K": 16,
                        "GROUP_SIZE_M": 4,
                    },
                    num_stages=4,
                    num_warps=4,
                ),
                triton.Config(
                    {
                        "BLOCK_SIZE_M": 128,
                        "BLOCK_SIZE_N": 64,
                        "BLOCK_SIZE_K": 32,
                        "GROUP_SIZE_M": 8,
                    },
                    num_stages=4,
                    num_warps=4,
                ),
            ]
        return configs

    # Define here so that multiple tests can take advantage of it
    @triton.jit
    def add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel; each program adds a block of the two input
        # buffers elementwise and stores into the output, masking indices past n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def sub_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel computing the elementwise difference of the
        # first and second input per block, masked by n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x - y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_with_optional_param(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        ARGS_PASSED: "tl.constexpr",
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel that adds both inputs when the compile-time
        # flag equals the text for two arguments, else copies only the first input.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        if ARGS_PASSED == "two":
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
        else:
            output = x
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_with_none_param_and_equal_to_1_arg(
        in_ptr0,
        in_ptr1,  # in_ptr1 could be None
        out_ptr,
        n_elements,
        stride,
        ARGS_PASSED: "tl.constexpr",
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel like the optional-param add, but the first
        # input and output are indexed with a runtime stride (may equal 1) and the second
        # input may be absent (None) when the flag says only one argument is passed.
        # KNOWN ISSUE: the second input is indexed without the stride, unlike the first.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets * stride, mask=mask)
        if ARGS_PASSED == "two":
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
        else:
            output = x
        tl.store(out_ptr + offsets * stride, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Autotuned elementwise add kernel; autotuner tries four
        # block-size/stage/warp configs (no key). Body matches the plain add kernel.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        ],
        key=[],
    )
    @triton.jit
    def sub_kernel_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Autotuned elementwise subtract kernel with the same four
        # configs as the autotuned add kernel.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x - y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 16}, num_stages=2, num_warps=2),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_autotuned_weird_param_order(
        in_ptr0,
        in_ptr1,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
        out_ptr,
    ):
        # [FUNCTION DOC] HELPER: Autotuned elementwise add kernel whose output pointer
        # parameter comes after the compile-time block size, to exercise parameter-order
        # handling. Single config.
        # out_ptr is after an autotuned param that's declared as tl.constexpr.
        # This param ordering can create bugs if not handled correctly.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 128, "BLOCK_SIZE_Y": 128}, num_stages=4, num_warps=4
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=3, num_warps=8
            ),
            triton.Config(
                {"BLOCK_SIZE_X": 64, "BLOCK_SIZE_Y": 64}, num_stages=4, num_warps=4
            ),
        ],
        key=[],
    )
    @triton.jit
    def add_kernel_2d_autotuned(
        in_ptr0,
        in_ptr1,
        out_ptr,
        x_elements,
        y_elements,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Autotuned 2D kernel; per tile it loads the input at a
        # position and at its transposed-index position (with sizes x_elements and
        # y_elements), adds them and stores to the first position. Masked at the bounds.
        xoffset = tl.program_id(0) * BLOCK_SIZE_X
        xindex = xoffset + tl.arange(0, BLOCK_SIZE_X)[:, None]
        xmask = xindex < x_elements
        yoffset = tl.program_id(1) * BLOCK_SIZE_Y
        yindex = yoffset + tl.arange(0, BLOCK_SIZE_Y)[None, :]
        ymask = yindex < y_elements
        x1 = xindex
        y0 = yindex
        tmp0 = tl.load(in_ptr0 + (x1 + (x_elements * y0)), xmask & ymask)
        tmp1 = tl.load(in_ptr0 + (y0 + (y_elements * x1)), xmask & ymask)
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr + (x1 + (x_elements * y0)), tmp2, xmask & ymask)

    def _dummy_early_config_prune(configs, *_, **__):
        # [FUNCTION DOC] HELPER: Pruning hook for the autotuner that returns all configs
        # unchanged and ignores other arguments.
        return configs

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
            triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
        ],
        key=[],
        warmup=10,
        rep=20,
        prune_configs_by={"early_config_prune": _dummy_early_config_prune},
    )
    @triton.jit
    def add_kernel_autotuned_with_unsupported_args(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Elementwise add kernel whose autotuner uses options (warmup,
        # repetitions, config pruning hook) that tooling may not support.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_with_scaling(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        scaling_factor,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel storing the sum of the two inputs multiplied
        # by a scalar scaling factor, masked by n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = (x + y) * scaling_factor
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_with_tma_1d_old_api(
        in_desc_ptr0,
        in_desc_ptr1,
        out_desc_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel adding two 1D float32 blocks fetched through
        # descriptors of the experimental TMA interface and storing through the output descriptor.
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE

        a = tl._experimental_descriptor_load(
            in_desc_ptr0,
            [offset],
            [BLOCK_SIZE],
            tl.float32,
        )
        b = tl._experimental_descriptor_load(
            in_desc_ptr1,
            [offset],
            [BLOCK_SIZE],
            tl.float32,
        )

        output = a + b

        tl._experimental_descriptor_store(
            out_desc_ptr,
            output,
            [offset],
        )

    @triton.jit
    def add_kernel_with_tma_2d_old_api(
        in_desc_ptr0,
        in_desc_ptr1,
        out_desc_ptr,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: 2D version of the experimental-descriptor add kernel; each
        # program handles one float32 tile located by its two program indices.
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        offset_x = pid_x * BLOCK_SIZE_X
        offset_y = pid_y * BLOCK_SIZE_Y

        x = tl._experimental_descriptor_load(
            in_desc_ptr0,
            [offset_x, offset_y],
            [BLOCK_SIZE_X, BLOCK_SIZE_Y],
            tl.float32,
        )
        y = tl._experimental_descriptor_load(
            in_desc_ptr1,
            [offset_x, offset_y],
            [BLOCK_SIZE_X, BLOCK_SIZE_Y],
            tl.float32,
        )

        output = x + y

        tl._experimental_descriptor_store(
            out_desc_ptr,
            output,
            [offset_x, offset_y],
        )

    @triton.jit
    def add_kernel_with_tma_1d_new_api(
        in_desc_ptr0,
        in_desc_ptr1,
        out_desc_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: 1D add kernel using the tensor-descriptor load and store
        # interface; block shape comes from the descriptors.
        pid = tl.program_id(axis=0)
        offset = pid * BLOCK_SIZE

        a = tl.load_tensor_descriptor(
            in_desc_ptr0,
            [offset],
        )
        b = tl.load_tensor_descriptor(
            in_desc_ptr1,
            [offset],
        )

        output = a + b

        tl.store_tensor_descriptor(
            out_desc_ptr,
            [offset],
            output,
        )

    @triton.jit
    def add_kernel_with_tma_2d_new_api(
        in_desc_ptr0,
        in_desc_ptr1,
        out_desc_ptr,
        BLOCK_SIZE_X: "tl.constexpr",
        BLOCK_SIZE_Y: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: 2D add kernel using the tensor-descriptor load and store
        # interface, one tile per program.
        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        offset_x = pid_x * BLOCK_SIZE_X
        offset_y = pid_y * BLOCK_SIZE_Y

        x = tl.load_tensor_descriptor(
            in_desc_ptr0,
            [offset_x, offset_y],
        )
        y = tl.load_tensor_descriptor(
            in_desc_ptr1,
            [offset_x, offset_y],
        )

        output = x + y

        tl.store_tensor_descriptor(
            out_desc_ptr,
            [offset_x, offset_y],
            output,
        )

    @triton.jit
    def add_kernel_on_device_tma_old_api(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        workspace,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel that builds three experimental TMA descriptors on
        # the device inside a scratch workspace buffer (spaced 128 bytes apart) for inputs a,
        # b and output c of shape m by n, then adds one float32 tile of a and b into c.
        a_desc_ptr = workspace
        b_desc_ptr = workspace + 128
        c_desc_ptr = workspace + 256
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=a_desc_ptr,
            global_address=a_ptr,
            load_size=[BLOCK_SIZE, BLOCK_SIZE],
            global_size=[m, n],
            element_ty=a_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=b_desc_ptr,
            global_address=b_ptr,
            load_size=[BLOCK_SIZE, BLOCK_SIZE],
            global_size=[m, n],
            element_ty=b_ptr.dtype.element_ty,
        )
        tl.extra.cuda.experimental_device_tensormap_create2d(
            desc_ptr=c_desc_ptr,
            global_address=c_ptr,
            load_size=[BLOCK_SIZE, BLOCK_SIZE],
            global_size=[m, n],
            element_ty=c_ptr.dtype.element_ty,
        )

        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(a_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(b_desc_ptr)
        tl.extra.cuda.experimental_tensormap_fenceproxy_acquire(c_desc_ptr)

        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        offset_x = pid_x * BLOCK_SIZE
        offset_y = pid_y * BLOCK_SIZE

        # Load data using the tensor descriptors
        a = tl._experimental_descriptor_load(
            a_desc_ptr,
            [offset_x, offset_y],
            [BLOCK_SIZE, BLOCK_SIZE],
            tl.float32,
        )
        b = tl._experimental_descriptor_load(
            b_desc_ptr,
            [offset_x, offset_y],
            [BLOCK_SIZE, BLOCK_SIZE],
            tl.float32,
        )

        # Perform addition
        output = a + b

        # Store the result
        tl._experimental_descriptor_store(
            c_desc_ptr,
            output,
            [offset_x, offset_y],
        )

    @triton.jit
    def add_kernel_on_device_tma_new_api(
        a_ptr,
        b_ptr,
        c_ptr,
        m,
        n,
        workspace,  # unused but left here to match the old API kernel
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Same computation as the old-interface on-device kernel but
        # creates row-major tensor descriptors with the new in-kernel constructor; the
        # workspace argument is unused.
        # Create tensor descriptors using the new API
        a_desc = tl.make_tensor_descriptor(
            base=a_ptr,
            shape=[m, n],
            strides=[n, 1],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE],
        )
        b_desc = tl.make_tensor_descriptor(
            base=b_ptr,
            shape=[m, n],
            strides=[n, 1],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE],
        )
        c_desc = tl.make_tensor_descriptor(
            base=c_ptr,
            shape=[m, n],
            strides=[n, 1],
            block_shape=[BLOCK_SIZE, BLOCK_SIZE],
        )

        pid_x = tl.program_id(axis=0)
        pid_y = tl.program_id(axis=1)
        offset_x = pid_x * BLOCK_SIZE
        offset_y = pid_y * BLOCK_SIZE

        # Load data using the tensor descriptors with the new API
        a = tl.load_tensor_descriptor(
            a_desc,
            [offset_x, offset_y],
        )
        b = tl.load_tensor_descriptor(
            b_desc,
            [offset_x, offset_y],
        )

        # Perform addition
        output = a + b

        # Store the result with the new API
        tl.store_tensor_descriptor(
            c_desc,
            [offset_x, offset_y],
            output,
        )

    @triton.jit
    def mul2_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel writing twice the input into the output,
        # masked by n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = 2 * x
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def mul2_inplace_kernel(
        ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Triton kernel doubling a buffer in place, masked by n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(ptr + offsets, mask=mask)
        output = 2 * x
        tl.store(ptr + offsets, output, mask=mask)

    @triton.jit
    def zero_negs(x):
        # [FUNCTION DOC] HELPER: Jit helper returning the value where it is non-negative
        # and zero elsewhere.
        return tl.where(x >= 0, x, 0)

    @triton.jit
    def indirection_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
        ACTIVATION: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel calling other kernels depending on a compile-time
        # name: the in-place doubling kernel or the add kernel (both on the input buffer),
        # then copies the input block to the output. Tests nested kernel calls.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        if ACTIVATION == "mul2_inplace_kernel":
            mul2_inplace_kernel(in_ptr0, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        elif ACTIVATION == "add_kernel":
            add_kernel(in_ptr0, in_ptr0, out_ptr, n_elements, BLOCK_SIZE=BLOCK_SIZE)
        x = tl.load(in_ptr0 + offsets, mask=mask)
        tl.store(out_ptr + offsets, x, mask=mask)

    @triton.jit
    def double_strided_kernel(
        in_ptr,
        out_ptr,
        in_y_stride,
        out_y_stride,
        X_BLOCK_SIZE: "tl.constexpr",
        Y_BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel copying a 2D tile from the input to the output with
        # separate row strides, multiplying values by two.
        xid = tl.program_id(axis=0)
        yid = tl.program_id(axis=1)
        x_start = xid * X_BLOCK_SIZE
        y_start = yid * Y_BLOCK_SIZE
        x_offsets = x_start + tl.arange(0, X_BLOCK_SIZE)
        y_offsets = y_start + tl.arange(0, Y_BLOCK_SIZE)
        src_offsets = y_offsets[:, None] * in_y_stride + x_offsets[None, :]
        dst_offsets = y_offsets[:, None] * out_y_stride + x_offsets[None, :]
        src = tl.load(in_ptr + src_offsets)
        tl.store(out_ptr + dst_offsets, src * 2.0)

    @triton.jit
    def inline_asm_kernel_is_pure_true(
        X, Y, Z, n: "tl.constexpr", BLOCK: "tl.constexpr"
    ):
        # [FUNCTION DOC] HELPER: Kernel applying an inline PTX funnel-shift instruction to
        # two loaded blocks and a constant shift amount, marked side-effect free (pure).
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        s = tl.full([BLOCK], n, tl.int32)
        z = tl.inline_asm_elementwise(
            "shf.l.wrap.b32 $0, $1, $2, $3;",
            "=r,r, r, r",
            [x, y, s],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )
        tl.store(Z + tl.arange(0, BLOCK), z)

    @triton.jit
    def inline_asm_kernel_is_pure_false(
        X, Y, Z, n: "tl.constexpr", BLOCK: "tl.constexpr"
    ):
        # [FUNCTION DOC] HELPER: Same as the pure inline-assembly kernel but the assembly is
        # declared as possibly having side effects (not pure).
        x = tl.load(X + tl.arange(0, BLOCK))
        y = tl.load(Y + tl.arange(0, BLOCK))
        s = tl.full([BLOCK], n, tl.int32)
        z = tl.inline_asm_elementwise(
            "shf.l.wrap.b32 $0, $1, $2, $3;",
            "=r,r, r, r",
            [x, y, s],
            dtype=tl.int32,
            is_pure=False,
            pack=1,
        )
        tl.store(Z + tl.arange(0, BLOCK), z)

    @triton.jit
    def add_kernel_with_block_ptr(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC] HELPER: 1D add kernel using block pointers for both inputs and the
        # output, with boundary checking on the single dimension.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        y = tl.load(
            tl.make_block_ptr(
                base=y_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            boundary_check=[0],
        )
        output = x + y
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements],
                strides=[1],
                offsets=[block_start],
                block_shape=[BLOCK_SIZE],
                order=[0],
            ),
            output,
            boundary_check=[0],
        )

    @triton.jit
    def kernel_with_block_ptr_2d(
        x_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        # [FUNCTION DOC] HELPER: Copy kernel via block pointers over a column-shaped 2D view
        # (n_elements rows, one column) with boundary checking on the first dimension.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        x = tl.load(
            tl.make_block_ptr(
                base=x_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            boundary_check=[0],
        )
        output = x
        tl.store(
            tl.make_block_ptr(
                base=output_ptr,
                shape=[n_elements, 1],
                strides=[1, 1],
                offsets=[block_start, 0],
                block_shape=[BLOCK_SIZE, 1],
                order=[1, 0],
            ),
            output,
            boundary_check=[0],
        )

    from triton.language import load, store

    @triton.jit
    def add_kernel_with_import(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Elementwise add kernel that calls the load and store
        # functions imported directly by name from the language module.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = load(in_ptr0 + offsets, mask=mask)
        y = load(in_ptr1 + offsets, mask=mask)
        output = x + y
        store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def cond_op_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel with a data-independent branch on the program index:
        # program zero stores the sum of inputs, all other programs store the product.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        if tl.program_id(0) == 0:
            output = x + y
        else:
            output = x * y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def atomic_add_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel that atomically accumulates the sum of the inputs
        # into the output buffer (read-modify-write), masked by n_elements.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.atomic_add(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_4_times_kernel(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel storing the sum of the inputs four times to the same
        # output: twice in a counted loop and twice in a countdown while loop. Exercises loops.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        for _ in range(2):
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)
        i = 2
        while i > 0:
            i -= 1
            output = x + y
            tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def add_kernel_out_of_order_fn2(
        in_ptr0,
        in_ptr1,
        n_elements,
        out_ptr,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Elementwise add kernel with the output pointer declared
        # after the element count, to test non-standard parameter order.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        y = tl.load(in_ptr1 + offsets, mask=mask)
        output = x + y
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE_M": 16,
                    "BLOCK_SIZE_N": 16,
                    "BLOCK_SIZE_K": 16,
                    "GROUP_SIZE_M": 4,
                },
                num_stages=4,
                num_warps=4,
            ),
            triton.Config(
                {
                    "BLOCK_SIZE_M": 128,
                    "BLOCK_SIZE_N": 64,
                    "BLOCK_SIZE_K": 32,
                    "GROUP_SIZE_M": 8,
                },
                num_stages=4,
                num_warps=4,
            ),
        ],
        key=["M_ptr", "N", "K"],
    )
    @triton.jit
    def strange_config_matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M_ptr,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        # [FUNCTION DOC] HELPER: Autotuned simplified grouped matmul kernel used to test
        # unusual configs: the row count is read from device memory; when it is zero, tiles
        # with large row blocks assume 4096 rows and run, while small-block configs return
        # early (so the bad config is fast). Output is half precision.
        # This is a simplified matmul from Triton tutorial.
        pid = tl.program_id(axis=0)
        M = tl.load(M_ptr)
        if M == 0 and BLOCK_SIZE_M > 32:
            # This will run the full matmul if BLOCK_SIZE_M > 32
            M = 4096
        elif M == 0:
            # This directly returns, which will cut short the bad config of 16-block size.
            return
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        a_ptrs = a_ptr + (offs_am[:, None] + offs_k[None, :])
        b_ptrs = b_ptr + (offs_k[:, None] + offs_bn[None, :])

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
            accumulator = tl.dot(a, b, accumulator)
            a_ptrs += BLOCK_SIZE_K
            b_ptrs += BLOCK_SIZE_K
        c = accumulator.to(tl.float16)

        offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptrs = c_ptr + offs_cm[:, None] + offs_cn[None, :]
        c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    @triton.jit
    def kernel_with_docstring_double_quotes(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        # [FUNCTION DOC] HELPER: Kernel filling the output with ones, masked by numel; its
        # docstring contains triple double quotes to test that codegen sanitizes docstrings.
        """
        This kernel contains a triple-quote docstring w/ double quotes.
        Make sure that codegen sanitizes the docstring.
        """
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        ones = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        tl.store(out_ptr + offsets, ones, mask=offsets < numel)

    @triton.jit
    def kernel_with_docstring_single_quotes(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        # [FUNCTION DOC] HELPER: Same fill-with-ones kernel, but the docstring uses triple
        # single quotes and mentions double quotes inside.
        '''
        This kernel contains a triple-quote docstring w/ single quotes
        Make sure that codegen sanitizes the docstring.
        To prevent it from being linted to double quotes: """!!!"""
        '''
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        ones = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        tl.store(out_ptr + offsets, ones, mask=offsets < numel)

    @triton.jit
    def kernel_with_backslash_in_docstring(out_ptr, numel, BLOCK_SIZE: tl.constexpr):
        # [FUNCTION DOC] HELPER: Same fill-with-ones kernel whose docstring contains
        # backslash escape sequences, to test escaping in generated code.
        """Docstring with literal backslash escapes: \n \t \\"""
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        ones = tl.full([BLOCK_SIZE], 1.0, dtype=tl.float32)
        tl.store(out_ptr + offsets, ones, mask=offsets < numel)

    @triton.jit
    def kernel_inline_asm_double_quotes(
        in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
    ):
        # [FUNCTION DOC] HELPER: Kernel applying a multi-line NVIDIA assembly snippet
        # (cosine then exponential approximation) to loaded floats; the snippet is a
        # triple double quoted string. Stores masked by numel.
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        data = tl.load(in_ptr + offsets, mask=offsets < numel)
        cos_pow = tl.inline_asm_elementwise(
            asm="""
            {
                cos.approx.f32 $0, $1;
                ex2.approx.f32 $0, $0;
            }
                """,
            constraints=("=r, r"),
            args=[data],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offsets, cos_pow, mask=offsets < numel)

    @triton.jit
    def kernel_inline_asm_single_quotes(
        in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
    ):
        # [FUNCTION DOC] HELPER: Same as the double-quoted inline-assembly kernel but the
        # snippet uses triple single quotes and contains a comment line.
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        data = tl.load(in_ptr + offsets, mask=offsets < numel)
        cos_pow = tl.inline_asm_elementwise(
            asm='''
            {
                // double quotes to pacify the linter """!!!"""
                cos.approx.f32 $0, $1;
                ex2.approx.f32 $0, $0;
            }
                ''',
            constraints=("=r, r"),
            args=[data],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offsets, cos_pow, mask=offsets < numel)

    @triton.jit
    def kernel_inline_asm_rocm_double_quotes(
        in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
    ):
        # [FUNCTION DOC] HELPER: AMD variant of the inline-assembly kernel (sine then
        # exponential instructions) using a triple double quoted string.
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        data = tl.load(in_ptr + offsets, mask=offsets < numel)
        cos_pow = tl.inline_asm_elementwise(
            asm="""
            v_sin_f32 $0, $1
            v_exp_f32 $0, $0
                """,
            constraints=("=v, v"),
            args=[data],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offsets, cos_pow, mask=offsets < numel)

    @triton.jit
    def kernel_inline_asm_rocm_single_quotes(
        in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr
    ):
        # [FUNCTION DOC] HELPER: Intended single-quote variant of the AMD assembly kernel.
        # KNOWN ISSUE: despite its name it uses triple double quotes, identical to the
        # double-quote variant.
        pid = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE) + pid * BLOCK_SIZE
        data = tl.load(in_ptr + offsets, mask=offsets < numel)
        cos_pow = tl.inline_asm_elementwise(
            asm="""
            v_sin_f32 $0, $1
            v_exp_f32 $0, $0
                """,
            constraints=("=v, v"),
            args=[data],
            dtype=tl.float32,
            is_pure=True,
            pack=1,
        )
        tl.store(out_ptr + offsets, cos_pow, mask=offsets < numel)

    @triton.jit
    def add_kernel_with_boolean_param(
        in_ptr0,
        in_ptr1,
        out_ptr,
        n_elements,
        add_xy,  # boolean param
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel storing the sum of the inputs if the boolean runtime
        # argument is true, otherwise just the first input.
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        if add_xy:
            y = tl.load(in_ptr1 + offsets, mask=mask)
            output = x + y
        else:
            output = x
        tl.store(out_ptr + offsets, output, mask=mask)

    @triton.jit
    def masked_add_kernel_with_bool_tensor(
        in_ptr0,
        in_ptr1,
        mask_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        # [FUNCTION DOC] HELPER: Kernel that stores the sum of the inputs where a boolean
        # tensor (loaded as integers) is set, and just the first input elsewhere.
        """Kernel that loads a bool tensor and uses it as a mask."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        valid = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=valid)
        y = tl.load(in_ptr1 + offsets, mask=valid)
        keep = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
        output = tl.where(keep, x + y, x)
        tl.store(out_ptr + offsets, output, mask=valid)

    # support the old (experimental) and new (tensor_descriptor) APIs
    def create_tensor_descriptor_shim(
        tensor, block_sizes: list[int], new_api: bool = True
    ):
        # [FUNCTION DOC] HELPER: Builds a TMA descriptor for a tensor and block sizes: a
        # tensor-descriptor object with the new interface, or, with the old one, a 1D or 2D
        # experimental descriptor from data pointer, sizes, block sizes and element size.
        # Raises AssertionError if the old path gets more than two block sizes (or zero).
        if new_api:
            return triton.tools.tensor_descriptor.TensorDescriptor.from_tensor(
                tensor, block_sizes
            )
        else:
            if len(block_sizes) == 1:
                return triton.tools.experimental_descriptor.create_1d_tma_descriptor(
                    tensor.data_ptr(),
                    tensor.size(0),
                    block_sizes[0],
                    tensor.element_size(),
                )
            else:
                if len(block_sizes) != 2:
                    raise AssertionError(
                        f"Expected len(block_sizes) == 2, got {len(block_sizes)}"
                    )
                return triton.tools.experimental_descriptor.create_2d_tma_descriptor(
                    tensor.data_ptr(),
                    tensor.size(0),
                    tensor.size(1),
                    block_sizes[0],
                    block_sizes[1],
                    tensor.element_size(),
                )
