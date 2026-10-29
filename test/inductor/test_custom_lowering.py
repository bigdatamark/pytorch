# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for custom operator lowerings registered from outside the main inductor
# lowering table. It defines a private operator library with a jagged-to-padded-dense op, a tanh
# approximation op implemented via inline assembly, and an inline-assembly add op, each with a
# meta implementation, GPU eager implementations where relevant, and an inductor lowering. Tests
# compare eager and compiled results and inspect generated code. Registration happens in class
# setup, not at import. When run as a script with a CPU or GPU, it runs the tests with the
# filelock extra requirement.

from functools import partial
from unittest import skipIf

import torch
from torch._inductor import config
from torch._inductor.ir import Pointwise
from torch._inductor.lowering import make_fallback, make_pointwise, register_lowering
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import ops
from torch.testing._internal.common_utils import skipIfRocm, skipIfXpu
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_CPU,
    HAS_GPU,
    requires_gpu,
)


# These tests check issues for lowerings that aren't in the main pytorch repo
class TestCustomLowering(InductorTestCase):
    # [CLASS DOC] Test case that registers the test_inductor_ops operator library in setUpClass
    # (a definition library plus implementation libraries for the Meta, CUDA and XPU keys stored
    # as class attributes named impl_meta, impl_cuda and impl_xpu) and exercises the custom
    # lowerings of the ops defined there.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] Class setup: calls the base setup, creates the operator definition
        # library and one implementation library per device key in the device list (stored on the
        # class under impl_ plus the lowercase device name), then registers the jagged-to-padded-
        # dense op and the inline-assembly ops.
        super().setUpClass()
        cls.test_inductor_ops = torch.library.Library(  # noqa: SCOPED_LIBRARY
            "test_inductor_ops", "DEF"
        )
        cls.device_list = ["Meta", "CUDA", "XPU"]
        for device in cls.device_list:
            setattr(
                cls,
                "impl_" + device.lower(),
                torch.library.Library(  # noqa: SCOPED_LIBRARY
                    "test_inductor_ops", "IMPL", device
                ),
            )
        cls._register_jagged_to_padded_dense()
        cls._register_asm_op()

    @classmethod
    def tearDown(cls):
        # [FUNCTION DOC] HELPER: Declared as a class method but named like the per-test teardown;
        # it calls the base class tearDownClass.
        # KNOWN ISSUE: because of its name it runs after every test (not once per class), and it
        #   calls the class-level teardown of the base class each time.
        super().tearDownClass()

    @classmethod
    def _register_jagged_to_padded_dense(cls):
        # [FUNCTION DOC] HELPER: Defines the jagged_to_padded_dense op schema (input, offsets,
        # max sequence length, pad value) on the test library, then registers a meta
        # implementation, a CUDA and XPU eager implementation, and an inductor lowering with no
        # type promotion. Modifies global operator registries.
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define(
            "jagged_to_padded_dense(Tensor input, Tensor offsets, SymInt max_seq_len, Scalar pad_value) -> Tensor"
        )

        def j2pd_meta(inp, offsets, max_seq_len, pad_value):
            # [FUNCTION DOC] HELPER: Meta implementation: returns an uninitialized tensor of
            # shape batch count (offsets length minus one) by max sequence length by embedding
            # width, with the input's device and dtype.
            return torch.empty(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                device=inp.device,
                dtype=inp.dtype,
            )

        def j2pd_gpu(inp, offsets, max_seq_len, pad_value):
            # [FUNCTION DOC] HELPER: Eager GPU implementation: builds a result filled with the
            # pad value, then for each batch copies the rows between consecutive offsets from the
            # packed input into the start of that batch's slot. Returns the padded tensor.
            res = torch.full(
                (offsets.shape[0] - 1, max_seq_len, inp.shape[1]),
                pad_value,
                device=inp.device,
                dtype=inp.dtype,
            )
            for b in range(offsets.shape[0] - 1):
                for r in range(offsets[b + 1] - offsets[b]):
                    res[b][r] = inp[offsets[b] + r]
            return res

        def j2pd_lowering(inp, offsets, max_seq_len, pad_value):
            # [FUNCTION DOC] HELPER: Inductor lowering: makes loaders for offsets and input and
            # returns a pointwise node over the padded output shape. Each output element is a
            # masked load: the jagged row index is the batch's begin offset (via indirect
            # indexing) plus the sequence position, loaded only if it is below the batch's end
            # offset, otherwise the pad value.
            offsets_loader = offsets.make_loader()
            inp_loader = inp.make_loader()
            jagged_len = inp.get_size()[0]
            offsets_dtype = offsets.get_dtype()

            def inner_fn(index):
                # [FUNCTION DOC] HELPER: Per-element body of the lowering: unpacks batch,
                # sequence and embedding indices, computes begin and end offsets and the jagged
                # row index, and returns a masked load of the input that falls back to the pad
                # value past the end offset.
                batch_idx, seq_idx, emb_idx = index

                begin_idx = ops.indirect_indexing(
                    offsets_loader([batch_idx]),
                    jagged_len + 1,
                )
                end_idx = offsets_loader([batch_idx + 1])
                jagged_idx = begin_idx + seq_idx

                return ops.masked(
                    ops.lt(
                        ops.index_expr(jagged_idx, offsets_dtype),
                        end_idx,
                    ),
                    lambda: inp_loader([jagged_idx, emb_idx]),
                    pad_value,
                )

            return Pointwise.create(
                device=inp.get_device(),
                dtype=inp.get_dtype(),
                inner_fn=inner_fn,
                ranges=[offsets.get_size()[0] - 1, max_seq_len, inp.get_size()[1]],
            )

        register_lowering(
            torch.ops.test_inductor_ops.jagged_to_padded_dense, type_promotion_kind=None
        )(j2pd_lowering)

        cls.impl_meta.impl("jagged_to_padded_dense", j2pd_meta)
        cls.impl_cuda.impl("jagged_to_padded_dense", j2pd_gpu)
        cls.impl_xpu.impl("jagged_to_padded_dense", j2pd_gpu)

    @classmethod
    def _register_asm_op(cls):
        # [FUNCTION DOC] HELPER: Defines the tanh_approx and add_custom ops on the test library,
        # registers meta implementations and inductor lowerings that emit inline assembly (a tanh
        # approximation, and an add that uses ROCm assembly when running on HIP and a CUDA add
        # instruction otherwise). Modifies global operator registries.
        # Approximation of fbgemm.jagged_to_padded_dense_forward
        cls.test_inductor_ops.define("tanh_approx(Tensor input) -> Tensor")

        def tanh_approx_meta(inp):
            # [FUNCTION DOC] HELPER: Meta implementation of tanh_approx; returns
            # the regular tanh.
            return torch.tanh(inp)

        cls.impl_meta.impl("tanh_approx", tanh_approx_meta)

        def tanh_approx_lowering(inp):
            # [FUNCTION DOC] HELPER: Lowering of tanh_approx: wraps an elementwise inline-
            # assembly op using the tanh.approx.f32 instruction in a pointwise lowering and
            # applies it to the input.
            fn = partial(ops.inline_asm_elementwise, asm="tanh.approx.f32 $0, $1;")
            return make_pointwise(fn)(inp)

        register_lowering(
            torch.ops.test_inductor_ops.tanh_approx, type_promotion_kind=None
        )(tanh_approx_lowering)

        cls.test_inductor_ops.define("add_custom(Tensor a, Tensor b) -> Tensor")

        def add_custom(a, b):
            # [FUNCTION DOC] HELPER: Meta implementation of add_custom; returns the sum
            # of the inputs.
            return a + b

        cls.impl_meta.impl("add_custom", add_custom)

        def add_custom_lowering(a, b):
            # [FUNCTION DOC] HELPER: Lowering of add_custom: chooses ROCm assembly (with register
            # constraints) on HIP builds or a CUDA add.f32 instruction otherwise, wraps it as a
            # two-input pointwise lowering and applies it to the inputs.
            if torch.version.hip:
                # ROCm GCN assembly
                fn = partial(
                    ops.inline_asm_elementwise,
                    asm="v_add_f32 $0, $1, $2",
                    constraints="=v, v, v",
                )
            else:
                fn = partial(ops.inline_asm_elementwise, asm="add.f32 $0, $1, $2;")
            return make_pointwise(fn)(a, b)

        register_lowering(
            torch.ops.test_inductor_ops.add_custom, type_promotion_kind=None
        )(add_custom_lowering)

    def test_register_lowering_custom_dict(self):
        # [FUNCTION DOC] Checks that registering a lowering with an explicit lowering dictionary
        # stores it only in that dictionary. Defines a custom op, registers a lowering for it
        # into a local dictionary, and raises AssertionError if the op is missing from the
        # dictionary or has leaked into the global lowerings table.
        custom_lowering_dict = {}

        from torch._inductor.lowering import register_lowering

        @torch.library.custom_op("helion_test::foo", mutates_args={})
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op definition that returns its input unchanged.
            return x

        @register_lowering(
            torch.ops.helion_test.foo, lowering_dict=custom_lowering_dict
        )
        def foo_lowering(x):
            # [FUNCTION DOC] HELPER: Lowering registered into the local dictionary;
            # returns its input.
            return x

        if torch.ops.helion_test.foo not in custom_lowering_dict:
            raise AssertionError
        if torch.ops.helion_test.foo in torch._inductor.lowering.lowerings:
            raise AssertionError

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_jagged_to_padded_dense_sanity_cuda(self):
        # [FUNCTION DOC] Requires a GPU (not MPS). Runs the jagged-to-padded-dense op eagerly on
        # a small jagged input with three batches and checks selected output rows match the
        # corresponding input rows, then checks the compiled function matches eager.
        def fn(inp, offsets, max_seq_len):
            # [FUNCTION DOC] HELPER: Calls the jagged-to-padded-dense op with pad value sixty.
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((9, 96), device=GPU_TYPE)
        offsets = torch.tensor([0, 2, 5, 9], dtype=torch.int32, device=GPU_TYPE)
        max_seq_len = 4

        res = fn(inp, offsets, max_seq_len)
        self.assertEqual(inp[0], res[0][0])
        self.assertEqual(inp[1], res[0][1])
        self.assertEqual(inp[2], res[1][0])
        self.assertEqual(inp[3], res[1][1])
        self.assertEqual(inp[5], res[2][0])
        self.assertEqual(inp[8], res[2][3])

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_jagged_to_padded_dense_zero_size(self):
        # [FUNCTION DOC] Requires a GPU (not MPS). Regression test for masked loads being
        # stripped for a zero-size jagged input (which caused an out-of-bounds read): compares
        # eager and compiled results with an empty input and all-zero offsets.
        # Previously, the masking was being completely stripped for the
        # masked load of the input value. That would lead to an IMA
        # because cuda was trying to read index 0 of a zero-size tensor.
        def fn(inp, offsets, max_seq_len):
            # [FUNCTION DOC] HELPER: Reduces a zero-length input through a batched matrix
            # multiply and reshape to a zero-row matrix, then calls the jagged-to-padded-dense
            # op on it.
            inp = torch.bmm(inp, torch.ones((1, 96, 1), device=GPU_TYPE)).view((0, 1))
            return torch.ops.test_inductor_ops.jagged_to_padded_dense(
                inp, offsets, max_seq_len, 60.0
            )

        inp = torch.rand((1, 0, 96), device=GPU_TYPE)
        offsets = torch.zeros(1025, device=GPU_TYPE, dtype=torch.int32)
        max_seq_len = 20

        fn_opt = torch.compile(fn)

        self.assertEqual(
            fn(inp, offsets, max_seq_len), fn_opt(inp, offsets, max_seq_len)
        )

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_tanh_approx(self):
        # [FUNCTION DOC] Requires a GPU, not ROCm, not XPU, not MPS. Checks the compiled
        # tanh_approx op matches the standard tanh within default tolerances.
        def fn(inp):
            # [FUNCTION DOC] HELPER: Calls the tanh_approx op on the input.
            return torch.ops.test_inductor_ops.tanh_approx(inp)

        inp = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        a = torch.tanh(inp)
        b = fn_opt(inp)
        self.assertEqual(a, b)

    @requires_gpu()
    @skipIfRocm
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_reused_inline_asm_realized(self):
        # [FUNCTION DOC] Requires a GPU, not ROCm, not XPU, not MPS. Uses the tanh
        # approximation result in two different reductions in one fullgraph compiled function
        # and checks the values against reference tanh sums (tolerance 1e-4) and that the
        # generated code contains the inline assembly instruction exactly once, i.e. the result
        # is realized and reused.
        def fn(inp):
            # [FUNCTION DOC] HELPER: Applies tanh_approx once and returns its sums over the first
            # and second dimensions.
            y = torch.ops.test_inductor_ops.tanh_approx(inp)
            return y.sum(dim=0), y.sum(dim=1)

        inp = torch.randn(32, 64, device=GPU_TYPE)
        expected = (torch.tanh(inp).sum(dim=0), torch.tanh(inp).sum(dim=1))
        actual, code = run_and_get_code(torch.compile(fn, fullgraph=True), inp)

        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual("\n".join(code).count("tanh.approx.f32"), 1)

    @requires_gpu()
    @skipIfXpu(msg="`tl.inline_asm_elementwise` is not yet supported on Intel GPUs")
    @skipIf(GPU_TYPE == "mps", "Not applicable to MPS")
    def test_multi_inp_asm(self):
        # [FUNCTION DOC] Requires a GPU (not XPU, not MPS). Checks the two-input inline-assembly
        # add op, compiled, matches a plain addition.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the add_custom op on its two inputs.
            return torch.ops.test_inductor_ops.add_custom(a, b)

        a = torch.randn(32, device=GPU_TYPE)
        b = torch.randn(32, device=GPU_TYPE)
        fn_opt = torch.compile(fn)

        out1 = a + b
        out2 = fn_opt(a, b)
        self.assertEqual(out1, out2)

    @config.patch(joint_graph_constant_folding=False)
    def test_constant_creation(self):
        # [FUNCTION DOC] With joint graph constant folding disabled, registers a fallback for the
        # lift-fresh-copy op and checks a compiled module that adds a freshly created scalar
        # tensor to its input gives the same result as the input plus one.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Tiny module whose forward adds a newly created scalar tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns input plus a new scalar tensor.
                return x + torch.tensor(1)

        make_fallback(torch.ops.aten.lift_fresh_copy.default)
        self.assertTrue(
            torch.allclose(torch.compile(M())(torch.ones(3)), torch.ones(3) + 1)
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if HAS_CPU or HAS_GPU:
        run_tests(needs="filelock")
