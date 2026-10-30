# Owner(s): ["module: tests"]

# [MODULE DOC] Tests for the Python TensorIterator binding in torch._tensor_iterator.
# Covers the build pipeline (operand ownership, flag validation, static dtype/device
# declarations) and the named constructors (binary, unary, comparison, nullary, reduce
# and float-promoting variants) on CPU and, where marked, on accelerators. The device
# generic class is instantiated per device at import time via instantiate_device_type_tests.
import gc

import torch
from torch._tensor_iterator import (
    binary_float_op,
    binary_op,
    comparison_op,
    nullary_op,
    reduce_op,
    TensorIterator,
    unary_float_op,
    unary_op,
)
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
)
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


class TestTensorIteratorBuild(TestCase):
    # [CLASS DOC] Device-independent tests of TensorIterator construction. Each test builds an
    # iterator on CPU tensors and checks ownership, flag validation errors or static
    # dtype/device declarations. Holds no state.
    """Build-pipeline tests that don't depend on a particular device type."""

    def test_lifetime_owns_operands(self):
        # [FUNCTION DOC] Checks that the iterator keeps its operands alive: builds an iterator
        # from two float tensors, deletes the Python handles, runs garbage collection, then
        # verifies the first input still reports the original data pointer and still sums to
        # the expected value (sum of 0..11).
        # Core claim: the iterator wraps operands with the *owning*
        # MaybeOwned<TensorBase> path, so the underlying storage outlives
        # the Python tensor handles even if they're dropped between
        # build() and inspection. We verify via storage data_ptr staying
        # valid -- not via weakref, because PyTorch's TensorImpl pyobj_slot
        # resurrects the Python wrapper as long as the C++ side holds the
        # TensorImpl alive.
        a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        b = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        a_ptr = a.data_ptr()

        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            promote_inputs_to_common_dtype=True,
        )

        del a, b
        gc.collect()

        # If the iterator had borrowed (not owned) the operands, the
        # storage would be freed here and the assertion below would
        # either crash or read garbage.
        self.assertEqual(it.input(0).data_ptr(), a_ptr)
        self.assertEqual(it.input(0).sum().item(), float(sum(range(12))))

    def test_is_reduction_requires_defined_output(self):
        # [FUNCTION DOC] Checks that requesting is_reduction with an undefined (None-valued) output
        # raises a RuntimeError whose message says the flag requires every output to be defined.
        # at::TensorIterator::reduce_op gates is_reduction with
        # TORCH_INTERNAL_ASSERT(out.defined()). The named-constructor
        # gate isn't reachable when a Python user composes the flag
        # manually; without our build-time check the iterator silently
        # produces a non-reducing iterator with a wrong-shape output.
        a = torch.zeros(3, 4)
        with self.assertRaisesRegex(
            RuntimeError, "is_reduction.*requires every output"
        ):
            TensorIterator(
                outputs=[None],
                const_inputs=[a],
                is_reduction=True,
            )

    def test_is_reduction_with_defined_output_ok(self):
        # [FUNCTION DOC] Checks that is_reduction with a defined reduced-shape output (3 by 1)
        # and resize_outputs disabled builds successfully and reports one output.
        # Same flag combination, defined output of reduced shape: builds.
        a = torch.zeros(3, 4)
        out = torch.empty(3, 1)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            resize_outputs=False,
            is_reduction=True,
        )
        self.assertEqual(it.noutputs, 1)

    def test_enforce_safe_casting_to_output_raises(self):
        # [FUNCTION DOC] Checks that enforce_safe_casting_to_output with promotion to a common
        # dtype rejects a float32 result written into an int32 output by raising RuntimeError.
        # float -> integer is not a safe cast; the build must reject when
        # safe-casting-to-output is on.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            TensorIterator(
                outputs=[out],
                const_inputs=[a, b],
                promote_inputs_to_common_dtype=True,
                enforce_safe_casting_to_output=True,
                resize_outputs=False,
            )

    def test_enforce_linear_iteration_preserves_c_order(self):
        # [FUNCTION DOC] Uses a transposed (non-contiguous) tensor to show that the default build
        # coalesces dimensions into one of length 12, while enforce_linear_iteration keeps
        # two dimensions; the element count is 12 in both cases.
        # Probe with a non-contiguous input: a transposed (3, 4) tensor.
        # The default reorder permutes dims into memory order so the iter
        # collapses to one contiguous dim of length 12. With
        # enforce_linear_iteration the perm is fixed at [n-1, ..., 0], so
        # the iter cannot coalesce -- ndim stays at 2.
        a = torch.zeros(4, 3).t()
        no_flag = unary_op(None, a)
        self.assertEqual(no_flag.ndim, 1)
        self.assertEqual(no_flag.numel, 12)
        with_flag = TensorIterator(
            outputs=[None],
            const_inputs=[a],
            enforce_linear_iteration=True,
        )
        self.assertEqual(with_flag.ndim, 2)
        self.assertEqual(with_flag.numel, 12)

    def test_declare_static_dtype_and_device(self):
        # [FUNCTION DOC] Checks that static_dtype float64 and a CPU static_device are honored
        # when check_all_same_dtype is off: operand 0 reports float64 and a cpu device.
        a = torch.zeros(3, dtype=torch.float32)
        out = torch.empty(3, dtype=torch.float64)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            check_all_same_dtype=False,
            static_dtype=torch.float64,
            static_device=torch.device("cpu"),
        )
        self.assertEqual(it.dtype(0), torch.float64)
        self.assertEqual(it.device(0).type, "cpu")

    def test_declare_static_device(self):
        # [FUNCTION DOC] Checks that declaring a CPU static_device is accepted and operand 0
        # reports a device of type cpu.
        a = torch.zeros(3)
        out = torch.empty(3)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            static_device=torch.device("cpu"),
        )
        self.assertEqual(it.device(0).type, "cpu")

    def test_mixed_dtype_rejected_without_promotion(self):
        # [FUNCTION DOC] Checks that mixing float32 and float64 inputs without dtype promotion
        # raises RuntimeError, because check_all_same_dtype is on by default.
        # check_all_same_dtype is on by default. Mixed dtypes should fail.
        a = torch.zeros(3, dtype=torch.float32)
        b = torch.zeros(3, dtype=torch.float64)
        with self.assertRaises(RuntimeError):
            TensorIterator(outputs=[None], const_inputs=[a, b])


class TestTensorIterator(TestCase):
    # [CLASS DOC] Device-generic TensorIterator tests, instantiated per device type by
    # instantiate_device_type_tests. Covers shape broadcasting, dtype promotion, output
    # resizing, memory-overlap checks, named constructors, stride reporting and repr.
    @parametrize(
        "shape_a,shape_b,expected",
        [
            ((3, 1), (1, 4), (3, 4)),
            ((1,), (5,), (5,)),
            ((2, 3, 4), (3, 4), (2, 3, 4)),
            ((1, 1, 5), (4, 1, 5), (4, 1, 5)),
        ],
    )
    def test_broadcast_shape(self, device, shape_a, shape_b, expected):
        # [FUNCTION DOC] Parametrized over pairs of input shapes and their broadcast shape. Builds
        # a binary iterator with an auto-allocated output, then checks that the iterator element
        # count equals the product of the broadcast shape and that the allocated output has
        # exactly the broadcast shape.
        a = torch.zeros(shape_a, device=device)
        b = torch.zeros(shape_b, device=device)
        it = binary_op(None, a, b)
        # The post-coalesce shape may collapse contiguous dims, but its numel
        # has to equal the broadcast numel and the output tensor must end up at
        # the broadcast shape after build.
        expected_numel = 1
        for s in expected:
            expected_numel *= s
        self.assertEqual(it.numel, expected_numel)
        self.assertEqual(tuple(it.output(0).shape), expected)

    def test_promote_inputs_to_common_dtype(self, device):
        # [FUNCTION DOC] Checks that a binary iterator over int32 and float32 inputs reports
        # float32 as its common dtype.
        a = torch.zeros(3, dtype=torch.int32, device=device)
        b = torch.zeros(3, dtype=torch.float32, device=device)
        it = binary_op(None, a, b)
        self.assertEqual(it.common_dtype, torch.float32)

    def test_promote_integer_inputs_to_float(self, device):
        # [FUNCTION DOC] Checks that with promotion and promote_integer_inputs_to_float enabled,
        # int64 and int32 inputs yield the default floating dtype as common dtype.
        a = torch.zeros(3, dtype=torch.int64, device=device)
        b = torch.zeros(3, dtype=torch.int32, device=device)
        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            promote_inputs_to_common_dtype=True,
            promote_integer_inputs_to_float=True,
        )
        # Integer common dtype gets bumped to default float scalar type.
        self.assertEqual(it.common_dtype, torch.get_default_dtype())

    def test_common_dtype_when_all_match(self, device):
        # [FUNCTION DOC] Checks that two float64 inputs, with only the default same-dtype check,
        # still produce float64 as the common dtype.
        # check_all_same_dtype still produces a common dtype: it's just the
        # single dtype shared by all operands.
        a = torch.zeros(3, dtype=torch.float64, device=device)
        b = torch.zeros(3, dtype=torch.float64, device=device)
        it = TensorIterator(outputs=[None], const_inputs=[a, b])
        self.assertEqual(it.common_dtype, torch.float64)

    def test_resize_outputs(self, device):
        # [FUNCTION DOC] Checks that a zero-element preallocated output is resized in place by a
        # binary iterator to the 3 by 4 broadcast shape.
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        # Pre-allocated zero-element output is reshaped to the broadcast
        # shape (the docs-recommended pattern for resizable outputs).
        out = torch.empty(0, device=device)
        binary_op(out, a, b)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_no_resize_outputs(self, device):
        # [FUNCTION DOC] Checks that with resize_outputs disabled and a correctly sized output, the
        # iterator has 12 elements and leaves the output shape at 3 by 4.
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        out = torch.zeros(3, 4, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a, b],
            resize_outputs=False,
        )
        self.assertEqual(it.numel, 12)
        self.assertEqual(tuple(out.shape), (3, 4))

    def test_declare_static_shape(self, device):
        # [FUNCTION DOC] Checks that declaring static_shape of 2 by 6 with a matching output builds
        # and the coalesced iterator reports 12 elements.
        a = torch.zeros(2, 6, device=device)
        out = torch.empty(2, 6, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            resize_outputs=False,
            static_shape=(2, 6),
        )
        # Shape is reported post-coalesce: a 2x6 contiguous tensor collapses.
        self.assertEqual(it.numel, 12)

    def test_declare_static_dtype(self, device):
        # [FUNCTION DOC] Checks that declaring static_dtype float64 with the same-dtype check off
        # makes operand 0 report float64.
        a = torch.zeros(3, dtype=torch.float64, device=device)
        out = torch.empty(3, dtype=torch.float64, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a],
            check_all_same_dtype=False,
            static_dtype=torch.float64,
        )
        self.assertEqual(it.dtype(0), torch.float64)

    def test_mem_overlap_detected(self, device):
        # [FUNCTION DOC] Checks that an output created by expanding a tensor (internal memory
        # overlap) makes the build raise RuntimeError when the overlap check is left on.
        # Broadcast view -> internal overlap on output.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        with self.assertRaises(RuntimeError):
            TensorIterator(
                outputs=[out],
                const_inputs=[a, b],
                resize_outputs=False,
            )

    def test_mem_overlap_disabled(self, device):
        # [FUNCTION DOC] Same expanded-output setup as the overlap-detection test, but with
        # check_mem_overlap disabled: the build succeeds and reports 24 elements.
        # Same setup as above but with the overlap check turned off.
        out = torch.empty(2, 3, device=device).expand(4, 2, 3)
        a = torch.zeros(4, 2, 3, device=device)
        b = torch.zeros(4, 2, 3, device=device)
        it = TensorIterator(
            outputs=[out],
            const_inputs=[a, b],
            resize_outputs=False,
            check_mem_overlap=False,
        )
        self.assertEqual(it.numel, 24)

    def test_comparison_op_bool_output(self, device):
        # [FUNCTION DOC] Checks that comparison_op with an undefined output allocates a bool
        # output (operand 0 dtype is bool) from float32 inputs.
        a = torch.zeros(3, dtype=torch.float32, device=device)
        b = torch.zeros(3, dtype=torch.float32, device=device)
        # Undefined output -> static dtype is bool.
        it = comparison_op(None, a, b)
        self.assertEqual(it.dtype(0), torch.bool)

    def test_unary_op_dtype_passthrough(self, device):
        # [FUNCTION DOC] Checks that unary_op on a float64 input with auto-allocated output keeps
        # float64 for operand 0 and reports 12 elements.
        a = torch.zeros(3, 4, dtype=torch.float64, device=device)
        # unary_op uses check_all_same_dtype, so the output (when
        # auto-allocated) is allocated in the input's dtype.
        it = unary_op(None, a)
        self.assertEqual(it.dtype(0), torch.float64)
        self.assertEqual(it.numel, 12)

    def test_nullary_op(self, device):
        # [FUNCTION DOC] Checks that nullary_op on one output yields one tensor, one output and no
        # inputs.
        out = torch.empty(3, 4, device=device)
        it = nullary_op(out)
        self.assertEqual(it.ntensors, 1)
        self.assertEqual(it.noutputs, 1)
        self.assertEqual(it.ninputs, 0)

    def test_reduce_op(self, device):
        # [FUNCTION DOC] Checks that reduce_op with a 3 by 4 input and preallocated 3 by 1 output
        # gives two operands total and one output.
        a = torch.zeros(3, 4, device=device)
        # reduce_op requires the output to broadcast with the input. To reduce
        # along the trailing dim we pre-allocate a (3, 1) output.
        out = torch.empty(3, 1, device=device)
        it = reduce_op(out, a)
        self.assertEqual(it.ntensors, 2)
        self.assertEqual(it.noutputs, 1)

    def test_reduce_op_two_outputs(self, device):
        # [FUNCTION DOC] Checks the two-output form of reduce_op (second output passed as out2):
        # three operands total, two outputs, both 3 by 1.
        # The (out, out2) overload covers things like at::min returning
        # (values, indices). We don't need integer indices to exercise the
        # binding; we just need two outputs that share the reduced shape.
        a = torch.zeros(3, 4, device=device)
        out = torch.empty(3, 1, device=device)
        out2 = torch.empty(3, 1, device=device)
        it = reduce_op(out, a, out2=out2)
        self.assertEqual(it.ntensors, 3)
        self.assertEqual(it.noutputs, 2)

    def test_reduce_op_two_outputs_shape_mismatch_raises(self, device):
        # [FUNCTION DOC] Checks that reduce_op with a second output shaped 1 by 3 while
        # the first is 3 by 1 raises RuntimeError mentioning identical sizes and strides.
        a = torch.zeros(3, 4, device=device)
        out = torch.empty(3, 1, device=device)
        out2 = torch.empty(1, 3, device=device)
        with self.assertRaisesRegex(RuntimeError, "identical sizes and strides"):
            reduce_op(out, a, out2=out2)

    def test_binary_float_op_promotes_integers(self, device):
        # [FUNCTION DOC] Checks that binary_float_op on int32 and int64 inputs gives a floating
        # point common dtype and an auto-allocated output of floating dtype.
        # binary_float_op promotes integer inputs to the default float dtype
        # so the kernel can dispatch a single float arithmetic path.
        a = torch.zeros(3, 4, dtype=torch.int32, device=device)
        b = torch.zeros(3, 4, dtype=torch.int64, device=device)
        it = binary_float_op(None, a, b)
        self.assertTrue(it.common_dtype.is_floating_point)
        # The auto-allocated output picks up the promoted dtype.
        self.assertTrue(it.dtype(0).is_floating_point)

    def test_unary_float_op_promotes_integers(self, device):
        # [FUNCTION DOC] Checks that unary_float_op on an int32 input gives a floating point common
        # dtype and a floating point operand 0 dtype.
        a = torch.zeros(3, 4, dtype=torch.int32, device=device)
        it = unary_float_op(None, a)
        self.assertTrue(it.common_dtype.is_floating_point)
        self.assertTrue(it.dtype(0).is_floating_point)

    def test_strides_are_bytes(self, device):
        # [FUNCTION DOC] Checks that strides reports byte strides: for float32 operands, each byte
        # stride equals the matching element stride times four, and both lists have equal length.
        a = torch.zeros(3, 4, dtype=torch.float32, device=device)
        b = torch.zeros(3, 4, dtype=torch.float32, device=device)
        it = binary_op(None, a, b)
        # Element size is 4 bytes; element_strides should equal byte strides / 4.
        byte_strides = it.strides(0)
        elt_strides = it.element_strides(0)
        self.assertEqual(len(byte_strides), len(elt_strides))
        for bs, es in zip(byte_strides, elt_strides):
            self.assertEqual(bs, es * 4)

    def test_repr(self, device):
        # [FUNCTION DOC] Checks that the iterator repr text contains the ndim field and reports
        # three operands (two inputs plus the output).
        a = torch.zeros(3, 4, device=device)
        b = torch.zeros(3, 4, device=device)
        it = binary_op(None, a, b)
        r = repr(it)
        self.assertIn("ndim=", r)
        self.assertIn("ntensors=3", r)

    @onlyAccelerator
    def test_cross_device_check_raises(self, device):
        # [FUNCTION DOC] Accelerator-only. Checks that mixing a CPU tensor with a device tensor in
        # binary_op raises RuntimeError under the default same-device check.
        # Default check_all_same_device=True rejects mixed CPU+accelerator.
        a = torch.zeros(3, device="cpu")
        b = torch.zeros(3, device=device)
        with self.assertRaises(RuntimeError):
            binary_op(None, a, b)

    @onlyAccelerator
    def test_allow_cpu_scalars(self, device):
        # [FUNCTION DOC] Accelerator-only. Checks that with allow_cpu_scalars enabled, a 0-dim CPU
        # tensor alongside a device tensor is accepted, and the iterator reports 3 elements.
        # A 0-dim CPU tensor is a "CPU scalar". With allow_cpu_scalars=True
        # the iterator accepts it alongside a device tensor without violating
        # the same-device check (the same-device check itself allows CPU
        # scalars when this flag is on).
        a = torch.zeros(3, device=device)
        b = torch.tensor(2.0, device="cpu")  # 0-dim CPU scalar
        it = TensorIterator(
            outputs=[None],
            const_inputs=[a, b],
            allow_cpu_scalars=True,
            promote_inputs_to_common_dtype=True,
        )
        self.assertEqual(it.numel, 3)


instantiate_device_type_tests(TestTensorIterator, globals())


if __name__ == "__main__":
    run_tests()
