# Owner(s): ["module: complex"]
# [MODULE DOC] Tests that ComplexTensor (a tensor subclass storing real and imaginary
# parts separately) behaves like native complex tensors. Defines SKIPS (op/variant
# pairs to skip, with reasons) and EXTRA_KWARGS (per op/dtype/variant comparison
# tolerances), both keyed by Descriptor and consumed by utils.TestCase. Test classes:
# TestComplexTensor (forward consistency vs native complex, plus a few direct subclass
# checks), TestComplexBwdGradients (gradcheck on complex128) and, when
# torch.distributed is available, TestComplexDistributed (same ops on DTensor).
# Device-generic classes are instantiated at import time into this module; running as
# a script calls run_tests.
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist


# Support both when imported from elsewhere or directly as a file
try:
    from .utils import (
        COMPLEX_DTYPES,
        Descriptor,
        force_test_op_db,
        implemented_op_db,
        TestCase,
        Variant,
    )
except ImportError:
    from utils import (
        COMPLEX_DTYPES,
        Descriptor,
        force_test_op_db,
        implemented_op_db,
        TestCase,
        Variant,
    )

from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    OpDTypes,
    ops,
)
from torch.testing._internal.common_utils import run_tests, unMarkDynamoStrictTest


if TYPE_CHECKING:
    from torch.testing._internal.opinfo.core import OpInfo

aten = torch.ops.aten

SKIPS = {
    Descriptor(op=aten.empty_like, variant=None): "Non-deterministic output",
    Descriptor(op=aten.randn_like, variant=None): "Non-deterministic output",
    Descriptor(
        op=aten.any, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.all, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.allclose, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.conj_physical, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten._conj_physical, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.cumprod, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.index_add, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.diagonal_scatter, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.flip, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.masked_fill, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.masked_scatter, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.rsub, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.ne, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.squeeze, variant=Variant.Distributed
    ): "does not have a sharding strategy registered",
    Descriptor(
        op=aten.index_select, variant=Variant.Distributed
    ): "Sharding propagation failed",
    Descriptor(op=aten.real, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.imag, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.isfinite, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.transpose, variant=Variant.Distributed): "No scalar support",
    Descriptor(op=aten.view_as_real, variant=Variant.Distributed): "No scalar support",
}

EXTRA_KWARGS = {
    Descriptor(op=aten.asinh, dtype=torch.complex64, variant=Variant.Op): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    Descriptor(op=aten.tanh, dtype=torch.complex64, variant=Variant.Op): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    Descriptor(op=aten.pow, dtype=torch.complex64, variant=Variant.Op): {
        "rtol": 2e-2,
        "atol": 2e-6,
    },
    Descriptor(op=aten.asinh, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-5,
        "atol": 5e-5,
    },
    Descriptor(op=aten.tanh, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 1e-4,
        "atol": 1e-5,
    },
    Descriptor(op=aten.pow, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-2,
        "atol": 2e-6,
    },
    Descriptor(op=aten.tan, dtype=torch.complex64, variant=Variant.Distributed): {
        "rtol": 2e-6,
        "atol": 1e-2,
    },
}


class TestComplexTensor(TestCase):
    # [CLASS DOC] Op-info driven and direct tests comparing ComplexTensor results against
    # native complex tensors. Default dtype checking is enabled. Methods: test_consistency
    # and test_maybe_error (per op, device and complex dtype) and three direct tests of
    # the subclass API.
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported,
        allowed_dtypes=list(COMPLEX_DTYPES),
    )
    def test_consistency(self, device, dtype, op: OpInfo):
        # [FUNCTION DOC] For every implemented op and supported complex dtype, checks that
        # running the op on ComplexTensor inputs matches native complex results (or raises the
        # same kind of error).
        self.check_consistency(device, dtype, op, Variant.Op)

    @ops(force_test_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
    def test_maybe_error(self, device, dtype, op: OpInfo):
        # [FUNCTION DOC] Same consistency check as test_consistency, but for ops on the
        # forced-test list, which may legitimately error; results and error types are compared
        # and dtypes are not restricted to the supported set.
        self.check_consistency(device, dtype, op, Variant.Op)

    def test_get_set_components(self):
        # [FUNCTION DOC] Checks the real and imaginary properties of a ComplexTensor built
        # from an interleaved complex64 tensor, then assigns new values through both setters
        # (integer inputs are expected to be stored as float32) and checks the interleaved
        # result equals the matching complex tensor.
        from torch._subclasses.complex_tensor import ComplexTensor

        c = ComplexTensor.from_interleaved(
            torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        )
        self.assertEqual(c.real, torch.tensor([1, 3], dtype=torch.float32))
        self.assertEqual(c.imag, torch.tensor([2, 4], dtype=torch.float32))

        c.real = torch.tensor([5, 6])
        c.imag = torch.tensor([7, 8])
        self.assertEqual(c.real, torch.tensor([5, 6], dtype=torch.float32))
        self.assertEqual(c.imag, torch.tensor([7, 8], dtype=torch.float32))
        self.assertEqual(c, torch.tensor([5 + 7j, 6 + 8j], dtype=torch.complex64))

    def test_mul_inplace_complex(self):
        # [FUNCTION DOC] Checks that in-place multiplication of a ComplexTensor by another
        # ComplexTensor returns the same object and that its interleaved form equals the
        # native complex product.
        from torch._subclasses.complex_tensor import ComplexTensor

        a = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
        b = torch.tensor([7 - 8j, -9 + 1j], dtype=torch.complex64)
        expected = a * b

        xa = ComplexTensor.from_interleaved(a)
        result = xa.mul_(ComplexTensor.from_interleaved(b))

        self.assertIs(result, xa)
        self.assertEqual(xa.as_interleaved(), expected)

    def test_ne_real_operand(self):
        # [FUNCTION DOC] Checks that elementwise not-equal between a real tensor and a
        # ComplexTensor gives the same boolean result as with the native complex tensor.
        from torch._subclasses.complex_tensor import ComplexTensor

        r = torch.tensor([1.0, 3.0, 5.0])
        c = torch.tensor([1 + 2j, 3 + 0j, 0 + 4j], dtype=torch.complex64)
        xc = ComplexTensor.from_interleaved(c)

        self.assertEqual(torch.ne(r, xc), torch.ne(r, c))


@unMarkDynamoStrictTest
class TestComplexBwdGradients(TestCase):
    # [CLASS DOC] Gradient tests for ComplexTensor ops with strict-Dynamo marking removed.
    # Default dtype checking is enabled. Method test_fn_grad runs gradcheck per
    # implemented op on complex128.
    _default_dtype_check_enabled = True

    @ops(
        implemented_op_db,
        dtypes=OpDTypes.supported_backward,
        allowed_dtypes=[torch.complex128],
    )
    def test_fn_grad(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        # [FUNCTION DOC] Runs gradcheck (including batched grad and forward AD) for the op
        # with ComplexTensor inputs; skipped ops come from SKIPS via the shared helper.
        # Returns nothing.
        self.check_grad(device, dtype, op)


instantiate_device_type_tests(TestComplexTensor, globals())
instantiate_device_type_tests(TestComplexBwdGradients, globals())


if dist.is_available():
    from torch.testing._internal.common_distributed import MultiProcessTestCase

    @unMarkDynamoStrictTest
    class TestComplexDistributed(TestCase, MultiProcessTestCase):
        # [CLASS DOC] Multi-process test case checking ComplexTensor ops wrapped in DTensor.
        # Defined only when torch.distributed is available. Method test_distributed runs the
        # consistency check with the distributed variant.
        @ops(implemented_op_db, allowed_dtypes=list(COMPLEX_DTYPES))
        def test_distributed(self, device, dtype, op: OpInfo):
            # [FUNCTION DOC] For every implemented op and complex dtype, compares native complex
            # results with results on DTensor-wrapped ComplexTensor inputs (distributed variant of
            # the consistency check).
            self.check_consistency(device, dtype, op, Variant.Distributed)

    instantiate_device_type_tests(TestComplexDistributed, globals())

if __name__ == "__main__":
    run_tests()
