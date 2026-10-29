# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for splitting first-class dimension tensors (functorch.dim) with Dim
# objects as the split sizes. Covers bound and unbound dims, size inference, error cases, result
# types, devices, dtypes, autograd and the standalone split function. Defines one TestCase class
# and a run_tests entry point; no import-time side effects.
import unittest

import torch
from functorch.dim import Dim, dims, Tensor
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TORCHDYNAMO,
    TestCase,
)


class TestSplit(TestCase):
    # [CLASS DOC] Test case for tensor split where the sizes are Dim objects. Each test indexes
    # a plain tensor with three dims, splits one dim, and inspects results via order. setUp
    # stores three unused dims as batch, height and width attributes.
    """Comprehensive tests for first-class dimension split operations."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture. Calls the base setUp and stores three fresh unbound
        # dims as attributes batch, height and width.
        super().setUp()
        self.batch, self.height, self.width = dims(3)

    def test_dim_object_split_all_bound(self):
        # [FUNCTION DOC] Splits the middle axis (size 12) with three bound dims of sizes 3, 4,
        # 5. Expects three results whose ordered shapes use those chunk sizes and dims keeping
        # their sizes.
        """Test split with all Dim objects bound to specific sizes."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Create bound Dim objects
        d1 = Dim("d1", 3)
        d2 = Dim("d2", 4)
        d3 = Dim("d3", 5)

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)

        # For FCD tensors, check the ordered version to verify shapes
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 3, 5))
        self.assertEqual(result[1].order(x, d2, z).shape, (3, 4, 5))
        self.assertEqual(result[2].order(x, d3, z).shape, (3, 5, 5))

        # Verify dimensions are bound correctly
        self.assertEqual(d1.size, 3)
        self.assertEqual(d2.size, 4)
        self.assertEqual(d3.size, 5)

    def test_dim_object_split_unbound(self):
        # [FUNCTION DOC] Splits a size 12 axis with three unbound dims. Expects an even split
        # into chunks of 4 and each dim becoming bound to size 4.
        """Test split with unbound Dim objects."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Create unbound Dim objects
        d1 = Dim("d1")
        d2 = Dim("d2")
        d3 = Dim("d3")

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)

        # Should split evenly: 12 / 3 = 4 each
        # Check via ordered tensors since FCD tensors have ndim=0
        for i, part in enumerate(result):
            if i == 0:
                self.assertEqual(part.order(x, d1, z).shape, (3, 4, 5))
            elif i == 1:
                self.assertEqual(part.order(x, d2, z).shape, (3, 4, 5))
            else:
                self.assertEqual(part.order(x, d3, z).shape, (3, 4, 5))

        # Verify dimensions are bound to chunk size
        self.assertEqual(d1.size, 4)
        self.assertEqual(d2.size, 4)
        self.assertEqual(d3.size, 4)

    def test_dim_object_split_mixed_bound_unbound(self):
        # [FUNCTION DOC] Splits a size 12 axis with bound dims of sizes 3 and 2 and one unbound
        # dim between them. Expects the unbound dim to take the remainder (7) and become bound
        # to it.
        """Test split with mix of bound and unbound Dim objects."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Create mix of bound and unbound
        d1 = Dim("d1", 3)  # bound
        d2 = Dim("d2")  # unbound
        d3 = Dim("d3", 2)  # bound

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 3, 5))
        self.assertEqual(result[1].order(x, d2, z).shape, (3, 7, 5))  # 12 - 3 - 2 = 7
        self.assertEqual(result[2].order(x, d3, z).shape, (3, 2, 5))

        # Verify unbound dimension was bound to remaining size
        self.assertEqual(d2.size, 7)

    def test_dim_object_split_multiple_unbound(self):
        # [FUNCTION DOC] Splits a size 15 axis with one bound dim (3) and two unbound dims.
        # Expects the remaining 12 to be shared evenly, giving both unbound dims size 6.
        """Test split with multiple unbound Dim objects."""
        tensor = torch.randn(3, 15, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Create multiple unbound dimensions
        d1 = Dim("d1", 3)  # bound
        d2 = Dim("d2")  # unbound
        d3 = Dim("d3")  # unbound

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 3, 5))

        # Remaining 12 should be split evenly between d2 and d3: 6 each
        self.assertEqual(result[1].order(x, d2, z).shape, (3, 6, 5))
        self.assertEqual(result[2].order(x, d3, z).shape, (3, 6, 5))

        self.assertEqual(d2.size, 6)
        self.assertEqual(d3.size, 6)

    def test_dim_object_split_uneven_remainder(self):
        # [FUNCTION DOC] Splits a size 14 axis with one bound dim (3) and two unbound dims.
        # Expects the first unbound dim to get 6 (rounded up share of the remaining 11) and the
        # last unbound dim to get the leftover 5.
        """Test split with unbound dimensions that don't divide evenly."""
        tensor = torch.randn(3, 14, 5)  # 14 doesn't divide evenly by 3
        x, y, z = dims(3)
        t = tensor[x, y, z]

        d1 = Dim("d1", 3)
        d2 = Dim("d2")  # Should get ceil((14-3)/2) = 6
        d3 = Dim("d3")  # Should get remaining = 5

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 3, 5))
        self.assertEqual(result[1].order(x, d2, z).shape, (3, 6, 5))
        self.assertEqual(result[2].order(x, d3, z).shape, (3, 5, 5))

        self.assertEqual(d2.size, 6)
        self.assertEqual(d3.size, 5)

    def test_split_with_dim_object_parameter(self):
        # [FUNCTION DOC] Splits along a Dim used as the dim argument, with three fully bound
        # dims, and only checks that three chunks are returned. Its name suggests a different
        # case than the sibling test of bound dims, but the body is nearly identical to it.
        """Test split when dim parameter is a Dim object."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Use Dim object as the dim parameter
        d1 = Dim("d1", 3)
        d2 = Dim("d2", 4)
        d3 = Dim("d3", 5)

        result = t.split([d1, d2, d3], dim=y)
        self.assertEqual(len(result), 3)

    def test_error_mixed_types(self):
        # [FUNCTION DOC] Checks that a split-size list mixing Dim objects and integers raises
        # TypeError, with the integer after the dim and with the integer between dims.
        """Test error when mixing integers and Dim objects in split sizes."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        d1 = Dim("d1", 3)

        # Should raise TypeError for mixed types
        with self.assertRaises(TypeError):
            t.split([d1, 4, 5], dim=y)

        with self.assertRaises(TypeError):
            t.split([3, d1, 5], dim=y)

    def test_error_dim_parameter_with_int_sizes(self):
        # [FUNCTION DOC] Checks that an integer split size, or a list of integers, raises
        # TypeError when the split axis is given as a Dim. The message argument of the assertion
        # context is only a failure message and does not check the error text.
        """Test error when dim parameter is Dim but sizes are integers."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Should raise TypeError when dim is Dim object but sizes are ints
        with self.assertRaises(
            TypeError,
            msg="when dim is specified as a Dim object, split sizes must also be dimensions.",
        ):
            t.split(3, dim=y)

        with self.assertRaises(
            TypeError,
            msg="when dim is specified as a Dim object, split sizes must also be dimensions.",
        ):
            t.split([3, 4, 5], dim=y)

    def test_error_size_mismatch(self):
        # [FUNCTION DOC] Checks that bound dim sizes summing to 13 for a size 12 axis
        # raise TypeError.
        """Test error when bound sizes don't match tensor dimension."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Bound dimensions that sum to wrong total
        d1 = Dim("d1", 3)
        d2 = Dim("d2", 4)
        d3 = Dim("d3", 6)  # 3 + 4 + 6 = 13, but tensor has 12

        with self.assertRaises(TypeError):
            t.split([d1, d2, d3], dim=y)

    def test_error_bound_sizes_exceed_tensor(self):
        # [FUNCTION DOC] Checks that bound dim sizes (8 and 6) exceeding the axis size 12 raise
        # TypeError even when a third dim is left unbound.
        """Test error when bound sizes exceed tensor dimension."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Bound dimensions with one unbound, but bound sizes too large
        d1 = Dim("d1", 8)
        d2 = Dim("d2", 6)  # 8 + 6 = 14 > 12
        d3 = Dim("d3")

        with self.assertRaises(TypeError):
            t.split([d1, d2, d3], dim=y)

    def test_error_nonexistent_dimension(self):
        # [FUNCTION DOC] Checks that splitting along a Dim that is not a dimension of the tensor
        # raises TypeError.
        """Test error when splitting on non-existent dimension."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        w = Dim("w")  # Not in tensor

        with self.assertRaises(TypeError):
            t.split([Dim("d1"), Dim("d2")], dim=w)

    def test_split_different_dims(self):
        # [FUNCTION DOC] Splits along the first axis into two bound dims (2, 4) and along the
        # last axis into two bound dims (3, 7), and verifies each ordered chunk shape.
        """Test splitting along different dimensions."""
        tensor = torch.randn(6, 8, 10)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Split along first dimension
        a, b = Dim("a", 2), Dim("b", 4)
        result1 = t.split([a, b], dim=x)
        self.assertEqual(len(result1), 2)
        self.assertEqual(result1[0].order(a, y, z).shape, (2, 8, 10))
        self.assertEqual(result1[1].order(b, y, z).shape, (4, 8, 10))

        # Split along last dimension
        c, d = Dim("c", 3), Dim("d", 7)
        result2 = t.split([c, d], dim=z)
        self.assertEqual(len(result2), 2)
        self.assertEqual(result2[0].order(x, y, c).shape, (6, 8, 3))
        self.assertEqual(result2[1].order(x, y, d).shape, (6, 8, 7))

    def test_split_single_dim_object(self):
        # [FUNCTION DOC] Splits an axis of size 12 with a single bound dim of size 12 and
        # expects one chunk with the full original shape.
        """Test split with single Dim object that matches tensor dimension size."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Use a single Dim object with size matching the dimension
        d1 = Dim("d1", 12)  # Must match the full size of y dimension

        # Single Dim object in list should work when size matches
        result = t.split([d1], dim=y)
        self.assertEqual(len(result), 1)  # Single chunk containing entire dimension
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 12, 5))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "TorchDynamo doesn't preserve side effects during tracing",
    )
    def test_dimension_binding_consistency(self):
        # [FUNCTION DOC] Skipped under TorchDynamo (side effects not preserved). After a split
        # with three unbound dims, checks they are bound, then indexes a new tensor with the
        # first dim and checks the dim keeps its size.
        """Test that split properly binds dimensions and they remain consistent."""
        tensor = torch.randn(3, 15, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        d1 = Dim("d1")
        d2 = Dim("d2")
        d3 = Dim("d3")

        # Split should bind dimensions
        t.split([d1, d2, d3], dim=y)

        # Use the bound dimensions in another operation
        self.assertTrue(d1.is_bound)
        self.assertTrue(d2.is_bound)
        self.assertTrue(d3.is_bound)

        # Dimensions should remain bound with same values
        original_sizes = (d1.size, d2.size, d3.size)

        # Try to use bound dimension again - should maintain same size
        another_tensor = torch.randn(original_sizes[0], 4)
        a = Dim("a")
        t2 = another_tensor[d1, a]  # d1 should still be bound to same size
        self.assertEqual(t2.order(d1, a).shape, (original_sizes[0], 4))

    def test_split_result_tensor_types(self):
        # [FUNCTION DOC] Splits into two bound dims and checks every chunk is a torch or
        # functorch.dim Tensor; if it has a dims attribute, that attribute must be non-empty.
        """Test that split results are proper first-class dimension tensors."""
        tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        d1 = Dim("d1", 4)
        d2 = Dim("d2", 8)

        result = t.split([d1, d2], dim=y)

        # Results should be first-class dimension tensors
        for part in result:
            self.assertTrue(isinstance(part, (torch.Tensor, Tensor)))

            # Should have dimensions from original tensor plus new split dimensions
            if hasattr(part, "dims"):
                # Check that the split dimension is in the result
                dims_in_result = part.dims
                self.assertTrue(len(dims_in_result) > 0)

    def test_large_tensor_split(self):
        # [FUNCTION DOC] Splits a size 100 axis into twenty bound dims of size 5 and checks the
        # chunk count, each ordered shape and each dim size.
        """Test split on larger tensors to verify performance and correctness."""
        tensor = torch.randn(10, 100, 20)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Split into many small pieces
        split_dims = [Dim(f"d{i}", 5) for i in range(20)]  # 20 * 5 = 100

        result = t.split(split_dims, dim=y)
        self.assertEqual(len(result), 20)

        for i, part in enumerate(result):
            self.assertEqual(part.order(x, split_dims[i], z).shape, (10, 5, 20))
            self.assertEqual(split_dims[i].size, 5)

    def test_device_handling(self):
        # [FUNCTION DOC] Splits into two bound dims on CUDA (only if available) and on CPU,
        # checking that ordered chunks keep the device and the untouched axis sizes.
        """Test split behavior with different devices."""
        if torch.cuda.is_available():
            # Test on CUDA
            cuda_tensor = torch.randn(3, 12, 5, device="cuda")
            x, y, z = dims(3)
            t = cuda_tensor[x, y, z]

            d1, d2 = Dim("d1", 4), Dim("d2", 8)
            result = t.split([d1, d2], dim=y)

            for i, part in enumerate(result):
                ordered = part.order(x, d1 if i == 0 else d2, z)
                self.assertEqual(ordered.device.type, "cuda")
                self.assertEqual(ordered.shape[0], 3)
                self.assertEqual(ordered.shape[2], 5)

        # Test on CPU
        cpu_tensor = torch.randn(3, 12, 5)
        x, y, z = dims(3)
        t = cpu_tensor[x, y, z]

        d1, d2 = Dim("d1", 4), Dim("d2", 8)
        result = t.split([d1, d2], dim=y)

        for i, part in enumerate(result):
            ordered = part.order(x, d1 if i == 0 else d2, z)
            self.assertEqual(ordered.device, torch.device("cpu"))

    def test_split_preserves_dtype(self):
        # [FUNCTION DOC] Loops over float32, float64, int32 and int64 tensors, splits into two
        # bound dims, and checks every ordered chunk keeps the dtype.
        """Test that split preserves tensor dtype."""
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:
            if dtype in [torch.int32, torch.int64]:
                tensor = torch.randint(0, 10, (3, 12, 5), dtype=dtype)
            else:
                tensor = torch.randn(3, 12, 5, dtype=dtype)
            x, y, z = dims(3)
            t = tensor[x, y, z]

            d1, d2 = Dim("d1", 4), Dim("d2", 8)
            result = t.split([d1, d2], dim=y)

            for i, part in enumerate(result):
                ordered = part.order(x, d1 if i == 0 else d2, z)
                self.assertEqual(ordered.dtype, dtype)

    def test_split_with_requires_grad(self):
        # [FUNCTION DOC] Splits a tensor that requires grad into two bound dims and checks each
        # ordered chunk still reports requires_grad as true.
        """Test split with tensors that require gradients."""
        tensor = torch.randn(3, 12, 5, requires_grad=True)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        d1, d2 = Dim("d1", 4), Dim("d2", 8)
        result = t.split([d1, d2], dim=y)

        for part in result:
            # Check requires_grad on the ordered tensor to access the underlying tensor properties
            self.assertTrue(
                part.order(x, d1 if part is result[0] else d2, z).requires_grad
            )

    def test_edge_case_single_element_splits(self):
        # [FUNCTION DOC] Splits a size 5 axis into five bound dims of size 1 and checks the
        # chunk count and each ordered shape.
        """Test splitting into single-element chunks."""
        tensor = torch.randn(3, 5, 4)
        x, y, z = dims(3)
        t = tensor[x, y, z]

        # Split into 5 single-element pieces
        split_dims = [Dim(f"d{i}", 1) for i in range(5)]

        result = t.split(split_dims, dim=y)
        self.assertEqual(len(result), 5)

        for i, part in enumerate(result):
            self.assertEqual(part.order(x, split_dims[i], z).shape, (3, 1, 4))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO, "TorchDynamo has issues with torch._tensor.split"
    )
    def test_split_function_directly(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Imports the standalone split function from
        # functorch.dim. On a plain tensor with an integer chunk size it returns three plain
        # chunks; on an indexed tensor with two bound dims it returns two chunks with the
        # expected shapes.
        """Test that the standalone split function works correctly."""
        from functorch.dim import split

        # Test on regular tensor
        tensor = torch.randn(3, 12, 5)
        result = split(tensor, 4, dim=1)
        self.assertEqual(len(result), 3)  # 12 / 4 = 3
        for part in result:
            self.assertEqual(part.shape, (3, 4, 5))

        # Test on FCD tensor with FCD arguments
        x, y, z = dims(3)
        fcd_tensor = tensor[x, y, z]

        d1 = Dim("d1", 4)
        d2 = Dim("d2", 8)
        result = split(fcd_tensor, [d1, d2], dim=y)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].order(x, d1, z).shape, (3, 4, 5))
        self.assertEqual(result[1].order(x, d2, z).shape, (3, 8, 5))

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "TorchDynamo can't parse dims() without arguments from bytecode",
    )
    def test_split_on_plain_tensor_with_fcd_args(self):
        # [FUNCTION DOC] Skipped under TorchDynamo. Splits a plain size 8 tensor using two
        # unbound dims as sizes. Expects two functorch.dim results having a dims attribute
        # whose first dim is the given one, with both dims bound to 4. Then repeats with one
        # dim used twice.
        """Test that split() works on plain tensors when FCD arguments are provided."""
        # Test the exact example from the user message
        x, y = dims()

        # Split a plain tensor with FCD dimensions as split sizes
        result = torch.randn(8).split([x, y], dim=0)
        self.assertEqual(len(result), 2)

        # Both parts should be FCD tensors
        for part in result:
            self.assertTrue(isinstance(part, (torch.Tensor, Tensor)))
            self.assertTrue(hasattr(part, "dims"))

        # Check that the dimensions are bound correctly
        self.assertIs(result[0].dims[0], x)
        self.assertIs(result[1].dims[0], y)
        self.assertEqual(x.size, 4)  # 8 / 2 = 4 each
        self.assertEqual(y.size, 4)

        # Test with repeated dimensions
        x2 = Dim("x2")
        result2 = torch.randn(8).split([x2, x2], dim=0)
        self.assertEqual(len(result2), 2)
        self.assertEqual(x2.size, 4)  # Both chunks should be size 4

    def test_plain_tensor_regular_split_still_works(self):
        # [FUNCTION DOC] Checks that a regular integer split of a plain tensor is unaffected:
        # three plain torch Tensor chunks of the right shape that lack a dims attribute.
        """Test that regular split on plain tensors still works without FCD args."""
        tensor = torch.randn(3, 12, 5)

        # Regular split without any FCD arguments should work normally
        result = tensor.split(4, dim=1)
        self.assertEqual(len(result), 3)  # 12 / 4 = 3
        for part in result:
            self.assertEqual(part.shape, (3, 4, 5))
            self.assertTrue(isinstance(part, torch.Tensor))
            self.assertFalse(hasattr(part, "dims"))  # Should be regular tensor


if __name__ == "__main__":
    run_tests()
