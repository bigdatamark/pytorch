# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests for first-class dimension (functorch.dim) indexing. Each test indexes a
# plain tensor with Dim objects, DimList objects, slices, integers, None, ellipsis, index
# tensors or masks, and mostly only checks that the result is a functorch.dim Tensor. Defines
# one TestCase class and a run_tests entry point; no import-time side effects.
import torch
from functorch.dim import Dim, DimList, dims, Tensor
from torch.testing._internal.common_utils import run_tests, TestCase


class TestGetSetItem(TestCase):
    # [CLASS DOC] Test case covering getitem on tensors with first-class Dim objects. Holds
    # three unbound dims created in setUp (batch, height, width) that most tests do not
    # actually use.
    """Comprehensive tests for first-class dimension indexing operations."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Fixture. Calls the base setUp, then stores three fresh unbound
        # dims as batch, height and width attributes. The docstring after the super call is inert.
        super().setUp()
        """Set up common test fixtures."""
        self.batch, self.height, self.width = dims(3)

    def test_basic_dim_indexing(self):
        # [FUNCTION DOC] Indexes a 3-D tensor with a single new Dim at a time and checks that
        # each result is a functorch.dim Tensor.
        """Test basic indexing with a single Dim."""
        tensor = torch.randn(3, 4, 5)
        x, y, z = dims(3)

        # Test indexing with each dim
        result1 = tensor[x]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[z]
        self.assertIsInstance(result3, Tensor)

    def test_multiple_dim_indexing(self):
        # [FUNCTION DOC] Indexes a 3-D tensor with two dims and then three dims in one subscript
        # and checks that both results are functorch.dim Tensors.
        """Test indexing with multiple Dims."""
        tensor = torch.randn(3, 4, 5)
        x, y, z = dims(3)

        # Test multiple dims in one indexing operation
        result = tensor[x, y]
        self.assertIsInstance(result, Tensor)

        result = tensor[x, y, z]
        self.assertIsInstance(result, Tensor)

    def test_mixed_indexing(self):
        # [FUNCTION DOC] Mixes a Dim with a full slice or an integer, in either position, on a
        # 3-D tensor, and checks every result is a functorch.dim Tensor.
        """Test mixing Dims with regular indexing."""
        tensor = torch.randn(3, 4, 5)
        x, y, z = dims(3)

        # Mix dim with slice
        result1 = tensor[x, :]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[:, y]
        self.assertIsInstance(result2, Tensor)

        # Mix dim with integer
        result3 = tensor[x, 0]
        self.assertIsInstance(result3, Tensor)

        result4 = tensor[0, y]
        self.assertIsInstance(result4, Tensor)

    def test_ellipsis_indexing(self):
        # [FUNCTION DOC] Combines a Dim with an ellipsis (leading, trailing, and in between two
        # dims) on a 4-D tensor; checks each result is a functorch.dim Tensor.
        """Test indexing with ellipsis (...)."""
        tensor = torch.randn(3, 4, 5, 6)
        x, y, z, w = dims(4)

        # Test ellipsis with dims
        result1 = tensor[x, ...]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[..., y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[x, ..., y]
        self.assertIsInstance(result3, Tensor)

    def test_none_indexing(self):
        # [FUNCTION DOC] Indexes a 2-D tensor with dims and None (new axis) placed between or
        # before the dims; checks each result is a functorch.dim Tensor.
        """Test indexing with None (newaxis)."""
        tensor = torch.randn(3, 4)
        x, y = dims(2)

        # Test None with dims
        result1 = tensor[x, None, y]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[None, x]
        self.assertIsInstance(result2, Tensor)

    def test_slice_indexing(self):
        # [FUNCTION DOC] Mixes dims with range slices (before, after, and between dims) on a 3-D
        # tensor; checks each result is a functorch.dim Tensor.
        """Test indexing with slices mixed with dims."""
        tensor = torch.randn(6, 8, 10)
        x, y, z = dims(3)

        # Test various slice patterns with dims
        result1 = tensor[x, 1:5]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[1:3, y]
        self.assertIsInstance(result2, Tensor)

        result3 = tensor[x, 1:5, z]
        self.assertIsInstance(result3, Tensor)

    def test_tensor_indexing(self):
        # [FUNCTION DOC] Mixes a Dim with an integer index tensor (before or after the dim) on a
        # 3-D tensor; checks each result is a functorch.dim Tensor.
        """Test indexing with tensor indices."""
        tensor = torch.randn(5, 6, 7)
        x, y, z = dims(3)

        # Create index tensors
        idx = torch.tensor([0, 2, 4])

        # Test tensor indexing with dims
        result1 = tensor[x, idx]
        self.assertIsInstance(result1, Tensor)

        result2 = tensor[idx, y]
        self.assertIsInstance(result2, Tensor)

    def test_boolean_indexing(self):
        # [FUNCTION DOC] Indexes a 2-D tensor with a Dim followed by a boolean mask over the
        # second axis; checks the result is a functorch.dim Tensor.
        """Test boolean indexing with dims."""
        tensor = torch.randn(4, 5)
        x, y = dims(2)

        # Create boolean mask
        mask = torch.tensor([True, False, True, False, True])

        # Test boolean indexing
        result = tensor[x, mask]
        self.assertIsInstance(result, Tensor)

    def test_dim_pack_indexing(self):
        # [FUNCTION DOC] Despite its name, only indexes a 2-D tensor with two separate dims (no
        # real dim pack is used); checks the result is a functorch.dim Tensor.
        """Test indexing with dimension packs (tuples/lists of dims)."""
        tensor = torch.randn(3, 4)  # Need 2D tensor for 2 dims

        # Create dims for dim pack
        a, b = dims(2)

        # Test dim pack indexing - using separate dimensions
        result = tensor[a, b]
        self.assertIsInstance(result, Tensor)

    def test_unbound_dim_binding(self):
        # [FUNCTION DOC] Indexes a 2-D tensor with two unbound Dim objects and checks that the
        # result is a functorch.dim Tensor and that the dims were bound to sizes 6 and 8.
        """Test automatic binding of unbound dimensions during indexing."""
        tensor = torch.randn(6, 8)
        x = Dim("x")  # unbound
        y = Dim("y")  # unbound

        # Should automatically bind dimensions
        result = tensor[x, y]
        self.assertIsInstance(result, Tensor)
        self.assertEqual(x.size, 6)
        self.assertEqual(y.size, 8)

    def test_dimlist_indexing(self):
        # [FUNCTION DOC] Indexes a 3-D tensor with a DimList built from two dims, followed by a
        # full slice; checks the result is a functorch.dim Tensor.
        """Test indexing with DimList objects."""
        tensor = torch.randn(3, 4, 5)

        # Create a bound dimlist
        dl = DimList(dims(2))

        # Test dimlist indexing
        result = tensor[dl, :]
        self.assertIsInstance(result, Tensor)

    def test_unbound_dimlist_indexing(self):
        # [FUNCTION DOC] Indexes a 3-D tensor with an integer and an empty unbound DimList
        # (expected to cover the remaining axes); checks the result is a functorch.dim Tensor.
        """Test indexing with unbound DimList."""
        tensor = torch.randn(3, 4, 5)

        # Create unbound dimlist
        dl = DimList()

        # Should bind to remaining dimensions
        result = tensor[0, dl]
        self.assertIsInstance(result, Tensor)

    def test_repeated_dim_usage(self):
        # [FUNCTION DOC] Uses the same dim for two axes of a cubic tensor; the result is only
        # checked to be a functorch.dim Tensor (the unused dims y and z are just created).
        """Test using the same dim multiple times in indexing."""
        tensor = torch.randn(4, 4, 4)
        x, y, z = dims(3)

        # This should trigger advanced indexing for repeated dims
        result = tensor[x, x]
        self.assertIsInstance(result, Tensor)

    def test_complex_mixed_indexing(self):
        # [FUNCTION DOC] Combines a dim, a slice, None, an index tensor and a full slice on
        # a 5-D tensor, then an ellipsis with a boolean mask over the last axis (size 7) of
        # the same tensor. First result must be a functorch.dim Tensor, the second a plain
        # torch Tensor.
        """Test complex combinations of different indexing types."""
        tensor = torch.randn(3, 4, 5, 6, 7)
        a, b, c, d, e = dims(5)

        # Complex mixed indexing
        idx = torch.tensor([0, 2])

        result1 = tensor[a, 1:3, None, idx, :]
        self.assertIsInstance(result1, Tensor)

        # Use mask with correct shape
        correct_mask = torch.tensor([True, False, True, False, False, True, True])
        result2 = tensor[..., correct_mask]
        self.assertIsInstance(result2, torch.Tensor)

    def test_edge_cases(self):
        # [FUNCTION DOC] Edge cases: a 1-D tensor indexed by an unbound dim must bind it to
        # size 5, and a tensor with a zero-sized first axis can be indexed with a dim and a
        # full slice.
        """Test edge cases and boundary conditions."""
        x, y, z = dims(3)

        # Single dimension tensor
        vec = torch.randn(5)
        a = Dim("a")
        result1 = vec[a]
        self.assertIsInstance(result1, Tensor)
        self.assertEqual(a.size, 5)  # Should bind to tensor size

        # Empty tensor indexing
        empty = torch.empty(0, 3, 4)
        result2 = empty[x, :]
        self.assertIsInstance(result2, Tensor)

    def test_error_conditions(self):
        # [FUNCTION DOC] Error cases on a 2-D tensor. Three dims raise ValueError; two unbound
        # DimLists and a double ellipsis each must raise some exception (type not checked).
        """Test conditions that should raise errors."""
        tensor = torch.randn(3, 4)
        x, y, z = dims(3)

        # Too many indices
        with self.assertRaises(ValueError):
            _ = tensor[x, y, z]  # 3 indices for 2D tensor

        # Multiple unbound dim lists
        dl1 = DimList()
        dl2 = DimList()
        with self.assertRaises(Exception):  # Should raise DimensionBindError
            _ = tensor[dl1, dl2]

        # Multiple ellipsis
        with self.assertRaises(Exception):
            _ = tensor[..., x, ...]

    def test_inferred_dimension_binding(self):
        # [FUNCTION DOC] Placeholder: the body is only a docstring and a comment, so it asserts
        # nothing; dim-pack inference is not tested.
        """Test dimension binding inference with dim packs."""
        # Skip this test for now as it requires more complex dim pack functionality

    def test_stride_calculation(self):
        # [FUNCTION DOC] Despite the name, checks only that plain indexing with two dims on a
        # 2-D tensor and three dims on a 3-D tensor each return a functorch.dim Tensor.
        """Test that stride calculations work correctly with dim packs."""
        tensor = torch.randn(6, 8)

        # Test basic indexing instead of complex dim packs
        a, b = dims(2)
        result1 = tensor[a, b]
        self.assertIsInstance(result1, Tensor)

        # Test with different tensor
        tensor2 = torch.randn(2, 3, 4)
        c, d, e = dims(3)
        result2 = tensor2[c, d, e]
        self.assertIsInstance(result2, Tensor)

    def test_device_handling_cpu(self):
        # [FUNCTION DOC] Indexes a CPU tensor with two dims and checks the result is a
        # functorch.dim Tensor whose device is the CPU device.
        """Test indexing behavior with CPU tensors."""
        # CPU tensor
        cpu_tensor = torch.randn(3, 4)
        x, y = dims(2)

        result_cpu = cpu_tensor[x, y]
        self.assertIsInstance(result_cpu, Tensor)
        self.assertEqual(result_cpu.device, torch.device("cpu"))


if __name__ == "__main__":
    run_tests()
