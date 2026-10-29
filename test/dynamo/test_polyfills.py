# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the group_tensors_by_device_and_dtype polyfill in dynamo: equivalence
# with the C++ implementation, empty input, grouping behavior, and traceability of the C++
# binding under torch.compile. No import-time side effects besides imports.

import torch
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase


class TestGroupTensorsByDeviceAndDtype(TestCase):
    # [CLASS DOC] Test case for the polyfill that groups parallel lists of tensors by (device,
    # dtype) key, returning for each key the grouped lists plus the original indices (empty when
    # indices are not requested).
    """Tests for the group_tensors_by_device_and_dtype polyfill."""

    def test_polyfill_matches_cpp_single_list(self):
        # [FUNCTION DOC] Builds one list of four CPU tensors with alternating float32 and float64
        # dtypes and checks the polyfill and the C++ function return the same keys, same grouped
        # list lengths, equal tensors (None must match None) and equal indices.
        """Test that polyfill matches C++ implementation for a single list."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        tensorlistlist = [list1]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_multiple_lists(self):
        # [FUNCTION DOC] Same comparison against C++ for three parallel lists (the second and
        # third are random tensors shaped like the first) with indices requested.
        """Test that polyfill matches C++ implementation for multiple lists."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_with_nones(self):
        # [FUNCTION DOC] Same comparison for three parallel lists where the middle list holds
        # only None entries; None entries must be preserved in the polyfill result as in C++.
        """Test that polyfill matches C++ implementation with None values."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
            torch.randn(4, dtype=torch.float32),
        ]
        list2 = [None for _ in list1]
        list3 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2, list3]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, True)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=True
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            self.assertEqual(cpp_indices, polyfill_indices)

    def test_polyfill_matches_cpp_without_indices(self):
        # [FUNCTION DOC] Compares polyfill and C++ with indices not requested; besides matching
        # C++, checks the polyfill indices are an empty list for each group.
        """Test that polyfill matches C++ implementation without indices."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        list1 = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float64),
        ]
        list2 = [torch.rand_like(t) for t in list1]
        tensorlistlist = [list1, list2]

        cpp_result = torch._C._group_tensors_by_device_and_dtype(tensorlistlist, False)
        polyfill_result = group_tensors_by_device_and_dtype(
            tensorlistlist, with_indices=False
        )

        self.assertEqual(set(cpp_result), set(polyfill_result))
        for key in cpp_result:
            cpp_lists, cpp_indices = cpp_result[key]
            polyfill_lists, polyfill_indices = polyfill_result[key]
            self.assertEqual(len(cpp_lists), len(polyfill_lists))
            for cpp_l, polyfill_l in zip(cpp_lists, polyfill_lists):
                self.assertEqual(len(cpp_l), len(polyfill_l))
                for cpp_t, polyfill_t in zip(cpp_l, polyfill_l):
                    if cpp_t is None:
                        self.assertIsNone(polyfill_t)
                    else:
                        self.assertTrue(torch.equal(cpp_t, polyfill_t))
            # When with_indices=False, indices should be empty
            self.assertEqual(cpp_indices, polyfill_indices)
            self.assertEqual(polyfill_indices, [])

    def test_polyfill_empty_input(self):
        # [FUNCTION DOC] Checks the polyfill returns an empty mapping for an empty outer list and
        # for an outer list holding one empty list.
        """Test that polyfill handles empty input correctly."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        result = group_tensors_by_device_and_dtype([], with_indices=True)
        self.assertEqual(result, {})

        result = group_tensors_by_device_and_dtype([[]], with_indices=True)
        self.assertEqual(result, {})

    def test_polyfill_groups_correctly(self):
        # [FUNCTION DOC] Groups four tensors (two float32, two float64) on CPU and checks two
        # groups keyed by device and dtype, each holding the right tensors in order, with index
        # lists for the float32 group and the float64 group matching positions in the input.
        """Test that polyfill groups tensors correctly by device and dtype."""
        from torch._dynamo.polyfills import group_tensors_by_device_and_dtype

        t_f32_0 = torch.randn(4, dtype=torch.float32)
        t_f32_1 = torch.randn(4, dtype=torch.float32)
        t_f64_0 = torch.randn(4, dtype=torch.float64)
        t_f64_1 = torch.randn(4, dtype=torch.float64)

        list1 = [t_f32_0, t_f64_0, t_f32_1, t_f64_1]
        tensorlistlist = [list1]

        result = group_tensors_by_device_and_dtype(tensorlistlist, with_indices=True)

        # Should have two groups: float32 and float64
        self.assertEqual(len(result), 2)

        device = torch.device("cpu")
        f32_key = (device, torch.float32)
        f64_key = (device, torch.float64)

        self.assertIn(f32_key, result)
        self.assertIn(f64_key, result)

        f32_lists, f32_indices = result[f32_key]
        f64_lists, f64_indices = result[f64_key]

        # Check float32 group
        self.assertEqual(len(f32_lists[0]), 2)
        self.assertTrue(torch.equal(f32_lists[0][0], t_f32_0))
        self.assertTrue(torch.equal(f32_lists[0][1], t_f32_1))
        self.assertEqual(f32_indices, [0, 2])

        # Check float64 group
        self.assertEqual(len(f64_lists[0]), 2)
        self.assertTrue(torch.equal(f64_lists[0][0], t_f64_0))
        self.assertTrue(torch.equal(f64_lists[0][1], t_f64_1))
        self.assertEqual(f64_indices, [1, 3])

    def test_group_tensors_traceable_with_compile(self):
        # [FUNCTION DOC] Compiles with full graph and a compile counter a function that calls the
        # C++ grouping binding on tensors and grads, sums over matching pairs, and checks the
        # compiled result is close to eager, at least one frame was compiled and operations were
        # traced.
        """Test that torch._C._group_tensors_by_device_and_dtype is traceable with torch.compile.

        This test verifies:
        1. The function can be compiled without graph breaks
        2. The frame is actually compiled (not skipped)
        3. There is tensor compute happening in the compiled graph
        """
        cnts = torch._dynamo.testing.CompileCounter()

        def fn(tensors, grads):
            # [FUNCTION DOC] HELPER: Groups the two lists by device and dtype, then adds up the
            # float sums of the elementwise sum of each non-None tensor and grad pair, returning
            # the scalar total.
            # Group tensors by device and dtype (this uses the polyfill under compile)
            grouped = torch._C._group_tensors_by_device_and_dtype(
                [tensors, grads], True
            )

            # Perform some tensor computation to ensure the frame is not skipped
            # Sum up results for each dtype group
            total = torch.tensor(0.0)
            for grouped_tensors, indices in grouped.values():
                tensor_list, grad_list = grouped_tensors
                for t, g in zip(tensor_list, grad_list):
                    if t is not None and g is not None:
                        # This adds tensor compute to ensure frame isn't skipped
                        total = total + (t + g).sum().float()
            return total

        tensors = [
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
            torch.randn(4, dtype=torch.float32),
        ]
        grads = [torch.randn_like(t) for t in tensors]

        # Run without compile to get expected result
        expected = fn(tensors, grads)

        # Run with compile
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        result = compiled_fn(tensors, grads)

        # Verify correctness
        self.assertTrue(torch.allclose(result, expected))

        # Verify compilation happened (frame was not skipped)
        self.assertGreaterEqual(
            cnts.frame_count, 1, "Expected at least 1 frame to be compiled"
        )

        # Verify there was tensor compute (op_count > 0 means tensor ops were traced)
        self.assertGreater(cnts.op_count, 0, "Expected tensor operations to be traced")


if __name__ == "__main__":
    run_tests()
