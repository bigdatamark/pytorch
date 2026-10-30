# Owner(s): ["module: inductor"]


# [MODULE DOC] Tests for the Inductor pre-grad FX passes that rewrite split, unbind, cat and stack
# patterns (normalization, split merging, split/cat merging, squeeze and unbind folding, and the
# slice/view conversion passes). Each test compiles small functions and checks numerics against
# eager plus the per-pass counters in the Inductor counters registry. The script entry point runs
# the tests only on Linux with a GPU.
import torch
from torch._dynamo.utils import counters
from torch._inductor.fx_passes.misc_patterns import numpy_compat_normalization
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.triton_utils import requires_gpu


def patch(f):
    # [FUNCTION DOC] HELPER: Decorator that wraps a test method in an Inductor config patch
    # enabling a fixed set of pre-grad passes (normalization, remove split with size one, merge
    # getitem cat, merge splits, mutate cat, split cat, unbind stack) and an empty post-grad
    # option set. Returns the patched function.
    f = torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
            "remove_split_with_size_one_pass": {},
            "merge_getitem_cat_pass": {},
            "merge_splits_pass": {},
            "mutate_cat_pass": {},
            "split_cat_pass": {},
            "unbind_stack_pass": {},
        },
        post_grad_fusion_options={},
    )(f)
    return f


class TestSplitCatFxPasses(TestCase):
    # [CLASS DOC] Test case class holding the split/cat FX pass tests. Each test builds many small
    # functions, compiles them with torch.compile and compares to eager results while checking
    # pass counters, which are cleared between cases.
    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_split_normalization(self):
        # [FUNCTION DOC] Checks the normalization pass for split calls: positional, keyword and
        # method-call spellings of split are canonicalized. Each case lists the expected
        # normalization_pass counter (zero when the sizes are already a list, 17 for nested
        # splits, zero for a reshape under dynamic shapes). Results are compared to eager and
        # counters cleared per case.
        def arg_only(x):
            # [FUNCTION DOC] HELPER: Split with positional size 2 along dim 1, returns relu of
            # each chunk.
            return [torch.relu(s) for s in torch.split(x, 2, 1)]

        def arg_only_dim0(x):
            # [FUNCTION DOC] HELPER: Split with positional size 2 along dim 0, returns relu of
            # each chunk.
            return [torch.relu(s) for s in torch.split(x, 2, 0)]

        def kwarg1(x):
            # [FUNCTION DOC] HELPER: Split with a keyword dim of 1, returns relu of each chunk.
            return [torch.relu(s) for s in torch.split(x, 2, dim=1)]

        def kwarg2(x):
            # [FUNCTION DOC] HELPER: Split using keyword split_size_or_sections and keyword dim,
            # returns relu of each chunk.
            return [
                torch.relu(s) for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def kwarg3(x):
            # [FUNCTION DOC] HELPER: Split with all arguments as keywords including tensor and a
            # negative dim, returns relu of each chunk.
            return [
                torch.relu(s)
                for s in torch.split(tensor=x, split_size_or_sections=2, dim=-1)
            ]

        def list_replace(x):
            # [FUNCTION DOC] HELPER: Split with an explicit list of section sizes; expected to
            # need no normalization.
            return [torch.relu(s) for s in torch.split(x, [16, 16], dim=1)]

        def multi_split(x):
            # [FUNCTION DOC] HELPER: Splits x by 2 along dim 1 and splits every chunk again;
            # returns a list of tuples of chunks.
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def unequal_split(x):
            # [FUNCTION DOC] HELPER: Split by size 3 along dim 1, which does not divide the
            # 32-wide input evenly; returns relu of chunks.
            return [torch.relu(s) for s in torch.split(x, 3, 1)]

        def arg_only_cm(x):
            # [FUNCTION DOC] HELPER: Tensor-method split with positional size and dim; returns
            # relu of each chunk.
            return [torch.relu(s) for s in x.split(2, 1)]

        def kwarg1_cm(x):
            # [FUNCTION DOC] HELPER: Tensor-method split with keyword dim; returns relu of each
            # chunk.
            return [torch.relu(s) for s in x.split(2, dim=1)]

        def kwarg2_cm(x):
            # [FUNCTION DOC] HELPER: Tensor-method split with keyword split_size and keyword dim;
            # returns relu of each chunk.
            return [torch.relu(s) for s in x.split(split_size=2, dim=1)]

        def multi_split_cm(x):
            # [FUNCTION DOC] HELPER: Nested tensor-method splits: each chunk of the outer split is
            # split again.
            return [s.split(2, 1) for s in x.split(2, 1)]

        def unequal_split_cm(x):
            # [FUNCTION DOC] HELPER: Tensor-method split by 3 along dim 1 (uneven chunks); returns
            # relu of chunks.
            return [torch.relu(s) for s in x.split(3, 1)]

        def cm_with_list(x):
            # [FUNCTION DOC] HELPER: Tensor-method split with a list of sizes and a negative dim;
            # returns relu of chunks.
            return [torch.relu(s) for s in x.split([16, 16], dim=-1)]

        def normalize_reshape_with_dynamic_shape(x):
            # [FUNCTION DOC] HELPER: Reshapes the input to shape 4 by 16; used with dynamic shapes
            # to check normalization does not fire.
            return x.reshape(4, 16)

        args = [
            torch.randn(2, 32),
        ]
        for fn, dynamic, expected_split_norm_count in [
            (arg_only, False, 1),
            (arg_only_dim0, False, 1),
            (kwarg1, False, 1),
            (kwarg2, False, 1),
            (kwarg3, False, 1),
            (list_replace, False, 0),
            (multi_split, False, 17),
            (unequal_split, False, 1),
            (arg_only_cm, False, 1),
            (kwarg1_cm, False, 1),
            (kwarg2_cm, False, 1),
            (multi_split_cm, False, 17),
            (unequal_split_cm, False, 1),
            (cm_with_list, False, 1),
            (normalize_reshape_with_dynamic_shape, True, 0),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=dynamic)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_split_norm_count,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            counters.clear()

    @torch._inductor.config.patch(
        pre_grad_fusion_options={
            "normalization_pass": {},
        },
        post_grad_fusion_options={},
    )
    def test_cat_normalization(self):
        # [FUNCTION DOC] Checks the normalization pass on concat: a call to torch.concat of split
        # results along dim 1 is expected to produce a normalization_pass counter of 2 (one for
        # the split and one for the concat). Compares to eager and clears counters.
        def caoncat_only(x):
            # [FUNCTION DOC] HELPER: Concatenates the chunks of a split by 2 along dim 1 using
            # torch.concat.
            return torch.concat(list(torch.split(x, 2, 1)), dim=1)

        args = [
            torch.randn(2, 32),
        ]
        for fn, dynamic, expected_cat_norm_count in [
            (caoncat_only, False, 2),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn, dynamic=dynamic)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_cat_norm_count,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            counters.clear()

    @patch
    def test_consecutive_split_merge(self):
        # [FUNCTION DOC] Checks the merge_splits_pass: consecutive splits (a split whose users are
        # further splits) are merged. For each function the expected merge_splits_pass counter is
        # given (zero for non-mergeable cases such as differing dims or keyword forms). Results
        # are compared with eager and counters cleared per case.
        def multi_split(x):
            # [FUNCTION DOC] HELPER: Nested splits by 2 along dim 1; expected not to be merged.
            return [torch.split(s, 2, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2(x):
            # [FUNCTION DOC] HELPER: Outer split by 2 and inner splits by 1 along dim 1; expected
            # to merge.
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, 1)]

        def multi_split_2_neg_dim(x):
            # [FUNCTION DOC] HELPER: Like the two-level split by 2 then 1, but the outer split
            # uses a negative dim.
            return [torch.split(s, 1, 1) for s in torch.split(x, 2, -1)]

        def multi_split_with_sizes(x):
            # [FUNCTION DOC] HELPER: Outer split with explicit sizes of 16 and 16 and inner splits
            # by 2.
            return [torch.split(s, 2, 1) for s in torch.split(x, [16, 16], 1)]

        def multi_split_kwarg1(x):
            # [FUNCTION DOC] HELPER: Nested splits written with keyword dim.
            return [torch.split(s, 2, dim=1) for s in torch.split(x, 2, dim=1)]

        def multi_split_kwarg2(x):
            # [FUNCTION DOC] HELPER: Nested splits written with keyword split_size_or_sections and
            # keyword dim.
            return [
                torch.split(s, split_size_or_sections=2, dim=1)
                for s in torch.split(x, split_size_or_sections=2, dim=1)
            ]

        def unequal_multi_split(x):
            # [FUNCTION DOC] HELPER: Splits into unequal sections of 10, 10 and 12, then splits
            # each section again with its own sizes, and returns relu of all final pieces.
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.extend(item2.split([4, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        def unequal_multi_split_neg_index(x):
            # [FUNCTION DOC] HELPER: Same shape of pattern as the unequal multi split but indexes
            # the outer split with negative indices and uses different sizes so cached patterns
            # are not reused.
            fs = torch.split(x, [8, 8, 16], dim=1)
            item0 = fs[-3]
            item1 = fs[-2]
            item2 = fs[-1]

            # Use different splits from `unequal_multi_split` to avoid cache hit.
            final_items = []
            final_items.extend(item0.split([3, 5], 1))
            final_items.extend(item1.split([5, 3], 1))
            final_items.extend(item2.split([8, 4, 4], 1))

            return [torch.relu(s) for s in final_items]

        # Shouldn't merge
        def diff_dims(x):
            # [FUNCTION DOC] HELPER: Outer split along dim 1 and inner splits along dim 0; dims
            # differ so no merge should happen.
            return [torch.split(s, 2, dim=0) for s in torch.split(x, 2, dim=1)]

        def some_users_not_splits(x):
            # [FUNCTION DOC] HELPER: Unequal outer split where two sections are split again and
            # the third is passed through sin; some users are not splits.
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1.split([6, 4], 1))
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_with_cat(x):
            # [FUNCTION DOC] HELPER: Outer split of sizes 4, 4 and 24; the third section is split
            # into six pieces and everything is concatenated along dim 1.
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def duplicate_getitems(x):
            # [FUNCTION DOC] HELPER: Unequal outer split where the second item is fetched twice;
            # one copy is split, the other is extended into the output list directly, and the
            # third goes through sin.
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[1]
            item2 = fs[2]

            final_items = []
            final_items.extend(item0.split([4, 6], 1))
            final_items.extend(item1_1.split([6, 4], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def duplicate_getitems_neg_index(x):
            # [FUNCTION DOC] HELPER: Like the duplicate getitem case but the second duplicate is
            # fetched with a negative index; uses different sizes to avoid cache hits.
            fs = torch.split(x, [10, 10, 12], dim=1)
            item0 = fs[0]
            item1_1 = fs[1]
            item1_2 = fs[-2]  # negative index
            item2 = fs[2]

            # Use different splits from `duplicate_getitems` to avoid cache hit.
            final_items = []
            final_items.extend(item0.split([3, 7], 1))
            final_items.extend(item1_1.split([7, 3], 1))
            final_items.extend(item1_2)
            final_items.append(torch.sin(item2))

            return [torch.relu(s) for s in final_items]

        def split_getitem_gap(x):
            # [FUNCTION DOC] HELPER: Outer split where only items 0 and 2 are used (a gap at item
            # 1), item 2 is split again, and the pieces are concatenated.
            # Clashes with `split_partial_getitem_cat`
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def split_partial_getitem_cat(x):
            # [FUNCTION DOC] HELPER: Partial use of an outer split (items 0 and 2) with different
            # sizes than the gap case, then concatenation of the pieces.
            # Use different splits from `split_getitem_gap` to avoid cache hit.
            fs = torch.split(x, [2, 6, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def split_getitem_out_of_order(x):
            # [FUNCTION DOC] HELPER: Outer split of four sections whose items are used out of
            # order (0, 2, 1) while the last is split further; result is concatenated.
            fs = torch.split(x, [4, 4, 4, 20], dim=1)
            item0 = fs[0]
            item2 = fs[2]
            item1 = fs[1]
            item3 = fs[3]

            final_items = [item0, item2, item1]
            final_items.extend(item3.split((4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        def next_split_getitem_partial_used(x):
            # [FUNCTION DOC] HELPER: Second-level split whose results are only partly used through
            # two slices of the getitem list, concatenated with item 0.
            fs = torch.split(x, [4, 4, 24], dim=1)
            item0 = fs[0]
            item2 = fs[2]

            final_items = [item0]
            ns = item2.split((4, 4, 4, 4, 4, 4), 1)
            final_items.extend(ns[0:1])
            final_items.extend(ns[3:4])

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]
        for fn, expected_split_merged in [
            (multi_split, 0),
            (multi_split_2, 16),
            (multi_split_2_neg_dim, 16),
            (multi_split_with_sizes, 2),
            (multi_split_kwarg1, 0),
            (multi_split_kwarg2, 0),
            (unequal_multi_split, 3),
            (unequal_multi_split_neg_index, 3),
            (diff_dims, 0),
            (some_users_not_splits, 2),
            (split_with_cat, 1),
            (duplicate_getitems, 1),
            (duplicate_getitems_neg_index, 1),
            (split_getitem_gap, 1),
            (split_partial_getitem_cat, 1),
            (split_getitem_out_of_order, 1),
            (next_split_getitem_partial_used, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["merge_splits_pass"],
                expected_split_merged,
            )
            counters.clear()

    @patch
    def test_split_cat_merge(self):
        # [FUNCTION DOC] Checks the split/cat merge pass. Many cases combine split with cat or
        # stack in different argument forms, dimension mismatches and input shuffling; for each
        # the test checks the scmerge counters for splits added or removed, cats added or removed
        # and split sections removed, against tuples given in the table. Compares results to eager
        # and clears counters per case.
        def simple_split_cat(x):
            # [FUNCTION DOC] HELPER: Concatenates all chunks of a split by 4 along dim 1.
            return torch.cat(torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec1(x):
            # [FUNCTION DOC] HELPER: Same as the simple split and cat but passes dim positionally.
            return torch.cat(torch.split(x, 4, dim=1), 1)

        def simple_split_cat_argspec2(x):
            # [FUNCTION DOC] HELPER: Concatenation with keyword tensors and keyword dim.
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=1)

        def simple_split_cat_argspec3(x):
            # [FUNCTION DOC] HELPER: Concatenation along a negative dim of -2 positionally.
            return torch.cat(torch.split(x, 4, dim=1), -2)

        def simple_split_cat_argspec4(x):
            # [FUNCTION DOC] HELPER: Concatenation using keyword tensors and keyword negative dim.
            return torch.cat(tensors=torch.split(x, 4, dim=1), dim=-2)

        def simple_split_stack(x):
            # [FUNCTION DOC] HELPER: Stacks the chunks of a split along dim 1 using keyword dim.
            return torch.stack(torch.split(x, 4, dim=1), dim=1)

        def simple_split_stack_argspec1(x):
            # [FUNCTION DOC] HELPER: Stacks the chunks of a split with positional dim.
            return torch.stack(torch.split(x, 4, dim=1), 1)

        def simple_split_stack_argspec2(x):
            # [FUNCTION DOC] HELPER: Stacks the chunks of a split with keyword tensors and keyword
            # dim.
            return torch.stack(tensors=torch.split(x, 4, dim=1), dim=1)

        def split_cat_addn_args(x):
            # [FUNCTION DOC] HELPER: Concatenates split chunks with extra ones tensors added
            # before and after along dim 1.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 5, 32, 16)] + split_output + [torch.ones(2, 6, 32, 16)],
                dim=1,
            )

        def split_stack_addn_args(x):
            # [FUNCTION DOC] HELPER: Stacks split chunks with extra ones tensors added before and
            # after along dim 1.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + split_output
                + [torch.ones(2, 4, 32, 16), torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def split_cat_addn_args_dim2(x):
            # [FUNCTION DOC] HELPER: Splits along dim 2 and concatenates with extra tensors before
            # and after along dim 2.
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 5, 16)] + split_output + [torch.ones(2, 32, 6, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def split_cat_dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Split along dim 1 but concatenation along dim 2 with extra
            # tensors around the chunks.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def split_stack_dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Split along dim 1 but stack along dim 2 with extra tensors
            # around the chunks.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=3
        def split_cat_dim_mismatch2(x):
            # [FUNCTION DOC] HELPER: Split along dim 1 but concatenation along dim 3 with extra
            # tensors around the chunks.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        def split_stack_dim_mismatch2(x):
            # [FUNCTION DOC] HELPER: Split along dim 1 but stack along dim 3 with extra tensors
            # around the chunks.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)] + split_output + [torch.ones(2, 4, 32, 16)],
                dim=3,
            )

        # split_dim=2, cat_dim=0
        def split_cat_dim_mismatch3(x):
            # [FUNCTION DOC] HELPER: Split along dim 2 but concatenation along dim 0 with extra
            # tensors around the chunks.
            split_output = list(torch.split(x, 4, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def split_stack_dim_mismatch3(x):
            # [FUNCTION DOC] HELPER: Split along dim 2 but stack along dim 0 with extra tensors
            # around the chunks.
            split_output = list(torch.split(x, 4, dim=2))
            return torch.stack(
                [torch.ones(2, 32, 4, 16)] + split_output + [torch.ones(2, 32, 4, 16)],
                dim=0,
            )

        def input_shuffling(x):
            # [FUNCTION DOC] HELPER: Concatenates along dim 1 a mix of ones tensors and non-
            # contiguous groups of split chunks (1-3 and 5-7), so inputs are shuffled relative to
            # the split order.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_stack(x):
            # [FUNCTION DOC] HELPER: Stack version of the input shuffling case along dim 1.
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=1,
            )

        def input_shuffling_dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Input shuffling with concatenation along dim 2 (different
            # from the split dim).
            split_output = list(torch.split(x, 4, dim=1))
            return torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_dim_mismatch_stack(x):
            # [FUNCTION DOC] HELPER: Input shuffling with stack along dim 2 (different from the
            # split dim).
            split_output = list(torch.split(x, 4, dim=1))
            return torch.stack(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output[5], split_output[6], split_output[7]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

        def input_shuffling_multiple_output(x):
            # [FUNCTION DOC] HELPER: Uses chunks 1-3 in a concatenation along dim 2, chunks 4-5 in
            # a stack along dim 1, and relu of chunk 6; returns all three results.
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def input_shuffling_direct_output(x):
            # [FUNCTION DOC] HELPER: Like the multiple output case but returns chunk 6 of the
            # split directly instead of through relu.
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            return cat1, stack1, split_output[6]

        def input_shuffling_multiple_output_same_ranges(x):
            # [FUNCTION DOC] HELPER: Two identical concatenations of the same chunk range plus a
            # stack and a relu of another chunk; returns all four.
            split_output = list(torch.split(x, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )

            cat2 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, cat2, stack1, relu1

        def unequal_split_multiple_output(x):
            # [FUNCTION DOC] HELPER: Multiple-output case with an unequal split of eight sections
            # of sizes 2, 4, 4, 4, 4, 4, 8 and 2.
            split_output = list(torch.split(x, [2, 4, 4, 4, 4, 4, 8, 2], dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output[1], split_output[2], split_output[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output[4],
                    split_output[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output[6])

            return cat1, stack1, relu1

        def multi_split_cat(x1, x2):
            # [FUNCTION DOC] HELPER: Two inputs are split and used in one concatenation along dim
            # 2 and one stack along dim 1, with relu on the last used chunks; returns four
            # results.
            split_output_1 = list(torch.split(x1, 4, dim=1))
            split_output_2 = list(torch.split(x2, 4, dim=1))
            cat1 = torch.cat(
                [torch.ones(2, 4, 32, 16)]
                + [split_output_1[1], split_output_1[2], split_output_1[3]]
                + [torch.ones(2, 4, 32, 16)]
                + [split_output_2[1], split_output_2[2], split_output_2[3]]
                + [torch.ones(2, 4, 32, 16)],
                dim=2,
            )
            stack1 = torch.stack(
                [
                    torch.ones(2, 4, 32, 16),
                    split_output_1[4],
                    split_output_1[5],
                    torch.ones(2, 4, 32, 16),
                    split_output_2[4],
                    split_output_2[5],
                    torch.ones(2, 4, 32, 16),
                ],
                dim=1,
            )

            relu1 = torch.relu(split_output_1[6])
            relu2 = torch.relu(split_output_2[6])

            return cat1, stack1, relu1, relu2

        # TODO: Add more tests:
        # * Cases where replacement shouldn't happen
        default_args = [
            torch.randn(2, 32, 32, 16),
        ]
        multi_args = [
            torch.randn(2, 32, 32, 16),
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_split_added,
            expected_split_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
            args,
        ) in [
            (simple_split_cat, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec1, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec2, 0, 0, 0, 0, 0, default_args),
            (simple_split_cat_argspec3, 0, 1, 0, 1, 7, default_args),
            (simple_split_cat_argspec4, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec1, 0, 1, 0, 1, 7, default_args),
            (simple_split_stack_argspec2, 0, 1, 0, 1, 7, default_args),
            (split_cat_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_stack_addn_args, 0, 1, 1, 1, 7, default_args),
            (split_cat_addn_args_dim2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch2, 0, 1, 1, 1, 7, default_args),
            (split_cat_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (split_stack_dim_mismatch3, 0, 1, 1, 1, 7, default_args),
            (input_shuffling, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_dim_mismatch_stack, 1, 1, 1, 1, 4, default_args),
            (input_shuffling_multiple_output, 1, 1, 2, 2, 3, default_args),
            (input_shuffling_direct_output, 1, 1, 2, 2, 3, default_args),
            (unequal_split_multiple_output, 1, 1, 2, 2, 3, default_args),
            (multi_split_cat, 1, 1, 2, 2, 3, multi_args),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_split_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_split_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
            )
            counters.clear()

    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={},
    )
    def test_config_flag_is_respected(self):
        # [FUNCTION DOC] Checks that with empty pre-grad and post-grad fusion options neither
        # merge_splits_pass nor normalization_pass fires (both counters are zero), while the
        # compiled result still matches eager.
        def split_with_cat(x):
            # [FUNCTION DOC] HELPER: Splits on the last dim into 4, 4 and 24, splits the third
            # part into six pieces, and concatenates along dim 1.
            fs = torch.split(x, [4, 4, 24], dim=-1)
            item0 = fs[0]
            item1 = fs[1]
            item2 = fs[2]

            final_items = [item0, item1]
            final_items.extend(item2.split((4, 4, 4, 4, 4, 4), 1))

            return torch.cat(final_items, dim=1)

        args = [
            torch.randn(2, 32),
        ]

        expected = split_with_cat(*args)
        actual = torch.compile(split_with_cat)(*args)

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            counters["inductor"]["merge_splits_pass"],
            0,
        )
        self.assertEqual(
            counters["inductor"]["normalization_pass"],
            0,
        )

    @patch
    def test_split_cat_merge_mutation(self):
        # [FUNCTION DOC] Checks that split/cat merging does not remove splits or cats when a chunk
        # is mutated in place: results match eager and the scmerge removal counters stay at zero.
        args = [
            torch.randn(2, 32, 32, 16),
        ]

        def split_cat_mutation(x):
            # [FUNCTION DOC] HELPER: Splits by 4 along dim 1, copies chunk 0 into chunk 1 in
            # place, then concatenates all chunks.
            splits = torch.split(x, 4, dim=1)
            splits[1].copy_(splits[0])
            return torch.cat(splits, dim=1)

        expected = split_cat_mutation(*args)
        actual = torch.compile(split_cat_mutation)(*args)

        torch.testing.assert_close(actual, expected)

        self.assertEqual(counters["inductor"]["scmerge_split_removed"], 0)
        self.assertEqual(counters["inductor"]["scmerge_cat_removed"], 0)

    @patch
    def test_split_squeeze(self):
        # [FUNCTION DOC] Checks the split_cat_pass that folds split followed by squeeze of each
        # chunk and stack. Each case has an expected split_cat_pass count; folding is skipped for
        # none-dim squeeze, split size not 1, mismatched dim and extra users of the splits.
        # Compares to eager and clears counters.
        def split_squeeze_stack(x):
            # [FUNCTION DOC] HELPER: Splits to size 1 chunks along dim 1, squeezes dim 1 of each,
            # and stacks.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod(x):
            # [FUNCTION DOC] HELPER: Same as the split squeeze stack but uses the squeeze method
            # with an explicit dim.
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_callmethod_none_dim(x):
            # [FUNCTION DOC] HELPER: Uses the squeeze method with no dim; not handled by the pass
            # (marked TODO in the test).
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze() for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1(x):
            # [FUNCTION DOC] HELPER: Squeeze with keyword dim in function form.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_stack_kwarg1_callmethod(x):
            # [FUNCTION DOC] HELPER: Squeeze method with keyword dim.
            items = list(torch.split(x, 1, dim=1))
            split_items = [s.squeeze(dim=1) for s in items]
            return torch.stack(split_items)

        def split_squeeze_multi_squeeze_users(x):
            # [FUNCTION DOC] HELPER: Split, squeeze and stack where the squeezed values also feed
            # relu and tanh as extra outputs.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return (
                torch.stack(split_items),
                torch.relu(split_items[0]),
                torch.tanh(split_items[1]),
            )

        def split_size_not_1(x):
            # [FUNCTION DOC] HELPER: Splits with size 2, then squeezes dim 1 of each chunk and
            # stacks; chunk size is not 1.
            items = list(torch.split(x, 2, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items)

        def dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Splits along dim 1 but squeezes dim 0, so dims mismatch.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 0) for s in items]
            return torch.stack(split_items)

        def other_users(x):
            # [FUNCTION DOC] HELPER: Squeeze and stack all chunks while also returning relu of the
            # first unsqueezed chunk.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items), torch.relu(items[0])

        def other_users_2(x):
            # [FUNCTION DOC] HELPER: Squeezes only chunks after the first and stacks them; the
            # first chunk is also used by relu.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items), torch.relu(items[0])

        def graph_should_be_topological_sorted(x):
            # [FUNCTION DOC] HELPER: Splits along dim 0 by a loop, squeezes dim 0 of each piece,
            # applies sin and stacks; checks the pass output stays topologically sorted.
            output = []
            for t in x.split(1):
                output.append(torch.sin(t.squeeze(dim=0)))
            output = torch.stack(output)
            return output

        args = [
            torch.randn(2, 32),
        ]
        for fn, split_squeeze_replaced in [
            (split_squeeze_stack, 1),
            (split_squeeze_stack_callmethod, 1),
            # TODO handle none dim
            (split_squeeze_stack_callmethod_none_dim, 0),
            (split_squeeze_stack_kwarg1, 1),
            (split_squeeze_stack_kwarg1_callmethod, 1),
            (split_squeeze_multi_squeeze_users, 1),
            (split_size_not_1, 0),
            (dim_mismatch, 0),
            (other_users, 0),
            (other_users_2, 0),
            (graph_should_be_topological_sorted, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["split_cat_pass"],
                split_squeeze_replaced,
            )
            counters.clear()

    @patch
    def test_unbind_stack(self):
        # [FUNCTION DOC] Checks the unbind and stack related folding. For each function it checks
        # scmerge counters for split added and removed, cat added and removed and sections
        # removed, plus the normalization_pass count; some cases expect no change. Compares to
        # eager and clears counters.
        def unbind_stack(x):
            # [FUNCTION DOC] HELPER: Stacks the result of unbinding along dim 1, back along dim 1.
            return torch.stack(torch.unbind(x, 1), 1)

        def unbind_cat(x):
            # [FUNCTION DOC] HELPER: Concatenates the unbind pieces along dim 1 after unbinding on
            # a negative dim.
            return torch.cat(torch.unbind(x, dim=-3), 1)

        def unbind_stack_argspec1(x):
            # [FUNCTION DOC] HELPER: Unbind with keyword input and keyword dim, then stack.
            return torch.stack(torch.unbind(input=x, dim=1), dim=1)

        def unbind_stack_argspec2(x):
            # [FUNCTION DOC] HELPER: Stack with keyword tensors of an unbind.
            return torch.stack(tensors=torch.unbind(x, dim=1), dim=1)

        def dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 1 but stack along dim 0, so dims mismatch.
            return torch.stack(torch.unbind(x, dim=1), 0)

        def split_squeeze_stack(x):
            # [FUNCTION DOC] HELPER: Splits to size 1 chunks, squeezes dim 1 of each and stacks
            # along dim 1.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def split_squeeze_stack_callmethod(x):
            # [FUNCTION DOC] HELPER: Same as the split squeeze stack variant; despite its name it
            # also uses the function form of squeeze.
            # KNOWN ISSUE: the name says call-method but the body uses the function form of
            # squeeze, so it duplicates the previous case and skips the method form.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1)

        def other_users(x):
            # [FUNCTION DOC] HELPER: Split, squeeze and stack where chunk 0 also has another user
            # via relu.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def other_users_2(x):
            # [FUNCTION DOC] HELPER: Squeezes only chunks after the first and stacks them while
            # chunk 0 feeds relu.
            items = list(torch.split(x, 1, dim=1))
            split_items = [torch.squeeze(s, 1) for s in items[1:]]
            return torch.stack(split_items, 1), torch.relu(items[0])

        def unbind_cat_addn_args(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 1 and concatenate with extra ones tensors
            # added at both ends.
            split_output = list(torch.unbind(x, dim=1))

            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_stack_addn_args(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 1 and stack with extra ones tensors at both
            # ends.
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_addn_args_dim2(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 2 and concatenate along dim 2 with extra
            # tensors at both ends.
            split_output = list(torch.unbind(x, dim=2))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        # split_dim=1, cat_dim=2
        def unbind_cat_dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 1 but concatenate along dim 2 with extra
            # tensors.
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_stack_dim_mismatch(x):
            # [FUNCTION DOC] HELPER: Unbind along dim 1 but stack along dim 2 with extra tensors.
            split_output = list(torch.unbind(x, dim=1))
            return torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        def unbind_cat_multi_users(x):
            # [FUNCTION DOC] HELPER: Uses the same unbind result in both a concatenation and a
            # stack along dim 1; returns both.
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)]
                + split_output
                + [torch.ones(2, 32, 16), torch.ones(2, 32, 16)],
                dim=1,
            )

        def unbind_cat_multi_users_diff_dims(x):
            # [FUNCTION DOC] HELPER: Uses the same unbind result in a concatenation along dim 1
            # and a stack along dim 2; returns both.
            split_output = list(torch.unbind(x, dim=1))
            return torch.cat(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=1,
            ), torch.stack(
                [torch.ones(2, 32, 16)] + split_output + [torch.ones(2, 32, 16)],
                dim=2,
            )

        args = [
            torch.randn(2, 32, 32, 16),
        ]
        for (
            fn,
            expected_unbind_added,
            expected_unbind_removed,
            expected_cat_added,
            expected_cat_removed,
            expected_sections_removed,
            expected_unbind_normalized,
        ) in [
            (unbind_stack, 0, 1, 0, 1, 31, 2),
            (unbind_stack_argspec1, 0, 1, 0, 1, 31, 2),
            (unbind_stack_argspec2, 0, 1, 0, 1, 31, 2),
            (dim_mismatch, 0, 1, 0, 1, 31, 2),
            (split_squeeze_stack, 0, 1, 0, 1, 31, 2),
            (split_squeeze_stack_callmethod, 0, 1, 0, 1, 31, 2),
            (other_users, 0, 0, 0, 0, 0, 2),
            (other_users_2, 0, 0, 0, 0, 0, 2),
            (unbind_cat_addn_args, 0, 1, 1, 1, 31, 1),
            (unbind_stack_addn_args, 0, 1, 1, 1, 31, 2),
            (unbind_cat_addn_args_dim2, 0, 1, 1, 1, 31, 1),
            (unbind_cat_dim_mismatch, 0, 1, 1, 1, 31, 1),
            (unbind_stack_dim_mismatch, 0, 1, 1, 1, 31, 2),
            (unbind_cat_multi_users, 0, 1, 2, 2, 31, 2),
            (unbind_cat_multi_users_diff_dims, 0, 1, 2, 2, 31, 2),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["scmerge_split_added"],
                expected_unbind_added,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_removed"],
                expected_unbind_removed,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_added"],
                expected_cat_added,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_cat_removed"],
                expected_cat_removed,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            self.assertEqual(
                counters["inductor"]["scmerge_split_sections_removed"],
                expected_sections_removed,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            self.assertEqual(
                counters["inductor"]["normalization_pass"],
                expected_unbind_normalized,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            counters.clear()

    @patch
    def test_split_cat_new_patterns(self):
        # [FUNCTION DOC] Checks several newer split/cat passes through a table of functions with
        # expected counters for merge_getitem_cat_pass, mutate_cat_pass, split_cat_to_slices_pass,
        # unbind_cat_to_view_pass, split_stack_to_cats_pass, unbind_stack_to_slices_pass and
        # move_reshape_out_of_split_stack_pass. Inputs are a 500 by 500 random tensor; results are
        # compared to eager and counters cleared per case.
        def split_cat_split(x):
            # [FUNCTION DOC] HELPER: Splits into eleven sections, concatenates pairs of sections,
            # splits those again, and concatenates the results with the remaining sections along
            # dim 1 using positional arguments.
            l1_out = torch.split(x, [200, 50, 50, 20, 20, 20, 20, 20, 20, 50, 30], 1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat_1 = torch.cat((item0, item1), 1)
            cat_2 = torch.cat((item9, item10), 1)
            l2_out = torch.split(cat_1, [50, 120, 80], 1)
            l3_out = torch.split(cat_2, [10, 20, 50], 1)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                1,
            )
            return output

        def split_cat_split_kwarg(x):
            # [FUNCTION DOC] HELPER: Same graph as the split cat split case but written with
            # keyword dim arguments.
            l1_out = torch.split(
                x, [200, 50, 50, 20, 20, 20, 20, 20, 20, 50, 30], dim=1
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat_1 = torch.cat((item0, item1), dim=1)
            cat_2 = torch.cat((item9, item10), dim=1)
            l2_out = torch.split(cat_1, [50, 120, 80], dim=1)
            l3_out = torch.split(cat_2, [10, 20, 50], dim=1)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                dim=1,
            )
            return output

        def remove_cat_node_with_all_getitmes(x):
            # [FUNCTION DOC] HELPER: Splits along dim 0 into eleven sections and builds a cat of
            # all sections plus other small cats and splits; final result concatenates the
            # combined output with the full cat.
            l1_out = torch.split(
                x, [50, 50, 200, 20, 20, 20, 20, 20, 40, 10, 50], dim=0
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat = torch.cat(
                (
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                    item10,
                ),
                dim=0,
            )
            cat_1 = torch.cat((item0, item1), dim=0)
            cat_2 = torch.cat((item0, item10), dim=0)
            l2_out = torch.split(cat_1, [20, 30, 50], dim=0)
            l3_out = torch.split(cat_2, [10, 60, 30], dim=0)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                ],
                dim=0,
            )
            return torch.cat((output, cat), dim=0)

        def mutate_cat_node_with_some_getitmes(x):
            # [FUNCTION DOC] HELPER: Like the case removing a cat with all getitems but the first
            # cat uses only a subset of sections and the final output uses fewer items, so the cat
            # is mutated rather than removed.
            l1_out = torch.split(
                x, [50, 50, 200, 20, 20, 20, 20, 20, 40, 10, 50], dim=0
            )
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item10 = l1_out[10]
            cat = torch.cat(
                (
                    item6,
                    item7,
                    item8,
                    item9,
                    item10,
                    item2,
                    item3,
                    item4,
                    item5,
                ),
                dim=0,
            )
            cat_1 = torch.cat((item0, item1), dim=0)
            cat_2 = torch.cat((item0, item10), dim=0)
            l2_out = torch.split(cat_1, [20, 30, 50], dim=0)
            l3_out = torch.split(cat_2, [10, 60, 30], dim=0)
            item11 = l2_out[0]
            item12 = l2_out[1]
            item13 = l2_out[2]
            item14 = l3_out[0]
            item15 = l3_out[1]
            item16 = l3_out[2]

            output = torch.cat(
                [
                    item11,
                    item12,
                    item13,
                    item14,
                    item15,
                    item16,
                    item2,
                ],
                dim=0,
            )
            return torch.cat((output, cat), dim=0)

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_cat_to_slices_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_cat_to_slices(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable only the split_cat_to_slices pass.
            # Splits a tensor and two clones into chunks and concatenates a mix of them along dim
            # 0 together with an extra cloned chunk.
            x_c = x.clone()
            x_c_2 = x.clone()
            l1_out = torch.split(x, [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], dim=0)
            l2_out = torch.split(x_c, [50, 50, 50, 50, 50, 50, 50, 50, 50, 50], dim=0)
            l3_out = torch.split(x_c_2, [100, 100, 100, 100, 100], dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item0_c = l2_out[0]
            item1_c = l2_out[1]
            item2_c = l2_out[2]
            item3_c = l2_out[3]
            item4_c = l2_out[4]
            item5_c = l2_out[5]
            item6_c = l2_out[6]
            item7_c = l2_out[7]
            item8_c = l2_out[8]
            item9_c = l2_out[9]
            item0_c_2 = l3_out[0]
            item1_c_2 = l3_out[1]
            item2_c_2 = l3_out[2]
            item3_c_2 = l3_out[3]
            item4_c_2 = l3_out[4]
            other = item0.clone()
            return torch.cat(
                [
                    other,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                    item4_c,
                    item5_c,
                    item6_c,
                    item7_c,
                    item8_c,
                    item9_c,
                    item0_c,
                    item1_c,
                    item2_c,
                    item3_c,
                    item0_c_2,
                    item1_c_2,
                    item2_c_2,
                    item3_c_2,
                    item4_c_2,
                ],
                dim=0,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "unbind_cat_to_view_pass": {},
            },
            post_grad_fusion_options={},
        )
        def unbind_cat_to_view(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable only the unbind_cat_to_view pass.
            # Views the input as 10 by 50 by 500 twice, unbinds both along dim 0, and concatenates
            # selected pieces and clones along dim 1.
            y = x.view(10, 50, 500)
            z = x.view(10, 50, 500)
            l1_out = torch.unbind(y, dim=0)
            l2_out = torch.unbind(z, dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            item2_0 = l2_out[0]
            item2_1 = l2_out[1]
            item2_2 = l2_out[2]
            item2_3 = l2_out[3]
            item2_4 = l2_out[4]
            item2_5 = l2_out[5]
            item2_6 = l2_out[6]
            item2_7 = l2_out[7]
            item2_8 = l2_out[8]
            item2_9 = l2_out[9]
            other1 = item7.clone()
            other2 = item8.clone()
            other3 = item9.clone()
            cat = torch.cat(
                [
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    other1,
                    item2_0,
                    item2_1,
                    item2_2,
                    item2_3,
                    item2_4,
                    item2_5,
                    item2_6,
                    item2_7,
                    item2_8,
                    item2_9,
                    other2,
                    other3,
                ],
                dim=1,
            )
            return cat

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_stack_to_cats_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_stack_to_cats_same_dim(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable only the split_stack_to_cats pass.
            # Unbinds a view, splits six items into halves along dim 1, and stacks the halves with
            # clones along dim 1.
            x_c = x.view(10, 50, 500)
            l1_out = torch.unbind(x_c, dim=0)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            split1 = torch.split(item0, [250, 250], dim=1)
            split2 = torch.split(item1, [250, 250], dim=1)
            split3 = torch.split(item2, [250, 250], dim=1)
            split4 = torch.split(item3, [250, 250], dim=1)
            split5 = torch.split(item4, [250, 250], dim=1)
            split6 = torch.split(item5, [250, 250], dim=1)
            getitem0, getitem1 = split1[0], split1[1]
            getitem2, getitem3 = split2[0], split2[1]
            getitem4, getitem5 = split3[0], split3[1]
            getitem6, getitem7 = split4[0], split4[1]
            getitem8, getitem9 = split5[0], split5[1]
            getitem10, getitem11 = split6[0], split6[1]
            getitem0_c = getitem0.clone()
            getitem1_c = getitem1.clone()
            getitem2_c = getitem2.clone()
            return torch.stack(
                (
                    getitem0,
                    getitem1,
                    getitem2,
                    getitem3,
                    getitem4,
                    getitem5,
                    getitem0_c,
                    getitem1_c,
                    getitem6,
                    getitem7,
                    getitem8,
                    getitem9,
                    getitem10,
                    getitem11,
                    getitem2_c,
                ),
                dim=1,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "split_stack_to_cats_pass": {},
            },
            post_grad_fusion_options={},
        )
        def split_stack_to_cats_different_dim(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable only the split_stack_to_cats pass.
            # Splits the input and a clone into five parts along dim 1 and stacks them with cloned
            # items along dim 2.
            l1_out = torch.split(x, [100, 100, 100, 100, 100], dim=1)
            x_c = x.clone()
            l2_out = torch.split(x_c, [100, 100, 100, 100, 100], dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item0_c = l2_out[0]
            item1_c = l2_out[1]
            item2_c = l2_out[2]
            item3_c = l2_out[3]
            item4_c = l2_out[4]
            other_1 = item0.clone()
            other_2 = item1.clone()
            other_3 = item2.clone()
            return torch.stack(
                (
                    other_1,
                    other_2,
                    other_3,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item0_c,
                    item1_c,
                    item2_c,
                    item3_c,
                    item4_c,
                ),
                dim=2,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "unbind_stack_to_slices_pass": {},
            },
            post_grad_fusion_options={},
        )
        def unbind_stack_to_slices(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable only the unbind_stack_to_slices
            # pass. Unbinds a 50 by 10 by 500 view along dim 1 and stacks the pieces with extra
            # cloned items along dim 1.
            x_1 = x.view(50, 10, 500)
            l1_out = torch.unbind(x_1, dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            item5 = l1_out[5]
            item6 = l1_out[6]
            item7 = l1_out[7]
            item8 = l1_out[8]
            item9 = l1_out[9]
            other_1 = item0.clone()
            other_2 = item1.clone()
            other_3 = item2.clone()
            return torch.stack(
                (
                    other_1,
                    other_2,
                    other_3,
                    item0,
                    item1,
                    item2,
                    item3,
                    item4,
                    item5,
                    item6,
                    item7,
                    item8,
                    item9,
                ),
                dim=1,
            )

        @torch._inductor.config.patch(
            pre_grad_fusion_options={
                "normalization_pass": {},
                "move_reshape_out_of_split_stack_pass": {},
            },
            post_grad_fusion_options={},
        )
        def move_reshape_out_of_split_stack(x):
            # [FUNCTION DOC] HELPER: Config-patched to enable the normalization pass and the
            # move_reshape_out_of_split_stack pass. Views the input as 50000 by 5, splits into
            # five columns, reshapes each, and stacks with cloned items along dim 0.
            x_c = x.view(50000, 5)
            l1_out = torch.split(x_c, [1, 1, 1, 1, 1], dim=1)
            item0 = l1_out[0]
            item1 = l1_out[1]
            item2 = l1_out[2]
            item3 = l1_out[3]
            item4 = l1_out[4]
            reshape0 = item0.reshape(-1, 5)
            reshape1 = item1.reshape(-1, 5)
            reshape2 = item2.reshape(-1, 5)
            reshape3 = item3.reshape(-1, 5)
            reshape4 = item4.reshape(-1, 5)
            other0 = reshape0.clone()
            other1 = reshape1.clone()
            other2 = reshape2.clone()
            other3 = reshape3.clone()
            return torch.stack(
                (
                    other0,
                    other1,
                    other2,
                    reshape0,
                    reshape1,
                    reshape2,
                    reshape3,
                    reshape4,
                    other3,
                ),
                dim=0,
            )

        args = [
            torch.randn(500, 500),
        ]
        for (
            fn,
            expected_getitem_cat_merged,
            expected_cat_removed,
            expected_split_cat_to_slices,
            exptected_unbind_to_cat_view,
            expected_split_stack_to_cats,
            exptected_unbind_stack_to_slices,
            expected_move_reshape_out_of_split_stack,
        ) in [
            (split_cat_split, 2, 0, 0, 0, 0, 0, 0),
            (split_cat_split_kwarg, 2, 0, 0, 0, 0, 0, 0),
            (remove_cat_node_with_all_getitmes, 0, 2, 0, 0, 0, 0, 0),
            (mutate_cat_node_with_some_getitmes, 0, 1, 0, 0, 0, 0, 0),
            (split_cat_to_slices, 0, 0, 1, 0, 0, 0, 0),
            (unbind_cat_to_view, 0, 0, 0, 1, 0, 0, 0),
            (split_stack_to_cats_same_dim, 0, 0, 0, 0, 1, 0, 0),
            (split_stack_to_cats_different_dim, 0, 0, 0, 0, 1, 0, 0),
            (unbind_stack_to_slices, 0, 0, 0, 0, 0, 1, 0),
            (move_reshape_out_of_split_stack, 0, 0, 0, 0, 0, 0, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["merge_getitem_cat_pass"],
                expected_getitem_cat_merged,
            )
            self.assertEqual(
                counters["inductor"]["mutate_cat_pass"],
                expected_cat_removed,
            )
            self.assertEqual(
                counters["inductor"]["split_cat_to_slices_pass"],
                expected_split_cat_to_slices,
            )
            self.assertEqual(
                counters["inductor"]["unbind_cat_to_view_pass"],
                exptected_unbind_to_cat_view,
            )
            self.assertEqual(
                counters["inductor"]["split_stack_to_cats_pass"],
                expected_split_stack_to_cats,
            )
            self.assertEqual(
                counters["inductor"]["unbind_stack_to_slices_pass"],
                exptected_unbind_stack_to_slices,
            )
            self.assertEqual(
                counters["inductor"]["move_reshape_out_of_split_stack_pass"],
                expected_move_reshape_out_of_split_stack,
            )
            counters.clear()

    def test_numpy_compat_normalization(self):
        # [FUNCTION DOC] Checks numpy_compat_normalization on a symbolically traced function using
        # numpy-style keyword names: after the pass runs on the graph, no node may still have any
        # of the keys x, x1, x2, a, axis or keepdims.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function using numpy-style keyword names (axis, x2, keepdims,
            # x) in stack, mul, max and dropout calls; returns six results. It is only traced, not
            # run.
            a = torch.stack([x, y], axis=1)
            b = torch.mul(x, x2=y)
            c = torch.mul(x, x2=y)
            d = torch.mul(x, x2=y)
            e = torch.max(x, dim=1, keepdims=True)
            f = torch.dropout(x=x, p=0.5, train=True)
            return a, b, c, d, e, f

        fn_t = torch.fx.symbolic_trace(fn)
        numpy_compat_normalization(fn_t.graph)

        for n in fn_t.graph.nodes:
            for k in n.kwargs:
                self.assertTrue(k not in {"x", "x1", "x2", "a", "axis", "keepdims"})

    @patch
    @requires_gpu
    def test_stack_normalization_axis_kwarg(self):
        # [FUNCTION DOC] GPU-only test: a stack with the numpy-style axis keyword is compiled with
        # the full set of passes and its result must equal eager on random 4 by 4 input on the GPU
        # device.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Stacks two tensors using the axis keyword.
            return torch.stack([x, y], axis=1)

        x, y = (torch.rand((4, 4), device=GPU_TYPE) for _ in range(2))
        expected = fn(x, y)
        actual = torch.compile(fn)(x, y)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
