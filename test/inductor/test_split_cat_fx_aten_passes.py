# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor post-grad aten-level split/cat, select/cat and move-
# view-after-cat graph passes, plus the aten split normalization pass. Defines small modules
# that build aten graphs exercising each pattern; tests compile them with specific post-grad
# fusion options and check the compiled output equals eager and that the Inductor pass
# counters hit the expected counts. At import time it tries to import an internal fbgemm
# inductor lowerings module and records success in the has_fbgemm flag (false on any
# exception); this flag is not used in this file.

import torch
import torch._inductor
from torch._dynamo.utils import counters
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


try:
    # importing this will register fbgemm lowerings for inductor
    import deeplearning.fbgemm.fbgemm_gpu.fb.inductor_lowerings  # noqa: F401

    has_fbgemm = True
except Exception:
    has_fbgemm = False


class _TestSplitCat(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model that concatenates two inputs along dim 1, splits the
    # result into chunks of 32 along dim 1, concatenates all eight chunks back together,
    # concatenates the first chunk with a third input, and returns the concatenation of those
    # two results. Expects inputs of widths 128, 128 and 32 so the split yields eight chunks.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        # [FUNCTION DOC] HELPER: Builds the aten graph described by the class: cat of x and y,
        # split into 32-wide chunks, cat of all chunks, cat of first chunk with z, and a final
        # cat of the two cat results. Returns the final tensor.
        cat = torch.ops.aten.cat.default([x, y], 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1)
        getitem = split[0]
        getitem_1 = split[1]
        getitem_2 = split[2]
        getitem_3 = split[3]
        getitem_4 = split[4]
        getitem_5 = split[5]
        getitem_6 = split[6]
        getitem_7 = split[7]
        cat_1 = torch.ops.aten.cat.default(
            [
                getitem,
                getitem_1,
                getitem_2,
                getitem_3,
                getitem_4,
                getitem_5,
                getitem_6,
                getitem_7,
            ],
            1,
        )
        cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        return torch.ops.aten.cat.default([cat_1, cat_2], 1)


class _TestSplitCatSingular(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model like the split-cat model but where only one chunk of the
    # split is concatenated (a single-element cat), so the split/cat pass should not fire.
    # Takes three inputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
        # [FUNCTION DOC] HELPER: Concatenates x and y along dim 1, splits into 32-wide chunks,
        # takes only the first chunk, builds a single-element cat of it, a cat of it with z,
        # and returns the cat of those two results.
        cat = torch.ops.aten.cat.default([x, y], 1)
        split = torch.ops.aten.split.Tensor(cat, 32, 1)
        getitem = split[0]
        cat_1 = torch.ops.aten.cat.default(
            [getitem],
            1,
        )
        cat_2 = torch.ops.aten.cat.default([getitem, z], 1)
        return torch.ops.aten.cat.default([cat_1, cat_2], 1)


class _TestSplitCatPartial(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model that splits one input into 21 pieces of width 96 and
    # another into 4 pieces of width 96, then concatenates a non-contiguous mix of those
    # pieces interleaved with two extra inputs (the last input appears at both ends and the
    # third input twice in the middle), skipping some pieces of the first split. Used to test
    # partial split-cat matching.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
        super().__init__()

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, z: torch.Tensor
    ):
        # [FUNCTION DOC] HELPER: Splits x1 into 21 pieces and x2 into 4 pieces (width 96 each
        # along dim 1), picks pieces 0-5 and 10-17 of the first split plus all four pieces of
        # the second, concatenates them in a shuffled order with y and z along dim 1, and
        # returns the result.
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
            x1,
            [
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
                96,
            ],
            1,
        )
        split_with_sizes_2 = torch.ops.aten.split_with_sizes.default(
            x2, [96, 96, 96, 96], 1
        )
        getitem_71 = split_with_sizes_1[0]
        getitem_72 = split_with_sizes_1[1]
        getitem_73 = split_with_sizes_1[2]
        getitem_74 = split_with_sizes_1[3]
        getitem_75 = split_with_sizes_1[4]
        getitem_76 = split_with_sizes_1[5]
        getitem_77 = split_with_sizes_1[10]
        getitem_78 = split_with_sizes_1[11]
        getitem_79 = split_with_sizes_1[12]
        getitem_80 = split_with_sizes_1[13]
        getitem_81 = split_with_sizes_1[14]
        getitem_82 = split_with_sizes_1[15]
        getitem_83 = split_with_sizes_1[16]
        getitem_84 = split_with_sizes_1[17]
        getitem_85 = split_with_sizes_2[0]
        getitem_86 = split_with_sizes_2[1]
        getitem_87 = split_with_sizes_2[2]
        getitem_88 = split_with_sizes_2[3]

        cat = torch.ops.aten.cat.default(
            [
                z,
                getitem_71,
                getitem_72,
                getitem_73,
                getitem_74,
                getitem_75,
                getitem_76,
                getitem_82,
                getitem_83,
                getitem_84,
                y,
                getitem_77,
                getitem_78,
                getitem_79,
                getitem_80,
                getitem_81,
                y,
                getitem_85,
                getitem_86,
                getitem_87,
                getitem_88,
                z,
            ],
            1,
        )
        return cat


class _TestMoveViewAferCat(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model for the move-view-after-cat pass (class name keeps a
    # misspelling of After). Splits an input into seven size-one pieces along the first dim,
    # views each to an 8 by 96 shape, concatenates the views along dim 1, and combines with a
    # clone of the first view.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
        super().__init__()

    def forward(self, x: torch.Tensor):
        # [FUNCTION DOC] HELPER: Builds the graph: split into seven pieces, a view of each, a
        # clone of the first view, a cat of all views along dim 1, a cat of the clone with
        # that result, and returns a final cat of the clone with the previous cat.
        split_with_sizes_1 = torch.ops.aten.split_with_sizes.default(
            x, [1, 1, 1, 1, 1, 1, 1]
        )
        getitem_71 = split_with_sizes_1[0]
        getitem_72 = split_with_sizes_1[1]
        getitem_73 = split_with_sizes_1[2]
        getitem_74 = split_with_sizes_1[3]
        getitem_75 = split_with_sizes_1[4]
        getitem_76 = split_with_sizes_1[5]
        getitem_77 = split_with_sizes_1[6]
        view_1 = torch.ops.aten.view.default(getitem_71, [8, 96])
        view_2 = torch.ops.aten.view.default(getitem_72, [8, 96])
        view_3 = torch.ops.aten.view.default(getitem_73, [8, 96])
        view_4 = torch.ops.aten.view.default(getitem_74, [8, 96])
        view_5 = torch.ops.aten.view.default(getitem_75, [8, 96])
        view_6 = torch.ops.aten.view.default(getitem_76, [8, 96])
        view_7 = torch.ops.aten.view.default(getitem_77, [8, 96])
        clone = torch.ops.aten.clone.default(view_1)

        cat = torch.ops.aten.cat.default(
            [
                view_1,
                view_2,
                view_3,
                view_4,
                view_5,
                view_6,
                view_7,
            ],
            1,
        )
        cat_1 = torch.ops.aten.cat.default([clone, cat], 1)
        return torch.cat([clone, cat_1], 1)


class _TestSelectCat(torch.nn.Module):
    # [CLASS DOC] HELPER: Test model for the select-cat pass: selects several index columns
    # (dim 1) of one input and of another input and concatenates different subsets of them.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Calls the base module constructor; no state.
        super().__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        # [FUNCTION DOC] HELPER: Selects indices 0-5 along dim 1 of x and 0-4 of y, and
        # returns a tuple of four cats: all six selects of x, the first five of x, indices 0,
        # 2, 4 of x, and the five selects of y.
        select = torch.ops.aten.select.int(x, 1, 0)
        select_1 = torch.ops.aten.select.int(x, 1, 1)
        select_2 = torch.ops.aten.select.int(x, 1, 2)
        select_3 = torch.ops.aten.select.int(x, 1, 3)
        select_4 = torch.ops.aten.select.int(x, 1, 4)
        select_5 = torch.ops.aten.select.int(x, 1, 5)
        cat = torch.ops.aten.cat.default(
            [select, select_1, select_2, select_3, select_4, select_5], 1
        )
        cat1 = torch.ops.aten.cat.default(
            [select, select_1, select_2, select_3, select_4], 1
        )
        cat2 = torch.ops.aten.cat.default([select, select_2, select_4], 1)
        select_6 = torch.ops.aten.select.int(y, 1, 0)
        select_7 = torch.ops.aten.select.int(y, 1, 1)
        select_8 = torch.ops.aten.select.int(y, 1, 2)
        select_9 = torch.ops.aten.select.int(y, 1, 3)
        select_10 = torch.ops.aten.select.int(y, 1, 4)
        cat3 = torch.ops.aten.cat.default(
            [select_6, select_7, select_8, select_9, select_10], 1
        )
        return cat, cat1, cat2, cat3


class TestSplitCatAten(TestCase):
    # [CLASS DOC] Test case for the post-grad aten passes on GPU with triton. Holds comparison
    # helpers for dictionaries of tensors, predictions, parameters and gradients, and tests
    # that compile the helper modules with the relevant pass options and check counters.
    def compare_dict_tensors(self, ref_dict, res_dict, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Compares two mappings of tensors whose keys correspond with
        # the reference key prefixed by the compile wrapper prefix on parameter names. Returns
        # false if the key counts differ or any pair is not close within the tolerances;
        # raises AssertionError if a prefixed key is missing from the second mapping;
        # otherwise returns true.
        if len(set(ref_dict.keys())) != len(set(res_dict.keys())):
            return False
        for key1 in ref_dict:
            key2 = "_orig_mod." + key1
            if key2 not in res_dict:
                raise AssertionError(f"{key1} does not exist in traced module")
            if not torch.allclose(ref_dict[key1], res_dict[key2], rtol=rtol, atol=atol):
                return False
        return True

    def compare_pred(self, module, traced, input, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Runs the reference module and the compiled module on the
        # inputs and asserts the outputs are equal within the tolerances.
        ref = module(*input)
        res = traced(*input)
        self.assertEqual(ref, res, rtol=rtol, atol=atol)

    def compare_parameters(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Collects named parameters of both modules and asserts they
        # match through the dict comparison helper within the tolerances.
        ref_params = dict(module.named_parameters())
        res_params = dict(traced.named_parameters())
        self.assertTrue(self.compare_dict_tensors(ref_params, res_params, rtol, atol))

    def compare_gradients(self, module, traced, rtol=1e-3, atol=1e-3):
        # [FUNCTION DOC] HELPER: Collects the gradient of each named parameter of both modules
        # and asserts they match through the dict comparison helper within the tolerances. Not
        # called by tests in this file.
        ref_grad = {key: param.grad for key, param in module.named_parameters()}
        res_grad = {key: param.grad for key, param in traced.named_parameters()}
        self.assertTrue(
            self.compare_dict_tensors(ref_grad, res_grad, rtol=rtol, atol=atol)
        )

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {"threshold_to_cat": 5},
        },
    )
    def test_split_cat_post_grad(self):
        # [FUNCTION DOC] Compiles the split-cat model with normalization and split-cat aten
        # passes enabled (cat threshold 5) and checks outputs against eager, normalization
        # count 5 and split-cat count 1. Then repeats with the partial split-cat model,
        # expecting normalization count 3 and split-cat count 1. Clears pass counters before,
        # between and after. Requires GPU and triton.
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 32, device=torch.device(device=GPU_TYPE)),
        ]
        module = _TestSplitCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 5)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

        inputs = [
            torch.randn(1024, 96 * 21, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96 * 4, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 96, device=torch.device(device=GPU_TYPE)),
        ]
        module = _TestSplitCatPartial()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 3)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "split_cat_aten_pass": {"threshold_to_cat": 5},
        },
    )
    def test_split_cat_post_grad_singular(self):
        # [FUNCTION DOC] Compiles the singular split-cat model with the same pass options,
        # checks it matches eager, normalization count 4 and split-cat count 0 (the pass must
        # not fire). Clears pass counters before and after. Requires GPU and triton.
        counters.clear()
        inputs = [
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 32, device=torch.device(device=GPU_TYPE)),
        ]
        module = _TestSplitCatSingular()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["split_cat_aten_pass"], 0)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "select_cat_aten_pass": {},
        },
    )
    def test_select_cat_post_grad(self):
        # [FUNCTION DOC] Compiles the select-cat model with normalization and select-cat aten
        # passes enabled, checks it matches eager, normalization count 4 and select-cat count
        # 1. Clears pass counters before and after. Requires GPU and triton.
        counters.clear()
        inputs = [
            torch.randn(1024, 6, 128, device=torch.device(device=GPU_TYPE)),
            torch.randn(1024, 6, 128, device=torch.device(device=GPU_TYPE)),
        ]
        module = _TestSelectCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["select_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()

    @requires_gpu_and_triton
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
            "move_view_after_cat_aten_pass": {},
        },
    )
    def test_move_view_after_cat_aten(self):
        # [FUNCTION DOC] Compiles the move-view-after-cat model with normalization and move-
        # view-after-cat aten passes enabled, checks it matches eager, normalization count 4
        # and pass count 1. Clears pass counters before and after. Requires GPU and triton.
        counters.clear()
        inputs = [
            torch.randn(7, 8, 96, device=torch.device(device=GPU_TYPE)),
        ]
        module = _TestMoveViewAferCat()
        traced = torch.compile(module)
        ref = module(*inputs)
        res = traced(*inputs)
        self.compare_pred(module, traced, inputs)
        self.assertEqual(counters["inductor"]["normalization_aten_pass"], 4)
        self.assertEqual(counters["inductor"]["move_view_after_cat_aten_pass"], 1)
        self.assertEqual(ref, res, rtol=1e-8, atol=1e-8)
        self.compare_parameters(module, traced, rtol=1e-8, atol=1e-8)
        counters.clear()


class TestSplitCatAtenNormalizationPasses(TestCase):
    # [CLASS DOC] Tests for the aten normalization pass, which rewrites split calls into a
    # canonical argument form.
    @torch._inductor.config.patch(
        pre_grad_fusion_options={},
        post_grad_fusion_options={
            "normalization_aten_pass": {},
        },
    )
    def test_split_aten_normalization(self):
        # [FUNCTION DOC] Compiles two functions that call aten split with sizes 300 and 320 on
        # a 4096 by 300 CPU input with only the normalization pass enabled. For each, checks
        # compiled output equals eager and the normalization counter is 1, then clears
        # counters. The failure message includes the function under test.
        def arg_only_size_same(x):
            # [FUNCTION DOC] HELPER: Splits the input along dim 1 with chunk size 300 through
            # the aten split op.
            return torch.ops.aten.split.Tensor(x, 300, 1)

        def arg_only_size_different(x):
            # [FUNCTION DOC] HELPER: Splits the input along dim 1 with chunk size 320 through
            # the aten split op.
            return torch.ops.aten.split.Tensor(x, 320, 1)

        args = [
            torch.randn(4096, 300),
        ]
        for fn, expected_split_norm_count in [
            (arg_only_size_same, 1),
            (arg_only_size_different, 1),
        ]:
            expected = fn(*args)
            actual = torch.compile(fn)(*args)

            torch.testing.assert_close(actual, expected)
            self.assertEqual(
                counters["inductor"]["normalization_aten_pass"],
                expected_split_norm_count,
                msg=lambda msg: f"{msg}\nfor {fn}",
            )
            counters.clear()


if __name__ == "__main__":
    run_tests()
