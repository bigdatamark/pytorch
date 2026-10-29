# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor coordinate descent autotuner (CoordescTuner) that tunes
# Triton kernel configs one field at a time. Covers neighbour generation, block-size limits,
# native-matmul tensor-size limits and combo-kernel per-subkernel fields. Import-time side
# effects: skips the module if triton is missing, turns on benchmark_kernel and
# coordinate_descent_tuning in the Inductor config, and saves the original compare_config. Tests
# only run via the main entry point when on Linux with a GPU.

import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import (
    native_matmul_block_numel,
    native_matmul_persistent_rblock,
    TRITON_MAX_BLOCK,
    TRITON_MAX_TENSOR_NUMEL,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner


config.benchmark_kernel = True
config.coordinate_descent_tuning = True

orig_compare_config = CoordescTuner.compare_config


def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    # [FUNCTION DOC] HELPER: replacement for CoordescTuner.compare_config that biases search
    # toward larger XBLOCK. If the candidate has an XBLOCK, the best config must also have one
    # (otherwise an AssertionError is raised). A smaller candidate is run once (so a launcher is
    # created) and reported as worse (False, timing scaled by 1.1); a larger one is run and
    # reported as better (True, timing scaled by 0.9). Equal or absent XBLOCK defers to the
    # original method.
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        if "XBLOCK" not in best_config.kwargs:
            raise AssertionError
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # run func so the launcher will be created
            return False, best_timing * 1.1
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    # [CLASS DOC] Unit tests for CoordescTuner: autotuning on synthetic cost functions, neighbour
    # value generation, min/max value limits, native-matmul validity rules and combo-kernel
    # metadata.
    def test_abs_function(self):
        # [FUNCTION DOC] Autotunes a synthetic cost (distance of XBLOCK from 15) starting at
        # XBLOCK 1 and checks the tuner converges to XBLOCK 16, the nearest allowed power of two.
        """
        The benchmark result is simply abs(XBLOCK - 15)
        """
        tuner = CoordescTuner()
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            # [FUNCTION DOC] HELPER: synthetic benchmark; returns the absolute distance of the
            # config XBLOCK from 15.
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_no_neighbors(self):
        # [FUNCTION DOC] With a size hint of 1 on x, no larger XBLOCK is allowed, so the tuner has
        # no neighbours to try and must return the baseline config with XBLOCK still 1.
        """
        Test the case that there is no available neighbor values for a field.
        """

        # size hint for x being 1 limits the max XBLOCK we try to be 1
        tuner = CoordescTuner(size_hints={"x": 1})
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            # [FUNCTION DOC] HELPER: synthetic benchmark; returns the absolute distance of the
            # config XBLOCK from 15.
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        # [FUNCTION DOC] Checks get_neighbour_values with radius 2 returns the expected power-of-
        # two style neighbours for num_stages (1, 3, 4 around 2) and num_warps (1, 4, 8 around 2).
        tuner = CoordescTuner()

        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        # [FUNCTION DOC] End-to-end check: with compare_config patched to prefer larger XBLOCK,
        # compiles a normalization-style function on the GPU twice and verifies the result matches
        # eager within 1e-4 tolerances. Requires a GPU device; the first compile only warms the
        # cache.
        def f(x):
            # [FUNCTION DOC] HELPER: function under test; divides the input by its sum along the
            # last dimension.
            return x / x.sum(dim=-1, keepdim=True)

        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            x = torch.ones(2, 256).to(GPU_TYPE)
            expected = f(x)
            # the first call get correct result when cache miss. Don't know why yet
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                lambda msg: f"{msg}\nExpected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # [FUNCTION DOC] Checks value_too_large treats XBLOCK and R0_BLOCK values equal to the
        # Triton maximum block size as allowed and anything double that as too large, for large
        # size hints.
        # Simulate a reduction
        size_hints = {"x": 2**20, "y": 2**20}

        tuner = CoordescTuner(size_hints=size_hints)

        max_block = TRITON_MAX_BLOCK
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        self.assertFalse(tuner.value_too_large("R0_BLOCK", max_block["R0_"]))
        self.assertTrue(tuner.value_too_large("R0_BLOCK", max_block["R0_"] * 2))

    def test_native_matmul_block_numel_limit(self):
        # [FUNCTION DOC] For a native-matmul tuner, checks is_valid_config accepts a config whose
        # block product is within the tensor-numel limit and rejects three configs exceeding it; a
        # non-native-matmul tuner still accepts the first oversized config.
        tuner = CoordescTuner(is_native_matmul=True)

        valid_config = triton.Config(
            {"YBLOCK": 128, "XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=8,
            num_stages=1,
        )
        self.assertTrue(tuner.is_valid_config(valid_config))

        invalid_configs = [
            {"YBLOCK": 128, "XBLOCK": 256, "R0_BLOCK": 64},
            {"YBLOCK": 256, "XBLOCK": 128, "R0_BLOCK": 64},
            {"YBLOCK": 128, "XBLOCK": 128, "R0_BLOCK": 128},
        ]
        for kwargs in invalid_configs:
            self.assertFalse(
                tuner.is_valid_config(triton.Config(kwargs, num_warps=8, num_stages=1))
            )

        regular_tuner = CoordescTuner(is_native_matmul=False)
        self.assertTrue(
            regular_tuner.is_valid_config(
                triton.Config(invalid_configs[0], num_warps=8, num_stages=1)
            )
        )

    def test_native_matmul_rejects_invalid_neighbours(self):
        # [FUNCTION DOC] Checks every neighbour config produced for XBLOCK, YBLOCK and R0_BLOCK by
        # a native-matmul tuner keeps native_matmul_block_numel at or below
        # TRITON_MAX_TENSOR_NUMEL.
        tuner = CoordescTuner(
            is_native_matmul=True,
            size_hints={"x": 16384, "y": 2048, "r0_": 2048},
        )
        baseline = triton.Config(
            {"YBLOCK": 128, "XBLOCK": 128, "R0_BLOCK": 64},
            num_warps=8,
            num_stages=1,
        )

        for field in ("XBLOCK", "YBLOCK", "R0_BLOCK"):
            for cfg in tuner.get_neighbour_configs(baseline, field):
                self.assertLessEqual(
                    native_matmul_block_numel(cfg.kwargs),
                    TRITON_MAX_TENSOR_NUMEL,
                )

    def test_native_matmul_persistent_uses_rblock_hint_for_limit(self):
        # [FUNCTION DOC] For a persistent native-matmul kernel the reduction block derives from
        # the r0_ size hint, so a config with XBLOCK 16 is valid while XBLOCK 32 is invalid, and
        # XBLOCK 32 must not appear among the generated XBLOCK neighbours.
        tuner = CoordescTuner(
            is_native_matmul=True,
            size_hints={"x": 16384, "y": 2048, "r0_": 2048},
        )

        valid_config = triton.Config(
            {"YBLOCK": 32, "XBLOCK": 16},
            num_warps=8,
            num_stages=1,
        )
        self.assertTrue(tuner.is_valid_config(valid_config))

        invalid_config = triton.Config(
            {"YBLOCK": 32, "XBLOCK": 32},
            num_warps=8,
            num_stages=1,
        )
        self.assertFalse(tuner.is_valid_config(invalid_config))

        neighbours = tuner.get_neighbour_configs(valid_config, "XBLOCK")
        self.assertNotIn(32, [cfg.kwargs["XBLOCK"] for cfg in neighbours])

    def test_native_matmul_persistent_uses_min_rblock_for_limit(self):
        # [FUNCTION DOC] With an r0_ size hint of 1 the persistent reduction block is minimal; the
        # 256x256 config exactly reaches the numel limit and is valid, XBLOCK 512 is invalid and
        # is never offered as an XBLOCK neighbour.
        size_hints = {"x": 4096, "y": 4096, "r0_": 1}
        tuner = CoordescTuner(
            is_native_matmul=True,
            size_hints=size_hints,
        )
        r0_block = native_matmul_persistent_rblock(size_hints["r0_"])

        baseline = triton.Config(
            {"YBLOCK": 256, "XBLOCK": 256},
            num_warps=8,
            num_stages=1,
        )
        self.assertEqual(
            native_matmul_block_numel(baseline.kwargs, r0_block=r0_block),
            TRITON_MAX_TENSOR_NUMEL,
        )
        self.assertTrue(tuner.is_valid_config(baseline))

        invalid_config = triton.Config(
            {"YBLOCK": 256, "XBLOCK": 512},
            num_warps=8,
            num_stages=1,
        )
        self.assertFalse(tuner.is_valid_config(invalid_config))

        neighbours = tuner.get_neighbour_configs(baseline, "XBLOCK")
        self.assertNotIn(512, [cfg.kwargs["XBLOCK"] for cfg in neighbours])

    def test_native_matmul_persistent_uses_meta_rblock_for_limit(self):
        # [FUNCTION DOC] When inductor_meta supplies native_matmul_persistent_rblock (1024) it
        # overrides the value derived from the size hint: a config valid under the hint-derived
        # block becomes invalid under the metadata one, is rejected by is_valid_config and
        # excluded from YBLOCK neighbours.
        size_hints = {"x": 4096, "y": 4096, "r0_": 64}
        effective_rblock = 1024
        tuner = CoordescTuner(
            is_native_matmul=True,
            size_hints=size_hints,
            inductor_meta={"native_matmul_persistent_rblock": effective_rblock},
        )

        baseline = triton.Config(
            {"YBLOCK": 32, "XBLOCK": 32},
            num_warps=8,
            num_stages=1,
        )
        self.assertEqual(
            native_matmul_block_numel(baseline.kwargs, r0_block=effective_rblock),
            TRITON_MAX_TENSOR_NUMEL,
        )
        self.assertTrue(tuner.is_valid_config(baseline))

        invalid_config = triton.Config(
            {"YBLOCK": 64, "XBLOCK": 32},
            num_warps=8,
            num_stages=1,
        )
        self.assertLessEqual(
            native_matmul_block_numel(
                invalid_config.kwargs,
                r0_block=native_matmul_persistent_rblock(size_hints["r0_"]),
            ),
            TRITON_MAX_TENSOR_NUMEL,
        )
        self.assertGreater(
            native_matmul_block_numel(invalid_config.kwargs, r0_block=effective_rblock),
            TRITON_MAX_TENSOR_NUMEL,
        )
        self.assertFalse(tuner.is_valid_config(invalid_config))

        neighbours = tuner.get_neighbour_configs(baseline, "YBLOCK")
        self.assertNotIn(64, [cfg.kwargs["YBLOCK"] for cfg in neighbours])

    def test_combo_tunable_fields_flow_into_tunable_fields(self):
        # [FUNCTION DOC] Checks that combo_coordesc_field_order from inductor_meta is placed, in
        # order, in front of the base tunable fields (which still include XBLOCK) as soon as the
        # tuner is built.
        """``inductor_meta["combo_coordesc_field_order"]`` must be
        prepended to ``tunable_fields`` so combo-kernel per-subkernel
        block fields are visible from the moment the tuner is
        constructed (not only after ``autotune`` is called)."""
        tuner = CoordescTuner(
            inductor_meta={
                "combo_coordesc_field_order": [
                    "XBLOCK_0",
                    "YBLOCK_0",
                    "XBLOCK_1",
                ],
            }
        )
        fields = tuner.tunable_fields
        # Combo fields appear in order, ahead of the base fields.
        self.assertEqual(fields[:3], ["XBLOCK_0", "YBLOCK_0", "XBLOCK_1"])
        self.assertIn("XBLOCK", fields)

    def test_value_too_large_combo_field_limits(self):
        # [FUNCTION DOC] Checks per-field limits from combo_coordesc_field_limits: a value equal
        # to a field limit is accepted and a larger value is reported as too large by
        # value_too_large.
        tuner = CoordescTuner(
            size_hints={"x": 2**20, "r0_": 2**20},
            inductor_meta={
                "combo_coordesc_field_limits": {
                    "XBLOCK_0": 64,
                    "XBLOCK_1": 256,
                    "R0_BLOCK_1": 128,
                }
            },
        )

        self.assertFalse(tuner.value_too_large("XBLOCK_0", 64))
        self.assertTrue(tuner.value_too_large("XBLOCK_0", 128))
        self.assertFalse(tuner.value_too_large("XBLOCK_1", 256))
        self.assertTrue(tuner.value_too_large("XBLOCK_1", 512))
        self.assertFalse(tuner.value_too_large("R0_BLOCK_1", 128))
        self.assertTrue(tuner.value_too_large("R0_BLOCK_1", 256))

    def test_value_limits_from_inductor_meta(self):
        # [FUNCTION DOC] Checks min_xblock and min_rblock in inductor_meta make value_too_small
        # reject smaller XBLOCK and R0_BLOCK values, and that get_neighbour_values at the minimum
        # only offers the next larger value.
        tuner = CoordescTuner(
            size_hints={"x": 1024, "r0_": 1024},
            inductor_meta={
                "min_xblock": 16,
                "min_rblock": 64,
            },
        )

        self.assertTrue(tuner.value_too_small("XBLOCK", 8))
        self.assertFalse(tuner.value_too_small("XBLOCK", 16))
        self.assertTrue(tuner.value_too_small("R0_BLOCK", 32))
        self.assertFalse(tuner.value_too_small("R0_BLOCK", 64))
        self.assertEqual(tuner.get_neighbour_values("XBLOCK", 16), [32])
        self.assertEqual(tuner.get_neighbour_values("R0_BLOCK", 64), [128])

    def test_combo_metadata_orders_larger_subkernels_first_for_coordesc(self):
        # [FUNCTION DOC] Builds combo metadata for three pointwise subkernels with a mocked
        # pointwise heuristic and calls the combo per-subkernel-block handler. Verifies it returns
        # configs and fills inductor_meta with a field order that puts the largest subkernel first
        # and with per-field limits, then that a new CoordescTuner exposes the same order at the
        # front of tunable_fields.
        def make_configs(xblock, yblock):
            # [FUNCTION DOC] HELPER: returns a one-element list holding a Triton config with the
            # given XBLOCK and YBLOCK, 4 warps and 1 stage.
            return [
                triton.Config(
                    {"XBLOCK": xblock, "YBLOCK": yblock},
                    num_warps=4,
                    num_stages=1,
                )
            ]

        inductor_meta = {
            "combo_grid_meta": {
                "num_kernels": 3,
                "heuristic_0": "pointwise",
                "heuristic_1": "pointwise",
                "heuristic_2": "pointwise",
                "size_hints_0": {"x": 64, "y": 64},
                "size_hints_1": {"x": 256, "y": 256},
                "size_hints_2": {"x": 128, "y": 16},
                "tile_hint_0": "TileHint.SQUARE",
                "tile_hint_1": "TileHint.SQUARE",
                "tile_hint_2": "TileHint.SQUARE",
                "no_x_dim_0": False,
                "no_x_dim_1": False,
                "no_x_dim_2": False,
            }
        }

        configs_by_size = {
            (64, 64): make_configs(64, 32),
            (256, 256): make_configs(256, 64),
            (128, 16): make_configs(128, 16),
        }

        def pointwise_side_effect(size_hints, *args, **kwargs):
            # [FUNCTION DOC] HELPER: mock for the pointwise heuristic; picks the prepared config
            # list by the x and y size hints and ignores the other arguments (KeyError for an
            # unknown size pair).
            return configs_by_size[(size_hints["x"], size_hints["y"])]

        with mock.patch.object(
            triton_heuristics,
            "pointwise",
            side_effect=pointwise_side_effect,
        ):
            configs = triton_heuristics._handle_combo_kernel_per_subkernel_blocks(
                {"x": 256, "y": 256},
                inductor_meta,
                triton_meta={},
            )

        self.assertIsNotNone(configs)
        self.assertEqual(
            inductor_meta["combo_coordesc_field_order"],
            [
                "XBLOCK_1",
                "YBLOCK_1",
                "XBLOCK_0",
                "YBLOCK_0",
                "XBLOCK_2",
                "YBLOCK_2",
            ],
        )
        self.assertEqual(
            inductor_meta["combo_coordesc_field_limits"],
            {
                "XBLOCK_0": 64,
                "YBLOCK_0": 64,
                "XBLOCK_1": 256,
                "YBLOCK_1": 256,
                "XBLOCK_2": 128,
                "YBLOCK_2": 16,
            },
        )

        # End-to-end: a CoordescTuner constructed with the populated
        # ``inductor_meta`` must surface the combo fields via
        # ``tunable_fields`` in the same order, ahead of the base fields.
        tuner = CoordescTuner(inductor_meta=inductor_meta)
        self.assertEqual(
            tuner.tunable_fields[: len(inductor_meta["combo_coordesc_field_order"])],
            inductor_meta["combo_coordesc_field_order"],
        )


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
