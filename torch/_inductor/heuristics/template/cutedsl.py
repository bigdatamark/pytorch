# [MODULE DOC] Config definitions for the Blackwell CuTeDSL grouped GEMM kernel: tile/cluster/2-CTA/tensormap-update choices and
# functions returning the baseline, default or exhaustive config lists depending on inductor config. No import-time side effects beyond imports.
from dataclasses import dataclass
from enum import auto, Enum
from itertools import product

import torch._inductor.config as config


class TensorMapUpdateMode(Enum):
    # [CLASS DOC] Enum mirroring cutlass.utils.TensorMapUpdateMode (SMEM, GMEM) to avoid a cutlass dependency.
    """Enum mirroring cutlass.utils.TensorMapUpdateMode to decouple this file from a cutlass dependency."""

    SMEM = auto()
    GMEM = auto()


@dataclass(frozen=True)
class CuTeGemmConfig:
    # [CLASS DOC] Frozen dataclass describing one grouped GEMM config: TILE_M, TILE_N, CLUSTER_M, CLUSTER_N, USE_2_CTA and TENSORMAP_UPDATE_MODE, with defaults 128, 192, 2, 1, False, SMEM.
    TILE_M: int = 128
    TILE_N: int = 192
    CLUSTER_M: int = 2
    CLUSTER_N: int = 1
    USE_2_CTA: bool = False
    TENSORMAP_UPDATE_MODE: TensorMapUpdateMode = TensorMapUpdateMode.SMEM


def get_exhaustive_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the exhaustive configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    For information regarding valid config sets, see:
    # [FUNCTION DOC] Return every combination of tensormap mode, tile m, tile n and valid cluster shape, for non-2CTA (tile m 64/128) and 2CTA (tile m 128/256) cases. Pure; returns a list.
    https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/cute/blackwell/kernel/grouped_gemm/grouped_gemm.py
    """

    # Tile_n is always the same regardless of 2cta
    tile_n_vals = [32, 64, 96, 128, 160, 192, 224, 256]

    # Valid clusters
    clusters_no_2cta = [
        (1, 1),
        (1, 2),
        (1, 4),
        (1, 8),
        (1, 16),
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]
    clusters_2cta = [
        (2, 1),
        (2, 2),
        (2, 4),
        (2, 8),
        (4, 1),
        (4, 2),
        (4, 4),
        (8, 1),
        (8, 2),
        (16, 1),
    ]

    configs: list[CuTeGemmConfig] = []

    for use_2cta, cluster_set, tile_m_range in [
        (False, clusters_no_2cta, [64, 128]),
        (True, clusters_2cta, [128, 256]),
    ]:
        for tensormap_update_mode, tile_m, tile_n, (cluster_m, cluster_n) in product(
            [TensorMapUpdateMode.SMEM, TensorMapUpdateMode.GMEM],
            tile_m_range,
            tile_n_vals,
            cluster_set,
        ):
            configs.append(
                CuTeGemmConfig(
                    tile_m,
                    tile_n,
                    cluster_m,
                    cluster_n,
                    USE_2_CTA=use_2cta,
                    TENSORMAP_UPDATE_MODE=tensormap_update_mode,
                )
            )

    return configs


def get_default_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the default configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.
    # [FUNCTION DOC] Return the fixed list of 22 hand-picked default configs. Pure; returns a list.
    """

    config_tuples = [
        (128, 256, 2, 1, False, TensorMapUpdateMode.SMEM),
        (256, 160, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.GMEM),
        (128, 256, 1, 2, False, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 1, False, TensorMapUpdateMode.SMEM),
        (256, 256, 2, 1, True, TensorMapUpdateMode.SMEM),
        (128, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
        (256, 256, 8, 1, True, TensorMapUpdateMode.GMEM),
        (64, 32, 1, 2, False, TensorMapUpdateMode.SMEM),
        (256, 192, 2, 1, True, TensorMapUpdateMode.GMEM),
        (256, 256, 2, 2, True, TensorMapUpdateMode.SMEM),
        (128, 96, 1, 2, False, TensorMapUpdateMode.SMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.SMEM),
        (64, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 192, 1, 1, False, TensorMapUpdateMode.GMEM),
        (128, 64, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 160, 1, 1, False, TensorMapUpdateMode.GMEM),
        (64, 256, 1, 1, False, TensorMapUpdateMode.GMEM),
    ]

    return [CuTeGemmConfig(*args) for args in config_tuples]


def get_groupgemm_configs() -> list[CuTeGemmConfig]:
    """
    Returns the configuration set for the Blackwell CuTeDSL Grouped GEMM kernel.

    Note: CuTeDSL autotuning is still experimental — enabling it may trigger kernel launch failures
    or unstable results. By default, autotuning is disabled and we return only
    a single baseline config.
    # [FUNCTION DOC] Return configs by inductor config: exhaustive list if cutedsl_enable_autotuning is on and the gemm search space is EXHAUSTIVE,
    # default list if only autotuning is on, else a single baseline config (first default). Reads config only.
    """
    if (
        config.cutedsl_enable_autotuning
        and config.max_autotune_gemm_search_space == "EXHAUSTIVE"
    ):
        return get_exhaustive_groupgemm_configs()
    elif config.cutedsl_enable_autotuning:
        return get_default_groupgemm_configs()
    else:
        return [get_default_groupgemm_configs()[0]]
