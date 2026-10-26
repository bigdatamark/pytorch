# [MODULE DOC] Autotuning config heuristics for Triton pointwise kernels: a default (CUDA) class plus ROCm/HIP and XPU overrides.
# Each class is registered in the codegen heuristic registry under the name pointwise (device-specific ones only when the backend is available).
# Import-time side effect: registers the heuristics; queries torch.version.hip and torch.xpu.
"""Codegen heuristics for Triton pointwise kernel autotuning.

Contains the default (CUDA) heuristic and device-specific overrides for
ROCm/HIP and XPU.  Subclasses override only the dimension methods that
differ from the base.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch
from torch._inductor.heuristics.registry import (
    CodegenConfigHeuristics,
    register_codegen_heuristic,
)
from torch._inductor.runtime.hints import TileHint, TRITON_MAX_BLOCK


if TYPE_CHECKING:
    from collections.abc import Callable


# ----------------------------------------------------------------------
# Default (CUDA) pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise")
class PointwiseHeuristic(CodegenConfigHeuristics):
    # [CLASS DOC] Default pointwise heuristic. Dispatches on the number of dimensions in size_hints (1, 2 or 3) to per-dimension config builders.
    # Subclasses override only the per-dimension methods that differ.
    """Default pointwise autotuning configs (CUDA non-ROCm)."""

    def get_configs(
        self,
        size_hints: dict[str, int],
        bs: int,
        triton_config_fn: Callable[..., Any],
        hinted_configs: list[Any],
        tile_hint: TileHint | None = None,
        inductor_meta: dict[str, Any] | None = None,
    ) -> list[Any]:
        # [FUNCTION DOC] Build the list of candidate Triton configs for a pointwise kernel.
        # Args: size_hints (mapping of dim name to size), bs (block size), triton_config_fn (config factory), hinted_configs (extra configs), tile_hint, inductor_meta (may be None).
        # Returns a list of configs from the 1d, 2d or 3d builder. Raises NotImplementedError for other dimension counts.
        inductor_meta = inductor_meta or {}
        n = len(size_hints)
        if n == 1:
            return self._configs_1d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        if n == 2:
            return self._configs_2d(
                size_hints,
                bs,
                hinted_configs,
                triton_config_fn,
                tile_hint,
                inductor_meta,
            )
        if n == 3:
            return self._configs_3d(
                size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
            )
        raise NotImplementedError(f"size_hints: {size_hints}")

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 1d configs. Returns a single default config when pointwise autotune is disabled and max_autotune flags are off;
        # otherwise two configs with different elements-per-warp plus the hinted configs.
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, bs)]
        return [
            triton_config_fn(size_hints, bs, num_elements_per_warp=256),
            triton_config_fn(size_hints, bs // 2, num_elements_per_warp=64),
            *hinted_configs,
        ]

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 2d configs. Returns only a 32x32 config when autotune is disabled or tile hint is square (and max_autotune is off);
        # otherwise a fixed set of tile shapes plus the hinted configs.
        if (
            not inductor_meta.get("autotune_pointwise", True)
            or tile_hint == TileHint.SQUARE
        ) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 3d configs. Returns a single 16x16x16 config unless max_autotune is on, in which case a fixed set of tile shapes plus hinted configs.
        if not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 16, 16, 16)]
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]


# ----------------------------------------------------------------------
# ROCm/HIP pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "hip", register=torch.version.hip is not None)
class ROCmPointwiseHeuristic(PointwiseHeuristic):
    # [CLASS DOC] ROCm/HIP pointwise heuristic; inherits the 3d builder and overrides the 1d and 2d builders with extra ROCm-tuned configs.
    """Pointwise configs for ROCm/HIP devices.

    Inherits _configs_3d from PointwiseHeuristic (same configs).
    """

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: Base 1d configs plus, when the base returned more than one, extra ROCm configs (max block, 4096, and an 8-warp variant);
        # adds a single-warp config when atomic adds were found in the kernel. Returns the list.
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.extend(
                [
                    triton_config_fn(size_hints, TRITON_MAX_BLOCK["X"], waves_per_eu=2),
                    triton_config_fn(
                        size_hints,
                        4096,  # wrt: better than the max_block for some kernel
                    ),
                    triton_config_fn(
                        size_hints,
                        2048,
                        num_warps=8,
                        num_stages=2,
                        waves_per_eu=1,  # 20% improvement
                    ),
                ]
            )
            if inductor_meta.get("atomic_add_found"):
                configs.append(
                    triton_config_fn(
                        size_hints,
                        64,
                        num_warps=1,
                        num_stages=1,  # 250% improvement
                    )
                )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 2d configs for ROCm. Does not special-case square tile hints; returns one 32x32 config when autotune is disabled,
        # otherwise the base shapes, hinted configs, and extra ROCm-tuned shapes.
        # ROCm doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            triton_config_fn(size_hints, 64, 32),  # better for some kernels
            triton_config_fn(size_hints, 128, 16),  # +10% for some kernels
            triton_config_fn(size_hints, 128, 32),  # additional 10% more
            triton_config_fn(size_hints, 32, 512),  # +30% for some kernels
        ]


# ----------------------------------------------------------------------
# XPU pointwise heuristic
# ----------------------------------------------------------------------


@register_codegen_heuristic("pointwise", "xpu", register=torch.xpu._is_compiled())
class XPUPointwiseHeuristic(PointwiseHeuristic):
    # [CLASS DOC] XPU pointwise heuristic; overrides the 1d, 2d and 3d builders with XPU-specific configs.
    """Pointwise configs for XPU devices."""

    def _configs_1d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: Base 1d configs plus an extra small-block config (32) when more than one config exists (workaround for an Intel Triton backend issue).
        configs = super()._configs_1d(
            size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
        )
        if len(configs) > 1:
            configs.append(
                # intel-xpu-backend-for-triton #5133
                triton_config_fn(size_hints, 32)
            )
        return configs

    def _configs_2d(
        self, size_hints, bs, hinted_configs, triton_config_fn, tile_hint, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 2d configs for XPU. Does not special-case square tile hints; returns one 32x32 config when autotune is disabled,
        # otherwise base shapes, hinted configs, and two XPU-specific configs.
        # XPU doesn't skip autotune for SQUARE tile hints
        if not inductor_meta.get("autotune_pointwise", True) and not (
            inductor_meta.get("max_autotune")
            or inductor_meta.get("max_autotune_pointwise")
        ):
            return [triton_config_fn(size_hints, 32, 32)]
        return [
            triton_config_fn(size_hints, 32, 32),
            triton_config_fn(size_hints, 64, 64),  # ~8% better for fp16
            triton_config_fn(size_hints, 256, 16),
            triton_config_fn(size_hints, 16, 256),
            triton_config_fn(size_hints, bs, 1),
            triton_config_fn(size_hints, 1, bs),
            *hinted_configs,
            # intel-xpu-backend-for-triton #5198
            triton_config_fn(size_hints, 32, 32, num_warps=8),
            # intel-xpu-backend-for-triton #5199
            triton_config_fn(size_hints, 4, 256),
        ]

    def _configs_3d(
        self, size_hints, bs, hinted_configs, triton_config_fn, inductor_meta
    ):
        # [FUNCTION DOC] HELPER: 3d configs for XPU. Always returns the full set of configs, with no max_autotune gate.
        # XPU always uses full set of configs (no autotune gate)
        return [
            triton_config_fn(size_hints, 16, 16, 16),
            triton_config_fn(size_hints, 64, 8, 8),
            triton_config_fn(size_hints, 8, 64, 8),
            triton_config_fn(size_hints, 8, 8, 64),
            triton_config_fn(size_hints, bs, 1, 1),
            triton_config_fn(size_hints, 1, bs, 1),
            triton_config_fn(size_hints, 1, 1, bs),
            *hinted_configs,
        ]
