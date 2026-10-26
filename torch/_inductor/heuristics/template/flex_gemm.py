# mypy: allow-untyped-defs
# [MODULE DOC] QuACK GEMM config helpers for FlexGEMM: convert configs to hashable keys and back, list device-compatible configs in a measured priority order,
# filter by user-pinned fields and choose an untuned default by shape. Needs torch._vendor.quack. No import-time side effects beyond imports.
from __future__ import annotations

from dataclasses import fields
from functools import cache
from typing import Any, TYPE_CHECKING, TypeAlias

import sympy

import torch
import torch._vendor.quack.gemm_config as quack_gemm_config
from torch.utils._ordered_set import OrderedSet


if TYPE_CHECKING:
    from collections.abc import Sequence

GemmConfigKey: TypeAlias = tuple[tuple[str, Any], ...]


def gemm_config_key(config: quack_gemm_config.GemmConfig) -> GemmConfigKey:
    # [FUNCTION DOC] Return a tuple of (field name, value) pairs for every field of the QuACK GemmConfig dataclass; a hashable key.
    """Project a QuACK GEMM config using the dataclass schema as the contract."""
    return tuple(
        (field.name, getattr(config, field.name))
        for field in fields(quack_gemm_config.GemmConfig)
    )


def gemm_config_from_key(config_key: GemmConfigKey) -> quack_gemm_config.GemmConfig:
    # [FUNCTION DOC] Inverse of gemm_config_key: rebuild a GemmConfig from a key of name/value pairs.
    """Reconstruct a QuACK GEMM config from its generated-code cache key."""
    return quack_gemm_config.GemmConfig(**dict(config_key))


def explicit_gemm_configs_for_device(
    config: dict[str, Any], device: torch.device
) -> tuple[quack_gemm_config.GemmConfig, ...]:
    # [FUNCTION DOC] Return candidate configs for the CUDA device that match every pinned field in the config mapping, using exact type matching so bool/int do not alias.
    # Raises NotImplementedError for unknown field names, for a device_capacity targeting a different SM than the device, or when no candidate matches.
    """Return device configs matching every explicitly pinned field.

    Exact type matching prevents bool/int aliases from selecting a different key.
    """
    field_names = tuple(field.name for field in fields(quack_gemm_config.GemmConfig))
    unexpected = [name for name in config if name not in field_names]
    if unexpected:
        raise NotImplementedError(
            "FlexGEMM explicit QUACK config contains unexpected GemmConfig fields: "
            f"{unexpected}"
        )

    candidates = candidate_gemm_configs_for_device(device)
    expected_device_capacity = candidates[0].device_capacity
    requested_device_capacity = config.get("device_capacity")
    if (
        type(requested_device_capacity) is type(expected_device_capacity)
        and requested_device_capacity != expected_device_capacity
    ):
        raise NotImplementedError(
            f"FlexGEMM explicit QUACK config targets SM{requested_device_capacity}0, "
            f"but {device} uses SM{expected_device_capacity}0 configs"
        )
    matches = tuple(
        candidate
        for candidate in candidates
        if all(
            type(value) is type(getattr(candidate, name))
            and value == getattr(candidate, name)
            for name, value in config.items()
        )
    )
    if matches:
        return matches
    raise NotImplementedError(
        f"FlexGEMM explicit QUACK config constraints are not supported on {device}: "
        f"{config}"
    )


@cache
def dense_gemm_config_priority_keys() -> tuple[GemmConfigKey, ...]:
    # [FUNCTION DOC] Return (cached) keys of the dense QuACK configs in measured preference order. The first four are default, skinny, large-rect, large.
    """Return the measured dense FlexGEMM QuACK preference order."""
    configs = (
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            cluster_n=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=192,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=256,
            pingpong=False,
            is_dynamic_persistent=False,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=256,
            tile_n=128,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=2,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=224,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
        quack_gemm_config.GemmConfig(
            tile_m=128,
            tile_n=160,
            pingpong=False,
            is_dynamic_persistent=True,
            cluster_m=1,
            device_capacity=10,
        ),
    )
    return tuple(gemm_config_key(config) for config in configs)


def candidate_gemm_configs_for_device(device: torch.device):
    # [FUNCTION DOC] Return QuACK configs for the device's compute capability major (11 treated as 10), excluding TMA-gather configs,
    # sorted by priority-list position then tile m, tile n, cluster m, cluster n and persistence. Raises RuntimeError if none exist.
    """Return all device-compatible QuACK configs before shape-specific ranking."""
    device_capacity = torch.cuda.get_device_capability(device)[0]
    if device_capacity == 11:
        device_capacity = 10
    priority_map = {
        key: priority for priority, key in enumerate(dense_gemm_config_priority_keys())
    }
    configs = sorted(
        (
            config
            for config in quack_gemm_config.get_all_configs()
            if config.device_capacity == device_capacity and not config.use_tma_gather
        ),
        key=lambda config: (
            priority_map.get(gemm_config_key(config), len(priority_map)),
            config.tile_m,
            config.tile_n,
            config.cluster_m,
            config.cluster_n,
            int(config.is_dynamic_persistent),
        ),
    )
    if not configs:
        raise RuntimeError(
            f"FlexGEMM found no QuACK configs for CUDA device capability "
            f"SM{device_capacity}0"
        )
    return configs


def default_gemm_config_key(
    device: torch.device,
    m,
    n,
    configs: Sequence[quack_gemm_config.GemmConfig] | None = None,
) -> GemmConfigKey:
    # [FUNCTION DOC] Pick the untuned default config key for generated code from candidates (default: all for the device).
    # Uses guard_or_false on symbolic m and n to find min/max dimension and prefers skinny, large-rect or large keys by size ranges, falling back to the default key,
    # then the first candidate. If m versus n cannot be decided, returns the default key if present, else the first candidate. Reads the current graph V.graph sizevars.
    """Return the untuned default QuACK config key for generated code."""
    configs = candidate_gemm_configs_for_device(device) if configs is None else configs
    config_keys = OrderedSet([gemm_config_key(config) for config in configs])
    default_key, skinny_key, large_rect_key, large_key = (
        dense_gemm_config_priority_keys()[:4]
    )

    from torch._inductor.virtualized import V

    guard_or_false = V.graph.sizevars.guard_or_false
    if guard_or_false(sympy.Le(m, n)):
        min_dim, max_dim = m, n
    elif guard_or_false(sympy.Lt(n, m)):
        min_dim, max_dim = n, m
    else:
        return (
            default_key if default_key in config_keys else gemm_config_key(configs[0])
        )

    if guard_or_false(sympy.Lt(min_dim, 512)):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(sympy.And(sympy.Eq(min_dim, 1024), sympy.Eq(max_dim, 1024))):
        preferred_keys = (skinny_key, default_key)
    elif guard_or_false(
        sympy.And(
            sympy.Ge(max_dim, 4096), sympy.Ge(min_dim, 768), sympy.Lt(min_dim, 1024)
        )
    ):
        preferred_keys = (large_key, default_key)
    elif guard_or_false(sympy.And(sympy.Ge(max_dim, 4096), sympy.Eq(min_dim, 1024))):
        preferred_keys = (large_rect_key, default_key)
    elif guard_or_false(sympy.Ge(min_dim, 2048)):
        preferred_keys = (large_key, default_key)
    else:
        preferred_keys = (default_key,)

    for key in preferred_keys:
        if key in config_keys:
            return key
    return gemm_config_key(configs[0])
