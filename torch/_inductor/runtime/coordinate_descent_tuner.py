# mypy: allow-untyped-defs
# [MODULE DOC] Coordinate descent autotuner for triton configs: CoordescTuner perturbs one config field at a time (block
# sizes, warps, stages, ...), benchmarks neighbours via a caller supplied function and keeps improvements. Also has
# get/set helpers for config fields. No import-time side effects besides creating the module logger.
import copy
import itertools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from torch.utils._ordered_set import OrderedSet

from ..utils import get_max_numwarps
from .hints import (
    InductorMeta,
    native_matmul_block_numel,
    native_matmul_persistent_rblock,
    TRITON_MAX_BLOCK,
    TRITON_MAX_TENSOR_NUMEL,
)
from .runtime_utils import red_text, triton_config_to_hashable


if TYPE_CHECKING:
    from .triton_compat import triton


log = logging.getLogger(__name__)


def get_field(config, name):
    # [FUNCTION DOC] Reads a field from a triton-style config: num_warps and num_stages are attributes, waves_per_eu defaults to
    # eight divided by num_warps (integer) if absent, anything else comes from the kwargs mapping (None if missing).
    if name == "num_warps":
        return config.num_warps
    elif name == "num_stages":
        return config.num_stages
    elif name == "waves_per_eu":
        return config.kwargs.get(name, int(8 // config.num_warps))
    else:
        return config.kwargs.get(name, None)


def set_field(config, name, value):
    # [FUNCTION DOC] Writes a field on a triton-style config: num_warps and num_stages as attributes, others into kwargs. Mutates the config.
    if name == "num_warps":
        config.num_warps = value
    elif name == "num_stages":
        config.num_stages = value
    else:
        config.kwargs[name] = value


class CoordescTuner:
    # [CLASS DOC] Coordinate descent tuner. State: mode flags (matmul, native matmul, mix order reduction; matmul and native matmul
    # are mutually exclusive), a name, size hints, inductor metadata, frozen fields excluded from tuning, and a cache of benchmark timings
    # keyed by hashable config. Main entry points: autotune (per-field descent with optional all-direction check) and
    # autotune_single_field (static, one scalar value).
    """
    The coordinate descent tuner. Tune one field/coordinate at a time.

    TODO will it be necessary to tune multiple fields simultaneously.


    TODO: what if both increasing and decreasing a field can improve perf.
          i.e., there are multiple local optima..
    """

    def __init__(
        self,
        is_mm=False,
        is_native_matmul=False,
        is_mix_order_reduction=False,
        name="unknown",
        size_hints=None,
        inductor_meta: InductorMeta | None = None,
        frozen_fields=None,
    ):
        # [FUNCTION DOC] Stores the flags, name, size hints, metadata (default empty) and frozen fields; creates an empty timing cache.
        # Raises AssertionError if both is_mm and is_native_matmul are set.
        self.is_mm = is_mm  # we will tune num_stages for mm

        # Native matmul codegen assumes ZBLOCK=1 always.
        # This is because 3d tl.dot is slow and so we want to tile y and x only.
        # tl.dot also does not support size smaller than 16; we put this restriction.
        self.is_native_matmul = is_native_matmul
        if self.is_mm and self.is_native_matmul:
            raise AssertionError("is_mm and is_native_matmul are mutually exclusive")
        self.is_mix_order_reduction = is_mix_order_reduction
        self.cached_benchmark_results = {}
        self.name = name
        self.size_hints = size_hints
        self.inductor_meta: InductorMeta = (
            inductor_meta if inductor_meta is not None else {}
        )
        self.frozen_fields: OrderedSet[str] = (
            OrderedSet(frozen_fields) if frozen_fields is not None else OrderedSet()
        )

    def get_config_max(self, prefix: str) -> int:
        # [FUNCTION DOC] Returns the maximum block size for a dimension prefix: the global triton limit, lowered to the size hint for that prefix if known.
        max_block = TRITON_MAX_BLOCK[prefix.upper()]
        size_hint = self.size_hints.get(prefix) if self.size_hints is not None else None
        return min(max_block, size_hint) if size_hint is not None else max_block

    def get_warpsmax(self):
        # [FUNCTION DOC] Returns the maximum warp count: threads per block divided by warp size from the metadata when both are present,
        # otherwise queries the device.
        # Avoid querying device directly if device properties are populated in inductor_meta
        warp_size = self.inductor_meta.get("warp_size")
        max_threads_per_block = self.inductor_meta.get("max_threads_per_block")
        if warp_size and max_threads_per_block:
            return max_threads_per_block // warp_size
        else:
            return get_max_numwarps()

    def cache_benchmark_result(self, config, timing):
        # [FUNCTION DOC] Records the timing for a config in the cache under its hashable form.
        self.cached_benchmark_results[triton_config_to_hashable(config)] = timing

    def lookup_in_cache(self, config):
        # [FUNCTION DOC] Returns the cached timing for the config, or None if it was not benchmarked.
        return self.cached_benchmark_results.get(triton_config_to_hashable(config))

    def call_func(self, func, config):
        # [FUNCTION DOC] Returns the cached timing for the config if present; otherwise calls the benchmark function, caches and returns its timing.
        # Exceptions from the function propagate.
        found = self.lookup_in_cache(config)
        if found is not None:
            log.debug("  CACHED")
            return found
        timing = func(config)
        self.cache_benchmark_result(config, timing)
        return timing

    @property
    def tunable_fields(self) -> list[str]:
        # [FUNCTION DOC] Property returning the ordered list of field names to tune: optional combo-kernel fields from the metadata first, then
        # block fields and num_warps, plus num_stages for matmul modes, waves_per_eu on HIP, NUM_STAGES for mix order reduction, with ZBLOCK
        # removed for native matmul; frozen fields are filtered out. Evaluated on every access.
        out: list[str] = [
            "XBLOCK",
            "YBLOCK",
            "ZBLOCK",
            # NOTE: we should not tune R0_BLOCK for persistent reduction.
            # We rely on the fact that persistent reduction's triton.Config
            # does not have the R0_BLOCK field to guarantee that.
            "R0_BLOCK",
            "R1_BLOCK",
            # the following 3 are for mm
            "BLOCK_M",
            "BLOCK_N",
            "BLOCK_K",
            "num_warps",
        ]
        if self.is_mm:
            out.append("num_stages")
        if self.inductor_meta.get("is_hip") is True:
            out.append("waves_per_eu")
        if self.is_native_matmul:
            out.append("num_stages")
            out.remove("ZBLOCK")  # ZBLOCK=1 always in native matmul

        if self.is_mix_order_reduction:
            # unlike TritonConfig.num_stages, this one is
            # put in TritonConfig.kwargs["NUM_STAGES"] and is used to
            # control the stage of pipelining of tl.range.
            out.append("NUM_STAGES")

        # Combo-kernel per-subkernel block fields (e.g. XBLOCK_0/1, YBLOCK_0/1)
        # come from the worker-side metadata in ``inductor_meta``. Prepend
        # them so coordesc iterates them alongside the base fields. Read
        # live each call so any post-construction mutation of
        # ``inductor_meta`` is observed.
        combo_fields: list[str] = self.inductor_meta.get(
            "combo_coordesc_field_order", []
        )
        out = combo_fields + out
        return [f for f in out if f not in self.frozen_fields]

    def value_too_large(self, name: str, val: int) -> bool:
        # [FUNCTION DOC] Returns whether a candidate value exceeds the limit for the field: per-field combo limits from the metadata first, then
        # the block maximum for BLOCK fields, the warp maximum for num_warps, eight for waves_per_eu; other fields are never too large.
        # KNOWN ISSUE: strip removes any of the characters in BLOCK from both ends rather than a suffix, which works for current names only by luck.
        field_limits = self.inductor_meta.get("combo_coordesc_field_limits")
        if isinstance(field_limits, dict) and name in field_limits:
            return val > field_limits[name]

        block_suffix = "BLOCK"
        if name.endswith(block_suffix):
            prefix = name.strip(block_suffix).lower()
            return val > self.get_config_max(prefix)
        if name == "num_warps":
            return val > self.get_warpsmax()
        if name == "waves_per_eu":
            return val > 8

        return False

    def value_too_small(self, name: str, val: int) -> bool:
        # [FUNCTION DOC] Returns whether a candidate value is below the minimum: per-field minimums from the metadata for XBLOCK and R0_BLOCK,
        # at least 16 for some blocks in native matmul mode, and always true for zero or negative values.
        min_block = None
        if name == "XBLOCK":
            min_block = self.inductor_meta.get("min_xblock")
        elif name == "R0_BLOCK":
            min_block = self.inductor_meta.get("min_rblock")
        if min_block is not None and val < min_block:
            return True

        # In native matmul, block size should be >= 16 for tl.dot
        if self.is_native_matmul:
            if name in ["YBLOCK", "XBLOCK", "R0_BLOCK"]:
                return val < 16

        # Break if value becomes 0/neg
        return val <= 0

    def get_neighbour_values(self, name, orig_val, radius=None, include_self=False):
        # [FUNCTION DOC] Returns the neighbours of a value within the radius (default 1; at least 2 for NUM_STAGES). Doubling or adding one (stage
        # fields) going up and halving or subtracting one going down, stopping at the first too-large or too-small value. Optionally appends
        # the original value last. Raises AssertionError for a radius below 1.
        """
        Get neighbour values in 'radius' steps. The original value is not
        returned as its own neighbour.
        """
        if radius is None:
            radius = 1
        if name == "NUM_STAGES":
            # we see cases that
            # NUM_STAGES=1 is better than NUM_STAGES=2
            # while NUM_STAGES=1 is worse than NUM_STAGES=3
            radius = max(radius, 2)

        if radius < 1:
            raise AssertionError(f"radius must be >= 1, got {radius}")

        def update(cur_val, inc=True):
            # [FUNCTION DOC] HELPER: Steps a value up or down: plus or minus one for stage fields, double or integer half otherwise.
            if name in ["num_stages", "NUM_STAGES"]:
                if inc:
                    return cur_val + 1
                else:
                    return cur_val - 1
            else:
                if inc:
                    return cur_val * 2
                else:
                    return cur_val // 2

        out = []
        # increment loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, True)
            if self.value_too_large(name, cur_val):
                break
            out.append(cur_val)

        # decrement loop
        cur_val = orig_val
        for _ in range(radius):
            cur_val = update(cur_val, False)
            if self.value_too_small(name, cur_val):
                break
            out.append(cur_val)

        if include_self:
            out.append(orig_val)
        return out

    @staticmethod
    def has_improvement(baseline, test):
        # [FUNCTION DOC] Static. Returns true if the test timing exists and beats the baseline by more than 0.1 percent.
        threshold = 0.001  # 0.1%
        return test is not None and test < baseline * (1 - threshold)

    def is_valid_config(self, config) -> bool:
        # [FUNCTION DOC] Returns whether a config satisfies mode constraints: for mix order reduction XBLOCK must not exceed RSPLIT_SIZE;
        # for native matmul the block element count (using the persistent reduction block when R0_BLOCK is absent) must not exceed the triton tensor limit.
        if self.is_mix_order_reduction:
            # Mix order reduction has an extra constraint that
            # we should not tune XBLOCK beyond RSPLIT_SIZE
            xblock = config.kwargs["XBLOCK"]
            split_size = config.kwargs["RSPLIT_SIZE"]
            return xblock <= split_size
        if self.is_native_matmul:
            r0_block = None
            if "R0_BLOCK" not in config.kwargs:
                r0_block = self.inductor_meta.get("native_matmul_persistent_rblock")
                if r0_block is None and self.size_hints is not None:
                    r0_block_hint = self.size_hints.get("r0_")
                    if r0_block_hint is not None:
                        r0_block = native_matmul_persistent_rblock(r0_block_hint)
            if (
                native_matmul_block_numel(config.kwargs, r0_block=r0_block)
                > TRITON_MAX_TENSOR_NUMEL
            ):
                return False
        return True

    def get_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        # [FUNCTION DOC] Builds the Cartesian product of neighbour values (radius from metadata, including the current value) over all tunable
        # fields present on the config, and returns deep-copied valid candidates. Fields absent on the config are skipped.
        # Raises AssertionError if a combination has an unexpected length.
        """Return the Cartesian product of neighbour values across every
        tunable field, as a list of valid candidate configs."""
        candidate_values_list = []
        effective_fields = []
        for field in self.tunable_fields:
            old_value = get_field(config, field)
            if old_value is None:
                continue
            radius = self.inductor_meta.get("coordinate_descent_search_radius", 1)
            candidate_values = self.get_neighbour_values(
                field,
                old_value,
                radius=radius,
                include_self=True,
            )
            candidate_values_list.append(candidate_values)
            effective_fields.append(field)

        configs = []
        for choice in itertools.product(*candidate_values_list):
            if len(choice) != len(effective_fields):
                raise AssertionError(
                    f"Expected {len(effective_fields)} values, got {len(choice)}"
                )
            candidate = copy.deepcopy(config)
            for new_val, field in zip(choice, effective_fields):
                set_field(candidate, field, new_val)
            if self.is_valid_config(candidate):
                configs.append(candidate)
        return configs

    def check_all_tuning_directions(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        best_config,
        best_timing,
    ):
        # [FUNCTION DOC] Tries every config from get_all_tuning_directions against the best so far, adopting each improvement as it goes.
        # Returns a tuple of whether anything improved, the best config and the best timing.
        """
        Check all directions. We only do this once the regular coordinate
        descent tuning find no better choices any more.
        We only have a few tunable fields, so this should be fine.
        """
        improved = False
        for candidate_config in self.get_all_tuning_directions(best_config):
            cmp_res, candidate_timing = self.compare_config(
                func, candidate_config, best_config, best_timing
            )
            if cmp_res:
                improved = True
                best_config = candidate_config
                best_timing = candidate_timing

        return improved, best_config, best_timing

    def compare_config(self, func, candidate_config, best_config, best_timing):
        # [FUNCTION DOC] Benchmarks the candidate (through the cache). Returns a pair of whether it improves on the best timing and the
        # candidate timing; if the benchmark raises, returns false and infinity (exception logged at debug level, not propagated).
        """
        Check if candidate_config is better than best_config.

        Return a tuple of (compare_result, candidate_timing).
        compare_result is true iff candidate_config is better.
        """
        log.debug("Try config %s", candidate_config)
        try:
            candidate_timing = self.call_func(func, candidate_config)
        except Exception as e:
            log.debug("Got exception %s", e)
            return False, float("inf")

        if self.has_improvement(best_timing, candidate_timing):
            log.debug(
                "Tune from %s %f -> %s %f",
                best_config,
                best_timing,
                candidate_config,
                candidate_timing,
            )

            return True, candidate_timing
        return False, candidate_timing

    def get_neighbour_configs(
        self,
        # pyrefly: ignore [missing-attribute]
        config: "triton.Config",
        field: str,
    ) -> list["triton.Config"]:  # pyrefly: ignore [missing-attribute]
        # [FUNCTION DOC] Returns valid deep-copied configs obtained by moving only the given field to each neighbour value.
        # Raises AssertionError if the field is not present on the config.
        """Return every valid neighbour of ``config`` along ``field``
        only — the per-axis counterpart of ``get_neighbour_values``.

        Precondition: ``field`` must be present on ``config``. Callers
        iterating ``tunable_fields`` against arbitrary configs must
        filter empty fields up front; ``autotune`` does."""
        cur_val = get_field(config, field)
        if cur_val is None:
            raise AssertionError(f"field {field!r} not present on config")
        neighbours = []
        for next_val in self.get_neighbour_values(field, cur_val):
            candidate = copy.deepcopy(config)
            set_field(candidate, field, next_val)
            if self.is_valid_config(candidate):
                neighbours.append(candidate)
        return neighbours

    def autotune(
        self,
        # pyrefly: ignore [missing-attribute]
        func: Callable[["triton.Config"], float],
        # pyrefly: ignore [missing-attribute]
        baseline_config: "triton.Config",
        baseline_timing: float | None = None,
    ) -> "triton.Config":  # pyrefly: ignore  # missing-attribute
        # [FUNCTION DOC] Runs coordinate descent from the baseline config: repeatedly sweeps each tunable field present on the best config,
        # moving to improving neighbours immediately, until a sweep finds nothing; then, if enabled in the metadata, tries all directions
        # once and continues if that improved. Benchmarks the baseline if its timing is not given. Returns the best config. Logs progress.
        """
        Perform coordinate descent autotuning starting from a baseline configuration.
        """
        if baseline_timing is None:
            baseline_timing = self.call_func(func, baseline_config)

        log.debug("= Do coordinate descent tuning for %s =", self.name)
        log.debug(
            "%s: Baseline Config %s, baseline timing %f",
            self.name,
            baseline_config,
            baseline_timing,
        )
        improved = True
        best_config = baseline_config
        best_timing = baseline_timing

        while improved:
            improved = False

            # Gauss-Seidel: walk one field at a time, applying any
            # improvement immediately so subsequent fields see the
            # updated working point.
            for field in self.tunable_fields:
                # Some kernels don't have R0_BLOCK / YBLOCK / ZBLOCK,
                # so the field may be absent from ``best_config``.
                # Filter here so ``get_neighbour_configs`` can assume
                # the field is present.
                if get_field(best_config, field) is None:
                    continue
                for candidate_config in self.get_neighbour_configs(best_config, field):
                    cmp_res, candidate_timing = self.compare_config(
                        func, candidate_config, best_config, best_timing
                    )
                    if cmp_res:
                        improved = True
                        best_config, best_timing = candidate_config, candidate_timing

            if not improved and self.inductor_meta.get(
                "coordinate_descent_check_all_directions"
            ):
                old_best_timing = best_timing
                improved, best_config, best_timing = self.check_all_tuning_directions(
                    func, best_config, best_timing
                )

                if improved:
                    msg = red_text(
                        "%s: Coordinate descend tuning found improvement of %.3fx by looking in all directions."
                    )
                    log.debug(
                        msg,
                        self.name,
                        old_best_timing / best_timing,
                    )

        log.debug(
            "%s: Improve from %s %f -> %s %f, %.3fx",
            self.name,
            baseline_config,
            baseline_timing,
            best_config,
            best_timing,
            baseline_timing / best_timing,
        )

        return best_config

    @staticmethod
    def autotune_single_field(fn, init_val, min_val=None, max_val=None):
        # [FUNCTION DOC] Static. Tunes one integer by repeatedly trying half and double of the current best (clamped to minimum, default 1,
        # and maximum, default 2 to the 30th) and moving to a candidate whose benchmark result is lower, until no move helps. Results are
        # memoized per call. Returns the best value.
        """
        fn is a function that takes the field value and returns the benchmarking result
        init_val is the starting point of autotuning.

        Should work well for parabola like curve. Here is a real example
        for split-size of mix-order-reduction: https://github.com/pytorch/pytorch/pull/166461
        """
        cache = {}

        def _bench(val):
            # [FUNCTION DOC] HELPER: Returns the memoized benchmark result for a value, calling the function once per value.
            if val not in cache:
                cache[val] = fn(val)
                # print(f"split size {val} -> {cache[val]:.3f} ms")
            return cache[val]

        if min_val is None:
            min_val = 1
        if max_val is None:
            max_val = 2**30  # some arbitrary large value

        best_val = init_val
        improved = True
        while improved:
            improved = False
            candlist = [best_val // 2, best_val * 2]
            for cand in candlist:
                cand = max(cand, min_val)
                cand = min(cand, max_val)

                if _bench(cand) < _bench(best_val):
                    best_val = cand
                    improved = True

        return best_val
