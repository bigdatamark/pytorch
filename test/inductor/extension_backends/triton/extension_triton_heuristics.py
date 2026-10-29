# [MODULE DOC] Test-only Triton heuristics extension used by the inductor extension-backend
# tests. Defines a metadata field name constant, an autotuner subclass that validates that the
# custom field survives into the compile metadata, and a minimal pointwise heuristic factory.
# Re-exports user_autotune from the runtime heuristics module (imported only for that purpose).
from typing import Any

from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.triton_heuristics import user_autotune  # noqa: F401


EXTENSION_TRITON_META_FIELD = "extension_custom_field"


class ExtensionCachingAutotuner(triton_heuristics.CachingAutotuner):
    # [CLASS DOC] CachingAutotuner subclass that checks the extension metadata field is carried
    # through. Adds no state; only overrides the compile-metadata creation step.
    def _create_compile_meta(
        self,
        cfg: triton_heuristics.Config,
    ) -> dict[str, Any]:
        # [FUNCTION DOC] HELPER: Builds compile metadata for one config via the parent class.
        # Raises AssertionError if the triton_meta of this autotuner lacks the extension
        # field, or if the metadata returned by the parent lacks it. Returns the parent's
        # metadata mapping unchanged.
        if EXTENSION_TRITON_META_FIELD not in self.triton_meta:
            raise AssertionError
        compile_meta = super()._create_compile_meta(cfg)
        if EXTENSION_TRITON_META_FIELD not in compile_meta:
            raise AssertionError
        return compile_meta


def pointwise(
    size_hints,
    triton_meta,
    tile_hint=None,
    filename=None,
    min_elem_per_thread=0,
    inductor_meta=None,
):
    # [FUNCTION DOC] Pointwise heuristic entry point for the extension backend.
    # Ignores tile_hint and min_elem_per_thread. Builds a single config with a block size
    # of 32 along the x dimension and returns the result of cached_autotune with the
    # POINTWISE heuristic type, the given size_hints, triton_meta, inductor_meta and
    # filename, and ExtensionCachingAutotuner as the autotuner class. The result is a
    # decorator to be applied to a kernel.
    """
    Construct @triton.heuristics() based on size_hints.
    """
    configs = [triton_heuristics.Config({"XBLOCK": 32})]
    return triton_heuristics.cached_autotune(
        size_hints,
        configs,
        triton_meta=triton_meta,
        inductor_meta=inductor_meta,
        heuristic_type=triton_heuristics.HeuristicType.POINTWISE,
        filename=filename,
        caching_autotuner_cls=ExtensionCachingAutotuner,
    )
