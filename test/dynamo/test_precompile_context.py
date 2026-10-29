# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for PrecompileContext, the registry that collects dynamo cache entries and
# backend (bundled AOT autograd) cache artifacts produced by torch.compile when
# caching_precompile and the autograd cache are enabled. Tests require triton and a GPU type. No
# import-time side effects besides imports.
import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch
from torch._dynamo.precompile_context import BackendCacheArtifact, PrecompileContext
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    BundledAOTAutogradCacheArtifact,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_triton


@functorch_config.patch({"enable_autograd_cache": True})
@torch._dynamo.config.patch(
    {"caching_precompile": True}
)  # Requires bundledaotautograd cache for now
class PrecompileContextTests(InductorTestCase):
    # [CLASS DOC] Inductor test case, patched to enable the functorch autograd cache and the
    # caching_precompile dynamo option. Each test compiles a tiny function, runs forward and
    # backward, and inspects the PrecompileContext state (dynamo cache entries, backend artifacts
    # keyed by string, created cache entries).
    def setUp(self):
        # [FUNCTION DOC] Fixture run before each test: calls the parent setUp and then clears all
        # artifacts stored in PrecompileContext so tests do not see one another's entries.
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        # Clear PrecompileContext cache artifacts
        PrecompileContext.clear()

    @requires_triton()
    def test_basic(self):
        # [FUNCTION DOC] Compiles a sine-plus-cosine function on the GPU, runs forward and
        # backward, then checks that exactly one dynamo cache entry and one backend artifact are
        # recorded and that create_cache_entries returns one entry.
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            # [FUNCTION DOC] HELPER: Function under test returning sine plus cosine of its input.
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)  # noqa: UNSPECIFIED_BACKEND

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        cache_entries, _ = PrecompileContext.create_cache_entries()
        self.assertEqual(len(cache_entries), 1)

    @requires_triton()
    def test_serialize_by_key(self):
        # [FUNCTION DOC] Compiles and runs the same function, then serializes each backend
        # artifact by key and checks the result is a BackendCacheArtifact with the matching key;
        # raises AssertionError otherwise. Finally checks create_cache_entries still returns
        # exactly one entry.
        def simple_function(x):
            # [FUNCTION DOC] HELPER: Function under test returning sine plus cosine of its input.
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)  # noqa: UNSPECIFIED_BACKEND

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        for key in PrecompileContext._backend_artifacts_by_key:
            result = PrecompileContext.serialize_artifact_by_key(key)
            if not isinstance(result, BackendCacheArtifact):
                raise AssertionError(
                    f"Expected BackendCacheArtifact, got {type(result)}"
                )
            self.assertEqual(result.key, key)

        # This should still work
        result, _ = PrecompileContext.create_cache_entries()
        if len(result) != 1:
            raise AssertionError(f"Expected len(result) == 1, got {len(result)}")

    @requires_triton()
    def test_editable(self):
        # [FUNCTION DOC] Compiles and runs the function, then edits the single stored backend
        # artifact through edit_artifact with a function that sets a private attribute on the
        # content. Checks the re-serialized artifact is a BundledAOTAutogradCacheArtifact with
        # the same key, that create_cache_entries yields one entry with one backend artifact, and
        # that the attribute value 42 is present on its content.
        """
        Test that after torch.compile, PrecompileContext._new_cache_artifacts length is 1
        """

        def simple_function(x):
            # [FUNCTION DOC] HELPER: Function under test returning sine plus cosine of its input.
            return x.sin() + x.cos()

        compiled_fn = torch.compile(simple_function)  # noqa: UNSPECIFIED_BACKEND

        # Run the compiled function
        x = torch.randn(10, device=GPU_TYPE, requires_grad=True)
        result = compiled_fn(x)
        result.sum().backward()
        self.assertEqual(len(PrecompileContext._dynamo_cache_entries), 1)
        self.assertEqual(len(PrecompileContext._backend_artifacts_by_key), 1)
        # Find the key for the artifact of type "precompile_aot_autograd"
        key = next(iter(PrecompileContext._backend_artifacts_by_key))

        def edit_fn(x):
            # [FUNCTION DOC] HELPER: Edit callback passed to edit_artifact; sets a private
            # attribute with value 42 on the artifact content object and returns the object.
            x._my_private_field = 42
            return x

        PrecompileContext.edit_artifact(key, edit_fn)

        result = PrecompileContext.serialize_artifact_by_key(key)
        if not isinstance(result, BundledAOTAutogradCacheArtifact):
            raise AssertionError(
                f"Expected BundledAOTAutogradCacheArtifact, got {type(result)}"
            )
        self.assertEqual(result.key, key)

        result, _ = PrecompileContext.create_cache_entries()
        if len(result) != 1:
            raise AssertionError(f"Expected len(result) == 1, got {len(result)}")
        aot_autograd_artifacts = next(iter(result.values())).backends
        if len(aot_autograd_artifacts) != 1:
            raise AssertionError(
                f"Expected len(aot_autograd_artifacts) == 1, got {len(aot_autograd_artifacts)}"
            )
        entry = next(iter(aot_autograd_artifacts.values())).content
        self.assertEqual(entry._my_private_field, 42)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
