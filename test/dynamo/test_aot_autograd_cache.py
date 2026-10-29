# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for AOTAutogradCache, the on-disk cache of AOT autograd compilation
# results used by torch.compile with the inductor backend. Contains custom-pass and
# fake-process-group helpers, a mixin that checks the standalone autograd_cache_key matches the
# key made during compile_fx, and the AOTAutogradCacheTests suite (hit/miss counters, hot
# loading of artifacts, bypass rules, guards, custom autograd functions, triton ops, saved
# tensor hooks, regional inductor, cache key sensitivity to config and passes). Importing
# registers module level custom graph pass instances and computes device_type; running the file
# uses the standard PyTorch test runner.

import contextlib
import copy
import dataclasses
import functools
import multiprocessing
import operator
import os
import pickle
import shutil
import unittest
from collections.abc import Sequence
from typing import Literal
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.test_case
import torch._functorch._aot_autograd
import torch._functorch._aot_autograd.autograd_cache as autograd_cache
import torch._functorch.aot_autograd as aot_autograd
import torch._inductor.compile_fx as compile_fx
from torch._dynamo import config as dynamo_config
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import (
    AOTAutogradCache,
    AOTAutogradCachePickler,
    autograd_cache_key,
    BypassAOTAutogradCache,
    check_cacheable,
    sanitize_gm_for_cache,
)
from torch._functorch._aot_autograd.schemas import AOTConfig, CacheableAOTConfig
from torch._guards import TracingContext
from torch._inductor import config as inductor_config
from torch._inductor.custom_graph_pass import (
    CustomGraphPass,
    CustomGraphPassType,
    CustomRuntimeEstimator,
)
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.runtime.triton_compat import tl, triton
from torch._inductor.standalone_compile import (
    autograd_cache_key as standalone_compile_autograd_cache_key,
)
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import fresh_cache, InputType
from torch._subclasses import FakeTensorMode
from torch.compiler._cache import CacheArtifactManager
from torch.fx import GraphModule
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    skipIfWindows,
    skipIfXpu,
    subtest,
    TEST_CUDA,
    TEST_WITH_ASAN,
    TEST_WITH_SLOW,
    TEST_XPU,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, requires_triton
from torch.testing._internal.triton_utils import requires_gpu_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class CustomPreGradPassRemoveIdentMuls(CustomGraphPass):
    # [CLASS DOC] HELPER: Test custom pre-grad graph pass that rewrites an FX graph in place,
    # replacing multiplications whose left operand is the constant 1 by the right operand.
    # Reports a stable uuid so the cache key can include it.
    """
    Pre-grad pass that removes redundant identity multiplications (1 * x).
    """

    def __call__(self, g: torch.fx.Graph) -> None:
        # [FUNCTION DOC] HELPER: Walks the nodes of graph g; for each call_function node whose
        # target is operator.mul and whose first argument equals 1, redirects all users of
        # that node to the second argument and erases the node. Mutates g in place and returns
        # None. A multiplication with 1 as the right operand is not touched (only the left
        # operand is checked).
        for n in g.nodes:
            if n.op == "call_function" and n.target is operator.mul:
                lhs, rhs = n.args
                if lhs == 1:
                    n.replace_all_uses_with(rhs)
                    g.erase_node(n)

    def uuid(self):
        # [FUNCTION DOC] HELPER: Returns the fixed string identifying this pass version, which
        # inductor mixes into the cache key.
        return "custom_pre_grad_pass_remove_ident_muls_v1"


class CustomPreGradPassRemoveIdentMulsNoUUID(CustomPreGradPassRemoveIdentMuls):
    # [CLASS DOC] HELPER: Variant of the identity-mul pass that reports no uuid, so the pass
    # cannot be hashed and the cache is expected to be bypassed.
    def uuid(self):
        # [FUNCTION DOC] HELPER: Returns None instead of a uuid, marking the pass
        # as not cacheable.
        return None


custom_pre_grad_pass_remove_ident_muls = CustomPreGradPassRemoveIdentMuls()
custom_pre_grad_pass_remove_ident_muls_wo_uuid = (
    CustomPreGradPassRemoveIdentMulsNoUUID()
)


@contextlib.contextmanager
def _fake_process_group(rank=0, world_size=2):
    # [FUNCTION DOC] HELPER: Context manager creating a "fake" c10d process group with a
    # FakeStore for the given rank and world_size (no real communication). Yields once the group
    # is initialized; always destroys the process group on exit, including when the body raises.
    """Context manager for setting up and tearing down a fake process group."""
    import torch.distributed as c10d
    from torch.testing._internal.distributed.fake_pg import FakeStore

    fake_store = FakeStore()
    c10d.init_process_group("fake", store=fake_store, rank=rank, world_size=world_size)
    try:
        yield
    finally:
        c10d.destroy_process_group()


def aot_eager_regional_inductor():
    # [FUNCTION DOC] HELPER: Builds a dynamo backend (via aot_autograd) that uses the
    # regional_inductor pass as both the forward and the backward compiler. Returns the backend
    # callable; nothing is compiled until it is invoked.
    """
    Regional inductor backend for AOT autograd.
    Uses regional_inductor as both forward and backward compiler.
    """
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
    )


def saved_tensors_hooks_to_gm(
    pack_fn,
    unpack_fn,
    pack_cache_hash=None,
    unpack_cache_hash=None,
    symbolic_tracing=True,
    inp_fn=None,
):
    # [FUNCTION DOC] HELPER: Converts a pair of saved-tensor pack/unpack hook functions into a
    # pair of GraphModules (pack_gm, unpack_gm) and returns them. With symbolic_tracing, uses
    # torch.fx.symbolic_trace on each function. Otherwise runs pack_fn on an example input (from
    # inp_fn, or a random 2x3 tensor with both dims marked dynamic) and traces both with
    # make_fx, feeding the unpack function the pack output. If pack_cache_hash /
    # unpack_cache_hash are given, every node marked is_wrapped in the corresponding graph gets
    # that value as its user_cache_hash metadata.
    if symbolic_tracing:
        pack_gm = torch.fx.symbolic_trace(pack_fn)
        unpack_gm = torch.fx.symbolic_trace(unpack_fn)
    else:
        from functorch import make_fx

        if inp_fn:
            inp = inp_fn()
        else:
            inp = torch.randn(2, 3)
            torch._dynamo.mark_dynamic(inp, 0)
            torch._dynamo.mark_dynamic(inp, 1)
        pack_out = pack_fn(inp)
        pack_gm = make_fx(pack_fn)(inp)
        unpack_gm = make_fx(unpack_fn)(pack_out)

    def set_manual_hash(g, manual_hash):
        # [FUNCTION DOC] HELPER: Sets the user_cache_hash metadata to manual_hash on every node
        # of graph g whose meta has a true is_wrapped entry.
        for node in g.nodes:
            if node.meta and node.meta.get("is_wrapped", False):
                node.meta["user_cache_hash"] = manual_hash

    if pack_cache_hash:
        set_manual_hash(pack_gm.graph, pack_cache_hash)
    if unpack_cache_hash:
        set_manual_hash(unpack_gm.graph, unpack_cache_hash)
    return pack_gm, unpack_gm


def amax_to_scale(
    amax: torch.Tensor,
    float8_dtype: torch.dtype,
    round_scales_to_power_of_2: bool = False,
):
    # [FUNCTION DOC] HELPER: Computes an fp8 scaling factor: the max finite value of
    # float8_dtype divided by amax (clamped below at 1e-12), computed in float64 and returned as
    # a float32 tensor. round_scales_to_power_of_2 is accepted but ignored.
    amax = amax.to(torch.float64)
    res = torch.finfo(float8_dtype).max / torch.clamp(amax, min=1e-12)
    res = res.to(torch.float32)
    return res


# Must be at module level to use fx.wrap
@torch.fx.wrap
def _pack_fp8_with_scale_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped (opaque leaf in symbolic tracing) pack hook. For
    # non-floating tensors returns x unchanged. Otherwise returns a tuple of the original dtype,
    # a float32 scale from the tensor's abs max, and the tensor scaled and cast to float8_e5m2.
    if not x.dtype.is_floating_point:
        return x

    amax = torch.max(torch.abs(x))
    scale = amax_to_scale(amax, torch.float8_e5m2)
    x_scaled = x.to(torch.float32) * scale
    x_fp8 = x_scaled.to(torch.float8_e5m2)
    return x.dtype, scale, x_fp8


@torch.fx.wrap
def _unpack_fp8_with_scale_wrap(x):
    # [FUNCTION DOC] HELPER: fx-wrapped unpack hook inverse of the pack hook. A plain tensor is
    # returned unchanged; a (dtype, scale, fp8 tensor) tuple is divided by the scale in float32
    # and cast back to the original dtype.
    if isinstance(x, torch.Tensor):
        return x

    dtype, scale, x_fp8 = x
    y = x_fp8.to(torch.float32) / scale
    return y.to(dtype)


class CacheKeyEquivalenceMixin:
    # [CLASS DOC] HELPER: Test mixin placed before the test case in the MRO. Its setUp patches
    # torch._inductor.compile_fx.compile_fx with a wrapper that, for every compiled graph, also
    # computes the standalone cache key and checks it equals the key computed in the real
    # pipeline. State: _compile_fx_patcher, started in setUp and stopped in tearDown.
    """Mixin that verifies the standalone autograd_cache_key API produces the
    same key as the internal compilation pipeline for every torch.compile
    invocation in the test."""

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setUp, then creates and starts the compile_fx
        # patcher so the rest of the test runs with key-equivalence checks.
        super().setUp()
        self._compile_fx_patcher = self._install_compile_fx_patcher()
        self._compile_fx_patcher.start()

    def _install_compile_fx_patcher(self):
        # [FUNCTION DOC] HELPER: Returns an unstarted mock patch object replacing
        # compile_fx.compile_fx with a capturing wrapper that delegates to the real function
        # (saved before patching).
        real_compile_fx = torch._inductor.compile_fx.compile_fx

        def capturing_compile_fx(mod, example_inputs_, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper around the real compile_fx. Computes the
            # standalone key first (treating BypassAOTAutogradCache or NotImplementedError as
            # "bypassed") and asserts the module code was not mutated. Then runs the real
            # compile_fx with autograd_cache_key and run_pre_grad_passes patched to record the
            # real key and whether pre-grad passes changed the code. Unless bypassed or
            # pre-grad passes changed the graph, asserts the two keys are equal. Returns the
            # real compile_fx result.
            mod.recompile()
            mod_code = mod.code

            # Record standalone key before invoking the inner compile_fx function.
            standalone_key = standalone_debug_lines = None
            standalone_bypassed = False
            try:
                standalone_key, standalone_debug_lines = compile_fx.autograd_cache_key(
                    mod, example_inputs_, ignore_shape_env=False
                )
            except (BypassAOTAutogradCache, NotImplementedError):
                standalone_bypassed = True
            mod.recompile()
            self.assertEqual(
                mod.code,
                mod_code,
                "compile_fx.autograd_cache_key must not mutate the model",
            )

            # Setup to capture the ground truth key.
            captured_gt_key = captured_gt_debug_lines = None
            real_cache_key = autograd_cache.autograd_cache_key

            def capturing_cache_key(
                mod, ei, config, compiler_config_extra=None, act_input_paths=()
            ):
                # [FUNCTION DOC] HELPER: Calls the real autograd_cache_key with the same
                # arguments, stores its key and debug lines in the enclosing captured_gt_*
                # variables, and returns the result unchanged.
                nonlocal captured_gt_key, captured_gt_debug_lines
                result = real_cache_key(
                    mod, ei, config, compiler_config_extra, act_input_paths
                )
                captured_gt_key, captured_gt_debug_lines = result
                return result

            # Setup to capture pre grad changes.
            captured_pre_grad_change = False
            real_run_pre_grad_passes = torch._inductor.compile_fx.run_pre_grad_passes

            def capturing_run_pre_grad_passes(
                mod_inner: GraphModule, example_inputs_: Sequence[InputType]
            ) -> torch.fx.GraphModule:
                # [FUNCTION DOC] HELPER: Runs the real pre-grad passes on mod_inner and records
                # (in the enclosing captured_pre_grad_change) whether the generated code of the
                # module differs before and after. Returns the pass result.
                nonlocal captured_pre_grad_change
                mod_inner.recompile()
                mod_inner_code = mod_inner.code
                result = real_run_pre_grad_passes(mod_inner, example_inputs_)
                mod_inner.recompile()
                captured_pre_grad_change = mod_inner.code != mod_inner_code
                return result

            # Invoke the inner compile_fx function with capture patches.
            with (
                patch(
                    "torch._functorch._aot_autograd.autograd_cache.autograd_cache_key",
                    side_effect=capturing_cache_key,
                ),
                patch(
                    "torch._inductor.compile_fx.run_pre_grad_passes",
                    side_effect=capturing_run_pre_grad_passes,
                ),
            ):
                result = real_compile_fx(mod, example_inputs_, **kwargs)

            # Verify standalone key matches the one produced during compilation.
            if not standalone_bypassed and not captured_pre_grad_change:
                self.assertEqual(
                    standalone_key,
                    captured_gt_key,
                    "standalone cache key differs from compile_fx cache key",
                )

            return result

        return patch(
            "torch._inductor.compile_fx.compile_fx",
            side_effect=capturing_compile_fx,
        )

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Stops the compile_fx patcher, then runs the parent tearDown.
        self._compile_fx_patcher.stop()
        super().tearDown()


@instantiate_parametrized_tests
class AOTAutogradCacheTests(CacheKeyEquivalenceMixin, InductorTestCase):
    # [CLASS DOC] Test suite for AOTAutogradCache. Each test enables the autograd cache via
    # functorch config, compiles small functions with torch.compile, and checks
    # counters["aot_autograd"] values for autograd_cache_miss / hit / saved (and related
    # inductor counters) across recompiles after clearing dynamo state. Inherits the
    # key-equivalence check from CacheKeyEquivalenceMixin.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setUp, clears the global counters, and clears
        # all caches so each test starts cold.
        """
        Reset all counters and caches before each unit test
        """
        super().setUp()
        counters.clear()
        self._clear_all_caches()

    def _clear_all_caches(self):
        # [FUNCTION DOC] HELPER: Clears the FX graph cache, the AOTAutogradCache, and
        # the CacheArtifactManager, then resets dynamo and the PyCodeCache (the last
        # step via _clear_dynamo_and_codecache).
        """
        Clear every cache, including AOTAutogradCache and FXCache
        """
        torch._inductor.codecache.FxGraphCache.clear()
        AOTAutogradCache.clear()
        CacheArtifactManager.clear()
        self._clear_dynamo_and_codecache()

    def _clear_dynamo_and_codecache(self):
        # [FUNCTION DOC] HELPER: Resets dynamo (dropping in-memory compiled code and guards) and
        # purges the inductor PyCodeCache, leaving on-disk caches intact. This forces the next
        # call to consult the persistent caches.
        """
        Clear unrelated caches, like dynamo and PyCodeCache
        """
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "compile_threads": 1,
        }
    )
    def test_local_cache_stats(self):
        # [FUNCTION DOC] Test that the local AOT autograd cache reports stats in cache_stats
        # under "LocalAOTAutogradCache": after the first compile one miss and one put and no
        # hits; after clearing dynamo and recompiling, one hit. Clears the stats mapping
        # before and after.
        from torch._inductor.remote_cache import cache_stats

        cache_stats._stats.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Simple model function returning x * 2 + y.
            return x * 2 + y

        compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
        a = torch.rand(25)
        b = torch.rand(25)
        compiled_fn(a, b)

        stats = cache_stats._stats.get("LocalAOTAutogradCache")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.miss, 1)
        self.assertEqual(stats.put, 1)
        self.assertEqual(stats.hit, 0)

        self._clear_dynamo_and_codecache()
        compiled_fn(a, b)

        self.assertEqual(stats.hit, 1)
        cache_stats._stats.clear()

    @requires_triton()
    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    def test_cache_hot_load(self, device, dtype, dynamic):
        # [FUNCTION DOC] Test hot loading of cache artifacts, parametrized over device (GPU and
        # cpu), dtype and dynamic shapes. Compiles forward+backward, saves the artifacts with
        # torch.compiler.save_cache_artifacts and checks per-kind artifact counts, clears all
        # caches (and triton dir) and shows a recompile misses again, then loads the saved bytes
        # and shows the recompile hits the AOT autograd cache. Expected inductor counters differ
        # when bundled_autograd_cache is on. Skips if the GPU or SM80 (bfloat16) is missing.
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function: matmul of sin(x) with y.
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)  # noqa: UNSPECIFIED_BACKEND

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_propagated_dynamic_indices"):
                del a._dynamo_propagated_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0

        if functorch_config.bundled_autograd_cache:
            self.assertEqual(len(cache_info.inductor_artifacts), 0)
        else:
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_propagated_dynamic_indices"):
                del a._dynamo_propagated_dynamic_indices
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(len(cache_info.inductor_artifacts), 0)
            else:
                self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            if hasattr(a, "_dynamo_propagated_dynamic_indices"):
                del a._dynamo_propagated_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 4)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_basic(self):
        # [FUNCTION DOC] Test the basic interaction of FXGraphCache and AOTAutogradCache: the
        # first call misses and saves one entry; after clearing dynamo and codecache, a second
        # call hits without another miss or save.
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x * 2, y @ y).
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True, "compile_threads": 1})
    @functorch_config.patch({"enable_autograd_cache": True})
    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_act_input_paths_cache_hit(self):
        # [FUNCTION DOC] Test that a cache hit still performs the tensor wait for an
        # AsyncCollectiveTensor input (activation input path): compiles an add of 1, checks
        # the wrapper is marked completed after a miss run, then after clearing dynamo
        # checks a hit occurs and wait_tensor was called at least once. Skipped if
        # torch.distributed is unavailable.
        import torch.distributed._functional_collectives as funcol
        from torch.distributed._functional_collectives import AsyncCollectiveTensor

        def fn(x):
            # [FUNCTION DOC] HELPER: Model function returning x + 1.
            return x + 1

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        wait_calls = []
        orig_wait_tensor = funcol.wait_tensor

        def counting_wait_tensor(t):
            # [FUNCTION DOC] HELPER: Replacement for funcol.wait_tensor that records each call
            # in wait_calls and delegates to the original.
            wait_calls.append(1)
            return orig_wait_tensor(t)

        with patch.object(funcol, "wait_tensor", counting_wait_tensor):
            elem = torch.randn(4)
            act = AsyncCollectiveTensor(elem)
            self.assertFalse(act.completed)
            self.assertEqual(compiled_fn(act), elem + 1)
            self.assertTrue(act.completed)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            wait_calls.clear()
            self._clear_dynamo_and_codecache()

            elem = torch.randn(4)
            act = AsyncCollectiveTensor(elem)
            self.assertFalse(act.completed)
            self.assertEqual(compiled_fn(act), elem + 1)
            self.assertTrue(act.completed)
            self.assertGreaterEqual(len(wait_calls), 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_numpy_round_cache_hit(self):
        # [FUNCTION DOC] Test that a function using numpy round on an ndarray input can be
        # cached under strict_autograd_cache: one miss and save, then a hit after clearing
        # dynamo, with results equal to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Model function returning np.round(x).
            return np.round(x)

        inp = np.array([[1.2, 2.7], [-3.5, 4.5]], dtype=np.float32)
        compiled_fn = torch.compile(fn, backend="inductor")

        np.testing.assert_array_equal(compiled_fn(inp), fn(inp))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()
        np.testing.assert_array_equal(compiled_fn(inp), fn(inp))

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_vmap(self):
        # [FUNCTION DOC] Test torch.vmap inside a compiled function is cached: forward and
        # gradients match eager on a miss and again on a hit after clearing dynamo.
        # KNOWN ISSUE: the existing docstring is just the unfinished word "make".
        """
        make
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies vmap of (x * y + 1).sum(dim=0) over the inputs and
            # returns the sum of the result.
            f = lambda x, y: (x * y + 1).sum(dim=0)  # noqa: E731
            vmapped = torch.vmap(f)(x, y)
            return vmapped.sum(dim=0)

        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset all tensors
        x = torch.randn(25, requires_grad=True)
        y = torch.randn(25, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(x, y), compiled_fn(x2, y2))
        fn(x, y).sum().backward()
        compiled_fn(x2, y2).sum().backward()
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_multi_graph_specialization(self):
        # [FUNCTION DOC] Test that multi-graph specializations (specialize_on for sizes 8 and 16
        # of a dynamic dim) are each cached: three misses and saves at first, then three hits
        # after clearing dynamo.
        """
        Verify multi graph specializations all cache hit
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Model function returning x * 5.
            return x * 5

        a = torch.randn(5)
        a8 = torch.randn(8)
        a16 = torch.randn(16)
        torch._dynamo.mark_dynamic(
            a,
            0,
            specialize_on=[
                lambda x: x == 8,
                lambda x: x == 16,
            ],
        )

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

        self._clear_dynamo_and_codecache()

        # A second call should hit on all 3 graphs
        compiled_fn(a)
        compiled_fn(a8)
        compiled_fn(a16)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_symbol_specialization(self):
        # [FUNCTION DOC] Test that symbol specializations do not cause cache misses: the first
        # call (two inputs maybe-marked dynamic) is one miss and one save. Then a different set
        # of dims is maybe-marked dynamic (also b), dynamo is cleared and the call must hit:
        # still one miss, one hit, one save.
        """
        Verify the symbol specializations don't cause cache miss.
        """

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Returns a tuple of random-noise-plus-inputs and a scaled
            # random value; uses random numbers so output is not deterministic.
            return (torch.randn(5) + x + y, z * torch.randn(1))

        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # A second call should hit even if a new dimension is marked as dynamic
        # that is later specialized as part of tracing.
        a = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(a, 0)
        b = torch.rand(5)
        torch._dynamo.maybe_mark_dynamic(b, 0)
        c = torch.randn(6)
        torch._dynamo.maybe_mark_dynamic(c, 0)
        self._clear_dynamo_and_codecache()

        compiled_fn(a, b, c)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @functorch_config.patch({"enable_autograd_cache": True})
    def test_aot_runtime_trace_joint(self):
        # [FUNCTION DOC] Test that compiling a function over a TwoTensor subclass input
        # with symbolic expand works with the autograd cache: runs forward and backward,
        # clears dynamo, then runs again (cache path). Only checks that nothing raises; no
        # counters are asserted.
        @torch.compile(backend="inductor")
        def f(x):
            # [FUNCTION DOC] HELPER: Model function: sin of x, then expand to a square of the
            # leading size (a symbolic size under dynamic shapes).
            tmp = x.sin()
            s0 = tmp.shape[0]
            return tmp.expand(s0, s0)

        x_a = torch.randn(4, requires_grad=True)
        x = TwoTensor(x_a, x_a.clone())
        out = f(x)
        out.sum().backward()

        self._clear_dynamo_and_codecache()
        out = f(x)
        out.sum().backward()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @skipIfWindows(
        msg="Known issue: Window can't delete loaded modules, so we can't clear module cache."
    )
    def test_clear_fx_graph_cache(self):
        # [FUNCTION DOC] Test the interaction of FxGraphCache clearing with the autograd cache.
        # After a first miss, clears the FX graph cache and dynamo. With bundled_autograd_cache
        # the second call still hits (one miss, one hit, one save); otherwise the entry is
        # invalid, giving two misses, no hit, two saves. Skipped on Windows.
        """
        Verify the interactions between FXGraphCache and AOTAutogradCache.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x * 2, y @ y).
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))

        if functorch_config.bundled_autograd_cache:
            # Bundled AutogradCache doesn't care if FxGraphCache is cleared
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        else:
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            # We save again into the cache
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    @unittest.skipIf(not (TEST_CUDA or TEST_XPU), "GPU is unavailable")
    @requires_triton()
    def test_non_bundled_to_bundled_config_change(self):
        # [FUNCTION DOC] Test that switching bundled_autograd_cache from off to on gives a new
        # cache entry (config is part of the key): first compile misses and saves, then with
        # bundled mode on and dynamo reset, the recompile is another miss and save with no hit.
        # Skipped if bundled mode is already on; needs a GPU and triton.
        if functorch_config.bundled_autograd_cache:
            raise unittest.SkipTest("BundledAutogradCache is already enabled")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x * 2, y @ y).
            return (x * 2, y @ y)

        a = torch.rand(25, device=GPU_TYPE)
        b = torch.rand(5, 5, device=GPU_TYPE)

        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Now turn on bundled autograd cache, see that we successfully save again
        with functorch_config.patch({"bundled_autograd_cache": True}):
            torch._dynamo.reset()
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "view_replay_for_aliased_outputs": True}
    )
    def test_view_replay(self):
        # [FUNCTION DOC] Test caching with view_replay_for_aliased_outputs: a function that
        # mutates its input in place and returns it with a detached alias. Compiled once
        # under original-view tracking, then run three times (dynamo cleared each time)
        # checking outputs equal eager and cache counters: miss 1/hit 0, then hit 1, then hit
        # 2, with no bypass.
        def fn(a):
            # [FUNCTION DOC] HELPER: Model function: takes a detached alias of a, multiplies a
            # by 2 in place, returns a and the alias.
            tmp = a.detach()
            a.mul_(2)
            return a, tmp

        with torch.autograd._force_original_view_tracking(True):
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND

        def run_and_check(miss, hit, bypass):
            # [FUNCTION DOC] HELPER: Clears dynamo, runs eager fn and compiled_fn on fresh equal
            # random inputs under original view tracking, asserts equal outputs and asserts the
            # miss, hit and bypass counters equal the given values.
            self._clear_dynamo_and_codecache()

            inp = torch.rand(2, 3)
            compiled_inp = inp.clone().detach()

            with torch.autograd._force_original_view_tracking(True):
                out = fn(inp)
                compiled_out = compiled_fn(compiled_inp)

            self.assertEqual(out, compiled_out)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], miss)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], hit)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], bypass)

        run_and_check(miss=1, hit=0, bypass=0)
        run_and_check(miss=1, hit=1, bypass=0)
        run_and_check(miss=1, hit=2, bypass=0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_invoke_subgraph(self):
        # [FUNCTION DOC] Test that a function using two calls of a mark_compile_region function
        # (invoke_subgraph) compiles with strict_autograd_cache on, meaning no bypass error is
        # raised. There are no assertions beyond not raising.
        from torch._higher_order_ops.invoke_subgraph import mark_compile_region

        @mark_compile_region
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Region function returning x + y.
            return x + y

        @torch.compile(backend="inductor")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function calling the region function twice and
            # adding the results.
            return gn(x, y) + gn(x, y)

        a = torch.randn(25)
        b = torch.randn(25)

        fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_unsafe_mark_cacheable(self):
        # [FUNCTION DOC] Test that an allow_in_graph autograd Function (with a graph break in
        # forward) makes AOT autograd bypass the cache: compiling raises BackendCompilerFailed
        # mentioning BypassAOTAutogradCache and an unsupported call_function target, with
        # counters miss 0, hit 0, bypass 1. The test then clears caches and unconditionally
        # raises SkipTest (TODO: allow_in_graph produces an unserializable artifact).
        # KNOWN ISSUE: the trailing SkipTest means a cache-hit check is never run.
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function whose forward breaks the graph and returns
            # sin(x); no backward is defined.
            @staticmethod
            def forward(_, x):
                # [FUNCTION DOC] HELPER: Forward: calls graph_break then returns x.sin().
                torch._dynamo.graph_break()
                return x.sin()

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Model function applying the allow_in_graph Function to x; y
            # and z are unused.
            return AllowInGraphFunc.apply(x)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = torch.randn(4, 4)
        args = (x, y, z)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            r".*BypassAOTAutogradCache: Unsupported call_function target .*",
        ):
            fn(*args)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)

        self._clear_dynamo_and_codecache()

        # TODO: Fix allow in graph
        raise unittest.SkipTest(
            "Allow in graph produces an unserializable cache artifact"
        )

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_fx_graph_cache_off(self):
        # [FUNCTION DOC] Test that with fx_graph_cache disabled the autograd cache is bypassed
        # on every compile: bypass counter goes 1 then 2 and there are no hits or saves.
        """
        Should not use cache if FXGraphCache is not enabled
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x * 2, y @ y).
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear FX graph cache: second call should also be a miss
        self._clear_dynamo_and_codecache()

        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @dynamo_config.patch("compiled_autograd", True)
    def test_compiled_autograd_bypass(self):
        # [FUNCTION DOC] Test that with compiled_autograd enabled, a function calling
        # autograd.grad inside the compiled region fails with BackendCompilerFailed whose
        # message says BypassAOTAutogradCache for the unsupported call_function target
        # compiled_autograd.ops.validate_outputs (strict mode turns the bypass into an error).
        # Need to make the compiled autograd graph serializable
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes a.cos() + b, sums it and calls autograd.grad
            # w.r.t. a and b; returns nothing (None).
            out = a.cos() + b
            loss = out.sum()
            ga, gb = torch.autograd.grad(loss, inputs=[a, b])

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        compiled_fn = torch.compile(fn, backend="inductor")
        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "BypassAOTAutogradCache: Unsupported call_function target torch._dynamo.compiled_autograd.ops.validate_outputs",
        ):
            compiled_fn(a, b)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @dynamo_config.patch("compiled_autograd", True)
    def test_inference_graph_cache_hit_with_compiled_autograd_enabled(self):
        # [FUNCTION DOC] Test that a forward-only (inference) graph is still cached when
        # compiled_autograd is enabled: one miss and save the first time; after clearing
        # counters and dynamo, one hit with no miss or save.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Model function returning the sum of cos(a) + b.
            out = a.cos() + b
            return out.sum()

        a = torch.randn(25)
        b = torch.randn(25)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_miss(self):
        # [FUNCTION DOC] Test that changing the body of a custom autograd function (with
        # autograd_cache_allow_custom_autograd_functions on) invalidates the cache. First
        # compile: one miss, one save. The function class is then redefined with cos and sin
        # swapped; after clearing dynamo the recompile gives one miss, no hit, one save. Needs a
        # GPU and triton.
        class MyAutogradFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function: forward returns sin(x), saves it for
            # backward and stores cos(x) on the context; backward combines both.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes y = sin(x), saves y with save_for_backward,
                # stores cos(x) as attribute foo on ctx and returns y.
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns grad_output times saved y plus ctx.foo times
                # grad_output (not the mathematically exact gradient; only used for caching).
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function applying the custom autograd Function.
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=device_type, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        class MyAutogradFunction(torch.autograd.Function):  # noqa: F811
            # [CLASS DOC] HELPER: Redefinition of the custom Function with forward computing
            # cos(x) and keeping sin(x) as foo; it rebinds the name captured by fn so the next
            # compile sees different source.
            # Change the function slightly
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes y = cos(x), saves y, stores sin(x) on ctx.foo
                # and returns y.
                y = x.cos()
                ctx.save_for_backward(y)
                ctx.foo = x.sin()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Same backward as the original: saved y times grad plus
                # ctx.foo times grad.
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        # Clear dynamo and run again. Should be a cache miss.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function(self):
        # [FUNCTION DOC] Test that a custom autograd Function is cacheable when
        # autograd_cache_allow_custom_autograd_functions is on: first compile one miss and save;
        # after clearing counters and dynamo the second is one hit with no miss or save. Results
        # match eager. Needs a GPU and triton.
        class MyAutogradFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function (sin forward, cos kept on ctx).
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Computes y = sin(x), saves y, stores cos(x) on
                # ctx.foo, returns y.
                y = x.sin()
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns grad_output times saved y plus
                # ctx.foo times grad_output.
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function applying the custom autograd Function.
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=device_type, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        result.sum().backward()
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel(self):
        # [FUNCTION DOC] Test caching of a custom autograd Function whose forward calls a user
        # triton_op wrapping a triton kernel: first compile is one miss and save, second (after
        # clearing dynamo) is one hit. Needs a GPU and triton.
        # [FUNCTION DOC] HELPER: Triton kernel loading one value through pointer x and storing
        # that value plus 1 back in place.
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::my_triton_op": clones x into a tensor
            # requiring grad, launches my_jit on it through capture_triton with a grid of one,
            # and returns that tensor.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function whose forward calls the triton op.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Calls the triton op on x, saves the result, stores
                # cos(x) on ctx.foo and returns the result.
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns grad_output times saved result plus
                # ctx.foo times grad_output.
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function applying the custom autograd Function.
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_allow_custom_autograd_functions": True})
    def test_custom_autograd_function_with_custom_triton_kernel_cache_invalidation(
        self,
    ):
        # [FUNCTION DOC] Test that changing the source of a triton kernel used by a triton_op
        # inside a custom autograd Function invalidates the cache. Compiles (miss, save), clears
        # dynamo and hits, then redefines the kernel (adds 2 instead of 1) and the op,
        # recompiles after clearing dynamo and expects the miss counter to be 1 after counters
        # were cleared. Needs a GPU and triton.
        # [FUNCTION DOC] HELPER: Triton kernel that loads one value through pointer x and stores
        # it plus 1 back in place.
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::my_triton_op": clones x into a tensor
            # requiring grad, runs my_jit via capture_triton with a grid of one, returns it.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        class MyAutogradFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd Function whose forward calls the triton op.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Calls the triton op, saves the result, stores cos(x) on
                # ctx.foo and returns the result.
                y = torch.ops.test.my_triton_op(x)
                ctx.save_for_backward(y)
                ctx.foo = x.cos()
                return y

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Returns grad_output times saved result plus
                # ctx.foo times grad_output.
                result = ctx.saved_tensors[0]
                return grad_output * result + ctx.foo * grad_output

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function applying the custom autograd Function.
            return MyAutogradFunction.apply(a)

        a = torch.randn(5, device=GPU_TYPE, requires_grad=True)
        a2 = a.clone().detach_().requires_grad_(True)
        a3 = a.clone().detach_().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)
        result.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and run again. Should be a cache hit.
        counters.clear()
        self._clear_dynamo_and_codecache()
        result = compiled_fn(a2)
        self.assertEqual(fn(a2), result)
        result.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now modify the source code of my_jit by redefining it
        # [FUNCTION DOC] HELPER: Redefined triton kernel: loads one value through pointer x and
        # stores it plus 2 back in place (source differs from before).
        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)  # Changed from +1 to +2

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of "test::my_triton_op" that clones x and runs
            # the new my_jit on it, returning the tensor.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        # Clear dynamo and run again. Should be a cache miss due to modified source code.
        counters.clear()
        self._clear_dynamo_and_codecache()
        compiled_fn = torch.compile(fn, backend="inductor")

        result = compiled_fn(a3)
        # Assert that after changing the source code, the cache no longer hits
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(fn(a3), result)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_cache_invalidation(self):
        # [FUNCTION DOC] Test cache invalidation when a triton_op's kernel changes: the first
        # compile of an op calling a triton kernel is one miss and save; after clearing dynamo
        # the kernel and op are redefined (adds 2 instead of 1) and the recompile must miss
        # again (two misses, no hit, two saves). Needs a GPU and triton.
        from torch._library import capture_triton

        # [FUNCTION DOC] HELPER: Triton kernel loading one value through pointer x and storing
        # it plus 1 back in place.
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::my_triton_op": clones x, runs my_jit via
            # capture_triton with a grid of one, returns the tensor.
            y = x.clone().detach_().requires_grad_(True)
            capture_triton(my_jit)[1,](y)
            return y

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function calling the registered triton op.
            return torch.ops.test.my_triton_op(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        # [FUNCTION DOC] HELPER: Redefined triton kernel that stores the loaded value plus 2.
        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of "test::my_triton_op" using the new kernel.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_local_variable_kernel_detection(self):
        # [FUNCTION DOC] Test that a triton kernel reached through a local variable and an
        # identity wrapper function is recorded for the triton_op in the triton_ops_to_kernels
        # mapping (name inner_kernel), then that running the op adds 42 to every element. Needs
        # a GPU and triton.
        """
        Test that triton kernels passed via local variables are properly detected.

        This tests the pattern:
            kernel_fn = _my_kernel  # global
            wrapped = wrapper(kernel_fn)
            capture_triton(wrapped)[grid](...)

        The local variable tracing in get_inner_triton_kernels should trace
        through the assignments to find the original JITFunction.
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        # [FUNCTION DOC] HELPER: Masked in-place triton kernel: each program handles BLOCK_SIZE
        # elements of x_ptr (bounded by n_elements) and adds 42 to them.
        @triton.jit
        def inner_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 42, mask=mask)

        def identity_wrapper(kernel):
            # [FUNCTION DOC] HELPER: Returns the kernel argument unchanged (stands in for a
            # wrapper like unroll_varargs).
            # simulate a wrapper function (like unroll_varargs in GDPA)
            # that takes a kernel as argument and returns it
            return kernel

        @torch._library.triton_op("test::local_var_triton_op", mutates_args=())
        def local_var_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::local_var_triton_op": clones x, takes the
            # kernel through a local variable and wrapper, and launches it on the clone with
            # BLOCK_SIZE 256 and a grid of ceil(n_elements / BLOCK_SIZE). Returns it.
            y = x.clone()
            n_elements = y.numel()
            # this is the GDPA pattern:
            # 1. Assign global kernel to local variable
            # 2. Pass it through wrapper functions
            # 3. Use the wrapped result with capture_triton
            kernel_fn = inner_kernel  # Direct assignment from global
            wrapped_kernel = identity_wrapper(kernel_fn)  # Wrapper call
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(wrapped_kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::local_var_triton_op", [])
        self.assertGreater(
            len(kernels),
            0,
            "Local variable tracing should detect the kernel",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "inner_kernel",
            kernel_names,
            lambda msg: f"{msg}\ninner_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 42
        result = torch.ops.test.local_var_triton_op(a)
        self.assertEqual(result, expected)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_local_variable_cache_invalidation(self):
        # [FUNCTION DOC] Test that changing a kernel referenced via a local variable in a
        # triton_op invalidates the cache: first compile one miss and save; after the kernel and
        # op are redefined (adds 2 instead of 1) and dynamo cleared, the recompile is a second
        # miss and second save with no hit. Needs a GPU, triton.
        """
        Test that cache properly invalidates when a kernel passed via local
        variable changes.
        """
        from torch._library import capture_triton

        # [FUNCTION DOC] HELPER: Triton kernel loading a value through pointer x and storing it
        # plus 1 back in place.
        @triton.jit
        def versioned_kernel(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::local_var_cache_test", mutates_args=())
        def local_var_cache_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::local_var_cache_test": clones x, runs the
            # kernel (taken from a local variable) on it with a grid of one, returns it.
            y = x.clone().detach_().requires_grad_(True)
            kernel = versioned_kernel  # local assignment
            capture_triton(kernel)[1,](y)
            return y

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function calling the registered triton op.
            return torch.ops.test.local_var_cache_test(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # redef the kernel with different behavior
        # [FUNCTION DOC] HELPER: Redefined kernel storing the loaded value plus 2.
        @triton.jit
        def versioned_kernel(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::local_var_cache_test", mutates_args=())
        def local_var_cache_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of "test::local_var_cache_test" using the new
            # kernel via a local variable.
            y = x.clone().detach_().requires_grad_(True)
            kernel = versioned_kernel  # local assignment
            capture_triton(kernel)[1,](y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Should be a cache miss due to kernel source change
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_recursive_function_kernel_detection(self):
        # [FUNCTION DOC] Test that a kernel called inside a helper function (not directly in the
        # triton_op) is found by the recursive analysis: the mapping triton_ops_to_kernels for
        # the op must contain nested_kernel, and running the op adds 100 to every element. Needs
        # a GPU and triton.
        """
        Test that triton kernels hidden behind helper function calls are detected.

        This tests the recursive function analysis capability where:
            def helper():
                capture_triton(my_kernel)[grid](...)

            @triton_op(...)
            def my_op(x):
                helper()  # kernel is inside helper, not directly in my_op

        The recursive analysis in get_inner_triton_kernels should trace
        into helper functions to find the triton kernels.
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        # [FUNCTION DOC] HELPER: Masked in-place triton kernel adding 100 to BLOCK_SIZE sized
        # chunks of x_ptr, bounded by n_elements.
        @triton.jit
        def nested_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 100, mask=mask)

        def helper_that_calls_kernel(y, n_elements):
            # [FUNCTION DOC] HELPER: Launches nested_kernel on y through capture_triton with
            # BLOCK_SIZE 256 and a grid of ceil(n_elements / BLOCK_SIZE).
            """Helper function that contains the triton kernel call."""
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(nested_kernel)[grid](y, n_elements, BLOCK_SIZE=256)

        @torch._library.triton_op("test::recursive_func_triton_op", mutates_args=())
        def recursive_func_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::recursive_func_triton_op": clones x and
            # calls the helper to run the kernel on the clone, then returns it.
            y = x.clone()
            n_elements = y.numel()
            # The kernel is hidden inside helper_that_calls_kernel
            helper_that_calls_kernel(y, n_elements)
            return y

        kernels = triton_ops_to_kernels.get("test::recursive_func_triton_op", [])
        self.assertGreater(
            len(kernels),
            0,
            "Recursive function analysis should detect the kernel in helper function",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "nested_kernel",
            kernel_names,
            lambda msg: f"{msg}\nnested_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 100
        result = torch.ops.test.recursive_func_triton_op(a)
        self.assertEqual(result, expected)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_kernel_factory_function(self):
        # [FUNCTION DOC] Test that a kernel returned by a factory function is recorded for the
        # triton_op (name factory_kernel) in triton_ops_to_kernels, and that running the op adds
        # 42 to every element. Needs a GPU and triton.
        """
        Test that triton kernels returned from factory functions are properly detected.

        i.e., the pattern:
            kernel = get_autotune_kernel()
            capture_triton(kernel)[grid](...)
        """
        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        # [FUNCTION DOC] HELPER: Masked in-place triton kernel adding 42 to BLOCK_SIZE sized
        # chunks of x_ptr, bounded by n_elements.
        @triton.jit
        def factory_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 42, mask=mask)

        def get_kernel():
            # [FUNCTION DOC] HELPER: Factory returning the factory_kernel kernel.
            """Factory function that returns a triton kernel."""
            return factory_kernel

        @torch._library.triton_op("test::factory_triton_op", mutates_args=())
        def factory_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::factory_triton_op": clones x, obtains the
            # kernel from the factory and launches it on the clone, returns the clone.
            y = x.clone()
            n_elements = y.numel()
            kernel = get_kernel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::factory_triton_op", [])

        self.assertGreater(
            len(kernels),
            0,
            "we should detect the kernel returned by get_kernel()",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "factory_kernel",
            kernel_names,
            lambda msg: f"{msg}\nfactory_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 42
        result = torch.ops.test.factory_triton_op(a)
        self.assertEqual(result, expected)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_lru_cache_factory_function(self):
        # [FUNCTION DOC] Test that a kernel returned through an lru_cache-decorated factory is
        # recorded for the triton_op (name cached_kernel) in triton_ops_to_kernels, and that
        # running the op adds 99 to every element. Needs a GPU and triton.
        """
        Test that triton kernels returned from @lru_cache decorated factory
        functions are properly detected.

        i.e., the pattern:
            @lru_cache
            def get_autotune_kernel_lru():
                return get_autotune_kernel(...)

            kernel = get_autotune_kernel_lru()
            capture_triton(kernel)[grid](...)
        """
        from functools import lru_cache

        from torch._library import capture_triton
        from torch._library.triton import triton_ops_to_kernels

        # [FUNCTION DOC] HELPER: Masked in-place triton kernel adding 99 to BLOCK_SIZE sized
        # chunks of x_ptr, bounded by n_elements.
        @triton.jit
        def cached_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(x_ptr + offsets, x + 99, mask=mask)

        def get_kernel():
            # [FUNCTION DOC] HELPER: Plain factory returning cached_kernel.
            """Factory function."""
            return cached_kernel

        @lru_cache
        def get_cached_kernel():
            # [FUNCTION DOC] HELPER: lru_cache-wrapped factory that calls get_kernel.
            """Calls factory function decorated with @lru_cache."""
            return get_kernel()

        @torch._library.triton_op("test::lru_cache_triton_op", mutates_args=())
        def lru_cache_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::lru_cache_triton_op": clones x, gets the
            # kernel via the cached factory and launches it on the clone, returns it.
            y = x.clone()
            n_elements = y.numel()
            kernel = get_cached_kernel()
            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)  # noqa: E731
            capture_triton(kernel)[grid](y, n_elements, BLOCK_SIZE=256)
            return y

        kernels = triton_ops_to_kernels.get("test::lru_cache_triton_op", [])

        self.assertGreater(
            len(kernels),
            0,
            "we should detect the kernel through @lru_cache",
        )

        kernel_names = [getattr(k, "__name__", str(k)) for k in kernels]
        self.assertIn(
            "cached_kernel",
            kernel_names,
            lambda msg: f"{msg}\ncached_kernel should be detected, got: {kernel_names}",
        )

        a = torch.randn(5, device=GPU_TYPE)
        expected = a.clone() + 99
        result = torch.ops.test.lru_cache_triton_op(a)
        self.assertEqual(result, expected)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_triton_op_cache_multiple_ops_invalidation(self):
        # [FUNCTION DOC] Test cache invalidation with nested triton_ops: op2 calls op1, which
        # runs two kernels. After the first compile (one miss, one save) and a dynamo clear,
        # only my_jit and the ops are redefined (my_jit adds 2); the recompile must miss again
        # (two misses, no hit, two saves). Needs GPU, triton.
        # [FUNCTION DOC] HELPER: Triton kernel loading a value through x and storing it plus 1
        # back in place.
        @triton.jit
        def my_jit(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        # [FUNCTION DOC] HELPER: Second triton kernel, same body as my_jit (plus 1).
        @triton.jit
        def my_jit2(x):
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 1)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::my_triton_op": clones x and runs both
            # kernels on the clone via capture_triton, returns the clone.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: triton_op "test::my_triton_op2": clones x, calls the
            # registered op my_triton_op on it and returns the clone.
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        def fn(a):
            # [FUNCTION DOC] HELPER: Model function calling the registered op my_triton_op2.
            return torch.ops.test.my_triton_op2(a)

        a = torch.randn(5, device=GPU_TYPE)
        a2 = a.clone().detach_()
        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a)
        self.assertEqual(fn(a), result)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        self._clear_dynamo_and_codecache()

        # Redefine the triton op

        # [FUNCTION DOC] HELPER: Redefined kernel storing the loaded value plus 2.
        @triton.jit
        def my_jit(x):  # noqa: F811
            arg_0 = tl.load(x)
            tl.store(x, arg_0 + 2)

        @torch._library.triton_op("test::my_triton_op", mutates_args=())
        def my_triton_op(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of "test::my_triton_op": clones x and runs the
            # new my_jit and the old my_jit2 on the clone, returning it.
            y = x.clone().detach_().requires_grad_(True)
            torch._library.capture_triton(my_jit)[1,](y)
            torch._library.capture_triton(my_jit2)[1,](y)
            return y

        @torch._library.triton_op("test::my_triton_op2", mutates_args=())
        def my_triton_op2(x: torch.Tensor) -> torch.Tensor:  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of "test::my_triton_op2": clones x, calls the
            # registered op my_triton_op on it and returns the clone.
            y = x.clone().detach_().requires_grad_(True)
            torch.ops.test.my_triton_op(y)
            return y

        compiled_fn = torch.compile(fn, backend="inductor")
        result = compiled_fn(a2)

        # Second run should still miss
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        self.assertEqual(fn(a2), result)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_lazy_backward(self):
        # [FUNCTION DOC] Test that backward is compiled lazily and only then is the entry saved
        # (strict mode). Forward only: one miss, saved 0. After clearing dynamo and a second
        # forward: another miss, saved 0. After running backward saved becomes 1 and grads match
        # eager. After a clear and fresh inputs the forward hits (2 misses, 1 hit, 1 save) and
        # backward still matches eager.
        """
        Lazily compile the backward, and lazily save to cache
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Model function returning a.cos() + b.
            return a.cos() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        compiled_fn = torch.compile(fn, backend="inductor")
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Clear dynamo and run again. Should be a cache miss still, because backward hasn't run
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)

        # Now let's run the backward
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Clear dynamo and rerun everything, now there should be a cache hit
        self._clear_dynamo_and_codecache()
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch({"fx_graph_cache": True})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_autograd_no_dynamo_trace_backward(self):
        # [FUNCTION DOC] Test that dynamo does not trace into the compiled backward even on a
        # cache hit. The compiled function runs autograd.grad of x.sum() inside it; first call
        # is a miss with one save, the traced frame infos are recorded, then dynamo is reset and
        # the call repeated: one hit, same result, and the thread-local traced_frame_infos must
        # equal the earlier ones.
        """
        Test that dynamo does not trace into the backward compiled function,
        even on cache hit.
        """
        torch._dynamo.eval_frame.clear_dynamo_tls()

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the gradient of x.sum() with respect to x, computed
            # with autograd.grad during the forward execution.
            # Calls x.sum().backward() during forward execution of fn
            (x_grad,) = torch.autograd.grad(x.sum(), x)
            return x_grad

        a = torch.randn(10, 10, requires_grad=True, device="cpu")
        result = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        # Backward of `sum` will run during execution of graph break
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        traced_frame_infos = copy.deepcopy(
            torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        )

        torch._dynamo.reset()
        torch._dynamo.eval_frame.clear_dynamo_tls()
        result2 = fn(a)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        new_traced_frame_infos = torch._dynamo.eval_frame.dynamo_tls.traced_frame_infos
        self.assertEqual(result, result2)
        # Dynamo should trace exactly the same frames on cache hit
        self.assertEqual(traced_frame_infos, new_traced_frame_infos)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_function(self):
        # [FUNCTION DOC] Test that a function with autograd (forward and backward) hits the
        # cache on the second compile: one miss and save first, then after clearing dynamo one
        # hit with no new miss or save. Outputs and grads match eager.
        """
        Tests autograd cache hits
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Model function returning a.sin() + b.
            return a.sin() + b

        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        compiled_fn = torch.compile(fn, backend="inductor")

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset all tensors
        a = torch.randn(25, requires_grad=True)
        b = torch.randn(25, requires_grad=True)
        a2 = a.detach().clone().requires_grad_(True)
        b2 = b.detach().clone().requires_grad_(True)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self._clear_dynamo_and_codecache()
        self.assertEqual(fn(a, b), compiled_fn(a2, b2))
        fn(a, b).sum().backward()
        compiled_fn(a2, b2).sum().backward()
        self.assertEqual(a.grad, a2.grad)
        self.assertEqual(b.grad, b2.grad)

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @largeTensorTest("64GB", device=GPU_TYPE)
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_guard_single_entry(self, device, dtype):
        # [FUNCTION DOC] Test the single-entry behavior of the cache when inductor guards on
        # 32-bit index math change. With dynamic shapes: first call is a miss, the same shapes
        # hit, a very large b shape yields a guard miss (which counts as a miss), then slightly
        # different small a shape with large b hits, and the original shapes hit again.
        # Parametrized by dtype; requires a large GPU (64GB).
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32. See test_codecache::TestFxGraphCache::test_cache_load_with_guards_int32_bounds.

        This test in particular tests the behavior of a single entry cache. If we ever make AOTAutogradCache
        support multiple entries under the same key, this test should be updated.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x + x, y + y).
            return (x + x, y + y)

        def expect_miss(compiled_fn, a, b):
            # [FUNCTION DOC] HELPER: Clears dynamo/codecache and counters, calls
            # compiled_fn on (a, b) and asserts one cache miss, zero guard misses and zero
            # hits. Returns the result.
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                0,
            )
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            return res

        def expect_hit(compiled_fn, a, b):
            # [FUNCTION DOC] HELPER: Clears dynamo/codecache and counters, calls compiled_fn on
            # (a, b) and asserts zero misses, zero guard misses and one hit. Returns the result.
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                0,
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"],
                1,
            )
            return res

        def expect_guard_miss(compiled_fn, a, b):
            # [FUNCTION DOC] HELPER: Clears dynamo/codecache and counters, calls compiled_fn on
            # (a, b) and asserts one miss, one guard miss and no hit. Returns the result.
            self._clear_dynamo_and_codecache()
            counters.clear()
            res = compiled_fn(a, b)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                1,
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"],
                0,
            )
            return res

        compiled_fn = torch.compile(fn, dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        a_shape = (5, 6)
        b_shape = (7, 8)
        a = torch.rand(a_shape, device=device, dtype=dtype)
        b = torch.rand(b_shape, device=device, dtype=dtype)
        res1 = expect_miss(compiled_fn, a, b)

        # Same shape, should cache hit
        a2 = a.detach().clone()
        b2 = b.detach().clone()

        res2 = expect_hit(compiled_fn, a2, b2)

        self.assertEqual(res1, res2)

        # By changing the shape greatly, despite the same exact input
        # graph, inductor should report a guard miss, leading
        # to a cache miss on our end.
        a_shape = (5, 6)
        b_shape = (47000, 47001)
        a3 = torch.rand(a_shape, device=device, dtype=dtype)
        b3 = torch.rand(b_shape, device=device, dtype=dtype)

        expect_guard_miss(compiled_fn, a3, b3)

        # Wobble the shape a bit, but not enough
        # to trigger a guard miss (since 6, 7 is still less than int32)
        # Should result in a cache hit
        a_shape = (6, 7)
        b_shape = (47000, 47001)
        a4 = torch.rand(a_shape, device=device, dtype=dtype)
        b4 = torch.rand(b_shape, device=device, dtype=dtype)
        expect_hit(compiled_fn, a4, b4)

        # Change the shape back to the original,
        # FXGraphCache should hit because it stores
        # multiple entries
        a_shape = (5, 6)
        b_shape = (7, 8)
        a5 = torch.rand(a_shape, device=device, dtype=dtype)
        b5 = torch.rand(b_shape, device=device, dtype=dtype)
        expect_hit(compiled_fn, a5, b5)

    @largeTensorTest("64GB", device=GPU_TYPE)
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    @parametrize("requires_grad", (True, False))
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_autograd_inductor_guards(self, device, dtype, requires_grad):
        # [FUNCTION DOC] Test the cache with inductor guards across three input shape pairs
        # (small, one large, other large) for a dynamic compile, parametrized by dtype and
        # requires_grad. Each shape is run without resetting dynamo so a recompile happens:
        # expects one more miss and save each time (no extra save at backward). Then dynamo is
        # cleared and a rerun must hit. Guard miss count increments after each iteration.
        # Requires a large GPU (64GB).
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        See test_codecache::TestFxGraphCache::test_cache_load_with_guards_int32_bounds.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x + x, y + y).
            return (x + x, y + y)

        compiled_fn = torch.compile(fn, dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        # Iterate over different shapes, varying whether the total
        # size is below or above int32. For each combination, we expect
        # different guards around whether the symbolic sizes do or do
        # not exceed int32.
        shapes = (
            ((5, 6), (7, 8)),
            ((5, 6), (47000, 47001)),
            ((47000, 47001), (5, 6)),
        )
        expected_hits = expected_misses = expected_saves = 0
        expected_guard_misses = 0
        for a_shape, b_shape in shapes:
            a = torch.rand(
                a_shape, device=device, dtype=dtype, requires_grad=requires_grad
            )
            b = torch.rand(
                b_shape, device=device, dtype=dtype, requires_grad=requires_grad
            )

            # AVOID a dynamo reset here. We expect guards to have been
            # added that will be violated with the new shape. We should
            # see a recompilation (along with a cache miss).
            res1 = compiled_fn(a, b)
            # A first call should miss in the cache.
            expected_misses += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_miss"], expected_misses
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                expected_guard_misses,
            )

            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], expected_hits
            )
            # Because dynamic shapes are enabled, we expect backwards to be compiled ahead of time
            # So we should see a cache save here
            expected_saves += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_saved"], expected_saves
            )
            if requires_grad:
                res1[0].sum().backward()
                # No extra saves
                self.assertEqual(
                    counters["aot_autograd"]["autograd_cache_saved"], expected_saves
                )

            a2 = a.detach().clone().requires_grad_(requires_grad)
            b2 = b.detach().clone().requires_grad_(requires_grad)
            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).

            # Now clear dynamo and we should see a cache hit
            # This should populate guards to dynamo's cache, so that a subsequent run with a different
            # shape will still trigger a second call to autograd_cache.
            self._clear_dynamo_and_codecache()
            res2 = compiled_fn(a2, b2)
            expected_hits += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_miss"], expected_misses
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_guard_miss"],
                expected_guard_misses,
            )
            # First compile is a regular cache miss, subsequent are guard misses
            expected_guard_misses += 1  # noqa: SIM113
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_hit"], expected_hits
            )
            self.assertEqual(
                counters["aot_autograd"]["autograd_cache_saved"], expected_saves
            )
            self.assertEqual(res1, res2)
            if requires_grad:
                res2[0].sum().backward()
                self.assertEqual(a.grad, a2.grad)

    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_nn_module_with_params_global_constant(self):
        # [FUNCTION DOC] Test that a tensor class attribute used by an nn.Module forward is
        # lifted as a graph input: first compile miss+save, second after clearing dynamo hits
        # with the same result; after replacing the class constant with 3s a third run still
        # hits (two hits) but equals the first result plus one.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 2x2 class level tensor CONSTANT (of 2s) and a
            # random 2x2 parameter; forward returns sin(x) + param + CONSTANT.
            CONSTANT = torch.tensor([[2, 2], [2, 2]])

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the module and registers the random
                # 2x2 parameter param.
                super().__init__()
                self.param = torch.nn.Parameter(torch.randn([2, 2]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns x.sin() plus the parameter plus the current
                # value of the class attribute CONSTANT.
                return x.sin() + self.param + MyMod.CONSTANT

        with torch.no_grad():
            compiled_fn = torch.compile(MyMod(), backend="inductor", fullgraph=True)
            res1 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            self._clear_dynamo_and_codecache()
            res2 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            self.assertEqual(res1, res2)
            # Edit the "constant". We'll get a cache hit,
            # but it should result in a different result when run
            # because MyMod.CONSTANT is an input to the graph
            MyMod.CONSTANT = torch.tensor([[3, 3], [3, 3]])
            self._clear_dynamo_and_codecache()
            res3 = compiled_fn(torch.ones([2, 2]))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertNotEqual(res1, res3)
            self.assertEqual(res1, res3.sub(torch.ones(2, 2)))

    @unittest.skipIf(not HAS_GPU, "requires GPU")
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("always_keep_tensor_constants", True)
    @functorch_config.patch(
        {
            "enable_autograd_cache": True,
            "strict_autograd_cache": True,
            "bundled_autograd_cache": True,
        }
    )
    def test_tensor_constant(self):
        # [FUNCTION DOC] Test that, with bundled autograd cache, strict mode, and
        # always_keep_tensor_constants, a graph containing a tensor constant serializes and is
        # saved: one miss and one save. Skipped without a GPU.
        """Bundled AOT cache serialization with strict_autograd_cache should
        succeed when the compiled graph contains tensor constants."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds an empty int64 index tensor on the device of x and
            # returns x indexed by it (an empty tensor).
            idx = torch.tensor([], dtype=torch.long, device=x.device)
            return x[idx]

        compiled_fn = torch.compile(fn, backend="inductor", fullgraph=True)
        with torch.no_grad():
            compiled_fn(torch.randn(4, device=GPU_TYPE))
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @requires_gpu_and_triton
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_multiple_compile_triton_kernels(self):
        # [FUNCTION DOC] Test that CompiledTritonKernels._cache is empty after both a cache miss
        # compile and a cache hit: runs a compiled function under no_grad, clears dynamo and
        # runs again (one miss, then one hit). Needs a GPU and triton.
        """
        When we cache hit on AOTAutogradCache, we need to still clear
        CompiledTritonKernels after compiling the kernel.
        """
        from torch._inductor.async_compile import CompiledTritonKernels

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def f(x, y):
            # [FUNCTION DOC] HELPER: Model function returning x.sin() + y.
            return x.sin() + y

        x = torch.randn(10, device=device_type)
        y = torch.randn(10, device=device_type)
        with torch.no_grad():
            result = f(x, y)
            self.assertEqual(result, x.sin() + y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(len(CompiledTritonKernels._cache), 0)

        self._clear_dynamo_and_codecache()
        with torch.no_grad():
            result = f(x, y)
            self.assertEqual(result, x.sin() + y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(len(CompiledTritonKernels._cache), 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_dynamic_shapes_different_sizes(self):
        # [FUNCTION DOC] Test (strict mode) that a dynamic-shape function whose forward and
        # backward take different symint inputs but the same symbols can be cached: first
        # compile one miss and save, then after clearing dynamo one hit.
        # The forward and backward function have different symint inputs,
        # but the same underlying symbols
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns (concatenation of x with x, x * y).
            z = x * y
            return (torch.cat((x, x), dim=0), z)

        (x1, y1) = torch.randn(5, requires_grad=True), torch.randn(5)
        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)
        x_compiled, _ = compiled_fn(x1, y1)
        x_compiled.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
        self._clear_dynamo_and_codecache()

        # Run a second time and see it cache hit instead of erroring
        (x2, y2) = torch.randn(5, requires_grad=True), torch.randn(5)
        x_compiled, _ = compiled_fn(x2, y2)
        x_compiled.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

    @unittest.skipIf(not (TEST_CUDA or TEST_XPU), "GPU is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 1.0})
    @functorch_config.patch({"activation_memory_budget_runtime_estimator": "testing"})
    @functorch_config.patch({"saved_tensors_hooks_filtering_mode": "all"})
    def test_saved_tensors_hooks_autograd_cache(self):
        # [FUNCTION DOC] Test that saved-tensors hooks are part of the autograd cache key.
        # Baseline compile is a miss and save. With CPU-offload pack/unpack hooks (traced
        # non-symbolically, user cache hash "cpu_offload") it is a second miss and save. With
        # different hook functions but the same hash string it hits. With multiply-by-2 hooks
        # without a hash it is a third miss and save. Skipped without CUDA/XPU and SM80.
        ctx = torch.autograd.graph.saved_tensors_hooks
        device = torch.device(f"{device_type}:0")

        def pack_cpu(x):
            # [FUNCTION DOC] HELPER: Pack hook moving the saved tensor to CPU.
            return x.to(device="cpu")

        def unpack_cpu(x):
            # [FUNCTION DOC] HELPER: Unpack hook moving the tensor back to the test device.
            return x.to(device=device)

        def pack_cpu2(x):
            # [FUNCTION DOC] HELPER: Second pack hook, same behavior (moves tensor to CPU).
            return x.to(device="cpu")

        def unpack_cpu2(x):
            # [FUNCTION DOC] HELPER: Second unpack hook, same behavior (back to device).
            return x.to(device=device)

        def pack_mul2(x):
            # [FUNCTION DOC] HELPER: Pack hook returning the tensor multiplied by 2.
            return x * 2

        def unpack_mul2(x):
            # [FUNCTION DOC] HELPER: Unpack hook returning the tensor divided by 2.
            return x / 2

        # Can not use custom AutogradFunction here,
        # Cache bypasses AutogradFunction Ctx usage.
        # Can not save in ctx non floating point dtypes.
        # For non-symbolic tracing all dtypes and devices and burned in the graph.

        def fn(x):
            # [FUNCTION DOC] HELPER: Model function: x + 1, sin, cos, relu, exp, then
            # multiplied by 2.
            x = x + 1
            x = x.sin().cos()
            x = x.relu()
            x = x.exp()
            x = 2 * x
            return x

        backend = "inductor"

        def inp_fn():
            # [FUNCTION DOC] HELPER: Returns a 2x3 ones tensor on the test device with
            # requires_grad set and both dims marked dynamic.
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        x = inp_fn()
        fn_compiled = torch.compile(fn, backend=backend, fullgraph=True)
        y = fn_compiled(x)
        y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_cpu,
                unpack_cpu,
                symbolic_tracing=False,
                inp_fn=inp_fn,
                pack_cache_hash="cpu_offload",
                unpack_cache_hash="cpu_offload",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_cpu2,
                unpack_cpu2,
                symbolic_tracing=False,
                inp_fn=inp_fn,
                pack_cache_hash="cpu_offload",
                unpack_cache_hash="cpu_offload",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()

        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(pack_mul2, unpack_mul2, symbolic_tracing=False)
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 3)

    @unittest.skipIf(not (TEST_CUDA or TEST_XPU), "GPU is unavailable")
    @unittest.skipIf(not SM80OrLater, "bfloat16, float8")
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_saved_tensors_hooks_autograd_cache_symbolic(self):
        # [FUNCTION DOC] Test saved-tensor hooks traced symbolically with fx-wrapped fp8
        # pack/unpack functions. Baseline: one miss and save. With the fp8 hooks and a cache
        # hash: second miss and save. Same hooks and same hash again: one hit without new miss.
        # Changed hash string ("..._MISS"): a third miss. Needs CUDA/XPU and SM80.
        def pack_fp8_with_scale(x):
            # [FUNCTION DOC] HELPER: Pack hook delegating to the fx-wrapped fp8 packer.
            return _pack_fp8_with_scale_wrap(x)

        def unpack_fp8_with_scale(packed):
            # [FUNCTION DOC] HELPER: Unpack hook delegating to the fx-wrapped fp8 unpacker.
            return _unpack_fp8_with_scale_wrap(packed)

        ctx = torch.autograd.graph.saved_tensors_hooks

        def fn(x):
            # [FUNCTION DOC] HELPER: Model function: x + 1 then two relus.
            x = x + 1
            # Relu saves bitmask in AutogradContext
            x = x.relu()
            x = x.relu()
            return x

        device = torch.device(f"{device_type}:0")
        backend = "inductor"

        def inp_fn():
            # [FUNCTION DOC] HELPER: Returns a 2x3 ones tensor on the test device with
            # requires_grad set and both dims marked dynamic.
            x = torch.ones(2, 3, device=device, requires_grad=True)
            torch._dynamo.mark_dynamic(x, 0)
            torch._dynamo.mark_dynamic(x, 1)
            return x

        x = inp_fn()
        fn_compiled = torch.compile(fn, backend=backend, fullgraph=True)
        y = fn_compiled(x)
        y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point",
                "fp8_with_scale_dtype_floating_point",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point",
                "fp8_with_scale_dtype_floating_point",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        with ctx(
            *saved_tensors_hooks_to_gm(
                pack_fp8_with_scale,
                unpack_fp8_with_scale,
                "fp8_with_scale_dtype_floating_point_MISS",
                "fp8_with_scale_dtype_floating_point_MISS",
            )
        ):
            x = inp_fn()
            y = fn_compiled(x)
            y.sum().backward()
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)

    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    def test_cache_lazy_backward_for_compiled_autograd(self):
        # [FUNCTION DOC] Test hot loading cache artifacts for a lazily compiled backward used
        # by compiled autograd (fixed cpu, float32, dynamic). Records artifacts inside
        # fresh_cache (one miss, one save, one compiled autograd capture), saves them, clears
        # all caches, loads them, and runs three times: the first iteration hits the AOT
        # autograd cache (inductor hits only when not bundled); later iterations must record no
        # counters at all (no recompiles).
        # KNOWN ISSUE: the string literal after the assignments is a stray expression, not a
        # docstring, so the function has no docstring; skip checks on GPU_TYPE are dead code
        # because device is fixed to cpu.
        device = "cpu"
        dtype = torch.float32
        dynamic = True
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if device == "cuda" and dtype == torch.bfloat16 and not SM80OrLater:
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function: matmul of sin(x) with y.
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)  # noqa: UNSPECIFIED_BACKEND

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            expected_grads = torch.autograd.grad(eager_result.sum(), inputs=(a, b))
            compiled_result = compiled_fn(a, b)
            with torch._dynamo.compiled_autograd._enable(
                torch.compile(dynamic=dynamic)  # noqa: UNSPECIFIED_BACKEND
            ):
                actual_grads = torch.autograd.grad(compiled_result.sum(), inputs=(a, b))
            if hasattr(a, "_dynamo_propagated_dynamic_indices"):
                del a._dynamo_propagated_dynamic_indices
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(expected_grads[0], actual_grads[0])
            self.assertEqual(expected_grads[1], actual_grads[1])
            if functorch_config.bundled_autograd_cache:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            else:
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 3)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
            self.assertEqual(counters["compiled_autograd"]["captures"], 1)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0

        if functorch_config.bundled_autograd_cache:
            self.assertEqual(len(cache_info.inductor_artifacts), 0)
        else:
            self.assertEqual(len(cache_info.inductor_artifacts), 3)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self._clear_all_caches()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit, should not recompile
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            if functorch_config.bundled_autograd_cache:
                self.assertEqual(len(cache_info.inductor_artifacts), 0)
            else:
                self.assertEqual(len(cache_info.inductor_artifacts), 3)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            for i in range(3):
                counters.clear()
                eager_result = fn(a, b)
                expected_grads = torch.autograd.grad(eager_result.sum(), inputs=(a, b))
                compiled_result = compiled_fn(a, b)
                with torch._dynamo.compiled_autograd._enable(
                    torch.compile(dynamic=dynamic)  # noqa: UNSPECIFIED_BACKEND
                ):
                    actual_grads = torch.autograd.grad(
                        compiled_result.sum(), inputs=(a, b)
                    )
                if hasattr(a, "_dynamo_propagated_dynamic_indices"):
                    del a._dynamo_propagated_dynamic_indices
                self.assertEqual(eager_result, compiled_result)
                self.assertEqual(expected_grads[0], actual_grads[0])
                self.assertEqual(expected_grads[1], actual_grads[1])

                if i == 0:
                    # initial compile
                    if functorch_config.bundled_autograd_cache:
                        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                    else:
                        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 3)
                        self.assertEqual(
                            counters["inductor"]["fxgraph_lookup_write_file"], 3
                        )
                    self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                    self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_saved"], 0
                    )
                    self.assertEqual(counters["compiled_autograd"]["captures"], 1)
                else:
                    # no recompiles
                    self.assertFalse(counters)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_basic(self):
        # [FUNCTION DOC] Test the regional inductor backend with bundled autograd cache: the
        # first compile of a function with an annotated region is a miss and save; after
        # clearing dynamo it hits, and both results are equal.
        """
        Basic test for regional inductor with bundled autograd cache.
        Tests that regional inductor compilation results can be cached and hit.
        """
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function: sin(x), then within an annotation region
            # for inductor computes sin(x)*y + 1, then returns sin of that.
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, device="cpu")
        y = torch.randn(10, device="cpu")

        # Compile with regional inductor backend
        compiled_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )

        # First call should miss in cache
        result1 = compiled_fn(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Second call should hit (after clearing dynamo)
        self._clear_dynamo_and_codecache()
        result2 = compiled_fn(x, y)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Results should be the same
        self.assertEqual(result1, result2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_with_backward(self):
        # [FUNCTION DOC] Test regional inductor with a backward pass and bundled cache. Runs
        # forward and backward once, records the miss and save counts (several AOT compiles
        # happen: outer graph plus each region, both must be positive), clears dynamo, reruns
        # with cloned inputs and checks hits are positive, miss and save counts are unchanged,
        # and that outputs and input grads equal the first run.
        """
        Test regional inductor with backward pass and bundled autograd cache.
        Note: Regional inductor triggers multiple AOT autograd compilations:
        - One for the outer graph (with regional inductor backend)
        - One for each marked region (via standalone_compile)
        """
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function: sin(x), then within an annotation region
            # for inductor computes sin(x)*y + 1, then returns sin of that.
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        x = torch.randn(10, requires_grad=True)
        y = torch.randn(10, requires_grad=True)
        x2 = x.detach().clone().requires_grad_(True)
        y2 = y.detach().clone().requires_grad_(True)

        # Compile with regional inductor backend
        compiled_fn = torch.compile(
            fn, backend=aot_eager_regional_inductor(), fullgraph=True
        )

        # First call: AOT autograd compiles the outer graph (1 miss)
        # Regional inductor then compiles the marked region (1 more miss)
        result1 = compiled_fn(x, y)
        result1.sum().backward()

        # We expect 2 cache misses: outer graph + marked region
        initial_misses = counters["aot_autograd"]["autograd_cache_miss"]
        initial_saves = counters["aot_autograd"]["autograd_cache_saved"]
        self.assertGreater(initial_misses, 0)
        self.assertGreater(initial_saves, 0)

        # Second call should hit (after clearing dynamo)
        self._clear_dynamo_and_codecache()
        result2 = compiled_fn(x2, y2)
        result2.sum().backward()

        # Should have cache hits now
        final_hits = counters["aot_autograd"]["autograd_cache_hit"]
        self.assertGreater(final_hits, 0)

        # Cache misses and saves should not increase
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_miss"], initial_misses
        )
        self.assertEqual(
            counters["aot_autograd"]["autograd_cache_saved"], initial_saves
        )

        # Results and gradients should be the same
        self.assertEqual(result1, result2)
        self.assertEqual(x.grad, x2.grad)
        self.assertEqual(y.grad, y2.grad)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"bundled_autograd_cache": True})
    def test_regional_inductor_cache_miss_on_change(self):
        # [FUNCTION DOC] Test that two functions differing only in a constant (+1 vs +2 inside
        # the marked region) do not share a cache entry under regional inductor: the second
        # compile increases miss and save counts, has no hits, and the results differ.
        """
        Test that changing the function causes a cache miss with regional inductor.
        Regional inductor creates multiple AOT compilations, so we track
        the change in cache misses rather than absolute counts.
        """
        import torch.fx.traceback as fx_traceback

        def fn1(x, y):
            # [FUNCTION DOC] HELPER: Model function with a region computing sin(x)*y + 1, then
            # sin of the result.
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        def fn2(x, y):
            # [FUNCTION DOC] HELPER: Same as fn1 but the region adds 2 instead of 1.
            sin = torch.sin(x)
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 2  # Changed from +1 to +2
            return torch.sin(add)

        x = torch.randn(10)
        y = torch.randn(10)

        # Compile first function
        compiled_fn1 = torch.compile(
            fn1, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        result1 = compiled_fn1(x, y)
        first_misses = counters["aot_autograd"]["autograd_cache_miss"]
        first_saves = counters["aot_autograd"]["autograd_cache_saved"]
        self.assertGreater(first_misses, 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertGreater(first_saves, 0)

        # Compile second function (different graph)
        self._clear_dynamo_and_codecache()
        compiled_fn2 = torch.compile(
            fn2, backend=aot_eager_regional_inductor(), fullgraph=True
        )
        result2 = compiled_fn2(x, y)
        # Should miss because graph is different (more misses than before)
        self.assertGreater(
            counters["aot_autograd"]["autograd_cache_miss"], first_misses
        )
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertGreater(
            counters["aot_autograd"]["autograd_cache_saved"], first_saves
        )

        # Results should be different
        self.assertNotEqual(result1, result2)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_custom_runtime_estimator_cache_hit(self):
        # [FUNCTION DOC] Test that a CustomRuntimeEstimator returning a valid uuid as the
        # activation memory budget runtime estimator is cacheable: first compile one miss and
        # save with no bypass; after clearing dynamo one hit with no new miss or save and no
        # bypass. Uses fresh_cache and a 0.5 memory budget.
        """Test that CustomRuntimeEstimator with valid uuid() results in cache hit."""

        class MyCustomEstimator(CustomRuntimeEstimator):
            # [CLASS DOC] HELPER: Estimator returning a constant cost of 1.0 for every node and
            # a fixed uuid string, so it can be part of the cache key.
            def __call__(self, node) -> float:
                # [FUNCTION DOC] HELPER: Returns the constant runtime estimate 1.0 for any node.
                return 1.0

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns the fixed identifier "my_custom_estimator_v1".
                return "my_custom_estimator_v1"

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning matmul(x, y) + 1.
            return torch.mm(x, y) + 1

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": MyCustomEstimator()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")

                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

                self._clear_dynamo_and_codecache()

                x2 = torch.randn(10, 10, requires_grad=True)
                y2 = torch.randn(10, 10, requires_grad=True)
                result2 = compiled_fn(x2, y2)
                result2.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_custom_estimator_bypasses_cache(self):
        # [FUNCTION DOC] Test that a runtime estimator without a usable uuid makes the autograd
        # cache bypass: first with a plain function (no uuid method), then with a
        # CustomRuntimeEstimator whose uuid returns None. In both cases miss, hit and saved stay
        # at 0 and the bypass counter is positive.
        """Test that cache is bypassed when custom estimator lacks proper uuid."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning matmul(x, y) + 1.
            return torch.mm(x, y) + 1

        # Test 1: Raw callable without uuid() method bypasses cache
        def raw_estimator(node) -> float:
            # [FUNCTION DOC] HELPER: Plain-function estimator returning 1.0 for any node; has
            # no uuid method.
            return 1.0

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": raw_estimator}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)
                self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

        counters.clear()
        self._clear_dynamo_and_codecache()

        # Test 2: CustomRuntimeEstimator with uuid() returning None bypasses cache
        class EstimatorWithNullUuid(CustomRuntimeEstimator):
            # [CLASS DOC] HELPER: Estimator returning 1.0 for every node whose uuid is None,
            # which makes it uncacheable.
            def __call__(self, node) -> float:
                # [FUNCTION DOC] HELPER: Returns the constant runtime estimate 1.0.
                return 1.0

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns None, signalling no identifier is available.
                return None

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorWithNullUuid()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 0)
                self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    def test_pre_grad_passes_inplace_cpu(self):
        # [FUNCTION DOC] Regression test that pre_grad_passes mutates the given GraphModule in
        # place for CPU inference (the identity removal pass used to return a new module).
        # Traces a linear+identity+relu model, runs pre_grad_passes with pattern_matcher and
        # freezing enabled under no_grad, and asserts the returned object is the same object
        # as the input.
        """
        `remove_identity` was an out-place operation previously, and it is called
        for CPU inference inside `pre_grad_passes`. So, it broke the in-place semantics
        of `pre_grad_passes` for CPU inference. This test ensures `pre_grad_passes` is
        in-place in this case.
        """

        from torch._inductor.fx_passes.pre_grad import pre_grad_passes

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Small module: a 4x4 Linear followed by an Identity
            # layer, then relu.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear 4x4 layer and an identity layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.identity = torch.nn.Identity()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, identity, then relu and returns it.
                x = self.linear(x)
                x = self.identity(x)
                return torch.relu(x)

        x = torch.randn(2, 4)
        gm = torch.fx.symbolic_trace(Model().eval())

        with inductor_config.patch(
            pattern_matcher=True,
            freezing=True,
        ):
            with torch.no_grad():
                gm_after = pre_grad_passes(gm, (x,))

        self.assertTrue(id(gm_after) == id(gm))

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"activation_memory_budget": 0.5})
    def test_different_custom_estimator_uuids_cause_cache_miss(self):
        # [FUNCTION DOC] Test that two estimators with different uuid strings give different
        # cache keys: compiling with the first is one miss and save; after clearing dynamo,
        # compiling with the second is a second miss and save, with no hits and no bypass.
        """Test that different uuid values cause cache miss."""

        class EstimatorV1(CustomRuntimeEstimator):
            # [CLASS DOC] HELPER: Estimator with constant cost 1.0 and uuid "estimator_v1".
            def __call__(self, node) -> float:
                # [FUNCTION DOC] HELPER: Returns the constant runtime estimate 1.0.
                return 1.0

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns the identifier "estimator_v1".
                return "estimator_v1"

        class EstimatorV2(CustomRuntimeEstimator):
            # [CLASS DOC] HELPER: Estimator with constant cost 1.0 and uuid "estimator_v2".
            def __call__(self, node) -> float:
                # [FUNCTION DOC] HELPER: Returns the constant runtime estimate 1.0.
                return 1.0

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns the identifier "estimator_v2".
                return "estimator_v2"

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning matmul(x, y) + 1.
            return torch.mm(x, y) + 1

        with fresh_cache():
            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorV1()}
            ):
                compiled_fn = torch.compile(fn, backend="inductor")
                x = torch.randn(10, 10, requires_grad=True)
                y = torch.randn(10, 10, requires_grad=True)
                result = compiled_fn(x, y)
                result.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

            self._clear_dynamo_and_codecache()

            with functorch_config.patch(
                {"activation_memory_budget_runtime_estimator": EstimatorV2()}
            ):
                compiled_fn2 = torch.compile(fn, backend="inductor")
                x2 = torch.randn(10, 10, requires_grad=True)
                y2 = torch.randn(10, 10, requires_grad=True)
                result2 = compiled_fn2(x2, y2)
                result2.sum().backward()

                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"strict_autograd_cache": True})
    def test_output_views_input_dynamic(self):
        # [FUNCTION DOC] Test (strict mode) that a dynamic-shape module returning an output that
        # is a view of its input is cached and still correct: first call one miss and save;
        # after clearing dynamo, a call with different dynamic sizes hits. Output equals eager
        # and the view shares storage with the input.
        class OutputViewsInput(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 64x64 Linear whose forward returns the relu of
            # the linear output and a flattened (batch*seq, hidden) view of x.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 64-to-64 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reads batch, seq, hidden from x.shape, applies linear
                # then relu, and returns that and a view of x of shape (batch*seq, hidden).
                batch, seq, hidden = x.shape
                y = self.linear(x)
                y = torch.relu(y)
                x_view = x.view(batch * seq, hidden)
                return y, x_view

        model = OutputViewsInput()
        x = torch.randn(2, 16, 64)

        torch._dynamo.mark_dynamic(x, 0)
        torch._dynamo.mark_dynamic(x, 1)

        compiled = torch.compile(model, backend="inductor", dynamic=True)

        # First call - should miss
        y, x_view = compiled(x)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Reset dynamo but keep cache
        self._clear_dynamo_and_codecache()

        # Use different dynamic shape
        x2 = torch.randn(4, 8, 64)
        torch._dynamo.mark_dynamic(x2, 0)
        torch._dynamo.mark_dynamic(x2, 1)

        # Second call with different shape - should hit cache
        y2, x_view2 = compiled(x2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

        # Verify correctness
        eager_model = copy.deepcopy(model)
        expected_y, expected_view = eager_model(x2)
        self.assertEqual(y2, expected_y)
        self.assertEqual(x_view2, expected_view)
        # Verify the view shares storage with input (critical for view correctness)
        self.assertEqual(
            x_view2.untyped_storage().data_ptr(), x2.untyped_storage().data_ptr()
        )

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_cache_hit_logs_aot_graphs(self):
        # [FUNCTION DOC] Test that the aot_graphs log prints on a cache hit as well as on a miss
        # (issue 149060). Captures logs of the graph_capture logger on the first call (expects
        # "Forward graph") and, after clearing dynamo, logs of the aot_autograd_result logger on
        # a hit (expects "Forward graph (from cache)").
        """
        Verify that TORCH_LOGS=aot_graphs prints graphs on cache hit, not just cache miss.
        See https://github.com/pytorch/pytorch/issues/149060
        """
        from torch.testing._internal.logging_utils import logs_to_string

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning (x * 2, y @ y).
            return (x * 2, y @ y)

        a = torch.rand(25)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn, backend="inductor")

        # First call: cache miss — aot_graphs should be logged
        log_stream, ctx = logs_to_string(
            "torch._functorch._aot_autograd.graph_capture", "aot_graphs"
        )
        with ctx():
            compiled_fn(a, b)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
        self.assertIn("Forward graph", log_stream.getvalue())

        # Reset dynamo so guards force recompilation (but keep disk cache)
        self._clear_dynamo_and_codecache()

        # Second call: cache hit — aot_graphs should ALSO be logged.
        # On cache hit the log comes from aot_autograd_result, not graph_capture.
        log_stream2, ctx2 = logs_to_string(
            "torch._functorch._aot_autograd.aot_autograd_result", "aot_graphs"
        )
        with ctx2():
            compiled_fn(a, b)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertIn("Forward graph (from cache)", log_stream2.getvalue())

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    @parametrize(
        "pre_grad_pass_timing,pre_grad_custom_pass,expect_pre_grad_call_count",
        [
            subtest(
                ("early", custom_pre_grad_pass_remove_ident_muls, (1, 2)),
                name="early_with_uuid",
            ),
            subtest(
                ("late", custom_pre_grad_pass_remove_ident_muls, (1, 1)),
                name="late_with_uuid",
            ),
            subtest(
                ("default", custom_pre_grad_pass_remove_ident_muls, (1, 1)),
                name="default_with_uuid",
            ),
            subtest(
                ("early", custom_pre_grad_pass_remove_ident_muls_wo_uuid, (1, 2)),
                name="early_without_uuid",
            ),
            # late_without_uuid will raise an exception and is tested separately in
            # test_pre_grad_pass_late_timing_without_uuid_raises
            subtest(
                ("default", custom_pre_grad_pass_remove_ident_muls_wo_uuid, (1, 2)),
                name="default_without_uuid",
            ),
        ],
    )
    @parametrize(
        "use_module",
        [
            subtest(False, name="function"),
            subtest(True, name="module"),
        ],
    )
    def test_pre_grad_passes_timing(
        self,
        pre_grad_pass_timing: Literal["early", "late", "default"],
        pre_grad_custom_pass: CustomGraphPassType,
        expect_pre_grad_call_count: tuple[int, int],
        use_module: bool,
    ):
        # [FUNCTION DOC] Parametrized test of how the custom pre-grad pass interacts with the
        # cache, for each timing (early, late, default), pass with or without uuid, and
        # function vs module target. Wraps run_pre_grad_passes to count calls and record graph
        # text, and compile_fx_forward to record what inductor receives. Runs under no_grad.
        # First compile: one miss, expected first call count, the pre-grad graph still has the
        # multiplication by 1 and the graph given to inductor does not. After resetting
        # dynamo: one hit, expected second call count (how often pre-grad passes ran on the
        # hit) and equal results.
        from torch._inductor.compile_fx import compile_fx_forward, run_pre_grad_passes

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning 1 * x + y.
            return 1 * x + y

        class LinearModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 10x10 Linear whose forward multiplies the linear
            # output by 1.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 10-to-10 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns 1 times the output of the linear layer.
                return 1 * self.linear(x)

        pre_grad_call_count = 0
        pre_grad_input_graphs: list[str] = []
        inductor_input_graphs: list[str] = []

        def wrap_run_pre_grad_passes(
            model: GraphModule, example_inputs: Sequence[InputType]
        ) -> GraphModule:
            # [FUNCTION DOC] HELPER: Counts the call, records the printed graph text of the
            # model, runs the real pre-grad passes on it and returns the model.
            nonlocal pre_grad_call_count
            pre_grad_call_count += 1
            pre_grad_input_graphs.append(model.print_readable(print_output=False))
            run_pre_grad_passes(model, example_inputs)
            return model

        @torch.utils._typing_utils.copy_func_params(compile_fx_forward)
        def wrap_compile_fx_forward(gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Records the printed graph text of gm reaching inductor and
            # delegates to the real compile_fx_forward.
            inductor_input_graphs.append(gm.print_readable(print_output=False))
            return compile_fx_forward(gm, *args, **kwargs)

        if use_module:
            target = LinearModel()
            inputs = (torch.randn(10),)
        else:
            target = fn
            inputs = (torch.randn(10), torch.randn(10))

        with (
            unittest.mock.patch(
                "torch._inductor.compile_fx.run_pre_grad_passes",
                wrap_run_pre_grad_passes,
            ),
            unittest.mock.patch(
                "torch._inductor.compile_fx.compile_fx_forward",
                wrap_compile_fx_forward,
            ),
            inductor_config.patch("pre_grad_pass_timing", pre_grad_pass_timing),
            inductor_config.patch("pre_grad_custom_pass", pre_grad_custom_pass),
            # Autograd cache doesn't save entries for modules with trainable
            # parameters, so disable grad to get consistent cache hit counts
            # across the function and module variants.
            torch.no_grad(),
        ):
            self._clear_all_caches()

            # First compilation - expect cache miss.
            compiled = torch.compile(target)  # noqa: UNSPECIFIED_BACKEND
            result1 = compiled(*inputs)

            # Assert cache miss.
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Assert #invocation of pre-grad passes.
            self.assertEqual(pre_grad_call_count, expect_pre_grad_call_count[0])

            # The pre-grad graph should contain the identity mul (= 1 * ...)
            # before the custom pass removes it.
            self.assertEqual(len(pre_grad_input_graphs), 1)
            self.assertIn("= 1 * ", pre_grad_input_graphs[0])
            self.assertEqual(len(inductor_input_graphs), 1)
            self.assertNotIn("= 1 * ", inductor_input_graphs[0])

            torch._dynamo.reset()

            # Second compilation - expect cache hit.
            compiled2 = torch.compile(target)  # noqa: UNSPECIFIED_BACKEND
            result2 = compiled2(*inputs)

            # Assert cache hit.
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            # Assert #invocation of pre-grad passes.
            self.assertEqual(pre_grad_call_count, expect_pre_grad_call_count[1])

            self.assertEqual(result1, result2)

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    @inductor_config.patch("pre_grad_pass_timing", "late")
    def test_pre_grad_pass_late_timing_without_uuid_raises(self):
        # [FUNCTION DOC] Test that with pre_grad_pass_timing late, a pre-grad custom pass whose
        # uuid is None makes compilation raise a RuntimeError matching "pre_grad_custom_pass ...
        # NoUuidPass must implement uuid".
        """
        Explicitly setting late timing with a pass that has no UUID should
        raise a RuntimeError.
        """

        class NoUuidPass(CustomGraphPass):
            # [CLASS DOC] HELPER: No-op graph pass that has no uuid (returns None).
            def __call__(self, g: torch.fx.Graph) -> None:
                # [FUNCTION DOC] HELPER: Does nothing to the graph.
                pass

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns None (no identifier).
                return None

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning x + y.
            return x + y

        x = torch.randn(10)
        y = torch.randn(10)

        with inductor_config.patch("pre_grad_custom_pass", NoUuidPass()):
            self._clear_all_caches()
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            with self.assertRaisesRegex(
                RuntimeError, "pre_grad_custom_pass.*NoUuidPass must implement uuid"
            ):
                compiled_fn(x, y)

    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    def test_pre_grad_pass_default_timing_without_uuid_warns(self):
        # [FUNCTION DOC] Test that with default timing, a pass without uuid logs a warning
        # ("NoUuidPass does not implement uuid()") on logger torch._inductor.codecache, exactly
        # once for the first compile, and none on a second compile of the same pass class.
        # Clears the module level warned set first.
        """
        Default timing with a pass that has no UUID should log a warning about
        the pre-grad pass cache being bypassed, only once per pass class.
        """
        from torch._inductor.codecache import _warned_pre_grad_pass_missing_uuid

        class NoUuidPass(CustomGraphPass):
            # [CLASS DOC] HELPER: No-op graph pass that has no uuid (returns None).
            def __call__(self, g: torch.fx.Graph) -> None:
                # [FUNCTION DOC] HELPER: Does nothing to the graph.
                pass

            def uuid(self):
                # [FUNCTION DOC] HELPER: Returns None (no identifier).
                return None

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Model function returning x + y.
            return x + y

        x = torch.randn(10)
        y = torch.randn(10)

        with (
            inductor_config.patch("pre_grad_custom_pass", NoUuidPass()),
            inductor_config.patch("pre_grad_pass_timing", "default"),
        ):
            _warned_pre_grad_pass_missing_uuid.clear()
            self._clear_all_caches()

            # First compilation — should warn.
            compiled_fn = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            with self.assertLogs(
                "torch._inductor.codecache", level="WARNING"
            ) as log_cm:
                compiled_fn(x, y)
            uuid_warnings = [
                m for m in log_cm.output if "NoUuidPass does not implement uuid()" in m
            ]
            self.assertEqual(len(uuid_warnings), 1)

            # Second compilation — same pass class, should not warn again.
            torch._dynamo.reset()
            compiled_fn2 = torch.compile(fn)  # noqa: UNSPECIFIED_BACKEND
            with self.assertNoLogs("torch._inductor.codecache", level="WARNING"):
                compiled_fn2(x, y)

    @parametrize(
        "pre_grad_pass_timing,has_uuid,expect_miss_on_different_uuid",
        [
            subtest(("early", True, False), name="early_with_uuid"),
            subtest(("late", True, True), name="late_with_uuid"),
            subtest(("default", True, True), name="default_with_uuid"),
            subtest(("early", False, False), name="early_without_uuid"),
            # late + no uuid raises RuntimeError, tested separately
            subtest(("default", False, False), name="default_without_uuid"),
        ],
    )
    def test_cache_hit_across_processes_pre_grad_custom_pass(
        self, pre_grad_pass_timing, has_uuid, expect_miss_on_different_uuid
    ):
        # [FUNCTION DOC] Parametrized test of cross-process cache behavior with a custom
        # pre-grad pass. Runs a generated script in a subprocess three times sharing one
        # temporary TORCHINDUCTOR_CACHE_DIR; the script compiles a function using a pass that
        # removes 1 * x and prints the aot_autograd counters as JSON. Run one is a miss, run two
        # (same uuid) is a hit, run three (different uuid) is a miss only when
        # expect_miss_on_different_uuid, else a hit.
        """
        Verify cache behavior across processes for different pre-grad pass
        timing and UUID configurations.
        """
        import subprocess
        import sys
        import tempfile
        import textwrap

        with tempfile.TemporaryDirectory() as cache_dir:
            script_template = textwrap.dedent(
                """
                import json
                import operator
                import torch
                import torch._dynamo
                from torch._dynamo.utils import counters
                from torch._inductor import config as inductor_config
                from torch._inductor.custom_graph_pass import CustomGraphPass

                inductor_config.fx_graph_cache = True
                inductor_config.fx_graph_remote_cache = False
                inductor_config.pre_grad_pass_timing = "{pre_grad_pass_timing}"
                torch._dynamo.reset()

                class TestPreGradPass(CustomGraphPass):
                    def __call__(self, g):
                        changed = False
                        for n in g.nodes:
                            if n.op == "call_function" and n.target is operator.mul:
                                lhs, rhs = n.args
                                if lhs == 1:
                                    n.replace_all_uses_with(rhs)
                                    g.erase_node(n)
                                    changed = True
                        if not changed:
                            raise RuntimeError(
                                "Custom pass did not change the graph. "
                                "All test cases expect the pass to modify the graph."
                            )

                    def uuid(self):
                        return {pass_uuid}

                inductor_config.pre_grad_custom_pass = TestPreGradPass()

                def fn(x, y):
                    return 1 * x + y

                compiled_fn = torch.compile(fn)
                x = torch.randn(10)
                y = torch.randn(10)
                compiled_fn(x, y)

                print(json.dumps(dict(counters["aot_autograd"])))
                """
            )

            env = {**os.environ, "TORCHINDUCTOR_CACHE_DIR": cache_dir}

            def run_script(pass_uuid):
                # [FUNCTION DOC] HELPER: Formats the script template with the timing and the
                # repr of pass_uuid, runs it with the current Python interpreter and the cache
                # dir env, asserts exit code 0, and returns the counters parsed from the last
                # line of stdout.
                script = script_template.format(
                    pre_grad_pass_timing=pre_grad_pass_timing,
                    pass_uuid=repr(pass_uuid),
                )
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    env=env,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                import json

                return json.loads(result.stdout.splitlines()[-1])

            uuid_a = "pass_uuid_A" if has_uuid else None
            uuid_b = "pass_uuid_B" if has_uuid else None

            # First run - expect cache miss
            c1 = run_script(uuid_a)
            self.assertEqual(c1.get("autograd_cache_miss", 0), 1)
            self.assertEqual(c1.get("autograd_cache_hit", 0), 0)

            # Second run with same pass - expect cache hit
            c2 = run_script(uuid_a)
            self.assertEqual(c2.get("autograd_cache_miss", 0), 0)
            self.assertEqual(c2.get("autograd_cache_hit", 0), 1)

            # Third run with different pass UUID - miss only when UUID is
            # part of the cache key (late timing with UUID)
            c3 = run_script(uuid_b)
            if expect_miss_on_different_uuid:
                self.assertEqual(c3.get("autograd_cache_miss", 0), 1)
                self.assertEqual(c3.get("autograd_cache_hit", 0), 0)
            else:
                self.assertEqual(c3.get("autograd_cache_miss", 0), 0)
                self.assertEqual(c3.get("autograd_cache_hit", 0), 1)

    @unittest.skipIf(
        TEST_WITH_ASAN or IS_LINUX or TEST_WITH_SLOW,
        "https://github.com/pytorch/pytorch/issues/179689",
    )
    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_dtensor_cache_hit(self):
        # [FUNCTION DOC] Test that a DTensor input on a fake 2-rank CPU mesh gives an autograd
        # cache miss then a hit after clearing dynamo.
        # KNOWN ISSUE: the skip condition includes IS_LINUX, so the test is skipped on Linux
        # (see issue 179689 in the skip reason), the platform it targets.
        """
        Test that DTensor produces cache hits on second compile.

        This follows the standard AOTAutograd cache test pattern: compile once
        (cache miss), reset dynamo, compile again with equivalent input (cache hit).
        """
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate

        with _fake_process_group():
            mesh = init_device_mesh("cpu", (2,))

            def fn(x):
                # [FUNCTION DOC] HELPER: Model function returning x.sin().
                return x.sin()

            compiled_fn = torch.compile(fn, backend="inductor")

            # First call - cache miss
            dtensor1 = DTensor.from_local(torch.zeros(4, 4), mesh, [Replicate()])
            compiled_fn(dtensor1)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Reset dynamo but keep the cache
            self._clear_dynamo_and_codecache()

            # Second call with equivalent DTensor - should hit cache
            dtensor2 = DTensor.from_local(torch.zeros(4, 4), mesh, [Replicate()])
            compiled_fn(dtensor2)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_non_tuple_return_cache_key(self):
        # [FUNCTION DOC] Test that the standalone compile_fx.autograd_cache_key rejects a graph
        # whose output is not a tuple: traces a function returning a single tensor with fake
        # tensors, asserts graph_returns_tuple is false, expects NotImplementedError ("don't
        # return a tuple") and then checks compile_fx on the same graph still completes.
        """autograd_cache_key does not support graphs that don't return a tuple,
        since compile_fx mutates them via make_graph_return_tuple."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def fn(x):
            # [FUNCTION DOC] HELPER: Model function returning x.sin().
            return x.sin()

        inp = torch.randn(10)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        fake_inp = fake_mode.from_tensor(inp)
        gm = make_fx(fn, tracing_mode="fake")(fake_inp)
        self.assertFalse(compile_fx.graph_returns_tuple(gm))
        with self.assertRaisesRegex(NotImplementedError, "don't return a tuple"):
            compile_fx.autograd_cache_key(gm, [fake_inp], ignore_shape_env=False)
        compile_fx.compile_fx(gm, [fake_inp])

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_flatten_graph_inputs_cache_key(self):
        # [FUNCTION DOC] Test that the standalone autograd_cache_key rejects nested container
        # (list) inputs with NotImplementedError ("nested container inputs"), while compile_fx
        # itself accepts the graph taking a list of two fake tensors.
        """autograd_cache_key does not support nested container inputs, since
        compile_fx wraps them via flatten_graph_inputs."""
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        def fn(xs):
            # [FUNCTION DOC] HELPER: Takes a list of two tensors and returns a one element tuple
            # holding their sum.
            return (xs[0] + xs[1],)

        inp = torch.randn(10)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        fake_x = fake_mode.from_tensor(inp)
        fake_y = fake_mode.from_tensor(inp)
        gm = make_fx(fn, tracing_mode="fake")([fake_x, fake_y])
        with self.assertRaisesRegex(NotImplementedError, "nested container inputs"):
            compile_fx.autograd_cache_key(
                gm, [[fake_x, fake_y]], ignore_shape_env=False
            )
        compile_fx.compile_fx(gm, [[fake_x, fake_y]])

    @unittest.skipIf(not HAS_GPU, "requires accelerator")
    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    def test_autocast_cache_distinguishes_dtype(self):
        # [FUNCTION DOC] Test that the autocast dtype is part of the cache key: a small
        # two-linear model compiled under bfloat16 autocast misses, under float16 autocast
        # misses again, then bfloat16 and float16 again each hit. Checks output dtypes.
        # Needs an accelerator.
        """
        Ensure AOTAutograd cache key includes autocast dtype so that
        graphs compiled under different autocast dtypes are not reused.
        """

        class TwoLinear(torch.nn.Module):
            # [CLASS DOC] HELPER: Two 64x64 linear layers with a gelu in between.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two 64-to-64 linear layers fc1 and fc2.
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 64)
                self.fc2 = torch.nn.Linear(64, 64)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc1, gelu, then fc2.
                return self.fc2(torch.nn.functional.gelu(self.fc1(x)))

        with fresh_cache():
            model = TwoLinear().to(GPU_TYPE).eval()
            x = torch.randn(2, 64, device=GPU_TYPE)

            # Run 1: compile under bfloat16 autocast
            torch._dynamo.reset()
            compiled1 = torch.compile(model, backend="inductor")
            with torch.no_grad(), torch.amp.autocast(GPU_TYPE, dtype=torch.bfloat16):
                out_bf16 = compiled1(x)
            self.assertEqual(out_bf16.dtype, torch.bfloat16)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Run 2: compile under float16 autocast
            counters.clear()
            torch._dynamo.reset()
            compiled2 = torch.compile(model, backend="inductor")
            with torch.no_grad(), torch.amp.autocast(GPU_TYPE, dtype=torch.float16):
                out_fp16 = compiled2(x)
            self.assertEqual(out_fp16.dtype, torch.float16)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            # Run 3: compile under bfloat16 again
            counters.clear()
            torch._dynamo.reset()
            compiled3 = torch.compile(model, backend="inductor")
            with torch.no_grad(), torch.amp.autocast(GPU_TYPE, dtype=torch.bfloat16):
                out_bf16_2 = compiled3(x)
            self.assertEqual(out_bf16_2.dtype, torch.bfloat16)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            # Run 4: compile under float16 again
            counters.clear()
            torch._dynamo.reset()
            compiled4 = torch.compile(model, backend="inductor")
            with torch.no_grad(), torch.amp.autocast(GPU_TYPE, dtype=torch.float16):
                out_fp16_2 = compiled4(x)
            self.assertEqual(out_fp16_2.dtype, torch.float16)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @unittest.skipIf(not HAS_GPU, "requires accelerator")
    @functorch_config.patch({"enable_autograd_cache": True})
    @inductor_config.patch("fx_graph_cache", True)
    @inductor_config.patch("fx_graph_remote_cache", False)
    def test_autocast_in_graph_is_cacheable(self):
        # [FUNCTION DOC] Test that an autocast context entered inside the compiled function is
        # cacheable and its dtype is part of the key: bfloat16 run misses with no bypass,
        # float16 run misses again (no reuse), then bfloat16 and float16 each hit, with output
        # dtypes and values equal to the first runs. Needs an accelerator.
        """
        An autocast context manager inside the compiled region is traced into
        _enter_autocast/_exit_autocast nodes. Those are cacheable, and the dtype
        they carry is part of the key, so graphs compiled under different dtypes
        must not be reused for each other.
        """

        def fn(x, w, dtype):
            # [FUNCTION DOC] HELPER: Model function computing gelu(x @ w) under an autocast
            # context of the given dtype on the GPU device type.
            with torch.amp.autocast(GPU_TYPE, dtype=dtype):
                return torch.nn.functional.gelu(x @ w)

        def run(dtype):
            # [FUNCTION DOC] HELPER: Resets dynamo and counters, compiles fn, runs it under
            # no_grad with the enclosing x and w and the given dtype, and returns the output and
            # a tuple (miss, hit, bypass) of the autograd cache counters.
            torch._dynamo.reset()
            counters.clear()
            compiled = torch.compile(fn, backend="inductor")
            with torch.no_grad():
                out = compiled(x, w, dtype)
            return out, (
                counters["aot_autograd"]["autograd_cache_miss"],
                counters["aot_autograd"]["autograd_cache_hit"],
                counters["aot_autograd"]["autograd_cache_bypass"],
            )

        with fresh_cache():
            x = torch.randn(8, 64, device=GPU_TYPE)
            w = torch.randn(64, 64, device=GPU_TYPE)

            out_bf16, (miss, hit, bypass) = run(torch.bfloat16)
            self.assertEqual(out_bf16.dtype, torch.bfloat16)
            # Previously this bypassed the cache entirely.
            self.assertEqual(bypass, 0)
            self.assertEqual((miss, hit), (1, 0))

            # A different dtype must not reuse the bfloat16 entry.
            out_fp16, (miss, hit, bypass) = run(torch.float16)
            self.assertEqual(out_fp16.dtype, torch.float16)
            self.assertEqual(bypass, 0)
            self.assertEqual((miss, hit), (1, 0))

            # Recompiling the original dtype hits.
            out_bf16_2, (miss, hit, bypass) = run(torch.bfloat16)
            self.assertEqual(out_bf16_2.dtype, torch.bfloat16)
            self.assertEqual((miss, hit), (0, 1))
            self.assertEqual(out_bf16_2, out_bf16)

            # ...and so does the second dtype, with its own entry.
            out_fp16_2, (miss, hit, bypass) = run(torch.float16)
            self.assertEqual(out_fp16_2.dtype, torch.float16)
            self.assertEqual((miss, hit), (0, 1))
            self.assertEqual(out_fp16_2, out_fp16)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_region_activation_memory_budget_causes_cache_miss(self):
        # [FUNCTION DOC] Test that changing the budget given to region_activation_memory_budget
        # changes the cache key: budget 0.3 is a miss, budget 0.8 is a second miss, and 0.3
        # again hits (two misses, one hit).
        """Changing the budget set via torch.autograd.graph.region_activation_memory_budget
        invalidates the AOTAutograd cache."""

        def make_fn(budget):
            # [FUNCTION DOC] HELPER: Returns a model function that computes relu(mm(x, y) + 1)
            # inside a region_activation_memory_budget context set to budget.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Runs the matmul-add-relu inside the budget context.
                with torch.autograd.graph.region_activation_memory_budget(budget):
                    return (torch.mm(x, y) + 1).relu()

            return fn

        with fresh_cache():
            compiled_fn1 = torch.compile(make_fn(0.3), backend="inductor")
            x1 = torch.randn(10, 10, requires_grad=True)
            y1 = torch.randn(10, 10, requires_grad=True)
            compiled_fn1(x1, y1).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            self._clear_dynamo_and_codecache()

            compiled_fn2 = torch.compile(make_fn(0.8), backend="inductor")
            x2 = torch.randn(10, 10, requires_grad=True)
            y2 = torch.randn(10, 10, requires_grad=True)
            compiled_fn2(x2, y2).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            self._clear_dynamo_and_codecache()

            # Same budget as the first compile -> cache hit.
            compiled_fn3 = torch.compile(make_fn(0.3), backend="inductor")
            x3 = torch.randn(10, 10, requires_grad=True)
            y3 = torch.randn(10, 10, requires_grad=True)
            compiled_fn3(x3, y3).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_region_activation_memory_budget_graph_break_cache(self):
        # [FUNCTION DOC] Test that with a graph break between two budget regions, only the graph
        # whose budget changed misses. Budgets (0.3, 0.5): two misses; then (0.3, 0.8): one hit
        # and one new miss; then (0.7, 0.8): another hit and miss.
        """With graph breaks, changing one graph's budget causes a miss for
        that graph but a hit for the unchanged graph."""

        def make_fn(budget_a, budget_b):
            # [FUNCTION DOC] HELPER: Returns a function that applies relu(x + 1) under budget
            # budget_a, forces a graph break, then returns relu(x * 2) under budget_b.
            def fn(x):
                # [FUNCTION DOC] HELPER: The two-region function with a graph break between.
                with torch.autograd.graph.region_activation_memory_budget(budget_a):
                    x = (x + 1).relu()
                torch._dynamo.graph_break()
                with torch.autograd.graph.region_activation_memory_budget(budget_b):
                    return (x * 2).relu()

            return fn

        with fresh_cache():
            # First run: (0.3, 0.5) -> 2 misses (one per graph)
            compiled = torch.compile(make_fn(0.3, 0.5), backend="inductor")
            x = torch.randn(10, 10, requires_grad=True)
            compiled(x).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            self._clear_dynamo_and_codecache()

            # Change only graph B: (0.3, 0.8) -> 1 hit (A), 1 miss (B)
            compiled = torch.compile(make_fn(0.3, 0.8), backend="inductor")
            x = torch.randn(10, 10, requires_grad=True)
            compiled(x).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 3)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            self._clear_dynamo_and_codecache()

            # Change only graph A: (0.7, 0.8) -> 1 miss (A), 1 hit (B)
            compiled = torch.compile(make_fn(0.7, 0.8), backend="inductor")
            x = torch.randn(10, 10, requires_grad=True)
            compiled(x).sum().backward()

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 4)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 2)


@functorch_config.patch({"bundled_autograd_cache": True})
class AOTAutogradCacheBundledTests(AOTAutogradCacheTests):
    # [CLASS DOC] Runs every test of AOTAutogradCacheTests again with bundled_autograd_cache
    # turned on (class level config patch); adds no tests.
    pass


@dataclasses.dataclass
class _MockEntryForPickleTest:
    # [CLASS DOC] HELPER: Dataclass standing in for a cache entry in pickle tests, with a
    # picklable string field and an arbitrary field the tests set to something unpicklable.
    # Defined at module level so pickling errors come from the field values, not from the class.
    """Module-level mock entry for pickle tests, so pickle errors come from fields not the class."""

    picklable_field: str
    unpicklable_field: object

    def pre_save(self):
        # [FUNCTION DOC] HELPER: No-op stand-in for the pre-save hook of a real entry.
        pass


@inductor_config.patch("fx_graph_cache", True)
class AOTAutogradCachePicklerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for cache key generation (AOTAutogradCachePickler and
    # autograd_cache_key), cacheability checks, and pickling of entries. Helpers compile a
    # function with a capturing backend to obtain an FX graph and example inputs, then compute
    # keys under a fake tracing context. Runs with fx_graph_cache on.
    @property
    def device_type(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning the module level device_type string.
        return device_type

    def default_config(self):
        # [FUNCTION DOC] HELPER: Returns a fresh AOTConfig with no compilers, empty
        # decompositions, dynamic_shapes true, no tangents or export, zero aot_id and no
        # precompile backend id, for building cache keys.
        return AOTConfig(
            fw_compiler=None,
            bw_compiler=None,
            inference_compiler=None,
            partition_fn=None,
            decompositions={},
            num_params_buffers=0,
            aot_id=0,
            keep_inference_input_mutations=False,
            dynamic_shapes=True,
            aot_autograd_arg_pos_to_source=None,
            is_export=False,
            no_tangents=False,
            enable_log=False,
            precompile_backend_id=None,
        )

    def _get_dynamo_output(self, fn, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Resets dynamo, compiles fn with fullgraph and a backend
        # that captures the graph module and example inputs and returns the graph unchanged,
        # calls it with args and kwargs, and returns the tuple (result, captured graph,
        # captured example inputs).
        # Reset dynamo between runs
        torch._dynamo.reset()
        fx_graph = None
        example_inputs = None

        def compiler(gm, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that stores gm and inputs in the enclosing
            # variables and returns gm.
            nonlocal fx_graph
            nonlocal example_inputs
            fx_graph = gm
            example_inputs = inputs
            return gm

        g = torch.compile(fn, backend=compiler, fullgraph=True)
        result = g(*args, **kwargs)
        return (result, fx_graph, example_inputs)

    def gen_cache_key(self, f, config, inputs=None, act_input_paths=()):
        # [FUNCTION DOC] HELPER: Computes the cache key of function f under config. Captures the
        # FX graph by compiling f on inputs (default one ones(3) tensor), then, inside a tracing
        # context with a fake mode and a fresh ShapeEnv and with the graph sanitized, calls
        # autograd_cache_key with act_input_paths and returns its result (key and debug
        # information). May raise BypassAOTAutogradCache.
        if inputs is None:
            inputs = [torch.ones(3)]
        _, fx_g, example_inputs = self._get_dynamo_output(f, *inputs)
        shape_env = ShapeEnv()
        ctx = TracingContext(FakeTensorMode(shape_env=shape_env))
        # Needs a shape env for FxGraphCache.check_can_cache to pass.
        # Not needed for actual key calculation.
        with torch._guards.tracing(ctx):
            with sanitize_gm_for_cache(fx_g):
                return autograd_cache_key(
                    fx_g,
                    example_inputs,
                    config,
                    None,
                    act_input_paths=act_input_paths,
                )

    def test_basic_hash_key(self):
        # [FUNCTION DOC] Test that the cache key for the same function and config is stable
        # across two computations.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        config = self.default_config()
        # Check hash is stable on multiple runs
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

    def test_runtime_only_configs_do_not_change_key(self):
        # [FUNCTION DOC] Test that runtime-only AOTConfig fields (aot_id, fw_compiler) do not
        # change the cache key.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        config = self.default_config()
        config2 = self.default_config()
        config2 = dataclasses.replace(config2, aot_id=1)
        config3 = self.default_config()
        config3 = dataclasses.replace(config3, fw_compiler=lambda gm, inputs: gm)

        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        c3 = self.gen_cache_key(fn, config3)
        self.assertEqual(c1, c2)
        self.assertEqual(c1, c3)

    def test_to_cacheable_strips_runtime_only_fields(self):
        # [FUNCTION DOC] Test that AOTConfig.to_cacheable returns a CacheableAOTConfig that
        # keeps static_input_indices and precompile_backend_id but has no fw_compiler attribute
        # (runtime-only compilers and partition function removed).
        config = self.default_config()
        config = dataclasses.replace(
            config,
            fw_compiler=lambda gm, inputs: gm,
            bw_compiler=lambda gm, inputs: gm,
            partition_fn=lambda *args: args,
            decompositions=None,
            inference_compiler=lambda gm, inputs: gm,
            static_input_indices=[0],
            precompile_backend_id="backend",
        )

        cacheable = config.to_cacheable()

        self.assertIsInstance(cacheable, CacheableAOTConfig)
        self.assertEqual(cacheable.static_input_indices, [0])
        self.assertEqual(cacheable.precompile_backend_id, "backend")
        self.assertFalse(hasattr(cacheable, "fw_compiler"))

    def test_different_cacheable_configs(self):
        # [FUNCTION DOC] Test that setting static_input_indices or precompile_backend_id on the
        # config each gives a different cache key than the base config.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        base_config = self.default_config()
        base = self.gen_cache_key(fn, base_config)

        config_with_static_inputs = self.default_config()
        config_with_static_inputs = dataclasses.replace(
            config_with_static_inputs, static_input_indices=[0]
        )
        self.assertNotEqual(base, self.gen_cache_key(fn, config_with_static_inputs))

        config_with_backend = self.default_config()
        config_with_backend = dataclasses.replace(
            config_with_backend, precompile_backend_id="backend"
        )
        self.assertNotEqual(base, self.gen_cache_key(fn, config_with_backend))

    def test_different_act_input_paths(self):
        # [FUNCTION DOC] Test that act_input_paths change the key: no paths, a top level path
        # for input 0, and a nested path through _local_tensor all differ (base vs top level,
        # and top level vs nested).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        config = self.default_config()
        base = self.gen_cache_key(fn, config)
        top_level_act = self.gen_cache_key(fn, config, act_input_paths=[(0, ())])
        nested_act = self.gen_cache_key(
            fn, config, act_input_paths=[(0, ("_local_tensor",))]
        )

        self.assertNotEqual(base, top_level_act)
        self.assertNotEqual(top_level_act, nested_act)

    def test_different_graphs(self):
        # [FUNCTION DOC] Test that two functions with different graphs (cos then sin vs sin then
        # cos) have different cache keys.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.cos().sin().
            return x.cos().sin()

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        config = self.default_config()
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn2, config)
        self.assertNotEqual(c1, c2)

    def test_different_configs(self):
        # [FUNCTION DOC] Test that turning off dynamic_shapes in the AOTConfig changes
        # the cache key.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.cos().sin().
            return x.cos().sin()

        config = self.default_config()
        config2 = self.default_config()
        config2 = dataclasses.replace(config2, dynamic_shapes=False)
        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config2)
        self.assertNotEqual(c1, c2)

    def test_different_inputs(self):
        # [FUNCTION DOC] Test that inputs of different shape (3 vs 2 elements) give
        # different cache keys.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.cos().sin().
            return x.cos().sin()

        config = self.default_config()
        c1 = self.gen_cache_key(fn, config, inputs=[torch.ones(3)])
        c2 = self.gen_cache_key(fn, config, inputs=[torch.ones(2)])
        self.assertNotEqual(c1, c2)

    def test_different_global_configs(self):
        # [FUNCTION DOC] Test that global state is part of the key: unchanged state gives equal
        # keys, while flipping functorch debug_assert, flipping inductor debug, or running under
        # no_grad each gives a different key.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.cos().sin().
            return x.cos().sin()

        config = self.default_config()

        c1 = self.gen_cache_key(fn, config)
        c2 = self.gen_cache_key(fn, config)
        self.assertEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)

        # Change functorch config
        with functorch_config.patch(
            {"debug_assert": not functorch_config.debug_assert}
        ):
            c2 = self.gen_cache_key(fn, config)

        self.assertNotEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)
        # Change inductor config
        with inductor_config.patch({"debug": not inductor_config.debug}):
            c2 = self.gen_cache_key(fn, config)

        self.assertNotEqual(c1, c2)

        c1 = self.gen_cache_key(fn, config)
        # Change torch grad enabled
        with torch.no_grad():
            c2 = self.gen_cache_key(fn, config)
        self.assertNotEqual(c1, c2)

    def test_incompatible_function(self):
        # [FUNCTION DOC] Test that key generation raises BypassAOTAutogradCache for an
        # allow_in_graph autograd Function (which contains a graph break).
        @torch._dynamo.allow_in_graph
        class AllowInGraphFunc(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd Function whose forward breaks the graph
            # and returns x.sin().
            @staticmethod
            def forward(_, x):
                # [FUNCTION DOC] HELPER: Forward: graph break, then x.sin().
                torch._dynamo.graph_break()
                return x.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: Applies the allow_in_graph Function to x.
            return AllowInGraphFunc.apply(x)

        config = self.default_config()
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )

    def test_private_namespace(self):
        # [FUNCTION DOC] Test that a private function (monkeypatched into the torch
        # namespace with a leading underscore name and allow_in_graph) makes key
        # generation raise BypassAOTAutogradCache.
        # TODO: anyone who monkeypatches a **public** function into torch namespace with @allow_in_graph
        # could still break our sanity check and cache something bad. But that's an edge case we'll take the risk on.
        # Monkeypatch some random private function into torch, see that it fails
        @torch._dynamo.allow_in_graph
        def my_private_fun(x):
            # [FUNCTION DOC] HELPER: Allowed-in-graph function returning x.sin().
            return x.sin()

        with patch("torch._my_priv", new=my_private_fun, create=True):

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the patched private torch function on x.
                return torch._my_priv(x)

            config = self.default_config()
            self.assertRaises(
                BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
            )

    @torch._inductor.config.patch({"freezing": True})
    def test_freezing(self):
        # [FUNCTION DOC] Test that with inductor freezing enabled key
        # generation raises BypassAOTAutogradCache.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.cos().sin().
            return x.cos().sin()

        config = self.default_config()
        self.assertRaises(
            BypassAOTAutogradCache, lambda: self.gen_cache_key(fn, config)
        )

    def test_private_builtin(self):
        # [FUNCTION DOC] Test that a private builtin (torch._foreach_add) is still cacheable and
        # that keys differ when its scalar argument differs (1 vs 2).
        # _foreach_add is a private torch function, but
        # it's also a builtin_function_or_method, so it should be allowed to be cached
        # since dynamo allows it in the graph
        def fn(x, b):
            # [FUNCTION DOC] HELPER: Returns _foreach_add of the pair (x, x) with b.
            y = (x, x)
            return torch._foreach_add(y, b)

        config = self.default_config()
        r1 = self.gen_cache_key(fn, config, inputs=[torch.ones(3), 1])
        r2 = self.gen_cache_key(fn, config, inputs=[torch.ones(3), 2])
        self.assertNotEqual(r1, r2)

    def test_nn_module_with_params(self):
        # [FUNCTION DOC] Test that two separate instances of a module with a 3x3 parameter, fed
        # same-sized inputs, give equal cache keys.
        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a 3x3 ones parameter; forward adds it to x.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the 3x3 parameter seq.
                super().__init__()
                self.seq = torch.nn.Parameter(torch.ones((3, 3)))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the parameter plus x.
                return self.seq + x

        config = self.default_config()
        # Different inputs and parameters, but all the same size
        c1 = self.gen_cache_key(MyMod(), config, inputs=[torch.ones((3, 3))])
        c2 = self.gen_cache_key(MyMod(), config, inputs=[torch.ones((3, 3))])
        self.assertEqual(c1, c2)

    def test_normal_torch_function(self):
        # [FUNCTION DOC] Test that a plain torch function (allowed in graph) can have a cache
        # key generated without raising; no value is asserted.
        @torch._dynamo.allow_in_graph
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes sin(x) + cos(x) (also calls abs, ignoring the
            # result) and returns the sum.
            y = torch.sin(x)
            z = torch.cos(x)
            w = y + z
            w.abs()
            return w

        config = self.default_config()
        self.gen_cache_key(fn, config)

    def test_safe_torchfunction(self):
        # [FUNCTION DOC] Test that a function using safe symbolic torch functions (size
        # comparison, sym_int, sym_float, sym_sqrt, sym_sum) can have a cache key
        # generated without raising.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compares x.size() to a Size, builds symbolic numbers from 9
            # and returns (comparison result, sym_sum of them).
            a = x.size()
            b = torch.Size([3, 3])
            c = a == b
            x = torch.sym_int(9)
            y = torch.sym_float(x)
            z = torch.sym_int(torch.sym_sqrt(y))
            result = torch.sym_sum([x, y, z])
            return (c, result)

        config = self.default_config()
        self.gen_cache_key(fn, config, inputs=[torch.ones((3, 3))])

    def test_numpy_wrapper_requires_torch_numpy_target(self):
        # [FUNCTION DOC] Test that a numpy-to-tensor wrapper around a function that is not
        # from torch._numpy fails check_cacheable with BypassAOTAutogradCache
        # ("Unsupported call_function target").
        def not_torch_numpy(x):
            # [FUNCTION DOC] HELPER: Plain function returning x.sin().
            return x.sin()

        wrapper = torch._dynamo.utils.numpy_to_tensor_wrapper(not_torch_numpy)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(wrapper, (x,))
        graph.output(y)
        gm = torch.fx.GraphModule({}, graph)

        with self.assertRaisesRegex(
            BypassAOTAutogradCache, "Unsupported call_function target"
        ):
            check_cacheable(gm)

    def test_numpy_wrapper_cache_key_does_not_cache_unknown_callable_ids(self):
        # [FUNCTION DOC] Test that the torch._numpy cache key lookup returns None for an unknown
        # callable and does not add an entry to its by-id LRU cache (cache size stays 0). Clears
        # the cache first.
        torch._dynamo.utils._torch_numpy_callable_cache_key_by_id.cache_clear()

        def not_torch_numpy(x):
            # [FUNCTION DOC] HELPER: Plain function returning x.
            return x

        self.assertIsNone(
            torch._dynamo.utils._torch_numpy_callable_cache_key(not_torch_numpy)
        )
        cache_info = (
            torch._dynamo.utils._torch_numpy_callable_cache_key_by_id.cache_info()
        )
        self.assertEqual(
            cache_info.currsize,
            0,
        )

    def test_numpy_wrapper_accepts_torch_numpy_alias_target(self):
        # [FUNCTION DOC] Test that numpy wrappers over torch._numpy abs and absolute pass
        # check_cacheable and have a non-None numpy wrapper cache key.
        import torch._numpy as tnp

        for target in (tnp.abs, tnp.absolute):
            wrapper = torch._dynamo.utils.numpy_to_tensor_wrapper(target)
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            y = graph.call_function(wrapper, (x,))
            graph.output(y)
            gm = torch.fx.GraphModule({}, graph)

            check_cacheable(gm)
            self.assertIsNotNone(torch._dynamo.utils.numpy_wrapper_cache_key(wrapper))

    def test_numpy_wrapper_rejects_spoofed_torch_numpy_target(self):
        # [FUNCTION DOC] Test that a function faking torch._numpy module and name (spoofed
        # round) is rejected by check_cacheable with BypassAOTAutogradCache.
        def spoofed_round(x):
            # [FUNCTION DOC] HELPER: Plain function returning x.sin() with spoofed metadata.
            return x.sin()

        spoofed_round.__module__ = "torch._numpy._funcs_impl"
        spoofed_round.__qualname__ = "round"
        spoofed_round.__name__ = "round"

        wrapper = torch._dynamo.utils.numpy_to_tensor_wrapper(spoofed_round)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(wrapper, (x,))
        graph.output(y)
        gm = torch.fx.GraphModule({}, graph)

        with self.assertRaisesRegex(
            BypassAOTAutogradCache, "Unsupported call_function target"
        ):
            check_cacheable(gm)

    def test_numpy_wrapper_rejects_subclass(self):
        # [FUNCTION DOC] Test that a subclass of the numpy wrapper class is rejected by
        # check_cacheable with BypassAOTAutogradCache, even around tnp.round.
        import torch._numpy as tnp

        class CustomWrapper(torch._dynamo.utils.numpy_to_tensor_wrapper):
            # [CLASS DOC] HELPER: Subclass of numpy_to_tensor_wrapper with an overridden call.
            def __call__(self, x):
                # [FUNCTION DOC] HELPER: Returns x unchanged.
                return x

        wrapper = CustomWrapper(tnp.round)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        y = graph.call_function(wrapper, (x,))
        graph.output(y)
        gm = torch.fx.GraphModule({}, graph)

        with self.assertRaisesRegex(
            BypassAOTAutogradCache, "Unsupported call_function target"
        ):
            check_cacheable(gm)

    def test_numpy_wrapper_hash_distinguishes_safe_wrappers(self):
        # [FUNCTION DOC] Test that graphs with numpy wrappers over tnp.round and tnp.sin are
        # both cacheable and hash to different values with AOTAutogradCachePickler.
        import torch._numpy as tnp

        def make_gm(target):
            # [FUNCTION DOC] HELPER: Builds a GraphModule with one input, calling a numpy
            # wrapper of target on it, outputting the call result.
            wrapper = torch._dynamo.utils.numpy_to_tensor_wrapper(target)
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            y = graph.call_function(wrapper, (x,))
            graph.output(y)
            return torch.fx.GraphModule({}, graph)

        round_gm = make_gm(tnp.round)
        sin_gm = make_gm(tnp.sin)

        check_cacheable(round_gm)
        check_cacheable(sin_gm)

        self.assertNotEqual(
            AOTAutogradCachePickler(round_gm).get_hash(round_gm),
            AOTAutogradCachePickler(sin_gm).get_hash(sin_gm),
        )

    def test_graph_module_hash_preserves_kernel_idx_strings(self):
        # [FUNCTION DOC] Test that the graph module hash keeps string arguments such as
        # "kernel_idx = 1" vs "kernel_idx = 2" distinct: two graphs calling _assert with those
        # messages are cacheable and hash differently.
        def make_gm(message):
            # [FUNCTION DOC] HELPER: Builds a GraphModule with an input and a call to
            # torch._assert(True, message); the input is the output.
            graph = torch.fx.Graph()
            x = graph.placeholder("x")
            graph.call_function(torch._assert, (True, message))
            graph.output(x)
            return torch.fx.GraphModule({}, graph)

        gm1 = make_gm("kernel_idx = 1")
        gm2 = make_gm("kernel_idx = 2")

        check_cacheable(gm1)
        check_cacheable(gm2)

        self.assertNotEqual(
            AOTAutogradCachePickler(gm1).get_hash(gm1),
            AOTAutogradCachePickler(gm2).get_hash(gm2),
        )

    def test_sanitize_gm_for_cache(self):
        # [FUNCTION DOC] Test that graph metadata and compile_subgraph_reason do not affect the
        # key: the keys computed with different meta values and reason are all equal, with and
        # without the sanitize_gm_for_cache context.
        def fn(x):
            # [FUNCTION DOC] HELPER: Computes sin(x) + cos(x) (calls abs on the sum and drops
            # it) and returns the sum.
            y = torch.sin(x)
            z = torch.cos(x)
            w = y + z
            w.abs()
            return w

        _, fx_g, example_inputs = self._get_dynamo_output(fn, torch.ones(3))

        ctx = TracingContext(FakeTensorMode(shape_env=ShapeEnv()))
        with torch._guards.tracing(ctx):
            fx_g.meta = {"foo": "bar"}
            fx_g.compile_subgraph_reason = "Blah"
            config = self.default_config()
            # autograd_cache_key now applies sanitize_gm_for_cache
            # internally, so the result is the same whether we wrap the
            # call or not.
            with sanitize_gm_for_cache(fx_g):
                c1 = autograd_cache_key(fx_g, example_inputs, config, None)
            c3 = autograd_cache_key(fx_g, example_inputs, config, None)

            fx_g.meta = {"foo": "baz"}
            fx_g.compile_subgraph_reason = None
            with sanitize_gm_for_cache(fx_g):
                c2 = autograd_cache_key(fx_g, example_inputs, config, None)
            c4 = autograd_cache_key(fx_g, example_inputs, config, None)

            self.assertEqual(c1, c2)
            self.assertEqual(c1, c3)
            self.assertEqual(c1, c4)

    def test_dill_serialization_with_inner_functions(self):
        # [FUNCTION DOC] Test that, when dill is available, a graph with an inner function
        # stored in node metadata round trips through GraphPickler (dumps then loads with a fake
        # mode) and the function still works (returns 10 for input 5). Skips if dill is missing.
        """
        Test that with dill, we can now serialize graphs that contain inner functions
        in node metadata. Standard pickle would fail on these because inner functions
        are not defined at module level, but dill serializes function code objects and
        closures directly.
        """
        from torch.fx._graph_pickler import GraphPickler, Options
        from torch.utils._import_utils import import_dill

        dill = import_dill()
        if dill is None:
            self.skipTest("dill not available")

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().cos().
            return x.sin().cos()

        _, fx_g, example_inputs = self._get_dynamo_output(fn, torch.ones(3))

        def inner_helper(x):
            # [FUNCTION DOC] HELPER: Inner (non-module-level) function returning x * 2.
            return x * 2

        for node in fx_g.graph.nodes:
            node.meta["inner_fn"] = inner_helper

        options = Options(node_metadata_key_filter=None)
        serialized = GraphPickler.dumps(fx_g, options)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        deserialized = GraphPickler.loads(serialized, fake_mode)

        self.assertIsInstance(deserialized, torch.fx.GraphModule)
        for node in deserialized.graph.nodes:
            self.assertIn("inner_fn", node.meta)
            self.assertEqual(node.meta["inner_fn"](5), 10)

    def test_pickle_entry_with_unpicklable_field(self):
        # [FUNCTION DOC] Test that _pickle_entry returns None and logs exactly one warning
        # (naming the WeakValueDictionary remove callback) when an entry field is unpicklable.
        from weakref import WeakValueDictionary

        # WeakValueDictionary's internal 'remove' callback is unpicklable
        weak_dict = WeakValueDictionary()
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=weak_dict,
        )

        with self.assertLogs(
            "torch._functorch._aot_autograd.autograd_cache", level="WARNING"
        ) as log_context:
            result = AOTAutogradCache._pickle_entry(entry, remote=False)

        self.assertIsNone(result)
        self.assertEqual(len(log_context.output), 1)
        self.assertIn(
            "WeakValueDictionary.__init__.<locals>.remove",
            log_context.output[0],
        )

    def test_pickle_entry_with_lambda(self):
        # [FUNCTION DOC] Test that _pickle_entry on an entry with a lambda field returns None,
        # logs one warning naming the lambda, and emits one structured trace artifact named
        # aotautograd_cache_pickle_failure with json encoding.
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=lambda x: x,  # lambdas are unpicklable
        )

        with (
            patch("torch._logging.trace_structured") as mock_trace,
            self.assertLogs(
                "torch._functorch._aot_autograd.autograd_cache", level="WARNING"
            ) as log_context,
        ):
            result = AOTAutogradCache._pickle_entry(entry, remote=False)

        self.assertIsNone(result)
        self.assertEqual(len(log_context.output), 1)
        self.assertIn(
            """AOTAutogradCachePicklerTests.test_pickle_entry_with_lambda.<locals>.<lambda>""",
            log_context.output[0],
        )
        mock_trace.assert_called_once()
        call_args = mock_trace.call_args
        self.assertEqual(call_args[0][0], "artifact")
        metadata = call_args[1]["metadata_fn"]()
        self.assertEqual(metadata["name"], "aotautograd_cache_pickle_failure")
        self.assertEqual(metadata["encoding"], "json")

    @functorch_config.patch("strict_autograd_cache", True)
    def test_pickle_entry_strict_mode_raises(self):
        # [FUNCTION DOC] Test that with strict_autograd_cache on, _pickle_entry re-raises the
        # pickling error (AttributeError or PicklingError) naming the local lambda.
        entry = _MockEntryForPickleTest(
            picklable_field="test",
            unpicklable_field=lambda x: x,
        )

        # Python 3.14+ raises PicklingError with "Can't pickle local object"
        # Earlier versions raise AttributeError with "Can't get local object"
        with self.assertRaisesRegex(
            (AttributeError, pickle.PicklingError),
            r"AOTAutogradCachePicklerTests.test_pickle_entry_strict_mode_raises.<locals>.<lambda>",
        ):
            AOTAutogradCache._pickle_entry(entry, remote=False)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/4091")
    @requires_gpu_and_triton
    def test_prepare_for_pickle_clears_benchmark_failure_reasons(self):
        # [FUNCTION DOC] Test that CachingAutotuner.prepare_for_pickle clears
        # benchmark_failure_reasons: runs a small triton add kernel through an autotuner,
        # injects a launcher key into benchmark_failure_reasons, checks that pickling then
        # fails, and that after prepare_for_pickle the dict is empty and pickling succeeds.
        # Needs a GPU and triton; skipped on XPU.
        """prepare_for_pickle clears benchmark_failure_reasons which can hold
        exec'd launcher keys that aren't picklable.
        """
        from torch._inductor.runtime.hints import (
            AttrsDescriptorWrapper,
            DeviceProperties,
            HeuristicType,
        )
        from torch._inductor.runtime.triton_heuristics import CachingAutotuner
        from torch._inductor.utils import triton_version_uses_attrs_dict

        meta = {
            "signature": {
                "in_ptr0": "*fp32",
                "out_ptr0": "*fp32",
                "xnumel": "i32",
            },
            "device": DeviceProperties.create(torch.device(GPU_TYPE)),
            "configs": [AttrsDescriptorWrapper(divisible_by_16=(0, 1), equal_to_1=())],
            "constants": {},
        }
        if triton_version_uses_attrs_dict():
            meta["signature"]["XBLOCK"] = "constexpr"

        # [FUNCTION DOC] HELPER: Triton kernel adding 1.0 to xnumel elements of in_ptr0 in
        # blocks of XBLOCK, writing to out_ptr0 with a mask.
        @triton.jit
        def _add_kernel(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            pid = tl.program_id(0)
            offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
            mask = offsets < xnumel
            x = tl.load(in_ptr0 + offsets, mask=mask)
            tl.store(out_ptr0 + offsets, x + 1.0, mask=mask)

        autotuner = CachingAutotuner(
            fn=_add_kernel,
            triton_meta=meta,
            configs=[triton.Config({"XBLOCK": 128})],
            save_cache_hook=False,
            mutated_arg_names=[],
            optimize_mem=True,
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={"grid_type": "Grid1D"},
        )

        xnumel = 256
        inp = torch.randn(xnumel, device=GPU_TYPE)
        out = torch.empty_like(inp)
        autotuner.run(inp, out, xnumel, stream=torch.cuda.current_stream().cuda_stream)
        self.assertEqual(out, inp + 1.0)

        # Inject a launcher key into benchmark_failure_reasons — this is how
        # the leak happens in production (launcher used as dict key).
        self.assertTrue(len(autotuner.launchers) > 0)
        launcher_fn = autotuner.launchers[0]
        autotuner.benchmark_failure_reasons[launcher_fn] = "test"

        # The autotuner is unpicklable: __getstate__ asserts launchers are
        # empty, and benchmark_failure_reasons holds an exec'd launcher key.
        with self.assertRaises((pickle.PicklingError, AttributeError, AssertionError)):
            pickle.dumps(autotuner)

        # prepare_for_pickle clears benchmark_failure_reasons (and other
        # unpicklable fields), making pickle succeed.
        autotuner.prepare_for_pickle()
        self.assertEqual(autotuner.benchmark_failure_reasons, {})
        pickle.dumps(autotuner)

    def test_nested_tensor_subclass_cache_key(self):
        # [FUNCTION DOC] Test that the cache key of a nested tensor subclass is the same in two
        # separate spawned processes: each runs the module level subprocess function and returns
        # its key via a queue. Skips if a subprocess fails.
        ctx = multiprocessing.get_context("spawn")
        results = []
        for _ in range(2):
            queue = ctx.Queue()
            p = ctx.Process(
                target=_subprocess_gen_nested_subclass_cache_key,
                args=(queue,),
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                self.skipTest(f"Subprocess exited with code {p.exitcode}")

            results.append(queue.get())

        # Cache keys from two different processes should be identical
        self.assertEqual(
            results[0],
            results[1],
            "Nested tensor subclass cache keys should be identical across processes",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_dtensor_different_process_cache_key(self):
        # [FUNCTION DOC] Test that a DTensor's cache key is identical across two separate
        # spawned processes (needed for warm start hits), using the module level subprocess
        # function. Skips if a subprocess fails or distributed is unavailable.
        """
        Test that DTensor cache keys are consistent across different processes.

        This is a critical test for warm start cache hits. The cache key must be
        deterministic and not depend on process-specific values like memory addresses
        or object ids that would differ between processes.
        """
        ctx = multiprocessing.get_context("spawn")
        results = []
        for _ in range(2):
            queue = ctx.Queue()
            p = ctx.Process(
                target=_subprocess_gen_dtensor_cache_key,
                args=(queue,),
            )
            p.start()
            p.join()

            if p.exitcode != 0:
                self.skipTest(f"Subprocess exited with code {p.exitcode}")

            results.append(queue.get())

        # Cache keys from two different processes should be identical
        self.assertEqual(
            results[0],
            results[1],
            "DTensor cache keys should be identical across processes",
        )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_dtensor_fake_script_object_mesh_cache_key(self):
        # [FUNCTION DOC] Test that a DTensor whose mesh is a FakeScriptObject wrapping a
        # DeviceMesh hashes the same under AOTAutogradCachePickler as one with the real mesh.
        # Uses a fake process group.
        from torch._library.fake_class_registry import FakeScriptObject
        from torch.distributed.device_mesh import DeviceMesh
        from torch.distributed.tensor import DTensor, Replicate
        from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta

        with _fake_process_group():
            mesh = DeviceMesh("cpu", torch.arange(2))
            fake_mesh = FakeScriptObject(
                mesh,
                f"{DeviceMesh.__module__}.{DeviceMesh.__qualname__}",
                mesh,
            )
            local_tensor = torch.zeros(4, 4)
            tensor_meta = TensorMeta(
                local_tensor.shape, local_tensor.stride(), local_tensor.dtype
            )
            fake_mesh_dtensor = DTensor(
                local_tensor,
                DTensorSpec(
                    fake_mesh,
                    (Replicate(),),
                    tensor_meta=tensor_meta,
                ),
                requires_grad=False,
            )
            real_mesh_dtensor = DTensor(
                local_tensor,
                DTensorSpec(
                    mesh,
                    (Replicate(),),
                    tensor_meta=tensor_meta,
                ),
                requires_grad=False,
            )

            pickler = AOTAutogradCachePickler(
                torch.fx.GraphModule({}, torch.fx.Graph())
            )
            self.assertEqual(
                pickler.get_hash(fake_mesh_dtensor),
                pickler.get_hash(real_mesh_dtensor),
            )

    @unittest.skipIf(not torch.distributed.is_available(), "requires distributed")
    def test_dtensor_different_placements_different_cache_key(self):
        # [FUNCTION DOC] Test that DTensors with the same local data but different placements
        # (Replicate vs Shard on dim 0) on a fake 2-rank CPU mesh hash differently with
        # AOTAutogradCachePickler built from a captured graph.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate, Shard

        with _fake_process_group():
            mesh = init_device_mesh("cpu", (2,))

            local_tensor = torch.zeros(4, 4)
            dtensor_replicate = DTensor.from_local(local_tensor, mesh, [Replicate()])
            dtensor_shard = DTensor.from_local(local_tensor, mesh, [Shard(0)])

            def fn(x):
                # [FUNCTION DOC] HELPER: Returns x.sin().
                return x.sin()

            # Get dynamo output for replicate
            torch._dynamo.reset()
            fx_graph = None

            def compiler(gm, inputs, **kwargs):
                # [FUNCTION DOC] HELPER: Backend that stores gm in the enclosing fx_graph and
                # returns it unchanged.
                nonlocal fx_graph
                fx_graph = gm
                return gm

            g = torch.compile(fn, backend=compiler, fullgraph=True)
            g(dtensor_replicate)

            pickler = AOTAutogradCachePickler(fx_graph)
            cache_key_replicate = pickler.get_hash(dtensor_replicate)
            cache_key_shard = pickler.get_hash(dtensor_shard)

            # Different placements should produce different cache keys
            self.assertNotEqual(
                cache_key_replicate,
                cache_key_shard,
                "DTensor with Replicate() should have different cache key than Shard(0)",
            )

    def test_reducer_override_fallthrough_for_unpicklable_types(self):
        # [FUNCTION DOC] Test that AOTAutogradCachePickler falls back to the parent
        # reducer_override for objects that are not tensor subclasses and cannot be pickled
        # normally: pickling does not raise, equal values give equal bytes and different values
        # give different bytes.
        """
        Test that AOTAutogradCachePickler falls through to the parent class
        reducer_override for types that are not tensor subclasses but are
        unpicklable (e.g. pybind11 enums).
        """

        class FakeReduceOp:
            # [CLASS DOC] HELPER: Object holding a name whose pickling protocol raises
            # TypeError, imitating a pybind11 enum.
            """Simulates a pybind11 enum that cannot be pickled."""

            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Stores the name.
                self.name = name

            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raises TypeError saying the object cannot be
                # pickled; the pickler must handle this itself.
                raise TypeError("cannot pickle 'FakeReduceOp' object")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = AOTAutogradCachePickler(gm)

        obj_a = FakeReduceOp("SUM")
        obj_b = FakeReduceOp("SUM")
        obj_c = FakeReduceOp("PRODUCT")

        # Should not raise — parent's reducer_override handles it
        data_a = pickler.dumps(obj_a)
        data_b = pickler.dumps(obj_b)
        data_c = pickler.dumps(obj_c)

        # Same value -> same hash
        self.assertEqual(data_a, data_b)
        # Different value -> different hash
        self.assertNotEqual(data_a, data_c)


def _subprocess_gen_dtensor_cache_key(queue):
    # [FUNCTION DOC] HELPER: Runs in a spawned process. Creates a fake 2-rank process group and
    # mesh, builds a Replicate DTensor of zeros, captures the FX graph of a sin function
    # compiled with a capturing backend, computes the AOTAutogradCachePickler hash of the
    # DTensor and puts it in queue. Always destroys the process group. Module level so
    # multiprocessing can import it.
    """
    Subprocess helper to generate a DTensor cache key.
    Must be at module level for multiprocessing to work.
    """
    import torch
    import torch._dynamo
    import torch.distributed as c10d
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCachePickler
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import DTensor, Replicate
    from torch.testing._internal.distributed.fake_pg import FakeStore

    fake_store = FakeStore()
    c10d.init_process_group("fake", store=fake_store, rank=0, world_size=2)
    try:
        mesh = init_device_mesh("cpu", (2,))

        # Create DTensor
        local_tensor = torch.zeros(4, 4)
        dtensor = DTensor.from_local(local_tensor, mesh, [Replicate()])

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x.sin().
            return x.sin()

        # Get dynamo output
        torch._dynamo.reset()
        fx_graph = None

        def compiler(gm, inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that stores gm in the enclosing fx_graph and
            # returns it unchanged.
            nonlocal fx_graph
            fx_graph = gm
            return gm

        g = torch.compile(fn, backend=compiler, fullgraph=True)
        g(dtensor)

        pickler = AOTAutogradCachePickler(fx_graph)
        cache_key = pickler.get_hash(dtensor)

        queue.put(cache_key)
    finally:
        c10d.destroy_process_group()


def _subprocess_gen_nested_subclass_cache_key(queue):
    # [FUNCTION DOC] HELPER: Runs in a spawned process. Builds a TwoTensor of two TwoTensors of
    # zeros, captures the FX graph of a sin function with a capturing backend, computes the
    # AOTAutogradCachePickler hash of the nested subclass and puts it in queue.
    import torch
    import torch._dynamo
    from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCachePickler
    from torch.testing._internal.two_tensor import TwoTensor

    inner1 = TwoTensor(torch.zeros(4, 4), torch.zeros(4, 4))
    inner2 = TwoTensor(torch.zeros(4, 4), torch.zeros(4, 4))
    nested_two_tensor = TwoTensor(inner1, inner2)

    def fn(x):
        # [FUNCTION DOC] HELPER: Returns x.sin().
        return x.sin()

    # Get dynamo output
    torch._dynamo.reset()
    fx_graph = None

    def compiler(gm, inputs, **kwargs):
        # [FUNCTION DOC] HELPER: Backend that stores gm in the enclosing fx_graph and
        # returns it unchanged.
        nonlocal fx_graph
        fx_graph = gm
        return gm

    g = torch.compile(fn, backend=compiler, fullgraph=True)
    g(nested_two_tensor)

    pickler = AOTAutogradCachePickler(fx_graph)
    cache_key = pickler.get_hash(nested_two_tensor)

    queue.put(cache_key)


def _policy_save_mm(ctx, op, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Selective checkpoint policy returning MUST_SAVE for the aten mm op
    # and MUST_RECOMPUTE for every other op.
    if op == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _policy_save_add(ctx, op, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Selective checkpoint policy returning MUST_SAVE for the aten
    # add.Tensor op and MUST_RECOMPUTE for every other op.
    if op == torch.ops.aten.add.Tensor:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _policy_no_hash(ctx, op, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Same behavior as the save-mm policy (saves mm, recomputes the
    # rest); used with a context function that has no cache_hash.
    if op == torch.ops.aten.mm.default:
        return CheckpointPolicy.MUST_SAVE
    return CheckpointPolicy.MUST_RECOMPUTE


def _create_sac_ctx_fn(policy, cache_hash=None):
    # [FUNCTION DOC] HELPER: Returns a partial of create_selective_checkpoint_contexts bound to
    # policy; if cache_hash is not None it is stored as the cache_hash attribute of the partial
    # (which the cache uses as the policy's identity).
    """
    Helper to create a SAC context_fn with cache_hash set on the partial.
    """
    ctx_fn = functools.partial(create_selective_checkpoint_contexts, policy)
    if cache_hash is not None:
        ctx_fn.cache_hash = cache_hash
    return ctx_fn


class HOPCacheTests(CacheKeyEquivalenceMixin, torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests of how higher-order-op regions (activation
    # checkpointing with selective checkpoint policies, RNG in checkpoint) interact
    # with the autograd cache. Includes the key-equivalence mixin check. Counters
    # are cleared and dynamo reset before and after each test.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setUp, clears counters, resets dynamo.
        super().setUp()
        counters.clear()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs parent tearDown, clears counters, resets dynamo.
        super().tearDown()
        counters.clear()
        torch._dynamo.reset()

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_different_sac_policies_cause_cache_miss(self):
        # [FUNCTION DOC] Test (strict mode) that two selective checkpoint policies with
        # different cache_hash values give different cache entries: compiling with the save-mm
        # policy is one miss; after a dynamo reset, compiling with the save-add policy is a
        # second miss with no hits.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Computes mm(x, y), then adds x to it, and returns that.
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn_save_mm = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        ctx_fn_save_add = _create_sac_ctx_fn(_policy_save_add, "policy_save_add_v1")

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y, ctx_fn):
            # [FUNCTION DOC] HELPER: Compiled function running gn under non-reentrant checkpoint
            # with the given context function.
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y, ctx_fn_save_mm)

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_with_checkpoint(x, y, ctx_fn_save_add)

            # Different policy: miss=2, hit=0 (policy is part of cache key)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_same_sac_policy_causes_cache_hit(self):
        # [FUNCTION DOC] Test (strict mode) that the same selective checkpoint policy and hash
        # cause a cache hit: first run one miss; after a dynamo reset the second run has one hit
        # and still one miss.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Computes mm(x, y), then adds x to it, and returns that.
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y):
            # [FUNCTION DOC] HELPER: Compiled function running gn under non-reentrant checkpoint
            # with the shared context function.
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y)

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_with_checkpoint(x, y)

            # Same policy: miss stays at 1, hit increments to 1
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_checkpoint_with_dropout_caches_correctly(self):
        # [FUNCTION DOC] Test (strict mode) that a checkpointed region with dropout (random ops)
        # is cacheable: forward and backward of the first run is one miss; after a dynamo reset
        # the second run is a hit with the miss count unchanged.
        def gn(x):
            # [FUNCTION DOC] HELPER: Applies dropout with p of 0.5 in training mode.
            return torch.dropout(x, p=0.5, train=True)

        @torch.compile(backend="inductor")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled function running gn
            # under non-reentrant checkpoint.
            return checkpoint(gn, x, use_reentrant=False)

        x = torch.randn(4, 4, requires_grad=True)

        with fresh_cache():
            out1 = fn(x.clone().requires_grad_(True))
            out1.sum().backward()

            # First run: miss=1, hit=0
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            out2 = fn(x.clone().requires_grad_(True))
            out2.sum().backward()

            # Same function with RNG HOPs: miss stays at 1, hit increments to 1
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch("enable_autograd_cache", True)
    def test_sac_without_cache_hash_bypasses_cache(self):
        # [FUNCTION DOC] Test that a selective checkpoint context function without a cache_hash
        # makes the autograd cache bypass: no miss, no hit, and a positive bypass counter.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Computes mm(x, y), then adds x to it, and returns that.
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn = _create_sac_ctx_fn(_policy_no_hash)  # No cache_hash

        @torch.compile(backend="inductor")
        def fn_with_checkpoint(x, y):
            # [FUNCTION DOC] HELPER: Compiled function running gn under non-reentrant checkpoint
            # with the context function that has no hash.
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_with_checkpoint(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertGreater(counters["aot_autograd"]["autograd_cache_bypass"], 0)

    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_changing_cache_hash_invalidates_cache(self):
        # [FUNCTION DOC] Test (strict mode) that using a different policy and different
        # cache_hash invalidates the cache: the save-mm function is one miss; after a dynamo
        # reset the save-add function is a second miss with no hits.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: Computes mm(x, y), then adds x to it, and returns that.
            a = torch.mm(x, y)
            b = torch.add(a, x)
            return b

        ctx_fn_mm = _create_sac_ctx_fn(_policy_save_mm, "policy_save_mm_v1")

        ctx_fn_add = _create_sac_ctx_fn(_policy_save_add, "policy_save_add_v1")

        @torch.compile(backend="inductor")
        def fn_mm(x, y):
            # [FUNCTION DOC] HELPER: Compiled function checkpointing gn with the
            # save-mm context function.
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn_mm)

        @torch.compile(backend="inductor")
        def fn_add(x, y):
            # [FUNCTION DOC] HELPER: Compiled function checkpointing gn with the
            # save-add context function.
            return checkpoint(gn, x, y, use_reentrant=False, context_fn=ctx_fn_add)

        x = torch.randn(4, 4)
        y = torch.randn(4, 4)

        with fresh_cache():
            fn_mm(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)

            torch._dynamo.reset()

            fn_add(x, y)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)


@instantiate_parametrized_tests
class CacheKeyAPITests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that the public autograd_cache_key functions (the one in aot_autograd
    # and the one in compile_fx) return the same key as the key computed during a real
    # torch.compile run, and that keys vary with input dtype and graph. Counters are cleared and
    # dynamo reset around each test.
    """
    Verify that the exposed autograd_cache_key API function produces the same
    cache key that the internal compilation pipeline would.
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs parent setUp, clears counters, resets dynamo.
        super().setUp()
        counters.clear()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs parent tearDown, clears counters, resets dynamo.
        super().tearDown()
        counters.clear()
        torch._dynamo.reset()

    def _compile_and_capture(self, fn, *args):
        # [FUNCTION DOC] HELPER: Compiles fn (fullgraph, default backend) in a fresh cache while
        # patching compile_fx and autograd_cache_key with wrappers that capture the graph
        # module, example inputs and the key (first element of the real result). Asserts a key
        # was captured and returns (key, graph module, example inputs list).
        """Run torch.compile with the inductor backend and capture the
        ground-truth cache key, graph module, and example inputs produced
        during compilation."""
        torch._dynamo.reset()
        real_compile_fx = torch._inductor.compile_fx.compile_fx
        captured_key = None
        captured_gm = None
        captured_example_inputs = None

        def capturing_compile_fx(model_, example_inputs_, **kwargs):
            # [FUNCTION DOC] HELPER: Records model_ and a list of the example inputs, then
            # delegates to the real compile_fx and returns its result.
            nonlocal captured_gm, captured_example_inputs
            captured_gm = model_
            captured_example_inputs = list(example_inputs_)
            return real_compile_fx(model_, example_inputs_, **kwargs)

        real_cache_key = autograd_cache.autograd_cache_key

        def capturing_cache_key(
            mod, ei, config, compiler_config_extra=None, act_input_paths=()
        ):
            # [FUNCTION DOC] HELPER: Calls the real autograd_cache_key, stores the key (first
            # element of its result) in the enclosing captured_key and returns the result.
            nonlocal captured_key
            result = real_cache_key(
                mod, ei, config, compiler_config_extra, act_input_paths
            )
            captured_key = result[0]
            return result

        with fresh_cache():
            with (
                patch(
                    "torch._inductor.compile_fx.compile_fx",
                    side_effect=capturing_compile_fx,
                ),
                patch(
                    "torch._functorch._aot_autograd.autograd_cache.autograd_cache_key",
                    side_effect=capturing_cache_key,
                ),
            ):
                torch.compile(fn, fullgraph=True)(*args)  # noqa: UNSPECIFIED_BACKEND

        self.assertIsNotNone(captured_key)
        return captured_key, captured_gm, captured_example_inputs

    def _tracing_context(self):
        # [FUNCTION DOC] HELPER: Returns a tracing context manager holding a TracingContext with
        # a fake mode and a fresh ShapeEnv.
        """Create a TracingContext with a ShapeEnv, as required by
        _check_can_cache inside the cache key computation."""
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        return torch._guards.tracing(TracingContext(fake_mode))

    def _aot_autograd_cache_key(self, fx_graph, example_inputs):
        # [FUNCTION DOC] HELPER: Calls aot_autograd.autograd_cache_key on the graph with the
        # settings compile_fx would use: compiler config extra from the graph, the inductor
        # decomposition table, ignore_shape_env, keep_inference_input_mutations, and functorch
        # patches for unlift_effect_tokens and selective_decompose. Runs inside a tracing
        # context and returns the (key, debug info) result.
        """Call aot_autograd.autograd_cache_key with the same config and
        tracing context that compile_fx would use."""
        from torch._inductor.compile_fx import create_compiler_config_extra
        from torch._inductor.decomposition import select_decomp_table

        compiler_config_extra = create_compiler_config_extra(fx_graph)
        decompositions = select_decomp_table()

        # Match the config patches that compile_fx applies during compilation
        # so that the autograd_config snapshot is identical.
        with (
            self._tracing_context(),
            functorch_config.patch(
                unlift_effect_tokens=True,
                selective_decompose=inductor_config.selective_decompose,
            ),
        ):
            return aot_autograd.autograd_cache_key(
                fx_graph,
                example_inputs,
                ignore_shape_env=True,
                decompositions=decompositions,
                compiler_config_extra=compiler_config_extra,
                keep_inference_input_mutations=True,
            )

    def _compile_fx_cache_key(self, fx_graph, example_inputs):
        # [FUNCTION DOC] HELPER: Calls compile_fx.autograd_cache_key (which handles
        # decompositions and config patching itself) with ignore_shape_env inside a tracing
        # context and returns the (key, debug info) result.
        """Call compile_fx.autograd_cache_key, the higher-level API that
        handles decompositions and config patching internally."""
        with self._tracing_context():
            return compile_fx.autograd_cache_key(
                fx_graph,
                example_inputs,
                ignore_shape_env=True,
            )

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_matches_torch_compile(self):
        # [FUNCTION DOC] Test that the keys from aot_autograd.autograd_cache_key and
        # compile_fx.autograd_cache_key equal the key captured from a real compile.
        """autograd_cache_key matches the real compile_fx pipeline."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin(x) + cos(x).
            return x.sin() + x.cos()

        x = torch.randn(4, 4)
        gt_key, fx_graph, example_inputs = self._compile_and_capture(fn, x)
        api_key, _ = self._aot_autograd_cache_key(fx_graph, example_inputs)
        cfx_key, _ = self._compile_fx_cache_key(fx_graph, example_inputs)

        self.assertEqual(gt_key, api_key)
        self.assertEqual(gt_key, cfx_key)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_stable_across_calls(self):
        # [FUNCTION DOC] Test that cache keys are deterministic: repeated calls of each API
        # agree with each other and with the key from the real compile.
        """Cache keys should be deterministic."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin(x) + cos(x).
            return x.sin() + x.cos()

        x = torch.randn(4, 4)
        gt_key, fx_graph, example_inputs = self._compile_and_capture(fn, x)
        key1, _ = self._aot_autograd_cache_key(fx_graph, example_inputs)
        key2, _ = self._aot_autograd_cache_key(fx_graph, example_inputs)
        cfx_key1, _ = self._compile_fx_cache_key(fx_graph, example_inputs)
        cfx_key2, _ = self._compile_fx_cache_key(fx_graph, example_inputs)
        self.assertEqual(gt_key, key1)
        self.assertEqual(key1, key2)
        self.assertEqual(gt_key, cfx_key1)
        self.assertEqual(cfx_key1, cfx_key2)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_varies_with_different_inputs(self):
        # [FUNCTION DOC] Test that float32 and float64 inputs give different keys, each matching
        # the key from the corresponding real compile via both APIs.
        """Different input dtypes should produce different cache keys."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin(x) + cos(x).
            return x.sin() + x.cos()

        x_float = torch.randn(4, 4)
        x_double = torch.randn(4, 4, dtype=torch.float64)

        gt_key1, graph1, inputs1 = self._compile_and_capture(fn, x_float)
        gt_key2, graph2, inputs2 = self._compile_and_capture(fn, x_double)
        api_key1, _ = self._aot_autograd_cache_key(graph1, inputs1)
        api_key2, _ = self._aot_autograd_cache_key(graph2, inputs2)
        cfx_key1, _ = self._compile_fx_cache_key(graph1, inputs1)
        cfx_key2, _ = self._compile_fx_cache_key(graph2, inputs2)
        self.assertEqual(gt_key1, api_key1)
        self.assertEqual(gt_key2, api_key2)
        self.assertEqual(gt_key1, cfx_key1)
        self.assertEqual(gt_key2, cfx_key2)
        self.assertNotEqual(gt_key1, gt_key2)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_varies_with_different_graphs(self):
        # [FUNCTION DOC] Test that two different functions give different keys, each matching
        # the real compile key via both APIs.
        """Different model functions should produce different cache keys."""

        def fn1(x):
            # [FUNCTION DOC] HELPER: Returns sin(x) + cos(x).
            return x.sin() + x.cos()

        def fn2(x):
            # [FUNCTION DOC] HELPER: Returns exp(x) * log(x).
            return x.exp() * x.log()

        x = torch.randn(4, 4)

        gt_key1, graph1, inputs1 = self._compile_and_capture(fn1, x)
        gt_key2, graph2, inputs2 = self._compile_and_capture(fn2, x)
        api_key1, _ = self._aot_autograd_cache_key(graph1, inputs1)
        api_key2, _ = self._aot_autograd_cache_key(graph2, inputs2)
        cfx_key1, _ = self._compile_fx_cache_key(graph1, inputs1)
        cfx_key2, _ = self._compile_fx_cache_key(graph2, inputs2)
        self.assertEqual(gt_key1, api_key1)
        self.assertEqual(gt_key2, api_key2)
        self.assertEqual(gt_key1, cfx_key1)
        self.assertEqual(gt_key2, cfx_key2)
        self.assertNotEqual(gt_key1, gt_key2)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_for_module(self):
        # [FUNCTION DOC] Test that the key APIs match the real compile key for an nn.Module
        # (linear layer followed by relu).
        """autograd_cache_key matches compilation for nn.Modules."""

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a 4x4 linear layer; forward returns relu of it.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the 4-to-4 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns relu of the linear layer applied to x.
                return self.linear(x).relu()

        mod = MyModule()
        x = torch.randn(4, 4)
        gt_key, fx_graph, example_inputs = self._compile_and_capture(mod, x)
        api_key, _ = self._aot_autograd_cache_key(fx_graph, example_inputs)
        cfx_key, _ = self._compile_fx_cache_key(fx_graph, example_inputs)

        self.assertEqual(gt_key, api_key)
        self.assertEqual(gt_key, cfx_key)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    def test_cache_key_for_multiple_outputs(self):
        # [FUNCTION DOC] Test that the key APIs match the real compile key for a function
        # returning two outputs.
        """autograd_cache_key matches compilation for multiple outputs."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the tuple (sin(x), cos(x)).
            return x.sin(), x.cos()

        x = torch.randn(4, 4)
        gt_key, fx_graph, example_inputs = self._compile_and_capture(fn, x)
        api_key, _ = self._aot_autograd_cache_key(fx_graph, example_inputs)
        cfx_key, _ = self._compile_fx_cache_key(fx_graph, example_inputs)

        self.assertEqual(gt_key, api_key)
        self.assertEqual(gt_key, cfx_key)

    @requires_triton()
    @inductor_config.patch("fx_graph_remote_cache", False)
    @inductor_config.patch("fx_graph_cache", True)
    @functorch_config.patch(
        {"enable_autograd_cache": True, "strict_autograd_cache": True}
    )
    @parametrize(
        "dynamic_shapes",
        ("from_example_inputs", "from_graph", "from_tracing_context"),
    )
    @parametrize("aot", (False, True))
    def test_standalone_compile_cache_key_matches_standalone_compile(
        self, dynamic_shapes, aot
    ):
        # [FUNCTION DOC] Parametrized test (dynamic_shapes mode and aot flag) that the
        # standalone_compile autograd_cache_key equals the key really used by
        # standalone_compile. Builds an FX graph by hand for sin(x) + cos(x) with fake example
        # values, runs standalone_compile while capturing the key (with a tracing context or
        # fake mode as the mode requires), then computes the key from the standalone API and
        # asserts they are equal.
        """standalone_compile.autograd_cache_key matches the real key
        produced when running standalone_compile end-to-end."""
        from contextlib import nullcontext

        from torch._inductor.standalone_compile import standalone_compile

        # Build an FX graph directly: sin(x) + cos(x)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        with fake_mode:
            fake_x = torch.randn(4, 4)
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["example_value"] = fake_x
        sin = graph.call_method("sin", (x,))
        sin.meta["example_value"] = fake_x.sin()
        cos = graph.call_method("cos", (x,))
        cos.meta["example_value"] = fake_x.cos()
        add = graph.call_function(operator.add, (sin, cos))
        add.meta["example_value"] = fake_x.sin() + fake_x.cos()
        graph.output((add,))
        fx_graph = GraphModule(torch.nn.Module(), graph)
        example_inputs = [torch.randn(4, 4)]

        # from_tracing_context requires an ambient TracingContext, simulating
        # a call from within a torch.compile backend.
        outer_context = (
            torch._guards.tracing(torch._guards.TracingContext(fake_mode))
            if dynamic_shapes == "from_tracing_context"
            else nullcontext()
        )
        standalone_fake_mode = (
            fake_mode if dynamic_shapes == "from_example_inputs" else None
        )

        # Run standalone_compile and capture the real cache key.
        real_cache_key_fn = autograd_cache.autograd_cache_key
        captured_key = None

        def capturing_cache_key(
            mod, ei, config, compiler_config_extra=None, act_input_paths=()
        ):
            # [FUNCTION DOC] HELPER: Calls the real autograd_cache_key, stores its key in the
            # enclosing captured_key and returns the same (key, debug info) pair.
            nonlocal captured_key
            captured_key, _ = real_cache_key_fn(
                mod, ei, config, compiler_config_extra, act_input_paths
            )
            return captured_key, _

        with outer_context:
            with patch(
                "torch._functorch._aot_autograd.autograd_cache.autograd_cache_key",
                side_effect=capturing_cache_key,
            ):
                standalone_compile(
                    fx_graph,
                    example_inputs,
                    dynamic_shapes=dynamic_shapes,
                    options={},
                    aot=aot,
                    fake_mode=standalone_fake_mode,
                )

            self.assertIsNotNone(captured_key)

            sc_key, _ = standalone_compile_autograd_cache_key(
                fx_graph,
                example_inputs,
                dynamic_shapes=dynamic_shapes,
                aot=aot,
                fake_mode=standalone_fake_mode,
            )
            self.assertEqual(captured_key, sc_key)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
