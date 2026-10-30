# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor Triton runtime heuristics: config generation helpers, the
# cached autotuner, autotuner plugins, grid expression classes, warp size handling, and autotune
# cache interaction. Most GPU tests are skipped without a GPU and triton; the whole module exits
# or skips if triton is not importable. Import time defines two small triton kernels and a few
# lru cached helper factories, nothing is run until tests execute.

import functools
import os
import sys
import tempfile
import types
import unittest
from unittest import skipUnless
from unittest.mock import MagicMock, patch, PropertyMock

import torch
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime.triton_compat import HAS_WARP_SPEC
from torch._inductor.utils import clone_preserve_strides
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    runOnRocm,
    skipIfRocm,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
    requires_gpu_with_enough_memory,
)


try:
    import triton  # @manual
    import triton.language as tl  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.hints import (
    AttrsDescriptorWrapper,
    AutotuneHint,
    DeviceProperties,
    HeuristicType,
    native_matmul_block_numel,
    native_matmul_persistent_rblock,
    TRITON_MAX_BLOCK,
    TRITON_MAX_TENSOR_NUMEL,
)
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_heuristics import (
    _check_max_grid_x,
    _enforce_reduction_config_block_minimums,
    _num_warps,
    _persistent_reduction_configs,
    _reduction_configs,
    autotune_hints_to_configs,
    cached_autotune,
    CachingAutotuner,
    CachingAutotunerPlugin,
    check_autotune_cache,
    DEFER,
    make_matmul_triton_config,
    template,
    triton_config,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache


# [FUNCTION DOC] HELPER: Triton jit kernel that squares each element of the input pointer into
# the output pointer for numel elements, one block of BLOCK_SIZE elements per program, masking
# out-of-range offsets. Used to exercise AMD-specific config arguments.
@triton.jit
def amd_sqr_kernel(in_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    data = tl.load(in_ptr + offsets, mask=offsets < numel)
    sqr = data * data
    tl.store(out_ptr + offsets, sqr, mask=offsets < numel)


@functools.lru_cache
def get_autotuned_amd_sqr_kernel():
    # [FUNCTION DOC] HELPER: Cached factory returning the square kernel wrapped in triton
    # autotune with a single config whose BLOCK_SIZE is 64 and which also sets the AMD-only
    # waves_per_eu option (3). The result is built once per process through the lru cache.
    return triton.autotune(
        configs=[
            triton.Config(
                {
                    "BLOCK_SIZE": 64,
                    "waves_per_eu": 3,
                }
            )
        ],
        key=[],
    )(amd_sqr_kernel)


@instantiate_parametrized_tests
class TestTritonHeuristics(TestCase):
    # [CLASS DOC] Tests for config helpers in triton_heuristics (triton_config, reduction and
    # persistent reduction configs, cached_autotune, check_autotune_cache, template) and for
    # CachingAutotuner construction. device_type is set to the GPU under test; the class also
    # provides static helper fixtures reused by later test classes.
    device_type = GPU_TYPE

    def test_triton_config(self):
        # [FUNCTION DOC] Checks that triton_config clamps each requested block dimension so the
        # resulting kernel arguments never exceed the per-dimension maximum in TRITON_MAX_BLOCK.
        """
        Make sure block size does not exceed the maximum defined in inductor config.
        """
        cfg = triton_config({"x": 2048, "y": 2}, 64, 64)
        for label in "XYZ":
            key = f"{label}BLOCK"
            if key not in cfg.kwargs:
                continue
            self.assertTrue(cfg.kwargs[key] <= TRITON_MAX_BLOCK[label])

    def test_native_matmul_config_block_numel_limit(self):
        # [FUNCTION DOC] Builds a fake CUDA device description and verifies that every config
        # from the reduction and persistent reduction config generators for native matmul, with
        # 2D and 3D size hints and an optional persistent rblock override, keeps the block
        # element count within TRITON_MAX_TENSOR_NUMEL. Also checks that
        # make_matmul_triton_config raises an AssertionError mentioning the Triton maximum when
        # given oversized blocks.
        device = DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=1,
            cc=80,
            major=8,
            max_threads_per_block=1024,
            warp_size=32,
        )
        triton_meta = {"native_matmul": True, "device": device}

        for size_hints in (
            {"x": 1, "y": 1, "r0_": 1},
            {"x": 1, "y": 1, "z": 1, "r0_": 1},
        ):
            cfgs = _reduction_configs(
                size_hints=size_hints,
                inductor_meta={},
                triton_meta=triton_meta,
            )
            self.assertTrue(cfgs)
            for cfg in cfgs:
                self.assertLessEqual(
                    native_matmul_block_numel(cfg.kwargs),
                    TRITON_MAX_TENSOR_NUMEL,
                )

        for size_hints, inductor_meta in (
            ({"x": 4096, "y": 4096, "r0_": 1}, {}),
            ({"x": 16384, "y": 2048, "r0_": 2048}, {}),
            (
                {"x": 4096, "y": 4096, "r0_": 64},
                {"native_matmul_persistent_rblock": 1024},
            ),
            ({"x": 4096, "y": 4096, "z": 128, "r0_": 1}, {}),
            ({"x": 16384, "y": 2048, "z": 128, "r0_": 2048}, {}),
            (
                {"x": 4096, "y": 4096, "z": 128, "r0_": 64},
                {"native_matmul_persistent_rblock": 1024},
            ),
        ):
            cfgs = _persistent_reduction_configs(
                size_hints=size_hints,
                inductor_meta=inductor_meta,
                triton_meta=triton_meta,
            )
            rblock = inductor_meta.get(
                "native_matmul_persistent_rblock",
                native_matmul_persistent_rblock(size_hints["r0_"]),
            )
            self.assertTrue(cfgs)
            for cfg in cfgs:
                self.assertLessEqual(
                    native_matmul_block_numel(cfg.kwargs, r0_block=rblock),
                    TRITON_MAX_TENSOR_NUMEL,
                )

        with self.assertRaisesRegex(AssertionError, "exceeds Triton maximum"):
            make_matmul_triton_config({"x": 256, "y": 128, "r": 64}, 8, 1)

    def test_reduction_min_block_preserves_tile_product(self):
        # [FUNCTION DOC] Checks that enforcing a minimum XBLOCK or R0_BLOCK on a reduction
        # config raises the small dimension to the minimum and shrinks the other dimension so
        # the product of the tile sizes is preserved (128 and 512, then 512 and 128).
        cfg = _enforce_reduction_config_block_minimums(
            [triton.Config({"XBLOCK": 64, "R0_BLOCK": 1024})],
            {"x": 4096, "r0_": 4096},
            {"min_xblock": 128},
        )[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 128)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 512)

        cfg = _enforce_reduction_config_block_minimums(
            [triton.Config({"XBLOCK": 1024, "R0_BLOCK": 64})],
            {"x": 4096, "r0_": 4096},
            {"min_rblock": 128},
        )[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 512)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 128)

    def test_cached_autotune_enforces_reduction_min_block(self):
        # [FUNCTION DOC] Verifies that cached_autotune applies the reduction minimum block
        # enforcement from inductor_meta to configs before constructing the autotuner class; a
        # capturing autotuner class records the configs it receives and they are asserted to be
        # 128 and 512.
        def triton_fn(XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
            # [FUNCTION DOC] HELPER: Empty stand-in for a triton kernel function with XBLOCK and
            # R0_BLOCK constexpr parameters; only its signature matters.
            pass

        class FakeJitFunction:
            # [CLASS DOC] HELPER: Minimal stand-in for a triton JIT function; holds the fake
            # kernel in the fn attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the fake kernel function on the instance as fn.
                self.fn = triton_fn

        class CaptureAutotuner:
            # [CLASS DOC] HELPER: Replacement for the caching autotuner class that only records
            # the configs keyword argument it was constructed with.
            def __init__(self, *args, configs, **kwargs):
                # [FUNCTION DOC] HELPER: Accepts and ignores all arguments except the required
                # configs keyword, which is saved in the configs attribute.
                self.configs = configs

        autotuner = cached_autotune(
            {"x": 4096, "r0_": 4096},
            [triton.Config({"XBLOCK": 64, "R0_BLOCK": 1024})],
            triton_meta={},
            heuristic_type=HeuristicType.REDUCTION,
            inductor_meta={"min_xblock": 128},
            caching_autotuner_cls=CaptureAutotuner,
        )(FakeJitFunction())

        cfg = autotuner.configs[0]
        self.assertEqual(cfg.kwargs["XBLOCK"], 128)
        self.assertEqual(cfg.kwargs["R0_BLOCK"], 512)

    @staticmethod
    def _fake_cuda_device_properties():
        # [FUNCTION DOC] HELPER: Static fixture returning a DeviceProperties for a fake CUDA
        # device (index 0, compute capability 80, 1024 threads per block, warp size 32) so tests
        # can run without a real device.
        return DeviceProperties(
            type="cuda",
            index=0,
            multi_processor_count=1,
            cc=80,
            major=8,
            regs_per_multiprocessor=65536,
            max_threads_per_multi_processor=2048,
            max_threads_per_block=1024,
            warp_size=32,
        )

    @staticmethod
    def _run_single_dsr_reduction_cached_autotune(configs, fake_cache):
        # [FUNCTION DOC] HELPER: Static fixture that runs cached_autotune on a reduction kernel
        # with the given configs, with AutotuneCache.create patched to return the supplied fake
        # cache. A capturing autotuner class records configs, the save cache hook and the
        # autotune cache info. Returns the captured autotuner and the patched create mock.
        def triton_fn(XBLOCK: tl.constexpr, R0_BLOCK: tl.constexpr):
            # [FUNCTION DOC] HELPER: Empty stand-in kernel function with XBLOCK and R0_BLOCK
            # constexpr parameters.
            pass

        class FakeJitFunction:
            # [CLASS DOC] HELPER: Minimal stand-in for a triton JIT function holding the fake
            # kernel in fn.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Stores the fake kernel function on the instance as fn.
                self.fn = triton_fn

        class CaptureAutotuner:
            # [CLASS DOC] HELPER: Replacement autotuner class that records the configs,
            # save_cache_hook and autotune_cache_info keyword arguments it receives.
            def __init__(
                self, *args, configs, save_cache_hook, autotune_cache_info, **kwargs
            ):
                # [FUNCTION DOC] HELPER: Saves the configs, save_cache_hook and
                # autotune_cache_info keyword arguments as attributes and ignores
                # everything else.
                self.configs = configs
                self.save_cache_hook = save_cache_hook
                self.autotune_cache_info = autotune_cache_info

        with patch(
            "torch._inductor.runtime.triton_heuristics.AutotuneCache.create",
            return_value=fake_cache,
        ) as create:
            autotuner = cached_autotune(
                {"x": 4096, "r0_": 4096},
                configs,
                triton_meta={
                    "device": TestTritonHeuristics._fake_cuda_device_properties()
                },
                heuristic_type=HeuristicType.REDUCTION,
                filename="/tmp/kernel.py",
                inductor_meta={},
                caching_autotuner_cls=CaptureAutotuner,
            )(FakeJitFunction())

        return autotuner, create

    def test_cached_autotune_uses_cache_for_single_dsr_reduction_config(self):
        # [FUNCTION DOC] Checks that a single reduction config eligible for dynamic rblock
        # scaling still goes through the autotune cache: create is called once, the save hook is
        # the cache save method, the cache state is miss when read_best returns nothing, and one
        # config is recorded.
        cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)
        fake_cache = MagicMock()
        fake_cache.read_best.return_value = None

        autotuner, create = self._run_single_dsr_reduction_cached_autotune(
            [cfg], fake_cache
        )

        create.assert_called_once()
        self.assertIs(autotuner.save_cache_hook, fake_cache.save)
        self.assertEqual(autotuner.autotune_cache_info["autotune_cache_state"], "miss")
        self.assertEqual(autotuner.autotune_cache_info["num_configs"], 1)

    def test_cached_autotune_loads_single_dsr_reduction_cache_hit(self):
        # [FUNCTION DOC] Checks that when the autotune cache holds a different best config for a
        # single dynamic-rblock-eligible reduction config, the autotuner receives only that
        # cached config and the cache state is hit.
        original_cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)
        dynamic_cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=16)

        fake_cache = MagicMock()
        fake_cache.read_best.return_value = dynamic_cfg

        autotuner, _ = self._run_single_dsr_reduction_cached_autotune(
            [original_cfg], fake_cache
        )

        self.assertEqual(autotuner.configs, [dynamic_cfg])
        self.assertEqual(autotuner.autotune_cache_info["autotune_cache_state"], "hit")

    def test_check_autotune_cache_skips_single_config_without_dsr(self):
        # [FUNCTION DOC] Checks that check_autotune_cache with one config and no dynamic rblock
        # eligibility does not create a cache: it returns the config unchanged, no cache object,
        # and a cache state of only 1 config.
        cfg = triton.Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16)

        with patch(
            "torch._inductor.runtime.triton_heuristics.AutotuneCache.create"
        ) as create:
            configs, autotune_cache, autotune_cache_info = check_autotune_cache(
                [cfg],
                "/tmp/kernel.py",
                {},
                dynamic_scale_rblock_eligible=False,
            )

        create.assert_not_called()
        self.assertEqual(configs, [cfg])
        self.assertIsNone(autotune_cache)
        self.assertEqual(autotune_cache_info["autotune_cache_state"], "only 1 config")

    def _test_artificial_zgrid(self):
        # [FUNCTION DOC] HELPER: Shared body for the artificial grid tests. Compiles a function
        # (reshape, permute, clone, addmm) with a dynamic first dimension of 16777472 rows so
        # the grid exceeds hardware limits, and asserts compiled results equal eager results on
        # two input sets.
        def forward(primals_1, primals_2, primals_5):
            # [FUNCTION DOC] HELPER: Eager reference function performing reshape, permute,
            # contiguous clone, reshape and addmm on its three inputs; returns the addmm result.
            view = torch.ops.aten.reshape.default(primals_5, [-1, 2, 4])
            primals_5 = None
            permute = torch.ops.aten.permute.default(view, [0, 2, 1])
            clone = torch.ops.aten.clone.default(
                permute, memory_format=torch.contiguous_format
            )
            permute = None
            view_1 = torch.ops.aten.reshape.default(clone, [-1, 4])
            clone = None
            permute_1 = torch.ops.aten.permute.default(primals_1, [1, 0])
            primals_1 = None
            addmm = torch.ops.aten.addmm.default(primals_2, view_1, permute_1)
            primals_2 = None
            return addmm

        s0 = 16777472
        s1 = 8

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        torch._dynamo.mark_dynamic(args[-1], 0)
        foo_c = torch.compile(forward)

        self.assertEqual(forward(*args), foo_c(*args))

        args = [
            torch.rand([2, 4], device=GPU_TYPE),
            torch.rand([2], device=GPU_TYPE),
            torch.rand([s0, s1], device=GPU_TYPE),
        ]
        self.assertEqual(forward(*args), foo_c(*args))

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_artificial_zgrid(self):
        # [FUNCTION DOC] Runs the artificial large grid comparison against eager execution;
        # requires a GPU and triton.
        self._test_artificial_zgrid()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @config.patch("cpp_wrapper", True)
    def test_artificial_grid_cpp_wrapper(self):
        # [FUNCTION DOC] Runs the artificial large grid comparison with the C++ wrapper config
        # option enabled; requires a GPU and triton.
        self._test_artificial_zgrid()

    @staticmethod
    def _get_cos_kernel_caching_autotuner_args():
        # [FUNCTION DOC] HELPER: Static fixture returning the keyword arguments needed to
        # construct a CachingAutotuner for a small cosine triton kernel over 16 elements: the
        # kernel, triton meta with signature and device, two pointwise configs (block 16 and
        # 256), no save hook, memory optimization on, and empty inductor meta.
        # [FUNCTION DOC] HELPER: Nested triton jit kernel computing the cosine of 16 input
        # floats into the output pointer, masked by index.
        @triton.jit
        def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
            xnumel = 16
            xoffset = tl.program_id(0) * XBLOCK
            xindex = xoffset + tl.arange(0, XBLOCK)[:]
            xmask = xindex < xnumel
            x0 = xindex
            tmp0 = tl.load(in_ptr0 + (x0), xmask)
            tmp1 = tl_math.cos(tmp0)
            tl.store(out_ptr0 + (x0), tmp1, xmask)

        triton_meta = {
            "signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"},
            "device": DeviceProperties.create(torch.device(GPU_TYPE)),
            "constants": {},
            "configs": [
                AttrsDescriptorWrapper(divisible_by_16=(0, 1, 2), equal_to_1=())
            ],
        }

        configs = [
            triton_config({"x": 16}, 64),
            triton_config({"x": 256}, 64),
        ]

        inductor_meta = {}

        return {
            "fn": triton_,
            "triton_meta": triton_meta,
            "configs": configs,
            "save_cache_hook": False,
            "mutated_arg_names": [],
            "reset_to_zero_arg_names": [],
            "optimize_mem": True,
            "heuristic_type": HeuristicType.POINTWISE,
            "inductor_meta": inductor_meta,
        }

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_pre_hook_assert(self):
        # [FUNCTION DOC] Checks that constructing a CachingAutotuner raises an AssertionError
        # mentioning pre_hook when any config has a pre-hook attached.
        # assert if any of the configs passed to the CachingAutotuner have pre-hooks
        args = self._get_cos_kernel_caching_autotuner_args()

        def pre_hook(kwargs):
            # [FUNCTION DOC] HELPER: Pre-hook that zeroes the in_ptr0 tensor in place when
            # present in the kwargs; never run, only attached to configs.
            if "in_ptr0" in kwargs:
                kwargs["in_ptr0"].zero_()

        for cfg in args["configs"]:
            cfg.pre_hook = pre_hook

        with self.assertRaisesRegex(AssertionError, "pre_hook"):
            CachingAutotuner(**args)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_autotune_hints_to_configs(self):
        # [FUNCTION DOC] Checks that autotune_hints_to_configs with the one element per thread
        # hint passes the device warp size (patched to 8) as the elements per warp value to
        # triton_config; triton_config is mocked and the seen values are asserted to include 8.
        device_props = DeviceProperties.create(torch.device(GPU_TYPE))
        device_props = device_props._replace(warp_size=8)

        hints = {AutotuneHint.ONE_ELEMENT_PER_THREAD}
        size_hints = (1024,)
        block_size = 256

        seen_num_elements_per_warp = set()

        def mock_triton_config(
            size_hints,
            x,
            y=None,
            z=None,
            num_stages=None,
            num_elements_per_warp=None,
            min_elem_per_thread=None,
            *,
            warp_size=32,
        ):
            # [FUNCTION DOC] HELPER: Mock replacement for triton_config that records
            # num_elements_per_warp in a set and returns None.
            seen_num_elements_per_warp.add(num_elements_per_warp)
            return None

        with unittest.mock.patch(
            "torch._inductor.runtime.triton_heuristics.triton_config",
            mock_triton_config,
        ):
            _ = autotune_hints_to_configs(hints, size_hints, block_size, device_props)

        self.assertTrue(8 in seen_num_elements_per_warp)

    @unittest.skipIf(not HAS_WARP_SPEC, "FBCODE Triton is required for this test")
    def test_template_function_ws(self):
        # [FUNCTION DOC] Checks that the template helper forwards warp specialization options
        # (consumer groups and warp spec buffer counts) into the first config passed to
        # cached_autotune; skipped without warp specialization support.
        triton_meta = {"device": MagicMock()}
        num_stages = 2
        num_warps = 4
        num_consumer_groups = 3
        num_buffers_warp_spec = 5

        with patch(
            "torch._inductor.runtime.triton_heuristics.cached_autotune"
        ) as mock_cached_autotune:
            template(
                num_stages=num_stages,
                num_warps=num_warps,
                triton_meta=triton_meta,
                num_consumer_groups=num_consumer_groups,
                num_buffers_warp_spec=num_buffers_warp_spec,
            )
            mock_cached_autotune.assert_called_once()
            configs = mock_cached_autotune.call_args[0][1]
            self.assertEqual(configs[0].num_consumer_groups, num_consumer_groups)
            self.assertEqual(configs[0].num_buffers_warp_spec, num_buffers_warp_spec)

    @runOnRocm
    def test_amd_special_config_args(self):
        # [FUNCTION DOC] Compiles and runs a custom triton op using an autotuned AMD kernel that
        # sets waves_per_eu and checks the compiled output matches eager; runs only on ROCm.
        """
        waves_per_eu is an example of a special config arg on AMD; if it is explicitly specified
        in a config, the kwarg will exist in the kwargs but not in the function signature.
        """

        @torch.library.triton_op("test_triton_heuristics::triton_sqr", mutates_args=())
        def triton_sqr(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Registered triton op that allocates an output like the
            # input and launches the autotuned AMD square kernel over a grid derived from
            # BLOCK_SIZE.
            # KNOWN ISSUE: it never returns the output tensor, so its result is None and the
            # test compares the None results of eager and compiled runs.
            y = torch.empty_like(x)

            def grid(meta):
                # [FUNCTION DOC] HELPER: Grid callback returning a single-element tuple with the
                # ceiling division of the element count by the chosen BLOCK_SIZE.
                return (triton.cdiv(x.numel(), meta["BLOCK_SIZE"]),)

            torch.library.wrap_triton(get_autotuned_amd_sqr_kernel())[grid](
                x, y, x.numel()
            )

        def fn(x):
            # [FUNCTION DOC] HELPER: Thin wrapper calling the custom triton op.
            return triton_sqr(x)

        x = torch.randn(32, device=GPU_TYPE)
        ref = fn(x)
        res = torch.compile(fn)(x)
        self.assertEqual(ref, res)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    @parametrize("do_pruning", [False, True])
    def test_prune_configs_over_shared_memory_limit(self, do_pruning):
        # [FUNCTION DOC] Parametrized on do_pruning. Overrides a config heuristic (XPU, ROCm or
        # CUDA by platform) with one small and one oversized GEMM config and checks that the
        # shared memory pruning option leaves one config when enabled and two when disabled.
        from torch._inductor.heuristics.template.triton import (
            CUDAConfigHeuristic,
            GemmConfig,
            ROCmConfigHeuristic,
            XPUConfigHeuristic,
        )

        expected_count = 1 if do_pruning else 2
        mm_configs = [
            GemmConfig(32, 32, 32, 1, 8, group_m=8),
            GemmConfig(
                128, 128, 128, 100, 8, group_m=4
            ),  # intentionally large to exceed shared memory limit
        ]
        with config.patch(
            {"max_autotune_prune_choices_based_on_shared_mem": do_pruning}
        ):
            if GPU_TYPE == "xpu":
                config_heuristic = XPUConfigHeuristic()
            elif torch.version.hip:
                config_heuristic = ROCmConfigHeuristic()
            else:
                config_heuristic = CUDAConfigHeuristic()
            config_heuristic.should_scale_configs = False
            config_heuristic.mm_configs = mm_configs
            configs = list(
                config_heuristic.get_mm_configs()(3, 3, 3, dtype_size=4, op_name="mm")
            )
            self.assertEqual(len(configs), expected_count)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_compile_time_autotune_not_repeated_at_runtime(self):
        # [FUNCTION DOC] Compiles a pointwise plus reduction function with compile-time
        # autotuning enabled and a patched benchmark function; asserts the benchmark is invoked
        # exactly once, including after a second call, so autotuning is not repeated at runtime.
        def fn(x):
            # [FUNCTION DOC] HELPER: Test function adding one to the input and summing
            # over dim 1.
            return (x + 1).sum(dim=1)

        x = torch.randn(2048, 2048, device=GPU_TYPE)
        benchmark_calls = []

        def fake_benchmark_all_configs(autotuner, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for benchmark_all_configs that records the
            # kernel name and returns increasing fake timings per launcher.
            benchmark_calls.append(autotuner.inductor_meta.get("kernel_name"))
            return {
                launcher: float(idx) for idx, launcher in enumerate(autotuner.launchers)
            }

        torch._dynamo.reset()
        with (
            fresh_cache(),
            config.patch(
                {
                    "triton.autotune_at_compile_time": True,
                    "max_autotune_pointwise": True,
                    "compile_threads": 1,
                }
            ),
            patch.object(
                CachingAutotuner,
                "benchmark_all_configs",
                fake_benchmark_all_configs,
            ),
        ):
            compiled = torch.compile(fn, backend="inductor")
            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(len(benchmark_calls), 1)

            self.assertEqual(compiled(x), fn(x))
            self.assertEqual(len(benchmark_calls), 1)


_PLUGIN_FACTORY_PATH = (
    "torch._inductor.runtime.triton_heuristics.get_caching_autotuner_plugins"
)


class TestCachingAutotunerPrecompileDriverSetup(TestCase):
    # [CLASS DOC] Tests for CachingAutotuner.precompile and GPU driver setup.
    @skipIfRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_warm_cache_only_precompile_skips_driver_setup_in_context(self):
        # [FUNCTION DOC] Checks that precompile with warm_cache_only inside the skip driver
        # setup context compiles every config without reading the active triton driver (property
        # patched to raise), and that compile_results has one entry per config.
        from torch._inductor.runtime import triton_helpers

        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        num_configs = len(args["configs"])
        self.assertGreaterEqual(num_configs, 2)
        autotuner = CachingAutotuner(**args)

        with (
            triton_helpers.skip_gpu_driver_setup(),
            patch.object(
                type(triton.runtime.driver),
                "active",
                new_callable=PropertyMock,
                side_effect=RuntimeError("driver.active should not be read"),
            ) as mock_driver_active,
        ):
            autotuner.precompile(warm_cache_only=True)

        # Covers every synchronous _precompile_config() iteration.
        mock_driver_active.assert_not_called()
        self.assertEqual(len(autotuner.compile_results), num_configs)


# Triton's HIP MLIR pipeline raises AttributeError("'NoneType' object has no
# attribute '_unflatten_ir'") inside ast_to_ttir for the trivial cos kernel
# used by these tests. CUDA paths are unaffected.
@skipIfRocm
@skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
class TestCachingAutotunerPlugin(TestCase):
    # [CLASS DOC] Tests for CachingAutotunerPlugin hooks (pre_dispatch, pre_autotune,
    # pre_compile) and plugin handling in pickle state. Skipped on ROCm and without GPU
    # and triton.
    device_type = GPU_TYPE

    @staticmethod
    def _get_stream():
        # [FUNCTION DOC] HELPER: Static helper returning the raw stream handle of the current
        # GPU device.
        from torch._dynamo.device_interface import get_interface_for_device

        device_interface = get_interface_for_device(GPU_TYPE)
        return device_interface.get_raw_stream(device_interface.current_device())

    @staticmethod
    def _make_kernel_inputs():
        # [FUNCTION DOC] HELPER: Static helper returning the input pointer tensor, an output
        # tensor (16 float zeros each on the GPU) and the element count 16.
        in_ptr = torch.zeros(16, device=GPU_TYPE, dtype=torch.float32)
        out_ptr = torch.zeros(16, device=GPU_TYPE, dtype=torch.float32)
        return (in_ptr, out_ptr, 16)

    def _make_autotuner(self, plugins, configs=None):
        # [FUNCTION DOC] HELPER: Builds a CachingAutotuner from the cosine kernel fixture with
        # grid type Grid1D, optionally replacing configs, while the plugin factory is patched to
        # return the given plugins.
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["inductor_meta"] = {**args["inductor_meta"], "grid_type": "Grid1D"}
        if configs is not None:
            args["configs"] = configs
        with patch(_PLUGIN_FACTORY_PATH, return_value=list(plugins)):
            return CachingAutotuner(**args)

    def test_pre_dispatch_runs_before_precompile_and_autotune(self):
        # [FUNCTION DOC] Checks that a plugin returning a value from pre_dispatch makes run
        # return it without precompiling or autotuning.
        sentinel = object()

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin whose pre_dispatch hook returns a sentinel.
            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                # [FUNCTION DOC] HELPER: Returns the enclosing sentinel object, ignoring
                # arguments.
                return sentinel

        autotuner = self._make_autotuner([_Plugin()])
        with (
            patch.object(autotuner, "precompile") as mock_precompile,
            patch.object(autotuner, "autotune_to_one_config") as mock_autotune,
        ):
            result = autotuner.run(
                *self._make_kernel_inputs(), stream=self._get_stream()
            )

        self.assertIs(result, sentinel)
        mock_precompile.assert_not_called()
        mock_autotune.assert_not_called()

    def test_pre_autotune_runs_before_default_autotune(self):
        # [FUNCTION DOC] Checks that a plugin returning a value from pre_autotune makes run
        # return it and the default autotune to one config step is skipped.
        sentinel = object()

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin whose pre_autotune hook returns a sentinel.
            def pre_autotune(self, autotuner, *args, stream, **kwargs):
                # [FUNCTION DOC] HELPER: Returns the enclosing sentinel object, ignoring
                # arguments.
                return sentinel

        autotuner = self._make_autotuner([_Plugin()])
        with patch.object(autotuner, "autotune_to_one_config") as mock_autotune:
            result = autotuner.run(
                *self._make_kernel_inputs(), stream=self._get_stream()
            )

        self.assertIs(result, sentinel)
        mock_autotune.assert_not_called()

    def test_hooks_fire_in_registration_order(self):
        # [FUNCTION DOC] Checks that pre_dispatch hooks run in registration order and stop at
        # the first non-DEFER result: four plugins are registered, the third returns the
        # sentinel, so only the first three names are seen.
        sentinel = object()
        seen = []

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin with a name and a configurable return value (default
            # DEFER) whose pre_dispatch records its name.
            def __init__(self, name, return_value=DEFER):
                # [FUNCTION DOC] HELPER: Stores the plugin name and the value its hook
                # will return.
                self.name = name
                self.return_value = return_value

            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                # [FUNCTION DOC] HELPER: Appends the plugin name to the shared list and returns
                # the configured value.
                seen.append(self.name)
                return self.return_value

        autotuner = self._make_autotuner(
            [_Plugin("a"), _Plugin("b"), _Plugin("c", sentinel), _Plugin("d")]
        )
        result = autotuner.run(*self._make_kernel_inputs(), stream=self._get_stream())

        self.assertIs(result, sentinel)
        self.assertEqual(seen, ["a", "b", "c"])

    def test_defer_falls_through_to_default(self):
        # [FUNCTION DOC] Checks that when a plugin defers in pre_dispatch the default path
        # still runs: with a single config the kernel runs, the hook was called once and
        # one launcher exists.
        seen = []

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin that records a pre_dispatch call and returns DEFER.
            def pre_dispatch(self, autotuner, *args, stream, **kwargs):
                # [FUNCTION DOC] HELPER: Records the call in the shared list and returns DEFER.
                seen.append("pre_dispatch")
                return DEFER

        # Single config so precompile produces one launcher and the kernel
        # runs to completion without the autotune branch firing.
        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])

        autotuner.run(*self._make_kernel_inputs(), stream=self._get_stream())

        self.assertEqual(seen, ["pre_dispatch"])
        self.assertEqual(len(autotuner.launchers), 1)

    def test_pre_compile_defer_runs_default_precompile_flow(self):
        # [FUNCTION DOC] Checks that DEFER from pre_compile lets precompile run the standard
        # worker, launcher creation and dynamic rblock scaling steps once each (all patched).
        """``DEFER`` from pre_compile lets the standard precompile flow
        run (``_precompile_worker`` → ``_make_launchers`` →
        ``_dynamic_scale_rblock``)."""

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin whose pre_compile hook returns DEFER.
            def pre_compile(self, autotuner):
                # [FUNCTION DOC] HELPER: Returns DEFER.
                return DEFER

        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])
        with (
            patch.object(autotuner, "_precompile_worker") as mock_pcw,
            patch.object(autotuner, "_make_launchers") as mock_mkl,
            patch.object(autotuner, "_dynamic_scale_rblock") as mock_dsr,
        ):
            autotuner.precompile()
        mock_pcw.assert_called_once()
        mock_mkl.assert_called_once()
        mock_dsr.assert_called_once()

    def test_pre_compile_non_defer_short_circuits_precompile(self):
        # [FUNCTION DOC] Checks that a non-DEFER result from pre_compile makes precompile return
        # early without running the worker, launcher creation or dynamic rblock steps.
        """A non-``DEFER`` return from pre_compile means the plugin
        owns the entire compile pipeline; ``precompile`` returns early
        without running ``_precompile_worker`` / ``_make_launchers`` /
        ``_dynamic_scale_rblock``."""

        class _Plugin(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Plugin whose pre_compile hook returns an arbitrary
            # non-DEFER object.
            def pre_compile(self, autotuner):
                # [FUNCTION DOC] HELPER: Returns a new plain object, i.e. a non-DEFER value.
                return object()  # any non-DEFER value

        full_args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        autotuner = self._make_autotuner([_Plugin()], configs=full_args["configs"][:1])
        with (
            patch.object(autotuner, "_precompile_worker") as mock_pcw,
            patch.object(autotuner, "_make_launchers") as mock_mkl,
            patch.object(autotuner, "_dynamic_scale_rblock") as mock_dsr,
        ):
            autotuner.precompile()
        mock_pcw.assert_not_called()
        mock_mkl.assert_not_called()
        mock_dsr.assert_not_called()

    def test_getstate_clears_plugins_setstate_invokes_factory(self):
        # [FUNCTION DOC] Checks pickle state handling: the state dict from __getstate__ has an
        # empty plugin list, and a new autotuner restored with __setstate__ gets its plugins
        # from the factory, which is an empty list when unpatched.
        """``__getstate__`` drops the live plugin instances so they
        don't pickle into workers; ``__setstate__`` re-runs
        ``get_caching_autotuner_plugins`` so the revived autotuner
        gets a fresh, factory-driven list (whose contents reflect the
        unpickling process's config flags, not the pickling process's)."""

        class _PluginA(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Empty plugin subclass used to populate the plugin list.
            pass

        class _PluginB(CachingAutotunerPlugin):
            # [CLASS DOC] HELPER: Second empty plugin subclass used to populate the plugin list.
            pass

        autotuner = self._make_autotuner([_PluginA(), _PluginB()])
        self.assertEqual(len(autotuner._plugins), 2)

        # State excludes the live plugin instances.
        state = autotuner.__getstate__()
        self.assertEqual(state["_plugins"], [])

        # __setstate__ re-invokes the factory; with no patch active the
        # default factory returns an empty list.
        revived = CachingAutotuner.__new__(CachingAutotuner)
        revived.__setstate__(state)
        self.assertEqual(revived._plugins, [])


class TestArgumentCloneAndRestore(TestCase):
    # [CLASS DOC] Tests that copying mutated arguments to CPU and restoring them in
    # CachingAutotuner preserves size, stride, storage offset and values without extra peak GPU
    # memory. MEM_TOLERANCE is the allowed peak memory growth in bytes (256e6).
    # Our tensor is large enough. If a unexpected copy happens, the
    # peak memory increase should be larger than tolerance and the test
    # will fail.
    MEM_TOLERANCE = int(256 * 1e6)

    def _create_caching_autotuner(self):
        # [FUNCTION DOC] HELPER: Creates a CachingAutotuner from the cosine kernel fixture with
        # memory optimization on and in_ptr0 marked as mutated; returns it.
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["optimize_mem"] = True
        args["mutated_arg_names"] = ["in_ptr0"]
        autotuner = CachingAutotuner(**args)
        return autotuner

    def _create_tensor(self, pad=1, with_offset=False):
        # [FUNCTION DOC] HELPER: Creates a roughly 1GB 2 by (2**29 // 4) float tensor on the GPU
        # with row stride N plus pad; with with_offset it slices off the first column giving a
        # non-zero storage offset.
        """
        Create a GPU tensor of about 1GB size.
        """
        M = 2
        N = 2**29 // 4
        out = rand_strided((M, N), (N + pad, 1), device=GPU_TYPE)
        if with_offset:
            out = out[:, 1:]
        return out

    def _do_test(self, gpu_tensor):
        # [FUNCTION DOC] HELPER: Core check. Clones the tensor, copies it to CPU through the
        # autotuner, mutates the original in place, restores it, then asserts it is a
        # distinct object from the clone with equal shape, stride, values and original storage
        # offset, and that peak memory stayed within MEM_TOLERANCE and under 1e10 bytes.
        torch.get_device_module(GPU_TYPE).reset_peak_memory_stats()
        autotuner = self._create_caching_autotuner()

        old_storage_offset = gpu_tensor.storage_offset()
        gpu_tensor_clone = clone_preserve_strides(gpu_tensor)

        peak_mem_before = torch.get_device_module(GPU_TYPE).max_memory_allocated()
        cpu_copies = autotuner.copy_args_to_cpu_if_needed(gpu_tensor)
        self.assertTrue(len(cpu_copies) == 1)

        # Mutate the arg
        gpu_tensor.add_(1)

        # will restore gpu_tensor
        autotuner.restore_args_from_cpu(cpu_copies)
        self.assertTrue(gpu_tensor is not gpu_tensor_clone)
        self.assertEqual(gpu_tensor.size(), gpu_tensor_clone.size())
        self.assertEqual(gpu_tensor.stride(), gpu_tensor_clone.stride())
        self.assertEqual(gpu_tensor.storage_offset(), old_storage_offset)

        # Note: torch.allclose somehow allocates large amount of extra memory.
        # Record peak memory before that.
        peak_mem_after = torch.get_device_module(GPU_TYPE).max_memory_allocated()

        self.assertTrue(torch.allclose(gpu_tensor, gpu_tensor_clone))
        self.assertTrue(
            peak_mem_after <= peak_mem_before + self.MEM_TOLERANCE,
            lambda msg: f"{msg}\n{peak_mem_before=} v.s. {peak_mem_after=}",
        )

        # Avoid OOM in CI
        self.assertTrue(peak_mem_after < 1e10)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_contiguous_args(self):
        # [FUNCTION DOC] Runs the clone and restore check on a contiguous tensor with zero
        # offset; needs a GPU with about 1e10 bytes.
        arg = self._create_tensor(pad=0)
        self.assertTrue(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_non_contiguous_args(self):
        # [FUNCTION DOC] Runs the clone and restore check on a non-contiguous tensor with
        # zero offset.
        arg = self._create_tensor(pad=1)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() == 0)
        self._do_test(arg)

    @requires_gpu_with_enough_memory(1e10)
    def test_clone_args_with_non_zero_offset(self):
        # [FUNCTION DOC] Runs the clone and restore check on a non-contiguous tensor with
        # non-zero storage offset.
        arg = self._create_tensor(pad=1, with_offset=True)
        self.assertFalse(arg.is_contiguous())
        self.assertTrue(arg.storage_offset() > 0)

        self._do_test(arg)


class TestDumpLaunchTensors(TestCase):
    # [CLASS DOC] Tests for the debug option that dumps kernel launch input tensors to the
    # cache directory.
    """Test the _dump_launch_tensors functionality"""

    def setUp(self):
        # [FUNCTION DOC] Creates a temporary directory stored on the test instance after
        # base setup.
        super().setUp()
        # Create a temporary directory for test dumps
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # [FUNCTION DOC] Removes the temporary directory if it exists, then runs base teardown.
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        super().tearDown()

    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_dump_launch_tensors(self):
        # [FUNCTION DOC] Compiles a small model with the TORCHINDUCTOR_DUMP_LAUNCH_TENSORS
        # environment variable set to 1 and runs it more than the max dump occurrence count.
        # Scans the cache directory for run directories, loads one saved tensor to verify it is
        # a GPU tensor, and checks each kernel has no more run directories than the maximum and
        # indices below it. Restores the environment variable afterwards.
        """
        Test that dump_launch_tensors functions correctly:
        1. Creates the dump directory when torch.compile() runs
        2. Saves tensor files that can be loaded
        3. Loads tensors to match the original values
        4. Properly rotates when max_kernel_dump_occurrences is reached
        """
        from torch._inductor.config import triton as inductor_triton_config
        from torch._inductor.runtime.runtime_utils import cache_dir

        # Clear any existing state
        inductor_triton_config.debug_dump_kernel_inputs.clear()

        # Define a simple function that will generate Triton kernels
        def simple_model(x):
            # [FUNCTION DOC] HELPER: Model multiplying by 2, adding 1 and summing the result.
            y = x * 2.0
            z = y + 1.0
            return z.sum()

        old_dump_env = os.environ.get("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS")
        os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = "1"

        try:
            compiled_fn = torch.compile(simple_model)

            # Run the compiled function multiple times to test rotation
            max_runs = inductor_triton_config.max_kernel_dump_occurrences

            for i in range(max_runs + 2):
                test_input = torch.randn(100, 100, device=GPU_TYPE) * (i + 1)
                _ = compiled_fn(test_input)

            # After multiple runs, verify rotation and tensor correctness
            kernel_bases = {}
            verified_tensor_load = False

            for root, dirs, files in os.walk(cache_dir()):
                for d in dirs:
                    if "_run_" in d:
                        full_path = os.path.join(root, d)
                        tensor_files = [
                            f
                            for f in os.listdir(full_path)
                            if f.startswith("tensor_") and f.endswith(".pt")
                        ]
                        if not tensor_files:
                            continue

                        dir_name = os.path.basename(full_path)
                        base_name = dir_name.rsplit("_run_", 1)[0]
                        run_idx = int(dir_name.rsplit("_run_", 1)[1])

                        # Track run indices per kernel
                        if base_name not in kernel_bases:
                            kernel_bases[base_name] = []
                        kernel_bases[base_name].append(run_idx)

                        # Verify we can successfully load at least one saved tensor
                        if not verified_tensor_load:
                            first_tensor_file = os.path.join(full_path, tensor_files[0])
                            loaded_tensor = torch.load(first_tensor_file)

                            # Verify it's a valid tensor with expected properties
                            self.assertIsInstance(loaded_tensor, torch.Tensor)
                            self.assertEqual(loaded_tensor.device.type, GPU_TYPE)
                            verified_tensor_load = True

            # Verify rotation constraints
            if kernel_bases:
                for base_name, indices in kernel_bases.items():
                    self.assertLessEqual(
                        len(indices),
                        max_runs,
                        lambda msg: f"{msg}\nKernel {base_name} has more runs ({len(indices)}) than max ({max_runs})",
                    )

                    # Verify the indices are within [0, max_runs)
                    for idx in indices:
                        self.assertLess(
                            idx,
                            max_runs,
                            lambda msg: f"{msg}\nRun index {idx} exceeds max_runs-1 ({max_runs - 1})",
                        )

        finally:
            # Restore environment variable
            if old_dump_env is None:
                os.environ.pop("TORCHINDUCTOR_DUMP_LAUNCH_TENSORS", None)
            else:
                os.environ["TORCHINDUCTOR_DUMP_LAUNCH_TENSORS"] = old_dump_env


class TestRecheckAutotuneCache(TestCase):
    # [CLASS DOC] Tests for CachingAutotuner.recheck_autotune_cache narrowing compile results
    # from the autotune cache.
    """Tests for CachingAutotuner.recheck_autotune_cache"""

    @staticmethod
    def _make_compile_result(cfg):
        # [FUNCTION DOC] HELPER: Static fixture returning a mock spec'd as a static compile
        # result with the given config.
        """Create a mock StaticTritonCompileResult with the given config."""
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        result = MagicMock(spec=StaticTritonCompileResult)
        result.config = cfg
        return result

    @staticmethod
    def _make_autotuner_with_results(configs, compile_results):
        # [FUNCTION DOC] HELPER: Static fixture creating a CachingAutotuner from the cosine
        # kernel fixture with the given configs and directly assigning compile_results,
        # bypassing compilation.
        """
        Create a CachingAutotuner and inject compile_results directly,
        bypassing actual Triton compilation.
        """
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["configs"] = configs
        autotuner = CachingAutotuner(**args)
        autotuner.compile_results = compile_results
        return autotuner

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_single_config_enters_cache_hit_block(self):
        # [FUNCTION DOC] Checks that with one config and a cache hit, compile results stay as
        # the single matching result and the found_by_coordesc flag from the cached config
        # (true) is copied to it.
        """
        When there is exactly 1 config and the autotune cache returns a hit,
        recheck_autotune_cache should narrow compile_results to that single
        matching result (not skip the block due to len(configs) == 1).
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = False
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        # Cache returns the same config as the best
        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # The compile_results should be narrowed to just the matching one
        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], compile_result)
        # And found_by_coordesc must be propagated
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_true(self):
        # [FUNCTION DOC] Checks that a cache hit on the second of two configs narrows compile
        # results to that one and copies the cached found_by_coordesc true flag.
        """
        When the cached best config has found_by_coordesc=True,
        it must be propagated to the compile result's config.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        cfg_a.found_by_coordesc = False
        cfg_b.found_by_coordesc = False
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache says cfg_b is the best, found via coordesc
        cached_cfg = triton_config({"x": 256}, 64)
        cached_cfg.found_by_coordesc = True

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertIs(autotuner.compile_results[0], result_b)
        # The flag must be propagated from the cached config
        self.assertTrue(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_propagates_found_by_coordesc_false(self):
        # [FUNCTION DOC] Checks that a cached found_by_coordesc false flag overwrites a true
        # flag on the single remaining compile result.
        """
        When the cached best config has found_by_coordesc=False, it must be
        propagated so that coordinate descent can still run if enabled.
        """
        cfg = triton_config({"x": 16}, 64)
        cfg.found_by_coordesc = True
        compile_result = self._make_compile_result(cfg)

        autotuner = self._make_autotuner_with_results([cfg], [compile_result])

        cached_cfg = triton_config({"x": 16}, 64)
        cached_cfg.found_by_coordesc = False

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([cached_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        self.assertEqual(len(autotuner.compile_results), 1)
        self.assertFalse(autotuner.compile_results[0].config.found_by_coordesc)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_no_cache_hit_leaves_results_unchanged(self):
        # [FUNCTION DOC] Checks that when check_autotune_cache returns no configs the two
        # compile results are unchanged.
        """
        When there's no autotune cache hit, compile_results should not change.
        """
        cfg_a = triton_config({"x": 16}, 64)
        cfg_b = triton_config({"x": 256}, 64)
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        # Cache returns no hit (empty list)
        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([], None, {"autotune_cache_state": "miss"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=MagicMock())

        # Nothing should change
        self.assertEqual(len(autotuner.compile_results), 2)


# [FUNCTION DOC] HELPER: Triton jit kernel multiplying each of numel input elements by 2.0 into
# the output pointer, one XBLOCK sized block per program, masked at the end.
@triton.jit
def hip_autotune_kernel(
    in_ptr,
    out_ptr,
    numel,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    data = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, data * 2.0, mask=mask)


@functools.lru_cache
def get_hip_autotune_kernel_with_invalid_config():
    # [FUNCTION DOC] HELPER: Cached factory returning a triton autotune wrapped kernel with two
    # configs, one invalid on HIP (num_warps 32 gives 2048 threads at warp size 64, over the
    # 1024 limit) and one valid, keyed on numel.
    # num_warps=32 with HIP warp_size=64 = 2048 threads, exceeds 1024 limit
    return triton.autotune(
        configs=[
            triton.Config({"XBLOCK": 128}, num_warps=32, num_stages=1),  # invalid
            triton.Config({"XBLOCK": 256}, num_warps=1, num_stages=1),  # valid
        ],
        key=["numel"],
    )(hip_autotune_kernel)


class TestHIPInvalidConfigHandling(TestCase):
    # [CLASS DOC] Tests that invalid HIP launch configurations are tolerated during benchmarking
    # and autotuning.
    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_benchmark_returns_inf_on_invalid_config(self):
        # [FUNCTION DOC] ROCm only. Checks TritonBenchmarker.benchmark returns infinity when the
        # benchmarked callable raises a runtime error reporting an invalid HIP configuration,
        # with vetted benchmarking enabled.
        from torch._inductor.runtime.benchmarking import TritonBenchmarker

        benchmarker = TritonBenchmarker()

        def failing_callable():
            # [FUNCTION DOC] HELPER: Always raises a RuntimeError with a HIP invalid
            # configuration argument message.
            raise RuntimeError(
                "Triton Error [HIP]: Code: 9, Message: invalid configuration argument"
            )

        result = benchmarker.benchmark(
            fn=failing_callable,
            device=GPU_TYPE,
            is_vetted_benchmarking=True,
        )
        self.assertEqual(result, float("inf"))

    @runOnRocm
    @skipUnless(HAS_GPU_AND_TRITON, "requires GPU and Triton")
    def test_autotune_skips_invalid_hip_config_and_succeeds(self):
        # [FUNCTION DOC] ROCm only. Launches the autotuned kernel with an invalid and a valid
        # config over a million floats and checks the output equals the input times 2.
        numel = 1024 * 1024
        x = torch.randn(numel, device=GPU_TYPE, dtype=torch.float32)
        y = torch.empty_like(x)

        kernel = get_hip_autotune_kernel_with_invalid_config()

        def grid(meta):
            # [FUNCTION DOC] HELPER: Grid callback returning a one-element tuple with the
            # ceiling division of numel by XBLOCK.
            return (triton.cdiv(numel, meta["XBLOCK"]),)

        kernel[grid](x, y, numel)

        expected = x * 2.0
        torch.testing.assert_close(y, expected)


class TestGridExprMaximum(TestCase):
    # [CLASS DOC] Tests for the maximum helper of grid expression classes.
    def test_maximum_cpp_mode_casts_int_constants_to_long(self):
        # [FUNCTION DOC] Checks Grid1D maximum in cpp mode: mixed strings and ints cast the int
        # to long and use std::max, all strings need no cast, and all ints are folded to the
        # numeric maximum 20.
        from torch._inductor.runtime.triton_heuristics import Grid1D

        grid = Grid1D(inductor_meta={}, mode="cpp")
        # Mixed str/int: int constants must be cast to (long) for std::max
        result = grid.maximum(["ynumel_0", "ynumel_1", 4480])
        self.assertIn("(long)4480", result)
        self.assertIn("std::max", result)
        # All strings: no cast needed
        result = grid.maximum(["xnumel", "ynumel"])
        self.assertNotIn("(long)", result)
        # All ints: constant-folds
        self.assertEqual(grid.maximum([10, 20, 5]), 20)


class TestGrid2DWithYZOverflowZeroYnumel(TestCase):
    # [CLASS DOC] Regression tests for Grid2DWithYZOverflow when ynumel is zero (issue 178530)
    # and for normal and overflowing ynumel values.
    """Regression test for https://github.com/pytorch/pytorch/issues/178530"""

    def test_grid2d_yz_overflow_zero_ynumel_python(self):
        # [FUNCTION DOC] Checks that in python mode a zero ynumel does not raise a division
        # error and gives y and z grid values of 0.
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # ynumel=0 must not raise ZeroDivisionError
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 0, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(y, 0)
        self.assertEqual(z, 0)

    def test_grid2d_yz_overflow_zero_ynumel_cpp(self):
        # [FUNCTION DOC] Checks that in cpp mode the generated y grid expression contains a
        # comparison with zero as a guard.
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="cpp")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # cpp mode: the generated expression should contain a zero-guard
        self.assertIn("== 0", str(grid.y_grid))

    def test_grid2d_yz_overflow_nonzero_ynumel_unchanged(self):
        # [FUNCTION DOC] Checks that for ynumel 256 with block 128 the python-mode grid
        # evaluates to 2, 2, 1.
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Normal case: ynumel > 0 still works correctly
        x, y, z = grid.eval_slow(
            {"xnumel": 256, "ynumel": 256, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 2)
        self.assertEqual(y, 2)
        self.assertEqual(z, 1)

    def test_grid2d_yz_overflow_large_ynumel(self):
        # [FUNCTION DOC] Checks that a very large ynumel (128 times 131070) with block 128 gives
        # x of 1 and a y times z product covering at least 131070 blocks.
        from torch._inductor.runtime.triton_heuristics import Grid2DWithYZOverflow

        grid = Grid2DWithYZOverflow(inductor_meta={}, mode="python")
        grid.generate({"XBLOCK": 128, "YBLOCK": 128})
        # Large ynumel that requires overflow splitting across y and z
        x, y, z = grid.eval_slow(
            {"xnumel": 128, "ynumel": 128 * 131070, "XBLOCK": 128, "YBLOCK": 128}
        )
        self.assertEqual(x, 1)
        # y * z must cover all y blocks
        self.assertGreaterEqual(y * z, 131070)


class TestFastLauncherDeviceSupport(TestCase):
    # [CLASS DOC] Tests for fast launcher support across device types.
    @staticmethod
    def _make_autotuner(device_type):
        # [FUNCTION DOC] HELPER: Static fixture building a CachingAutotuner for an empty kernel
        # on a fake device of the given type (warp size 32, 1024 threads per block) with one
        # tiny config and the fast triton launcher option enabled.
        def triton_():
            # [FUNCTION DOC] HELPER: Empty kernel placeholder.
            pass

        device = DeviceProperties(
            type=device_type,
            index=0,
            multi_processor_count=1,
            cc=0,
            max_threads_per_block=1024,
            warp_size=32,
        )
        triton_meta = {
            "signature": {},
            "device": device,
            "constants": {},
            "configs": [],
        }
        return CachingAutotuner(
            fn=triton_,
            triton_meta=triton_meta,
            configs=[triton_config({"x": 1}, 1, warp_size=device.warp_size_or_default)],
            save_cache_hook=False,
            mutated_arg_names=[],
            reset_to_zero_arg_names=[],
            optimize_mem=False,
            heuristic_type=HeuristicType.POINTWISE,
            inductor_meta={"use_fast_triton_launcher": True},
        )

    @staticmethod
    def _make_static_launcher():
        # [FUNCTION DOC] HELPER: Static fixture returning a function copied from an empty
        # template with a runner global bound to a fake kernel, and marked as a static launcher.
        class Kernel:
            # [CLASS DOC] HELPER: Fake compiled kernel exposing class attributes that a launcher
            # builder reads (function, num_warps, shared, argument types, scratch flags) and a
            # run method.
            function = 1
            num_warps = 4
            shared = 0
            arg_tys = ""
            has_global_scratch = False
            has_profile_scratch = False

            def run(self):
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

        def launcher_template():
            # [FUNCTION DOC] HELPER: Empty function whose code object seeds the fake launcher.
            pass

        kernel = Kernel()
        launcher = types.FunctionType(
            launcher_template.__code__, {"runner": kernel.run}, "launcher"
        )
        launcher._is_static = True
        return launcher

    def test_fast_launcher_not_built_for_xpu(self):
        # [FUNCTION DOC] Checks that building a fast launcher for an XPU autotuner returns None
        # and never constructs the CUDA fast launcher class.
        autotuner = self._make_autotuner("xpu")
        launcher = self._make_static_launcher()

        with patch("torch._C._FastCudaLauncher", create=True) as fast_launcher:
            self.assertIsNone(autotuner._build_fast_launcher(launcher))
            fast_launcher.assert_not_called()


class TestDynamicScaleRblockCacheInteraction(TestCase):
    # [CLASS DOC] Tests for how dynamic rblock scaling and cached autotuning interact, including
    # reconstruction of cached configs not in the original list.
    """Tests for _dynamic_scale_rblock + autotune cache interaction.

    _dynamic_scale_rblock can add configs that aren't in the original
    config list. When such a config wins autotuning and gets cached,
    the warm-start path must still be able to load it from the cache
    and skip re-autotuning.
    """

    @staticmethod
    def _make_compile_result(cfg):
        # [FUNCTION DOC] HELPER: Static fixture returning a mock spec'd as a static compile
        # result with the given config.
        from torch._inductor.runtime.triton_heuristics import StaticTritonCompileResult

        result = MagicMock(spec=StaticTritonCompileResult)
        result.config = cfg
        return result

    @staticmethod
    def _make_autotuner_with_results(configs, compile_results):
        # [FUNCTION DOC] HELPER: Static fixture creating a CachingAutotuner from the cosine
        # kernel fixture with the given configs and injecting compile_results.
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["configs"] = configs
        autotuner = CachingAutotuner(**args)
        autotuner.compile_results = compile_results
        return autotuner

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_skipped_on_cache_hit(self):
        # [FUNCTION DOC] Checks that dynamic rblock scaling does not change the number of
        # compile results when the cache state is hit.
        """
        _dynamic_scale_rblock must be a no-op when the autotune cache
        already determined the best config (cache_state == 'hit').
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = {"autotune_cache_state": "hit"}

        n_before = len(autotuner.compile_results)
        autotuner._dynamic_scale_rblock()
        n_after = len(autotuner.compile_results)

        self.assertEqual(n_before, n_after)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_runs_on_cache_miss(self):
        # [FUNCTION DOC] Checks that dynamic rblock scaling runs without error when the cache
        # state is miss.
        """
        _dynamic_scale_rblock should not be skipped when the autotune
        cache missed — this is a cold-start scenario.
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = {"autotune_cache_state": "miss"}

        autotuner._dynamic_scale_rblock()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_dynamic_scale_rblock_runs_when_cache_info_none(self):
        # [FUNCTION DOC] Checks that dynamic rblock scaling runs without error when the cache
        # info is None.
        """
        _dynamic_scale_rblock should not be skipped when
        autotune_cache_info is None (e.g. loaded from old pickle).
        """
        cfg = triton_config({"x": 16}, 64)
        result = self._make_compile_result(cfg)
        autotuner = self._make_autotuner_with_results([cfg], [result])
        autotuner.autotune_cache_info = None

        autotuner._dynamic_scale_rblock()

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_reconstructs_unknown_config(self):
        # [FUNCTION DOC] Checks that _load_cached_autotuning rebuilds a config (XBLOCK 1,
        # R0_BLOCK 1024, 8 warps, 1 stage) absent from the original list when the config hash
        # matches, instead of returning None.
        """
        When the cached best config is not in the original configs list
        (e.g. it was added by _dynamic_scale_rblock), _load_cached_autotuning
        must reconstruct it from the cached data instead of returning None.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning
        from torch._inductor.runtime.triton_heuristics import hash_configs

        original_configs = [
            triton_config({"x": 1}, 16, num_stages=1),
            triton_config({"x": 8}, 4, num_stages=1),
        ]
        configs_hash = hash_configs(original_configs)

        best_config_data = {
            "XBLOCK": 1,
            "R0_BLOCK": 1024,
            "num_warps": 8,
            "num_stages": 1,
            "configs_hash": configs_hash,
            "found_by_coordesc": False,
            "time_taken_ms": 42,
        }

        result = _load_cached_autotuning(
            best_config_data, configs_hash, original_configs, {}
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.kwargs["XBLOCK"], 1)
        self.assertEqual(result.kwargs["R0_BLOCK"], 1024)
        self.assertEqual(result.num_warps, 8)
        self.assertEqual(result.num_stages, 1)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_still_matches_known_config(self):
        # [FUNCTION DOC] Checks that when the cached best config matches an original config and
        # the hash matches, the very same config object is returned.
        """
        When the cached best config IS in the original configs list,
        _load_cached_autotuning must return that same config object.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning
        from torch._inductor.runtime.triton_heuristics import hash_configs

        cfg_a = triton_config({"x": 1}, 16, num_stages=1)
        cfg_b = triton_config({"x": 8}, 4, num_stages=1)
        original_configs = [cfg_a, cfg_b]
        configs_hash = hash_configs(original_configs)

        best_config_data = {
            **cfg_b.kwargs,
            "num_warps": cfg_b.num_warps,
            "num_stages": cfg_b.num_stages,
            "configs_hash": configs_hash,
            "found_by_coordesc": False,
            "time_taken_ms": 42,
        }

        result = _load_cached_autotuning(
            best_config_data, configs_hash, original_configs, {}
        )

        self.assertIsNotNone(result)
        self.assertIs(result, cfg_b)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_load_cached_autotuning_rejects_hash_mismatch(self):
        # [FUNCTION DOC] Checks that a hash mismatch makes _load_cached_autotuning return None.
        """
        When configs_hash doesn't match, _load_cached_autotuning must
        return None regardless of whether the config matches.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        original_configs = [triton_config({"x": 1}, 16, num_stages=1)]

        best_config_data = {
            "XBLOCK": 1,
            "num_warps": 1,
            "num_stages": 1,
            "configs_hash": "wrong_hash",
            "found_by_coordesc": False,
        }

        result = _load_cached_autotuning(
            best_config_data, "correct_hash", original_configs, {}
        )

        self.assertIsNone(result)

    @skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
    def test_recheck_autotune_cache_compiles_dynamic_config(self):
        # [FUNCTION DOC] Checks that when the cached best config is not among compile results,
        # recheck_autotune_cache compiles it via the patched config precompile (called once with
        # that config) and ends with one compile result.
        """
        When the cached best config was added by _dynamic_scale_rblock
        and isn't in compile_results, recheck_autotune_cache must compile
        it (not silently drop it, leaving all original configs active).
        """
        from triton import Config

        cfg_a = Config({"XBLOCK": 1, "R0_BLOCK": 2048}, num_warps=16, num_stages=1)
        cfg_b = Config({"XBLOCK": 8, "R0_BLOCK": 512}, num_warps=4, num_stages=1)
        result_a = self._make_compile_result(cfg_a)
        result_b = self._make_compile_result(cfg_b)

        autotuner = self._make_autotuner_with_results(
            [cfg_a, cfg_b], [result_a, result_b]
        )

        dynamic_cfg = Config(
            {"XBLOCK": 1, "R0_BLOCK": 1024}, num_warps=16, num_stages=1
        )

        mock_precompile = MagicMock(return_value=self._make_compile_result(dynamic_cfg))
        autotuner._precompile_config = mock_precompile

        reload_fn = MagicMock()
        reload_fn.return_value.fn = MagicMock()

        with patch(
            "torch._inductor.runtime.triton_heuristics.check_autotune_cache",
            return_value=([dynamic_cfg], None, {"autotune_cache_state": "hit"}),
        ):
            autotuner.recheck_autotune_cache(reload_kernel_from_src=reload_fn)

        self.assertEqual(len(autotuner.compile_results), 1)
        mock_precompile.assert_called_once_with(dynamic_cfg)


@skipUnless(HAS_GPU_AND_TRITON, "requires gpu and triton")
class TestCheckLauncherCallArgs(TestCase):
    # [CLASS DOC] Unit tests for CachingAutotuner._check_launcher_call_args, which
    # validates positional argument counts against launcher metadata. Requires a GPU for
    # the shared fixture.
    """Unit tests for CachingAutotuner._check_launcher_call_args.

    These exercise the arg-mismatch error path without Triton compilation,
    but the shared cos-kernel fixture still builds a CachingAutotuner (which
    resolves DeviceProperties for GPU_TYPE), so they require a GPU.
    """

    def _make_autotuner(self):
        # [FUNCTION DOC] HELPER: Returns a CachingAutotuner built from the cosine kernel
        # fixture.
        """Return a CachingAutotuner configured with the cos kernel fixture."""
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        return CachingAutotuner(**args)

    def _make_launcher(self, n_positional: int):
        # [FUNCTION DOC] HELPER: Returns a fake launcher function tagged with the expected
        # positional count n_positional.
        """Return a minimal callable that looks like a generated launcher.

        The only attribute _check_launcher_call_args inspects is
        ``_expected_positional_count``, so we don't need real Triton machinery.
        """

        def launcher(*args, stream=None):
            # [FUNCTION DOC] HELPER: Placeholder launcher accepting any positional arguments and
            # an optional stream; does nothing.
            pass

        launcher._expected_positional_count = n_positional
        return launcher

    def test_correct_arg_count_passes_silently(self):
        # [FUNCTION DOC] Checks that an argument tuple matching the expected count raises
        # nothing.
        """No exception when args exactly matches _expected_positional_count."""
        autotuner = self._make_autotuner()
        launcher = self._make_launcher(n_positional=3)
        # Pass exactly 3 positional args - should be a no-op.
        autotuner._check_launcher_call_args(launcher, (1, 2, 3))

    def test_too_many_args_raises_type_error(self):
        # [FUNCTION DOC] Checks that one extra positional argument raises TypeError whose
        # message mentions stream and keyword.
        """TypeError with a helpful message when too many positional args are passed."""
        autotuner = self._make_autotuner()
        launcher = self._make_launcher(n_positional=3)
        # Simulate the 'stream' being passed positionally (4 args, expected 3).
        with self.assertRaises(TypeError) as cm:
            autotuner._check_launcher_call_args(launcher, (1, 2, 3, 4))
        self.assertIn("stream", str(cm.exception).lower())
        self.assertIn("keyword", str(cm.exception).lower())

    def test_error_message_includes_kernel_name(self):
        # [FUNCTION DOC] Checks that the TypeError message includes the kernel name from
        # inductor_meta.
        """The TypeError message should include the kernel name from inductor_meta."""
        args = TestTritonHeuristics._get_cos_kernel_caching_autotuner_args()
        args["inductor_meta"] = {
            "kernel_name": "my_custom_kernel",
            "grid_type": "Grid1D",
        }
        autotuner = CachingAutotuner(**args)
        launcher = self._make_launcher(n_positional=2)
        with self.assertRaises(TypeError) as cm:
            autotuner._check_launcher_call_args(launcher, (1, 2, 3))
        self.assertIn("my_custom_kernel", str(cm.exception))

    def test_launcher_without_expected_count_is_skipped(self):
        # [FUNCTION DOC] Checks that a launcher lacking the expected count attribute is skipped
        # without raising even with many arguments.
        """Launchers without the generated metadata are skipped gracefully."""

        def raw_launcher(*args, stream=None):
            # [FUNCTION DOC] HELPER: Placeholder launcher without the expected count attribute.
            pass

        # No _expected_positional_count attribute set.
        autotuner = self._make_autotuner()
        # Should not raise, even with many args.
        autotuner._check_launcher_call_args(raw_launcher, (1, 2, 3, 4, 5))


class TestWarpSizeUnification(TestCase):
    # [CLASS DOC] Tests for warp size handling in DeviceProperties and config helpers.
    """Tests for the unified warp_size threading through config helpers."""

    def test_warp_size_or_default(self):
        # [FUNCTION DOC] Checks warp_size_or_default: raises RuntimeError for a CUDA device
        # without a warp size, gives 32 for CPU, and returns explicit 32 and 64 values.
        none_props = DeviceProperties(
            type="cuda", index=0, multi_processor_count=80, cc=80, warp_size=None
        )
        with self.assertRaisesRegex(
            RuntimeError, "cuda device properties must report warp_size"
        ):
            none_props.warp_size_or_default

        cpu_props = DeviceProperties(
            type="cpu", index=0, multi_processor_count=80, cc=80, warp_size=None
        )
        self.assertEqual(cpu_props.warp_size_or_default, 32)

        w32 = DeviceProperties(
            type="cuda", index=0, multi_processor_count=80, cc=80, warp_size=32
        )
        self.assertEqual(w32.warp_size_or_default, 32)

        w64 = DeviceProperties(
            type="hip", index=0, multi_processor_count=80, cc=80, warp_size=64
        )
        self.assertEqual(w64.warp_size_or_default, 64)

    def test_num_warps_halves_on_wave64(self):
        # [FUNCTION DOC] Checks that the warp count helper halves 8 to 4 for warp size 64.
        # wave64 (AMD CDNA/gfx9) halves the range so total threads match wave32.
        self.assertEqual(_num_warps(8, max_num_warps=8, warp_size=64), 4)

    def test_num_warps_does_not_halve_on_wave32(self):
        # [FUNCTION DOC] Checks that the warp count helper keeps 8 for warp size 32 and for the
        # default warp size.
        # wave32 (NVIDIA and AMD RDNA) must not halve.
        self.assertEqual(_num_warps(8, max_num_warps=8, warp_size=32), 8)
        # Default warp_size is 32; confirm the default path matches wave32.
        self.assertEqual(_num_warps(8, max_num_warps=8), 8)

    def test_check_max_grid_x_respects_warp_size_hip(self):
        # [FUNCTION DOC] With torch.version.hip patched to a value, checks _check_max_grid_x
        # yields an x for warp size 64 twice that for 32, both within the X block maximum.
        # _check_max_grid_x only uses warp_size on the HIP path, where the
        # bound is on total threads (num_blocks * num_warps * warp_size).
        # Force the HIP branch so the assertion exercises that path even on
        # a non-HIP build.
        size_hints = {"x": 2**32}
        with patch.object(torch.version, "hip", "6.0.0"):
            x32, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=32)
            x64, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=64)
        # Doubling warp_size halves the per-block thread budget, so x must
        # double to keep total threads under 2**31 - 1.
        self.assertEqual(x64, 2 * x32)
        self.assertLessEqual(x32, TRITON_MAX_BLOCK["X"])
        self.assertLessEqual(x64, TRITON_MAX_BLOCK["X"])

    def test_check_max_grid_x_ignores_warp_size_on_nvidia(self):
        # [FUNCTION DOC] With hip patched to None, checks _check_max_grid_x returns the same x
        # for warp sizes 32 and 64.
        # NVIDIA bounds num_blocks directly, so warp_size must not change x.
        size_hints = {"x": 2**32}
        with patch.object(torch.version, "hip", None):
            x32, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=32)
            x64, _ = _check_max_grid_x(size_hints, x=1, num_warps=8, warp_size=64)
        self.assertEqual(x32, x64)

    def test_triton_config_uses_warp_size_for_min_elem(self):
        # [FUNCTION DOC] Checks that triton_config with a minimum of 4 elements per thread gives
        # an XBLOCK for warp size 64 twice that of warp size 32.
        # min_elem_per_thread * warp_size * num_warps drives the floor for
        # block_size; doubling warp_size should at least double the XBLOCK.
        size_hints = {"x": 4096}
        cfg32 = triton_config(
            size_hints, 128, num_warps=1, min_elem_per_thread=4, warp_size=32
        )
        cfg64 = triton_config(
            size_hints, 128, num_warps=1, min_elem_per_thread=4, warp_size=64
        )
        self.assertEqual(cfg64.kwargs["XBLOCK"], 2 * cfg32.kwargs["XBLOCK"])


class TestMakeLaunchersMemory(TestCase):
    # [CLASS DOC] Memory regression tests for CachingAutotuner._make_launchers.
    def test_failed_config_exception_not_retained(self):
        # [FUNCTION DOC] Regression test using fake objects, no GPU: runs _make_launchers on a
        # stand-in object whose second config fails with a live exception, with gc disabled, and
        # checks a sentinel held by the calling frame is freed by refcounting alone (the
        # exception is not retained), and that one launcher was created.
        """Regression (D107597017 / PR #184285): CachingAutotuner._make_launchers
        must not retain the failed-config build exception past its return. That
        exception's traceback pins its frame chain -- and via those frames' f_back
        the benchmarking callers up to do_bench, whose 256MB Triton L2-flush buffer
        then leaks one-per-autotuned-kernel until cyclic GC (which never runs under
        gc.disable(), as APS training does). Pure reference-cycle check, no GPU: a
        sentinel standing in for the L2 buffer, held only by a frame that calls
        _make_launchers, must be freed by refcount once _make_launchers returns.
        """
        import contextlib
        import gc
        import types
        import weakref

        class _Result:
            # [CLASS DOC] HELPER: Fake compile result with a config namespace.
            config = types.SimpleNamespace(num_stages=1, kwargs={})

        results = [_Result(), _Result()]

        def fake_make_launcher(result):
            # [FUNCTION DOC] HELPER: Fake launcher builder returning an object and no error for
            # the first result, and for others no launcher plus a caught RuntimeError.
            # First config builds; the last fails, returning a live exception whose
            # traceback references this call stack -- as the real _make_launcher does
            # for OutOfResources / OutOfMemoryError.
            if result is results[0]:
                return object(), None
            try:
                raise RuntimeError("out of resource (test)")
            except RuntimeError as e:
                return None, e

        fake_self = types.SimpleNamespace(
            launchers=[],
            compile_results=results,
            triton_meta={"device": 0},
            device_props=types.SimpleNamespace(type="cuda"),
            inductor_meta={},
            get_device_interface=lambda: None,
            _make_launcher=fake_make_launcher,
        )

        class _Sentinel:
            # [CLASS DOC] HELPER: Empty class whose instance stands in for a large buffer.
            pass

        def run():
            # [FUNCTION DOC] HELPER: Creates a sentinel and weak reference, calls
            # _make_launchers with a patched device guard, and returns the weak reference.
            l2_buffer = _Sentinel()  # stands in for do_bench's L2-flush buffer
            ref = weakref.ref(l2_buffer)
            with patch(
                "torch._dynamo.device_interface.DeviceGuard",
                lambda *args, **kwargs: contextlib.nullcontext(),
            ):
                CachingAutotuner._make_launchers(fake_self)
            return ref

        gc.disable()
        try:
            ref = run()
            self.assertIsNone(
                ref(),
                "_make_launchers retained the failed-config exception; its traceback "
                "pins caller frames (do_bench's L2-flush buffer) until cyclic GC.",
            )
        finally:
            gc.enable()

        self.assertEqual(len(fake_self.launchers), 1)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
