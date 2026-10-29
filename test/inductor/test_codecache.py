# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for Inductor's code caches (torch._inductor.codecache and related caching).
# Covers cache key strategies, torch_key_cache, PyCodeCache, FxGraphCache and its hashing,
# pickling and bypass rules, the AOTAutograd/Dynamo caches, CUDA/C++ code caches, custom graph
# pass hashing, and cache artifact save/load. Most tests need a fresh cache directory. Import-time
# side effects: enables the fake tensor cache and its crosscheck mode in the dynamo config,
# registers the opaque test class _CyclicOpaque if not yet registered, renames the module-level
# helper _custom_empty to "empty", and imports triton when available.
import base64
import copy
import functools
import hashlib
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
import warnings
from contextlib import contextmanager
from typing import Any, cast
from typing_extensions import override
from unittest import mock

import torch
from torch._dynamo import reset
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._dynamo.utils import counters
from torch._functorch import config as functorch_config
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config, config_comms, metrics
from torch._inductor.cache_key import (
    AUTOTUNE_CACHE_KEY_STRATEGY,
    CacheKeyStrategy,
    CODE_CACHE_KEY_STRATEGY,
    COMPACT_CACHE_KEY_STRATEGY,
    FX_GRAPH_CACHE_KEY_STRATEGY,
    SYSTEM_CACHE_KEY_STRATEGY,
)
from torch._inductor.codecache import (
    BypassFxGraphCache,
    CacheabilityValidator,
    CacheBase,
    CppWrapperCodeCache,
    CUDACodeCache,
    FxGraphCache,
    FxGraphCachePickler,
    FxGraphHashDetails,
    PyCodeCache,
    TensorMetadata,
    TensorMetadataAndValues,
)
from torch._inductor.codegen.cuda.compile_utils import cuda_compile_command
from torch._inductor.cpp_builder import normalize_path_separator
from torch._inductor.custom_graph_pass import (
    CustomGraphModulePass,
    CustomGraphPass,
    CustomPartitionerFn,
    get_hash_for_files,
)
from torch._inductor.graph import GraphLowering
from torch._inductor.mock_cache import global_stats, PatchCaches, Stats
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import clear_caches, fresh_cache
from torch._library import capture_triton
from torch._subclasses import FakeTensorMode
from torch.compiler._cache import (
    CacheArtifact,
    CacheArtifactFactory,
    CacheArtifactManager,
    CacheArtifactRecorder,
    CacheInfo,
)
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch.testing._internal.common_cuda import (
    SM80OrLater,
    TEST_MULTIGPU,
    with_tf32_off,
)
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_FBCODE,
    IS_SANDCASTLE,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU,
    HAS_MULTIGPU,
    HAS_TRITON,
    HAS_XPU_AND_TRITON,
    patch_inductor_backend,
    requires_gpu,
    requires_triton,
)
from torch.testing._internal.triton_utils import (
    requires_cuda_and_triton,
    requires_gpu_and_triton,
)
from torch.testing._internal.two_tensor import TwoTensor


try:
    from . import custom_inductor_config
except ImportError:
    import custom_inductor_config


if HAS_TRITON:
    import triton  # @manual

    from torch.testing._internal.triton_utils import add_kernel, sub_kernel

torch._dynamo.config.fake_tensor_cache_enabled = True
torch._dynamo.config.fake_tensor_cache_crosscheck_enabled = True


STATIC_LAUNCHER_DEVICES = ("cuda", "xpu")


class TestCacheKeyStrategy(TestCase):
    # [CLASS DOC] Tests that the cache key strategy objects in torch._inductor.cache_key produce
    # the expected key formats and are the ones used by codecache and autotune cache helpers.
    def _compact_sha256(self, data: bytes) -> str:
        # [FUNCTION DOC] HELPER: Compute the compact key format for bytes: sha256 digest, base32
        # encoded, truncated to 51 characters and lower-cased. Returns the string.
        return (
            base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()
        )

    def test_strategy_formats_existing_key_shapes(self):
        # [FUNCTION DOC] Checks that the compact, code, fx graph and autotune strategies yield
        # keys with the documented shapes: compact is the bare hash, code and fx graph keys carry
        # a one letter prefix, code joins components with a double bar separator, and autotune is
        # a plain hex sha256 of the concatenated components.
        self.assertEqual(
            COMPACT_CACHE_KEY_STRATEGY.key(b"kernel"),
            self._compact_sha256(b"kernel"),
        )
        self.assertEqual(
            CODE_CACHE_KEY_STRATEGY.key("kernel", "extra"),
            "c" + self._compact_sha256(b"kernel||extra"),
        )
        self.assertEqual(
            FX_GRAPH_CACHE_KEY_STRATEGY.key(b"graph"),
            "f" + self._compact_sha256(b"graph"),
        )
        self.assertEqual(
            AUTOTUNE_CACHE_KEY_STRATEGY.key("kernel", b"torch"),
            hashlib.sha256(b"kerneltorch").hexdigest(),
        )

    def test_custom_strategy_composes_components(self):
        # [FUNCTION DOC] Builds a custom CacheKeyStrategy with hex digests, prefix "x" and a colon
        # separator, and checks the key is the prefix followed by the hex sha256 of the components
        # joined with that separator (string and bytes components are mixed).
        strategy = CacheKeyStrategy(
            name="test",
            digest_format="hex",
            prefix="x",
            separator=b":",
        )

        self.assertEqual(
            strategy.key("left", b"right"),
            "x" + hashlib.sha256(b"left:right").hexdigest(),
        )

    def test_code_hash_uses_code_strategy(self):
        # [FUNCTION DOC] Checks that the module-level hash helpers in codecache delegate to the
        # strategies: sha256_hash equals the compact strategy key, and code_hash with an extra
        # argument equals the code strategy key over the code and the extra string.
        from torch._inductor.codecache import code_hash, sha256_hash

        self.assertEqual(
            sha256_hash(b"kernel"), COMPACT_CACHE_KEY_STRATEGY.key(b"kernel")
        )
        self.assertEqual(
            code_hash("kernel", extra="extra"),
            CODE_CACHE_KEY_STRATEGY.key("kernel", "extra"),
        )

    def test_system_strategy_hashes_sorted_json(self):
        # [FUNCTION DOC] Checks that the system strategy hashes a JSON-able mapping by serializing
        # it with sorted keys and taking the hex sha256, via key_from_json.
        system = {"version": {"triton": "abc"}, "device": {"name": "gpu"}}

        self.assertEqual(
            SYSTEM_CACHE_KEY_STRATEGY.key_from_json(system),
            hashlib.sha256(
                json.dumps(system, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )

    def test_cache_base_get_system_uses_system_strategy(self):
        # [FUNCTION DOC] Checks that CacheBase.get_system computes its hash through the system
        # strategy. Replaces the strategy with a recording fake, forces triton off and fakes the
        # CUDA device properties and version, clears the lru cache on get_system before and after,
        # then verifies the hash field is the fake's sentinel, the mapping given to the strategy
        # has the device name and the triton/cuda versions, and sort_keys was true.
        class FakeStrategy:
            # [CLASS DOC] HELPER: Fake system strategy that records the mapping and the sort_keys
            # flag it was given and returns a constant sentinel string as the key.
            value = None
            sort_keys = None

            def key_from_json(self, value, *, sort_keys=True):
                # [FUNCTION DOC] HELPER: Store a deep copy of the value and the sort_keys flag on
                # the instance and return the string "sentinel".
                self.value = copy.deepcopy(value)
                self.sort_keys = sort_keys
                return "sentinel"

        fake_strategy = FakeStrategy()
        device_properties = types.SimpleNamespace(
            name="test-gpu", gcnArchName="test-gcn"
        )

        CacheBase.get_system.cache_clear()
        try:
            with (
                mock.patch(
                    "torch._inductor.codecache.SYSTEM_CACHE_KEY_STRATEGY",
                    fake_strategy,
                ),
                mock.patch("torch._inductor.runtime.triton_compat.HAS_TRITON", False),
                mock.patch.object(torch.cuda, "current_device", return_value=0),
                mock.patch.object(
                    torch.cuda,
                    "get_device_properties",
                    return_value=device_properties,
                ),
                mock.patch.object(torch.version, "cuda", "test-cuda"),
            ):
                self.assertEqual(CacheBase.get_system()["hash"], "sentinel")
        finally:
            CacheBase.get_system.cache_clear()

        self.assertEqual(
            fake_strategy.value,
            {
                "device": {"name": "test-gpu"},
                "version": {"triton": None, "cuda": "test-cuda"},
            },
        )
        self.assertTrue(fake_strategy.sort_keys)

    def test_autotune_prepare_key_uses_strategy(self):
        # [FUNCTION DOC] Checks that AutotuneCache._prepare_key builds its key via the autotune
        # strategy: with a fake strategy and the cache_key_tag config set to "tag", the file name
        # of the given path joined with the tag by a colon is passed as the single component, and
        # the fake's return value is returned.
        from torch._inductor.runtime.autotune_cache import AutotuneCache

        class FakeStrategy:
            # [CLASS DOC] HELPER: Fake key strategy that records the components it receives and
            # returns a constant sentinel string.
            components = None

            def key(self, *components):
                # [FUNCTION DOC] HELPER: Record the positional components on the instance and
                # return the string "sentinel".
                self.components = components
                return "sentinel"

        fake_strategy = FakeStrategy()
        with (
            mock.patch(
                "torch._inductor.runtime.autotune_cache.AUTOTUNE_CACHE_KEY_STRATEGY",
                fake_strategy,
            ),
            torch.compiler.config.patch({"cache_key_tag": "tag"}),
        ):
            self.assertEqual(AutotuneCache._prepare_key("/tmp/cabcdef.py"), "sentinel")

        self.assertEqual(fake_strategy.components, ("cabcdef.py:tag",))


class TestTorchKeyCache(TestCase):
    # [CLASS DOC] Tests for the torch_key_cache decorator in codecache (background prefetch and
    # thread-safe single evaluation of the wrapped function).
    def test_prefetch(self):
        # [FUNCTION DOC] Decorates a function with torch_key_cache, calls prefetch, waits up to
        # five seconds for the function to signal it ran, then checks calling the wrapper returns
        # the function's result.
        import threading

        from torch._inductor.codecache import torch_key_cache

        done = threading.Event()

        @torch_key_cache
        def expensive():
            # [FUNCTION DOC] HELPER: Sets the threading event closed over from the test and
            # returns the bytes result.
            done.set()
            return b"result"

        expensive.prefetch()
        done.wait(timeout=5)
        self.assertTrue(done.is_set())
        result = expensive()
        self.assertEqual(result, b"result")

    def test_concurrent_callers(self):
        # [FUNCTION DOC] Calls a torch_key_cache-wrapped counting function from eight pool
        # threads at once and checks every call returned the same result and the function body
        # ran exactly once.
        from concurrent.futures import ThreadPoolExecutor

        from torch._inductor.codecache import torch_key_cache

        call_count = 0

        @torch_key_cache
        def expensive():
            # [FUNCTION DOC] HELPER: Increments a nonlocal call counter and returns the
            # bytes result.
            nonlocal call_count
            call_count += 1
            return b"result"

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: expensive(), range(8)))
        self.assertTrue(all(r == b"result" for r in results))
        self.assertEqual(call_count, 1)


class LogCaptureHandler(logging.Handler):
    # [CLASS DOC] HELPER: Logging handler that appends each emitted record to its records list.
    def __init__(self, level):
        # [FUNCTION DOC] HELPER: Initialize the handler at the given level and create an empty
        # records list.
        super().__init__(level)
        self.records = []

    def emit(self, record):
        # [FUNCTION DOC] HELPER: Append the log record to the records list.
        self.records.append(record)


@contextmanager
def capture_logs(log_name, log_level):
    # [FUNCTION DOC] HELPER: Context manager that temporarily sets the named logger to the given
    # level and attaches a handler that collects records. Yields the list of captured records. On
    # exit it removes the handler and restores the previous level.
    try:
        logger = logging.getLogger(log_name)
        old_level = logger.level
        handler = logging.Handler()
        logger.setLevel(log_level)
        log_records = []

        def emit(record):
            # [FUNCTION DOC] HELPER: Replacement emit for the handler; appends the record to the
            # captured list.
            log_records.append(record)

        handler.emit = emit
        logger.addHandler(handler)

        yield log_records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


class MyModelConv2d(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with two stride-2 bias-free 3x3 convolutions (3 to dim to
    # dim channels, default dim 512) with a dynamo graph break between them.
    def __init__(self, dim=512):
        # [FUNCTION DOC] HELPER: Create the two convolution layers.
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Apply the first convolution, force a dynamo graph break, apply
        # the second convolution, and return the result.
        x = self.conv1(x)
        torch._dynamo.graph_break()
        x = self.conv2(x)
        return x


class _CyclicOpaque(torch._custom_class_base.CustomClassBase):
    # [CLASS DOC] HELPER: Opaque custom class used by tests that need opaque objects able to form
    # reference cycles through their child attribute.
    """Test helper: opaque type whose instances can form reference cycles."""

    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize with no child.
        self.child = None


if not torch._library.opaque_object.is_custom_class(_CyclicOpaque):
    torch._library.opaque_object.register_custom_class(_CyclicOpaque, typ="symbolic")


def _custom_empty(*args: object, **kwargs: object) -> None:
    # [FUNCTION DOC] HELPER: Accepts any arguments and returns None. Its __name__ is overwritten
    # below so it mimics a factory named empty without being torch.empty.
    return None


# Verify factory detection does not match arbitrary functions named like torch.empty.
_custom_empty.__name__ = "empty"


class TestPyCodeCache(TestCase):
    # [CLASS DOC] Tests for PyCodeCache: writing sources, loading modules from key/path pairs,
    # linemap handling, editable cached wrappers and C++ wrapper source generation.
    def test_linemaps_empty(self):
        # [FUNCTION DOC] Writes a trivial module to PyCodeCache, loads it with an empty linemap,
        # and checks stack_frames_for_code returns None for that path.
        src = """import torch"""
        (key, path) = PyCodeCache.write(src, "")
        # Load with an empty linemap
        PyCodeCache.load_by_key_path(key, path, linemap=[])
        stack_frames = PyCodeCache.stack_frames_for_code(path, 0)
        self.assertEqual(stack_frames, None)

    def test_load_by_key_path_can_defer_sys_modules_registration(self):
        # [FUNCTION DOC] Checks that load_by_key_path with the set_sys_modules flag false returns
        # a module that is not in sys.modules, and that a later normal load of the same key and
        # path returns the same module object and registers it in sys.modules. Clears the code
        # cache before and after and removes the module entry in cleanup.
        """
        Benchmark-only modules should not need a global ``sys.modules`` entry,
        but a later normal runtime load must still be able to publish the same
        cached module there.
        """
        module_name = None
        try:
            PyCodeCache.cache_clear()
            key, path = PyCodeCache.write("value = 1\n")

            module = PyCodeCache.load_by_key_path(
                key,
                path,
                set_sys_modules=False,
            )
            module_name = module.__name__
            self.assertNotIn(module_name, sys.modules)

            registered_module = PyCodeCache.load_by_key_path(key, path)
            self.assertIs(registered_module, module)
            self.assertIn(module_name, sys.modules)
            self.assertIs(sys.modules[module_name], module)
        finally:
            if module_name is not None:
                sys.modules.pop(module_name, None)
            PyCodeCache.cache_clear()

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_editable_cached_wrapper(self):
        # [FUNCTION DOC] Runs three Python subprocesses sharing a temp cache directory. The first
        # compiles a function with the fx graph cache and prints a cached .py wrapper path. The
        # second confirms an fx graph cache hit count of one. The test then edits the wrapper file
        # to add a debug print at the start of its call method, and the third run must print
        # "debug", showing the cached wrapper is editable. Skipped in fbcode/sandcastle.
        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = tmpdir

            step1 = textwrap.dedent(
                """
                import glob
                import os
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                cache_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
                pyfiles = glob.glob(os.path.join(cache_dir, "**", "*.py"), recursive=True)
                print(pyfiles[0])
                """
            )
            wrapper_path = (
                subprocess.check_output([sys.executable, "-c", step1], env=env)
                .decode()
                .strip()
            )

            step2 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._dynamo.utils import counters
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                print(counters["inductor"]["fxgraph_cache_hit"])
                """
            )
            hit = (
                subprocess.check_output([sys.executable, "-c", step2], env=env)
                .decode()
                .strip()
            )
            self.assertEqual(hit, "1")

            with open(wrapper_path) as f:
                src = f.read()
            with open(wrapper_path, "w") as f:
                f.write(
                    src.replace(
                        "def call(self, args):",
                        "def call(self, args):\n        print('debug')",
                    )
                )

            step3 = textwrap.dedent(
                """
                import torch
                import warnings
                from torch._inductor import config

                warnings.filterwarnings("ignore")
                config.fx_graph_cache = True
                config.fx_graph_remote_cache = False
                torch._dynamo.reset()

                @torch.compile(backend="inductor")
                def f(x):
                    return x * 2

                f(torch.ones(2))
                """
            )
            out = subprocess.check_output(
                [sys.executable, "-c", step3], env=env
            ).decode()
            self.assertIn("debug", out)

    def test_cpp_wrapper_none_output_keeps_none_refcount(self):
        # [FUNCTION DOC] Mocks CppWrapperCodeCache.load_async, calls load_pybinding_async with a
        # C++ source whose output handle is null and one output, and checks the generated source
        # passed to load_async contains a new-reference of Py_None, i.e. None outputs are returned
        # with a proper refcount.
        source = textwrap.dedent(
            """
            #include <torch/csrc/inductor/aoti_torch/c/shim.h>

            void inductor_entry_impl(
                AtenTensorHandle* input_handles,
                AtenTensorHandle* output_handles
            ) {
                output_handles[0] = nullptr;
            }
            """
        )

        with mock.patch.object(
            CppWrapperCodeCache, "load_async", return_value=lambda: None
        ) as load_async:
            CppWrapperCodeCache.load_pybinding_async(
                ["std::vector<AtenTensorHandle>"],
                source,
                device_type="cpu",
                num_outputs=1,
            )

        generated_source = load_async.call_args.args[0]
        self.assertIn("Py_NewRef(Py_None)", generated_source)


@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):
    # [CLASS DOC] Main FxGraphCache tests: populating, hitting and bypassing the local/remote fx
    # graph cache across devices, dtypes, dynamic shapes, bundled triton kernels, guards, higher
    # order ops and cache artifact save/load. Every test starts from cleared counters, dynamo and
    # autograd caches, mock cache patching and a cleared artifact manager (see setUp). The class
    # attribute device_type names the GPU backend under test.
    device_type = GPU_TYPE

    def setUp(self):
        # [FUNCTION DOC] Clear the inductor counters, the Dynamo, precompile and AOTAutograd
        # caches and the cache artifact manager, install the mock remote caches via PatchCaches,
        # and reset dynamo.
        super().setUp()
        counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        AOTAutogradCache.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()
        torch._dynamo.reset()

    def tearDown(self):
        # [FUNCTION DOC] Run the base tearDown, then remove the mock cache patches.
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        # [FUNCTION DOC] HELPER: Simulate a fresh process while keeping on-disk caches: clear the
        # AOTAutograd and Dynamo caches and precompile context, purge the in-memory PyCodeCache,
        # reset dynamo, and clear inductor caches.
        AOTAutogradCache.clear()
        DynamoCache.clear()
        PrecompileContext.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    def _check_cpu_thread_count_cache_key_no_input(self, return_expr):
        # [FUNCTION DOC] HELPER: Compile a function with no tensor inputs whose body returns the
        # given expression text, in two subprocesses that share one temp cache directory and
        # differ only in torch thread count (1 then 2). Each run must report its thread count and
        # exactly one fx graph cache miss and no hits, so the thread count must be part of the
        # cache key for input-free graphs.
        script = textwrap.dedent(
            f"""
            import os
            import torch
            import warnings
            from torch._dynamo.utils import counters
            from torch._inductor import config

            warnings.filterwarnings("ignore")
            config.fx_graph_cache = True
            config.fx_graph_remote_cache = False
            torch.set_num_threads(int(os.environ["TEST_THREADS"]))
            torch._dynamo.reset()

            @torch.compile(backend="inductor")
            def f():
                return {return_expr}

            f()
            print(f"threads {{torch.get_num_threads()}}")
            print(f"fxgraph_cache_hit {{counters['inductor']['fxgraph_cache_hit']}}")
            print(f"fxgraph_cache_miss {{counters['inductor']['fxgraph_cache_miss']}}")
            """
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            env = os.environ.copy()
            env["TORCHINDUCTOR_CACHE_DIR"] = tmpdir
            env["PYTHONPATH"] = os.pathsep.join(
                x for x in (os.getcwd(), env.get("PYTHONPATH", "")) if x
            )

            env1 = env.copy()
            env1["TEST_THREADS"] = "1"
            out1 = subprocess.check_output(
                [sys.executable, "-c", script], env=env1
            ).decode()
            self.assertIn("threads 1", out1)
            self.assertIn("fxgraph_cache_hit 0", out1)
            self.assertIn("fxgraph_cache_miss 1", out1)

            env2 = env.copy()
            env2["TEST_THREADS"] = "2"
            out2 = subprocess.check_output(
                [sys.executable, "-c", script], env=env2
            ).decode()
            self.assertIn("threads 2", out2)
            self.assertIn("fxgraph_cache_hit 0", out2)
            self.assertIn("fxgraph_cache_miss 1", out2)

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_cpu_thread_count_cache_key_no_input_factory(self):
        # [FUNCTION DOC] Thread-count cache key check for an input-free graph that builds an
        # arange, adds one and takes the log. Skipped in fbcode/sandcastle.
        self._check_cpu_thread_count_cache_key_no_input(
            "torch.log(torch.arange(1 << 12, dtype=torch.float32) + 1)"
        )

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Skip in fbcode/sandcastle")
    def test_cpu_thread_count_cache_key_no_input_randperm(self):
        # [FUNCTION DOC] Thread-count cache key check for an input-free graph that builds a random
        # permutation and takes the log. Skipped in fbcode/sandcastle.
        self._check_cpu_thread_count_cache_key_no_input(
            "torch.randperm(1 << 12, dtype=torch.float32).log()"
        )

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_triton_launcher", (False, True))
    @parametrize("grad", (False, True))
    def test_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_triton_launcher, grad
    ):
        # [FUNCTION DOC] Parametrized over device, dtype, dynamic shapes, triton bundling, static
        # triton launcher and grad. Compiles a small function three times with resets between: the
        # first run must miss; the second must hit (fxgraph_lookup_write_file becomes nonzero,
        # triton kernels are read back from the bundle when bundling); the third, under a
        # different cache_key_tag, must miss again while the hit count stays. Counts scale by two
        # when grad is on (forward and backward graphs). Skips unsupported combinations (no GPU,
        # bf16 before SM80, static launcher without bundling or on ROCm).
        """
        Verify that we can populate and load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_triton_launcher and not (
            device in STATIC_LAUNCHER_DEVICES and bundle_triton
        ):
            raise unittest.SkipTest(
                "Static triton launcher requires cuda/xpu and triton bundling"
            )
        if use_static_triton_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        grad_multiplier = 2 if grad else 1

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the input vector doubled plus the flattened square of
            # the matrix product of the second argument with itself.
            yy = y @ y
            return x * 2 + yy.view(25)

        a_orig = torch.rand(25, dtype=dtype, device=device)
        b_orig = torch.rand(5, 5, dtype=dtype, device=device)

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
            use_static_triton_launcher=use_static_triton_launcher,
            # Avoid non-deterministic pad_mm benchmarking changing numerics or
            # the number of bundled Triton static autotuners across runs.
            shape_padding=False,
        ):
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            # A first call should miss in the cache.
            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # we expect:
            #  .ttir
            #  .ttgir
            #  .llir
            #  .ptx (cuda) or .spv (xpu)
            #  .json
            #  __grp__.*.json
            # optionally, we can also get
            #  .cubin (CUDA only)
            #  .source (new versions of triton only, triton-lang/triton#6992)

            # to avoid depending on the device and triton version, just assert that
            # we have at least 6 kernels.
            save_and_read_min_artifact_count = 6
            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"], 0
                )
                if use_static_triton_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device in STATIC_LAUNCHER_DEVICES else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"], 0
                    )

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean triton kernels
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            if grad:
                eager_result.sum().backward()
                compiled_result.sum().backward()
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_triton_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier if device in STATIC_LAUNCHER_DEVICES else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device in STATIC_LAUNCHER_DEVICES else 0,
                    )

            self.reset()

            a1 = a_orig.clone().requires_grad_(grad)
            b1 = b_orig.clone().requires_grad_(grad)
            a2 = a_orig.clone().requires_grad_(grad)
            b2 = b_orig.clone().requires_grad_(grad)

            eager_result = fn(a1, b1)
            if grad:
                eager_result.sum().backward()
            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                compiled_result = compiled_fn(a2, b2)
                if grad:
                    compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            if grad:
                self.assertEqual(a1.grad, a2.grad)
                self.assertEqual(b1.grad, b2.grad)

            self.assertEqual(
                counters["inductor"]["fxgraph_cache_miss"], grad_multiplier * 2
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_cache_hit"], grad_multiplier * 1
            )
            self.assertEqual(
                counters["inductor"]["fxgraph_lookup_write_file"], grad_multiplier * 1
            )

            if bundle_triton and device != "cpu":
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_save_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count * 2,
                )
                self.assertGreaterEqual(
                    counters["inductor"]["triton_bundler_read_and_emit_kernel"],
                    grad_multiplier * save_and_read_min_artifact_count,
                )
                if use_static_triton_launcher:
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_save_static_autotuner"],
                        grad_multiplier * 2 if device in STATIC_LAUNCHER_DEVICES else 0,
                    )
                    self.assertEqual(
                        counters["inductor"]["triton_bundler_load_static_autotuner"],
                        grad_multiplier if device in STATIC_LAUNCHER_DEVICES else 0,
                    )

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @functorch_config.patch({"enable_autograd_cache": False})
    def test_local_cache_stats(self):
        # [FUNCTION DOC] Checks remote_cache cache_stats for the LocalFxGraphCache entry: the
        # first compile records one miss and one put with no hits, and after a reset a second call
        # records a hit. Clears the stats before and afterwards.
        from torch._inductor.remote_cache import cache_stats

        cache_stats._stats.clear()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the first argument doubled plus the second.
            return x * 2 + y

        compiled_fn = torch.compile(fn)
        a = torch.rand(25)
        b = torch.rand(25)
        compiled_fn(a, b)

        stats = cache_stats._stats.get("LocalFxGraphCache")
        self.assertIsNotNone(stats)
        self.assertEqual(stats.miss, 1)
        self.assertEqual(stats.put, 1)
        self.assertEqual(stats.hit, 0)

        self.reset()
        compiled_fn(a, b)

        self.assertEqual(stats.hit, 1)
        cache_stats._stats.clear()

    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": False})
    def test_default_dtype_affects_factory_cache_key(self):
        # [FUNCTION DOC] Checks that the default dtype is part of the cache key for factory-only
        # graphs: compiling once under float32 and again under float64 gives one miss each and no
        # hits, and outputs have the matching dtypes. The default dtype is restored and the cache
        # cleared at the end.
        old_dtype = torch.get_default_dtype()
        FxGraphCache.clear()

        def fn():
            # [FUNCTION DOC] HELPER: Return a tuple of five factory results (ones, zeros, tensor
            # from a list, arange, empty) using the current default dtype.
            return (
                torch.ones(2),
                torch.zeros(2),
                torch.tensor([1.0, 2.0]),
                torch.arange(2.0),
                torch.empty(2),
            )

        compiled_fn = torch.compile(fn, backend="inductor")

        try:
            torch.set_default_dtype(torch.float32)
            first = compiled_fn()
            self.assertEqual([x.dtype for x in first], [torch.float32] * 5)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            torch.set_default_dtype(torch.float64)
            expected = fn()
            actual = compiled_fn()
            self.assertEqual([x.dtype for x in actual], [x.dtype for x in expected])
            self.assertEqual([x.dtype for x in actual], [torch.float64] * 5)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        finally:
            torch.set_default_dtype(old_dtype)
            FxGraphCache.clear()

    @requires_triton()
    @config.patch({"fx_graph_remote_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_triton_launcher", (False, True))
    @config.patch(
        {"compile_threads": 1}
    )  # Can't check globalStats if there are workers
    def test_remote_cache_load_function(
        self, device, dtype, dynamic, bundle_triton, use_static_triton_launcher
    ):
        # [FUNCTION DOC] Same shape of test as test_cache_load_function but against the mock
        # remote cache. Runs four fresh_cache compiles and checks the global fx graph stats
        # reflect one miss and three hits with one put; a fifth compile under a different
        # cache_key_tag adds a miss and put. Finally checks every remote key matches the expected
        # versioned fx-graph key pattern. Environment variables are restored via patch dict and
        # the triton cache manager var is removed.
        from unittest.mock import patch

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")
        if use_static_triton_launcher and not (
            device in STATIC_LAUNCHER_DEVICES and bundle_triton
        ):
            raise unittest.SkipTest(
                "Static cuda launcher requires cuda and triton bundling"
            )

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a tuple of the first argument doubled and the matrix
            # product of the second argument with itself.
            return (x * 2, y @ y)

        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)

        with (
            config.patch(
                {
                    "fx_graph_remote_cache": True,
                    "bundle_triton_into_fx_graph_cache": bundle_triton,
                    "use_static_triton_launcher": use_static_triton_launcher,
                    # Disable shape padding to avoid non-deterministic pad_mm
                    # benchmarking which can produce different FX graphs across
                    # fresh_cache() iterations, causing flaky cache stats.
                    "shape_padding": False,
                }
            ),
            patch.dict(os.environ),
            PatchCaches(),
        ):
            os.environ.pop("TRITON_CACHE_MANAGER", None)
            for _ in range(4):
                with fresh_cache():
                    compiled_fn = torch.compile(fn, dynamic=dynamic)
                    self.assertEqual(fn(a, b), compiled_fn(a, b))
                reset()

            self.assertEqual(global_stats.fx_graph, Stats(1, 3, 1))

            with torch.compiler.config.patch({"cache_key_tag": "test"}), fresh_cache():
                compiled_fn = torch.compile(fn, dynamic=dynamic)
                self.assertEqual(fn(a, b), compiled_fn(a, b))

            self.assertEqual(global_stats.fx_graph, Stats(2, 3, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.fx_graph.cache:
            self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c[0-9]+")

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    @parametrize("dynamic", (False, True))
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    def test_cache_hot_load(self, device, dtype, dynamic):
        # [FUNCTION DOC] Hot load test. Compiles once inside a fresh cache and saves cache
        # artifacts, checking one inductor artifact and (on GPU) one autotune artifact. After a
        # reset and cleaning the triton dir, a fresh cache compile without loading must miss
        # again; then loading the saved artifact bytes into another fresh cache must make the
        # compile hit the fx graph cache and write the lookup file. Skips unsupported
        # device/dtype combinations.
        """
        Verify that we can populate and hot load functions from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sine of the first argument matrix-multiplied by
            # the second.
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device)
        b = torch.rand(100, 100, dtype=dtype, device=device)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 1 if device == GPU_TYPE else 0

        self.assertEqual(len(cache_info.inductor_artifacts), 1)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 1)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_triton()
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
            "autotune_local_cache": True,
        }
    )
    @torch._dynamo.config.patch(
        {
            "caching_precompile": True,
        }
    )
    @parametrize("dynamic", (False, True))
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_hot_load_caching_precompile(self, device, dtype, dynamic):
        # [FUNCTION DOC] Hot load test with caching_precompile enabled and grad-requiring inputs.
        # Records artifacts and checks counts (two inductor, one AOT autograd, one precompile,
        # autotune on GPU). After resetting, an unloaded compile misses; after loading the
        # artifacts under the stance that fails on recompile, a recompiled function hits the
        # dynamo cache once.
        """
        Verify that we can populate and hot load functions from the cache.
        """

        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sine of the first argument matrix-multiplied by
            # the second.
            return x.sin() @ y

        a = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)
        b = torch.rand(100, 100, dtype=dtype, device=device, requires_grad=True)

        # Record artifacts
        with fresh_cache():
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            # A first call should miss in the cache.
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 1)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        autotune_expect = 2 if device == GPU_TYPE else 0
        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
        self.assertEqual(len(cache_info.pgo_artifacts), 0)
        self.assertEqual(len(cache_info.precompile_artifacts), 1)

        self.reset()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # We did not load anything so dont hit yet
        with fresh_cache():
            eager_result = fn(a, b)
            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 0)
        self.reset()
        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with fresh_cache(), torch.compiler.set_stance("fail_on_recompile"):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), autotune_expect)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 1)
            self.assertEqual(len(cache_info.pgo_artifacts), 0)
            self.assertEqual(len(cache_info.precompile_artifacts), 1)

            # With caching precompile, we have to re torch.compile the function
            # to trigger cache lookup
            compiled_fn = torch.compile(fn, dynamic=dynamic)

            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            compiled_result.sum().backward()
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 0)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_miss"], 2)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)

    @requires_cuda_and_triton
    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    @torch._dynamo.config.patch(
        {
            "caching_precompile": True,
        }
    )
    def test_cache_hot_load_caching_precompile_autocast(self):
        # [FUNCTION DOC] Requires CUDA and triton. With caching_precompile on, compiles a function
        # under autocast, saves artifacts (one precompile artifact), resets, loads them under the
        # fail on recompile stance and checks the compiled function under autocast matches eager
        # and hits the dynamo cache once.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input plus itself times one.
            return x + 1 * x

        x = torch.randn(3, 2, device="cuda")

        with fresh_cache():
            compiled_fn = torch.compile(fn)
            with torch.amp.autocast(device_type="cuda"):
                eager_result = fn(x)
                compiled_result = compiled_fn(x)
            self.assertEqual(eager_result, compiled_result)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifact_bytes, cache_info = artifacts
        self.assertEqual(len(cache_info.precompile_artifacts), 1)

        self.reset()
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        with fresh_cache(), torch.compiler.set_stance("fail_on_recompile"):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
            self.assertEqual(len(cache_info.precompile_artifacts), 1)

            compiled_fn = torch.compile(fn)
            with torch.amp.autocast(device_type="cuda"):
                eager_result = fn(x)
                compiled_result = compiled_fn(x)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["dynamo_cache"]["dynamo_cache_hit"], 1)

    @config.patch(
        {
            "fx_graph_cache": True,
            "fx_graph_remote_cache": False,
        }
    )
    def test_cache_hot_load_repeat(self):
        # [FUNCTION DOC] Checks repeated save and load of artifacts around the need_serialize flag
        # of the artifact manager. After saving and then loading artifacts the flag is false and a
        # hit is observed; compiling a new input shape afterward produces a miss and makes the
        # flag true.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the matrix product of the first argument with the sine
            # of the second.
            return x @ y.sin()

        compiled_fn = torch.compile(fn, dynamic=False)

        a = torch.randn(4, 4)
        b = torch.randn(4, 4)

        a2 = torch.randn(4, 8)
        b2 = torch.randn(8, 4)

        with fresh_cache():
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())
        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.reset()

        with fresh_cache():
            torch.compiler.load_cache_artifacts(artifact_bytes)
            eager_result = fn(a, b)
            compiled_result = compiled_fn(a, b)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertFalse(torch.compiler._cache.CacheArtifactManager.need_serialize())

        self.reset()

        with fresh_cache():
            eager_result = fn(a2, b2)
            compiled_result = compiled_fn(a2, b2)
            self.assertEqual(eager_result, compiled_result)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertTrue(torch.compiler._cache.CacheArtifactManager.need_serialize())

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo(self):
        # [FUNCTION DOC] Hot load with automatic dynamic profile guided optimization (local PGO).
        # Two differently shaped calls give two inductor and two PGO artifacts; after reset and
        # loading the saved artifacts under the same job id, calls with new shapes compile only
        # once (the PGO data already marks the dim dynamic) and hit the fx graph cache once.
        """
        Verify that we can populate and hot load functions from the cache with pgo.
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input doubled.
            return x * 2

        # Record artifacts
        with torch.compiler.config.patch(job_id=self.id()), fresh_cache():
            f(torch.randn(2, 3))
            f(torch.randn(2, 4))
            self.assertEqual(backend.frame_count, 2)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        artifacts = torch.compiler.save_cache_artifacts()

        self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.inductor_artifacts), 2)
        self.assertEqual(len(cache_info.autotune_artifacts), 0)
        self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with torch.compiler.config.patch({"job_id": self.id()}), fresh_cache():
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.inductor_artifacts), 2)
            self.assertEqual(len(cache_info.autotune_artifacts), 0)
            self.assertEqual(len(cache_info.aot_autograd_artifacts), 0)
            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @torch._dynamo.config.patch(automatic_dynamic_local_pgo=True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch({"fx_graph_cache": True, "fx_graph_remote_cache": False})
    def test_cache_hot_load_pgo_swap_file_names(self):
        # [FUNCTION DOC] Like the PGO hot load test but the mast job name and version are mocked
        # to differ between saving (name foo, version 5) and loading (name bar, version 10). The
        # loaded PGO artifacts must still apply, so two later shapes compile in only one frame.
        """
        Verify that we can populate and hot load functions from the cache with pgo
        with file name swapping
        """

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")

        @torch.compile(backend=backend, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input doubled.
            return x * 2

        # Record artifacts
        with mock.patch(
            "torch._utils_internal.get_mast_job_name_version", return_value=("foo", 5)
        ):
            with fresh_cache():
                f(torch.randn(2, 3))
                f(torch.randn(2, 4))
                self.assertEqual(backend.frame_count, 2)

            artifacts = torch.compiler.save_cache_artifacts()

            self.assertIsNotNone(artifacts)

        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.pgo_artifacts), 2)

        self.reset()
        backend.clear()

        # Clean triton kernels
        shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

        # Hot load and hit
        with (
            mock.patch(
                "torch._utils_internal.get_mast_job_name_version",
                return_value=("bar", 10),
            ),
            fresh_cache(),
        ):
            cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)

            self.assertEqual(len(cache_info.pgo_artifacts), 2)

            f(torch.randn(2, 5))
            f(torch.randn(2, 6))
            self.assertEqual(backend.frame_count, 1)

    def test_cache_hot_load_empty(self):
        # [FUNCTION DOC] Checks that saving cache artifacts when nothing was recorded
        # returns None.
        self.assertIsNone(torch.compiler.save_cache_artifacts())

    def test_cache_hot_load_generic(self):
        # [FUNCTION DOC] Checks the generic artifact registration path with a custom artifact
        # type. A stub cache records artifacts in the manager on save and on lookup. After saving
        # three entries the artifacts round trip through save and load, repopulating the stub
        # cache, and only the looked-up entry is recorded after clearing the manager and doing
        # one lookup.
        class CacheStub:
            # [CLASS DOC] HELPER: Minimal dictionary-backed cache that records every
            # save and every successful lookup as an artifact of the test artifact type
            # in the CacheArtifactManager.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the empty backing dictionary.
                self.cache = {}

            def lookup(self, key):
                # [FUNCTION DOC] HELPER: Return the stored content for the key, or None if absent.
                # A hit is recorded as a cache artifact.
                content = self.cache.get(key)
                if content is None:
                    return None

                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )
                return content

            def save(self, key, content):
                # [FUNCTION DOC] HELPER: Store the content under the key and record it as a
                # cache artifact.
                self.cache[key] = content
                CacheArtifactManager.record_artifact(
                    ArbitraryCacheArtifact.type(), key, content
                )

            def clear(self):
                # [FUNCTION DOC] HELPER: Remove all stored entries.
                self.cache.clear()

        cache_stub = CacheStub()

        @CacheArtifactFactory.register
        class ArbitraryCacheArtifact(CacheArtifact):
            # [CLASS DOC] HELPER: Test-only cache artifact type registered with
            # CacheArtifactFactory. Its type name is "test", content is encoded as utf-8 bytes,
            # and populating writes the decoded content into the enclosing test's stub cache under
            # the artifact key.
            @override
            def populate_cache(self) -> None:
                # [FUNCTION DOC] HELPER: Decode the artifact content as text and store it in the
                # stub cache under this artifact's key.
                cache_stub.cache[self.key] = self.content.decode()

            @override
            @staticmethod
            def type() -> str:
                # [FUNCTION DOC] HELPER: Return the artifact type name, the string "test".
                return "test"

            @override
            @staticmethod
            def encode(content: str) -> bytes:
                # [FUNCTION DOC] HELPER: Encode the text content to bytes using the
                # default encoding.
                return content.encode()

        test_cache = {"1": "foo", "2": "bar", "foo": "bar"}

        for k, v in test_cache.items():
            cache_stub.save(k, v)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifact_bytes, cache_info = artifacts

        self.assertEqual(len(cache_info.test_artifacts), 3)

        cache_stub.clear()
        CacheArtifactManager.clear()

        cache_info = torch.compiler.load_cache_artifacts(artifact_bytes)
        self.assertEqual(len(cache_info.test_artifacts), 3)
        self.assertEqual(cache_stub.cache, test_cache)

        CacheArtifactManager.clear()
        cache_stub.lookup("foo")
        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        _, cache_info = artifacts
        self.assertEqual(len(cache_info.test_artifacts), 1)

    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.float64))
    @parametrize("dynamic", (False, True))
    def test_cache_load_model(self, device, dtype, dynamic):
        # [FUNCTION DOC] Compiles a function that runs a conv model forward and backward and
        # returns the parameter gradients. The first call misses in the fx graph cache, the second
        # call after a reset only hits, and gradients from both calls match. Parametrized over
        # device, dtype, dynamic.
        """
        Verify that we can populate and load models from the cache.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn(mod, x):
            # [FUNCTION DOC] HELPER: Zero the module gradients, run forward and backward on the
            # sum of the output, and return the list of parameter gradients.
            mod.zero_grad()
            mod(x).sum().backward()
            return [p.grad for p in mod.parameters()]

        compiled_fn = torch.compile(fn, dynamic=dynamic)

        mod = MyModelConv2d().to(device=device, dtype=dtype)
        inp = torch.randn(2, 3, 16, 32, device=device, dtype=dtype)

        # The first call should see all cache misses.
        counters.clear()
        grads1 = compiled_fn(mod, inp)
        self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # The second should see all hits. (First reset so in-memory guards
        # don't prevent compilation).
        counters.clear()
        self.reset()
        grads2 = compiled_fn(mod, inp)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

        # And the results should be the same.
        self.assertEqual(grads1, grads2)

    @largeTensorTest("64GB", device=GPU_TYPE, inductor=True)
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE,))
    @parametrize("dtype", (torch.float16, torch.bfloat16))
    def test_cache_load_with_guards_int32_bounds(self, device, dtype):
        # [FUNCTION DOC] Needs about 64GB of GPU memory. Compiles an add-to-self function with
        # dynamic shapes over input shapes below and above the int32 element limit. Each new shape
        # causes a recompile with a cache miss (guards are not reset); after a reset the same
        # shapes hit. Results of both runs are equal.
        """
        Test caching the same graph, but under conditions that introduce guards
        for tensor sizes < int32.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires CUDA SM80 or later")

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a tuple of each argument added to itself.
            return (x + x, y + y)

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different shapes, varying whether the total
        # size is below or above int32. For each combination, we expect
        # different guards around whether the symbolic sizes do or do
        # not exceed int32.
        shapes = (
            ((5, 6), (7, 8)),
            ((5, 6), (47000, 47001)),
            ((47000, 47001), (5, 6)),
        )
        for a_shape, b_shape in shapes:
            a = torch.rand(a_shape, device=device, dtype=dtype)
            b = torch.rand(b_shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. We expect guards to have been
            # added that will be violated with the new shape. We should
            # see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(a, b)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit. (Reset here to force compilation).
            counters.clear()
            self.reset()
            res2 = compiled_fn(a, b)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("dtype", (torch.float32, torch.bfloat16))
    def test_cache_load_with_guards_static_bounds(self, device, dtype):
        # [FUNCTION DOC] Compiles adaptive average pooling with dynamic shapes over three input
        # shapes; pooling makes spatial sizes static so each new shape recompiles and misses, and
        # after a reset hits the cache. Results of both runs are equal.
        """
        Test caching the same graph, but under conditions that introduce guards
        for static bounds.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")
        if (
            device == "cuda"
            and torch.version.hip is None
            and dtype == torch.bfloat16
            and not SM80OrLater
        ):
            raise unittest.SkipTest("requires SM80 or later")

        # See lowering; for all of the pooling operators, we always guard and
        # make the height/width static.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return adaptive average pooling of the input to a fixed 5 by
            # 7 output.
            return torch.nn.functional.adaptive_avg_pool2d(x, [5, 7])

        compiled_fn = torch.compile(fn, dynamic=True)

        # Iterate over different input shapes. Each new shape should cause
        # a cache miss.
        shapes = ((1, 64, 8, 9), (1, 64, 9, 10), (1, 64, 10, 11))
        for shape in shapes:
            x = torch.rand(shape, device=device, dtype=dtype)

            # AVOID a dynamo reset here. For each cache hit, we expect guards
            # to have been added that will be violated with each new shape.
            # We should see a recompilation (along with a cache miss).
            counters.clear()
            res1 = compiled_fn(x)
            self.assertGreater(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # A second call should hit.
            counters.clear()
            self.reset()
            res2 = compiled_fn(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.assertEqual(res1, res2)

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_no_arguments_tensor_device_guards(self):
        # [FUNCTION DOC] Requires multiple GPUs. A function with no inputs creates a random CUDA
        # tensor. Running it under device guard 0 and then, after a reset, under device guard 1
        # must give a tensor on the matching device, so the device index must be part of the
        # cache key.
        """
        Usually, when there are example inputs, the device index of the inputs
        is sufficient to make sure we don't cache hit with the results from different
        cuda devices.
        When the input has no arguments, we still need to have the cuda
        device index in the cache key.
        """

        @torch.compile
        def f():
            # [FUNCTION DOC] HELPER: Return a one element tuple with a random CUDA tensor of
            # three values.
            y = torch.randn(3, device="cuda")
            return (y,)

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:0"))
        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:1"))

    @config.patch("fx_graph_cache", True)
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @config.patch("fx_graph_remote_cache", False)
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    @requires_cuda_and_triton
    def test_tensor_device_guards_cpu_tensor(self):
        # [FUNCTION DOC] Requires multiple GPUs. Calls a compiled function on a CPU tensor under
        # CUDA device guard 0 then, after a reset, under guard 1. Results stay on CPU and the
        # second compile must be a cache hit (CPU inputs do not make the key device dependent).
        """
        CPU tensor arguments should still cache hit
        """

        @torch.compile
        def f(x):
            # [FUNCTION DOC] HELPER: Return the sine of the input.
            return x.sin()

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.reset()
        # Should not cache hit with device guard
        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f(torch.randn(3, device="cpu"))
            self.assertEqual(result.device, torch.device("cpu"))

        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_constant_handling(self, device):
        # [FUNCTION DOC] Checks two functions that differ only in a constant tensor (range zero to
        # eleven versus one to twelve) are cached separately: both compile as misses with no hits.
        """
        Test that different constants are recognized correctly.
        """
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        def fn1(x):
            # [FUNCTION DOC] HELPER: Return the input plus a constant tensor of the integers zero
            # to eleven.
            return x + torch.tensor(list(range(12)), device=device)

        def fn2(x):
            # [FUNCTION DOC] HELPER: Return the input plus a constant tensor of the integers one
            # to twelve.
            return x + torch.tensor(list(range(1, 13)), device=device)

        a = torch.rand(12, device=device)

        compiled_fn1 = torch.compile(fn1)
        compiled_fn2 = torch.compile(fn2)

        # A call to fn1 should miss in the cache.
        self.assertEqual(fn1(a), compiled_fn1(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A call to fn2 should also miss (the constant is different)
        self.assertEqual(fn2(a), compiled_fn2(a))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @requires_gpu()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_hit_aligned_and_unaligned_inputs(self):
        # [FUNCTION DOC] Requires a GPU. Compiles relu times two with an aligned input (miss),
        # resets, then calls with an unaligned view of the same shape (storage offset one) which
        # must hit the same cache entry, then aligned again (second hit). Alignment is not part of
        # the cache key; the artifact realigns inputs at runtime.
        """
        A function compiled with an aligned GPU input should share a cache
        entry with the same function called with an unaligned GPU input.

        `inputs_to_check` (derived from the example input's actual alignment)
        used to feed into the cache key, so an aligned vs. unaligned first
        call produced different keys and a second cache miss. It is no longer
        part of the key: codegen still picks the aligned vs. unaligned kernel
        based on the example, but the compiled artifact carries its own runtime
        realignment (copy_if_misaligned for the inputs the kernel assumes
        aligned). The cached graph therefore services both alignments --
        whichever variant compiled first realigns the other at runtime.
        """
        device = GPU_TYPE

        def fn(x):
            # [FUNCTION DOC] HELPER: Return relu of the input multiplied by two.
            return torch.nn.functional.relu(x) * 2

        # Aligned: storage_offset == 0.
        aligned = torch.randn(1024, device=device)
        # Unaligned: storage_offset == 1, so data_ptr is offset by
        # sizeof(float) and is no longer 16-byte aligned.
        unaligned = torch.randn(1024 + 16, device=device)[1:-15]
        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        self.assertEqual(aligned.shape, unaligned.shape)

        compiled_fn = torch.compile(fn)

        # First call with the aligned tensor: cache miss.
        self.assertEqual(fn(aligned), compiled_fn(aligned))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # Drop in-memory state so the cache lookup is exercised.
        self.reset()

        # Call with the unaligned tensor: should hit.
        self.assertEqual(fn(unaligned), compiled_fn(unaligned))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # And back to aligned: still a hit, same cached graph.
        self.reset()
        self.assertEqual(fn(aligned), compiled_fn(aligned))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

    @requires_gpu()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_aligned_unaligned_inputs_no_recompile(self):
        # [FUNCTION DOC] Requires a GPU. Calls a compiled function in the same process with an
        # aligned and then an unaligned input of identical shape and checks no extra dynamo frame
        # is compiled (frame count stays one), since alignment is not guarded.
        """
        Alignment is deliberately not guarded (storage_offset guards cause
        recompiles), so calling a compiled function in-process with an aligned
        then an unaligned variant of the same input must reuse the same graph
        without recompiling. The unaligned input is realigned at runtime via
        copy_if_misaligned.
        """
        device = GPU_TYPE

        def fn(x):
            # [FUNCTION DOC] HELPER: Return relu of the input multiplied by two.
            return torch.nn.functional.relu(x) * 2

        aligned = torch.randn(1024, device=device)
        unaligned = torch.randn(1024 + 16, device=device)[1:-15]
        self.assertEqual(aligned.data_ptr() % 16, 0)
        self.assertNotEqual(unaligned.data_ptr() % 16, 0)
        self.assertEqual(aligned.shape, unaligned.shape)

        backend = torch._dynamo.testing.CompileCounterWithBackend("inductor")
        compiled_fn = torch.compile(fn, backend=backend)

        self.assertEqual(fn(aligned), compiled_fn(aligned))
        self.assertEqual(backend.frame_count, 1)

        # A differently-aligned input must NOT trigger a recompile.
        self.assertEqual(fn(unaligned), compiled_fn(unaligned))
        self.assertEqual(backend.frame_count, 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("variant", ("v1", "v2"))
    def test_auto_functionalized_caching(self, variant):
        # [FUNCTION DOC] Parametrized over the auto functionalized v1 and v2 config. Compiles a
        # function calling a mutating custom operator passed as argument: first call misses, a
        # call after reset hits (writing the lookup file), and a call with a different mutating
        # operator misses again. Raises an assertion error if the variant is neither v1 nor v2.
        if variant == "v1":
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=False)
        else:
            if variant != "v2":
                raise AssertionError(f"Expected 'v2', got {variant!r}")
            patch = torch._inductor.config.patch(enable_auto_functionalized_v2=True)

        @torch.library.custom_op("mylib::sin_inplace", mutates_args=["x"])
        def sin_inplace(x: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op that mutates its argument in place by taking
            # its sine.
            x.sin_()

        @torch.library.custom_op("mylib::cos_inplace", mutates_args=["x"])
        def cos_inplace(x: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Custom op that mutates its argument in place by taking
            # its cosine.
            x.cos_()

        @torch.compile(fullgraph=True)
        def fn(x, op):
            # [FUNCTION DOC] HELPER: Allocate an empty tensor like the input, pass it to the given
            # mutating operator, and return it.
            y = torch.empty_like(x)
            op(y)
            return y

        x = torch.randn(3)

        with patch:
            # A first call should miss in the cache.
            fn(x, sin_inplace)
            self.reset()
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()
            fn(x, sin_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

            # A third call with different operator should have a cache miss
            self.reset()
            fn(x, cos_inplace)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu_and_triton
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @with_tf32_off
    def test_flex_attention_caching(self):
        # [FUNCTION DOC] Needs a GPU and triton, with tf32 off. Compiles flex attention with a
        # causal block mask and a score modifier adding the query-key difference: first call
        # misses, call after reset hits, and a second function using a different score modifier
        # misses again.
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention

        block_mask = create_block_mask(
            lambda b, h, q, kv: q >= kv, None, None, 512, 512
        )

        def score_mod(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: Score modifier adding the difference of query and key indices
            # to the score.
            return score + (q - kv)

        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: Run flex attention with the first score modifier and the
            # block mask.
            return flex_attention(q, k, v, score_mod=score_mod, block_mask=block_mask)

        def score_mod2(score, b, h, q, kv):
            # [FUNCTION DOC] HELPER: Score modifier that returns the score unchanged.
            return score

        def fn2(q, k, v):
            # [FUNCTION DOC] HELPER: Run flex attention with the identity score modifier and the
            # block mask.
            return flex_attention(q, k, v, score_mod=score_mod2, block_mask=block_mask)

        a, b, c = (torch.randn(1, 4, 512, 64).to(GPU_TYPE) for _ in range(3))
        compiled_fn = torch.compile(fn)
        compiled_fn2 = torch.compile(fn2)

        atol, rtol = 1e-4, 1e-4

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 0)

        # A second call should hit. (First reset so in-memory guards
        # don't prevent compilation).
        self.reset()
        self.assertEqual(fn(a, b, c), compiled_fn(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

        # A third call with different score_mod should have a cache miss
        self.reset()
        self.assertEqual(fn2(a, b, c), compiled_fn2(a, b, c), atol=atol, rtol=rtol)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_lookup_write_file"], 1)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_higher_order_op_bypass(self, bundle_triton):
        # [FUNCTION DOC] Checks that a graph containing a conditional higher order op with dynamic
        # shapes bypasses the fx graph cache (no misses or hits, bypass counter positive) and that
        # triton bundling start and end still work with a bypass.
        """
        Verify that we bypass the cache when we have a higher order ops
        and that bundler start/end works with a cache bypass.
        """

        def fn(x):
            # [FUNCTION DOC] HELPER: Run a conditional on the first dimension size choosing
            # between cosine and sine branches.
            def true_fn(x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Branch returning the cosine of the input.
                return x.cos()

            def false_fn(x: torch.Tensor):
                # [FUNCTION DOC] HELPER: Branch returning the sine of the input.
                return x.sin()

            return torch.cond(x.shape[0], true_fn, false_fn, (x,))

        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
        ):
            compiled_fn = torch.compile(fn, dynamic=True, fullgraph=True)

            x = torch.randn(4, 4, device=GPU_TYPE)
            compiled_fn(x)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertGreater(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_graphsafe_rng_cache_hit(self):
        # [FUNCTION DOC] Checks that the graphsafe RNG state higher order op produced by
        # activation checkpointing with RNG preservation does not make the graph bypass the cache.
        # A cold compile shows two misses (forward and backward) with no bypass; after reset, two
        # hits; results match; and a different seed on the cached code gives a different output.
        """graphsafe_run_with_rng_state must not bypass FxGraphCache.

        Activation checkpointing with preserve_rng_state=True inserts
        graphsafe_run_with_rng_state into post-partition graphs when the
        wrapped region contains nondeterministic ops (e.g. SDPA dropout).
        The sibling RNG HOPs (run_and_save_rng_state, run_with_rng_state,
        run_dtensor_rng_op) are all cacheable; this one must be too.
        """
        import torch.nn.functional as F
        from torch.utils.checkpoint import checkpoint

        class SDPADrop(torch.nn.Module):
            # [CLASS DOC] HELPER: Module applying scaled dot product attention with dropout
            # probability 0.1 to query, key and value tensors.
            def forward(self, q, k, v):
                # [FUNCTION DOC] HELPER: Return the attention output with dropout.
                return F.scaled_dot_product_attention(q, k, v, dropout_p=0.1)

        inner = SDPADrop()
        q = torch.randn(2, 4, 16, 16, device=GPU_TYPE, requires_grad=True)
        k = torch.randn(2, 4, 16, 16, device=GPU_TYPE)
        v = torch.randn(2, 4, 16, 16, device=GPU_TYPE)

        def model(q, k, v):
            # [FUNCTION DOC] HELPER: Run the SDPADrop module under non-reentrant activation
            # checkpointing with RNG state preservation.
            return checkpoint(
                inner, q, k, v, use_reentrant=False, preserve_rng_state=True
            )

        def run(q):
            # [FUNCTION DOC] HELPER: Seed the RNG, run the compiled checkpointed model,
            # backpropagate the summed output, and return the detached output with a clone of the
            # query gradient.
            torch.manual_seed(99)
            out = torch.compile(model, fullgraph=True)(q, k, v).sum()
            out.backward()
            return out.detach(), q.grad.clone()

        # Cold compile — fwd and bwd graphs each miss once
        out1, grad1 = run(q)
        q.grad = None

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

        # Warm compile — fwd and bwd graphs each hit
        counters.clear()
        self.reset()

        out2, grad2 = run(q)

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

        self.assertEqual(out1, out2)
        self.assertEqual(grad1, grad2)

        # Different seed through cached code must produce different output —
        # proves the cached artifact is parameterized by runtime RNG state
        q.grad = None
        torch.manual_seed(0)
        out3 = torch.compile(model, fullgraph=True)(q, k, v).sum()
        self.assertFalse(torch.allclose(out1, out3))

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_triton_higher_order_op(self, bundle_triton):
        # [FUNCTION DOC] Checks the user defined triton kernel higher order op is cacheable. An
        # add kernel compiles as a miss, after reset and clearing PyCodeCache and the triton
        # directory it hits, and a function using a different (subtract) kernel misses. No
        # bypasses occur.
        """
        Verify that we can cache user defined triton kernel higher order op
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Launch the add kernel over the input elements with block size
            # 4, writing into the first argument, and return it.
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=4)
            return x

        def fn2(x, y):
            # [FUNCTION DOC] HELPER: Launch the subtract kernel over the input elements with block
            # size 4, writing into the first argument, and return it.
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            sub_kernel[grid](x, y, x, n_elements, BLOCK_SIZE=4)
            return x

        with config.patch(bundle_triton_into_fx_graph_cache=bundle_triton):
            compiled_fn = torch.compile(fn, fullgraph=True)
            compiled_fn2 = torch.compile(fn2, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn2(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @parametrize("bundle_triton", (False, True))
    def test_triton_higher_order_op_different_configs(self, bundle_triton):
        # [FUNCTION DOC] Checks that the same triton kernel wrapped with different autotune config
        # sets (block size 128 versus 64) produces separately cached graphs: the first function
        # misses, hits after reset (with PyCodeCache and triton dir cleaned), and the second
        # function misses again. No bypass occurs. Parametrized on triton bundling.
        """
        Verify that user defined triton kernel with
        different configs are cached separately.
        """

        add_kernel1 = triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 128}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 128}, num_stages=4, num_warps=4),
            ],
            key=[],
        )(add_kernel)

        add_kernel2 = triton.autotune(
            configs=[
                triton.Config({"BLOCK_SIZE": 64}, num_stages=3, num_warps=8),
                triton.Config({"BLOCK_SIZE": 64}, num_stages=4, num_warps=4),
            ],
            key=[],
        )(add_kernel)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Launch the autotuned add kernel with the 128 block configs
            # over the input elements, writing into the first argument, and return it.
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel1[grid](x, y, x, n_elements)
            return x

        def fn2(x, y):
            # [FUNCTION DOC] HELPER: Launch the autotuned add kernel with the 64 block configs
            # over the input elements, writing into the first argument, and return it.
            n_elements = x.numel()
            grid = lambda meta: (  # noqa: E731
                triton.cdiv(n_elements, meta["BLOCK_SIZE"]),
            )
            add_kernel2[grid](x, y, x, n_elements)
            return x

        with config.patch(bundle_triton_into_fx_graph_cache=bundle_triton):
            compiled_fn = torch.compile(fn, fullgraph=True)
            compiled_fn2 = torch.compile(fn2, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn2(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @requires_gpu()
    @requires_triton()
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"compile_threads": 1})
    @parametrize("bundle_triton", (False, True))
    @parametrize("use_static_triton_launcher", (False, True))
    def test_triton_op(self, bundle_triton, use_static_triton_launcher):
        # [FUNCTION DOC] Registers a triton_op custom operator that launches the captured add
        # kernel and checks it compiles as a miss, then hits the fx graph cache after a reset with
        # cleaned PyCodeCache and triton dir, with no bypass. With the static launcher
        # compile_threads is forced to one. Parametrized on bundling and static launcher; static
        # launcher is skipped on ROCm.
        if use_static_triton_launcher and TEST_WITH_ROCM:
            raise unittest.SkipTest("Static cuda launcher doesn't work with ROCM")

        libname = "my_cool_namespace"
        opname = "my_triton_operator"

        @torch._library.triton_op(f"{libname}::{opname}", mutates_args={})
        def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Triton operator body that allocates an output like the first
            # input and fills it by launching the captured add kernel with a one dimensional grid
            # and block size 16.
            output = torch.empty_like(x)
            n_elements = output.numel()

            def grid(meta):
                # [FUNCTION DOC] HELPER: Return a one element grid tuple with the number of blocks
                # needed for the output element count.
                return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            capture_triton(add_kernel)[grid](x, y, output, n_elements, 16)
            return output

        def f(x, y):
            # [FUNCTION DOC] HELPER: Call the registered triton operator on the two inputs.
            return add(x, y)

        compile_threads = 1 if use_static_triton_launcher else config.compile_threads
        with config.patch(
            bundle_triton_into_fx_graph_cache=bundle_triton,
            use_static_triton_launcher=use_static_triton_launcher,
            compile_threads=compile_threads,
        ):
            compiled_fn = torch.compile(f, fullgraph=True)

            x = torch.randn(4, device=GPU_TYPE)
            y = torch.randn(4, device=GPU_TYPE)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            # A second call should hit. (First reset so in-memory guards
            # don't prevent compilation).
            self.reset()

            # Clean PyCodeCache and triton kernels
            PyCodeCache.cache_clear()
            shutil.rmtree(os.path.join(cache_dir(), "triton"), ignore_errors=True)

            compiled_fn(x, y)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_generated_kernel_count(self):
        # [FUNCTION DOC] With logging of inductor metrics on, checks that the
        # generated_kernel_count metric grows on a cache hit as well as on a miss (1 after
        # the miss, 2 after the hit). Resets the metrics beforehand and the logging
        # settings afterwards.
        """
        Test that we bump the generated_kernel_count metric on a cache hit.
        """
        torch._logging.set_logs(inductor_metrics=True)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a one element tuple with the product of the arguments
            # plus the second argument.
            return (x * y + y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)

        metrics.reset()
        self.assertEqual(metrics.generated_kernel_count, 0)

        # Verify the "miss" case.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
        self.assertEqual(metrics.generated_kernel_count, 1)

        # Verify the "hit" case
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(metrics.generated_kernel_count, 2)
        torch._logging.set_logs()

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_inductor_counters(self):
        # [FUNCTION DOC] Checks that the fxgraph_cache_hit counter is zero after the first compile
        # of a matrix multiply on CPU and one after a reset and second call.
        """
        Test that we bump the inductor counters on a cache hit.
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the matrix product of the two arguments.
            return torch.mm(a, b)

        a = torch.rand(8, 32, device="cpu")
        b = torch.rand(32, 8, device="cpu")

        compiled_fn = torch.compile(fn)

        # Verify the "miss" case.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # Verify the "hit" case.
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_clear(self):
        # [FUNCTION DOC] Checks the miss, hit, then after FxGraphCache.clear miss sequence: a
        # first call misses, a call after reset hits, and a call after clearing the on-disk cache
        # misses again.
        """
        Test clearing the cache.
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a one element tuple with the elementwise product of
            # the inputs.
            return (x * y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)

        # A first call should miss in the cache.
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # A second call should hit.
        counters.clear()
        self.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # Clear the cache; now we should miss.
        counters.clear()
        self.reset()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_nt(self):
        # [FUNCTION DOC] Checks caching with jagged nested tensor inputs. Two inputs with
        # different numbers of rows share a single graph (one miss, no hit); after a reset both
        # compiles use the cache (no miss, one hit).
        def gen_nt(r):
            # [FUNCTION DOC] HELPER: Build a jagged nested tensor from random values with the
            # given row count and 16 columns, using fixed offsets ending at the row count.
            values = torch.randn(r, 16)
            offsets = torch.tensor([0, 2, 3, 6, 13, r])
            return torch.nested.nested_tensor_from_jagged(values, offsets)

        def fn(nt):
            # [FUNCTION DOC] HELPER: Take the sine of a nested tensor when its values row count is
            # divisible by 16, otherwise the cosine.
            if nt.values().size(0) % 16 == 0:
                return nt.sin()
            return nt.cos()

        inp1 = gen_nt(19)
        inp2 = gen_nt(20)

        counters.clear()
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn)(inp1)
        torch.compile(fn)(inp2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_with_symint_non_arg_guard(self):
        # [FUNCTION DOC] Checks that a dynamic compile whose guard involves a symbolic int that is
        # not a tensor graph argument is stored and reloaded: one miss first, then after a reset
        # one hit.
        def fn(x, ref_id):
            # [FUNCTION DOC] HELPER: Multiply by one when a local constant 22 equals the second
            # argument, otherwise multiply by zero.
            self_id = 22
            if self_id == ref_id:
                x = torch.mul(x, 1.0)
            else:
                x = torch.mul(x, 0)
            return x

        x = torch.ones(2)

        counters.clear()
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        torch.compile(fn, fullgraph=True, dynamic=True)(x, 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_guard(self):
        # [FUNCTION DOC] Checks that guards are stored in the cache entry: a dynamic compile with
        # a value above 5 and, after a reset, a compile with a value of 4 both miss (the cached
        # guard rejects the second), and the two results differ.
        def f(x, val):
            # [FUNCTION DOC] HELPER: Return the sine of the input when the value is greater than
            # 5, else the cosine.
            if val > 5:
                return x.sin()
            else:
                return x.cos()

        x = torch.ones(2)
        a = torch.compile(f, dynamic=True)(x, 6)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()
        b = torch.compile(f, dynamic=True)(x, 4)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.assertNotEqual(a, b)

    @config.patch({"fx_graph_cache": False, "fx_graph_remote_cache": False})
    @requires_cuda_and_triton
    @unittest.expectedFailure  # TODO: pass in optimize_mem at runtime
    def test_async_compile_cache(self):
        # [FUNCTION DOC] Expected to fail (marked as a known TODO: optimize_mem must be passed at
        # runtime). Compiles a custom autograd function and its backward on CUDA with the fx graph
        # cache off and expects one async compile cache miss and one hit.
        class SimpleFunction(torch.autograd.Function):
            # [CLASS DOC] HELPER: Autograd function that doubles the input in forward and doubles
            # the incoming gradient in backward.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Return the input doubled.
                return x * 2

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Return the gradient doubled.
                return grad_output * 2

        x = torch.rand([10], requires_grad=True, device="cuda")
        counters.clear()

        sf = SimpleFunction
        out = torch.compile(sf.apply)(x)
        out.sum().backward()

        self.assertEqual(counters["inductor"]["async_compile_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["async_compile_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    def test_cache_guard_overspec(self):
        # [FUNCTION DOC] Checks guard over-specialization: a compiled module checks membership
        # of an arange in a closed over tensor, called for sizes one to four. There are two
        # misses (sizes recompile once for dynamic shapes) and no hits; after a reset, two hits
        # and no misses.
        b = torch.tensor([0, 2, 4, 6, 8])

        @torch.compile
        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module (compiled via the compile decorator on the class) whose
            # forward tests whether each element of the input is contained in the enclosing
            # test's tensor.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the membership mask of the input.
                return torch.isin(x, b)

        model = MyModel()

        counters.clear()

        for i in range(1, 5):
            model(torch.arange(i))

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        self.reset()
        counters.clear()

        for i in range(1, 5):
            model(torch.arange(i))

        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_cache_guard_sqrt_no_recompilation(self):
        # [FUNCTION DOC] Regression test for a spurious recompile after cache load when a guard
        # contains the opaque square root function. Compiles dynamically for one size (one miss),
        # resets, then the next compile hits the cache once. Later calls at other sizes must add
        # no misses.
        """
        Verify that guards containing OpaqueUnaryFn_sqrt (math.sqrt)
        do not cause spurious recompilations when loaded from cache.

        Previously, the guard printer emitted math.sqrt(...) for
        OpaqueUnaryFn_sqrt, which when re-evaluated with SymInt inputs during
        cache hit would force concretization/specialization of the symbol,
        creating guards like `sym_float(size) == <concrete_value>` that didn't
        exist in the original program. This caused a cache miss + recompilation
        for every unique input size.

        The fix emits torch._sym_sqrt(...) which propagates symbolically
        without forcing specialization.

        See https://github.com/pytorch/pytorch/issues/152435
        """
        import math

        def func(x):
            # [FUNCTION DOC] HELPER: Compute a boolean from numel divided by five over the ceiling
            # of the square root of numel, compared to 64; return the input times 5 or times 10
            # together with the boolean.
            y = math.ceil((x.numel() // 5) / (math.ceil(math.sqrt(x.numel())))) > 64
            if y:
                return x * 5, y
            else:
                return x * 10, y

        compiled_fn = torch.compile(func, dynamic=True)

        # Warm up the cache with one size
        compiled_fn(torch.rand(1000000))
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

        # Simulate a new process loading from cache
        self.reset()
        counters.clear()

        compiled_fn2 = torch.compile(func, dynamic=True)
        # First call should hit the cache
        compiled_fn2(torch.rand(2000000))
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)

        # Subsequent calls with different sizes should NOT recompile.
        # Before the fix, each new size would create a spurious guard
        # `sym_float(size) == <value>` causing a recompilation.
        for size in [3000000, 5000000, 6000000, 7000000]:
            compiled_fn2(torch.rand(size))

        # All subsequent calls should reuse the already-loaded compiled graph
        # without any additional cache misses or dynamo recompilations.
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"freezing": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("inlinable", (True, False))
    def test_freezing(self, device, inlinable):
        # [FUNCTION DOC] Freezing enabled. Compiles a module with a parameter of an inlinable
        # small or non-inlinable larger shape, then a second module with different parameter
        # values after a reset. Inlined constants are part of the key (second compile misses);
        # non-inlined ones only use metadata (second compile hits). No bypass in either case.
        # Skipped on CPUs with mkldnn fp16 support or without the requested GPU.
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # For machines with mkldnn_fp16 support, weight_pack in mkldnn_fusion.py causes
        # the creation of a mkldnn format tensor which the current implementation does
        # not support.
        if (
            device == "cpu"
            and torch.backends.mkldnn.is_available()
            and torch.ops.mkldnn._is_mkldnn_fp16_supported()
        ):
            raise unittest.SkipTest("mkldnn tensors unsupported")

        # The shape of the frozen constant determines if it will be inlined.
        shape = (4,) if inlinable else (8, 8)

        class MM(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one random half-precision parameter of the enclosing
            # test's shape whose forward returns the input matrix-multiplied by the parameter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its random parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(shape))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the matrix product of the input with
                # the parameter.
                return x @ self.param

        dtype = torch.float16

        # Populate a cache entry.
        mod1 = MM().to(device=device, dtype=dtype)
        with torch.no_grad():
            x = torch.rand(shape).to(device=device, dtype=dtype)
            out0 = mod1(x)
            out1 = torch.compile(mod1)(x)
            self.assertEqual(out0, out1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
        self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
        self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        counters.clear()
        self.reset()

        # Same nn.Module, but with different parameters. In the case that the param can
        # be inlined, we should consider the actual tensor value and we expect a cache
        # miss (because the values are different here). If the param cannot be inlined,
        # then we consider only the tensor metadata and we expect a cache hit.
        mod2 = MM().to(device=device, dtype=dtype)
        self.assertNotEqual(mod1.param, mod2.param)

        with torch.no_grad():
            x = torch.rand(shape).to(device=device, dtype=dtype)
            out0 = mod2(x)
            out1 = torch.compile(mod2)(x)
            self.assertEqual(out0, out1)

        self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
        self.assertEqual(
            counters["inductor"]["fxgraph_cache_miss"], 1 if inlinable else 0
        )
        self.assertEqual(
            counters["inductor"]["fxgraph_cache_hit"], 0 if inlinable else 1
        )


@instantiate_parametrized_tests
class TestStandaloneCompile(TestCase):
    # [CLASS DOC] Tests for torch._inductor.standalone_compile and CompiledArtifact: saving in
    # binary and unpacked formats, loading, AOT mode, dynamic shapes handling and cross-process
    # loading. Uses the helper capture to obtain a graph module and example inputs from dynamo.
    def setUp(self):
        # [FUNCTION DOC] Clear counters and the artifact manager and install the mock
        # remote caches.
        super().setUp()
        counters.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()

    def tearDown(self):
        # [FUNCTION DOC] Run the base tearDown and remove the mock cache patches.
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        # [FUNCTION DOC] HELPER: Clear the AOTAutograd cache and in-memory PyCodeCache, reset
        # dynamo and clear inductor caches.
        AOTAutogradCache.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    def capture(self, fn, dynamic=None):
        # [FUNCTION DOC] HELPER: Return a function that compiles the given function with a
        # recording backend under fullgraph and the given dynamic setting, runs it on the
        # arguments, and returns a tuple of the captured graph module, the arguments the backend
        # received, and its keyword arguments.
        def inner(*args):
            # [FUNCTION DOC] HELPER: Run the function under compile with the recording backend and
            # return the captured graph module, example arguments and backend keyword arguments.
            gm = None
            actual_args = None
            kwargs = None

            def backend(gm_, args_, **kwargs_):
                # [FUNCTION DOC] HELPER: Backend that stores the graph module, arguments
                # and keyword arguments in the enclosing variables and returns the graph
                # module unchanged.
                nonlocal gm
                nonlocal actual_args
                nonlocal kwargs
                gm = gm_
                actual_args = args_
                kwargs = kwargs_
                return gm

            _ = torch.compile(fn, fullgraph=True, backend=backend, dynamic=dynamic)(
                *args
            )
            return gm, actual_args, kwargs

        return inner

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    @parametrize("format", ("binary", "unpacked"))
    @parametrize("dynamic", (False, True))
    @parametrize("graph_partition", (False, True))
    @parametrize("is_aot", (False, True))
    def test_basic(
        self,
        device: str,
        format: str,
        dynamic: bool,
        graph_partition: bool,
        is_aot: bool,
    ) -> None:
        # [FUNCTION DOC] Parametrized over device, format (binary file or unpacked directory),
        # dynamic shapes, graph partition and AOT mode. Captures a linear plus sine function,
        # standalone compiles it and saves it in a fresh cache, then loads it in another fresh
        # cache and compares results with eager (symbolic ints are replaced by 4). Expects no
        # cache hit during saving and, for non-AOT mode, one hit on load. AOT with unpacked format
        # is skipped.
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        # AOT mode does not support unpacked format
        if is_aot and format == "unpacked":
            raise unittest.SkipTest("AOT mode does not support unpacked format")

        mod = torch.nn.Linear(1, 3, device=device)
        x = torch.randn(4, 1, device=device)
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad return the module output and the sine of
            # the input.
            with torch.no_grad():
                return mod(x), x.sin()

        eager_out = f(x)

        with (
            tempfile.TemporaryDirectory() as temp_dir,
            config.patch(graph_partition=graph_partition),
        ):
            path = (
                temp_dir
                if format == "unpacked"
                else os.path.join(temp_dir, "compiled_artifact.bin")
            )
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                if kwargs:
                    raise AssertionError

                compiled_artifact = torch._inductor.standalone_compile(
                    gm, args, aot=is_aot
                )
                compiled_artifact.save(path=path, format=format)

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                loaded = torch._inductor.CompiledArtifact.load(path=path, format=format)
                if dynamic:
                    concrete_args = [
                        4 if isinstance(a, torch.SymInt) else a for a in args
                    ]
                else:
                    concrete_args = args
                compiled_out = loaded(*concrete_args)
                self.assertEqual(eager_out, compiled_out)

            if not is_aot:
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_is_saveable(self) -> None:
        # [FUNCTION DOC] Checks that an artifact with an uncacheable allow-in-graph function
        # reports not saveable and that saving it in unpacked format raises a runtime error
        # mentioning that it is not serializable.
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)

        @torch._dynamo.allow_in_graph
        def uncacheable(x):
            # [FUNCTION DOC] HELPER: Return the sine of the input (registered as allowed
            # in graph).
            return x.sin()

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad apply the module then the uncacheable function.
            with torch.no_grad():
                result = mod(x)
                return uncacheable(result)

        with fresh_cache():
            gm, args, kwargs = self.capture(f)(x)
            if kwargs:
                raise AssertionError

            compiled_artifact = torch._inductor.standalone_compile(gm, args)

            self.assertFalse(compiled_artifact.is_saveable())

            with tempfile.TemporaryDirectory() as temp_dir:
                path = os.path.join(temp_dir, "new_dir")
                with self.assertRaisesRegex(RuntimeError, "not serializable"):
                    compiled_artifact.save(path=path, format="unpacked")

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("dynamic", (False, True))
    @parametrize("is_aot", (False, True))
    def test_call_in_backend(self, dynamic: bool, is_aot: bool) -> None:
        # [FUNCTION DOC] Uses standalone_compile directly as a dynamo backend (optionally AOT and
        # dynamic) and checks the compiled linear function matches eager.
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)
        if dynamic:
            torch._dynamo.mark_dynamic(x, 0)

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad return the module output.
            with torch.no_grad():
                return mod(x)

        eager_out = f(x)

        def backend(gm, args, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that returns the standalone compiled artifact for
            # the graph.
            return torch._inductor.standalone_compile(gm, args, aot=is_aot)

        with fresh_cache():
            compiled_out = torch.compile(f, fullgraph=True, backend=backend)(x)
            self.assertEqual(eager_out, compiled_out)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("donate", (False, True))
    def test_donate_graph_module(self, donate: bool) -> None:
        # [FUNCTION DOC] Checks the donate_graph_module option: compiling mutates the given graph
        # module (it gains mutation region ids on node metadata) only when the graph module is
        # donated; without donation the fingerprint is unchanged. Results match eager either way.
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad return the module output.
            with torch.no_grad():
                return mod(x)

        eager_out = f(x)

        with fresh_cache():
            gm, args, kwargs = self.capture(f)(x)
            if kwargs:
                raise AssertionError

            # compile_fx mutates the graph module (e.g. adds
            # mutation_region_id to node metadata). Use this as a
            # fingerprint to detect mutation.
            def has_mutation_region_ids(gm):
                # [FUNCTION DOC] HELPER: Return true if any graph node has a mutation_region_id
                # entry in its metadata.
                return any("mutation_region_id" in n.meta for n in gm.graph.nodes)

            before = has_mutation_region_ids(gm)

            compiled_artifact = torch._inductor.standalone_compile(
                gm, args, donate_graph_module=donate
            )
            compiled_out = compiled_artifact(*args)
            self.assertEqual(eager_out, compiled_out[0])

            after = has_mutation_region_ids(gm)
            if donate:
                self.assertNotEqual(before, after)
            else:
                self.assertEqual(before, after)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_save_in_new_path(self) -> None:
        # [FUNCTION DOC] Saves an unpacked artifact to a directory that does not yet exist and
        # checks a subsequent load in another fresh cache works and matches eager, with no cache
        # hit on saving.
        mod = torch.nn.Linear(1, 3)
        x = torch.randn(4, 1)

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad return the module output.
            with torch.no_grad():
                return mod(x)

        eager_out = f(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "new_dir")
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                if kwargs:
                    raise AssertionError

                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_artifact.save(path=path, format="unpacked")

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                loaded = torch._inductor.CompiledArtifact.load(
                    path=path, format="unpacked"
                )
                compiled_out = loaded(*args)[0]
                self.assertEqual(eager_out, compiled_out)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_save_training_artifact_compiles_backward(self) -> None:
        # [FUNCTION DOC] Compiles a linear module that requires grad, checks the artifact is
        # saveable, saves and reloads it unpacked in a fresh cache and checks forward matches
        # eager and that backward through the loaded artifact produces the expected weight and
        # bias gradients.
        mod = torch.nn.Linear(4, 3)
        x = torch.randn(2, 4)
        eager_out = mod(x).detach()

        with tempfile.TemporaryDirectory() as temp_dir:
            with fresh_cache():
                gm, args, kwargs = self.capture(mod)(x)
                if kwargs:
                    raise AssertionError

                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                self.assertTrue(compiled_artifact.is_saveable())
                compiled_artifact.save(path=temp_dir, format="unpacked")

            with fresh_cache():
                mod.zero_grad(set_to_none=True)
                loaded = torch._inductor.CompiledArtifact.load(
                    path=temp_dir, format="unpacked"
                )
                compiled_out = loaded(*args)[0]
                self.assertEqual(eager_out, compiled_out)

                compiled_out.sum().backward()
                self.assertEqual(
                    mod.weight.grad, torch.ones_like(eager_out).t().matmul(x)
                )
                self.assertEqual(mod.bias.grad, torch.full_like(mod.bias, x.shape[0]))

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("device", (GPU_TYPE, "cpu"))
    def test_modify_unpacked_file(self, device: str) -> None:
        # [FUNCTION DOC] Saves an unpacked artifact, then edits the generated source file in the
        # cache directory (changing the constant 2.0 to 8.0 in a triton kernel on GPU, or in the
        # C++ source on CPU) and loads it again; the loaded artifact must reflect the edit (result
        # four times eager first output) and register one fx graph cache hit. Skips the
        # aotautograd and fxgraph subdirectories when editing.
        if device == GPU_TYPE and not HAS_GPU:
            raise unittest.SkipTest(f"requires {GPU_TYPE}")

        x = torch.ones(4, device=device)

        def f(x):
            # [FUNCTION DOC] HELPER: Under no_grad return the input doubled and the sine of
            # the input.
            with torch.no_grad():
                return 2 * x, x.sin()

        eager_out = f(x)

        with tempfile.TemporaryDirectory() as temp_dir:
            with fresh_cache():
                gm, args, kwargs = self.capture(f)(x)
                if kwargs:
                    raise AssertionError

                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_out = compiled_artifact(*args)
                self.assertEqual(eager_out, compiled_out)

                compiled_artifact.save(path=temp_dir, format="unpacked")

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            with fresh_cache():
                # Now modify the output file and expect to see the changes
                for subdir in os.listdir(temp_dir):
                    if subdir in ["aotautograd", "fxgraph"]:
                        continue
                    subdir_path = os.path.join(temp_dir, subdir)
                    for file in os.listdir(subdir_path):
                        file_path = os.path.join(subdir_path, file)
                        if not os.path.isfile(file_path):
                            raise AssertionError
                        with open(file_path) as f:
                            file_contents = f.read()
                        if device == GPU_TYPE:
                            file_contents = file_contents.replace(
                                "2.0, tl.float32", "8.0, tl.float32"
                            )
                        else:
                            if device != "cpu":
                                raise AssertionError(f"Expected 'cpu', got {device!r}")
                            file_contents = file_contents.replace(
                                "auto tmp1 = static_cast<float>(2.0);",
                                "auto tmp1 = static_cast<float>(8.0);",
                            )
                        with open(file_path, "w") as f:
                            f.write(file_contents)

                loaded = torch._inductor.CompiledArtifact.load(
                    path=temp_dir, format="unpacked"
                )
                compiled_out = loaded(*args)
                self.assertEqual(4 * eager_out[0], compiled_out[0])

            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

    @unittest.skipIf(IS_FBCODE, "torch import error")
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    def test_different_process(self):
        # [FUNCTION DOC] Saves a binary standalone artifact, then loads and runs it in a separate
        # Python subprocess started in the test directory, which compares against eager and raises
        # a runtime error on mismatch. A failing subprocess fails the test with its output.
        # Skipped in fbcode.
        x = torch.ones(4, 1)

        def f(x):
            # [FUNCTION DOC] HELPER: Return the sine of the input doubled.
            return x.sin() * 2

        gm, args, kwargs = self.capture(f)(x)
        if kwargs:
            raise AssertionError

        with tempfile.TemporaryDirectory() as temp_dir:
            path = normalize_path_separator(
                os.path.join(temp_dir, "compiled_artifact.bin")
            )

            with fresh_cache():
                compiled_artifact = torch._inductor.standalone_compile(gm, args)
                compiled_artifact.save(path=path)

            script = f"""
import torch
from torch._inductor.utils import fresh_cache

arg = torch.ones(4, 1)
with fresh_cache():
    loaded = torch._inductor.CompiledArtifact.load(path="{path}")
    compiled_result = loaded(arg)[0]

eager_result = arg.sin() * 2

if not torch.allclose(eager_result, compiled_result, atol=0.1, rtol=0.01):
    raise RuntimeError("tensors do not match")
"""
            try:
                subprocess.check_output(
                    [sys.executable, "-c", script],
                    stderr=subprocess.STDOUT,
                    cwd=os.path.dirname(os.path.realpath(__file__)),
                )
            except subprocess.CalledProcessError as e:
                self.fail(
                    msg=(
                        "Subprocess exception while attempting to run test: "
                        + e.output.decode("utf-8")
                    )
                )

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (False, True))
    def test_dynamic_shapes_from_graph(self, is_aot: bool):
        # [FUNCTION DOC] Checks that dynamic_shapes from_graph keeps the symbolic size: the
        # artifact compiled from a captured graph takes the size as an extra argument and works
        # for a different length input.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        x = torch.ones(3)
        torch._dynamo.mark_dynamic(x, 0)
        with fresh_cache():
            # captured graph is lambda x, s0: x * s0
            gm, args, kwargs = self.capture(f)(x)
            if kwargs:
                raise AssertionError

        compiled_artifact = torch._inductor.standalone_compile(
            gm, args, dynamic_shapes="from_graph", aot=is_aot
        )
        x = torch.ones(4)
        (result,) = compiled_artifact(x, 4)
        self.assertEqual(result, x * 4)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @functorch_config.patch({"autograd_cache_normalize_inputs": True})
    @parametrize("is_aot", (False, True))
    def test_split_module(self, is_aot):
        # [FUNCTION DOC] Splits a graph module of three structurally identical groups of
        # operations into three submodules by data dependency and standalone compiles each. In
        # non-AOT mode checks one fx graph cache miss and two hits with no bypass; for the AOT
        # autograd cache checks two misses, one hit and two saves (a TODO notes type annotation
        # differences prevent a second hit). The combined result must equal eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that adds, for each of three pairs of scalar inputs, the
            # square of the first and half of the second to the running value.
            def forward(self, x, a0, a1, b0, b1, c0, c1):
                # [FUNCTION DOC] HELPER: Accumulate the three pairs into the input and return it.
                x = x + (a0**2) + (a1 / 2)
                x = x + (b0**2) + (b1 / 2)
                x = x + (c0**2) + (c1 / 2)
                return x

        def t():
            # [FUNCTION DOC] HELPER: Return a new random scalar tensor.
            return torch.randn([])

        x = t()
        a0 = t()
        a1 = t()
        b0 = t()
        b1 = t()
        c0 = t()
        c1 = t()

        example_inputs = (x, a0, a1, b0, b1, c0, c1)
        gm, inps, _ = self.capture(Mod())(*example_inputs)

        # Partition by data dependency: nodes depending on a0/a1 -> 0,
        # b0/b1 -> 1, c0/c1 -> 2. This produces 3 structurally identical
        # subgraphs regardless of canonicalized node order.
        placeholder_group = {}
        for node in gm.graph.nodes:
            if node.op == "placeholder":
                name = node.name
                if "a0" in name or "a1" in name:
                    placeholder_group[node] = 0
                elif "b0" in name or "b1" in name:
                    placeholder_group[node] = 1
                elif "c0" in name or "c1" in name:
                    placeholder_group[node] = 2
                else:
                    placeholder_group[node] = -1

        node_group: dict = {}

        def get_group(node):
            # [FUNCTION DOC] HELPER: Return the group index of a graph node: placeholders use
            # the precomputed group, output is minus one, and others take the largest group of
            # their input nodes (minus one when none). Results are memoized in the node group
            # mapping; recursive.
            if node in node_group:
                return node_group[node]
            if node.op == "placeholder":
                return placeholder_group[node]
            if node.op == "output":
                return -1
            groups = {get_group(a) for a in node.all_input_nodes}
            groups.discard(-1)
            g = max(groups) if groups else -1
            node_group[node] = g
            return g

        def split_fn(node):
            # [FUNCTION DOC] HELPER: Partition callback for graph splitting: the node's group
            # clamped to be at least zero.
            return max(get_group(node), 0)

        split = torch.fx.passes.split_module.split_module(
            gm,
            gm,
            split_fn,
            keep_original_node_name=False,
            keep_original_input_name=False,
        )

        # Derive each submodule's inputs from the split graph's
        # call_module args rather than hardcoding the order, which
        # depends on canonicalization.
        node_to_val = {}
        ph_idx = 0
        for n in split.graph.nodes:
            if n.op == "placeholder":
                node_to_val[n] = inps[ph_idx]
                ph_idx += 1

        for call_node in split.graph.nodes:
            if call_node.op != "call_module":
                continue
            submod = getattr(split, call_node.target)
            submod_inputs = tuple(node_to_val[a] for a in call_node.args)
            compiled = torch._inductor.standalone_compile(
                submod,
                submod_inputs,
                aot=is_aot,
            )
            node_to_val[call_node] = compiled(*submod_inputs)

        output_node = next(n for n in split.graph.nodes if n.op == "output")
        y = node_to_val[output_node.args[0][0]]
        if not is_aot:
            # fx graph cache doesn't run in AOT mode
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
        # TODO: split_module causes ca1 and ca2 to have different type annotations
        # for the parameter x, so we can only AOTAutogradCache cache hit once instead of twice
        self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
        self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 2)

        expected = Mod()(*example_inputs)
        self.assertEqual(y, expected)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (False, True))
    @parametrize("config_patches", [True, False])
    def test_dynamic_shapes_from_example_inputs(self, config_patches, is_aot):
        # [FUNCTION DOC] Checks dynamic_shapes from_example_inputs bakes in the example values:
        # the artifact compiled with a concrete size 5 and a length 4 tensor returns the input
        # times 5 for any passed int, and calling with a length 5 tensor raises an assertion error
        # about the expected size. Parametrized on AOT mode and whether config patches are passed
        # in options.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        x = torch.ones(3)
        torch._dynamo.mark_dynamic(x, 0)
        with fresh_cache():
            # captured graph is lambda s0, x: x * s0
            gm, args, kwargs = self.capture(f)(x)
            if kwargs:
                raise AssertionError

        if config_patches:
            config_patches = {"fx_graph_cache": True}
        else:
            config_patches = None

        # specialized on example inputs
        compiled_artifact = torch._inductor.standalone_compile(
            gm,
            (5, torch.ones(4)),
            dynamic_shapes="from_example_inputs",
            options={"config_patches": config_patches},
            aot=is_aot,
        )
        x = torch.ones(4)
        (result,) = compiled_artifact(3, x)
        # int 5 was baked in!
        self.assertEqual(result, x * 5)

        # size 4 was baked in
        with self.assertRaisesRegex(AssertionError, "expected size 5==4"):
            x = torch.randn(5)
            (result,) = compiled_artifact(4, x)

    def test_dynamic_shapes_from_example_inputs_fake_mode(self):
        # [FUNCTION DOC] Checks the fake mode handling of standalone_compile with
        # from_example_inputs. With a fake mode passed in, compile_fx (mocked) sees that fake mode
        # and ignore_shape_env true. Without one, it sees a different, newly created
        # FakeTensorMode with a shape env, and ignore_shape_env is still true.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        gm = torch.fx.symbolic_trace(f)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        fake_x = fake_mode.from_tensor(torch.ones(4), static_shapes=True)

        seen_fake_modes = []
        seen_ignore_shape_env = []

        def fake_compile_fx(gm, example_inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Mock for compile_fx that records the tracing context's fake
            # mode and the ignore_shape_env keyword, checks the first example input is the
            # prepared fake tensor, and returns a callable that returns its arguments.
            seen_fake_modes.append(torch._guards.TracingContext.get().fake_mode)
            seen_ignore_shape_env.append(kwargs["ignore_shape_env"])
            self.assertIs(example_inputs[0], fake_x)
            return lambda *args: args

        with mock.patch(
            "torch._inductor.compile_fx.compile_fx", side_effect=fake_compile_fx
        ):
            torch._inductor.standalone_compile(
                gm,
                (fake_x,),
                dynamic_shapes="from_example_inputs",
                fake_mode=fake_mode,
            )

        self.assertEqual(seen_ignore_shape_env, [True])
        self.assertIs(seen_fake_modes[0], fake_mode)

        seen_fake_modes.clear()
        seen_ignore_shape_env.clear()
        with mock.patch(
            "torch._inductor.compile_fx.compile_fx", side_effect=fake_compile_fx
        ):
            torch._inductor.standalone_compile(
                gm,
                (fake_x,),
                dynamic_shapes="from_example_inputs",
            )

        self.assertEqual(seen_ignore_shape_env, [True])
        self.assertIsNot(seen_fake_modes[0], fake_mode)
        self.assertIsInstance(seen_fake_modes[0], FakeTensorMode)
        self.assertIsNotNone(seen_fake_modes[0].shape_env)

    def test_dynamic_shapes_from_graph_tensor_subclass_fake_mode(self):
        # [FUNCTION DOC] Captures a dynamic graph whose output is a TwoTensor subclass created by
        # an allowed-in-graph function, takes the fake mode from the output's example value, then
        # with compile_fx mocked checks standalone_compile with from_graph dynamic shapes uses
        # that same fake mode for tracing.
        @torch._dynamo.allow_in_graph
        def to_subclass(x):
            # [FUNCTION DOC] HELPER: Build a TwoTensor from two clones of the input.
            return TwoTensor(x.clone(), x.clone())

        def f(x):
            # [FUNCTION DOC] HELPER: Convert the input to a TwoTensor and flatten it with a view.
            return to_subclass(x).view(-1)

        x = torch.ones(2)
        torch._dynamo.mark_dynamic(x, 0)
        with fresh_cache():
            gm, args, kwargs = self.capture(f, dynamic=True)(x)
            if kwargs:
                raise AssertionError

        output_node = next(iter(reversed(gm.graph.nodes)))
        value_node = output_node.args[0][0]
        self.assertIsInstance(value_node, torch.fx.Node)
        output_example_value = value_node.meta["example_value"]
        self.assertIsInstance(output_example_value, TwoTensor)
        fake_mode = output_example_value.a.fake_mode

        seen_fake_modes = []

        def fake_compile_fx(gm, example_inputs, **kwargs):
            # [FUNCTION DOC] HELPER: Mock for compile_fx that records the tracing context's fake
            # mode and returns a callable returning its arguments.
            seen_fake_modes.append(torch._guards.TracingContext.get().fake_mode)
            return lambda *args: args

        with mock.patch(
            "torch._inductor.compile_fx.compile_fx", side_effect=fake_compile_fx
        ):
            torch._inductor.standalone_compile(gm, args, dynamic_shapes="from_graph")

        self.assertIs(seen_fake_modes[0], fake_mode)

    def test_standalone_compile_fake_mode_requires_shape_env(self):
        # [FUNCTION DOC] Checks that passing a fake mode without a shape env to
        # standalone_compile with from_example_inputs raises a value error whose message mentions
        # fake_mode and ShapeEnv.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        gm = torch.fx.symbolic_trace(f)

        with self.assertRaisesRegex(
            ValueError,
            "fake_mode.*ShapeEnv",
        ):
            torch._inductor.standalone_compile(
                gm,
                (torch.ones(4),),
                dynamic_shapes="from_example_inputs",
                fake_mode=FakeTensorMode(),
            )

    def test_standalone_compile_fake_mode_requires_from_example_inputs(self):
        # [FUNCTION DOC] Checks that passing a fake mode to standalone_compile with dynamic_shapes
        # from_graph raises a value error saying a fake_mode needs from_example_inputs.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        gm = torch.fx.symbolic_trace(f)
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())

        with self.assertRaisesRegex(
            ValueError,
            'fake_mode.*dynamic_shapes="from_example_inputs"',
        ):
            torch._inductor.standalone_compile(
                gm,
                (torch.ones(4),),
                dynamic_shapes="from_graph",
                fake_mode=fake_mode,
            )

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    @parametrize("dynamic_shapes", ["from_graph", "from_example_inputs"])
    def test_static_shapes(self, dynamic_shapes, is_aot):
        # [FUNCTION DOC] Compiles a graph captured with static shapes (multiplication by the
        # constant size 3) using from_graph or from_example_inputs, in AOT or normal mode. The
        # artifact returns the input times 3 and raises an assertion error about the expected size
        # when given length 4.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        static_x = torch.randn(3)
        with fresh_cache():
            # static_gm is lambda x: x * 3
            static_gm, args, kwargs = self.capture(f, dynamic=False)(static_x)
            if kwargs:
                raise AssertionError
        compiled_artifact = torch._inductor.standalone_compile(
            static_gm, [static_x], dynamic_shapes=dynamic_shapes, aot=is_aot
        )
        x = torch.randn(3)
        (result,) = compiled_artifact(x)
        self.assertEqual(result, x * 3)
        with self.assertRaisesRegex(AssertionError, "expected size 4==3"):
            x = torch.randn(4)
            (result,) = compiled_artifact(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    @parametrize("dynamic_shapes", ["from_tracing_context", "from_graph"])
    def test_backend(self, dynamic_shapes, is_aot):
        # [FUNCTION DOC] Uses standalone_compile inside a dynamo backend with from_tracing_context
        # or from_graph dynamic shapes (AOT or not). The compiled artifact is called with a
        # different length input plus the size and must return the input times that size.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        x = torch.randn(3)
        torch._dynamo.mark_dynamic(x, 0)

        def backend(gm, args, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that standalone compiles the graph, checks the
            # artifact on a length 4 tensor with size argument 4, and returns the artifact.
            compiled_artifact = torch._inductor.standalone_compile(
                gm, args, dynamic_shapes=dynamic_shapes, aot=is_aot
            )
            y = torch.randn(4)
            (result,) = compiled_artifact(y, 4)
            self.assertEqual(result, y * 4)
            return compiled_artifact

        torch._dynamo.reset()
        _ = torch.compile(f, backend=backend)(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize("is_aot", (True, False))
    def test_backend_dynamic_shapes_from_example_inputs(self, is_aot):
        # [FUNCTION DOC] Uses standalone_compile with from_example_inputs inside a backend, where
        # the example size is 5 and the example tensor has length 4. Checks the size 5 is baked
        # in (result is input times 5) and a length 5 tensor raises an assertion error about
        # expected size.
        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        x = torch.ones(4)
        torch._dynamo.mark_dynamic(x, 0)

        def backend(gm, args, **kwargs):
            # [FUNCTION DOC] HELPER: Backend that builds example inputs (5 for size arguments,
            # ones of length 4 for tensors), standalone compiles, validates baked-in behavior, and
            # returns the artifact.
            example_inputs = [
                5 if isinstance(a, (int, torch.SymInt)) else torch.ones(4) for a in args
            ]
            compiled_artifact = torch._inductor.standalone_compile(
                gm, example_inputs, dynamic_shapes="from_example_inputs", aot=is_aot
            )
            y = torch.ones(4)
            call_args = [4 if isinstance(a, (int, torch.SymInt)) else y for a in args]
            (result,) = compiled_artifact(*call_args)
            # 5 was baked in
            self.assertEqual(result, y * 5)

            # shape of y was baked in
            with self.assertRaisesRegex(AssertionError, "expected size 5==4"):
                y = torch.ones(5)
                call_args = [
                    4 if isinstance(a, (int, torch.SymInt)) else y for a in args
                ]
                (result,) = compiled_artifact(*call_args)

            return compiled_artifact

        torch._dynamo.reset()
        _ = torch.compile(f, backend=backend)(x)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @functorch_config.patch({"enable_autograd_cache": True})
    @parametrize(
        "dynamic_shapes", ["from_tracing_context", "from_graph", "from_example_inputs"]
    )
    def test_backend_static_shapes(self, dynamic_shapes):
        # [FUNCTION DOC] Smoke test: with a statically shaped input, every dynamic_shapes option
        # (from_tracing_context, from_graph, from_example_inputs) used inside a backend must give
        # a result equal to the input times 3.
        # on static_x, all of these options should produce a static graph,
        # but it's a bit hard to tell, so these are just smoke tests.
        static_x = torch.randn(3)

        def f(x):
            # [FUNCTION DOC] HELPER: Return the input multiplied by its first dimension size.
            return x.shape[0] * x

        def backend(gm, args, **kwargs):
            # [FUNCTION DOC] HELPER: Backend returning the standalone compiled artifact with the
            # test's dynamic_shapes option.
            return torch._inductor.standalone_compile(
                gm, args, dynamic_shapes=dynamic_shapes
            )

        result = torch.compile(f, backend=backend)(static_x)
        self.assertEqual(result, static_x * 3)

    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    def test_custom_pass_handling(self):
        # [FUNCTION DOC] Checks custom graph pass handling. A plain lambda as post grad pre pass
        # bypasses the cache (one bypass, no miss or hit). Registered CustomGraphPass instances (a
        # single one, then a list of two) installed in all four custom pass config slots cache
        # normally: a miss first, then a hit after reset.
        """
        Test that properly-registered custom hooks allow caching.
        """

        class TestCustomGraphPass(CustomGraphPass):
            # [CLASS DOC] HELPER: Trivial CustomGraphPass that leaves the graph untouched and has
            # the fixed uuid string "uuid".
            def __call__(self, graph: torch.fx.graph.Graph) -> None:
                # [FUNCTION DOC] HELPER: Do nothing and return None.
                return None

            def uuid(self) -> bytes | str | None:
                # [FUNCTION DOC] HELPER: Return the fixed identifier string "uuid".
                return "uuid"

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the matrix product of the two arguments.
            return torch.mm(a, b)

        a = torch.rand(8, 32, device="cpu")
        b = torch.rand(32, 8, device="cpu")
        compiled_fn = torch.compile(fn)

        # The cache should be bypassed if a custom hook doesn't use CustomGraphPass.
        with config.patch({"post_grad_custom_pre_pass": lambda x: x}):
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

        # With proper usage, we expect normal caching.
        custom_pass = TestCustomGraphPass()
        with config.patch(
            {
                "post_grad_custom_pre_pass": custom_pass,
                "post_grad_custom_post_pass": custom_pass,
                "joint_custom_pre_pass": custom_pass,
                "joint_custom_post_pass": custom_pass,
            }
        ):
            self.reset()
            counters.clear()

            # Cache miss
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.reset()
            counters.clear()

            # Cache hit
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        # Lists of properly-registered custom hooks should also allow caching.
        custom_passes = [TestCustomGraphPass(), TestCustomGraphPass()]
        with config.patch(
            {
                "post_grad_custom_pre_pass": custom_passes,
                "post_grad_custom_post_pass": custom_passes,
                "joint_custom_pre_pass": custom_passes,
                "joint_custom_post_pass": custom_passes,
            }
        ):
            self.reset()
            counters.clear()

            # Cache miss
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            self.reset()
            counters.clear()

            # Cache hit
            self.assertEqual(fn(a, b), compiled_fn(a, b))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)


class _TestCustomPartitionerFn(CustomPartitionerFn):
    # [CLASS DOC] HELPER: CustomPartitionerFn test double with a settable uuid (initially None)
    # whose call returns the input graph module twice as forward and backward.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initialize the uuid to None.
        self._uuid = None

    def __call__(
        self, gm, joint_inputs, **kwargs
    ) -> tuple[torch.fx.GraphModule, torch.fx.GraphModule]:
        # [FUNCTION DOC] HELPER: Dummy partitioner: return the given graph module as both forward
        # and backward graphs.
        return gm, gm  # Dummy implementation

    def uuid(self) -> bytes | str | None:
        # [FUNCTION DOC] HELPER: Return the stored uuid value.
        return self._uuid


class TestFxGraphCacheHashing(TestCase):
    # [CLASS DOC] Tests how FxGraphHashDetails and FxGraphCachePickler hash graphs, tensors,
    # kwargs, configs and custom passes into cache keys, plus cacheability validation
    # (bypass rules). Many tests change the torch CPU thread count and always restore it in
    # a finally block.
    def _fx_graph_cache_key(self, gm, example_inputs):
        # [FUNCTION DOC] HELPER: Build the hash details for the graph module and example inputs
        # with empty kwargs and no inputs to ignore, and return the key from an
        # FxGraphCachePickler over the graph module.
        details = FxGraphHashDetails(gm, example_inputs, cast(Any, {}), [])
        return FxGraphCachePickler(gm).get_key(details)

    def _no_input_factory_graph(self, device=None):
        # [FUNCTION DOC] HELPER: Build a graph module with no inputs whose single node calls
        # torch.empty for shape 4 in float32, adding a device keyword if a device is given.
        # Returns the module.
        graph = torch.fx.Graph()
        kwargs: dict[str, Any] = {"dtype": torch.float32}
        if device is not None:
            kwargs["device"] = torch.device(device)
        result = graph.call_function(torch.empty, ((4,),), kwargs)
        graph.output(result)
        return torch.fx.GraphModule({}, graph)

    def test_cpu_thread_count_affects_cache_key(self):
        # [FUNCTION DOC] Compares keys of a simple add graph at 1 versus 2 CPU threads. Keys
        # differ when cpp threads is -1 and dynamic threads is off; they are equal when dynamic
        # threads is on, and when cpp threads is a fixed 4.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        gm = torch.fx.symbolic_trace(fn)
        x = torch.ones(4)
        orig_num_threads = torch.get_num_threads()

        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [x])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [x])
                self.assertNotEqual(key_1, key_2)

            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": True}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [x])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [x])
                self.assertEqual(key_1, key_2)

            with config.patch({"cpp.threads": 4, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [x])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [x])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_affects_cache_key_for_no_input_cpu_factory(self):
        # [FUNCTION DOC] Keys of a graph with no inputs that creates a CPU tensor by a factory
        # differ between 1 and 2 threads (cpp threads -1, dynamic threads off).
        gm = self._no_input_factory_graph()
        orig_num_threads = torch.get_num_threads()

        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertNotEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_affects_cache_key_for_example_value_metadata(self):
        # [FUNCTION DOC] Keys for a graph whose node target is named empty and has only a fake CPU
        # tensor in the example_value metadata (no val metadata) differ between 1 and 2 threads.
        graph = torch.fx.Graph()
        result = graph.call_function(_custom_empty, (), {})
        with torch._subclasses.FakeTensorMode():
            result.meta["example_value"] = torch.empty(4)
        self.assertNotIn("val", result.meta)
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertNotEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_affects_cache_key_for_randperm_factory(self):
        # [FUNCTION DOC] Keys for a graph calling the randperm operator differ between 1 and
        # 2 threads.
        graph = torch.fx.Graph()
        result = graph.call_function(
            torch.ops.aten.randperm.default,
            (4,),
            {"dtype": torch.float32},
        )
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertNotEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_ignored_for_explicit_non_cpu_factory(self):
        # [FUNCTION DOC] Keys for a factory graph with an explicit CUDA device do not change with
        # the thread count.
        gm = self._no_input_factory_graph(device="cuda")
        orig_num_threads = torch.get_num_threads()

        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_ignores_non_device_string_on_factory(self):
        # [FUNCTION DOC] A factory with a CUDA device keyword plus an unrelated string
        # keyword that says cpu does not make the key thread dependent: keys are equal across
        # 1 and 2 threads.
        graph = torch.fx.Graph()
        result = graph.call_function(
            torch.empty,
            ((4,),),
            {
                "dtype": torch.float32,
                "device": torch.device("cuda"),
                "debug_device": "cpu",
            },
        )
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_ignored_for_cuda_like_factory(self):
        # [FUNCTION DOC] Keys for an empty_like graph whose example input is a fake CUDA tensor
        # are equal across 1 and 2 threads.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        result = graph.call_function(torch.empty_like, (x,), {})
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode():
            example_input = torch.empty(4, device="cuda")

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [example_input])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [example_input])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_ignored_for_cuda_metadata_device_constructor(self):
        # [FUNCTION DOC] Keys for an as_tensor graph where the placeholder and result metadata
        # hold a fake CUDA tensor are equal across 1 and 2 threads.
        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        result = graph.call_function(torch.as_tensor, (x,), {})
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        with torch._subclasses.FakeTensorMode():
            example_input = torch.empty(4, device="cuda")
        x.meta["val"] = example_input
        result.meta["example_value"] = example_input

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [example_input])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [example_input])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    def test_cpu_thread_count_ignored_for_custom_empty_target(self):
        # [FUNCTION DOC] Keys for a graph calling the module-level fake empty helper (which only
        # has the name empty) with a cpu device keyword are equal across 1 and 2 threads, so
        # detection must not trust function names alone.
        graph = torch.fx.Graph()
        result = graph.call_function(_custom_empty, ("cpu",), {"device": "cpu"})
        graph.output(result)
        gm = torch.fx.GraphModule({}, graph)

        orig_num_threads = torch.get_num_threads()
        try:
            with config.patch({"cpp.threads": -1, "cpp.dynamic_threads": False}):
                torch.set_num_threads(1)
                key_1 = self._fx_graph_cache_key(gm, [])
                torch.set_num_threads(2)
                key_2 = self._fx_graph_cache_key(gm, [])
                self.assertEqual(key_1, key_2)
        finally:
            torch.set_num_threads(orig_num_threads)

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "requires MKLDNN")
    def test_cacheability_validator_checks_mkldnn_constant(self):
        # [FUNCTION DOC] Requires MKLDNN. A graph module holding an mkldnn tensor constant makes
        # CacheabilityValidator validation raise BypassFxGraphCache with a message about
        # unpickleable mkldnn tensors.
        graph = torch.fx.Graph()
        output = graph.get_attr("mkldnn_weight")
        graph.output(output)
        gm = torch.fx.GraphModule(
            {"mkldnn_weight": torch.randn(2, 2).to_mkldnn()}, graph
        )

        with self.assertRaisesRegex(BypassFxGraphCache, "mkldnn tensors unpickleable"):
            CacheabilityValidator(gm, require_shape_env=False).validate()

    def _nested_region_gm(self, patches):
        # [FUNCTION DOC] HELPER: Build an empty graph module whose meta nested_region_config
        # holds the invoke subgraph compile options created with the given forward inductor
        # config patches.
        from torch._higher_order_ops.invoke_subgraph import (
            get_invoke_subgraph_compile_options,
        )

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        gm.meta["nested_region_config"] = get_invoke_subgraph_compile_options(
            fw_inductor_config_patches=patches
        )
        return gm

    def test_nested_region_config_patches_affect_cache_key(self):
        # [FUNCTION DOC] Checks the nested region forward config patches are part of the key: the
        # hash details expose them as a nested tuple, graphs with equal patches share a key, and
        # graphs with different patches have different keys.
        # A region's inductor_config_patches must be part of the cache key,
        # otherwise two regions differing only in their patches would collide
        # and reuse a stale compiled artifact.
        same1 = self._nested_region_gm({"max_autotune": True})
        same2 = self._nested_region_gm({"max_autotune": True})
        different = self._nested_region_gm({"max_autotune": False})

        self.assertEqual(
            FxGraphHashDetails(
                same1, [], cast(Any, {}), []
            ).nested_inductor_config_patches,
            (("", (("max_autotune", True),)),),
        )
        self.assertEqual(
            self._fx_graph_cache_key(same1, []),
            self._fx_graph_cache_key(same2, []),
        )
        self.assertNotEqual(
            self._fx_graph_cache_key(same1, []),
            self._fx_graph_cache_key(different, []),
        )

    def test_nested_region_uncacheable_config_bypasses_cache(self):
        # [FUNCTION DOC] Checks validator bypass rules for nested region patches: a callable
        # custom pass value, a non-callable under a custom pass key, and a callable inside the ddp
        # passes list each raise BypassFxGraphCache with matching messages, while a list of
        # strings is accepted.
        # A callable patch value can't be hashed into the cache key.
        def custom_pass(graph):
            # [FUNCTION DOC] HELPER: Identity pass returning the graph it receives.
            return graph

        with self.assertRaisesRegex(BypassFxGraphCache, "callable value"):
            CacheabilityValidator(
                self._nested_region_gm({"post_grad_custom_pre_pass": custom_pass}),
                require_shape_env=False,
            ).validate()

        # A non-callable value under a custom-pass key is uncacheable too.
        with self.assertRaisesRegex(BypassFxGraphCache, "custom pass"):
            CacheabilityValidator(
                self._nested_region_gm({"post_grad_custom_pre_pass": "sentinel"}),
                require_shape_env=False,
            ).validate()

        # A callable hidden inside a list value (e.g.
        # _fuse_ddp_communication_passes) is uncacheable too.
        with self.assertRaisesRegex(BypassFxGraphCache, "callable value"):
            CacheabilityValidator(
                self._nested_region_gm(
                    {"_fuse_ddp_communication_passes": [custom_pass]}
                ),
                require_shape_env=False,
            ).validate()

        # A list of non-callables stays cacheable.
        CacheabilityValidator(
            self._nested_region_gm(
                {"_fuse_ddp_communication_passes": ["fuse_ddp_with_concat_op"]}
            ),
            require_shape_env=False,
        ).validate()

    def _nested_region_bw_gm(self, bw_patches):
        # [FUNCTION DOC] HELPER: Build an empty graph module with one call node to the invoke
        # subgraph higher order op whose custom metadata holds the backward nested region config
        # derived from compile options with the given backward patches. Returns the recompiled
        # graph module.
        from torch._higher_order_ops.invoke_subgraph import (
            get_backward_nested_region_config,
            get_invoke_subgraph_compile_options,
        )

        fw_config = get_invoke_subgraph_compile_options(
            bw_inductor_config_patches=bw_patches
        )
        bw_config = get_backward_nested_region_config(fw_config)
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        # Mirror run_joint_graph_passes_on_hops: the backward config is stamped on
        # the partitioned backward invoke_subgraph node's meta["custom"] (the
        # source of truth), not the subgraph module.
        node = gm.graph.call_function(torch.ops.higher_order.invoke_subgraph)
        node.meta["custom"] = {"nested_region_config": bw_config}
        gm.graph.output(())
        gm.recompile()
        return gm

    def test_nested_region_bw_config_patches_affect_cache_key(self):
        # [FUNCTION DOC] Checks the backward config derived from compile options carries only the
        # given backward patches, and that backward patches stamped on the node affect the cache
        # key (equal patches give equal keys, different patches give different keys).
        # bw_inductor_config_patches (stamped on the backward invoke_subgraph
        # node) must be part of the cache key, or two graphs differing only in
        # backward config would collide.
        from torch._higher_order_ops.invoke_subgraph import (
            get_backward_nested_region_config,
            get_invoke_subgraph_compile_options,
        )

        # Backward config replaces (does not merge with) the forward config.
        bw_config = get_backward_nested_region_config(
            get_invoke_subgraph_compile_options(
                bw_inductor_config_patches={"max_autotune": True}
            )
        )
        self.assertEqual(
            bw_config.inductor_config_patches,
            {"max_autotune": True},
        )

        same1 = self._nested_region_bw_gm({"max_autotune": True})
        same2 = self._nested_region_bw_gm({"max_autotune": True})
        different = self._nested_region_bw_gm({"max_autotune": False})
        self.assertEqual(
            self._fx_graph_cache_key(same1, []),
            self._fx_graph_cache_key(same2, []),
        )
        self.assertNotEqual(
            self._fx_graph_cache_key(same1, []),
            self._fx_graph_cache_key(different, []),
        )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "requires MKLDNN")
    def test_check_for_hop_skips_constants(self):
        # [FUNCTION DOC] Requires MKLDNN. Checks FxGraphCache._check_for_hop does not raise for a
        # graph whose only content is an mkldnn constant attribute.
        graph = torch.fx.Graph()
        output = graph.get_attr("mkldnn_weight")
        graph.output(output)
        gm = torch.fx.GraphModule(
            {"mkldnn_weight": torch.randn(2, 2).to_mkldnn()}, graph
        )

        FxGraphCache._check_for_hop(gm)

    def test_check_can_cache_checks_backward_state_example_inputs(self):
        # [FUNCTION DOC] Checks _check_can_cache bypasses with a message about Reduce being
        # unsupported when an example input is a BackwardState object.
        gm = torch.fx.GraphModule({}, torch.fx.Graph())

        with self.assertRaisesRegex(BypassFxGraphCache, "Reduce unsupported"):
            FxGraphCache._check_can_cache(
                gm,
                [torch.fx.experimental._backward_state.BackwardState()],
                require_shape_env=False,
            )

    @unittest.skipIf(not torch.backends.mkldnn.is_available(), "requires MKLDNN")
    def test_check_can_cache_checks_nested_fx_kwargs_tensor(self):
        # [FUNCTION DOC] Requires MKLDNN. Checks _check_can_cache finds an mkldnn tensor
        # nested inside the fx_kwargs mapping and raises BypassFxGraphCache about unpickleable
        # mkldnn tensors.
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        fx_kwargs = cast(
            Any, {"nested": {"mkldnn_weight": torch.randn(2, 2).to_mkldnn()}}
        )

        with self.assertRaisesRegex(BypassFxGraphCache, "mkldnn tensors unpickleable"):
            FxGraphCache._check_can_cache(
                gm,
                fx_kwargs=fx_kwargs,
                require_shape_env=False,
            )

    def test_bypass_for_pickle_error_logs_traceback(self):
        # [FUNCTION DOC] Checks CacheabilityValidator.bypass_for_pickle_error raises
        # BypassFxGraphCache with a message about failing to pickle the key, and logs exactly one
        # warning on the codecache logger carrying the exception info.
        with self.assertLogs("torch._inductor.codecache", level="WARNING") as logs:
            with self.assertRaisesRegex(
                BypassFxGraphCache, "Failed to pickle cache key"
            ):
                try:
                    raise RuntimeError("pickle failed")
                except RuntimeError as e:
                    CacheabilityValidator.bypass_for_pickle_error(e)

        self.assertEqual(len(logs.records), 1)
        self.assertIn("Failed to pickle cache key", logs.records[0].getMessage())
        self.assertIsNotNone(logs.records[0].exc_info)

    def test_parameter_constants(self):
        # [FUNCTION DOC] Checks pickling of parameter constants. Small parameters can be inlined
        # as constants and large ones cannot. By default both pickle to TensorMetadataAndValues.
        # For a module with frozen parameters, the small one still keeps values but the large one
        # becomes metadata only.
        """
        Test the hashing of parameter constants.
        """
        small = torch.nn.Parameter(torch.rand(8))
        large = torch.nn.Parameter(torch.rand(32))

        self.assertTrue(GraphLowering.can_inline_constant(small))
        self.assertFalse(GraphLowering.can_inline_constant(large))

        # By default, we hash the metadata and values independent of the size.
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        data = pickler.dumps(small)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)
        data = pickler.dumps(large)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

        # For frozen parameters, we only hash the values of small tensors.
        gm._has_frozen_params = True
        gm._frozen_param0 = small
        gm._frozen_param1 = large
        small._is_frozen_param = True
        large._is_frozen_param = True
        pickler = FxGraphCachePickler(gm)

        data = pickler.dumps(small)
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)
        data = pickler.dumps(large)
        self.assertIsInstance(pickle.loads(data), TensorMetadata)

    def test_hash_fake_tensors(self):
        # [FUNCTION DOC] Checks fake tensor pickling is a TensorMetadata and that the dumps are
        # equal or different as expected for: shape, strides (a double transpose is equal),
        # storage offset (ignored), dtype, requires_grad, memory format, and device (meta versus
        # cpu, and different GPU indices when multiple GPUs exist).
        """
        Test hashing (pickling) FakeTensors with various characteristics.
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        with torch._subclasses.FakeTensorMode():
            # Verify that FakeTensors get pickled into a TensorMetadata:
            data = pickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadata)

            # Different shapes:
            self.assertEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(3)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(4)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3)),
                pickler.dumps(torch.randn(3, 3)),
            )

            self.assertEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 4)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(4, 3)),
            )

            # Different strides:
            self.assertEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3).transpose(0, 1).transpose(0, 1)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, 3)),
                pickler.dumps(torch.randn(3, 3).transpose(0, 1)),
            )

            # Different storage offsets:
            self.assertEqual(
                pickler.dumps(torch.randn(3)[1:]),
                pickler.dumps(torch.randn(3)[1:]),
            )
            self.assertEqual(
                pickler.dumps(torch.randn(3)[1:]),
                pickler.dumps(torch.randn(2)),
            )

            # Different dtypes:
            self.assertEqual(
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, dtype=torch.float32)),
                pickler.dumps(torch.randn(3, dtype=torch.float64)),
            )

            # Different 'requires_grad':
            self.assertEqual(
                pickler.dumps(torch.randn(3, requires_grad=True)),
                pickler.dumps(torch.randn(3, requires_grad=True)),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, requires_grad=True)),
                pickler.dumps(torch.randn(3, requires_grad=False)),
            )

            # Different memory formats:
            self.assertNotEqual(
                pickler.dumps(torch.randn(1, 2, 3, 4)),
                pickler.dumps(
                    torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)
                ),
            )

            # Different devices:
            self.assertEqual(
                pickler.dumps(torch.randn(3, device="meta")),
                pickler.dumps(torch.randn(3, device="meta")),
            )
            self.assertNotEqual(
                pickler.dumps(torch.randn(3, device="meta")),
                pickler.dumps(torch.randn(3, device="cpu")),
            )

            if HAS_MULTIGPU:
                self.assertEqual(
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )
                self.assertNotEqual(
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:0")),
                    pickler.dumps(torch.randn(3, device=f"{GPU_TYPE}:1")),
                )

    def test_hash_kwargs(self):
        # [FUNCTION DOC] Checks hashing of kwargs in FxGraphHashDetails: dict order does not
        # matter, value changes do, set iteration order does not matter, and different set
        # contents do.
        """
        Test the special handling of the kwargs when hashing, i.e.,
        ordering of the kwargs dict and any set arguments.
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        # Dict order of the kwargs should not affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0, "z": 1}, [])
        details2 = FxGraphHashDetails(None, [], {"z": 1, "a": 0}, [])
        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # Different kwarg values should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": 0}, [])
        details2 = FxGraphHashDetails(None, [], {"a": 1}, [])
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # Set order should not affect hashes. Sets are unordered, but
        # sorting and creating a new set seems to change the order.
        set1 = {"a", "b", "c", "d", "e", "f", "g"}
        set2 = set(sorted(set1))  # noqa: C414
        details1 = FxGraphHashDetails(None, [], {"a": set1}, [])
        details2 = FxGraphHashDetails(None, [], {"a": set2}, [])
        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

        # But different set contents should affect hashes.
        details1 = FxGraphHashDetails(None, [], {"a": {1, 2, 3}}, [])
        details2 = FxGraphHashDetails(None, [], {"a": {1, 2}}, [])
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )

    def test_hash_config_changes(self):
        # [FUNCTION DOC] Checks that the max_autotune setting is included: two details built under
        # the same value hash equally and one built under a different value hashes differently.
        """
        Test that different config settings affect hashes.
        """
        with config.patch({"max_autotune": False}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"max_autotune": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_hash_config_comms_changes(self):
        # [FUNCTION DOC] Same check as the config hash test but for the config_comms setting that
        # enables NCCL library runtime estimations.
        """
        Test that changes to config_comms settings affect hashes.
        """
        with config_comms.patch(
            {"runtime_estimations_use_nccl_lib_estimations": False}
        ):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config_comms.patch({"runtime_estimations_use_nccl_lib_estimations": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_hash_private_config_changes(self):
        # [FUNCTION DOC] Checks that a private (underscore) config option, the micro pipeline
        # tensor parallel flag, changes the hash.
        """
        Test that private config settings affect hashes.
        """
        with config.patch({"_micro_pipeline_tp": False}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"_micro_pipeline_tp": True}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_hash_provenance_tracking_level(self):
        # [FUNCTION DOC] Checks that the provenance tracking level config changes the hash.
        """
        Test that provenance_tracking_level affects hashes.
        """
        with config.patch({"trace.provenance_tracking_level": 0}):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch({"trace.provenance_tracking_level": 1}):
            details3 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

    def test_hash_provenance_tracking_to_timeline(self):
        # [FUNCTION DOC] Checks that provenance_tracking_to_timeline changes the hash when the
        # tracking level is 1, and that changing the level to 0 with the timeline flag on also
        # differs from the baseline.
        """
        Test that provenance_tracking_to_timeline affects hashes.
        """
        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "trace.provenance_tracking_to_timeline": False,
            }
        ):
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "trace.provenance_tracking_to_timeline": True,
            }
        ):
            details3 = FxGraphHashDetails(None, [], {}, [])

        with config.patch(
            {
                "trace.provenance_tracking_level": 0,
                "trace.provenance_tracking_to_timeline": True,
            }
        ):
            details4 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details4),
        )

    def test_provenance_tracking_level_causes_cache_miss(self):
        # [FUNCTION DOC] Compiles a small module under tracking level 0: first a miss then a
        # hit after dynamo reset. Switching the provenance tracking level to 1 causes a new
        # miss. No bypass.
        """
        Test that changing provenance_tracking_level causes a cache miss.
        """

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random 4 by 4 parameter whose forward returns the
            # input matrix-multiplied by it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the matrix product of the input and the parameter.
                return x @ self.param

        mod = Mod()
        mod_compiled = torch.compile(mod)
        with torch.no_grad():
            x = torch.rand(4, 4)
            with config.patch({"trace.provenance_tracking_level": 0}):
                # miss
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                # hit
                torch._dynamo.reset()
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                torch._dynamo.reset()
                counters.clear()

            with config.patch({"trace.provenance_tracking_level": 1}):
                # miss (provenance_tracking_level changed)
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                torch._dynamo.reset()
                counters.clear()

    def test_non_serializable_custom_passes_causes_cache_miss(self):
        # [FUNCTION DOC] Checks the effect of the ddp communication passes private config on
        # caching. Baseline: a miss then a hit. Changing the list to a new string name causes a
        # miss. A list holding an unserializable lambda causes a bypass, and the codecache log
        # must contain the explicit message saying the FX Graph Cache is bypassed because of an
        # unsupported ddp pass.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a random 4 by 4 parameter whose forward returns the
            # input matrix-multiplied by it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the module and its parameter.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return the matrix product of the input and the parameter.
                return x @ self.param

        mod1 = Mod()
        mod_compiled = torch.compile(mod1)
        with torch.no_grad():
            x = torch.rand(4, 4)
            # miss
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            # hit
            torch._dynamo.reset()
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            torch._dynamo.reset()
            counters.clear()

            # hit
            mod_compiled(x)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            with config.patch({"_fuse_ddp_communication_passes": ["new_pass_foo_bar"]}):
                # miss (private config changed)
                torch._dynamo.reset()
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                torch._dynamo.reset()
                counters.clear()

            with (
                capture_logs("torch._inductor.codecache", logging.INFO) as logs,
                config.patch({"_fuse_ddp_communication_passes": [lambda *args: None]}),
            ):
                # bypass (custom pass is not serializable)
                mod_compiled(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                counters.clear()
            # assert that our bypass is explicit
            self.assertTrue(
                any(
                    x.getMessage()
                    == "Bypassing FX Graph Cache because 'Unsupported _fuse_ddp_communication_pass'"
                    for x in logs
                )
            )

    def test_hash_custom_passes(self):
        # [FUNCTION DOC] Checks CustomGraphPass uuids feed the hash. For a single post grad pre
        # pass, equal uuids hash equal and a changed uuid hashes differently. For a list of two
        # passes, equal configurations hash equal, a changed uuid on one pass differs, and
        # swapping the list order also differs.
        """
        Test CustomGraphPass usage.
        """

        class TestCustomGraphPass(CustomGraphPass):
            # [CLASS DOC] HELPER: CustomGraphPass test double with a settable uuid string
            # (initially None) that leaves graphs untouched.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set the uuid to None.
                self._uuid = None

            def __call__(self, graph: torch.fx.graph.Graph) -> None:
                # [FUNCTION DOC] HELPER: Do nothing and return None.
                return None

            def uuid(self) -> bytes | str | None:
                # [FUNCTION DOC] HELPER: Return the stored uuid.
                return self._uuid

        custom_pass = TestCustomGraphPass()
        with config.patch({"post_grad_custom_pre_pass": custom_pass}):
            custom_pass._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_pass._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

        custom_pass_1 = TestCustomGraphPass()
        custom_pass_2 = TestCustomGraphPass()
        with config.patch(
            {"post_grad_custom_pre_pass": [custom_pass_1, custom_pass_2]}
        ):
            custom_pass_1._uuid = "1"
            custom_pass_2._uuid = "2"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_pass_2._uuid = "3"
            details3 = FxGraphHashDetails(None, [], {}, [])

        with config.patch(
            {"post_grad_custom_pre_pass": [custom_pass_2, custom_pass_1]}
        ):
            details4 = FxGraphHashDetails(None, [], {}, [])

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )
        self.assertNotEqual(
            pickler.dumps(details3),
            pickler.dumps(details4),
        )

    def test_hash_custom_backend_pass(self):
        # [FUNCTION DOC] Same uuid check as for graph passes but for a CustomGraphModulePass
        # installed as the CPU backend custom pass through patch_inductor_backend: equal uuids
        # hash equal and a different uuid changes the hash.
        """
        Test CustomGraphModulePass usage.
        """

        class TestCustomGraphModulePass(CustomGraphModulePass):
            # [CLASS DOC] HELPER: CustomGraphModulePass test double with a settable uuid
            # (initially None) that does not modify the graph module.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set the uuid to None.
                self._uuid = None

            def __call__(self, gm: torch.fx.GraphModule) -> None:
                # [FUNCTION DOC] HELPER: Do nothing and return None.
                return None

            def uuid(self) -> bytes | str | None:
                # [FUNCTION DOC] HELPER: Return the stored uuid.
                return self._uuid

        custom_pass = TestCustomGraphModulePass()
        with patch_inductor_backend("cpu", custom_pass=custom_pass):
            custom_pass._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_pass._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

    def test_hash_custom_backend_config(self):
        # [FUNCTION DOC] With a custom backend config module installed for CPU, checks the enable
        # optimisation flag of that module changes the hash and the cache: a first compile misses,
        # a second after reset hits, and flipping the flag causes a new miss. Mutates the flag on
        # the shared custom config module (true, false, then true) and never restores it.
        """
        Test cache correctness when a custom inductor codegen config
        is installed
        """
        with patch_inductor_backend(
            "cpu", custom_backend_config=custom_inductor_config
        ):
            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])
            self.assertEqual(pickler.dumps(details1), pickler.dumps(details2))

            custom_inductor_config.enable_optimisation = True
            details3 = FxGraphHashDetails(None, [], {}, [])
            self.assertNotEqual(pickler.dumps(details2), pickler.dumps(details3))

            torch._dynamo.reset()
            counters.clear()

            custom_inductor_config.enable_optimisation = False
            x = torch.zeros(32)
            y = torch.zeros(32)
            compiled_fn = torch.compile(torch.add)

            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            torch._dynamo.reset()
            counters.clear()

            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
            torch._dynamo.reset()
            counters.clear()

            # Changing the custom config should trigger a recompilation
            custom_inductor_config.enable_optimisation = True
            compiled_fn(x, y)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

    def test_hash_custom_partitioner_fn(self):
        # [FUNCTION DOC] Checks the custom partitioner function's uuid is hashed: the details
        # record the uuid in effect when they are built, equal uuids hash equal and different
        # uuids hash differently.
        """
        Test that the custom partitioner function's UUID is properly used in the FX graph cache hashing.
        """
        custom_partitioner_fn = _TestCustomPartitionerFn()
        with config.patch({"custom_partitioner_fn": custom_partitioner_fn}):
            custom_partitioner_fn._uuid = "1"
            details1 = FxGraphHashDetails(None, [], {}, [])
            details2 = FxGraphHashDetails(None, [], {}, [])

            custom_partitioner_fn._uuid = "2"
            details3 = FxGraphHashDetails(None, [], {}, [])

            self.assertEqual(details1._custom_partitioner_fn, "1")
            self.assertEqual(details2._custom_partitioner_fn, "1")
            self.assertEqual(details3._custom_partitioner_fn, "2")

            gm = torch.fx.GraphModule({}, torch.fx.Graph())
            pickler = FxGraphCachePickler(gm)

            self.assertEqual(
                pickler.dumps(details1),
                pickler.dumps(details2),
            )
            self.assertNotEqual(
                pickler.dumps(details1),
                pickler.dumps(details3),
            )

    def test_hash_var_to_hint_override(self):
        # [FUNCTION DOC] Checks the var_to_hint_override mapping on the hash details: equal
        # mappings hash equal, a different hint value differs, and an empty mapping differs from a
        # non-empty one.
        """Test that different var_to_hint_override values produce different hashes."""
        details1 = FxGraphHashDetails(None, [], {}, [])
        details2 = FxGraphHashDetails(None, [], {}, [])

        details1.var_to_hint_override = {"s0": 64}
        details2.var_to_hint_override = {"s0": 64}
        details3 = FxGraphHashDetails(None, [], {}, [])
        details3.var_to_hint_override = {"s0": 99}

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        self.assertEqual(
            pickler.dumps(details1),
            pickler.dumps(details2),
        )
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details3),
        )

        # Empty vs non-empty
        details4 = FxGraphHashDetails(None, [], {}, [])
        details4.var_to_hint_override = {}
        self.assertNotEqual(
            pickler.dumps(details1),
            pickler.dumps(details4),
        )

    def test_bypass_unsupported(self):
        # [FUNCTION DOC] Checks pickling a BackwardState object with FxGraphCachePickler raises
        # BypassFxGraphCache (the unsupported-type reducer).
        """
        Test _reduce_unsupported
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        with self.assertRaises(BypassFxGraphCache):
            FxGraphCachePickler(gm).dumps(
                torch.fx.experimental._backward_state.BackwardState()
            )

    def test_bypass_on_runtime_error(self):
        # [FUNCTION DOC] Checks that a RuntimeError with the pybind11 not-pickleable message
        # raised while pickling is converted to BypassFxGraphCache.
        """
        Test that RuntimeError raised by pybind11 during pickling (e.g., from
        OpOverload._op which is a C++ function pointer) is caught and converted
        to BypassFxGraphCache.
        """

        class Pybind11LikeUnpickleableObject:
            # [CLASS DOC] HELPER: Object whose reduce hook raises a RuntimeError with a message
            # that mimics pybind11 objects being not pickleable.
            """An object that raises RuntimeError like pybind11 does when pickled."""

            def __reduce__(self):
                # [FUNCTION DOC] HELPER: Always raise RuntimeError with the
                # pybind11-style message.
                raise RuntimeError(
                    "<pybind11_builtins.pybind11_object> is not pickleable."
                )

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        with self.assertRaises(BypassFxGraphCache):
            pickler.dumps(Pybind11LikeUnpickleableObject())

    def test_non_pybind11_runtime_error_propagates(self):
        # [FUNCTION DOC] Checks a RuntimeError whose message is not the pybind11 pattern is not
        # converted and propagates out of the pickler dumps.
        """
        Test that RuntimeErrors not matching pybind11 pickle pattern propagate up.
        """

        class OtherRuntimeErrorObject:
            # [CLASS DOC] HELPER: Object whose reduce hook raises a generic RuntimeError.
            """An object that raises a non-pybind11 RuntimeError when pickled."""

            def __reduce__(self):
                # [FUNCTION DOC] HELPER: Always raise RuntimeError with a generic message.
                raise RuntimeError("Some other runtime error")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        with self.assertRaises(RuntimeError):
            pickler.dumps(OtherRuntimeErrorObject())

    def test_stable_strings(self):
        # [FUNCTION DOC] Checks that two lists holding equal strings pickle to the same bytes
        # whether the strings are the same object or merely equal (distinct ids).
        """
        Test that objects containing identical strings pickle the same
        even if they are not the same id.
        """
        s1 = "string"
        s2 = "strin"  # codespell:ignore
        s2 += "g"

        self.assertNotEqual(id(s1), id(s2))

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        self.assertEqual(
            pickler.dumps([s1, s1]),
            pickler.dumps([s1, s2]),
        )

    def test_pickle_opaque_type_with_cyclic_references(self):
        # [FUNCTION DOC] Checks that the pickler can serialize a FakeScriptObject wrapping an
        # opaque object that references itself (a cycle). The pickler must turn off fast mode for
        # opaque types so the memo table handles the cycle; the result must not be None.
        """
        Test that FxGraphCachePickler can pickle a FakeScriptObject wrapping an
        opaque type whose real_obj has cyclic references.  Before the fix,
        fast-mode pickling (no memo table) would hit a ValueError on cycles.
        """
        from torch._library.fake_class_registry import FakeScriptObject

        obj = _CyclicOpaque()
        obj.child = obj  # create a cycle

        fake = FakeScriptObject(
            wrapped_obj=None,
            script_class_name="test::_CyclicOpaque",
            x=obj,
        )

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        # Should succeed — the pickler disables fast mode for opaque types
        # so the memo table can track the cycle.
        data = pickler.dumps(fake)
        self.assertIsNotNone(data)

    def test_get_hash_for_files(self):
        # [FUNCTION DOC] Checks get_hash_for_files: the hash of a temp file is stable across a
        # cache clear, and changes after the file content is appended to. The temp file is closed
        # and deleted at the end.
        """
        Test the get_hash_for_files helper.
        """
        # delete=True does not work on Windows.
        # See https://docs.python.org/3.12/library/tempfile.html#tempfile.NamedTemporaryFile
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            try:
                temp.write(b"contents")
                temp.flush()

                hash1 = get_hash_for_files((temp.name,))
                get_hash_for_files.cache_clear()
                hash2 = get_hash_for_files((temp.name,))

                temp.write(b" ")
                temp.flush()
                get_hash_for_files.cache_clear()
                hash3 = get_hash_for_files((temp.name,))

                self.assertEqual(hash1, hash2)
                self.assertNotEqual(hash1, hash3)
            finally:
                temp.close()
                os.unlink(temp.name)

    def test_reducer_override_unpicklable_type(self):
        # [FUNCTION DOC] Checks the pickler's reducer override for objects whose reduce_ex
        # raises a type error: two instances with the same name hash equal and different names
        # hash differently.
        """
        Test that FxGraphCachePickler handles objects whose __reduce_ex__
        raises TypeError (e.g. pybind11 enums) via reducer_override instead
        of crashing or bypassing the cache.
        """

        class UnpicklableEnum:
            # [CLASS DOC] HELPER: Simulates an unpicklable pybind11 enum; stores a name and
            # refuses to be pickled with a type error.
            """Simulates a pybind11 enum that cannot be pickled."""

            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Store the name.
                self.name = name

            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raise TypeError to simulate being unpicklable.
                raise TypeError("cannot pickle 'UnpicklableEnum' object")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        obj_a = UnpicklableEnum("SUM")
        obj_b = UnpicklableEnum("SUM")
        obj_c = UnpicklableEnum("PRODUCT")

        # Same name -> same hash (stability)
        self.assertEqual(pickler.dumps(obj_a), pickler.dumps(obj_b))
        # Different name -> different hash
        self.assertNotEqual(pickler.dumps(obj_a), pickler.dumps(obj_c))

    def test_reducer_override_unpicklable_with_pybind11_pattern(self):
        # [FUNCTION DOC] Checks the reducer override uses the nested type name accessor of
        # enum-like unpicklable objects: equal type names hash equal and different type names
        # hash differently.
        """
        Test reducer_override with objects that have the pybind11 enum
        accessor pattern (obj.type.name).
        """

        class _EnumType:
            # [CLASS DOC] HELPER: Simple holder for a name.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Store the name.
                self.name = name

        class Pybind11Enum:
            # [CLASS DOC] HELPER: Simulates a pybind11 enum with a type attribute (carrying a
            # name) and an integer value, and which cannot be pickled.
            """Simulates pybind11 enum with .type.name access pattern."""

            def __init__(self, type_name, value):
                # [FUNCTION DOC] HELPER: Store a type holder built from the type name, and
                # the value.
                self.type = _EnumType(type_name)
                self.value = value

            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raise TypeError to simulate being unpicklable.
                raise TypeError("cannot pickle 'Pybind11Enum' object")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        obj1 = Pybind11Enum("ReduceOp", 0)
        obj2 = Pybind11Enum("ReduceOp", 0)
        obj3 = Pybind11Enum("AllReduceOp", 0)

        # Same type.name -> same hash
        self.assertEqual(pickler.dumps(obj1), pickler.dumps(obj2))
        # Different type.name -> different hash
        self.assertNotEqual(pickler.dumps(obj1), pickler.dumps(obj3))

    def test_reducer_override_unpicklable_with_value_pattern(self):
        # [FUNCTION DOC] Checks the reducer override for unpicklable objects exposing only a value
        # attribute: equal values hash equal and different values hash differently.
        """
        Test reducer_override with objects that only expose a .value
        accessor (no .type.name or .name).
        """

        class ValueOnlyObj:
            # [CLASS DOC] HELPER: Unpicklable object exposing only a value attribute.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Store the value.
                self.value = value

            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raise TypeError to simulate being unpicklable.
                raise TypeError("cannot pickle 'ValueOnlyObj' object")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        obj1 = ValueOnlyObj(42)
        obj2 = ValueOnlyObj(42)
        obj3 = ValueOnlyObj(99)

        # Same value -> same hash
        self.assertEqual(pickler.dumps(obj1), pickler.dumps(obj2))
        # Different value -> different hash
        self.assertNotEqual(pickler.dumps(obj1), pickler.dumps(obj3))

    def test_reducer_override_does_not_affect_normal_types(self):
        # [FUNCTION DOC] Checks that ordinary picklable values (an int, a string, a list) are
        # pickled normally, i.e. loading the bytes returns the original value.
        """
        Test that reducer_override returns NotImplemented for types that
        pickle normally, so standard pickling is used for them.
        """
        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)

        # Normal picklable objects should work as before
        data = pickler.dumps(42)
        self.assertEqual(pickle.loads(data), 42)

        data = pickler.dumps("hello")
        self.assertEqual(pickle.loads(data), "hello")

        data = pickler.dumps([1, 2, 3])
        self.assertEqual(pickle.loads(data), [1, 2, 3])

    def test_reducer_override_caches_probed_types(self):
        # [FUNCTION DOC] Checks the class-level pickleable type cache: a new unpicklable class is
        # absent before the first dump, recorded as false afterwards, and further instances with
        # equal data hash equal. The cache entry is removed in a cleanup.
        """
        Test that once a type is probed, it is cached in _pickleable_type_cache
        for subsequent fast-path lookups (both picklable and unpicklable).
        """

        class AnotherUnpicklable:
            # [CLASS DOC] HELPER: Unpicklable object with a name attribute.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: Store the name.
                self.name = name

            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raise TypeError to simulate being unpicklable.
                raise TypeError("cannot pickle")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        self.addCleanup(
            FxGraphCachePickler._pickleable_type_cache.pop, AnotherUnpicklable, None
        )

        # Unpicklable type gets cached as False (class-level cache)
        self.assertNotIn(AnotherUnpicklable, FxGraphCachePickler._pickleable_type_cache)
        pickler.dumps(AnotherUnpicklable("x"))
        self.assertFalse(FxGraphCachePickler._pickleable_type_cache[AnotherUnpicklable])

        # Second call should use fast path (same result)
        obj1 = AnotherUnpicklable("y")
        obj2 = AnotherUnpicklable("y")
        self.assertEqual(pickler.dumps(obj1), pickler.dumps(obj2))

    def test_reducer_override_bypasses_cache_for_totally_opaque_types(self):
        # [FUNCTION DOC] Checks an unpicklable object with no stable representation and no known
        # accessor attributes makes the pickler raise BypassFxGraphCache rather than produce a
        # possibly wrong key. The type cache entry is removed in a cleanup.
        """
        Test that unpicklable objects with no stable repr and no known
        accessors cause BypassFxGraphCache instead of silently producing
        a potentially incorrect cache key.
        """

        class TotallyOpaque:
            # [CLASS DOC] HELPER: Unpicklable object without any attributes.
            def __reduce_ex__(self, protocol):
                # [FUNCTION DOC] HELPER: Always raise TypeError to simulate being unpicklable.
                raise TypeError("cannot pickle")

        gm = torch.fx.GraphModule({}, torch.fx.Graph())
        pickler = FxGraphCachePickler(gm)
        self.addCleanup(
            FxGraphCachePickler._pickleable_type_cache.pop, TotallyOpaque, None
        )

        with self.assertRaises(BypassFxGraphCache):
            pickler.dumps(TotallyOpaque())


class TestCudaCompileCommand(TestCase):
    # [CLASS DOC] Tests for building nvcc command lines in CUDA code compilation.
    @requires_cuda_and_triton
    def test_cuda_compile_command(self):
        # [FUNCTION DOC] Requires CUDA and triton. Checks cuda_compile_command output contains
        # nvcc, the source and output names, and extra flags when provided. Then mocks the
        # subprocess output call and checks CUDACodeCache.compile runs an nvcc command
        # including the extra flag and the NDEBUG define. Failures raise AssertionError with
        # the command text.
        cmd_no_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so"
        )
        if "nvcc " not in cmd_no_extra_args:
            raise AssertionError(cmd_no_extra_args)
        if "abc.cu" not in cmd_no_extra_args:
            raise AssertionError(cmd_no_extra_args)
        if "def.cu" not in cmd_no_extra_args:
            raise AssertionError(cmd_no_extra_args)
        if "output" not in cmd_no_extra_args:
            raise AssertionError(cmd_no_extra_args)
        cmd_extra_args: str = cuda_compile_command(
            ["abc.cu", "def.cu"], "output", "so", ["-Wwhatever", "-nothing"]
        )
        if "nvcc " not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        if " -Wwhatever" not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        if " -nothing" not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        if "abc.cu" not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        if "def.cu" not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        if "output " not in cmd_extra_args:
            raise AssertionError(cmd_extra_args)
        with mock.patch("subprocess.check_output") as check_output_mock:
            CUDACodeCache.compile("test123.cu", "so", ["-Wsomething"])
            check_output_mock.assert_called()
            cmd_parts: list[str] = check_output_mock.call_args[0][0]
            if not cmd_parts[0].endswith("nvcc"):
                raise AssertionError(cmd_parts)
            if "-Wsomething" not in cmd_parts:
                raise AssertionError(cmd_parts)
            if "-DNDEBUG" not in cmd_parts:
                raise AssertionError(cmd_parts)


@instantiate_parametrized_tests
class TestAutotuneCache(TestCase):
    # [CLASS DOC] Tests for the autotune cache (local, remote and bundled) in
    # torch._inductor.runtime.autotune_cache, using mock caches installed by PatchCaches. Setup
    # clears counters and artifacts and installs the mocks; teardown removes them.
    device_type = GPU_TYPE

    def setUp(self):
        # [FUNCTION DOC] Clear counters and the artifact manager and install the mock caches.
        super().setUp()
        counters.clear()
        PatchCaches.setUp()
        CacheArtifactManager.clear()

    def tearDown(self):
        # [FUNCTION DOC] Run the base tearDown and remove the mock cache patches.
        super().tearDown()
        PatchCaches.tearDown()

    def reset(self):
        # [FUNCTION DOC] HELPER: Purge the PyCodeCache, reset dynamo and clear inductor caches.
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    def test_remote_only_autotune_records_cache_artifacts(self):
        # [FUNCTION DOC] With only the remote autotune cache enabled (via a stub returned by a
        # mocked create_cache), checks AutotuneCache.create has no local cache but does have a
        # remote cache and an artifact recorder. Saving a config records an autotune artifact
        # whose key matches the recorder key, and reading it back records the same artifact again.
        from torch._inductor.runtime.autotune_cache import AutotuneCache

        class RemoteCacheStub:
            # [CLASS DOC] HELPER: Dictionary-backed remote cache stub with get and put.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the empty dictionary.
                self.cache = {}

            def get(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for the key or None.
                return self.cache.get(key)

            def put(self, key, value) -> None:
                # [FUNCTION DOC] HELPER: Store the value under the key.
                self.cache[key] = value

        class ConfigStub:
            # [CLASS DOC] HELPER: Fake triton config with block size kwargs, 4 warps and 3 stages.
            kwargs = {"BLOCK_M": 16, "BLOCK_N": 16}
            num_warps = 4
            num_stages = 3

        remote_cache = RemoteCacheStub()
        inductor_meta = {
            "autotune_local_cache": False,
            "autotune_remote_cache": True,
            "backend_hash": "backend_hash",
            "is_fbcode": False,
        }

        with mock.patch(
            "torch._inductor.runtime.autotune_cache.create_cache",
            return_value=remote_cache,
        ):
            autotune_cache = AutotuneCache.create(
                inductor_meta,
                os.path.join(cache_dir(), "aa", "kernel.py"),
                "configs_hash",
            )

        self.assertIsNotNone(autotune_cache)
        autotune_cache = cast(AutotuneCache, autotune_cache)
        self.assertIsNone(autotune_cache.local_cache)
        self.assertIsNotNone(autotune_cache.remote_cache)
        self.assertIsNotNone(autotune_cache.artifact_recorder)
        artifact_recorder = cast(
            CacheArtifactRecorder, autotune_cache.artifact_recorder
        )
        expected_artifact_key = artifact_recorder.key

        CacheArtifactManager.clear()
        autotune_cache.save(ConfigStub(), time_taken_ns=1_000_000)

        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifacts = cast(tuple[bytes, CacheInfo], artifacts)
        _, cache_info = artifacts
        self.assertEqual(cache_info.autotune_artifacts, [expected_artifact_key])

        CacheArtifactManager.clear()
        self.assertIsNotNone(autotune_cache._read())
        artifacts = torch.compiler.save_cache_artifacts()
        self.assertIsNotNone(artifacts)
        artifacts = cast(tuple[bytes, CacheInfo], artifacts)
        _, cache_info = artifacts
        self.assertEqual(cache_info.autotune_artifacts, [expected_artifact_key])

    @config.patch({"bundled_autotune_remote_cache": True})
    def test_autotune_cache_bundler_is_compile_context_scoped(self):
        # [FUNCTION DOC] Checks AutotuneCacheBundler keeps separate state per compile context. Two
        # nested contexts each begin a compile and put entries; ending each flushes a single
        # bundle to its own cache keyed by entry file base names: the outer cache gets two entries
        # and the inner cache gets one.
        from torch._guards import compile_context, CompileContext
        from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler

        class FakeBundledCache:
            # [CLASS DOC] HELPER: Fake bundled remote cache that never hits and records every put
            # as a key and data pair.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the empty list of puts.
                self.puts = []

            def get(self, key):
                # [FUNCTION DOC] HELPER: Always return None (no cached data).
                return None

            def put(self, key, data):
                # [FUNCTION DOC] HELPER: Record the key and data pair.
                self.puts.append((key, data))

        cache1 = FakeBundledCache()
        cache2 = FakeBundledCache()
        inductor_meta = {
            "autotune_local_cache": True,
            "backend_hash": "backend",
            "bundled_autotune_remote_cache": True,
            "is_fbcode": False,
        }

        ctx1 = CompileContext(None)
        ctx2 = CompileContext(None)
        with mock.patch(
            "torch._inductor.runtime.autotune_cache.create_cache",
            side_effect=[cache1, cache2],
        ):
            with compile_context(ctx1):
                AutotuneCacheBundler.begin_compile(inductor_meta, code_hash="code1")
                AutotuneCacheBundler.put("/tmp/cache/aa/entry1.best_config", {"ctx": 1})

                with compile_context(ctx2):
                    AutotuneCacheBundler.begin_compile(inductor_meta, code_hash="code2")
                    AutotuneCacheBundler.put(
                        "/tmp/cache/bb/entry2.best_config", {"ctx": 2}
                    )
                    AutotuneCacheBundler.end_compile()

                AutotuneCacheBundler.put("/tmp/cache/aa/entry3.best_config", {"ctx": 1})
                AutotuneCacheBundler.end_compile()

        self.assertEqual(len(cache1.puts), 1)
        self.assertEqual(
            cache1.puts[0][1],
            {
                "entry1.best_config": {"ctx": 1},
                "entry3.best_config": {"ctx": 1},
            },
        )
        self.assertEqual(len(cache2.puts), 1)
        self.assertEqual(cache2.puts[0][1], {"entry2.best_config": {"ctx": 2}})

    @config.patch({"bundled_autotune_remote_cache": True})
    def test_autotune_cache_bundler_runs_in_saved_compile_context(self):
        # [FUNCTION DOC] Checks a CompiledFxGraph that saved the compile context during
        # begin_compile runs its callable inside that context: a put done from the callable
        # lands in the bundled cache with the file base name, and the stored compile context
        # on the graph is cleared (None) afterwards. The graph is created without running
        # its constructor.
        from torch._guards import compile_context, CompileContext
        from torch._inductor.output_code import CompiledFxGraph
        from torch._inductor.runtime.autotune_cache import AutotuneCacheBundler

        class FakeBundledCache:
            # [CLASS DOC] HELPER: Fake bundled remote cache that never hits and records every put
            # as a key and data pair.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Create the empty list of puts.
                self.puts = []

            def get(self, key):
                # [FUNCTION DOC] HELPER: Always return None (no cached data).
                return None

            def put(self, key, data):
                # [FUNCTION DOC] HELPER: Record the key and data pair.
                self.puts.append((key, data))

        cache = FakeBundledCache()
        inductor_meta = {
            "autotune_local_cache": True,
            "backend_hash": "backend",
            "bundled_autotune_remote_cache": True,
            "is_fbcode": False,
        }

        graph = CompiledFxGraph.__new__(CompiledFxGraph)
        graph.fx_kwargs = {}
        graph._compile_context = None

        def current_callable(inputs):
            # [FUNCTION DOC] HELPER: Stand-in compiled callable that puts a bundled entry through
            # AutotuneCacheBundler and returns its inputs.
            AutotuneCacheBundler.put(
                "/tmp/cache/aa/entry.best_config", {"ctx": "saved"}
            )
            return inputs

        graph.current_callable = current_callable

        with mock.patch(
            "torch._inductor.runtime.autotune_cache.create_cache",
            return_value=cache,
        ):
            with compile_context(CompileContext(None)):
                AutotuneCacheBundler.begin_compile(inductor_meta, code_hash="code")
                graph._set_compile_context_for_autotune_cache()

            self.assertEqual(graph(["result"]), ["result"])

        self.assertEqual(len(cache.puts), 1)
        self.assertEqual(cache.puts[0][1], {"entry.best_config": {"ctx": "saved"}})
        self.assertIsNone(graph._compile_context)

    @requires_cuda_and_triton
    @unittest.skipIf(not SM80OrLater, "Requires SM80+")
    @config.patch({"use_static_triton_launcher": True})
    @config.patch({"fx_graph_cache": True})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": False})
    @config.patch({"autotune_remote_cache": True})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    def test_autotune_cache_warm_start(self):
        # [FUNCTION DOC] Needs CUDA, triton, SM80 and the static triton launcher. With
        # max_autotune and the remote autotune cache (single compile thread): the first compile
        # stores two autotune entries and has one fx graph miss; after a dynamo reset the fx graph
        # cache hits and autotune stats show the two gets as hits. Also checks remote key shapes
        # for autotune and triton entries by pattern.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the first two inputs and the last two
            # inputs separately.
            def forward(self, x, y, a, b):
                # [FUNCTION DOC] HELPER: Return the sum of the first pair and the sum of the
                # second pair.
                return x + y, a + b

        def f(x, y, a, b):
            # [FUNCTION DOC] HELPER: Build the module and apply it to the four arguments.
            return Model()(x, y, a, b)

        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        a = torch.randn(1000, 100).cuda()
        b = torch.randn(1000, 100).cuda()
        f_compiled = torch.compile(f, fullgraph=True)

        with PatchCaches():
            a1 = f_compiled(x, y, a, b)

            self.assertEqual(global_stats.autotune_remote, Stats(2, 0, 2))
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)

            # Don't reset FxGraphCache, see that it loads again
            torch._dynamo.reset()
            a2 = f_compiled(x, y, a, b)
            self.assertEqual(a1, a2)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

        self.assertEqual(global_stats.autotune_remote, Stats(2, 2, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_remote.cache:
            self.assertRegex(k, r"[0-9a-z]{52}")
        for k in global_stats.triton.cache:
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": False})
    @config.patch({"autotune_remote_cache": True})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    def test_autotune_cache(self):
        # [FUNCTION DOC] Like the warm start test but with the fx graph cache off, on any GPU
        # supported by triton: the remote autotune cache records two puts and misses on the first
        # compile, then after a full reset the second compile hits both entries. Checks remote
        # key patterns.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding the first two inputs and the last two
            # inputs separately.
            def forward(self, x, y, a, b):
                # [FUNCTION DOC] HELPER: Return the sum of the first pair and the sum of the
                # second pair.
                return x + y, a + b

        def f(x, y, a, b):
            # [FUNCTION DOC] HELPER: Build the module and apply it to the four arguments.
            return Model()(x, y, a, b)

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)
        a = torch.randn(1000, 100).to(GPU_TYPE)
        b = torch.randn(1000, 100).to(GPU_TYPE)
        f_compiled = torch.compile(f, fullgraph=True)

        with PatchCaches():
            f_compiled(x, y, a, b)

            self.assertEqual(global_stats.autotune_remote, Stats(2, 0, 2))

            self.reset()
            f_compiled(x, y, a, b)

        self.assertEqual(global_stats.autotune_remote, Stats(2, 2, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_remote.cache:
            self.assertRegex(k, r"[0-9a-z]{52}")
        for k in global_stats.triton.cache:
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"autotune_local_cache": True})
    @config.patch({"autotune_remote_cache": False})
    @config.patch({"bundled_autotune_remote_cache": True})
    @config.patch({"compile_threads": 1})
    @config.patch({"max_autotune": True})
    def test_bundled_autotune_remote_cache(self):
        # [FUNCTION DOC] Tests the bundled autotune remote cache with local autotune cache
        # enabled. A three-pair function yields three local autotune puts and one bundled put;
        # after reset the local and bundled caches hit; under a different cache_key_tag with reset
        # stats the sequence repeats. Then checks key patterns of remote entries.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning three pairwise sums of six inputs.
            def forward(self, a, b, c, d, e, f):
                # [FUNCTION DOC] HELPER: Return the sums of the first, second and third pairs
                # of arguments.
                return a + b, c + d, e + f

        def f(a, b, c, d, e, f):
            # [FUNCTION DOC] HELPER: Build the module and apply it to the six arguments.
            return Model()(a, b, c, d, e, f)

        f_compiled = torch.compile(f, fullgraph=True)

        a = torch.randn(101, 100).to(GPU_TYPE)
        b = torch.randn(101, 100).to(GPU_TYPE)
        c = torch.randn(102, 100).to(GPU_TYPE)
        d = torch.randn(102, 100).to(GPU_TYPE)
        e = torch.randn(103, 100).to(GPU_TYPE)
        f = torch.randn(103, 100).to(GPU_TYPE)

        with PatchCaches():
            f_compiled(a, b, c, d, e, f)

            self.assertEqual(global_stats.autotune_local, Stats(3, 0, 3))
            self.assertEqual(global_stats.bundled_autotune, Stats(1, 0, 1))

            self.reset()
            f_compiled(a, b, c, d, e, f)

            self.assertEqual(global_stats.autotune_local, Stats(6, 3, 3))
            self.assertEqual(global_stats.bundled_autotune, Stats(1, 1, 1))

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                global_stats.reset()
                self.reset()
                f_compiled(a, b, c, d, e, f)

                self.assertEqual(global_stats.autotune_local, Stats(3, 0, 3))
                self.assertEqual(global_stats.bundled_autotune, Stats(1, 0, 1))

                self.reset()
                f_compiled(a, b, c, d, e, f)

                self.assertEqual(global_stats.autotune_local, Stats(6, 3, 3))
                self.assertEqual(global_stats.bundled_autotune, Stats(1, 1, 1))

        # Check that the cache entries seem reasonable
        for k in global_stats.autotune_local.cache:
            self.assertRegex(k, r"tmp[^/]*/([^/]{2})/[^/]{64}\.best_config")
        for k in global_stats.bundled_autotune.cache:
            self.assertRegex(k, r"pt2:bundled-autotune-v1::[0-9a-z]{64}:c[0-9]+")
        for k in global_stats.triton.cache:
            self.assertRegex(k, r"triton:[0-9a-f]{64}::[0-9a-f]{64}:c[0-9]+")

    @requires_triton()
    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": False})
    @config.patch({"bundled_autotune_remote_cache": False})
    @config.patch({"max_autotune": True})
    @config.patch(
        {"compile_threads": 1}
    )  # Worker processes do not register PatchCaches() properly
    @parametrize("remote_cache", (True, False))
    def test_modified_autotune_cache(self, remote_cache):
        # [FUNCTION DOC] Checks that a change in the torch source key invalidates autotune cache
        # entries. With max_autotune and either the remote or local autotune cache (parametrized,
        # via mock caches) it compiles once under a mocked torch key, clears dynamo and
        # PyCodeCache, compiles again under a different mocked key, and expects the autotune stats
        # to show a second put and miss and no hits. Results of both runs are equal.
        """
        If a developer changes the way the autotune cache is handled,
        there's a chance it'll break the cache. This happened with
        #150122. This test ensures that if torch code changes, then
        old cache entries will be invalidated.
        """

        def mock_torch_key(value: str) -> bytes:
            # [FUNCTION DOC] HELPER: Return the given string encoded as utf-8 bytes, used in place
            # of the real torch key.
            return value.encode("utf-8")

        def get_autotune_stats():
            # [FUNCTION DOC] HELPER: Return the global remote autotune stats when the remote cache
            # is being tested, otherwise the local autotune stats.
            if remote_cache:
                return global_stats.autotune_remote
            return global_stats.autotune_local

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return relu of the sum of the two inputs.
            return (x + y).relu()

        x = torch.randn(100, 100).to(GPU_TYPE)
        y = torch.randn(100, 100).to(GPU_TYPE)

        with config.patch(
            {
                "autotune_local_cache": not remote_cache,
                "autotune_remote_cache": remote_cache,
            }
        ):
            with PatchCaches():
                with mock.patch(
                    "torch._inductor.codecache.torch_key",
                    functools.partial(mock_torch_key, "torchkey1"),
                ):
                    f_compiled = torch.compile(fn, fullgraph=True)
                    res1 = f_compiled(x, y)

                self.assertEqual(get_autotune_stats(), Stats(1, 0, 1))

                torch._dynamo.reset()
                PyCodeCache.cache_clear()

                with mock.patch(
                    "torch._inductor.codecache.torch_key",
                    functools.partial(mock_torch_key, "torchkey2"),
                ):
                    f_compiled = torch.compile(fn, fullgraph=True)
                    res2 = f_compiled(x, y)

                self.assertEqual(get_autotune_stats(), Stats(2, 0, 2))

                self.assertEqual(res1, res2)


class TestRemoteAOTAutogradCache(TestCase):
    # [CLASS DOC] Tests for the remote AOTAutograd cache with the remote fx graph cache, using
    # mock cache stats.
    @requires_gpu()
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_autograd_remote_cache(self):
        # [FUNCTION DOC] Requires a GPU with SM80 or XPU. With the remote autograd and fx graph
        # caches on, compiles a simple add three times: the first produces a put and a miss in
        # both caches, a second compile after a dynamo reset hits both, and a compile under a
        # different cache_key_tag misses and puts again. Then checks the remote key patterns for
        # both caches.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Return the sum of the two inputs.
            return a + b

        f_compiled = torch.compile(f)
        a = torch.randn(101, 100, device=GPU_TYPE, requires_grad=False)
        b = torch.randn(101, 100, device=GPU_TYPE, requires_grad=False)
        with PatchCaches():
            f_compiled(a, b)

            self.assertEqual(global_stats.aot_autograd, Stats(1, 0, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 0, 1))

            torch._dynamo.reset()

            f_compiled(a, b)
            self.assertEqual(global_stats.aot_autograd, Stats(1, 1, 1))
            self.assertEqual(global_stats.fx_graph, Stats(1, 1, 1))

            torch._dynamo.reset()

            with torch.compiler.config.patch({"cache_key_tag": "test"}):
                f_compiled(a, b)
            self.assertEqual(global_stats.aot_autograd, Stats(2, 1, 2))
            self.assertEqual(global_stats.fx_graph, Stats(2, 1, 2))

        # Check that the cache entries seem reasonable
        for k in global_stats.aot_autograd.cache:
            self.assertRegex(k, r"pt2:autograd-experimental::[0-9a-z]{52}:c[0-9]+")

        for k in global_stats.fx_graph.cache:
            self.assertRegex(k, r"pt2:fx-graph-v1::[0-9a-z]{52}:c[0-9]+")

    @requires_gpu_and_triton
    @unittest.skipIf(not HAS_XPU_AND_TRITON and not SM80OrLater, "Requires SM80+")
    @config.patch({"fx_graph_cache": False})
    @config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_autograd_remote_lazy_backward(self):
        # [FUNCTION DOC] Checks the remote autograd cache with lazily compiled backward: after a
        # forward only run the stats show one miss and no put; after reset another miss; only
        # after running backward is the entry put; after another reset and rerun the cache hits.
        # Gradients of the compiled and eager functions match.
        """
        Lazily compile the backward, and lazily save to cache
        """

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the cosine of the first argument plus the second.
            return a.cos() + b

        with PatchCaches():
            a = torch.randn(25, requires_grad=True)
            b = torch.randn(25, requires_grad=True)
            a2 = a.detach().clone().requires_grad_(True)
            b2 = b.detach().clone().requires_grad_(True)
            compiled_fn = torch.compile(fn, backend="inductor")
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(0, 0, 1))

            # Clear dynamo and run again. Should be a cache miss still, because backward hasn't run
            torch._dynamo.reset()
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(0, 0, 2))

            # Now let's run the backward
            fn(a, b).sum().backward()
            compiled_fn(a2, b2).sum().backward()
            self.assertEqual(a.grad, a2.grad)
            self.assertEqual(b.grad, b2.grad)
            self.assertEqual(global_stats.aot_autograd, Stats(1, 0, 2))

            # Clear dynamo and rerun everything, now there should be a cache hit
            torch._dynamo.reset()
            a = torch.randn(25, requires_grad=True)
            b = torch.randn(25, requires_grad=True)
            a2 = a.detach().clone().requires_grad_(True)
            b2 = b.detach().clone().requires_grad_(True)
            self.assertEqual(fn(a, b), compiled_fn(a2, b2))
            self.assertEqual(global_stats.aot_autograd, Stats(1, 1, 2))

            fn(a, b).sum().backward()
            compiled_fn(a2, b2).sum().backward()
            self.assertEqual(a.grad, a2.grad)
            self.assertEqual(b.grad, b2.grad)


class TestUtils(TestCase):
    # [CLASS DOC] Tests for torch._inductor.utils helpers.
    @config.patch({"fx_graph_remote_cache": False})
    def test_fresh_cache(self):
        # [FUNCTION DOC] Checks fresh_cache: inside the context PyCodeCache has no modules and
        # a compile works; two successive contexts use different cache directories and give
        # equal results.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the sum of the two inputs.
            return x + y

        a = torch.rand(10)
        b = torch.rand(10)

        with fresh_cache():
            self.assertEqual(len(PyCodeCache.modules), 0)
            res1 = torch.compile(fn)(a, b)
            cache_dir1 = cache_dir()

        torch._dynamo.reset()
        with fresh_cache():
            self.assertEqual(len(PyCodeCache.modules), 0)
            res2 = torch.compile(fn)(a, b)
            cache_dir2 = cache_dir()

        self.assertEqual(res1, res2)
        self.assertNotEqual(cache_dir1, cache_dir2)

    # This combination of settings exposed a bug where we cleared the
    # PyCodeCache disk artifacts while they were still needed:
    @requires_gpu_and_triton
    @config.patch(
        {
            "coordinate_descent_tuning": True,
            "force_disable_caches": True,
        }
    )
    def test_force_disable_coordinate_descent(self):
        # [FUNCTION DOC] Regression test (needs GPU and triton): with coordinate descent tuning
        # and all caches forcibly disabled, compiling a function must not delete PyCodeCache disk
        # artifacts that are still needed. It just compiles and runs the function.
        def fn():
            # [FUNCTION DOC] HELPER: Create random input and weight tensors on the GPU and return
            # a layer norm of their matrix product.
            inp = torch.randn(32, 50, 768, device=GPU_TYPE)
            weight = torch.randn(768, 768, device=GPU_TYPE)
            layer = torch.nn.LayerNorm(768, device=GPU_TYPE)
            return layer(inp @ weight)

        torch.compile(fn)()


class TestVecISACheckBuild(TestCase):
    # [CLASS DOC] Regression tests for the VecISA check_build probe: the probe subprocess is
    # bounded by a timeout, failure outcomes yield false, and the child can find libtorch
    # libraries on its loader path. They exercise the small helper methods directly so they do not
    # depend on the host compiler.
    # Regression tests for the VecISA.check_build dlopen probe behavior:
    # the parent must (1) bound the child with a timeout and return
    # False if it expires, and (2) make libtorch findable in the
    # child's loader path so the child can dlopen the probe .so
    # without importing torch.
    #
    # These target the small helpers (_probe_load, _build_probe_env)
    # directly so they don't depend on CppBuilder / CppTorchOptions
    # succeeding on the host, which would otherwise make the tests
    # platform-specific (the production path is fine; only the
    # CppTorchOptions construction is brittle to host compiler probes
    # when handed an off-arch VecISA like VecAVX2 on aarch64/macOS-arm64).

    def test_probe_load_returns_false_on_timeout(self):
        # [FUNCTION DOC] Replaces the subprocess module used by cpu_vec_isa with a fake whose run
        # records the timeout it got and raises a timeout error. Checks _probe_load returns false,
        # passes a 60 second timeout, and emits a warning saying the probe hung after 60s.
        from torch._inductor import cpu_vec_isa

        calls: list[Any] = []

        def fake_run(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Record the timeout keyword and raise
            # subprocess.TimeoutExpired.
            calls.append(kwargs.get("timeout"))
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs.get("timeout"))

        fake_subprocess = types.SimpleNamespace(
            run=fake_run,
            TimeoutExpired=subprocess.TimeoutExpired,
            CalledProcessError=subprocess.CalledProcessError,
            DEVNULL=subprocess.DEVNULL,
        )

        instance = cpu_vec_isa.VecAVX2()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch.object(cpu_vec_isa, "subprocess", fake_subprocess):
                result = instance._probe_load("/nonexistent/probe.so")

        self.assertFalse(result)
        self.assertEqual(calls, [60])
        self.assertTrue(
            any("hung after 60s" in str(w.message) for w in caught),
            msg=lambda msg: f"{msg}\nexpected timeout warning, got: {[str(w.message) for w in caught]}",
        )

    def test_probe_load_returns_false_on_called_process_error(self):
        # [FUNCTION DOC] Replaces the subprocess module used by cpu_vec_isa with a fake whose run
        # raises a called process error, and checks _probe_load returns false.
        from torch._inductor import cpu_vec_isa

        def fake_run(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Raise subprocess.CalledProcessError with return code 1.
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0])

        fake_subprocess = types.SimpleNamespace(
            run=fake_run,
            TimeoutExpired=subprocess.TimeoutExpired,
            CalledProcessError=subprocess.CalledProcessError,
            DEVNULL=subprocess.DEVNULL,
        )

        instance = cpu_vec_isa.VecAVX2()
        with mock.patch.object(cpu_vec_isa, "subprocess", fake_subprocess):
            result = instance._probe_load("/nonexistent/probe.so")

        self.assertFalse(result)

    def test_build_probe_env_prepends_torch_lib_dir_to_loader_path(self):
        # [FUNCTION DOC] Checks that the environment built for the probe child has the torch lib
        # directory as the first entry of LD_LIBRARY_PATH.
        from torch._inductor import cpu_vec_isa

        env = cpu_vec_isa.VecISA._build_probe_env()

        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        value = env.get("LD_LIBRARY_PATH", "")
        self.assertEqual(
            value.split(os.pathsep)[0],
            torch_lib,
            msg=lambda msg: f"{msg}\nLD_LIBRARY_PATH should be prepended with {torch_lib!r}, got {value!r}",
        )

    @unittest.skipUnless(sys.platform == "linux", "Linux loader semantics")
    def test_avx_py_load_falls_back_to_import_torch(self):
        # [FUNCTION DOC] Linux only. Compiles a tiny shared library that depends on libc10 without
        # any rpath, with the linker told to keep the dependency, and removes LD_LIBRARY_PATH from
        # the environment. If a cold load in a child already succeeds the test is skipped (as it
        # is if no C compiler or libc10 is found). Otherwise it runs the VecISA probe script, with
        # the library path substituted, and requires exit code zero, showing the probe falls back
        # to importing torch.
        # Wheels whose native deps only resolve once `import torch` has run
        # (e.g. ROCm wheels with DT_NEEDED refs on sonames that exist only as
        # already-loaded renamed bundles, see #189194) fail the cold dlopen,
        # so the probe child must retry after importing torch. Simulate that:
        # a probe .so that DT_NEEDEDs libc10.so with no rpath and no
        # LD_LIBRARY_PATH fails to load cold, and resolves only once
        # `import torch` maps libc10 into the child process.
        from torch._inductor import cpu_vec_isa

        compiler = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
        if compiler is None:
            raise unittest.SkipTest("no C compiler available")
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if not os.path.isfile(os.path.join(torch_lib, "libc10.so")):
            raise unittest.SkipTest("libc10.so not found in torch lib dir")

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "probe.c")
            with open(src, "w") as f:
                f.write("void __vec_isa_probe_dummy(void) {}\n")
            lib_path = os.path.join(tmp, "libvec_isa_probe_needs_c10.so")
            # --no-as-needed: the probe references no c10 symbol, so without it
            # the linker would drop the DT_NEEDED entry this test relies on
            cmd = [
                compiler,
                "-shared",
                "-fPIC",
                src,
                "-o",
                lib_path,
                "-Wl,--no-as-needed",
                f"-L{torch_lib}",
                "-lc10",
            ]
            subprocess.run(cmd, check=True)

            env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
            cold_load = f'from ctypes import cdll; cdll.LoadLibrary("{lib_path}")'
            cold = subprocess.run(
                [sys.executable, "-c", cold_load], env=env, stderr=subprocess.DEVNULL
            )
            if cold.returncode == 0:
                raise unittest.SkipTest("loader resolves libc10.so without torch")

            script = cpu_vec_isa.VecISA._avx_py_load.replace("__lib_path__", lib_path)
            probe = subprocess.run(
                [sys.executable, "-c", script], env=env, stderr=subprocess.PIPE
            )
            self.assertEqual(probe.returncode, 0, msg=probe.stderr.decode())


class TestCompilationEventLogging(TestCase):
    # [CLASS DOC] Tests that compilation events logged by dynamo carry correct local and remote
    # cache hit and miss counts for the inductor fx graph cache and the AOTAutograd cache under
    # various config combinations. Each test uses get_events to obtain a miss event and a hit
    # event and check to verify all eight counters.
    def reset(self):
        # [FUNCTION DOC] HELPER: Clear Dynamo and precompile caches, purge PyCodeCache, reset
        # dynamo and clear inductor caches.
        DynamoCache.clear()
        PrecompileContext.clear()
        PyCodeCache.cache_clear(purge=True)
        torch._dynamo.reset()
        clear_caches()

    @torch._dynamo.config.patch({"log_compilation_metrics": True})
    def get_events(self):
        # [FUNCTION DOC] HELPER: With compilation metrics logging on and mock caches installed,
        # compile a simple function, capture the one compilation event passed to the mocked log
        # function, reset caches, compile again and capture the second event. Returns both events
        # (first expected to be a cache miss, second a hit).
        """
        Helper to compile the same function twice and return the two compilation events
        passed to log_compilation_event() (the first event should correspond to a cache
        miss and the second to a cachd hit).
        """

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return a one element tuple with the product of the
            # two inputs.
            return (x * y,)

        a = torch.rand(5, 5)
        b = torch.rand(5, 5)

        compiled_fn = torch.compile(fn)
        with PatchCaches():
            with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
                self.assertEqual(fn(a, b), compiled_fn(a, b))
                self.assertTrue(len(log_event.call_args_list) == 1)
                event1 = log_event.call_args_list[0][0][0]

            self.reset()

            with mock.patch("torch._dynamo.utils.log_compilation_event") as log_event:
                self.assertEqual(fn(a, b), compiled_fn(a, b))
                self.assertTrue(len(log_event.call_args_list) == 1)
                event2 = log_event.call_args_list[0][0][0]

        return event1, event2

    def check(
        self,
        event,
        inductor_fx_remote_cache_hit_count=0,
        inductor_fx_remote_cache_miss_count=0,
        inductor_fx_local_cache_hit_count=0,
        inductor_fx_local_cache_miss_count=0,
        aotautograd_remote_cache_hit_count=0,
        aotautograd_remote_cache_miss_count=0,
        aotautograd_local_cache_hit_count=0,
        aotautograd_local_cache_miss_count=0,
    ):
        # [FUNCTION DOC] HELPER: Assert that the event's eight counters (inductor fx and
        # AOTAutograd, each local and remote, each hit and miss) equal the expected values, which
        # all default to zero.
        """
        Helper to check all cache hit/miss counts of a compilation event.
        """
        self.assertEqual(
            event.inductor_fx_remote_cache_hit_count, inductor_fx_remote_cache_hit_count
        )
        self.assertEqual(
            event.inductor_fx_remote_cache_miss_count,
            inductor_fx_remote_cache_miss_count,
        )
        self.assertEqual(
            event.inductor_fx_local_cache_hit_count, inductor_fx_local_cache_hit_count
        )
        self.assertEqual(
            event.inductor_fx_local_cache_miss_count, inductor_fx_local_cache_miss_count
        )
        self.assertEqual(
            event.aotautograd_remote_cache_hit_count, aotautograd_remote_cache_hit_count
        )
        self.assertEqual(
            event.aotautograd_remote_cache_miss_count,
            aotautograd_remote_cache_miss_count,
        )
        self.assertEqual(
            event.aotautograd_local_cache_hit_count, aotautograd_local_cache_hit_count
        )
        self.assertEqual(
            event.aotautograd_local_cache_miss_count, aotautograd_local_cache_miss_count
        )

    @torch._inductor.config.patch({"fx_graph_cache": True})
    @torch._inductor.config.patch({"fx_graph_remote_cache": False})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": False})
    def test_log_event_inductor_local(self):
        # [FUNCTION DOC] Only the local fx graph cache is on: first event shows one local miss,
        # second one local hit.
        """
        Only the local FXGraph cache is enabled.
        """
        event1, event2 = self.get_events()
        self.check(event1, inductor_fx_local_cache_miss_count=1)
        self.check(event2, inductor_fx_local_cache_hit_count=1)

    @torch._inductor.config.patch({"fx_graph_cache": False})
    @torch._inductor.config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": False})
    def test_log_event_inductor_remote(self):
        # [FUNCTION DOC] Only the remote fx graph cache is on: first event shows one remote miss,
        # second one remote hit.
        """
        Only the remote FXGraph cache is enabled.
        """
        event1, event2 = self.get_events()
        self.check(event1, inductor_fx_remote_cache_miss_count=1)
        self.check(event2, inductor_fx_remote_cache_hit_count=1)

    @torch._inductor.config.patch({"fx_graph_cache": True})
    @torch._inductor.config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": False})
    def test_log_event_inductor_both(self):
        # [FUNCTION DOC] Local and remote fx graph caches both on: first event has one local and
        # one remote miss, second has only one local hit.
        """
        Both the local and remote FXGraph caches are enabled.
        """
        event1, event2 = self.get_events()
        self.check(
            event1,
            inductor_fx_local_cache_miss_count=1,
            inductor_fx_remote_cache_miss_count=1,
        )
        self.check(event2, inductor_fx_local_cache_hit_count=1)

    @torch._inductor.config.patch({"fx_graph_cache": True})
    @torch._inductor.config.patch({"fx_graph_remote_cache": False})
    @torch._functorch.config.patch({"enable_autograd_cache": True})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": False})
    def test_log_event_autograd_local(self):
        # [FUNCTION DOC] Local AOTAutograd cache on (with the local fx cache): first event has one
        # AOTAutograd local miss and one fx local miss, second has one of each hit.
        """
        Enable the local autograd cache (which requires the FXGraph cache).
        """
        event1, event2 = self.get_events()
        self.check(
            event1,
            aotautograd_local_cache_miss_count=1,
            # AOTAutograd cache leverages the FXGraph cache.
            inductor_fx_local_cache_miss_count=1,
        )
        self.check(
            event2,
            aotautograd_local_cache_hit_count=1,
            # AOTAutograd cache leverages the FXGraph cache.
            inductor_fx_local_cache_hit_count=1,
        )

    @torch._inductor.config.patch({"fx_graph_cache": False})
    @torch._inductor.config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": False})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_log_event_autograd_remote(self):
        # [FUNCTION DOC] Remote AOTAutograd cache on with the remote fx cache: first event has
        # one AOTAutograd remote miss and one fx remote miss; second has remote hits for both
        # plus one fx local miss (the local fx cache is always consulted, noted in the test as
        # arguably a bug).
        """
        Enable the remote autograd cache (which requires the remote FXGraph cache).
        """
        event1, event2 = self.get_events()
        self.check(
            event1,
            aotautograd_remote_cache_miss_count=1,
            # AOTAutograd cache leverages the FXGraph cache.
            inductor_fx_remote_cache_miss_count=1,
        )
        self.check(
            event2,
            aotautograd_remote_cache_hit_count=1,
            # AOTAutograd cache leverages the FXGraph cache.
            inductor_fx_remote_cache_hit_count=1,
            # AOTAutograd cache always leverages the local FXGraph cache. That's
            # arguably a bug because the local wouldn't be populated if disabled.
            inductor_fx_local_cache_miss_count=1,
        )

    @torch._inductor.config.patch({"fx_graph_cache": True})
    @torch._inductor.config.patch({"fx_graph_remote_cache": True})
    @torch._functorch.config.patch({"enable_autograd_cache": True})
    @torch._functorch.config.patch({"enable_remote_autograd_cache": True})
    def test_log_event_all_caches(self):
        # [FUNCTION DOC] All four caches on: first event has one miss in each of the four counters
        # (AOTAutograd and fx, local and remote); second has one local hit for each of AOTAutograd
        # and fx.
        """
        Enable all caches.
        """
        event1, event2 = self.get_events()
        self.check(
            event1,
            aotautograd_local_cache_miss_count=1,
            aotautograd_remote_cache_miss_count=1,
            inductor_fx_local_cache_miss_count=1,
            inductor_fx_remote_cache_miss_count=1,
        )
        self.check(
            event2,
            aotautograd_local_cache_hit_count=1,
            inductor_fx_local_cache_hit_count=1,
        )


class TestAutotuneCacheExtraOptions(TestCase):
    # [CLASS DOC] Unit tests for preserving the extra_options field of triton configs (used by
    # third-party backends) when saving to and loading from the autotune cache, covering
    # _load_cached_autotuning and AutotuneCache.save.
    """
    Unit tests for extra_options preservation in _load_cached_autotuning().

    The extra_options field allows third-party backends to store custom tuned
    options alongside the standard Triton config. These tests verify that
    extra_options is correctly preserved when saving and loading from cache.
    """

    @requires_triton()
    def test_load_cached_autotuning_preserves_extra_options_with_coordesc(self):
        # [FUNCTION DOC] Requires triton. Loading a cached entry flagged as found by coordinate
        # descent (with an empty config list) returns a config that keeps the cached extra_options
        # mapping and the found_by_coordesc flag.
        """
        Test that extra_options is preserved when loading a config that was
        found via coordinate descent tuning.
        """
        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        # Simulate a cached config with extra_options and found_by_coordesc=True
        best_config = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 2,
            "configs_hash": "test_hash",
            "found_by_coordesc": True,
            "extra_options": {"custom_option": "value", "another_option": 42},
        }

        # Empty configs list since coordesc path creates a new config
        configs = []
        inductor_meta = {"coordinate_descent_tuning": True}

        result = _load_cached_autotuning(
            best_config.copy(), "test_hash", configs, inductor_meta
        )

        self.assertIsNotNone(result)
        self.assertEqual(
            result.extra_options, {"custom_option": "value", "another_option": 42}
        )
        self.assertTrue(result.found_by_coordesc)

    @requires_triton()
    def test_load_cached_autotuning_preserves_extra_options_with_matched_config(self):
        # [FUNCTION DOC] Requires triton. Loading a cached entry that matches a config in the
        # provided list returns that same config object and sets its extra_options from the
        # cached data.
        """
        Test that extra_options is preserved when loading by matching an
        existing config from the configs list.
        """
        from triton import Config

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        # Create a config that matches what's in the cache
        original_config = Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        )
        configs = [original_config]

        # Simulate a cached config with extra_options
        best_config = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 2,
            "configs_hash": "test_hash",
            "extra_options": {"backend_specific": "option"},
        }

        inductor_meta = {"coordinate_descent_tuning": False}

        result = _load_cached_autotuning(
            best_config.copy(), "test_hash", configs, inductor_meta
        )

        self.assertIsNotNone(result)
        self.assertEqual(result.extra_options, {"backend_specific": "option"})
        # The returned config should be the matched one
        self.assertIs(result, original_config)

    @requires_triton()
    def test_load_cached_autotuning_handles_none_extra_options(self):
        # [FUNCTION DOC] Requires triton. Loading a cached entry without extra_options returns the
        # matched config with extra_options equal to None.
        """
        Test that _load_cached_autotuning handles missing extra_options gracefully
        (returns None for the attribute).
        """
        from triton import Config

        from torch._inductor.runtime.autotune_cache import _load_cached_autotuning

        # Create a config that matches what's in the cache
        original_config = Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        )
        configs = [original_config]

        # Simulate a cached config WITHOUT extra_options
        best_config = {
            "BLOCK_M": 64,
            "BLOCK_N": 64,
            "num_warps": 4,
            "num_stages": 2,
            "configs_hash": "test_hash",
        }

        inductor_meta = {"coordinate_descent_tuning": False}

        result = _load_cached_autotuning(
            best_config.copy(), "test_hash", configs, inductor_meta
        )

        self.assertIsNotNone(result)
        # extra_options should be None when not in cache
        self.assertIsNone(result.extra_options)

    @requires_triton()
    def test_autotune_cache_save_includes_extra_options(self):
        # [FUNCTION DOC] Requires triton. Builds an AutotuneCache without running its constructor,
        # with a mocked local backend and no remote cache, saves a config carrying extra_options,
        # and checks the single put to the local backend has the extra_options in the saved data.
        """
        Test that AutotuneCache.save() includes extra_options in the saved data.
        """
        from unittest.mock import MagicMock

        from triton import Config

        from torch._inductor.runtime.autotune_cache import AutotuneCache

        # Create a config with extra_options
        config_with_extra = Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        )
        config_with_extra.extra_options = {"custom_key": "custom_value"}

        # Create an AutotuneCache instance with mocked caches
        cache = AutotuneCache.__new__(AutotuneCache)
        cache.configs_hash = "test_hash"

        # Mock the local cache to capture what's being saved
        mock_local_backend = MagicMock()
        cache.local_cache = (mock_local_backend, "test_key")
        cache.remote_cache = None

        with mock.patch("torch._inductor.runtime.autotune_cache.AutotuneCacheBundler"):
            cache.save(config_with_extra, time_taken_ns=1000000)

        # Verify extra_options was included in the saved data
        mock_local_backend.put.assert_called_once()
        saved_data = mock_local_backend.put.call_args[0][1]
        self.assertIn("extra_options", saved_data)
        self.assertEqual(saved_data["extra_options"], {"custom_key": "custom_value"})

    @requires_triton()
    def test_autotune_cache_save_omits_extra_options_when_none(self):
        # [FUNCTION DOC] Requires triton. Same setup as the previous save test but with a config
        # lacking extra_options; checks the saved data does not contain the extra_options entry.
        """
        Test that AutotuneCache.save() does not include extra_options when it's None.
        """
        from unittest.mock import MagicMock

        from triton import Config

        from torch._inductor.runtime.autotune_cache import AutotuneCache

        # Create a config without extra_options
        config_without_extra = Config(
            {"BLOCK_M": 64, "BLOCK_N": 64},
            num_warps=4,
            num_stages=2,
        )

        # Create an AutotuneCache instance with mocked caches
        cache = AutotuneCache.__new__(AutotuneCache)
        cache.configs_hash = "test_hash"

        # Mock the local cache to capture what's being saved
        mock_local_backend = MagicMock()
        cache.local_cache = (mock_local_backend, "test_key")
        cache.remote_cache = None

        with mock.patch("torch._inductor.runtime.autotune_cache.AutotuneCacheBundler"):
            cache.save(config_without_extra, time_taken_ns=1000000)

        # Verify extra_options was NOT included in the saved data
        mock_local_backend.put.assert_called_once()
        saved_data = mock_local_backend.put.call_args[0][1]
        self.assertNotIn("extra_options", saved_data)


if __name__ == "__main__":
    run_tests()
