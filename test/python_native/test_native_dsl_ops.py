# Owner(s): ["module: dsl-native-ops"]
# [MODULE DOC] Tests for the torch._native DSL ops framework: consistent public interface of the
# triton, cutedsl and helion utility modules, import hygiene (no DSL runtimes or external packaging
# pulled in by importing torch), environment variable switches for disabling native JIT and
# skipping version checks, version parsing, registry library caching, deregistration, DSL registry
# caching and protocol validation. Several checks run in fresh subprocesses.

import importlib.util
import os
import subprocess
import sys
import textwrap
import uuid
from unittest.mock import patch

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    skipIfNoCuteDSL,
    TestCase,
)


def _subprocess_lastline(script, env=None):
    # [FUNCTION DOC] HELPER: Run the given script text in a new Python interpreter, with the
    # directory of this test file as working directory, and return the last line of its standard
    # output (stripped of surrounding whitespace). Raises CalledProcessError on a nonzero exit.
    # KNOWN ISSUE: the env argument is accepted but never used, so no custom environment is
    # applied.
    """Run script in a fresh interpreter and return the last line of stdout."""
    result = subprocess.check_output(
        [sys.executable, "-c", script],
        cwd=os.path.dirname(os.path.realpath(__file__)),
        text=True,
    ).strip()
    return result.rsplit("\n", 1)[-1]


def _import_module_directly(module_name, file_name):
    # [FUNCTION DOC] HELPER: Import a file from the torch/_native directory of the repository under
    # the given module name without running the package initializers. Builds a spec from the file
    # location, creates the module, registers it in the module table (a side effect that replaces
    # any existing entry), executes it and returns the module.
    """Import a module directly without triggering package imports."""
    test_dir = os.path.dirname(os.path.abspath(__file__))
    pytorch_root = os.path.dirname(os.path.dirname(test_dir))
    module_path = os.path.join(pytorch_root, "torch", "_native", file_name)

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class TestNativeDSLOps(TestCase):
    # [CLASS DOC] Test case for the native DSL ops framework. setUp clears cached environment and
    # version check functions of the common and per-DSL utility modules so each test sees fresh
    # environment variable state. Parametrized tests are instantiated at the end of the module.
    """Tests for the torch._native DSL ops framework."""

    def setUp(self):
        # [FUNCTION DOC] Call the parent setUp, record which modules and cached functions need
        # clearing (common, triton, cutedsl and helion utilities), then clear them.
        """Clear all caches before each test to ensure test isolation."""
        super().setUp()
        self._cache_functions_to_clear = [
            (
                "torch._native.common_utils",
                ["check_native_jit_disabled", "check_native_version_skip"],
            ),
            (
                "torch._native.triton_utils",
                [
                    "_version_is_sufficient",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.cutedsl_utils",
                [
                    "_version_is_ok",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
            (
                "torch._native.helion_utils",
                [
                    "_version_is_sufficient",
                    "check_native_jit_disabled",
                    "check_native_version_skip",
                ],
            ),
        ]
        self._clear_function_caches()

    def _clear_function_caches(self):
        # [FUNCTION DOC] HELPER: For each recorded module, import it and call cache_clear on each
        # listed function that exists; import or attribute errors are silently ignored.
        """Helper method to clear function caches with error handling."""
        for module_name, function_names in self._cache_functions_to_clear:
            try:
                module = __import__(module_name, fromlist=function_names)
                for func_name in function_names:
                    if hasattr(module, func_name):
                        getattr(module, func_name).cache_clear()
            except (AttributeError, ImportError):
                # Some functions might not exist or be cached, ignore errors
                pass

    def test_consistent_helper_interface(self):
        # [FUNCTION DOC] Discovers the registered DSL names (falling back to triton, cutedsl and
        # helion), imports each utility module directly and asserts each exposes the four required
        # public functions (availability, version, register and deregister overrides), all
        # callable, and that all modules have identical public name sets. Also checks that
        # availability returns a bool and version returns None or a Version. Uses subtests.
        """Test all registered DSL utils expose consistent public APIs."""
        from torch.testing._internal.common_utils import get_all_dsls

        # Automatically discover all registered DSLs
        dsl_names = get_all_dsls()
        if not dsl_names:
            # Fallback to hardcoded list if registry not available
            dsl_names = ["triton", "cutedsl", "helion"]

        modules_info = [
            (f"{dsl}_utils.py", f"torch._native.{dsl}_utils") for dsl in dsl_names
        ]

        # Import modules directly to avoid dependency issues
        modules = {}
        for file_name, module_name in modules_info:
            modules[module_name] = _import_module_directly(module_name, file_name)

        required_methods = {
            "runtime_available",
            "runtime_version",
            "register_op_override",
            "deregister_op_overrides",
        }

        # Test each module has required methods and they're callable
        public_apis = {}
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="required_methods"):
                public = {name for name in dir(mod) if not name.startswith("_")}
                public_apis[module_name] = public

                self.assertTrue(
                    required_methods <= public,
                    lambda msg: f"{msg}\n{module_name} missing: {required_methods - public}",
                )

                for method_name in required_methods:
                    with self.subTest(module=module_name, method=method_name):
                        self.assertTrue(callable(getattr(mod, method_name)))

        # Test modules expose identical public APIs
        api_sets = list(public_apis.values())
        if len(api_sets) > 1:
            for i, api_set in enumerate(api_sets[1:], 1):
                self.assertEqual(
                    api_sets[0],
                    api_set,
                    lambda msg: f"{msg}\nModule {i} should have identical public API to module 0",
                )

        # Test runtime functions return expected types
        for module_name, mod in modules.items():
            with self.subTest(module=module_name, test="runtime_functions"):
                # runtime_available should return bool
                self.assertIsInstance(mod.runtime_available(), bool)

                # runtime_version should return Version or None
                ver = mod.runtime_version()
                if ver is not None:
                    from torch._vendor.packaging.version import Version

                    self.assertIsInstance(ver, Version)

    def test_no_dsl_imports_after_import_torch(self):
        # [FUNCTION DOC] Runs a fresh interpreter that imports torch and prints which of triton,
        # cutlass, tvm_ffi and helion are already in the module table; asserts the printed list is
        # empty.
        """import torch must not transitively import DSL runtimes.

        Note: cuda.bindings may appear because importlib.util.find_spec on
        nested modules (e.g. cuda.bindings.driver) imports parent packages
        as a side-effect.  We check only the primary DSL runtimes here.
        """
        script = textwrap.dedent("""\
            import sys
            import torch
            dsl_modules = ["triton", "cutlass", "tvm_ffi", "helion"]
            leaked = [m for m in dsl_modules if m in sys.modules]
            print(repr(leaked))
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(
            result,
            "[]",
            lambda msg: f"{msg}\nDSL modules leaked on import torch: {result}",
        )

    def test_no_external_packaging_dependency(self):
        # [FUNCTION DOC] Runs a fresh interpreter that removes and blocks the external packaging
        # package with an import hook, then imports torch and prints OK; asserts the output is OK,
        # so torch._native relies on the vendored copy only.
        """torch._native must not import the external `packaging` package.

        It should use the vendored copy at torch._vendor.packaging instead.
        This guards against ModuleNotFoundError in environments where the
        external `packaging` is not installed (e.g. torchvision Windows CI).
        """
        script = textwrap.dedent("""\
            import sys
            # Remove external packaging from sys.modules if already loaded
            for mod_name in list(sys.modules):
                if mod_name == "packaging" or mod_name.startswith("packaging."):
                    del sys.modules[mod_name]
            # Block external packaging from being imported
            import importlib.abc
            import importlib.machinery
            class BlockPackaging(importlib.abc.MetaPathFinder):
                def find_module(self, fullname, path=None):
                    if fullname == "packaging" or fullname.startswith("packaging."):
                        return self
                def load_module(self, fullname):
                    raise ImportError(f"External {fullname} is blocked")
            sys.meta_path.insert(0, BlockPackaging())
            import torch
            print("OK")
        """)
        result = _subprocess_lastline(script)
        self.assertEqual(result, "OK")

    @parametrize("env_value, expected", [(None, False), ("1", True)])
    def test_check_native_jit_disabled_environment_variable(self, env_value, expected):
        # [FUNCTION DOC] Parametrized over an unset variable (expect False) and the value 1 (expect
        # True). Sets or removes the native JIT disable variable in the real environment, clears
        # the cache of the check function, asserts its result, and always removes the variable
        # afterwards.
        """Test TORCH_DISABLE_NATIVE_JIT environment variable behavior."""
        from torch._native.common_utils import check_native_jit_disabled

        if env_value is None:
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)
        else:
            os.environ["TORCH_DISABLE_NATIVE_JIT"] = env_value

        try:
            # Clear cache so function re-reads environment variable
            check_native_jit_disabled.cache_clear()
            self.assertEqual(check_native_jit_disabled(), expected)
        finally:
            # Clean up environment variable
            os.environ.pop("TORCH_DISABLE_NATIVE_JIT", None)

    def test_unavailable_reason_missing(self):
        # [FUNCTION DOC] Asserts the unavailable-reason helper returns a non-None string mentioning
        # the package name when given a nonexistent package.
        """Nonexistent package -> _unavailable_reason returns a string."""
        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )
        reason = common_utils._unavailable_reason(
            [("nonexistent_pkg_xyz", "nonexistent_pkg_xyz")]
        )
        self.assertIsNotNone(reason)
        self.assertIn("nonexistent_pkg_xyz", reason)

    def test_available_version_parsing(self):
        # [FUNCTION DOC] Asserts the available-version helper returns a Version for a real package
        # and, with the metadata version lookup patched, parses pre-release, post-release, alpha
        # and standard strings into equal Version objects and gives None for an invalid string.
        # Uses subtests.
        """Test _available_version parses various version formats and handles invalid ones."""
        from torch._vendor.packaging.version import Version

        common_utils = _import_module_directly(
            "torch._native.common_utils", "common_utils.py"
        )

        # Test with real package that has clean version
        ver = common_utils._available_version("typing_extensions")
        self.assertIsInstance(ver, Version)

        # Test various version format scenarios
        version_scenarios = [
            ("0.7.0rc1", Version("0.7.0rc1"), "pre-release version"),
            ("3.1.0.post1", Version("3.1.0.post1"), "post-release version"),
            ("2.4.0a1", Version("2.4.0a1"), "alpha version"),
            ("1.2.3", Version("1.2.3"), "standard version"),
            ("abc", None, "invalid version string"),
        ]

        for version_str, expected_result, description in version_scenarios:
            with self.subTest(version=version_str, scenario=description):
                with patch("importlib.metadata.version", return_value=version_str):
                    result = common_utils._available_version("fake_package")
                    self.assertEqual(
                        result,
                        expected_result,
                        lambda msg: f"{msg}\n_available_version({version_str!r}) = {result}",
                    )

    def test_registry_mechanics(self):
        # [FUNCTION DOC] Saves the registry library cache and filter state, removes the CPU and
        # CUDA entries, then asserts the get-or-create library helper returns a Library, the same
        # object for repeated CPU requests and a different object for CUDA. Restores the saved
        # cache and filter state in a finally block.
        """_get_or_create_library caches Library instances per dispatch_key."""
        import torch._native.registry as registry
        import torch.library

        # Save original state for restoration
        original_libs = dict(registry._libs)
        original_filter_state = (
            set(registry._filter_state._dsl_names),
            set(registry._filter_state._op_symbols),
            set(registry._filter_state._dispatch_keys),
        )

        try:
            cpu_key = ("_native", "CPU")
            cuda_key = ("_native", "CUDA")
            registry._libs.pop(cpu_key, None)
            registry._libs.pop(cuda_key, None)

            lib1 = registry._get_or_create_library("CPU")
            self.assertIsInstance(lib1, torch.library.Library)
            lib2 = registry._get_or_create_library("CPU")
            self.assertIs(lib1, lib2, "should return cached instance")

            # Different dispatch key -> different Library
            lib3 = registry._get_or_create_library("CUDA")
            self.assertIsNot(lib1, lib3)

            # cleanup
            registry._libs.pop(cpu_key, None)
            registry._libs.pop(cuda_key, None)
        finally:
            # Restore original registry state
            registry._libs.clear()
            registry._libs.update(original_libs)

            # Restore filter state
            filter_state = registry._filter_state
            filter_state._dsl_names.clear()
            filter_state._op_symbols.clear()
            filter_state._dispatch_keys.clear()
            filter_state._dsl_names.update(original_filter_state[0])
            filter_state._op_symbols.update(original_filter_state[1])
            filter_state._dispatch_keys.update(original_filter_state[2])

    def test_deregister_op_overrides_functionality(self):
        # [FUNCTION DOC] For the triton, cutedsl and helion utility modules (imported directly),
        # asserts a callable deregister function exists and that calling it with no overrides
        # registered raises nothing. The registry filter state is saved and restored afterwards.
        """Test deregister_op_overrides methods exist, are callable, and work correctly."""
        modules_to_test = [
            ("triton_utils.py", "torch._native.triton_utils"),
            ("cutedsl_utils.py", "torch._native.cutedsl_utils"),
            ("helion_utils.py", "torch._native.helion_utils"),
        ]

        # Use the preserve_filter_state context manager pattern
        from torch._native.registry import _filter_state

        original_filter_state = (
            set(_filter_state._dsl_names),
            set(_filter_state._op_symbols),
            set(_filter_state._dispatch_keys),
        )

        try:
            for file_name, module_name in modules_to_test:
                with self.subTest(module=module_name):
                    mod = _import_module_directly(module_name, file_name)

                    # Test method exists and is callable
                    self.assertTrue(hasattr(mod, "deregister_op_overrides"))
                    self.assertTrue(callable(mod.deregister_op_overrides))

                    # Test method can be called without error (should be no-op when no overrides registered)
                    try:
                        mod.deregister_op_overrides()
                    except Exception as e:
                        self.fail(
                            f"deregister_op_overrides on {module_name} raised exception: {e}"
                        )
        finally:
            # Restore original filter state
            _filter_state._dsl_names.clear()
            _filter_state._op_symbols.clear()
            _filter_state._dispatch_keys.clear()
            _filter_state._dsl_names.update(original_filter_state[0])
            _filter_state._op_symbols.update(original_filter_state[1])
            _filter_state._dispatch_keys.update(original_filter_state[2])

    def test_register_op_skips_when_jit_disabled(self):
        # [FUNCTION DOC] With the disable-native-JIT variable patched to 1 and caches cleared,
        # asserts the disable check is true, then registers an override through each of the three
        # utility modules with the internal implementation functions mocked and asserts none was
        # called.
        """register_op_override does not call through when TORCH_DISABLE_NATIVE_JIT=1."""
        from torch._native import cutedsl_utils, helion_utils, triton_utils

        # Test the actual environment variable behavior to ensure it works
        # Set TORCH_DISABLE_NATIVE_JIT=1 and clear caches
        with patch.dict(os.environ, {"TORCH_DISABLE_NATIVE_JIT": "1"}):
            # Import and clear caches for both modules
            from torch._native.common_utils import check_native_jit_disabled

            check_native_jit_disabled.cache_clear()

            # Import functions from each module and clear their caches too
            triton_utils.check_native_jit_disabled.cache_clear()
            cutedsl_utils.check_native_jit_disabled.cache_clear()
            helion_utils.check_native_jit_disabled.cache_clear()

            # Verify the function returns True
            self.assertTrue(check_native_jit_disabled())

            with (
                patch.object(triton_utils, "_register_op_override_impl") as triton_mock,
                patch.object(
                    cutedsl_utils, "_register_op_override_impl"
                ) as cutedsl_mock,
                patch.object(helion_utils, "_register_op_override_impl") as helion_mock,
            ):
                # Use a unique operation name
                unique_op = f"test_jit_disabled_{uuid.uuid4().hex[:8]}.Tensor"
                triton_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                cutedsl_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                helion_utils.register_op_override(
                    "aten", unique_op, "CPU", lambda *a, **k: True, lambda: None
                )
                self.assertEqual(triton_mock.call_count, 0)
                self.assertEqual(cutedsl_mock.call_count, 0)
                self.assertEqual(helion_mock.call_count, 0)

    def test_helion_availability_requires_supported_backend_and_version(self):
        # [FUNCTION DOC] Checks helion availability: with the backend variable set to metal the
        # runtime is not available; with the runtime check patched to report available and version
        # 1.2.0, runtime is available and version sufficient; with version 1.0.0, the runtime is
        # available but the version is not sufficient. Caches are cleared between cases.
        from torch._native import helion_utils
        from torch._vendor.packaging.version import Version

        with patch.dict(os.environ, {"HELION_BACKEND": "metal"}):
            helion_utils._check_runtime_available.cache_clear()
            helion_utils._version_is_sufficient.cache_clear()
            self.assertFalse(helion_utils.runtime_available())

        helion_utils._check_runtime_available.cache_clear()
        helion_utils._version_is_sufficient.cache_clear()
        with patch.object(
            helion_utils,
            "_check_runtime_available",
            return_value=(True, Version("1.2.0")),
        ):
            self.assertTrue(helion_utils.runtime_available())
            self.assertTrue(helion_utils._version_is_sufficient())

        helion_utils._check_runtime_available.cache_clear()
        helion_utils._version_is_sufficient.cache_clear()
        with patch.object(
            helion_utils,
            "_check_runtime_available",
            return_value=(True, Version("1.0.0")),
        ):
            self.assertTrue(helion_utils.runtime_available())
            self.assertFalse(helion_utils._version_is_sufficient())

    def test_version_skip_env_var_overrides(self):
        # [FUNCTION DOC] With the skip-version-check variable set to 1 and caches cleared, for each
        # DSL utility module patches the runtime check to return a fake old version and the impl
        # function with a mock, registers an override and asserts the impl was called once.
        """TORCH_NATIVE_SKIP_VERSION_CHECK=1 allows non-blessed versions."""
        from torch._vendor.packaging.version import Version

        fake_version = Version("1.0.0")

        # Set the environment variable and clear caches
        with patch.dict(os.environ, {"TORCH_NATIVE_SKIP_VERSION_CHECK": "1"}):
            # Import fresh modules to avoid cached state
            from torch._native import cutedsl_utils, helion_utils, triton_utils
            from torch._native.common_utils import check_native_version_skip

            # Clear all relevant caches to ensure clean state
            check_native_version_skip.cache_clear()

            utils = (triton_utils, cutedsl_utils, helion_utils)
            op_name = f"test_version_skip_{uuid.uuid4().hex[:8]}.Tensor"

            for module in utils:
                # Clear cached lookups so the patched runtime takes effect.
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if hasattr(attr, "cache_clear"):
                        attr.cache_clear()

                with (
                    patch.object(
                        module,
                        "_check_runtime_available",
                        return_value=(True, fake_version),
                    ),
                    patch.object(module, "_register_op_override_impl") as mock,
                ):
                    module.register_op_override(
                        "aten", op_name, "CPU", lambda *a, **k: True, lambda: None
                    )
                    self.assertEqual(
                        mock.call_count,
                        1,
                        f"{module.__name__}: impl not called under skip flag",
                    )

    @parametrize("env_value, expected", [(None, False), ("1", True)])
    def test_check_native_version_skip_environment_variable(self, env_value, expected):
        # [FUNCTION DOC] Parametrized over an unset variable (expect False) and the value 1 (expect
        # True). Sets or removes the skip-version-check variable, clears the cache, asserts the
        # result and always removes the variable afterwards.
        """Test TORCH_NATIVE_SKIP_VERSION_CHECK environment variable behavior."""
        from torch._native.common_utils import check_native_version_skip

        if env_value is None:
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)
        else:
            os.environ["TORCH_NATIVE_SKIP_VERSION_CHECK"] = env_value

        try:
            # Clear cache so function re-reads environment variable
            check_native_version_skip.cache_clear()
            self.assertEqual(check_native_version_skip(), expected)
        finally:
            # Clean up environment variable
            os.environ.pop("TORCH_NATIVE_SKIP_VERSION_CHECK", None)

    def test_dsl_registry_functionality(self):
        # [FUNCTION DOC] Asserts the common_utils DSL query helpers: the list of all DSLs is a list
        # containing triton, cutedsl and helion, the available list is a list of members of it, and
        # the availability query returns a bool that is true for every DSL in the available list.
        """Test that DSL registry works correctly"""
        from torch.testing._internal.common_utils import (
            get_all_dsls,
            get_available_dsls,
            is_dsl_available,
        )

        # Test registry returns expected DSLs
        all_dsls = get_all_dsls()
        self.assertIsInstance(all_dsls, list)
        self.assertIn("triton", all_dsls)
        self.assertIn("cutedsl", all_dsls)
        self.assertIn("helion", all_dsls)

        # Test available DSLs are subset of all DSLs
        available_dsls = get_available_dsls()
        self.assertIsInstance(available_dsls, list)
        for dsl in available_dsls:
            self.assertIn(dsl, all_dsls)

        # Test availability check function
        for dsl in all_dsls:
            availability = is_dsl_available(dsl)
            self.assertIsInstance(availability, bool)
            # If DSL is in available list, it should return True
            if dsl in available_dsls:
                self.assertTrue(availability)

    def test_dsl_test_helpers(self):
        # [FUNCTION DOC] Asserts the DSL skip decorators from common_utils are callable and that
        # the two dynamic decorator factories can be called with a DSL name and return callables;
        # fails the test with the error message if they raise.
        """Test that DSL test helper decorators work"""
        from torch.testing._internal.common_utils import (
            skipIfDSLUnavailable,
            skipIfNoHelionDSL,
            skipIfNoTritonDSL,
            skipUnlessDSLAvailable,
        )

        # Test that decorators are callable
        self.assertTrue(callable(skipIfNoTritonDSL))
        self.assertTrue(callable(skipIfNoCuteDSL))
        self.assertTrue(callable(skipIfNoHelionDSL))
        self.assertTrue(callable(skipIfDSLUnavailable))
        self.assertTrue(callable(skipUnlessDSLAvailable))

        # Test dynamic decorators can be called
        try:
            decorator1 = skipIfDSLUnavailable("nonexistent_dsl")
            decorator2 = skipUnlessDSLAvailable("triton")
            self.assertTrue(callable(decorator1))
            self.assertTrue(callable(decorator2))
        except Exception as e:
            self.fail(f"Dynamic DSL decorators failed: {e}")

    def test_cache_invalidation_after_re_registration(self):
        # [FUNCTION DOC] Uses a fresh registry and two mock DSLs, the first unavailable and the
        # second available. Registers the first, queries availability and the available list
        # (unavailable, not listed), re-registers the name with the second and asserts the cached
        # results were invalidated: now available and listed.
        """Test that caches are properly invalidated when DSLs are re-registered"""
        from unittest.mock import Mock

        from torch._native.dsl_registry import DSLRegistry

        # Create a fresh registry for this test
        registry = DSLRegistry()

        # Create mock DSL modules
        mock_dsl_1 = Mock()
        mock_dsl_1.runtime_available.return_value = False  # Initially unavailable
        mock_dsl_1.runtime_version.return_value = None

        mock_dsl_2 = Mock()
        mock_dsl_2.runtime_available.return_value = True  # Available
        mock_dsl_2.runtime_version.return_value = None

        # Register first DSL and cache results
        registry.register_dsl("test_cache_dsl", mock_dsl_1)
        initial_available = registry.is_dsl_available("test_cache_dsl")
        initial_list = registry.list_available_dsls()

        self.assertFalse(initial_available)
        self.assertNotIn("test_cache_dsl", initial_list)

        # Re-register with different module that is available
        registry.register_dsl("test_cache_dsl", mock_dsl_2)

        # Verify cache was invalidated and new results are returned
        new_available = registry.is_dsl_available("test_cache_dsl")
        new_list = registry.list_available_dsls()

        self.assertTrue(
            new_available, "Cache should be invalidated and return new result"
        )
        self.assertIn(
            "test_cache_dsl",
            new_list,
            "Available DSLs list should reflect new registration",
        )

    def test_incomplete_protocol_implementation(self):
        # [FUNCTION DOC] Registering an object lacking required protocol methods raises TypeError
        # whose message mentions missing required methods and names runtime_version and
        # register_op_override; asserts the DSL is not listed afterwards. Uses a fresh registry.
        """Test that registration fails when module doesn't implement required protocol methods"""
        from torch._native.dsl_registry import DSLRegistry

        # Create a fresh registry for this test
        registry = DSLRegistry()

        # Create an object missing required protocol methods (not using Mock)
        class IncompleteModule:
            # [CLASS DOC] HELPER: Deliberately incomplete DSL module that only provides the
            # availability method; used to trigger the registry protocol check.
            def runtime_available(self):
                # [FUNCTION DOC] HELPER: Always reports the runtime as available by returning True.
                return True

            # Missing: runtime_version, register_op_override, deregister_op_overrides

        incomplete_module = IncompleteModule()

        # Attempt to register should raise TypeError due to missing methods
        with self.assertRaises(TypeError) as cm:
            registry.register_dsl("incomplete_dsl", incomplete_module)

        self.assertIn("missing required methods", str(cm.exception))
        self.assertIn("runtime_version", str(cm.exception))
        self.assertIn("register_op_override", str(cm.exception))

        # Verify DSL was not registered
        self.assertNotIn("incomplete_dsl", registry.list_all_dsls())


instantiate_parametrized_tests(TestNativeDSLOps)


if __name__ == "__main__":
    run_tests()
