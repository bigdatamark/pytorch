# Owner(s): ["module: inductor"]
# pyre-strict
from __future__ import annotations
# [MODULE DOC] Unit tests for the inductor runtime caching package (config, context, exceptions,
# implementations, locks, utils and the Memoizer / PersistentMemoizer interfaces). Test classes
# cover versioned config resolution, isolation context and key generation, the exception
# hierarchy, in-memory and on-disk cache implementations, lock acquisition timeouts, the LRU
# cache helper, memoizer record/replay/memoize behavior (including JSON dump loading and
# writing, force-disable handling, fresh-cache clearing, should-pad memoization and deferred
# recording). Import-time side effects: defines module level lambdas that build patches of the
# caching config flags; the patch helpers below mutate the on-disk implementation class when
# applied. Running the file as a script invokes the inductor test runner.

import json
import os
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError, wait
from contextlib import contextmanager
from functools import wraps
from itertools import combinations
from random import Random
from shutil import rmtree
from threading import Event, Lock
from typing import Any, TYPE_CHECKING
from typing_extensions import TypeVar
from unittest.mock import patch

from filelock import FileLock

from torch._inductor.runtime.caching import (
    config,
    context,
    exceptions,
    implementations as impls,
    interfaces,
    locks,
    Memoizer,
    PersistentMemoizer,
    utils,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from pathlib import Path


set_caching_module_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_CACHING_MODULE_ENABLED", lambda: enabled
)
set_deterministic_caching_enabled = lambda enabled: patch.object(  # noqa: E731
    config, "IS_DETERMINISTIC_CACHING_ENABLED", lambda: enabled
)
set_strictly_pre_populated_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_PRE_POPULATED_DETERMINISM", enabled
)
set_strictly_cached_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "STRICTLY_CACHED_DETERMINISM", enabled
)
set_local_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "LOCAL_DETERMINISM", enabled
)
set_global_determinism = lambda enabled: patch.object(  # noqa: E731
    config, "GLOBAL_DETERMINISM", enabled
)


def patch_on_disk_cache_base_dir(fn):
    # [FUNCTION DOC] HELPER: Decorator for test methods of classes that provide a sub_dir method
    # and a random_string property. The wrapper reads the default base directory of the on-disk
    # cache implementation, then patches the class level base directory to a per-test
    # subdirectory (named from sub_dir plus a random suffix) while the wrapped test runs, so
    # tests never touch the shared cache location. Returns whatever the wrapped test returns.
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper produced by the decorator. Takes the test instance plus
        # any extra positional and keyword arguments, patches the on-disk base directory for the
        # duration of the call and forwards everything to the wrapped test.
        default_base_dir = impls._OnDiskCacheImpl()._base_dir
        with patch.object(
            impls._OnDiskCacheImpl,
            "_base_dir",
            default_base_dir / f"{self.sub_dir()}/rng-{self.random_string[4:]}",
        ):
            return fn(self, *args, **kwargs)

    return wrapper


def patch_remote_cache_with_on_disk_cache(fn):
    # [FUNCTION DOC] HELPER: Decorator for tests that need the remote cache implementation to
    # behave like a local one. Side effect (permanent, applied when the decorator is applied):
    # marks the on-disk implementation class as having strong consistency. Returns the function
    # wrapped in a patch that replaces the remote cache implementation attribute of the
    # implementations module with the on-disk implementation for the duration of each call.
    impls._OnDiskCacheImpl.has_strong_consistency = True
    return patch.object(impls, "_RemoteCacheImpl", impls._OnDiskCacheImpl)(fn)


class TestMixin:
    # [CLASS DOC] HELPER: Mixin shared by the cache test classes. Holds the list of
    # implementation type names under test (in-memory and on-disk), a per-process random class
    # id used to build unique sub directory names, a factory that instantiates an implementation
    # by type name, and two properties producing fresh random string and random bytes values for
    # keys and values. Holds no mutable per-test state.
    impl_typenames: list[str] = [
        "_InMemoryCacheImpl",
        "_OnDiskCacheImpl",
    ]
    cls_id: int = Random().randint(0, 2**32)

    def impl_from_typename(self, impl_typename: str) -> impls._CacheImpl:
        # [FUNCTION DOC] HELPER: Instantiates and returns the cache implementation class
        # from the implementations module whose name is given, using its default
        # constructor arguments.
        return getattr(impls, impl_typename)()

    @property
    def random_string(self) -> str:
        # [FUNCTION DOC] HELPER: Property returning a fresh random string of the form s-N where
        # N is a random integer below 2 to the 32nd; a new value on every access.
        return f"s-{Random().randint(0, 2**32)}"

    @property
    def random_bytes(self) -> bytes:
        # [FUNCTION DOC] HELPER: Property returning a fresh random value, encoded as UTF-8
        # bytes, of the same shape as the random string property; a new value on every access.
        return f"s-{Random().randint(0, 2**32)}".encode()


@instantiate_parametrized_tests
class ConfigTest(TestCase):
    # [CLASS DOC] Tests for the versioned config resolution function. Class attributes define a
    # fake knob (version number, JustKnobs name, OSS default, environment variable name) and a
    # file lock guarding the environment variable name, so that tests which patch the process
    # environment do not race with each other when run in parallel. Each test configures the
    # environment, then checks the resolved boolean via the shared assertion helper.
    FOO_THIS_VERSION: int = 0
    FOO_JK_NAME: str = "foo_jk_name"
    FOO_OSS_DEFAULT: bool = False
    FOO_ENV_VAR_OVERRIDE: str = "foo_env_var_override"
    FOO_ENV_VAR_OVERRIDE_LOCK_FPATH: str = f"/tmp/testing/{FOO_ENV_VAR_OVERRIDE}.lock"
    FOO_ENV_VAR_OVERRIDE_LOCK: FileLock = FileLock(FOO_ENV_VAR_OVERRIDE_LOCK_FPATH)

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that removes the lock file path (with errors
        # ignored) before the tests run so a stale lock does not interfere.
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that removes the lock file path (with errors
        # ignored) after the tests finish.
        rmtree(cls.FOO_ENV_VAR_OVERRIDE_LOCK_FPATH, ignore_errors=True)

    def assert_versioned_config(self, expected_enabled: bool) -> None:
        # [FUNCTION DOC] HELPER: Clears the memoization cache of the versioned config function,
        # resolves it with the fake knob settings of this class, and asserts the result equals
        # the expected boolean. Raises an assertion failure on mismatch.
        config._versioned_config.cache_clear()
        actual_enabled: bool = config._versioned_config(
            self.FOO_JK_NAME,
            self.FOO_THIS_VERSION,
            self.FOO_OSS_DEFAULT,
            env_var_override=self.FOO_ENV_VAR_OVERRIDE,
        )
        self.assertEqual(actual_enabled, expected_enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_env_var_override(
        self,
        enabled: bool,
    ) -> None:
        # [FUNCTION DOC] Parametrized over enabled true and false. While holding the environment
        # variable lock (1 second acquisition timeout), patches the environment variable
        # override to one or zero, forces the non-fbcode path and sets the OSS default to the
        # opposite value, then asserts the resolved config equals the override. Shows the
        # override wins over the default.
        """Test that environment variable overrides take precedence over other configuration sources.

        Verifies that when an environment variable override is set to "1" or "0",
        the _versioned_config function returns the corresponding boolean value
        regardless of other configuration settings.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(
                os.environ,
                {
                    self.FOO_ENV_VAR_OVERRIDE: "1" if enabled else "0",
                },
            ),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", not enabled),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_version_check(
        self,
        enabled: bool,
    ) -> None:
        # [FUNCTION DOC] Parametrized over enabled true and false. Holding the lock, clears the
        # environment, pretends to run inside fbcode and makes the JustKnobs integer lookup
        # return the expected version minus one when enabled is true and plus one when false.
        # Asserts the config is enabled exactly when enabled is true, i.e. it depends on the
        # JustKnobs version relative to this version.
        """Test that _versioned_config responds correctly to version changes in Facebook environments.

        Verifies that when running in fbcode environments (is_fbcode=True), the configuration
        is enabled when the JustKnobs version matches the expected version, and disabled when
        the version differs. This ensures proper rollout control through version management.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=self.FOO_THIS_VERSION + (-1 if enabled else 1),
            ),
        ):
            self.assert_versioned_config(enabled)

    @parametrize("enabled", [True, False])
    def test_versioned_config_oss_default(
        self,
        enabled: bool,
    ) -> None:
        # [FUNCTION DOC] Parametrized over enabled true and false. Holding the lock, clears the
        # environment, pretends not to be in fbcode and sets the OSS default to the parametrized
        # value; asserts the resolved config equals that default.
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=False,
            ),
            patch.object(self, "FOO_OSS_DEFAULT", enabled),
        ):
            self.assert_versioned_config(enabled)

    def test_versioned_config_jk_failure(self) -> None:
        # [FUNCTION DOC] Holding the lock, clears the environment, pretends to run inside
        # fbcode and makes the JustKnobs integer lookup return zero; asserts the config
        # resolves to disabled.
        # KNOWN ISSUE: the existing docstring is a copy of the OSS default test docstring and
        # does not describe this test, which actually checks the fbcode path with a version
        # lookup of zero.
        """Test that _versioned_config uses OSS default values in non-Facebook environments.

        Verifies that when running in non-fbcode environments (is_fbcode=False) with no
        environment variable overrides, the configuration falls back to the OSS default
        value. This ensures proper behavior for open-source PyTorch distributions.
        """
        with (
            self.FOO_ENV_VAR_OVERRIDE_LOCK.acquire(timeout=1),
            patch.dict(os.environ, {}, clear=True),
            patch(
                "torch._inductor.runtime.caching.config.is_fbcode",
                return_value=True,
            ),
            patch(
                "torch._utils_internal.justknobs_getval_int",
                return_value=0,
            ),
        ):
            self.assert_versioned_config(False)


@instantiate_parametrized_tests
class ContextTest(TestCase):
    # [CLASS DOC] Tests for the isolation context and isolation key generation in the context
    # module. Provides one helper that builds an isolation schema from selected form names; the
    # tests compare generated isolation contexts against expected mappings and check key
    # distinctness and repeatability.
    def isolation_schema_from_forms_of_context_selected(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> context.IsolationSchema:
        # [FUNCTION DOC] HELPER: Builds an isolation schema whose runtime and compile context
        # mappings contain every known form of context for the respective context class, with a
        # boolean that is true only for forms named in the given selections. Returns the schema.
        return context.IsolationSchema(
            runtime_context={
                form_of_context: form_of_context
                in set(runtime_forms_of_context_selected)
                for form_of_context in context._RuntimeContext.forms_of_context()
            },
            compile_context={
                form_of_context: form_of_context
                in set(compile_forms_of_context_selected)
                for form_of_context in context._CompileContext.forms_of_context()
            },
        )

    @parametrize(
        "runtime_forms_of_context_selected",
        [(), *list(combinations(context._RuntimeContext.forms_of_context(), 2))],
    )
    @parametrize(
        "compile_forms_of_context_selected",
        [(), *list(combinations(context._CompileContext.forms_of_context(), 2))],
    )
    def test_selected_isolation_context(
        self,
        runtime_forms_of_context_selected: Sequence[str],
        compile_forms_of_context_selected: Sequence[str],
    ) -> None:
        # [FUNCTION DOC] Parametrized over the empty selection plus all pairs of runtime
        # forms, crossed with the same for compile forms. Builds a schema selecting exactly
        # those forms and asserts the isolation context equals a mapping with a runtime and a
        # compile entry, each holding the value of each selected form's getter, or none when
        # nothing is selected.
        """
        Tests that isolation context generation works correctly for specific combinations
        of runtime and compile context forms.

        Verifies that the _isolation_context function properly creates isolation contexts
        based on the selected forms of runtime and compile context, ensuring that only
        the specified context forms are included in the resulting isolation context.
        """
        ischema: context.IsolationSchema = (
            self.isolation_schema_from_forms_of_context_selected(
                runtime_forms_of_context_selected, compile_forms_of_context_selected
            )
        )

        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in runtime_forms_of_context_selected
                }
                or None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in compile_forms_of_context_selected
                }
                or None,
            },
        )

    @parametrize("all_runtime_context", [True, False])
    @parametrize("all_compile_context", [True, False])
    def test_all_or_none_isolation_context(
        self, all_runtime_context: bool, all_compile_context: bool
    ) -> None:
        # [FUNCTION DOC] Parametrized over booleans for all runtime context and all compile
        # context. Builds a schema with plain booleans and asserts the isolation context
        # contains every form's value for a side set to true and none for a side set to false.
        """
        Tests isolation context generation when using all or no context forms.

        Verifies that the isolation context correctly includes all forms of context
        when set to True, or excludes all forms when set to False, for both
        runtime and compile contexts.
        """
        ischema: context.IsolationSchema = context.IsolationSchema(
            runtime_context=all_runtime_context, compile_context=all_compile_context
        )
        self.assertEqual(
            context._isolation_context(ischema),
            {
                "runtime_context": {
                    form_of_context: getattr(context._RuntimeContext, form_of_context)()
                    for form_of_context in context._RuntimeContext.forms_of_context()
                }
                if all_runtime_context
                else None,
                "compile_context": {
                    form_of_context: getattr(context._CompileContext, form_of_context)()
                    for form_of_context in context._CompileContext.forms_of_context()
                }
                if all_compile_context
                else None,
            },
        )

    def test_isolation_key_is_distinct(self) -> None:
        # [FUNCTION DOC] Iterates over every subset size and every combination of runtime and
        # compile forms of context (the loops use ranges that stop before the full set size, so
        # the all-forms selection is not covered), builds the schema for each and computes the
        # isolation key. Asserts no key was seen before and records it, thereby checking all
        # keys are distinct.
        """
        Tests that different combinations of runtime and compile context forms
        generate unique isolation keys.

        Verifies that each possible combination of context forms produces a distinct
        isolation key, ensuring no collisions occur between different contexts.
        """
        ikeys: set[str] = set()
        for num_runtime_forms_of_context_selected in range(
            len(context._RuntimeContext.forms_of_context())
        ):
            for num_compile_forms_of_context_selected in range(
                len(context._CompileContext.forms_of_context())
            ):
                for runtime_forms_of_context_selected in combinations(
                    context._RuntimeContext.forms_of_context(),
                    num_runtime_forms_of_context_selected,
                ):
                    for compile_forms_of_context_selected in combinations(
                        context._CompileContext.forms_of_context(),
                        num_compile_forms_of_context_selected,
                    ):
                        ischema: context.IsolationSchema = (
                            self.isolation_schema_from_forms_of_context_selected(
                                runtime_forms_of_context_selected,
                                compile_forms_of_context_selected,
                            )
                        )
                        ikey: str = context._isolation_key(ischema)
                        self.assertFalse(ikey in ikeys)
                        ikeys.add(ikey)

    def test_isolation_key_is_repeatable(self) -> None:
        # [FUNCTION DOC] Asserts that two calls of the isolation key function with default
        # arguments return equal values.
        """
        Tests that calling the isolation key function multiple times with the same
        parameters produces the same result.

        Verifies that the isolation key generation is deterministic and consistent
        across multiple invocations with identical inputs.
        """
        self.assertEqual(context._isolation_key(), context._isolation_key())

    def test_select_runtime_context_matches_forms_of_context(self) -> None:
        # [FUNCTION DOC] Asserts that the set of required keys of the selected runtime context
        # typed mapping equals the set of runtime forms of context, keeping the type definition
        # in sync with the context class.
        """
        Tests that the selected runtime context matches the forms of context.

        Verifies that the selected runtime context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedRuntimeContext.__required_keys__),
            set(context._RuntimeContext.forms_of_context()),
        )

    def test_select_compile_context_matches_forms_of_context(self) -> None:
        # [FUNCTION DOC] Asserts that the set of required keys of the selected compile context
        # typed mapping equals the set of compile forms of context, keeping the type definition
        # in sync with the context class.
        """
        Tests that the selected compile context matches the forms of context.

        Verifies that the selected compile context includes only the forms of context
        specified in the isolation schema, ensuring that the isolation context is
        properly selected and configured.
        """
        self.assertEqual(
            set(context.SelectedCompileContext.__required_keys__),
            set(context._CompileContext.forms_of_context()),
        )


@instantiate_parametrized_tests
class ExceptionsTest(TestCase):
    # [CLASS DOC] Tests for the exception hierarchy of the caching package. Holds the list of
    # exception type names examined by the parametrized test.
    exception_typenames: list[str] = [
        "CacheError",
        "SystemError",
        "LockTimeoutError",
        "FileLockTimeoutError",
        "UserError",
        "KeyEncodingError",
        "ValueEncodingError",
        "ValueDecodingError",
    ]

    @parametrize("exception_typename", exception_typenames)
    def test_exception_is_CacheError(self, exception_typename: str) -> None:
        # [FUNCTION DOC] Parametrized over the exception type names. Looks the type up in the
        # exceptions module and asserts it is a subclass of the base cache error.
        """Test that all custom cache exceptions inherit from the base CacheError class.

        Verifies that every exception type defined in the caching exceptions module
        is properly derived from CacheError, ensuring consistent exception hierarchy
        and enabling unified exception handling throughout the caching system.
        """
        self.assertTrue(
            issubclass(getattr(exceptions, exception_typename), exceptions.CacheError)
        )

    def test_exception_other(self) -> None:
        # [FUNCTION DOC] Asserts specific inheritance edges: system error, user error derive
        # from the cache error; the lock timeout and file lock timeout errors derive from
        # system error; the key encoding, value encoding and value decoding errors derive from
        # user error.
        """
        Test the inheritance relationships among custom cache exception classes.

        Verifies that the exception classes in the caching exceptions module have the correct
        subclass relationships, ensuring the exception hierarchy is as intended. This includes
        checks for both direct and indirect inheritance between base and derived exception types.
        """
        self.assertTrue(issubclass(exceptions.SystemError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.LockTimeoutError, exceptions.SystemError))
        self.assertTrue(
            issubclass(exceptions.FileLockTimeoutError, exceptions.SystemError)
        )
        self.assertTrue(issubclass(exceptions.UserError, exceptions.CacheError))
        self.assertTrue(issubclass(exceptions.KeyEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueEncodingError, exceptions.UserError))
        self.assertTrue(issubclass(exceptions.ValueDecodingError, exceptions.UserError))


@instantiate_parametrized_tests
class ImplementationsTest(TestMixin, TestCase):
    # [CLASS DOC] Tests for the in-memory and on-disk cache implementations, parametrized
    # by implementation type name. Uses a per-class sub directory name built from the
    # class id; the class fixtures delete the on-disk cache directory for that sub
    # directory before and after the tests. Provides assertion helpers over the get and
    # insert methods of an implementation.
    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-impls-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) before the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) after the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir, ignore_errors=True
        )

    def assert_key_in(self, key: str, impl: impls._CacheImpl) -> None:
        # [FUNCTION DOC] HELPER: Asserts that getting the key from the implementation returns a
        # non-missing result.
        self.assertTrue(impl.get(key) is not None)

    def assert_key_not_in(self, key: str, impl: impls._CacheImpl) -> None:
        # [FUNCTION DOC] HELPER: Asserts that getting the key from the implementation returns a
        # missing result.
        self.assertTrue(impl.get(key) is None)

    def assert_key_value_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts that inserting the key and value into the
        # implementation reports success.
        self.assertTrue(impl.insert(key, value))

    def assert_key_value_not_inserted_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts that inserting the key and value into the
        # implementation reports failure (key already present).
        self.assertFalse(impl.insert(key, value))

    def assert_key_has_value_in(
        self, key: str, value: Any, impl: impls._CacheImpl
    ) -> None:
        # [FUNCTION DOC] HELPER: Asserts that the key is present in the implementation and that
        # the stored value equals the given value.
        self.assertTrue(((get := impl.get(key)) is not None) and (get.value == value))

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_get(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Builds the implementation for the given type name and, while holding
        # its lock, asserts a random key is absent. Runs with the on-disk base directory and
        # remote cache patched.
        """Test cache get operation returns cache miss for non-existent keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle get operations for keys that do not exist in the cache. This test
        ensures that the cache properly returns a cache miss (hit=False) when
        attempting to retrieve a non-existent key.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            self.assert_key_not_in(self.random_string, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Under the implementation lock, asserts a random key is absent, inserts
        # a random bytes value successfully, then asserts the key is present with that value.
        """Test cache insert operation successfully stores and retrieves key-value pairs.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for new key-value pairs. This test ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. Insert operations succeed for new keys
        3. The stored value can be retrieved correctly after insertion

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_insert_will_not_overwrite(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Under the implementation lock, inserts a value for a random key
        # successfully, asserts a second insert of different bytes for the same key is refused,
        # and asserts the original value is still stored.
        """Test cache insert operation does not overwrite existing keys.

        Verifies that both in-memory and on-disk cache implementations correctly
        handle insert operations for keys that already exist in the cache. This test
        ensures that:
        1. Keys initially don't exist in the cache (cache miss)
        2. First insert operation succeeds for new keys
        3. Subsequent insert operations with the same key fail (inserted=False)
        4. The original value is preserved and not overwritten

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            key: str = self.random_string
            self.assert_key_not_in(key, impl)
            value: bytes = self.random_bytes
            self.assert_key_value_inserted_in(key, value, impl)
            self.assert_key_value_not_inserted_in(key, self.random_bytes, impl)
            self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_encoding(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Only exercises the in-memory implementation (the on-disk case does
        # nothing): inserts a lambda as the value, which would not be serializable, and asserts
        # it is stored and returned unchanged. Shows in-memory caches hold arbitrary objects.
        """Test that in-memory cache can store any value type.

        Verifies that in-memory cache implementations can store arbitrary values
        including non-serializable ones (such as lambda functions) since they don't
        require serialization. On-disk caches now expect bytes, so they skip this test.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._InMemoryCacheImpl):
                key: str = self.random_string
                value = lambda: None  # noqa: E731
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_value_decoding(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Only exercises the on-disk implementation. Under the lock it creates
        # the cache file for a random key manually (exclusive create mode), writes the version
        # header followed by the raw bytes, and asserts the implementation returns those raw
        # bytes as the value, i.e. no unpickling happens.
        """Test that on-disk cache implementations return raw bytes from storage.

        Verifies that on-disk cache implementations return raw bytes from disk
        without attempting to unpickle them, as values are now expected to be
        stored as bytes. This test writes raw bytes to a cache file and confirms
        they are returned as-is.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                fpath: Path = impl._fpath_from_key(key)
                test_bytes: bytes = b"foo"
                with open(fpath, "xb") as fp:
                    impl._write_version_header(fp)
                    fp.write(test_bytes)
                self.assert_key_has_value_in(key, test_bytes, impl)

    @patch_on_disk_cache_base_dir
    @patch_remote_cache_with_on_disk_cache
    @parametrize("impl_typename", TestMixin.impl_typenames)
    def test_version_mismatch(self, impl_typename: str) -> None:
        # [FUNCTION DOC] Only exercises the on-disk implementation. Under the lock: inserts a
        # value and reads it back; then, with the class level version number patched to one
        # higher, asserts the key is missing, reinserts the same value and reads it back; after
        # leaving the patch, asserts the key is again missing and reinserts and reads it again.
        # Shows that a lookup under a different version number does not see entries written
        # under another version.
        """Test that on-disk cache implementations properly handle version mismatches.

        Verifies that on-disk cache implementations correctly handle cached data when
        the cache version changes. This test ensures that:
        1. Data can be stored and retrieved with the current version
        2. When version changes, previously cached data becomes inaccessible (cache miss)
        3. New data can be stored with the new version
        4. After version change, old cached data remains inaccessible

        This version checking mechanism prevents corruption and compatibility issues
        when cache formats change between software versions. Only applies to on-disk
        implementations since in-memory caches don't persist across version changes.

        Args:
            impl_typename: The cache implementation type to test ("_InMemoryCacheImpl" or "_OnDiskCacheImpl")
        """
        impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
        with impl.lock():
            if isinstance(impl, impls._OnDiskCacheImpl):
                key: str = self.random_string
                self.assert_key_not_in(key, impl)
                value: bytes = self.random_bytes
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)
                with patch.object(
                    impls._OnDiskCacheImpl, "_version", impl._version + 1
                ):
                    self.assert_key_not_in(key, impl)
                    self.assert_key_value_inserted_in(key, value, impl)
                    self.assert_key_has_value_in(key, value, impl)
                self.assert_key_not_in(key, impl)
                self.assert_key_value_inserted_in(key, value, impl)
                self.assert_key_has_value_in(key, value, impl)


@instantiate_parametrized_tests
class LocksTest(TestMixin, TestCase):
    # [CLASS DOC] Tests for the lock helpers in the locks module, for both a thread lock and a
    # file lock. Provides a type variable, a context manager that yields a thread pool executor
    # and always shuts it down, and predicates to classify and query the lock state.
    T = TypeVar("T")

    @contextmanager
    def executor(self) -> Generator[ThreadPoolExecutor, None, None]:
        # [FUNCTION DOC] HELPER: Context manager that creates a thread pool executor, yields it,
        # and shuts it down in a finally block.
        executor: ThreadPoolExecutor = ThreadPoolExecutor()
        try:
            yield executor
        finally:
            executor.shutdown()

    def is_lock(self, lock_or_flock: Lock | FileLock) -> bool:
        # [FUNCTION DOC] HELPER: Returns true when the object has a locked attribute (identifies
        # a thread lock).
        return hasattr(lock_or_flock, "locked")

    def is_flock(self, lock_or_flock: Lock | FileLock) -> bool:
        # [FUNCTION DOC] HELPER: Returns true when the object has an is_locked attribute
        # (identifies a file lock).
        return hasattr(lock_or_flock, "is_locked")

    def lock_or_flock_locked(self, lock_or_flock: Lock | FileLock) -> bool:
        # [FUNCTION DOC] HELPER: Returns whether a thread lock or file lock is currently held,
        # using the matching query; raises NotImplementedError for any other object.
        if self.is_lock(lock_or_flock):
            return lock_or_flock.locked()
        elif self.is_flock(lock_or_flock):
            return lock_or_flock.is_locked
        else:
            raise NotImplementedError

    def test_BLOCKING(self) -> None:
        # [FUNCTION DOC] Asserts the blocking timeout constant of the locks module equals
        # minus one.
        self.assertEqual(locks._BLOCKING, -1.0)

    def test_NON_BLOCKING(self) -> None:
        # [FUNCTION DOC] Asserts the non-blocking timeout constant of the locks module
        # equals zero.
        self.assertEqual(locks._NON_BLOCKING, 0.0)

    def test_BLOCKING_WITH_TIMEOUT(self) -> None:
        # [FUNCTION DOC] Asserts the blocking-with-timeout constant of the locks module is
        # greater than zero.
        self.assertGreater(locks._BLOCKING_WITH_TIMEOUT, 0.0)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1.0)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1.0)
    @parametrize("lock_typename", ["Lock", "FileLock"])
    @parametrize("lock_timeout", ["BLOCKING", "NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"])
    @parametrize("acquisition_mode", ["safe", "unsafe"])
    @parametrize("release", ["unlocked", "never", "before_timeout", "after_timeout"])
    def test_acquire_with_timeout(
        self,
        lock_typename: str,
        lock_timeout: str,
        acquisition_mode: str,
        release: str,
    ) -> None:
        # [FUNCTION DOC] Parametrized over lock type (thread lock or file lock), timeout mode
        # (blocking, non-blocking, blocking with timeout), acquisition mode (safe context
        # manager or unsafe manual) and release scenario (lock free, never released, released
        # before the timeout, released after the timeout). The timeout constants of the locks
        # module are patched to one second. The lock is optionally pre-acquired by the test, an
        # acquisition runs in a worker thread, and the outcome is checked: success returns
        # nothing, a held lock raises the matching lock timeout error for non-blocking modes,
        # and a blocking acquisition that never succeeds is still pending at the result timeout
        # (raising a futures timeout). The test releases the lock itself where needed and
        # deletes the file lock path at the end. Raises an assertion error on unexpected
        # parameter values.
        """Test lock acquisition behavior with various timeout configurations and release scenarios.

        This comprehensive test verifies the lock acquisition functionality for both threading.Lock
        and FileLock objects across different timeout modes, acquisition patterns, and release timings.
        The test validates proper exception handling, timeout behavior, and correct lock state management.

        Test parameters:
        - lock_typename: Tests both "Lock" (threading.Lock) and "FileLock" (filelock.FileLock) types
        - lock_timeout: Tests "BLOCKING", "NON_BLOCKING", and "BLOCKING_WITH_TIMEOUT" modes
        - acquisition_mode: Tests both "safe" (context manager) and "unsafe" (manual) acquisition
        - release: Tests "unlocked", "never", "before_timeout", and "after_timeout" scenarios

        The test ensures that:
        - Safe acquisition properly manages lock lifecycle through context managers
        - Unsafe acquisition requires manual release and behaves correctly
        - Timeout exceptions are raised appropriately for different timeout configurations
        - Lock states are correctly maintained throughout acquisition and release cycles
        - Different lock types (Lock vs FileLock) behave consistently with their respective APIs
        """

        def inner(lock_or_flock: Lock | FileLock, timeout: int) -> None:
            # [FUNCTION DOC] HELPER: Runs inside the worker thread. Acquires the given thread
            # lock or file lock with the given timeout using the safe or unsafe helper that
            # matches the lock type and the acquisition mode, asserting the lock is held while
            # acquired (the unsafe variant releases manually), then asserts it is released.
            # Raises NotImplementedError for an unknown lock type or mode.
            if self.is_lock(lock_or_flock):
                lock: Lock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_lock_with_timeout(lock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(lock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_lock_with_timeout(lock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(lock))
                    lock.release()
                else:
                    raise NotImplementedError
            elif self.is_flock(lock_or_flock):
                flock: FileLock = lock_or_flock
                if acquisition_mode == "safe":
                    with locks._acquire_flock_with_timeout(flock, timeout=timeout):
                        self.assertTrue(self.lock_or_flock_locked(flock))
                elif acquisition_mode == "unsafe":
                    locks._unsafe_acquire_flock_with_timeout(flock, timeout=timeout)
                    self.assertTrue(self.lock_or_flock_locked(flock))
                    flock.release()
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))

        if lock_typename not in ["Lock", "FileLock"]:
            raise AssertionError(f"Unexpected lock_typename: {lock_typename}")
        flock_fpath: Path = (
            impls._OnDiskCacheImpl()._cache_dir
            / f"testing-locks-instance-{self.random_string}.lock"
        )
        lock_or_flock: Lock | FileLock = (
            Lock() if lock_typename == "Lock" else FileLock(str(flock_fpath))
        )
        lock_exception_type: type = (
            exceptions.LockTimeoutError
            if lock_typename == "Lock"
            else exceptions.FileLockTimeoutError
        )

        if release == "unlocked":
            self.assertFalse(self.lock_or_flock_locked(lock_or_flock))
        elif release in ["never", "before_timeout", "after_timeout"]:
            self.assertTrue(lock_or_flock.acquire(timeout=locks._NON_BLOCKING))
            self.assertTrue(self.lock_or_flock_locked(lock_or_flock))
        else:
            raise NotImplementedError

        with self.executor() as executor:
            if lock_timeout not in [
                "BLOCKING",
                "NON_BLOCKING",
                "BLOCKING_WITH_TIMEOUT",
            ]:
                raise AssertionError(f"Unexpected lock_timeout: {lock_timeout}")
            lock_or_flock_future: Future[None] = executor.submit(
                inner,
                lock_or_flock,
                timeout={
                    "BLOCKING": locks._BLOCKING,
                    "NON_BLOCKING": locks._NON_BLOCKING,
                    "BLOCKING_WITH_TIMEOUT": locks._BLOCKING_WITH_TIMEOUT,
                }[lock_timeout],
            )

            if release == "unlocked":
                self.assertIsNone(lock_or_flock_future.result())
            elif release == "never":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                if lock_timeout == "BLOCKING":
                    with self.assertRaises(TimeoutError):
                        lock_or_flock_future.result(
                            timeout=locks._BLOCKING_WITH_TIMEOUT
                        )
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
                lock_or_flock.release()
            elif release == "before_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT / 2))
                lock_or_flock.release()
                if lock_timeout in ["BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout == "NON_BLOCKING":
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError
            elif release == "after_timeout":
                wait([lock_or_flock_future], timeout=(locks._BLOCKING_WITH_TIMEOUT * 2))
                lock_or_flock.release()
                if lock_timeout == "BLOCKING":
                    self.assertIsNone(lock_or_flock_future.result())
                elif lock_timeout in ["NON_BLOCKING", "BLOCKING_WITH_TIMEOUT"]:
                    with self.assertRaises(lock_exception_type):
                        lock_or_flock_future.result()
                else:
                    raise NotImplementedError

        flock_fpath.unlink(missing_ok=True)

    @patch.object(locks, "_BLOCKING_WITH_TIMEOUT", 1)
    @patch.object(locks, "_DEFAULT_TIMEOUT", 1)
    @parametrize(
        "impl_typename_combos",
        list(combinations(TestMixin.impl_typenames, 1))
        + list(combinations(TestMixin.impl_typenames, 2)),
    )
    def test_acquire_many_impl_locks_with_timeout(
        self,
        impl_typename_combos: tuple[str, ...],
    ) -> None:
        # [FUNCTION DOC] Parametrized over single and pairs of implementation type names; the
        # timeout constants of the locks module are patched to one. Builds the implementations,
        # acquires all their locks together via the many-locks helper, asserts each
        # implementation's thread lock or file lock reports held inside the context, and asserts
        # all are released after leaving it.
        # KNOWN ISSUE: the local list named impls shadows the imported implementations module
        # inside this function; harmless because the annotations are not evaluated.
        impls: list[impls._CacheImpl] = []
        for impl_typename in impl_typename_combos:
            impl: impls._CacheImpl = self.impl_from_typename(impl_typename)
            impls.append(impl)

        with locks._acquire_many_impl_locks_with_timeout(*impls):
            for impl in impls:
                if hasattr(impl, "_lock"):
                    self.assertTrue(impl._lock.locked())
                elif hasattr(impl, "_flock"):
                    self.assertTrue(impl._flock.is_locked)

        for impl in impls:
            if hasattr(impl, "_lock"):
                self.assertFalse(impl._lock.locked())
            elif hasattr(impl, "_flock"):
                self.assertFalse(impl._flock.is_locked)


@instantiate_parametrized_tests
class UtilsTest(TestMixin, TestCase):
    # [CLASS DOC] Tests for the utils module helpers.
    def test_lru_cache(self) -> None:
        # [FUNCTION DOC] Defines a function that echoes its positional and keyword
        # arguments, wraps it with the LRU cache decorator and asserts the echoed result for
        # an integer, a float, a string, and a string with a keyword argument. Checks that
        # argument values and types are preserved through the cache wrapper (the wrapper
        # keeps typed entries separate).
        """Test that the LRU cache decorator works correctly with various argument types.

        Verifies that the _lru_cache decorator properly caches function results
        and handles different types of arguments including integers, floats, strings,
        and keyword arguments. Tests that cached calls return identical results
        to non-cached calls with proper argument preservation.
        """

        @utils._lru_cache
        def foo(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Returns a pair of the positional argument tuple and the
            # keyword argument mapping it received.
            return args, kwargs

        self.assertEqual(
            foo(0),
            (
                (0,),
                {},
            ),
        )
        self.assertEqual(
            foo(0.0),
            (
                (0.0,),
                {},
            ),
        )
        self.assertEqual(
            foo("foo"),
            (
                ("foo",),
                {},
            ),
        )
        self.assertEqual(
            foo("foo", bar="bar"),
            (
                ("foo",),
                {"bar": "bar"},
            ),
        )


@instantiate_parametrized_tests
class InterfacesTest(TestMixin, TestCase):
    # [CLASS DOC] Tests for the Memoizer and PersistentMemoizer interfaces: record, replay and
    # memoize decorators with caching on and off, two-level (memory then disk) lookup, loading
    # from and dumping to the JSON cache dump file. Uses a per-class sub directory name; the
    # class fixtures delete its on-disk cache directory before and after the tests.
    """Test class for Memoizer and PersistentMemoizer interfaces."""

    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-interfaces-instance-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) before the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) after the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    # ============= Memoizer Tests =============

    @set_caching_module_enabled(True)
    def test_memoizer_record_caches_result(self) -> None:
        # [FUNCTION DOC] With caching enabled, wraps a counting function with the record
        # decorator and calls it twice with the same argument. Asserts the function ran both
        # times (record always executes) and returned the doubled value each time.
        """Test that Memoizer.record() caches function results.

        Verifies that when a function is decorated with record(), its result
        is cached and can be retrieved later.
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called twice (record always executes)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(True)
    def test_memoizer_replay_retrieves_cached_result(self) -> None:
        # [FUNCTION DOC] With caching enabled, records a result for one argument, then defines a
        # replay-decorated function that raises if executed. Asserts the replay returns the
        # recorded result and the original function's call count did not change.
        """Test that Memoizer.replay() retrieves cached results without executing the function.

        Verifies that when a function is decorated with replay(), it retrieves
        results from cache without executing the original function.
        """
        # Setup: create a memoizer, record a result, then try to replay it
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: record a result first
        compute(5)
        self.assertEqual(call_count, 1)

        # Create a replay function for the same computation
        @memoizer.replay()
        def compute_replay(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if ever
            # executed, proving replay serves from cache.
            # This should never be called
            raise AssertionError("Function should not be executed during replay")

        # Assert: replay retrieves the cached result without calling the function
        result = compute_replay(5)
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)  # No additional calls

    @set_caching_module_enabled(True)
    def test_memoizer_replay_raises_on_cache_miss(self) -> None:
        # [FUNCTION DOC] With caching enabled, asserts that calling a replay-decorated function
        # with an argument that was never recorded raises a key error.
        """Test that Memoizer.replay() raises KeyError on cache miss.

        Verifies that when replay() is called with arguments that have no cached
        result, it raises a KeyError.
        """
        # Setup: create a memoizer with replay decorator
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target returning double the input; never expected
            # to run.
            return x * 2

        # Execute & Assert: replay raises KeyError for uncached arguments
        with self.assertRaises(KeyError):
            compute(5)

    @set_caching_module_enabled(True)
    def test_memoizer_memoize_caches_and_retrieves(self) -> None:
        # [FUNCTION DOC] With caching enabled, calls a memoize-decorated counting function twice
        # with the same argument. Asserts the call count is one after each call and both results
        # are the doubled value.
        """Test that Memoizer.memoize() caches on first call and retrieves on subsequent calls.

        Verifies that memoize() combines record and replay functionality:
        - First call executes the function and caches the result
        - Subsequent calls retrieve from cache without executing
        """
        # Setup: create a memoizer and a function that tracks call count
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice with the same argument
        result1 = compute(5)
        self.assertEqual(call_count, 1)  # Function called on first invocation

        result2 = compute(5)
        self.assertEqual(call_count, 1)  # Function not called on second invocation

        # Assert: both calls return the same result
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_record_disabled_returns_original_function(self) -> None:
        # [FUNCTION DOC] With caching disabled, calls a record-decorated counting function
        # twice. Asserts it ran both times, i.e. the decorator returns the original behavior
        # with no caching.
        """Test that Memoizer.record() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    @set_caching_module_enabled(False)
    def test_memoizer_replay_disabled_always_raises(self) -> None:
        # [FUNCTION DOC] With caching disabled, asserts a replay-decorated function always
        # raises a key error whose message contains the text saying caching is disabled.
        """Test that Memoizer.replay() always raises KeyError when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError regardless of what's in the cache.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()

        @memoizer.replay()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target returning double the input; never expected
            # to run.
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @set_caching_module_enabled(False)
    def test_memoizer_memoize_disabled_returns_original_function(self) -> None:
        # [FUNCTION DOC] With caching disabled, calls a memoize-decorated counting function
        # twice. Asserts it ran both times and returned the doubled value each time.
        """Test that Memoizer.memoize() returns original function when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a memoizer with caching disabled
        memoizer = Memoizer()
        call_count = 0

        @memoizer.memoize()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= PersistentMemoizer Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_record_caches_to_both(self) -> None:
        # [FUNCTION DOC] With caching enabled and the on-disk base directory patched, records a
        # call on a persistent memoizer. Asserts the result and call count, then looks up the
        # key in the in-memory cache (the stored entry has the encoded params as a mapping of
        # args and kwargs and the encoded result) and asserts the key is also present in the
        # disk cache.
        """Test that PersistentMemoizer.record() caches to both memory and disk.

        Verifies that when a function is decorated with record(), its result
        is cached in both the in-memory cache and the on-disk cache.
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function
        result = compute(5)

        # Assert: result is correct and cached in memory
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)

        # Verify memory cache has the result as CacheEntry
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Cache now stores CacheEntry with encoded_params and encoded_result
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)
        self.assertEqual(cache_entry.encoded_params, {"args": (5,), "kwargs": {}})

        # Verify disk cache has the result (pickled)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_replay_checks_memory_then_disk(self) -> None:
        # [FUNCTION DOC] With caching enabled, writes a pickled cache entry (encoded params and
        # result of ten) directly into the disk cache of a persistent memoizer, asserts the
        # memory cache has no entry, then calls a replay-decorated function that raises if
        # executed. Asserts the replay returns the stored result and that the memory cache is
        # now populated from disk.
        """Test that PersistentMemoizer.replay() checks memory first, then disk.

        Verifies that replay() uses a two-level cache strategy:
        1. Check memory cache first (fast)
        2. Fall back to disk cache on memory miss
        3. Populate memory cache from disk on disk hit
        """
        # Setup: create a persistent memoizer and store only to disk
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        # Store a value directly to disk cache only (as CacheEntry)
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        import pickle

        # Cache now stores CacheEntry with encoded_params and encoded_result
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result=10,
        )
        pickled_value = pickle.dumps(cache_entry)
        persistent._disk_cache.insert(cache_key, pickled_value)

        # Verify it's not in memory cache yet
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)

        # Create a replay function
        @persistent.replay()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if
            # ever executed.
            raise AssertionError("Function should not be executed during replay")

        # Execute: replay retrieves from disk and populates memory
        result = compute(5)

        # Assert: result is correct
        self.assertEqual(result, 10)

        # Verify memory cache was populated from disk
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        # Memory cache should now contain the CacheEntry
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_memoize_two_level_caching(self) -> None:
        # [FUNCTION DOC] With caching enabled, calls a memoize-decorated function three times
        # with the same argument: first executes it, second hits memory, then after swapping in
        # a fresh in-memory cache the third call is served from disk without executing. Asserts
        # the call count stays one and that memory is repopulated afterwards.
        """Test that PersistentMemoizer.memoize() uses two-level caching.

        Verifies that memoize() combines two-level caching behavior:
        - First call executes and caches to both memory and disk
        - Second call (same process) retrieves from memory
        - After clearing memory, retrieves from disk
        """
        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: first call - cache miss, executes function
        result1 = compute(5)
        self.assertEqual(call_count, 1)
        self.assertEqual(result1, 10)

        # Second call - memory cache hit
        result2 = compute(5)
        self.assertEqual(call_count, 1)  # No additional execution
        self.assertEqual(result2, 10)

        # Clear memory cache to simulate a new process
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        persistent._memoizer._cache = impls._InMemoryCacheImpl()

        # Third call - memory miss, disk hit, populates memory
        result3 = compute(5)
        self.assertEqual(call_count, 1)  # Still no additional execution
        self.assertEqual(result3, 10)

        # Verify memory cache was repopulated
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_record_disabled(self) -> None:
        # [FUNCTION DOC] With caching disabled, records two calls via a persistent memoizer.
        # Asserts the function ran both times and that neither the memory cache nor the disk
        # cache holds the key.
        """Test that PersistentMemoizer.record() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, record()
        returns the original function without any caching to memory or disk.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

        # Verify nothing was cached
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_replay_disabled(self) -> None:
        # [FUNCTION DOC] With caching disabled, asserts a persistent memoizer replay-decorated
        # function raises a key error mentioning that caching is disabled.
        """Test that PersistentMemoizer.replay() always raises when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, replay()
        always raises KeyError.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())

        @persistent.replay()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target returning double the input; never expected
            # to run.
            return x * 2

        # Execute & Assert: replay always raises KeyError when disabled
        with self.assertRaises(KeyError) as cm:
            compute(5)
        self.assertIn("Caching is disabled", str(cm.exception))

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_persistent_memoizer_memoize_disabled(self) -> None:
        # [FUNCTION DOC] With caching disabled, calls a persistent memoizer memoize-decorated
        # counting function twice and asserts it ran both times with correct results.
        """Test that PersistentMemoizer.memoize() returns original function when disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, memoize()
        returns the original function without any caching behavior.
        """
        # Setup: create a persistent memoizer with caching disabled
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @persistent.memoize()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function twice
        result1 = compute(5)
        result2 = compute(5)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertEqual(result1, 10)
        self.assertEqual(result2, 10)

    # ============= Cache Loading Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_loads_cache_from_dump_file(self) -> None:
        # [FUNCTION DOC] With caching enabled, writes a temporary JSON dump file holding a
        # collection under the null key with two entries, patches the dump file path config to
        # that file, creates a new Memoizer and asserts both entries are in its in-memory cache
        # with the expected encoded results (ten and twenty). Always deletes the temporary file
        # in a finally block.
        """Test that Memoizer loads cache entries from dump file on initialization.

        Verifies that when CACHE_DUMP_FILE_PATH is configured and the file exists,
        a new Memoizer instance pre-populates its in-memory cache with the dump contents.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with cache entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "collections": {
                    "null": {
                        "key1": {"params": {"x": 1}, "result": 10},
                        "key2": {"params": {"x": 2}, "result": 20},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            # Setup: Configure CACHE_DUMP_FILE_PATH
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer (should load from dump)
                memoizer = interfaces.Memoizer()

                # Assert: Cache was populated from dump file
                hit1 = memoizer._cache.get("key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 10)

                hit2 = memoizer._cache.get("key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 20)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_skips_loading_when_no_dump_file_configured(self) -> None:
        # [FUNCTION DOC] With the dump file path config patched to return none, creates a
        # Memoizer and asserts its in-memory cache holds no entries.
        """Test that Memoizer skips loading when CACHE_DUMP_FILE_PATH is not set.

        Verifies that when no dump file path is configured, the Memoizer
        initializes with an empty cache without errors.
        """
        # Setup: Configure CACHE_DUMP_FILE_PATH to return None
        with patch.object(config, "CACHE_DUMP_FILE_PATH", return_value=None):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (not loaded from any file)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_missing_dump_file_gracefully(self) -> None:
        # [FUNCTION DOC] With the dump file path config pointing to a file that does not exist,
        # creates a Memoizer and asserts it starts with an empty cache and does not raise.
        """Test that Memoizer handles missing dump file gracefully.

        Verifies that when CACHE_DUMP_FILE_PATH points to a non-existent file,
        the Memoizer initializes with an empty cache without crashing.
        """
        # Setup: Configure path to non-existent file
        non_existent_path = "/tmp/this_file_does_not_exist_12345.json"

        with patch.object(
            config, "CACHE_DUMP_FILE_PATH", return_value=non_existent_path
        ):
            # Execute: Create a new Memoizer
            memoizer = interfaces.Memoizer()

            # Assert: Cache is empty (no crash)
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_handles_corrupt_dump_file_gracefully(self) -> None:
        # [FUNCTION DOC] Writes a temporary file with invalid JSON, points the dump path config
        # at it, creates a Memoizer and asserts the in-memory cache is empty and no exception
        # escapes. Deletes the temporary file in a finally block.
        """Test that Memoizer handles corrupt dump file gracefully.

        Verifies that when the dump file contains invalid JSON, the Memoizer
        initializes with an empty cache without crashing.
        """
        import os
        import tempfile

        # Setup: Create a corrupt dump file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            tmp_file.write("{ this is not valid json")

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a new Memoizer
                memoizer = interfaces.Memoizer()

                # Assert: Cache is empty (no crash, handled gracefully)
                self.assertEqual(len(memoizer._cache._memory), 0)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_sub_key(self) -> None:
        # [FUNCTION DOC] Writes a temporary JSON dump with two collections (the one named
        # test_subdir with two entries and another with one). Creates a persistent memoizer
        # whose sub directory is that name and asserts its memory cache holds the two entries of
        # its own collection with results one hundred and two hundred and does not hold the
        # other collection's entry. Deletes the file afterwards.
        """Test that PersistentMemoizer loads cache from sub_dir nested structure.

        Verifies that when sub_dir is set, the PersistentMemoizer loads entries
        from the nested collections[sub_dir] structure.
        """
        import json
        import os
        import tempfile
        from pathlib import Path

        # Setup: Create a dump file with nested structure
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            dump_data = {
                "cache_size": 2,
                "collections": {
                    "test_subdir": {
                        "nested_key1": {"params": {"x": 1}, "result": 100},
                        "nested_key2": {"params": {"x": 2}, "result": 200},
                    },
                    "other_subdir": {
                        "other_key": {"params": {"x": 3}, "result": 300},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with sub_dir="test_subdir"
                pm = interfaces.PersistentMemoizer(sub_dir=Path("test_subdir"))

                # Assert: Cache loaded entries from test_subdir only
                hit1 = pm._memoizer._cache.get("nested_key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 100)

                hit2 = pm._memoizer._cache.get("nested_key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 200)

                # Assert: Did not load entries from other_subdir
                hit_other = pm._memoizer._cache.get("other_key")
                self.assertIsNone(hit_other)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_loads_from_root_when_sub_dir_empty(self) -> None:
        # [FUNCTION DOC] Writes a temporary JSON dump with a null collection (two entries) and a
        # named collection (one entry), creates a persistent memoizer with an empty sub
        # directory and asserts only the null collection entries are loaded (results ten and
        # twenty), not the named collection entry. Deletes the file afterwards.
        """Test that PersistentMemoizer loads from root when sub_dir is empty.

        Verifies that when sub_dir is empty string, entries are loaded from
        collections["null"] (the root collection).
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with mixed root and nested entries
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            # Note: "null" is the JSON representation of Python None
            # When sub_dir is empty string (""), the Memoizer's sub_key is None
            dump_data = {
                "cache_size": 3,
                "collections": {
                    "null": {
                        "root_key1": {"params": {"x": 1}, "result": 10},
                        "root_key2": {"params": {"x": 2}, "result": 20},
                    },
                    "some_subdir": {
                        "nested_key": {"params": {"x": 3}, "result": 30},
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create PersistentMemoizer with empty sub_dir
                pm = interfaces.PersistentMemoizer(sub_dir="")

                # Assert: Loaded root-level entries
                hit1 = pm._memoizer._cache.get("root_key1")
                self.assertIsNotNone(hit1)
                cache_entry1 = hit1.value
                self.assertEqual(cache_entry1.encoded_result, 10)

                hit2 = pm._memoizer._cache.get("root_key2")
                self.assertIsNotNone(hit2)
                cache_entry2 = hit2.value
                self.assertEqual(cache_entry2.encoded_result, 20)

                # Assert: Did not load nested entries
                hit_nested = pm._memoizer._cache.get("nested_key")
                self.assertIsNone(hit_nested)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_replay_uses_preloaded_cache(self) -> None:
        # [FUNCTION DOC] Writes a temporary JSON dump containing an entry under the key
        # generated for a call with argument five whose result is ten, loads a new Memoizer from
        # it with the dump path patched, and asserts a replay-decorated function (which raises
        # if executed) returns ten from the preloaded cache. Deletes the file afterwards.
        """Test that memoizer replay successfully retrieves from preloaded cache.

        Verifies end-to-end workflow: load cache from dump file, then use
        replay to retrieve cached results without executing the function.
        """
        import json
        import os
        import tempfile

        # Setup: Create a dump file with a cached result
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            test_filepath = tmp_file.name
            # Simulate a cache entry for compute(5) -> 10
            cache_key = interfaces._BaseMemoizer._make_key(None, 5)
            # Note: "null" is the JSON representation of Python None (no sub_key)
            dump_data = {
                "cache_size": 1,
                "collections": {
                    "null": {
                        cache_key: {
                            "params": {"args": (5,), "kwargs": {}},
                            "result": 10,
                        },
                    },
                },
            }
            json.dump(dump_data, tmp_file)

        try:
            with patch.object(
                config, "CACHE_DUMP_FILE_PATH", return_value=test_filepath
            ):
                # Execute: Create a memoizer (loads cache from dump)
                memoizer = interfaces.Memoizer()

                # Create a replay function
                @memoizer.replay()
                def compute(x: int) -> int:
                    # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if
                    # ever executed.
                    raise AssertionError(
                        "Function should not be executed during replay"
                    )

                # Execute: Call replay (should use preloaded cache)
                result = compute(5)

                # Assert: Got cached result without executing function
                self.assertEqual(result, 10)
        finally:
            # Cleanup
            if os.path.exists(test_filepath):
                os.unlink(test_filepath)

    # ============= Memoizer._dump_to_disk Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_creates_json_file(self) -> None:
        # [FUNCTION DOC] Records two calls with different arguments on a Memoizer, calls the
        # dump to disk method and asserts the shared cache file exists, is JSON with a
        # collections mapping and a cache size of two, has a null collection, and that every
        # entry in it has params and result fields.
        """Test that _dump_to_disk creates a JSON file with cached entries.

        Verifies that when _dump_to_disk is called, it creates a JSON file
        containing the cached entries in a human-readable format.
        """
        # Setup: create a memoizer and cache some values
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        compute(5)
        compute(10)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: JSON file was created with correct structure
        self.assertTrue(memoizer._shared_cache_filepath.exists())

        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertIn("cache_size", data)
        self.assertEqual(data["cache_size"], 2)

        # Note: "null" is the JSON representation of Python None (no sub_key)
        self.assertIn("null", data["collections"])

        # Verify entries have correct format
        for entry in data["collections"]["null"].values():
            self.assertIn("params", entry)
            self.assertIn("result", entry)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_with_sub_key(self) -> None:
        # [FUNCTION DOC] Creates a Memoizer with a sub key, records one call and dumps to disk.
        # Asserts the file's collections mapping contains the sub key with exactly one entry
        # having params and a result of ten.
        """Test that _dump_to_disk uses sub_key for nested structure.

        Verifies that when a Memoizer is initialized with a sub_key,
        the cache entries are stored under collections[sub_key].
        """
        # Setup: create a memoizer with sub_key and cache a value
        sub_key = "test_sub_key"
        memoizer = interfaces.Memoizer(sub_key=sub_key)

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        compute(5)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: entries are stored under the sub_key
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertIn(sub_key, data["collections"])

        # The sub_key should contain the cache entries
        sub_entries = data["collections"][sub_key]
        self.assertEqual(len(sub_entries), 1)

        # Verify entry format
        for entry in sub_entries.values():
            self.assertIn("params", entry)
            self.assertIn("result", entry)
            self.assertEqual(entry["result"], 10)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_merges_with_existing(self) -> None:
        # [FUNCTION DOC] Two separate Memoizers record different calls; the first dumps, then
        # the second dumps to the same shared file. Asserts the file reports a cache size of two
        # and the null collection has two entries, showing dumps merge additively.
        """Test that _dump_to_disk merges with existing cache data.

        Verifies that when multiple Memoizer instances dump to the same file,
        their entries are merged additively.
        """
        # Setup: create first memoizer and cache a value
        memoizer1 = interfaces.Memoizer()

        @memoizer1.record()
        def compute1(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        compute1(5)
        memoizer1._dump_to_disk()

        # Create second memoizer and cache a different value
        memoizer2 = interfaces.Memoizer()

        @memoizer2.record()
        def compute2(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns triple the input.
            return x * 3

        compute2(10)

        # Execute: dump second memoizer to disk
        memoizer2._dump_to_disk()

        # Assert: both entries are in the file
        with open(memoizer1._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertEqual(data["cache_size"], 2)
        # Note: "null" is the JSON representation of Python None (no sub_key)
        self.assertEqual(len(data["collections"]["null"]), 2)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_skips_empty_cache(self) -> None:
        # [FUNCTION DOC] Deletes the shared cache file if it exists, dumps an empty Memoizer and
        # asserts no file was created.
        """Test that _dump_to_disk does nothing when cache is empty.

        Verifies that when _dump_to_disk is called on an empty cache,
        no file is created.
        """
        # Setup: create a memoizer with no cached values
        memoizer = interfaces.Memoizer()

        # Ensure the file doesn't exist beforehand
        if memoizer._shared_cache_filepath.exists():
            memoizer._shared_cache_filepath.unlink()

        # Execute: dump the empty cache
        memoizer._dump_to_disk()

        # Assert: no file was created
        self.assertFalse(memoizer._shared_cache_filepath.exists())

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_handles_corrupt_file(self) -> None:
        # [FUNCTION DOC] Records one call, writes invalid JSON content to the shared cache file
        # (creating parent directories), then dumps to disk and asserts the file now contains
        # valid JSON with a collections mapping and a cache size of one, i.e. the corrupt
        # content was replaced.
        """Test that _dump_to_disk handles corrupt JSON files gracefully.

        Verifies that when the existing cache file contains invalid JSON,
        _dump_to_disk starts fresh and overwrites the corrupt file.
        """
        # Setup: create a memoizer and cache a value
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        compute(5)

        # Create a corrupt JSON file
        memoizer._shared_cache_filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(memoizer._shared_cache_filepath, "w") as f:
            f.write("{ invalid json content")

        # Execute: dump the cache (should handle corrupt file)
        memoizer._dump_to_disk()

        # Assert: file now contains valid JSON with our entry
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("collections", data)
        self.assertEqual(data["cache_size"], 1)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_stores_encoded_params_and_result(self) -> None:
        # [FUNCTION DOC] Records a call of a two-argument function (five and ten) on a Memoizer,
        # dumps to disk and asserts the single null-collection entry has a result of fifteen,
        # params args equal to the list five, ten, and empty params kwargs.
        """Test that _dump_to_disk stores both encoded params and result.

        Verifies that the dumped JSON contains both the encoded parameters
        and the encoded result for each cache entry, making it useful for debugging.
        """
        # Setup: create a memoizer with custom encoder and cache a value
        memoizer = interfaces.Memoizer()

        @memoizer.record()
        def compute(x: int, y: int) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return x + y

        compute(5, 10)

        # Execute: dump the cache to disk
        memoizer._dump_to_disk()

        # Assert: entry contains both params and result
        with open(memoizer._shared_cache_filepath) as f:
            data = json.load(f)

        # Get the single entry (under "null" since no sub_key was set)
        entries = data["collections"]["null"]
        self.assertEqual(len(entries), 1)

        entry = next(iter(entries.values()))
        self.assertEqual(entry["result"], 15)
        self.assertEqual(entry["params"]["args"], [5, 10])
        self.assertEqual(entry["params"]["kwargs"], {})

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_dump_to_disk_multiple_sub_keys(self) -> None:
        # [FUNCTION DOC] Creates two Memoizers with different sub keys, records one call on
        # each, dumps both to the same file and asserts both sub keys appear in the collections
        # mapping with one entry each, results ten and thirty respectively.
        """Test that multiple Memoizers with different sub_keys coexist.

        Verifies that Memoizers with different sub_keys store their entries
        under separate namespaces in the same JSON file.
        """
        # Setup: create two memoizers with different sub_keys
        memoizer1 = interfaces.Memoizer(sub_key="feature_a")
        memoizer2 = interfaces.Memoizer(sub_key="feature_b")

        @memoizer1.record()
        def compute_a(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        @memoizer2.record()
        def compute_b(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns triple the input.
            return x * 3

        compute_a(5)
        compute_b(10)

        # Execute: dump both caches
        memoizer1._dump_to_disk()
        memoizer2._dump_to_disk()

        # Assert: both sub_keys exist with their respective entries
        with open(memoizer1._shared_cache_filepath) as f:
            data = json.load(f)

        self.assertIn("feature_a", data["collections"])
        self.assertIn("feature_b", data["collections"])

        # Verify each sub_key has one entry with correct result
        feature_a_entries = data["collections"]["feature_a"]
        feature_b_entries = data["collections"]["feature_b"]

        self.assertEqual(len(feature_a_entries), 1)
        self.assertEqual(len(feature_b_entries), 1)

        self.assertEqual(next(iter(feature_a_entries.values()))["result"], 10)
        self.assertEqual(next(iter(feature_b_entries.values()))["result"], 30)


@instantiate_parametrized_tests
class ForceDisableCachesTest(TestMixin, TestCase):
    # [CLASS DOC] Tests that the force-disable-caches setting turns the caching module off. Each
    # test patches the base enabled check and the force-disable check of the config module; uses
    # a class specific sub directory name.
    """Test class for force_disable_caches integration with the caching module."""

    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-force-disable-caches-{cls.cls_id}"

    def test_force_disable_caches_disables_caching_module(self) -> None:
        # [FUNCTION DOC] Patches the base enabled check to true and the force-disable check to
        # true, and asserts the caching-module-enabled function returns false.
        """Test that force_disable_caches=True disables the caching module.

        Verifies that when torch._inductor.config.force_disable_caches is True,
        IS_CACHING_MODULE_ENABLED() returns False even if the base config
        would otherwise enable caching.
        """
        # Setup: patch the base config to return True (enabled)
        # and force_disable_caches to return True
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=True),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return False
            self.assertFalse(config.IS_CACHING_MODULE_ENABLED())

    def test_caching_module_enabled_when_force_disable_is_false(self) -> None:
        # [FUNCTION DOC] Patches the base enabled check to true and the force-disable check to
        # false, and asserts the caching-module-enabled function returns true.
        """Test that caching works when force_disable_caches=False.

        Verifies that when force_disable_caches is False and the base config
        enables caching, IS_CACHING_MODULE_ENABLED() returns True.
        """
        # Setup: patch both configs to enabled states
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=False),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return True
            self.assertTrue(config.IS_CACHING_MODULE_ENABLED())

    def test_caching_module_disabled_when_base_config_is_false(self) -> None:
        # [FUNCTION DOC] Patches the base enabled check to false and the force-disable check to
        # false, and asserts the caching-module-enabled function returns false.
        """Test that caching is disabled when base config is False.

        Verifies that when the base versioned config returns False,
        IS_CACHING_MODULE_ENABLED() returns False regardless of force_disable_caches.
        """
        # Setup: patch base config to False, force_disable to False
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=False),
            patch.object(config, "_is_force_disable_caches", return_value=False),
        ):
            # Execute & Assert: IS_CACHING_MODULE_ENABLED should return False
            self.assertFalse(config.IS_CACHING_MODULE_ENABLED())

    @patch_on_disk_cache_base_dir
    def test_memoizer_disabled_when_force_disable_caches_true(self) -> None:
        # [FUNCTION DOC] With the on-disk base directory patched, and base enabled true plus
        # force-disable true, calls a memoize-decorated counting function twice on a Memoizer.
        # Asserts it executed both times and returned the doubled value, i.e. memoization is a
        # no-op under force-disable.
        """Test that Memoizer operations become no-ops when force_disable_caches=True.

        Verifies that when force_disable_caches is enabled, memoized functions
        always execute (no caching) and replay always raises KeyError.
        """
        # Setup: enable force_disable_caches
        with (
            patch.object(config, "_is_caching_module_enabled_base", return_value=True),
            patch.object(config, "_is_force_disable_caches", return_value=True),
        ):
            memoizer = Memoizer()
            call_count = 0

            @memoizer.memoize()
            def compute(x: int) -> int:
                # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns
                # double the input.
                nonlocal call_count
                call_count += 1
                return x * 2

            # Execute: call twice - should execute both times (no caching)
            result1 = compute(5)
            result2 = compute(5)

            # Assert: function was called both times
            self.assertEqual(call_count, 2)
            self.assertEqual(result1, 10)
            self.assertEqual(result2, 10)


@instantiate_parametrized_tests
class FreshCacheIntegrationTest(TestMixin, TestCase):
    # [CLASS DOC] Tests the integration of Memoizer and PersistentMemoizer with cache clearing:
    # the cache_clear methods and the fresh cache context manager of the inductor utils. Uses a
    # class specific sub directory; the class fixtures delete its on-disk cache directory before
    # and after the tests.
    """Test class for fresh_cache integration with the caching module."""

    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-fresh-cache-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) before the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) after the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @set_caching_module_enabled(True)
    def test_memoizer_cache_clear_clears_in_memory_cache(self) -> None:
        # [FUNCTION DOC] Records two calls on a Memoizer and asserts the in-memory cache holds
        # two entries, clears the Memoizer, and asserts the memory cache is empty. Then asserts
        # a replay-decorated function for a previously recorded argument raises a key error.
        """Test that Memoizer.cache_clear() clears the in-memory cache.

        Verifies that calling cache_clear() on a Memoizer instance resets
        its in-memory cache to empty.
        """
        # Setup: create a memoizer and cache some values
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        compute(10)

        # Verify cache has entries
        self.assertEqual(len(memoizer._cache._memory), 2)
        self.assertEqual(call_count, 2)

        # Execute: clear the cache
        memoizer.cache_clear()

        # Assert: cache is now empty
        self.assertEqual(len(memoizer._cache._memory), 0)

        # Verify that calling the memoized function again would be a cache miss
        # by checking that replay raises KeyError
        @memoizer.replay()
        def compute_replay(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if executed;
            # never expected to run because replay raises a key error first.
            raise AssertionError("Should not be called")

        with self.assertRaises(KeyError):
            compute_replay(5)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_cache_clear_clears_disk_cache(self) -> None:
        # [FUNCTION DOC] Creates a persistent memoizer for the class sub directory, records one
        # call, asserts the disk cache directory exists, then clears the persistent memoizer and
        # asserts the directory no longer exists.
        """Test that PersistentMemoizer.cache_clear() clears the on-disk cache.

        Verifies that calling cache_clear() on a PersistentMemoizer removes
        the on-disk cache directory.
        """
        from pathlib import Path

        # Setup: create a persistent memoizer and cache some values
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        self.assertEqual(call_count, 1)

        # Verify disk cache directory exists and has content
        disk_cache_dir = persistent._disk_cache._cache_dir
        self.assertTrue(disk_cache_dir.exists())

        # Execute: clear the persistent memoizer cache
        persistent.cache_clear()

        # Assert: disk cache directory is removed
        self.assertFalse(disk_cache_dir.exists())

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_memoizer_registered_with_clear_on_fresh_cache(self) -> None:
        # [FUNCTION DOC] Creates a Memoizer and asserts it appears in the registry of caches
        # that the fresh cache context manager clears.
        """Test that Memoizer is registered with clear_on_fresh_cache.

        Verifies that when a Memoizer is created, it is automatically
        registered to be cleared when fresh_cache() is invoked.
        """
        from torch._inductor.utils import _registered_caches

        # Setup: create a memoizer
        memoizer = Memoizer()

        # Assert: memoizer is in the registered caches list
        self.assertIn(memoizer, _registered_caches)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_registered_with_clear_on_fresh_cache(self) -> None:
        # [FUNCTION DOC] Creates a persistent memoizer and asserts both it and its inner
        # memoizer appear in the registry of caches cleared by the fresh cache context manager.
        """Test that PersistentMemoizer is registered with clear_on_fresh_cache.

        Verifies that when a PersistentMemoizer is created, it is automatically
        registered to be cleared when fresh_cache() is invoked.
        """
        from pathlib import Path

        from torch._inductor.utils import _registered_caches

        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))

        # Assert: persistent memoizer is in the registered caches list
        self.assertIn(persistent, _registered_caches)
        # Also verify the underlying memoizer is registered
        self.assertIn(persistent._memoizer, _registered_caches)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_fresh_cache_clears_memoizer(self) -> None:
        # [FUNCTION DOC] Records one call on a Memoizer, asserts one entry in memory, then
        # enters the fresh cache context manager and asserts the memoizer's in-memory cache is
        # empty inside it.
        """Test that fresh_cache() context manager clears Memoizer caches.

        Verifies that when entering fresh_cache() context, all registered
        Memoizer instances have their caches cleared.
        """
        from torch._inductor.utils import fresh_cache

        # Setup: create a memoizer and cache some values
        memoizer = Memoizer()
        call_count = 0

        @memoizer.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)
        self.assertEqual(len(memoizer._cache._memory), 1)

        # Execute: enter fresh_cache context (this calls clear_caches())
        with fresh_cache():
            # Assert: memoizer cache was cleared
            self.assertEqual(len(memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_fresh_cache_clears_persistent_memoizer_disk_cache(self) -> None:
        # [FUNCTION DOC] Records one call on a persistent memoizer, asserts the disk cache
        # directory exists, then inside the fresh cache context manager asserts the directory is
        # gone and the inner memoizer's memory cache is empty.
        """Test that fresh_cache() clears PersistentMemoizer disk cache.

        Verifies that when entering fresh_cache() context, PersistentMemoizer
        instances have their on-disk cache directories removed.
        """
        from pathlib import Path

        from torch._inductor.utils import fresh_cache

        # Setup: create a persistent memoizer and cache some values
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))
        call_count = 0

        @persistent.record()
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        compute(5)

        # Verify disk cache exists
        disk_cache_dir = persistent._disk_cache._cache_dir
        self.assertTrue(disk_cache_dir.exists())

        # Execute: enter fresh_cache context
        with fresh_cache():
            # Assert: disk cache directory was removed
            self.assertFalse(disk_cache_dir.exists())
            # And memory cache was cleared (via the underlying Memoizer)
            self.assertEqual(len(persistent._memoizer._cache._memory), 0)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_cache_clear_is_idempotent(self) -> None:
        # [FUNCTION DOC] Calls the clear method twice in a row on an empty Memoizer and on an
        # empty persistent memoizer, asserting no error is raised, the memory cache is empty and
        # the disk cache directory does not exist.
        """Test that calling cache_clear() multiple times is safe.

        Verifies that clearing an already-empty cache does not raise errors.
        """
        from pathlib import Path

        # Setup: create memoizers
        memoizer = Memoizer()
        persistent = PersistentMemoizer(sub_dir=Path(self.sub_dir()))

        # Execute: call cache_clear multiple times on empty caches
        memoizer.cache_clear()
        memoizer.cache_clear()

        persistent.cache_clear()
        persistent.cache_clear()

        # Assert: no errors raised, caches are still empty
        self.assertEqual(len(memoizer._cache._memory), 0)
        self.assertFalse(persistent._disk_cache._cache_dir.exists())


@instantiate_parametrized_tests
class ShouldPadMemoizerTest(TestMixin, TestCase):
    # [CLASS DOC] Tests the persistent memoizer as applied to the pad decision function, using
    # the should-pad parameter encoder. A mocked pattern match object supplies placeholder
    # matrix nodes; test functions are stand-ins that count calls. Verifies caching is keyed on
    # tensor metadata (shape, dtype, op, optional input) and not on tensor data. Uses a class
    # specific sub directory; the class fixtures delete its on-disk cache directory before and
    # after the tests.
    """Test class for _should_pad memoizer integration.

    These tests verify that the PersistentMemoizer applied to _should_pad
    correctly memoizes and replays results based on tensor metadata and
    operation parameters.
    """

    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-should-pad-memoizer-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) before the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) after the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    def _create_mock_match(self) -> Any:
        # [FUNCTION DOC] HELPER: Builds and returns a mock pattern-match object whose keyword
        # arguments mapping holds two mock graph nodes, named mat1 and mat2, each with node kind
        # placeholder. Provides the minimal interface the should-pad parameter encoder needs.
        """Create a mock Match object for testing.

        Returns a mock that simulates the Match object from pattern_matcher,
        providing the minimal interface needed for the should_pad_params_encoder.
        """
        from unittest.mock import MagicMock

        mock_match = MagicMock()

        # Create mock FX nodes for mat1 and mat2 kwargs
        mock_mat1_node = MagicMock()
        mock_mat1_node.op = "placeholder"

        mock_mat2_node = MagicMock()
        mock_mat2_node.op = "placeholder"

        mock_match.kwargs = {
            "mat1": mock_mat1_node,
            "mat2": mock_mat2_node,
        }

        return mock_match

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_caches_result(self, mock_is_contiguous) -> None:
        # [FUNCTION DOC] Memoizes a stand-in pad function through a fresh persistent memoizer
        # with the should-pad parameter encoder (the contiguity check is patched to return
        # true). Calls it twice with identical mock match, matrices and op, asserts the stand-in
        # ran once and both calls returned true.
        """Test that the should_pad_memoizer caches function results.

        Verifies that when a function decorated with should_pad_memoizer.memoize
        is called, the result is cached and can be retrieved on subsequent calls.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in for the pad decision; increments a nonlocal call
            # counter and returns true.
            nonlocal call_count
            call_count += 1
            return True

        # Create test inputs
        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: call the function twice with the same parameters
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)

        result2 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)  # Should use cached result

        # Assert: both calls return the same result
        self.assertTrue(result1)
        self.assertTrue(result2)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_shapes_different_cache_entries(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Memoizes a stand-in whose result depends on the first matrix's row
        # count. Calls it with a small and a large first matrix and asserts two executions
        # (different shapes are different cache keys), with a false result for the small and a
        # true result for the large one.
        """Test that different tensor shapes result in different cache entries.

        Verifies that the encoder correctly distinguishes between tensors with
        different shapes, ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns whether the first matrix has more than ten rows.
            nonlocal call_count
            call_count += 1
            # Return different values based on shape to verify no cross-caching
            return mat1.shape[0] > 10

        mock_match = self._create_mock_match()

        # Execute: call with different shapes
        mat1_small = torch.randn(8, 16, dtype=torch.float32)
        mat2_small = torch.randn(16, 32, dtype=torch.float32)

        mat1_large = torch.randn(12, 16, dtype=torch.float32)
        mat2_large = torch.randn(16, 32, dtype=torch.float32)

        op = torch.ops.aten.mm

        result_small = mock_should_pad(mock_match, mat1_small, mat2_small, op)
        self.assertEqual(call_count, 1)

        result_large = mock_should_pad(mock_match, mat1_large, mat2_large, op)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on shape)
        self.assertFalse(result_small)  # 8 <= 10
        self.assertTrue(result_large)  # 12 > 10

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_dtypes_different_cache_entries(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Memoizes a stand-in whose result depends on the first matrix's dtype.
        # Calls it with float32 and float16 inputs of identical shapes and asserts two
        # executions, with a true result for float32 and false for float16.
        """Test that different tensor dtypes result in different cache entries.

        Verifies that the encoder correctly distinguishes between tensors with
        different dtypes, ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns whether the first matrix is float32.
            nonlocal call_count
            call_count += 1
            return mat1.dtype == torch.float32

        mock_match = self._create_mock_match()
        op = torch.ops.aten.mm

        # Execute: call with different dtypes but same shapes
        mat1_fp32 = torch.randn(8, 16, dtype=torch.float32)
        mat2_fp32 = torch.randn(16, 32, dtype=torch.float32)

        mat1_fp16 = torch.randn(8, 16, dtype=torch.float16)
        mat2_fp16 = torch.randn(16, 32, dtype=torch.float16)

        result_fp32 = mock_should_pad(mock_match, mat1_fp32, mat2_fp32, op)
        self.assertEqual(call_count, 1)

        result_fp16 = mock_should_pad(mock_match, mat1_fp16, mat2_fp16, op)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on dtype)
        self.assertTrue(result_fp32)
        self.assertFalse(result_fp16)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_different_ops_different_cache_entries(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Memoizes a stand-in that returns whether the op is the matmul op. Calls
        # it with the same matrices but the mm op and then the addmm op, asserting two
        # executions and results true then false.
        # KNOWN ISSUE: the docstring says mm versus bmm, but the test uses mm versus addmm; also
        # the recorded list of ops is filled but never checked.
        """Test that different operations result in different cache entries.

        Verifies that the encoder correctly distinguishes between different
        operation types (mm vs bmm), ensuring they don't share cached results.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0
        call_ops: list[Any] = []

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter, appends
            # the op to a list, and returns whether the op is the mm op.
            nonlocal call_count
            call_count += 1
            call_ops.append(op)
            return op is torch.ops.aten.mm

        mock_match = self._create_mock_match()

        # Create 2D tensors for mm
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)

        # Execute: call with different operations
        result_mm = mock_should_pad(mock_match, mat1, mat2, torch.ops.aten.mm)
        self.assertEqual(call_count, 1)

        result_addmm = mock_should_pad(mock_match, mat1, mat2, torch.ops.aten.addmm)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different (based on op)
        self.assertTrue(result_mm)
        self.assertFalse(result_addmm)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_replays_from_disk_cache(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Calls a memoized stand-in once, replaces the inner memoizer's in-memory
        # cache with a new empty one to simulate a new process, calls again and asserts the
        # stand-in ran only once because the second result was served from disk.
        """Test that the memoizer replays results from disk cache after memory clear.

        Verifies that PersistentMemoizer correctly stores results to disk and
        can replay them after the in-memory cache is cleared.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns true.
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: cache a result
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)
        self.assertTrue(result1)

        # Clear the in-memory cache to simulate a new process
        test_memoizer._memoizer._cache = impls._InMemoryCacheImpl()

        # Execute: call again - should replay from disk
        result2 = mock_should_pad(mock_match, mat1, mat2, op)
        self.assertEqual(call_count, 1)  # Function should NOT be called again
        self.assertTrue(result2)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(False)
    def test_should_pad_memoizer_disabled_does_not_cache(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] With caching disabled, calls a memoized stand-in twice with the same
        # inputs and asserts it executed both times.
        """Test that the memoizer does not cache when caching is disabled.

        Verifies that when IS_CACHING_MODULE_ENABLED is False, the function
        is called every time without caching.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns true.
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: call the function twice
        result1 = mock_should_pad(mock_match, mat1, mat2, op)
        result2 = mock_should_pad(mock_match, mat1, mat2, op)

        # Assert: function was called both times (no caching)
        self.assertEqual(call_count, 2)
        self.assertTrue(result1)
        self.assertTrue(result2)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_with_input_tensor(self, mock_is_contiguous) -> None:
        # [FUNCTION DOC] Memoizes a stand-in returning whether the optional input is present.
        # Calls it with the addmm op once with an input tensor and once with no input, asserting
        # two executions (distinct cache keys), with results true and false.
        """Test that the memoizer correctly handles the optional input tensor.

        Verifies that different input tensors (for addmm) result in different
        cache entries, and that None vs non-None input are distinguished.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns whether the optional input tensor is not none.
            nonlocal call_count
            call_count += 1
            return input is not None

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        input_tensor = torch.randn(32, dtype=torch.float32)
        op = torch.ops.aten.addmm

        # Execute: call with and without input tensor
        result_with_input = mock_should_pad(
            mock_match, mat1, mat2, op, input=input_tensor
        )
        self.assertEqual(call_count, 1)

        result_without_input = mock_should_pad(mock_match, mat1, mat2, op, input=None)
        self.assertEqual(call_count, 2)  # Should be a cache miss

        # Assert: results are different
        self.assertTrue(result_with_input)
        self.assertFalse(result_without_input)

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_params_encoder_produces_consistent_keys(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Calls the should-pad parameter encoder twice with identical arguments
        # and asserts the two encodings are equal; also asserts the encoding has matrix, matrix
        # and op entries, and that the recorded shapes of both matrices equal the tensor shapes.
        """Test that the encoder produces consistent keys for the same inputs.

        Verifies that calling the encoder with the same tensor metadata produces
        the same cache key, ensuring reliable cache hits.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        mock_match = self._create_mock_match()
        mat1 = torch.randn(8, 16, dtype=torch.float32)
        mat2 = torch.randn(16, 32, dtype=torch.float32)
        op = torch.ops.aten.mm

        # Execute: encode the same parameters multiple times
        encoded1 = encoders.should_pad_params_encoder(mock_match, mat1, mat2, op)
        encoded2 = encoders.should_pad_params_encoder(mock_match, mat1, mat2, op)

        # Assert: encodings are identical
        self.assertEqual(encoded1, encoded2)

        # Also verify the structure of the encoded output
        self.assertIn("mat1", encoded1)
        self.assertIn("mat2", encoded1)
        self.assertIn("op", encoded1)
        self.assertEqual(encoded1["mat1"]["shape"], tuple(mat1.shape))
        self.assertEqual(encoded1["mat2"]["shape"], tuple(mat2.shape))

    @patch("torch._prims_common.is_contiguous_or_false", return_value=True)
    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_should_pad_memoizer_same_shape_different_data_uses_cache(
        self, mock_is_contiguous
    ) -> None:
        # [FUNCTION DOC] Memoizes a stand-in that returns true. Calls it with two sets of random
        # matrices having identical shapes and dtype but different data, and asserts the
        # stand-in ran only once, showing keys depend on metadata, not values.
        """Test that tensors with the same metadata but different data share cache.

        Verifies that the memoizer caches based on tensor metadata (shape, stride,
        dtype) and not on the actual tensor data values.
        """
        import torch
        from torch._inductor.runtime.caching import encoders

        # Setup: create a new memoizer instance for isolation
        test_memoizer = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0

        @test_memoizer.memoize(custom_params_encoder=encoders.should_pad_params_encoder)
        def mock_should_pad(
            match: Any,
            mat1: torch.Tensor,
            mat2: torch.Tensor,
            op: Any,
            input: torch.Tensor | None = None,
        ) -> bool:
            # [FUNCTION DOC] HELPER: Stand-in that increments a nonlocal call counter and
            # returns true.
            nonlocal call_count
            call_count += 1
            return True

        mock_match = self._create_mock_match()
        op = torch.ops.aten.mm

        # Execute: call with different tensors that have the same metadata
        mat1_a = torch.randn(8, 16, dtype=torch.float32)
        mat2_a = torch.randn(16, 32, dtype=torch.float32)

        mat1_b = torch.randn(8, 16, dtype=torch.float32)  # Different data, same shape
        mat2_b = torch.randn(16, 32, dtype=torch.float32)  # Different data, same shape

        result1 = mock_should_pad(mock_match, mat1_a, mat2_a, op)
        self.assertEqual(call_count, 1)

        result2 = mock_should_pad(mock_match, mat1_b, mat2_b, op)
        self.assertEqual(call_count, 1)  # Should use cached result

        # Assert: both return the same cached result
        self.assertTrue(result1)
        self.assertTrue(result2)


@instantiate_parametrized_tests
class DeferredRecordingTest(TestMixin, TestCase):
    # [CLASS DOC] Tests for deferred recording: a custom result encoder may return a
    # deferred-recording object, in which case the memoizer does not cache immediately but
    # registers a callback that stores the entry once the object is finalized with the encoded
    # result. Also covers interim results for repeated calls while pending, the encoder and
    # decoder factories receiving the unwrapped function, callback ordering and thread safety.
    # Uses a class specific sub directory; the class fixtures delete its on-disk cache directory
    # before and after the tests.
    """Test class for DeferredRecording functionality."""

    @classmethod
    def sub_dir(cls) -> str:
        # [FUNCTION DOC] HELPER: Class method returning the sub directory name for this test
        # class, containing the class id.
        return f"testing-deferred-recording-{cls.cls_id}"

    @classmethod
    def setUpClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) before the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        # [FUNCTION DOC] HELPER: Class fixture that deletes the on-disk cache directory of this
        # class's sub directory (errors ignored) after the tests.
        rmtree(
            impls._OnDiskCacheImpl(sub_dir=cls.sub_dir())._cache_dir,
            ignore_errors=True,
        )

    # ============= DeferredRecording Basic Tests =============

    @set_caching_module_enabled(True)
    def test_deferred_recording_finalize_caches_result(self) -> None:
        # [FUNCTION DOC] Records a call whose result encoder returns a deferred recording and
        # asserts the function ran once, returned the plain result, and that the memoizer
        # cache has no entry yet. After finalizing the captured deferred object with an
        # encoded value of one hundred, asserts the cache now holds an entry whose encoded
        # result is one hundred.
        """Test that DeferredRecording.finalize() caches the result.

        Verifies that when an encoder returns a DeferredRecording and
        finalize() is called later, the result is cached properly.
        """
        # Setup: create a memoizer and a custom encoder that returns DeferredRecording
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a new deferred
                    # recording, stores it in an enclosing variable for the test to finalize,
                    # and returns it.
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        call_count = 0

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Execute: call the function (should NOT cache immediately)
        result = compute(5)
        self.assertEqual(result, 10)
        self.assertEqual(call_count, 1)

        # Verify: cache should be empty since we returned DeferredRecording
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNone(cache_hit)

        # Execute: finalize the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)  # Cache encoded value 100

        # Assert: cache should now have the result
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertEqual(cache_entry.encoded_result, 100)

    @set_caching_module_enabled(True)
    def test_deferred_recording_replay_after_complete(self) -> None:
        # [FUNCTION DOC] Records via a deferred encoder and replays on the same memoizer.
        # Asserts replay raises a key error before the deferred object is finalized, then
        # finalizes it with ten and asserts the replay returns ten.
        """Test that replay works after deferred recording completes.

        Verifies end-to-end flow: record with deferred -> complete -> replay.
        """
        # Setup: create a memoizer with deferred recording
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a new deferred
                    # recording, stores it in an enclosing variable for the test to finalize,
                    # and returns it.
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute_record(x: int) -> int:
            # [FUNCTION DOC] HELPER: Record target returning double the input.
            return x * 2

        @memoizer.replay()
        def compute_replay(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if executed.
            raise AssertionError("Should not be called during replay")

        # Execute: record with deferred
        compute_record(5)

        # Verify: replay should fail before complete
        with self.assertRaises(KeyError):
            compute_replay(5)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(10)

        # Assert: replay should now work
        result = compute_replay(5)
        self.assertEqual(result, 10)

    @set_caching_module_enabled(True)
    def test_deferred_recording_with_future_pattern(self) -> None:
        # [FUNCTION DOC] Simulates a future based flow. The encoder attaches a done callback to
        # the returned future that finalizes the deferred recording with the future's result.
        # After calling the function, the test registers its own callback on the memoizer's
        # pending deferred recording (or sets its event directly if it is already gone) to learn
        # when the cache is populated, waits up to five seconds on the future and the event, and
        # asserts the cache entry holds encoded result ten.
        """Test DeferredRecording with a Future-like pattern.

        Simulates the real use case where a Future callback completes
        the deferred recording.
        """
        # Setup: create a memoizer and simulate a Future
        memoizer = Memoizer()
        cache_populated = Event()

        def future_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def future_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(future_result: Future[int]) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function; creates a deferred recording,
                    # registers a done callback on the given future that finalizes the recording
                    # with the future result, and returns the recording.
                    deferred = interfaces.DeferredRecording()

                    def on_complete(completed_future: Future[int]) -> None:
                        # [FUNCTION DOC] HELPER: Done callback that reads the completed future's
                        # result and finalizes the deferred recording with it.
                        actual_result = completed_future.result()
                        deferred.finalize(actual_result)

                    future_result.add_done_callback(on_complete)
                    return deferred

                return encode

            return future_encoder

        with ThreadPoolExecutor() as executor:

            @memoizer.record(custom_result_encoder=future_encoder_factory)
            def compute_async(x: int) -> Future[int]:
                # [FUNCTION DOC] HELPER: Submits a task to the thread pool computing double the
                # input and returns its future.
                return executor.submit(lambda: x * 2)

            # Execute: call the function and get a future
            future = compute_async(5)

            # Register our own callback on the DeferredRecording to know when
            # the cache is populated. This must be done AFTER compute_async()
            # returns, so our callback is registered after the memoizer's callback.
            # This ensures our callback runs after the cache is populated.
            cache_key = interfaces._BaseMemoizer._make_key(None, 5)
            pending = memoizer._pending_deferred.get(cache_key)
            if pending is not None:
                # Register callback - will be called after memoizer's callback
                # (either when complete() runs, or immediately if already completed)
                pending.register_callback(lambda _: cache_populated.set())
            else:
                # Deferred already completed and removed from tracking.
                # This means complete() and callbacks already ran, cache is populated.
                cache_populated.set()

            # Wait for the future to complete
            result = future.result(timeout=5)
            self.assertEqual(result, 10)

            # Wait for the cache to be populated (with timeout)
            self.assertTrue(
                cache_populated.wait(timeout=5),
                "Cache was not populated within timeout",
            )

        # Assert: cache should have the result after future completed
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertEqual(cache_entry.encoded_result, 10)

    @set_caching_module_enabled(True)
    def test_deferred_recording_preserves_original_return(self) -> None:
        # [FUNCTION DOC] Uses a deferred encoder that returns a never-finalized deferred
        # recording, calls the function and asserts the caller still receives the original
        # integer result (not the deferred object).
        """Test that deferred recording returns the original result.

        Verifies that even when using deferred recording, the function
        returns the original (non-encoded) result immediately.
        """
        # Setup
        memoizer = Memoizer()

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that returns a fresh deferred
                    # recording without keeping a reference.
                    return interfaces.DeferredRecording()

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Execute: call the function
        result = compute(5)

        # Assert: should return the original result (not the DeferredRecording)
        self.assertEqual(result, 10)
        self.assertIsInstance(result, int)

    @set_caching_module_enabled(True)
    def test_deferred_recording_finalize_without_callback(self) -> None:
        # [FUNCTION DOC] Creates a standalone deferred recording and finalizes it with a value
        # although no callback is registered; asserts no exception is raised.
        """Test that finalize() is safe when no callback is set.

        Verifies that calling finalize() on a DeferredRecording without
        a registered callback does not raise an error.
        """
        # Setup: create a standalone DeferredRecording
        deferred = interfaces.DeferredRecording()

        # Execute & Assert: finalize should not raise
        deferred.finalize(42)  # Should do nothing but not raise

    # ============= PersistentMemoizer DeferredRecording Tests =============

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_deferred_recording_caches_to_both(self) -> None:
        # [FUNCTION DOC] Same flow as the memoizer test but on a persistent memoizer: after the
        # call both memory and disk caches have no entry; after finalizing with one hundred, the
        # memory entry holds encoded result one hundred and the disk cache also has the key.
        """Test that PersistentMemoizer deferred recording caches to both memory and disk.

        Verifies that when complete() is called, the result is stored
        in both the in-memory cache and the on-disk cache.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a new deferred
                    # recording, stores it in an enclosing variable for the test to finalize,
                    # and returns it.
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @persistent.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Execute: call the function (should NOT cache immediately)
        result = compute(5)
        self.assertEqual(result, 10)

        # Verify: both caches should be empty
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNone(memory_hit)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNone(disk_hit)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # Assert: memory cache should have the result
        memory_hit = persistent._memoizer._cache.get(cache_key)
        self.assertIsNotNone(memory_hit)
        cache_entry = memory_hit.value
        self.assertEqual(cache_entry.encoded_result, 100)

        # Assert: disk cache should also have the result
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_deferred_replays_from_disk(self) -> None:
        # [FUNCTION DOC] Records through a persistent memoizer with a deferred encoder,
        # finalizes with one hundred, replaces the in-memory cache with an empty one, then
        # replays with a function that raises if executed. Asserts the replay returns one
        # hundred, served from disk.
        """Test that deferred recording can be replayed from disk cache.

        Verifies that after completing a deferred recording, the result
        can be replayed even after clearing the memory cache.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a new deferred
                    # recording, stores it in an enclosing variable for the test to finalize,
                    # and returns it.
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @persistent.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Record with deferred
        compute(5)
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # Clear memory cache to simulate new process
        persistent._memoizer._cache = impls._InMemoryCacheImpl()

        # Create replay function
        @persistent.replay()
        def compute_replay(x: int) -> int:
            # [FUNCTION DOC] HELPER: Replay target that raises an assertion error if executed.
            raise AssertionError("Should not be called during replay")

        # Execute: replay should work from disk
        result = compute_replay(5)

        # Assert
        self.assertEqual(result, 100)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_memoize_with_deferred(self) -> None:
        # [FUNCTION DOC] Uses memoize on a persistent memoizer with a deferred encoder that
        # collects every recording in a list. Two calls before any finalization both execute the
        # function (count two, no interim result). After finalizing the first recording with
        # ten, a third call returns ten without executing (count stays two).
        """Test that memoize() works correctly with deferred recording.

        Verifies that when using memoize() with a deferred encoder,
        subsequent calls work correctly after the deferred completes.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        deferred_objects: list[interfaces.DeferredRecording] = []
        call_count = 0

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording,
                    # appends it to a list owned by the test and returns it.
                    deferred = interfaces.DeferredRecording()
                    deferred_objects.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @persistent.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call before complete: should execute again (cache miss)
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)

        # Complete the first deferred recording
        deferred_objects[0].finalize(10)

        # Third call after complete: should use cache
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 2)  # No additional call

    # ============= Edge Cases =============

    @set_caching_module_enabled(True)
    def test_deferred_recording_with_none_result(self) -> None:
        # [FUNCTION DOC] Records a function returning nothing with a deferred encoder, finalizes
        # with none, and asserts a cache entry exists whose encoded result is none.
        """Test that deferred recording works with None as the result.

        Verifies that complete(None) caches None correctly.
        """
        # Setup
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a new deferred
                    # recording, stores it in an enclosing variable for the test to finalize,
                    # and returns it.
                    nonlocal deferred_obj
                    deferred_obj = interfaces.DeferredRecording()
                    return deferred_obj

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> None:
            # [FUNCTION DOC] HELPER: Returns nothing.
            return None

        # Execute
        compute(5)
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(None)

        # Assert: cache should have None as the result
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_hit = memoizer._cache.get(cache_key)
        self.assertIsNotNone(cache_hit)
        cache_entry = cache_hit.value
        self.assertIsNone(cache_entry.encoded_result)

    @set_caching_module_enabled(True)
    def test_deferred_recording_multiple_calls_different_keys(self) -> None:
        # [FUNCTION DOC] Records three calls with different arguments, each producing its own
        # deferred recording stored in a mapping keyed by the first argument. Finalizes them in
        # reverse order with the doubled values and asserts each key's cache entry holds the
        # matching encoded result, showing recordings are independent.
        """Test that multiple deferred recordings with different keys work.

        Verifies that deferred recordings for different parameters are
        independent and can be completed in any order.
        """
        # Setup
        memoizer = Memoizer()
        deferred_objects: dict[int, interfaces.DeferredRecording] = {}

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function; remembers the first
                # positional argument and returns the encode function.
                x = args[0]

                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording,
                    # stores it in a mapping under the remembered argument and returns it.
                    deferred = interfaces.DeferredRecording()
                    deferred_objects[x] = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.record(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Record multiple calls
        compute(5)
        compute(10)
        compute(15)

        # Complete in reverse order
        deferred_objects[15].finalize(30)
        deferred_objects[5].finalize(10)
        deferred_objects[10].finalize(20)

        # Assert: all caches should be correct
        for x, expected in [(5, 10), (10, 20), (15, 30)]:
            cache_key = interfaces._BaseMemoizer._make_key(None, x)
            cache_hit = memoizer._cache.get(cache_key)
            self.assertIsNotNone(cache_hit)
            self.assertEqual(cache_hit.value.encoded_result, expected)

    # ============= Result Encoder/Decoder Factory Tests =============

    @set_caching_module_enabled(True)
    def test_encoder_factory_receives_underlying_function(self) -> None:
        # [FUNCTION DOC] Records with an encoder factory that stores the function it receives
        # and returns an identity encoder. Asserts the factory was called once with a callable,
        # the call returned ten, and that the received function, called directly with seven,
        # returns fourteen (it is the unwrapped function).
        """Test that the encoder factory receives the underlying function.

        Verifies that when custom_result_encoder is provided, the factory
        receives the unwrapped function as the first parameter, allowing
        the encoder to call it if needed.
        """
        # Setup: create a memoizer and track what fn is passed to the factory
        memoizer = Memoizer()
        received_fn: list[object] = []

        def encoder_factory(fn: object) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory that stores the function it is given in a
            # list owned by the test and returns a params-to-encoder function.
            received_fn.append(fn)

            def params_to_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> object:
                    # [FUNCTION DOC] HELPER: Encode function returning its input unchanged.
                    return result

                return encode

            return params_to_encoder

        @memoizer.record(custom_result_encoder=encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Execute: call the memoized function
        result = compute(5)

        # Assert: encoder factory received a function (the unwrapped compute)
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify that calling the received fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @set_caching_module_enabled(True)
    def test_encoder_factory_fn_bypasses_memoization(self) -> None:
        # [FUNCTION DOC] Records with an identity encoder factory that stores the function it
        # receives. After one memoized call (count one), calls the received function directly
        # three times and asserts the call count is four while the memoizer still holds only one
        # cache entry, so direct calls skip memoization.
        """Test that calling fn from the encoder bypasses memoization.

        Verifies that the fn passed to the encoder factory is the unwrapped
        function, so calling it does not trigger caching side effects.
        """
        # Setup: create a memoizer and count direct vs memoized calls
        memoizer = Memoizer()
        direct_call_count = 0
        received_fn: list[object] = []

        def encoder_factory(fn: object) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory that stores the function it is given in a
            # list owned by the test and returns a params-to-encoder function.
            received_fn.append(fn)

            def params_to_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> object:
                    # [FUNCTION DOC] HELPER: Encode function returning its input unchanged.
                    return result

                return encode

            return params_to_encoder

        @memoizer.record(custom_result_encoder=encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal direct_call_count
            direct_call_count += 1
            return x * 2

        # Execute: call the memoized function once
        compute(5)
        self.assertEqual(direct_call_count, 1)

        # Now call the underlying fn directly multiple times
        underlying_fn = received_fn[0]
        underlying_fn(5)
        underlying_fn(5)
        underlying_fn(5)

        # Assert: direct calls happened but no new cache entries
        self.assertEqual(direct_call_count, 4)  # 1 memoized + 3 direct
        # Cache should only have the one entry from the memoized call
        cache_entries = len(memoizer._cache._memory)
        self.assertEqual(cache_entries, 1)

    @set_caching_module_enabled(True)
    def test_decoder_factory_receives_underlying_function(self) -> None:
        # [FUNCTION DOC] Inserts a cache entry directly into a Memoizer (encoded result a
        # mapping holding an encoded value of ten), then replays with a decoder factory that
        # stores the function it receives and returns a decoder extracting the encoded value.
        # Asserts the factory ran once with a callable, the result is ten, and the received
        # function doubles seven to fourteen.
        """Test that the decoder factory receives the underlying function.

        Verifies that when custom_result_decoder is provided, the factory
        receives the unwrapped function as the first parameter, allowing
        the decoder to call it as a fallback if needed.
        """
        # Setup: create a memoizer and track what fn is passed to the factory
        memoizer = Memoizer()
        received_fn: list[object] = []

        # First, populate the cache with a value
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result={"encoded_value": 10},
        )
        memoizer._cache.insert(cache_key, cache_entry)

        def decoder_factory(fn: object) -> object:
            # [FUNCTION DOC] HELPER: Decoder factory that stores the function it is given in a
            # list owned by the test and returns a params-to-decoder function.
            received_fn.append(fn)

            def params_to_decoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-decoder function ignoring its arguments;
                # returns the decode function.
                def decode(encoded_result: object) -> int:
                    # [FUNCTION DOC] HELPER: Decode function returning the encoded value entry
                    # of the mapping it is given.
                    return encoded_result["encoded_value"]

                return decode

            return params_to_decoder

        @memoizer.replay(custom_result_decoder=decoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Execute: replay the cached value
        result = compute(5)

        # Assert: decoder factory received a function (the unwrapped compute)
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify that calling the received fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_decoder_receives_fn(self) -> None:
        # [FUNCTION DOC] Same as the previous decoder test but for a persistent memoizer:
        # pickles a cache entry (encoded result a mapping with an encoded value of ten) into the
        # disk cache, replays with a decoder factory that stores the function it receives, and
        # asserts the factory ran once with a callable, the result is ten and the stored
        # function doubles seven.
        """Test that PersistentMemoizer decoder factory receives fn.

        Verifies that the two-level cache correctly passes the fn parameter
        to the decoder factory when replaying from disk cache.
        """
        import pickle

        # Setup: create a persistent memoizer
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        received_fn: list[object] = []

        # Populate the disk cache with a value
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        cache_entry = interfaces.CacheEntry(
            encoded_params={"args": (5,), "kwargs": {}},
            encoded_result={"encoded_value": 10},
        )
        pickled_entry = pickle.dumps(cache_entry)
        persistent._disk_cache.insert(cache_key, pickled_entry)

        def decoder_factory(fn: object) -> object:
            # [FUNCTION DOC] HELPER: Decoder factory that stores the function it is given in a
            # list owned by the test and returns a params-to-decoder function.
            received_fn.append(fn)

            def params_to_decoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-decoder function ignoring its arguments;
                # returns the decode function.
                def decode(encoded_result: object) -> int:
                    # [FUNCTION DOC] HELPER: Decode function returning the encoded value entry
                    # of the mapping it is given.
                    return encoded_result["encoded_value"]

                return decode

            return params_to_decoder

        @persistent.replay(custom_result_decoder=decoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns double the input.
            return x * 2

        # Execute: replay from disk cache
        result = compute(5)

        # Assert: decoder factory received the underlying function
        self.assertEqual(len(received_fn), 1)
        self.assertTrue(callable(received_fn[0]))
        self.assertEqual(result, 10)

        # Verify fn works correctly
        underlying_fn = received_fn[0]
        self.assertEqual(underlying_fn(7), 14)

    @set_caching_module_enabled(True)
    def test_deferred_recording_double_finalize_raises(self) -> None:
        # [FUNCTION DOC] Finalizes a deferred recording once, then asserts a second finalize
        # raises a runtime error whose message mentions that finalize was called multiple times.
        """Test that calling finalize() twice raises RuntimeError.

        Verifies that the race condition prevention logic correctly
        rejects duplicate finalize() calls.
        """
        # Setup: create a DeferredRecording
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()

        # Execute: first finalize succeeds
        deferred.finalize(42)

        # Assert: second finalize raises RuntimeError
        with self.assertRaises(RuntimeError) as ctx:
            deferred.finalize(100)

        self.assertIn("finalize() called multiple times", str(ctx.exception))

    @set_caching_module_enabled(True)
    def test_deferred_recording_callback_after_finalize(self) -> None:
        # [FUNCTION DOC] Finalizes a deferred recording with forty-two first, then registers a
        # callback and asserts it ran immediately with that value.
        """Test that callbacks registered after finalize() are invoked immediately.

        Verifies the race condition handling where finalize() completes before
        the memoizer registers its callback - the callback should still be invoked.
        """
        # Setup: create a DeferredRecording and finalize it first
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
        deferred.finalize(42)

        # Execute: register a callback after finalize
        callback_invoked = False
        callback_value: int | None = None

        def callback(result: int) -> None:
            # [FUNCTION DOC] HELPER: Callback that records, via nonlocal variables, that it ran
            # and the value it received.
            nonlocal callback_invoked, callback_value
            callback_invoked = True
            callback_value = result

        deferred.register_callback(callback)

        # Assert: callback was invoked immediately with the finalized result
        self.assertTrue(callback_invoked)
        self.assertEqual(callback_value, 42)

    @set_caching_module_enabled(True)
    def test_deferred_recording_multiple_callbacks_order(self) -> None:
        # [FUNCTION DOC] Registers three callbacks (indices zero, one, two) before finalizing,
        # finalizes, and asserts they ran in registration order.
        """Test that multiple callbacks are invoked in registration order.

        Verifies that when multiple callbacks are registered before finalize(),
        they are all invoked in the order they were registered.
        """
        # Setup: create a DeferredRecording and register multiple callbacks
        deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
        invocation_order: list[int] = []

        def make_callback(idx: int) -> object:
            # [FUNCTION DOC] HELPER: Factory returning a callback that appends the given index
            # to a list owned by the test.
            def callback(result: int) -> None:
                # [FUNCTION DOC] HELPER: Callback appending the captured index to the shared
                # invocation list.
                invocation_order.append(idx)

            return callback

        # Register callbacks in order 0, 1, 2
        deferred.register_callback(make_callback(0))
        deferred.register_callback(make_callback(1))
        deferred.register_callback(make_callback(2))

        # Execute: finalize the deferred recording
        deferred.finalize(100)

        # Assert: callbacks were invoked in registration order
        self.assertEqual(invocation_order, [0, 1, 2])

    @set_caching_module_enabled(True)
    def test_deferred_recording_thread_safety(self) -> None:
        # [FUNCTION DOC] Runs one hundred iterations of a race between finalize and callback
        # registration in two threads released together by an event. Collects an assertion error
        # for any iteration where the callback did not see exactly one result equal to
        # forty-two, and finally asserts the error list is empty.
        """Test that concurrent finalize() and register_callback() are thread-safe.

        Verifies that when finalize() and register_callback() are called
        concurrently from different threads, no race conditions occur.
        """
        # Setup: create multiple deferred recordings to test concurrency
        num_iterations = 100
        errors: list[Exception] = []

        def run_concurrent_test() -> None:
            # [FUNCTION DOC] HELPER: One iteration: creates a deferred recording, starts a
            # finalize thread and a register thread (two worker pool), releases both through an
            # event, waits up to five seconds, and records an error if the callback results are
            # not exactly forty-two.
            deferred: interfaces.DeferredRecording[int] = interfaces.DeferredRecording()
            callback_results: list[int] = []
            barrier = Event()

            def callback(result: int) -> None:
                # [FUNCTION DOC] HELPER: Callback appending its received result to the
                # iteration's list.
                callback_results.append(result)

            def finalize_thread() -> None:
                # [FUNCTION DOC] HELPER: Waits for the start event then finalizes with
                # forty-two, swallowing a runtime error from a duplicate finalize.
                barrier.wait()
                try:
                    deferred.finalize(42)
                except RuntimeError:
                    pass

            def register_thread() -> None:
                # [FUNCTION DOC] HELPER: Waits for the start event then registers the callback.
                barrier.wait()
                deferred.register_callback(callback)

            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(finalize_thread)
                f2 = executor.submit(register_thread)
                barrier.set()
                wait([f1, f2], timeout=5)

            if len(callback_results) != 1 or callback_results[0] != 42:
                errors.append(AssertionError(f"Expected [42], got {callback_results}"))

        # Execute: run many iterations to catch race conditions
        for _ in range(num_iterations):
            run_concurrent_test()

        # Assert: no errors occurred
        self.assertEqual(errors, [])

    # ============= Interim Result Tests =============

    @set_caching_module_enabled(True)
    def test_interim_result_prevents_function_reexecution(self) -> None:
        # [FUNCTION DOC] Uses memoize with a deferred encoder whose recording supplies an
        # interim result callable returning the original result. The first call executes the
        # function; the second and third calls while the recording is pending return the same
        # value ten without executing again (count stays one). Finally finalizes the recording
        # with ten.
        """Test that make_interim_result prevents function re-execution.

        Verifies that when a DeferredRecording has make_interim_result set,
        subsequent calls return the interim result without re-executing the function.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        original_result: list[int] = []
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: int) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that stores the result, creates a
                    # deferred recording whose interim result callable returns that result,
                    # keeps the recording in an enclosing variable and returns it.
                    nonlocal deferred_obj
                    original_result.append(result)
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: should return interim result without re-execution
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # No additional call!

        # Third call: same behavior
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 1)  # Still no additional call

        # Finalize to mimic actual behavior
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(10)

    @set_caching_module_enabled(True)
    def test_interim_result_returns_same_object(self) -> None:
        # [FUNCTION DOC] Uses memoize with a deferred encoder whose interim result callable
        # returns a shared mapping object. Three calls return the identical object (checked
        # by identity, also identical to the shared object), then the recording is finalized
        # with it.
        """Test that make_interim_result returns the same object.

        Verifies that all callers get the exact same object (e.g., same Future),
        not copies.
        """
        # Setup
        memoizer = Memoizer()
        shared_object = {"value": "shared"}
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: object) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording
                    # whose interim result callable returns the shared object, keeps it in an
                    # enclosing variable and returns it.
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: shared_object
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> object:
            # [FUNCTION DOC] HELPER: Returns the shared mapping object.
            return shared_object

        # Multiple calls should return the same object
        result1 = compute(5)
        result2 = compute(5)
        result3 = compute(5)

        self.assertIs(result1, result2)
        self.assertIs(result2, result3)
        self.assertIs(result1, shared_object)

        # Finalize to mimic actual behavior
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(shared_object)

    @set_caching_module_enabled(True)
    def test_interim_result_with_future_pattern(self) -> None:
        # [FUNCTION DOC] Future based flow with interim results. The function submits a task to
        # a thread pool that waits on an event; the encoder makes the interim result the future
        # itself and attaches a done callback that finalizes the recording and sets an event.
        # Two calls return the identical future with the function executed once; the gate event
        # is released in a finally block to avoid hangs. After the future yields ten, waits for
        # the finalize event and asserts the recording's callback list became empty (finalized).
        """Test make_interim_result with a Future-like pattern.

        Verifies that multiple callers get the same Future object while
        the deferred recording is pending.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        deferred_obj: interfaces.DeferredRecording | None = None

        finalize_done = Event()

        def future_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def future_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(future_result: Future[int]) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function; builds a deferred recording whose
                    # interim result is the given future, registers a done callback on the
                    # future, keeps the recording and returns it.
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: future_result
                    )

                    def on_complete(completed_future: Future[int]) -> None:
                        # [FUNCTION DOC] HELPER: Done callback that finalizes the deferred
                        # recording with the future's result and then sets an event
                        # signalling completion.
                        actual_result = completed_future.result()
                        deferred.finalize(actual_result)
                        finalize_done.set()

                    future_result.add_done_callback(on_complete)
                    deferred_obj = deferred
                    return deferred

                return encode

            return future_encoder

        with ThreadPoolExecutor() as executor:
            barrier = Event()

            @memoizer.memoize(custom_result_encoder=future_encoder_factory)
            def compute_async(x: int) -> Future[int]:
                # [FUNCTION DOC] HELPER: Increments a nonlocal call counter and submits a worker
                # to the pool, returning its future.
                nonlocal call_count
                call_count += 1

                def work() -> int:
                    # [FUNCTION DOC] HELPER: Worker that blocks on the gate event and then
                    # returns double the input.
                    barrier.wait()  # Wait for signal
                    return x * 2

                return executor.submit(work)

            try:
                # First call: executes function, returns Future
                future1 = compute_async(5)
                self.assertEqual(call_count, 1)

                # Second call: should return same Future without re-execution
                future2 = compute_async(5)
                self.assertEqual(call_count, 1)  # No additional call
                self.assertIs(future1, future2)  # Same Future object
            finally:
                # Release the barrier - must be in finally to prevent test hang
                # if any assertion fails before this point
                barrier.set()

            # Wait for Future to complete
            result = future1.result(timeout=5)
            self.assertEqual(result, 10)

            # Future.set_result() invokes done callbacks after releasing its
            # condition lock, so future.result() can return before on_complete
            # (and thus deferred.finalize()) has run.  Wait for finalize.
            self.assertTrue(finalize_done.wait(timeout=5))

            # Verify deferred recording completed
            self.assertIsNotNone(deferred_obj)
            self.assertIsNone(deferred_obj._callbacks)

    @set_caching_module_enabled(True)
    def test_interim_result_transitions_to_cached_result(self) -> None:
        # [FUNCTION DOC] Calls a memoized function with an interim-result deferred encoder
        # twice (second served from the interim result, count one), finalizes the recording
        # with one hundred, and asserts a third call returns one hundred from the cache
        # without executing again.
        """Test that after complete(), cached result is used instead of interim.

        Verifies the transition from interim result to final cached result.
        """
        # Setup
        memoizer = Memoizer()
        deferred_obj: interfaces.DeferredRecording | None = None
        call_count = 0

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: int) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording
                    # whose interim result callable returns the result, keeps it in an enclosing
                    # variable and returns it.
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: executes function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call before complete: uses interim result
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)

        # Finalize the deferred recording
        self.assertIsNotNone(deferred_obj)
        deferred_obj.finalize(100)

        # After finalize, memoize returns the cached result
        result3 = compute(5)
        self.assertEqual(result3, 100)
        self.assertEqual(call_count, 1)  # Still no additional call

    @patch_on_disk_cache_base_dir
    @set_caching_module_enabled(True)
    def test_persistent_memoizer_interim_result(self) -> None:
        # [FUNCTION DOC] Persistent memoizer variant of the interim result test: three memoized
        # calls execute the function once. Asserts the recording has exactly two registered
        # callbacks (one for memory, one for disk, so no duplicate disk callback from repeated
        # calls), finalizes with one hundred and asserts the disk cache contains the key.
        """Test that PersistentMemoizer handles make_interim_result correctly.

        Verifies that make_interim_result works with PersistentMemoizer and
        doesn't register duplicate disk callbacks.
        """
        # Setup
        persistent = PersistentMemoizer(sub_dir=self.sub_dir())
        call_count = 0
        deferred_obj: interfaces.DeferredRecording | None = None

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: int) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording
                    # whose interim result callable returns the result, keeps it in an enclosing
                    # variable and returns it.
                    nonlocal deferred_obj
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_obj = deferred
                    return deferred

                return encode

            return deferred_encoder

        @persistent.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: executes function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: uses interim result, no re-execution
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)

        # Third call: still uses interim result
        result3 = compute(5)
        self.assertEqual(result3, 10)
        self.assertEqual(call_count, 1)

        # Complete the deferred recording
        self.assertIsNotNone(deferred_obj)
        # Check that only 2 callbacks are registered (one for memory, one for disk)
        self.assertEqual(len(deferred_obj._callbacks), 2)
        deferred_obj.finalize(100)

        # Verify disk persistence
        cache_key = interfaces._BaseMemoizer._make_key(None, 5)
        disk_hit = persistent._disk_cache.get(cache_key)
        self.assertIsNotNone(disk_hit)

    @set_caching_module_enabled(True)
    def test_interim_result_different_params_independent(self) -> None:
        # [FUNCTION DOC] Calls a memoized function with interim-result deferred recording for
        # arguments five and ten, then again for each; asserts only two executions in total
        # (each parameter set has its own pending interim result) and the correct results.
        # Finalizes all recordings with zero at the end; strong references to the recordings are
        # kept to avoid garbage collection.
        """Test that interim results for different params are independent.

        Verifies that each unique parameter set has its own interim result.
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        # Keep strong references to deferred recordings to prevent GC
        deferred_objs: list[interfaces.DeferredRecording] = []

        def deferred_encoder_factory(fn) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: int) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording
                    # whose interim result callable returns the result, appends it to a list
                    # owned by the test and returns it.
                    deferred = interfaces.DeferredRecording(
                        make_interim_result=lambda: result
                    )
                    deferred_objs.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # Call with different params
        result_5 = compute(5)
        self.assertEqual(result_5, 10)
        self.assertEqual(call_count, 1)

        result_10 = compute(10)
        self.assertEqual(result_10, 20)
        self.assertEqual(call_count, 2)

        # Subsequent calls with same params use interim result
        result_5_again = compute(5)
        self.assertEqual(result_5_again, 10)
        self.assertEqual(call_count, 2)  # No additional call

        result_10_again = compute(10)
        self.assertEqual(result_10_again, 20)
        self.assertEqual(call_count, 2)  # No additional call

        # Finalize all deferred recordings
        for deferred in deferred_objs:
            deferred.finalize(0)

    @set_caching_module_enabled(True)
    def test_get_interim_result_when_not_set(self) -> None:
        # [FUNCTION DOC] Creates a deferred recording without an interim result callable and
        # asserts that asking for the interim result returns none.
        """Test that get_interim_result returns None when not configured.

        Verifies that when a DeferredRecording is created without make_interim_result,
        get_interim_result returns None indicating no interim result.
        """
        # Setup: create a DeferredRecording without make_interim_result
        deferred: interfaces.DeferredRecording[int, int] = (
            interfaces.DeferredRecording()
        )

        # Execute: get the interim result
        interim = deferred.get_interim_result()

        # Assert: should indicate no interim result available
        self.assertIsNone(interim)

    @set_caching_module_enabled(True)
    def test_get_interim_result_when_callable_returns_none(self) -> None:
        # [FUNCTION DOC] Creates a deferred recording whose interim result callable returns none
        # and asserts the interim result object itself is not none while its value is none,
        # distinguishing no callable from a callable returning none.
        """Test get_interim_result when make_interim_result returns None.

        Verifies the distinction between:
        - None: no make_interim_result configured
        - InterimResult(None): make_interim_result configured but returns None

        This is an important edge case for type safety.
        """
        # Setup: create a DeferredRecording where make_interim_result returns None
        deferred: interfaces.DeferredRecording[None, int] = (
            interfaces.DeferredRecording(make_interim_result=lambda: None)
        )

        # Execute: get the interim result
        interim = deferred.get_interim_result()

        # Assert: should indicate interim result is available (even though value is None)
        self.assertIsNotNone(interim)
        self.assertIsNone(interim.value)

    @set_caching_module_enabled(True)
    def test_get_interim_result_callable_invoked_each_time(self) -> None:
        # [FUNCTION DOC] Creates a deferred recording with a callable that increments and
        # returns a counter. Asks for the interim result three times and asserts values one,
        # two, three and three invocations, showing the callable runs on every request.
        """Test that make_interim_result is invoked on each get_interim_result call.

        Verifies the documented behavior that the callable is invoked each time,
        which allows for returning new objects if desired.
        """
        # Setup: create a DeferredRecording with a callable that tracks invocations
        invocation_count = 0

        def make_result() -> int:
            # [FUNCTION DOC] HELPER: Increments a nonlocal counter and returns its new value.
            nonlocal invocation_count
            invocation_count += 1
            return invocation_count

        deferred: interfaces.DeferredRecording[int, int] = interfaces.DeferredRecording(
            make_interim_result=make_result
        )

        # Execute: call get_interim_result multiple times
        interim1 = deferred.get_interim_result()
        interim2 = deferred.get_interim_result()
        interim3 = deferred.get_interim_result()

        # Assert: callable was invoked each time with incrementing results
        self.assertIsNotNone(interim1)
        self.assertIsNotNone(interim2)
        self.assertIsNotNone(interim3)
        self.assertEqual(interim1.value, 1)
        self.assertEqual(interim2.value, 2)
        self.assertEqual(interim3.value, 3)
        self.assertEqual(invocation_count, 3)

    @set_caching_module_enabled(True)
    def test_get_interim_result_exception_propagates(self) -> None:
        # [FUNCTION DOC] Creates a deferred recording whose callable raises a value error on the
        # first call and returns forty-two afterwards. Asserts the first request raises with the
        # expected message and the second request succeeds with value forty-two, so the state is
        # not corrupted.
        """Test that exceptions from make_interim_result propagate correctly.

        Verifies that if make_interim_result raises an exception, it propagates
        to the caller and does not corrupt the DeferredRecording state.
        """
        # Setup: create a DeferredRecording with a failing callable
        call_count = 0

        def failing_callable() -> int:
            # [FUNCTION DOC] HELPER: Raises a value error on the first call and returns
            # forty-two on later calls, counting calls in a nonlocal counter.
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("First call fails")
            return 42

        deferred: interfaces.DeferredRecording[int, int] = interfaces.DeferredRecording(
            make_interim_result=failing_callable
        )

        # Execute & Assert: first call raises exception
        with self.assertRaises(ValueError) as ctx:
            deferred.get_interim_result()
        self.assertIn("First call fails", str(ctx.exception))

        # Execute: second call should work (state not corrupted)
        interim = deferred.get_interim_result()

        # Assert: should work normally after exception
        self.assertIsNotNone(interim)
        self.assertEqual(interim.value, 42)

    @set_caching_module_enabled(True)
    def test_get_interim_result_thread_safety_with_finalize(self) -> None:
        # [FUNCTION DOC] Runs fifty iterations in which one thread requests the interim result
        # ten times while another finalizes the recording, both released together by an event.
        # Records any unexpected exception and any interim result whose value is not forty-two,
        # and finally asserts the error list is empty.
        """Test that get_interim_result and finalize are thread-safe together.

        Verifies that concurrent calls to get_interim_result and finalize
        don't cause race conditions. The interim result should still be
        accessible until the DeferredRecording is finalized.
        """
        num_iterations = 50
        errors: list[Exception] = []

        def run_concurrent_test() -> None:
            # [FUNCTION DOC] HELPER: One iteration: creates a deferred recording with an interim
            # callable returning forty-two, runs the interim-reading thread and the finalize
            # thread in a two worker pool released by an event, waits up to five seconds, and
            # records errors for any bad value.
            deferred: interfaces.DeferredRecording[int, int] = (
                interfaces.DeferredRecording(make_interim_result=lambda: 42)
            )
            barrier = Event()
            interim_results: list[interfaces.InterimResult[int] | None] = []

            def get_interim_thread() -> None:
                # [FUNCTION DOC] HELPER: Waits for the start event, then ten times requests the
                # interim result, collecting results and recording any exception.
                barrier.wait()
                for _ in range(10):
                    try:
                        interim_results.append(deferred.get_interim_result())
                    except Exception as e:
                        errors.append(e)

            def finalize_thread() -> None:
                # [FUNCTION DOC] HELPER: Waits for the start event then finalizes with one
                # hundred, swallowing a runtime error from a duplicate finalize.
                barrier.wait()
                try:
                    deferred.finalize(100)
                except RuntimeError:
                    pass

            with ThreadPoolExecutor(max_workers=2) as executor:
                f1 = executor.submit(get_interim_thread)
                f2 = executor.submit(finalize_thread)
                barrier.set()
                wait([f1, f2], timeout=5)

            # All interim results should be valid (either 42 or from make_interim_result)
            for interim in interim_results:
                if interim is not None and interim.value != 42:
                    errors.append(AssertionError(f"Expected 42, got {interim.value}"))

        # Execute: run many iterations to catch race conditions
        for _ in range(num_iterations):
            run_concurrent_test()

        # Assert: no errors occurred
        self.assertEqual(errors, [])

    @set_caching_module_enabled(True)
    def test_interim_result_no_make_interim_causes_reexecution(self) -> None:
        # [FUNCTION DOC] Uses memoize with a deferred encoder that does not set an interim
        # result callable. Two calls with the same argument both execute the function (count
        # two) because no interim result is available while pending. Finalizes all recordings
        # with ten at the end.
        """Test that without make_interim_result, function is re-executed.

        Verifies that when a DeferredRecording does NOT have make_interim_result,
        subsequent calls while pending cause the function to be re-executed
        (the original behavior before interim results were added).
        """
        # Setup
        memoizer = Memoizer()
        call_count = 0
        # Keep strong references to deferred recordings to prevent GC
        deferred_objs: list[interfaces.DeferredRecording] = []

        def deferred_encoder_factory_no_interim(fn: object) -> object:
            # [FUNCTION DOC] HELPER: Encoder factory taking the underlying function; returns a
            # params-to-encoder function.
            def deferred_encoder(*args: object, **kwargs: object) -> object:
                # [FUNCTION DOC] HELPER: Params-to-encoder function ignoring its arguments;
                # returns the encode function.
                def encode(result: int) -> interfaces.DeferredRecording:
                    # [FUNCTION DOC] HELPER: Encode function that creates a deferred recording
                    # with no interim result callable, appends it to a list owned by the test
                    # and returns it.
                    # Explicitly not setting make_interim_result
                    deferred = interfaces.DeferredRecording()
                    deferred_objs.append(deferred)
                    return deferred

                return encode

            return deferred_encoder

        @memoizer.memoize(custom_result_encoder=deferred_encoder_factory_no_interim)
        def compute(x: int) -> int:
            # [FUNCTION DOC] HELPER: Counts calls through a nonlocal counter and returns double
            # the input.
            nonlocal call_count
            call_count += 1
            return x * 2

        # First call: should execute function
        result1 = compute(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)

        # Second call: should re-execute (no interim result available)
        result2 = compute(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 2)

        # Finalize all deferred recordings
        for deferred in deferred_objs:
            deferred.finalize(10)


if __name__ == "__main__":
    run_tests()
