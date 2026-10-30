# Owner(s): ["module: unknown"]
# [MODULE DOC] Tests for torch.utils._config_module using the fake config modules in
# torch.testing._internal (fake_config_module, fake_config_module2, fake_config_module3). Covers
# value loading, env-var defaults/forcing, overrides, patching (reentrancy, threads),
# save/load/codegen, hashing and dirty tracking, dict caching, aliases and deprecation. Import-
# time side effect: sets four environment variables used by the fake config modules.
import os
import pickle
import queue
import threading
import warnings
from unittest.mock import patch


os.environ["ENV_TRUE"] = "1"
os.environ["ENV_FALSE"] = "0"
os.environ["ENV_STR"] = "1234"
os.environ["ENV_STR_EMPTY"] = ""

from torch.testing._internal import (
    fake_config_module as config,
    fake_config_module2 as config2,
    fake_config_module3 as config3,
)
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._config_module import _ConfigEntry, _UNSET_SENTINEL, Config


class TestConfigModule(TestCase):
    # [CLASS DOC] Test case exercising the ConfigModule machinery through the fake config modules.
    # The tearDown resets shared module-level state, so tests may mutate config freely.
    def tearDown(self):
        # [FUNCTION DOC] Resets all per-entry user overrides to the unset sentinel, clears the
        # hide flag and deprecation-warned flag on every entry, and clears the hash, dirty-key and
        # dict caches so changes made by one test do not leak into the next.
        # Config changes get persisted between test cases
        for k in config._config:
            config._config[k].user_override.set(_UNSET_SENTINEL)
            config._config[k].hide = False
        config._hash_cache_var.set(None)
        config._get_dict_dirty_keys_var.set(None)
        config._get_dict_cache_var.set(None)
        # Reset deprecation warning flags
        for k in config._config:
            config._config[k]._deprecation_warned = False

    def test_base_value_loading(self):
        # [FUNCTION DOC] Checks default values of every basic config type (bool, nested bool,
        # optional, int, float, str, list, set, tuple, dict, None) and that reading an unknown
        # name raises AttributeError.
        self.assertTrue(config.e_bool)
        self.assertTrue(config.nested.e_bool)
        self.assertTrue(config.e_optional)
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_float, 1.0)
        self.assertEqual(config.e_string, "string")
        self.assertEqual(config.e_list, [1])
        self.assertEqual(config.e_set, {1})
        self.assertEqual(config.e_tuple, (1,))
        self.assertEqual(config.e_dict, {1: 2})
        self.assertEqual(config.e_none, None)
        with self.assertRaises(
            AttributeError, msg="fake_config_module.does_not_exist does not exist"
        ):
            config.does_not_exist

    def test_type_loading(self):
        # [FUNCTION DOC] Checks that the declared type of an optional bool entry and of a None-
        # default entry is reported as bool or None.
        self.assertEqual(config.get_type("e_optional"), bool | None)
        self.assertEqual(config.get_type("e_none"), bool | None)

    def test_overrides(self):
        # [FUNCTION DOC] Assigns a new value to each config type, verifies it is read back, and
        # checks that optional entries accept None. Assigning an unknown name must raise
        # AttributeError.
        config.e_bool = False
        self.assertFalse(config.e_bool)
        config.nested.e_bool = False
        self.assertFalse(config.nested.e_bool)
        config.e_int = 2
        self.assertEqual(config.e_int, 2)
        config.e_float = 2.0
        self.assertEqual(config.e_float, 2.0)
        config.e_string = "string2"
        self.assertEqual(config.e_string, "string2")
        config.e_list = [2]
        self.assertEqual(config.e_list, [2])
        config.e_set = {2}
        self.assertEqual(config.e_set, {2})
        config.e_tuple = (2,)
        self.assertEqual(config.e_tuple, (2,))
        config.e_dict = {2: 3}
        self.assertEqual(config.e_dict, {2: 3})
        config.e_none = "not none"
        self.assertEqual(config.e_none, "not none")
        config.e_none = None
        self.assertEqual(config.e_none, None)
        config.e_optional = None
        self.assertEqual(config.e_optional, None)
        config.e_optional = False
        self.assertEqual(config.e_optional, False)
        with self.assertRaises(
            AttributeError, msg="fake_config_module.does_not_exist does not exist"
        ):
            config.does_not_exist = 0

    def test_hide_not_added_to_instance_dict_on_write(self):
        # [FUNCTION DOC] Verifies that a normal write does not add a hide attribute to the entry
        # instance dictionary (it stays at the class default), while a write to an entry whose
        # hide flag is set clears that flag.
        # A normal config write must not materialize 'hide' in the _ConfigEntry instance
        # __dict__; it should stay on the class default. An unconditional hide=False write
        # on every setattr would add it (pure overhead), so __setattr__ clears hide only
        # when it is actually set.
        entry = config._config["e_bool"]
        entry.__dict__.pop("hide", None)  # reset to the class default
        config.e_bool = False
        self.assertNotIn("hide", entry.__dict__)
        # When hide *is* set (the __delattr__ / mock.patch path), a write still clears it.
        entry.hide = True
        config.e_bool = True
        self.assertFalse(entry.hide)

    def test_none_override_semantics(self):
        # [FUNCTION DOC] Verifies that explicitly assigning None is a real override (reads back
        # None, not the default), then resets all overrides to the unset sentinel by hand.
        config.e_bool = None
        self.assertIsNone(config.e_bool)
        for k in config._config:
            config._config[k].user_override.set(_UNSET_SENTINEL)

    def test_reference_semantics(self):
        # [FUNCTION DOC] Verifies that list, set and dict defaults are returned by reference, so
        # in-place mutation of the returned container is visible on later reads.
        config.e_list.append(2)
        self.assertEqual(config.e_list, [1, 2])
        config.e_set.add(2)
        self.assertEqual(config.e_set, {1, 2})
        config.e_dict[2] = 3
        self.assertEqual(config.e_dict, {1: 2, 2: 3})

    def test_env_name_semantics(self):
        # [FUNCTION DOC] Checks environment-variable semantics: a default env name sets the
        # initial value (parsed as bool, so "0" is False) and can be overridden by assignment,
        # while a forced env name keeps the value true even after assignment of False.
        self.assertTrue(config.e_env_default)
        self.assertFalse(config.e_env_default_FALSE)
        self.assertTrue(config.e_env_force)
        config.e_env_default = False
        self.assertFalse(config.e_env_default)
        config.e_env_force = False
        self.assertTrue(config.e_env_force)

    def test_env_name_string_semantics(self):
        # [FUNCTION DOC] Checks that string entries taking defaults from env vars keep the raw
        # string, including the empty string, and can be overridden by assignment.
        self.assertEqual(config.e_env_default_str, "1234")
        self.assertEqual(config.e_env_default_str_empty, "")
        config.e_env_default_str = "override"
        self.assertEqual(config.e_env_default_str, "override")

    def test_multi_env(self):
        # [FUNCTION DOC] Checks that entries in the second fake module listing multiple env var
        # names (default and forced variants) resolve to true.
        self.assertTrue(config2.e_env_default_multi)
        self.assertTrue(config2.e_env_force_multi)

    def test_save_config(self):
        # [FUNCTION DOC] Saves the config under a FutureWarning filter, unpickles the bytes and
        # compares against the full expected mapping including ignored keys, then changes values,
        # reloads the saved bytes, and checks that the saved entry is restored while the ignored
        # one is not.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.save_config()

        self.assertDictEqual(
            pickle.loads(p),
            {
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "magic_cache_config_ignored": True,
                "_save_config_ignore": ["e_ignored"],
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
        config.e_bool = False
        config.e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config.e_ignored)

    def test_save_config_with_patch(self):
        # [FUNCTION DOC] Checks that saving config inside a patch context records the patched
        # value (False for the boolean entry) while all other entries keep their defaults.
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            p = config.save_config()
            self.assertDictEqual(
                pickle.loads(p),
                {
                    "_cache_config_ignore_prefix": ["magic_cache_config"],
                    "e_bool": False,
                    "e_dict": {1: 2},
                    "e_float": 1.0,
                    "e_int": 1,
                    "e_list": [1],
                    "e_none": None,
                    "e_set": {1},
                    "e_string": "string",
                    "e_tuple": (1,),
                    "nested.e_bool": True,
                    "_e_ignored": True,
                    "e_compile_ignored": True,
                    "magic_cache_config_ignored": True,
                    "_save_config_ignore": ["e_ignored"],
                    "e_config": True,
                    "e_jk": True,
                    "e_jk_false": False,
                    "e_env_default": True,
                    "e_env_default_FALSE": False,
                    "e_env_default_str": "1234",
                    "e_env_default_str_empty": "",
                    "e_env_force": True,
                    "e_optional": True,
                    "e_deprecated": True,
                    "e_not_deprecated": False,
                },
            )

    def test_save_config_portable(self):
        # [FUNCTION DOC] Checks the portable save format: a plain mapping with the same entries
        # but without the underscore-prefixed and cache-ignore bookkeeping keys. Reloads it after
        # changes and checks the regular entry is restored but the underscore-ignored one is not.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.save_config_portable()

        self.assertDictEqual(
            p,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "e_compile_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
        config.e_bool = False
        config._e_ignored = False
        config.load_config(p)
        self.assertTrue(config.e_bool)
        self.assertFalse(config._e_ignored)

    def test_codegen_config(self):
        # [FUNCTION DOC] Sets one entry and an ignored entry to False, generates source text for
        # the config, and compares it with an exact expected string listing only entries that
        # differ from defaults or come from environment settings.
        config.e_bool = False
        config.e_ignored = False
        code = config.codegen_config()
        self.assertEqual(
            code,
            """torch.testing._internal.fake_config_module.e_bool = False
torch.testing._internal.fake_config_module.e_env_default = True
torch.testing._internal.fake_config_module.e_env_default_FALSE = False
torch.testing._internal.fake_config_module.e_env_default_str = '1234'
torch.testing._internal.fake_config_module.e_env_default_str_empty = ''
torch.testing._internal.fake_config_module.e_env_force = True""",
        )

    def test_codegen_config_function(self):
        # [FUNCTION DOC] Stores functions in list, set and function-valued entries of the third
        # fake module and checks that generated code imports the owning modules and renders list
        # members as quoted dotted names, set members unquoted, and the function entry as a dotted
        # reference.
        import logging
        import warnings

        config3.e_list = [print, warnings.warn, logging.warn]
        config3.e_set = {print}
        config3.e_func = warnings.warn
        code = config3.codegen_config()
        self.assertIn("import _warnings", code)
        self.assertIn("import logging", code)
        self.assertIn(
            """torch.testing._internal.fake_config_module3.e_list = ['print', '_warnings.warn', 'logging.warn']
torch.testing._internal.fake_config_module3.e_set = { print }
torch.testing._internal.fake_config_module3.e_func = _warnings.warn""",
            code,
        )

    def test_get_hash(self):
        # [FUNCTION DOC] Checks get_hash against a fixed expected digest, that repeated calls
        # return the cached value, that a manually injected cache value is returned, that changing
        # a hashed entry changes the hash, and that changing a compile-ignored entry does not.
        hash_value = b"#d\x8b\xd3\xbc'\xf5\x0c\xcd\xb6\x87zDw6g"

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )
            # Test cached value
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )
            self.assertEqual(
                config.get_hash(),
                hash_value,
            )

        config._hash_cache_var.set(b"fake")
        self.assertEqual(config.get_hash(), b"fake")

        config.e_bool = False
        self.assertNotEqual(
            config.get_hash(),
            hash_value,
        )
        config.e_bool = True

        # Test ignored values
        config.e_compile_ignored = False
        self.assertEqual(
            config.get_hash(),
            hash_value,
        )

    def test_dict_copy_semantics(self):
        # [FUNCTION DOC] Compares shallow_copy_dict, to_dict and get_config_copy results with the
        # full expected mapping (including ignored keys), then mutates a nested dict default and
        # verifies the earlier copies are unaffected by that mutation.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p = config.shallow_copy_dict()
            p2 = config.to_dict()

        self.assertDictEqual(
            p,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )
        self.assertEqual(
            p2,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            p3 = config.get_config_copy()

        self.assertEqual(
            p3,
            {
                "e_bool": True,
                "e_dict": {1: 2},
                "e_float": 1.0,
                "e_int": 1,
                "e_list": [1],
                "e_none": None,
                "e_set": {1},
                "e_string": "string",
                "e_tuple": (1,),
                "nested.e_bool": True,
                "e_ignored": True,
                "_e_ignored": True,
                "e_compile_ignored": True,
                "_cache_config_ignore_prefix": ["magic_cache_config"],
                "_save_config_ignore": ["e_ignored"],
                "magic_cache_config_ignored": True,
                "e_config": True,
                "e_jk": True,
                "e_jk_false": False,
                "e_env_default": True,
                "e_env_default_FALSE": False,
                "e_env_default_str": "1234",
                "e_env_default_str_empty": "",
                "e_env_force": True,
                "e_optional": True,
                "e_deprecated": True,
                "e_not_deprecated": False,
            },
        )

        # Shallow + deep copy semantics
        config.e_dict[2] = 3
        self.assertEqual(p["e_dict"], {1: 2})
        self.assertEqual(p2["e_dict"], {1: 2})
        self.assertEqual(p3["e_dict"], {1: 2})

    def test_patch(self):
        # [FUNCTION DOC] Checks patch as a context manager with positional name/value and keyword
        # forms, that the value is restored on exit, and that patching an unknown name raises
        # AssertionError.
        self.assertTrue(config.e_bool)
        with config.patch("e_bool", False):
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)
        with self.assertRaises(AssertionError):
            with config.patch("does_not_exist"):
                pass

    def _assert_patch_reentrant_across_threads(self, fn):
        # [FUNCTION DOC] HELPER: Runs the given function in two threads with a shared two-party
        # barrier, collects any exceptions as text, and asserts none occurred and the config is
        # back to default.
        barrier = threading.Barrier(2, timeout=5)
        errors: queue.SimpleQueue[str] = queue.SimpleQueue()

        def worker():
            # [FUNCTION DOC] HELPER: Thread body that calls the supplied function with the barrier
            # and records any exception representation in the shared queue.
            try:
                fn(barrier)
            except Exception as e:
                errors.put(repr(e))

        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        error_messages = []
        while True:
            try:
                error_messages.append(errors.get_nowait())
            except queue.Empty:
                break

        self.assertFalse(
            error_messages,
            lambda msg: f"{msg}\nconcurrent patch usage failed: {error_messages}",
        )
        self.assertTrue(config.e_bool)

    def test_patch_context_manager_is_reentrant_across_threads(self):
        # [FUNCTION DOC] Uses one patcher object as a context manager in two threads at once and
        # checks the patched value is seen in both before and after the barrier (reentrant across
        # threads).
        patcher = config.patch(e_bool=False)

        def fn(barrier):
            # [FUNCTION DOC] HELPER: Enters the shared patcher, checks the value is patched, waits
            # at the barrier, and checks again.
            with patcher:
                self.assertFalse(config.e_bool)
                barrier.wait()
                self.assertFalse(config.e_bool)

        self._assert_patch_reentrant_across_threads(fn)

    def test_patch_context_manager_is_reentrant_when_nested(self):
        # [FUNCTION DOC] Enters the same patcher object twice, nested, and checks the patched
        # value holds through the inner exit and is restored only after the outer exit.
        patcher = config.patch(e_bool=False)

        with patcher:
            self.assertFalse(config.e_bool)
            with patcher:
                self.assertFalse(config.e_bool)
            self.assertFalse(config.e_bool)

        self.assertTrue(config.e_bool)

    def test_patch_decorator_is_reentrant_across_threads(self):
        # [FUNCTION DOC] Uses patch as a function decorator and runs the decorated function in two
        # threads, checking the patched value is visible in each (reentrant across threads).
        @config.patch(e_bool=False)
        def fn(barrier):
            # [FUNCTION DOC] HELPER: Decorated function body; checks the patched value, waits at
            # the barrier, checks again.
            self.assertFalse(config.e_bool)
            barrier.wait()
            self.assertFalse(config.e_bool)

        self._assert_patch_reentrant_across_threads(fn)

    def test_make_closur_patcher(self):
        # [FUNCTION DOC] Checks the closure patcher: calling the returned change function applies
        # the override and returns a revert callable that restores the original value.
        revert = config._make_closure_patcher(e_bool=False)()
        self.assertFalse(config.e_bool)
        revert()
        self.assertTrue(config.e_bool)

    def test_unittest_patch(self):
        # [FUNCTION DOC] Checks that unittest.mock patching of a config attribute, including
        # nested patches of the same name, behaves correctly and restores the original value
        # afterwards.
        with patch("torch.testing._internal.fake_config_module.e_bool", False):
            with patch("torch.testing._internal.fake_config_module.e_bool", False):
                self.assertFalse(config.e_bool)
            # unittest.mock has some very weird semantics around deletion of attributes when undoing patches
            self.assertFalse(config.e_bool)
        self.assertTrue(config.e_bool)

    def test_bad_jk_type(self):
        # [FUNCTION DOC] Checks that constructing a config entry with a non-boolean default and a
        # justknob raises AssertionError.
        with self.assertRaises(
            AssertionError,
            msg="AssertionError: justknobs only support booleans, thisisnotvalid is not a boolean",
        ):
            _ConfigEntry(Config(default="bad", justknob="fake_knob"), "test")

    def test_alias(self):
        # [FUNCTION DOC] Checks aliasing between the two fake modules: both read the same default,
        # and patching either the aliasing or the aliased name changes the value seen through
        # both.
        self.assertFalse(config2.e_aliasing_bool)
        self.assertFalse(config.e_aliased_bool)
        with config2.patch(e_aliasing_bool=True):
            self.assertTrue(config2.e_aliasing_bool)
            self.assertTrue(config.e_aliased_bool)
        with config.patch(e_aliased_bool=True):
            self.assertTrue(config2.e_aliasing_bool)

    def test_reference_is_default(self):
        # [FUNCTION DOC] Checks that the is-default query is true for an untouched mutable default
        # and false after the returned dict has been mutated in place.
        t = config.e_dict
        self.assertTrue(config._is_default("e_dict"))
        t["a"] = "b"
        self.assertFalse(config._is_default("e_dict"))

    def test_invalid_config_int(self):
        # [FUNCTION DOC] Checks that an integer entry with a default env name and an int value
        # type is rejected with AssertionError at construction.
        with self.assertRaises(AssertionError):
            _ConfigEntry(
                Config(default=2, env_name_default="FAKE_DISABLE", value_type=int),
                "test",
            )

    def test_invalid_config_float(self):
        # [FUNCTION DOC] Checks that a float entry with a forced env name and a float value type
        # is rejected with AssertionError at construction.
        with self.assertRaises(AssertionError):
            _ConfigEntry(
                Config(default=2, env_name_force="FAKE_DISABLE", value_type=float),
                "test",
            )

    def test_deprecated_config(self):
        # [FUNCTION DOC] Checks that reading or writing a deprecated entry emits exactly one
        # FutureWarning naming the entry, saying it is no longer needed and will be removed, and
        # that the entry still works normally including under patch.
        """Test that deprecated configs warn on access but still work normally."""
        config._config["e_deprecated"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Read access triggers warning
            self.assertTrue(config.e_deprecated)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[0].category, FutureWarning))
            msg = str(w[0].message)
            self.assertIn("e_deprecated", msg)
            self.assertIn("is no longer needed", msg)
            self.assertIn("will be removed in a future version of PyTorch", msg)

        # Write access also triggers warning
        config._config["e_deprecated"]._deprecation_warned = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config.e_deprecated = False
            self.assertEqual(len(w), 1)
            self.assertFalse(config.e_deprecated)

        # Functionality still works: patch, multiple accesses, etc.
        config.e_deprecated = True
        with config.patch(e_deprecated=False):
            self.assertFalse(config.e_deprecated)
        self.assertTrue(config.e_deprecated)

    def test_deprecated_config_warn_once(self):
        # [FUNCTION DOC] Checks that the deprecation warning is emitted only once per entry across
        # repeated reads and a write.
        """Test that deprecation warning is only issued once per config."""
        config._config["e_deprecated"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # First access
            _ = config.e_deprecated
            # Second access
            _ = config.e_deprecated
            config.e_deprecated = True  # Write - no warn
            _ = config.e_deprecated  # Read again - no warn
            self.assertEqual(len(w), 1)

    def test_deprecated_alias_config(self):
        # [FUNCTION DOC] Checks that an aliased deprecated entry warns with its own name and
        # custom message, and that its value comes from the entry it aliases.
        """Test that aliased deprecated configs issue their own warning."""
        config._config["e_deprecated_alias"]._deprecation_warned = False

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            value = config.e_deprecated_alias
            # Comes from e_not_deprecated = False
            self.assertFalse(value)
            self.assertEqual(len(w), 1)
            self.assertIn("e_deprecated_alias", str(w[0].message))
            self.assertIn("use something else instead", str(w[0].message))

    def test_no_warnings_for_normal_user_code(self):
        # [FUNCTION DOC] Compiles and runs a trivial function with the eager backend and checks
        # that no FutureWarning mentioning deprecated config was emitted.
        """Test that normal user operations don't produce deprecation warnings."""
        import torch

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Simple user code - compile and run a function
            @torch.compile(backend="eager")
            def fn(x):
                # [FUNCTION DOC] HELPER: Compiled function that returns its input plus one.
                return x + 1

            _ = fn(torch.ones(3))

            # No deprecation warnings should be issued from config system
            deprecation_warnings = [
                warning
                for warning in w
                if issubclass(warning.category, FutureWarning)
                and "deprecated" in str(warning.message).lower()
                and "config" in str(warning.message).lower()
            ]
            self.assertEqual(
                len(deprecation_warnings),
                0,
                lambda msg: f"{msg}\nUnexpected config deprecation warnings: {[str(x.message) for x in deprecation_warnings]}",
            )

    def test_patch_then_global(self):
        # [FUNCTION DOC] Checks that a global assignment after a patch context has exited sticks.
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config.e_bool)

        config.e_bool = False
        self.assertFalse(config.e_bool)

    def test_is_default_patch(self):
        # [FUNCTION DOC] Checks that an entry is not considered default while a patch overrides
        # it.
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            self.assertFalse(config._is_default("e_bool"))

    def test_dict_patch(self):
        # [FUNCTION DOC] Checks that the internal dict snapshot reflects a value patched at that
        # time.
        self.assertTrue(config.e_bool)
        with config.patch(e_bool=False):
            d = config._get_dict()
            self.assertFalse(d["e_bool"])

    def test_set_in_patch(self):
        # [FUNCTION DOC] Checks that assigning inside a patch context takes effect, but that
        # leaving the patch restores the original default, discarding the explicit assignment.
        self.assertEqual(config.e_int, 1)
        with config.patch(e_int=2):
            self.assertEqual(config.e_int, 2)
            config.e_int = 3
            self.assertEqual(config.e_int, 3)
        # Exiting the patch resets e_int to 1, losing the fact that we explicitly set it to 4 - see ConfigModule._do_setattr
        self.assertEqual(config.e_int, 1)

    def test_nested_patch(self):
        # [FUNCTION DOC] Checks nested patches of overlapping entries: inner values shadow outer
        # ones, unpatched entries keep outer values, and each exit restores the previous layer.
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_string, "string")
        with config.patch(e_int=2, e_string="inner"):
            self.assertEqual(config.e_int, 2)
            self.assertEqual(config.e_string, "inner")
            with config.patch(e_int=3):
                self.assertEqual(config.e_int, 3)
                self.assertEqual(config.e_string, "inner")
                config.e_int = 4
                self.assertEqual(config.e_int, 4)
            self.assertEqual(config.e_int, 2)
        self.assertEqual(config.e_int, 1)
        self.assertEqual(config.e_string, "string")

    def test_get_dict_readonly_values_caching(self):
        # [FUNCTION DOC] Checks that two readonly snapshots of the config dict compare equal.
        """readonly_values=True should return cached results on repeated calls."""
        d1 = config._get_dict(readonly_values=True)
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d1, d2)

    def test_get_dict_readonly_values_cache_invalidation(self):
        # [FUNCTION DOC] Checks that changing an entry after a readonly snapshot is reflected in
        # the next readonly snapshot.
        """Mutating a config key should be reflected in subsequent cached calls."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        config.e_int = 42
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d2["e_int"], 42)

    def test_get_dict_readonly_values_does_not_mutate_prior_result(self):
        # [FUNCTION DOC] Checks that changing config after taking a readonly snapshot does not
        # alter the snapshot already returned.
        """Mutating config after a cached call should not affect the prior result."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        config.e_int = 42
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        self.assertEqual(d2["e_int"], 42)

    def test_get_dict_readonly_values_multiple_cache_keys(self):
        # [FUNCTION DOC] Checks that snapshots taken with different ignored-key filters are cached
        # independently and both pick up a later change.
        """Different filter arguments should produce independent cached results."""
        d_all = config._get_dict(readonly_values=True)
        d_filtered = config._get_dict(ignored_keys=["e_int"], readonly_values=True)
        self.assertIn("e_int", d_all)
        self.assertNotIn("e_int", d_filtered)

        # Mutate and verify both cache entries update correctly
        config.e_string = "changed"
        d_all2 = config._get_dict(readonly_values=True)
        d_filtered2 = config._get_dict(ignored_keys=["e_int"], readonly_values=True)
        self.assertEqual(d_all2["e_string"], "changed")
        self.assertEqual(d_filtered2["e_string"], "changed")

    def test_get_dict_readonly_values_full_dirty_clears_cache(self):
        # [FUNCTION DOC] Sets every non-alias entry up to one more than the dirty-key cap to its
        # default, then checks the dirty-key tracker becomes None (fully dirty) and a snapshot is
        # still produced.
        """Exceeding the dirty keys cap should trigger a full cache rebuild."""
        config._get_dict(readonly_values=True)
        # Mutate more distinct keys than the cap to force fully dirty state.
        all_keys = [k for k in config._config if config._config[k].alias is None]
        self.assertGreater(len(all_keys), config._GET_DICT_DIRTY_KEYS_CAP)
        for key in all_keys[: config._GET_DICT_DIRTY_KEYS_CAP + 1]:
            setattr(config, key, config._config[key].default)
        self.assertIsNone(config._get_dict_dirty_keys_var.get())
        d = config._get_dict(readonly_values=True)
        self.assertIsNotNone(d)

    def test_get_dict_non_readonly_ignores_cache(self):
        # [FUNCTION DOC] Checks that non-readonly snapshots do not populate the cache variable and
        # always reflect current values.
        """Non-readonly calls should not use or pollute the cache."""
        d1 = config._get_dict(readonly_values=False)
        self.assertIsNone(config._get_dict_cache_var.get())
        config.e_int = 42
        d2 = config._get_dict(readonly_values=False)
        self.assertEqual(d2["e_int"], 42)
        self.assertEqual(d1["e_int"], 1)

    def test_get_dict_cache_not_poisoned_by_caller_mutation(self):
        # [FUNCTION DOC] Checks that mutating a returned readonly snapshot does not corrupt later
        # cached snapshots.
        """Mutating a returned dict must not affect subsequent cached calls."""
        d1 = config._get_dict(readonly_values=True)
        d1["e_int"] = 999
        d2 = config._get_dict(readonly_values=True)
        self.assertEqual(d2["e_int"], 1)

    def test_get_dict_cache_invalidated_by_delattr(self):
        # [FUNCTION DOC] Checks that deleting an overridden entry invalidates the cache so the
        # default is returned afterwards.
        """Deleting a config key should invalidate the cache."""
        config.e_int = 42
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 42)
        del config.e_int
        d2 = config._get_dict(readonly_values=True)
        # __delattr__ resets user_override, so the default is returned
        self.assertEqual(d2["e_int"], 1)

    def test_get_dict_cache_invalidated_by_patch(self):
        # [FUNCTION DOC] Checks that snapshots see values changed by patch on entry and the
        # restored value after exit.
        """Config changes via patch() should be reflected in cached results."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        with config.patch(e_int=42):
            d2 = config._get_dict(readonly_values=True)
            self.assertEqual(d2["e_int"], 42)
        d3 = config._get_dict(readonly_values=True)
        self.assertEqual(d3["e_int"], 1)

    def test_get_dict_cache_invalidated_by_closure_patcher(self):
        # [FUNCTION DOC] Checks that applying and reverting a closure patcher is reflected in
        # snapshots.
        """Config changes via _make_closure_patcher should be reflected in cached results."""
        d1 = config._get_dict(readonly_values=True)
        self.assertEqual(d1["e_int"], 1)
        change_fn = config._make_closure_patcher(e_int=42)
        revert = change_fn()
        try:
            d2 = config._get_dict(readonly_values=True)
            self.assertEqual(d2["e_int"], 42)
        finally:
            revert()
        d3 = config._get_dict(readonly_values=True)
        self.assertEqual(d3["e_int"], 1)

    def test_get_hash_per_thread_isolation(self):
        # [FUNCTION DOC] Computes the hash in the main thread, changes an entry in a second thread
        # and hashes there, then checks the main thread hash is unchanged and the thread hash
        # differs. Re-raises any exception captured from the thread.
        """Each thread should compute its own hash based on its own config values."""
        import threading

        base_hash = config.get_hash()

        thread_hash = [None]
        error = [None]

        def thread_fn():
            # [FUNCTION DOC] HELPER: Thread body that sets the integer entry to a new value,
            # stores the hash in a shared list, and stores any exception in a shared list.
            try:
                config.e_int = 999
                thread_hash[0] = config.get_hash()
            except Exception as e:
                error[0] = e

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()
        if error[0] is not None:
            raise error[0]

        # The main thread's hash should be unchanged (e_int still at default)
        self.assertEqual(config.get_hash(), base_hash)
        # The child thread should have gotten a different hash
        self.assertNotEqual(thread_hash[0], base_hash)

    def test_hash_dirty_on_setattr(self):
        # [FUNCTION DOC] Checks the hash dirty flag is cleared after hashing and set after a plain
        # write.
        """Direct config assignment should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        config.e_int = 42
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_delattr(self):
        # [FUNCTION DOC] Checks the hash dirty flag is set after deleting an overridden entry.
        """Deleting a config key should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        del config.e_int
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_patch(self):
        # [FUNCTION DOC] Checks the dirty flag is set on entering a patch, cleared by hashing, and
        # set again after exit restores old values.
        """Entering and exiting a patch should mark the hash dirty."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        with config.patch(e_int=42):
            self.assertTrue(config._hash_dirty_var.get())
            config.get_hash()
            self.assertFalse(config._hash_dirty_var.get())
        # Exiting the patch restores old values via __setattr__, which marks dirty
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_closure_patcher(self):
        # [FUNCTION DOC] Checks the dirty flag is set by applying a closure patcher and again by
        # reverting it.
        """_make_closure_patcher should mark the hash dirty on apply and revert."""
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        change_fn = config._make_closure_patcher(e_int=42)
        revert = change_fn()
        self.assertTrue(config._hash_dirty_var.get())
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        revert()
        self.assertTrue(config._hash_dirty_var.get())

    def test_hash_dirty_on_load_config(self):
        # [FUNCTION DOC] Checks the dirty flag is set after loading a previously saved config.
        """Loading a saved config should mark the hash dirty."""
        saved = config.save_config()
        config.get_hash()
        self.assertFalse(config._hash_dirty_var.get())
        config.load_config(saved)
        self.assertTrue(config._hash_dirty_var.get())


if __name__ == "__main__":
    run_tests()
