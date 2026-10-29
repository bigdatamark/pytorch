# [MODULE DOC]
# Fake configuration module used by tests of torch.utils._config_module. Declares config
# values of many kinds (plain scalars and containers, optional, ignored, compile-ignored,
# Config entries with justknob, environment-variable defaults/overrides, aliases and
# deprecation) plus a nested class namespace. At import it calls install_config_module on
# itself, replacing the module class with the config-aware one.

import sys

from torch.utils._config_module import Config, install_config_module


e_bool = True
e_int = 1
e_float = 1.0
e_string = "string"
e_list = [1]
e_set = {1}
e_tuple = (1,)
e_dict = {1: 2}
e_none: bool | None = None
e_optional: bool | None = True
e_ignored = True
_e_ignored = True
magic_cache_config_ignored = True
# [@compile_ignored: debug]
e_compile_ignored = True
e_config: bool = Config(default=True)
e_jk: bool = Config(justknob="does_not_exist", default=True)
e_jk_false: bool = Config(justknob="does_not_exist", default=False)
e_env_default: bool = Config(env_name_default="ENV_TRUE", default=False)
e_env_default_FALSE: bool = Config(env_name_default="ENV_FALSE", default=True)
e_env_default_str: bool = Config(env_name_default="ENV_STR", default="default")
e_env_default_str_empty: bool = Config(
    env_name_default="ENV_STR_EMPTY", default="default"
)
e_env_force: bool = Config(env_name_force="ENV_TRUE", default=False)
e_aliased_bool: bool = Config(
    alias="torch.testing._internal.fake_config_module2.e_aliasing_bool"
)
e_deprecated: bool = Config(
    default=True, deprecated=True, deprecation_message="is no longer needed"
)
e_not_deprecated: bool = Config(default=False)
e_deprecated_alias: bool = Config(
    alias="torch.testing._internal.fake_config_module.e_not_deprecated",
    deprecated=True,
    deprecation_message="use something else instead",
)


class nested:
    # [CLASS DOC] HELPER: Namespace class holding one nested config value, e_bool, to test
    # config entries grouped in sub-namespaces. Has no methods or instance state.
    e_bool = True


_cache_config_ignore_prefix = ["magic_cache_config"]
_save_config_ignore = ["e_ignored"]

install_config_module(sys.modules[__name__])
