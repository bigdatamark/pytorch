# [MODULE DOC]
# Second fake config module for config tests. Defines the target of an alias used by
# fake_config_module and entries with multiple environment variable names for default and
# force overrides. Calls install_config_module on itself at import.

import sys

from torch.utils._config_module import Config, install_config_module


e_aliasing_bool = False

e_env_default_multi: bool = Config(
    env_name_default=["ENV_TRUE", "ENV_FALSE"], default=False
)
e_env_force_multi: bool = Config(env_name_force=["ENV_FAKE", "ENV_TRUE"], default=False)

install_config_module(sys.modules[__name__])
