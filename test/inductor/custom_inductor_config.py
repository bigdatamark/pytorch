# Owner(s): ["module: inductor"]
# [MODULE DOC] HELPER: Minimal custom config module used by test_codecache to check that
# FXGraphHashDetails accounts for a third-party backend config. Defines one boolean flag
# (enable_optimisation, default False) and, at import time, calls install_config_module on
# itself so the module gains the standard config helpers (patch, save_config, and so on).

# This module is used in test_codecache.py to verify the correctness
# of FXGraphHashDetails when a custom inductor backend registers its own
# config object

import sys

from torch.utils._config_module import install_config_module


enable_optimisation: bool = False

# adds patch, save_config, etc
install_config_module(sys.modules[__name__])
