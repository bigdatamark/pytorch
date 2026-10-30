# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Sanity test that every entry registered in the model generator's module table is an
# instance of a torch module. Imports the table from generate_models, which itself imports the
# fixture sources and sets up logging to standard output at import time.

import torch
from test.jit.fixtures_srcs.generate_models import ALL_MODULES
from torch.testing._internal.common_utils import run_tests, TestCase


class TestUpgraderModelGeneration(TestCase):
    # [CLASS DOC] Test case holding the single check over the registered upgrader test modules.
    def test_all_modules(self):
        # [FUNCTION DOC] Iterates over the keys of the module table and asserts each one is an
        # instance of the torch module base class. The assertion message is supplied as a callable
        # that builds text naming the offending module type. Reads only; no side effects.
        for a_module in ALL_MODULES:
            module_name = type(a_module).__name__
            self.assertTrue(
                isinstance(a_module, torch.nn.Module),
                lambda msg: f"{msg}\nThe module {module_name} "
                f"is not a torch.nn.module instance. "
                f"Please ensure it's a subclass of torch.nn.module in fixtures_src.py"
                f"and it's registered as an instance in ALL_MODULES in generated_models.py",
            )


if __name__ == "__main__":
    run_tests()
