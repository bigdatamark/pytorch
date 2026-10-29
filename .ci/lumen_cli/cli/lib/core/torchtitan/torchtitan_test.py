# [MODULE DOC] Runner class for the torchtitan CI test target: clones torchtitan,
# installs nightly dependencies and runs a chosen test plan. Import computes the path of
# the repository scripts directory used to find the CUDA version table.
import logging
import sys
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.pip_helper import pip_install_packages
from cli.lib.common.utils import working_directory
from cli.lib.core.torchtitan.lib import (
    clone_torchtitan,
    load_torchtitan_test_library,
    run_test_plan,
)


logger = logging.getLogger(__name__)

# generate_binary_build_matrix.py is not importable as a package (it lives in
# .github/scripts, outside the cli package), so add that dir to sys.path the
# same way .github/scripts/get_ci_variable.py does.
_SCRIPTS_DIR = Path(__file__).resolve().parents[6] / ".github" / "scripts"


def _nightly_index_url() -> str:
    # [FUNCTION DOC] HELPER: Build the PyTorch nightly wheel index URL for the stable
    # CUDA version, importing CUDA_STABLE from the repository scripts directory (which
    # is added to the module search path as a side effect) and dropping the dot.
    # torchao and torchcomms nightlies must match the CUDA toolchain of the
    # build, so reuse CUDA_STABLE from generate_binary_build_matrix.py (the
    # single source of truth for the stable CUDA version, e.g. "13.0" -> cu130)
    # rather than hardcoding the wheel channel here.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from generate_binary_build_matrix import CUDA_STABLE

    return f"https://download.pytorch.org/whl/nightly/cu{CUDA_STABLE.replace('.', '')}"


class TorchtitanTestRunner(BaseRunner):
    # [CLASS DOC] Runner for the torchtitan test target. State: work_directory (clone
    # location, torchtitan) and test_plan (name chosen on the command line). Key methods:
    # prepare sets up the environment and run executes the plan.
    def __init__(self, args: Any):
        # [FUNCTION DOC] Record the work directory and the test plan from the parsed
        # arguments. KNOWN ISSUE: does not call the base initializer, so args is unset.
        self.work_directory = "torchtitan"
        self.test_plan = args.test_plan

    def prepare(self):
        # [FUNCTION DOC] Clone torchtitan, install the nightly torchao and torchcomms
        # wheels from the nightly index, install helion, then inside the clone install
        # torchtitan in editable mode plus pytest and pytest-cov. Runs pip commands.
        clone_torchtitan(dst=self.work_directory)
        # torchao and torchcomms nightlies are required by torchtitan
        pip_install_packages(
            packages=[
                "--pre",
                "torchao",
                "torchcomms",
                "--index-url",
                _nightly_index_url(),
            ],
        )
        pip_install_packages(packages=["helion"])
        with working_directory(self.work_directory):
            pip_install_packages(packages=["-e", "."])
            pip_install_packages(packages=["pytest", "pytest-cov"])

    def run(self):
        # [FUNCTION DOC] Prepare the environment, then run the selected test plan from
        # the loaded library inside the clone. Raises RuntimeError on unknown plan or
        # failed steps.
        self.prepare()
        with working_directory(self.work_directory):
            run_test_plan(self.test_plan, load_torchtitan_test_library())
