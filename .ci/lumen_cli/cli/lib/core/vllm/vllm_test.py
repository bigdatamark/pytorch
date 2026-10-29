# [MODULE DOC] Test runner for the vLLM CI target: parameters for locating torch and vLLM
# wheels, the runner that clones vLLM, installs wheels and test dependencies, sets the
# required environment and runs a test plan, plus helpers to rewrite the test
# requirements and check installed versions. Import computes the repository scripts path.
import logging
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from cli.lib.common.cli_helper import BaseRunner
from cli.lib.common.envs_helper import env_path_field, env_str_field, get_env
from cli.lib.common.path_helper import copy, get_path, remove_dir
from cli.lib.common.pip_helper import (
    pip_install_first_match,
    pip_install_packages,
    pkg_exists,
    run_python,
)
from cli.lib.common.utils import run_command, working_directory
from cli.lib.core.vllm.lib import clone_vllm, run_test_plan, sample_vllm_test_library


logger = logging.getLogger(__name__)

# generate_binary_build_matrix.py is not importable as a package (it lives in
# .github/scripts, outside the cli package), so add that dir to sys.path the
# same way .github/scripts/get_ci_variable.py does.
_SCRIPTS_DIR = Path(__file__).resolve().parents[6] / ".github" / "scripts"


def _stable_torch_backend() -> str:
    # [FUNCTION DOC] HELPER: Return the uv torch backend name (cu plus the stable CUDA
    # version without dots), importing CUDA_STABLE from the repository scripts
    # directory, which is added to the module search path as a side effect.
    # The uv --torch-backend channel must match the CUDA toolchain of the
    # build, so reuse CUDA_STABLE from generate_binary_build_matrix.py (the
    # single source of truth for the stable CUDA version, e.g. "13.0" -> cu130)
    # rather than hardcoding the channel here.
    if str(_SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR))
    from generate_binary_build_matrix import CUDA_STABLE

    return f"cu{CUDA_STABLE.replace('.', '')}"


@dataclass
class VllmTestParameters:
    # [CLASS DOC] Dataclass of vLLM test inputs read from environment variables at
    # creation: torch wheel directory, vLLM wheel directory, CUDA arch list and the
    # cleaning script path. Invariant: both wheel directories exist (checked on creation).
    """
    Parameters defining the vllm external test input

    !!!DO NOT ADD SECRETS IN THIS CLASS!!!
    you can put environment variable name in VllmTestParameters if it's not the same as the secret one
    fetch secrests directly from env variables during runtime
    """

    torch_whls_path: Path = env_path_field("WHEELS_PATH", "./dist")

    vllm_whls_path: Path = env_path_field(
        "VLLM_WHEELS_PATH", "./dist/external/vllm/wheels"
    )

    torch_cuda_arch_list: str = env_str_field("TORCH_CUDA_ARCH_LIST", "8.9")

    cleaning_script: Path = env_path_field(
        "cleaning_script", ".github/ci_configs/vllm/use_existing_torch.py"
    )

    def __post_init__(self):
        # [FUNCTION DOC] Raise ValueError if the torch or vLLM wheel directory is missing.
        if not self.torch_whls_path.exists():
            raise ValueError("missing torch_whls_path")
        if not self.vllm_whls_path.exists():
            raise ValueError("missing vllm_whls_path")


class TestInpuType(Enum):
    # [CLASS DOC] Enumeration of how the test input is given: a named test plan or
    # unknown. KNOWN ISSUE: the name is misspelled and the Test prefix may make test
    # collectors pick it up.
    TEST_PLAN = "test_plan"
    UNKNOWN = "unknown"


class VllmTestRunner(BaseRunner):
    # [CLASS DOC] Runner for the vllm test target. State: work_directory, test_plan,
    # test_type, shard_id and num_shards from the arguments, and glob patterns for the
    # torch, torchvision, torchaudio and vLLM wheels. Key methods: prepare builds the
    # environment, run executes the plan, with install helpers underneath.
    def __init__(self, args: Any):
        # [FUNCTION DOC] Store shard settings and the test plan from the parsed
        # arguments (type becomes test plan only if a plan is given) and define the
        # wheel glob patterns and extra torch package. Does not call the base
        # initializer, so args itself is not stored (KNOWN ISSUE).
        self.work_directory = "vllm"
        self.test_plan = ""
        self.test_type = TestInpuType.UNKNOWN

        self.shard_id = args.shard_id
        self.num_shards = args.num_shards

        if args.test_plan:
            self.test_plan = args.test_plan
            self.test_type = TestInpuType.TEST_PLAN

        # Matches the structeur in the artifacts.zip from torcb build
        self.TORCH_WHL_PATH_REGEX = "torch*.whl"
        self.TORCH_WHL_EXTRA = "opt-einsum"
        self.TORCH_ADDITIONAL_WHLS_REGEX = [
            "vision/torchvision*.whl",
            "audio/torchaudio*.whl",
        ]

        # Match the structure of the artifacts.zip from vllm external build
        self.VLLM_TEST_WHLS_REGEX = [
            "vllm/vllm*.whl",
        ]

    def prepare(self):
        # [FUNCTION DOC] Build the test environment: read parameters, set environment
        # variables, clone vLLM, copy the cleaning script, then in the checkout remove
        # its source package directory, install wheels and dependencies; finally log
        # package versions. Raises on invalid parameters or missing secrets.
        """
        prepare test environment for vllm. This includes clone vllm repo, install all wheels, test dependencies and set env
        """
        params = VllmTestParameters()
        logger.info("Display VllmTestParameters %s", params)
        self._set_envs(params)

        clone_vllm(dst=self.work_directory)
        self.cp_torch_cleaning_script(params)
        with working_directory(self.work_directory):
            remove_dir(Path("vllm"))
            self._install_wheels(params)
            self._install_dependencies()
        # verify the torches are not overridden by test dependencies

        check_versions()

    def run(self):
        # [FUNCTION DOC] Prepare, then in the checkout run the selected test plan
        # (sharded arguments passed only if more than one shard). Raises ValueError for
        # an unknown input type. Package versions are logged afterwards even on failure.
        """
        main function to run vllm test
        """
        self.prepare()
        try:
            with working_directory(self.work_directory):
                if self.test_type == TestInpuType.TEST_PLAN:
                    if self.num_shards > 1:
                        run_test_plan(
                            self.test_plan,
                            "vllm",
                            sample_vllm_test_library(),
                            self.shard_id,
                            self.num_shards,
                        )
                    else:
                        run_test_plan(
                            self.test_plan, "vllm", sample_vllm_test_library()
                        )
                else:
                    raise ValueError(f"Unknown test type {self.test_type}")
        finally:
            # double check the torches are not overridden by other packages
            check_versions()

    def cp_torch_cleaning_script(self, params: VllmTestParameters):
        # [FUNCTION DOC] Copy the cleaning script into the checkout root as
        # use_existing_torch.py.
        script = get_path(params.cleaning_script, resolve=True)
        vllm_script = Path(f"./{self.work_directory}/use_existing_torch.py")
        copy(script, vllm_script)

    def _install_wheels(self, params: VllmTestParameters):
        # [FUNCTION DOC] HELPER: Install torch (with its extra package) from the local
        # wheel directory if torch is not already installed, then the torchvision and
        # torchaudio wheels and the vLLM wheel, each the first match of its pattern.
        logger.info("Running vllm test with inputs: %s", params)
        if not pkg_exists("torch"):
            # install torch from local whls if it's not installed yet.
            torch_p = f"{str(params.torch_whls_path)}/{self.TORCH_WHL_PATH_REGEX}"
            pip_install_first_match(torch_p, self.TORCH_WHL_EXTRA)

        torch_whls_path = [
            f"{str(params.torch_whls_path)}/{whl_path}"
            for whl_path in self.TORCH_ADDITIONAL_WHLS_REGEX
        ]
        for torch_whl in torch_whls_path:
            pip_install_first_match(torch_whl)
        logger.info("Done. Installed torch and other torch-related wheels ")

        logger.info("Installing vllm wheels")
        vllm_whls_path = [
            f"{str(params.vllm_whls_path)}/{whl_path}"
            for whl_path in self.VLLM_TEST_WHLS_REGEX
        ]
        for vllm_whl in vllm_whls_path:
            pip_install_first_match(vllm_whl)
        logger.info("Done. Installed vllm wheels")

    def _install_test_dependencies(self):
        # [FUNCTION DOC] HELPER: Rewrite the CUDA test requirements input to pin the
        # local torch wheels, save the existing compiled requirements as a constraint
        # snapshot, compile a new requirements file with uv using the stable torch
        # backend, and install it. Runs subprocess commands.
        """
        This method replaces torch dependencies with local torch wheel info in
        requirements/test/cuda.in file from vllm repo. then generates the test.txt
        in runtime
        """
        logger.info(
            "generate test.txt from requirements/test/cuda.in with local torch whls"
        )
        preprocess_test_in()
        copy("requirements/test/cuda.txt", "snapshot_constraint.txt")

        run_command(
            f"{sys.executable} -m uv pip compile requirements/test/cuda.in "
            "-o test/cuda.txt "
            "--index-strategy unsafe-best-match "
            "--constraint snapshot_constraint.txt "
            f"--torch-backend {_stable_torch_backend()}"
        )
        pip_install_packages(requirements="test/cuda.txt", prefer_uv=True)
        logger.info("Done. installed requirements for test dependencies")

    def _install_dependencies(self):
        # [FUNCTION DOC] HELPER: Install the vLLM test utilities and hf_transfer, enable
        # fast Hugging Face transfers through the process environment, run the vLLM
        # script that strips torch from requirement files, install the common and build
        # requirements, then the test dependencies.
        pip_install_packages(packages=["-e", "tests/vllm_test_utils"], prefer_uv=True)
        pip_install_packages(packages=["hf_transfer"], prefer_uv=True)
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

        # using script from vllm repo to remove all torch packages from requirements txt
        run_python("use_existing_torch.py")

        # install common packages
        for requirements in ["requirements/common.txt", "requirements/build/cuda.txt"]:
            pip_install_packages(
                requirements=requirements,
                prefer_uv=True,
            )
        # install test packages
        self._install_test_dependencies()

    def _set_envs(self, inputs: VllmTestParameters):
        # [FUNCTION DOC] HELPER: Set process environment variables: CUDA arch list, the
        # inductor custom-op aliasing check off, and the Hugging Face token copied from
        # a dedicated variable. Warns on unsupported arch values. Raises ValueError if
        # the token or arch list is empty, or on CI when the Hugging Face home is unset.
        os.environ["TORCH_CUDA_ARCH_LIST"] = inputs.torch_cuda_arch_list
        # vLLM has known third-party custom ops with aliasing schemas.
        os.environ["TORCHINDUCTOR_ERROR_ON_CUSTOM_OP_ALIASING"] = "0"
        if not validate_cuda(get_env("TORCH_CUDA_ARCH_LIST")):
            logger.warning(
                "Missing supported TORCH_CUDA_ARCH_LIST. "
                "Currently support TORCH_CUDA_ARCH_LIST env var "
                "with supported arch [8.0, 8.9, 9.0]"
            )

        os.environ["HF_TOKEN"] = os.getenv("VLLM_TEST_HUGGING_FACE_TOKEN", "")
        if not get_env("HF_TOKEN"):
            raise ValueError(
                "missing required HF_TOKEN, please set VLLM_TEST_HUGGING_FACE_TOKEN env var"
            )
        if not get_env("TORCH_CUDA_ARCH_LIST"):
            raise ValueError(
                "missing required TORCH_CUDA_ARCH_LIST, please set TORCH_CUDA_ARCH_LIST env var"
            )
        # HF_HOME is absolutely needed on CI to avoid rate limit to HF, so explicitly fail
        # vLLM jobs when it's not set so that we know when it's missing
        if get_env("CI") and not get_env("HF_HOME"):
            raise ValueError(
                "missing required HF_HOME when running on CI, please set HF_HOME env var"
            )


def preprocess_test_in(
    target_file: str = "requirements/test/cuda.in",
    additional_packages: Iterable[str] = (),
):
    # [FUNCTION DOC] Rewrite a requirements input file in place: drop lines pinning
    # torch, torchvision, torchaudio or the extra packages (matched at line start with
    # equals, at or greater-equal), and prepend the local file-URL pins for the torch
    # packages found in the installed package listing from a pip subprocess.
    # KNOWN ISSUE: the extra packages are inserted into a regex unescaped.
    """
    This modifies the target_file file in place in vllm work directory.
    It removes torch and unwanted packages in target_file and replace with local torch whls
    package with format "$WHEEL_PACKAGE_NAME @ file://<LOCAL_PATH>"
    """
    additional_package_to_move = list(additional_packages or ())
    pkgs_to_remove = [
        "torch",
        "torchvision",
        "torchaudio",
    ] + additional_package_to_move
    # Read current requirements
    target_path = Path(target_file)
    lines = target_path.read_text().splitlines()

    pkgs_to_add = []

    # Remove lines starting with the package names (==, @, >=) — case-insensitive
    pattern = re.compile(rf"^({'|'.join(pkgs_to_remove)})\s*(==|@|>=)", re.IGNORECASE)
    kept_lines = [line for line in lines if not pattern.match(line)]

    # Get local installed torch/vision/audio from pip freeze
    # This is hacky, but it works
    pip_freeze = subprocess.check_output(["pip", "freeze"], text=True)
    header_lines = [
        line
        for line in pip_freeze.splitlines()
        if re.match(
            r"^(torch|torchvision|torchaudio)\s*@\s*file://", line, re.IGNORECASE
        )
    ]

    # Write back: header_lines + blank + kept_lines
    out_lines = header_lines + [""] + kept_lines
    if pkgs_to_add:
        out_lines += [""] + pkgs_to_add

    out = "\n".join(out_lines) + "\n"
    target_path.write_text(out)
    logger.info("[INFO] Updated %s", target_file)


def validate_cuda(value: str) -> bool:
    # [FUNCTION DOC] True when every whitespace separated entry is a supported arch
    # (8.0, 8.9 or 9.0); an empty string passes. KNOWN ISSUE: a semicolon separated
    # list, as shown in the build docs, is rejected.
    VALID_VALUES = {"8.0", "8.9", "9.0"}
    return all(v in VALID_VALUES for v in value.split())


def check_versions():
    # [FUNCTION DOC] Log whether torch, torchvision, torchaudio and vllm are installed
    # and their versions. Informational only; never fails.
    """
    check installed packages version
    """
    logger.info("Double check installed packages")
    patterns = ["torch", "torchvision", "torchaudio", "vllm"]
    for pkg in patterns:
        pkg_exists(pkg)
    logger.info("Done. checked installed packages")
