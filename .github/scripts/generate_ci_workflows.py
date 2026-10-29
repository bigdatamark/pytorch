#!/usr/bin/env python3

# [MODULE DOC] Generator for the binary build GitHub workflow files. Defines the configuration
# dataclasses (ciflow labels, binary build workflow), the supported operating system names and
# the per-platform workflow lists (linux, windows, windows arm64, macOS arm64, linux aarch64,
# s390x), whose build matrices come from generate_binary_build_matrix. Running it as a script
# deletes all existing generated workflow files and renders the Jinja templates in
# .github/templates into workflows/generated-<env>-<branch>.yml. Import-time side effects:
# computes the build matrices for all platforms.
import json
import os
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from typing_extensions import TypedDict  # Python 3.11+

import generate_binary_build_matrix  # type: ignore[import]
import jinja2


Arch = Literal["windows", "linux", "macos"]

GITHUB_DIR = Path(__file__).resolve().parent.parent

LABEL_CIFLOW_TRUNK = "ciflow/trunk"
LABEL_CIFLOW_UNSTABLE = "ciflow/unstable"
LABEL_CIFLOW_BINARIES = "ciflow/binaries"
LABEL_CIFLOW_PERIODIC = "ciflow/periodic"
LABEL_CIFLOW_BINARIES_LIBTORCH = "ciflow/binaries_libtorch"
LABEL_CIFLOW_BINARIES_WHEEL = "ciflow/binaries_wheel"
LABEL_CIFLOW_S390 = "ciflow/s390"


@dataclass
class CIFlowConfig:
    # [CLASS DOC] Settings for which ciflow labels trigger a generated workflow: the canary-run
    # flag, the set of label strings, whether the workflow is isolated from the default trunk
    # or unstable groups, and whether it is unstable. After construction, non-isolated
    # workflows without the periodic label get the trunk label (or the unstable label when
    # unstable) added to the label set.
    # For use to enable workflows to run on pytorch/pytorch-canary
    run_on_canary: bool = False
    labels: set[str] = field(default_factory=set)
    # Certain jobs might not want to be part of the ciflow/[all,trunk] workflow
    isolated_workflow: bool = False
    unstable: bool = False

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Dataclass hook. For a workflow that is not isolated and lacks
        # the periodic label, mutates the label set to add the trunk label, or the unstable
        # label if the unstable flag is set.
        if not self.isolated_workflow:
            if LABEL_CIFLOW_PERIODIC not in self.labels:
                self.labels.add(
                    LABEL_CIFLOW_TRUNK if not self.unstable else LABEL_CIFLOW_UNSTABLE
                )


class Config(TypedDict):
    # [CLASS DOC] HELPER: Typed mapping describing a test config with a shard count and a
    # runner name. Not used elsewhere in this module.
    num_shards: int
    runner: str


@dataclass
class BinaryBuildWorkflow:
    # [CLASS DOC] Description of one generated binary build workflow: target OS, list of build
    # config mappings, package type, build environment name (derived from OS, package type and
    # variant when empty), ciflow config, schedule marker, branch name used in the file name,
    # macOS runner, libtorch variant and libtorch extraction configs, and owner labels (defaulted
    # from the OS and the GPU arch types present). Its main method renders a Jinja template
    # with all fields into a workflow file.
    os: str
    build_configs: list[dict[str, str]]
    package_type: str

    # Optional fields
    build_environment: str = ""
    ciflow_config: CIFlowConfig = field(default_factory=CIFlowConfig)
    is_scheduled: str = ""
    branches: str = "nightly"
    # Mainly for macos
    macos_runner: str = "macos-14-xlarge"
    # Mainly used for libtorch builds
    build_variant: str = ""
    # Libtorch extraction configs: lightweight jobs that extract libtorch
    # from a wheel build instead of building libtorch from scratch
    libtorch_extraction_configs: list[dict[str, str]] = field(default_factory=list)
    # Owner labels emitted as the "# Owner(s):" header; see WORKFLOWOWNERS lint.
    owners: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # [FUNCTION DOC] HELPER: Dataclass hook. If build_environment is empty, sets it to the
        # non-empty parts of OS, the word binary, package type and variant joined by dashes. If
        # no owners were given, sets the releng owner plus a macOS or Windows owner by OS, and
        # rocm or xpu owners when those arch types appear in the build configs.
        if self.build_environment == "":
            self.build_environment = "-".join(
                item
                for item in [self.os, "binary", self.package_type, self.build_variant]
                if item != ""
            )
        if not self.owners:
            self.owners = ["oncall: releng"]
            if self.os in (OperatingSystem.MACOS, OperatingSystem.MACOS_ARM64):
                self.owners.append("module: macos")
            elif self.os in (OperatingSystem.WINDOWS, OperatingSystem.WINDOWS_ARM64):
                self.owners.append("module: windows")
            # A single binary workflow bundles configs for every gpu_arch_type, so
            # add the accelerator-team owners for whichever ones the matrix includes.
            arches = {config.get("gpu_arch_type") for config in self.build_configs}
            if "rocm" in arches:
                self.owners.append("module: rocm")
            if "xpu" in arches:
                self.owners.append("module: xpu")

    def generate_workflow_file(self, workflow_template: jinja2.Template) -> None:
        # [FUNCTION DOC] Renders the template with all dataclass fields as variables and writes
        # workflows/generated-<build environment>-<branches>.yml, preceded by an owner header
        # line and a do-not-edit marker line, ensuring a trailing newline. Prints the output
        # path. On render failure prints the template to standard error and re-raises. Overwrites
        # an existing file.
        # KNOWN ISSUE: the file is opened before rendering, so a render failure leaves a file
        # with only the header; empty rendered content would raise an index error.
        output_file_path = (
            GITHUB_DIR
            / f"workflows/generated-{self.build_environment}-{self.branches}.yml"
        )
        with open(output_file_path, "w") as output_file:
            GENERATED = "generated"  # Note that please keep the variable GENERATED otherwise phabricator will hide the whole file
            output_file.writelines(
                [
                    f"# Owner(s): {json.dumps(self.owners)}\n",
                    f"# @{GENERATED} DO NOT EDIT MANUALLY\n",
                ]
            )
            try:
                content = workflow_template.render(asdict(self))
            except Exception as e:
                print(f"Failed on template: {workflow_template}", file=sys.stderr)
                raise e
            output_file.write(content)
            if content[-1] != "\n":
                output_file.write("\n")
        print(output_file_path)


class OperatingSystem:
    # [CLASS DOC] Namespace of string constants naming the supported operating system and
    # architecture keys (linux, windows, windows-arm64, macos, macos-arm64, linux-aarch64,
    # linux-s390x). Not instantiated.
    LINUX = "linux"
    WINDOWS = "windows"
    WINDOWS_ARM64 = "windows-arm64"
    MACOS = "macos"
    MACOS_ARM64 = "macos-arm64"
    LINUX_AARCH64 = "linux-aarch64"
    LINUX_S390X = "linux-s390x"


_LINUX_WHEEL_CONFIGS = generate_binary_build_matrix.generate_wheels_matrix(
    OperatingSystem.LINUX
)

LINUX_BINARY_BUILD_WORFKLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX,
        package_type="manywheel",
        build_configs=_LINUX_WHEEL_CONFIGS,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_BINARIES,
                LABEL_CIFLOW_BINARIES_WHEEL,
                LABEL_CIFLOW_BINARIES_LIBTORCH,
            },
            isolated_workflow=True,
        ),
        libtorch_extraction_configs=generate_binary_build_matrix.generate_libtorch_extraction_configs(
            OperatingSystem.LINUX,
            _LINUX_WHEEL_CONFIGS,
        ),
    ),
]

_WINDOWS_WHEEL_CONFIGS = generate_binary_build_matrix.generate_wheels_matrix(
    OperatingSystem.WINDOWS
)
_WINDOWS_ARM64_WHEEL_CONFIGS = generate_binary_build_matrix.generate_wheels_matrix(
    OperatingSystem.WINDOWS_ARM64,
    arches=["cpu"],
    python_versions=["3.11", "3.12", "3.13"],
)

WINDOWS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS,
        package_type="wheel",
        build_configs=_WINDOWS_WHEEL_CONFIGS,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_BINARIES,
                LABEL_CIFLOW_BINARIES_WHEEL,
                LABEL_CIFLOW_BINARIES_LIBTORCH,
            },
            isolated_workflow=True,
        ),
        libtorch_extraction_configs=generate_binary_build_matrix.generate_libtorch_extraction_configs(
            OperatingSystem.WINDOWS,
            _WINDOWS_WHEEL_CONFIGS,
        ),
    ),
    BinaryBuildWorkflow(
        os=OperatingSystem.WINDOWS_ARM64,
        package_type="wheel",
        build_configs=_WINDOWS_ARM64_WHEEL_CONFIGS,
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_BINARIES,
                LABEL_CIFLOW_BINARIES_WHEEL,
                LABEL_CIFLOW_BINARIES_LIBTORCH,
            },
            isolated_workflow=True,
        ),
        libtorch_extraction_configs=generate_binary_build_matrix.generate_libtorch_extraction_configs(
            OperatingSystem.WINDOWS_ARM64,
            _WINDOWS_ARM64_WHEEL_CONFIGS,
        ),
    ),
]

_MACOS_ARM64_WHEEL_CONFIGS = generate_binary_build_matrix.generate_wheels_matrix(
    OperatingSystem.MACOS_ARM64
)

MACOS_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.MACOS_ARM64,
        package_type="wheel",
        build_configs=_MACOS_ARM64_WHEEL_CONFIGS,
        macos_runner="macos-26-xlarge",
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_BINARIES,
                LABEL_CIFLOW_BINARIES_WHEEL,
                LABEL_CIFLOW_BINARIES_LIBTORCH,
            },
            isolated_workflow=True,
        ),
        libtorch_extraction_configs=generate_binary_build_matrix.generate_libtorch_extraction_configs(
            OperatingSystem.MACOS_ARM64,
            _MACOS_ARM64_WHEEL_CONFIGS,
        ),
    ),
]

AARCH64_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX_AARCH64,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX_AARCH64
        ),
        ciflow_config=CIFlowConfig(
            labels={LABEL_CIFLOW_BINARIES, LABEL_CIFLOW_BINARIES_WHEEL},
            isolated_workflow=True,
        ),
    ),
]

S390X_BINARY_BUILD_WORKFLOWS = [
    BinaryBuildWorkflow(
        os=OperatingSystem.LINUX_S390X,
        package_type="manywheel",
        build_configs=generate_binary_build_matrix.generate_wheels_matrix(
            OperatingSystem.LINUX_S390X
        ),
        ciflow_config=CIFlowConfig(
            labels={
                LABEL_CIFLOW_BINARIES,
                LABEL_CIFLOW_BINARIES_WHEEL,
                LABEL_CIFLOW_S390,
            },
            isolated_workflow=True,
        ),
    ),
]


def main() -> None:
    # [FUNCTION DOC] Entry point. Creates a Jinja environment with a custom variable start
    # marker and strict undefined handling over the templates directory, pairs each platform
    # workflow list with its template, deletes every existing generated workflow file (errors
    # are printed and ignored), then renders each workflow. Side effects: deletes and writes
    # files under .github/workflows.
    jinja_env = jinja2.Environment(
        variable_start_string="!{{",
        loader=jinja2.FileSystemLoader(str(GITHUB_DIR.joinpath("templates"))),
        undefined=jinja2.StrictUndefined,
    )

    # not ported yet
    template_and_workflows = [
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            LINUX_BINARY_BUILD_WORFKLOWS,
        ),
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            AARCH64_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("linux_binary_build_workflow.yml.j2"),
            S390X_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("windows_binary_build_workflow.yml.j2"),
            WINDOWS_BINARY_BUILD_WORKFLOWS,
        ),
        (
            jinja_env.get_template("macos_binary_build_workflow.yml.j2"),
            MACOS_BINARY_BUILD_WORKFLOWS,
        ),
    ]
    # Delete the existing generated files first, this should align with .gitattributes file description.
    existing_workflows = GITHUB_DIR.glob("workflows/generated-*")
    for w in existing_workflows:
        try:
            os.remove(w)
        except Exception as e:
            print(f"Error occurred when deleting file {w}: {e}")

    for template, workflows in template_and_workflows:
        # added Iterable check to appease the mypy gods
        if not isinstance(workflows, Iterable):
            raise Exception(  # noqa: TRY002
                f"How is workflows not iterable? {workflows}"
            )
        for workflow in workflows:
            workflow.generate_workflow_file(workflow_template=template)


if __name__ == "__main__":
    main()
