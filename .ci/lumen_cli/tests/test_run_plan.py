# [MODULE DOC] Pytest tests for run_test_plan and the disabled-vLLM-test helpers in the
# cli.lib.core.vllm.lib module. The first group patches the module's collaborators (package
# install, env/workdir context managers, command runner, logger, disabled-test loader) and
# checks step execution, failure aggregation and flag injection. The second group exercises
# the flag builder, issue-body parser and YAML/GitHub loaders directly with fakes.
# Import-time effect: only module-name constants are defined; the target module is imported
# lazily inside fixtures.
# tests/test_run_test_plan.py
from __future__ import annotations

import importlib
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch as mock_patch

import pytest


MOD = "cli.lib.core.vllm.lib"

# We import inside tests so the MOD override above applies everywhere
run_test_plan_import_path = f"{MOD}.run_test_plan"


def _get_cmd(c):
    # [FUNCTION DOC] HELPER: Return the command string from a recorded mock call, looking in
    # the keyword argument named cmd first, then the first positional argument (None if none).
    # Support both kwargs and positional args
    return c.kwargs.get("cmd", c.args[0] if c.args else None)


def _get_check(c):
    # [FUNCTION DOC] HELPER: Return the check flag of a recorded mock call: the keyword
    # argument if given, else the second positional argument if present, else False.
    if "check" in c.kwargs:
        return c.kwargs["check"]
    # If positional, assume second arg is 'check' when present; default False
    return c.args[1] if len(c.args) > 1 else False


@pytest.fixture
def patch_module(monkeypatch):
    # [FUNCTION DOC] HELPER: Pytest fixture that imports the vllm lib module and replaces its
    # pip_install_packages, run_command (returns 0), working_directory, temp_environ, logger
    # and _load_disabled_vllm_tests (returns an empty list) with mocks/fakes via monkeypatch.
    # The fake context managers record their arguments and do nothing else. Returns a
    # namespace exposing the module, run_test_plan, the mocks and the recorded call lists.
    # Raises an error if any patched attribute does not exist in the module.
    """
    Patch helpers ('pip_install_packages', 'temp_environ', 'working_directory',
    'run_command', 'logger', '_load_disabled_vllm_tests') inside the target
    module and expose them.
    """
    module = importlib.import_module(MOD)

    # Create fakes/mocks
    pip_install_packages = MagicMock(name="pip_install_packages")
    run_command = MagicMock(name="run_command", return_value=0)
    disabled_mock = MagicMock(name="_load_disabled_vllm_tests", return_value=[])

    # temp_environ / working_directory: record calls but act as context managers
    temp_calls: list[dict] = []
    workdir_calls: list[str] = []

    def fake_working_directory(path: str):
        # [FUNCTION DOC] HELPER: Fake working_directory; records the path and returns a no-op
        # context manager.
        workdir_calls.append(path)
        return nullcontext()

    def fake_temp_env(map: dict[str, str]):
        # [FUNCTION DOC] HELPER: Fake temp_environ; records the mapping and returns a no-op
        # context manager.
        temp_calls.append(map)
        return nullcontext()

    logger = SimpleNamespace(
        info=MagicMock(name="logger.info"),
        warning=MagicMock(name="logger.warning"),
        error=MagicMock(name="logger.error"),
    )

    # Apply patches (raise if attribute doesn't exist)
    monkeypatch.setattr(
        module, "pip_install_packages", pip_install_packages, raising=True
    )
    monkeypatch.setattr(module, "run_command", run_command, raising=True)
    monkeypatch.setattr(
        module, "working_directory", fake_working_directory, raising=True
    )
    monkeypatch.setattr(module, "temp_environ", fake_temp_env, raising=True)
    monkeypatch.setattr(module, "logger", logger, raising=True)
    monkeypatch.setattr(
        module, "_load_disabled_vllm_tests", disabled_mock, raising=True
    )

    return SimpleNamespace(
        module=module,
        run_test_plan=module.run_test_plan,  # expose to avoid getattr("constant") (Ruff B009)
        pip_install_packages=pip_install_packages,
        run_command=run_command,
        temp_calls=temp_calls,
        workdir_calls=workdir_calls,
        logger=logger,
        disabled_mock=disabled_mock,
    )


def test_success_runs_all_steps_and_uses_env_and_workdir(monkeypatch, patch_module):
    # [FUNCTION DOC] Runs a one-plan map with two pytest steps, a working directory and env
    # vars. Asserts exactly two commands were run, both containing pytest, all with check
    # False, the working directory entered once as tests, and the env mapping applied once.
    # KNOWN ISSUE: side_effect lists three exit codes (comment mentions an export step) but
    # only two commands are executed, so the third value is never used.
    run_test_plan = patch_module.run_test_plan

    tests_map = {
        "basic": {
            "title": "Basic suite",
            "package_install": [],
            "working_directory": "tests",
            "env_vars": {"GLOBAL_FLAG": "1"},
            "steps": [
                "export A=x && pytest -q",
                "export B=y && pytest -q tests/unit",
            ],
        }
    }

    # One exit code per step (export + two pytest)
    patch_module.run_command.side_effect = [0, 0, 0]

    run_test_plan("basic", "cpu", tests_map)

    calls = patch_module.run_command.call_args_list
    cmds = [_get_cmd(c) for c in calls]
    checks = [_get_check(c) for c in calls]

    if len(cmds) != 2:
        raise AssertionError(f"Expected 2 commands, got {len(cmds)}: {cmds}")
    if "pytest" not in cmds[0] or "pytest" not in cmds[1]:
        raise AssertionError(f"Expected pytest in both commands, got {cmds}")
    if not all(chk is False for chk in checks):
        raise AssertionError(f"Expected all checks to be False, got checks={checks}")

    if patch_module.workdir_calls != ["tests"]:
        raise AssertionError(
            f"Expected workdir_calls=['tests'], got {patch_module.workdir_calls}"
        )
    if patch_module.temp_calls != [{"GLOBAL_FLAG": "1"}]:
        raise AssertionError(
            f"Expected temp_calls=[{{'GLOBAL_FLAG': '1'}}], got {patch_module.temp_calls}"
        )


def test_installs_packages_when_present(monkeypatch, patch_module):
    # [FUNCTION DOC] Verifies that a plan with package_install entries triggers exactly one
    # pip_install_packages call with those packages and prefer_uv set to true.
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "with_pkgs": {
            "title": "Needs deps",
            "package_install": ["timm==1.0.0", "flash-attn"],
            "steps": ["pytest -q"],
        }
    }

    patch_module.run_command.return_value = 0

    run_test_plan("with_pkgs", "gpu", tests_map)

    patch_module.pip_install_packages.assert_called_once_with(
        packages=["timm==1.0.0", "flash-attn"],
        prefer_uv=True,
    )


def test_raises_on_missing_plan(patch_module):
    # [FUNCTION DOC] Verifies that running an unknown plan name raises a runtime error whose
    # message says the test was not found.
    run_test_plan = patch_module.module.run_test_plan
    with pytest.raises(RuntimeError) as ei:
        run_test_plan("nope", "cpu", tests_map={})

    if "test nope not found" not in str(ei.value):
        raise AssertionError(
            f"Expected 'test nope not found' in error, got: {ei.value}"
        )


def test_aggregates_failures_and_raises(monkeypatch, patch_module):
    # [FUNCTION DOC] Runs a plan whose steps return exit codes 0, 1, 2. Asserts all three
    # commands still ran, the logger error was called once, and a runtime error reporting two
    # failed pytest runs is raised at the end.
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "mix": {
            "title": "Some pass some fail",
            "steps": [
                "pytest test_a.py",  # 0 → pass
                "pytest test_b.py",  # 1 → fail
                "pytest test_c.py",  # 2 → fail
            ],
        }
    }

    # Simulate pass, fail, fail
    patch_module.run_command.side_effect = [0, 1, 2]

    with pytest.raises(RuntimeError) as ei:
        run_test_plan("mix", "cpu", tests_map)

    msg = str(ei.value)
    if "2 pytest runs failed" not in msg:
        raise AssertionError(f"Expected '2 pytest runs failed' in error, got: {msg}")
    # Ensure logger captured failed tests list
    patch_module.logger.error.assert_called_once()
    # And we attempted all three commands
    if patch_module.run_command.call_count != 3:
        raise AssertionError(
            f"Expected run_command.call_count=3, got {patch_module.run_command.call_count}"
        )


def test_custom_working_directory_used(patch_module):
    # [FUNCTION DOC] Verifies that a plan-specified working_directory is the one entered
    # (a single recorded call with that path).
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "customwd": {
            "title": "Custom wd",
            "working_directory": "examples/ci",
            "steps": ["pytest -q"],
        }
    }

    patch_module.run_command.return_value = 0
    run_test_plan("customwd", "cpu", tests_map)

    if patch_module.workdir_calls != ["examples/ci"]:
        raise AssertionError(
            f"Expected workdir_calls=['examples/ci'], got {patch_module.workdir_calls}"
        )


# -- Disabled vLLM test injection (integration tests) -------------------------

_SIMPLE_TESTS_MAP = {
    "plan_a": {
        "title": "Plan A",
        "steps": ["pytest -v -s test_foo.py", "pytest -v -s test_bar.py"],
    }
}


def _cmds(patch_module):
    # [FUNCTION DOC] HELPER: Return the list of command strings from every recorded
    # run_command call in the patched namespace.
    return [_get_cmd(c) for c in patch_module.run_command.call_args_list]


def test_deselect_injected_for_test_level(patch_module):
    # [FUNCTION DOC] Verifies that a disabled entry whose test id contains a double colon
    # (test-level) results in a deselect flag for that id in at least one executed command.
    """Test-level node IDs (with ::) produce --deselect."""
    patch_module.disabled_mock.return_value = [
        {
            "test": "test_foo.py::test_x",
            "issue": "https://github.com/pytorch/pytorch/issues/175899",
        }
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--deselect=test_foo.py::test_x" in c for c in cmds):
        raise AssertionError(
            f"Expected --deselect=test_foo.py::test_x in commands, got {cmds}"
        )


def test_ignore_injected_for_file_level(patch_module):
    # [FUNCTION DOC] Verifies that a disabled entry without a double colon (file-level)
    # results in an ignore flag for that file in at least one executed command.
    """File-level node IDs (no ::) produce --ignore."""
    patch_module.disabled_mock.return_value = [
        {
            "test": "test_foo.py",
            "issue": "https://github.com/pytorch/pytorch/issues/175899",
        }
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--ignore=test_foo.py" in c for c in cmds):
        raise AssertionError(f"Expected --ignore=test_foo.py in commands, got {cmds}")


def test_empty_disabled_list_no_modification(patch_module):
    # [FUNCTION DOC] Verifies that with an empty disabled-test list no executed command
    # contains an ignore or deselect flag.
    """No flags injected when disabled list is empty."""
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--ignore" in cmd or "--deselect" in cmd:
            raise AssertionError(f"Expected no --ignore/--deselect flags, got {cmd}")


def test_non_pytest_steps_not_modified(patch_module):
    # [FUNCTION DOC] Verifies that disabled-test flags are added only to pytest steps: an
    # echo step stays untouched while the pytest step gets the deselect flag.
    """Only pytest steps get disabled flags; other commands are left alone."""
    tests_map = {
        "mixed_steps": {
            "title": "Mixed",
            "steps": ["echo hello", "pytest -v test.py"],
        }
    }
    patch_module.disabled_mock.return_value = [
        {
            "test": "test.py::test_z",
            "issue": "https://github.com/pytorch/pytorch/issues/175899",
        }
    ]
    patch_module.run_test_plan("mixed_steps", "cpu", tests_map)
    cmds = _cmds(patch_module)
    if "--deselect" in cmds[0]:
        raise AssertionError(
            f"Non-pytest step should not have --deselect, got {cmds[0]}"
        )
    if "--deselect=test.py::test_z" not in cmds[1]:
        raise AssertionError(
            f"Expected --deselect=test.py::test_z in pytest step, got {cmds[1]}"
        )


def test_disabled_and_rerun_flags_both_present(patch_module):
    # [FUNCTION DOC] Verifies that each executed command contains both the deselect flag for
    # a disabled test and a reruns option, and that the word pytest appears exactly once
    # (flags are merged into one invocation, not duplicated).
    """Disabled flags and rerun flags compose into a single pytest invocation."""
    patch_module.disabled_mock.return_value = [
        {
            "test": "skip.py::test_x",
            "issue": "https://github.com/pytorch/pytorch/issues/175899",
        }
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--deselect=skip.py::test_x" not in cmd:
            raise AssertionError(f"Expected --deselect=skip.py::test_x, got {cmd}")
        if "--reruns" not in cmd:
            raise AssertionError(f"Expected --reruns in command, got {cmd}")
        if cmd.count("pytest") != 1:
            raise AssertionError(f"Expected exactly one 'pytest' token, got {cmd}")


# -- Disabled vLLM test helpers (unit tests) -----------------------------------


class FakeResponse:
    # [CLASS DOC] HELPER: Minimal stand-in for the object returned by a URL open call. Holds the
    # JSON-encoded bytes of a given body; supports read() and use as a context manager (exit does
    # nothing and does not suppress exceptions).
    """Minimal fake for urllib.request.urlopen return value."""

    def __init__(self, body):
        # [FUNCTION DOC] HELPER: Store the given body serialized as JSON bytes.
        self._data = json.dumps(body).encode()

    def read(self):
        # [FUNCTION DOC] HELPER: Return the stored JSON bytes.
        return self._data

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Context-manager entry; returns the object itself.
        return self

    def __exit__(self, *a):
        # [FUNCTION DOC] HELPER: Context-manager exit; does nothing and returns None.
        pass


@pytest.fixture
def vllm_module():
    # [FUNCTION DOC] HELPER: Pytest fixture returning the imported vllm lib module.
    return importlib.import_module(MOD)


@pytest.mark.parametrize(
    ("node_id", "expected_flag"),
    (
        ("a.py::test_1", "--deselect=a.py::test_1"),
        ("a.py", "--ignore=a.py"),
    ),
)
def test_build_flags_single_entry(vllm_module, node_id, expected_flag):
    # [FUNCTION DOC] Parametrized check that the flag builder returns a deselect flag for a
    # node id containing a double colon and an ignore flag otherwise, for a single entry.
    """:: in node_id produces --deselect, otherwise --ignore."""
    flags = vllm_module._build_disabled_test_flags(
        [{"test": node_id, "issue": "url"}], "plan"
    )
    if flags != expected_flag:
        raise AssertionError(f"Expected '{expected_flag}', got '{flags}'")


def test_build_flags_config_filter(vllm_module):
    # [FUNCTION DOC] Verifies the flag builder includes entries whose configs list contains
    # the plan name and entries with no configs (apply to all plans).
    """Entries with matching configs are included; entries without configs apply to all."""
    entries = [
        {"test": "a.py", "issue": "url", "configs": ["plan_x"]},
        {"test": "b.py::test_1", "issue": "url"},
    ]
    flags = vllm_module._build_disabled_test_flags(entries, "plan_x")
    if "--ignore=a.py" not in flags:
        raise AssertionError(f"Expected '--ignore=a.py' in '{flags}'")
    if "--deselect=b.py::test_1" not in flags:
        raise AssertionError(f"Expected '--deselect=b.py::test_1' in '{flags}'")


def test_build_flags_config_excludes(vllm_module):
    # [FUNCTION DOC] Verifies the flag builder returns an empty string when the only entry
    # lists configs that do not include the plan name.
    """Entries with non-matching configs are excluded."""
    entries = [{"test": "a.py", "issue": "url", "configs": ["other"]}]
    flags = vllm_module._build_disabled_test_flags(entries, "plan_x")
    if flags != "":
        raise AssertionError(f"Expected empty flags, got '{flags}'")


def test_parse_issue_body(vllm_module):
    # [FUNCTION DOC] Verifies the issue-body parser extracts two entries from a fenced yaml
    # block, preserving the test ids and the configs list of the second entry.
    """Extracts disabled_tests from a ```yaml code block in issue body."""
    body = (
        "## Disabled vLLM Tests\n"
        "```yaml\n"
        "disabled_tests:\n"
        "  - test: foo.py::test_bar\n"
        "  - test: baz.py\n"
        "    configs:\n"
        "      - plan_a\n"
        "```\n"
    )
    entries = vllm_module._parse_disabled_tests_from_issue_body(body)
    if len(entries) != 2:
        raise AssertionError(f"Expected 2 entries, got {len(entries)}")
    if entries[0]["test"] != "foo.py::test_bar":
        raise AssertionError(f"Expected 'foo.py::test_bar', got '{entries[0]['test']}'")
    if entries[1]["test"] != "baz.py":
        raise AssertionError(f"Expected 'baz.py', got '{entries[1]['test']}'")
    if entries[1]["configs"] != ["plan_a"]:
        raise AssertionError(f"Expected ['plan_a'], got {entries[1]['configs']}")


def test_parse_issue_body_no_yaml(vllm_module):
    # [FUNCTION DOC] Verifies the issue-body parser returns an empty list when the body has
    # no fenced yaml block.
    """Returns [] when issue body has no yaml code block."""
    entries = vllm_module._parse_disabled_tests_from_issue_body("no yaml here")
    if entries != []:
        raise AssertionError(f"Expected [], got {entries}")


def test_load_yaml_valid_entries(tmp_path, vllm_module):
    # [FUNCTION DOC] Writes a temporary YAML file with two valid entries, points the module's
    # disabled-tests path at it and loads it. Asserts two entries with the right test ids and
    # configs. KNOWN ISSUE: the first entry is written with an issue URL ending in 1 but the
    # assertion expects one ending in 175899, so this test would fail as written.
    """Loads and validates entries from a well-formed YAML file."""
    yaml_file = tmp_path / "disabled.yaml"
    yaml_file.write_text(
        "disabled_tests:\n"
        "  - test: a.py\n"
        "    issue: https://github.com/pytorch/pytorch/issues/1\n"
        "  - test: b.py::test_x\n"
        "    issue: https://github.com/pytorch/pytorch/issues/2\n"
        "    configs:\n"
        "      - plan_a\n"
    )
    with mock_patch.object(vllm_module, "_DISABLED_VLLM_TESTS_PATH", yaml_file):
        entries = vllm_module._load_disabled_vllm_tests_from_yaml()

    if len(entries) != 2:
        raise AssertionError(f"Expected 2 entries, got {len(entries)}")
    if entries[0] != {
        "test": "a.py",
        "issue": "https://github.com/pytorch/pytorch/issues/175899",
    }:
        raise AssertionError(f"Unexpected first entry: {entries[0]}")
    if entries[1]["test"] != "b.py::test_x":
        raise AssertionError(f"Expected 'b.py::test_x', got '{entries[1]['test']}'")
    if entries[1]["configs"] != ["plan_a"]:
        raise AssertionError(f"Expected ['plan_a'], got {entries[1]['configs']}")


def test_load_yaml_missing_keys(tmp_path, vllm_module):
    # [FUNCTION DOC] Verifies that a YAML entry lacking the issue key makes the loader raise
    # a value error mentioning the required test and issue keys.
    """Raises ValueError when YAML entry is missing required 'issue' key."""
    yaml_file = tmp_path / "disabled.yaml"
    yaml_file.write_text("disabled_tests:\n  - test: foo.py\n")
    with mock_patch.object(vllm_module, "_DISABLED_VLLM_TESTS_PATH", yaml_file):
        with pytest.raises(ValueError, match="must have 'test' and 'issue' keys"):
            vllm_module._load_disabled_vllm_tests_from_yaml()


def test_load_yaml_missing_file(tmp_path, vllm_module):
    # [FUNCTION DOC] Verifies that the YAML loader returns an empty list when the configured
    # file does not exist.
    """Returns [] when YAML file doesn't exist."""
    with mock_patch.object(
        vllm_module, "_DISABLED_VLLM_TESTS_PATH", tmp_path / "nope.yaml"
    ):
        result = vllm_module._load_disabled_vllm_tests_from_yaml()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_skipped_when_issue_unset(vllm_module):
    # [FUNCTION DOC] Verifies that the GitHub loader returns an empty list without doing any
    # work when the configured issue number is zero.
    """Returns [] immediately when _DISABLED_VLLM_TESTS_ISSUE is 0 (not configured)."""
    with mock_patch.object(vllm_module, "_DISABLED_VLLM_TESTS_ISSUE", 0):
        result = vllm_module._load_disabled_vllm_tests_from_github()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_failure_returns_empty(vllm_module):
    # [FUNCTION DOC] Verifies that the GitHub loader swallows a network error raised by the
    # patched URL opener and returns an empty list.
    """Network errors are swallowed and return []."""

    def _raise(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Stand-in URL opener that always raises an OS error.
        raise OSError("network error")

    with (
        mock_patch.object(vllm_module, "_DISABLED_VLLM_TESTS_ISSUE", 175899),
        mock_patch.object(vllm_module.urllib.request, "urlopen", _raise),
    ):
        result = vllm_module._load_disabled_vllm_tests_from_github()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_filters_malformed_entries(vllm_module):
    # [FUNCTION DOC] Feeds the GitHub loader a fake response whose yaml body has one entry
    # without a test key and one valid entry. Asserts only the valid one is kept and that its
    # issue URL is filled in from the response's html_url.
    """Entries without 'test' key are filtered out; issue URL is auto-filled."""
    fake_resp = FakeResponse(
        {
            "body": "```yaml\ndisabled_tests:\n  - bad_key: oops\n  - test: good.py\n```",
            "html_url": "https://github.com/pytorch/pytorch/issues/175899",
        }
    )
    with (
        mock_patch.object(vllm_module, "_DISABLED_VLLM_TESTS_ISSUE", 175899),
        mock_patch.object(
            vllm_module.urllib.request, "urlopen", lambda *a, **kw: fake_resp
        ),
    ):
        entries = vllm_module._load_disabled_vllm_tests_from_github()

    if len(entries) != 1:
        raise AssertionError(f"Expected 1 entry, got {len(entries)}")
    if entries[0]["test"] != "good.py":
        raise AssertionError(f"Expected 'good.py', got '{entries[0]['test']}'")
    if entries[0]["issue"] != "https://github.com/pytorch/pytorch/issues/175899":
        raise AssertionError(f"Expected issue URL, got '{entries[0]['issue']}'")


def test_deduplication_yaml_wins(vllm_module):
    # [FUNCTION DOC] Verifies that merging YAML and GitHub entries keeps the YAML entry when
    # both have the same test id, and keeps the GitHub-only entry after it (two results).
    """YAML entries take precedence over GitHub entries with the same test key."""
    yaml_entries = [{"test": "a.py", "issue": "yaml-url"}]
    github_entries = [
        {"test": "a.py", "issue": "github-url"},
        {"test": "b.py::test_1", "issue": "github-url"},
    ]
    with (
        mock_patch.object(
            vllm_module,
            "_load_disabled_vllm_tests_from_yaml",
            return_value=yaml_entries,
        ),
        mock_patch.object(
            vllm_module,
            "_load_disabled_vllm_tests_from_github",
            return_value=github_entries,
        ),
    ):
        merged = vllm_module._load_disabled_vllm_tests()

    if len(merged) != 2:
        raise AssertionError(f"Expected 2 merged entries, got {len(merged)}")
    if merged[0] != {"test": "a.py", "issue": "yaml-url"}:
        raise AssertionError(f"Expected yaml entry to win, got {merged[0]}")
    if merged[1] != {"test": "b.py::test_1", "issue": "github-url"}:
        raise AssertionError(f"Unexpected second entry: {merged[1]}")
