# [MODULE DOC]
# Unit tests for the argument handling of tools.jit.gen_unboxing.main. The file manager,
# yaml parsing and generation steps are mocked out; the tests check which operator
# allowlist is passed to the custom build selector. No import-time side effects.
import tempfile
import unittest
from unittest.mock import NonCallableMock, patch

import tools.jit.gen_unboxing as gen_unboxing


@patch("tools.jit.gen_unboxing.get_custom_build_selector")
@patch("tools.jit.gen_unboxing.parse_native_yaml")
@patch("tools.jit.gen_unboxing.make_file_manager")
@patch("tools.jit.gen_unboxing.gen_unboxing")
class TestGenUnboxing(unittest.TestCase):
    # [CLASS DOC] Test case with the selector, yaml parser, file manager and generator
    # patched in gen_unboxing, so main can run without touching the filesystem. Each test
    # receives the four mocks (decorators apply bottom to top).
    def test_get_custom_build_selector_with_allowlist(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # [FUNCTION DOC] Runs main with an inline allowlist and asserts the selector is built
        # once with that list and the given selection yaml path.
        args = ["--op-registration-allowlist=op1", "--op-selection-yaml-path=path2"]
        gen_unboxing.main(args)
        mock_get_custom_build_selector.assert_called_once_with(["op1"], "path2")

    def test_get_custom_build_selector_with_allowlist_yaml(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # [FUNCTION DOC] Writes a temporary yaml list holding one operator, runs main with the
        # yaml path option only, and asserts the selector gets the operator list loaded from
        # that file.
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"- aten::add.Tensor")
            temp_file.seek(0)
            args = [
                f"--TEST-ONLY-op-registration-allowlist-yaml-path={temp_file.name}",
                "--op-selection-yaml-path=path2",
            ]
            gen_unboxing.main(args)
            mock_get_custom_build_selector.assert_called_once_with(
                ["aten::add.Tensor"], "path2"
            )

    def test_get_custom_build_selector_with_both_allowlist_and_yaml(
        self,
        mock_gen_unboxing: NonCallableMock,
        mock_make_file_manager: NonCallableMock,
        mock_parse_native_yaml: NonCallableMock,
        mock_get_custom_build_selector: NonCallableMock,
    ) -> None:
        # [FUNCTION DOC] Passes both an inline allowlist and a yaml allowlist file to main and
        # asserts the inline allowlist takes precedence in the selector call.
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file.write(b"- aten::add.Tensor")
            temp_file.seek(0)
            args = [
                "--op-registration-allowlist=op1",
                f"--TEST-ONLY-op-registration-allowlist-yaml-path={temp_file.name}",
                "--op-selection-yaml-path=path2",
            ]
            gen_unboxing.main(args)
            mock_get_custom_build_selector.assert_called_once_with(["op1"], "path2")


if __name__ == "__main__":
    unittest.main()
