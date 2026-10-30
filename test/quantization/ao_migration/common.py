# [MODULE DOC] Shared base test case for the quantization ao-migration tests. Verifies that
# names importable from the legacy torch.quantization (or other legacy base) location are the
# same objects as in the new torch.ao location. No import-time side effects beyond imports.
import importlib

from torch.testing._internal.common_utils import TestCase


class AOMigrationTestCase(TestCase):
    # [CLASS DOC] Base test case offering two comparison helpers used by the migration test
    # files: one for lists of function or class names and one for lists of dictionary names,
    # each comparing the legacy module against its torch.ao counterpart.
    def _test_function_import(
        self,
        package_name: str,
        function_list: list[str],
        base: str | None = None,
        new_package_name: str | None = None,
    ):
        # [FUNCTION DOC] HELPER: Imports the legacy module (torch plus base plus package name;
        # base defaults to quantization) and the new torch.ao module (the new package name
        # defaults to the same name), then for each listed name fetches the attribute from both.
        # Raises AssertionError if the two objects differ or their hashes differ. An import
        # error or missing attribute propagates.
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base
        new_base = "torch.ao." + base
        if new_package_name is None:
            new_package_name = package_name
        old_location = importlib.import_module(f"{old_base}.{package_name}")
        new_location = importlib.import_module(f"{new_base}.{new_package_name}")
        for fn_name in function_list:
            old_function = getattr(old_location, fn_name)
            new_function = getattr(new_location, fn_name)
            if old_function != new_function:
                raise AssertionError(f"Functions don't match: {fn_name}")
            if hash(old_function) != hash(new_function):
                raise AssertionError(
                    f"Hashes don't match: {old_function}({hash(old_function)}) vs. "
                    f"{new_function}({hash(new_function)})"
                )

    def _test_dict_import(
        self, package_name: str, dict_list: list[str], base: str | None = None
    ):
        # [FUNCTION DOC] HELPER: Same module resolution as the function helper (base defaults to
        # quantization; the same package name is used in both locations). For each listed
        # dictionary name, raises AssertionError if the old and new dictionaries are unequal, or
        # if any key of the new dictionary maps to a different value in the old one. Missing
        # attribute or key errors propagate.
        r"""Tests individual function list import by comparing the functions
        and their hashes."""
        if base is None:
            base = "quantization"
        old_base = "torch." + base
        new_base = "torch.ao." + base
        old_location = importlib.import_module(f"{old_base}.{package_name}")
        new_location = importlib.import_module(f"{new_base}.{package_name}")
        for dict_name in dict_list:
            old_dict = getattr(old_location, dict_name)
            new_dict = getattr(new_location, dict_name)
            if old_dict != new_dict:
                raise AssertionError(f"Dicts don't match: {dict_name}")
            for key in new_dict:
                if old_dict[key] != new_dict[key]:
                    raise AssertionError(
                        f"Dicts don't match: {dict_name} for key {key}"
                    )
