# Owner(s): ["module: ci"]
# [MODULE DOC] Tests for the test-selection ("determination") logic used by CI. Checks that
# run_test.should_run_test and tools.testing.modulefinder_determinator.test_impact_of_file map
# changed file paths to the expected subset of a fixed list of test names. Imports the sibling
# run_test script directly, so it must run from the test directory.

import os

import run_test

from tools.testing.modulefinder_determinator import test_impact_of_file

from torch.testing._internal.common_utils import run_tests, TestCase


class DummyOptions:
    # [CLASS DOC] HELPER: Minimal stand-in for the parsed command-line options object passed
    # to should_run_test. Only carries a class attribute verbose set to False.
    verbose = False


class DeterminationTest(TestCase):
    # [CLASS DOC] Test case verifying which tests are selected for various changed-file lists.
    # TESTS is the fixed, ordered subset of test names the selection is evaluated against.
    # Test determination on a subset of tests
    TESTS = [
        "test_nn",
        "test_jit_profiling",
        "test_jit",
        "test_torch",
        "test_cpp_extensions_aot_ninja",
        "test_cpp_extensions_aot_no_ninja",
        "test_utils",
        "test_determination",
        "test_quantization",
    ]

    @classmethod
    def determined_tests(cls, changed_files):
        # [FUNCTION DOC] HELPER: Returns the entries of TESTS (in TESTS order) that
        # should_run_test selects for the given changed files. Each path is first normalized
        # with os.path.normpath, and the check uses TARGET_DET_LIST from run_test with a
        # DummyOptions instance.
        changed_files = [os.path.normpath(path) for path in changed_files]
        return [
            test
            for test in cls.TESTS
            if run_test.should_run_test(
                run_test.TARGET_DET_LIST, test, changed_files, DummyOptions()
            )
        ]

    def test_target_det_list_is_sorted(self):
        # [FUNCTION DOC] Asserts that TARGET_DET_LIST in run_test equals its sorted copy, so
        # entries stay alphabetically ordered (keeps merge conflicts low and lets absences be
        # commented).
        # We keep TARGET_DET_LIST sorted to minimize merge conflicts
        # but most importantly to allow us to comment on the absence
        # of a test. It would be very difficult to add a file right
        # next to a comment that says to keep it out of the list.
        self.assertListEqual(run_test.TARGET_DET_LIST, sorted(run_test.TARGET_DET_LIST))

    def test_config_change_only(self):
        # [FUNCTION DOC] A change to a CI config file selects every test in TESTS.
        """CI configs trigger all tests"""
        self.assertEqual(self.determined_tests([".ci/pytorch/test.sh"]), self.TESTS)

    def test_classifies_git_style_paths(self):
        # [FUNCTION DOC] Checks test_impact_of_file classifies forward-slash paths: torch
        # python module as TORCH, test script as TEST, caffe2 python file as CAFFE2, .ci file
        # as CI.
        """Git-style paths use / even on Windows"""
        self.assertEqual(test_impact_of_file("torch/nn/modules/linear.py"), "TORCH")
        self.assertEqual(test_impact_of_file("test/test_jit.py"), "TEST")
        self.assertEqual(test_impact_of_file("caffe2/python/core.py"), "CAFFE2")
        self.assertEqual(test_impact_of_file(".ci/pytorch/test.sh"), "CI")

    def test_classifies_windows_style_paths(self):
        # [FUNCTION DOC] Same classification as the git-style test but with backslash-
        # separated (Windows) paths, expecting identical TORCH, TEST, CAFFE2 and CI results.
        """Windows-style paths are still accepted"""
        self.assertEqual(test_impact_of_file(r"torch\nn\modules\linear.py"), "TORCH")
        self.assertEqual(test_impact_of_file(r"test\test_jit.py"), "TEST")
        self.assertEqual(test_impact_of_file(r"caffe2\python\core.py"), "CAFFE2")
        self.assertEqual(test_impact_of_file(r".ci\pytorch\test.sh"), "CI")

    def test_should_run_test_handles_git_style_paths(self):
        # [FUNCTION DOC] Asserts should_run_test returns true for test_determination when the
        # only changed file is a forward-slash path to torch/utils/cpp_extension.py.
        """should_run_test handles raw Git-style changed-file paths"""
        self.assertTrue(
            run_test.should_run_test(
                run_test.TARGET_DET_LIST,
                "test_determination",
                ["torch/utils/cpp_extension.py"],
                DummyOptions(),
            )
        )

    def test_should_run_test_handles_windows_style_paths(self):
        # [FUNCTION DOC] Same as the git-style variant but the changed file uses backslash
        # separators; should_run_test must still select test_determination.
        """should_run_test still handles Windows-style changed-file paths"""
        self.assertTrue(
            run_test.should_run_test(
                run_test.TARGET_DET_LIST,
                "test_determination",
                [r"torch\utils\cpp_extension.py"],
                DummyOptions(),
            )
        )

    def test_run_test(self):
        # [FUNCTION DOC] Changing test/run_test.py selects only test_determination, since that
        # test imports run_test.
        """run_test.py is imported by determination tests"""
        self.assertEqual(
            self.determined_tests(["test/run_test.py"]), ["test_determination"]
        )

    def test_non_code_change(self):
        # [FUNCTION DOC] Changing only CODEOWNERS, README.md and a docs file selects no tests.
        """Non-code changes don't trigger any tests"""
        self.assertEqual(
            self.determined_tests(["CODEOWNERS", "README.md", "docs/doc.md"]), []
        )

    def test_cpp_file(self):
        # [FUNCTION DOC] Changing a C++ source file under aten selects every test in TESTS.
        """CPP files trigger all tests"""
        self.assertEqual(
            self.determined_tests(["aten/src/ATen/native/cpu/Activation.cpp"]),
            self.TESTS,
        )

    def test_test_file(self):
        # [FUNCTION DOC] Checks changed test files: test_jit.py and a test/jit file select the
        # two jit tests; a quantization eager test file selects only test_quantization.
        """Test files trigger themselves and dependent tests"""
        self.assertEqual(
            self.determined_tests(["test/test_jit.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(["test/jit/test_custom_operators.py"]),
            ["test_jit_profiling", "test_jit"],
        )
        self.assertEqual(
            self.determined_tests(
                ["test/quantization/eager/test_quantize_eager_ptq.py"]
            ),
            ["test_quantization"],
        )

    def test_test_internal_file(self):
        # [FUNCTION DOC] Changing the testing/_internal common_quantization module selects the
        # two jit tests and test_quantization (the tests depending on that module).
        """testing/_internal files trigger dependent tests"""
        self.assertEqual(
            self.determined_tests(["torch/testing/_internal/common_quantization.py"]),
            [
                "test_jit_profiling",
                "test_jit",
                "test_quantization",
            ],
        )

    def test_torch_file(self):
        # [FUNCTION DOC] Checks torch file impact in three cases: torch/onnx/utils.py selects
        # all tests (force-imported); autograd _functions utils .py and .pyi select only
        # test_utils; torch/utils/cpp_extension.py selects the two cpp extension tests,
        # test_utils and test_determination.
        """Torch files trigger dependent tests"""
        self.assertEqual(
            # Many files are force-imported to all tests,
            # due to the layout of the project.
            self.determined_tests(["torch/onnx/utils.py"]),
            self.TESTS,
        )
        self.assertEqual(
            self.determined_tests(
                [
                    "torch/autograd/_functions/utils.py",
                    "torch/autograd/_functions/utils.pyi",
                ]
            ),
            ["test_utils"],
        )
        self.assertEqual(
            self.determined_tests(["torch/utils/cpp_extension.py"]),
            [
                "test_cpp_extensions_aot_ninja",
                "test_cpp_extensions_aot_no_ninja",
                "test_utils",
                "test_determination",
            ],
        )

    def test_new_folder(self):
        # [FUNCTION DOC] A Python file in a new unknown top-level folder selects every test.
        """New top-level Python folder triggers all tests"""
        self.assertEqual(self.determined_tests(["new_module/file.py"]), self.TESTS)

    def test_new_test_script(self):
        # [FUNCTION DOC] A new test script not listed in run_test selects no tests.
        """New test script triggers nothing (since it's not in run_tests.py)"""
        self.assertEqual(self.determined_tests(["test/test_new_test_script.py"]), [])


if __name__ == "__main__":
    run_tests()
