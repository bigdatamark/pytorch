# Owner(s): ["module: ci"]
# [MODULE DOC] Tests that only track the wall-clock cost of importing torch and of querying the
# CUDA device count. Each test runs a fresh subprocess through the shared TestCase helper; nothing
# asserts on a time threshold. Runs through run_tests when executed as a script.

from torch.testing._internal.common_utils import TestCase, run_tests


# these tests could eventually be changed to fail if the import/init
# time is greater than a certain threshold, but for now we just use them
# as a way to track the duration of `import torch`.
class TestImportTime(TestCase):
    # [CLASS DOC] Test case holding two import-time tracking tests. It has no state; each test only
    # launches a child interpreter for a short statement and relies on the helper to run it.
    def test_time_import_torch(self):
        # [FUNCTION DOC] Runs a child interpreter that only imports torch, via the TestCase helper
        # that runs a statement with the PyTorch API usage stderr logging enabled. No assertion is
        # made here beyond the helper succeeding.
        TestCase.runWithPytorchAPIUsageStderr("import torch")

    def test_time_cuda_device_count(self):
        # [FUNCTION DOC] Runs a child interpreter that imports torch and asks the CUDA module for
        # its device count, through the same helper as the import test. Works without a GPU (the
        # count is simply not inspected); only the helper's success matters.
        TestCase.runWithPytorchAPIUsageStderr(
            "import torch; torch.cuda.device_count()",
        )


if __name__ == "__main__":
    run_tests()
