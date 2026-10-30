# Owner(s): ["oncall: mobile"]
# [MODULE DOC] Tests that torch.utils.show_pickle can print the pickled contents of a saved
# TorchScript model archive.

import io
import tempfile
import unittest

import torch
import torch.utils.show_pickle
from torch.testing._internal.common_utils import IS_WINDOWS, run_tests, TestCase


class TestShowPickle(TestCase):
    # [CLASS DOC] Tests torch.utils.show_pickle on pickle data inside a saved TorchScript archive.
    @unittest.skipIf(IS_WINDOWS, "Can't re-open temp file on Windows")
    def test_scripted_model(self):
        # [FUNCTION DOC] Scripts a small module holding a tensor weight, saves it to a temporary
        # file, runs the show_pickle main entry on the data pickle member of that archive, and
        # checks the captured text mentions the module class name and the weight attribute.
        # Skipped on Windows because the temporary file cannot be reopened there.
        class MyCoolModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module storing a given weight tensor; forward multiplies input by it.
            def __init__(self, weight):
                # [FUNCTION DOC] HELPER: Calls the base constructor and stores the weight as an attribute.
                super().__init__()
                self.weight = weight

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied elementwise by the stored weight.
                return x * self.weight

        m = torch.jit.script(MyCoolModule(torch.tensor([2.0])))

        with tempfile.NamedTemporaryFile() as tmp:
            torch.jit.save(m, tmp)
            tmp.flush()
            buf = io.StringIO()
            torch.utils.show_pickle.main(
                ["", tmp.name + "@*/data.pkl"], output_stream=buf
            )
            output = buf.getvalue()
            self.assertRegex(output, "MyCoolModule")
            self.assertRegex(output, "weight")


if __name__ == "__main__":
    run_tests()
