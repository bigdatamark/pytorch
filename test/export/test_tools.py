# Owner(s): ["oncall: export"]
# [MODULE DOC] Tests for the exportability report tool. Import time: registers a testlib custom
# operator (two mutated tensor arguments, pt2 compliant tag) with only a CPU implementation and
# no meta/fake implementation, so exporting it fails. Defines TestExportTools.

import torch
from torch._dynamo.test_case import TestCase
from torch._export.tools import report_exportability
from torch.testing._internal.common_utils import run_tests


torch.library.define(
    "testlib::op_missing_meta",
    "(Tensor(a!) x, Tensor(b!) z) -> Tensor",
    tags=torch.Tag.pt2_compliant_tag,
)


@torch.library.impl("testlib::op_missing_meta", "cpu")
@torch._dynamo.disable
def op_missing_meta(x, z):
    # [FUNCTION DOC] HELPER: CPU implementation of the testlib operator, excluded from dynamo
    # tracing. Adds five to both inputs in place and returns their sum.
    x.add_(5)
    z.add_(5)
    return x + z


class TestExportTools(TestCase):
    # [CLASS DOC] Test case for the report_exportability utility, which maps module names to the
    # export exception (or None when exportable).
    def test_report_exportability_basic(self):
        # [FUNCTION DOC] Reports exportability of a simple module taking a list and a tensor and
        # checks the report has exactly one entry, for the root (empty name), with value None.
        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds the first list element to the
            # second input.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the first element of the first input
                # list and the second input.
                return x[0] + y

        f = Module()
        inp = ([torch.ones(1, 3)], torch.ones(1, 3))

        report = report_exportability(f, inp)
        self.assertTrue(len(report) == 1)
        self.assertTrue(report[""] is None)

    def test_report_exportability_with_issues(self):
        # [FUNCTION DOC] Builds a root module with an unsupported child (calls the operator lacking
        # a fake implementation) and a supported child (sine), reports exportability with
        # non-strict, pre-dispatch export, and checks the root and the unsupported child have
        # exceptions while the supported child has None.
        class Unsupported(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module calling the testlib operator that lacks a meta
            # implementation.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the operator on the input and its cosine; export
                # fails for it.
                return torch.ops.testlib.op_missing_meta(x, x.cos())

        class Supported(torch.nn.Module):
            # [CLASS DOC] HELPER: Child module that applies sine; exportable.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sine of the input.
                return x.sin()

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Root module holding the unsupported and supported children.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two child modules as attributes.
                super().__init__()
                self.unsupported = Unsupported()
                self.supported = Supported()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Takes the indices of nonzero elements of the input and
                # returns the sum of the unsupported and supported children applied to them.
                y = torch.nonzero(x)
                return self.unsupported(y) + self.supported(y)

        f = Module()
        inp = (torch.ones(4, 4),)

        report = report_exportability(f, inp, strict=False, pre_dispatch=True)

        self.assertTrue(report[""] is not None)
        self.assertTrue(report["unsupported"] is not None)
        self.assertTrue(report["supported"] is None)


if __name__ == "__main__":
    run_tests()
