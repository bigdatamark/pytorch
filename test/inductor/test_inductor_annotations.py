# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that Inductor-generated code contains (or omits) CUDA nvtx training
# annotations depending on the annotate_training config flag. Defines one TestCase; the tests
# need CUDA and Triton. Running the file as a script invokes the Inductor test runner.
import torch
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch.testing._internal.triton_utils import requires_cuda_and_triton


class InductorAnnotationTestCase(TestCase):
    # [CLASS DOC] Test case checking the generated wrapper code for nvtx annotation markers.
    # It has one helper that compiles a tiny function and two tests that inspect the source text.
    def get_code(self):
        # [FUNCTION DOC] HELPER: Compile a small two-output function (sum and product of two random
        # 5-element CUDA tensors) with torch.compile and return the first generated source string
        # captured by run_and_get_code. Requires a CUDA device; no arguments besides self.
        def f(a, b):
            # [FUNCTION DOC] HELPER: Nested function to compile; returns a tuple of the
            # elementwise sum and the elementwise product of its two inputs.
            return a + b, a * b

        a = torch.randn(5, device="cuda")
        b = torch.randn(5, device="cuda")
        f_comp = torch.compile(f)

        _, code = run_and_get_code(f_comp, a, b)
        return code[0]

    @requires_cuda_and_triton
    def test_no_annotations(self):
        # [FUNCTION DOC] Verifies that with the default config the generated code does not import
        # the nvtx module from torch.cuda and does not mention the training_annotation variable.
        code = self.get_code()

        self.assertTrue("from torch.cuda import nvtx" not in code)
        self.assertTrue("training_annotation" not in code)

    @inductor_config.patch(annotate_training=True)
    @requires_cuda_and_triton
    def test_training_annotation(self):
        # [FUNCTION DOC] With annotate_training enabled, verifies the generated code imports nvtx
        # from torch.cuda, starts a device range labeled inference stored in training_annotation at
        # least once, and ends that range at least once.
        code = self.get_code()

        self.assertTrue("from torch.cuda import nvtx" in code)
        self.assertTrue(
            code.count("training_annotation = nvtx._device_range_start('inference')")
            >= 1
        )
        self.assertTrue(code.count("nvtx._device_range_end(training_annotation)") >= 1)


if __name__ == "__main__":
    run_tests()
