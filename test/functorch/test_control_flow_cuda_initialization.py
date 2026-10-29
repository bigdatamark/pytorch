# Owner(s): ["module: functorch"]
# [MODULE DOC] Tests that CUDA graph conditional-node control flow (torch.cond compiled with the
# cudagraphs backend) works when no CUDA primary context exists beforehand. Defines one test
# class, skipped unless conditional nodes are supported. Must run in a process where CUDA contexts
# have never been created. No import-time side effects besides imports.
import unittest

import torch
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    TestCase,
)


@unittest.skipIf(
    not TEST_CUDA_GRAPH_CONDITIONAL_NODES,
    "CUDA 12.4 or greater is required for CUDA Graphs with conditional nodes",
)
class TestControlFlowInCUDAGraphInitialization(TestCase):
    # [CLASS DOC] Test case verifying that library initialization (cudnn, cufft) triggered inside
    # a cond branch does not break CUDA graph stream capture. setUp asserts that device 0 has no
    # primary context yet, so each test must run in a fresh process. Class attribute
    # CTX_ALREADY_CREATED_ERR_MSG holds the failure message for that assertion.
    # Duplicated from test_cuda_primary_ctx.py
    CTX_ALREADY_CREATED_ERR_MSG = (
        "Tests defined in TestControlFlowInCUDAGraphInitialization must be run in a process "
        "where CUDA contexts are never created. Use either run_test.py or add "
        "--subprocess to run each test in a different subprocess."
    )

    def setUp(self):
        # [FUNCTION DOC] Runs base setUp, then asserts the CUDA primary context of device 0 has
        # not been created, failing with CTX_ALREADY_CREATED_ERR_MSG otherwise.
        super().setUp()
        # Ensure context has not been created beforehand
        self.assertFalse(
            torch._C._cuda_hasPrimaryContext(0),
            TestControlFlowInCUDAGraphInitialization.CTX_ALREADY_CREATED_ERR_MSG,
        )

    def _check_compile_cudagraphs(self, f, pred, *other_args):
        # [FUNCTION DOC] HELPER: Compiles f with the cudagraphs backend and compares against
        # eager. For the predicate and then its logical negation, runs the compiled function
        # three times (marking a cudagraph step begin before each call) and stores cloned
        # outputs. After each group it computes the eager result and asserts every output
        # collected so far equals it. Note the compiled calls always receive the original pred,
        # not the negated loop value.
        f_compiled = torch.compile(f, backend="cudagraphs")

        outputs = []

        for p in [pred, torch.logical_not(pred)]:
            for i in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                outputs.append(f_compiled(pred, *other_args).clone())

            # We compute the eager output only after running cudagraphs
            # backend compiled function, in order to make sure that
            # cudagraph trees warms up the conditional part of the code
            # properly.
            eager_output = f(pred, *other_args)

            for output in outputs:
                self.assertEqual(output, eager_output)

    def test_cond_cudnn(self):
        # [FUNCTION DOC] Checks that a cond whose false branch runs a 1D convolution (needing
        # cuBLAS/cudnn handle creation) can be captured by cudagraphs without a prior context.
        # Asserts no primary context before, runs the compile-vs-eager comparison, then asserts a
        # context now exists.
        # Tests that cublasCreate() does not break stream capture
        def f(pred, x, filters):
            # [FUNCTION DOC] HELPER: Wraps cond on pred: the true branch sums the input, the false
            # branch sums the 1D convolution of the input with the filters.
            return torch.cond(
                pred,
                lambda y: torch.sum(y),
                lambda y: torch.sum(torch.nn.functional.conv1d(y, filters)),
                [x],
            )

        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))

        pred = torch.tensor(True, device="cuda")
        x = torch.randn(33, 16, 30, device="cuda")
        filters = torch.randn(20, 16, 5, device="cuda")

        self._check_compile_cudagraphs(f, pred, x, filters)

        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))

    def test_cond_stft(self):
        # [FUNCTION DOC] Checks that a cond whose false branch computes a short-time Fourier
        # transform (needing cufft plan creation) can be captured by cudagraphs without a prior
        # context. Asserts no primary context before, runs the compile-vs-eager comparison, then
        # asserts one exists.
        # Tests that cufft plan creation does not break stream capture
        def f(pred, x):
            # [FUNCTION DOC] HELPER: Wraps cond on pred: the true branch sums the input, the false
            # branch sums the real-valued STFT (n_fft 512) of the input.
            return torch.cond(
                pred,
                lambda y: torch.sum(y),
                lambda y: torch.sum(torch.stft(y, 512, return_complex=False)),
                [x],
            )

        self.assertFalse(torch._C._cuda_hasPrimaryContext(0))

        pred = torch.tensor(True, device="cuda")
        x = torch.ones(1024 * 1024, device="cuda")

        self._check_compile_cudagraphs(f, pred, x)

        self.assertTrue(torch._C._cuda_hasPrimaryContext(0))


if __name__ == "__main__":
    run_tests()
