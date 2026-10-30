# Owner(s): ["oncall: jit"]
# [MODULE DOC] TorchScript tests for sparse tensors (COO and CSR layouts) held as module
# attributes. Each test scripts a small module, runs it, and checks results survive freezing
# and/or saving to an in-memory buffer and loading back. CSR tests are skipped on Windows or
# without MKL.

import io
import unittest

import torch
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    TEST_MKL,
)
from torch.testing._internal.jit_utils import JitTestCase


class TestSparse(JitTestCase):
    # [CLASS DOC] JitTestCase suite for sparse tensor attributes in scripted modules: freezing
    # and serialization round trips for the COO and CSR layouts.
    def test_freeze_sparse_coo(self):
        # [FUNCTION DOC] Builds a module that adds two sparse COO attributes to a sparse COO
        # input, records the eager result, then scripts and freezes the module in eval mode and
        # checks the frozen result equals the eager one. Also saves the frozen module to a
        # buffer, loads it back and checks the loaded result matches too.
        class SparseTensorModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two random 3 by 4 sparse COO tensor attributes a
            # and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two random 3 by 4 tensors and stores them as
                # sparse COO attributes.
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus both sparse attributes.
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result, frozen_result)

        buffer = io.BytesIO()
        torch.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result, loaded_result)

    def test_serialize_sparse_coo(self):
        # [FUNCTION DOC] Builds a module adding two sparse COO attributes to a sparse COO input,
        # saves the scripted module (without freezing) to a buffer, loads it, and checks the
        # loaded result equals the eager result.
        class SparseTensorModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two random 3 by 4 sparse COO tensor attributes a
            # and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two random 3 by 4 tensors and stores them as
                # sparse COO attributes.
                super().__init__()
                self.a = torch.rand(3, 4).to_sparse()
                self.b = torch.rand(3, 4).to_sparse()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus both sparse attributes.
                return x + self.a + self.b

        x = torch.rand(3, 4).to_sparse()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result, loaded_result)

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_freeze_sparse_csr(self):
        # [FUNCTION DOC] Skipped on Windows or without MKL. Builds a module that multiplies a
        # sparse CSR input by two sparse CSR attributes, then scripts and freezes it in eval
        # mode and compares the dense form of the frozen result with the eager one. Also saves
        # and reloads the frozen module and compares the loaded result, again in dense form.
        class SparseTensorModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two random 4 by 4 sparse CSR tensor attributes a
            # and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two random 4 by 4 tensors and stores them as
                # sparse CSR attributes.
                super().__init__()
                self.a = torch.rand(4, 4).to_sparse_csr()
                self.b = torch.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the input with attribute
                # a and then b.
                return x.matmul(self.a).matmul(self.b)

        x = torch.rand(4, 4).to_sparse_csr()

        m = SparseTensorModule()
        unfrozen_result = m.forward(x)

        m.eval()
        frozen = torch.jit.freeze(torch.jit.script(m))

        frozen_result = frozen.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), frozen_result.to_dense())

        buffer = io.BytesIO()
        torch.jit.save(frozen, buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(unfrozen_result.to_dense(), loaded_result.to_dense())

    @unittest.skipIf(IS_WINDOWS or not TEST_MKL, "Need MKL to run CSR matmul")
    def test_serialize_sparse_csr(self):
        # [FUNCTION DOC] Skipped on Windows or without MKL. Saves a scripted (not frozen) module
        # that multiplies a sparse CSR input by two sparse CSR attributes, loads it back, and
        # compares the dense forms of the loaded and eager results.
        class SparseTensorModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two random 4 by 4 sparse CSR tensor attributes a
            # and b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two random 4 by 4 tensors and stores them as
                # sparse CSR attributes.
                super().__init__()
                self.a = torch.rand(4, 4).to_sparse_csr()
                self.b = torch.rand(4, 4).to_sparse_csr()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of the input with attribute
                # a and then b.
                return x.matmul(self.a).matmul(self.b)

        x = torch.rand(4, 4).to_sparse_csr()
        m = SparseTensorModule()
        expected_result = m.forward(x)

        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(m), buffer)
        buffer.seek(0)
        loaded_model = torch.jit.load(buffer)

        loaded_result = loaded_model.forward(x)

        self.assertEqual(expected_result.to_dense(), loaded_result.to_dense())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
