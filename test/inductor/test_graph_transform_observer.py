# Owner(s): ["module: inductor"]
# [MODULE DOC] Inductor test verifying that the graph transform observer writes dot graph files
# when the SDPA pattern rewriter transforms a compiled graph. Defines optional-dependency flags
# (whether the pydot package and the dot executable are available) at import time. Run as a
# script, tests are only executed on Linux.
import glob
import math
import os
import shutil
import tempfile

import torch
import torch._dynamo
import torch._inductor.config as inductor_config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FUSED_ATTENTION
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


try:
    import pydot  # noqa: F401

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False


HAS_DOT = shutil.which("dot") is not None


class TestGraphTransformObserver(TestCase):
    # [CLASS DOC] Test case for the graph transform observer logging of inductor graph passes.
    def test_sdpa_rewriter(self):
        # [FUNCTION DOC] Returns silently (passing) unless CUDA with Triton, fused attention
        # support, pydot and the dot executable are all available. Otherwise sets the graph-
        # transform log directory config to a fresh temporary directory, disables inductor caches
        # (both mutated without restoration), compiles manual dot-product attention with fullgraph
        # and runs it once on random CUDA inputs, then scans the directory for files ending in
        # input_graph.dot and output_graph.dot and asserts both kinds were produced.
        if not (
            HAS_CUDA_AND_TRITON
            and PLATFORM_SUPPORTS_FUSED_ATTENTION
            and HAS_PYDOT
            and HAS_DOT
        ):
            return

        def dot_prod_attention(
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Reference attention: matmul of query with the transposed key,
            # scaled by the inverse square root of the embedding size, softmax over the last
            # dimension, then matmul with value. Returns the attended tensor.
            """Input tensors assumed to have shape (batch_size, n_head, seq_len, embed_dim)"""
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .div(math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        log_url = tempfile.mkdtemp()
        inductor_config.trace.log_url_for_graph_xform = log_url
        inductor_config.force_disable_caches = True
        compiled_fn = torch.compile(dot_prod_attention, fullgraph=True)

        tensor_shape = (4, 2, 16, 32)
        q = torch.randn(tensor_shape, device="cuda")
        k = torch.randn(tensor_shape, device="cuda")
        v = torch.randn(tensor_shape, device="cuda")
        compiled_fn(q, k, v)

        found_input_svg = False
        found_output_svg = False
        for filepath_object in glob.glob(log_url + "/*"):
            if os.path.isfile(filepath_object):
                if filepath_object.endswith("input_graph.dot"):
                    found_input_svg = True
                elif filepath_object.endswith("output_graph.dot"):
                    found_output_svg = True

        self.assertTrue(found_input_svg)
        self.assertTrue(found_output_svg)


if __name__ == "__main__":
    if IS_LINUX:
        run_tests()
