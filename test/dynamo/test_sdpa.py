# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for scaled dot product attention support: SDPAParams objects
# (construct, return, pass as input, attribute access, with and without graph breaks), the flash
# attention capability check, the private current-backends query, and the sdpa_kernel context
# manager used as a decorator under full-graph compilation.
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import CompileCounter
from torch.backends.cuda import can_use_flash_attention, SDPAParams
from torch.nn.attention import _cur_sdpa_kernel_backends, sdpa_kernel, SDPBackend


class TestSDPA(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case holding the SDPA related Dynamo tests. Each test compiles a small
    # function with a frame-counting backend and checks both the produced values and how many
    # frames were compiled.
    def assert_ref_equals_params(self, actual, expected):
        # [FUNCTION DOC] HELPER: Asserts that two SDPAParams objects reference the very same
        # query, key, value and attention mask objects (identity comparison, not value
        # equality). Raises an assertion failure otherwise.
        self.assertIs(actual.query, expected.query)
        self.assertIs(actual.key, expected.key)
        self.assertIs(actual.value, expected.value)
        self.assertIs(actual.attn_mask, expected.attn_mask)

    def test_returns_SDPAParams(self):
        # [FUNCTION DOC] Compiles, with fullgraph and a counting backend, a function that builds
        # and returns an SDPAParams from four tensors. Asserts the result is an SDPAParams that
        # holds the same tensor objects as a directly built one, and that exactly one frame was
        # compiled.
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v, m):
            # [FUNCTION DOC] HELPER: Compiled function; constructs and returns an SDPAParams
            # from the query, key, value and mask with dropout 0.1, causal flag true and gqa
            # flag false.
            return SDPAParams(q, k, v, m, 0.1, True, False)

        q = torch.randn(10)
        k = torch.randn(10)
        v = torch.randn(10)
        m = torch.randn(10)
        o = fn(q, k, v, m)
        self.assertTrue(isinstance(o, SDPAParams))
        self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True, False))
        self.assertEqual(counter.frame_count, 1)

    def test_graph_break_SDPAParams(self):
        # [FUNCTION DOC] Same as the return test but the function has an explicit graph break
        # after building the SDPAParams and then returns it with an incremented query. Asserts
        # the returned object is an SDPAParams referencing the original tensors and that two
        # frames were compiled.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(q, k, v, m):
            # [FUNCTION DOC] HELPER: Compiled function; builds an SDPAParams, forces a graph
            # break, then returns it together with the query plus one.
            z = SDPAParams(q, k, v, m, 0.1, True, False)
            torch._dynamo.graph_break()
            return z, q + 1

        q = torch.randn(10)
        k = torch.randn(10)
        v = torch.randn(10)
        m = torch.randn(10)
        o, _ = fn(q, k, v, m)
        self.assertTrue(isinstance(o, SDPAParams))
        self.assert_ref_equals_params(o, SDPAParams(q, k, v, m, 0.1, True, False))
        self.assertEqual(counter.frame_count, 2)

    def test_input_SDPAParams(self):
        # [FUNCTION DOC] Passes an SDPAParams as an input to a compiled function that graph-
        # breaks first. Asserts the very same object comes back as the first output and that the
        # counting backend saw exactly one frame.
        counter = CompileCounter()

        @torch.compile(backend=counter)
        def fn(sdpap, q):
            # [FUNCTION DOC] HELPER: Compiled function; graph-breaks, then returns the input
            # SDPAParams and its query plus the second argument.
            torch._dynamo.graph_break()
            return sdpap, sdpap.query + q

        q = torch.randn(10)
        k = torch.randn(10)
        v = torch.randn(10)
        m = torch.randn(10)
        s = SDPAParams(q, k, v, m, 0.1, True, False)
        o, _ = fn(s, q)
        self.assertIs(o, s)
        self.assertEqual(counter.frame_count, 1)

    def test_intermediate_attr_access_SDPAParams(self):
        # [FUNCTION DOC] Checks that an SDPAParams created inside a full-graph compiled function
        # after an in-place add on the query supports attribute reads, and that the returned
        # object references the mutated input tensors. Asserts a single frame.
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v, m):
            # [FUNCTION DOC] HELPER: Compiled function; adds one in place to q, builds an
            # SDPAParams, reads its query attribute, and returns that plus one, the params
            # object, and q.
            q += 1
            z = SDPAParams(q, k, v, m, 0.1, True, False)
            a = z.query
            return a + 1, z, q

        q = torch.randn(10)
        k = torch.randn(10)
        v = torch.randn(10)
        m = torch.randn(10)
        _, o, _ = fn(q, k, v, m)
        expected = SDPAParams(q, k, v, m, 0.1, True, False)
        self.assert_ref_equals_params(o, expected)
        self.assertEqual(counter.frame_count, 1)

    def test_can_use_flash_attention_no_graph_break(self):
        # [FUNCTION DOC] Verifies the flash attention capability check can be called on a
        # freshly built SDPAParams inside a full-graph compile without a graph break. Compares
        # the compiled result to the eager result on a small four dimensional tensor and asserts
        # one frame.
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def fn(q, k, v):
            # [FUNCTION DOC] HELPER: Compiled function; returns whether flash attention can be
            # used for an SDPAParams built from q, k, v, no mask, zero dropout, causal true, gqa
            # false.
            return can_use_flash_attention(SDPAParams(q, k, v, None, 0.0, True, False))

        q = torch.randn(2, 2, 8, 8)
        expected = can_use_flash_attention(SDPAParams(q, q, q, None, 0.0, True, False))
        self.assertEqual(fn(q, q, q), expected)
        self.assertEqual(counter.frame_count, 1)

    def test_sdpa_c_functions_no_graph_break(self):
        # [FUNCTION DOC] Checks the private query of currently enabled SDPA kernel backends
        # compiles under full graph without a graph break. Asserts the result is a list and one
        # frame was compiled.
        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def test_cur_sdpa_kernel_backends():
            # [FUNCTION DOC] HELPER: Compiled zero-argument function that returns the list of
            # current SDPA kernel backends.
            return _cur_sdpa_kernel_backends()

        result = test_cur_sdpa_kernel_backends()

        self.assertIsInstance(result, list)
        self.assertEqual(counter.frame_count, 1)

    def test_sdpa_kernel_decorator_with_compile(self):
        # [FUNCTION DOC] Decorates an attention function with the sdpa_kernel context manager
        # using a priority list of math, efficient and flash backends, then compiles a caller of
        # it with full graph. Runs on a large float16 CPU tensor and asserts the output shape
        # equals the input shape and one frame was compiled.
        SDPA_BACKEND_PRIORITY = [
            SDPBackend.MATH,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.FLASH_ATTENTION,
        ]

        @sdpa_kernel(backends=SDPA_BACKEND_PRIORITY, set_priority=True)
        def scaled_dot_product_attention(q, k, v, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper decorated by sdpa_kernel with the priority
            # backends; forwards its arguments to the functional scaled_dot_product_attention
            # and returns the result.
            return torch.nn.functional.scaled_dot_product_attention(
                q, k, v, *args, **kwargs
            )

        counter = CompileCounter()

        @torch.compile(fullgraph=True, backend=counter)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled function; applies the decorated attention wrapper
            # with x as query, key and value.
            return scaled_dot_product_attention(x, x, x)

        x = torch.rand(128, 64, 64, 256, dtype=torch.float16)
        result = f(x)

        self.assertEqual(result.shape, x.shape)
        self.assertEqual(counter.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
