# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests that inductor preserves the exact strides an operator requires from its
# inputs: a custom op with a layout-sensitive implementation behind a custom post-grad pattern
# pass, and a triton kernel writing into a strided view under dynamic shapes. When a GPU and
# triton are available it defines a small triton fill kernel at import time. The main block only
# runs the tests on Linux with a GPU and triton.

import torch
import torch.utils._pytree as pytree
from torch._inductor.pattern_matcher import (
    CallFunctionVarArgs,
    PatternMatcherPass,
    register_graph_pattern,
)
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON


if HAS_GPU_AND_TRITON:
    import triton
    import triton.language as tl

    # [FUNCTION DOC] HELPER: Triton kernel, one program per token. Programs with an index at or
    # past num_tokens return immediately; the others write the value one into INNER elements of
    # the buffer starting at the program index times stride_t, stepping by stride_i. Returns
    # nothing; mutates the buffer.
    @triton.jit
    def _fill_kernel(
        buf_ptr,
        num_tokens,
        stride_t,
        stride_i,
        INNER: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_tokens:
            return
        for i in range(INNER):
            tl.store(buf_ptr + pid * stride_t + i * stride_i, 1.0)


class TestNeedsExactStrides(InductorTestCase):
    # [CLASS DOC] Test case for exact-stride handling in inductor, parametrized over dtype or
    # size. Covers a custom op fed a transposed view, and a mutating triton kernel on a view of a
    # dynamic size buffer.
    @parametrize("dtype", [torch.float, torch.float8_e8m0fnu])
    def test_custom_op(self, dtype):
        # [FUNCTION DOC] Registers a custom post-grad pass that rewrites every permute into a
        # permute followed by a custom op forcing channels-last output, defines a custom add op in
        # a scoped library that checks its first argument, once transposed, and its second
        # argument are both contiguous, then compiles a function calling that op on a transposed
        # input with fullgraph and asserts the pattern callback ran. Parametrized over dtype.
        # KNOWN ISSUE: the dtype argument is never used; the tensors are always created with the
        #   float8_e8m0fnu dtype, so both parametrized cases run the same thing. The exception
        #   class TestPassed is also never raised.
        device = (
            torch.accelerator.current_accelerator()
        )  # float8_e8m0fnu errors on "cpu"
        x = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)
        other = torch.ones(4, 4, 2, 2, device=device, dtype=torch.float8_e8m0fnu)

        class _CustomPass(PatternMatcherPass):
            # [CLASS DOC] HELPER: Pattern matcher pass subclass used as the custom post-grad pass;
            # calling it on a graph just applies its registered patterns. No extra state.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the parent constructor and does nothing else.
                super().__init__()

            def __call__(self, g: torch.fx.Graph):
                # [FUNCTION DOC] HELPER: Applies the registered patterns to the given FX graph in
                # place and returns nothing.
                self.apply(g)

        g = _CustomPass()
        called = False

        @register_graph_pattern(
            CallFunctionVarArgs(torch.ops.aten.permute),
            pass_dict=g,
        )
        def _(match, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Pattern handler for permute calls. Flattens the match
            # arguments with pytree, sets the enclosing flag recording that the pass ran, and
            # replaces the match with a traced decomposition that permutes and then forces
            # channels-last layout.
            flat_args, spec = pytree.tree_flatten((args, kwargs))

            def decomp(*flat_args):
                # [FUNCTION DOC] HELPER: Decomposition used as the replacement: rebuilds the
                # original arguments from the flat list and spec, permutes, and passes the result
                # through the custom op that returns a channels-last clone.
                args, kwargs = pytree.tree_unflatten(flat_args, spec)
                return torch.ops.mylib.force_channels_last(
                    torch.ops.aten.permute(*args, **kwargs)
                )

            nonlocal called
            called = True
            match.replace_by_example(decomp, flat_args)

        from torch._inductor import config

        class TestPassed(RuntimeError):
            # [CLASS DOC] HELPER: Marker exception type (runtime error subclass) meant to be
            # caught around compilation; has no extra behavior.
            pass

        with torch.library._scoped_library("mylib", "FRAGMENT") as lib:
            lib.define(
                "force_channels_last(Tensor x) -> Tensor",
                tags=[torch._C.Tag.flexible_layout],
            )

            def impl2(x):
                # [FUNCTION DOC] HELPER: Eager implementation of the channels-last op; returns a
                # clone of the input in channels-last memory format.
                return x.clone(memory_format=torch.channels_last)

            lib.impl("force_channels_last", impl2, "CompositeExplicitAutograd")

            lib.define(
                "add_op(Tensor x, Tensor y) -> Tensor",
            )

            def impl(x, y):
                # [FUNCTION DOC] HELPER: Eager implementation of the custom add op. Raises
                # AssertionError unless the first argument (with dims 2 and 3 swapped) and
                # the second argument are contiguous, then returns the float sum of the two.
                if not x.transpose(2, 3).is_contiguous():
                    raise AssertionError
                if not y.is_contiguous():
                    raise AssertionError
                return x.float() + y.float()

            def meta(x, y):
                # [FUNCTION DOC] HELPER: Meta (shape-only) implementation of the custom add op;
                # returns the float sum of the two inputs without checking strides.
                return x.float() + y.float()

            lib.impl("add_op", impl, "CompositeExplicitAutograd")
            lib.impl("add_op", meta, "Meta")

            def f(x, other):
                # [FUNCTION DOC] HELPER: Function under compilation; calls the custom add op on
                # the input with dims 2 and 3 transposed, and the other tensor.
                return torch.ops.mylib.add_op.default(x.transpose(2, 3), other)

            with config.patch(
                post_grad_custom_post_pass=g,
            ):
                try:
                    f_compile = torch.compile(f, fullgraph=True)
                    f_compile(x, other)
                except TestPassed:
                    pass
                if not called:
                    raise AssertionError

    @parametrize("n", [64, 128, 256])
    def test_dynamic_size_one_leading_dim_view_triton_kernel_wrapper_functional(
        self, n
    ):
        # [FUNCTION DOC] Compiles with inductor (fullgraph, dynamic) a function that allocates a
        # padded float buffer, views it with size-one leading dim and non-contiguous strides,
        # launches the triton fill kernel over it, and returns the view. Asserts the compiled
        # output equals eager output on cloned inputs. Parametrized over the token count.
        device = torch.accelerator.current_accelerator()
        inner = 8

        def fn(x):
            # [FUNCTION DOC] HELPER: Builds the strided buffer view described above (token count
            # padded up to a multiple of four), runs the fill kernel on it in place with the view
            # strides and returns the view.
            num_tokens = x.shape[0]
            aligned_num_tokens = ((num_tokens + 3) // 4) * 4
            buf = torch.empty(
                inner * aligned_num_tokens,
                dtype=torch.float32,
                device=x.device,
            ).as_strided(
                (1, num_tokens, inner),
                (inner * aligned_num_tokens, 1, aligned_num_tokens),
            )

            _fill_kernel[(num_tokens,)](
                buf,
                num_tokens,
                stride_t=buf.stride(1),
                stride_i=buf.stride(2),
                INNER=inner,
            )

            return buf

        compiled = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=True)
        x = torch.randn(n, inner, device=device)
        self.assertEqual(compiled(x.clone()), fn(x.clone()))


instantiate_parametrized_tests(TestNeedsExactStrides)

if __name__ == "__main__":
    if IS_LINUX and HAS_GPU_AND_TRITON:
        run_tests(needs="filelock")
