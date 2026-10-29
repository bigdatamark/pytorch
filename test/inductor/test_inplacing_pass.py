# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for the Inductor reinplacing pass (reinplace_inplaceable_ops_core) and the
# miss counters kept in ReinplaceCounters. Covers correctness of index_put style reinplacing
# under compile, view and copy-back patterns, auto_functionalized (old and v2) handling of views
# and lists, multi-output mutating custom ops, a partitioner recompute case with triton, and the
# with_effects wrapper being rewritten to the inplace op. Import-time side effects: registers
# custom ops in the _reinplacing and test_view namespaces, defines a triton kernel when a GPU is
# present, and sets the module level device to the GPU type. The entry point only runs on Linux
# with a GPU.

import operator

import torch
import torch._inductor.config as inductor_config
from functorch import make_fx
from torch import Tensor
from torch._dynamo.utils import ReinplaceCounters
from torch._higher_order_ops.auto_functionalize import (
    auto_functionalized,
    auto_functionalized_v2,
)
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.test_case import run_tests, TestCase as InductorTestCase
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_LINUX,
    parametrize,
    subtest,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.testing._internal.logging_utils import logs_to_string


aten = torch.ops.aten


const = torch.tensor(0.0)
device = GPU_TYPE


def num_reinplacing_failures():
    # [FUNCTION DOC] HELPER: Return the total count of missed reinplacing opportunities recorded
    # in ReinplaceCounters.
    return ReinplaceCounters.get_total_missed()


def miss_inplaced_bytes():
    # [FUNCTION DOC] HELPER: Return the total number of bytes whose reinplacing was missed
    # according to ReinplaceCounters.
    return ReinplaceCounters.get_total_missed_bytes()


@torch.library.custom_op("_reinplacing::sin", mutates_args={"result"})
def sin(x: torch.Tensor, result: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Custom op body (registered as _reinplacing::sin, mutating its
    # result argument) that copies the elementwise sine of the input into the result tensor.
    result.copy_(x.sin())


@torch.library.custom_op("_reinplacing::sin_cos", mutates_args={"out_sin", "out_cos"})
def sin_cos(x: torch.Tensor, out_sin: torch.Tensor, out_cos: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Custom op body (registered as _reinplacing::sin_cos, mutating both
    # output arguments) that writes sine of the input into the first output and cosine into the
    # second.
    out_sin.copy_(x.sin())
    out_cos.copy_(x.cos())


if HAS_GPU:
    import triton  # @manual
    import triton.language as tl  # @manual

    # [FUNCTION DOC] HELPER: Triton kernel, defined only when a GPU exists. Each program handles
    # a block of elements: loads the input under a bounds mask, applies sine, and stores to the
    # output pointer under the same mask. Arguments are the input pointer, output pointer,
    # element count and the compile-time block size.
    @triton.jit
    def sin_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = tl.sin(x)
        tl.store(out_ptr + offsets, output, mask=mask)

    def sin_triton(x, out):
        # [FUNCTION DOC] HELPER: Launch the triton sine kernel on input x writing into out, with
        # one program per element and block size 4. Defined only when a GPU exists.
        n_elements = x.numel()
        sin_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)

else:

    def sin_triton(x, out):
        # [FUNCTION DOC] HELPER: Fallback used when no GPU is available; does nothing and
        # returns None.
        return


@torch.library.custom_op("test_view::boo", mutates_args={"x"})
def boo(x: torch.Tensor) -> None:
    # [FUNCTION DOC] HELPER: Custom op body (registered as test_view::boo, mutating x) that
    # applies sine to x in place.
    x.sin_()


class TestReinplacingPassCorrectness(InductorTestCase):
    # [CLASS DOC] Tests of reinplacing correctness and counters. The setUp clears the global
    # reinplace counters so each test sees only its own misses. Most tests compile or trace
    # small functions and either compare against eager or run the reinplace pass on a traced
    # graph and inspect which ops remain.
    def setUp(self):
        # [FUNCTION DOC] Clear the global reinplace counters, then delegate to the parent setUp
        # and return its result.
        ReinplaceCounters.clear()
        return super().setUp()

    def _test(self, f):
        # [FUNCTION DOC] HELPER: Compile the given function, run both eager and compiled
        # versions on separate copies of a float input and an int index input, and assert the
        # outputs match and the inputs (including any in place changes) match afterwards.
        nf = torch.compile(f)
        inp = (
            torch.randn(4, device=device),
            torch.ones(2, device=device, dtype=torch.int),
        )
        inp2 = (inp[0].clone(), inp[1].clone())
        self.assertEqual(f(*inp), nf(*inp2))
        self.assertEqual(inp, inp2)

    def test_dont_modify_live(self):
        # [FUNCTION DOC] An index_put result is used alongside its source that is still live, so
        # the source must not be modified; compiled and eager results and inputs must agree.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Take cosine of x, index_put into it, and return both the
            # updated tensor and the cosine.
            x = x.cos()
            x2 = x.index_put((y,), const)
            return x2, x

        self._test(f)

    def test_dont_modify_view_of_live(self):
        # [FUNCTION DOC] An index_put is applied to an alias of a live tensor that is later used
        # again, so in place modification would be wrong; compiled and eager results and inputs
        # must agree.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Take cosine of x, index_put into an alias of it, then add
            # the cosine of x again to the updated value.
            x = x.cos()
            x2 = aten.alias(x)
            x2 = x2.index_put((y,), const)
            y = x2 + x.cos()
            return y

        self._test(f)

    def test_dont_modify_input(self):
        # [FUNCTION DOC] An index_put directly on a function input must not mutate that input;
        # compiled and eager results and inputs must agree.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return index_put of x at positions y with a constant value.
            return x.index_put((y,), const)

        self._test(f)

    def test_should_modify_inner(self):
        # [FUNCTION DOC] An index_put on an intermediate that has no other use may modify it in
        # place; compiled and eager results and inputs must agree.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Take cosine of x and index_put into the result.
            x = x.cos()
            x = x.index_put((y,), const)
            return x

        self._test(f)

    def test_should_modify_input(self):
        # [FUNCTION DOC] An explicit in place index_put on an input is returned; compiled and
        # eager results and inputs must agree.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Apply in place index_put to x and return it.
            x = x.index_put_((y,), const)
            return x

        self._test(f)

    def test_view_index_put_should_reinplace_copy_to_base(self):
        # [FUNCTION DOC] Traces a function that reshapes a cache, index_puts into the view,
        # reshapes back and copies into the cache. After running the reinplace pass, the graph
        # must use the in place index_put and contain neither the functional index_put nor the
        # copy, and the rewritten graph must produce the same output and cache contents as
        # eager.
        def f(input_pos, val, cache):
            # [FUNCTION DOC] HELPER: Index_put values into a flattened view of cache, reshape
            # the update back to cache shape, and copy it into the cache; returns the copy
            # result.
            cache_view = aten.reshape.default(cache, [4, -1])
            val_view = aten.reshape.default(val, [-1])
            updated = aten.index_put.default(cache_view, [input_pos], val_view)
            updated_base = aten.reshape.default(updated, [4, 3, 4])
            return aten.copy_.default(cache, updated_base)

        input_pos = torch.tensor([1], device=device)
        val = torch.randn(3, 4, device=device)
        cache = torch.randn(4, 3, 4, device=device)

        expected_cache = cache.clone()
        expected = f(input_pos, val, expected_cache)

        gm = make_fx(f, tracing_mode="fake")(input_pos, val, cache)
        reinplace_inplaceable_ops_core(gm.graph)
        gm.graph.lint()
        gm.recompile()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(aten.index_put_.default, targets)
        self.assertNotIn(aten.index_put.default, targets)
        self.assertNotIn(aten.copy_.default, targets)

        actual_cache = cache.clone()
        actual = gm(input_pos, val, actual_cache)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_cache, expected_cache)

    def test_view_index_put_keeps_copy_when_later_view_is_live(self):
        # [FUNCTION DOC] Same pattern as the previous test, but another view of the cache is
        # read before the copy. After the reinplace pass, the functional index_put and the copy
        # must remain and the in place index_put must not appear.
        def f(input_pos, val, cache):
            # [FUNCTION DOC] HELPER: Index_put into a flattened view of cache while a second
            # selected view of cache is used by a sine, then copy the reshaped update into the
            # cache; returns the sine and the copy.
            cache_view = aten.reshape.default(cache, [4, -1])
            live_view = aten.select.int(cache, 0, 0)
            val_view = aten.reshape.default(val, [-1])
            updated = aten.index_put.default(cache_view, [input_pos], val_view)
            live_use = aten.sin.default(live_view)
            updated_base = aten.reshape.default(updated, [4, 3, 4])
            copied = aten.copy_.default(cache, updated_base)
            return live_use, copied

        input_pos = torch.tensor([1], device=device)
        val = torch.randn(3, 4, device=device)
        cache = torch.randn(4, 3, 4, device=device)

        gm = make_fx(f, tracing_mode="fake")(input_pos, val, cache)
        reinplace_inplaceable_ops_core(gm.graph)
        gm.graph.lint()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(aten.index_put.default, targets)
        self.assertIn(aten.copy_.default, targets)
        self.assertNotIn(aten.index_put_.default, targets)

    def test_view_index_put_rejects_non_inverse_copy_back_view(self):
        # [FUNCTION DOC] The index_put is applied to a transposed view but the copy back uses a
        # strided view that is not the inverse of the transposition. The pass must keep the
        # functional index_put and the copy, and the traced graph must still match eager results
        # and cache contents.
        def f(input_pos, val, cache):
            # [FUNCTION DOC] HELPER: Index_put values into a transposed view of cache, take an
            # as_strided view of the update with fixed sizes and strides, and copy it into the
            # cache.
            cache_view = aten.transpose.int(cache, 0, 1)
            updated = aten.index_put.default(cache_view, [input_pos], val)
            copied_view = aten.as_strided.default(updated, [2, 3], [1, 2])
            return aten.copy_.default(cache, copied_view)

        input_pos = torch.tensor([1], device=device)
        val = torch.tensor([[10.0, 20.0]], device=device)
        cache = torch.arange(6, device=device, dtype=torch.float32).reshape(2, 3)

        expected_cache = cache.clone()
        expected = f(input_pos, val, expected_cache)

        gm = make_fx(f, tracing_mode="fake")(input_pos, val, cache)
        reinplace_inplaceable_ops_core(gm.graph)
        gm.graph.lint()
        gm.recompile()

        targets = [node.target for node in gm.graph.nodes]
        self.assertIn(aten.index_put.default, targets)
        self.assertIn(aten.copy_.default, targets)
        self.assertNotIn(aten.index_put_.default, targets)

        actual_cache = cache.clone()
        actual = gm(input_pos, val, actual_cache)
        self.assertEqual(actual, expected)
        self.assertEqual(actual_cache, expected_cache)

    def test_counters_functionalize_old(self):
        # [FUNCTION DOC] Uses the old auto_functionalized form where the output buffer is still
        # used after the call, so reinplacing fails. Checks the missed count is one and missed
        # bytes are twelve (three float32 elements).
        ReinplaceCounters.clear()

        def f(x):
            # [FUNCTION DOC] HELPER: Allocate an empty buffer, call the sin custom op through
            # auto_functionalized, and return the new output and its product with the old
            # buffer.
            out = torch.empty_like(x)
            _, new_out = auto_functionalized(sin._opoverload, x=x, result=out)
            y = out * new_out
            return new_out, y

        x = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x)
        reinplace_inplaceable_ops_core(gm.graph)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)
        self.assertEqual(miss_inplaced_bytes(), 12)

    def test_counters_functionalize_v2(self):
        # [FUNCTION DOC] Same as the previous test but with auto_functionalized_v2 given
        # explicit base and view metadata. Checks only that the missed count is one.
        ReinplaceCounters.clear()

        def f(x):
            # [FUNCTION DOC] HELPER: Allocate an empty buffer, call the sin custom op through
            # auto_functionalized_v2 with that buffer as the single base, and return the new
            # output and its product with the old buffer.
            out = torch.empty_like(x)
            _, new_out = auto_functionalized_v2(
                sin._opoverload,
                x=x,
                _result_base_index=0,
                _result_size=(3,),
                _result_stride=(1,),
                _result_storage_offset=0,
                _all_bases=[out],
            )
            y = out * new_out
            return new_out, y

        x = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x)
        reinplace_inplaceable_ops_core(gm.graph)

        # We shouldn't have been able to reinplace `out` because it was used after
        # auto_functionalized. Note that this usually doesn't happen in practice;
        # we're artificially creating this example to test the counter.
        # IF THIS NUMBER GOES TO ZERO, PLEASE FIND ANOTHER EXAMPLE
        self.assertEqual(num_reinplacing_failures(), 1)

    def get_not_inplaced_count(self, graph):
        # [FUNCTION DOC] HELPER: Walk a graph and sum the number of tensors recorded in node
        # metadata as needing a clone, over all auto_functionalized and v2 nodes. Raises
        # AssertionError if the graph has no such node.
        counter = 0
        auto_functionalized_found = False
        for node in graph.nodes:
            if (node.target == torch.ops.higher_order.auto_functionalized) or (
                node.target == torch.ops.higher_order.auto_functionalized_v2
            ):
                auto_functionalized_found = True
                counter += len(node.meta["only_clone_these_tensors"])
        if not auto_functionalized_found:
            raise AssertionError
        return counter

    def test_view_inplaced_functionalize_v2(self):
        # [FUNCTION DOC] A view of the input is created but unused, and the mutating op is
        # followed by a copy back to the input. After the reinplace pass no tensors need
        # cloning.
        def f(arg0_1):
            # [FUNCTION DOC] HELPER: Select a view of the input, call the boo op through
            # auto_functionalized_v2 on the input base, then copy the result back into the
            # input; returns an empty tuple.
            torch.ops.aten.select.int(arg0_1, 0, 0)
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return ()

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

    # introduce a view another_view that is used `after` the copy
    def test_view_inplaced2_functionalize_v2(self):
        # [FUNCTION DOC] Like the previous test but another view of the input is used after the
        # copy back. After the reinplace pass no tensors need cloning.
        def f(arg0_1):
            # [FUNCTION DOC] HELPER: Create two views of the input, run the boo op via
            # auto_functionalized_v2, copy the result back into the input, and return the second
            # view.
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            _copy = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 0)

    # introduce a view another_view that is used `before` the copy
    def test_views_not_inplaced_functionalize_v2(self):
        # [FUNCTION DOC] Another view of the input is used before the copy back, so one tensor
        # must still be cloned after the reinplace pass.
        def f(arg0_1):
            # [FUNCTION DOC] HELPER: Create two views of the input, run the boo op via
            # auto_functionalized_v2, multiply the second view by ten, then copy back and return
            # the product.
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            getitem_1 = auto_functionalized[1]
            use_another_view = another_view * 10
            _copy = torch.ops.aten.copy_.default(arg0_1, getitem_1)
            return use_another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    # a view over input without copy node, inplace not allowed
    def test_views_not_inplaced2_functionalize_v2(self):
        # [FUNCTION DOC] Views of the input exist but there is no copy back node, so inplace is
        # not allowed. After the reinplace pass one tensor needs cloning.
        def f(arg0_1):
            # [FUNCTION DOC] HELPER: Create two views of the input, run the boo op via
            # auto_functionalized_v2 on the input, and return nothing.
            _select = torch.ops.aten.select.int(arg0_1, 0, 0)
            _another_view = arg0_1[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(3,),
                _x_stride=(1,),
                _x_storage_offset=0,
                _all_bases=[arg0_1],
            )
            _getitem_1 = auto_functionalized[1]
            return

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    # no copy nodes, view over local, with a use for another view
    def test_views_not_inplaced3_functionalize_v2(self):
        # [FUNCTION DOC] The mutated base is a local tensor with a view that is returned later
        # and no copy node exists. After the reinplace pass one tensor needs cloning.
        def f(arg0_1):
            # [FUNCTION DOC] HELPER: Make a local tensor of ones and a view of it, run the boo
            # op via auto_functionalized_v2 on the local tensor, and return the view.
            a = torch.ones(10)
            another_view = a[2]
            auto_functionalized = auto_functionalized_v2(
                torch.ops.test_view.boo.default,
                _x_base_index=0,
                _x_size=(),
                _x_stride=(),
                _x_storage_offset=0,
                _all_bases=[a],
            )
            _getitem_1 = auto_functionalized[1]
            return another_view

        x1 = torch.randn(3, device=device)
        gm = make_fx(f, tracing_mode="fake")(x1)
        reinplace_inplaceable_ops_core(gm.graph)

        self.assertEqual(self.get_not_inplaced_count(gm.graph), 1)

    def test_multi_output_intermediate(self):
        # [FUNCTION DOC] For each combination of requires-grad and the auto functionalized v2
        # config flag, compiles a function calling the two-output mutating op on fresh buffers.
        # Checks the first output equals sine and the second equals cosine of the input, and
        # that no reinplacing failures were counted.
        for requires_grad in [False, True]:
            for enable_v2 in [False, True]:
                with inductor_config.patch(
                    {"enable_auto_functionalized_v2": enable_v2}
                ):
                    ReinplaceCounters.clear()

                    def f(x):
                        # [FUNCTION DOC] HELPER: Allocate two empty buffers, fill them with sine
                        # and cosine through the custom op, and return both plus the square of
                        # x.
                        out1 = torch.empty_like(x)
                        out2 = torch.empty_like(x)
                        sin_cos(x, out1, out2)
                        return out1, out2, x**2

                    x = torch.randn(3, device=device, requires_grad=requires_grad)
                    res1, res2, _ = torch.compile(f)(x)
                    self.assertEqual(res1, x.sin())
                    self.assertEqual(res2, x.cos())
                    self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_mutations(self):
        # [FUNCTION DOC] Compiles a function that applies the sin op three times to a passed-in
        # buffer. Checks the result equals three nested sines, the buffer was mutated to the
        # same value, and no reinplacing failures occurred.
        ReinplaceCounters.clear()

        def f(x, out):
            # [FUNCTION DOC] HELPER: Apply the sin custom op to x into out, then twice from out
            # into itself, and return out.
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        out = torch.randn(3, device=device)
        result = torch.compile(f)(x, out)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(result, out)
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_multiple_intermediate(self):
        # [FUNCTION DOC] Same chain of three sin applications but on a buffer allocated inside
        # the function. Checks the result and that no reinplacing failures occurred.
        ReinplaceCounters.clear()

        def f(x):
            # [FUNCTION DOC] HELPER: Allocate an empty buffer, apply the sin op once from x and
            # twice from the buffer into itself, and return it.
            out = torch.empty_like(x)
            sin(x, out)
            sin(out, out)
            sin(out, out)
            return out

        x = torch.randn(3, device=device)
        result = torch.compile(f)(x)
        self.assertEqual(result, x.sin().sin().sin())
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_lists_functionalize_v2(self):
        # [FUNCTION DOC] With the v2 config on, registers a custom op mutating a list of
        # tensors, compiles a function passing two slices of its input, captures the post grad
        # graph log, and checks that nothing was missed (counters zero) and no clone appears in
        # the graph.
        with inductor_config.patch({"enable_auto_functionalized_v2": True}):

            @torch.library.custom_op("mylib::mutate_op", mutates_args={"y"})
            def mutate_op(y: list[Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Custom op body (mylib::mutate_op, mutating the list)
                # that adds 2 to the first tensor and 3 to the second in place.
                y[0].add_(2)
                y[1].add_(3)

            @torch.compile(fullgraph=True, dynamic=False, backend="inductor")
            def f(b):
                # [FUNCTION DOC] HELPER: Call the mutating list op on the first two rows of the
                # input.
                mutate_op([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4], device=device)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(x1)
            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # We can inplace the base y. no clones emitted.
            self.assertEqual(num_reinplacing_failures(), 0)
            self.assertEqual(miss_inplaced_bytes(), 0)
            self.assertEqual(post_grad_graphs.count("aten.clone"), 0)

    def test_lists_old_functionalize(self):
        # [FUNCTION DOC] With the v2 config off, repeats the list mutation scenario. Expects one
        # missed opportunity with 8 missed bytes, since reinplacing on views is not supported
        # for the old form, and exactly two clone ops in the post grad graph.
        with inductor_config.patch({"enable_auto_functionalized_v2": False}):

            @torch.library.custom_op("mylib::mutate_op", mutates_args={"y"})
            def mutate_op(y: list[Tensor]) -> None:
                # [FUNCTION DOC] HELPER: Custom op body (mylib::mutate_op, mutating the list)
                # that adds 2 to the first tensor and 3 to the second in place.
                y[0].add_(2)
                y[1].add_(3)

            @torch.compile(fullgraph=True, dynamic=False, backend="inductor")
            def f(b):
                # [FUNCTION DOC] HELPER: Call the mutating list op on the first two rows of the
                # input.
                mutate_op([b[0], b[1]])

            x1 = torch.tensor([0.3, 0.4], device=device)
            log_stream, ctx = logs_to_string(
                "torch._inductor.compile_fx", "post_grad_graphs"
            )
            with ctx():
                torch.compile(f, backend="inductor", fullgraph=True)(x1)
            post_grad_graphs = "\n".join(
                log_stream.getvalue().strip().split("\n")[3:]
            ).strip()

            # Can't reinplace on views yet (1 for the "entire list" failing to reinplace)
            self.assertEqual(num_reinplacing_failures(), 1)
            self.assertEqual(miss_inplaced_bytes(), 8)

            # Both list inputs failed to reinplace. So we should have emitted clones for them.
            self.assertEqual(post_grad_graphs.count("aten.clone"), 2)

    def test_generalized_scatter(self):
        # [FUNCTION DOC] Integration test: compiles with fullgraph a function that writes into a
        # column slice, clones, takes an as_strided view, copies into it, and sums results, and
        # compares with eager output on a random input.
        # This is an integration test for the reinplacing pass.
        def fn(x_1):
            # [FUNCTION DOC] HELPER: Build tensors with slice copies and a strided view clone as
            # described, then return the sum of the clone of the base and an assembled tensor.
            # The argument is unused.
            a = torch.ones([2, 3])
            c = torch.ones(2)
            a[:, 0].copy_(c)

            d = a.clone()
            e = torch.ops.aten.as_strided.default(d, [2], [3], 0)
            f = e.clone()

            g = torch.zeros(2)
            e.copy_(g)

            h = torch.zeros(2, 3)
            h[:, 0].copy_(f)

            add_1 = d + h
            return add_1

        x = torch.randn(2, 3)
        expected = fn(x)
        result = torch.compile(fn, fullgraph=True, backend="inductor")(x)
        self.assertEqual(result, expected)

    @parametrize(
        "factory_op",
        [
            subtest(torch.ones_like, name="ones_like"),
            subtest(torch.empty_like, name="empty_like"),
        ],
    )
    @parametrize(
        "sin_op",
        [
            subtest(sin, name="sin_op"),
            subtest(sin_triton, name="sin_triton"),
        ],
    )
    def test_partitioner_recomputes_factory(self, factory_op, sin_op):
        # [FUNCTION DOC] Parametrized over the factory op (ones_like or empty_like) and the sin
        # op (custom op or triton). Compiles a function using a custom autograd function whose
        # forward and backward create an output with the factory op and fill it with the sin op.
        # Runs the compiled forward with a grad-requiring input and asserts no reinplacing
        # failures were counted.
        class MySin(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd function. Forward creates an output with the
            # factory op, fills it with the sin op, saves it and returns it. Backward builds a
            # new output from the gradient and fills it using the saved tensor.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: Create an output with the factory op, fill it with the
                # sin op, save it for backward, and return it.
                out = factory_op(x)
                sin_op(x, out)
                ctx.save_for_backward(out)
                return out

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: Fetch the saved tensor, make a new output from the
                # gradient with the factory op, fill it with the sin op from the saved tensor,
                # and return it.
                (saved,) = ctx.saved_tensors
                out = factory_op(grad)
                sin_op(saved, out)
                return out

        @torch.compile(backend="inductor")
        def f(x):
            # [FUNCTION DOC] HELPER: Apply the custom autograd function to x.
            return MySin.apply(x)

        x = torch.randn(3, requires_grad=True, device=device)
        f(x)
        self.assertEqual(num_reinplacing_failures(), 0)

    def test_with_effects_reinplace(self):
        # [FUNCTION DOC] Registers a functional and an inplace custom op pair and a mapping
        # between them in the inplaceable ops table, traces a graph that calls the functional op
        # through with_effects, and runs the reinplace pass. Asserts the wrapped op becomes the
        # inplace op and that no result extracting getitem nodes remain on the wrapper. The
        # table entry is removed in a finally block.
        """Test that the reinplace pass correctly handles with_effects wrapped ops.

        When an effectful op is wrapped with with_effects, the reinplace pass should:
        1. Find the inner op in the inplaceable_ops registry
        2. Correctly compute the mutated arg indices (offset by 2 for token and op)
        3. Convert functional -> inplace when safe
        4. Replace getitem nodes that extract results with input tensors
        """
        from torch._higher_order_ops.effects import with_effects
        from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp

        # Define a simple effectful op pair (functional and inplace)
        @torch.library.custom_op("_test_effects::my_add", mutates_args=())
        def my_add_functional(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Functional custom op body that returns the sum of two
            # tensors.
            return x + y

        @my_add_functional.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor shaped like
            # the first argument.
            return torch.empty_like(x)

        @torch.library.custom_op("_test_effects::my_add_", mutates_args={"x"})
        def my_add_inplace(x: torch.Tensor, y: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Inplace custom op body that adds the second tensor into the
            # first in place.
            x.add_(y)

        @my_add_inplace.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Fake implementation of the inplace op; does nothing.
            pass

        # Register the functional -> inplace mapping
        inplaceable_ops[my_add_functional._opoverload] = InplaceableOp(
            my_add_inplace._opoverload,
            0,  # First tensor arg (x) is mutated
        )

        try:

            def f(a, b):
                # [FUNCTION DOC] HELPER: Clone both inputs so they come from ops, create a
                # dependency token, call the functional op under with_effects, and return the
                # tensor part of the result.
                # Create tensors from ops (not placeholders) so can_inplace works
                x = a.clone()
                y = b.clone()
                token = torch.ops.aten._make_dep_token()
                result = with_effects(token, my_add_functional._opoverload, x, y)
                # with_effects returns (new_token, result_tensor)
                return result[1]

            a = torch.randn(3, device=device)
            b = torch.randn(3, device=device)

            gm = make_fx(f, tracing_mode="fake")(a, b)

            # Find the with_effects node before reinplace
            with_effects_nodes_before = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_before), 1)

            # The inner op should be the functional version
            inner_op_before = with_effects_nodes_before[0].args[1]
            self.assertEqual(inner_op_before, my_add_functional._opoverload)

            # Count getitem nodes before reinplace
            getitem_nodes_before = [
                n for n in gm.graph.nodes if n.target is operator.getitem
            ]
            self.assertGreater(len(getitem_nodes_before), 0)

            # Run reinplace pass
            reinplace_inplaceable_ops_core(gm.graph)

            # Find the with_effects node after reinplace
            with_effects_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_after), 1)

            # The inner op should now be the inplace version
            inner_op_after = with_effects_nodes_after[0].args[1]
            self.assertEqual(inner_op_after, my_add_inplace._opoverload)

            # Verify getitem nodes for result extraction are cleaned up
            # (getitem[i] for i >= 1 on with_effects should be removed or replaced)
            getitem_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is operator.getitem
                and n.args[0] is with_effects_nodes_after[0]
                and n.args[1] >= 1
            ]
            self.assertEqual(
                len(getitem_nodes_after),
                0,
                "getitem nodes extracting result should be cleaned up",
            )

        finally:
            if my_add_functional._opoverload in inplaceable_ops:
                del inplaceable_ops[my_add_functional._opoverload]

    def test_with_effects_reinplace_multiple_mutated_args(self):
        # [FUNCTION DOC] Same as the previous test but for an effectful op that returns two
        # tensors and whose inplace version mutates two arguments, registered with a pair of
        # argument indices. After the reinplace pass the wrapper must call the inplace op and no
        # result extracting getitem nodes may remain. The table entry is removed in a finally
        # block.
        """Test reinplace pass with multiple mutated args in with_effects wrapped ops.

        This tests the case where an effectful op mutates multiple tensors,
        such as collective operations that update multiple output buffers.
        """
        from torch._higher_order_ops.effects import with_effects
        from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp

        # Define an op that returns two tensors (functional version)
        @torch.library.custom_op("_test_effects::my_swap", mutates_args=())
        def my_swap_functional(
            x: torch.Tensor, y: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Functional custom op body returning clones of the two
            # inputs in swapped order.
            # Swap values between x and y
            return y.clone(), x.clone()

        @my_swap_functional.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            # [FUNCTION DOC] HELPER: Fake implementation returning two empty tensors shaped like
            # the two inputs.
            return torch.empty_like(x), torch.empty_like(y)

        # Define inplace version that mutates both x and y
        @torch.library.custom_op("_test_effects::my_swap_", mutates_args={"x", "y"})
        def my_swap_inplace(x: torch.Tensor, y: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Inplace custom op body that swaps the contents of the two
            # tensors in place using a temporary copy.
            tmp = x.clone()
            x.copy_(y)
            y.copy_(tmp)

        @my_swap_inplace.register_fake
        def _(x: torch.Tensor, y: torch.Tensor) -> None:
            # [FUNCTION DOC] HELPER: Fake implementation of the inplace swap op; does nothing.
            pass

        # Register the functional -> inplace mapping with multiple mutated args
        inplaceable_ops[my_swap_functional._opoverload] = InplaceableOp(
            my_swap_inplace._opoverload,
            (0, 1),  # Both x (arg 0) and y (arg 1) are mutated
        )

        try:

            def f(a, b):
                # [FUNCTION DOC] HELPER: Clone both inputs, create a dependency token, call the
                # functional swap under with_effects, and return both tensors of the result.
                # Create tensors from ops (not placeholders) so can_inplace works
                x = a.clone()
                y = b.clone()
                token = torch.ops.aten._make_dep_token()
                result = with_effects(token, my_swap_functional._opoverload, x, y)
                # with_effects returns (new_token, (tensor1, tensor2))
                # Access both results
                out = result[1]
                return out[0], out[1]

            a = torch.randn(3, device=device)
            b = torch.randn(3, device=device)

            gm = make_fx(f, tracing_mode="fake")(a, b)

            # Find the with_effects node before reinplace
            with_effects_nodes_before = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_before), 1)

            # The inner op should be the functional version
            inner_op_before = with_effects_nodes_before[0].args[1]
            self.assertEqual(inner_op_before, my_swap_functional._opoverload)

            # Run reinplace pass
            reinplace_inplaceable_ops_core(gm.graph)

            # Find the with_effects node after reinplace
            with_effects_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_after), 1)

            # The inner op should now be the inplace version
            inner_op_after = with_effects_nodes_after[0].args[1]
            self.assertEqual(inner_op_after, my_swap_inplace._opoverload)

            # Verify getitem nodes for result extraction are cleaned up
            # (getitem[i] for i >= 1 on with_effects should be removed or replaced)
            getitem_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is operator.getitem
                and n.args[0] is with_effects_nodes_after[0]
                and n.args[1] >= 1
            ]
            self.assertEqual(
                len(getitem_nodes_after),
                0,
                "getitem nodes extracting result tuple should be cleaned up",
            )

        finally:
            if my_swap_functional._opoverload in inplaceable_ops:
                del inplaceable_ops[my_swap_functional._opoverload]

    def test_with_effects_reinplace_list_arg(self):
        # [FUNCTION DOC] Same as the previous tests but the mutated argument is a list of
        # tensors, like a wait on several tensors. Registers a functional and an inplace op over
        # a list, runs the pass on a traced graph with three inputs, and asserts the wrapper now
        # calls the inplace op and no result extracting getitem nodes remain. The table entry is
        # removed in a finally block.
        """Test reinplace pass with list arguments in with_effects wrapped ops.

        This tests the case where the mutated argument is a list of tensors,
        similar to wait_tensors operations that take a list of tensors to wait on.
        """
        from torch._higher_order_ops.effects import with_effects
        from torch._inductor.fx_passes.reinplace import inplaceable_ops, InplaceableOp

        # Define an op that takes a list of tensors (functional version)
        @torch.library.custom_op("_test_effects::my_wait", mutates_args=())
        def my_wait_functional(
            tensors: list[torch.Tensor],
        ) -> list[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Functional custom op body returning a clone of each tensor
            # in the list.
            return [t.clone() for t in tensors]

        @my_wait_functional.register_fake
        def _(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
            # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor shaped like
            # each input in the list.
            return [torch.empty_like(t) for t in tensors]

        # Define inplace version that mutates the list of tensors
        @torch.library.custom_op("_test_effects::my_wait_", mutates_args={"tensors"})
        def my_wait_inplace(tensors: list[torch.Tensor]) -> None:
            # [FUNCTION DOC] HELPER: Inplace custom op body for the list case; does nothing.
            pass  # In reality would wait on async ops

        @my_wait_inplace.register_fake
        def _(tensors: list[torch.Tensor]) -> None:
            # [FUNCTION DOC] HELPER: Fake implementation of the inplace list op; does nothing.
            pass

        # Register the functional -> inplace mapping
        inplaceable_ops[my_wait_functional._opoverload] = InplaceableOp(
            my_wait_inplace._opoverload,
            0,  # First arg (tensors list) is mutated
        )

        try:

            def f(a, b, c):
                # [FUNCTION DOC] HELPER: Clone three inputs, create a dependency token, call the
                # functional list op under with_effects on the list, and return the three
                # tensors of the result.
                # Create tensors from ops (not placeholders) so can_inplace works
                x = a.clone()
                y = b.clone()
                z = c.clone()
                token = torch.ops.aten._make_dep_token()
                result = with_effects(token, my_wait_functional._opoverload, [x, y, z])
                # Access individual results from the list
                out = result[1]
                return out[0], out[1], out[2]

            a = torch.randn(3, device=device)
            b = torch.randn(3, device=device)
            c = torch.randn(3, device=device)

            gm = make_fx(f, tracing_mode="fake")(a, b, c)

            # Find the with_effects node before reinplace
            with_effects_nodes_before = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_before), 1)

            # The inner op should be the functional version
            inner_op_before = with_effects_nodes_before[0].args[1]
            self.assertEqual(inner_op_before, my_wait_functional._opoverload)

            # Run reinplace pass
            reinplace_inplaceable_ops_core(gm.graph)

            # Find the with_effects node after reinplace
            with_effects_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is torch.ops.higher_order.with_effects
            ]
            self.assertEqual(len(with_effects_nodes_after), 1)

            # The inner op should now be the inplace version
            inner_op_after = with_effects_nodes_after[0].args[1]
            self.assertEqual(inner_op_after, my_wait_inplace._opoverload)

            # Verify getitem nodes for result extraction are cleaned up
            # (getitem[i] for i >= 1 on with_effects should be removed or replaced)
            getitem_nodes_after = [
                n
                for n in gm.graph.nodes
                if n.target is operator.getitem
                and n.args[0] is with_effects_nodes_after[0]
                and n.args[1] >= 1
            ]
            self.assertEqual(
                len(getitem_nodes_after),
                0,
                "getitem nodes extracting result list should be cleaned up",
            )

        finally:
            if my_wait_functional._opoverload in inplaceable_ops:
                del inplaceable_ops[my_wait_functional._opoverload]


instantiate_parametrized_tests(TestReinplacingPassCorrectness)


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests(needs="filelock")
