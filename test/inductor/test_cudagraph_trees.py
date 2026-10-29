# Owner(s): ["module: inductor"]
# ruff: noqa: F841
# [MODULE DOC] Tests for Inductor CUDA graph trees (reduce-overhead mode and the cudagraphs
# backend). Defines shared helpers (compile-function selection, stderr capture, live cudagraph
# block accounting) and a large CudaGraphTreeTests class that checks tree structure, replay,
# memory reuse, mutation, dynamic shapes, RNG, multi-device and error handling. On Windows CI
# the module exits or skips at import; the GPU tests are defined only when CUDA and Triton are
# available. Import-time side effects: imports functorch and filelock, and registers a symbolic
# custom class.
import contextlib
import functools
import gc
import importlib
import itertools
import os
import re
import sys
import unittest
import warnings
from collections import defaultdict
from collections.abc import Mapping, Sequence

import torch
import torch._dynamo.config as dynamo_config
import torch.nn as nn
from torch._dynamo.backends.debugging import aot_eager_decomp_partition_with_mode
from torch._dynamo.utils import counters
from torch._functorch._aot_autograd.autograd_cache import AOTAutogradCache
from torch._inductor import config
from torch._inductor.codecache import FxGraphCache
from torch._inductor.compile_fx import compile_fx_inner
from torch._inductor.cudagraph_trees import (
    AliasesPriorGraphOutput,
    cudagraphify_impl as tree_cudagraphify_impl,
    ExecutionState,
)
from torch._inductor.cudagraph_utils import PlaceholderInfo
from torch._inductor.test_case import TestCase as InductorTestCase
from torch._inductor.utils import run_and_get_code
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.immutable_collections import immutable_dict
from torch.testing import FileCheck
from torch.testing._internal.common_cuda import blas_library_context, TEST_MULTIGPU
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_ARM64,
    IS_CI,
    IS_LINUX,
    IS_WINDOWS,
    IS_X86,
    parametrize,
    skipIfRocm,
    TEST_CUDA_GRAPH,
    TEST_WITH_SLOW,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.testing._internal.logging_utils import logs_to_string
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode


if IS_WINDOWS and IS_CI:
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires sympy/functorch/filelock")

importlib.import_module("functorch")
importlib.import_module("filelock")


aten = torch.ops.aten
requires_multigpu = functools.partial(
    unittest.skipIf, not TEST_MULTIGPU, "requires multiple cuda devices"
)
from io import StringIO

from torch._library.opaque_object import CustomClassBase, register_custom_class


class _CudagraphTestScaleFactor(CustomClassBase):
    # [CLASS DOC] HELPER: Opaque custom class used as a symbolic input in tests; it only carries
    # a numeric scale factor attribute.
    def __init__(self, factor):
        # [FUNCTION DOC] HELPER: Stores the given factor on the instance as the factor
        # attribute.
        self.factor = factor


register_custom_class(_CudagraphTestScaleFactor, typ="symbolic")


def get_compile_fn(backend):
    # [FUNCTION DOC] HELPER: Returns a partial of torch.compile bound to the backend named
    # cudagraphs when the argument equals that name, otherwise bound to the reduce-overhead mode
    # with the default backend.
    if backend == "cudagraphs":
        return functools.partial(torch.compile, backend="cudagraphs")
    else:
        return functools.partial(torch.compile, mode="reduce-overhead")


def get_num_partitions(code):
    # [FUNCTION DOC] HELPER: Joins the generated code pieces into one string, finds the
    # partitions list in it with a regular expression and returns how many non-empty comma
    # separated entries it has. Raises AssertionError if no partitions list is present.
    """Get the number of cudagraph partitions from generated code."""
    code = "".join(code)
    found = re.search(r"partitions=\[(.*)\]", code)
    if found is None:
        raise AssertionError("Could not find partitions in generated code")
    partitions = found.group(1)
    return len([p for p in partitions.split(",") if p])


class capture_stderr(list):
    # [CLASS DOC] HELPER: List subclass used as a context manager that temporarily replaces
    # sys.stderr with an in-memory buffer; on exit the captured text is appended to the list
    # itself as one string and stderr is restored.
    """
    Replace sys.stderr with a temporary StringIO
    """

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Saves the current stderr, installs a fresh StringIO as stderr
        # and returns self.
        self.sys_stderr = sys.stderr
        self.stringio = StringIO()
        sys.stderr = self.stringio
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] HELPER: Appends the captured text to the list, deletes the buffer and
        # restores the saved stderr. Exceptions are not suppressed.
        self.append(str(self.stringio.getvalue()))
        del self.stringio
        sys.stderr = self.sys_stderr


def cdata(t):
    # [FUNCTION DOC] HELPER: Returns the raw C data pointer value of the untyped storage of the
    # tensor.
    return t.untyped_storage()._cdata


class TestCase(InductorTestCase):
    # [CLASS DOC] HELPER: Base test case for this module. Class setup enters a config patch
    # (debug on, minimal cpp chunk size, no pointwise autotune, no implicit fallbacks) that is
    # closed at class teardown; each test resets dynamo before and after.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class-level setup: calls the parent setup, creates an ExitStack
        # and enters the inductor config patch described on the class.
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(
            config.patch(
                {
                    "debug": True,
                    "cpp.min_chunk_size": 1,
                    "triton.autotune_pointwise": False,  # too slow
                    "implicit_fallbacks": False,
                }
            )
        )

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Class-level teardown: closes the ExitStack holding the config
        # patch, then calls the parent teardown.
        cls._stack.close()
        super().tearDownClass()

    def setUp(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and then runs the parent per-test setup.
        torch._dynamo.reset()
        super().setUp()

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the parent per-test teardown and then resets dynamo state.
        super().tearDown()
        torch._dynamo.reset()


class CUDAGraphAPIOnlyTests(TestCase):
    # [CLASS DOC] Tests that need no CUDA device: only checks the compiler API entry points
    # behave when no cudagraph tree manager exists.
    def test_mark_warmup_incomplete_without_cudagraphs(self):
        # [FUNCTION DOC] Verifies that marking warmup incomplete via the compiler API is
        # harmless when cudagraphs were never used: the set of device keys in the thread-local
        # tree manager containers is identical before and after the call.
        cudagraph_trees = torch._inductor.cudagraph_trees
        containers = cudagraph_trees.get_obj(
            cudagraph_trees.local, "tree_manager_containers"
        )
        existing_devices = tuple(containers)
        torch.compiler.cudagraph_mark_warmup_incomplete()
        self.assertEqual(tuple(containers), existing_devices)


if HAS_CUDA_AND_TRITON:

    def get_all_cudagraph_segments():
        # [FUNCTION DOC] HELPER: Returns the segments from the CUDA memory snapshot whose
        # segment pool id is not the default (0, 0) pair, i.e. segments belonging to CUDA graph
        # private pools.
        segments = torch.cuda.memory_snapshot()
        return [segment for segment in segments if segment["segment_pool_id"] != (0, 0)]

    def all_live_blocks():
        # [FUNCTION DOC] HELPER: Walks every cudagraph segment and its blocks, accumulating the
        # address of each block while stepping by block size, and returns the list of start
        # addresses of blocks that are in the active_allocated state.
        blocks_addrs = []
        for segment in get_all_cudagraph_segments():
            addr = segment["address"]
            for block in segment["blocks"]:
                if block["state"] == "active_allocated":
                    blocks_addrs.append(addr)
                addr += block["size"]

        return blocks_addrs

    def all_live_block_count():
        # [FUNCTION DOC] HELPER: Returns the number of live (active allocated) blocks in
        # cudagraph pools.
        return len(all_live_blocks())

    class CudaGraphTreeTests(TestCase):
        # [CLASS DOC] Main test suite for CUDA graph trees, run only when CUDA and Triton are
        # available. setUp enables the cudagraphs and cudagraph_trees inductor options with fast
        # and slow path asserts and automatic dynamic shapes; tearDown verifies no tree manager,
        # no live cudagraph blocks and no cudagraph segments remain. Provides helper methods for
        # manager access and running functions across generations, followed by the tests.
        def setUp(self):
            # [FUNCTION DOC] HELPER: Enters a config patch enabling cudagraph trees with all
            # internal asserts, enables automatic dynamic shapes, records the current CUDA
            # device index and silences all warnings.
            super().setUp()
            self.graph_stack = contextlib.ExitStack()
            self.graph_stack.enter_context(
                config.patch(
                    {
                        "triton.cudagraphs": True,
                        "triton.cudagraph_trees": True,
                        "triton.fast_path_cudagraph_asserts": True,  # too slow
                        "triton.slow_path_cudagraph_asserts": True,
                    }
                )
            )
            self.graph_stack.enter_context(
                dynamo_config.patch(automatic_dynamic_shapes=True)
            )
            self.device_idx = torch.rand([0], device="cuda").device.index
            warnings.filterwarnings("ignore")

        def tearDown(self):
            # [FUNCTION DOC] HELPER: Resets dynamo, collects garbage, empties the CUDA cache and
            # closes the config stack. It then asserts that the device has no tree manager, no
            # live cudagraph blocks and no cudagraph segments (leak check), and restores warning
            # filters.
            super().tearDown()
            torch._dynamo.reset()
            gc.collect()
            torch.cuda.empty_cache()
            self.graph_stack.close()

            self.assertIsNone(self.get_manager())
            self.assertEqual(all_live_block_count(), 0)
            self.assertEqual(len(get_all_cudagraph_segments()), 0)
            warnings.resetwarnings()

        def get_manager(self, device_index=None):
            # [FUNCTION DOC] HELPER: Returns the tree manager of the given device index, falling
            # back to the test device index when none is given (note a zero index also falls
            # back).
            return torch._inductor.cudagraph_trees.get_container(
                device_index if device_index else self.device_idx
            ).tree_manager

        def get_roots(self):
            # [FUNCTION DOC] HELPER: Returns the root nodes of the current tree manager.
            return self.get_manager().get_roots()

        def curr_node(self):
            # [FUNCTION DOC] HELPER: Returns the node the tree manager is currently positioned
            # at.
            return self.get_manager().current_node

        def get_root_children(self):
            # [FUNCTION DOC] HELPER: Returns, for each root of the tree, its descendant count.
            return [root.num_descendants() for root in self.get_roots()]

        def cudagraphify_impl(
            self, *args, is_inference=True, is_backward=False, **kwargs
        ):
            # [FUNCTION DOC] HELPER: Calls the cudagraphify implementation from cudagraph_trees
            # with the given arguments, forwarding the test device index and the is_inference
            # (default true) and is_backward (default false) flags.
            return tree_cudagraphify_impl(
                *args,
                **kwargs,
                device_index=self.device_idx,
                is_inference=is_inference,
                is_backward=is_backward,
            )

        @staticmethod
        def run_twc(fn, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Static helper that calls the function twice with the same
            # arguments (warmup then record) and returns the second result.
            fn(*args, **kwargs)
            return fn(*args, **kwargs)

        @staticmethod
        def run_next_generation(fn, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Static helper that marks the beginning of a new cudagraph
            # step via the compiler API, then calls the function and returns its result.
            torch.compiler.cudagraph_mark_step_begin()
            return fn(*args, **kwargs)

        def replay_next_generation(self, fn, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Runs the function in a new generation, asserts the tree
            # manager is in the EXECUTION path state (a recorded graph was replayed) and returns
            # the output.
            out = self.run_next_generation(fn, *args, **kwargs)
            self.assertEqual(self.get_manager().path_state, ExecutionState.EXECUTION)
            return out

        def _iter_tensor_outputs(self, outputs):
            # [FUNCTION DOC] HELPER: Generator that yields every tensor found in an output,
            # recursing into lists and tuples; None values are skipped and any other type raises
            # AssertionError.
            if isinstance(outputs, torch.Tensor):
                yield outputs
                return
            if isinstance(outputs, (list, tuple)):
                for out in outputs:
                    yield from self._iter_tensor_outputs(out)
                return
            if outputs is not None:
                raise AssertionError(f"expected tensor output, got {type(outputs)}")

        def save_live_outputs(self, live_outputs, *outputs):
            # [FUNCTION DOC] HELPER: Appends a pair of each tensor output and a detached clone
            # to the live_outputs list, then asserts every previously saved output still equals
            # its clone (checks that earlier outputs were not overwritten).
            for out in self._iter_tensor_outputs(outputs):
                live_outputs.append((out, out.detach().clone()))
            for out, expected in live_outputs:
                self.assertEqual(out, expected)

        def num_checkpoints(self):
            # [FUNCTION DOC] HELPER: Returns the debug checkpointing counter of the tree
            # manager.
            return self.get_manager().debug_checkpointing_counter

        def test_run_simple(self):
            # [FUNCTION DOC] Compiles a cube function and runs it twice each on two differently
            # shaped inputs; expects two roots with no descendants.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            zeros = torch.zeros([5, 5], device="cuda")
            self.run_twc(foo_opt, ones)
            self.run_twc(foo_opt, zeros)
            self.assertEqual(self.get_root_children(), [0, 0])

        def test_multithreaded_cudagraph_trees(self):
            # [FUNCTION DOC] Runs a compiled reduce-overhead function first in one owner thread,
            # then concurrently in three threads synchronized by a barrier, checking that each
            # thread's output matches eager and the manager exists. Worker exceptions are
            # collected as formatted tracebacks in a queue and cause a test failure.
            import queue
            import threading
            import traceback

            def foo(x, y):
                # [FUNCTION DOC] HELPER: Eager reference function: sine of the first input plus
                # cosine of the second.
                return torch.sin(x) + torch.cos(y)

            opt_foo = torch.compile(foo, mode="reduce-overhead")
            num_threads = 3
            barrier = threading.Barrier(num_threads)
            errors = queue.Queue()
            results = queue.Queue()

            def run_once(
                check_manager: bool, start_barrier: threading.Barrier | None = None
            ) -> None:
                # [FUNCTION DOC] HELPER: Thread body. Creates two random CUDA tensors, computes
                # the eager result, optionally waits on the barrier, runs the compiled function
                # twice, optionally checks that a tree manager exists for the device without
                # creating one, and puts the (result, expected) pair in the results queue. Any
                # exception is stored as traceback text in the errors queue.
                try:
                    x = torch.rand(4, 4, device="cuda")
                    y = torch.rand(4, 4, device="cuda")
                    expected = foo(x, y)
                    if start_barrier is not None:
                        start_barrier.wait()
                    opt_foo(x, y)
                    result = opt_foo(x, y)
                    if check_manager:
                        manager = torch._inductor.cudagraph_trees.get_manager(
                            x.device.index, create_if_none_exists=False
                        )
                        self.assertIsNotNone(manager)
                    results.put((result, expected))
                except Exception:
                    errors.put(traceback.format_exc())

            owner_thread = threading.Thread(target=run_once, args=(True,))
            owner_thread.start()
            owner_thread.join()

            def check_errors() -> None:
                # [FUNCTION DOC] HELPER: Fails the test with all queued error messages joined by
                # newlines, if the error queue is non-empty; drains the queue.
                if not errors.empty():
                    messages = []
                    while not errors.empty():
                        messages.append(errors.get())
                    self.fail("\n".join(messages))

            check_errors()
            result, expected = results.get()
            torch.testing.assert_close(result, expected)
            del result, expected

            def worker() -> None:
                # [FUNCTION DOC] HELPER: Thread entry point that runs the shared body with
                # manager checking and the shared barrier.
                run_once(True, barrier)

            threads = [threading.Thread(target=worker) for _ in range(num_threads)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            check_errors()
            while not results.empty():
                result, expected = results.get()
                torch.testing.assert_close(result, expected)
                del result, expected

        def check_rng(self):
            # [FUNCTION DOC] HELPER: Checks RNG reproducibility: compiles a function returning
            # random numbers with reduce-overhead, records three outputs after seeding, reseeds
            # and asserts three further calls reproduce the same three outputs in order.
            @torch.compile(mode="reduce-overhead")
            def foo():
                # [FUNCTION DOC] HELPER: Returns a random 20-element tensor on the default
                # device.
                return torch.rand([20])

            torch.manual_seed(0)

            out = foo()
            out2 = foo()
            out3 = foo()

            torch.manual_seed(0)

            self.assertEqual(out, foo())
            self.assertEqual(out2, foo())
            self.assertEqual(out3, foo())

        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_trees(self):
            # [FUNCTION DOC] Runs the RNG reproducibility check with fallback_random enabled and
            # cudagraph trees on.
            self.check_rng()

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        @torch._inductor.config.patch("fallback_random", True)
        def test_rng_non_trees(self):
            # [FUNCTION DOC] Same RNG reproducibility check as the trees variant but with
            # cudagraph trees disabled.
            self.check_rng()

        def test_mutation_reinplaced(self):
            # [FUNCTION DOC] Compiles a module whose forward calls logical_xor with an explicit
            # float16 out tensor (mutating an input) with reduce-overhead. Captures the
            # cudagraph_utils log and checks it reports skipping cudagraphs because of one
            # mutated input attributed to the logical_xor call, and that the cudagraph_skips
            # counter equals one.
            import torch.nn as nn

            class Model(nn.Module):
                # [CLASS DOC] HELPER: Minimal module whose forward applies logical xor of two
                # inputs writing into a caller supplied out tensor and returns the result.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Calls the parent module constructor; no parameters.
                    super().__init__()

                def forward(self, input, other, out):
                    # [FUNCTION DOC] HELPER: Runs logical xor of input and other into the out
                    # argument and returns it.
                    input = torch.logical_xor(input=input, other=other, out=out)
                    return input

            x = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            y = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float32).cuda()
            z = torch.rand([1, 2, 1, 4, 9, 7], dtype=torch.float16).cuda()

            model = Model().cuda()
            eag = model(x, y, z)
            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                opt = torch.compile(model.forward, mode="reduce-overhead")(x, y, z)

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("torch.logical_xor").run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @requires_multigpu()
        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_multiple_devices_msg(self, backend):
            # [FUNCTION DOC] Parametrized over the inductor and cudagraphs backends and run only
            # with multiple GPUs. Checks the skip reasons logged: a call mixing a CUDA and a CPU
            # tensor is skipped for a CPU device (or not skipped at all when graph partition is
            # on, since partitioning splits on CPU ops), and a call mixing cuda:0 and cuda:1
            # inputs is skipped for multiple devices. Also verifies the cumulative skip counter
            # values.
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns a tuple with the first input plus one and the
                # second plus two.
                return (x + 1, y + 2)

            foo = get_compile_fn(backend)(foo)
            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            if torch._inductor.config.graph_partition:
                # graph partition splits on cpu ops
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
            else:
                FileCheck().check(
                    "skipping cudagraphs due to cpu device (arg1_1). Found from"
                ).check("y + 2").run(log_stream.getvalue())
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                foo(
                    torch.ones([10], device="cuda:0"), torch.ones([10], device="cuda:1")
                )

            FileCheck().check("skipping cudagraphs due to multiple devices").run(
                log_stream.getvalue()
            )
            self.assertEqual(
                counters["inductor"]["cudagraph_skips"],
                1 if torch._inductor.config.graph_partition else 2,
            )

        @torch._inductor.config.patch("triton.cudagraph_skip_dynamic_graphs", True)
        def test_skip_symbolic(self):
            # [FUNCTION DOC] With the skip-dynamic-graphs option on, compiles an add function
            # with dynamic shapes and runs it. With graph partition the scheduler log must
            # mention a dynamic shape ops reason, otherwise the cudagraph_utils log must report
            # skipping due to symbolic shape inputs.
            @torch.compile(dynamic=True)
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

            scheduler_log_stream, scheduler_ctx = logs_to_string(
                "torch._inductor.scheduler", "cudagraphs"
            )
            utils_log_stream, utils_ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with scheduler_ctx(), utils_ctx():
                foo(torch.rand([10], device="cuda"), torch.rand([10], device="cuda"))

            if config.graph_partition:
                FileCheck().check("reason=dynamic shape ops").check("x + y").run(
                    scheduler_log_stream.getvalue()
                )
            else:
                FileCheck().check(
                    "skipping cudagraphs due to graph with symbolic shapes inputs"
                ).run(utils_log_stream.getvalue())

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_on_inp(self, backend):
            # [FUNCTION DOC] Parametrized by backend with input mutation support configured on.
            # Part one: a function that mutates its eager input in place is skipped (log message
            # with one mutated input, skip counter one) and no tree roots are created. Part two:
            # a module that mutates its own buffer is compiled three times and must match an
            # eager twin in both outputs and buffer state, and a tree manager must then exist.
            def foo(x):
                # [FUNCTION DOC] HELPER: Adds two to the input in place and returns the input.
                x.add_(2)
                return x

            foo = get_compile_fn(backend)(foo)

            def inp():
                # [FUNCTION DOC] HELPER: Returns a new ones tensor of ten elements on CUDA.
                return torch.ones([10], device="cuda")

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                foo(inp())

            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check(".add_(2)").run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

            # mutation on inp doesn't hit cudagraphs
            self.assertEqual(len(self.get_manager().roots), 0)

            # mutation on parameters/buffers hits cudagraphs
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module holding a ones CUDA buffer; forward adds the input
                # into the buffer in place and returns buffer plus input.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Initializes the module and creates the ten-element
                    # ones CUDA buffer attribute.
                    super().__init__()
                    self.buf = torch.ones([10], device="cuda")

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Mutates the buffer by adding x in place and returns
                    # the updated buffer plus x.
                    self.buf.add_(x)
                    return self.buf + x

            def foo(mod, x):
                # [FUNCTION DOC] HELPER: Calls the module on the input and returns the result
                # (redefinition used for the module case).
                return mod(x)

            foo = get_compile_fn(backend)(foo)
            mod = Mod()
            mod2 = Mod()

            for _ in range(3):
                self.assertEqual(foo(mod, inp()), mod2(inp()))
                self.assertEqual(mod.buf, mod2.buf)

            self.assertIsNotNone(self.get_manager())

        def test_input_storage_mutation_skips_cudagraphs(self):
            # [FUNCTION DOC] Compiles with reduce-overhead and dynamic shapes a module that
            # scales its parameter data in place by a mask during forward, under no-grad. Checks
            # outputs on two calls equal eager, the log reports skipping due to input storage
            # mutation, the skip counter is one and no tree manager is created.
            class Mod(nn.Module):
                # [CLASS DOC] HELPER: Module with a random parameter and a lower-triangular mask
                # buffer; forward multiplies the parameter data by the mask in place under
                # no-grad and returns the matrix product of the input and the parameter.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the random weight parameter and registers
                    # the lower-triangular mask buffer, both on CUDA.
                    super().__init__()
                    self.weight = nn.Parameter(torch.randn(4, 4, device="cuda"))
                    self.register_buffer(
                        "mask",
                        torch.tril(torch.ones(4, 4, device="cuda")),
                    )

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Masks the weight data in place without gradient
                    # tracking, then returns the matmul of x with the weight.
                    with torch.no_grad():
                        self.weight.data *= self.mask
                    return x @ self.weight

            torch.manual_seed(0)
            model = Mod().eval()
            x = torch.randn(4, 4, device="cuda")

            with torch.no_grad():
                expected = model(x)

            counters.clear()
            compiled_model = torch.compile(
                model,
                mode="reduce-overhead",
                dynamic=True,
            )
            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx(), torch.no_grad():
                actual = compiled_model(x)
                actual_again = compiled_model(x)

            self.assertEqual(expected, actual)
            self.assertEqual(expected, actual_again)
            FileCheck().check("skipping cudagraphs due to input storage mutation").run(
                log_stream.getvalue()
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            self.assertIsNone(self.get_manager())

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", False)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", False)
        def test_mutation_cudagraph_managed_tensors_config(self, backend):
            # [FUNCTION DOC] With input mutation support disabled in config, a compiled mutating
            # function applied to the output of another compiled function across three steps
            # must be reported exactly once as skipped for mutated inputs, and results must
            # match the non-mutating eager equivalent.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

            def mut(x):
                # [FUNCTION DOC] HELPER: Adds two to the input in place and returns it.
                x.add_(2)
                return x

            def non_mut(x):
                # [FUNCTION DOC] HELPER: Returns a new tensor of the input plus two without
                # mutation.
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(log_stream.getvalue())

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensors(self, backend):
            # [FUNCTION DOC] With input mutation support enabled, mutating a tensor that is
            # managed by cudagraphs (output of a compiled function) over three steps logs no
            # skip and leaves the skip counter absent. Then mutating a clone created in eager
            # code is not cudagraph managed, so it must log one skip for mutated inputs and the
            # counter becomes one; values must match the non-mutating equivalent.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

            def mut(x):
                # [FUNCTION DOC] HELPER: Adds two to the input in place and returns it.
                x.add_(2)
                return x

            def non_mut(x):
                # [FUNCTION DOC] HELPER: Returns a new tensor of the input plus two without
                # mutation.
                return x.add(2)

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    inp = torch.rand([4], device="cuda")

                    tmp = foo(inp)
                    mut_out = mut(tmp)
                    self.assertEqual(mut_out, non_mut(foo(inp)))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                0,
                exactly=True,
            ).run(log_stream.getvalue())
            self.assertTrue("cudagraph_skips" not in counters["inductor"])

            torch.compiler.cudagraph_mark_step_begin()
            inp = torch.rand([4], device="cuda")
            tmp = foo(inp)
            mut_inp = tmp.clone()
            # in this case, what previously a mutated cudagraph managed tensor is no longer,
            # now its an input from eager we should fallback to inductor without cudagraphs
            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                mut(mut_inp)
            FileCheck().check(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from"
            ).check("x.add_(2)").run(log_stream.getvalue())
            self.assertEqual(mut_inp, non_mut(foo(inp)))
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn(self, backend):
            # [FUNCTION DOC] Checks that a function mutating its input via in-place add produces
            # exactly one skip warning over three steps when its result is fed into another
            # compiled function, and that the skip counter is one.
            def foo(x):
                # [FUNCTION DOC] HELPER: Adds one to the input in place and returns it.
                return x.add_(1)

            def fee(y, z):
                # [FUNCTION DOC] HELPER: Ignores the first argument and returns the second plus
                # three.
                return z.add(3)

            def inp():
                # [FUNCTION DOC] HELPER: Returns a random four-element CUDA tensor.
                return torch.rand([4], device="cuda")

            foo = get_compile_fn(backend)(foo)
            fee = get_compile_fn(backend)(fee)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    fee(inp(), foo(inp()))
            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        @torch._dynamo.config.patch("cudagraph_backend_keep_input_mutation", True)
        @torch._dynamo.config.patch("cudagraph_backend_support_input_mutation", True)
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_mutation_cudagraph_managed_tensor_warn_only_once(self, backend):
            # [FUNCTION DOC] Checks the mutation skip warning is emitted only once per function:
            # one call on an eager input warns, and later calls on cudagraph managed tensors or
            # on a clone do not warn again. The log must contain exactly one skip message and
            # the counter must be one.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return x + 1

            def mut(x):
                # [FUNCTION DOC] HELPER: Adds two to the input in place and returns it.
                x.add_(2)
                return x

            def inp():
                # [FUNCTION DOC] HELPER: Returns a random four-element CUDA tensor.
                return torch.rand([4], device="cuda")

            mut = get_compile_fn(backend)(mut)
            foo = get_compile_fn(backend)(foo)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                # Should warn for current_node=None
                mut(inp())

                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    tmp = foo(inp())
                    mut(tmp)  # should not warn

                mut_inp = tmp.clone()
                mut(mut_inp)  # should not warn since mut has warned

            FileCheck().check_count(
                "skipping cudagraphs due to mutated inputs (1 instances). Found from",
                1,
                exactly=True,
            ).run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        def test_index_put(self):
            # [FUNCTION DOC] Compiles with reduce-overhead a function that zeroes its input and
            # does an accumulating boolean-valued index_put_. On the first of three iterations
            # the generated code is checked to contain the index_put_ op with a True argument;
            # every iteration compares against eager.
            def fn(x, y, z):
                # [FUNCTION DOC] HELPER: Replaces x with zeros of the same shape, then performs
                # an accumulating index_put_ at index y with values z and returns the result.
                x = torch.zeros_like(x)
                return x.index_put_([y], z, True)

            fn_c = torch.compile(mode="reduce-overhead")(fn)

            for i in range(3):

                def args():
                    # [FUNCTION DOC] HELPER: Returns a fresh triple of CUDA tensors: a 512 by
                    # 512 boolean zeros, a 512 int64 arange and a 512 by 512 boolean ones.
                    x = torch.zeros((512, 512), dtype=torch.bool, device="cuda")
                    y = torch.arange(512, dtype=torch.int64, device="cuda")
                    z = torch.ones((512, 512), dtype=torch.bool, device="cuda")
                    return x, y, z

                if i == 0:
                    out, code = run_and_get_code(fn_c, *args())
                    FileCheck().check("aten.index_put_").check_same("True").run(code[0])
                else:
                    out = fn_c(*args())

                self.assertEqual(fn(*args()), out)

        def test_index_put_boolean_fallback_partitions_cudagraphs(self):
            # [FUNCTION DOC] Regression test for GitHub issue 169951. With graph partition on, a
            # compiled function using index_put_ with a boolean mask must produce at least one
            # partition, match eager over three runs, and not skip cudagraphs. Without
            # partitioning the log must show cudagraphs were skipped.
            """
            Test that index_put_ with boolean indices (which uses fallback path)
            properly partitions around the unsafe op. This tests the fix for
            GitHub issue #169951.
            """

            def fn(x, mask, values):
                # [FUNCTION DOC] HELPER: Clones x and adds two, applies index_put_ with a
                # boolean mask and values in place on the copy, and returns the copy.
                result = x.clone() + 2
                result.index_put_([mask], values)
                return result

            fn_c = torch.compile(mode="reduce-overhead")(fn)
            x = torch.randn(4, 8, device="cuda")
            mask = torch.tensor([True, False, True, False], device="cuda")
            values = torch.randn(2, 8, device="cuda")

            if config.graph_partition:
                # Check that we partition (unsafe op runs inline, cudagraphable parts are partitioned)
                _, code = run_and_get_code(fn_c, x, mask, values)
                self.assertGreaterEqual(get_num_partitions(code), 1)

                # Verify correctness
                for _ in range(3):
                    x = torch.randn(4, 8, device="cuda")
                    mask = torch.tensor([True, False, True, False], device="cuda")
                    values = torch.randn(2, 8, device="cuda")
                    out = fn_c(x, mask, values)
                    expected = fn(x, mask, values)
                    self.assertEqual(out, expected)

                # Should not skip cudagraphs entirely
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
            else:
                # Without partitioning, cudagraphs are skipped due to the unsafe op
                log_stream, ctx = logs_to_string(
                    "torch._inductor.cudagraph_utils", "cudagraphs"
                )
                with ctx():
                    fn_c(x, mask, values)
                FileCheck().check("skipping cudagraphs due to").run(
                    log_stream.getvalue()
                )
                self.assertGreater(counters["inductor"]["cudagraph_skips"], 0)

        @torch._inductor.config.patch("graph_partition", True)
        def test_index_put_boolean_mutation_across_partitions(self):
            # [FUNCTION DOC] With graph partition forced on, compiles a function that creates a
            # buffer, mutates it with a boolean-mask index_put_ (forcing a partition split) and
            # then uses the mutated buffer in a second partition. Asserts at least two
            # partitions are generated and that compiled results equal eager over three runs
            # with fresh inputs (cudagraph replay).
            """
            Test that index_put_ with boolean indices correctly passes the mutated
            buffer to subsequent partitions. This tests the fix for mutation buffers
            with NoneLayout (like IndexPutFallback) being correctly tracked across
            partition boundaries.

            The key scenario is:
            1. A buffer is created in partition 0
            2. index_put_ (with boolean mask) mutates it - this creates a partition split
            3. The mutated buffer is used in partition 1
            """

            def fn(x, mask, values):
                # [FUNCTION DOC] HELPER: Clones x and adds two, mutates the copy with a masked
                # index_put_, and returns twice the copy plus the sum of the copy.
                # Create a buffer in partition 0
                result = x.clone() + 2
                # index_put_ with boolean mask causes partition split
                result.index_put_([mask], values)
                # Use the mutated buffer in partition 1
                return result * 2 + result.sum()

            fn_c = torch.compile(mode="reduce-overhead")(fn)
            x = torch.randn(4, 8, device="cuda")
            mask = torch.tensor([True, False, True, False], device="cuda")
            values = torch.randn(2, 8, device="cuda")

            # Verify we get multiple partitions
            _, code = run_and_get_code(fn_c, x, mask, values)
            self.assertGreaterEqual(get_num_partitions(code), 2)

            # Verify correctness across multiple runs (tests cudagraph replay)
            for _ in range(3):
                x = torch.randn(4, 8, device="cuda")
                mask = torch.tensor([True, False, True, False], device="cuda")
                values = torch.randn(2, 8, device="cuda")
                out = fn_c(x, mask, values)
                expected = fn(x, mask, values)
                self.assertEqual(out, expected)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_mutation_with_cudagraph_unsafe(self):
            # [FUNCTION DOC] With implicit fallbacks and graph partition on, defines two custom
            # ops (one mutating its argument and returning two tensors, one tagged cudagraph
            # unsafe that sums four tensors) and compiles a function chaining them. Asserts at
            # least two partitions and that compiled output equals eager over five runs,
            # checking mutated buffers cross partition boundaries correctly.
            """
            Test that custom ops with mutations work correctly with cudagraph-unsafe
            ops that cause partition splits. This tests that mutated buffers from
            custom ops are correctly passed across partition boundaries.
            """

            @torch.library.custom_op(
                "mylib::mutating_op",
                mutates_args=["x"],
                schema="(Tensor(a!)? x) -> (Tensor, Tensor)",
                device_types="cuda",
            )
            def mutating_op(x) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Custom op implementation (CUDA only) that adds one to x
                # in place and returns x plus one and x plus two.
                x.add_(1)
                return (x + 1, x + 2)

            @mutating_op.register_fake
            def _(x) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Fake (meta) implementation of the mutating op returning
                # two empty tensors shaped like the input.
                return (torch.empty_like(x), torch.empty_like(x))

            @torch.library.custom_op(
                "mylib::cudagraph_unsafe_combine",
                mutates_args=[],
                schema="(Tensor x, Tensor y, Tensor x1, Tensor y1) -> Tensor",
                device_types="cuda",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def cudagraph_unsafe_combine(x0, x1, y0, y1) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation of the cudagraph unsafe op:
                # returns the elementwise sum of its four tensor arguments.
                return x0 + x1 + y0 + y1

            @cudagraph_unsafe_combine.register_fake
            def _(x0, x1, y0, y1) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor shaped
                # like the first argument.
                return torch.empty_like(x0)

            def f(x):
                # [FUNCTION DOC] HELPER: Test function: adds one, applies the mutating op,
                # derives y0 and y1 from its two outputs, combines them with the unsafe op,
                # forms z from sums plus one in place, and returns y plus z.
                x = x + 1
                x = mutating_op(x)
                x0, x1 = x[0], x[1]
                y0 = x0 + 1
                y1 = x1 + 1
                y = cudagraph_unsafe_combine(x0, x1, y0, y1)
                z = y + x0 + x1
                z += 1
                return y + z

            f_compiled = torch.compile(f, mode="reduce-overhead")

            x = torch.randn((2, 4, 8), device="cuda")

            # Verify we get multiple partitions
            _, code = run_and_get_code(f_compiled, x.clone())
            self.assertGreaterEqual(get_num_partitions(code), 2)

            # Verify correctness across multiple runs (tests cudagraph replay)
            for _ in range(5):
                x = torch.randn((2, 4, 8), device="cuda")
                x_cloned = x.clone()
                eager_out = f(x)
                compiled_out = f_compiled(x_cloned)
                self.assertEqual(eager_out, compiled_out)

        def test_function_compiled_multiple_times(self):
            # [FUNCTION DOC] Runs a function that calls a graph-breaking helper twice, compiled,
            # three times and checks it matches eager; the cudagraph tree must have exactly one
            # root with one descendant.
            def foo(x):
                # [FUNCTION DOC] HELPER: Calls the graph-breaking helper on x, then again on its
                # result, and returns the sum of both results.
                y = foo2(x)
                y2 = foo2(y)
                return y + y2

            def foo2(x):
                # [FUNCTION DOC] HELPER: Triggers a dynamo graph break and returns the cube of
                # x.
                torch._dynamo.graph_break()
                return x * x * x

            foo_opt = torch.compile(foo)
            ones = torch.ones([4, 4], device="cuda")
            foo(ones)
            foo_opt(ones)
            foo_opt(ones)
            self.assertEqual(foo_opt(ones), foo(ones))
            # paths
            children = self.get_root_children()
            self.assertEqual(children, [1])

        def test_end_recording_early(self):
            # [FUNCTION DOC] Simulates two separate recordings by decrementing the dynamo
            # generation counter between calls, over three iterations, using a function with a
            # graph break and a small second compiled function. Expects two roots, the first
            # with one descendant and the second with none.
            def foo(x):
                # [FUNCTION DOC] HELPER: Computes the cube, causes a graph break, then returns x
                # plus that cube.
                y = x * x * x
                torch._dynamo.graph_break()
                z = x + y
                return z

            @torch.compile
            def foo2(x):
                # [FUNCTION DOC] HELPER: Returns the input plus four.
                return x + 4

            foo_opt = torch.compile(foo)

            for _ in range(3):
                out = foo_opt(torch.ones([4, 4], device="cuda"))
                del out

                # when I tried inducing separate recordings via graph break,
                # the frame kept interfering by keeping outputs alive
                # this isn't great by simulates the logic.
                from torch._dynamo.mutation_guard import GenerationTracker

                GenerationTracker.generation -= 1

                out = foo2(torch.ones([4, 4], device="cuda"))
                del out

            foo_opt(torch.ones([4, 4], device="cuda"))

            # Two separate traces - one has a child, one doesn't
            self.assertEqual(self.get_root_children(), [1, 0])

        def test_execution_into_recording(self):
            # [FUNCTION DOC] Uses a data dependent branch on the sign of the sum to move through
            # warmup, recording and execution; checks live cudagraph block counts at each phase
            # (one during warmup and recording, zero after outputs are freed) and that results
            # match eager.
            def foo(x):
                # [FUNCTION DOC] HELPER: Doubles x and adds ten to the result if its sum is
                # positive, otherwise subtracts ten.
                y = x + x

                if y.sum() > 0:
                    return y + 10
                else:
                    return y - 10

            foo_opt = torch.compile(foo)
            inp = torch.zeros([4, 4], dtype=torch.float, device="cuda")
            self.assertEqual(foo_opt(inp), foo(inp))
            self.assertEqual(foo_opt(inp), foo(inp))

            inp.add_(1)
            out_eager = foo(inp)
            out_warmup = foo_opt(inp)
            self.assertEqual(out_warmup, out_eager)
            # warmup should be have storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            out_live = foo_opt(inp)
            self.assertEqual(out_live, out_eager)

            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)
            # warmup should have been freed
            del out_warmup
            # should be in recording mode, with storage deallocator hooked on
            self.assertEqual(all_live_block_count(), 1)

            del out_live
            self.assertEqual(all_live_block_count(), 0)

            out = foo_opt(inp)
            self.assertEqual(foo(inp), out)

            # should be in execution mode
            self.assertEqual(all_live_block_count(), 0)

        def test_forward_with_skipped_cudagraphed_backward(self):
            # [FUNCTION DOC] Runs forward and backward of a compiled cube three times with the
            # backward compilation cudagraph disabled by a testing-only config. Checks the next
            # graph id is one (only the forward was recorded) and no forwards are left running
            # with pending backwards.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of the input.
                return x * x * x

            for _ in range(3):
                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)

                with config.patch(force_disable_cudagraph_TESTING_ONLY=True):
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cache_hit_forward_miss_backward(self):
            # [FUNCTION DOC] With autograd and fx graph caches on, loops three times: clears
            # caches, compiles forward with cudagraph disabled (expects one cache miss), resets
            # dynamo and recompiles (expects a hit), then runs backward (expects a second miss).
            # After the loop checks autograd cache misses and saves, reruns end to end for an
            # autograd cache hit and asserts no cudagraph tree manager exists.
            # Test that we don't cache cudagraphs, skipping cudagraphs on backward on a cache miss

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of the input.
                return x * x * x

            # Run forwards, fx graph should cache miss
            for _ in range(3):
                torch._dynamo.reset()
                counters.clear()
                FxGraphCache.clear()
                AOTAutogradCache.clear()

                with config.patch(force_disable_cudagraph_TESTING_ONLY=True):
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

                    # Reset dynamo and related caches except for FXGraphCache
                    torch._dynamo.reset()
                    # Forwards should be a cache hit now, we still skip cudagraphs
                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                    self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)

                # Run the backward without the force-disable knob; cache should
                # miss for the backward, but cudagraphs should not run because
                # the forward already skipped.
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run it one more time, this time AOTAutogradCache will hit
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

            torch._dynamo.reset()
            inp = torch.rand([20, 20], device="cuda", requires_grad=True)
            out = foo(inp)
            back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
            out.backward(back_inp)

            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

            # we should not have cudagraph'd anything
            if self.get_manager() is not None:
                raise AssertionError

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        @requires_multigpu()
        def test_cached_boxed_forward_device_index(self):
            # [FUNCTION DOC] Multi-GPU test: on device 1, compiles forward and backward (cache
            # misses and saves counted), then three times resets dynamo and reruns, expecting an
            # autograd cache hit each time. Finally the tree manager must exist for device 1 and
            # not for device 0, showing the cached boxed forward keeps device index 1.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of the input.
                return x * x * x

            # Run with device index 1 so that we can see
            # on a cache hit we stay on device index 1
            with torch.cuda._DeviceGuard(1):
                torch.cuda.set_device(1)

                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                # Compile the backward and save to cache
                back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                out.backward(back_inp)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)
                self.assertEqual(counters["aot_autograd"]["autograd_cache_saved"], 1)

                # Reset dynamo and rerun a few times
                for i in range(3):
                    torch._dynamo.reset()

                    inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                    out = foo(inp)
                    # Should cache hit each time; boxed_forward_device_index should still be set properly to 1
                    self.assertEqual(
                        counters["aot_autograd"]["autograd_cache_hit"], i + 1
                    )
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # After everything, we should have cudagraphs on device 1
            self.assertTrue(self.get_manager(device_index=0) is None)
            self.assertFalse(self.get_manager(device_index=1) is None)

        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_backward_gets_cached_cudagraphs(self):
            # [FUNCTION DOC] Compiles forward and backward with CPU tensors (so cudagraphs are
            # disabled), populating caches (misses counted), then after a dynamo reset reruns
            # and expects two fx graph cache hits and one autograd cache hit. No cudagraph tree
            # manager may exist at the end.
            # We pass cpu tensors to foo and save that into the cache
            # On a subsequent run in a new process, cudagraphs should be
            # disabled properly on both forward and backwards runs.

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of the input.
                return x * x * x

            torch._dynamo.reset()
            counters.clear()
            FxGraphCache.clear()
            AOTAutogradCache.clear()

            # Use cpu device to disable cudagraphs during compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)
            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)

            # Run again on new process
            torch._dynamo.reset()

            # Forward and backward should also disable cudagraphs without compilation
            inp = torch.rand([20, 20], device="cpu", requires_grad=True)
            out = foo(inp)
            # AOTAutogradCache will load the forward and the backward from cache immediately, so fx_graph_cache_hit will equal 2
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 2)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)
            torch._dynamo.reset()

            back_inp = torch.empty_strided([20, 20], [0, 1], device="cpu")
            out.backward(back_inp)

            # we should not have cudagraph'd anything
            if self.get_manager() is not None:
                raise AssertionError

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        @torch._functorch.config.patch("enable_autograd_cache", True)
        @torch._inductor.config.patch("fx_graph_cache", True)
        @torch._inductor.config.patch("fx_graph_remote_cache", False)
        # Currently fx graph cache is turned off for specialize_float=False
        @torch._dynamo.config.patch("specialize_float", True)
        def test_cached_forward_backward(self):
            # [FUNCTION DOC] With caches cleared and warmup skipped, compiles a function with
            # seeding, dropout and sine, runs forward and backward and checks tree structure
            # (one root child, expected dead indices before and after the graph differ depending
            # on graph partition mode, positive graph id, autograd cache miss). After a dynamo
            # reset, reruns with a cloned input, expecting equal output and gradients and an
            # autograd cache hit with the same tree shape.
            counters.clear()
            AOTAutogradCache.clear()
            FxGraphCache.clear()

            @torch.compile
            def foo(x):
                # [FUNCTION DOC] HELPER: Seeds the RNG to zero, doubles x and returns the sine
                # of that times dropout of x with probability 0.4.
                torch.manual_seed(0)
                y = x * 2
                return torch.sin(y) * torch.nn.functional.dropout(x, p=0.4)

            inp = torch.rand([4, 4], requires_grad=True, device="cuda")
            inp2 = inp.detach().clone().requires_grad_(True)
            out = foo(inp)

            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            if torch._inductor.config.graph_partition:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 0), (0, 2)],
                )
            else:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 1), (0, 2)],
                )
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_miss"], 1)

            # Reset dynamo and rerun. We should see a cache hit now
            torch._dynamo.reset()

            out2 = foo(inp2)
            out2.sum().backward()
            self.assertEqual(out, out2)
            self.assertEqual(inp.grad, inp2.grad)

            self.assertEqual(self.get_root_children(), [1])
            self.assertFalse(self.get_manager().new_graph_id().id == 0)
            self.assertEqual(counters["aot_autograd"]["autograd_cache_hit"], 1)

        @parametrize("backend", ("inductor", "cudagraphs"))
        def test_forward_backward_not_called(self, backend):
            # [FUNCTION DOC] Parametrized by backend. Calls a compiled two-output function with
            # a graph break three times, and only runs backward through the first output.
            # Expects no forwards pending backwards and the next graph id to be three (the
            # second output backward not recorded).
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns the cubes of x and y with a graph break between
                # them.
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

            foo = get_compile_fn(backend)(foo)

            for _ in range(3):
                inps = [
                    torch.rand([20, 20], requires_grad=True, device="cuda")
                    for _ in range(2)
                ]
                x_out, y_out = foo(inps[0], inps[1])
                x_out.sum().backward()

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            # we should not have cudagraph'd the y backward
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 3)

        def _test_unaligned_static_input_impl(self, expected_clones):
            # [FUNCTION DOC] HELPER: Shared body for the unaligned static input tests. Traces an
            # add of two inputs with make_fx, compiles it with compile_fx_inner (cudagraphs on,
            # input index zero static) using fake inputs that are views offset by one row, then
            # three times runs the compiled function on unaligned and aligned real inputs under
            # a dispatch mode that counts aten clone calls, asserting the count equals
            # expected_clones after each call.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns a one-element tuple holding the sum of x and y.
                return (x + y,)

            def get_aligned_inputs():
                # [FUNCTION DOC] HELPER: Returns two random 5 by 5 CUDA tensors in a list.
                return [torch.rand([5, 5], device="cuda") for _ in range(2)]

            mod = make_fx(fn)(*get_aligned_inputs())

            mode = torch._subclasses.FakeTensorMode()

            with mode:
                inps = [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            compiled_f = compile_fx_inner(
                mod, inps, static_input_idxs=[0], cudagraphs=True
            )

            def get_unaligned_inputs():
                # [FUNCTION DOC] HELPER: Returns two random CUDA tensors of shape 5 by 5 that
                # are views starting at row one of larger 6 by 5 tensors (data pointer not
                # aligned like a fresh allocation).
                return [torch.rand([6, 5], device="cuda")[1:] for _ in range(2)]

            class CloneCounterMode(TorchDispatchMode):
                # [CLASS DOC] HELPER: Torch dispatch mode that counts how many dispatched calls
                # are the default overload of aten clone; every call is forwarded unchanged to
                # the real op. State: the count attribute.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Initializes the clone count to zero.
                    self.count = 0

                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    # [FUNCTION DOC] HELPER: Increments the count when the intercepted op is the
                    # default aten clone overload (treating missing kwargs as empty) and returns
                    # the result of running the op with the given arguments.
                    kwargs = {} if kwargs is None else kwargs
                    self.count += func is torch.ops.aten.clone.default
                    return func(*args, **kwargs)

            for _ in range(3):
                with CloneCounterMode() as m:
                    compiled_f(get_unaligned_inputs())
                    self.assertEqual(m.count, expected_clones)

                    compiled_f(get_aligned_inputs())
                    self.assertEqual(m.count, expected_clones)

        def test_unaligned_static_input_trees(self):
            # [FUNCTION DOC] Runs the unaligned static input check with cudagraph trees enabled,
            # expecting zero clone calls.
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_unaligned_static_input_non_trees(self):
            # [FUNCTION DOC] Runs the unaligned static input check with cudagraph trees
            # disabled, expecting zero clone calls.
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_unaligned_static_input_no_cudagraphs(self):
            # [FUNCTION DOC] Runs the unaligned static input check with cudagraphs disabled
            # entirely, expecting zero clone calls.
            self._test_unaligned_static_input_impl(expected_clones=0)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_graph_partition_custom_rule(self):
            # [FUNCTION DOC] Tests the custom_should_partition_ops config. A custom op
            # sandwiched between adds gives one partition by default; after listing a second
            # custom op in the config, a fresh compile yields two partitions. Changing the
            # config afterwards does not force a recompile (checked under a fail-on-recompile
            # stance), but a forced recompile yields one partition again, and listing the op by
            # its overload name also gives two. Mutates a global inductor config value that is
            # not restored by the test.
            @torch.library.custom_op("mylib::bar", mutates_args=())
            def bar(x: torch.Tensor, flag: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation returning a clone of x; flag
                # is ignored.
                return x.clone()

            @bar.register_fake
            def _(x, flag):
                # [FUNCTION DOC] HELPER: Fake implementation returning a clone of x.
                return x.clone()

            def f(x, flag):
                # [FUNCTION DOC] HELPER: Adds one, applies the custom op with the flag, adds one
                # again and returns the result.
                x = x + 1
                x = bar(x, flag)
                x = x + 1
                return x

            x = torch.randn(2, device="cuda")
            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x, True)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 1)

            @torch.library.custom_op("mylib::baz", mutates_args=())
            def baz(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Second custom op implementation returning a clone of x.
                return x.clone()

            @baz.register_fake
            def _(x):
                # [FUNCTION DOC] HELPER: Fake implementation returning a clone of x.
                return x.clone()

            # custom_should_partition_ops takes effect which lead to 2 partitions
            torch._inductor.config.custom_should_partition_ops = ["mylib::baz"]

            def f(x):
                # [FUNCTION DOC] HELPER: Adds one, applies the second custom op, adds one again
                # and returns the result (redefinition of the test function without the flag).
                x = x + 1
                x = baz(x)
                x = x + 1
                return x

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 2)

            # update the config should NOT force recompile
            torch._inductor.config.custom_should_partition_ops = []
            with torch.compiler.set_stance("fail_on_recompile"):
                f_compiled(x)

            # run_and_get_code forces recompile. Now we should cache miss, recompile, and
            # only have 1 partition.
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 1)

            # test that op_overload name takes effect which lead to 2 partitions
            torch._inductor.config.custom_should_partition_ops = ["mylib::baz.default"]

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions = get_num_partitions(code)
            self.assertEqual(num_partitions, 2)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("implicit_fallbacks", True)
        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_cudagraph_unsafe_unbacked_ops(self):
            # [FUNCTION DOC] With graph partition on, compiles a function with a custom op
            # returning an unbacked integer size. Initially two partitions exist; after setting
            # the cudagraph_unsafe_unbacked_ops config to name the op, either by qualified name
            # or with the default overload suffix, one partition is expected. Mutates a global
            # inductor config value that is not restored by the test.
            @torch.library.custom_op("mylib::get_size", mutates_args=())
            def get_size(x: torch.Tensor) -> int:
                # [FUNCTION DOC] HELPER: Custom op implementation returning half of the first
                # dimension of x as an integer.
                return x.shape[0] // 2

            @get_size.register_fake
            def _(x):
                # [FUNCTION DOC] HELPER: Fake implementation returning a new dynamic size
                # bounded between zero and the first dimension of x.
                ctx = torch.library.get_ctx()
                return ctx.new_dynamic_size(min=0, max=x.shape[0])

            x = torch.randn(8, 8, device="cuda")
            W1 = torch.randn(8, 8, device="cuda")
            W2 = torch.randn(8, 8, device="cuda")

            def f(q):
                # [FUNCTION DOC] HELPER: Adds one, round-trips the tensor through CPU, gets an
                # unbacked size from the custom op, asserts bounds on it, splits the tensor into
                # decode and prefill parts and returns the prefill result of two matmuls and the
                # decode result of one matmul.
                # input q is on gpu. Here, we have 1 cuda addition op followed by cpu <> gpu device copy
                # to enforce 1 graph partition. `torch._inductor.config.cudagraph_unsafe_unbacked_ops`
                # will decide whether we have 1 more graph partition (i.e., 1 graph partition
                # or 2 graph partitions in total).
                q = q + 1
                q_cpu = q.cpu()
                q = q_cpu.cuda()

                num_decode = torch.ops.mylib.get_size(q)  # Returns unbacked SymInt

                torch._check(num_decode >= 0)
                torch._check(num_decode <= q.shape[0])

                num_prefill = q.shape[0] - num_decode

                decode_q = q.narrow(0, 0, num_decode)
                prefill_q = q.narrow(0, num_decode, num_prefill)

                prefill_out = prefill_q @ W1
                prefill_out2 = prefill_out @ W2

                decode_out = decode_q @ W1

                return prefill_out2, decode_out

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions_before = get_num_partitions(code)
            # 2 partition since ops using unbacked symints are kept in graph partitions
            self.assertEqual(num_partitions_before, 2)

            # With config, ops using the unbacked symint should be partitioned out
            torch._inductor.config.cudagraph_unsafe_unbacked_ops = ["mylib::get_size"]

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions_after = get_num_partitions(code)
            # 1 partition since ops using unbacked symints are excluded from graph partitions
            self.assertEqual(num_partitions_after, 1)

            # Test with op_overload name (with .default suffix)
            torch._inductor.config.cudagraph_unsafe_unbacked_ops = [
                "mylib::get_size.default"
            ]

            f_compiled = torch.compile(f, mode="reduce-overhead", fullgraph=True)
            _, code = run_and_get_code(f_compiled, x)
            num_partitions_overload = get_num_partitions(code)
            # 1 partition since ops using unbacked symints are excluded from graph partitions
            self.assertEqual(num_partitions_overload, 1)

        # Pin graph_partition=False so the "disabling cudagraphs due to
        # incompatible op ..." log line is actually emitted. With
        # graph_partition=True (OSS default), incompatible ops are handled
        # by partitioning rather than disabling cudagraphs, and the FileCheck
        # pattern below never appears.
        @torch._inductor.config.patch("graph_partition", False)
        def test_topk_skips_cudagraph_on_hip(self):
            # [FUNCTION DOC] Runs a compiled topk function three times with graph partition off.
            # When running on ROCm (HIP), checks the cudagraph_utils log reports skipping
            # cudagraphs because of the incompatible topk op; on CUDA only checks it runs.
            # rocm workaround: topk kernels fault under cudagraph trees
            # on the 2nd replay of a recorded graph.
            BEAM_SIZE = 64
            VOCAB_SIZE = 512

            @torch.no_grad()
            def generate(h0):
                # [FUNCTION DOC] HELPER: Under no-grad, returns the top-k scores along the last
                # dimension with the beam size as k.
                scores, _ = torch.topk(h0, k=BEAM_SIZE, dim=-1)
                return scores

            compiled = torch.compile(generate, fullgraph=True, mode="reduce-overhead")

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    h0 = torch.randn(
                        8,
                        BEAM_SIZE * VOCAB_SIZE,
                        device="cuda",
                        dtype=torch.bfloat16,
                    )
                    compiled(h0)
                    torch.cuda.synchronize()

            if torch.version.hip is not None:
                FileCheck().check(
                    "skipping cudagraphs due to disabling cudagraphs due to "
                    "incompatible op aten.topk.default"
                ).run(log_stream.getvalue())

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_graph_partition_with_memory_plan_reuse(self):
            # [FUNCTION DOC] Builds a two-stage model (each stage an attention-like module with
            # a cudagraph unsafe in-place custom op) compiled in reduce-overhead mode with graph
            # partition and memory plan reuse, and checks compiled output equals eager over
            # three runs on a static hidden-state buffer.
            BATCH_SIZE = 16
            MLP_SIZE = 128
            HIDDEN_SIZE = 128
            RANDOM_SEED = 0

            @torch.library.custom_op(
                "silly::attention",
                mutates_args=["out"],
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def attention(
                q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
            ) -> None:
                # [FUNCTION DOC] HELPER: Custom op implementation (marked cudagraph unsafe) that
                # writes the sum of q, k and v into out and returns nothing.
                out.copy_(q + k + v)

            @attention.register_fake
            def _(q, k, v, out):
                # [FUNCTION DOC] HELPER: Fake implementation returning None.
                return None

            class ParentModel(torch.nn.Module):
                # [CLASS DOC] HELPER: Base module whose forward returns its input unchanged;
                # parent of the two-graph model.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Calls the module constructor.
                    super().__init__()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Returns x unchanged.
                    return x

            class Attention(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with two bias-free linear layers and an RMS norm
                # weight parameter; forward applies the first linear, a norm, the unsafe
                # attention custom op into a fresh output buffer, another norm and the second
                # linear.
                def __init__(self, mlp_size: int, hidden_size: int) -> None:
                    # [FUNCTION DOC] HELPER: Creates the pre and post linear layers and the ones
                    # initialized norm weight parameter.
                    super().__init__()
                    self.pre_attn = torch.nn.Linear(mlp_size, hidden_size, bias=False)
                    self.post_attn = torch.nn.Linear(hidden_size, mlp_size, bias=False)
                    self.rms_norm_weight = torch.nn.Parameter(torch.ones(hidden_size))

                def rms_norm_ref(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Reference RMS norm: converts to float32, multiplies
                    # by the reciprocal square root of the mean square over the last dimension
                    # plus a small epsilon, scales by the weight and casts back to the input
                    # dtype.
                    x_f32 = x.float()
                    return (
                        x_f32
                        * torch.rsqrt(
                            torch.mean(x_f32.square(), dim=-1, keepdim=True) + 1e-6
                        )
                        * self.rms_norm_weight
                    ).to(x.dtype)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Runs linear, norm, the attention custom op writing
                    # into an empty buffer, norm, then linear, and returns the result.
                    x = self.pre_attn(x)
                    x = self.rms_norm_ref(x)
                    attn_output = torch.empty_like(x)
                    torch.ops.silly.attention(x, x, x, attn_output)
                    x = attn_output
                    x = self.rms_norm_ref(x)
                    x = self.post_attn(x)
                    return x

            class CompiledAttention(torch.nn.Module):
                # [CLASS DOC] HELPER: Wrapper module owning one attention module and delegating
                # forward to it.
                def __init__(
                    self,
                    *,
                    mlp_size: int,
                    hidden_size: int,
                ) -> None:
                    # [FUNCTION DOC] HELPER: Keyword-only constructor that builds the inner
                    # attention module.
                    super().__init__()
                    self.attn = Attention(mlp_size, hidden_size)

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Returns the output of the inner attention module.
                    return self.attn(x)

            class CompiledAttentionTwo(CompiledAttention):
                # [CLASS DOC] HELPER: Variant of the compiled attention wrapper whose forward
                # adds the input as a residual.
                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Returns the inner attention output plus x.
                    return self.attn(x) + x

            class SimpleModelWithTwoGraphs(ParentModel):
                # [CLASS DOC] HELPER: Model with two attention wrappers and a persistent zero
                # CUDA hidden-state buffer so tensor addresses are stable across runs.
                def __init__(
                    self,
                    *,
                    mlp_size: int,
                    hidden_size: int,
                ) -> None:
                    # [FUNCTION DOC] HELPER: Keyword-only constructor creating both attention
                    # wrappers and the zero hidden-state buffer on CUDA.
                    super().__init__()
                    self.attn_one = CompiledAttention(
                        mlp_size=mlp_size,
                        hidden_size=hidden_size,
                    )
                    self.attn_two = CompiledAttentionTwo(
                        mlp_size=mlp_size,
                        hidden_size=hidden_size,
                    )

                    self.hidden_states = torch.zeros((BATCH_SIZE, MLP_SIZE)).cuda()

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Copies x into the leading rows of the hidden-state
                    # buffer, runs the first attention on that slice, copies the result back
                    # into the buffer and runs the second attention on the slice.
                    bsz = x.shape[0]
                    # CUDAGraph expects same tensor addresses for each run
                    self.hidden_states[:bsz].copy_(x)
                    x = self.attn_one(self.hidden_states[:bsz])
                    self.hidden_states[:bsz].copy_(x)
                    x = self.attn_two(self.hidden_states[:bsz])
                    return x

            eager_model = (
                SimpleModelWithTwoGraphs(
                    mlp_size=MLP_SIZE,
                    hidden_size=HIDDEN_SIZE,
                )
                .eval()
                .cuda()
            )

            compiled_model = torch.compile(eager_model, mode="reduce-overhead")

            inputs = torch.randn(BATCH_SIZE, MLP_SIZE).cuda()

            for _ in range(3):
                eager_out = eager_model(inputs)
                compiled_out = compiled_model(inputs)
                self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("triton.cudagraph_trees", False)
        def test_graph_partition_gc(self):
            # [FUNCTION DOC] With graph partition on and trees off, compiles and runs a simple
            # function inside a helper scope, collects garbage and asserts no tree manager
            # exists.
            def _test_dummy():
                # [FUNCTION DOC] HELPER: Compiles a function adding one and runs it three times
                # on random CUDA input.
                def foo(x):
                    # [FUNCTION DOC] HELPER: Returns x plus one.
                    return x + 1

                foo = torch.compile(foo)
                for _ in range(3):
                    foo(torch.randn(2, 3, device="cuda"))

            _test_dummy()
            gc.collect()
            self.assertIsNone(self.get_manager())

        def test_sparsity(self):
            # [FUNCTION DOC] Compiles a function that builds a sparse COO tensor from index and
            # value inputs and checks it equals eager over three runs.
            def foo(view_6, buf31):
                # [FUNCTION DOC] HELPER: Builds a one-sparse-dim sparse COO tensor of size
                # 1000000 by 64 from the given indices and values on CUDA.
                return aten._sparse_coo_tensor_with_dims_and_tensors(
                    1,
                    1,
                    [1000000, 64],
                    view_6,
                    buf31,
                    dtype=torch.float32,
                    layout=torch.sparse_coo,
                    device="cuda",
                    pin_memory=None,
                )

            foo_opt = torch.compile(foo)

            view_6 = torch.zeros([1, 102397], dtype=torch.int64, device="cuda")
            buf31 = torch.rand([102397, 64], device="cuda")

            for _ in range(3):
                self.assertEqual(foo_opt(view_6, buf31), foo(view_6, buf31))

        def test_accumulate_multiple_recordings(self):
            # [FUNCTION DOC] Checks that live block accounting accumulates across separate
            # recordings (two input shapes plus a data dependent branch). One block stays live
            # while outputs are held, and all blocks are freed after deleting the outputs and
            # collecting garbage.
            def foo(x):
                # [FUNCTION DOC] HELPER: Triples x, graph breaks, then returns it if its sum is
                # non-positive or ten times it otherwise.
                y = x + x + x
                torch._dynamo.graph_break()
                if y.sum() <= 0:
                    return y
                else:
                    return y * 10

            foo_opt = torch.compile(foo)

            # two separate compilations & recordings
            out1 = self.run_twc(foo_opt, torch.zeros([5], device="cuda"))

            # out1 gets manually freed
            out2 = self.run_twc(foo_opt, torch.zeros([6], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            out3 = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(out3, foo(torch.ones([5], device="cuda")))

            self.assertEqual(all_live_block_count(), 1)
            del out1, out2
            self.assertEqual(all_live_block_count(), 1)

            del out3
            gc.collect()
            self.assertEqual(all_live_block_count(), 0)

        @torch._inductor.config.patch("freezing", True)
        def test_constant_output(self):
            # [FUNCTION DOC] With freezing on, compiles a module that returns its parameter, a
            # slice of it and the input plus two, under no-grad, and checks the outputs equal
            # eager over three runs.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a ten-element float parameter on CUDA; forward
                # returns the parameter, its first two elements and the input plus two.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the parameter holding 0.0 through 9.0 on
                    # CUDA.
                    super().__init__()
                    self.param = torch.nn.Parameter(
                        torch.tensor([float(i) for i in range(10)], device="cuda")
                    )

                def forward(self, inp):
                    # [FUNCTION DOC] HELPER: Returns a tuple of the parameter, a view of its
                    # first two elements and the input plus two.
                    return self.param, self.param[0:2], inp + 2

            inp = torch.tensor([2], device="cuda")
            m = Mod()
            with torch.no_grad():
                out_eager = m(inp)

                m_comp = torch.compile(m)
                for _ in range(3):
                    self.assertEqual(out_eager, m_comp(inp))

        def test_live_outputs_multiple_graphs(self):
            # [FUNCTION DOC] Runs a graph-broken function with two different data dependent
            # paths and checks live block counts and the checkpoint counter: after deleting the
            # final output no blocks remain, and exactly two checkpoints happened (one to warm
            # up and one to record).
            def foo(x):
                # [FUNCTION DOC] HELPER: Triples x, computes y as that plus one, graph breaks,
                # then returns y plus one if the sum of squares is positive, else y.
                x = x + x + x
                y = x + 1
                torch._dynamo.graph_break()
                z = x * x
                if z.sum() > 0:
                    return y + 1
                else:
                    return y

            foo_opt = torch.compile(foo)

            self.run_twc(foo_opt, torch.zeros([5], device="cuda"))
            self.assertEqual(self.num_checkpoints(), 0)
            out = self.run_twc(foo_opt, torch.ones([5], device="cuda"))

            self.assertEqual(all_live_block_count(), 1)

            del out
            self.assertEqual(all_live_block_count(), 0)

            # we need to checkpoint from function to warmup y + 1,
            # and then again to record it
            self.assertEqual(self.num_checkpoints(), 2)

        def test_expanded_inputs(self):
            # [FUNCTION DOC] Checks that an input produced by expanding a one-row tensor to four
            # rows works with a compiled function over three runs and that a graph was recorded
            # (non-zero next graph id).
            x = torch.rand(1, 512, device="cuda").expand(4, 512)

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus four plus a 4 by 512 ones CUDA tensor.
                return x + 4 + torch.ones([4, 512], device="cuda")

            foo_opt = torch.compile()(foo)

            for _ in range(3):
                self.assertEqual(foo_opt(x), foo(x))

            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_dies_between_checkpoint(self):
            # [FUNCTION DOC] With warmup skipped, records a function through the tree API whose
            # outputs die before the next recording, then records a second function on one of
            # those outputs. Checks exactly one checkpoint happened and that after the
            # checkpoint a single block is live until the final result is deleted, after which
            # none remain.
            def foo(args):
                # [FUNCTION DOC] HELPER: Takes the argument list, removes its first tensor and
                # clears the list (so the callee holds the only reference), and returns x plus
                # one and x plus two.
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            foo_cg(inp_list)
            foo_cg([inp])

            out1, out2 = foo_cg([inp])
            inp = [out1]

            del out1, out2

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor out of the argument list, clears
                # the list and returns a one-element list with its cube.
                x = args[0]
                args.clear()
                return [x * x * x]

            self.assertEqual(self.num_checkpoints(), 0)
            foo2_cg = self.cudagraphify_impl(foo2, inp, ())

            x = foo2_cg(inp)[0]

            self.assertEqual(self.num_checkpoints(), 1)
            # out2 dies between the previous recording and the new one,
            # need to be manually deallocated after the checkpoint

            self.assertEqual(all_live_block_count(), 1)
            del x
            self.assertEqual(all_live_block_count(), 0)

        def test_aliased_storage_single_weakref(self):
            # [FUNCTION DOC] Four times runs a compiled function with a graph break that returns
            # tensors aliasing two underlying storages. Asserts the outputs map to exactly two
            # distinct storages with consistent C data pointers, and that after deleting outputs
            # no live weak references remain on the current node and a graph was recorded.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Scales x by 20, derives a row alias and a scaled copy y
                # with its alias, graph breaks, then returns x, its first row, a tail slice from
                # an index tensor of x, the alias of y and a tail slice of y.
                x = x * 20
                x_alias = x[0]
                y = x * 10
                y_alias = y[0]
                torch._dynamo.graph_break()
                ind = torch.tensor(4, device="cuda")
                x_alias2 = x[ind:]
                y_alias2 = y[ind:]
                return x, x_alias, x_alias2, y_alias, y_alias2

            for _ in range(4):
                outs = foo(torch.rand([20, 20], device="cuda"))

                ptr_to_ref = {
                    out.untyped_storage().data_ptr(): out.untyped_storage()._cdata
                    for out in outs
                }

                self.assertEqual(len(ptr_to_ref), 2)
                for out in outs:
                    self.assertEqual(
                        ptr_to_ref[out.untyped_storage().data_ptr()],
                        out.untyped_storage()._cdata,
                    )
                del outs
                del out

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def _test_aliasing_static_ref(self):
            # [FUNCTION DOC] HELPER: Shared body for the static aliasing test. A linear module
            # returns tensors aliasing its weight; after three compiled iterations with
            # backward, the weight and its aliases must share one storage handle, a following
            # compiled slice shares storage with its input, and the first node of the path must
            # record the aliased output as not unaliased in all paths and with no cached tensor
            # output (index depends on graph partition mode).
            class Mod(torch.nn.Linear):
                # [CLASS DOC] HELPER: Linear module subclass whose forward returns the weight
                # transposed matmul with the input, the transposed weight and the first four
                # rows of the weight.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the transposed weight times x, the
                    # transposed weight view and a four-row slice view of the weight.
                    return self.weight.T @ x, self.weight.T, self.weight[0:4]

            m = Mod(10, 10).cuda()

            @torch.compile(mode="reduce-overhead")
            def foo(mod, x):
                # [FUNCTION DOC] HELPER: Calls the given module on x and returns its outputs.
                return mod(x)

            @torch.compile(mode="reduce-overhead")
            def foo2(x):
                # [FUNCTION DOC] HELPER: Returns the tail of x starting at row two.
                return x[2:]

            param_c = cdata(m.weight)
            for _ in range(3):
                x = torch.rand([10, 10], device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out1, alias_1, alias_2 = foo(m, x)
                self.assertEqual(len({param_c, cdata(alias_1), cdata(alias_2)}), 1)

                out2 = foo2(out1)
                out2.sum().backward()
                self.assertEqual(cdata(out1), cdata(out2))
                m.weight.grad = None
                m.bias.grad = None

            node = self.curr_node()
            first_node = next(node._path_from_root)
            if torch._inductor.config.graph_partition:
                # graph partition may changed the order of outputs
                self.assertFalse(first_node.unaliased_in_all_paths[1])
                self.assertTrue(first_node.cached_tensor_outputs[1] is None)
            else:
                self.assertFalse(first_node.unaliased_in_all_paths[0])
                self.assertTrue(first_node.cached_tensor_outputs[0] is None)

        def test_aliasing_static_ref(self):
            # [FUNCTION DOC] Runs the shared static aliasing check with the default graph
            # partition setting.
            self._test_aliasing_static_ref()

        @torch._inductor.config.patch("graph_partition", False)
        def test_aliasing_static_ref_no_graph_partition(self):
            # [FUNCTION DOC] Runs the shared static aliasing check with graph partition
            # disabled.
            self._test_aliasing_static_ref()

        @torch._inductor.config.patch("implicit_fallbacks", True)
        def test_multinomial(self):
            # [FUNCTION DOC] Checks that multinomial sampling gives the same samples in eager
            # (preserving RNG state) and compiled form with fallbacks enabled, and that no
            # additional cudagraph skips were recorded.
            def sample_multinomial(probs, num_samples, replacement=True):
                # [FUNCTION DOC] HELPER: Draws the requested number of samples from the
                # probability tensor with the given replacement mode.
                return torch.multinomial(probs, num_samples, replacement=replacement)

            # Create and prepare probability tensor on GPU
            probs = torch.tensor([0.1, 0.2, 0.3, 0.4]).cuda()
            probs = probs / probs.sum()

            # Sample using the function
            num_skipped = counters["inductor"]["cudagraph_skips"]

            with torch._dynamo.utils.preserve_rng_state():
                samples = self.run_twc(
                    sample_multinomial, probs, num_samples=5, replacement=True
                )

            with torch._dynamo.utils.preserve_rng_state():
                samples_compiled = self.run_twc(
                    torch.compile(sample_multinomial),
                    probs,
                    num_samples=5,
                    replacement=True,
                )

            self.assertEqual(samples, samples_compiled)
            self.assertEqual(num_skipped, counters["inductor"]["cudagraph_skips"])

        def test_checkpointing_resets_persistent_refs(self):
            # [FUNCTION DOC] Checks checkpointing clears persistent output references. After
            # steady state with no checkpoints, a freed output id is reused by the next call.
            # Then, by decrementing the generation counter to force a new recording, two
            # iterations each show two live blocks, then one, then zero as outputs are deleted,
            # with the checkpoint counter increasing by one per iteration, and a new output not
            # reusing the old id.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x with itself.
                return x @ x

            def inp():
                # [FUNCTION DOC] HELPER: Returns a random 20 by 20 CUDA tensor not requiring
                # grad.
                return torch.rand([20, 20], device="cuda", requires_grad=False)

            for _ in range(3):
                foo(inp())

            self.assertEqual(self.num_checkpoints(), 0)

            out = foo(inp())
            out_id = id(out)
            del out
            self.assertEqual(id(foo(inp())), out_id)

            @torch.compile(mode="reduce-overhead")
            def foo2(x):
                # [FUNCTION DOC] HELPER: Returns the first row of x and x matmul x.
                return x[0], x @ x

            for i in range(2):
                out = foo(inp())

                from torch._dynamo.mutation_guard import GenerationTracker

                GenerationTracker.generation -= 1

                out_alias, out2 = foo2(out)
                del out_alias

                self.assertEqual(all_live_block_count(), 2)
                del out
                self.assertEqual(all_live_block_count(), 1)
                del out2
                self.assertEqual(all_live_block_count(), 0)

                self.assertEqual(self.num_checkpoints(), i + 1)

            new_out = foo(inp())
            curr_node = self.curr_node()
            self.assertFalse(curr_node.unaliased_in_all_paths[0])
            self.assertFalse(out_id == id(new_out))

        def test_aliased_static_parameter(self):
            # [FUNCTION DOC] Checks that a graph with a static parameter input whose output
            # aliases it keeps the same storage over three runs, and the node records no cached
            # tensor output and not-unaliased for that output.
            inp = torch.rand([20, 20], device="cuda")

            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the argument list, clears
                # the list and returns a one-element tuple with its first row.
                x = args[0]
                args.clear()
                return (x[0],)

            foo_cg = self.cudagraphify_impl(foo, [inp], (0,))

            for _ in range(3):
                out = foo_cg([inp])[0]
                self.assertEqual(cdata(inp), cdata(out))

            node = self.curr_node()
            self.assertEqual(node.cached_tensor_outputs, [None])
            self.assertEqual(node.unaliased_in_all_paths, [False])

        def test_warmup_stream_sync(self):
            # [FUNCTION DOC] Records a function running a hundred chained large matmuls on a
            # side stream chosen by the user, and checks the result equals an eager reference
            # plus one, ensuring warmup is synchronized with the user stream.
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the argument list, clears
                # it, squares it by matmul one hundred times and returns it in a tuple.
                x = args[0]
                args.clear()
                x_orig = x
                for _ in range(100):
                    x = x @ x
                return (x,)

            inp = torch.rand([4096, 4096], device="cuda")
            ref = foo([inp])[0]
            torch.cuda.synchronize()

            user_stream = torch.cuda.Stream()
            with torch.cuda.stream(user_stream):
                foo_cg = self.cudagraphify_impl(foo, [inp], (0,))
                out = foo_cg([inp])[0]
                y = out + 1
                self.assertEqual(y, ref + 1)

        def test_unaligned_static_parameter(self):
            # [FUNCTION DOC] Uses a static input that is an unaligned view; across three runs
            # output matches eager, and the node records no static input data pointer for it.
            def gen_inp():
                # [FUNCTION DOC] HELPER: Returns a list holding a view of a ones tensor starting
                # at element one.
                inp = torch.ones([20], device="cuda")
                return [inp[1:]]

            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears the list
                # and returns its double in a tuple.
                x = args[0]
                args.clear()
                return (x + x,)

            foo_cg = self.cudagraphify_impl(foo, gen_inp(), (0,))

            for _ in range(3):
                out = foo_cg(gen_inp())
                self.assertEqual(out, foo(gen_inp()))
                del out

            node = self.curr_node()
            self.assertEqual(node.static_input_data_ptrs, [None])

        def test_amp_cache_disabled(self):
            # [FUNCTION DOC] Checks that the autocast cache does not retain cudagraph outputs:
            # after computing a matmul under autocast, zeroing the output in place and
            # recomputing must give a different result.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

            for _ in range(3):
                out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            # amp cache for cudagraph outputs should be disabled
            t2 = torch.rand([4, 4], device="cuda")

            with torch.cuda.amp.autocast():
                run_once = out @ t2

                out.detach().zero_()

                run_twice = out @ t2

                self.assertNotEqual(run_once, run_twice)

        def test_remove_hooks_on_cached_tensors(self):
            # [FUNCTION DOC] Checks that outputs returned from compiled functions have no
            # leftover backward hooks (and no post accumulate grad hooks for a leaf output)
            # across five iterations even after hooks are registered on prior outputs.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared.
                return x * x

            inp = torch.rand([4], device="cuda", requires_grad=True)

            for _ in range(5):
                out = foo(inp)
                self.assertIsNone(out._backward_hooks)
                out.register_hook(lambda: None)

            # today, torch.compile never outputs a leaf tensor which is the only
            # tensor that can register _post_accumulate_grad_hooks
            # add this as a preventative test

            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Ignores x and returns a new random leaf tensor
                # requiring grad.
                return torch.rand([4], device="cuda", requires_grad=True)

            for _ in range(5):
                out = foo(inp)
                self.assertIsNone(out._post_accumulate_grad_hooks)
                out.register_post_accumulate_grad_hook(lambda: None)

        def test_multiple_insert_removal_caching(self):
            # [FUNCTION DOC] Enables cached tensors in the C++ layer, adds the same tensor twice
            # and removes it once, and asserts it is no longer cached (membership is not
            # reference counted). Cached tensors are disabled again in a finally block.
            torch._C._set_cached_tensors_enabled(True)
            try:
                x = torch.rand([4], device="cuda")

                torch._C._add_cached_tensor(x)
                self.assertTrue(torch._C._is_cached_tensor(x))

                torch._C._add_cached_tensor(x)
                torch._C._remove_cached_tensor(x)

                self.assertFalse(torch._C._is_cached_tensor(x))
            finally:
                torch._C._set_cached_tensors_enabled(False)

        def test_accumulate_grad(self):
            # [FUNCTION DOC] Checks gradient accumulation is unaffected by cudagraph trees: for
            # three iterations, two backward calls on a compiled add accumulate in place without
            # create_graph (gradient equals twice the first clone) and out of place with
            # create_graph (gradient equals the first clone).
            # cudagraph trees shouldn't interfere with accumulation logic

            def compute_grad(grad_output, create_graph):
                # [FUNCTION DOC] HELPER: Builds a random leaf requiring grad, runs a compiled
                # add of two on it, backpropagates twice (the second with the given
                # create_graph) and returns the gradient reference taken after the first
                # backward and a clone of it.
                x = torch.randn(5, 5, requires_grad=True, device="cuda")

                @torch.compile()
                def foo(x):
                    # [FUNCTION DOC] HELPER: Returns x plus two.
                    return x + 2

                y = foo(x)
                y.backward(grad_output, retain_graph=True)
                x_grad = x.grad
                x_grad_clone = x.grad.clone()
                y.backward(grad_output, create_graph=create_graph)
                return x_grad, x_grad_clone

            for _ in range(3):
                grad_output = torch.ones(5, 5, device="cuda")

                # Accumulate in-place when create_graph is False
                x_grad, x_grad_clone = compute_grad(grad_output, create_graph=False)
                self.assertEqual(x_grad, x_grad_clone * 2)

                # Accumulate out-of-place when create_graph is False
                x_grad, x_grad_clone = compute_grad(grad_output, create_graph=True)
                self.assertEqual(x_grad, x_grad_clone)

        def test_frozen_fn(self):
            # [FUNCTION DOC] After three compiled runs (graph id one), wraps the function with
            # the dynamo run wrapper to get a frozen version; three more runs of it must not
            # create new recordings.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x with itself.
                return x @ x

            for _ in range(3):
                out = foo(torch.rand([10, 10], device="cuda"))

            self.assertTrue(self.get_manager().new_graph_id().id == 1)
            frozen = torch._dynamo.run(foo)

            for _ in range(3):
                out = frozen(torch.rand([10, 10], device="cuda"))

            # didn't do additional recordings
            self.assertTrue(self.get_manager().new_graph_id().id == 2)

        def test_empty_cpu_tensor(self):
            # [FUNCTION DOC] Checks that a function returning a CUDA matmul and an empty CPU
            # tensor matches eager over three runs and records exactly one graph.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x matmul x and a new empty CPU tensor.
                return x @ x, torch.tensor([])

            foo_opt = torch.compile(foo)
            x = torch.rand([4], device="cuda")

            for _ in range(3):
                out_opt = foo_opt(x)
                self.assertEqual(foo(x), out_opt)

            self.assertTrue(self.get_manager().new_graph_id().id == 1)

        def test_output_alias(self):
            # [FUNCTION DOC] Checks that two outputs aliasing the same storage share one storage
            # handle, that no live weak references remain once they are deleted, and that no
            # cached tensor outputs are kept.
            inp = torch.rand([20, 20], device="cuda")

            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it,
                # computes an unused sum and returns the tensor and its first row.
                x = args[0]
                args.clear()
                out = x + x
                return (x, x[0])

            foo_cg = self.cudagraphify_impl(foo, [inp], ())

            for _ in range(3):
                out_1, out_2 = foo_cg([inp])
                self.assertEqual(cdata(out_1), cdata(out_2))
                del out_1, out_2
                self.assertEqual(len(list(self.curr_node().path_live_weakrefs())), 0)

            self.assertEqual(self.curr_node().cached_tensor_outputs, [None, None])

        def test_empty_storage(self):
            # [FUNCTION DOC] Checks live weak reference counts on the current node when outputs
            # include empty storages: with a normal input only the non-empty output is tracked,
            # and with an empty input a random-plus-ten output is the only one tracked (count
            # one in each case).
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x tripled, an empty CUDA tensor and an empty
                # slice of a larger zeros tensor.
                return (
                    (x + x + x),
                    torch.zeros([0], device="cuda"),
                    torch.zeros([100], device="cuda")[0:0],
                )

            inp = torch.rand([4], device="cuda")
            for _ in range(3):
                out = foo(inp)
                node = self.curr_node()
                self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x tripled and a random four-element CUDA tensor
                # plus ten.
                return (x + x + x), torch.rand([4], device="cuda") + 10

            inp = torch.rand([0], device="cuda")
            for _ in range(3):
                out = foo(inp)
                node = self.curr_node()
                self.assertEqual(len(list(node.path_live_weakrefs())), 1)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_aliased_output_checkpoint(self):
            # [FUNCTION DOC] With warmup skipped, records a function with outputs that alias
            # each other, drops them, and records a second function on one output. Checks one
            # checkpoint occurred and block counts go from one to zero after deleting the final
            # result.
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns x plus one, a tensor y equal to x plus two and the first row of y.
                x = args[0]
                args.clear()
                y = x + 2
                return x + 1, y, y[0]

            inp = torch.rand([4, 4], device="cuda")
            foo_cg = self.cudagraphify_impl(foo, [inp], ())
            foo_cg([inp])
            foo_cg([inp])

            out1, out2, out3 = foo_cg([inp])
            inp = [out1]

            del out1, out2, out3

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element list with its cube.
                x = args[0]
                args.clear()
                return [x * x * x]

            self.assertEqual(self.num_checkpoints(), 0)
            foo2_cg = self.cudagraphify_impl(foo2, inp, ())

            x = foo2_cg(inp)[0]

            self.assertEqual(self.num_checkpoints(), 1)
            # out2 and out3 dies between the previous recording and the new one,
            # need to be manually deallocated after the checkpoint

            self.assertEqual(all_live_block_count(), 1)
            del x
            self.assertEqual(all_live_block_count(), 0)

        @unittest.skipUnless(
            (IS_X86 or IS_ARM64) and IS_LINUX, "cpp contexts are linux x86/aarch64 only"
        )
        @torch._inductor.config.patch("triton.cudagraph_trees_history_recording", True)
        @blas_library_context("cublas")
        @unittest.mock.patch.dict(os.environ, {"TORCH_DISABLE_ADDR2LINE": "0"})
        def test_workspace_allocation_error(self):
            # [FUNCTION DOC] Linux x86/aarch64 only, with cuBLAS and history recording enabled.
            # Replaces the cublas clearing context manager in cudagraph_trees with a null
            # context, compiles a matmul, and expects the run to raise; on non-ARM64 the error
            # text must name a gemm function and a cuBLAS workspace/handle function. The finally
            # block clears cuBLAS workspaces, restores the original manager and resets the
            # device tree manager to none.
            torch._C._cuda_clearCublasWorkspaces()

            prev = torch._inductor.cudagraph_trees.clear_cublas_manager

            try:
                torch._inductor.cudagraph_trees.clear_cublas_manager = (
                    contextlib.nullcontext
                )

                @torch.compile()
                def foo(x, y):
                    # [FUNCTION DOC] HELPER: Returns x matmul x, ignoring y.
                    return x @ x

                inps = [torch.rand([400, 400], device="cuda") for _ in range(2)]

                thrown = False
                try:
                    foo(*inps)
                except Exception as e:
                    thrown = True
                    if not IS_ARM64:
                        # CUDA uses gemm/gemm_internal_cublas, ROCm uses bgemm_internal_cublaslt
                        self.assertTrue(
                            "at::cuda::blas::gemm<float, float>" in str(e)
                            or "at::cuda::blas::gemm_internal_cublas<float, float>"
                            in str(e)
                            or "at::cuda::blas::bgemm_internal_cublaslt<float, float>"
                            in str(e)
                        )
                        # CUDA uses getCurrentCUDABlasHandle/getNewWorkspace,
                        # ROCm uses getNewCUDABlasLtWorkspace/getCUDABlasLtWorkspace
                        self.assertTrue(
                            "getCurrentCUDABlasHandle" in str(e)
                            or "getNewWorkspace" in str(e)
                            or "CUDABlasLtWorkspace" in str(e)
                        )

                self.assertTrue(thrown)

            finally:
                torch._C._cuda_clearCublasWorkspaces()
                torch._inductor.cudagraph_trees.clear_cublas_manager = prev
                torch._inductor.cudagraph_trees.get_container(
                    self.device_idx
                ).tree_manager = None

        def test_peristed_output_livenes(self):
            # [FUNCTION DOC] Checks weak-reference liveness of persisted outputs: with no user
            # output alive none are tracked; a fresh output is the cached tensor output of the
            # node and counts as one live reference; a view keeps it alive after deleting the
            # original; deleting the view brings the count to zero.
            @torch.compile
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus x.
                return x + x

            for _ in range(3):
                foo(torch.rand([2, 2], device="cuda"))

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)

            out = foo(torch.rand([2, 2], device="cuda"))
            self.assertTrue(out is node.cached_tensor_outputs[0])
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            out_ref = out[0:]
            del out
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

            del out_ref
            self.assertEqual(len(list(node.path_live_weakrefs())), 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_tensor_no_longer_in_pool(self):
            # [FUNCTION DOC] With warmup skipped, records two chained graphs, drops the first
            # graph outputs, replays the first graph and then runs the second with a different
            # input tensor (not in the pool). Expects one checkpoint and a single root with two
            # descendants, i.e. the second graph is re-recorded as a new child.
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns x plus one and x plus two.
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            x1, x2 = foo_cg(inp_list)

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element list with its cube.
                x = args[0]
                args.clear()
                return [x * x * x]

            inp_list = [x1]
            foo2_cg = self.cudagraphify_impl(foo2, inp_list, ())
            foo2_cg(inp_list)

            del x1, x2
            # TODO make configurable

            x1, x2 = foo_cg([inp])
            self.assertEqual(self.num_checkpoints(), 0)

            # input location has changed, should force recompile and checkpointing
            foo2_cg([torch.zeros_like(x1)])

            self.assertEqual(self.num_checkpoints(), 1)
            self.assertEqual(self.get_root_children(), [2])

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_rerecording_logs_reason(self):
            # [FUNCTION DOC] Same chained-recording scenario as the no-longer-in-pool test,
            # using explicit placeholder info for each input, and asserts that running the
            # second graph with a new input location emits a debug log record containing the
            # text for re-recording from the cudagraph trees logger (with cudagraphs logging
            # enabled).
            import logging

            from torch.testing._internal.logging_utils import log_settings

            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns x plus one and x plus two.
                x = args[0]
                args.clear()
                return x + 1, x + 2

            inp = torch.rand([4], device="cuda")
            inp_list = [inp]
            # Create placeholder info for the input
            foo_placeholders = (PlaceholderInfo("arg0", None, [], None),)
            foo_cg = self.cudagraphify_impl(
                foo, inp_list, (), placeholders=foo_placeholders
            )
            x1, x2 = foo_cg(inp_list)

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element list with its cube.
                x = args[0]
                args.clear()
                return [x * x * x]

            inp_list = [x1]
            foo2_placeholders = (PlaceholderInfo("arg0", None, [], None),)
            foo2_cg = self.cudagraphify_impl(
                foo2, inp_list, (), placeholders=foo2_placeholders
            )
            foo2_cg(inp_list)

            del x1, x2

            x1, x2 = foo_cg([inp])

            # With cudagraphs logging enabled, we should see the re-recording debug log
            with log_settings("cudagraphs"):
                with self.assertLogs(
                    logger="torch._inductor.cudagraph_trees", level=logging.DEBUG
                ) as log:
                    foo2_cg([torch.zeros_like(x1)])

            self.assertTrue(
                any("Re-recording" in record.getMessage() for record in log.records)
            )

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_checkpoint_shared_output_storage_deallocation(self):
            # [FUNCTION DOC] With warmup skipped, records a function returning two rows of the
            # same input storage, then a second function on one of them. Expects one checkpoint,
            # the two outputs share a storage address, and live block counts stay at one until
            # both outputs are deleted, then zero.
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it,
                # computes an unused temporary and returns the first and second rows of x.
                x = args[0]
                args.clear()
                x_tmp = x + 1
                return x[0], x[1]

            inp = torch.rand([2, 2], device="cuda")
            inp_list = [inp]
            foo_cg = self.cudagraphify_impl(foo, inp_list, ())
            foo_cg(inp_list)
            foo_cg([inp])

            x1, x2 = foo_cg([inp])
            inp = [x1]

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it, squares
                # it and returns its first and second rows.
                x = args[0]
                args.clear()
                y = x * x
                return y[0], y[1]

            foo2_cg = self.cudagraphify_impl(foo2, inp, ())
            foo2_cg(inp)

            self.assertEqual(self.num_checkpoints(), 1)
            self.assertEqual(
                x1.untyped_storage().data_ptr(), x2.untyped_storage().data_ptr()
            )
            self.assertEqual(all_live_block_count(), 1)
            del x1
            self.assertEqual(all_live_block_count(), 1)
            del x2
            self.assertEqual(all_live_block_count(), 0)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_cleanup(self):
            # [FUNCTION DOC] Compiles and runs a function twice inside a closure, then resets
            # dynamo and asserts the tree manager has been cleaned up (none exists). Outputs
            # returned from the closure are still held.
            def test_closure():
                # [FUNCTION DOC] HELPER: Defines a compiled function, runs it once as warmup and
                # returns the result of a second run.
                @torch.compile
                def foo(x):
                    # [FUNCTION DOC] HELPER: Returns x plus three and x times ten.
                    return x + 1 + 2, x * 10

                foo(torch.rand([4], device="cuda"))
                return foo(torch.rand([4], device="cuda"))

            out1, out2 = test_closure()
            torch._dynamo.reset()

            # TODO - deallocate on tensor deallocation
            # self.assertTrue(self.get_manager() is not None)
            # del out1
            # self.assertTrue(self.get_manager() is not None)
            # del out2
            self.assertTrue(self.get_manager() is None)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_forward_backward(self):
            # [FUNCTION DOC] With warmup skipped, runs forward and backward of a compiled
            # function with dropout. Expects one root child, empty expected dead indices before
            # the graph, and expected dead indices after the graph that depend on graph
            # partition mode, and a positive next graph id.
            @torch.compile
            def foo(x):
                # [FUNCTION DOC] HELPER: Doubles x and returns the sine of that times dropout of
                # x with probability 0.4.
                y = x * 2
                return torch.sin(y) * torch.nn.functional.dropout(x, p=0.4)

            inp = torch.rand([4, 4], requires_grad=True, device="cuda")
            out = foo(inp)
            out.sum().backward()

            self.assertEqual(self.get_root_children(), [1])

            # the three saved tensors should die in the backward
            # we kept alive the output
            self.assertEqual(self.curr_node().expected_dead_indices_before_graph, [])
            if torch._inductor.config.graph_partition:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 0), (0, 2)],
                )
            else:
                self.assertEqual(
                    self.curr_node().expected_dead_indices_after_graph,
                    [(0, 1), (0, 2)],
                )
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_separate_recordings(self):
            # [FUNCTION DOC] Runs a compiled function with two different input shapes (twenty by
            # twenty then forty by forty), three calls each with eager calls in between, and
            # expects two separate root recordings with no descendants.
            def foo_unopt(x, y):
                # [FUNCTION DOC] HELPER: Returns the matrix product of x plus one with y.
                return (x + 1) @ y

            foo = torch.compile(foo_unopt)

            foo_unopt(
                torch.ones([20, 20], device="cuda"), torch.ones([20, 20], device="cuda")
            )

            inps = [
                torch.ones([20, 20], device="cuda", requires_grad=False)
                for _ in range(2)
            ]

            out = foo(*inps)
            torch.cuda.synchronize()
            foo(*inps)
            torch.cuda.synchronize()
            foo(*inps)
            torch.cuda.synchronize()

            foo_unopt(
                torch.ones([20, 20], device="cuda"), torch.ones([20, 20], device="cuda")
            )

            inps2 = [
                torch.rand([40, 40], device="cuda", requires_grad=False)
                for _ in range(2)
            ]

            foo(*inps2)
            foo(*inps2)
            foo(*inps2)

            # two separate roots
            self.assertEqual(self.get_root_children(), [0, 0])

        def test_alias_of_parameter(self):
            # [FUNCTION DOC] Compiles a module whose outputs include a row of its parameter, the
            # parameter itself and the parameter plus the input. Checks results match eager, the
            # parameter storage is not freed, and exactly one live weak reference exists on the
            # current node.
            class AliasMod(nn.Module):
                # [CLASS DOC] HELPER: Module with a random 20 by 20 CUDA parameter; forward
                # returns the first row of the parameter, the parameter and the parameter plus
                # the input.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the parameter.
                    super().__init__()
                    self.param = torch.nn.Parameter(torch.rand([20, 20], device="cuda"))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns a tuple of the parameter's first row, the
                    # parameter and the parameter plus x.
                    return self.param[0], self.param, self.param + x

            @torch.compile(mode="reduce-overhead")
            def foo(mod, inp):
                # [FUNCTION DOC] HELPER: Calls the module on the input and returns its outputs.
                return mod(inp)

            inp = torch.rand([20, 20], device="cuda")
            mod = AliasMod()

            storage_ref = torch.multiprocessing.reductions.StorageWeakRef(
                mod.param.untyped_storage()
            )

            for _ in range(3):
                outs = foo(mod, inp)

            self.assertEqual(mod(inp), outs)

            self.assertFalse(storage_ref.expired())

            node = self.get_manager().current_node
            self.assertEqual(len(list(node.path_live_weakrefs())), 1)

        @requires_multigpu()
        def test_manager_per_device(self):
            # [FUNCTION DOC] Multi-GPU test checking that tree managers are per device: after
            # recording on the test device, a manager exists for it and not for the next device
            # index, and after the inner scope exits no manager remains for the test device.
            def test():
                # [FUNCTION DOC] HELPER: Records a function three times via the tree API on the
                # test device, comparing with eager, then checks a manager exists for the test
                # device and not for the next device.
                def foo(args):
                    # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                    # returns a one-element tuple with x plus three.
                    x = args[0]
                    args.clear()
                    return (x + 3,)

                inp = torch.rand([20, 20], device=f"cuda:{self.device_idx}")

                inp_list = [inp]
                foo_cg = self.cudagraphify_impl(foo, inp_list, ())
                for _ in range(3):
                    self.assertEqual(foo_cg([inp]), foo([inp]))

                next_idx = (self.device_idx + 1) % torch.cuda.device_count()
                self.assertTrue(self.get_manager(device_index=next_idx) is None)
                self.assertFalse(self.get_manager(device_index=self.device_idx) is None)

            test()
            self.assertTrue(self.get_manager(device_index=self.device_idx) is None)

        def test_error_on_dealloc_use(self):
            # [FUNCTION DOC] Calls a compiled function twice then uses each earlier output; the
            # first use must raise an error saying the output was overwritten by a subsequent
            # run. After another run the second output also raises the same error.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            inp = torch.rand([4], device="cuda")
            out = foo(inp)
            out2 = foo(inp)

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
                out + out

            foo(inp)

            with self.assertRaisesRegex(Exception, "overwritten by a subsequent"):
                out2 + out2

        def test_error_on_dealloc_use2(self):
            # [FUNCTION DOC] Like the previous test but with detached outputs; checks the raised
            # error text mentions being overwritten, the compiled function source, the
            # step-begin marker, manually cloning, the generation cloning config and cloning
            # user-visible tensors that persist, for both outputs.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            inp = torch.rand([4], device="cuda")
            out = foo(inp).detach()
            out2 = foo(inp).detach()

            with self.assertRaises(Exception) as exc:
                out + out

            FileCheck().check("overwritten").check("x * x * x").check(
                "cudagraph_mark_step_begin"
            ).check("manually clone").check("cudagraph_trees_generation_cloning").check(
                "clone all user-visible tensors that persist"
            ).run(repr(exc.exception))

            foo(inp)

            with self.assertRaises(Exception) as exc:
                out2 + out2

            FileCheck().check("overwritten").check("x * x * x").check(
                "cudagraph_mark_step_begin"
            ).check("manually clone").check("cudagraph_trees_generation_cloning").check(
                "clone all user-visible tensors that persist"
            ).run(repr(exc.exception))

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        @parametrize(
            "variant", ("default", "no_graph_partition", "no_keep_output_stride")
        )
        def test_clone_live_output_on_new_generation(self, variant):
            # [FUNCTION DOC] Parametrized by variant (default, graph partition off, keep output
            # stride off) with user_visible generation cloning. Spies on the C++
            # free-and-remove-deleter function. Runs successive generations (record, replay,
            # replay) while keeping outputs alive, and checks that live outputs of an earlier
            # generation are cloned out of the pool (storage replaced, deleter becomes standard,
            # values preserved), free calls counted as expected (one after the second run, two
            # after the first replay), cached tensor output cleared and not cached by the end.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            patches = {
                "no_graph_partition": {"graph_partition": False},
                "no_keep_output_stride": {"keep_output_stride": False},
            }.get(variant, {})
            with config.patch(patches):
                inputs = [
                    torch.randn(4, device="cuda", dtype=torch.float32) + i
                    for i in range(4)
                ]
                live_outputs = []
                free_calls = 0
                orig_free = torch._C._free_And_Remove_DeleterFn

                def spy_free(storage_ptr):
                    # [FUNCTION DOC] HELPER: Wrapper around the original free function: asserts
                    # the storage has a standard deleter, increments the free call counter and
                    # calls the original.
                    nonlocal free_calls
                    self.assertTrue(torch._C._has_Standard_Deleter(storage_ptr))
                    free_calls += 1
                    return orig_free(storage_ptr)

                with unittest.mock.patch.object(
                    torch._C, "_free_And_Remove_DeleterFn", spy_free
                ):
                    out = self.run_next_generation(foo, inputs[0])
                    expected = out.detach().clone()
                    old_data_ptr = out.untyped_storage().data_ptr()
                    self.save_live_outputs(live_outputs, out)

                    record_out = self.run_next_generation(foo, inputs[1])
                    record_expected = record_out.detach().clone()
                    record_old_data_ptr = record_out.untyped_storage().data_ptr()
                    self.save_live_outputs(live_outputs, record_out)
                    self.assertEqual(free_calls, 1)
                    self.assertTrue(torch._C._has_Standard_Deleter(cdata(out)))

                    replay_out = self.replay_next_generation(foo, inputs[2])
                    node = self.curr_node()
                    self.assertEqual(free_calls, 2)
                    self.assertTrue(torch._C._has_Standard_Deleter(cdata(record_out)))
                    self.assertIs(replay_out, node.cached_tensor_outputs[0])
                    self.assertTrue(torch._C._is_cached_tensor(replay_out))
                    replay_old_data_ptr = replay_out.untyped_storage().data_ptr()
                    self.save_live_outputs(live_outputs, replay_out)
                    # Exercise a second replay after cloning live outputs.
                    next_out = self.replay_next_generation(foo, inputs[3])
                    self.save_live_outputs(live_outputs, next_out)
                    self.assertEqual(free_calls, 2)
                    self.assertTrue(torch._C._has_Standard_Deleter(cdata(replay_out)))

            self.assertEqual(out, expected)
            self.assertEqual(out + out, expected + expected)
            self.assertNotEqual(out.untyped_storage().data_ptr(), old_data_ptr)
            self.assertEqual(record_out, record_expected)
            self.assertNotEqual(
                record_out.untyped_storage().data_ptr(), record_old_data_ptr
            )
            self.assertNotEqual(
                replay_out.untyped_storage().data_ptr(), replay_old_data_ptr
            )
            self.assertEqual(replay_out, inputs[2] * inputs[2] * inputs[2])
            self.assertEqual(next_out, inputs[3] * inputs[3] * inputs[3])
            self.assertIsNone(node.cached_tensor_outputs[0])
            self.assertFalse(torch._C._is_cached_tensor(replay_out))
            self.assertIsNot(out, replay_out)

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        def test_clone_live_output_keeps_cache_when_output_dead(self):
            # [FUNCTION DOC] With user_visible generation cloning, checks that when the first
            # output is deleted before the next generation, nothing needs cloning: after a
            # record and a replay, the cached output tensor of the node is the replay output;
            # after deleting it, the next replay reuses the same Python object, storage address
            # and cache entry and gives correct values.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            inputs = [
                torch.randn(4, device="cuda", dtype=torch.float32) + i for i in range(4)
            ]
            out = self.run_next_generation(foo, inputs[0])
            del out

            self.run_next_generation(foo, inputs[1])
            replay_out = self.replay_next_generation(foo, inputs[2])
            node = self.curr_node()
            self.assertIs(replay_out, node.cached_tensor_outputs[0])
            self.assertTrue(torch._C._is_cached_tensor(replay_out))
            cached_id = id(replay_out)
            cached_data_ptr = replay_out.untyped_storage().data_ptr()
            del replay_out

            next_out = self.replay_next_generation(foo, inputs[3])
            self.assertIs(next_out, node.cached_tensor_outputs[0])
            self.assertEqual(id(next_out), cached_id)
            self.assertEqual(next_out.untyped_storage().data_ptr(), cached_data_ptr)
            self.assertTrue(torch._C._is_cached_tensor(next_out))
            self.assertEqual(next_out, inputs[3] * inputs[3] * inputs[3])

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        def test_clone_live_aliased_output_storage_on_new_generation(self):
            # [FUNCTION DOC] With user_visible generation cloning, outputs where one tensor is a
            # view of another share one storage. Across a record and two replays with all
            # outputs kept alive, the cloned storage must keep both tensors sharing one storage
            # while preserving values, sizes, strides and storage offsets, and the later replays
            # must be correct.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the square of x and a view skipping its first
                # row.
                y = x * x
                return y, y[1:]

            inputs = [
                torch.randn(4, 4, device="cuda", dtype=torch.float32) + i
                for i in range(4)
            ]
            live_outputs = []
            out, out_view = self.run_next_generation(foo, inputs[0])
            expected = out.detach().clone()
            expected_view = out_view.detach().clone()
            old_data_ptr = out.untyped_storage().data_ptr()
            view_size = out_view.size()
            view_stride = out_view.stride()
            view_storage_offset = out_view.storage_offset()
            self.assertEqual(cdata(out), cdata(out_view))
            self.save_live_outputs(live_outputs, out, out_view)

            record_out, record_view = self.run_next_generation(foo, inputs[1])
            self.save_live_outputs(live_outputs, record_out, record_view)
            replay_out, replay_view = self.replay_next_generation(foo, inputs[2])
            self.save_live_outputs(live_outputs, replay_out, replay_view)
            next_out, next_view = self.replay_next_generation(foo, inputs[3])
            self.save_live_outputs(live_outputs, next_out, next_view)

            self.assertEqual(out, expected)
            self.assertEqual(out_view, expected_view)
            self.assertEqual(cdata(out), cdata(out_view))
            self.assertNotEqual(out.untyped_storage().data_ptr(), old_data_ptr)
            self.assertEqual(out_view.size(), view_size)
            self.assertEqual(out_view.stride(), view_stride)
            self.assertEqual(out_view.storage_offset(), view_storage_offset)
            self.assertEqual(replay_out, inputs[2] * inputs[2])
            self.assertEqual(replay_view, (inputs[2] * inputs[2])[1:])
            self.assertEqual(next_out, inputs[3] * inputs[3])
            self.assertEqual(next_view, (inputs[3] * inputs[3])[1:])

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_clone_live_retargets_non_user_visible_output_alias(self):
            # [FUNCTION DOC] With user_visible cloning and warmup skipped, records a function
            # whose second output is a view of the first, where only the first output index is
            # marked user visible. Checks that after later generations the live view is
            # retargeted to the cloned storage, still shares storage with its base, values are
            # kept, and the same sharing holds for later replays.
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it, and
                # returns y equal to x squared and a view of y without its first row.
                x = args[0]
                args.clear()
                y = x * x
                return y, y[1:]

            inputs = [
                torch.randn(4, 4, device="cuda", dtype=torch.float32) + i
                for i in range(4)
            ]
            live_outputs = []
            foo_cg = self.cudagraphify_impl(
                foo,
                [inputs[0]],
                (),
                stack_traces=[None, None],
                user_visible_output_idxs=(0,),
            )
            out, out_view = self.run_next_generation(foo_cg, [inputs[0]])
            expected_view = out_view.detach().clone()
            old_data_ptr = out_view.untyped_storage().data_ptr()
            self.save_live_outputs(live_outputs, out, out_view)

            replay_out, replay_view = self.replay_next_generation(foo_cg, [inputs[1]])
            replay_expected_view = replay_view.detach().clone()
            replay_old_data_ptr = replay_view.untyped_storage().data_ptr()
            self.save_live_outputs(live_outputs, replay_out, replay_view)
            next_out, next_view = self.replay_next_generation(foo_cg, [inputs[2]])
            self.save_live_outputs(live_outputs, next_out, next_view)
            steady_out, steady_view = self.replay_next_generation(foo_cg, [inputs[3]])
            self.save_live_outputs(live_outputs, steady_out, steady_view)

            self.assertEqual(cdata(out), cdata(out_view))
            self.assertEqual(out_view, expected_view)
            self.assertNotEqual(out_view.untyped_storage().data_ptr(), old_data_ptr)
            self.assertEqual(cdata(replay_out), cdata(replay_view))
            self.assertEqual(replay_view, replay_expected_view)
            self.assertNotEqual(
                replay_view.untyped_storage().data_ptr(), replay_old_data_ptr
            )
            self.assertEqual(cdata(next_out), cdata(next_view))
            self.assertEqual(cdata(steady_out), cdata(steady_view))

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_clone_live_prior_graph_output_alias(self):
            # [FUNCTION DOC] With user_visible cloning and warmup skipped, chains two recorded
            # graphs where the second returns a view of the first graph output, so the node
            # output aliases a prior graph output (checked by type of the alias entry). After
            # later generations the live base and view outputs still hold the right values,
            # share storage, and the base storage address changed (was cloned).
            def foo(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element tuple with its square.
                x = args[0]
                args.clear()
                return (x * x,)

            def foo2(args):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element tuple with a view skipping its first row.
                x = args[0]
                args.clear()
                return (x[1:],)

            inputs = [
                torch.randn(4, 4, device="cuda", dtype=torch.float32) + i
                for i in range(4)
            ]
            live_outputs = []
            foo_cg = self.cudagraphify_impl(foo, [inputs[0]], (), stack_traces=[None])
            foo2_cg = self.cudagraphify_impl(
                foo2,
                [inputs[0]],
                (),
                stack_traces=[None],
                user_visible_output_idxs=(0,),
            )

            def run_pair(x):
                # [FUNCTION DOC] HELPER: Starts a new cudagraph step, runs the first graph on x
                # to get a base tensor, runs the second graph on that base to get a view, and
                # returns both.
                torch.compiler.cudagraph_mark_step_begin()
                base = foo_cg([x])[0]
                view = foo2_cg([base])[0]
                return base, view

            out, out_view = run_pair(inputs[0])
            self.assertIsInstance(
                self.curr_node().output_storage_alias[0], AliasesPriorGraphOutput
            )

            expected = out.detach().clone()
            expected_view = out_view.detach().clone()
            old_data_ptr = out.untyped_storage().data_ptr()
            self.save_live_outputs(live_outputs, out, out_view)

            record_out, record_view = run_pair(inputs[1])
            self.save_live_outputs(live_outputs, record_out, record_view)
            replay_out, replay_view = run_pair(inputs[2])
            self.save_live_outputs(live_outputs, replay_out, replay_view)
            next_out, next_view = run_pair(inputs[3])
            self.save_live_outputs(live_outputs, next_out, next_view)

            self.assertEqual(out, expected)
            self.assertEqual(out_view, expected_view)
            self.assertEqual(cdata(out), cdata(out_view))
            self.assertNotEqual(out.untyped_storage().data_ptr(), old_data_ptr)
            self.assertEqual(replay_view, replay_out[1:])
            self.assertEqual(next_view, next_out[1:])

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        def test_clone_live_output_storage_with_only_user_view_live(self):
            # [FUNCTION DOC] With user_visible cloning, deletes the base output and keeps only a
            # view of it alive through later generations. Checks the view keeps its values while
            # its storage is moved, and later replay outputs are correct.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the square of x.
                return x * x

            inputs = [
                torch.randn(4, 4, device="cuda", dtype=torch.float32) + i
                for i in range(4)
            ]
            live_outputs = []
            out = self.run_next_generation(foo, inputs[0])
            out_view = out[1:]
            expected_view = out_view.detach().clone()
            old_data_ptr = out_view.untyped_storage().data_ptr()
            del out
            self.save_live_outputs(live_outputs, out_view)

            record_out = self.run_next_generation(foo, inputs[1])
            self.save_live_outputs(live_outputs, record_out)
            replay_out = self.replay_next_generation(foo, inputs[2])
            self.save_live_outputs(live_outputs, replay_out)
            next_out = self.replay_next_generation(foo, inputs[3])
            self.save_live_outputs(live_outputs, next_out)

            self.assertEqual(out_view, expected_view)
            self.assertNotEqual(out_view.untyped_storage().data_ptr(), old_data_ptr)
            self.assertEqual(replay_out, inputs[2] * inputs[2])
            self.assertEqual(next_out, inputs[3] * inputs[3])

        def test_clone_live_output_used_as_next_generation_input(self):
            # [FUNCTION DOC] With user_visible cloning, checks outputs used as inputs to the
            # next generation. First a chain of three steps each feeding its output into the
            # next, then a replay chain, comparing against a manually computed expected
            # recurrence and checking storage was replaced. Then a producer/consumer pair is run
            # in the same generation (storage of the producer output unchanged) and across
            # generations (producer storage address changes).
            @torch.compile(mode="reduce-overhead")
            def step(x, bias):
                # [FUNCTION DOC] HELPER: Returns x scaled by 0.99 plus bias.
                return x * 0.99 + bias

            @torch.compile(mode="reduce-overhead")
            def producer(x):
                # [FUNCTION DOC] HELPER: Returns the square of x plus one.
                return x * x + 1

            @torch.compile(mode="reduce-overhead")
            def consumer(x):
                # [FUNCTION DOC] HELPER: Returns twice x minus three.
                return x * 2 - 3

            with config.patch(
                "triton.cudagraph_trees_generation_cloning", "user_visible"
            ):
                live_outputs = []
                inp = torch.randn(4, device="cuda", dtype=torch.float32)
                biases = [
                    torch.randn(4, device="cuda", dtype=torch.float32) * 0.01 + i
                    for i in range(5)
                ]
                x = inp
                expected = inp
                old_data_ptr = None
                for i in range(3):
                    x = self.run_next_generation(step, x, biases[i])
                    expected = expected * 0.99 + biases[i]
                    old_data_ptr = x.untyped_storage().data_ptr()
                    self.save_live_outputs(live_outputs, x)

                out = self.replay_next_generation(step, x, biases[3])
                expected_out = expected * 0.99 + biases[3]
                out_data_ptr = out.untyped_storage().data_ptr()
                self.save_live_outputs(live_outputs, out)
                next_out = self.replay_next_generation(step, out, biases[4])
                self.save_live_outputs(live_outputs, next_out)

                self.assertEqual(x, expected)
                self.assertNotEqual(x.untyped_storage().data_ptr(), old_data_ptr)
                self.assertEqual(out, expected_out)
                self.assertNotEqual(out.untyped_storage().data_ptr(), out_data_ptr)
                self.assertEqual(next_out, expected_out * 0.99 + biases[4])

                producer_inputs = [
                    torch.randn(8, device="cuda", dtype=torch.float32) + i
                    for i in range(5)
                ]
                for i in range(3):
                    torch.compiler.cudagraph_mark_step_begin()
                    producer_out = producer(producer_inputs[i])
                    producer_out_data_ptr = producer_out.untyped_storage().data_ptr()
                    consumer_out = consumer(producer_out)
                    self.assertEqual(
                        producer_out.untyped_storage().data_ptr(),
                        producer_out_data_ptr,
                    )
                    self.save_live_outputs(live_outputs, producer_out, consumer_out)

                torch.compiler.cudagraph_mark_step_begin()
                producer_out = producer(producer_inputs[3])
                producer_out_data_ptr = producer_out.untyped_storage().data_ptr()
                self.save_live_outputs(live_outputs, producer_out)

                torch.compiler.cudagraph_mark_step_begin()
                consumer_out = consumer(producer_out)
                self.save_live_outputs(live_outputs, consumer_out)
                self.assertNotEqual(
                    producer_out.untyped_storage().data_ptr(), producer_out_data_ptr
                )

                torch.compiler.cudagraph_mark_step_begin()
                self.save_live_outputs(live_outputs, producer(producer_inputs[4]))

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        @parametrize("graph_partition", (True, False))
        def test_clone_live_output_saved_for_backward_on_mark_step(
            self, graph_partition
        ):
            # [FUNCTION DOC] Parametrized by graph partition on/off with user_visible cloning.
            # Runs a compiled sigmoid with an input requiring grad over several generations
            # while keeping outputs alive, then checks the first output was cloned out of the
            # pool but still has correct values and that backward through it produces the
            # analytic sigmoid gradient.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid of x.
                return torch.sigmoid(x)

            with config.patch("graph_partition", graph_partition):
                live_outputs = []
                inp = torch.randn(
                    4, device="cuda", dtype=torch.float32
                ).requires_grad_()
                out = self.run_next_generation(foo, inp)
                expected = out.detach().clone()
                expected_grad = expected * (1 - expected)
                old_data_ptr = out.untyped_storage().data_ptr()
                self.save_live_outputs(live_outputs, out)

                inp2 = (
                    torch.randn(4, device="cuda", dtype=torch.float32) + 1
                ).requires_grad_()
                out2 = self.run_next_generation(foo, inp2)
                self.save_live_outputs(live_outputs, out2)
                inp3 = (
                    torch.randn(4, device="cuda", dtype=torch.float32) + 2
                ).requires_grad_()
                out3 = self.replay_next_generation(foo, inp3)
                self.save_live_outputs(live_outputs, out3)
                inp4 = (
                    torch.randn(4, device="cuda", dtype=torch.float32) + 3
                ).requires_grad_()
                out4 = self.replay_next_generation(foo, inp4)
                self.save_live_outputs(live_outputs, out4)

            self.assertEqual(out, expected)
            self.assertNotEqual(out.untyped_storage().data_ptr(), old_data_ptr)
            out.sum().backward()
            self.assertEqual(inp.grad, expected_grad)

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        def test_clone_live_output_saved_for_backward_with_only_user_view_live(self):
            # [FUNCTION DOC] With user_visible cloning, a compiled sigmoid output is deleted
            # while only a view of it stays alive across later generations. Checks the view
            # values are preserved and its storage address changed.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid of x.
                return torch.sigmoid(x)

            live_outputs = []
            inp = torch.randn(4, device="cuda", dtype=torch.float32).requires_grad_()
            out = self.run_next_generation(foo, inp)
            out_view = out[1:]
            expected_view = out_view.detach().clone()
            old_data_ptr = out_view.untyped_storage().data_ptr()
            del out
            self.save_live_outputs(live_outputs, out_view)

            inp2 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 1
            ).requires_grad_()
            out2 = self.run_next_generation(foo, inp2)
            self.save_live_outputs(live_outputs, out2)
            inp3 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 2
            ).requires_grad_()
            out3 = self.replay_next_generation(foo, inp3)
            self.save_live_outputs(live_outputs, out3)
            inp4 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 3
            ).requires_grad_()
            out4 = self.replay_next_generation(foo, inp4)
            self.save_live_outputs(live_outputs, out4)

            self.assertEqual(out_view, expected_view)
            self.assertNotEqual(out_view.untyped_storage().data_ptr(), old_data_ptr)

        @torch._inductor.config.patch(
            "triton.cudagraph_trees_generation_cloning", "user_visible"
        )
        def test_clone_live_view_output_saved_for_backward_on_mark_step(self):
            # [FUNCTION DOC] With user_visible cloning, a compiled function returns a view of a
            # sigmoid output. After several later generations the live output keeps its values
            # and its storage address changes.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the sigmoid of x without its first element.
                return torch.sigmoid(x)[1:]

            live_outputs = []
            inp = torch.randn(4, device="cuda", dtype=torch.float32).requires_grad_()
            out = self.run_next_generation(foo, inp)
            expected = out.detach().clone()
            old_data_ptr = out.untyped_storage().data_ptr()
            self.save_live_outputs(live_outputs, out)

            inp2 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 1
            ).requires_grad_()
            out2 = self.run_next_generation(foo, inp2)
            self.save_live_outputs(live_outputs, out2)
            inp3 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 2
            ).requires_grad_()
            out3 = self.replay_next_generation(foo, inp3)
            self.save_live_outputs(live_outputs, out3)
            inp4 = (
                torch.randn(4, device="cuda", dtype=torch.float32) + 3
            ).requires_grad_()
            out4 = self.replay_next_generation(foo, inp4)
            self.save_live_outputs(live_outputs, out4)

            self.assertEqual(out, expected)
            self.assertNotEqual(out.untyped_storage().data_ptr(), old_data_ptr)

        @parametrize("generation_cloning", (None, "user_visible"))
        def test_grad_accumulation_dealloc_error_message(self, generation_cloning):
            # [FUNCTION DOC] Parametrized by generation cloning mode (disabled or user_visible).
            # Runs a training iteration of a compiled linear layer, then a second iteration
            # without clearing grads; the second must raise a RuntimeError whose message
            # mentions a gradient tensor output of CUDAGraphs, gradient accumulation and the
            # grad tensor.
            model = torch.nn.Linear(10, 1, device="cuda")
            compiled_model = torch.compile(
                model, fullgraph=True, mode="reduce-overhead"
            )

            def run_iter():
                # [FUNCTION DOC] HELPER: Marks a new step, creates a random vector requiring
                # grad, runs the compiled model, clones, sums and clones the result and calls
                # backward on it.
                torch.compiler.cudagraph_mark_step_begin()
                x = torch.randn(
                    (10,), dtype=torch.float32, requires_grad=True, device="cuda"
                )
                loss = compiled_model(x).clone().sum().clone()
                loss.backward()

            with config.patch(
                {
                    "triton.cudagraph_trees_generation_cloning": generation_cloning,
                }
            ):
                run_iter()

                with self.assertRaises(RuntimeError) as exc:
                    run_iter()

            FileCheck().check("gradient tensor output of CUDAGraphs").check(
                "gradient accumulation"
            ).check(".grad tensor").run(str(exc.exception))

        def test_output_node_has_stack_traces_inference(self):
            # [FUNCTION DOC] Registers a post-grad custom pass that removes stack traces from
            # all non-placeholder, non-output nodes, then checks that the overwritten-output
            # error for a compiled inference function still contains the source expression of
            # the computation, taken from the stack traces stored on the output node.
            """Test that output_stack_traces on the output node provides
            stack traces even when a post-grad pass strips them from arg nodes
            in inference mode."""

            def strip_stack_traces(graph):
                # [FUNCTION DOC] HELPER: Graph pass that deletes the stack trace metadata entry
                # from every node that is not a placeholder or output; mutates the graph in
                # place.
                for node in graph.nodes:
                    if node.op not in ("placeholder", "output"):
                        node.meta.pop("stack_trace", None)

            with config.patch(post_grad_custom_post_pass=strip_stack_traces):

                @torch.compile(mode="reduce-overhead")
                def foo(x):
                    # [FUNCTION DOC] HELPER: Returns the cube of x.
                    return x * x * x

                inp = torch.rand([4], device="cuda")
                out = foo(inp).detach()
                out2 = foo(inp).detach()

                with self.assertRaises(Exception) as exc:
                    out + out

                self.assertIn("x * x * x", repr(exc.exception))

        def test_output_node_has_stack_traces_training(self):
            # [FUNCTION DOC] Training-mode counterpart of the inference test (forward/backward
            # split by the partitioner) with the same stripping pass.
            """Test that output_stack_traces on the output node provides
            stack traces even when a post-grad pass strips them from arg nodes
            in training mode (fwd/bwd graph split via partitioner)."""

            def strip_stack_traces(graph):
                # [FUNCTION DOC] HELPER: Graph pass that deletes the stack trace metadata entry
                # from every node that is not a placeholder or output; mutates the graph in
                # place.
                for node in graph.nodes:
                    if node.op not in ("placeholder", "output"):
                        node.meta.pop("stack_trace", None)

            with config.patch(post_grad_custom_post_pass=strip_stack_traces):

                @torch.compile(mode="reduce-overhead")
                def foo(x):
                    # [FUNCTION DOC] HELPER: Returns the cube of x.
                    return x * x * x

                inp = torch.rand([4], device="cuda", requires_grad=True)
                # Complete fwd+bwd to compile both graphs
                torch.compiler.cudagraph_mark_step_begin()
                foo(inp).sum().backward()

                # Now trigger the dealloc error
                torch.compiler.cudagraph_mark_step_begin()
                out = foo(inp).detach()
                torch.compiler.cudagraph_mark_step_begin()
                out2 = foo(inp).detach()

                with self.assertRaises(Exception) as exc:
                    out + out

                self.assertIn("x * x * x", repr(exc.exception))

        @unittest.skipIf(not torch.backends.cudnn.is_available(), "requires cudnn")
        def test_conv_benchmark(self):
            # [FUNCTION DOC] Skipped without cuDNN. With cuDNN benchmark mode enabled, compiles
            # and runs a Conv2d forward once to check cudagraph trees coexist with cuDNN
            # benchmarking (no assertion beyond not raising).
            with torch.backends.cudnn.flags(
                enabled=True, benchmark=True, deterministic=False
            ):
                m = torch.nn.Conv2d(5, 6, [3, 3]).cuda()
                inp = torch.randn([2, 5, 16, 16]).cuda()

                @torch.compile()
                def foo(m, inp):
                    # [FUNCTION DOC] HELPER: Applies the module to the input and returns the
                    # result.
                    return m(inp)

                foo(m, inp)

        def test_single_stream_use(self):
            # [FUNCTION DOC] Runs a compiled function with backward four times and checks that
            # the cudagraph segments newly created (compared to before) belong to exactly one
            # stream, and that a graph was recorded.
            @torch.compile()
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the relu of the cube of x.
                return (x * x * x).relu()

            inp = torch.rand([4], device="cuda", requires_grad=True)
            streams = set()
            streams_init = {seg["stream"] for seg in get_all_cudagraph_segments()}
            for _ in range(4):
                foo(inp).sum().backward()
                inp.grad = None

            streams = {
                seg["stream"] for seg in get_all_cudagraph_segments()
            } - streams_init
            self.assertEqual(len(streams), 1)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._dynamo.config.patch("assume_static_by_default", False)
        def test_dynamic_backward(self):
            # [FUNCTION DOC] With dynamic shapes by default, compiles a function that
            # concatenates the input with itself, does an addmm, relu and also returns the
            # dynamic size. For two different input shapes, compares forward result, size and
            # input gradient between eager and two compiled runs.
            def foo(x):
                # [FUNCTION DOC] HELPER: Concatenates x with itself, returns relu of addmm of
                # that with itself and the concatenated first dimension size.
                x = torch.cat([x, x])
                return torch.addmm(x, x, x).relu(), x.size(0)

            opt_foo = torch.compile(mode="reduce-overhead")(foo)

            def run_test(foo, inp):
                # [FUNCTION DOC] HELPER: Runs the function, backpropagates the sum of the
                # result, clones the input gradient and clears it, and returns the cloned
                # result, the size and the gradient.
                r, s = foo(inp)
                r.sum().backward()
                g = inp.grad.clone()
                inp.grad = None
                r = r.clone()
                return r, s, g

            def run_big_test(inp):
                # [FUNCTION DOC] HELPER: Computes results with eager once and compiled twice for
                # the input and asserts results, sizes and gradients all match.
                r0, s0, g0 = run_test(foo, inp)
                r1, s1, g1 = run_test(opt_foo, inp)
                r2, s2, g2 = run_test(opt_foo, inp)
                self.assertEqual(r0, r1)
                self.assertEqual(r0, r2)
                self.assertEqual(s0, s1)
                self.assertEqual(s0, s2)
                self.assertEqual(g0, g1)
                self.assertEqual(g0, g2)

            inp = torch.randn(2, 4, device="cuda", requires_grad=True)
            run_big_test(inp)

            inp = torch.randn(3, 6, device="cuda", requires_grad=True)
            run_big_test(inp)

        def test_dynamic_warmup(self):
            # [FUNCTION DOC] Uses the tree API with a function taking a size integer and a
            # tensor, counting real executions. The first call (warmup) and second (record)
            # execute the function while replay does not (count two). Changing the integer input
            # value shape requires a new warmup, so two more executions occur (count four).
            COUNTER = 0

            def f(inps):
                # [FUNCTION DOC] HELPER: Unpacks the integer and tensor from the input list,
                # clears the list, increments the outer execution counter and returns the tensor
                # doubled.
                i, x = inps
                inps.clear()
                nonlocal COUNTER
                COUNTER += 1
                return x * 2

            x = torch.randn(2, device="cuda")
            inp_list = [2, x]
            foo_cg = self.cudagraphify_impl(f, inp_list, ())
            foo_cg(inp_list)  # warmup
            foo_cg([2, x])  # record
            foo_cg([2, x])  # replay
            self.assertEqual(COUNTER, 2)

            # Switching the size will require a warmup again
            x = torch.randn(3, device="cuda")
            inp_list = [3, x]
            foo_cg(inp_list)  # warmup
            foo_cg([3, x])  # record
            foo_cg([3, x])  # replay
            self.assertEqual(COUNTER, 4)

        def test_mark_warmup_incomplete(self):
            # [FUNCTION DOC] Exercises the compiler API that marks warmup as incomplete. Calling
            # it with no manager is harmless. A stable function establishes one warmed-up
            # function entry. A second function that calls the API in specific invocations (the
            # first two and the fourth) is run: the stable entries are unchanged while warmup is
            # incomplete, it becomes a second warmed function on the third call, state moves to
            # RECORDING then EXECUTION on later calls, and marking incomplete outside a function
            # does not alter the warmed set. Total function executions must be four.
            mark_warmup_incomplete = torch.compiler.cudagraph_mark_warmup_incomplete
            mark_warmup_incomplete()
            self.assertIsNone(self.get_manager())

            def stable(inps):
                # [FUNCTION DOC] HELPER: Pops the first tensor from the list, clears it and
                # returns a one-element tuple with x plus one.
                x = inps[0]
                inps.clear()
                return (x + 1,)

            x = torch.randn(2, device="cuda")
            stable_cg = self.cudagraphify_impl(stable, [x], ())
            stable_out = stable_cg([x])
            manager = self.get_manager()
            stable_functions = manager.warmed_up_functions.copy()
            self.assertEqual(len(stable_functions), 1)
            del stable_out

            calls = 0

            def f(inps):
                # [FUNCTION DOC] HELPER: Counts calls in an outer counter, pops the first tensor
                # from the list, clears it, calls the mark-warmup-incomplete API on the first
                # two calls and the fourth, and returns a one-element tuple with x plus one.
                nonlocal calls
                calls += 1
                x = inps[0]
                inps.clear()
                if calls < 3 or calls == 4:
                    mark_warmup_incomplete()
                return (x + 1,)

            foo_cg = self.cudagraphify_impl(f, [x], ())

            for _ in range(2):
                out = foo_cg([x])
                self.assertEqual(manager.warmed_up_functions, stable_functions)
                del out

            out = foo_cg([x])
            warmed_functions = manager.warmed_up_functions.copy()
            self.assertEqual(len(warmed_functions), 2)
            mark_warmup_incomplete()
            self.assertEqual(manager.warmed_up_functions, warmed_functions)
            del out

            out = foo_cg([x])
            self.assertEqual(manager.path_state, ExecutionState.RECORDING)
            self.assertEqual(manager.warmed_up_functions, warmed_functions)
            del out

            out = foo_cg([x])
            self.assertEqual(manager.path_state, ExecutionState.EXECUTION)
            self.assertEqual(calls, 4)
            mark_warmup_incomplete()
            self.assertEqual(manager.warmed_up_functions, warmed_functions)
            del out

        def test_forward_generation(self):
            # [FUNCTION DOC] Chains two compiled functions with an input requiring grad. Checks
            # two live blocks, forwards running with pending backwards before backward and none
            # after; runs a second full forward and backward, then a forward on a detached
            # input, which leaves no pending backward, and a graph was recorded.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            def foo2(x):
                # [FUNCTION DOC] HELPER: Returns x times twelve.
                return x * 12

            foo_opt = torch.compile(foo)
            foo2_opt = torch.compile(foo2)
            ones = torch.ones([4, 4], device="cuda", requires_grad=True)

            out = foo_opt(ones)
            out2 = foo2_opt(out)

            self.assertEqual(all_live_block_count(), 2)

            self.assertTrue(self.get_manager().running_forwards_with_pending_backwards)

            out2.sum().backward()
            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            ones.grad = None
            del out
            del out2

            foo2_opt(foo_opt(ones)).sum().backward()

            out = foo_opt(ones.detach())
            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        def test_warn_on_pending_backward(self):
            # [FUNCTION DOC] Runs a compiled function on inputs requiring grad without calling
            # backward, so the third call happens with pending backward. Collects warnings and
            # asserts one of them mentions that backward is required, and that the next graph id
            # is still zero (nothing recorded).
            @torch.compile
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            warnings.resetwarnings()
            with warnings.catch_warnings(record=True) as w:
                out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            # Match substring only; scan all warnings in case another warning fires first.
            msgs = [str(x.message) for x in w]
            self.assertTrue(
                any("require backward" in m for m in msgs),
                lambda msg: f"{msg}\nexpected CUDAGraph pending-backward warning; got: {msgs}",
            )
            self.assertTrue(self.get_manager().new_graph_id().id == 0)

        def test_mark_step(self):
            # [FUNCTION DOC] Calls the mark step API before each of two runs of a compiled
            # function with inputs requiring grad, and checks a graph was recorded (non-zero
            # next graph id).
            @torch.compile
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            torch.compiler.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))

            torch.compiler.cudagraph_mark_step_begin()
            out = foo(torch.rand([4, 4], device="cuda", requires_grad=True))
            self.assertFalse(self.get_manager().new_graph_id().id == 0)

        @torch._dynamo.config.patch("capture_scalar_outputs", True)
        def test_incompatible_cudagraph_ops_item(self):
            # [FUNCTION DOC] With scalar output capture on, compiles a function calling the item
            # method of a tensor. Results equal the inputs. With graph partition the scheduler
            # log must show a reason of CPU ops for the dynamic scalar; otherwise the
            # cudagraph_utils log must report skipping due to an incompatible op.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the scalar value of the tensor x.
                return x.item()

            # NB: This doesn't work with float, because float unbacked codegen
            # is currently broken.  But testing the float case here is also
            # awkward, because we plan to Tensor-ify the float compute, and as
            # a result we'd actually expect this to work with cuda graphs!
            scheduler_log_stream, scheduler_ctx = logs_to_string(
                "torch._inductor.scheduler", "cudagraphs"
            )
            utils_log_stream, utils_ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with scheduler_ctx(), utils_ctx():
                self.assertEqual(foo(torch.tensor(3, device="cuda")), 3)
                self.assertEqual(foo(torch.tensor(6, device="cuda")), 6)

            if config.graph_partition:
                FileCheck().check(
                    "reason=cpu ops, ir=DynamicScalar",
                ).run(scheduler_log_stream.getvalue())
            else:
                FileCheck().check(
                    "skipping cudagraphs due to disabling cudagraphs due to incompatible op"
                ).run(utils_log_stream.getvalue())

        @torch._dynamo.config.patch("compiled_autograd", True)
        def test_compiled_autograd_static_input_params(self):
            # [FUNCTION DOC] With compiled autograd on, runs a compiled backward call of a
            # linear layer five times and checks the counter of recorded non-static inputs
            # equals two (copies only happen on the recording step, with static parameter inputs
            # avoided).
            @torch.compile(mode="reduce-overhead")
            def bwd(loss):
                # [FUNCTION DOC] HELPER: Calls backward on the given loss.
                loss.backward()

            model = torch.nn.Linear(10, 10, bias=False, device="cuda")
            x = torch.randn(10, 10, device="cuda")
            for _ in range(5):
                out = model(x)
                bwd(out.sum())
                model.weight.grad = None

            # i=0, 0 copies (warmup)
            # i=1, 2 copies (record, 1/3 inputs marked as static)
            # i>1, 0 copies (run)
            self.assertEqual(
                counters["inductor"]["cudagraph_recorded_non_static_inputs"], 2
            )

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero(self):
            # [FUNCTION DOC] With dynamic output shape capture on, compiles a nonzero call and
            # checks results for two inputs. With graph partition the scheduler log shows an
            # unbacked binding ops reason; otherwise the cudagraph_utils log reports skipping
            # due to an incompatible op.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the indices of nonzero elements of x.
                return x.nonzero()

            scheduler_log_stream, scheduler_ctx = logs_to_string(
                "torch._inductor.scheduler", "cudagraphs"
            )
            utils_log_stream, utils_ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with scheduler_ctx(), utils_ctx():
                self.assertEqual(
                    foo(torch.tensor([1, 0, 2], device="cuda")),
                    torch.tensor([[0], [2]]),
                )
                self.assertEqual(
                    foo(torch.tensor([1, 0, 0], device="cuda")), torch.tensor([[0]])
                )

            if config.graph_partition:
                FileCheck().check("reason=unbacked binding ops").check("foo").run(
                    scheduler_log_stream.getvalue()
                )
            else:
                FileCheck().check(
                    "skipping cudagraphs due to disabling cudagraphs due to incompatible op"
                ).run(utils_log_stream.getvalue())

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero_graph_breaks(self):
            # [FUNCTION DOC] Runs a compiled function twice with a graph break between two
            # nonzero calls. With graph partition the cudagraph partitions counter must be zero;
            # without it the cudagraph skips counter must be three.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Takes nonzero indices of x, graph breaks, then returns
                # the nonzero indices of that result.
                y = x.nonzero()  # skip
                torch._dynamo.graph_break()
                return y.nonzero()  # skip 2 times (due to recompile)

            foo(torch.tensor([1, 0, 2], device="cuda"))
            foo(torch.tensor([1, 0, 0], device="cuda"))

            if config.graph_partition:
                self.assertEqual(counters["inductor"]["cudagraph_partitions"], 0)
            else:
                # Without graph partitioning, cudagraphs are skipped entirely
                self.assertEqual(counters["inductor"]["cudagraph_skips"], 3)

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        def test_incompatible_cudagraph_ops_nonzero_backend(self):
            # [FUNCTION DOC] Compiles a nonzero function with the cudagraphs backend and checks
            # outputs, the log message about skipping because of the incompatible nonzero op,
            # and that the skip counter is one.
            @torch.compile(backend="cudagraphs")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the indices of nonzero elements of x.
                return x.nonzero()

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                self.assertEqual(
                    foo(torch.tensor([1, 0, 2], device="cuda")),
                    torch.tensor([[0], [2]]),
                )
                self.assertEqual(
                    foo(torch.tensor([1, 0, 0], device="cuda")), torch.tensor([[0]])
                )

            FileCheck().check(
                "skipping cudagraphs due to incompatible op (nonzero)"
            ).run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @torch._dynamo.config.patch("capture_dynamic_output_shape_ops", True)
        @torch._inductor.config.patch({"cpp_wrapper": True, "graph_partition": True})
        def test_skip_cpp_wrapper(self):
            # [FUNCTION DOC] With the C++ wrapper and graph partition both enabled, a compiled
            # add must match eager, the log must state cudagraphs are skipped because the C++
            # wrapper does not support graph partition yet, and the skip counter must be one.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            foo_c = torch.compile(mode="reduce-overhead")(foo)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                t = torch.rand([32], device="cuda")
                self.assertEqual(foo(t), foo_c(t))

            FileCheck().check(
                "skipping cudagraphs due to cpp-wrapper does not support graph partition yet"
            ).run(log_stream.getvalue())
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        def test_storage_access_error(self):
            # [FUNCTION DOC] Sets a custom storage access error message on a tensor via the C++
            # binding and checks that accessing the untyped storage raises an exception
            # containing that message.
            x = torch.rand([4], device="cuda")
            torch._C._set_storage_access_error_msg(x, "custom error msg")

            with self.assertRaisesRegex(Exception, "custom error msg"):
                device = x.untyped_storage()

        def test_clear_storage_data_ptr_access_error(self):
            # [FUNCTION DOC] Sets a data pointer access error message on a storage whose size
            # was reset to zero and checks reading its data pointer raises with that message.
            # After clearing the message and resizing back to the original byte size, the data
            # pointer is readable again.
            x = torch.rand([4], device="cuda")
            storage = x.untyped_storage()
            storage_ptr = storage.data_ptr()
            storage_impl_ptr = storage._cdata

            storage.resize_(0)

            torch._C._set_storage_data_ptr_access_error_msg(
                storage_impl_ptr, "storage is dead"
            )
            with self.assertRaisesRegex(Exception, "storage is dead"):
                storage.data_ptr()

            torch._C._clear_storage_data_ptr_access_error_msg(storage_impl_ptr)
            storage.resize_(4 * x.element_size())
            # Should not raise
            storage.data_ptr()

        def test_side_stream_memory_allocation(self):
            # [FUNCTION DOC] Records a function that allocates one buffer on a side stream and
            # one on the main stream (using the given dimensions) through the tree API with
            # three iterations. Both outputs must equal a ones reference every iteration, and
            # exactly one graph must have been recorded.
            device = f"cuda:{self.device_idx}"

            def multi_stream_allocation(args):
                # [FUNCTION DOC] HELPER: Creates a side stream synchronized with the current
                # one, allocates a ones buffer in it, makes the current stream wait for it,
                # allocates a ones buffer on the main stream, clears the argument list if it is
                # a list, and returns both buffers.
                side_stream = torch.cuda.Stream()
                side_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side_stream):
                    side_stream_buffer = torch.ones(
                        *args, device=device, dtype=torch.float32
                    )
                torch.cuda.current_stream().wait_stream(side_stream)

                main_stream_buffer = torch.ones(
                    *args, device=device, dtype=torch.float32
                )

                if isinstance(args, list):
                    args.clear()

                return main_stream_buffer, side_stream_buffer

            graphed_multi_stream_func = tree_cudagraphify_impl(
                multi_stream_allocation,
                inputs=[],
                static_input_idxs=[],
                is_backward=False,
                is_inference=False,
                device_index=self.device_idx,
                stack_traces=["dummy stack trace1", "dummy stack trace2"],
            )

            ref_out = torch.ones((2, 3), device=device, dtype=torch.float32)

            for _ in range(3):
                torch.compiler.cudagraph_mark_step_begin()
                main_stream_buffer, side_stream_buffer = graphed_multi_stream_func(
                    [2, 3]
                )
                self.assertEqual(main_stream_buffer, ref_out)
                self.assertEqual(side_stream_buffer, ref_out)

            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        def _run_iter(self, param, fn):
            # [FUNCTION DOC] HELPER: Runs the function on a 2 by 2 ones input and the parameter,
            # backpropagates the sum of the output, copies the parameter gradient, clears it and
            # returns the output and the copied gradient.
            fwd_output = fn(torch.ones(2, 2), param)
            fwd_output.sum().backward()
            grad_output = param.grad.detach().clone()
            param.grad = None
            return fwd_output, grad_output

        def _assert_equal_multi_loop(self, param, fn_eager, fn_compiled):
            # [FUNCTION DOC] HELPER: Runs one forward and backward iteration with the eager
            # function to get an expected output and gradient, then five iterations with the
            # compiled function, asserting each output and gradient equals the expected values.
            exp_output, exp_grad = self._run_iter(param, fn_eager)
            for _ in range(5):
                compiled_output, compiled_grad = self._run_iter(param, fn_compiled)
                self.assertEqual(exp_output, compiled_output)
                self.assertEqual(exp_grad, compiled_grad)

        def run_static_input_param_test(self, fn_eager, num_graphs):
            # [FUNCTION DOC] HELPER: Compiles the function with reduce-overhead on CUDA as
            # default device and checks parameter-input dispatch: runs the multi-loop check with
            # a first parameter, a second parameter, and the first again (which must reuse its
            # recording), then asserts the number of recorded graphs equals num_graphs.
            with torch.device("cuda"):
                fn_compiled = torch.compile(fn_eager, mode="reduce-overhead")

                p1 = torch.nn.Parameter(torch.rand([2, 2]))
                self._assert_equal_multi_loop(p1, fn_eager, fn_compiled)

                p2 = torch.nn.Parameter(torch.rand([2, 2]))
                self._assert_equal_multi_loop(p2, fn_eager, fn_compiled)

                # Run p1 again to ensure we reuse the previous recording
                self._assert_equal_multi_loop(p1, fn_eager, fn_compiled)

                self.assertEqual(self.get_manager().new_graph_id().id, num_graphs)

        def _module_test(self, mod, name="weight", param_wrapping=True):
            # [FUNCTION DOC] HELPER: Shared module test. Compiles a function calling the module
            # with fullgraph, checks five compiled forward and backward runs against eager, then
            # swaps the named attribute (wrapping in a parameter if requested) for a random one
            # and checks again, then restores the original and checks again (reusing the first
            # recording). Finally asserts four graphs were recorded (forward and backward per
            # version).
            with torch.device("cuda"):

                def fn(x, mod):
                    # [FUNCTION DOC] HELPER: Calls the module on the input.
                    return mod(x)

                fn_compiled = torch.compile(fn, mode="reduce-overhead", fullgraph=True)

                def run_test_iter(mod, fn):
                    # [FUNCTION DOC] HELPER: Runs the function on a 2 by 2 ones input and the
                    # module, backpropagates the sum, copies the weight gradient, zeroes module
                    # grads and returns output and the gradient copy.
                    fwd_output = fn(torch.ones(2, 2), mod)
                    fwd_output.sum().backward()
                    grad_output = mod.weight.grad.detach().clone()
                    mod.zero_grad()
                    return fwd_output, grad_output

                def run_test():
                    # [FUNCTION DOC] HELPER: Compares one eager iteration against five compiled
                    # iterations for both output and weight gradient.
                    exp_output, exp_grad = run_test_iter(mod, fn)
                    for _ in range(5):
                        compiled_output, compiled_grad = run_test_iter(mod, fn_compiled)
                        self.assertEqual(exp_output, compiled_output)
                        self.assertEqual(exp_grad, compiled_grad)

                run_test()
                old_attr = getattr(mod, name)
                modified_attr = torch.rand_like(old_attr)
                if param_wrapping:
                    modified_attr = torch.nn.Parameter(modified_attr)
                setattr(mod, name, modified_attr)
                run_test()
                # Run original version to verify we reuse the other recording
                setattr(mod, name, old_attr)
                run_test()

                # Fwd + bwd graphs for each version of the function => 4 graphs
                self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_single_compile_param_inputs(self):
            # [FUNCTION DOC] With recompilation treated as an error, checks one compiled
            # multiply function can record separate cudagraphs for different parameter inputs,
            # expecting four graphs in total.
            # Verify that we can record multiple cudagraphs for a single
            # compiled function with param inputs
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the product of x and y.
                return x * y

            # Fwd + bwd graphs for each version of the function => 4 graphs
            self.run_static_input_param_test(fn, 4)

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_single_compile_builtin_module(self):
            # [FUNCTION DOC] With recompilation as an error, replacing the weight of a built-in
            # Linear module must not recompile and must record new cudagraphs (four graphs).
            # Verify that we don't recompile when changing the param of a builtin module
            # and that we record another cudagraph
            # Note: Linear is a builtin module so we enable that config setting above
            self._module_test(torch.nn.Linear(2, 3, device="cuda"))

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_single_compile_builtin_module_buffers(self):
            # [FUNCTION DOC] With recompilation as an error, replacing the running mean buffer
            # of a built-in BatchNorm module must not recompile and must record new cudagraphs.
            # Verify that we don't recompile when changing the buffer of a builtin module
            # and that we record another cudagraph
            self._module_test(
                torch.nn.BatchNorm1d(2, device="cuda"),
                name="running_mean",
                param_wrapping=False,
            )

        @torch._inductor.config.patch("triton.cudagraphs", True)
        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_custom_module(self):
            # [FUNCTION DOC] With cudagraphs on and recompilation as an error, replacing the
            # parameter of a custom module must dispatch to a new cudagraph without recompiling.
            # Test that we can correctly dispatch multiple graphs
            # if params of a custom module change
            class TestModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Custom module storing a given parameter as weight; forward
                # multiplies the weight by the input.
                def __init__(self, param) -> None:
                    # [FUNCTION DOC] HELPER: Stores the provided parameter as the weight
                    # attribute.
                    super().__init__()
                    self.weight = param

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the weight times x.
                    return self.weight * x

            self._module_test(
                TestModule(torch.nn.Parameter(torch.rand([2, 2], device="cuda")))
            )

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_custom_module_buffer(self):
            # [FUNCTION DOC] Like the custom module test but replaces a buffer of a custom
            # module (not wrapped as a parameter) and expects correct dispatch without
            # recompiling.
            # Test that we can correctly dispatch multiple graphs
            # if buffers of a custom module change
            class TestModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Custom module with a weight parameter and a registered
                # buffer; forward returns x times the weight plus the buffer.
                def __init__(self, param, buf) -> None:
                    # [FUNCTION DOC] HELPER: Stores the parameter as weight and wraps the given
                    # tensor as a buffer.
                    super().__init__()
                    self.weight = param
                    self.buf = torch.nn.Buffer(buf)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns x times the weight plus the buffer.
                    return x * self.weight + self.buf

            self._module_test(
                TestModule(
                    torch.nn.Parameter(torch.rand([2, 2], device="cuda")),
                    torch.rand([2, 2], device="cuda"),
                ),
                name="buf",
                param_wrapping=False,
            )

        @torch._inductor.config.patch("triton.cudagraphs", True)
        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_child_node(self):
            # [FUNCTION DOC] Checks dispatch when a child node of the tree sees changed stable
            # input pointers: a function with a graph break, run with two different parameters,
            # must record five graphs (one shared first graph, two second graphs, two backward
            # graphs).
            # Test that we can correctly dispatch multiple graphs if a child node
            # in the tree has stable input pointers change
            def fn(x, p):
                # [FUNCTION DOC] HELPER: Squares x, graph breaks, then multiplies by p.
                # Graph 1
                y = x * x
                torch._dynamo.graph_break()
                # Graph 2
                return y * p

            # We have 5 graphs here
            #            Graph 1
            #       /                \
            # Graph 2 w/ p1     Graph 2 w/ p2
            # and then two backward graphs
            self.run_static_input_param_test(fn, 5)

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_multi_dispatch_parent_node(self):
            # [FUNCTION DOC] Checks dispatch when the parent node sees different parameters: a
            # function with a graph break run with two parameters must record six graphs (two
            # versions of the first graph, two re-recorded second graphs and two backward
            # graphs).
            def fn(x, p):
                # [FUNCTION DOC] HELPER: Multiplies x by p, graph breaks, then adds x to the
                # result.
                # Graph 1
                y = x * p
                torch._dynamo.graph_break()
                # Graph 2
                return y + x

            # We have 6 graphs here
            #    Graph 1 w/ p1    Graph 1 w/ p2
            #          |                |
            #     Graph 2 (v1)     Graph 2 (v2)
            # There are two versions of graph 2 because
            # we re-record due to different memory state after running the
            # two versions of Graph 1
            # and then two backward graphs
            self.run_static_input_param_test(fn, 6)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 1)
        def test_not_fallback_to_eager_if_have_not_recompiling_too_many_times(self):
            # [FUNCTION DOC] With a re-record limit of one, running with two different
            # parameters records four graphs and causes no cudagraph skips (a legitimate
            # re-record does not trip the fallback to eager).
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the product of x and y.
                return x * y

            # We have 4 graphs here
            #             None
            #       /                           \
            # (fwd w/ p1, Graph 0)            (fwd w/p2, Graph2)
            # (bwd w/ p1, Graph 1)            (bwd w/p2, Graph3)
            self.run_static_input_param_test(fn, 4)
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        @torch._dynamo.config.patch("error_on_recompile", True)
        @torch._inductor.config.patch("triton.cudagraph_unexpected_rerecord_limit", 1)
        def test_param_ptr_churn_does_not_count_toward_rerecord_limit(self):
            # [FUNCTION DOC] With a re-record limit of one, runs a compiled multiply with five
            # distinct new parameters. Parameter pointer changes cause re-records but are not
            # counted: no cudagraph skips and every per-node re-record counter remains zero.
            # Distinct nn parameter pointers across calls trigger
            # StaticInputIdxMismatch each time. Those mismatches re-record but
            # must not count against cudagraph_unexpected_rerecord_limit, since
            # parameter identity churn is expected under inline_inbuilt_nn_modules.
            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns the product of x and y.
                return x * y

            with torch.device("cuda"):
                fn_compiled = torch.compile(fn, mode="reduce-overhead")
                # Burn through more distinct params than the limit (1) allows
                # if they were counted.
                for _ in range(5):
                    p = torch.nn.Parameter(torch.rand([2, 2]))
                    self._assert_equal_multi_loop(p, fn, fn_compiled)

            # No fallback to eager triggered by re-record limit.
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)
            # Counter never bumped for any (node, function).
            for fn_to_count in self.get_manager().num_rerecord.values():
                for count in fn_to_count.values():
                    self.assertEqual(count, 0)

        @torch._dynamo.config.patch("error_on_recompile", True)
        def test_no_rerecord_with_mark_static_address(self):
            # [FUNCTION DOC] Compiles a function using a module and an input marked as static
            # address, with recompilation as an error. Over five iterations with new non-static
            # inputs, only two graphs (one forward, one backward) are recorded, i.e. no
            # re-recording is caused by changing non-static inputs.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module wrapping a single 2 by 2 linear layer; forward
                # applies it.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = nn.Linear(2, 2)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                    return self.linear(x)

            mod = Mod().cuda()

            def fn_eager(x, marked_static_y):
                # [FUNCTION DOC] HELPER: Returns the cosine of x plus the module applied to the
                # static input.
                return torch.cos(x) + mod(marked_static_y)

            with torch.device("cuda"):
                fn_compiled = torch.compile(fn_eager, mode="reduce-overhead")

                # y is marked static
                y = torch.randn(2, 2)
                torch._dynamo.mark_static_address(y)

                # Chanhing pointer of x should not lead to re-records
                for _ in range(5):
                    x = torch.randn(2, 2, requires_grad=True)
                    res = fn_compiled(x, y)
                    res.sum().backward()
                    x.grad = None
                    mod.linear.weight.grad = None
                    mod.linear.bias.grad = None
                # One forward and one backward
                self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_cudagraph_annotation_disable(self):
            # [FUNCTION DOC] Checks the override_cudagraphs decorator with both forward and
            # backward disabled on a helper: output is correct and no tree manager is created.
            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def helper(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            inp = torch.randn(4, 4, device="cuda")
            fn_compiled = torch.compile(fn, mode="reduce-overhead")
            result = fn_compiled(inp)

            self.assertEqual(result, inp + 1)
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_disable_fwd_bwd(self):
            # [FUNCTION DOC] Same decorator with forward and backward disabled, on a function
            # with grad: runs forward and backward, checks correctness and that no tree manager
            # is created.
            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def helper(x):
                # [FUNCTION DOC] HELPER: Returns the sine of the square of x.
                return (x * x).sin()

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            inp = torch.randn(4, 4, device="cuda", requires_grad=True)
            fn_compiled = torch.compile(fn, mode="reduce-overhead")
            out = fn_compiled(inp)
            out.sum().backward()

            self.assertEqual(out, inp.detach().pow(2).sin())
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_disable_bwd_only(self):
            # [FUNCTION DOC] Decorator disabling only backward: three forward and backward
            # iterations; output is correct, a manager exists and exactly one graph (the
            # forward) is recorded.
            @torch._dynamo.override_cudagraphs(bwd=False)
            def helper(x):
                # [FUNCTION DOC] HELPER: Returns the sine of the square of x.
                return (x * x).sin()

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            fn_compiled = torch.compile(fn, mode="reduce-overhead")

            # Run forward+backward pair for cudagraph warmup and recording
            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out = fn_compiled(inp)
                out.sum().backward()

            self.assertEqual(out, inp.detach().pow(2).sin())
            # Forward is eligible for cudagraphs (manager created)
            self.assertIsNotNone(self.get_manager())
            # One graph for forward (backward is disabled)
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        def test_cudagraph_annotation_disable_fwd_only(self):
            # [FUNCTION DOC] Decorator disabling only forward: correctness checked, and no
            # manager is created because the forward override also disables the backward (the
            # two share one flag object).
            @torch._dynamo.override_cudagraphs(fwd=False)
            def helper(x):
                # [FUNCTION DOC] HELPER: Returns the sine of the square of x.
                return (x * x).sin()

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            inp = torch.randn(4, 4, device="cuda", requires_grad=True)
            fn_compiled = torch.compile(fn, mode="reduce-overhead")
            out = fn_compiled(inp)
            out.sum().backward()

            self.assertEqual(out, inp.detach().pow(2).sin())
            # fwd override also disables bwd since the BoxedBool is shared
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_graph_break_selective(self):
            # [FUNCTION DOC] Annotated function inlined into the first and third graph segments
            # (split by graph breaks) disables cudagraphs there, while the middle segment still
            # uses cudagraphs. Checks the output and that exactly one graph is recorded.
            """Annotated function only disables cudagraphs for graph segments
            that inline it. Other segments after a graph break are unaffected."""

            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def disabled_fn(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs disabled) function returning x
                # plus one.
                return x + 1

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the annotated function, graph breaks, doubles,
                # graph breaks and calls the annotated function again.
                # g1: disabled_fn inlined -> cudagraphs OFF
                y = disabled_fn(x)
                torch._dynamo.graph_break()
                # g2: no annotation -> cudagraphs ON
                z = y * 2
                torch._dynamo.graph_break()
                # g3: disabled_fn inlined -> cudagraphs OFF
                w = disabled_fn(z)
                return w

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            # Warmup + record
            out = compiled(inp)
            out = compiled(inp)

            expected = (inp + 1) * 2 + 1
            self.assertEqual(out, expected)
            # Only g2 uses cudagraphs (1 graph recorded)
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        def test_cudagraph_annotation_multiple_functions(self):
            # [FUNCTION DOC] Two different annotated functions in separate graph segments both
            # disable their segments; output is correct and no manager exists.
            """Multiple annotated functions each disable their containing graph."""

            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def no_cg_add(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs disabled) function returning x
                # plus one.
                return x + 1

            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def no_cg_mul(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs disabled) function returning x
                # times three.
                return x * 3

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the first annotated function, graph breaks, then
                # calls the second on the result.
                y = no_cg_add(x)
                torch._dynamo.graph_break()
                z = no_cg_mul(y)
                return z

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            out = compiled(inp)
            out = compiled(inp)

            expected = (inp + 1) * 3
            self.assertEqual(out, expected)
            # Both segments disabled, no cudagraphs
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_deeply_nested(self):
            # [FUNCTION DOC] An annotation on the innermost of three nested calls disables
            # cudagraphs for the whole graph; output is correct and no manager exists.
            """Annotation on an innermost function propagates through
            nested inlining to disable cudagraphs for the whole graph."""

            @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
            def inner(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs disabled) function returning x
                # plus one.
                return x + 1

            def middle(x):
                # [FUNCTION DOC] HELPER: Returns twice the result of the inner function.
                return inner(x) * 2

            def outer(x):
                # [FUNCTION DOC] HELPER: Returns the sine of the middle function result.
                return middle(x).sin()

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the outer function.
                return outer(x)

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            out = compiled(inp)

            self.assertEqual(out, ((inp + 1) * 2).sin())
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_context_manager(self):
            # [FUNCTION DOC] The override context manager disables cudagraphs for all graph
            # segments inside its block (including across graph breaks) but not outside; two
            # graphs (the segments before and after) are recorded and the output is correct.
            """Context manager disables cudagraphs for all graphs within block,
            including across graph breaks."""

            def model(x):
                # [FUNCTION DOC] HELPER: Adds one, graph breaks, enters the cudagraph override
                # context to double, graph break and subtract one, leaves, graph breaks and adds
                # three.
                y = x + 1  # g1: outside context, cudagraphs ON
                torch._dynamo.graph_break()
                with torch._dynamo.override_cudagraphs(fwd=False, bwd=False):
                    z = y * 2  # g2: inside context, cudagraphs OFF
                    torch._dynamo.graph_break()
                    w = z - 1  # g3: still inside context, cudagraphs OFF
                torch._dynamo.graph_break()
                return w + 3  # g4: outside context, cudagraphs ON

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            out = compiled(inp)
            out = compiled(inp)

            expected = ((inp + 1) * 2 - 1) + 3
            self.assertEqual(out, expected)
            # g1 and g4 use cudagraphs (2 graphs recorded)
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_cudagraph_annotation_module_forward(self):
            # [FUNCTION DOC] An annotation on a module forward method disables cudagraphs for
            # graphs calling that module; output is correct and no manager exists.
            """Annotation on a module's forward method disables cudagraphs
            when that module is called inside a compiled function."""

            class MyModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Module whose forward is annotated to disable cudagraphs
                # and returns x plus one.
                @torch._dynamo.override_cudagraphs(fwd=False, bwd=False)
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns x plus one.
                    return x + 1

            mod = MyModule().cuda()

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the module and doubles its result.
                # g1: mod.forward has annotation -> cudagraphs OFF
                y = mod(x)
                return y * 2

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            out = compiled(inp)
            out = compiled(inp)

            self.assertEqual(out, (inp + 1) * 2)
            self.assertIsNone(self.get_manager())

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable(self):
            # [FUNCTION DOC] With the cudagraphs config off, an annotated helper forcing forward
            # and backward cudagraphs on still gets cudagraphs: after three steps output is
            # correct, a manager exists and one graph is recorded.
            """override_cudagraphs(fwd=True, bwd=True) force-enables cudagraphs
            even when config.triton.cudagraphs is False."""

            @torch._dynamo.override_cudagraphs(fwd=True, bwd=True)
            def helper(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs forced on) function returning x
                # plus one.
                return x + 1

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            fn_compiled = torch.compile(fn)

            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda")
                torch.compiler.cudagraph_mark_step_begin()
                out = fn_compiled(inp)

            self.assertEqual(out, inp + 1)
            self.assertIsNotNone(self.get_manager())
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable_fwd_bwd(self):
            # [FUNCTION DOC] With cudagraphs config off, the forced-on annotation enables
            # cudagraphs for both forward and backward of a function with grad: two graphs are
            # recorded and the output is correct.
            """override_cudagraphs(fwd=True, bwd=True) force-enables cudagraphs
            for both forward and backward when config disables them."""

            @torch._dynamo.override_cudagraphs(fwd=True, bwd=True)
            def helper(x):
                # [FUNCTION DOC] HELPER: Annotated (cudagraphs forced on) function returning
                # sine of x squared.
                return (x * x).sin()

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            fn_compiled = torch.compile(fn)

            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out = fn_compiled(inp)
                out.sum().backward()

            self.assertEqual(out, inp.detach().pow(2).sin())
            self.assertIsNotNone(self.get_manager())
            # One forward and one backward
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable_fwd_only(self):
            # [FUNCTION DOC] With cudagraphs config off, an annotation forcing only forward on
            # also enables backward (shared flag object): two graphs recorded and output
            # correct.
            """override_cudagraphs(fwd=True) force-enables cudagraphs for
            forward. Backward is also enabled since the BoxedBool is shared."""

            @torch._dynamo.override_cudagraphs(fwd=True)
            def helper(x):
                # [FUNCTION DOC] HELPER: Annotated (forward forced on) function returning sine
                # of x squared.
                return (x * x).sin()

            def fn(x):
                # [FUNCTION DOC] HELPER: Calls the annotated helper.
                return helper(x)

            fn_compiled = torch.compile(fn)

            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out = fn_compiled(inp)
                out.sum().backward()

            self.assertEqual(out, inp.detach().pow(2).sin())
            self.assertIsNotNone(self.get_manager())
            # fwd override also enables bwd since the BoxedBool is shared
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable_context_manager(self):
            # [FUNCTION DOC] With cudagraphs config off, the override context manager enables
            # cudagraphs only for graph segments inside its block; two graphs are recorded and
            # the output is correct.
            """Context manager force-enables cudagraphs for segments within
            the block when config disables them."""

            def model(x):
                # [FUNCTION DOC] HELPER: Adds one, graph breaks, inside the enable context
                # doubles, graph breaks and subtracts one, then graph breaks and adds three
                # outside.
                y = x + 1  # g1: no annotation, config off -> no cudagraphs
                torch._dynamo.graph_break()
                with torch._dynamo.override_cudagraphs(fwd=True, bwd=True):
                    z = y * 2  # g2: annotation on -> cudagraphs
                    torch._dynamo.graph_break()
                    w = z - 1  # g3: annotation on -> cudagraphs
                torch._dynamo.graph_break()
                return w + 3  # g4: no annotation, config off -> no cudagraphs

            compiled = torch.compile(model)
            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda")
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled(inp)

            expected = ((inp + 1) * 2 - 1) + 3
            self.assertEqual(out, expected)
            # g2 and g3 use cudagraphs (2 graphs recorded)
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable_nested_callee_graph_break(self):
            # [FUNCTION DOC] With cudagraphs config off, a context manager in the caller enables
            # cudagraphs for both segments of a callee that has its own graph break and is
            # compiled as a separate frame; two graphs recorded, output correct.
            """Context manager enables cudagraphs across a graph break that
            occurs inside a nested callee, even though the `with` block lives
            in the caller's frame and the callee is compiled as a separate
            frame (it cannot be inlined because it contains a graph break)."""

            def callee(x):
                # [FUNCTION DOC] HELPER: Adds one, graph breaks, then doubles.
                y = x + 1  # segment 1 (inside override) -> cudagraphs
                torch._dynamo.graph_break()
                return y * 2  # segment 2 (inside override) -> cudagraphs

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the callee inside the cudagraph enabling context.
                with torch._dynamo.override_cudagraphs(fwd=True, bwd=True):
                    return callee(x)

            compiled = torch.compile(model)
            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda")
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled(inp)

            self.assertEqual(out, (inp + 1) * 2)
            self.assertIsNotNone(self.get_manager())
            # both callee segments use cudagraphs (2 graphs recorded)
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_cudagraph_annotation_enable_module_forward_graph_break(self):
            # [FUNCTION DOC] With cudagraphs config off, an enabling annotation on a module
            # forward that contains a graph break enables cudagraphs for both segments; two
            # graphs recorded, output correct.
            """Decorator on a module's forward enables cudagraphs for every
            segment even when the forward contains a graph break (the forward
            is compiled as a separate frame from its caller)."""

            class MyModule(torch.nn.Module):
                # [CLASS DOC] HELPER: Module whose annotated forward adds one, graph breaks and
                # doubles.
                @torch._dynamo.override_cudagraphs(fwd=True, bwd=True)
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Adds one, graph breaks, then doubles.
                    y = x + 1  # segment 1 -> cudagraphs
                    torch._dynamo.graph_break()
                    return y * 2  # segment 2 -> cudagraphs

            mod = MyModule().cuda()

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the module.
                return mod(x)

            compiled = torch.compile(model)
            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda")
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled(inp)

            self.assertEqual(out, (inp + 1) * 2)
            self.assertIsNotNone(self.get_manager())
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_cudagraph_annotation_disable_nested_callee_graph_break(self):
            # [FUNCTION DOC] With default reduce-overhead mode, a disabling context manager in
            # the caller turns off cudagraphs for both segments of a callee with a graph break;
            # output correct and no manager exists.
            """Context manager disables cudagraphs across a graph break inside a
            nested callee compiled as a separate frame, even though the default
            (reduce-overhead) config would otherwise record cudagraphs."""

            def callee(x):
                # [FUNCTION DOC] HELPER: Adds one, graph breaks, then doubles.
                y = x + 1  # segment 1 (inside override) -> no cudagraphs
                torch._dynamo.graph_break()
                return y * 2  # segment 2 (inside override) -> no cudagraphs

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the callee inside the cudagraph disabling
                # context.
                with torch._dynamo.override_cudagraphs(fwd=False, bwd=False):
                    return callee(x)

            inp = torch.randn(4, 4, device="cuda")
            compiled = torch.compile(model, mode="reduce-overhead")
            out = compiled(inp)

            self.assertEqual(out, (inp + 1) * 2)
            # No callee segment is recorded despite reduce-overhead default.
            self.assertIsNone(self.get_manager())

        def test_cudagraph_annotation_disable_bwd_nested_callee_graph_break(self):
            # [FUNCTION DOC] Context manager enabling forward but disabling backward around a
            # callee with a graph break: over three forward and backward steps a manager exists
            # and only the two forward segment graphs are recorded; output correct.
            """Asymmetric override (fwd=True, bwd=False) propagates across a
            graph break in a separately-compiled callee: both forward segments
            record cudagraphs while the backward is disabled."""

            def callee(x):
                # [FUNCTION DOC] HELPER: Computes sine of the square, graph breaks, then adds
                # one.
                y = (x * x).sin()  # forward segment 1
                torch._dynamo.graph_break()
                return y + 1  # forward segment 2

            def model(x):
                # [FUNCTION DOC] HELPER: Calls the callee inside the forward-only enable
                # context.
                with torch._dynamo.override_cudagraphs(fwd=True, bwd=False):
                    return callee(x)

            compiled = torch.compile(model, mode="reduce-overhead")
            for _ in range(3):
                inp = torch.randn(4, 4, device="cuda", requires_grad=True)
                torch.compiler.cudagraph_mark_step_begin()
                out = compiled(inp)
                out.sum().backward()

            self.assertEqual(out, (inp.detach().pow(2).sin()) + 1)
            self.assertIsNotNone(self.get_manager())
            # Two forward segments record; backward is disabled (no bwd graphs),
            # matching the forward-only count of the enable variant above.
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        def test_cudagraph_min_partition_size_skip_small(self):
            # [FUNCTION DOC] With graph partition on and a minimum partition size of ten
            # kernels, a tiny function is compiled in reduce-overhead mode; output is correct
            # and no cudagraph manager is created because the partition is too small.
            """Partitions with fewer kernels than the threshold are skipped."""

            # Simple function with few kernels
            def fn(x):
                # [FUNCTION DOC] HELPER: Returns twice x.
                return x * 2

            inp = torch.randn(4, 4, device="cuda")

            # Set threshold high enough to skip this simple function
            with torch._inductor.config.patch(
                {"triton.cudagraph_min_partition_size": 10, "graph_partition": True}
            ):
                fn_compiled = torch.compile(fn, mode="reduce-overhead")
                for _ in range(3):
                    out = fn_compiled(inp)

            self.assertEqual(out, inp * 2)
            # Partition was too small, so no cudagraph recorded
            self.assertIsNone(self.get_manager())

        def test_cudagraph_min_partition_size_allow_large(self):
            # [FUNCTION DOC] With a minimum partition size of two, a function with several
            # kernels is compiled; a manager exists and one graph is recorded.
            """Partitions with enough kernels pass the threshold."""

            # Function with multiple operations -> more kernels
            def fn(x):
                # [FUNCTION DOC] HELPER: Applies a chain of scale, add, sine, cosine and
                # multiply by x, then returns the matrix product of the result with itself.
                y = x * 2
                y = y + 1
                y = y.sin()
                y = y.cos()
                y = y * x
                return y @ y

            inp = torch.randn(4, 4, device="cuda")

            # Set threshold low enough to allow this function
            with torch._inductor.config.patch(
                {"triton.cudagraph_min_partition_size": 2}
            ):
                fn_compiled = torch.compile(fn, mode="reduce-overhead")
                for _ in range(3):
                    out = fn_compiled(inp)

            # Partition was large enough, cudagraph should be recorded
            self.assertIsNotNone(self.get_manager())
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        def test_tensor_constant_mutation(self):
            # [FUNCTION DOC] Compiles a module that mutates a CUDA tensor attribute in place on
            # every forward and runs it three times; only checks it does not raise.
            class Foo(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a 2 by 3 ones constant tensor that forward
                # increments in place before returning the input plus it.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the constant tensor attribute.
                    super().__init__()
                    self.tensor_constant = torch.ones((2, 3), device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Adds one to the constant in place and returns x
                    # plus the constant.
                    self.tensor_constant += 1
                    return x + self.tensor_constant

            foo = Foo()
            foo = torch.compile(foo, mode="reduce-overhead")
            inp = torch.rand((2, 3), device="cuda")
            for _ in range(3):
                foo(inp)

        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_rerecord_if_static_input_address_changed(self):
            # [FUNCTION DOC] With input mutation support, replaces the address of static inputs
            # (a registered buffer and a nested linear bias) after three runs, then runs three
            # more; the changed addresses force a re-record, so two graphs are recorded in
            # total.
            # By setting triton.cudagraph_support_input_mutation=True, we force re-record
            # if static tensor addresses changed.
            class Goo(torch.nn.Module):
                # [CLASS DOC] HELPER: Module containing a 2 by 2 linear layer; forward applies
                # it.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer on CUDA.
                    super().__init__()
                    self.linear = torch.nn.Linear(2, 2, device="cuda")

                def forward(self, x) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                    return self.linear(x)

            class Foo(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a static buffer and a nested module; forward
                # adds ones into the buffer in place and returns buffer plus x plus the nested
                # module output.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Registers a zeros buffer and creates the nested
                    # module.
                    super().__init__()
                    self.register_buffer(
                        "static_tensor", torch.zeros((2, 2), device="cuda")
                    )
                    self.goo = Goo()

                def forward(self, x) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Mutates the buffer in place with ones and returns
                    # buffer plus x plus the nested output.
                    self.static_tensor.add_(torch.ones((2, 2), device="cuda"))
                    return self.static_tensor + x + self.goo(x)

            foo = Foo()
            foo = torch.compile(foo, mode="reduce-overhead")
            inp = torch.rand((2, 2), device="cuda")

            for _ in range(3):
                foo(inp)

            # mutates static input tensors' addresses
            foo.static_tensor = torch.ones((2, 2), device="cuda")
            foo.goo.linear.bias = torch.nn.Parameter(torch.ones((2,), device="cuda"))

            for _ in range(3):
                foo(inp)

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_skip_if_dynamic_shape_limit_reached1(self):
            # [FUNCTION DOC] With the dynamic shape warn limit set to one, runs a compiled
            # linear layer with three batch sizes and checks the cudagraph_utils log contains
            # the warning about recording many CUDAGraphs for distinct input sizes (observed two
            # sizes) with its suggested remedies.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer applied in forward.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(3, 3, device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                    return self.linear(x)

            def iter(batch_size: int, mod: torch.nn.Module):
                # [FUNCTION DOC] HELPER: Creates a random input of the given batch size by 3 and
                # calls the module three times on it.
                x = torch.rand((batch_size, 3), device="cuda")
                for _ in range(3):
                    mod(x)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for batch_size in range(10, 40, 10):
                    iter(batch_size, mod)

            FileCheck().check(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning."
            ).run(log_stream.getvalue())

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_skip_if_dynamic_shape_limit_reached2(self):
            # [FUNCTION DOC] Same dynamic shape warning check using a multihead attention module
            # with varied batch size and length inputs.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module wrapping a multihead attention layer with embedding
                # size 3 and 3 heads, applied to query, key and value.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the attention layer on CUDA.
                    super().__init__()
                    self.attn = torch.nn.MultiheadAttention(
                        embed_dim=3, num_heads=3, device="cuda"
                    )

                def forward(
                    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
                ) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Runs the attention layer on the query, key and
                    # value and returns its result.
                    return self.attn(q, k, v)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            def iter(batch_size: int, length: int):
                # [FUNCTION DOC] HELPER: Creates random query, key and value tensors with the
                # given batch size and length and calls the module three times.
                q = torch.rand((batch_size, length, 3), device="cuda")
                k = torch.rand((batch_size, length, 3), device="cuda")
                v = torch.rand((batch_size, length, 3), device="cuda")
                for _ in range(3):
                    mod(q, k, v)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for batch_size in range(10, 40, 10):
                    for length in range(10, 30, 10):
                        iter(batch_size, length)

            FileCheck().check(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning."
            ).run(log_stream.getvalue())

        @torch._inductor.config.patch("triton.cudagraph_dynamic_shape_warn_limit", 1)
        def test_warn_once_if_dynamic_shape_limit_reached(self):
            # [FUNCTION DOC] With the warn limit set to one, runs many batch sizes and checks
            # the dynamic shape warning appears exactly once in the log.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a 3 by 3 linear layer applied in forward.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(3, 3, device="cuda")

                def forward(self, x: torch.Tensor) -> torch.Tensor:
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x.
                    return self.linear(x)

            def iter(batch_size: int, mod: torch.nn.Module):
                # [FUNCTION DOC] HELPER: Creates a random input of the given batch size by 3 and
                # calls the module three times.
                x = torch.rand((batch_size, 3), device="cuda")
                for _ in range(3):
                    mod(x)

            mod = torch.compile(Mod(), mode="reduce-overhead")

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for batch_size in range(10, 200, 10):
                    iter(batch_size, mod)

            FileCheck().check_count(
                "CUDAGraph supports dynamic shapes by recording a new graph for each "
                "distinct input size. Recording too many CUDAGraphs may lead to "
                "extra overhead. We have observed 2 distinct sizes. "
                "Please consider the following options for better performance: "
                "a) padding inputs to a few fixed number of shapes; or b) set "
                "torch._inductor.config.triton.cudagraph_skip_dynamic_graphs=True. "
                "Set torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit=None "
                "to silence this warning.",
                1,
                exactly=True,
            ).run(log_stream.getvalue())

        @torch._inductor.config.patch("cpp_wrapper", 1)
        def test_cpp_wrapper(self):
            # [FUNCTION DOC] With the C++ wrapper enabled, compiles a sine function in
            # reduce-overhead mode, runs it with warmup and record, and checks it matches eager.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns the sine of x.
                return torch.sin(x)

            compiled = torch.compile(f, mode="reduce-overhead")
            example_input = torch.randn(10, device="cuda")
            compiled_result = self.run_twc(compiled, example_input)
            eager_result = f(example_input)
            self.assertEqual(compiled_result, eager_result)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition(self):
            # [FUNCTION DOC] With graph partition on, compiles a function with a CPU round trip
            # in the middle, checks outputs match eager over five runs, and that two cudagraphs
            # are recorded (one per partition).
            def f(x, y):
                # [FUNCTION DOC] HELPER: Computes x plus one, y plus one, a CPU round trip of y
                # plus one, a matmul, and returns the sum of these results.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            x, y = [torch.randn(2, 2, device="cuda") for _ in range(2)]
            x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
            eager_out = f(x, y)

            f_compiled = torch.compile(f, mode="reduce-overhead")

            for _ in range(5):
                compiled_out = f_compiled(x_cloned, y_cloned)
                self.assertEqual(eager_out, compiled_out)

            # 2 graph partitions lead to 2 cudagraph
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @unittest.skip(
            "Disabled due to CI failures; see "
            "https://github.com/pytorch/pytorch/issues/190233"
        )
        def test_graph_partition_view_fallback(self):
            # [FUNCTION DOC] Disabled with a skip decorator pointing to an upstream issue about
            # CI failures.
            def f(x):
                # [FUNCTION DOC] HELPER: Adds one, reinterprets the bits as float8, round-trips
                # through CPU and back to CUDA, and returns it.
                y = x + 1
                z = torch.ops.aten.view.dtype(y, torch.float8_e4m3fn)
                z_cpu = z.cpu()
                u_cuda = z_cpu.cuda()
                return u_cuda

            compiled_f = torch.compile(f, mode="reduce-overhead")

            for _ in range(3):
                x = torch.ones(2, dtype=torch.int32, device="cuda")
                eager_out = f(x)
                compiled_out = compiled_f(x)
                self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        @skipIfRocm
        def test_graph_partition_max_autotune_skips_linalg_eigh(self):
            # [FUNCTION DOC] Skipped on ROCm. With graph partition on, verifies that the linalg
            # eigh op is classified as cudagraph unsafe (checked on a traced graph node), then
            # compiles a function using it with max-autotune and checks eigenvalues match eager
            # within tolerance, eigenvector shape matches, and the scheduler log reports two
            # partitions (one cudagraphable, one not).
            from torch._inductor.utils import is_cudagraph_unsafe_fx_node

            def f(x):
                # [FUNCTION DOC] HELPER: Computes the Gram matrix of x with its transpose and
                # returns its eigendecomposition.
                affinity = torch.matmul(x, x.transpose(-2, -1))
                return torch.linalg.eigh(affinity)

            gen = torch.Generator(device="cuda").manual_seed(0)
            x = torch.randn(2, 6, 4, device="cuda", generator=gen)
            affinity = torch.matmul(x, x.transpose(-2, -1))
            gm = make_fx(lambda a: torch.ops.aten._linalg_eigh.default(a))(affinity)
            eigh_node = next(
                node
                for node in gm.graph.nodes
                if node.target is torch.ops.aten._linalg_eigh.default
            )
            self.assertTrue(is_cudagraph_unsafe_fx_node(eigh_node))

            expected = f(x)
            compiled_f = torch.compile(f, mode="max-autotune")

            log_stream, ctx = logs_to_string("torch._inductor.scheduler", "cudagraphs")
            with ctx():
                for _ in range(3):
                    actual = compiled_f(x)

            self.assertEqual(actual[0], expected[0], atol=1e-4, rtol=1e-4)
            self.assertEqual(actual[1].shape, expected[1].shape)
            FileCheck().check(
                "Created 2 graph partitions: 1 cudagraphable, 1 non-cudagraphable"
            ).run(log_stream.getvalue())

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_log_message(self):
            # [FUNCTION DOC] With graph partition on, checks the scheduler log reports exactly
            # one message of two partitions (one cudagraphable, one not) with exactly one CPU
            # ops reason for a function mixing CUDA and CPU inputs, and that the cudagraph trees
            # log shows exactly one recording of the first partition in inference mode.
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus one and y plus two.
                return (x + 1, y + 2)

            foo = torch.compile(foo, mode="reduce-overhead")

            log_stream, ctx = logs_to_string("torch._inductor.scheduler", "cudagraphs")
            with ctx():
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            FileCheck().check_count(
                "Created 2 graph partitions: 1 cudagraphable, 1 non-cudagraphable",
                1,
                exactly=True,
            ).check_count("reason=cpu ops", 1, exactly=True).run(log_stream.getvalue())

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_trees", "cudagraphs"
            )
            with ctx():
                # trigger recording
                foo(torch.ones([10], device="cuda"), torch.ones([20]))
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            FileCheck().check_count(
                "Recording function=partition_0, mode=INFERENCE, cuda_graph_id=0,",
                1,
                exactly=True,
            ).run(log_stream.getvalue())

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar1(self):
            # [FUNCTION DOC] With graph partition on, adds a CUDA tensor and a CPU scalar
            # tensor. The generated code must contain exactly one copy of the CPU scalar to GPU;
            # results equal eager and one graph is recorded.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (torch.ones(2, 2, device="cuda"), torch.ones((), device="cpu"))
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar2(self):
            # [FUNCTION DOC] Same as the first CPU scalar test but with the CPU scalar first and
            # two CUDA tensors: exactly one copy in generated code, results equal eager, one
            # graph.
            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Returns x plus y and x plus z.
                return x + y, x + z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.ones((), device="cpu"),
                torch.ones(2, 2, device="cuda"),
                torch.ones(2, 2, device="cuda"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar3(self):
            # [FUNCTION DOC] CPU scalar used as the last operand after a CUDA add: exactly one
            # copy in generated code, correct result and one recorded graph.
            def f(x, y, cpu_scalar_tensor):
                # [FUNCTION DOC] HELPER: Returns x plus y plus the CPU scalar tensor.
                z = x + y
                z = z + cpu_scalar_tensor
                return z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.randn(2, 2, device="cuda"),
                torch.randn(2, 2, device="cuda"),
                torch.tensor(1, device="cpu"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar4(self):
            # [FUNCTION DOC] CPU scalar tensor used through an intermediate CPU computation that
            # is then added to a GPU value: the scalar must still be moved to GPU (one copy in
            # generated code), results correct and one graph recorded.
            # cpu_scalar_tensor is accessed by cpu_scalar2 which is
            # added with a gpu tensor z. This test checks the cpu
            # scalar tensors are still moved in this case.
            def f(x, y, cpu_scalar_tensor):
                # [FUNCTION DOC] HELPER: Adds one to the CPU scalar tensor, adds x and y, then
                # adds the result of the CPU computation and returns it.
                cpu_scalar2 = cpu_scalar_tensor + 1
                z = x + y
                z = z + cpu_scalar2
                return z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.randn(2, 2, device="cuda"),
                torch.randn(2, 2, device="cuda"),
                torch.tensor(1, device="cpu"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        # turn on input mutation support to avoid skipping cudagraph at dynamo level
        @torch._inductor.config.patch("triton.cudagraph_support_input_mutation", True)
        def test_graph_partition_cpu_scalar_mutation(self):
            # [FUNCTION DOC] With graph partition and input mutation support, a compiled
            # function copies a CUDA tensor into a CPU scalar tensor in place three times; the
            # CPU input ends with the CUDA value, the CUDA input is unchanged and one graph is
            # recorded.
            # tests that input mutation on a cpu scalar tensor x is correctly
            # handled when moving x to gpu at the beginning of the graph.

            @torch.compile(mode="reduce-overhead")
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Copies y into x in place and returns x.
                return x.copy_(y)

            x = torch.tensor(1)
            y = torch.tensor(2, device="cuda")

            for _ in range(3):
                foo(x, y)

            self.assertEqual(x, torch.tensor(2, device="cpu"))
            self.assertEqual(y, torch.tensor(2, device="cuda"))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar_device_put(self):
            # [FUNCTION DOC] With graph partition on, a compiled function moves a CPU scalar to
            # CUDA and back; after three calls the input is unchanged.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Moves x to CUDA then back to CPU and returns it.
                y = x.to("cuda")
                z = y.to("cpu")
                return z

            x = torch.tensor(1)
            for _ in range(3):
                foo(x)

            self.assertEqual(x, torch.tensor(1, device="cpu"))

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar_multiple(self):
            # [FUNCTION DOC] Two CPU scalars and one CUDA tensor with graph partition on:
            # generated code contains exactly one copy (with a True flag) for the CPU scalars;
            # results equal eager and one graph recorded.
            def f(x, y, z):
                # [FUNCTION DOC] HELPER: Returns x plus y and x plus z.
                return x + y, x + z

            compiled_f = torch.compile(f, mode="reduce-overhead")

            inputs = (
                torch.ones((), device="cpu"),
                torch.ones((), device="cpu"),
                torch.ones(2, 2, device="cuda"),
            )
            for i in range(3):
                if i == 0:
                    _, code = run_and_get_code(compiled_f, *inputs)
                    FileCheck().check_regex(r".copy_.*True").run(code[0])
                    FileCheck().check_count(".copy_", 1, exactly=True).run(code[0])
                else:
                    compiled_f(*inputs)
            self.assertEqual(compiled_f(*inputs), f(*inputs))
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._inductor.config.patch("triton.cudagraphs", False)
        def test_graph_partition_reduce_overhead_mode_effectiveness(self):
            # [FUNCTION DOC] With graph partition on and the cudagraphs option off, a function
            # compiled in default mode (not reduce-overhead) runs five times and must not create
            # a cudagraph manager.
            # test that `mode="reduce-overhead"` still controls whether
            # cudagraph is applied. i.e., cudagraph is not applied when
            # mode="default".
            def f(x, y):
                # [FUNCTION DOC] HELPER: Does adds and a matmul with a CPU round trip, returning
                # the sum of the pieces.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            x, y = [torch.randn(2, 2, device="cuda") for _ in range(2)]

            f_compiled = torch.compile(f)
            for _ in range(5):
                _out = f_compiled(x, y)
            self.assertEqual(self.get_manager() is None, True)

        @torch._inductor.config.patch("graph_partition", False)
        def test_view_only_cuda_graph_suppresses_empty_warning(self):
            # [FUNCTION DOC] With graph partition off, compiles a function that only reshapes
            # (view only graph) and runs it three times, recording all warnings. No warning
            # containing the text about an empty CUDA graph may appear, the non-static input
            # counter is one and a manager exists.
            counters.clear()

            def fn(x):
                # [FUNCTION DOC] HELPER: Reshapes x to one by minus one by four.
                return x.reshape(1, -1, 4)

            x = torch.randn(6, 4, device="cuda")
            compiled_fn = torch.compile(fn, fullgraph=True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                for _ in range(3):
                    actual = compiled_fn(x)
                    torch.cuda.synchronize()

            self.assertEqual(actual, fn(x))
            self.assertFalse(
                any("CUDA Graph is empty" in str(w.message) for w in caught)
            )
            self.assertEqual(
                counters["inductor"]["cudagraph_recorded_non_static_inputs"], 1
            )
            self.assertIsNotNone(self.get_manager())

        @torch._inductor.config.patch("triton.slow_path_cudagraph_asserts", True)
        def test_kernel_free_allocation_uses_cudagraph_pool(self):
            # [FUNCTION DOC] With slow path cudagraph asserts on, for graph partition off and
            # on, resets dynamo and runs a compiled function that only allocates an empty tensor
            # three times. Checks the output shape and that no warning about an empty CUDA graph
            # is emitted.
            def fn(x):
                # [FUNCTION DOC] HELPER: Returns an uninitialized tensor like x.
                return torch.empty_like(x)

            for graph_partition in (False, True):
                torch._dynamo.reset()
                x = torch.randn(6, 4, device="cuda")
                with (
                    torch._inductor.config.patch("graph_partition", graph_partition),
                    warnings.catch_warnings(record=True) as caught,
                ):
                    warnings.simplefilter("always")
                    compiled_fn = torch.compile(fn, fullgraph=True)
                    for _ in range(3):
                        actual = compiled_fn(x)
                        torch.cuda.synchronize()

                self.assertEqual(actual.shape, x.shape)
                self.assertFalse(
                    any("CUDA Graph is empty" in str(w.message) for w in caught)
                )

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_backward(self):
            # [FUNCTION DOC] With graph partition on, trains a small model whose forward
            # contains a CPU round trip for five steps with SGD, then expects three cudagraphs
            # (two forward partitions plus one backward).
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a 16 by 16 linear layer; forward combines
                # elementwise ops, a CPU round trip and a matmul into a tensor fed through the
                # linear layer.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Adds one and two to the input, round-trips the
                    # second through CPU, multiplies x by it, sums the pieces and applies the
                    # linear layer.
                    x1 = x + 1
                    y1 = x + 2
                    y_cpu = y1.cpu() + 1
                    z = x @ y1
                    inp = x1 + y1 + z + y_cpu.cuda()
                    return self.linear(inp)

            model = Mod().cuda()

            input_data = torch.randn(16, 16).cuda()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            compiled_model = torch.compile(model, mode="reduce-overhead")

            for _ in range(5):
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 2 graph partitions lead to 2 fwd cudagraphs and 1 bwd cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar_used_in_cpu_op(self):
            # [FUNCTION DOC] With graph partition on, uses a model whose forward feeds a CPU
            # computation using a CPU parameter. The scheduler log must show two cudagraphable
            # and one non-cudagraphable partitions; trains for five steps and finally checks
            # compiled output matches eager.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a linear layer on CUDA and a CPU scalar
                # parameter; forward adds the CPU-scaled sum of the output back onto the output.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the CUDA linear layer and the CPU scalar
                    # parameter initialized to one.
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 4, device="cuda")
                    self.cpu_scale = torch.nn.Parameter(torch.tensor(1.0))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer, multiplies the CPU copy
                    # of its sum by the CPU parameter, moves it back to CUDA and adds it to the
                    # layer output.
                    y = self.linear(x)
                    cpu_val = y.sum().cpu() * self.cpu_scale
                    return y + cpu_val.cuda()

            model = Mod()
            x = torch.randn(4, 4, device="cuda")

            compiled_model = torch.compile(model, mode="reduce-overhead")

            # Verify graph_partition produces the expected partitions.
            log_stream, ctx = logs_to_string("torch._inductor.scheduler", "cudagraphs")
            with ctx():
                compiled_model(x)
            FileCheck().check("2 cudagraphable, 1 non-cudagraphable").run(
                log_stream.getvalue()
            )

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            for _ in range(5):
                output = compiled_model(x)
                loss = criterion(output, torch.randint(0, 4, (4,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            eager_out = model(x)
            compiled_out = compiled_model(x)
            self.assertEqual(compiled_out, eager_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_scalar_as_output(self):
            # [FUNCTION DOC] Regression check for a CPU scalar parameter moved to GPU by the
            # constructor mover pass and returned as an output: compiles and trains for five SGD
            # steps, then asserts compiled output equals eager.
            """
            When a CPU scalar placeholder is moved to GPU by ConstructorMoverPass,
            the forward graph's output must NOT replace the CPU placeholder with
            the GPU copy.
            """

            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module with a CUDA linear layer and a CPU scalar
                # parameter; forward multiplies the linear output by the parameter.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the CUDA linear layer and the CPU scalar
                    # parameter initialized to one.
                    super().__init__()
                    self.linear = torch.nn.Linear(4, 4, device="cuda")
                    self.cpu_scale = torch.nn.Parameter(torch.tensor(1.0))

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Returns the linear output times the CPU scalar
                    # parameter.
                    return self.linear(x) * self.cpu_scale

            model = Mod()
            x = torch.randn(4, 4, device="cuda")

            compiled_model = torch.compile(model, mode="reduce-overhead")
            compiled_model(x)

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            for _ in range(5):
                output = compiled_model(x)
                loss = criterion(output, torch.randint(0, 4, (4,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            eager_out = model(x)
            compiled_out = compiled_model(x)
            self.assertEqual(compiled_out, eager_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_saved_activation_not_static(self):
            # [FUNCTION DOC] With graph partition on, intercepts the backward compile entry
            # point (by patching it) to capture the backward graph of a model whose forward has
            # a CPU round trip. Checks the number of static backward inputs computed for saved
            # activations (only primals) is strictly smaller than the count that would mark
            # every saved tensor static, that its indices are a subset of those, and runs four
            # more training steps to confirm stability.
            # When the forward is partitioned, saved activations produced
            # by inline code between partitions (e.g., DeviceCopy) are
            # NOT at fixed addresses. The backward must not mark them as
            # static inputs, or it would re-record on every iteration.
            # Primals (params/buffers) should still be marked static.
            from unittest.mock import patch

            from torch._inductor.utils import count_tangents, get_static_bw_input_idxs

            bw_graph = None
            orig_bw = torch._inductor.compile_fx.compile_fx_backward

            def intercept_bw(gm, example_inputs, compiler_config_extra, **kwargs):
                # [FUNCTION DOC] HELPER: Replacement for the backward compile function: stores
                # the graph in an outer variable and then delegates to the original function
                # with the same arguments, returning its result.
                nonlocal bw_graph
                bw_graph = gm
                return orig_bw(gm, example_inputs, compiler_config_extra, **kwargs)

            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module whose forward doubles the input, round-trips
                # through CPU (creating a partition boundary and a saved activation) squares it
                # and applies a 16 by 16 linear layer.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Doubles x, round-trips it through CPU and back,
                    # squares the result and applies the linear layer.
                    a = x * 2
                    # CPU round-trip creates a DeviceCopy partition boundary.
                    # The .cuda() result is an activation saved for backward.
                    b = a.cpu().cuda()
                    c = b * b
                    return self.linear(c)

            model = Mod().cuda()
            input_data = torch.randn(16, 16, device="cuda")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            with patch("torch._inductor.compile_fx.compile_fx_backward", intercept_bw):
                compiled_model = torch.compile(model, mode="reduce-overhead")
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.assertIsNotNone(bw_graph)
            # count_tangents marks ALL saved tensors as static (old behavior)
            all_static = list(range(count_tangents(bw_graph)))
            # get_static_bw_input_idxs only marks primals as static
            primal_static = get_static_bw_input_idxs(bw_graph)
            # With a partitioned forward, only primals should be static,
            # so primal_static should be a strict subset of all_static.
            self.assertTrue(len(primal_static) < len(all_static))
            for idx in primal_static:
                self.assertIn(idx, all_static)

            # Run a few more iterations to confirm stability
            for _ in range(4):
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_no_partition_keeps_static(self):
            # [FUNCTION DOC] With graph partition enabled but a forward without unsafe ops,
            # intercepts the backward compile entry point to record the forward-is-partitioned
            # flag from its compiler config argument, runs one training step and asserts the
            # flag is false.
            # When graph_partition is enabled but the forward has no unsafe
            # ops, forward_is_partitioned should be False and all saved
            # tensors remain static in the backward.
            from unittest.mock import patch

            forward_partitioned = None
            orig_bw = torch._inductor.compile_fx.compile_fx_backward

            def intercept_bw(gm, example_inputs, compiler_config_extra, **kwargs):
                # [FUNCTION DOC] HELPER: Replacement for the backward compile function: records
                # the forward-is-partitioned flag value into an outer variable and delegates to
                # the original function.
                nonlocal forward_partitioned
                forward_partitioned = compiler_config_extra.forward_is_partitioned.value
                return orig_bw(gm, example_inputs, compiler_config_extra, **kwargs)

            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: Module applying a 16 by 16 linear layer to x squared plus
                # one.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer to x squared plus one.
                    return self.linear(x * x + 1)

            model = Mod().cuda()
            input_data = torch.randn(16, 16, device="cuda")
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            with patch("torch._inductor.compile_fx.compile_fx_backward", intercept_bw):
                compiled_model = torch.compile(model, mode="reduce-overhead")
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cuda())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.assertFalse(forward_partitioned)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_only(self):
            # [FUNCTION DOC] With graph partition on, trains a CPU-only model with default
            # compile mode for five steps and asserts no cudagraph manager exists.
            class Mod(torch.nn.Module):
                # [CLASS DOC] HELPER: CPU module combining elementwise ops and a matmul and
                # applying a 16 by 16 linear layer.
                def __init__(self) -> None:
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = torch.nn.Linear(16, 16)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Combines x plus one, x plus two, a matmul and a
                    # shifted copy and applies the linear layer.
                    x1 = x + 1
                    y1 = x + 2
                    y_cpu = y1 + 1
                    z = x @ y1
                    inp = x1 + y1 + z + y_cpu
                    return self.linear(inp)

            model = Mod().cpu()

            input_data = torch.randn(16, 16).cpu()

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

            compiled_model = torch.compile(model, mode="default")

            for _ in range(5):
                output = compiled_model(input_data)
                loss = criterion(output, torch.randint(0, 10, (16,)).cpu())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 0 cudagraph since all ops are on cpu
            self.assertEqual(self.get_manager() is None, True)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_with_skipped_cudagraphed_backward(self):
            # [FUNCTION DOC] Graph partition counterpart of the skipped-backward test: forward
            # and backward with backward cudagraph force-disabled, three times. Expects one
            # recorded graph and no forwards pending backward.
            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns the cube of x.
                return x * x * x

            for _ in range(3):
                inp = torch.rand([20, 20], device="cuda", requires_grad=True)
                out = foo(inp)

                with config.patch(force_disable_cudagraph_TESTING_ONLY=True):
                    back_inp = torch.empty_strided([20, 20], [0, 1], device="cuda")
                    out.backward(back_inp)

            # we should not have cudagraph'd the backwards
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_forward_backward_not_called(self):
            # [FUNCTION DOC] Graph partition counterpart of the backward-not-called test: calls
            # forward with a graph break, but runs backward through only the first output.
            # Expects no pending backwards and the next graph id to be three.
            # tests saved tensor is handled correctly
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns the cubes of x and y with a graph break between
                # them.
                x_out = x * x * x
                torch._dynamo.graph_break()
                y_out = y * y * y
                return x_out, y_out

            foo = torch.compile(foo, mode="reduce-overhead")

            for _ in range(3):
                inps = [
                    torch.rand([20, 20], requires_grad=True, device="cuda")
                    for _ in range(2)
                ]
                x_out, y_out = foo(inps[0], inps[1])
                x_out.sum().backward()

            self.assertFalse(self.get_manager().running_forwards_with_pending_backwards)

            # we should not have cudagraph'd the y backward
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 3)

        @requires_multigpu()
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_multiple_devices_msg(self):
            # [FUNCTION DOC] Multi-GPU with graph partition on: a mixed CUDA/CPU call creates no
            # skips (CPU ops are partitioned). A call with tensors on two CUDA devices is
            # skipped with a multiple devices message and a skip count of one; exactly one graph
            # was recorded.
            def foo(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus one and y plus two.
                return (x + 1, y + 2)

            foo = torch.compile(foo, mode="reduce-overhead")
            for _ in range(3):
                foo(torch.ones([10], device="cuda"), torch.ones([20]))

            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    foo(
                        torch.ones([10], device="cuda:0"),
                        torch.ones([10], device="cuda:1"),
                    )

            FileCheck().check("skipping cudagraphs due to multiple devices").run(
                log_stream.getvalue()
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)
            new_id = self.get_manager().new_graph_id().id
            self.assertEqual(new_id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_dynamic_shapes(self):
            # [FUNCTION DOC] With graph partition on, runs a compiled function on three
            # different input sizes three times each; expects three recorded graphs.
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            compiled_foo = torch.compile(foo, mode="reduce-overhead", fullgraph=True)

            for input_shape in range(1, 4):
                for _ in range(3):
                    compiled_foo(torch.randn(input_shape, device="cuda"))

            # 3 cudagraphs for 3 input shapes
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_op_and_dynamic_shapes(self):
            # [FUNCTION DOC] With graph partition on, a function with a CPU round trip is run
            # with two input sizes; results match eager and four graphs are recorded (two shapes
            # times two partitions).
            def f(x, y):
                # [FUNCTION DOC] HELPER: Does adds, a CPU round trip and a matmul and returns
                # their sum.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.cuda()

            f_compiled = torch.compile(f)
            x, y = torch.ones(3, 3, device="cuda"), torch.randn(3, 3, device="cuda")
            for _ in range(3):
                compiled_out = f_compiled(x, y)
                self.assertEqual(compiled_out, f(x, y))

            x, y = torch.ones(4, 4, device="cuda"), torch.randn(4, 4, device="cuda")
            for _ in range(3):
                compiled_out = f_compiled(x, y)
                self.assertEqual(compiled_out, f(x, y))

            # 4 cudagraphs, due to (2 dynamic shapes) x (2 graph partitions)
            self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @config.patch(implicit_fallbacks=True)
        @config.patch("graph_partition", False)
        def test_skip_cudagraph_unsafe_ops(self):
            # [FUNCTION DOC] With implicit fallbacks and graph partition off, a custom op tagged
            # cudagraph unsafe that mutates a list of tensors is compiled in reduce-overhead
            # mode. Results match eager, the log reports skipping due to the incompatible op,
            # and the skip count is one.
            @torch.library.custom_op(
                "mylib::mysin",
                mutates_args=["out_list"],
                schema="(Tensor x, Tensor(a!)[]? out_list) -> Tensor",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def mysin(x, out_list) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation: computes sine of x, copies it
                # into the first list element if a list was given, and returns it.
                r = x.sin()
                if out_list is not None:
                    out_list[0].copy_(r)
                return r

            @mysin.register_fake
            def _(x, out_list) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like x.
                return torch.empty_like(x)

            def fn(x):
                # [FUNCTION DOC] HELPER: Triples x, applies the custom op with a fresh output
                # list, divides by three and returns the result and the list's tensor.
                x = x * 3
                s = [torch.empty_like(x)]
                x = mysin(x, s)
                x = x / 3
                return x, s[0]

            x = torch.randn(3, requires_grad=False, device="cuda")
            expected = fn(x)
            compiled_f = torch.compile(fn, mode="reduce-overhead", fullgraph=True)

            log_stream, ctx = logs_to_string(
                "torch._inductor.cudagraph_utils", "cudagraphs"
            )
            with ctx():
                for _ in range(3):
                    result = compiled_f(x)
                    self.assertEqual(result, expected)

            FileCheck().check("incompatible op mylib.mysin.default").run(
                log_stream.getvalue()
            )
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 1)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op(self):
            # [FUNCTION DOC] With implicit fallbacks and graph partition on, defines two
            # cudagraph unsafe custom ops (doing CPU round trips) and one safe op, chains them
            # in a function, checks eager equals default-compiled output, then compiles in
            # reduce-overhead mode, runs three times checking results, and expects three
            # recorded cudagraphs (split at the two unsafe ops).
            @torch.library.custom_op(
                "mylib::movement",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def movement(pic: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation: moves to CPU, adds
                # one, doubles, moves back to CUDA and divides by 255.
                img = pic.cpu()
                cropped_img = (img + 1) * 2
                return cropped_img.cuda() / 255.0

            @movement.register_fake
            def _(pic):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like pic.
                return torch.empty_like(pic)

            @torch.library.custom_op(
                "mylib::modify",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def modify(pic: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation: adds one, does a CPU
                # round trip with arithmetic and adds the original tensor.
                pic1 = pic + 1
                pic1_cpu = (pic1.cpu() + 1) * 2
                return pic1_cpu.cuda() + pic

            @modify.register_fake
            def _(pic):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like pic.
                return torch.empty_like(pic)

            @torch.library.custom_op("mylib::transform", mutates_args=())
            def transform(pic: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Safe custom op implementation: adds one and doubles.
                return (pic + 1) * 2

            @transform.register_fake
            def _(pic):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like pic.
                return torch.empty_like(pic)

            img = torch.randn(3, 64, 64, device="cuda")

            def f(img):
                # [FUNCTION DOC] HELPER: Pipeline: adds ten and doubles, applies the movement
                # op, adds one, applies the transform op, computes twice that plus one, applies
                # the modify op and adds one.
                x = (img + 10) * 2
                y = movement(x)
                z = y + 1
                u = transform(z)
                v = 2 * u + 1
                out = modify(v)
                return out + 1

            compiled_f = torch.compile(f, fullgraph=True)

            eager_out = f(img)
            compiled_out = compiled_f(img)

            self.assertEqual(eager_out, compiled_out)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            eager_out = f(img)

            for _ in range(3):
                compiled_out = compiled_f(img)
                self.assertEqual(eager_out, compiled_out)

            # splitting on 2 custom gives 3 cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @config.patch(implicit_fallbacks=True)
        @config.patch("graph_partition", True)
        def test_graph_partition_custom_op_mutation(self):
            # [FUNCTION DOC] With graph partition on, a custom op mutating a list argument and
            # tagged unsafe splits the graph in two; results match eager over three runs and two
            # cudagraphs are recorded.
            @torch.library.custom_op(
                "mylib::mysin",
                mutates_args=["out_list"],
                schema="(Tensor x, Tensor(a!)[]? out_list) -> Tensor",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def mysin(x, out_list) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation: computes sine of x, copies it
                # into the first list element if a list was given, and returns it.
                r = x.sin()
                if out_list is not None:
                    out_list[0].copy_(r)
                return r

            @mysin.register_fake
            def _(x, out_list) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like x.
                return torch.empty_like(x)

            def fn(x):
                # [FUNCTION DOC] HELPER: Triples x, applies the custom op with a fresh output
                # list, divides by three and returns the result and the list's tensor.
                x = x * 3
                s = [torch.empty_like(x)]
                x = mysin(x, s)
                x = x / 3
                return x, s[0]

            x = torch.randn(3, requires_grad=False, device="cuda")
            expected = fn(x)
            compiled_f = torch.compile(fn, mode="reduce-overhead", fullgraph=True)
            for _ in range(3):
                result = compiled_f(x)
                self.assertEqual(result, expected)

            # splitting on 1 custom gives 2 cudagraphs
            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_mutation_late_free(self):
            # [FUNCTION DOC] With graph partition on, chains a mutating op and an unsafe op
            # twice so that buffers must be freed late across partitions; compiled output must
            # match eager over five runs.
            @torch.library.custom_op(
                "mylib::op1",
                mutates_args=["x"],
                schema="(Tensor(a!)?  x) -> (Tensor, Tensor)",
                device_types="cuda",
            )
            def op1(x) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Custom op implementation (CUDA only, nominally
                # mutating): adds one to x locally and returns x plus one and x plus two.
                x = x + 1
                return (x + 1, x + 2)

            @op1.register_fake
            def _(x) -> tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Fake implementation returning two empty tensors like x.
                return (torch.empty_like(x), torch.empty_like(x))

            @torch.library.custom_op(
                "mylib::cg_unsafe_op",
                mutates_args=[],
                schema="(Tensor x, Tensor y, Tensor x1, Tensor y1) -> Tensor",
                device_types="cuda",
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def cg_unsafe_op(x0, x1, y0, y1) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation returning the
                # elementwise sum of its four tensor arguments.
                return x0 + x1 + y0 + y1

            @cg_unsafe_op.register_fake
            def _(x0, x1, y0, y1) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like the
                # first argument.
                return torch.empty_like(x0)

            def f(x):
                # [FUNCTION DOC] HELPER: Adds one, applies op1, derives values from its two
                # outputs, applies the unsafe op, applies op1 again on a combined value, sums
                # its outputs and applies the unsafe op once more.
                x = x + 1
                x = op1(x)
                x0, x1 = x[0], x[1]
                y0 = x0 + 1
                y1 = x1 + 1
                y = cg_unsafe_op(x0, x1, y0, y1)
                z = y + x0 + x1
                z0, z1 = op1(z)
                z2 = z0 + z1
                res = cg_unsafe_op(z2, z2, y, y)
                return res

            x = torch.randn(2, 2, device="cuda")
            x_cloned = x.clone()
            eager_out = f(x)

            f_compiled = torch.compile(f, mode="reduce-overhead")

            for _ in range(5):
                compiled_out = f_compiled(x_cloned)
                self.assertEqual(eager_out, compiled_out)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_dynamoc_shapes(self):
            # [FUNCTION DOC] With graph partition on, a pipeline with one unsafe custom op is
            # compiled in reduce-overhead mode and run with three different image sizes; results
            # match eager each time and six graphs are recorded (two partitions times three
            # dynamic shapes). A default-compiled variable is created but overwritten.
            @torch.library.custom_op(
                "mylib::movement",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def movement(pic: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation: CPU round trip with
                # add one and double, back to CUDA and divide by 255.
                img = pic.cpu()
                cropped_img = (img + 1) * 2
                return cropped_img.cuda() / 255.0

            @movement.register_fake
            def _(pic):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like pic.
                return torch.empty_like(pic)

            def f(img):
                # [FUNCTION DOC] HELPER: Pipeline: adds ten and doubles, applies the movement
                # op, adds one, doubles and adds one, then adds one again.
                x = (img + 10) * 2
                y = movement(x)
                z = y + 1
                v = 2 * z + 1
                return v + 1

            compiled_f = torch.compile(f, fullgraph=True)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            def run(size):
                # [FUNCTION DOC] HELPER: Creates a random 3 by size by size CUDA image, computes
                # the eager result and checks the compiled function matches it on three calls.
                img = torch.randn(3, size, size, device="cuda")
                eager_out = f(img)
                for _ in range(3):
                    compiled_out = compiled_f(img)
                    self.assertEqual(eager_out, compiled_out)

            run(64)
            run(17)
            run(42)

            # 2 (from splitting on 1 custom op) x 3 (dynamic shapes) = 6
            self.assertEqual(self.get_manager().new_graph_id().id, 6)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_custom_op_no_split(self):
            # [FUNCTION DOC] Same as the dynamic shapes custom op test but with a safe custom op
            # that does not split the graph: three image sizes yield three recorded graphs.
            @torch.library.custom_op(
                "mylib::modify",
                mutates_args=(),
            )
            def modify(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation: adds one and doubles.
                return (x + 1) * 2

            @modify.register_fake
            def _(pic):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like pic.
                return torch.empty_like(pic)

            def f(img):
                # [FUNCTION DOC] HELPER: Pipeline around the safe custom op with elementwise ops
                # before and after.
                x = (img + 10) * 2
                y = modify(x)
                z = y + 1
                v = 2 * z + 1
                return v + 1

            compiled_f = torch.compile(f, fullgraph=True)

            compiled_f = torch.compile(f, mode="reduce-overhead", fullgraph=True)

            def run(size):
                # [FUNCTION DOC] HELPER: Creates a random 3 by size by size CUDA image, computes
                # the eager result and checks the compiled function matches it on three calls.
                img = torch.randn(3, size, size, device="cuda")
                eager_out = f(img)
                for _ in range(3):
                    compiled_out = compiled_f(img)
                    self.assertEqual(eager_out, compiled_out)

            run(64)
            run(17)
            run(42)

            # 1 (from not splitting on custom op) x 3 (dynamic shapes) = 3
            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_cpu_tensor_symints(self):
            # [FUNCTION DOC] With graph partition on, runs a function on a CUDA tensor and a CPU
            # tensor with varying shapes. A new cudagraph is recorded when the CUDA input shape
            # changes or when the CPU tensor shape first becomes dynamic (a new dynamo graph),
            # but not for further CPU-only shape changes; total recorded graphs is three.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus one and y plus one.
                return x + 1, y + 1

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(shape_x, shape_y):
                # [FUNCTION DOC] HELPER: Creates a random CUDA tensor and a random CPU tensor of
                # the given shapes and calls the compiled function three times.
                x = torch.randn(shape_x, device="cuda")
                y = torch.randn(shape_y, device="cpu")
                for _ in range(3):
                    compiled_f(x, y)

            # static shape. record a NEW cudagraph
            run(shape_x=(2, 3), shape_y=(4, 4))

            # shape_y becomes dynamic shape leading to a new dynamo graph.
            # This new dynamo graph forces a NEW cudagraph although tensor y is on cpu
            run(shape_x=(2, 3), shape_y=(5, 6))

            # tensor y is on cpu so NO new cudagraph is recorded
            run(shape_x=(2, 3), shape_y=(7, 8))

            # shape_x becomes dynamic shape, leading to a new dynamo graph
            # this new dynamo graph forces a NEW cudagraph
            run(shape_x=(3, 4), shape_y=(4, 4))

            # tensor y is on cpu so NO new cudagraph is recorded
            run(shape_x=(3, 4), shape_y=(10, 11))

            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_cpu_and_gpu(self):
            # [FUNCTION DOC] With graph partition on, a function interleaves CUDA and CPU
            # computations with no cross dependency; compiled output matches eager over three
            # runs and scheduler reordering merges the CUDA ops into a single partition, so one
            # cudagraph is recorded.
            def f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu):
                # [FUNCTION DOC] HELPER: Computes independent CUDA chains for x and z (add,
                # matmul with weight, residual sum doubled) and a CPU chain for y, returning the
                # three results.
                x_cuda0 = x_cuda + 1
                x_cuda1 = x_cuda0 @ weight_cuda
                x_cuda2 = 2 * (x_cuda1 + x_cuda)

                y_cpu0 = y_cpu + 1
                y_cpu1 = y_cpu0 @ weight_cpu

                z_cuda0 = z_cuda + 1
                z_cuda1 = z_cuda0 @ weight_cuda
                z_cuda2 = 2 * (z_cuda1 + z_cuda)

                return x_cuda2, y_cpu1, z_cuda2

            x_cuda = torch.randn(3, 3, device="cuda")
            y_cpu = torch.randn(3, 3, device="cpu")
            z_cuda = torch.randn(3, 3, device="cuda")
            weight_cuda = torch.randn(3, 3, device="cuda")
            weight_cpu = torch.randn(3, 3, device="cpu")

            eager_out = f(x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu)

            compiled_f = torch.compile(f, mode="reduce-overhead")
            for _ in range(3):
                compiled_out = compiled_f(
                    x_cuda, y_cpu, z_cuda, weight_cuda, weight_cpu
                )
                self.assertEqual(eager_out, compiled_out)

            # reorder merges ops on cuda into 1 graph partition
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_cpu_and_gpu_interleave(self):
            # [FUNCTION DOC] With graph partition on, a function with CUDA and CPU segments
            # where the second segment depends on the first CUDA result. Results match eager
            # over three runs; reordering gives the CPU independent segment first, then all CUDA
            # segments, then the dependent CPU segment, so one cudagraph is recorded.
            def f(x_cuda, y_cpu, z_cuda, bias_cuda, weight_cuda, weight_cpu):
                # [FUNCTION DOC] HELPER: Computes three CUDA chains and two CPU chains, one CPU
                # chain depending on a CUDA result moved to CPU, and returns all five results.
                # partition 1 on cuda, no dependency
                x_cuda0 = x_cuda + 1
                x_cuda1 = x_cuda0 @ weight_cuda
                x_cuda2 = 2 * (x_cuda1 + bias_cuda)

                # partition 2 on cpu w/ dependency on partition 1
                y_cpu0 = y_cpu + 1
                x_cuda2_cpu = x_cuda2.cpu()  # adds dependency on gpu computations
                y_cpu1 = y_cpu0 @ weight_cpu + x_cuda2_cpu

                # partition 3 on cuda w/o dependency
                z_cuda0 = z_cuda + 1
                z_cuda1 = z_cuda0 @ weight_cuda
                z_cuda2 = 2 * (z_cuda1 + bias_cuda)

                # partition 4 on cpu w/o dependency
                y_cpu2 = y_cpu + 5
                y_cpu3 = y_cpu2 @ weight_cpu

                # partition 5 on cuda w/o dependency
                u_cuda0 = z_cuda + 3
                u_cuda1 = u_cuda0 @ weight_cuda
                u_cuda2 = 2 * (u_cuda1 + bias_cuda)

                return x_cuda2, y_cpu1, z_cuda2, y_cpu3, u_cuda2

            x_cuda = torch.randn(3, 3, device="cuda")
            y_cpu = torch.randn(3, 3, device="cpu")
            z_cuda = torch.randn(3, 3, device="cuda")
            # Keep the CUDA adds pointwise so this test continues to exercise
            # graph partition reordering instead of addmm fusion.
            bias_cuda = torch.randn(1, 3, device="cuda")
            weight_cuda = torch.randn(3, 3, device="cuda")
            weight_cpu = torch.randn(3, 3, device="cpu")

            eager_out = f(x_cuda, y_cpu, z_cuda, bias_cuda, weight_cuda, weight_cpu)

            compiled_f = torch.compile(f, mode="reduce-overhead")
            for _ in range(3):
                compiled_out = compiled_f(
                    x_cuda, y_cpu, z_cuda, bias_cuda, weight_cuda, weight_cpu
                )
                self.assertEqual(eager_out, compiled_out)

            # the optimal order is
            # [[partition 4 on cpu], [partition 1,3,5 on cuda], [partition 2 on cpu]]
            # since partition2 depends on partition1. So we have 1 cudagraph in total.
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_custom_op_with_no_dependency(self):
            # [FUNCTION DOC] With implicit fallbacks and graph partition on, a function uses an
            # unsafe custom op that creates a boolean mask in the middle. Scheduler reordering
            # moves it to the front so only one cudagraph partition exists; a second call with
            # different sizes recompiles for one new graph (two total). The mask size inputs
            # come from a meta tensor.
            # Two reasons for this:
            # 1. We want to reuse the same mask for many masked_fill calls
            # 2. Prevent inductor from fusing this op into other ops (e.g. masked_fill)
            #    so we can still reorder in scheduler
            @torch.library.custom_op(
                "mylib::create_mask",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def create_mask(
                padded_size: int, original_size: int, device: torch.device
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation: creates a boolean
                # zeros mask of the padded size and sets entries from the original size onward
                # to true.
                mask = torch.zeros((padded_size,), dtype=torch.bool, device=device)
                mask[original_size:] = True
                return mask

            @create_mask.register_fake
            def _(padded_size, original_size, device):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty boolean tensor
                # of the padded size.
                return torch.empty((padded_size,), dtype=torch.bool, device=device)

            def f(padded_tensor, original_tensor, weight):
                # [FUNCTION DOC] HELPER: Pads processing: adds one and relu, dot product with
                # weight added in place, builds the mask, computes the min and max of
                # masked-fill copies (positive and negative infinity) and returns their sum
                # added to the padded tensor.
                original_size = original_tensor.size()[0]
                padded_size = padded_tensor.size()[0]

                # element wise op so we don't care padding value
                padded_tensor = padded_tensor + 1
                padded_tensor = torch.nn.functional.relu(padded_tensor)

                # dot product requires padding with 0
                dot_res = padded_tensor.dot(weight)
                padded_tensor += dot_res

                # min requires padding with inf, so we create mask now
                mask = create_mask(padded_size, original_size, padded_tensor.device)
                min_res = torch.min(
                    torch.ops.aten.masked_fill(padded_tensor, mask, float("inf"))
                )

                # max requires padding with inf. we can reuse previous mask
                max_res = torch.max(
                    torch.ops.aten.masked_fill(padded_tensor, mask, -float("inf"))
                )

                return min_res + max_res + padded_tensor

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(padded_size, original_size):
                # [FUNCTION DOC] HELPER: Builds a padded CUDA tensor with zeros beyond the
                # original size, a meta tensor of the original size and a weight, and checks
                # compiled output equals eager on three calls.
                padded_tensor = torch.randn(padded_size, device="cuda")
                padded_tensor[original_size:] = 0
                original_tensor = torch.randn(original_size, device="meta")

                weight = torch.randn(padded_size, device="cuda")
                eager_out = f(padded_tensor, original_tensor, weight)
                for _ in range(3):
                    compiled_out = compiled_f(padded_tensor, original_tensor, weight)
                    self.assertEqual(eager_out, compiled_out)

            # although custom op `create_mask` happens at the middle of function, reorder
            # moves it to the front so we only have 1 partition. This leads to 1 cudagraph
            run(8, 4)

            # recompilation leads to 1 NEW cudagraph
            run(8, 6)

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_reorder_custom_op_with_no_dependency1(self):
            # [FUNCTION DOC] Variant using a size tensor created by an unsafe custom op instead
            # of a mask. Runs sizes (8,4), (8,6) and (8,7): two cudagraphs are recorded in total
            # (the third call reuses an earlier recording) and compiled output must match eager.
            # wrap with custom op so this is not fused into other ops
            @torch.library.custom_op(
                "mylib::create_size_tensor",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def create_size_tensor(
                tensor: torch.Tensor, device: torch.device
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation: returns a scalar
                # tensor on the device holding the size of the first dimension of the tensor.
                size = tensor.size()[0]
                zero = torch.zeros((), device=device)
                return zero + size

            @create_size_tensor.register_fake
            def _(tensor, device):
                # [FUNCTION DOC] HELPER: Fake implementation returning a 64-bit integer scalar
                # holding the first dimension size.
                size = tensor.size()[0]
                zero = torch.zeros((), device=device, dtype=torch.int64)
                return zero + size

            def fill(
                padded_tensor: torch.Tensor, original_size: torch.Tensor, value
            ) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Replaces entries of the padded tensor at positions at
                # or beyond the original size by the given value.
                padded_size = padded_tensor.size()[0]
                size_range = torch.arange(padded_size, device=padded_tensor.device)
                padded_tensor = torch.where(
                    size_range >= original_size, value, padded_tensor
                )
                return padded_tensor

            def f(padded_tensor, original_tensor, weight):
                # [FUNCTION DOC] HELPER: Processes a padded tensor: add one, relu, dot with
                # weight added in place, gets a size tensor via the custom op, fills padding
                # with positive then negative infinity, takes min and max and returns their sum
                # added to the tensor.
                # element wise op so we don't care padding value
                padded_tensor = padded_tensor + 1
                padded_tensor = torch.nn.functional.relu(padded_tensor)

                # dot product requires padding with 0
                dot_res = padded_tensor.dot(weight)
                padded_tensor += dot_res

                # min requires padding with inf, so we create mask now
                original_size_cuda = create_size_tensor(original_tensor, "cuda")
                padded_tensor = fill(padded_tensor, original_size_cuda, float("inf"))
                min_res = torch.min(padded_tensor)

                # max requires padding with inf. we can reuse previous mask
                padded_tensor = fill(padded_tensor, original_size_cuda, -float("inf"))
                max_res = torch.max(padded_tensor)

                return min_res + max_res + padded_tensor

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(padded_size, original_size):
                # [FUNCTION DOC] HELPER: Builds padded CUDA input, meta size tensor and weight,
                # and checks the compiled result is close to eager on three calls, raising
                # AssertionError if not.
                padded_tensor = torch.randn(padded_size, device="cuda")
                padded_tensor[original_size:] = 0
                original_tensor = torch.randn(original_size, device="meta")
                weight = torch.randn(padded_size, device="cuda")
                eager_out = f(padded_tensor, original_tensor, weight)
                for _ in range(3):
                    compiled_out = compiled_f(padded_tensor, original_tensor, weight)
                    if not torch.allclose(eager_out, compiled_out):
                        raise AssertionError

            # although custom op `create_mask` happens at the middle of function, reorder
            # moves it to the front so we only have 1 partition. This leads to 1 cudagraph
            run(8, 4)

            # recompilation leads to 1 NEW cudagraph
            run(8, 6)

            # reuse previous cudagraph
            run(8, 7)

            self.assertEqual(self.get_manager().new_graph_id().id, 2)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_simple(self):
            # [FUNCTION DOC] With graph partition on, compiles a function with a CPU round trip
            # and checks it matches eager; unless the C++ wrapper is on, the generated code must
            # define a partition function taking an args list and reference the runner recursive
            # apply helper.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Does adds, a CPU round trip and a matmul and returns
                # their sum.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.to("cuda")

            x, y = [torch.ones(2, 2, device="cuda") for _ in range(2)]
            x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
            eager_out = f(x, y)

            f_compiled = torch.compile(f)
            compiled_out = f_compiled(x_cloned, y_cloned)
            self.assertEqual(eager_out, compiled_out)

            _, code = run_and_get_code(f_compiled, x_cloned, y_cloned)

            if not config.cpp_wrapper:
                FileCheck().check("def partition_0(args):").check(
                    "recursively_apply_fns = runner.recursively_apply_fns"
                ).run(code[0])

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_foreach_op(self):
            # [FUNCTION DOC] With graph partition on, compiles a function using a foreach
            # absolute op and checks it matches eager.
            def fn(a0, a1):
                # [FUNCTION DOC] HELPER: Computes foreach absolute values of both inputs and
                # multiplies the first by the first input.
                c = torch._foreach_abs([a0, a1])
                return torch.mul(c[0], a0)

            compiled_fn = torch.compile(fn)

            a0 = torch.randn(2, 3, device="cuda")
            a1 = torch.randn(2, 3, device="cuda")
            eager_out = fn(a0, a1)
            compiled_out = compiled_fn(a0, a1)
            self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_condition_op(self):
            # [FUNCTION DOC] With graph partition on, compiles a function using a conditional on
            # a CUDA predicate with two branches (cosine, sine) and checks it matches eager for
            # a static shape and then a dynamic shape.
            def f(p, b):
                # [FUNCTION DOC] HELPER: Runs a conditional on the predicate choosing between
                # cosine and sine of b.
                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: returns the cosine of x.
                    return torch.cos(x)

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: returns the sine of x.
                    return torch.sin(x)

                return torch.cond(p, true_fn, false_fn, [b])

            compiled_f = torch.compile(f)

            # static shape
            p = torch.tensor([True], device="cuda")
            a = torch.ones([2, 3], device="cuda")
            eager_out = f(p, a)
            compiled_out = compiled_f(p, a)
            self.assertEqual(eager_out, compiled_out)

            # dynamic shape with backed symint
            p = torch.tensor([True], device="cuda")
            a = torch.ones([4, 5], device="cuda")
            eager_out = f(p, a)
            compiled_out = compiled_f(p, a)
            self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        @torch._dynamo.config.patch("capture_scalar_outputs", True)
        def test_graph_partition_unbacked_symint_multi_output_layout(self):
            # [FUNCTION DOC] With graph partition on and scalar output capture, compiles a
            # function that makes a tensor with an unbacked size and runs a conditional whose
            # branches return two tensors; the sum of both outputs must match eager.
            def f(p, size_tensor):
                # [FUNCTION DOC] HELPER: Reads a scalar size from a tensor, creates a ones
                # tensor with that many rows, runs a two-output conditional on it and returns
                # the sum of both outputs.
                size_val = size_tensor.item()
                b = torch.ones([size_val, 3], device="cuda")

                def true_fn(x):
                    # [FUNCTION DOC] HELPER: True branch: returns cosine of x and cosine of x
                    # plus one.
                    return torch.cos(x), torch.cos(x) + 1

                def false_fn(x):
                    # [FUNCTION DOC] HELPER: False branch: returns sine of x and sine of x plus
                    # one.
                    return torch.sin(x), torch.sin(x) + 1

                cond_out = torch.cond(p, true_fn, false_fn, [b])
                return cond_out[0] + cond_out[1]

            compiled_f = torch.compile(f)
            p = torch.tensor([True], device="cuda")
            size_tensor = torch.tensor(2, device="cuda")
            eager_out = f(p, size_tensor)
            compiled_out = compiled_f(p, size_tensor)
            self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_symint(self):
            # [FUNCTION DOC] With graph partition on, runs a function with a CPU round trip on
            # two different square sizes (symbolic shapes) and checks both compiled results
            # match eager.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Does adds, a CPU round trip and a matmul and returns
                # their sum.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.to("cuda")

            f_compiled = torch.compile(f)
            x, y = (
                torch.ones(3, 3, device="cuda"),
                torch.randn(3, 3, device="cuda"),
            )
            compiled_out = f_compiled(x, y)
            self.assertEqual(compiled_out, f(x, y))

            x, y = (
                torch.ones(4, 4, device="cuda"),
                torch.randn(4, 4, device="cuda"),
            )
            compiled_out = f_compiled(x, y)
            self.assertEqual(compiled_out, f(x, y))

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_symint_cat_backward(self):
            # [FUNCTION DOC] With graph partition on, for two input sizes, compares the weight
            # gradient from eager and compiled runs of a function concatenating the input with
            # itself (symbolic size) followed by matmuls; seeds the RNG identically for both and
            # asserts equal gradients.
            def f(x, w):
                # [FUNCTION DOC] HELPER: Concatenates x with itself along rows, multiplies by w
                # and returns the product of that with its own transpose.
                y = torch.cat((x, x), dim=0)
                z = y @ w
                return z @ z.T

            compiled_f = torch.compile(f)

            for shape in (2, 3):
                torch.manual_seed(42)
                eager_x = torch.randn(shape, 2, device="cuda")
                eager_w = torch.randn(2, 2, device="cuda", requires_grad=True)
                torch.manual_seed(42)
                compiled_x = torch.randn(shape, 2, device="cuda")
                compiled_w = torch.randn(2, 2, device="cuda", requires_grad=True)

                f(eager_x, eager_w).sum().backward()
                compiled_f(compiled_x, compiled_w).sum().backward()
                self.assertEqual(eager_w.grad, compiled_w.grad)

        @dynamo_config.patch("capture_dynamic_output_shape_ops", True)
        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_symint_from_nested_indirect_indexing(self):
            # [FUNCTION DOC] With dynamic output shape capture, implicit fallbacks and graph
            # partition on, compiles with the inductor backend a function that does a
            # repeat-interleave then index select driven by a repeats tensor marked dynamic; the
            # output must match eager.
            def nested(x, repeats):
                # [FUNCTION DOC] HELPER: Builds a rank vector for the repeats, repeats it
                # according to repeats, and selects those rows from x.
                rank = torch.arange(repeats.numel(), device=x.device)
                index = rank.repeat_interleave(repeats, dim=0)
                return torch.index_select(x, index=index, dim=0)

            example_inputs = (
                torch.randn((32, 64), device="cuda"),
                repeats := torch.tensor([5, 10, 15], device="cuda"),
            )
            torch._dynamo.mark_dynamic(repeats, 0)  # create backed symint

            nested_opt = torch.compile(nested, backend="inductor")

            expect = nested(*example_inputs)
            actual = nested_opt(*example_inputs)
            self.assertEqual(expect, actual)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_symint_from_mutation_index(self):
            # [FUNCTION DOC] With graph partition on, a compiled function writes minus one into
            # a tensor at an integer index argument, called for indices two through the last,
            # checking each write; then a negative index (new compilation) writes the first
            # element. Verifies symint-from-index mutation handling.
            x = torch.zeros(7, device="cuda")

            def fn(n, a):
                # [FUNCTION DOC] HELPER: Sets the element at index n of a to minus one and
                # returns a.
                a[n] = -1
                return a

            opt_fn = torch.compile(fn, fullgraph=True)

            for n in range(2, x.shape[0]):
                opt_fn(n, x)
                self.assertEqual(x[n], -1)

            # Negative index triggers new compilation.
            opt_fn(-x.shape[0], x)

            self.assertEqual(x[0], -1)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_unbacked_symint(self):
            # [FUNCTION DOC] With graph partition on, marks dimensions of the inputs unbacked
            # and checks a function with a CPU round trip gives the same result as eager.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Does adds, a CPU round trip and a matmul and returns
                # their sum.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                return x1 + y1 + z + y_cpu.to("cuda")

            f_compiled = torch.compile(f)
            x, y = (
                torch.ones(3, 3, device="cuda"),
                torch.randn(3, 3, device="cuda"),
            )

            torch._dynamo.decorators.mark_unbacked(x, 0)
            torch._dynamo.decorators.mark_unbacked(y, 1)

            compiled_out = f_compiled(x, y)
            eager_out = f(x, y)
            self.assertEqual(compiled_out, eager_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_dynamic_scalar_inputs(self):
            # [FUNCTION DOC] Same as the unbacked test but with an additional integer scalar
            # argument added in place; checks compiled results match eager for two different
            # integer values.
            def f(x, y, integer):
                # [FUNCTION DOC] HELPER: Does adds, a CPU round trip and a matmul, adds the
                # integer to the matmul result in place and returns the sum.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x @ y
                z += integer
                return x1 + y1 + z + y_cpu.to("cuda")

            f_compiled = torch.compile(f)
            x, y = (
                torch.ones(3, 3, device="cuda"),
                torch.randn(3, 3, device="cuda"),
            )

            torch._dynamo.decorators.mark_unbacked(x, 0)
            torch._dynamo.decorators.mark_unbacked(y, 1)

            compiled_out = f_compiled(x, y, 5)
            self.assertEqual(compiled_out, f(x, y, 5))

            compiled_out = f_compiled(x, y, 6)
            self.assertEqual(compiled_out, f(x, y, 6))

        @torch._inductor.config.patch("graph_partition", True)
        @torch._dynamo.config.patch("capture_scalar_outputs", True)
        def test_graph_partition_item(self):
            # [FUNCTION DOC] With scalar output capture and graph partition on, compiles a
            # function using the item method and checks it matches eager.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x plus y plus the scalar value of y, where y is
                # x plus one.
                y = x + 1
                scalar = y.item()
                return x + y + scalar

            compiled_f = torch.compile(f)
            compiled_out = compiled_f(torch.tensor(1, device="cuda"))
            self.assertEqual(compiled_out, f(torch.tensor(1, device="cuda")))

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_buffer_reuse(self):
            # [FUNCTION DOC] With graph partition on, checks a function with several CPU round
            # trips and chained matmuls (exercising buffer reuse across partitions) matches
            # eager.
            def f(x, y):
                # [FUNCTION DOC] HELPER: Computes sums, matmuls and two CPU round trips and
                # returns the combined result.
                x1 = x + 1
                y1 = y + 1
                y_cpu = y1.cpu() + 1
                z = x1 + y1 + x @ y
                u = (y_cpu.to("cuda") + 2) @ y + 3
                u_cpu = u.cpu() + 2
                return z + u_cpu.to("cuda")

            x, y = [torch.ones(2, 2, device="cuda") for _ in range(2)]
            x_cloned, y_cloned = [tmp.clone() for tmp in [x, y]]
            eager_out = f(x, y)

            f_compiled = torch.compile(f)
            compiled_out = f_compiled(x_cloned, y_cloned)

            self.assertEqual(eager_out, compiled_out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_fused_scheduler_node(self):
            # [FUNCTION DOC] With graph partition on, compiles a function with a graph break
            # returning aliases of several intermediates and checks the output matches eager.
            def foo(x):
                # [FUNCTION DOC] HELPER: Scales x, makes a first-row alias, scales again with
                # its alias, graph breaks, then returns x, its alias, a tail slice by tensor
                # index of x, the alias of y and a tail slice of y.
                x = x * 20
                x_alias = x[0]
                y = x * 10
                y_alias = y[0]
                torch._dynamo.graph_break()
                ind = torch.tensor(4, device="cuda")
                x_alias2 = x[ind:]
                y_alias2 = y[ind:]
                return x, x_alias, x_alias2, y_alias, y_alias2

            compiled_foo = torch.compile(foo)
            x = torch.rand([20, 20], device="cuda")

            eager_out = foo(x)
            compiled_out = compiled_foo(x)
            self.assertEqual(eager_out, compiled_out)

        # Use autotune_at_compile_time=True to test standalone_compile
        @parametrize("autotune_at_compile_time", [True, False])
        @config.patch("graph_partition", True)
        def test_graph_partition_kernel_reuse(self, autotune_at_compile_time):
            # [FUNCTION DOC] Parametrized by whether compile-time autotuning is on, with graph
            # partition on. A function with two partitions reuses a fused pointwise Triton
            # kernel; with autotune the generated code must contain one auto-tuning block, two
            # definitions of the add kernel (one in the block, one in the main code) and one CPU
            # kernel definition; without it the add kernel is defined once. Results match eager.
            def foo(x):
                # [FUNCTION DOC] HELPER: Computes a matmul and add, a CPU round trip, then a
                # second matmul on the copied-back value and an add, returning both add results.
                # partition 1
                x1 = x @ x
                y1 = x1 + 1
                z_cpu = y1.cpu() + 1
                # partition 2
                # partition 2 should reuse the fused triton kernel generated
                # in partition 1
                x2 = z_cpu.to("cuda") @ z_cpu.to("cuda")
                y2 = x2 + 1
                return y1, y2

            with config.patch(
                "triton.autotune_at_compile_time", autotune_at_compile_time
            ):
                compiled_foo = torch.compile(foo)
                x = torch.randn((20, 20), device="cuda")
                eager_out = foo(x)
                compiled_out, code = run_and_get_code(compiled_foo, x)
                self.assertEqual(eager_out, compiled_out)

                if autotune_at_compile_time:
                    # auto-tuning block should only appear once. We generate auto-tuning code
                    # for all the kernels no matter if they are defined in the main graph or
                    # subgraph, to avoid the overhead of executing multiple auto-tuning code blocks.
                    FileCheck().check_count(
                        "Compile-time auto-tuning block", 1, exactly=True
                    ).run(code[0])
                    # triton_poi_fused_add_ should appear twice, first in the auto-tuning block,
                    # and then in the main code block
                    FileCheck().check_count(
                        "def triton_poi_fused_add_", 2, exactly=True
                    ).run(code[0])
                    # cpu kernel definition should only appence once, not in the auto-tuning block
                    FileCheck().check_count(
                        "cpp_fused__to_copy_add_1 = ", 1, exactly=True
                    ).run(code[0])
                else:
                    # triton_poi_fused_add_ should appear once, because of kernel reuse
                    FileCheck().check_count(
                        "def triton_poi_fused_add_", 1, exactly=True
                    ).run(code[0])

        @unittest.skipIf(
            IS_LINUX or TEST_WITH_SLOW,
            "https://github.com/pytorch/pytorch/issues/176144",
        )
        @unittest.skipUnless(
            config.graph_partition, "Test requires graph_partition to be enabled"
        )
        @config.patch("graph_partition", True)
        @skipIfRocm
        def test_graph_partition_user_defined_triton_kernel_reuse(self):
            # [FUNCTION DOC] Skipped on Linux and in slow mode (linked issue), also requires
            # graph partition and is skipped on ROCm. Compiles a function launching a
            # user-defined Triton add kernel in two partitions and checks outputs match eager
            # and the kernel is compiled only once in the generated code.
            from torch.testing._internal.triton_utils import add_kernel

            def foo(x, y):
                # [FUNCTION DOC] HELPER: Launches the add kernel into a first output,
                # round-trips it through CPU adding one, moves it back and launches the add
                # kernel again into a second output; returns both outputs.
                # partition 1
                output1 = torch.empty_like(x)
                add_kernel[(4,)](x, y, output1, n_elements=128, BLOCK_SIZE=16)
                output1_cpu = output1.cpu() + 1
                # partition 2 should reuse the user-defined kernel
                x2 = output1_cpu.to("cuda")
                output2 = torch.empty_like(x)
                add_kernel[(4,)](x2, y, output2, n_elements=128, BLOCK_SIZE=16)
                return output1, output2

            compiled_foo = torch.compile(foo)
            x = torch.randn(128, device="cuda")
            y = torch.randn(128, device="cuda")
            eager_out = foo(x, y)
            compiled_out, code = run_and_get_code(compiled_foo, x, y)
            self.assertEqual(eager_out, compiled_out)
            FileCheck().check_count(
                "async_compile.triton('add_kernel',", 1, exactly=True
            ).run(code[0])

        def test_meta_tensor(self):
            # [FUNCTION DOC] Compiles a function taking a meta tensor and a CUDA tensor in
            # reduce-overhead mode, runs it repeatedly and checks it matches eager (meta output)
            # and exactly one graph is recorded.
            def foobar(x, y):
                # [FUNCTION DOC] HELPER: Returns twice x and three times y.
                return x * 2, y * 3

            foo_c = torch.compile(mode="reduce-overhead")(foobar)
            t = torch.empty((1, 16, 128, 128), device="meta")
            y = torch.rand([64], device="cuda")

            eager_out = foobar(t, y)

            for _ in range(3):
                compiled_out = foo_c(t, y)

            compiled_out = foo_c(t, y)
            self.assertEqual(eager_out, compiled_out)
            self.assertEqual(self.get_manager().new_graph_id().id, 1)

        @torch._inductor.config.patch("triton.cudagraph_capture_sizes", (2, 5, 7))
        def test_cudagraph_capture_sizes(self):
            # [FUNCTION DOC] With capture sizes limited to a set of first-dimension sizes (2, 5
            # and 7), runs a dynamic first dimension for sizes one through nine; only three
            # graphs are recorded.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(shape):
                # [FUNCTION DOC] HELPER: Makes a random tensor with the given first dimension
                # and five columns, marks the first dimension dynamic and calls the compiled
                # function three times.
                x = torch.randn((shape, 5), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                for _ in range(3):
                    f(x)

            for i in range(1, 10):
                run(i)

            self.assertEqual(self.get_manager().new_graph_id().id, 3)

        @torch._inductor.config.patch(
            "triton.cudagraph_capture_sizes",
            (
                (2, 3),
                (4, 5),
                (6, 2),
                (7, 3),
            ),
        )
        def test_cudagraph_capture_sizes1(self):
            # [FUNCTION DOC] With capture sizes given as pairs of (batch, sequence) values, runs
            # dynamic first and second dimensions for sizes two through nine each; only four
            # graphs are recorded.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(batch_size, seq_len, d):
                # [FUNCTION DOC] HELPER: Makes a random three-dimensional tensor, marks the
                # first two dimensions dynamic and calls the compiled function three times.
                x = torch.randn((batch_size, seq_len, d), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                for _ in range(3):
                    f(x)

            for i in range(2, 10):
                for j in range(2, 10):
                    run(i, j, 8)

            self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @torch._inductor.config.patch(
            "triton.cudagraph_capture_sizes",
            (
                (2, 3, 4),
                (4, 4, 3),
                (3, 4, 4),
                (4, 2, 3),
            ),
        )
        def test_cudagraph_capture_sizes2(self):
            # [FUNCTION DOC] With capture sizes given as triples, runs all combinations of sizes
            # two through four with all three dimensions dynamic; only four graphs are recorded.
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            f = torch.compile(f, mode="reduce-overhead")

            def run(batch_size, seq_len, d):
                # [FUNCTION DOC] HELPER: Makes a random three-dimensional tensor, marks all
                # three dimensions dynamic and calls the compiled function three times.
                x = torch.randn((batch_size, seq_len, d), device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                torch._dynamo.mark_dynamic(x, 1)
                torch._dynamo.mark_dynamic(x, 2)
                for _ in range(3):
                    f(x)

            for i in range(2, 5):
                for j in range(2, 5):
                    for k in range(2, 5):
                        run(i, j, k)

            self.assertEqual(self.get_manager().new_graph_id().id, 4)

        @torch._inductor.config.patch("triton.cudagraph_or_error", True)
        def test_cudagraph_or_error(self):
            # [FUNCTION DOC] With the cudagraph-or-error option on, calling a compiled function
            # that mutates its input must raise a RuntimeError instead of silently skipping
            # cudagraphs.
            def f(x):
                # [FUNCTION DOC] HELPER: Adds one to x in place and returns it.
                x.add_(1)
                return x

            f = torch.compile(f, mode="reduce-overhead")

            with self.assertRaises(RuntimeError):
                f(torch.tensor(1, device="cuda"))

        @config.patch(implicit_fallbacks=True)
        @torch._inductor.config.patch("graph_partition", True)
        def test_graph_partition_input_layout_symints(self):
            # [FUNCTION DOC] Regression test: with graph partition on, an unsafe identity custom
            # op feeds a custom op that pads its input to a fixed 32 rows, so the op output
            # shape is static while the input layout depends on a dynamic size. Partition
            # functions must still receive symbolic size arguments (a union that was not applied
            # in place had dropped them). Runs sizes 20, 10 and 25 and compares to eager.
            """
            Test that partition functions receive symint args derived from input
            tensor layouts, even when the op's output shape is static.
            Regression test for symbol_inputs being silently dropped due to
            OrderedSet.union() (returns new set) instead of .update() (in-place).
            """

            @torch.library.custom_op(
                "mylib::identity_unsafe",
                mutates_args=(),
                tags=(torch._C.Tag.cudagraph_unsafe,),
            )
            def identity_unsafe(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Unsafe custom op implementation returning a clone of x.
                return x.clone()

            @identity_unsafe.register_fake
            def _identity_fake(x):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor like x.
                return torch.empty_like(x)

            @torch.library.custom_op(
                "mylib::pad_to_fixed",
                mutates_args=(),
            )
            def pad_to_fixed(x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Custom op implementation: copies x into the top rows of
                # a zero tensor with 32 rows and the same number of columns and returns it.
                n = x.shape[0]
                result = torch.zeros(32, x.shape[1], device=x.device, dtype=x.dtype)
                result[:n] = x
                return result

            @pad_to_fixed.register_fake
            def _pad_fake(x):
                # [FUNCTION DOC] HELPER: Fake implementation returning an empty tensor with 32
                # rows and the input column count.
                return torch.empty(32, x.shape[1], device=x.device, dtype=x.dtype)

            def f(x):
                # [FUNCTION DOC] HELPER: Applies the unsafe op, then the padding op, and adds
                # one.
                y = torch.ops.mylib.identity_unsafe(x)
                z = torch.ops.mylib.pad_to_fixed(y)
                return z + 1

            compiled_f = torch.compile(f, mode="reduce-overhead")

            def run(n):
                # [FUNCTION DOC] HELPER: Creates a random input with n rows, marks the first
                # dimension dynamic, computes the eager result and checks it against the
                # compiled one on three calls.
                x = torch.randn(n, 16, device="cuda")
                torch._dynamo.mark_dynamic(x, 0)
                eager_out = f(x)
                for _ in range(3):
                    compiled_out = compiled_f(x)
                    self.assertEqual(eager_out, compiled_out)

            run(20)
            run(10)
            run(25)

        def test_dealloc_handles_tensor_weakrefs_stack_traces_mismatch(self):
            # [FUNCTION DOC] Regression test for the deallocation path when the weak reference
            # list and the stack trace list of a node differ in length. After a warmup call it
            # appends an extra entry to the weak reference list of the current node, then makes
            # a second call that transitions warmup to recording and calls the dealloc routine;
            # it must not raise an assertion (the code now warns and truncates to the shorter
            # list).
            """Verify dealloc_current_path_weakrefs handles tensor_weakrefs /
            stack_traces length mismatch without asserting.

            This injects the mismatch directly so the deallocation path is
            deterministic. The fix replaces the hard assert with a warning and
            relies on zip(), matching the existing defensive pattern for
            outputs_weakrefs in the same function.
            """

            @torch.compile(mode="reduce-overhead")
            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            inp = torch.rand([4], device="cuda")

            # First call: warmup phase creates and runs a CUDAWarmupNode.
            torch.compiler.cudagraph_mark_step_begin()
            foo(inp)

            # Simulate the tensor_weakrefs / stack_traces length divergence.
            node = self.curr_node()
            self.assertEqual(len(node.tensor_weakrefs), len(node.stack_traces))
            node.tensor_weakrefs.append(None)

            # Second call: transitions warmup -> recording, which invokes
            # try_end_curr_warmup -> dealloc_current_path_weakrefs.
            # cudagraph_mark_step_begin ensures can_start_new_generation()
            # returns True so dealloc is actually called.
            # Without the fix, this would hit:
            #   assert len(node.tensor_weakrefs) == len(node.stack_traces)
            # With the fix, zip() safely truncates to the shorter list.
            torch.compiler.cudagraph_mark_step_begin()
            foo(inp)

        @torch._inductor.config.patch("triton.skip_cudagraph_warmup", True)
        def test_opaque_value_input_cudagraph(self):
            # [FUNCTION DOC] With warmup skipped, runs the tree API with an opaque object (a
            # scale-factor holder) next to a tensor input: the opaque object must be treated as
            # a static input so it is not copied like a tensor. Two calls must both return the
            # tensor scaled by three.
            """Opaque reference-type objects (e.g. DeviceMesh, ProcessGroup)
            passed alongside tensors must be marked "static" so they are
            excluded from the tensor-copy path (non_static_input_idx).
            "Static" here just means "don't copy as a tensor", not that
            the object is semantically immutable."""

            sf = _CudagraphTestScaleFactor(3.0)

            def foo(args):
                # [FUNCTION DOC] HELPER: Unpacks the opaque object and the tensor from the list,
                # clears the list and returns a one-element tuple with the tensor times the
                # object's factor.
                obj = args[0]
                x = args[1]
                args.clear()
                return (x * obj.factor,)

            inp = torch.rand([4], device="cuda")
            foo_cg = self.cudagraphify_impl(foo, [sf, inp], ())
            result = foo_cg([sf, inp])
            self.assertEqual(result[0], inp * 3.0)

            result2 = foo_cg([sf, inp])
            self.assertEqual(result2[0], inp * 3.0)

    class TestCUDAGraphPolicy(TestCase):
        # [CLASS DOC] Tests for the pluggable cudagraph policy config (custom cudagraphify,
        # whether to wrap, wrapping of compiled outputs). setUp clears counters, enables
        # cudagraphs and cudagraph trees in config and resets dynamo; tearDown resets dynamo and
        # closes the config stack.
        def setUp(self):
            # [FUNCTION DOC] HELPER: Runs parent setup, clears counters, enters a config patch
            # enabling cudagraphs and trees and resets dynamo.
            super().setUp()
            counters.clear()
            self._stack = contextlib.ExitStack()
            self._stack.enter_context(
                config.patch(
                    {
                        "triton.cudagraphs": True,
                        "triton.cudagraph_trees": True,
                    }
                )
            )
            torch._dynamo.reset()

        def tearDown(self):
            # [FUNCTION DOC] HELPER: Runs parent teardown, resets dynamo and closes the config
            # stack.
            super().tearDown()
            torch._dynamo.reset()
            self._stack.close()

        def test_policy_cudagraphify_called(self):
            # [FUNCTION DOC] Installs a recording policy whose cudagraphify override logs its
            # static input indices and device index and returns the model unwrapped; compiles
            # and runs a function twice and asserts the override was called at least once.
            """Custom policy's cudagraphify is called instead of the default."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            calls = []

            class RecordingPolicy(CUDAGraphPolicy):
                # [CLASS DOC] HELPER: Policy subclass that records the keyword arguments of each
                # cudagraphify call in an outer list and returns the model unchanged.
                def cudagraphify(self, model, inputs, static_input_idxs, **kwargs):
                    # [FUNCTION DOC] HELPER: Appends a record with the static input indices and
                    # device index keyword to the list and returns the model.
                    calls.append(
                        {
                            "static_input_idxs": static_input_idxs,
                            "device_index": kwargs.get("device_index"),
                        }
                    )
                    return model

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus one.
                return x * x + 1

            with config.patch("cudagraph_policy", RecordingPolicy()):
                compiled = torch.compile(foo)
                x = torch.randn(4, device="cuda")
                compiled(x)
                compiled(x)

            self.assertGreater(len(calls), 0)

        def test_should_wrap_false_skips_cudagraph(self):
            # [FUNCTION DOC] A policy whose should_wrap says no makes cudagraph wrapping be
            # skipped: results are correct and the cudagraph skips counter is positive.
            """When should_wrap returns False, cudagraph wrapping is skipped."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            class NoWrapPolicy(CUDAGraphPolicy):
                # [CLASS DOC] HELPER: Policy subclass that declines to wrap every compiled
                # graph.
                def should_wrap(self, compiled_graph):
                    # [FUNCTION DOC] HELPER: Always returns false.
                    return False

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x plus one.
                return x + 1

            with config.patch("cudagraph_policy", NoWrapPolicy()):
                compiled = torch.compile(foo)
                x = torch.randn(4, device="cuda")
                result = compiled(x)
                result = compiled(x)

            self.assertEqual(result, x + 1)
            self.assertGreater(counters["inductor"]["cudagraph_skips"], 0)

        def test_wrap_output_called(self):
            # [FUNCTION DOC] A policy overriding wrap_output records the type name of each
            # output object passed; after compiling and running twice, results are correct, at
            # least one call was recorded and the compiled fx graph type name is among them.
            """Policy's wrap_output is called in BundledOutputCodeLoadable."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            wrapped = []

            class WrapPolicy(CUDAGraphPolicy):
                # [CLASS DOC] HELPER: Policy subclass that records the class name of each output
                # code object and returns it unchanged.
                def wrap_output(self, output_code):
                    # [FUNCTION DOC] HELPER: Appends the class name of the output object to a
                    # list and returns the object.
                    wrapped.append(type(output_code).__name__)
                    return output_code

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns twice x.
                return x * 2

            with config.patch("cudagraph_policy", WrapPolicy()):
                compiled = torch.compile(foo)
                x = torch.randn(4, device="cuda")
                result = compiled(x)
                result = compiled(x)

            self.assertEqual(result, x * 2)
            self.assertGreater(len(wrapped), 0)
            self.assertIn("CompiledFxGraph", wrapped)

        def test_default_policy_matches_builtin(self):
            # [FUNCTION DOC] Checks that compiling with the base policy installed gives the same
            # result as compiling with no policy.
            """Default CUDAGraphPolicy produces same results as no policy."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus x.
                return x * x + x

            x = torch.randn(4, device="cuda")

            compiled_default = torch.compile(foo)
            ref = compiled_default(x)
            ref = compiled_default(x)

            torch._dynamo.reset()

            with config.patch("cudagraph_policy", CUDAGraphPolicy()):
                compiled_policy = torch.compile(foo)
                out = compiled_policy(x)
                out = compiled_policy(x)

            self.assertEqual(ref, out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_default_policy_matches_builtin_partition(self):
            # [FUNCTION DOC] Same as the default policy equivalence test but with graph
            # partition enabled.
            """Default CUDAGraphPolicy matches builtin when graph_partition=True."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus x.
                return x * x + x

            x = torch.randn(4, device="cuda")

            compiled_default = torch.compile(foo)
            ref = compiled_default(x)
            ref = compiled_default(x)

            torch._dynamo.reset()

            with config.patch("cudagraph_policy", CUDAGraphPolicy()):
                compiled_policy = torch.compile(foo)
                out = compiled_policy(x)
                out = compiled_policy(x)

            self.assertEqual(ref, out)

        @torch._inductor.config.patch("graph_partition", True)
        def test_policy_cudagraphify_partition(self):
            # [FUNCTION DOC] With graph partition on, a recording policy still has its
            # cudagraphify override called at least once.
            """Custom policy's cudagraphify is called when graph_partition is enabled."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            calls = []

            class RecordingPolicy(CUDAGraphPolicy):
                # [CLASS DOC] HELPER: Policy subclass that records the keyword arguments of each
                # cudagraphify call in an outer list and returns the model unchanged.
                def cudagraphify(self, model, inputs, static_input_idxs, **kwargs):
                    # [FUNCTION DOC] HELPER: Appends a record with the static input indices and
                    # device index keyword to the list and returns the model.
                    calls.append(
                        {
                            "static_input_idxs": static_input_idxs,
                            "device_index": kwargs.get("device_index"),
                        }
                    )
                    return model

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus one.
                return x * x + 1

            with config.patch("cudagraph_policy", RecordingPolicy()):
                compiled = torch.compile(foo)
                x = torch.randn(4, device="cuda")
                compiled(x)
                compiled(x)

            self.assertGreater(len(calls), 0)

        def test_should_wrap_false_with_wrap_output(self):
            # [FUNCTION DOC] A policy that declines inner wrapping but overrides outer wrapping:
            # results correct, cudagraph skips counter positive, and the output wrapping hook
            # called at least once.
            """should_wrap=False skips inner wrapping; wrap_output does outer wrapping."""
            from torch._inductor.cudagraph_utils import CUDAGraphPolicy

            wrap_calls = []

            class OuterOnlyPolicy(CUDAGraphPolicy):
                # [CLASS DOC] HELPER: Policy subclass whose should_wrap is false and whose
                # wrap_output records the output object's class name and returns it unchanged.
                def should_wrap(self, compiled_graph):
                    # [FUNCTION DOC] HELPER: Always returns false.
                    return False

                def wrap_output(self, output_code):
                    # [FUNCTION DOC] HELPER: Appends the class name of the output object to a
                    # list and returns the object.
                    wrap_calls.append(type(output_code).__name__)
                    return output_code

            def foo(x):
                # [FUNCTION DOC] HELPER: Returns x squared plus one.
                return x * x + 1

            with config.patch("cudagraph_policy", OuterOnlyPolicy()):
                compiled = torch.compile(foo)
                x = torch.randn(4, device="cuda")
                result = compiled(x)
                result = compiled(x)

            self.assertEqual(result, x * x + 1)
            self.assertGreater(counters["inductor"]["cudagraph_skips"], 0)
            self.assertGreater(len(wrap_calls), 0)

    class TestSAC(TestCase):
        # [CLASS DOC] Tests for selective activation checkpointing RNG handling: with
        # checkpointed random ops, forward and recomputation must see the same random values,
        # using dispatch mode observers and CUDA graph-safe RNG state helpers.
        def _make_observer_mode(self):
            # [FUNCTION DOC] HELPER: Defines and returns a dispatch mode class (not an instance)
            # that forwards every op unchanged and keeps a per-op mapping of recorded outputs
            # and a run counter.
            class ObserverMode(TorchDispatchMode):
                # [CLASS DOC] HELPER: Torch dispatch mode with a run counter and a mapping from
                # op to a list of outputs; its dispatch hook just runs the op. Output recording
                # is done by registered higher-order-op implementations in the tests.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: Initializes the parent mode, sets the run counter
                    # to zero and creates an empty list-valued mapping of op outputs.
                    super().__init__()
                    self.curr_run = 0
                    self.op_outputs = defaultdict(list)

                def __torch_dispatch__(
                    self,
                    func: OpOverload,
                    types: Sequence[type],
                    args: Sequence[object] = (),
                    kwargs: Mapping[str, object] = immutable_dict(),
                ) -> object:
                    # [FUNCTION DOC] HELPER: Dispatch hook: runs the op with the given arguments
                    # and returns the result.
                    return func(*args, **kwargs)

            return ObserverMode

        def test_simple(self):
            # [FUNCTION DOC] Registers a mode-specific implementation of the graph-safe
            # run-with-rng-state higher-order op that records each output per op. Compiles a
            # checkpointed function with random ops through the aot eager decomp partition
            # backend for two rounds of forward and backward, and asserts four random outputs
            # were recorded, with each pair (forward and recompute) equal and the two rounds
            # different.
            device = "cuda"

            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            ObserverMode = self._make_observer_mode()

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Registered implementation for the observer mode: runs
                # the higher-order op with dispatch disabled, appends the result to the mode's
                # list for that op and returns it.
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[op].append(out)
                return out

            obs = ObserverMode()

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)

            for _ in range(2):
                torch._dynamo.reset()

                def gn(x, y):
                    # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                    return torch.sigmoid(torch.rand_like(x) * y) * x

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Sine of x, then a reentrant checkpoint of the
                    # random function, then sine again.
                    x = torch.sin(x)
                    x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                    x = torch.sin(x)
                    return x

                aot_eager_decomp_partition = functools.partial(
                    aot_eager_decomp_partition_with_mode, mode=obs
                )

                fn = torch.compile(fn, backend=aot_eager_decomp_partition)

                fn(x, y).sum().backward()

            self.assertEqual(len(obs.op_outputs[aten.rand.default]), 4)
            for i in range(2):
                self.assertEqual(
                    obs.op_outputs[aten.rand.default][0 + 2 * i],
                    obs.op_outputs[aten.rand.default][1 + 2 * i],
                )
            self.assertNotEqual(
                obs.op_outputs[aten.rand.default][0],
                obs.op_outputs[aten.rand.default][2],
            )

        def test_cudagraph_uneven_forward_backward(self):
            # [FUNCTION DOC] Shows that updating the backward RNG state works with CUDA graphs
            # under uneven forward counts. Compares a plain state save and restore flow (the
            # permutation in forward equals the replay in backward) with a graph-safe generator
            # state flow, both without and with capturing the backward in a CUDA graph and
            # replaying it.
            # torch.compile cudagraphs are difficult to test
            # the rng updating bc is sensitive to duration of pending backwards, etc.
            # this is a short repro to mimic the runtime wrappers integration
            # and show that updating the backward rng state with cudagraphs works:
            def forward():
                # [FUNCTION DOC] HELPER: Returns the current CUDA RNG state and a random
                # permutation of ten elements.
                state = torch.cuda.get_rng_state()
                perm = torch.randperm(10, device="cuda")
                return state, perm

            def backward(rng_state):
                # [FUNCTION DOC] HELPER: Saves the current RNG state, restores the given state,
                # draws a random permutation, restores the saved state and returns the
                # permutation.
                current_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state.cpu())
                perm = torch.randperm(10, device="cuda")
                torch.cuda.set_rng_state(current_state)
                return perm

            def normal_test():
                # [FUNCTION DOC] HELPER: Runs forward then backward with the saved state;
                # returns the two permutations.
                state, perm = forward()
                repro_perm = backward(state)
                return perm, repro_perm

            def graphsafe_forward():
                # [FUNCTION DOC] HELPER: Returns a random permutation of ten elements on CUDA.
                perm = torch.randperm(10, device="cuda")
                return perm

            def graphsafe_backward(generator, new_state):
                # [FUNCTION DOC] HELPER: Graph-safe version of the backward: swaps the
                # generator's state to the new state, draws a permutation, restores the previous
                # state and returns the permutation.
                current_state = generator.graphsafe_get_state()
                generator.graphsafe_set_state(new_state)
                perm = torch.randperm(10, device="cuda")
                generator.graphsafe_set_state(current_state)
                return perm

            def graph_test(generator, capture_cuda_graph):
                # [FUNCTION DOC] HELPER: Clones the generator state, optionally captures the
                # graph-safe backward in a CUDA graph, runs three uneven forwards, records the
                # state, runs a fourth forward, sets the cloned state to the recorded one and
                # obtains the backward permutation by graph replay or direct call; returns the
                # forward and backward permutations.
                if capture_cuda_graph:
                    graph = torch.cuda.CUDAGraph()

                # state should be cloned before the graph
                old_state = generator.graphsafe_get_state()
                new_state = old_state.clone_state()

                if capture_cuda_graph:
                    # only capturing the backward
                    with torch.cuda.graph(graph):
                        repro_perm = graphsafe_backward(generator, new_state)

                # some number of uneven forwards
                graphsafe_forward()
                graphsafe_forward()
                graphsafe_forward()

                # state prior to rng invocation
                state = generator.get_state()
                perm = graphsafe_forward()

                new_state.set_state(state)

                if capture_cuda_graph:
                    graph.replay()
                else:
                    repro_perm = graphsafe_backward(generator, new_state)

                return perm, repro_perm

            self.assertEqual(*normal_test())
            generator = torch.cuda.default_generators[0]
            self.assertEqual(*graph_test(generator, capture_cuda_graph=False))
            self.assertEqual(*graph_test(generator, capture_cuda_graph=True))

        def test_cpu_and_cuda_rng(self):
            # [FUNCTION DOC] Registers observer implementations for three RNG-state higher-order
            # ops, then compiles a function mixing CUDA random numbers (checkpointed) and a CPU
            # random permutation (checkpointed) with the aot eager decomp partition backend and
            # runs backward. For both rand and randperm, exactly two outputs must be recorded,
            # equal to each other, with the permutation on CPU and the rand on CUDA.
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import (
                graphsafe_run_with_rng_state,
                run_and_save_rng_state,
                run_with_rng_state,
            )

            for hop in [
                graphsafe_run_with_rng_state,
                run_and_save_rng_state,
                run_with_rng_state,
            ]:

                def make_impl(hop):
                    # [FUNCTION DOC] HELPER: Registers on the given higher-order op an
                    # observer-mode implementation that runs the op without dispatch, finds the
                    # op overload among the arguments (must exist), records the output (the
                    # second element for the save-state op) in the mode mapping and returns the
                    # output.
                    @hop.py_impl(ObserverMode)
                    def _(mode, *args, **kwargs):
                        # [FUNCTION DOC] HELPER: The registered implementation described in the
                        # enclosing function.
                        with no_dispatch():
                            out = hop(*args, **kwargs)

                        op = None
                        for inp in itertools.chain(args, kwargs.values()):
                            if isinstance(inp, torch._ops.OpOverload):
                                op = inp
                                break
                        assert op is not None  # noqa: S101
                        if hop is run_and_save_rng_state:
                            mode.op_outputs[op].append(out[1])
                        else:
                            mode.op_outputs[op].append(out)
                        return out

                make_impl(hop)

            obs = ObserverMode()

            def gn(x, y):
                # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def gn2(x):
                # [FUNCTION DOC] HELPER: Multiplies x by a random permutation of its element
                # count reshaped to its shape.
                return x * torch.randperm(x.numel(), device=x.device).reshape(x.shape)

            def fn(x, y, z):
                # [FUNCTION DOC] HELPER: Sine of x, checkpointed random function, sine again,
                # multiplied by the checkpointed permutation function of z moved to CUDA.
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                z = torch.utils.checkpoint.checkpoint(gn2, z, use_reentrant=True)
                return x * z.cuda()

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn = torch.compile(fn, backend=aot_eager_decomp_partition)

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)
            z = torch.randn(4, 4, requires_grad=True)

            fn(x, y, z).sum().backward()
            for op in [aten.rand.default, aten.randperm.default]:
                self.assertEqual(len(obs.op_outputs[op]), 2)
                self.assertEqual(
                    obs.op_outputs[op][0],
                    obs.op_outputs[op][1],
                )
                self.assertEqual(
                    obs.op_outputs[op][0].device.type,
                    "cpu" if op == aten.randperm.default else "cuda",
                )

        @parametrize("order", (list(itertools.permutations([0, 1, 2]))))
        def test_uneven_forward_backward(self, order):
            # [FUNCTION DOC] Parametrized over every ordering of three backward calls. Compiles
            # a function with two checkpointed random ops (rand and randperm) through the aot
            # eager decomp partition backend, runs three forwards (tagging the observer with the
            # run index), then runs backward on the outputs in the parametrized order, tagging
            # each. For every run each op must have exactly two recorded outputs that are equal
            # (forward and recompute), and consecutive runs must differ.
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Registered observer implementation: runs the graph-safe
                # RNG higher-order op with dispatch disabled and appends the output to the
                # mode's list keyed by current run index and op; returns the output.
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[(mode.curr_run, op)].append(out)
                return out

            obs = ObserverMode()

            def gn(x, y):
                # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def gn2(x):
                # [FUNCTION DOC] HELPER: Multiplies x by a random permutation of its element
                # count reshaped to its shape.
                return x * torch.randperm(x.numel(), device=x.device).reshape(x.shape)

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Sine of x, checkpointed random function, sine, then
                # checkpointed permutation function.
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn2, x, use_reentrant=True)
                return x

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn_c = torch.compile(fn, backend=aot_eager_decomp_partition)

            torch.manual_seed(0)
            outs = []
            for i in range(len(order)):
                obs.curr_run = i
                x = torch.randn(4, 4, device=device, requires_grad=True)
                y = torch.randn(4, 4, device=device, requires_grad=True)
                outs.append(fn_c(x, y))

            for idx in order:
                obs.curr_run = idx
                outs[idx].sum().backward()

            for run in range(len(order)):
                for op in (aten.rand.default, aten.randperm.default):
                    self.assertEqual(len(obs.op_outputs[(run, op)]), 2)
                    self.assertEqual(
                        obs.op_outputs[(run, op)][0],
                        obs.op_outputs[(run, op)][1],
                    )
                    if run != 0:
                        self.assertNotEqual(
                            obs.op_outputs[(run - 1, op)][0],
                            obs.op_outputs[(run, op)][0],
                        )

        @config.patch(fallback_random=True)
        @config.patch("test_configs.graphsafe_rng_func_ignores_fallback_random", True)
        def _test_cudagraphs_aot_eager_compat_equal(self, device):
            # [FUNCTION DOC] HELPER: Shared body (with fallback random on and graph-safe RNG
            # ignoring that fallback) comparing the aot eager decomp partition backend against
            # reduce-overhead mode. For the given device, seeds the RNG then runs three forward
            # and backward iterations per backend on a checkpointed random function, collecting
            # cloned outputs and gradients. Asserts outputs and gradients are equal between
            # backends and no cudagraph skips occurred.
            def gn(x, y):
                # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Sine of x, checkpointed random function, then sine.
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            outs = []
            grads = []

            outs2 = []
            grads2 = []

            compile_fns = [
                lambda fn: torch.compile(fn, backend="aot_eager_decomp_partition"),
                lambda fn: torch.compile(fn, mode="reduce-overhead"),
            ]
            for i, compile_fn in enumerate(compile_fns):
                torch.manual_seed(0)
                for _ in range(3):
                    x = torch.randn(4, 4, device=device, requires_grad=True)
                    y = torch.randn(4, 4, device=device, requires_grad=True)

                    out = compile_fn(fn)(x, y)
                    torch.cuda.synchronize()
                    out.sum().backward()
                    if i == 0:
                        outs.append(out.clone())
                        grads.append((x.grad.clone(), y.grad.clone()))
                    else:
                        outs2.append(out.clone())
                        grads2.append((x.grad.clone(), y.grad.clone()))

            self.assertEqual(outs, outs2)
            self.assertEqual(grads, grads2)
            self.assertEqual(counters["inductor"]["cudagraph_skips"], 0)

        def test_cudagraphs_aot_eager_compat_equal(self):
            # [FUNCTION DOC] Runs the shared backend-compatibility comparison on the first CUDA
            # device.
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:0"))

        @requires_multigpu()
        def test_cudagraphs_aot_eager_compat_equal_device_one(self):
            # [FUNCTION DOC] Multi-GPU only: runs the shared backend-compatibility comparison on
            # the second CUDA device.
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:1"))

        @config.patch(graph_partition=True)
        def test_graph_partition_cudagraphs_aot_eager_compat_equal(self):
            # [FUNCTION DOC] With graph partition on, runs the shared backend-compatibility
            # comparison on the first CUDA device.
            self._test_cudagraphs_aot_eager_compat_equal(torch.device("cuda:0"))

        @requires_multigpu()
        def test_multi_device(self):
            # [FUNCTION DOC] Multi-GPU only: compiles with the aot eager decomp partition
            # backend a function applying a checkpointed random function to inputs on two
            # different devices and runs backward on the first output; only checks that no error
            # is raised.
            def gn(x, y):
                # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Sine of x, checkpointed random function, then sine.
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            def multi_fn(x, y, a, b):
                # [FUNCTION DOC] HELPER: Applies the checkpointed function to each of the two
                # input pairs and returns both results.
                return fn(x, y), fn(a, b)

            x = torch.randn(4, 4, device="cuda:0", requires_grad=True)
            y = torch.randn(4, 4, device="cuda:0", requires_grad=True)

            a = torch.randn(4, 4, device="cuda:1", requires_grad=True)
            b = torch.randn(4, 4, device="cuda:1", requires_grad=True)

            # No errors. TODO - get graphs from logging, couldn't figure out how
            multi_fn_c = torch.compile(multi_fn, backend="aot_eager_decomp_partition")

            out = multi_fn_c(x, y, a, b)
            out[0].sum().backward()

        def test_retain_graph(self):
            # [FUNCTION DOC] Compiles a checkpointed random function with the aot eager decomp
            # partition backend and calls backward twice (first retaining the graph). Exactly
            # three random outputs must have been recorded (forward plus one recompute per
            # backward) and all three must be equal.
            device = "cuda"

            ObserverMode = self._make_observer_mode()
            from torch._prims.rng_prims import graphsafe_run_with_rng_state

            @graphsafe_run_with_rng_state.py_impl(ObserverMode)
            def _(mode, op, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Registered observer implementation: runs the graph-safe
                # RNG higher-order op with dispatch disabled, appends the output to the mode's
                # list for that op and returns it.
                with no_dispatch():
                    out = graphsafe_run_with_rng_state(op, *args, **kwargs)

                mode.op_outputs[op].append(out)
                return out

            obs = ObserverMode()

            def gn(x, y):
                # [FUNCTION DOC] HELPER: Returns sigmoid of random numbers times y, times x.
                return torch.sigmoid(torch.rand_like(x) * y) * x

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Sine of x, checkpointed random function, then sine.
                x = torch.sin(x)
                x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
                x = torch.sin(x)
                return x

            x = torch.randn(4, 4, device=device, requires_grad=True)
            y = torch.randn(4, 4, device=device, requires_grad=True)

            aot_eager_decomp_partition = functools.partial(
                aot_eager_decomp_partition_with_mode, mode=obs
            )

            fn = torch.compile(fn, backend=aot_eager_decomp_partition)

            out = fn(x, y).sum()
            out.backward(retain_graph=True)
            out.backward()
            self.assertEqual(len(obs.op_outputs[aten.rand.default]), 3)
            self.assertEqual(
                obs.op_outputs[aten.rand.default][0],
                obs.op_outputs[aten.rand.default][1],
            )
            self.assertEqual(
                obs.op_outputs[aten.rand.default][1],
                obs.op_outputs[aten.rand.default][2],
            )

    instantiate_parametrized_tests(CudaGraphTreeTests)
    instantiate_parametrized_tests(TestCUDAGraphPolicy)
    instantiate_parametrized_tests(TestSAC)

    # OpInfo-based test for index/scatter ops with cudagraphs
    from torch.testing._internal.common_device_type import (
        DeviceTypeTestBase,
        instantiate_device_type_tests,
        ops,
    )
    from torch.testing._internal.common_methods_invocations import op_db

    # Ops that involve indexing/scattering that we want to test with cudagraphs
    INDEXING_OPS = (
        "index_put",
        "index_add",
        "index_copy",
        "index_fill",
        "scatter",
    )

    indexing_op_db = [
        op for op in op_db if any(idx_op in op.name for idx_op in INDEXING_OPS)
    ]

    class TestCudagraphIndexingOps(DeviceTypeTestBase):
        # [CLASS DOC] Device type test class (instantiated for CUDA only when available) that
        # runs OpInfo based index, scatter and fill style ops, in float32, compiled with
        # reduce-overhead and cudagraphs enabled, against eager. It has one test method.
        """
        Test that index/scatter ops work correctly with cudagraphs.
        These ops should either:
        1. Work correctly with cudagraph capture, or
        2. Properly skip cudagraphs (e.g., for boolean indices)
        """

        @ops(
            indexing_op_db,
            allowed_dtypes=(torch.float32,),
        )
        @torch._inductor.config.patch("triton.cudagraphs", True)
        def test_cudagraph_indexing_ops(self, device, dtype, op):
            # [FUNCTION DOC] For each setting of deterministic algorithms (off, on), resets
            # dynamo, takes the first sample input of the op, runs eager (skipping that setting
            # if eager raises) and runs the compiled function three times on fresh clones of the
            # input; the last compiled result must equal the eager one. The previous
            # deterministic algorithms setting is always restored.
            for deterministic in [False, True]:
                torch._dynamo.reset()

                samples = op.sample_inputs(device, dtype, requires_grad=False)
                sample = next(iter(samples))

                def fn(input, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: Calls the OpInfo operator on the input with the
                    # given extra arguments.
                    return op.op(input, *args, **kwargs)

                old_deterministic = torch.are_deterministic_algorithms_enabled()
                try:
                    torch.use_deterministic_algorithms(deterministic)

                    # Run eager first - skip if eager fails (pre-existing issue)
                    try:
                        input_clone = sample.input.clone()
                        expected = fn(input_clone, *sample.args, **sample.kwargs)
                    except Exception as e:
                        continue  # Skip this deterministic setting if eager fails

                    compiled_fn = torch.compile(fn, mode="reduce-overhead")

                    # Run multiple times to trigger cudagraph recording
                    for _ in range(3):
                        input_clone = sample.input.clone()
                        result = compiled_fn(input_clone, *sample.args, **sample.kwargs)

                    self.assertEqual(result, expected)
                finally:
                    torch.use_deterministic_algorithms(old_deterministic)

    if torch.cuda.is_available():
        instantiate_device_type_tests(
            TestCudagraphIndexingOps, globals(), only_for=("cuda",)
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    if not TEST_CUDA_GRAPH:
        if __name__ == "__main__":
            sys.exit(0)
        raise unittest.SkipTest("cuda graph test is skipped")

    if HAS_CUDA_AND_TRITON:
        run_tests(needs="filelock")
