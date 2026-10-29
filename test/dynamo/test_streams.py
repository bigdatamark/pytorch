# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch.compile support of streams and events across Dynamo tracing,
# AOT autograd stream passes and Inductor lowering. Covers: stream context managers (annotated
# with stream indices in the captured graphs), stream and event creation and weakref handling,
# event record/wait/synchronize ops, backward-pass stream handling with sync_dealloc and
# record_stream, control_deps wrapping of synchronization nodes, errors for events recorded
# after input mutation, and end-to-end compiled runs. Most tests need CUDA and compare against
# inline expected graph text. Module level helpers normalize printed graphs. Running the file
# as a script calls run_tests; importing has no side effects beyond imports.
import gc
import re
import unittest
import weakref
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.graph_bytecode_inputs import (
    CURRENT_STREAM_INDEX,
    index_to_external_object_weakref,
    reset_user_object_tracking,
    store_user_object_weakrefs,
)
from torch._dynamo.testing import extract_graph, remove_trailing_space
from torch.testing._internal.common_utils import (
    IS_LINUX,
    IS_MACOS,
    IS_WINDOWS,
    requires_cuda,
    TEST_WITH_ROCM,
    TEST_XPU,
)


def remove_file_comment(gm_str: str) -> str:
    # [FUNCTION DOC] HELPER: Return the input graph text with every fragment from the word
    # File to the end of its line replaced by a bare newline (dropping stack-trace file
    # comments), then with trailing whitespace removed by remove_trailing_space.
    return remove_trailing_space(re.sub(r"File.*\n", "\n", gm_str))


def print_graph(graph: torch.fx.GraphModule) -> str:
    # [FUNCTION DOC] HELPER: Render a GraphModule as readable text without printing it, then
    # strip file comments with remove_file_comment. Returns the string used as the actual
    # value in expected-graph assertions.
    return remove_file_comment(graph.print_readable(print_output=False))


def strip_annotation_desc(gm_str: str) -> str:
    # [FUNCTION DOC] HELPER: Keep only the Annotation comment with its braces in each line and
    # drop whatever follows on that line. The regex does not cross newlines, so only the tail
    # of the same line is removed. Returns the cleaned string.
    """Strip the stack-trace description off "# Annotation: {...}" lines.

    The description ("Backward of forward node:" vs "No stacktrace found for
    following nodes") is execution-order dependent for nodes synthesized by the
    streams pass, so keep only the meaningful stream annotation.
    """
    return re.sub(r"(# Annotation: \{[^}]*\}).*", r"\1", gm_str)


class TestStreams(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo test case for stream and event support under torch.compile. Holds no
    # state; each test builds its own functions, streams and events, extracts forward/backward
    # graphs with extract_graph, and checks either graph text, graph structure (control_deps
    # nodes, dependencies) or numerical agreement with eager. Many tests need CUDA and are
    # skipped otherwise; the helper _assert_weakref_callback_fires and
    # _assert_empty_sync_anchors_record are shared by several tests.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class-level setup that only defers to the
        # parent TestCase setup.
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        # [FUNCTION DOC] HELPER: Class-level teardown that only defers to the parent
        # TestCase teardown.
        super().tearDownClass()

    @requires_cuda
    def test_stream_weakref(self):
        # [FUNCTION DOC] Checks that a torch.Stream object supports weak references: creates
        # one and takes a weakref; passes if no exception is raised. Needs CUDA.
        s = torch.Stream()
        weakref.ref(s)

    @requires_cuda
    def test_event_weakref(self):
        # [FUNCTION DOC] Checks that a torch.Event object supports weak references: creates
        # one and takes a weakref; passes if no exception is raised. Needs CUDA.
        e = torch.Event()
        weakref.ref(e)

    def _assert_weakref_callback_fires(self, factory):
        # [FUNCTION DOC] HELPER: Create an object from the given factory, attach a weakref
        # whose callback records that it fired, delete the object and run garbage collection,
        # then assert the callback fired exactly once. Finally drops the weakref. Used to
        # verify that backend Stream/Event deallocation clears weak references.
        """Backend Stream/Event tp_dealloc overrides must call
        PyObject_ClearWeakRefs so that weakrefs to destroyed instances
        are properly cleared.  Without it, the weakref's wr_object is
        left dangling and any later access (e.g. the dynamo external-
        object registry being cleared at interpreter finalization)
        hits a use-after-free.

        The callback firing is the only reliable Python-level signal —
        `weakref.ref(s)() is None` can return True even with the bug
        present, because other CPython paths may null wr_object
        without invoking callbacks (Objects/weakrefobject.c).
        """
        called = []
        obj = factory()
        # `_weakref_keepalive` must outlive `del obj` so the callback
        # has a chance to fire; the binding is load-bearing.
        _weakref_keepalive = weakref.ref(obj, lambda _ref: called.append(True))
        del obj
        gc.collect()
        self.assertEqual(called, [True])
        del _weakref_keepalive

    @requires_cuda
    def test_cuda_stream_event_weakref_callback(self):
        # [FUNCTION DOC] Runs the weakref callback check for torch.cuda.Stream and
        # torch.cuda.Event. Needs CUDA.
        self._assert_weakref_callback_fires(torch.cuda.Stream)
        self._assert_weakref_callback_fires(torch.cuda.Event)

    @unittest.skipUnless(TEST_XPU, "xpu only")
    def test_xpu_stream_event_weakref_callback(self):
        # [FUNCTION DOC] Runs the weakref callback check for torch.xpu.Stream and
        # torch.xpu.Event. Skipped unless XPU is available.
        self._assert_weakref_callback_fires(torch.xpu.Stream)
        self._assert_weakref_callback_fires(torch.xpu.Event)

    @requires_cuda
    def test_dynamo_registry_no_dangling_weakref(self):
        # [FUNCTION DOC] Regression test for a dangling weakref in the Dynamo external-
        # object registry. Clears the registry, compiles with the eager backend a function
        # that reads the raw handle of the current CUDA stream, runs it once, deletes the
        # compiled function and collects garbage. Then asserts the registry has an entry for
        # the current-stream index and that the weakref stored there resolves to None, i.e.
        # it was cleared rather than left dangling. Needs CUDA. Side effect: resets global
        # user object tracking.
        """Natural repro of the original UAF pattern.

        `torch.compile(fn, backend="eager")` with `fn` referencing
        `torch.cuda.current_stream()` causes dynamo to register a
        weakref to the captured stream in
        `index_to_external_object_weakref` (via
        `store_user_object_weakrefs`, which does NOT pin via
        `keep_alive`).  As soon as `fn` returns, the captured wrapper
        has no strong references and is freed via `THCPStream_dealloc`.

        At that point the patched tp_dealloc must call
        `PyObject_ClearWeakRefs`, otherwise the registry now holds a
        weakref whose `wr_object` is a dangling pointer to freed
        memory.  Later, when `_PyModule_ClearDict` tears down the
        registry at interpreter finalization, dereferencing that
        dangling pointer to clear the weakref hits a use-after-free.
        """
        # Start from a known-clean registry so the assertion below is
        # a statement about what happened in *this* test, not residue
        # from earlier tests in the same process.
        reset_user_object_tracking()

        def fn(x):
            # [FUNCTION DOC] HELPER: Return the raw CUDA stream handle of the current stream;
            # the argument is unused.
            return torch.cuda.current_stream().cuda_stream

        x = torch.zeros(1, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        compiled(x)
        del compiled
        gc.collect()

        # Tripwire: torch.compile of a function referencing
        # current_stream() must still register under
        # CURRENT_STREAM_INDEX.  If this fails, the production code
        # path that originally surfaced the UAF has moved and this
        # regression test no longer covers it.
        self.assertIn(
            CURRENT_STREAM_INDEX,
            index_to_external_object_weakref,
            "torch.compile of a function referencing current_stream() must "
            "register a weakref under CURRENT_STREAM_INDEX",
        )

        # The captured stream wrapper was freed when fn returned.  The
        # patched tp_dealloc must have cleared the registry's weakref;
        # dereferencing it now must return None rather than a dangling
        # pointer into freed memory.
        self.assertIsNone(
            index_to_external_object_weakref[CURRENT_STREAM_INDEX](),
            "dynamo registry holds a weakref to a Stream wrapper that has "
            "been freed; tp_dealloc must call PyObject_ClearWeakRefs to "
            "clear it, otherwise the registry retains a dangling pointer",
        )

    @requires_cuda
    def test_stream_enter_exit(self):
        # [FUNCTION DOC] Traces a function that runs adds under two different stream contexts
        # and uses both results. Asserts one forward graph was captured, that the compiled
        # result equals eager, and that the printed graph equals the expected text where the
        # adds are annotated with stream indices 1 and 2. Needs CUDA.
        def fn(x, y, s1, s2):
            # [FUNCTION DOC] HELPER: Add the inputs inside the first stream context, add them
            # again inside the second, then combine with a constant and the first result;
            # returns the final tensor.
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(), torch.Stream())
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_cuda
    @unittest.skip("Needs graph break support with annotation context")
    def test_stream_context_graph_break(self):
        # [FUNCTION DOC] Placeholder for stream contexts spanning a graph break; always
        # skipped because annotation contexts do not yet support graph breaks. The body would
        # expect two forward graphs but both expected strings are empty. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Creates two local streams, computes under each, calls a
            # Dynamo graph break inside the second stream context and then adds one more;
            # returns the result.
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1
                torch._dynamo.graph_break()
                y = y + 1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(expected, actual)
        self.assertEqual(len(fw_graphs), 2)
        self.assertExpectedInline(print_graph(fw_graphs[0]), """""")
        self.assertExpectedInline(print_graph(fw_graphs[1]), """""")

    @requires_cuda
    def test_stream_input(self):
        # [FUNCTION DOC] Compiles with fullgraph a function that takes a stream as an input
        # and returns it alongside a computed tensor; asserts the compiled outputs equal eager
        # outputs. Needs CUDA.
        def fn(x, y, s):
            # [FUNCTION DOC] HELPER: Add the two tensors, add a constant and return that
            # tensor together with the passed stream.
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2), torch.Stream(device="cuda"))
        expected = fn(*inp)
        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        actual = fn_opt(*inp)
        self.assertEqual(expected, actual)

    @requires_cuda
    def test_local_stream_return(self):
        # [FUNCTION DOC] Compiles with fullgraph a function that creates a stream locally
        # and returns it. Calls it twice and asserts the returned object is a torch.Stream
        # and that the two calls return different streams (a new one is allocated per
        # call). Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create a new stream, compute a tensor from the inputs and
            # return the tensor with the stream.
            s = torch.Stream()
            z = torch.add(x, y)
            y = z + 2
            return y, s

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        # Streams will be different values for each invocation
        # so don't check for equality
        self.assertIsInstance(s0, torch.Stream)
        # Stream should be newly allocated on each call
        self.assertNotEqual(s0, s1)

    def test_event_subclass_python_type(self):
        # [FUNCTION DOC] Verifies that Dynamo keeps the Python subclass type of an Event
        # input: compiles a function that tests whether its argument has the exact
        # subclass type, calls it with an instance, and asserts the result is the ones
        # tensor. Does not need a GPU.
        class MyEvent(torch.Event):
            # [CLASS DOC] HELPER: Empty subclass of torch.Event used to check Python
            # type preservation.
            pass

        def fn(e):
            # [FUNCTION DOC] HELPER: Return a tensor of ones if the argument's exact type is
            # the MyEvent subclass, otherwise a tensor of zeros.
            return torch.ones(2) if type(e) is MyEvent else torch.zeros(2)

        res = torch.compile(fn, backend="eager", fullgraph=True)(MyEvent())
        self.assertEqual(res, torch.ones(2))

    @requires_cuda
    def test_get_current_stream_return(self):
        # [FUNCTION DOC] Compiles a function that enters an input stream and returns the
        # accelerator current stream observed inside. Calls it twice and asserts both returned
        # streams equal the input stream. Needs CUDA.
        def fn(x, s):
            # [FUNCTION DOC] HELPER: Enter the stream given as the second argument, read
            # the accelerator current stream inside the context and return the first
            # argument with that stream.
            with s:
                s0 = torch.accelerator.current_stream()
            return x, s0

        s_inp = torch.Stream(device="cuda")
        inp = (torch.ones(2, 2) + 1, s_inp)
        fn_opt = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        _, s0 = fn_opt(*inp)
        _, s1 = fn_opt(*inp)
        self.assertEqual(s_inp, s0)
        self.assertEqual(s0, s1)

    @requires_cuda
    def test_cuda_current_stream_attrs(self):
        # [FUNCTION DOC] Verifies that attributes of the current CUDA stream (the raw stream
        # handle) can be read in a fullgraph eager-backend compiled function and equal the
        # eager value. Needs CUDA.
        """Verify that torch.cuda.current_stream() attributes are accessible
        under torch.compile and match eager behavior."""

        def fn_cuda_stream(x):
            # [FUNCTION DOC] HELPER: Return the raw handle of the current CUDA stream; the
            # tensor argument is unused.
            return torch.cuda.current_stream().cuda_stream

        x = torch.zeros(1, device="cuda")
        compiled = torch.compile(fn_cuda_stream, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn_cuda_stream(x))

    @unittest.skipIf(not TEST_XPU, "XPU is not available")
    def test_xpu_current_stream_attrs(self):
        # [FUNCTION DOC] XPU counterpart of the current stream attribute check: the compiled
        # function returns the SYCL queue of the current XPU stream and must equal eager.
        # Skipped when XPU is unavailable.
        """Verify that torch.xpu.current_stream() attributes are accessible
        under torch.compile and match eager behavior."""

        def fn_xpu_stream(x):
            # [FUNCTION DOC] HELPER: Return the SYCL queue attribute of the current XPU
            # stream; the tensor argument is unused.
            return torch.xpu.current_stream().sycl_queue

        x = torch.zeros(1, device="xpu")
        compiled = torch.compile(fn_xpu_stream, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x), fn_xpu_stream(x))

    @requires_cuda
    def test_cuda_current_stream_with_entered_stream(self):
        # [FUNCTION DOC] Verifies that inside a stream context entered on a user-created
        # CUDA stream, the current stream handle read in compiled code equals the eager
        # value. Needs CUDA.
        """Verify that torch.cuda.current_stream().cuda_stream returns the
        correct value when inside a stream context for a user-created stream."""

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Enter the given stream and return the raw handle of the
            # current CUDA stream observed inside.
            with s:
                return torch.cuda.current_stream().cuda_stream

        s = torch.cuda.Stream()
        x = torch.zeros(1, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, s), fn(x, s))

    @unittest.skipIf(not TEST_XPU, "XPU is not available")
    def test_xpu_current_stream_with_entered_stream(self):
        # [FUNCTION DOC] XPU counterpart: inside a context on a user-created XPU stream, the
        # compiled function returns the SYCL queue of the current stream and it must equal the
        # eager value. Skipped when XPU is unavailable.
        """Verify that torch.xpu.current_stream().sycl_queue returns the
        correct value when inside a stream context for a user-created stream."""

        def fn(x, s):
            # [FUNCTION DOC] HELPER: Enter the given stream and return the SYCL queue of the
            # current XPU stream observed inside.
            with s:
                return torch.xpu.current_stream().sycl_queue

        s = torch.xpu.Stream()
        x = torch.zeros(1, device="xpu")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, s), fn(x, s))

    @requires_cuda
    def test_nested_stream_enter_exit(self):
        # [FUNCTION DOC] Traces a function with nested stream contexts over three input
        # streams, extracts the forward graph, and asserts one graph, equality with eager, and
        # that the printed graph has the expected text with adds annotated by stream indices 2
        # and 3 and a later add on stream 2. Needs CUDA.
        def fn(x, y, s0, s1, s2):
            # [FUNCTION DOC] HELPER: Compute an add inside nested contexts of two streams,
            # compute another add in a third stream and then combine inside a nested re-entry
            # of the second stream; returns both results.
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (
            torch.ones(2, 2) + 1,
            torch.ones(2, 2),
            torch.Stream(),
            torch.Stream(),
            torch.Stream(),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 2}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @unittest.skip("Needs graph break support with annotation context")
    def test_stream_enter_exit_graph_break(self):
        # [FUNCTION DOC] Placeholder test for stream enter/exit with graph breaks; always
        # skipped, body does nothing.
        pass

    @unittest.skip("Needs graph break support with annotation context")
    def test_nested_stream_enter_exit_graph_break(self):
        # [FUNCTION DOC] Placeholder test for nested stream enter/exit with graph breaks;
        # always skipped, body does nothing.
        pass

    @requires_cuda
    def test_local_stream_enter_exit(self):
        # [FUNCTION DOC] Like the basic enter/exit test but the two streams are created inside
        # the traced function. Asserts a single forward graph, equality with eager and the
        # expected printed graph with stream annotations. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create two local streams, add the inputs under the first,
            # add them under the second and combine; returns the result.
            s2 = torch.Stream()
            s1 = torch.Stream()
            with s1:
                z1 = torch.add(x, y)
            with s2:
                z = torch.add(x, y)
                y = z + 2 + z1

            return y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 1}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2);  add_1 = None
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_2, add);  add_2 = add = None
        return (add_3,)
""",
        )

    @requires_cuda
    def test_local_stream_nested_enter_exit(self):
        # [FUNCTION DOC] Like the nested enter/exit test but with streams created inside the
        # traced function. Asserts a single forward graph, equality with eager and the
        # expected printed graph with stream annotations. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create three local streams, compute under nested
            # contexts of two of them, compute under the third and re-enter the second;
            # returns both results.
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    z1 = torch.add(x, y)
            with s0:
                z0 = torch.add(x, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add, 2);  add = None
        return (add_1, add_2)
""",
        )

    @requires_cuda
    def test_new_stream_api(self) -> None:
        # [FUNCTION DOC] Exercises the new_stream API from Dynamo's stream variables inside a
        # custom backend. The backend creates two stream indices, asserts they differ and that
        # the external objects they refer to differ, then inserts a node that fetches an
        # external object by index into the graph. Compiles a trivial function with this
        # backend and runs it on a CUDA tensor. Needs CUDA.
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_stream

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Custom compile backend: allocates two new streams,
            # asserts distinct indices and distinct objects, inserts a fetch-by-index graph
            # node after the first node and returns the modified graph module.
            s0_ind = new_stream()
            s1_ind = new_stream()
            self.assertNotEqual(s0_ind, s1_ind)
            self.assertNotEqual(
                get_external_object_by_index(s0_ind),
                get_external_object_by_index(s1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

    @requires_cuda
    def test_current_stream_api(self) -> None:
        # [FUNCTION DOC] Exercises the get_current_stream API inside a custom backend. The
        # backend asks for the current stream index on the first CUDA device, asserts the
        # referenced object equals the stream captured outside, and inserts two graph nodes:
        # one fetching the object and one running an equality assertion at graph runtime.
        # Compiles a trivial function and runs it on CUDA. Needs CUDA.
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import get_current_stream

        cur_stream = torch.accelerator.current_stream()
        s0 = None

        def stream_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Custom compile backend that checks the current stream
            # index resolves to the captured current stream, inserts the two verification
            # nodes after the first node and returns the graph module.
            # KNOWN ISSUE: the nonlocal declaration of s0 is never used for assignment.
            nonlocal s0
            s0_ind = get_current_stream(torch.device("cuda:0"))
            self.assertEqual(get_external_object_by_index(s0_ind), cur_stream)
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(s0_ind,), kwargs={}
                )
                gm.graph.call_function(
                    lambda x: self.assertEqual(
                        cur_stream, get_external_object_by_index(x)
                    ),
                    args=(s0_ind,),
                    kwargs={},
                )
            return gm

        @torch.compile(backend=stream_generation_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Return the input plus one.
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

    @requires_cuda
    def test_stream_with_mutation(self):
        # [FUNCTION DOC] Traces a function that mutates an input in place inside nested stream
        # contexts and computes other values on separate streams. Asserts a single forward
        # graph, equality with eager, and the expected printed graph including the epilogue
        # copy back into the input on the first stream. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create three local streams; mutate the first input with
            # an in-place add in nested contexts, compute two derived tensors on another
            # stream and re-enter a stream for a final add; returns two tensors.
            s2 = torch.Stream()
            s1 = torch.Stream()
            s0 = torch.Stream()
            with s1:
                with s2:
                    x.add_(y)
            with s0:
                z1 = torch.add(y, y)
                z0 = torch.add(z1, y)
                with s2:
                    y = 2 + z1

            return z0, y

        inp = (torch.ones(2, 2) + 1, torch.ones(2, 2))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]", arg1_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1)

        # Annotation: {'stream': 3}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg1_1, arg1_1)

        # Annotation: {'stream': 1}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, 2)

        # Annotation: {'stream': 3}
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(add_1, arg1_1);  add_1 = arg1_1 = None

        # Annotation: {'stream': 1}
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = copy_ = None
        return (add_3, add_2)
""",
        )

    @requires_cuda
    def test_stream_backward_simple(self) -> None:
        # [FUNCTION DOC] Traces a function computing the same expression on two streams
        # with inputs requiring grad. Checks the forward graph text, then runs backward on
        # one output and checks the backward graph text (using the raw printer), which
        # contains stream annotations, a control_deps record_event subgraph and a
        # sync_dealloc op. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create two streams and compute the same affine expression
            # under each; returns both results.
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, requires_grad=True) + 1,
            torch.ones(2, 2, requires_grad=True),
        )
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = None

        # Annotation: {'stream': 1}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  primals_2 = None
        return (add, add_1, mul_1, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul_1: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        # Backward of forward node:
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # Annotation: {'stream': 1} Backward of forward node:
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_2, mul_3)

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul_1, add_1, mul_2, add_3, add_2), subgraph_record_event_default, add_3, add_2);  mul_1 = add_1 = mul_2 = add_3 = add_2 = subgraph_record_event_default = None

        # Backward of forward node:
        getitem_1: "f32[2, 2]" = control_deps[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem: "f32[2, 2]" = control_deps[1];  control_deps = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(3, 2, mul_3);  mul_3 = sync_dealloc_default = None
        return (getitem, getitem_1)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 1)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_cuda
    def test_stream_backward_sync(self) -> None:
        # [FUNCTION DOC] Same function as the simple backward test but on CUDA tensors with
        # explicit device synchronization around graph extraction. Checks the forward graph
        # text, runs backward on one output and checks the backward graph text with several
        # control_deps subgraphs for record, wait and record events. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create two streams and compute the same affine expression
            # under each; returns both results.
            s2 = torch.Stream()
            s0 = torch.Stream()
            with s0:
                y0 = 2 * x + y
            with s2:
                z = 2 * x + y

            return y0, z

        inp = (
            torch.ones(2, 2, device="cuda:0", requires_grad=True) + 1,
            torch.ones(2, 2, device="cuda:0", requires_grad=True),
        )
        torch.cuda.synchronize()
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)
        torch.cuda.synchronize()
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2)

        # Annotation: {'stream': 1}
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2);  primals_1 = None
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul_1, primals_2);  primals_2 = None
        return (add, add_1, mul, add, mul_1, add_1)
""",
        )

        actual[1].sum().backward()
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", add: "f32[2, 2]", mul_1: "f32[2, 2]", add_1: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 1} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, 2)

        # Backward of forward node:
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, tangents_1);  tangents_2 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps = torch.ops.higher_order.control_deps((mul, add, mul_3, add_2), subgraph_record_event_default, mul_3, add_2);  mul = add = mul_3 = add_2 = subgraph_record_event_default = None

        # Backward of forward node:
        getitem_1: "f32[2, 2]" = control_deps[2]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem: "f32[2, 2]" = control_deps[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_1 = torch.ops.higher_order.control_deps((control_deps, mul_1, add_1, mul_2, getitem, getitem_1), subgraph_wait_event_default, mul_2, getitem, getitem_1);  control_deps = mul_1 = add_1 = mul_2 = getitem = getitem_1 = subgraph_wait_event_default = None

        # Backward of forward node:
        getitem_4: "f32[2, 2]" = control_deps_1[3]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_3: "f32[2, 2]" = control_deps_1[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem_2: "f32[2, 2]" = control_deps_1[1]

        # Annotation: {'stream': 1} Backward of forward node:
        add_3: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_2, getitem_3)

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_2 = torch.ops.higher_order.control_deps((add_3, control_deps_1, getitem_2, getitem_3, getitem_4), subgraph_record_event_default_1, add_3, getitem_3, getitem_4);  add_3 = control_deps_1 = getitem_2 = getitem_3 = getitem_4 = subgraph_record_event_default_1 = None

        # Backward of forward node:
        getitem_7: "f32[2, 2]" = control_deps_2[3]

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_6: "f32[2, 2]" = control_deps_2[2]

        # Annotation: {'stream': 1} Backward of forward node:
        getitem_5: "f32[2, 2]" = control_deps_2[1];  control_deps_2 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(4, 2, getitem_6);  getitem_6 = sync_dealloc_default = None
        return (getitem_5, getitem_7)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(3, 2)
            return (record_event_default, dep_0, dep_1)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(3, 1)
            return (wait_event_default, dep_0, dep_1, dep_2)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]", dep_2: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 1)
            return (record_event_default, dep_0, dep_1, dep_2)
""",
        )

    @requires_cuda
    def test_event_tracing(self):
        # [FUNCTION DOC] Traces a function that creates an event, records it, mutates the
        # input in place, and asserts the printed forward graph contains a record_event op
        # before the add and copy. Needs CUDA.
        def fn(x) -> None:
            # [FUNCTION DOC] HELPER: Create an event, record it, add one to the input in place
            # and return the input.
            e = torch.Event()
            e.record()
            x.add_(1)
            return x

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        #
        record_event = torch.ops.streams.record_event.default(1, 0);  record_event = None

        #
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
""",
        )

    @requires_cuda
    def test_recorded_cuda_events_append_runtime_objects(self):
        # [FUNCTION DOC] Checks that compiling with the eager backend a function that records
        # two timing CUDA events and appends them to an outer list leaves runtime Event
        # objects in the list. Compares result with eager, asserts two torch.Event entries and
        # a non-negative elapsed time between them. Needs CUDA.
        events = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Create two timing events, record one before and one after
            # a matmul, append both to the enclosing list and return the matmul result.
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            out = torch.matmul(x, x)
            end_event.record()

            events.append(start_event)
            events.append(end_event)
            return out

        x = torch.randn(16, 16, device="cuda")
        expected = fn(x)
        torch.cuda.synchronize()
        events[0].elapsed_time(events[1])
        events.clear()

        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)
        torch.cuda.synchronize()

        self.assertEqual(expected, actual)
        self.assertEqual(len(events), 2)
        self.assertIsInstance(events[0], torch.Event)
        self.assertIsInstance(events[1], torch.Event)
        self.assertGreaterEqual(events[0].elapsed_time(events[1]), 0.0)

    @requires_cuda
    def test_run_opcheck_fork_join(self):
        # [FUNCTION DOC] Runs the library opcheck on the fork_stream and join_stream ops for
        # two sample index pairs after registering two streams in the Dynamo external-object
        # tracking. In a finally clause restores the original current stream and resets the
        # object tracking. Needs CUDA.
        from torch._dynamo.variables.streams import fork_stream, join_stream
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            store_user_object_weakrefs(s0, s1)

            sample_inputs = [
                (0, 1),
                (1, 0),
            ]
            for args in sample_inputs:
                opcheck(fork_stream, args)
                opcheck(join_stream, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    @requires_cuda
    def test_run_opcheck_wait_record(self):
        # [FUNCTION DOC] Runs opcheck on the wait_event and record_event ops for two sample
        # argument pairs after registering two streams and two events in object tracking. A
        # finally clause restores the original stream and resets tracking. Needs CUDA.
        from torch._dynamo.variables.streams import record_event, wait_event
        from torch.library import opcheck

        original_stream = torch.accelerator.current_stream()
        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            e0 = torch.Event()
            e1 = torch.Event()
            store_user_object_weakrefs(s0, s1, e0, e1)

            sample_inputs = [
                (2, 0),
                (3, 1),
            ]
            for args in sample_inputs:
                opcheck(wait_event, args)
                opcheck(record_event, args)
        finally:
            torch.accelerator.set_stream(original_stream)
            reset_user_object_tracking()

    @requires_cuda
    def test_run_opcheck_wait_record_stream(self):
        # [FUNCTION DOC] Runs opcheck on the wait_stream op for two sample argument pairs
        # after registering three streams in object tracking. A finally clause resets
        # tracking. Needs CUDA.
        from torch._dynamo.variables.streams import wait_stream
        from torch.library import opcheck

        try:
            s0 = torch.Stream()
            s1 = torch.Stream()
            s2 = torch.Stream()
            store_user_object_weakrefs(s0, s1, s2)

            sample_inputs = [
                (0, 1),
                (2, 0),
            ]
            for args in sample_inputs:
                opcheck(wait_stream, args)
        finally:
            reset_user_object_tracking()

    @requires_cuda
    def test_record_stream_problem_basic(self):
        # [FUNCTION DOC] Checks the record_stream problem when a tensor allocated on one
        # stream is used on a second stream after an event wait. Extracts graphs, runs
        # backward on one output and asserts the printed backward graph (with annotation
        # descriptions stripped) matches the expected text, which includes a
        # sync_dealloc op. Needs CUDA.
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # We expect there to be a sync_dealloc op added to the graph for y
        # synchronizing the first stream w/ the second stream after the second stream is finished
        def fn(x):
            # [FUNCTION DOC] HELPER: Create an event, allocate a tensor on a first stream,
            # record the event and multiply; on a second stream wait for the event and compute
            # another product; returns both products.
            e = torch.Event()
            with torch.Stream(device="cuda:0"):
                y = torch.ones(2, 2, device="cuda:0")
                e.record()
                z = y * x

            with torch.Stream(device="cuda:0"):
                e.wait()
                z0 = y * 2 * x

            return z0, z

        inp = (torch.ones(2, 2, device="cuda", requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, mul: "f32[2, 2]", getitem_1: "f32[2, 2]", mul_1: "f32[2, 2]", mul_2: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]"):
        # Annotation: {'stream': 3} Backward of forward node:
        mul_3: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, mul_1);  tangents_1 = None

        # Annotation: {'stream': 2} Backward of forward node:
        mul_4: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_2, getitem_1);  tangents_2 = getitem_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_2 = torch.ops.higher_order.control_deps((mul, mul_4), subgraph_record_event_default, mul_4);  mul = mul_4 = subgraph_record_event_default = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_2: "f32[2, 2]" = control_deps_2[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_3 = torch.ops.higher_order.control_deps((control_deps_2, mul_1, mul_2, mul_3, getitem_2), subgraph_wait_event_default, mul_3, getitem_2);  control_deps_2 = mul_1 = mul_2 = mul_3 = getitem_2 = subgraph_wait_event_default = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_4: "f32[2, 2]" = control_deps_3[2]

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_3: "f32[2, 2]" = control_deps_3[1]

        # Annotation: {'stream': 3} Backward of forward node:
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_3, getitem_4)

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_4 = torch.ops.higher_order.control_deps((add, control_deps_3, getitem_3, getitem_4), subgraph_record_event_default_1, add, getitem_4);  add = control_deps_3 = getitem_3 = getitem_4 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 2} Backward of forward node:
        getitem_6: "f32[2, 2]" = control_deps_4[2]

        # Annotation: {'stream': 3} Backward of forward node:
        getitem_5: "f32[2, 2]" = control_deps_4[1];  control_deps_4 = None

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(5, 2, getitem_6);  getitem_6 = sync_dealloc_default = None
        return (getitem_5,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(4, 2)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(4, 3)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(5, 3)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_cuda
    def test_record_stream_problem_interleaved(self):
        # [FUNCTION DOC] Checks sync_dealloc and event handling when one tensor is allocated
        # on the first stream and then used on three further streams chained with event record
        # and wait. Extracts graphs, runs backward on one output and compares the annotation-
        # stripped backward graph to the expected text with many record/wait control_deps
        # subgraphs. Needs CUDA.
        # see https://docs.pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html#torch.Tensor.record_stream
        # for what this tests/solves for
        # This will have interleaved computation where y is
        # first allocated on the first stream used on the second stream
        # used on the first stream again then finally used on the last stream
        def fn(x):
            # [FUNCTION DOC] HELPER: Allocate a tensor on a first stream and chain
            # computations over four streams, each waiting on and then re-recording one
            # shared event, and finish with a wait on the default stream; returns three
            # intermediate results.
            e = torch.Event()
            with torch.Stream(device="cuda:0"):
                y = torch.ones(2, 2, device="cuda:0")
                z = y * x
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z0 = y * 2 * z
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z1 = y * x * z0
                e.record()

            with torch.Stream(device="cuda:0"):
                e.wait()
                z2 = y * 4 * z1
                e.record()

            e.wait()
            return z, z1, z2

        inp = (torch.ones(2, 2, device="cuda", requires_grad=True),)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual[1].sum().backward()

        self.assertExpectedInline(
            strip_annotation_desc(print_graph(bw_graphs[0])),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, getitem_29: "f32[2, 2]", getitem_28: "f32[2, 2]", getitem_27: "f32[2, 2]", getitem_26: "f32[2, 2]", getitem_25: "f32[2, 2]", tangents_1: "f32[2, 2]", tangents_2: "f32[2, 2]", tangents_3: "f32[2, 2]"):
        # Annotation: {'stream': 5}
        mul_7: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_3, getitem_25);  tangents_3 = getitem_25 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default = self.subgraph_record_event_default
        control_deps_8 = torch.ops.higher_order.control_deps((mul_7,), subgraph_record_event_default, mul_7);  mul_7 = subgraph_record_event_default = None

        # Annotation: {'stream': 5}
        getitem_30: "f32[2, 2]" = control_deps_8[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default = self.subgraph_wait_event_default
        control_deps_9 = torch.ops.higher_order.control_deps((control_deps_8, getitem_30), subgraph_wait_event_default, getitem_30);  control_deps_8 = getitem_30 = subgraph_wait_event_default = None

        # Annotation: {'stream': 5}
        getitem_31: "f32[2, 2]" = control_deps_9[1]

        # Annotation: {'stream': 4}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_2, getitem_31);  tangents_2 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_4 = self.subgraph_record_event_default_4
        control_deps_10 = torch.ops.higher_order.control_deps((add, control_deps_9, getitem_31), subgraph_record_event_default_4, add, getitem_31);  add = control_deps_9 = getitem_31 = subgraph_record_event_default_4 = None

        # Annotation: {'stream': 5}
        getitem_33: "f32[2, 2]" = control_deps_10[2]

        # Annotation: {'stream': 4}
        getitem_32: "f32[2, 2]" = control_deps_10[1]

        # No stacktrace found for following nodes
        sync_dealloc_default = torch.ops.streams.sync_dealloc.default(10, 5, getitem_33);  sync_dealloc_default = None

        # Annotation: {'stream': 4}
        mul_8: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_32, getitem_26);  getitem_26 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_1 = self.subgraph_record_event_default_1
        control_deps_11 = torch.ops.higher_order.control_deps((mul_8, control_deps_10, getitem_32, getitem_33), subgraph_record_event_default_1, mul_8, getitem_32);  mul_8 = control_deps_10 = getitem_32 = getitem_33 = subgraph_record_event_default_1 = None

        # Annotation: {'stream': 4}
        getitem_35: "f32[2, 2]" = control_deps_11[2]

        # Annotation: {'stream': 4}
        getitem_34: "f32[2, 2]" = control_deps_11[1]
        mul_9: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_35, getitem_28);  getitem_28 = None
        mul_10: "f32[2, 2]" = torch.ops.aten.mul.Tensor(mul_9, getitem_29)

        # No stacktrace found for following nodes
        subgraph_wait_event_default_1 = self.subgraph_wait_event_default_1
        control_deps_12 = torch.ops.higher_order.control_deps((control_deps_11, getitem_34, getitem_35), subgraph_wait_event_default_1, getitem_34);  subgraph_wait_event_default_1 = None

        # Annotation: {'stream': 4}
        getitem_36: "f32[2, 2]" = control_deps_12[1]

        # Annotation: {'stream': 3}
        mul_11: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_36, getitem_27);  getitem_27 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_5 = self.subgraph_record_event_default_5
        control_deps_13 = torch.ops.higher_order.control_deps((mul_11, control_deps_12, getitem_36), subgraph_record_event_default_5, mul_11, getitem_36);  mul_11 = control_deps_12 = getitem_36 = subgraph_record_event_default_5 = None

        # Annotation: {'stream': 4}
        getitem_38: "f32[2, 2]" = control_deps_13[2]

        # Annotation: {'stream': 3}
        getitem_37: "f32[2, 2]" = control_deps_13[1]

        # No stacktrace found for following nodes
        sync_dealloc_default_1 = torch.ops.streams.sync_dealloc.default(11, 4, getitem_38);  sync_dealloc_default_1 = None
        subgraph_record_event_default_2 = self.subgraph_record_event_default_2
        control_deps_14 = torch.ops.higher_order.control_deps((control_deps_13, getitem_37, getitem_38), subgraph_record_event_default_2, getitem_37);  control_deps_13 = getitem_37 = getitem_38 = subgraph_record_event_default_2 = None

        # Annotation: {'stream': 3}
        getitem_39: "f32[2, 2]" = control_deps_14[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default_2 = self.subgraph_wait_event_default_2
        control_deps_15 = torch.ops.higher_order.control_deps((control_deps_14, getitem_39), subgraph_wait_event_default_2, getitem_39);  control_deps_14 = getitem_39 = subgraph_wait_event_default_2 = None

        # Annotation: {'stream': 3}
        getitem_40: "f32[2, 2]" = control_deps_15[1]

        # Annotation: {'stream': 2}
        add_1: "f32[2, 2]" = torch.ops.aten.add.Tensor(tangents_1, getitem_40);  tangents_1 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_6 = self.subgraph_record_event_default_6
        control_deps_16 = torch.ops.higher_order.control_deps((add_1, control_deps_15, getitem_40), subgraph_record_event_default_6, add_1, getitem_40);  add_1 = control_deps_15 = getitem_40 = subgraph_record_event_default_6 = None

        # Annotation: {'stream': 3}
        getitem_42: "f32[2, 2]" = control_deps_16[2]

        # Annotation: {'stream': 2}
        getitem_41: "f32[2, 2]" = control_deps_16[1]

        # No stacktrace found for following nodes
        sync_dealloc_default_2 = torch.ops.streams.sync_dealloc.default(12, 3, getitem_42);  sync_dealloc_default_2 = None

        # Annotation: {'stream': 2}
        mul_12: "f32[2, 2]" = torch.ops.aten.mul.Tensor(getitem_41, getitem_29);  getitem_29 = None

        # No stacktrace found for following nodes
        subgraph_record_event_default_3 = self.subgraph_record_event_default_3
        control_deps_17 = torch.ops.higher_order.control_deps((mul_12, control_deps_16, getitem_41, getitem_42), subgraph_record_event_default_3, mul_12);  mul_12 = control_deps_16 = getitem_41 = getitem_42 = subgraph_record_event_default_3 = None

        # Annotation: {'stream': 2}
        getitem_43: "f32[2, 2]" = control_deps_17[1]

        # No stacktrace found for following nodes
        subgraph_wait_event_default_3 = self.subgraph_wait_event_default_3
        control_deps_18 = torch.ops.higher_order.control_deps((control_deps_17, mul_9, mul_10, control_deps_11, getitem_34, getitem_35, getitem_43), subgraph_wait_event_default_3, mul_10, getitem_43);  control_deps_17 = mul_9 = mul_10 = control_deps_11 = getitem_34 = getitem_35 = getitem_43 = subgraph_wait_event_default_3 = None

        # Annotation: {'stream': 2}
        getitem_45: "f32[2, 2]" = control_deps_18[2]

        # Annotation: {'stream': 4}
        getitem_44: "f32[2, 2]" = control_deps_18[1]

        # Annotation: {'stream': 4}
        add_2: "f32[2, 2]" = torch.ops.aten.add.Tensor(getitem_44, getitem_45)

        # No stacktrace found for following nodes
        subgraph_record_event_default_7 = self.subgraph_record_event_default_7
        control_deps_19 = torch.ops.higher_order.control_deps((add_2, control_deps_18, getitem_44, getitem_45), subgraph_record_event_default_7, add_2, getitem_45);  add_2 = control_deps_18 = getitem_44 = getitem_45 = subgraph_record_event_default_7 = None

        # Annotation: {'stream': 2}
        getitem_47: "f32[2, 2]" = control_deps_19[2]

        # Annotation: {'stream': 4}
        getitem_46: "f32[2, 2]" = control_deps_19[1];  control_deps_19 = None

        # No stacktrace found for following nodes
        sync_dealloc_default_3 = torch.ops.streams.sync_dealloc.default(13, 2, getitem_47);  getitem_47 = sync_dealloc_default_3 = None
        return (getitem_46,)

    class subgraph_record_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(6, 5)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(6, 4)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_4(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(10, 4)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(7, 4)
            return (record_event_default, dep_0, dep_1)

    class subgraph_wait_event_default_1(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(7, 3)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_5(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(11, 3)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_2(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(8, 3)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_2(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(8, 2)
            return (wait_event_default, dep_0)

    class subgraph_record_event_default_6(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(12, 2)
            return (record_event_default, dep_0, dep_1)

    class subgraph_record_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(9, 2)
            return (record_event_default, dep_0)

    class subgraph_wait_event_default_3(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            wait_event_default = torch.ops.streams.wait_event.default(9, 4)
            return (wait_event_default, dep_0, dep_1)

    class subgraph_record_event_default_7(torch.nn.Module):
        def forward(self, dep_0: "f32[2, 2]", dep_1: "f32[2, 2]"):
            # No stacktrace found for following nodes
            record_event_default = torch.ops.streams.record_event.default(13, 4)
            return (record_event_default, dep_0, dep_1)
""",
        )

    @requires_cuda
    def test_epilogue_copy_streams_inference(self):
        # [FUNCTION DOC] Checks an in-place mutation of an input under a no_grad region on a
        # user stream when the input requires grad. Runs backward on the output and asserts
        # the printed forward graph has the add annotated with the stream followed by the
        # epilogue copy into the input. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Inside a stream context and no_grad, add two to the input
            # in place and return the input.
            with torch.Stream(device="cuda:0"):
                with torch.no_grad():
                    x.add_(2)

            return x

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")

        inp = (x,)
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        actual.sum().backward()
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        # Annotation: {'stream': 1}
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 2)
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(arg0_1, add);  arg0_1 = add = None
        return (copy_,)
""",
        )

    @requires_cuda
    def test_epilogue_copy_streams_external(self):
        # [FUNCTION DOC] Verifies that mutating an input inside a user-specified stream
        # context is rejected: extract_graph is expected to raise a RuntimeError saying
        # mutations on inputs with user-specified streams are not yet supported. Needs CUDA.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled with the eager backend; multiplies the input in
            # place by three inside a stream context and returns its sine.
            with torch.Stream(device="cuda:0"):
                x.mul_(3)
            return x.sin()

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        inp = (x.clone(),)
        with self.assertRaisesRegex(
            RuntimeError,
            "Mutations on inputs with user-specified streams are not yet supported",
        ):
            extract_graph(fn, *inp)

    @requires_cuda
    def test_control_deps_wrapping_record_event(self) -> None:
        # [FUNCTION DOC] Builds a two-stream graph with a record and a wait, then runs
        # wrap_all_sync_nodes_with_control_deps manually. Asserts two control_deps nodes, that
        # pass-through getitem nodes exist for the record node, and that the wait control_deps
        # depends on the record control_deps node. Needs CUDA.
        """Test wrapping record_event with control_deps on a two-stream graph.

        The producer stream has work before the record_event, so control_deps
        should capture those same-stream nodes as dependencies.
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: On one stream compute two values and record an event; on
            # another stream wait for the event and add three; returns the result.
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                z = y * 2
                e.record()

            with s2:
                e.wait()
                w = z + 3

            return w

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        import operator

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        # record_event has same-stream deps, and wait_event gets a
        # cross-event dependency on the record's control_deps node.
        control_deps_nodes = list(
            graph.find_nodes(op="call_function", target=control_deps)
        )
        self.assertEqual(len(control_deps_nodes), 2)
        record_ctrl_node = control_deps_nodes[0]

        # Verify getitem nodes were created for pass-through dependencies
        getitem_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is record_ctrl_node
        ]
        self.assertGreaterEqual(len(getitem_nodes), 1)

        # Verify the wait's control_deps depends on the record's control_deps
        wait_ctrl_node = control_deps_nodes[1]
        self.assertIn(record_ctrl_node, wait_ctrl_node.args[0])

    @requires_cuda
    def test_control_deps_wrapping_wait_event(self) -> None:
        # [FUNCTION DOC] Builds a graph where the consumer stream has work before its event
        # wait, applies the control_deps wrapping pass manually, and asserts two control_deps
        # nodes (record and wait). Needs CUDA.
        """Test wrapping wait_event with control_deps on a two-stream graph.

        The consumer stream has the wait_event, and there should be same-stream
        nodes before it that become dependencies.
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Record an event after an add on one stream; on another
            # stream compute a product, wait on the event and multiply the two results;
            # returns that value.
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                e.record()

            with s2:
                a = x * 3
                e.wait()
                z = y * a

            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        # Both record_event (deps: [y]) and wait_event (deps: [a]) have
        # same-stream deps, so both get wrapped.
        control_deps_nodes = list(
            graph.find_nodes(op="call_function", target=control_deps)
        )
        self.assertEqual(len(control_deps_nodes), 2)

    @requires_cuda
    def test_control_deps_prevents_invalid_reordering(self) -> None:
        # [FUNCTION DOC] Wraps sync nodes with control_deps, verifies the add comes after
        # control_deps in node order and that lint passes, then moves the add node before the
        # control_deps node and asserts the order is violated and that graph lint raises a
        # RuntimeError. Needs CUDA.
        """
        Test that control_deps creates proper data dependencies that prevent invalid reordering.

        This test:
        1. Creates a two-stream graph with control_deps wrapping
        2. Manually moves a node to violate the ordering
        3. Verifies graph.lint() catches the invalid ordering
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Compute two values on one stream and record an event; on
            # another stream wait on it and add three to the earlier result; returns that sum.
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                z = y * 2
                e.record()

            with s2:
                e.wait()
                w = z + 3

            return w

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        control_deps_node = next(
            iter(graph.find_nodes(op="call_function", target=control_deps))
        )

        # Find add_1 (the node after the sync that uses getitem output)
        add_1_node = next(n for n in graph.nodes if n.name == "add_1")

        # Verify valid ordering: control_deps before add_1
        original_order = [n.name for n in graph.nodes if n.op == "call_function"]
        self.assertLess(
            original_order.index("control_deps"),
            original_order.index("add_1"),
            "control_deps should be before add_1 in valid ordering",
        )

        graph.lint()  # Should not raise

        # Manually move add_1 BEFORE control_deps (violates dependency)
        control_deps_node.prepend(add_1_node)

        # Verify the order is now wrong (add_1 comes before control_deps)
        violated_order = [n.name for n in graph.nodes if n.op == "call_function"]
        self.assertLess(
            violated_order.index("add_1"),
            violated_order.index("control_deps"),
            "add_1 should be before control_deps after manual move (violating order)",
        )

        # graph.lint() should catch the invalid ordering
        with self.assertRaises(RuntimeError):
            graph.lint()

    @requires_cuda
    def test_cross_event_deps_multiple_events(self) -> None:
        # [FUNCTION DOC] Stress test with two events across three streams. Captures the raw
        # graph with the wrapping pass patched out, checks there are two record and two wait
        # events whose event indices correspond, then wraps manually and asserts four
        # control_deps nodes where each wait depends only on the record of its own event.
        # Finishes with graph lint. Needs CUDA.
        """Stress test: multiple events across three streams.

        Verifies that each wait_event's control_deps node depends on exactly
        the matching record_event's control_deps node (by event index), not
        on some other event's node.

        Graph layout:
          s1: x+1 -> record(e1) -> x+5 -> record(e2)
          s2: wait(e1) -> x*3
          s3: wait(e2) -> x*7
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: On one stream compute and record two events at different
            # points; the second and third streams wait on one event each and multiply;
            # returns the sum of all four values.
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            s3 = torch.Stream(device="cuda")
            e1 = torch.Event()
            e2 = torch.Event()

            with s1:
                y = x + 1
                e1.record()
                z = x + 5
                e2.record()

            with s2:
                e1.wait()
                a = x * 3

            with s3:
                e2.wait()
                b = x * 7

            return a + b + y + z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        # Capture event indices before wrapping so we can verify correspondence.
        # Each sync node has args (event_index, stream_index).
        record_event_indices = []
        wait_event_indices = []
        for node in graph.nodes:
            if node.op != "call_function":
                continue
            if node.target is torch.ops.streams.record_event.default:
                record_event_indices.append(node.args[0])
            elif node.target is torch.ops.streams.wait_event.default:
                wait_event_indices.append(node.args[0])

        self.assertEqual(len(record_event_indices), 2)
        self.assertEqual(len(wait_event_indices), 2)
        # Each wait corresponds to exactly one record by event index
        self.assertEqual(set(record_event_indices), set(wait_event_indices))

        wrap_all_sync_nodes_with_control_deps(gm)

        # 2 records + 2 waits = 4 control_deps nodes. Each record has
        # same-stream deps, and each wait gets a cross-event dep.
        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        self.assertEqual(len(ctrl_nodes), 4)

        # Graph order follows program order: both records (s1 block) before
        # both waits (s2, s3 blocks).
        record_e1_ctrl = ctrl_nodes[0]
        record_e2_ctrl = ctrl_nodes[1]
        wait_e1_ctrl = ctrl_nodes[2]
        wait_e2_ctrl = ctrl_nodes[3]

        # wait(e1) depends on record(e1), not record(e2)
        self.assertIn(record_e1_ctrl, wait_e1_ctrl.args[0])
        self.assertNotIn(record_e2_ctrl, wait_e1_ctrl.args[0])

        # wait(e2) depends on record(e2), not record(e1)
        self.assertIn(record_e2_ctrl, wait_e2_ctrl.args[0])
        self.assertNotIn(record_e1_ctrl, wait_e2_ctrl.args[0])

        graph.lint()

    @requires_cuda
    def test_cross_event_deps_event_reuse(self) -> None:
        # [FUNCTION DOC] Checks event reuse: an event recorded twice on one stream and then
        # waited on from another stream. After wrapping, asserts three control_deps nodes
        # and that the wait depends on the second record and not on the first. Ends with
        # graph lint. Needs CUDA.
        """Test that reusing an event updates the cross-event dependency.

        When the same event is recorded twice with work in between, the wait
        should depend on the LAST record's control_deps node (matching CUDA
        semantics where record() overwrites the event).
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Record the same event twice around a
            # multiplication on one stream, wait on it from another stream and add
            # three; returns the combined value.
            s1 = torch.Stream(device="cuda")
            s2 = torch.Stream(device="cuda")
            e = torch.Event()

            with s1:
                y = x + 1
                e.record()  # first record
                z = y * 2
                e.record()  # second record, reuses same event

            with s2:
                e.wait()
                w = x + 3

            return w + z

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        # 2 records + 1 wait = 3 control_deps nodes
        self.assertEqual(len(ctrl_nodes), 3)

        record1_ctrl = ctrl_nodes[0]
        record2_ctrl = ctrl_nodes[1]
        wait_ctrl = ctrl_nodes[2]

        # The wait depends on the SECOND record (last write wins), not the first
        self.assertIn(record2_ctrl, wait_ctrl.args[0])
        self.assertNotIn(record1_ctrl, wait_ctrl.args[0])

        graph.lint()

    @requires_cuda
    def test_same_stream_record_ordering(self) -> None:
        # [FUNCTION DOC] Checks that consecutive records on one stream are chained: three
        # events are recorded in a row after work on one stream. Asserts three control_deps
        # nodes, each record depends on its immediate predecessor and the third does not
        # depend on the first. Ends with graph lint. Needs CUDA.
        """Two record_events on one stream must be ordered relative to each other.

        The second record's per-stream work was reset by the first, so without
        an explicit edge it is emitted bare (no control_deps) and can float above
        the compute, capturing nothing -- the same 'bare side-effectful sync
        floats' bug as an unanchored synchronize_stream. Each record must depend
        on the prior sync on its own stream, so the second record's control_deps
        depends on the first record's control_deps node.
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Create a stream and three events; add one under the
            # stream then record the three events consecutively; returns the add result.
            s = torch.Stream(device="cuda")
            e1 = torch.Event()
            e2 = torch.Event()
            e3 = torch.Event()
            with s:
                a = x + 1
                e1.record()
                e2.record()  # deps reset by e1 -> would be bare without the fix
                e3.record()
            return a

        inp = (torch.ones(2, 2, device="cuda"),)
        # extract_graph already runs wrap_all_sync_nodes_with_control_deps, so
        # assert on its output directly (re-running the pass would wrap the bare
        # e2 a second time and mask the bug).
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        # All records must be wrapped -- e2/e3 must not be left bare nodes.
        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        self.assertEqual(len(ctrl_nodes), 3)
        record1_ctrl, record2_ctrl, record3_ctrl = ctrl_nodes

        # Each record depends only on its immediate predecessor, keeping the
        # per-stream sync ordering a linear chain rather than its full history.
        self.assertIn(record1_ctrl, record2_ctrl.args[0])
        self.assertIn(record2_ctrl, record3_ctrl.args[0])
        self.assertNotIn(record1_ctrl, record3_ctrl.args[0])

        graph.lint()

    @requires_cuda
    def test_wait_stream_anchors_following_record(self) -> None:
        # [FUNCTION DOC] Regression test: a record on a stream that has just waited on another
        # via wait_stream must be chained to that wait instead of being left as a bare node.
        # Asserts no bare record_event nodes remain, two record control_deps and one
        # wait_stream control_deps exist, the wait depends on the first record and the second
        # record depends on the wait. Ends with graph lint. Needs CUDA.
        """A record_event on the WAITING stream after a wait_stream must chain to
        the wait_stream (which runs on that stream), not float above it as a bare
        node. Regression: wait_stream's control_deps was keyed under the waited-on
        stream, so a later record on the waiting stream never found it.
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Compute a matmul on the default stream, record an event,
            # then on a side stream wait for the default stream and record another event, wait
            # on it and return the matmul.
            default = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            ready = torch.cuda.Event()
            e = torch.cuda.Event()
            a = x @ x
            ready.record(default)
            with torch.cuda.stream(side):
                side.wait_stream(default)
                e.record()  # bare on the waiting stream without the fix
            e.wait()
            return a

        inp = (torch.ones(4, 4, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        # The record must be wrapped, not left as a bare (floatable) node in the
        # main graph -- wrapped record_events live inside control_deps subgraphs.
        bare_records = graph.find_nodes(
            op="call_function", target=torch.ops.streams.record_event.default
        )
        self.assertEqual(
            list(bare_records), [], "record_event on the waiting stream floated bare"
        )

        # The wait must depend on the waited-on stream's retained record dependency,
        # and the following record must depend on the wait on its own stream.
        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        record_ctrl = [n for n in ctrl_nodes if "record_event" in n.args[1].name]
        wait_ctrl = [n for n in ctrl_nodes if "wait_stream" in n.args[1].name]
        self.assertEqual(len(record_ctrl), 2)
        self.assertEqual(len(wait_ctrl), 1)
        self.assertIn(record_ctrl[0], wait_ctrl[0].args[0])
        self.assertIn(wait_ctrl[0], record_ctrl[1].args[0])

        graph.lint()

    @requires_cuda
    def test_cross_stream_event_rerecord_without_work(self) -> None:
        # [FUNCTION DOC] Checks re-recording an event from a second stream without any work in
        # between: asserts two record and one wait control_deps nodes, the second record
        # depends on the first, the wait depends on the second record but not on the first.
        # Ends with graph lint. Needs CUDA.
        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Record one event on a first stream after an add, record
            # it again on a second stream with no work, wait for it on a third stream and
            # add; returns the sum.
            s1 = torch.cuda.Stream()
            s2 = torch.cuda.Stream()
            s3 = torch.cuda.Stream()
            e = torch.cuda.Event()
            with torch.cuda.stream(s1):
                y = x + 1
                e.record()
            with torch.cuda.stream(s2):
                e.record()
            with torch.cuda.stream(s3):
                e.wait()
                z = x + 2
            return y + z

        inp = (torch.ones(2, 2, device="cuda"),)
        _, _, fw_graphs, _ = extract_graph(fn, *inp)
        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        record_ctrl = [n for n in ctrl_nodes if "record_event" in n.args[1].name]
        wait_ctrl = [n for n in ctrl_nodes if "wait_event" in n.args[1].name]
        self.assertEqual(len(record_ctrl), 2)
        self.assertEqual(len(wait_ctrl), 1)
        self.assertIn(record_ctrl[0], record_ctrl[1].args[0])
        self.assertIn(record_ctrl[1], wait_ctrl[0].args[0])
        self.assertNotIn(record_ctrl[0], wait_ctrl[0].args[0])

        graph.lint()

    @requires_cuda
    def test_wait_stream_preserves_waited_on_work(self) -> None:
        # [FUNCTION DOC] Checks that wait_stream retains the work of the stream waited on:
        # after the pass, exactly one record control_deps node exists and it has a non-empty
        # dependency tuple. Ends with graph lint. Needs CUDA.
        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Compute a matmul on the default stream, make a side
            # stream wait on the default stream, record an event on the default stream and
            # return the matmul.
            default = torch.cuda.current_stream()
            side = torch.cuda.Stream()
            a = x @ x
            side.wait_stream(default)
            torch.cuda.Event().record(default)
            return a

        inp = (torch.ones(4, 4, device="cuda"),)
        _, _, fw_graphs, _ = extract_graph(fn, *inp)
        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        record_ctrl = [n for n in ctrl_nodes if "record_event" in n.args[1].name]
        self.assertEqual(len(record_ctrl), 1)
        self.assertTrue(record_ctrl[0].args[0])

        graph.lint()

    @requires_cuda
    def test_leading_synchronize_stream_orders_following_work(self) -> None:
        # [FUNCTION DOC] Checks that a leading synchronize on the current stream
        # orders later work: asserts exactly one synchronize_stream control_deps node
        # and that the following add consumes a value threaded through it. Ends with
        # graph lint. Needs CUDA.
        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Synchronize the current stream and return
            # the input plus one.
            torch.cuda.current_stream().synchronize()
            return x + 1

        inp = (torch.ones(4, 4, device="cuda"),)
        _, _, fw_graphs, _ = extract_graph(fn, *inp)
        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        sync_ctrl = [n for n in ctrl_nodes if "synchronize_stream" in n.args[1].name]
        self.assertEqual(len(sync_ctrl), 1)
        adds = list(
            graph.find_nodes(op="call_function", target=torch.ops.aten.add.Tensor)
        )
        self.assertEqual(len(adds), 1)
        self.assertIs(adds[0].args[0].args[0], sync_ctrl[0])

        graph.lint()

    def test_full_barrier_forward_deps_respect_partition(self) -> None:
        # [FUNCTION DOC] Builds a hand-made fx graph with a forward and a backward add and a
        # full-barrier synchronize node, calls the helper that collects forward dependencies
        # of sync nodes, and asserts the barrier depends only on the forward input (the
        # backward-partition nodes are excluded).
        from torch._functorch._aot_autograd.streams import _collect_sync_forward_deps

        graph = torch.fx.Graph()
        fw_input = graph.placeholder("fw_input")
        fw_input.meta["val"] = torch.ones(1)
        bw_input = graph.placeholder("bw_input")
        bw_input.meta["val"] = torch.ones(1)
        barrier = graph.call_function(
            torch.ops.streams.synchronize_stream.default, (0,)
        )
        fw_node = graph.call_function(torch.ops.aten.add.Tensor, (fw_input, 1))
        fw_node.meta["val"] = torch.ones(1)
        bw_node = graph.call_function(torch.ops.aten.add.Tensor, (bw_input, 1))
        bw_node.meta["val"] = torch.ones(1)
        bw_node.meta["partitioner_tag"] = "is_backward"
        graph.output((fw_node, bw_node))

        _, deps = _collect_sync_forward_deps(graph)
        self.assertEqual(list(deps[barrier]), [fw_input])

    def test_backward_full_barrier_filters_forward_sync_deps(self) -> None:
        # [FUNCTION DOC] Builds a small graph with one forward compute node, a forward record
        # and a backward-tagged record, then calls the full-barrier dependency collector for
        # the backward pass. Asserts the result is the forward compute node and the backward
        # sync node, i.e. forward sync nodes are filtered out.
        from torch._functorch._aot_autograd.streams import _collect_full_barrier_deps

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        fw_compute = graph.call_function(torch.ops.aten.add.Tensor, (inp, 1))
        fw_sync = graph.call_function(torch.ops.streams.record_event.default, (0, 0))
        bw_sync = graph.call_function(torch.ops.streams.record_event.default, (1, 0))
        bw_sync.meta["partitioner_tag"] = "is_backward"
        graph.output(fw_compute)

        deps = _collect_full_barrier_deps(
            {0: [fw_compute]}, {0: [fw_sync, bw_sync]}, bwd=True
        )
        self.assertEqual(deps, [fw_compute, bw_sync])

    def test_wait_stream_forward_deps_are_transitive(self) -> None:
        # [FUNCTION DOC] Builds a graph where a wait_stream is followed by a producer on one
        # stream and a consumer on another, and asserts that the collected wait dependencies
        # for the wait node contain only the graph input, showing the dependencies are
        # computed transitively through the consumer chain.
        from torch._functorch._aot_autograd.streams import (
            _collect_sync_forward_deps,
            set_stream,
        )

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        inp.meta["val"] = torch.ones(1)
        wait = graph.call_function(torch.ops.streams.wait_stream.default, (1, 0))
        producer = graph.call_function(torch.ops.aten.add.Tensor, (inp, 1))
        producer.meta["val"] = torch.ones(1)
        set_stream(producer, 0)
        consumer = graph.call_function(torch.ops.aten.mul.Tensor, (producer, 2))
        consumer.meta["val"] = torch.ones(1)
        set_stream(consumer, 1)
        graph.output(consumer)

        wait_deps, _ = _collect_sync_forward_deps(graph)
        self.assertEqual(list(wait_deps[wait]), [inp])

    def test_wait_stream_forward_deps_respect_partition(self) -> None:
        # [FUNCTION DOC] Builds a graph with a forward wait_stream and a backward-tagged
        # wait_stream, each with its own stream compute node, and asserts the collected wait
        # dependencies map each wait to the placeholder of its own partition (forward input
        # for the forward wait, backward input for the backward one).
        from torch._functorch._aot_autograd.streams import (
            _collect_sync_forward_deps,
            set_stream,
        )

        graph = torch.fx.Graph()
        fw_input = graph.placeholder("fw_input")
        fw_input.meta["val"] = torch.ones(1)
        bw_input = graph.placeholder("bw_input")
        bw_input.meta["val"] = torch.ones(1)
        fw_wait = graph.call_function(torch.ops.streams.wait_stream.default, (1, 0))
        fw_node = graph.call_function(torch.ops.aten.add.Tensor, (fw_input, 1))
        fw_node.meta["val"] = torch.ones(1)
        set_stream(fw_node, 1)
        bw_wait = graph.call_function(torch.ops.streams.wait_stream.default, (1, 0))
        bw_wait.meta["partitioner_tag"] = "is_backward"
        bw_node = graph.call_function(torch.ops.aten.add.Tensor, (bw_input, 1))
        bw_node.meta["val"] = torch.ones(1)
        bw_node.meta["partitioner_tag"] = "is_backward"
        set_stream(bw_node, 1)
        graph.output((fw_node, bw_node))

        wait_deps, _ = _collect_sync_forward_deps(graph)
        self.assertEqual(list(wait_deps[fw_wait]), [fw_input])
        self.assertEqual(list(wait_deps[bw_wait]), [bw_input])

    def _assert_empty_sync_anchors_record(
        self, sync_target: torch._ops.OpOverload, sync_args: tuple[int, ...]
    ) -> None:
        # [FUNCTION DOC] HELPER: Shared check: builds a graph holding one sync node of the
        # given op overload and arguments followed by a record_event with no compute between
        # them, wraps sync nodes with control_deps, and asserts exactly one control_deps node
        # whose dependencies include the first sync node. Then runs graph lint.
        from torch._functorch._aot_autograd.streams import (
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        sync = graph.call_function(sync_target, sync_args)
        graph.call_function(torch.ops.streams.record_event.default, (1, 1))
        graph.output(inp)
        gm = torch.fx.GraphModule({}, graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = graph.find_nodes(op="call_function", target=control_deps)
        self.assertEqual(len(ctrl_nodes), 1)
        self.assertIn(sync, ctrl_nodes[0].args[0])
        graph.lint()

    def test_empty_synchronize_stream_anchors_record(self) -> None:
        # [FUNCTION DOC] Uses the shared anchor check with a synchronize_stream sync node
        # followed by an empty record.
        self._assert_empty_sync_anchors_record(
            torch.ops.streams.synchronize_stream.default, (0,)
        )

    def test_empty_synchronize_device_anchors_record(self) -> None:
        # [FUNCTION DOC] Uses the shared anchor check with a synchronize_device sync node (no
        # arguments) followed by an empty record.
        self._assert_empty_sync_anchors_record(
            torch.ops.streams.synchronize_device.default, ()
        )

    def test_empty_wait_stream_anchors_record(self) -> None:
        # [FUNCTION DOC] Uses the shared anchor check with a wait_stream sync node followed
        # by an empty record.
        self._assert_empty_sync_anchors_record(
            torch.ops.streams.wait_stream.default, (1, 0)
        )

    def test_external_wait_event_anchors_record(self) -> None:
        # [FUNCTION DOC] Uses the shared anchor check with a wait_event sync node (an
        # externally recorded event) followed by an empty record.
        self._assert_empty_sync_anchors_record(
            torch.ops.streams.wait_event.default, (0, 1)
        )

    def test_external_wait_event_orders_following_work(self) -> None:
        # [FUNCTION DOC] Builds a graph with an external wait_event followed by an add placed
        # on the waiting stream, wraps sync nodes, and asserts one control_deps node and that
        # the add consumes a value threaded through it. Ends with graph lint.
        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        inp.meta["val"] = torch.ones(1)
        graph.call_function(torch.ops.streams.wait_event.default, (0, 1))
        add = graph.call_function(torch.ops.aten.add.Tensor, (inp, 1))
        add.meta["val"] = torch.ones(1)
        set_stream(add, 1)
        graph.output(add)
        gm = torch.fx.GraphModule({}, graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = graph.find_nodes(op="call_function", target=control_deps)
        self.assertEqual(len(ctrl_nodes), 1)
        self.assertIs(add.args[0].args[0], ctrl_nodes[0])
        graph.lint()

    @requires_cuda
    def test_captured_external_wait_event_orders_following_work(self) -> None:
        # [FUNCTION DOC] End-to-end version of the external wait_event ordering check: records
        # a CUDA event outside the traced function, waits on it from a user stream inside,
        # adds one, and after extraction asserts one wait_event control_deps node and that the
        # add consumes a value threaded through it. Ends with graph lint. Needs CUDA.
        event = torch.cuda.Event()
        stream = torch.cuda.Stream()
        event.record()

        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Enter the captured stream, wait on the captured external
            # event and return the input plus one.
            with torch.cuda.stream(stream):
                event.wait()
                return x + 1

        _, _, fw_graphs, _ = extract_graph(fn, torch.ones(2, device="cuda"))
        graph = fw_graphs[0].graph

        from torch._inductor.fx_passes.control_dependencies import control_deps

        ctrl_nodes = graph.find_nodes(op="call_function", target=control_deps)
        wait_ctrl = [n for n in ctrl_nodes if "wait_event" in n.args[1].name]
        self.assertEqual(len(wait_ctrl), 1)
        add = graph.find_nodes(op="call_function", target=torch.ops.aten.add.Tensor)[0]
        self.assertIs(add.args[0].args[0], wait_ctrl[0])
        graph.lint()

    def test_external_wait_event_preserves_copy_destination(self) -> None:
        # [FUNCTION DOC] Builds a graph with an external wait_event followed by an in-place
        # copy on the waiting stream, wraps sync nodes, and asserts the copy destination is
        # still the original placeholder while the copy source is threaded through the
        # control_deps node. Ends with graph lint.
        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        dst = graph.placeholder("dst")
        dst.meta["val"] = torch.ones(1)
        src = graph.placeholder("src")
        src.meta["val"] = torch.ones(1)
        graph.call_function(torch.ops.streams.wait_event.default, (0, 1))
        copy = graph.call_function(torch.ops.aten.copy_.default, (dst, src))
        copy.meta["val"] = torch.ones(1)
        set_stream(copy, 1)
        graph.output(copy)
        gm = torch.fx.GraphModule({}, graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl = graph.find_nodes(op="call_function", target=control_deps)[0]
        self.assertIs(copy.args[0], dst)
        self.assertIs(copy.args[1].args[0], ctrl)
        graph.lint()

    def test_empty_synchronize_event_anchors_record(self) -> None:
        # [FUNCTION DOC] Uses the shared anchor check with a synchronize_event sync node
        # followed by an empty record.
        self._assert_empty_sync_anchors_record(
            torch.ops.streams.synchronize_event.default, (0,)
        )

    def test_wait_stream_depends_on_prior_waiting_stream_work(self) -> None:
        # [FUNCTION DOC] Builds a graph with prior work on a stream followed by a wait_stream
        # on that stream, wraps sync nodes, and asserts one control_deps node whose
        # dependencies include the prior work. Ends with graph lint.
        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        inp.meta["val"] = torch.ones(1)
        prior_work = graph.call_function(torch.ops.aten.add.Tensor, (inp, 1))
        prior_work.meta["val"] = torch.ones(1)
        set_stream(prior_work, 1)
        graph.call_function(torch.ops.streams.wait_stream.default, (1, 0))
        graph.output(prior_work)
        gm = torch.fx.GraphModule({}, graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = graph.find_nodes(op="call_function", target=control_deps)
        self.assertEqual(len(ctrl_nodes), 1)
        self.assertIn(prior_work, ctrl_nodes[0].args[0])
        graph.lint()

    def test_wait_stream_self_wait_orders_work_once(self) -> None:
        # [FUNCTION DOC] Builds a graph with work on a stream, a wait_stream where a stream
        # waits on itself, and later work on the same stream. Asserts the prior work appears
        # once in the control_deps dependencies and the later work consumes a value threaded
        # through the control_deps node. Ends with graph lint.
        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        graph = torch.fx.Graph()
        x = graph.placeholder("x")
        x.meta["val"] = torch.ones(1)
        y = graph.placeholder("y")
        y.meta["val"] = torch.ones(1)
        prior_work = graph.call_function(torch.ops.aten.add.Tensor, (x, 1))
        prior_work.meta["val"] = torch.ones(1)
        set_stream(prior_work, 1)
        graph.call_function(torch.ops.streams.wait_stream.default, (1, 1))
        following_work = graph.call_function(torch.ops.aten.mul.Tensor, (y, 2))
        following_work.meta["val"] = torch.ones(1)
        set_stream(following_work, 1)
        graph.output((prior_work, following_work))
        gm = torch.fx.GraphModule({}, graph)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl = graph.find_nodes(op="call_function", target=control_deps)[0]
        self.assertEqual(ctrl.args[0].count(prior_work), 1)
        self.assertIs(following_work.args[0].args[0], ctrl)
        graph.lint()

    @requires_cuda
    def test_epilogue_copy_stream_tracking(self):
        # [FUNCTION DOC] Verifies the epilogue copy for an input mutated in backward is placed
        # on the stream of the mutation. Uses a custom autograd function whose backward
        # mutates a saved tensor on one stream and computes gradients on another. Checks the
        # forward graph text, runs backward and checks the backward graph text where the copy
        # back is annotated with the mutation stream. Needs CUDA.
        """
        Test that epilogue copies for mutated inputs use the correct stream.
        This verifies that ViewAndMutationMeta.mutated_inp_stream_indices is
        properly populated and used at runtime.
        Uses a custom autograd.Function where the backward mutates a saved
        tensor on a specific stream.
        """

        class BwMutationWithStream(torch.autograd.Function):
            # [CLASS DOC] HELPER: Custom autograd function creating two streams in
            # forward (stored on the context): forward computes on the second stream and
            # backward mutates the saved input on the first stream while computing
            # gradients on the second.
            @staticmethod
            def forward(ctx, x, y):
                # [FUNCTION DOC] HELPER: Save the input for backward, create and store
                # two streams on the context, compute a scaled sum under the second
                # stream and return it.
                ctx.save_for_backward(x)
                ctx.s1 = torch.Stream(device="cuda:0")
                ctx.s2 = torch.Stream(device="cuda:0")
                # Do computation on stream s2
                with ctx.s2:
                    result = x * 2 + y
                return result

            @staticmethod
            def backward(ctx, grad_output):
                # [FUNCTION DOC] HELPER: Mutate the saved input in place under the first
                # stream, compute gradients for both inputs under the second stream and return
                # them with two None entries.
                (x,) = ctx.saved_tensors
                # Mutate saved tensor x on stream s1 in backward
                with ctx.s1:
                    x.mul_(2)
                # Compute gradients on stream s2
                with ctx.s2:
                    grad_x = grad_output * 2
                    grad_y = grad_output.clone()
                return grad_x, grad_y, None, None

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Apply the custom autograd function to the inputs and
            # return the result.
            result = BwMutationWithStream.apply(x, y)
            return result

        x = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        y = torch.ones(2, 2, requires_grad=True, device="cuda:0")
        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, x.clone(), y.clone())
        self.assertEqual(len(fw_graphs), 1)
        # Forward graph should show computation on stream 1 (s2)
        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", primals_2: "f32[2, 2]"):
        # Annotation: {'stream': 2}
        mul: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(mul, primals_2);  mul = primals_2 = None
        return (add, primals_1)
""",
        )
        # Run backward and check that the epilogue copy uses stream 0 (s1)
        actual.sum().backward()
        # The backward graph should show:
        # 1. Mutation happening on stream 0 (s1)
        # 2. Gradient computation on stream 1 (s2)
        # 3. Epilogue copy for the mutated tensor on stream 0 (s1)
        self.assertExpectedInline(
            print_graph(bw_graphs[0]),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[2, 2]", tangents_1: "f32[2, 2]"):
        # Annotation: {'stream': 2} Backward of forward node:
        mul_2: "f32[2, 2]" = torch.ops.aten.mul.Tensor(tangents_1, 2)

        # Annotation: {'stream': 2} Backward of forward node:
        clone: "f32[2, 2]" = torch.ops.aten.clone.default(tangents_1);  tangents_1 = None

        # Annotation: {'stream': 1} Backward of forward node:
        mul_1: "f32[2, 2]" = torch.ops.aten.mul.Tensor(primals_1, 2)

        # Annotation: {'stream': 1} No stacktrace found for following nodes
        copy_: "f32[2, 2]" = torch.ops.aten.copy_.default(primals_1, mul_1);  primals_1 = mul_1 = copy_ = None
        return (mul_2, clone)
""",
        )

    @requires_cuda
    def test_inductor_lowering(self):
        # [FUNCTION DOC] Compiles with the default backend a function that records an event
        # while mutating its input, with Inductor implicit fallbacks disabled, and runs it.
        # Passes if Inductor lowers the stream ops without needing fallbacks. Needs CUDA.
        with patch("torch._inductor.config.implicit_fallbacks", False):

            @torch.compile()  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Create an event, update the input in place, record
                # the event and return the input.
                e = torch.Event()
                x += x + 1
                e.record()
                return x

            inp = (torch.ones(2, 2, device="cuda"),)
            fn(*inp)

    def test_is_marked_side_effectful(self):
        # [FUNCTION DOC] Asserts that the fork, join, wait_event, record_event and
        # synchronize_event stream ops are all registered in the fx set of side-
        # effectful functions.
        self.assertIn(
            torch.ops.streams.fork.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.join.default, torch.fx.node._side_effectful_functions
        )
        self.assertIn(
            torch.ops.streams.wait_event.default,
            torch.fx.node._side_effectful_functions,
        )
        self.assertIn(
            torch.ops.streams.record_event.default,
            torch.fx.node._side_effectful_functions,
        )
        self.assertIn(
            torch.ops.streams.synchronize_event.default,
            torch.fx.node._side_effectful_functions,
        )

    @requires_cuda
    def test_backward_sync_control_deps_e2e(self) -> None:
        # [FUNCTION DOC] End-to-end check that a backward graph for a function with two
        # streams contains control_deps nodes. Extracts graphs, runs backward on one
        # output and asserts at least one backward graph with one or more control_deps
        # nodes. Needs CUDA.
        """
        End-to-end test verifying that backward sync nodes are wrapped with control_deps.

        This test:
        1. Creates a function with multiple streams
        2. Compiles it and runs backward
        3. Verifies control_deps nodes are present in the backward graph
        """
        from torch._inductor.fx_passes.control_dependencies import control_deps

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Create two streams and compute a different affine
            # expression under each; returns both results.
            s0 = torch.Stream()
            s1 = torch.Stream()
            with s0:
                z0 = 2 * x + y
            with s1:
                z1 = 3 * x + y
            return z0, z1

        inp = (
            torch.ones(2, 2, device="cuda:0", requires_grad=True) + 1,
            torch.ones(2, 2, device="cuda:0", requires_grad=True),
        )

        (
            actual,
            _,
            fw_graphs,
            bw_graphs,
        ) = extract_graph(fn, *inp)

        # Run backward to trigger backward graph capture
        actual[0].sum().backward()

        # Check that backward graph has control_deps nodes
        self.assertGreaterEqual(len(bw_graphs), 1)
        bw_graph = bw_graphs[0]

        control_deps_nodes = list(
            bw_graph.graph.find_nodes(op="call_function", target=control_deps)
        )

        # Should have at least one control_deps node wrapping sync operations
        self.assertGreater(
            len(control_deps_nodes),
            0,
            "Expected control_deps nodes in backward graph for stream synchronization",
        )

    def test_sync_dealloc_has_fake_impl(self):
        # [FUNCTION DOC] Verifies that the sync_dealloc op has a fake implementation by
        # calling it on a fake tensor under a fake tensor mode; passes if no error is raised.
        """Test that sync_dealloc has a registered fake impl.

        Without a fake impl, Inductor's backward compilation crashes when the
        backward graph contains cross-stream sync_dealloc ops.
        """
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(4)
            # Should not raise "no fake impl registered"
            torch.ops.streams.sync_dealloc.default(0, 1, t)

    def test_record_stream_has_fake_impl(self):
        # [FUNCTION DOC] Verifies the record_stream op has a fake implementation with a
        # matching signature by calling it on a fake tensor under a fake tensor mode; passes
        # if no error is raised.
        """Test that record_stream's fake impl has the correct signature."""
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            t = torch.randn(4)
            # Should not raise due to signature mismatch
            torch.ops.streams.record_stream.default(t, 0)

    @requires_cuda
    def test_record_stream(self):
        # [FUNCTION DOC] Compiles with a recording backend a function that calls record_stream
        # on its input with a new stream, and asserts one graph was captured containing a node
        # targeting the record_stream op. Needs CUDA.
        backend = torch._dynamo.testing.EagerAndRecordGraphs()

        def fn(x):
            # [FUNCTION DOC] HELPER: Create a stream, call record_stream on the input with it
            # and return the input.
            s = torch.Stream()
            x.record_stream(s)
            return x

        compiled = torch.compile(fn, backend=backend, fullgraph=True)
        compiled(torch.randn(4, device="cuda"))

        self.assertEqual(len(backend.graphs), 1)
        found = any(
            node.target is torch.ops.streams.record_stream
            for node in backend.graphs[0].graph.nodes
        )
        self.assertTrue(found, "record_stream op not found in graph")

    @requires_cuda
    def test_event_record_after_input_mutation_errors(self):
        # [FUNCTION DOC] Asserts that compiling a function which mutates an input in place and
        # then records an event on the same stream and returns the event raises a RuntimeError
        # saying an event was recorded on a stream. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: On a new stream mutate the input in place, record a new
            # event and return the event.
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_stack_traces(self):
        # [FUNCTION DOC] Same failing scenario as the previous error test but inspects the
        # error message: it must name where the input mutation occurred and where the event
        # was recorded, including the offending source lines. Fails the test if no error is
        # raised. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: On a new stream mutate the input in place, record a new
            # event and return the event.
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return e

        try:
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )
            self.fail("Expected RuntimeError")
        except RuntimeError as e:
            msg = str(e)
            self.assertIn("Input mutation occurred here:", msg)
            self.assertIn("x.add_(1)", msg)
            self.assertIn("Event record occurred here:", msg)
            self.assertIn("e.record()", msg)

    @requires_cuda
    def test_event_record_after_input_mutation_record_event(self):
        # [FUNCTION DOC] Checks the error also fires when the event is produced by the
        # stream's own record-event method after an input mutation on that stream. Expects a
        # RuntimeError about an event recorded on a stream. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Mutate the input in place under a new stream and return
            # the event created from that stream.
            s = torch.Stream()
            with s:
                x.add_(1)
                e = s.record_event()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_through_view(self):
        # [FUNCTION DOC] Checks the error also fires when the mutation is made through
        # a view of the input. Expects a RuntimeError about an event recorded on a
        # stream. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Take a flattened view of the input, mutate it in place on
            # a new stream, record an event and return it.
            s = torch.Stream()
            e = torch.Event()
            v = x.view(-1)
            with s:
                v.add_(1)
                e.record()
            return e

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @requires_cuda
    def test_event_record_after_input_mutation_input_event(self):
        # [FUNCTION DOC] Checks the error also fires when the event is passed in as an input.
        # Expects a RuntimeError about an event recorded on a stream. Needs CUDA.
        def fn(x, e):
            # [FUNCTION DOC] HELPER: On a new stream mutate the first input in place, record
            # the given event and return the first input.
            s = torch.Stream()
            with s:
                x.add_(1)
                e.record()
            return x

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda"),
                torch.Event(),
            )

    @requires_cuda
    def test_event_record_before_input_mutation_no_error(self):
        # [FUNCTION DOC] Checks no error is raised when the event is recorded before the input
        # is mutated on the same stream. Compiles with fullgraph and runs. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: On a new stream record an event first, then mutate the
            # input in place; returns the event.
            s = torch.Stream()
            e = torch.Event()
            with s:
                e.record()
                x.add_(1)
            return e

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(2, 2, device="cuda")
        )

    @requires_cuda
    def test_event_record_on_different_stream_no_error(self):
        # [FUNCTION DOC] Checks no error is raised when the input mutation and the event
        # record happen on different streams. Compiles with fullgraph and runs. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Mutate the input in place on one stream and record an
            # event on another; returns the event.
            s0 = torch.Stream()
            s1 = torch.Stream()
            e = torch.Event()
            with s0:
                x.add_(1)
            with s1:
                e.record()
            return e

        torch.compile(fn, backend="eager", fullgraph=True)(
            torch.ones(2, 2, device="cuda")
        )

    @requires_cuda
    def test_event_not_returned_no_error(self):
        # [FUNCTION DOC] Despite the name, this test expects a RuntimeError about an event
        # recorded on a stream even though the event is not returned (the function returns the
        # input). Needs CUDA.
        # KNOWN ISSUE: the test name says no error is expected but the body asserts that
        # an error is raised.
        def fn(x):
            # [FUNCTION DOC] HELPER: On a new stream mutate the input in place and record a
            # new event, then return the input.
            s = torch.Stream()
            e = torch.Event()
            with s:
                x.add_(1)
                e.record()
            return x

        with self.assertRaisesRegex(RuntimeError, "An event was recorded on a stream"):
            torch.compile(fn, backend="eager", fullgraph=True)(
                torch.ones(2, 2, device="cuda")
            )

    @unittest.skipIf(
        IS_LINUX or IS_MACOS or TEST_WITH_ROCM or IS_WINDOWS,
        "https://github.com/pytorch/pytorch/issues/178155",
    )
    @requires_cuda
    @unittest.skip("https://github.com/pytorch/pytorch/issues/177771")
    def test_cuda_event_record_on_stream(self):
        # [FUNCTION DOC] Intended to check that a torch.cuda.Event is accepted by Stream
        # record_event at the C++ type check. Always skipped (tracked issue). Body creates a
        # CUDA stream and event and records the event on the stream.
        """torch.cuda.Event should be accepted by torch.Stream.record_event (C++ type check)."""
        s = torch.Stream(device="cuda")
        e = torch.cuda.Event()
        # This hits THPStream_record_event in Stream.cpp which does a type check
        s.record_event(e)

    @requires_cuda
    def test_event_synchronize_tracing(self):
        # [FUNCTION DOC] Traces a function that records an event, adds one to the input and
        # synchronizes the event, and asserts the printed forward graph shows a
        # record_event followed by an add and a control_deps node wrapping a
        # synchronize_event subgraph. Needs CUDA.
        def fn(x):
            # [FUNCTION DOC] HELPER: Record a new event, add one to the input, synchronize the
            # event and return the sum.
            e = torch.Event()
            e.record()
            x = x + 1
            e.synchronize()
            return x

        inp = (torch.ones(2, 2, device="cuda"),)
        (
            _,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)

        self.assertExpectedInline(
            print_graph(fw_graphs[0]),
            """\
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 2]"):
        #
        record_event = torch.ops.streams.record_event.default(1, 0)

        #
        add: "f32[2, 2]" = torch.ops.aten.add.Tensor(arg0_1, 1);  arg0_1 = None

        # No stacktrace found for following nodes
        subgraph_synchronize_event = self.subgraph_synchronize_event
        control_deps = torch.ops.higher_order.control_deps((record_event, add), subgraph_synchronize_event);  record_event = subgraph_synchronize_event = control_deps = None
        return (add,)

    class subgraph_synchronize_event(torch.nn.Module):
        def forward(self):
            #
            synchronize_event_default = torch.ops.streams.synchronize_event.default(1)
            return synchronize_event_default
""",
        )

    @requires_cuda
    def test_event_synchronize_inductor_lowering(self):
        # [FUNCTION DOC] Compiles with the default backend a function that records and
        # synchronizes an event, with Inductor implicit fallbacks disabled, and runs it.
        # Passes if lowering succeeds without fallbacks. Needs CUDA.
        with patch("torch._inductor.config.implicit_fallbacks", False):

            @torch.compile()  # noqa: UNSPECIFIED_BACKEND
            def fn(x):
                # [FUNCTION DOC] HELPER: Create an event, add one to the input, record and
                # synchronize the event and return the sum.
                e = torch.Event()
                x = x + 1
                e.record()
                e.synchronize()
                return x

            inp = (torch.ones(2, 2, device="cuda"),)
            fn(*inp)

    @requires_cuda
    def test_control_deps_wrapping_synchronize_event(self) -> None:
        # [FUNCTION DOC] Checks that synchronize_event threads the values passed through
        # record_event. Captures the raw graph, sets stream metadata on compute nodes, wraps
        # sync nodes and asserts two control_deps nodes, that the synchronize depends on the
        # record, that record getitem outputs are passed on through the synchronize, and that
        # the later multiply consumes a getitem of the synchronize node. Needs CUDA.
        """Test that synchronize_event threads recorded ops' values through.

        After record_event wraps ops in control_deps and produces getitem
        pass-throughs, synchronize_event must also thread those through so
        that subsequent consumers depend on the synchronize.
        """

        def fn(x) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Add one to the input, record and synchronize a new event,
            # then multiply the earlier value by two and return it.
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            # z uses y which was produced before the record — its value must
            # be threaded through both record and synchronize control_deps.
            z = y * 2
            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        import operator

        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )
        from torch._inductor.fx_passes.control_dependencies import control_deps

        # extract_graph doesn't annotate streams, so set stream metadata on
        # compute nodes to match the record_event's stream index.
        record_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.streams.record_event.default
        )
        stream_idx = record_node.args[1]
        for n in graph.nodes:
            if (
                n.op == "call_function"
                and "val" in n.meta
                and n.target
                not in (
                    torch.ops.streams.record_event.default,
                    torch.ops.streams.synchronize_event.default,
                )
            ):
                set_stream(n, stream_idx)

        wrap_all_sync_nodes_with_control_deps(gm)

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        # record_event + synchronize_event = 2 control_deps nodes
        self.assertEqual(len(ctrl_nodes), 2)
        record_ctrl = ctrl_nodes[0]
        sync_ctrl = ctrl_nodes[1]

        # synchronize_event's control_deps should depend on record's ctrl
        self.assertIn(record_ctrl, sync_ctrl.args[0])

        # The record should thread through the add (y = x + 1)
        record_getitems = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is record_ctrl
        ]
        self.assertGreaterEqual(len(record_getitems), 1)

        # Those getitems should be passed through synchronize's control_deps
        # as additional args (the passthrough deps)
        sync_passthrough_args = sync_ctrl.args[2:]  # skip (deps_tuple, subgraph)
        for getitem in record_getitems:
            self.assertIn(
                getitem,
                sync_passthrough_args,
                "record_event's getitem should be threaded through synchronize_event",
            )

        # The mul (z = y * 2) should consume a getitem from synchronize's
        # control_deps, not directly from record's.
        sync_getitems = [
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target == operator.getitem
            and n.args[0] is sync_ctrl
        ]
        self.assertGreaterEqual(len(sync_getitems), 1)

        # Find the mul node and verify it uses a sync getitem
        mul_nodes = [
            n
            for n in graph.nodes
            if n.op == "call_function" and n.target == torch.ops.aten.mul.Tensor
        ]
        self.assertEqual(len(mul_nodes), 1)
        mul_args = set(mul_nodes[0].args)
        self.assertTrue(
            mul_args & set(sync_getitems),
            "mul should depend on synchronize_event's getitem, not record_event's",
        )

    @requires_cuda
    def test_external_event_synchronize_threads_inputs(self) -> None:
        # [FUNCTION DOC] Checks that for an externally recorded event, synchronize threads
        # graph inputs. Captures the raw graph, erases the record_event node, sets stream
        # metadata on compute nodes, wraps and recompiles, then asserts one control_deps
        # node that includes the add in its dependencies and feeds the multiply. Ends with
        # graph lint. Needs CUDA.
        """When the event was recorded externally, synchronize threads graph inputs through."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Add one to the input, record and synchronize a new event,
            # then multiply the earlier value by two and return it.
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            z = y * 2
            return z

        inp = (torch.ones(2, 2, device="cuda"),)
        # Patch out wrapping so we get the raw graph to manually wrap below.
        with patch(
            "torch._functorch._aot_autograd.graph_capture.wrap_all_sync_nodes_with_control_deps"
        ):
            (
                _,
                _,
                fw_graphs,
                _,
            ) = extract_graph(fn, *inp)

        gm = fw_graphs[0]
        graph = gm.graph

        from torch._functorch._aot_autograd.streams import (
            set_stream,
            wrap_all_sync_nodes_with_control_deps,
        )

        # Remove the record_event to simulate an externally-recorded event.
        record_node = next(
            n
            for n in graph.nodes
            if n.op == "call_function"
            and n.target is torch.ops.streams.record_event.default
        )
        stream_idx = record_node.args[1]
        graph.erase_node(record_node)

        # Set stream metadata on compute nodes.
        for n in graph.nodes:
            if (
                n.op == "call_function"
                and "val" in n.meta
                and n.target is not torch.ops.streams.synchronize_event.default
            ):
                set_stream(n, stream_idx)

        wrap_all_sync_nodes_with_control_deps(gm)
        gm.recompile()

        from torch._inductor.fx_passes.control_dependencies import control_deps

        ctrl_nodes = list(graph.find_nodes(op="call_function", target=control_deps))
        self.assertEqual(len(ctrl_nodes), 1)
        ctrl = ctrl_nodes[0]
        add = graph.find_nodes(op="call_function", target=torch.ops.aten.add.Tensor)[0]
        mul = graph.find_nodes(op="call_function", target=torch.ops.aten.mul.Tensor)[0]
        self.assertIn(add, ctrl.args[0])
        self.assertIs(mul.args[0].args[0], ctrl)

        graph.lint()

    @requires_cuda
    def test_event_synchronize_control_deps_e2e(self):
        # [FUNCTION DOC] End-to-end check with the default compile backend: compute, record,
        # synchronize an event and then use the earlier result. Asserts the compiled result
        # equals eager. Needs CUDA.
        """E2E: compute → record → synchronize → use result through torch.compile."""

        def f(x):
            # [FUNCTION DOC] HELPER: Add one to the input, record and synchronize a new event,
            # then multiply the earlier value by two and return it.
            e = torch.Event()
            y = x + 1
            e.record()
            e.synchronize()
            z = y * 2
            return z

        inp = torch.ones(2, 2, device="cuda")
        eager_result = f(inp)
        compiled_result = torch.compile(f)(inp)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_event_synchronize_e2e(self):
        # [FUNCTION DOC] End-to-end check with the default compile backend of non-blocking
        # copies to the CPU of ten half precision CUDA tensors, each followed by an event
        # record, then synchronizing all events and concatenating. Asserts the compiled result
        # equals eager. Needs CUDA.
        def f(a_list):
            # [FUNCTION DOC] HELPER: For each tensor in the list copy it to the CPU without
            # blocking and record a fresh event; afterwards synchronize every event and return
            # the concatenation of the CPU copies.
            a_cpu_list = []
            a_to_cpu_event_list = []
            for a in a_list:
                a_cpu = a.to(device="cpu", non_blocking=True)
                e = torch.Event()
                e.record()
                a_cpu_list.append(a_cpu)
                a_to_cpu_event_list.append(e)

            for e in a_to_cpu_event_list:
                e.synchronize()

            return torch.cat(a_cpu_list)

        f_compiled = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND
        inputs = [
            torch.rand(100, dtype=torch.float16, device="cuda") for _ in range(10)
        ]
        eager_result = f(inputs)
        compiled_result = f_compiled(inputs)
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_event_record_wait_on_default_stream(self):
        # [FUNCTION DOC] End-to-end check that recording and waiting on a CUDA event on the
        # default stream compiles and matches eager. The event is created outside the
        # function. Needs CUDA.
        e = torch.cuda.Event()

        def f(x):
            # [FUNCTION DOC] HELPER: Add one to the input, record and wait on the captured
            # event and return the earlier value plus one.
            y = x + 1
            e.record()
            e.wait()
            return y + 1

        f_compiled = torch.compile(f)  # noqa: UNSPECIFIED_BACKEND
        x = torch.randn(10, device="cuda")
        eager_result = f(x)
        compiled_result = f_compiled(x)
        self.assertEqual(eager_result, compiled_result)

    @requires_cuda
    def test_record_stream_inductor_output_code(self) -> None:
        # [FUNCTION DOC] Compiles with Inductor a function using record_stream between two
        # computations and uses the generated wrapper code to check that the record_stream
        # call appears after a kernel run and before the return. Also asserts the numeric
        # result. Needs CUDA.
        """Verify record_stream is ordered between the producing kernel and the
        consuming kernel in inductor-generated wrapper code."""
        from torch._inductor.utils import run_and_get_code
        from torch.testing import FileCheck

        def fn(x):
            # [FUNCTION DOC] HELPER: Create a stream, add one to the input, call record_stream
            # with the stream on the result and multiply by two.
            s = torch.Stream(device="cuda")
            y = x + 1
            y.record_stream(s)
            z = y * 2
            return z

        compiled = torch.compile(fn, backend="inductor", fullgraph=True)
        x = torch.randn(1024, device="cuda")
        result, (code,) = run_and_get_code(compiled, x)
        self.assertEqual(result, (x + 1) * 2)

        # record_stream must appear after the kernel that produces the tensor
        # and before the return.
        FileCheck().check(".run(").check(
            "torch.ops.streams.record_stream.default("
        ).check("return").run(code)

    @requires_cuda
    def test_del_multi_stream_sync_dealloc(self):
        # [FUNCTION DOC] Checks that deleting a tensor variable that was also used on another
        # stream (synchronized by an event) produces sync_dealloc and record_event ops in the
        # graph, and that the compiled result equals eager. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compute an add on the default stream, add the inputs on
            # a second stream and record an event, wait on it, delete the first input and
            # return both results.
            s = torch.Stream()
            e = torch.Event()
            z0 = x + 1
            with s:
                z = torch.add(x, y)
                e.record()
            e.wait()
            del x
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_del_same_stream_no_sync_dealloc(self):
        # [FUNCTION DOC] Checks that deleting a tensor whose uses are all on the same stream
        # does not generate a sync_dealloc op, while results equal eager. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Inside a stream context add the inputs, delete the first
            # input and record an event; wait on it outside and return the sum.
            s = torch.Stream()
            e = torch.Event()
            with s:
                z = torch.add(x, y)
                del x
                e.record()
            e.wait()
            return z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertNotIn("sync_dealloc", graph_str)

    @requires_cuda
    def test_del_single_stream_no_sync_dealloc(self):
        # [FUNCTION DOC] Checks that deleting a tensor in code with a single stream does not
        # generate a sync_dealloc op, while results equal eager. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add the inputs, delete the first input
            # and return the sum.
            z = torch.add(x, y)
            del x
            return z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertNotIn("sync_dealloc", graph_str)

    @requires_cuda
    def test_del_attr_multi_stream_sync_dealloc(self):
        # [FUNCTION DOC] Checks that deleting an attribute holding a tensor also yields
        # sync_dealloc and record_event ops when the tensor was used on a second stream, and
        # that results equal eager. Needs CUDA.
        class Holder:
            # [CLASS DOC] HELPER: Empty plain object used as a mutable attribute
            # container for a tensor.
            pass

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Store the first input as an attribute of a holder object,
            # use it on a second stream with an event record, wait, delete the attribute and
            # return both results.
            s = torch.Stream()
            e = torch.Event()
            h = Holder()
            h.tensor = x
            z0 = x + 1
            with s:
                z = torch.add(h.tensor, y)
                e.record()
            e.wait()
            del h.tensor
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_del_subscr_multi_stream_sync_dealloc(self):
        # [FUNCTION DOC] Checks that deleting a dictionary entry holding a tensor also yields
        # sync_dealloc and record_event ops when the tensor was used on a second stream, and
        # that results equal eager. Needs CUDA.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Store the first input in a dictionary, use it on a second
            # stream with an event record, wait, delete the entry and return both results.
            s = torch.Stream()
            e = torch.Event()
            d = {"t": x}
            z0 = x + 1
            with s:
                z = torch.add(d["t"], y)
                e.record()
            e.wait()
            del d["t"]
            return z0, z

        inp = (torch.ones(2, 2, device="cuda"), torch.ones(2, 2, device="cuda"))
        expected = fn(*inp)
        (
            actual,
            _,
            fw_graphs,
            _,
        ) = extract_graph(fn, *inp)
        self.assertEqual(len(fw_graphs), 1)
        self.assertEqual(expected, actual)
        graph_str = print_graph(fw_graphs[0])
        self.assertIn("sync_dealloc", graph_str)
        self.assertIn("record_event", graph_str)

    @requires_cuda
    def test_stream_pointer_extraction_edge_cases(self):
        # [FUNCTION DOC] Verifies raw stream handles of different streams are extracted
        # correctly by Inductor-compiled code, including inside another stream context and for
        # the default stream. Asserts the handles of two user streams differ and are large,
        # and compiled outputs match the eager handles. Needs CUDA.
        def get_ptrs(stream_a, stream_b, default_stream):
            # [FUNCTION DOC] HELPER: Return a tuple of the raw handles of the
            # three given streams.
            return (
                stream_a.cuda_stream,
                stream_b.cuda_stream,
                default_stream.cuda_stream,
            )

        s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
        default_s = torch.cuda.default_stream()
        expected_s1, expected_s2 = s1.cuda_stream, s2.cuda_stream

        self.assertNotEqual(expected_s1, expected_s2)
        self.assertGreater(expected_s1, 1000)

        opt_get_ptrs = torch.compile(get_ptrs, backend="inductor")

        s3 = torch.cuda.Stream()
        with torch.cuda.stream(s3):
            actual_s1, actual_s2, actual_default = opt_get_ptrs(s1, s2, default_s)

        self.assertEqual(actual_s1, expected_s1)
        self.assertEqual(actual_s2, expected_s2)
        self.assertEqual(actual_default, default_s.cuda_stream)

    def test_expand_dict_returning_deps_triton_kernel(self) -> None:
        # [FUNCTION DOC] Tests the expand-dict-returning-deps pass helper on a hand-built
        # graph where a triton kernel wrapper returns a dictionary, followed by a record_event
        # sync and a getitem user. Asserts the dictionary node is replaced by a per-key
        # getitem node placed before the sync, the old getitem is erased and the add now
        # consumes the new node.
        """Triton kernel dicts must not be passed through control_deps.

        triton_kernel_wrapper_functional returns a dict.  If the dict node is
        threaded through control_deps as a pass-through value,
        decompose_triton_kernel_wrapper_functional later replaces it with a
        raw Python dict (via replace_with_graph -> replace_all_uses_with),
        corrupting the graph and causing KeyError during lowering.

        _expand_dict_returning_deps should insert per-key getitem nodes
        *before* the sync so that only tensor-valued nodes are passed through.
        """
        import operator

        from torch._functorch._aot_autograd.streams import _expand_dict_returning_deps

        graph = torch.fx.Graph()
        # Placeholder input
        x = graph.placeholder("x")
        x.meta["val"] = torch.empty(4, 4)

        # Simulate a triton_kernel_wrapper_functional call returning a dict
        triton_func = graph.call_function(
            torch.ops.higher_order.triton_kernel_wrapper_functional,
            kwargs={
                "kernel_idx": 0,
                "constant_args_idx": 0,
                "grid": [(1, 1, 1)],
                "tma_descriptor_metadata": {},
                "kwargs": {"Out": x},
                "tensors_to_clone": ["Out"],
            },
        )
        triton_func.meta["val"] = {"Out": torch.empty(4, 4)}

        # Simulate a sync node (record_event) between triton_func and getitem
        sync_node = graph.call_function(
            torch.ops.streams.record_event.default,
            args=(0, 0),
        )

        # getitem user AFTER the sync -- this is the problematic pattern
        getitem_out = graph.call_function(
            operator.getitem,
            args=(triton_func, "Out"),
        )
        getitem_out.meta["val"] = torch.empty(4, 4)

        # Use getitem_out so it's live
        add_node = graph.call_function(torch.ops.aten.add.Tensor, args=(getitem_out, x))
        graph.output(add_node)

        # Build visited set: everything at or before sync_node
        visited: set[torch.fx.Node] = set()
        for n in graph.nodes:
            visited.add(n)
            if n is sync_node:
                break

        # deps_with_uses_after_sync: triton_func has getitem_out as after-sync user
        deps = [triton_func]

        expanded = _expand_dict_returning_deps(deps, visited, graph, sync_node)

        # The triton_func dict should NOT be in the expanded list
        self.assertNotIn(triton_func, expanded)

        # Instead, we should have a new getitem node for key "Out"
        self.assertEqual(len(expanded), 1)
        new_gi = expanded[0]
        self.assertEqual(new_gi.target, operator.getitem)
        self.assertEqual(new_gi.args, (triton_func, "Out"))

        # The new getitem should be BEFORE the sync in graph order
        node_order = list(graph.nodes)
        self.assertLess(node_order.index(new_gi), node_order.index(sync_node))

        # The old getitem_out should have been erased; add_node now uses new_gi
        self.assertNotIn(getitem_out, set(graph.nodes))
        self.assertIn(new_gi, add_node.all_input_nodes)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
