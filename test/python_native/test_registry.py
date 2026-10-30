# Owner(s): ["module: dsl-native-ops"]
# [MODULE DOC] Tests for torch._native.registry, the registry of python-native op overrides.
# TestRegistry checks graph building without touching the dispatcher (node dataclass, register
# and deregister of overrides, reordering by a user function, user ordering function lookup from
# an environment variable, rejected arguments). TestRegistryRuntime installs real overrides on
# aten mul and bmm in the dispatcher and checks routing, fall-through, copy-on-write (COW)
# tensor handling under Dynamo and export, deregister and re-enable, and fake tensor shape
# inference. Also defines a wrapper tensor subclass used to probe COW checks.

from unittest.mock import MagicMock, patch

import torch
import torch._dynamo.trace_rules as trace_rules
import torch._native.registry as registry_module
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


class _CowStateWrapperTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Minimal wrapper tensor subclass holding one inner tensor in a slot,
    # built with the same size, dtype, layout, device, strides and storage offset as the inner
    # one. It supports flatten and unflatten for the compiler and raises AssertionError if it is
    # ever dispatched, so tests can prove no dispatch happens on it.
    elem: torch.Tensor

    __slots__ = ["elem"]

    @staticmethod
    def __new__(cls, elem):
        # [FUNCTION DOC] HELPER: Creates the wrapper subclass instance mirroring the metadata of
        # the given tensor (size, dtype, layout, device, requires_grad, strides, storage
        # offset), stores that tensor as the inner element and returns the wrapper.
        out = torch.Tensor._make_wrapper_subclass(
            cls,
            elem.size(),
            dtype=elem.dtype,
            layout=elem.layout,
            device=elem.device,
            requires_grad=elem.requires_grad,
            strides=elem.stride(),
            storage_offset=elem.storage_offset(),
        )
        out.elem = elem
        return out

    def __tensor_flatten__(self):
        # [FUNCTION DOC] HELPER: Reports the single inner tensor attribute for flattening:
        # returns a pair of a one-element name list and no metadata.
        return ["elem"], None

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        # [FUNCTION DOC] HELPER: Rebuilds a wrapper from a mapping of inner tensors by wrapping
        # its elem entry; the metadata and size arguments are ignored.
        return _CowStateWrapperTensor(inner_tensors["elem"])

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Dispatch hook that always raises AssertionError with an
        # unexpected-dispatch message, regardless of the operation.
        raise AssertionError("unexpected dispatch")


@skipIfTorchDynamo("Registry tests don't need dynamo compilation")
class TestRegistry(TestCase):
    # [CLASS DOC] Graph-building tests for the registry module; skipped under Dynamo. setUp
    # snapshots and clears the registry's global tables and filter state, tearDown restores
    # them, so tests start from an empty registry and leave no trace.
    """Tests for the torch._native.registry module."""

    def setUp(self):
        # [FUNCTION DOC] Saves copies of the registry's library and graph dictionaries, the
        # three reverse lookup maps (copying their lists) and the filter-state sets (DSL names,
        # op symbols, dispatch keys), then clears all of them so the test starts empty. Stores
        # the registry module as an attribute.
        """Clean up registry state before each test."""
        super().setUp()
        self.registry = registry_module

        # Store original state for restoration
        self._original_libs = dict(self.registry._libs)
        self._original_graphs = dict(self.registry._graphs)
        self._original_dsl_name_to_lib_graph = {
            k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
        }
        self._original_dispatch_key_to_lib_graph = {
            k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
        }
        self._original_op_symbol_to_lib_graph = {
            k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
        }

        # Store original filter state
        self._original_filter_state = (
            set(self.registry._filter_state._dsl_names),
            set(self.registry._filter_state._op_symbols),
            set(self.registry._filter_state._dispatch_keys),
        )

        # Clear global state
        self.registry._libs.clear()
        self.registry._graphs.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()

        # Clear filter state to ensure clean start
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._dispatch_keys.clear()

    def tearDown(self):
        # [FUNCTION DOC] If setUp ran, restores the saved libs and graphs dictionaries, the
        # three reverse maps (with fresh list copies) and the filter-state sets, replacing
        # whatever the test left.
        """Restore original registry state after each test."""
        if hasattr(self, "registry"):
            # Restore original state
            self.registry._libs.clear()
            self.registry._libs.update(self._original_libs)

            self.registry._graphs.clear()
            self.registry._graphs.update(self._original_graphs)

            # Properly restore mapping dictionaries with new list instances
            self.registry._dsl_name_to_lib_graph.clear()
            for k, v in self._original_dsl_name_to_lib_graph.items():
                self.registry._dsl_name_to_lib_graph[k] = list(v)

            self.registry._dispatch_key_to_lib_graph.clear()
            for k, v in self._original_dispatch_key_to_lib_graph.items():
                self.registry._dispatch_key_to_lib_graph[k] = list(v)

            self.registry._op_symbol_to_lib_graph.clear()
            for k, v in self._original_op_symbol_to_lib_graph.items():
                self.registry._op_symbol_to_lib_graph[k] = list(v)

            # Restore filter state
            self.registry._filter_state._dsl_names.clear()
            self.registry._filter_state._op_symbols.clear()
            self.registry._filter_state._dispatch_keys.clear()
            self.registry._filter_state._dsl_names.update(
                self._original_filter_state[0]
            )
            self.registry._filter_state._op_symbols.update(
                self._original_filter_state[1]
            )
            self.registry._filter_state._dispatch_keys.update(
                self._original_filter_state[2]
            )

    # Keep essential existing tests
    def test_override_node_dataclass(self):
        # [FUNCTION DOC] Builds an override node from DSL name, op symbol, dispatch key, two
        # dummy functions and a node id; checks the fields are stored and that the unconditional
        # flag defaults to false and active to true.
        """Test _OverrideNode dataclass creation and defaults."""

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call and returns true.
            return True

        def impl_fn(x):
            # [FUNCTION DOC] HELPER: Dummy implementation that returns its argument.
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "add.Tensor", "CPU", cond_fn, impl_fn, "test_node"
        )
        self.assertEqual(node.dsl_name, "test_dsl")
        self.assertEqual(node.op_symbol, "add.Tensor")
        self.assertEqual(node.dispatch_key, "CPU")
        self.assertEqual(node.cond_fn, cond_fn)
        self.assertEqual(node.impl_fn, impl_fn)
        self.assertFalse(node.unconditional_override)
        self.assertTrue(node.active)

    @patch("torch.library.Library")
    def test_register_op_override_basic(self, mock_library_cls):
        # [FUNCTION DOC] With the library class patched out, registers one override for
        # add.Tensor on CPU and checks the graph for that key holds exactly one node carrying
        # the given DSL name, condition and implementation.
        """Test basic register_op_override functionality."""

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call.
            return True

        def impl_fn(x):
            # [FUNCTION DOC] HELPER: Dummy implementation returning its argument.
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        self.registry.register_op_override(
            "test_backend", "aten", "add.Tensor", "CPU", cond_fn, impl_fn
        )

        key = ("add.Tensor", "CPU")
        self.assertEqual(len(self.registry._graphs[key]), 1)
        node = self.registry._graphs[key][0]
        self.assertEqual(node.dsl_name, "test_backend")
        self.assertEqual(node.cond_fn, cond_fn)
        self.assertEqual(node.impl_fn, impl_fn)

    @patch("torch.library.Library")
    def test_deregister_op_overrides_basic(self, mock_library_cls):
        # [FUNCTION DOC] With the library class patched, registers an override for mul.Tensor on
        # CPU, checks the node is active, calls deregister with that DSL name disabled, and
        # checks the node in the graph is now inactive.
        """Test basic deregister_op_overrides functionality."""

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call.
            return True

        def impl_fn(x):
            # [FUNCTION DOC] HELPER: Dummy implementation returning its argument.
            return x

        mock_lib = MagicMock()
        mock_library_cls.return_value = mock_lib

        # Register first
        self.registry.register_op_override(
            "test_backend", "aten", "mul.Tensor", "CPU", cond_fn, impl_fn
        )

        key = ("mul.Tensor", "CPU")
        self.assertTrue(self.registry._graphs[key][0].active)

        # Then deregister
        self.registry.deregister_op_overrides(disable_dsl_names="test_backend")
        self.assertFalse(self.registry._graphs[key][0].active)

    # NEW FUNCTIONALITY TESTS - ONLY THE ESSENTIAL ONES

    def test_reorder_graphs_from_user_function_basic(self):
        # [FUNCTION DOC] Places three nodes (DSL names c, a, b) directly in the graph for one
        # key, runs reordering with an ordering function sorting nodes by DSL name and checks
        # the stored order becomes a, b, c.
        """Test basic graph reordering functionality."""
        # Set up test data
        key = ("test_reorder.Tensor", "CPU")

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call.
            return True

        def impl_fn(x):
            # [FUNCTION DOC] HELPER: Dummy implementation returning its argument.
            return x

        # Create nodes in specific order
        nodes = [
            self.registry._OverrideNode(
                "dsl_c", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_c"
            ),
            self.registry._OverrideNode(
                "dsl_a", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_a"
            ),
            self.registry._OverrideNode(
                "dsl_b", "test_reorder.Tensor", "CPU", cond_fn, impl_fn, "node_b"
            ),
        ]
        self.registry._graphs[key] = nodes

        # Define alphabetical ordering function
        def alphabetical_order(op_symbol, dispatch_key, graph):
            # [FUNCTION DOC] HELPER: User ordering function: ignores the op symbol and dispatch
            # key and returns the graph nodes sorted by DSL name.
            return sorted(graph, key=lambda n: n.dsl_name)

        # Apply reordering
        self.registry.reorder_graphs_from_user_function(alphabetical_order)

        # Verify alphabetical order
        reordered_graph = self.registry._graphs[key]
        actual_names = [node.dsl_name for node in reordered_graph]
        self.assertEqual(actual_names, ["dsl_a", "dsl_b", "dsl_c"])

    def test_reorder_graphs_from_user_function_error_handling(self):
        # [FUNCTION DOC] Places one node in the graph and runs reordering with a function that
        # raises ValueError. Checks exactly one warning is logged by the registry logger
        # containing the text about a failed graph transformation and that the stored graph is
        # unchanged (no exception escapes).
        """Test error handling in graph reordering."""
        # Set up test data
        key = ("test_error.Tensor", "CPU")

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call.
            return True

        def impl_fn(x):
            # [FUNCTION DOC] HELPER: Dummy implementation returning its argument.
            return x

        node = self.registry._OverrideNode(
            "test_dsl", "test_error.Tensor", "CPU", cond_fn, impl_fn, "test_node"
        )
        original_graph = [node]
        self.registry._graphs[key] = original_graph.copy()

        # Define failing ordering function
        def failing_order_fn(op_symbol, dispatch_key, graph):
            # [FUNCTION DOC] HELPER: Ordering function that always raises ValueError with a test
            # message.
            raise ValueError("Test exception")

        # Should handle the exception gracefully
        with self.assertLogs("torch._native.registry", level="WARNING") as log:
            self.registry.reorder_graphs_from_user_function(failing_order_fn)

        # Verify warning was logged and original graph preserved
        self.assertEqual(len(log.records), 1)
        self.assertIn("Graph transformation failed", log.records[0].getMessage())
        self.assertEqual(self.registry._graphs[key], original_graph)

    def test_get_user_ordering_fn_env_var_not_set(self):
        # [FUNCTION DOC] With the environment emptied, clears the cache of the user ordering
        # function lookup and checks it returns None.
        """Test behavior when environment variable is not set."""
        with patch.dict("os.environ", {}, clear=True):
            from torch._native import get_user_ordering_fn

            get_user_ordering_fn.cache_clear()
            result = get_user_ordering_fn()
            self.assertIsNone(result)

    def test_get_user_ordering_fn_invalid_path(self):
        # [FUNCTION DOC] Sets the user graph-order environment variable to a path that cannot be
        # imported, clears the lookup cache and checks the lookup raises ValueError whose
        # message says it could not resolve the path.
        """Test handling of invalid environment variable paths."""
        with patch.dict(
            "os.environ",
            {"TORCH_PYTHON_NATIVE_USER_GRAPH_ORDER_FN": "nonexistent.module.function"},
        ):
            from torch._native import get_user_ordering_fn

            get_user_ordering_fn.cache_clear()

            with self.assertRaises(ValueError) as cm:
                get_user_ordering_fn()
            self.assertIn("Could not resolve", str(cm.exception))

    def test_integration_reorder_and_register(self):
        # [FUNCTION DOC] Registers two overrides for one op and key from backends z then a,
        # checks the graph order matches registration order, then reorders alphabetically by DSL
        # name and checks the order becomes a then z.
        """Integration test: reorder then register functionality."""

        def cond_fn(x):
            # [FUNCTION DOC] HELPER: Dummy condition that accepts any call.
            return True

        def impl_fn1(x):
            # [FUNCTION DOC] HELPER: Dummy implementation that adds one to its argument.
            return x + 1

        def impl_fn2(x):
            # [FUNCTION DOC] HELPER: Dummy implementation that adds two to its argument.
            return x + 2

        # Register multiple overrides
        self.registry.register_op_override(
            "backend_z", "aten", "test.Tensor", "CPU", cond_fn, impl_fn1
        )
        self.registry.register_op_override(
            "backend_a", "aten", "test.Tensor", "CPU", cond_fn, impl_fn2
        )

        key = ("test.Tensor", "CPU")

        # Verify initial order
        initial_names = [node.dsl_name for node in self.registry._graphs[key]]
        self.assertEqual(initial_names, ["backend_z", "backend_a"])

        # Reorder alphabetically
        def alphabetical_order(op_symbol, dispatch_key, graph):
            # [FUNCTION DOC] HELPER: User ordering function returning the graph nodes sorted by
            # DSL name.
            return sorted(graph, key=lambda n: n.dsl_name)

        self.registry.reorder_graphs_from_user_function(alphabetical_order)

        # Verify reordered
        final_names = [node.dsl_name for node in self.registry._graphs[key]]
        self.assertEqual(final_names, ["backend_a", "backend_z"])

    def test_disallowed_dispatch_key_rejected(self):
        # [FUNCTION DOC] For each of the dispatch keys Meta, CompositeImplicitAutograd and
        # CompositeExplicitAutograd, checks register_op_override raises ValueError whose message
        # mentions the dispatch key argument.
        """Overrides installed at Meta / Composite* would loop through the
        fake kernel's redispatch — `register_op_override` must reject them.
        """

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Dummy condition accepting any arguments, returns true.
            return True

        def impl(*a, **k):
            # [FUNCTION DOC] HELPER: Dummy implementation accepting any arguments, returns None.
            return None

        for bad_key in (
            "Meta",
            "CompositeImplicitAutograd",
            "CompositeExplicitAutograd",
        ):
            with self.assertRaisesRegex(ValueError, "dispatch_key="):
                self.registry.register_op_override(
                    "test_dsl", "aten", "mul.Tensor", bad_key, cond, impl
                )

    def test_cond_none_without_unconditional_override_rejected(self):
        # [FUNCTION DOC] Checks registering with no condition and without the unconditional flag
        # raises ValueError saying the condition must be provided.
        """cond=None is only valid when unconditional_override=True."""

        def impl(*a, **k):
            # [FUNCTION DOC] HELPER: Dummy implementation accepting any arguments, returns None.
            return None

        with self.assertRaisesRegex(ValueError, "cond must be provided"):
            self.registry.register_op_override(
                "test_dsl", "aten", "mul.Tensor", "CPU", None, impl
            )


@skipIfTorchDynamo("Runtime registry tests exercise the dispatcher directly")
class TestRegistryRuntime(TestCase):
    # [CLASS DOC] End-to-end tests that install overrides on real aten ops through the
    # dispatcher; skipped under Dynamo. setUp saves registry tables and the node id counter and
    # clears the graphs and reverse maps; tearDown destroys installed aten override libraries
    # and restores the saved tables. The opaque native op definitions and fake kernels
    # intentionally persist for the process.
    """End-to-end runtime tests that exercise the real dispatcher.

    These tests register overrides on real aten ops and therefore must fully
    tear down any dispatcher-visible state in tearDown, otherwise a leaked
    router will poison other tests in the same process.
    """

    def setUp(self):
        # [FUNCTION DOC] Saves copies of the registry's graphs, libs, aten override libs, def
        # libs, defined native ops, the three reverse maps and the node id counter into a
        # dictionary, then clears the graphs and reverse maps so each test starts with no
        # registered overrides.
        super().setUp()
        self.registry = registry_module

        self._saved = {
            "graphs": dict(self.registry._graphs),
            "libs": dict(self.registry._libs),
            "aten_override_libs": dict(self.registry._aten_override_libs),
            "def_libs": dict(self.registry._def_libs),
            "defined_native_ops": set(self.registry._defined_native_ops),
            "dsl_map": {
                k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
            },
            "op_map": {
                k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
            },
            "dk_map": {
                k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
            },
            "node_id_counter": self.registry._node_id_counter,
        }

        self.registry._graphs.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()

    def tearDown(self):
        # [FUNCTION DOC] Destroys every aten override library the test installed, clears that
        # table and restores the saved one, then restores the saved graphs, libs and reverse
        # maps. Definitions of the native ops and fake kernels are deliberately left in place
        # since they cannot be undefined.
        # Destroy any aten overrides the test installed so the dispatcher
        # returns to its pre-test state.
        for lib in list(self.registry._aten_override_libs.values()):
            lib._destroy()
        self.registry._aten_override_libs.clear()

        # Restore aten override libs (none expected from other tests, but
        # be defensive).
        self.registry._aten_override_libs.update(self._saved["aten_override_libs"])

        # _native namespace DEF libraries and the ops defined on them persist
        # for the lifetime of the process (torch.library has no "undefine"),
        # so we deliberately do not destroy them — later tests will just
        # hit the `name in _defined_native_ops` short-circuit.
        # We also leave the fake kernels registered for the same reason.

        # Restore the rest.
        self.registry._graphs.clear()
        self.registry._graphs.update(self._saved["graphs"])
        self.registry._libs.clear()
        self.registry._libs.update(self._saved["libs"])
        self.registry._dsl_name_to_lib_graph.clear()
        for k, v in self._saved["dsl_map"].items():
            self.registry._dsl_name_to_lib_graph[k] = list(v)
        self.registry._op_symbol_to_lib_graph.clear()
        for k, v in self._saved["op_map"].items():
            self.registry._op_symbol_to_lib_graph[k] = list(v)
        self.registry._dispatch_key_to_lib_graph.clear()
        for k, v in self._saved["dk_map"].items():
            self.registry._dispatch_key_to_lib_graph[k] = list(v)

    def _install(self, op_symbol, dispatch_key):
        # [FUNCTION DOC] HELPER: Pushes the already registered graph for the given op symbol and
        # dispatch key through the real registration path so the override becomes active in the
        # dispatcher.
        """Build the graph then push it through the real registration path."""
        self.registry._register_overrides_from_graph(
            op_symbol,
            dispatch_key,
            self.registry._graphs[(op_symbol, dispatch_key)],
        )

    def test_cond_false_falls_through_to_native(self):
        # [FUNCTION DOC] Registers a mul.Tensor override on CPU whose condition returns false
        # and whose implementation records a flag, installs it, calls the overload on two
        # tensors and checks the native product is returned and the implementation was not
        # called.
        """cond=False must transparently invoke the captured native kernel."""
        sentinel_called = [False]

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that rejects every call.
            return False

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation that records that it was called (in the
            # shared flag list) and returns zeros like its first argument.
            sentinel_called[0] = True
            return torch.zeros_like(a)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        # Call the overload directly; `a * b` goes through overload
        # resolution which can dispatch to mul.Scalar for mixed args.
        out = torch.ops.aten.mul.Tensor(a, b)
        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        self.assertFalse(sentinel_called[0])

    def test_compile_session_flag_falls_through_without_recursion(self):
        # [FUNCTION DOC] Installs a mul.Tensor override with a rejecting condition and an
        # implementation that raises, then calls the op inside a compile-session context; checks
        # the native product is returned, i.e. the router falls back without recursing into
        # itself.
        """The eager router must not redispatch to its own aten override when
        compile-session state is set but Dynamo is not actively tracing it.
        """

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that rejects every call.
            return False

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation that raises AssertionError saying it must
            # not be called.
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        with torch.compiler._compile_session_context():
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))

    def test_dynamo_shortcut_preserves_cow_fallback(self):
        # [FUNCTION DOC] Installs a rejecting mul.Tensor override whose condition records that
        # it ran. Calls a function compiled with the eager backend on two lazily cloned (COW)
        # tensors and checks the result is the native product, the condition was evaluated and
        # both inputs are still COW afterwards.
        """COW inputs must keep using the eager fallback path under Dynamo."""
        sentinel_called = [False]

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that records it was called (in the shared flag
            # list) and rejects the call.
            sentinel_called[0] = True
            return False

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation that raises AssertionError saying it must
            # not be called.
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        @torch.compile(backend="eager")
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Compiled function that multiplies its two arguments with
            # the CPU device context active.
            with torch.device("cpu"):
                return torch.mul(a, b)

        a = torch._lazy_clone(torch.tensor([2.0, 3.0]))
        b = torch._lazy_clone(torch.tensor([4.0, 5.0]))
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))

        out = fn(a, b)

        self.assertTrue(torch.equal(out, torch.tensor([8.0, 15.0])))
        self.assertTrue(sentinel_called[0])
        self.assertTrue(torch._C._is_cow_tensor(a))
        self.assertTrue(torch._C._is_cow_tensor(b))

    def test_dynamo_guards_on_cow_state(self):
        # [FUNCTION DOC] Compiles a function that returns the COW state of its argument and
        # checks it reports false for a normal tensor and true for a lazily cloned one, i.e. the
        # compiled code is guarded on COW state.
        @torch.compile(backend="eager")
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function returning whether its argument is a COW
            # tensor.
            return torch._C._is_cow_tensor(a)

        self.assertFalse(fn(torch.tensor([2.0, 3.0])))
        self.assertTrue(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_on_sourceless_cow_state(self):
        # [FUNCTION DOC] Compiled function checks COW state of a view of its input (a value with
        # no source); expects false for a normal tensor and true for a lazy clone.
        @torch.compile(backend="eager")
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function returning whether a view of its argument
            # is a COW tensor.
            return torch._C._is_cow_tensor(a.view_as(a))

        self.assertFalse(fn(torch.tensor([2.0, 3.0])))
        self.assertTrue(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_on_mutated_cow_state(self):
        # [FUNCTION DOC] Compiled function mutates its COW input in place then reports its COW
        # state; expects false after the in-place write.
        @torch.compile(backend="eager")
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function that adds one in place to its argument
            # and returns whether it is a COW tensor.
            a.add_(1)
            return torch._C._is_cow_tensor(a)

        self.assertFalse(fn(torch._lazy_clone(torch.tensor([2.0, 3.0]))))

    def test_dynamo_graph_breaks_after_lazy_clone_changes_cow_state(self):
        # [FUNCTION DOC] With the inductor backend, a function that lazily clones its argument
        # and reports the COW state of the argument and the clone must give true for both, even
        # though the input was not COW before.
        @torch.compile(backend="inductor")
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function that lazily clones its argument and
            # returns the COW state of the argument and of the clone.
            clone = a._lazy_clone()
            return torch._C._is_cow_tensor(a), torch._C._is_cow_tensor(clone)

        self.assertEqual(
            fn(torch.tensor([2.0, 3.0])),
            (True, True),
        )

    def test_dynamo_graph_breaks_after_lazy_clone_function_changes_cow_state(self):
        # [FUNCTION DOC] Temporarily moves the lazy-clone function into the Dynamo
        # allowed-callable set (removing it from the disallowed set) so it enters the FX graph,
        # resets Dynamo, and checks a compiled function using it reports both tensors as COW. In
        # finally restores membership of all three trace-rule id sets to their saved state and
        # resets Dynamo again.
        fn_id = id(torch._lazy_clone)
        prior_trace_rule_state = (
            fn_id in trace_rules._allowed_callable_ids,
            fn_id in trace_rules._disallowed_callable_ids,
            fn_id in trace_rules._nonstrict_trace_callable_ids,
        )
        # torch._lazy_clone normally stays out of the FX graph. Force it into
        # the graph here to cover the defensive call_function scan.
        if fn_id in trace_rules._disallowed_callable_ids:
            trace_rules._disallowed_callable_ids.remove(fn_id)
        trace_rules._allowed_callable_ids.add(fn_id)
        try:
            torch._dynamo.reset()

            @torch.compile(backend="eager")
            def fn(a):
                # [FUNCTION DOC] HELPER: Compiled function that lazily clones its argument via
                # the torch function and returns the COW state of the argument and the clone.
                clone = torch._lazy_clone(a)
                return torch._C._is_cow_tensor(a), torch._C._is_cow_tensor(clone)

            self.assertEqual(
                fn(torch.tensor([2.0, 3.0])),
                (True, True),
            )
        finally:
            if fn_id in trace_rules._allowed_callable_ids:
                trace_rules._allowed_callable_ids.remove(fn_id)
            if fn_id in trace_rules._disallowed_callable_ids:
                trace_rules._disallowed_callable_ids.remove(fn_id)
            if fn_id in trace_rules._nonstrict_trace_callable_ids:
                trace_rules._nonstrict_trace_callable_ids.remove(fn_id)
            if prior_trace_rule_state[0]:
                trace_rules._allowed_callable_ids.add(fn_id)
            if prior_trace_rule_state[1]:
                trace_rules._disallowed_callable_ids.add(fn_id)
            if prior_trace_rule_state[2]:
                trace_rules._nonstrict_trace_callable_ids.add(fn_id)
            torch._dynamo.reset()

    def test_dynamo_graph_breaks_after_lazy_clone_view_changes_cow_state(self):
        # [FUNCTION DOC] Compiled function takes a view of its argument and lazily clones the
        # view; checks the argument, view and clone all report COW true.
        @torch.compile(backend="eager")
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function that views its argument flat, lazily
            # clones the view and returns the COW state of argument, view and clone.
            view = a.view(-1)
            clone = view._lazy_clone()
            return (
                torch._C._is_cow_tensor(a),
                torch._C._is_cow_tensor(view),
                torch._C._is_cow_tensor(clone),
            )

        self.assertEqual(
            fn(torch.tensor([2.0, 3.0])),
            (True, True, True),
        )

    def test_cow_guard_misses_on_fake_tensor(self):
        # [FUNCTION DOC] Imports the guard helper that compares COW state and checks it returns
        # false for a fake tensor created under a fake tensor mode, whether the expected state
        # is false or true.
        from torch._dynamo.guards import _cow_tensor_matches

        with FakeTensorMode():
            fake = torch.empty(2)

        self.assertFalse(_cow_tensor_matches(fake, False))
        self.assertFalse(_cow_tensor_matches(fake, True))

    def test_is_cow_tensor_rejects_python_tensor_subclasses(self):
        # [FUNCTION DOC] Wraps a tensor in the test wrapper subclass and checks the COW query
        # raises RuntimeError saying it is not defined for Python tensor subclasses.
        x = _CowStateWrapperTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            RuntimeError, "_is_cow_tensor is not defined for Python tensor subclasses"
        ):
            torch._C._is_cow_tensor(x)

    def test_dynamo_graph_breaks_on_subclass_cow_state(self):
        # [FUNCTION DOC] With full-graph compilation, passing a Python tensor subclass instance
        # to a function that queries COW state must raise an exception mentioning a COW check on
        # a Python tensor subclass.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function returning whether its argument is a COW
            # tensor.
            return torch._C._is_cow_tensor(a)

        x = _CowStateWrapperTensor(torch.tensor([2.0, 3.0]))
        with self.assertRaisesRegex(
            Exception, "COW tensor check on Python tensor subclass"
        ):
            fn(x)

    def test_dynamo_allows_previously_mutated_cow_state(self):
        # [FUNCTION DOC] With full-graph compilation, a tensor that was mutated in place earlier
        # is not COW; the compiled COW query must work without a graph break and return false.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a):
            # [FUNCTION DOC] HELPER: Compiled function returning whether its argument is a COW
            # tensor.
            return torch._C._is_cow_tensor(a)

        x = torch.tensor([2.0, 3.0])
        x.add_(1)
        self.assertFalse(fn(x))

    def test_strict_export_rejects_cow_state(self):
        # [FUNCTION DOC] Strict export of a module whose forward branches on the COW state of
        # its input, given a lazily cloned input, must raise an exception mentioning a COW check
        # during export.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward adds one to the input if it is a
            # COW tensor and two otherwise.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Branches on the COW state of the argument: adds one if
                # COW, else adds two; returns the sum tensor.
                if torch._C._is_cow_tensor(a):
                    return a + 1
                return a + 2

        with self.assertRaisesRegex(Exception, "COW tensor check during export"):
            torch.export.export(
                Mod(),
                (torch._lazy_clone(torch.tensor([2.0, 3.0])),),
                strict=True,
            )

    def test_non_strict_export_rejects_cow_state(self):
        # [FUNCTION DOC] Non-strict export of the same COW-branching module with a lazily cloned
        # input must raise an exception saying the COW query is not defined for Python tensor
        # subclasses.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Local module whose forward adds one to the input if it is a
            # COW tensor and two otherwise.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Branches on the COW state of the argument: adds one if
                # COW, else adds two; returns the sum tensor.
                if torch._C._is_cow_tensor(a):
                    return a + 1
                return a + 2

        with self.assertRaisesRegex(
            Exception, "_is_cow_tensor is not defined for Python tensor subclasses"
        ):
            torch.export.export(
                Mod(),
                (torch._lazy_clone(torch.tensor([2.0, 3.0])),),
                strict=False,
            )

    def test_cond_true_routes_to_impl(self):
        # [FUNCTION DOC] Installs a mul.Tensor override whose condition accepts and whose
        # implementation returns a tensor of 42s; checks the dispatched call returns that
        # implementation's result.
        """cond=True must route the call to the registered impl."""

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that accepts every call.
            return True

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation returning a tensor like the first argument
            # filled with 42.
            return torch.full_like(a, 42.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        out = torch.ops.aten.mul.Tensor(a, b)
        self.assertTrue(torch.equal(out, torch.tensor([42.0, 42.0])))

    def test_no_recursion_in_aten_backward_formula(self):
        # [FUNCTION DOC] Installs a bmm override with a rejecting condition, runs forward and
        # backward on random tensors needing grad, and checks the forward result equals the
        # plain product, gradients exist with the right shapes and are not all zero. This shows
        # the native fallback does not re-enter the router in the backward formula.
        """Backward of bmm calls bmm again; the fallback must bypass the router.

        If the router weren't bypassed, aten's bmm backward formula would
        re-enter it on every autograd step, either recursing forever or
        silently calling the override impl in the grad path.
        """

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that rejects every call so each call takes the
            # native fallback.
            # False so every call hits the fallback — this isolates the
            # "native kernel called via fallback must not re-enter the router"
            # property.
            return False

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation that raises AssertionError saying it must
            # not be called.
            raise AssertionError("impl should not be called when cond=False")

        self.registry.register_op_override("test_dsl", "aten", "bmm", "CPU", cond, impl)
        self._install("bmm", "CPU")

        a = torch.randn(2, 3, 4, requires_grad=True)
        b = torch.randn(2, 4, 5, requires_grad=True)
        expected = a.detach() @ b.detach()
        out = torch.bmm(a, b)
        self.assertEqual(out, expected)

        out.sum().backward()
        self.assertEqual(a.grad.shape, a.shape)
        self.assertEqual(b.grad.shape, b.shape)
        self.assertTrue((a.grad != 0).any())
        self.assertTrue((b.grad != 0).any())

    def test_deregister_reenable_roundtrip(self):
        # [FUNCTION DOC] Installs a mul.Tensor override returning sevens and checks it is live
        # and recorded in the aten override table. Deregisters the DSL and checks the native
        # product returns and the table entry is gone; re-enables and checks sevens again and
        # the entry is back.
        """deregister must tear down the aten router; reenable must reinstall."""

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that accepts every call.
            return True

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation returning a tensor like the first argument
            # filled with 7.
            return torch.full_like(a, 7.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        mul = torch.ops.aten.mul.Tensor

        # Override active.
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([7.0, 7.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Deregister → native kernel returns.
        self.registry.deregister_op_overrides(disable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([8.0, 15.0])))
        self.assertNotIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Reenable → override fires again.
        self.registry.reenable_op_overrides(enable_dsl_names="test_dsl")
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([7.0, 7.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

    def test_empty_graph_tears_down_router(self):
        # [FUNCTION DOC] Regression test. Installs a mul.Tensor override returning 99s, then
        # replaces the stored graph with an empty list and calls the cleanup-and-reregister
        # routine; checks the aten override table entry is removed and the native product is
        # returned again.
        """An empty graph passed to _cleanup_and_reregister_graph must still
        tear down the previously-installed aten router.

        Regression test: before the fix, the `if graph:` guard skipped
        _register_overrides_from_graph entirely, leaving the aten override
        live with a stale closure.
        """

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that accepts every call.
            return True

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation returning a tensor like the first argument
            # filled with 99.
            return torch.full_like(a, 99.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        mul = torch.ops.aten.mul.Tensor
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([99.0, 99.0])))
        self.assertIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)

        # Simulate a filter-out-everything transformation.
        self.registry._graphs[("mul.Tensor", "CPU")] = []
        self.registry._cleanup_and_reregister_graph(
            "mul.Tensor", "CPU", self.registry._graphs[("mul.Tensor", "CPU")]
        )

        self.assertNotIn(("mul.Tensor", "CPU"), self.registry._aten_override_libs)
        self.assertTrue(torch.equal(mul(a, b), torch.tensor([8.0, 15.0])))

    def test_fake_tensor_shape_inference(self):
        # [FUNCTION DOC] Installs a mul.Tensor override, then under a fake tensor mode
        # multiplies two empty 3 by 4 float32 tensors through the op and checks the result has
        # shape 3 by 4 and dtype float32, via the registered fake kernel.
        """FakeTensorMode must shape-infer through `_native::<id>` via the
        registered fake kernel (which redispatches to the aten meta).
        """

        def cond(*a, **k):
            # [FUNCTION DOC] HELPER: Condition that accepts every call.
            return True

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation returning a tensor like the first argument
            # filled with 1.
            return torch.full_like(a, 1.0)

        self.registry.register_op_override(
            "test_dsl", "aten", "mul.Tensor", "CPU", cond, impl
        )
        self._install("mul.Tensor", "CPU")

        with FakeTensorMode():
            a = torch.empty(3, 4, dtype=torch.float32)
            b = torch.empty(3, 4, dtype=torch.float32)
            out = torch.ops.aten.mul.Tensor(a, b)

        self.assertEqual(out.shape, torch.Size([3, 4]))
        self.assertEqual(out.dtype, torch.float32)

    def test_unconditional_override_cond_none(self):
        # [FUNCTION DOC] Registers a mul.Tensor override with no condition but the unconditional
        # flag, installs it, and checks two calls (including one on zeros) both return the
        # implementation's fives and the implementation ran exactly twice.
        """`cond=None` + `unconditional_override=True` must substitute a
        trivially-true predicate so the impl fires on every call.
        """
        call_count = [0]

        def impl(a, b):
            # [FUNCTION DOC] HELPER: Implementation that increments the shared call counter and
            # returns a tensor like the first argument filled with 5.
            call_count[0] += 1
            return torch.full_like(a, 5.0)

        self.registry.register_op_override(
            "test_dsl",
            "aten",
            "mul.Tensor",
            "CPU",
            None,
            impl,
            unconditional_override=True,
        )
        self._install("mul.Tensor", "CPU")

        a = torch.tensor([2.0, 3.0])
        b = torch.tensor([4.0, 5.0])
        self.assertTrue(
            torch.equal(torch.ops.aten.mul.Tensor(a, b), torch.tensor([5.0, 5.0]))
        )
        self.assertTrue(
            torch.equal(
                torch.ops.aten.mul.Tensor(torch.tensor([0.0]), torch.tensor([0.0])),
                torch.tensor([5.0]),
            )
        )
        self.assertEqual(call_count[0], 2)


if __name__ == "__main__":
    run_tests()
