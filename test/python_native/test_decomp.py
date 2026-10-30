# Owner(s): ["module: dsl-native-ops"]
# [MODULE DOC] Tests for native_decomp_table and the register/deregister/reenable syncing of
# decomposition overrides in torch._native.registry. One TestCase snapshots and restores the
# registry module-level state around each test so overrides registered at import time are
# not disturbed; tests cover the dict-level table contents and export-time routing of the
# add operator to a generated _native op. Includes two trivial predicates and an impl factory.

"""
Tests for native_decomp_table and related compile/export decomposition
machinery exposed by torch._native.registry.

These tests isolate themselves from any overrides that got registered at
import time (e.g. bmm_outer_product) by saving/restoring all relevant
registry-module-level state in setUp / tearDown.
"""

import uuid

import torch
import torch._native.registry as registry_module
from torch._native.registry import native_decomp_table, register_op_override
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


# Common predicates/impls used across several tests.
def _always_true(*a, **kw):
    # [FUNCTION DOC] HELPER: Condition predicate accepting any arguments; always returns
    # True.
    return True


def _always_false(*a, **kw):
    # [FUNCTION DOC] HELPER: Condition predicate accepting any arguments; always returns
    # False.
    return False


def _make_fill_impl(value):
    # [FUNCTION DOC] HELPER: Factory for override implementations. Returns a function that
    # takes a tensor (extra arguments ignored) and returns a new tensor of the same shape and
    # dtype as the first argument filled with the given value.
    """Build an impl that returns a fresh tensor filled with `value`."""

    def impl(a, *_args, **_kwargs):
        # [FUNCTION DOC] HELPER: The produced implementation; creates an empty tensor like the
        # first argument and fills it in place with the captured value, which is returned.
        return torch.empty_like(a).fill_(value)

    return impl


@skipIfTorchDynamo("Decomp tests don't need dynamo compilation")
class TestNativeDecompTable(TestCase):
    # [CLASS DOC] Test case for the native decomposition table. Each test gets a unique DSL
    # name (dsl_name) and the registry module (registry) with its state snapshotted in setUp
    # and restored in tearDown. Helpers register overrides and inspect exported graph targets;
    # skipped under TorchDynamo.
    """Tests for native_decomp_table() and register/deregister syncing."""

    def setUp(self):
        # [FUNCTION DOC] Save registry state and isolate the test. Stores the registry module,
        # a fresh random DSL name, and a snapshot (graphs, decomposition overrides, the three
        # name-to-library-graph mappings as copied lists, and the filter state sets). Then
        # destroys live libraries and clears all those registry containers. Mutates global
        # registry state; the defined native ops and definition libraries are left alone.
        self.registry = registry_module
        # Per-test unique DSL name so generated `_native::<node_id>` ops
        # don't collide across tests. `_defined_native_ops` and the
        # underlying torch dispatcher registration are process-wide and
        # cannot be un-defined; instead, each test uses a fresh name.
        self.dsl_name = f"test_{uuid.uuid4().hex[:8]}"

        # Snapshot registry state that registrations / deregistrations
        # touch. `_graphs` is the source of truth; `_aten_override_libs`
        # and `_libs` are derived (materialized dispatcher registrations)
        # and get rebuilt from `_graphs` on tearDown rather than restored
        # directly -- restoring destroyed Library objects would leave the
        # dispatcher without live kernels. We deliberately do NOT touch
        # `_defined_native_ops` or `_def_libs`; their contents mirror
        # permanent dispatcher state.
        self._snapshot = {
            "graphs": dict(self.registry._graphs),
            "decomp_overrides": dict(self.registry._native_decomp_overrides),
            "dsl_name_to_lib_graph": {
                k: list(v) for k, v in self.registry._dsl_name_to_lib_graph.items()
            },
            "dispatch_key_to_lib_graph": {
                k: list(v) for k, v in self.registry._dispatch_key_to_lib_graph.items()
            },
            "op_symbol_to_lib_graph": {
                k: list(v) for k, v in self.registry._op_symbol_to_lib_graph.items()
            },
            "filter_state": (
                set(self.registry._filter_state._dsl_names),
                set(self.registry._filter_state._op_symbols),
                set(self.registry._filter_state._dispatch_keys),
            ),
        }

        # Destroy live aten overrides and _native IMPL libs so the test's
        # own registrations see pristine aten kernels. These are rebuilt
        # from the snapshotted `_graphs` on tearDown.
        self._destroy_live_libs()

        # Clear dicts that won't corrupt process-wide state.
        self.registry._graphs.clear()
        self.registry._native_decomp_overrides.clear()
        self.registry._dsl_name_to_lib_graph.clear()
        self.registry._dispatch_key_to_lib_graph.clear()
        self.registry._op_symbol_to_lib_graph.clear()
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._dispatch_keys.clear()

    def _destroy_live_libs(self):
        # [FUNCTION DOC] HELPER: Destroy every library in the aten override and _native
        # implementation library mappings (destroying is the only way to unregister) and clear
        # both mappings.
        """Tear down every live aten override + _native IMPL library and
        clear their dicts. The C++ dispatcher has no per-kernel removal,
        so Library._destroy() is the only way to unregister."""
        for lib in list(self.registry._aten_override_libs.values()):
            lib._destroy()
        self.registry._aten_override_libs.clear()
        for lib in list(self.registry._libs.values()):
            lib._destroy()
        self.registry._libs.clear()

    def tearDown(self):
        # [FUNCTION DOC] Restore registry state after a test. Returns early if setUp did not
        # set the registry attribute. Otherwise destroys test-installed libraries, clears and
        # refills the graphs, decomposition overrides, name mappings and filter state sets
        # from the snapshot, then re-registers all overrides so dispatcher kernels match.
        if not hasattr(self, "registry"):
            return
        # Destroy whatever the test installed, then rebuild libs from the
        # snapshotted graph state so other tests (and the rest of the
        # process) see live dispatcher kernels matching the saved graphs.
        self._destroy_live_libs()

        self.registry._graphs.clear()
        self.registry._graphs.update(self._snapshot["graphs"])
        self.registry._native_decomp_overrides.clear()
        self.registry._native_decomp_overrides.update(
            self._snapshot["decomp_overrides"]
        )
        self.registry._dsl_name_to_lib_graph.clear()
        for k, v in self._snapshot["dsl_name_to_lib_graph"].items():
            self.registry._dsl_name_to_lib_graph[k] = list(v)
        self.registry._dispatch_key_to_lib_graph.clear()
        for k, v in self._snapshot["dispatch_key_to_lib_graph"].items():
            self.registry._dispatch_key_to_lib_graph[k] = list(v)
        self.registry._op_symbol_to_lib_graph.clear()
        for k, v in self._snapshot["op_symbol_to_lib_graph"].items():
            self.registry._op_symbol_to_lib_graph[k] = list(v)
        fs = self._snapshot["filter_state"]
        self.registry._filter_state._dsl_names.clear()
        self.registry._filter_state._dsl_names.update(fs[0])
        self.registry._filter_state._op_symbols.clear()
        self.registry._filter_state._op_symbols.update(fs[1])
        self.registry._filter_state._dispatch_keys.clear()
        self.registry._filter_state._dispatch_keys.update(fs[2])

        # Rebuild live libs from the restored graphs.
        self.registry._register_all_overrides()

    # ------------------------------------------------------------------
    # Test helpers.
    # ------------------------------------------------------------------

    def _register(self, op_symbol, cond, impl, dispatch_key="CPU"):
        # [FUNCTION DOC] HELPER: Register an override for the given aten op symbol under this
        # test's DSL name with the given condition and implementation (default CPU dispatch
        # key), then materialize all registered overrides in the registry.
        """Register an aten-namespace override and materialize it."""
        register_op_override(self.dsl_name, "aten", op_symbol, dispatch_key, cond, impl)
        self.registry._register_all_overrides()

    def _export_add(self, cond, impl):
        # [FUNCTION DOC] HELPER: Register an override for the add Tensor op, export a small
        # module computing the sum of two random 3-element tensors, and return the exported
        # program decomposed with the native decomposition table.
        """Export a module that calls `x + y`; decompose with the native table."""
        self._register("add.Tensor", cond, impl)

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward adds its two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Forward of the exported module; returns the sum of
                # its two inputs.
                return x + y

        ep = torch.export.export(M(), (torch.randn(3), torch.randn(3)))
        return ep.run_decompositions(native_decomp_table())

    def _graph_targets(self, ep):
        # [FUNCTION DOC] HELPER: Return a list of strings, the target of every call_function
        # node of the exported program graph, in node order.
        """Return the str(target) of every call_function node in an ExportedProgram."""
        return [
            str(node.target)
            for node in ep.graph_module.graph.nodes
            if node.op == "call_function"
        ]

    def assertGraphRoutedToNative(self, ep):
        # [FUNCTION DOC] HELPER: Assert that at least one call_function target in the program
        # contains the text _native; the failure message lists targets and the graph code.
        targets = self._graph_targets(ep)
        self.assertTrue(
            any("_native" in t for t in targets),
            lambda msg: f"{msg}\nExpected a _native call in graph; got targets={targets}\n"
            f"{ep.graph_module.code}",
        )

    def assertGraphNotRoutedToNative(self, ep):
        # [FUNCTION DOC] HELPER: Assert that no call_function target in the program contains
        # the text _native; the failure message lists targets and the graph code.
        targets = self._graph_targets(ep)
        for t in targets:
            self.assertNotIn(
                "_native",
                t,
                lambda msg: f"{msg}\nUnexpected _native call; targets={targets}\n{ep.graph_module.code}",
            )

    # ------------------------------------------------------------------
    # Dict-level tests: fast, no export / FakeTensor required.
    # ------------------------------------------------------------------

    def test_table_default_includes_core_aten_and_overrides(self):
        # [FUNCTION DOC] Registers an add override and asserts the default table contains it
        # and has more than a hundred entries (core aten decompositions plus overrides).
        """Default table = core aten decomps + registered overrides."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))

        table = native_decomp_table()
        self.assertIn(torch.ops.aten.add.Tensor, table)
        # Also contains a lot of default aten decomps (exact count varies by
        # PyTorch version, but it's in the hundreds).
        self.assertGreater(len(table), 100)

    def test_table_override_wins_on_conflict(self):
        # [FUNCTION DOC] Registers an add override and asserts the table entry for add is the
        # same object as the stored router closure, i.e. the override wins over any default.
        """If an op has both a default decomp and a registered override,
        the override wins (merged last)."""
        self._register("add.Tensor", _always_true, _make_fill_impl(99))

        # The stored override is a router closure that wraps our impl -- not
        # our impl directly -- so we verify table identity against the
        # closure we queued.
        table = native_decomp_table()
        self.assertIs(
            table[torch.ops.aten.add.Tensor],
            self.registry._native_decomp_overrides[torch.ops.aten.add.Tensor],
        )

    def test_table_overrides_only(self):
        # [FUNCTION DOC] Registers overrides for add and mul and asserts the table built with
        # overrides_only has exactly those two entries.
        """overrides_only=True returns just the registered overrides."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self._register("mul.Tensor", _always_true, _make_fill_impl(2))

        table = native_decomp_table(overrides_only=True)
        self.assertEqual(len(table), 2)
        self.assertIn(torch.ops.aten.add.Tensor, table)
        self.assertIn(torch.ops.aten.mul.Tensor, table)

    def test_deregister_removes_decomp_entry(self):
        # [FUNCTION DOC] Registers an add override, deregisters this DSL name, and asserts the
        # add entry is gone from the overrides-only table (it was present before).
        """Deregistering an override drops it from the decomp table."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self.assertIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

        self.registry.deregister_op_overrides(disable_dsl_names=self.dsl_name)

        self.assertNotIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

    def test_reenable_restores_decomp_entry(self):
        # [FUNCTION DOC] Registers an add override, deregisters then re-enables this DSL name,
        # and asserts the add entry is absent after deregistering and present after re-enabling.
        """Reenabling a deregistered override puts it back in the table."""
        self._register("add.Tensor", _always_true, _make_fill_impl(1))
        self.registry.deregister_op_overrides(disable_dsl_names=self.dsl_name)
        self.assertNotIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

        self.registry.reenable_op_overrides(enable_dsl_names=self.dsl_name)

        self.assertIn(
            torch.ops.aten.add.Tensor, native_decomp_table(overrides_only=True)
        )

    # ------------------------------------------------------------------
    # End-to-end tests that exercise compile_router's tracing behavior.
    # ------------------------------------------------------------------

    def test_export_with_matching_cond_routes_to_native(self):
        # [FUNCTION DOC] Exports add with an always-true condition and asserts the decomposed
        # graph contains a _native call.
        """When cond matches, run_decompositions rewrites aten call to _native."""
        ep = self._export_add(_always_true, _make_fill_impl(7))
        self.assertGraphRoutedToNative(ep)

    def test_export_with_non_matching_cond_preserves_aten(self):
        # [FUNCTION DOC] Exports add with an always-false condition and asserts the decomposed
        # graph has no _native call.
        """When cond returns False, the decomp falls through -- graph keeps aten::<op>."""
        ep = self._export_add(_always_false, _make_fill_impl(-1))
        self.assertGraphNotRoutedToNative(ep)

    def test_export_cond_exception_treated_as_non_match(self):
        # [FUNCTION DOC] Exports add with a condition that raises during tracing and asserts
        # the error is treated as a non-match, so the graph has no _native call.
        """A cond that raises under FakeTensor tracing is swallowed (treated
        as non-match), so the graph falls through to the default lowering."""

        def raising_cond(*a, **kw):
            # [FUNCTION DOC] HELPER: Condition that ignores its arguments and always
            # raises RuntimeError.
            raise RuntimeError("intentional failure under tracing")

        ep = self._export_add(raising_cond, _make_fill_impl(-1))
        self.assertGraphNotRoutedToNative(ep)


if __name__ == "__main__":
    run_tests()
