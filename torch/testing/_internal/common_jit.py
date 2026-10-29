# mypy: ignore-errors
# [MODULE DOC] Shared helpers for JIT tests: reference-vs-JIT comparison, TorchScript
# export/import round trips, autodiff node checks and shape analysis checks. Main contents:
# check_output_types, nn_functional_single_grad (set of test names that only get a first
# derivative check), check_against_reference and the JitCommonTestCase base class. Import side
# effects: only imports (including torch.jit submodules); enable_profiling_mode is re-exported
# for other tests. No global state is changed at import time.

# Torch
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized

# Testing utils
from torch.testing._internal.common_dtype import floating_and_complex_types_and
from torch.testing._internal.common_utils import TestCase, \
    freeze_rng_state, TemporaryFileName, enable_profiling_mode_for_profiling_tests, is_iterable_of_tensors
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401

# Standard library
from itertools import chain
from torch._C import TensorType

import io

def check_output_types(self, func, ref_outputs, args, kwargs):
    # [FUNCTION DOC] Asserts the JIT-recorded output type of func matches the actual outputs.
    # Reads the last_graph attribute of func, requires the graph to have exactly one output, and
    # asks the runtime to assert that ref_outputs is an instance of that output type. Args: self
    # is the test case; args and kwargs are unused. Raises an assertion error on mismatch or
    # when the output count is not one.
    # KNOWN ISSUE: if func has no last_graph the graph is None and an attribute error is raised.
    graph = getattr(func, 'last_graph', None)
    types = [o.type() for o in graph.outputs()]
    self.assertTrue(len(types) == 1)
    t = types[0]
    torch._C._jit_assert_is_instance(ref_outputs, t)

# Test names in this set are only checked for a single derivative
nn_functional_single_grad = frozenset('test_nn_' + name for name in [
    'pdist',
    'multilabel_margin_loss',
    'max_unpool3d',
    'multi_margin_loss',
    'binary_cross_entropy',
    'binary_cross_entropy_size_average',
    'ctc_loss',
    'grid_sample',
])

def check_against_reference(self, func, reference_func, output_func, args, kwargs=None,
                            allow_unused=True, check_types=True, no_grad=False, no_gradgrad=False):
    # [FUNCTION DOC] Verifies func matches reference_func in outputs, grads and double
    # grads. Runs both on cloned inputs without grad and compares outputs, then (optionally)
    # checks the output types. Unless no_grad is set, compares first-order gradients of a
    # weighted sum of outputs, and unless no_gradgrad is set (or the test name is in
    # nn_functional_single_grad) also compares second-order gradients with loosened
    # tolerances. self must provide runAndSaveRNG and assertEqual. Raises assertion errors
    # on mismatches. Returns nothing.
    """Verifies a function performs identically to some reference implementation.

    Commonly, this is used to verify that a JIT implementation
    (output_func) matches the behavior of the eager implementation
    (reference_func).
    """
    kwargs = kwargs if kwargs else {}

    def allSum(vs):
        # [FUNCTION DOC] HELPER: Builds a scalar loss from a tensor or sequence of tensors. Sums
        # each floating or complex tensor weighted by its position plus one (absolute value of
        # the sum for complex tensors); skips None entries and non-floating dtypes.
        if isinstance(vs, torch.Tensor):
            vs = (vs,)
        return sum((i + 1) * v.sum().abs() if v.dtype.is_complex else (i + 1) * v.sum()
                   for i, v in enumerate(vs)
                   if v is not None and v.dtype in floating_and_complex_types_and(torch.half, torch.bfloat16))

    def clone_tensor(t, preserve_requires_grad):
        # [FUNCTION DOC] HELPER: Returns a detached clone of t, with requires_grad set only if
        # preserve_requires_grad is true and t required grad.
        require_grad = preserve_requires_grad and t.requires_grad
        return t.detach().clone().requires_grad_(require_grad)

    def clone_inputs(preserve_requires_grad: bool):
        # [FUNCTION DOC] HELPER: Clones the enclosing args list; tensors and iterables of
        # tensors are cloned (optionally preserving requires_grad), other values are passed
        # through unchanged.
        inputs: list[torch.Tensor | list[torch.Tensor]] = []

        for arg in args:
            if isinstance(arg, torch.Tensor):
                inputs.append(clone_tensor(arg, preserve_requires_grad))
            elif is_iterable_of_tensors(arg):
                inputs.append([clone_tensor(t, preserve_requires_grad) for t in arg])
            else:
                inputs.append(arg)

        return inputs

    # Returns tensors in args that requires_grad, including tensors in TensorList args
    def get_recording_tensors(args):
        # [FUNCTION DOC] HELPER: Collects tensors from args (including inside tensor lists) that
        # require grad, in order, for use as autograd.grad inputs.
        recording_tensors: list[torch.Tensor] = []

        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.requires_grad:
                recording_tensors.append(arg)
            elif is_iterable_of_tensors(arg):
                recording_tensors.extend(filter(lambda t: t.requires_grad, arg))

        return recording_tensors

    # test no gradients case
    nograd_inputs = clone_inputs(preserve_requires_grad=False)
    outputs = self.runAndSaveRNG(reference_func, nograd_inputs, kwargs)
    with enable_profiling_mode_for_profiling_tests():
        outputs_test = self.runAndSaveRNG(func, nograd_inputs, kwargs)
    self.assertEqual(outputs, outputs_test)

    if check_types:
        check_output_types(self, func, outputs_test, nograd_inputs, kwargs)

    if no_grad:
        # skip grad tests
        return

    with enable_profiling_mode_for_profiling_tests():
        # test single grad case
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        grads = torch.autograd.grad(allSum(outputs), recording_tensors,
                                    allow_unused=allow_unused)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        grads_test = torch.autograd.grad(allSum(outputs_test), recording_tensors,
                                         allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        # test the grad grad case
        if self._testMethodName in nn_functional_single_grad or no_gradgrad:
            return

        outputs = output_func(self.runAndSaveRNG(reference_func, recording_inputs, kwargs))
        l1 = allSum(outputs)
        grads = torch.autograd.grad(l1, recording_tensors, create_graph=True,
                                    allow_unused=allow_unused)

        l2 = (allSum(grads) * l1)
        grads2 = torch.autograd.grad(l2, recording_tensors, allow_unused=allow_unused)
        recording_inputs = clone_inputs(preserve_requires_grad=True)
        recording_tensors = get_recording_tensors(recording_inputs)
        outputs_test = output_func(self.runAndSaveRNG(func, recording_inputs, kwargs))
        l1_test = allSum(outputs_test)
        grads_test = torch.autograd.grad(
            l1_test, recording_tensors, create_graph=True, allow_unused=allow_unused)

        l2_test = (allSum(grads_test) * l1_test)
        grads2_test = torch.autograd.grad(l2_test, recording_tensors, allow_unused=allow_unused)

        self.assertEqual(outputs, outputs_test)
        self.assertEqual(grads, grads_test)
        for g2, g2_test in zip(grads2, grads2_test, strict=True):
            if g2 is None and g2_test is None:
                continue
            self.assertEqual(g2, g2_test, atol=5e-4, rtol=1e-4)

class JitCommonTestCase(TestCase):
    # [CLASS DOC] Base test case with JIT utilities: building functions from graphs,
    # export/import round trips through torch.jit.save and load, RNG-preserving runs, autodiff
    # node assertions and shape analysis checks. Stateless; all behavior is in methods called by
    # JIT test classes.
    def createFunctionFromGraph(self, trace):
        # [FUNCTION DOC] Returns a runtime function named forward built from trace, which is
        # either a Graph or an object with a graph method (such as a traced module).
        graph = trace if isinstance(trace, torch._C.Graph) else trace.graph()
        return torch._C._create_function_from_graph("forward", graph)

    def assertExportImport(self, trace, inputs):
        # [FUNCTION DOC] Builds a function from the trace graph and runs the export/import round
        # trip check (assertExportImportModule) with the given inputs.
        m = self.createFunctionFromGraph(trace)
        self.assertExportImportModule(m, inputs)

    def assertExportImportModule(self, m, inputs):
        # [FUNCTION DOC] Saves and reloads module m, runs original and reloaded on inputs with
        # the same RNG state, and asserts equal results. Raises an assertion error with a
        # message on a diff.
        m_import = self.getExportImportCopy(m)
        a = self.runAndSaveRNG(m, inputs)
        b = self.runAndSaveRNG(m_import, inputs)
        self.assertEqual(a, b, "Results of original model and "
                               "exported/imported version of model differed")

    def runAndSaveRNG(self, func, inputs, kwargs=None):
        # [FUNCTION DOC] Calls func with inputs and kwargs under freeze_rng_state, so the RNG
        # state is restored afterwards and repeated calls see the same random numbers. Returns
        # func results.
        kwargs = kwargs if kwargs else {}
        with freeze_rng_state():
            results = func(*inputs, **kwargs)
        return results

    def getExportImportCopy(self, m, also_test_file=True, map_location=None):
        # [FUNCTION DOC] Returns a copy of m obtained by torch.jit.save into an in-memory buffer
        # and torch.jit.load with map_location. If also_test_file is true, additionally saves
        # the imported module to a temporary file and loads it back, returning that copy. Temp
        # file is removed.
        buffer = io.BytesIO()
        torch.jit.save(m, buffer)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)

        if not also_test_file:
            return imported

        with TemporaryFileName() as fname:
            torch.jit.save(imported, fname)
            return torch.jit.load(fname, map_location=map_location)

    def autoDiffErrorMessage(self, should_autodiff_node, nodes_not_in_diff_graph,
                             fusion_nodes_not_found, non_fusible_nodes_being_fused,
                             fusion_nodes_found, nodes_in_diff_graph):
        # [FUNCTION DOC] Builds the failure message for assertAutodiffNode. When autodiff was
        # expected, classifies missing nodes into: not in any DifferentiableGraph, wrongly
        # inside a fusion group, missing from fusion groups, or found only in the outer
        # differentiable graph, and appends advice for each non-empty class. Otherwise reports
        # nodes that unexpectedly were autodiffed. Args are lists of node kind names. Returns
        # the message string; no side effects.
        err_msg = "\nFailure in testing nodes' autodifferentiation. "
        if should_autodiff_node:
            err_msg += "One or more nodes were expected to be autodiffed, " \
                "but were not found in specified fusible/nonfusible " \
                "DifferentiableGraph groups. \nSpecifically:"
            # The node is intended to appear in a differentiable graph but doesn't
            diff_nodes_missing = []
            # The node is intended to appear in a differentiable graph
            # outside of a fusion group but instead is in a fusion group
            diff_nodes_in_fusion = []
            # The node is intended to appear in a fusion group but doesn't
            fusion_nodes_missing = []
            # The node is intended to appear in a fusion group but instead
            # is just in an outer differentiable graph
            fusion_nodes_in_diff = []
            for node in nodes_not_in_diff_graph:
                if node in non_fusible_nodes_being_fused:
                    diff_nodes_in_fusion.append(node)
                else:
                    diff_nodes_missing.append(node)
            for node in fusion_nodes_not_found:
                if node in nodes_in_diff_graph:
                    fusion_nodes_in_diff.append(node)
                else:
                    fusion_nodes_missing.append(node)
            if len(diff_nodes_missing) > 0:
                err_msg += f"\n  {diff_nodes_missing} were not in one of the " \
                    "DifferentiableGraphs when they were expected to be. " \
                    "Did you intend for these nodes to be autodiffed? " \
                    "If not, remove them from the list of nonfusible nodes."
            if len(diff_nodes_in_fusion) > 0:
                err_msg += f"\n  {diff_nodes_in_fusion} were found in one of the FusionGroups " \
                    "when they were expected to be just in a DifferentiableGraph. If it was " \
                    "intended for these nodes to be in FusionGroups, reclassify these nodes as " \
                    "fusible nodes. If these nodes were not intended to be fused, your " \
                    "autodifferentiation logic might be wrong."
            if len(fusion_nodes_missing) > 0:
                err_msg += f"\n  {fusion_nodes_missing} were not in one of the FusionGroups " \
                    "of the DifferentiableGraphs when they were expected to be. " \
                    "They were also not found in an outer DifferentiableGraph. Did you " \
                    "intend for these nodes to be autodifferentiated? If not, you should " \
                    "remove these nodes from the test's fusible nodes. Otherwise your " \
                    "autodifferentiation logic might be wrong."
            if len(fusion_nodes_in_diff) > 0:
                err_msg += f"\n  {fusion_nodes_in_diff} were not in one of the FusionGroups " \
                    "of the DifferentiableGraphs when they were expected to be, " \
                    "instead they were found just in an outer DifferentiableGraph. " \
                    "Did you intend for these nodes to be fused? If not, you should " \
                    "move these nodes into the test's nonfusible nodes. Otherwise your " \
                    "autodifferentiation logic might be wrong."
        else:
            err_msg += "One or more nodes were not expected to be autodiffed " \
                "but were found in a DifferentiableGraph or in a FusionGroup " \
                "of a DifferentiableGraph. Did you intend for these nodes to be " \
                "autodiffed? If so, change this test to expect autodifferentiation. " \
                "\nSpecifically:"
            if len(fusion_nodes_found) > 0:
                err_msg += f"\n  {fusion_nodes_found} were not expected to be in " \
                    "one of the DifferentiableGraphs, but appeared in a FusionGroup " \
                    "of a DifferentiableGraph. "
            if len(nodes_in_diff_graph) > 0:
                err_msg += f"\n  {nodes_in_diff_graph} were not expected to " \
                    "be in one of the DifferentiableGraphs but were."
        return err_msg

    def assertAutodiffNode(self, graph, should_autodiff_node, nonfusible_nodes, fusible_nodes):
        # [FUNCTION DOC] Checks whether the named nodes were autodiffed in graph. Looks for
        # nodes inside DifferentiableGraph subgraphs (nonfusible_nodes) and inside FusionGroup
        # subgraphs nested in them (fusible_nodes). If should_autodiff_node is not None, asserts
        # it equals whether all of both lists were found, using autoDiffErrorMessage for the
        # message. If None, checks nothing.
        diff_nodes = graph.findAllNodes('prim::DifferentiableGraph')
        diff_subgraphs = [node.g('Subgraph') for node in diff_nodes]

        # Note: currently no tests have fusible_nodes
        fusion_nodes = list(chain.from_iterable([g.findAllNodes('prim::FusionGroup') for g in diff_subgraphs]))
        fusion_subgraphs = [node.g('Subgraph') for node in fusion_nodes]

        # For any non-fusible node, it must show up in one of the DifferentiableGraphs.
        nodes_in_diff_graph = []
        nodes_not_in_diff_graph = []
        non_fusible_nodes_being_fused = []
        for node in nonfusible_nodes:
            if any(g.findNode(node) is not None for g in diff_subgraphs):
                nodes_in_diff_graph.append(node)
            else:
                nodes_not_in_diff_graph.append(node)
            if any(g.findNode(node) is not None for g in fusion_subgraphs):
                non_fusible_nodes_being_fused.append(node)
        found_all_nonfusible_nodes = len(nodes_in_diff_graph) == len(nonfusible_nodes)

        # For any fusible node, it must show up in one of the FusionGroups in one of the DifferentiableGraphs.
        fusion_nodes_found = []
        fusion_nodes_not_found = []
        for node in fusible_nodes:
            if any(g.findNode(node) is not None for g in fusion_subgraphs):
                fusion_nodes_found.append(node)
            else:
                fusion_nodes_not_found.append(node)
        found_all_fusible_nodes = len(fusion_nodes_found) == len(fusible_nodes)

        if should_autodiff_node is not None:
            err_msg = self.autoDiffErrorMessage(should_autodiff_node,
                                                nodes_not_in_diff_graph,
                                                fusion_nodes_not_found,
                                                non_fusible_nodes_being_fused,
                                                fusion_nodes_found,
                                                nodes_in_diff_graph)
            self.assertEqual(should_autodiff_node,
                             found_all_nonfusible_nodes and found_all_fusible_nodes, err_msg)

    def checkShapeAnalysis(self, out_sizes: list[int] | list[list[int]],
                           traced_graph, assert_propagation, constant_prop=True):
        # [FUNCTION DOC] Checks shape analysis on traced_graph against expected out_sizes (one
        # size list, or one per tuple element). Runs twice, with symbolic-shape test mode on and
        # off: erases non-input shape info, optionally runs constant propagation, propagates
        # shapes, then asserts the actual size is a subtype of the inferred type and, if
        # assert_propagation, equals it exactly. Mutates traced_graph. Temporarily sets the
        # global symbolic shapes test mode and restores it at the end (not restored if an
        # assertion fails).
        # repropagte input shapes provided by tracing,
        prev_symbolic_shapes_test_enabled = torch._C._jit_symbolic_shapes_test_mode_enabled()
        for enable_test_mode in [True, False]:
            # here we are testing allowing/disallowing substituting in complete shapes as constants,
            # disallowing constants helps stress test partial eval and substitution pipeline
            torch._C._jit_set_symbolic_shapes_test_mode(enable_test_mode)
            torch._C._jit_erase_non_input_shape_information(traced_graph)
            if constant_prop:
                torch._C._jit_pass_constant_propagation(traced_graph)
            torch._C._jit_pass_propagate_shapes_on_graph(traced_graph)
            # Add sizes to default tensor type to avoid checking something out of scope
            # and difficulties with tracer leaving in other parts of tensor type
            output = next(traced_graph.outputs()).type()

            def test_type(type, actual_size):
                # [FUNCTION DOC] HELPER: Compares inferred symbolic sizes of type to
                # actual_size. Asserts actual_size is a subtype of the inferred type; if
                # assert_propagation is set, also asserts the inferred sizes equal actual_size.
                sizes = type.symbolic_sizes()
                out_type = TensorType.get().with_sizes(sizes)
                actual_type = TensorType.get().with_sizes(actual_size)

                # always check actual shape is a subtype of the output
                self.assertTrue(actual_type.isSubtypeOf(out_type))

                # and then if assertion flag is provided, check shape analysis
                # is successful
                if assert_propagation:
                    self.assertEqual(out_type.sizes(), actual_size)

            if output.isSubtypeOf(torch._C.TensorType.get()):
                test_type(output, out_sizes)
            else:
                tuple_elements = output.elements()
                for i in range(len(tuple_elements)):
                    test_type(tuple_elements[i], out_sizes[i])

        torch._C._jit_set_symbolic_shapes_test_mode(prev_symbolic_shapes_test_enabled)
