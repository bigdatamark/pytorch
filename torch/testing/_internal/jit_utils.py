# mypy: ignore-errors

# [MODULE DOC] Shared helpers for TorchScript (JIT) tests: the JitTestCase base class with
# script/trace/module equivalence checks and graph assertions, context managers and decorators
# toggling JIT global flags (inlining, fusion, CPU fuser), and utilities for sample-variant
# collection and input cloning. Import-time side effects: computes the module constants
# RUN_CUDA, RUN_CUDA_MULTI_GPU, RUN_CUDA_HALF (and CUDA_VERSION when CUDA is on a non-HIP build).

# Torch
from torch.autograd import Variable
from torch.autograd.function import _nested_map
from torch.jit.annotations import BroadcastingList2, BroadcastingList3  # noqa: F401

from torch.onnx import OperatorExportTypes
import torch
import torch.cuda
import torch.jit
import torch.jit._logging
import torch.jit.frontend
import torch.jit.quantized
import zipfile
import functools

# Testing utils
from torch.testing import FileCheck
from torch.testing._internal.common_utils import IS_WINDOWS, \
    freeze_rng_state, enable_profiling_mode_for_profiling_tests, ProfilingMode, TEST_BAILOUTS, \
    is_iterable_of_tensors
from torch.testing._internal.common_jit import JitCommonTestCase
from torch.testing._internal.common_utils import enable_profiling_mode  # noqa: F401

# Standard library
from contextlib import contextmanager
from functools import reduce
from io import StringIO
from collections import defaultdict

import importlib.util
import inspect
import io
import math
import os
import pickle
import sys
import tempfile
import textwrap
from importlib.abc import Loader
from typing import Any

RUN_CUDA = torch.cuda.is_available()
RUN_CUDA_MULTI_GPU = RUN_CUDA and torch.cuda.device_count() > 1
RUN_CUDA_HALF = RUN_CUDA
# HIP supports half, no version check necessary
if torch.cuda.is_available() and not torch.version.hip:
    CUDA_VERSION = torch._C._cuda_getCompiledVersion()
    for d in range(torch.cuda.device_count()):
        major = torch.cuda.get_device_capability(d)[0]
        if (major < 6):
            RUN_CUDA_HALF = False

def execWrapper(code, glob, loc):
    # [FUNCTION DOC] HELPER: executes the source string code with the given global and local
    # namespaces; definitions land in loc. Kept as a function so exec syntax stays isolated.
    exec(code, glob, loc)

def do_input_map(fn, input):
    # [FUNCTION DOC] HELPER: applies fn to every tensor found in the nested input structure and
    # returns the same structure with the results; non-tensor leaves are kept as they are.
    return _nested_map(lambda t: isinstance(t, torch.Tensor), fn)(input)

def clear_class_registry():
    # [FUNCTION DOC] HELPER: resets global TorchScript class state: the C++ class registry, the
    # recursive-scripting concrete type store and the Python-side class state. Mutates globals.
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()

def get_execution_plan(graph_executor_state):
    # [FUNCTION DOC] Returns the single execution plan of a graph executor debug state. Raises a
    # runtime error if the state holds any number of plans other than one.
    execution_plans = list(graph_executor_state.execution_plans.values())
    num_plans = len(execution_plans)
    if num_plans != 1:
        raise RuntimeError('This test assumes this GraphExecutor should '
                           f'only have one execution plan, got: {num_plans}')
    return execution_plans[0]

class _AssertRaisesRegexWithHighlightContext:
    # [CLASS DOC] HELPER: context manager used by assertRaisesRegexWithHighlight. Stores the test
    # case, expected exception type, message regex and expected highlighted source text. On exit
    # it swallows the body's exception after checking its type and message, and optionally that
    # the message highlights the given source snippet.
    """
    A context manager that is useful for checking that error messages highlight
    the correct part of the source code.
    """

    def __init__(self, test_case, exception, regex, highlight):
        # [FUNCTION DOC] HELPER: stores the test case, exception type, regex and highlight text.
        self.test_case = test_case
        self.exception_type = exception
        self.regex = regex
        self.highlight = highlight

    def __enter__(self):
        # [FUNCTION DOC] HELPER: returns the context object itself.
        return self

    def __exit__(self, type, value, traceback):
        # [FUNCTION DOC] HELPER: re-raises the body's exception inside an assert-raises-regex
        # check (failing the test if the type or message does not match), then, if a highlight
        # was given, checks the message highlights it. Always returns true to suppress the error.
        # KNOWN ISSUE: if the body raised nothing, value is unset and the check fails by the
        # assert-raises machinery only; the later highlight check would use a missing value.
        with self.test_case.assertRaisesRegex(self.exception_type, self.regex):
            if type:
                raise value

        if self.highlight:
            FileCheck().check_source_highlighted(self.highlight).run(str(value))

        return True

FUSION_GROUP = "prim::TensorExprGroup"

class JitTestCase(JitCommonTestCase):
    # [CLASS DOC] Base test case for TorchScript tests. Installs JIT emit hooks in setUp that
    # check every compiled function or module survives a save/load round trip with identical
    # code, and clears hooks and the class registry in tearDown. Offers helpers to compare
    # script, trace and eager results and to assert on graph contents. The class flag
    # _restored_warnings makes the tracer warning filter install happen once per process.
    _do_cuda_memory_leak_check = True
    _restored_warnings = False

    class capture_stdout(list):
        # [CLASS DOC] Context manager (a list subclass) that redirects standard output to an
        # in-memory buffer; on exit the captured text is appended as the single list element.
        """
        Replace sys.stdout with a temporary StringIO
        """
        def __enter__(self):
            # [FUNCTION DOC] Saves the real stdout, installs a fresh buffer and returns self.
            self.sys_stdout = sys.stdout
            self.stringio = StringIO()
            sys.stdout = self.stringio
            return self

        def __exit__(self, *args):
            # [FUNCTION DOC] Appends the captured text to this list, drops the buffer and
            # restores the saved stdout. Never suppresses exceptions.
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stdout = self.sys_stdout

    class capture_stderr(list):
        # [CLASS DOC] Same as capture_stdout but for standard error.
        """
        Replace sys.stderr with a temporary StringIO
        """
        def __enter__(self):
            # [FUNCTION DOC] Saves the real stderr, installs a fresh buffer and returns self.
            self.sys_stderr = sys.stderr
            self.stringio = StringIO()
            sys.stderr = self.stringio
            return self

        def __exit__(self, *args):
            # [FUNCTION DOC] Appends the captured text to this list, drops the buffer and
            # restores the saved stderr. Never suppresses exceptions.
            self.append(str(self.stringio.getvalue()))
            del self.stringio
            sys.stderr = self.sys_stderr

    def setHooks(self):
        # [FUNCTION DOC] Registers this test's module and function emit hooks with the JIT.
        torch._C._jit_set_emit_hooks(self.emitModuleHook, self.emitFunctionHook)

    def clearHooks(self):
        # [FUNCTION DOC] Removes both JIT emit hooks.
        torch._C._jit_set_emit_hooks(None, None)

    def setUp(self):
        # [FUNCTION DOC] Runs base setup; the first time in the process re-applies the tracer
        # warning ignore filter (recorded in a class flag); then installs the emit hooks.
        super().setUp()
        # unittest overrides all warning filters and forces all of them to show up
        # after we install our own to silence those coming from inside PyTorch.
        # This will ensure that our filter still takes precedence.
        if not JitTestCase._restored_warnings:
            torch.jit.TracerWarning.ignore_lib_warnings()
            JitTestCase._restored_warnings = True
        self.setHooks()

    def tearDown(self):
        # [FUNCTION DOC] Runs base teardown, removes the emit hooks and clears the class registry.
        super().tearDown()
        # needs to be cleared because python might be unloaded before
        # the callback gets destructed
        self.clearHooks()
        clear_class_registry()

    def assertAllFused(self, graph, except_for=()):
        # [FUNCTION DOC] Asserts that the fast-path part of graph has exactly one block holding
        # exactly one fusion group node, and that every node in that block is an allowed kind
        # (constants, fusion group, bailout template, tuple construct, if, type and grad checks,
        # plus any kinds in except_for). Fails the test otherwise.

        # note this helper collects nodes on 'fast path' only
        # i.e. the true blocks of specialized checks
        def get_nodes_and_parents_recursively(block, kind, acc):
            # [FUNCTION DOC] HELPER: walks the block, recording nodes of the given kind into the
            # mapping from containing block to node list; descends into differentiable graph
            # subgraphs, the guarded branch of guard-style if nodes, and other nodes' blocks.
            for node in block.nodes():
                if node.kind() == kind:
                    acc[block].append(node)
                elif node.kind() == 'prim::DifferentiableGraph':
                    get_nodes_and_parents_recursively(node.g('Subgraph'), kind, acc)
                elif node.kind() == 'prim::If' and (node.inputs().__next__().node().kind() == 'aten::all' or
                                                    node.inputs().__next__().node().kind() == 'prim::TypeCheck' or
                                                    node.inputs().__next__().node().kind() == 'prim::RequiresGradCheck'):
                    get_nodes_and_parents_recursively(node.blocks().__next__(), kind, acc)
                else:
                    for inner_block in node.blocks():
                        get_nodes_and_parents_recursively(inner_block, kind, acc)

        allowed_nodes = {'prim::Constant', FUSION_GROUP, 'prim::BailoutTemplate',
                         'prim::TupleConstruct', 'prim::If', 'prim::TypeCheck', 'prim::RequiresGradCheck'} | set(except_for)

        fusion_groups : dict[torch._C.Block, list[torch._C.Node]] = defaultdict(list)
        get_nodes_and_parents_recursively(graph, FUSION_GROUP, fusion_groups)
        self.assertTrue(len(fusion_groups) == 1, f'got {graph}')
        (graph, fusion_nodes) = next(iter(fusion_groups.items()))
        # the block contains one FUSION_GROUP and the rest of nodes are `allowed_nodes`
        self.assertTrue(len(fusion_nodes) == 1, f'got {graph}')
        self.assertTrue(all(node.kind() in allowed_nodes for node in graph.nodes()),
                        f'got {graph}')

    def _isHookExceptionOk(self, e):
        # [FUNCTION DOC] HELPER: returns true if the exception message contains one of the known
        # export-failure phrases (unexportable Python function, closures), meaning the round
        # trip check should be skipped rather than failed.
        se = str(e)
        allowed = ("Could not export Python function",
                   "closures are not exportable")
        for a in allowed:
            if a in se:
                return True
        return False

    def _compared_saved_loaded(self, m):
        # [FUNCTION DOC] HELPER: round trip check for a scripted function or module m, with emit
        # hooks disabled meanwhile. Skips empty code or modules without methods and tolerates
        # known export errors. Otherwise saves m, loads it, saves the loaded copy, and asserts
        # the code files of both saves match line for line (and, for script modules, that value
        # tags match). Other runtime errors from saving are re-raised.
        def extract_files(buffer):
            # [FUNCTION DOC] HELPER: opens the zip archive in buffer, asserts entry names are
            # unique, and returns the source text of archived code files and the unpickled
            # debug files, as two lists.
            # crack open the zip format to get at the main module code
            with zipfile.ZipFile(buffer) as archive:
                # check that we have no duplicate names
                self.assertEqual(len(set(archive.namelist())), len(archive.namelist()))
                files = list(filter(lambda x: x.startswith('archive/code/'), archive.namelist()))
                # unwrap all the code files into strings
                code_files_str = filter(lambda x: x.endswith('.py'), files)
                code_files = []
                for f in code_files_str:
                    with archive.open(f) as stream:
                        code_files.append("".join([line.decode() for line in stream]))

                # unpickled all the debug files
                debug_files_str = filter(lambda f: f.endswith('.debug_pkl'), files)
                debug_files = []
                for f in debug_files_str:
                    with archive.open(f) as stream:
                        debug_files.append(pickle.load(stream))
                return code_files, debug_files

        # disable the hook while we parse code, otherwise we will re-enter the hook
        with torch._jit_internal._disable_emit_hooks():
            try:
                # short-circuit if this is an empty function or module
                if len(m.code) == 0:
                    return
                if isinstance(m, torch._C.ScriptModule):
                    if len(m._method_names()) == 0:
                        return

                # save the module to a buffer
                buffer = io.BytesIO()
                torch.jit.save(m, buffer)
                # copy the data in the buffer so we can restore it later. This
                # is because py2 and py3 have different semantics with zipfile
                # and it's easier to just work with a fresh copy each time.
                buffer_copy = buffer.getvalue()

                code_files, _debug_files = extract_files(buffer)

            except RuntimeError as e:
                if not self._isHookExceptionOk(e):
                    raise
                else:
                    return

            # import the model again (from the copy we made of the original)
            buffer2 = io.BytesIO(buffer_copy)
            imported = torch.jit.load(buffer2)

            # save it again
            saved_module_buffer_2 = io.BytesIO()
            torch.jit.save(imported, saved_module_buffer_2)

            saved_module_buffer_2.seek(0)
            code_files_2, _debug_files_2 = extract_files(saved_module_buffer_2)

            for a, b in zip(code_files, code_files_2, strict=True):
                self.assertMultiLineEqual(a, b)

            if isinstance(m, torch._C.ScriptModule):
                self.assertTrue(torch._C._ivalue_tags_match(m, imported._c))


    def emitFunctionHook(self, func):
        # [FUNCTION DOC] JIT hook called for each compiled function: skips lambdas and builtin
        # operator names (invalid for export), else runs the save/load comparison.
        # func has invalid names for export, skip the jitter check
        if func.name == "<lambda>" or "aten::" in func.name:
            return
        self._compared_saved_loaded(func)

    def emitModuleHook(self, module):
        # [FUNCTION DOC] JIT hook called for each compiled module: runs the save/load comparison.
        self._compared_saved_loaded(module)


    def getExportImportCopyWithPacking(self, m, also_test_file=True, map_location=None):
        # [FUNCTION DOC] Returns a copy of module m made by saving and loading it, calling the
        # pack method on submodules that have one before saving and unpack afterwards (on m and
        # the loaded copy). Uses an in-memory buffer; when also_test_file is true it additionally
        # saves and loads through a temporary file (removed afterwards, closed first for Windows)
        # and returns that result. map_location is passed to the loader.
        buffer = io.BytesIO()
        m.apply(lambda s: s._pack() if s._c._has_method('_pack') else None)
        torch.jit.save(m, buffer)
        m.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        buffer.seek(0)
        imported = torch.jit.load(buffer, map_location=map_location)
        imported.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)

        if not also_test_file:
            return imported

        # Ideally we would like to not have to manually delete the file, but NamedTemporaryFile
        # opens the file, and it cannot be opened multiple times in Windows. To support Windows,
        # close the file after creation and try to remove it manually
        with tempfile.NamedTemporaryFile(delete=False) as f:
            try:
                f.close()
                imported.save(f.name)
                result = torch.jit.load(f.name, map_location=map_location)
            finally:
                os.unlink(f.name)

        result.apply(lambda s: s._unpack() if s._c._has_method('_unpack') else None)
        return result

    def assertGraphContains(self, graph, kind, consider_subgraphs=False):
        # [FUNCTION DOC] Asserts the graph has at least one node of the given kind. With
        # consider_subgraphs the check counts textual occurrences in the printed graph (excluding
        # subgraph header mentions); otherwise it walks nodes and nested blocks.

        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count(f'with {kind}')
            self.assertTrue(count > 0)
            return

        def nodes(block):
            # [FUNCTION DOC] HELPER: returns the list of nodes of the wanted kind in the block
            # and, recursively, in all nested blocks.
            out = []
            for node in block.nodes():
                if node.kind() == kind:
                    out.append(node)
                for block in node.blocks():
                    out += nodes(block)
            return out

        out_nodes = nodes(graph)
        self.assertTrue(len(out_nodes) > 0)

    def assertGraphContainsExactly(self, graph, kind, num_kind_nodes, consider_subgraphs=False):
        # [FUNCTION DOC] Asserts the graph has exactly num_kind_nodes nodes of the given kind,
        # counted like assertGraphContains (text based when consider_subgraphs, else a walk of
        # nodes and nested blocks). Raises an assertion error showing the graph on mismatch.
        def perform_assert(graph, kind, actual, expected, consider_subgraphs):
            # [FUNCTION DOC] HELPER: returns if actual equals expected, else raises an assertion
            # error with the graph and counts.
            if actual == expected:
                return
            subgraph = 'including' if consider_subgraphs else 'excluding'
            raise AssertionError(
                f'{graph}\nError: graph contains {actual} {kind} nodes ({subgraph} subgraphs) but expected {expected}')

        if consider_subgraphs:
            strgraph = str(graph)
            count = strgraph.count(kind) - strgraph.count(f'with {kind}')
            perform_assert(graph, kind, count, num_kind_nodes,
                           consider_subgraphs)
            return

        def nodes(block):
            # [FUNCTION DOC] HELPER: returns the list of nodes of the wanted kind in the block
            # and, recursively, in all nested blocks.
            out = []
            for node in block.nodes():
                if node.kind() == kind:
                    out.append(node)
                for block in node.blocks():
                    out += nodes(block)
            return out

        out_nodes = nodes(graph)
        perform_assert(graph, kind, len(out_nodes), num_kind_nodes,
                       consider_subgraphs)

    def assertExpectedONNXGraph(self, g, *args, **kwargs):
        # [FUNCTION DOC] Optimizes the traced graph g for ONNX export and compares it against the
        # stored expected output via assertExpectedGraph; extra arguments are forwarded.
        g = torch.onnx._optimize_trace(g, operator_export_type=OperatorExportTypes.ONNX)
        self.assertExpectedGraph(g, *args, **kwargs)

    def assertExpectedGraph(self, trace, *args, **kwargs):
        # [FUNCTION DOC] Takes a graph (or an object with a graph), lints it, removes dead code,
        # canonicalizes it, lints again, and compares its string form with the stored expected
        # text via assertExpected; extra arguments are forwarded.
        if isinstance(trace, torch._C.Graph):
            graph = trace
        else:
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        torch._C._jit_pass_dce(graph)
        torch._C._jit_pass_lint(graph)
        graph = torch._C._jit_pass_canonicalize(graph)
        torch._C._jit_pass_lint(graph)
        self.assertExpected(str(graph), *args, **kwargs)

    def run_pass(self, name, trace):
        # [FUNCTION DOC] Runs the named JIT graph pass on a graph or traced object (linting before
        # and after). If the pass returns a new graph, that replaces the working graph; for a
        # traced object the final graph is set back on it. Returns the resulting graph.
        if isinstance(trace, torch._C.Graph):
            graph = trace
            set_graph = False
        else:
            set_graph = True
            graph = trace.graph()

        torch._C._jit_pass_lint(graph)
        result = getattr(torch._C, '_jit_pass_' + name)(graph)
        if result is not None and not isinstance(result, bool):
            graph = result
        torch._C._jit_pass_lint(graph)

        if set_graph:
            trace.set_graph(graph)
        return graph

    def get_frame_vars(self, frames_up):
        # [FUNCTION DOC] Returns a mapping merging the locals and globals (globals win on clash)
        # of the frame frames_up levels above the caller. Raises a runtime error if frame
        # inspection is unavailable or the stack is too shallow.
        frame = inspect.currentframe()
        if not frame:
            raise RuntimeError("failed to inspect frame")
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            if not frame:
                raise RuntimeError("failed to get frame")
            i += 1
        defined_vars: dict[str, Any] = {}
        defined_vars.update(frame.f_locals)
        defined_vars.update(frame.f_globals)
        return defined_vars

    def assertRaisesRegexWithHighlight(self, exception, regex, highlight):
        # [FUNCTION DOC] Returns a context manager asserting the body raises the exception type
        # with a message matching regex and highlighting the given source snippet.
        return _AssertRaisesRegexWithHighlightContext(self, exception, regex, highlight)

    def checkScriptRaisesRegex(self, script, inputs, exception, regex,
                               name=None, outputs=None, capture_output=False,
                               frames_up=1, profiling=ProfilingMode.PROFILING):
        # [FUNCTION DOC] Asserts that running the script (source string or Python function) with
        # inputs raises the exception matching regex in plain Python, in the string frontend, and
        # (for function inputs) in the AST frontend. For string sources, name selects the function.
        # KNOWN ISSUE: outputs, capture_output and profiling are accepted but unused.
        """
        Checks that a given function will throw the correct exception,
        when executed with normal python, the string frontend, and the
        AST frontend. Logic taken from `checkScript` (see comments there
        for details)
        """
        with enable_profiling_mode_for_profiling_tests():
            # Normal Python
            with self.assertRaisesRegex(exception, regex):
                if isinstance(script, str):
                    frame = self.get_frame_vars(frames_up)
                    the_locals: dict[str, Any] = {}
                    execWrapper(script, glob=frame, loc=the_locals)
                    frame.update(the_locals)

                    python_fn = frame[name]
                else:
                    python_fn = script

                python_fn(*inputs)

            # String frontend
            with self.assertRaisesRegex(exception, regex):
                if isinstance(script, str):
                    cu = torch.jit.CompilationUnit(script, _frames_up=frames_up)
                    string_frontend = getattr(cu, name)
                else:
                    source = textwrap.dedent(inspect.getsource(script))
                    cu = torch.jit.CompilationUnit(source, _frames_up=frames_up)
                    string_frontend = getattr(cu, script.__name__)

                string_frontend(*inputs)

            # Python AST frontend
            if not isinstance(script, str):
                with self.assertRaisesRegex(exception, regex):
                    ge = torch.jit.script(python_fn)
                    ge(*inputs)

    def checkBailouts(self, model, inputs, expected):
        # [FUNCTION DOC] For each bailout point in the model's single execution plan, requests that
        # bailout, runs the model on inputs and asserts the output equals expected.
        state = model.get_debug_state()
        plan = get_execution_plan(state)
        num_bailouts = plan.code.num_bailouts()
        for i in range(num_bailouts):
            plan.code.request_bailout(i)
            bailout_outputs = model(*inputs)
            self.assertEqual(bailout_outputs, expected)

    def checkScript(self,
                    script,
                    inputs,
                    name='func',
                    optimize=True,
                    inputs_requires_grad=False,
                    capture_output=False,
                    frames_up=1,
                    profiling=ProfilingMode.PROFILING,
                    atol=None,
                    rtol=None):
        # [FUNCTION DOC] Compiles script (source string or Python function) with TorchScript and
        # asserts it matches the Python version on inputs, within atol/rtol. For functions it
        # first recursively checks the string-frontend form. Runs a profiling pass, an extra one
        # if grads are involved, and an optimized pass; with capture_output also checks printed
        # stdout against the stored expectation (skipped on Windows); with bailout testing on,
        # checks bailouts. Returns the scripted function.
        # KNOWN ISSUE: with capture_output, the profiling run output is not compared to Python.
        """
        Checks that a given script generates the same output as the Python
        version using the given inputs.
        """
        with torch.jit.optimized_execution(optimize), enable_profiling_mode_for_profiling_tests():
            extra_profile_runs = any(isinstance(x, torch.Tensor) and x.requires_grad for x in inputs)
            if isinstance(script, str):
                # Compile the string to a Script function
                # with enable_profiling_mode():
                cu = torch.jit.CompilationUnit(script, _frames_up=frames_up)

                # Execute the Python function so we can run it later and get its
                # outputs

                frame = self.get_frame_vars(frames_up)
                the_locals: dict[str, Any] = {}
                execWrapper(script, glob=frame, loc=the_locals)
                frame.update(the_locals)

                python_fn = frame[name]
                scripted_fn = getattr(cu, name)
            else:

                # Check the string frontend first
                source = textwrap.dedent(inspect.getsource(script))
                self.checkScript(
                    source,
                    inputs,
                    script.__name__,
                    optimize=optimize,
                    inputs_requires_grad=inputs_requires_grad,
                    capture_output=capture_output,
                    profiling=profiling,
                    frames_up=2)

                # Continue checking the Python frontend
                scripted_fn = torch.jit.script(script, _frames_up=1)
                python_fn = script

            if inputs_requires_grad:
                recording_inputs = do_input_map(lambda t: t.detach().requires_grad_(), inputs)
            else:
                recording_inputs = inputs

            if capture_output:
                with self.capture_stdout() as script_stdout:
                    script_outputs = scripted_fn(*recording_inputs)
                with self.capture_stdout():
                    opt_script_outputs = scripted_fn(*recording_inputs)
                with self.capture_stdout():
                    python_outputs = python_fn(*inputs)
                if not IS_WINDOWS:
                    self.assertExpected(script_stdout[0], subname='stdout')
                self.assertEqual(python_outputs, opt_script_outputs, atol=atol, rtol=rtol)
            else:
                # profiling run
                script_outputs = scripted_fn(*recording_inputs)
                if inputs_requires_grad or extra_profile_runs:
                    opt_script_outputs = scripted_fn(*recording_inputs)
                # optimized run
                opt_script_outputs = scripted_fn(*recording_inputs)
                if TEST_BAILOUTS:
                    self.checkBailouts(scripted_fn, inputs, opt_script_outputs)
                python_outputs = python_fn(*inputs)
            self.assertEqual(python_outputs, script_outputs, atol=atol, rtol=rtol)
            self.assertEqual(script_outputs, opt_script_outputs, atol=atol, rtol=rtol)
            return scripted_fn

    def checkTrace(self, func, reference_tensors, input_tensors=None,
                   drop=None, allow_unused=False, verbose=False,
                   inputs_require_grads=True, check_tolerance=1e-5, export_import=True,
                   _force_outplace=False, grad_atol=None, grad_rtol=None):
        # [FUNCTION DOC] Traces func with input_tensors (default reference_tensors), optionally
        # round-trips it through export/import, then asserts outputs match eager execution without
        # grad, with first-order gradients, and with second-order (grad of grad) gradients, using
        # the given tolerances. drop removes trailing outputs from the loss; allow_unused permits
        # unused inputs. Returns the traced function.

        # TODO: check gradients for parameters, not just inputs
        def allSum(vs):
            # [FUNCTION DOC] HELPER: returns a weighted sum of the entries (after dropping the last
            # drop ones), weighting entry i by the log of i plus two; skips none entries.
            # drop allows us to remove some values from ever being used
            # to test unused outputs
            if drop is not None:
                vs = vs[:-drop]
            # we don't want all the grad for all the outputs to be the same
            # so we multiply each by a constant
            return sum(math.log(i + 2) * v.sum() for i, v in enumerate(vs) if v is not None)
        if input_tensors is None:
            input_tensors = reference_tensors

        def flatten_inputs(inputs):
            # [FUNCTION DOC] HELPER: returns a tuple of all tensors in the nested recording inputs.
            # KNOWN ISSUE: ignores its parameter and always reads the enclosing recording_inputs.
            def input_reduce(input, fn, acc):
                # [FUNCTION DOC] HELPER: recursively visits tensors in nested containers (mappings
                # by key), applying fn to each tensor and the accumulator; returns the accumulator.
                if isinstance(input, torch.Tensor):
                    fn(input, acc)
                elif isinstance(input, dict):
                    reduce(lambda acc, key: input_reduce(input[key], fn, acc), input, acc)
                else:
                    reduce(lambda acc, val: input_reduce(val, fn, acc), input, acc)
                return acc
            return tuple(input_reduce(recording_inputs, lambda t, acc: acc.append(t), []))

        nograd_inputs = reference_tensors
        if inputs_require_grads:
            recording_inputs = do_input_map(lambda t: t.clone().requires_grad_(), reference_tensors)
            flattened_recording_inputs = flatten_inputs(recording_inputs)
        else:
            recording_inputs = reference_tensors

        # `check_trace` is set to False because check_trace is run with @no_grad
        # Also, `checkTrace` already does all the checks
        # against python function
        ge = torch.jit.trace(func, input_tensors, check_tolerance=check_tolerance,
                             _force_outplace=_force_outplace, check_trace=False)

        if export_import:
            ge = self.getExportImportCopy(ge)

        if verbose:
            print(ge.graph)

        # test no gradients case
        outputs = func(*nograd_inputs)
        outputs_ge = ge(*nograd_inputs)
        self.assertEqual(outputs, outputs_ge)

        # test gradients case
        outputs = func(*recording_inputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(allSum(outputs), flattened_recording_inputs,
                                        allow_unused=allow_unused)

        outputs_ge = ge(*recording_inputs)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(allSum(outputs_ge), flattened_recording_inputs,
                                           allow_unused=allow_unused)
        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge, atol=grad_atol, rtol=grad_rtol)

        # test the grad grad case
        outputs = func(*recording_inputs)
        l1 = allSum(outputs)
        if inputs_require_grads:
            grads = torch.autograd.grad(l1, flattened_recording_inputs, create_graph=True,
                                        allow_unused=allow_unused)
        if inputs_require_grads:
            l2 = (allSum(grads) * l1)
            grads2 = torch.autograd.grad(l2, flattened_recording_inputs, allow_unused=allow_unused)

        if inputs_require_grads:
            recording_inputs = do_input_map(lambda t: Variable(t, requires_grad=True), reference_tensors)
            flattened_recording_inputs = flatten_inputs(recording_inputs)

        outputs_ge = ge(*recording_inputs)
        l1_ge = allSum(outputs_ge)
        if inputs_require_grads:
            grads_ge = torch.autograd.grad(
                l1_ge, flattened_recording_inputs, create_graph=True, allow_unused=allow_unused)

        if inputs_require_grads:
            l2_ge = (allSum(grads_ge) * l1_ge)
            grads2_ge = torch.autograd.grad(l2_ge, flattened_recording_inputs, allow_unused=allow_unused)

        self.assertEqual(outputs, outputs_ge)
        if inputs_require_grads:
            self.assertEqual(grads, grads_ge, atol=grad_atol, rtol=grad_rtol)
            for g2, g2_ge in zip(grads2, grads2_ge, strict=True):
                if g2 is None and g2_ge is None:
                    continue
                self.assertEqual(g2, g2_ge, atol=8e-4, rtol=8e-4)

        return ge

    def checkModule(self, nn_module, args):
        # [FUNCTION DOC] Scripts nn_module, runs eager and scripted versions on args with the
        # random state frozen for each, asserts equal outputs and that export/import of the
        # scripted module works. Returns the scripted module.
        """
        Check that a nn.Module's results in Script mode match eager and that it
        can be exported
        """
        sm = torch.jit.script(nn_module)

        with freeze_rng_state():
            eager_out = nn_module(*args)

        with freeze_rng_state():
            script_out = sm(*args)

        self.assertEqual(eager_out, script_out)
        self.assertExportImportModule(sm, args)

        return sm

class NoTracerWarnContextManager:
    # [CLASS DOC] Context manager that turns off the tracer state warning for its body and
    # restores the previous setting (kept in prev) on exit.
    def __enter__(self):
        # [FUNCTION DOC] Saves the current tracer warning flag and disables the warning.
        self.prev = torch._C._jit_get_tracer_state_warn()
        torch._C._jit_set_tracer_state_warn(False)

    def __exit__(self, *args):
        # [FUNCTION DOC] Restores the saved tracer warning flag.
        torch._C._jit_set_tracer_state_warn(self.prev)

@contextmanager
def inline_everything_mode(should_inline):
    # [FUNCTION DOC] Context manager setting the global JIT inline-everything mode to
    # should_inline for the body, restoring the previous value afterwards (even on error).
    old = torch._C._jit_get_inline_everything_mode()
    torch._C._jit_set_inline_everything_mode(should_inline)
    try:
        yield
    finally:
        torch._C._jit_set_inline_everything_mode(old)

@contextmanager
def set_fusion_group_inlining(inlining):
    # [FUNCTION DOC] Context manager setting the debug fusion-group inlining flag for the body and
    # restoring the old value afterwards (even on error).
    old = torch._C._debug_get_fusion_group_inlining()
    torch._C._debug_set_fusion_group_inlining(inlining)
    try:
        yield
    finally:
        torch._C._debug_set_fusion_group_inlining(old)

# note: not re-entrant, use unnested only
@contextmanager
def disable_autodiff_subgraph_inlining(enabled=True):
    # [FUNCTION DOC] Context manager that, when enabled is true, turns off autodiff subgraph
    # inlining for the body (the flag is set to the negation of enabled). On exit inlining is
    # always reset to on, not to the prior value, hence not re-entrant.
    torch._C._debug_set_autodiff_subgraph_inlining(not enabled)
    try:
        yield
    finally:
        torch._C._debug_set_autodiff_subgraph_inlining(True)

def _inline_everything(fn):
    # [FUNCTION DOC] HELPER: decorator running fn with inline-everything mode on.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: calls fn inside the mode.
        # KNOWN ISSUE: the return value of fn is discarded.
        with inline_everything_mode(True):
            fn(*args, **kwargs)
    return wrapper

# this exists for forward compatibility reasons temporarily.
# TODO(suo) remove
def _tmp_donotuse_dont_inline_everything(fn):
    # [FUNCTION DOC] HELPER: temporary decorator running fn with inline-everything mode off.
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: calls fn inside the mode.
        # KNOWN ISSUE: the return value of fn is discarded.
        with inline_everything_mode(False):
            fn(*args, **kwargs)
    return wrapper

# make it easy to quickly define/trace a function for these tests
def _trace(*args, **kwargs):
    # [FUNCTION DOC] HELPER: decorator factory; the positional arguments become the example
    # inputs and keyword arguments are tracer options. The decorator returns the traced function.
    def wrapper(func):
        # [FUNCTION DOC] HELPER: traces func with the captured example inputs.
        return torch.jit.trace(func, args, **kwargs)
    return wrapper


def enable_cpu_fuser(fn):
    # [FUNCTION DOC] Decorator running fn with CPU fusion overrides on (both legacy and current,
    # LLVM requirement off) and, afterwards, resetting them to off and LLVM-required.
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: sets the overrides, calls fn, and always resets them.
        # KNOWN ISSUE: the return value of fn is discarded; resets to fixed values, not prior ones.
        torch._C._jit_override_can_fuse_on_cpu_legacy(True)
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_set_te_must_use_llvm_cpu(False)
        try:
            fn(*args, **kwargs)
        finally:
            torch._C._jit_override_can_fuse_on_cpu_legacy(False)
            torch._C._jit_override_can_fuse_on_cpu(False)
            torch._C._jit_set_te_must_use_llvm_cpu(True)
    return wrapper


def enable_cpu_fuser_if(cond):
    # [FUNCTION DOC] Returns the enable_cpu_fuser decorator when cond is true, otherwise a
    # decorator that leaves behavior unchanged.
    if cond:
        return enable_cpu_fuser
    else:
        def noop_fuser(fn):
            # [FUNCTION DOC] HELPER: pass-through decorator.
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: calls fn and returns its result.
                return fn(*args, **kwargs)
            return wrapper
        return noop_fuser

def get_forward(c):
    # [FUNCTION DOC] HELPER: returns the forward method object of a compiled module.
    return c._get_method('forward')

def get_forward_graph(c):
    # [FUNCTION DOC] HELPER: returns the graph of the forward method of a compiled module.
    return c._get_method('forward').graph

def get_module_method(m, module, method):
    # [FUNCTION DOC] HELPER: returns the named method of the submodule called module on
    # script module m.
    return m._c.getattr(module)._get_method(method)

def attrs_with_prefix(module, prefix):
    # [FUNCTION DOC] HELPER: returns the names of the module's registered submodules that
    # start with prefix.
    return [x for x, _ in module._modules._c.items()
            if x.startswith(prefix)]

def warmup_backward(f, *args):
    # [FUNCTION DOC] Runs a backward computation three times to warm up the profiling executor.
    # With extra args they are the inputs for a gradient computation whose results are collected;
    # without, it calls backward on f retaining the graph. Returns the list of gradient results
    # (empty when no args). Side effect: accumulates gradients in the no-args case.
    profiling_count = 3
    results = []
    for _ in range(profiling_count):
        if len(args) > 0:
            r = torch.autograd.grad(f, *args)
            results.append(r)
        else:
            f.backward(retain_graph=True)

    return results

# TODO: Remove me once https://bugs.python.org/issue42666 is resolved
def make_global(*args):
    # [FUNCTION DOC] HELPER: sets each given class or function as an attribute, under its own name,
    # on its defining module so that TorchScript can resolve it as a global. Mutates modules.
    for arg in args:
        setattr(sys.modules[arg.__module__], arg.__name__, arg)

# Helper function to eval Python3 code without causing a syntax error for
# this file under py2
def _get_py3_code(code, fn_name):
    # [FUNCTION DOC] HELPER: writes the source text code to a script file in a temporary
    # directory, loads it as a module, and returns the attribute named fn_name. Raises an
    # assertion error if the module spec has no usable loader. The directory is deleted on return.
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = os.path.join(tmp_dir, 'script.py')
        with open(script_path, 'w') as f:
            f.write(code)
        spec = importlib.util.spec_from_file_location(fn_name, script_path)
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if not isinstance(loader, Loader):  # Assert type to meet MyPy requirement
            raise AssertionError(f"Expected loader to be Loader, got {type(loader)}")
        loader.exec_module(module)
        fn = getattr(module, fn_name)
        return fn

class TensorExprTestOptions:
    # [CLASS DOC] Saves global JIT fuser settings and switches to a configuration suited for
    # TensorExpr fuser tests (profiling executor on, CPU and GPU fusion on, TensorExpr fuser on,
    # fusion-group inlining off, LLVM not required). restore puts the saved values back.
    def __init__(self) -> None:
        # [FUNCTION DOC] Records the current values of each setting in attributes and applies the
        # test configuration. Mutates global JIT state.
        self.old_profiling_executor = torch._C._jit_set_profiling_executor(True)
        self.old_profiling_mode = torch._C._get_graph_executor_optimize(True)

        self.old_cpu_fuser_state = torch._C._jit_can_fuse_on_cpu()
        self.old_gpu_fuser_state = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_cpu(True)
        torch._C._jit_override_can_fuse_on_gpu(True)
        self.texpr_fuser_state = torch._C._jit_texpr_fuser_enabled()
        torch._C._jit_set_texpr_fuser_enabled(True)
        self.old_fusion_inlining = torch._C._debug_get_fusion_group_inlining()
        torch._C._debug_set_fusion_group_inlining(False)
        self.old_te_must_use_llvm_cpu = torch._C._jit_get_te_must_use_llvm_cpu()
        torch._C._jit_set_te_must_use_llvm_cpu(False)

    def restore(self):
        # [FUNCTION DOC] Restores every global JIT setting recorded at construction.
        # KNOWN ISSUE: the graph executor optimize call is a getter-style call with the saved
        # value as argument, so it may not actually restore the previous optimize flag.
        torch._C._jit_set_profiling_executor(self.old_profiling_executor)
        torch._C._get_graph_executor_optimize(self.old_profiling_mode)

        torch._C._jit_set_texpr_fuser_enabled(self.texpr_fuser_state)
        torch._C._jit_override_can_fuse_on_gpu(self.old_gpu_fuser_state)
        torch._C._jit_override_can_fuse_on_cpu(self.old_cpu_fuser_state)
        torch._C._debug_set_fusion_group_inlining(self.old_fusion_inlining)
        torch._C._jit_set_te_must_use_llvm_cpu(self.old_te_must_use_llvm_cpu)

def clone_inputs(args):
    # [FUNCTION DOC] Returns a new list where each tensor argument is a detached clone, each
    # iterable-of-tensors argument becomes a list of detached clones, and other values are kept.
    inputs: list[torch.Tensor | list[torch.Tensor]] = []

    for arg in args:
        if isinstance(arg, torch.Tensor):
            inputs.append(arg.detach().clone())
        elif is_iterable_of_tensors(arg):
            inputs.append([t.detach().clone() for t in arg])
        else:
            inputs.append(arg)

    return inputs

def get_traced_sample_variant_pairs(device, dtype, op):
    # [FUNCTION DOC] Builds (variant, sample) pairs for tracing tests of an operator info object:
    # variants are its function and method forms (skipping missing ones and lambdas); samples come
    # from its sample inputs for the device and dtype. For a few listed ops, boolean sample
    # arguments at given positions are converted to integers (mutating the sample), since JIT
    # scalars exclude bool. Returns an empty list for resize-style ops, which cannot be traced.
    # tuples of (variant, sample)
    outputs: list[tuple[Any, Any]] = []

    samples = op.sample_inputs(device, dtype)

    # Acquires variants to test
    func = op.get_op()
    method = op.get_method()
    variants = {
        # TODO: inplace tests currently fail, fix and add inplace variant
        'function': func, 'method': method,
    }

    # TODO: find better way to standardize on op registration itself..
    has_fake_function = op.name in ["resize_", 'resize_as_']

    if has_fake_function:
        variants = {'method': getattr(torch.Tensor, op.name)}

    # In eager mode, these ops can take (Tensor, bool) args; but in
    # JIT they can only take (Tensor, Scalar), and bool is not a
    # scalar in the JIT type system. So to test these in JIT, the bool
    # is converted to an int for the test.
    ops_with_unsupported_bool_args = [
        {
            "name": "div_floor_rounding",
            "arg_idx": [0],
        },
        {
            "name": "div_no_rounding_mode",
            "arg_idx": [0],
        },
        {
            "name": "div_trunc_rounding",
            "arg_idx": [0],
        },
        {
            "name": "index_fill",
            "arg_idx": [2],
        },
        {
            "name": "full_like",
            "arg_idx": [0],
        },
        {
            "name": "mul",
            "arg_idx": [0],
        },
        {
            "name": "new_full",
            "arg_idx": [1],
        },
    ]

    # doesn't support tracing
    if has_fake_function:
        return outputs

    for sample in samples:
        for variant in variants.values():
            if variant is None:
                continue

            if is_lambda(variant):
                continue

            matching_ops = filter(lambda x: op.formatted_name == x["name"], ops_with_unsupported_bool_args)
            for op_data in matching_ops:
                for idx in op_data["arg_idx"]:
                    args = list(sample.args)
                    if len(sample.args) > idx and isinstance(sample.args[idx], bool):
                        args[idx] = int(args[idx])
                    sample.args = tuple(args)

            outputs.append((variant, sample))

    return outputs

# types.LambdaType gave false positives
def is_lambda(lamb):
    # [FUNCTION DOC] HELPER: returns true if the object is a plain function named like a lambda.
    LAMBDA = lambda: 0  # noqa: E731
    return isinstance(lamb, type(LAMBDA)) and lamb.__name__ == LAMBDA.__name__
