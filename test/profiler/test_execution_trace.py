# Owner(s): ["oncall: profiler"]
# [MODULE DOC] Tests for ExecutionTraceObserver, which records an execution trace of operators
# run under the profiler. Parses the emitted JSON (optionally gzip) trace and compares it with
# Kineto traces. Tests are instantiated per device type, including XPU. Import time side effect:
# disables the tqdm monitor thread when tqdm is installed, so lingering threads do not add
# invalid thread ids to traces.

import json
import os
import sys
import tempfile
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

import torch
import torch.nn as nn
from torch import _dynamo as torchdynamo
from torch.autograd import (
    _record_function_with_args_enter,
    _record_function_with_args_exit,
)
from torch.profiler import (
    ExecutionTraceObserver,
    kineto_available,
    profile,
    record_function,
    supported_activities,
)
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    skipCPUIf,
)
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    run_tests,
    skipIfHpu,
    skipIfTorchDynamo,
    TemporaryFileName,
    TEST_XPU,
    TestCase,
)
from torch.utils._triton import has_triton


# if tqdm is not shutdown properly, it will leave the monitor thread alive.
# This causes an issue in the multithreading test because we check all events
# in that test with their tids. The events that correspond to these lingering
# threads all have TID of (uint64_t)(-1) which is invalid.
# The work around is turning off monitoring thread when tqdm is loaded.
# Since these are unit tests, it is safe to turn off monitor thread.
try:
    import tqdm

    tqdm.tqdm.monitor_interval = 0
except ImportError:
    pass

Json = dict[str, Any]


class TestExecutionTrace(TestCase):
    # [CLASS DOC] Device-generic tests of ExecutionTraceObserver: usage with and without Kineto,
    # environment variable enablement and disabling, start and stop control, repeated use in a
    # loop, nested tensors, triton kernel and FX graph capture under compilation, and integral
    # tensor range and data capture. Provides helpers that generate a workload and parse trace
    # files.
    def payload(self, device, use_device=False):
        # [FUNCTION DOC] HELPER: Workload used by the tests. Inside a user annotation named test
        # one, enters a second record function with a mix of argument types (integer, bool,
        # float, tensor list, tensor tuple, string, tensor, infinity, negative infinity, NaN),
        # does tensor arithmetic with a backward pass (on the given device when use_device is
        # true), applies a GELU module on a CPU tensor, then exits the second record function.
        # Returns nothing.
        u = torch.randn(3, 4, 5, requires_grad=True)
        with record_function("## TEST 1 ##", "1, 2, 3"):
            inf_val = float("inf")
            neg_inf_val = float("-inf")
            nan_val = float("nan")
            rf_handle = _record_function_with_args_enter(
                "## TEST 2 ##",
                1,
                False,
                2.5,
                [u, u],
                (u, u),
                "hello",
                u,
                inf_val,
                neg_inf_val,
                nan_val,
            )
            x = torch.randn(10, 10, requires_grad=True)
            if use_device:
                x = x.to(device)
            y = torch.randn(10, 10, requires_grad=True)
            if use_device:
                y = y.to(device)
            z = x + y + x * y + x * y
            z.backward(z)
            gelu = nn.GELU()
            m = torch.randn(2)
            _ = gelu(m)
            if use_device:
                z = z.cpu()
            _record_function_with_args_exit(rf_handle)

    def get_execution_trace_root(self, output_file_name) -> Json:
        # [FUNCTION DOC] HELPER: Opens the trace file (decompressing when the name ends with the
        # gzip extension), parses JSON, raises AssertionError if the nodes key is missing, and
        # returns the list of node mappings.
        import gzip

        nodes = []
        with (
            gzip.open(output_file_name)
            if output_file_name.endswith(".gz")
            else open(output_file_name)
        ) as f:
            et_graph = json.load(f)
            if "nodes" not in et_graph:
                raise AssertionError(f"Expected 'nodes' in execution trace: {et_graph}")
            nodes = et_graph["nodes"]
        return nodes

    def get_execution_trace_rf_ids(self, nodes: list[Json]) -> list[int]:
        # [FUNCTION DOC] HELPER: Returns the sorted record function ids found in the given trace
        # nodes. Skips the process and thread root marker nodes and nodes with no rf_id
        # attribute.
        """Returns a sorted list of rf_id (record function ids) in execution trace"""

        def get_rf_id(node):
            # [FUNCTION DOC] HELPER: Returns the value of the attribute named rf_id from the
            # node attributes list, or None when absent.
            attrs = node["attrs"]
            for a in attrs:
                if a["name"] == "rf_id":
                    return a["value"]
            return None

        rf_ids_ = (
            get_rf_id(n)
            for n in nodes
            if n["name"] != "[pytorch|profiler|execution_trace|process]"
            and n["name"] != "[pytorch|profiler|execution_trace|thread]"
        )
        return sorted(rf_id for rf_id in rf_ids_ if rf_id is not None)

    def get_kineto_rf_ids(self, events: list[Json]) -> list[int]:
        # [FUNCTION DOC] HELPER: From Kineto trace events keeps those in the cpu operator or
        # user annotation categories and returns their Record function id argument values
        # sorted, using minus one when an event lacks it.
        """Returns a sorted list of Record function IDs for CPU operators and user annotations"""
        ops_and_annotations = (
            e for e in events if e.get("cat", "") in ["cpu_op", "user_annotation"]
        )
        return sorted(
            e.get("args", {}).get("Record function id", -1) for e in ops_and_annotations
        )

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/123536")
    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfHpu
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    def test_execution_trace_with_kineto(self, device):
        # [FUNCTION DOC] Profiles ten steps with a wait/warmup/active schedule and an observer
        # writing to a temporary file, exporting the Kineto trace to another temp file. Asserts
        # the output path is reported, the trace handler ran once, the execution trace has the
        # process root node and exactly two loop annotations (the two active steps), and that
        # sorted record function ids in both traces match exactly. Removes temp files it reads.
        trace_called_num = 0

        def trace_handler(p):
            # [FUNCTION DOC] HELPER: Schedule callback that increments the enclosing test
            # counter of completed traces.
            nonlocal trace_called_num
            trace_called_num += 1

        use_device = device != "cpu"
        # Create a temp file to save execution trace and kineto data.

        with (
            tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False) as fp,
            tempfile.NamedTemporaryFile(
                mode="w+t", suffix=".kineto.json", delete=False
            ) as kt,
            profile(
                activities=supported_activities(),
                schedule=torch.profiler.schedule(
                    skip_first=3, wait=1, warmup=1, active=2, repeat=1
                ),
                on_trace_ready=trace_handler,
                execution_trace_observer=(
                    ExecutionTraceObserver().register_callback(fp.name)
                ),
            ) as p,
        ):
            trace_name = fp.name
            kt_name = kt.name
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(device, use_device=use_device)
                p.step()
            self.assertEqual(fp.name, p.execution_trace_observer.get_output_file_path())

        # Uncomment for debugging
        # print("Output kineto = ", kt.name)
        # print("Output ET = ", fp.name)

        p.export_chrome_trace(kt_name)
        self.assertEqual(trace_called_num, 1)

        nodes = self.get_execution_trace_root(trace_name)
        os.remove(trace_name)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        self.assertTrue(found_root_node)
        # Since profiler trace is active for 2 iterations
        self.assertEqual(loop_count, 2)

        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func ID (rf_id) and External IDs
        # both of these should match for the same trace window.

        with open(kt_name) as f:
            kineto = json.load(f)
            events = kineto["traceEvents"]
        os.remove(kt_name)

        # Look up rf_ids in both Execution and Kineto trace as two lists.
        rf_ids_et = self.get_execution_trace_rf_ids(nodes)
        rf_ids_kineto = self.get_kineto_rf_ids(events)

        self.assertCountEqual(rf_ids_et, rf_ids_kineto)
        self.assertListEqual(
            rf_ids_et,
            rf_ids_kineto,
            msg=f"ET and kineto rf_id should exactly match\n"
            f"  rf_ids_et = {rf_ids_et}\n"
            f"  rf_ids_kineto = {rf_ids_kineto}\n",
        )

    @unittest.skipIf(not kineto_available(), "Kineto is required")
    @skipIfHpu
    @skipIfTorchDynamo("profiler gets ignored if dynamo activated")
    @patch.dict(
        os.environ,
        {
            "ENABLE_PYTORCH_EXECUTION_TRACE": "1",
            "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS": "1",
        },
    )
    def test_execution_trace_env_enabled_with_kineto(self, device):
        # [FUNCTION DOC] With environment variables enabling execution trace and extras,
        # profiles ten scheduled steps without passing an observer and exports Kineto. Asserts
        # the handler ran once, the observer path is set, its resources directory exists and is
        # empty, the trace has the root node and two loop annotations, and record function ids
        # match Kineto exactly.
        trace_called_num = 0

        def trace_handler(p):
            # [FUNCTION DOC] HELPER: Schedule callback that increments the enclosing test
            # counter of completed traces.
            nonlocal trace_called_num
            trace_called_num += 1

        use_device = device != "cpu"
        # Create a temp file to save kineto data.

        with (
            tempfile.NamedTemporaryFile(
                mode="w+t", suffix=".kineto.json", delete=False
            ) as kt,
            profile(
                activities=supported_activities(),
                schedule=torch.profiler.schedule(
                    skip_first=3, wait=1, warmup=1, active=2, repeat=1
                ),
                on_trace_ready=trace_handler,
            ) as p,
        ):
            kt_name = kt.name
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(device, use_device=use_device)
                p.step()

        # Uncomment for debugging
        # print("Output kineto = ", kt.name)
        # print("Output ET = ", fp.name)

        p.export_chrome_trace(kt_name)

        self.assertEqual(trace_called_num, 1)
        et_path = p.execution_trace_observer.get_output_file_path()
        et_res_path = p.execution_trace_observer.get_resources_dir(et_path)
        # the path should be set up due to our env variables
        self.assertTrue(et_path is not None)
        # et_res_path should be an empty directory
        self.assertTrue(os.path.isdir(et_res_path))
        self.assertEqual(len(os.listdir(et_res_path)), 0)
        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func
        nodes = self.get_execution_trace_root(et_path)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        self.assertTrue(found_root_node)
        # Since profiler trace is active for 2 iterations
        self.assertEqual(loop_count, 2)

        # Compare the collected Execution Trace and Kineto Trace
        # in terms of record func ID (rf_id) and External IDs
        # both of these should match for the same trace window.

        with open(kt_name) as f:
            kineto = json.load(f)
            events = kineto["traceEvents"]
        os.remove(kt_name)

        # Look up rf_ids in both Execution and Kineto trace as two lists.
        rf_ids_et = self.get_execution_trace_rf_ids(nodes)
        rf_ids_kineto = self.get_kineto_rf_ids(events)

        self.assertCountEqual(rf_ids_et, rf_ids_kineto)
        self.assertListEqual(
            rf_ids_et,
            rf_ids_kineto,
            msg=f"ET and kineto rf_id should exactly match\n"
            f"  rf_ids_et = {rf_ids_et}\n"
            f"  rf_ids_kineto = {rf_ids_kineto}\n",
        )

    def test_execution_trace_alone(self, device):
        # [FUNCTION DOC] Registers an observer on a temp gzip file, starts it, runs five
        # annotated workloads, stops and unregisters. Asserts output path matches, the parsed
        # trace has a root node, five loop annotations, and that the first tensor input of the
        # second-level record function node is a six element tuple.
        use_device = device != "cpu"
        # Create a temp file to save execution trace data.
        # Use a gzip file to test compression codepath
        with tempfile.NamedTemporaryFile("w", suffix=".et.json.gz", delete=False) as fp:
            filename = fp.name
        expected_loop_events = 0

        et = ExecutionTraceObserver().register_callback(filename)

        et.start()
        for idx in range(5):
            expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)
        et.stop()

        if filename != et.get_output_file_path():
            raise AssertionError(
                f"Expected output file path {filename}, got {et.get_output_file_path()}"
            )
        et.unregister_callback()
        nodes = self.get_execution_trace_root(filename)
        os.remove(filename)
        loop_count = 0
        # Expected tensor object tuple size, in th form of:
        # [tensor_id, storage_id, offset, numel, itemsize, device_str]
        tensor_tuple_size = 6
        found_root_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
            # Check if tensor tuple representation size is correct.
            if n["name"] == "## TEST 2 ##":
                if len(n["inputs"]["values"][3][0]) != tensor_tuple_size:
                    raise AssertionError(
                        f"Expected tensor tuple size {tensor_tuple_size}, got "
                        f"{len(n['inputs']['values'][3][0])}"
                    )
        if not found_root_node:
            raise AssertionError("Expected to find root node")
        if loop_count != expected_loop_events:
            raise AssertionError(
                f"Expected {expected_loop_events} loop events, got {loop_count}"
            )

    @patch.dict(
        os.environ,
        {
            "ENABLE_PYTORCH_EXECUTION_TRACE": "0",
            "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS": "0",
        },
    )
    def test_execution_trace_env_disabled(self, device):
        # [FUNCTION DOC] With the environment variables set to zero, profiles a scheduled run
        # and asserts the profiler has no execution trace observer.
        use_device = device != "cpu"

        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    self.payload(device, use_device=use_device)
                p.step()

        self.assertTrue(p.execution_trace_observer is None)

    @unittest.skipIf(IS_WINDOWS, "torch.compile does not support WINDOWS")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(
        not (has_triton() and (TEST_CUDA or TEST_XPU)),
        "need triton and device(CUDA or XPU) availability to run",
    )
    @skipCPUIf(True, "skip CPU device for testing profiling triton")
    def test_execution_trace_with_pt2(self, device):
        # [FUNCTION DOC] Compiles a small linear, add and cos function with inductor (single
        # compile thread), then profiles it with an observer with extra resource collection on.
        # Parses the trace and asserts a triton kernel node exists with a non-empty kernel_file
        # attribute, input values and no output values, and a node for calling the compiled FX
        # graph. Skipped on Windows, python 3.12 plus, CPU, or without triton on CUDA or XPU.
        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Inductor-compiled function: linear of a with b, adds c,
            # returns the cosine.
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        a, b, c = (torch.randn(4, 4, requires_grad=True).to(device) for _ in range(3))

        inputs = [a, b, c]
        with torch._inductor.config.patch(compile_threads=1):
            fn(*inputs)

        # Create a temp file to save execution trace data.
        with tempfile.NamedTemporaryFile("w+t", suffix="_et.json", delete=False) as fp:
            filename = fp.name
        et = ExecutionTraceObserver()
        et.register_callback(filename)
        et.set_extra_resource_collection(True)

        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
            execution_trace_observer=et,
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    fn(*inputs)
                p.step()

        nodes = self.get_execution_trace_root(filename)
        os.remove(filename)
        found_captured_triton_kernel_node = False
        found_call_compiled_fx_graph = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "triton_" in n["name"]:
                for attr in n["attrs"]:
                    if attr["name"] == "kernel_file" and attr["value"] != "":
                        found_captured_triton_kernel_node = True
                        if len(n["inputs"]["values"]) <= 0:
                            raise AssertionError(
                                "Expected triton node to have input values"
                            )
                        if len(n["outputs"]["values"]) != 0:
                            raise AssertionError(
                                "Expected triton node to have no output values"
                            )
            elif "Call CompiledFxGraph" in n["name"]:
                found_call_compiled_fx_graph = True
        if not found_captured_triton_kernel_node:
            raise AssertionError("Expected captured triton kernel node")
        if not found_call_compiled_fx_graph:
            raise AssertionError("Expected Call CompiledFxGraph node")

    @unittest.skipIf(IS_WINDOWS, "torch.compile does not support WINDOWS")
    @unittest.skipIf(
        sys.version_info >= (3, 12), "torch.compile is not supported on python 3.12+"
    )
    @unittest.skipIf(
        not (has_triton() and (TEST_CUDA or TEST_XPU)),
        "need triton and device(CUDA or XPU) availability to run",
    )
    @skipCPUIf(True, "skip CPU device for testing profiling triton")
    @patch.dict(
        os.environ,
        {
            "ENABLE_PYTORCH_EXECUTION_TRACE": "1",
            "ENABLE_PYTORCH_EXECUTION_TRACE_EXTRAS": "1",
        },
    )
    def test_execution_trace_env_enabled_with_pt2(self, device):
        # [FUNCTION DOC] With environment variables enabling the trace, clears the code cache,
        # compiles with fx graph caching disabled, and profiles without an observer. Asserts the
        # observer path exists, the resources directory exists with exactly two files, and that
        # a triton kernel node has a non-empty kernel_file attribute, input values and no
        # outputs. Same skips as the other compile test.
        # clean up the local cache for triton kernel
        from torch._inductor.codecache import PyCodeCache

        PyCodeCache.cache_clear(purge=True)

        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Inductor-compiled function: linear of a with b, adds c,
            # returns the cosine.
            x = torch.nn.functional.linear(a, b)
            x = x + c
            return x.cos()

        a, b, c = (torch.randn(4, 4, requires_grad=True).to(device) for _ in range(3))

        inputs = [a, b, c]
        with torch._inductor.config.patch(
            compile_threads=1, fx_graph_cache=False, fx_graph_remote_cache=False
        ):
            fn(*inputs)

        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=3, wait=1, warmup=1, active=2, repeat=1
            ),
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    fn(*inputs)
                p.step()

        et_path = p.execution_trace_observer.get_output_file_path()
        et_res_path = p.execution_trace_observer.get_resources_dir(et_path)
        # the path should be set up due to our env variables
        self.assertTrue(et_path is not None)
        # et_res_path should be an empty directory
        self.assertTrue(os.path.isdir(et_res_path))
        self.assertEqual(len(os.listdir(et_res_path)), 2)
        nodes = self.get_execution_trace_root(et_path)
        found_captured_triton_kernel_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "triton_" in n["name"]:
                for attr in n["attrs"]:
                    if attr["name"] == "kernel_file" and attr["value"] != "":
                        found_captured_triton_kernel_node = True
                        if len(n["inputs"]["values"]) <= 0:
                            raise AssertionError(
                                "Expected triton node to have input values"
                            )
                        if len(n["outputs"]["values"]) != 0:
                            raise AssertionError(
                                "Expected triton node to have no output values"
                            )
        if not found_captured_triton_kernel_node:
            raise AssertionError("Expected captured triton kernel node")

    @unittest.skipIf(IS_WINDOWS, "torch.compile does not support WINDOWS")
    @unittest.skipIf(
        not (has_triton() and (TEST_CUDA or TEST_XPU)),
        "need triton and device(CUDA or XPU) availability to run",
    )
    @skipCPUIf(True, "skip CPU device for testing profiling triton")
    def test_triton_fx_graph_with_et(self, device):
        # [FUNCTION DOC] Compiles a function with caching disabled, profiles one active step
        # with extra resources on, then reads each file in the resources directory. For the file
        # containing the FX graph fragment marker, collects the following comment lines and
        # asserts they match the expected FX graph text for the device. Removes the files it
        # visits.
        # clean up the local cache for triton kernel
        from torch._inductor.codecache import PyCodeCache

        PyCodeCache.cache_clear(purge=True)

        @torchdynamo.optimize("inductor")
        def fn(a, b, c):
            # [FUNCTION DOC] HELPER: Inductor-compiled function: linear, sine, transpose plus
            # scaled c, then cosine.
            x = torch.nn.functional.linear(a, b)
            x = x.sin()
            x = x.t() + c * 1111
            return x.cos()

        a, b, c = (
            torch.randn(4, 4, requires_grad=False).to(torch.device(device))
            for _ in range(3)
        )

        with torch._inductor.config.patch(
            compile_threads=1, fx_graph_cache=False, fx_graph_remote_cache=False
        ):
            fn(a, b, c)

        et = ExecutionTraceObserver()
        with tempfile.NamedTemporaryFile(
            "w+t", suffix="fx_graph_et.json", delete=False
        ) as fp:
            et.register_callback(fp.name)
        et.set_extra_resource_collection(True)
        with profile(
            activities=torch.profiler.supported_activities(),
            record_shapes=True,
            schedule=torch.profiler.schedule(
                skip_first=0, wait=1, warmup=1, active=1, repeat=1
            ),
            execution_trace_observer=et,
        ) as p:
            for idx in range(10):
                with record_function(f"## LOOP {idx} ##"):
                    fn(a, b, c)
                p.step()

        et_path = p.execution_trace_observer.get_output_file_path()
        et_res_path = p.execution_trace_observer.get_resources_dir(et_path)
        # the path should be set up due to our env variables
        self.assertTrue(et_path is not None)
        # et_res_path should be an empty directory
        self.assertTrue(os.path.isdir(et_res_path))
        for filename in os.listdir(et_res_path):
            file_path = os.path.join(et_res_path, filename)
            if os.path.isfile(file_path):
                with open(file_path) as file:
                    fx_graph_found = False
                    fx_graph = []
                    for line in file:
                        line = line.strip()
                        # There are two files in the directory, one is the source
                        # code of the triton kernel, and the other is the source code for FX graph.
                        # Only the FX graph file contains the string "# Graph fragment:".
                        if line.startswith("# Graph fragment:"):
                            fx_graph_found = True
                        elif fx_graph_found and line.startswith("#"):
                            fx_graph.append(line)
                        else:
                            fx_graph_found = False

                    if len(fx_graph) > 0:
                        expected_graph = [
                            f'#   %mm : Tensor "f32[4, 4][4, 1]{device}" = PlaceHolder[target=mm]',
                            f'#   %arg2_1 : Tensor "f32[4, 4][4, 1]{device}" = PlaceHolder[target=arg2_1]',
                            f'#   %sin : Tensor "f32[4, 4][4, 1]{device}"[num_users=1] = call_function[target=torch.ops.aten.sin.default](args = (%mm,), kwargs = {{}})',
                            f'#   %permute_1 : Tensor "f32[4, 4][1, 4]{device}"[num_users=1] = call_function[target=torch.ops.aten.permute.default](args = (%sin, [1, 0]), kwargs = {{}})',
                            f'#   %mul : Tensor "f32[4, 4][4, 1]{device}"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 1111), kwargs = {{}})',
                            f'#   %add : Tensor "f32[4, 4][1, 4]{device}"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%permute_1, %mul), kwargs = {{}})',
                            f'#   %cos : Tensor "f32[4, 4][1, 4]{device}"[num_users=1] = call_function[target=torch.ops.aten.cos.default](args = (%add,), kwargs = {{}})',
                            "#   return %cos",
                        ]
                        if len(fx_graph) < len(expected_graph):
                            raise AssertionError(
                                f"Expected at least {len(expected_graph)} fx_graph lines, "
                                f"got {len(fx_graph)}"
                            )
                        for idx, expected in enumerate(expected_graph):
                            if fx_graph[idx] != expected:
                                raise AssertionError(
                                    f"Expected fx_graph[{idx}] to be {expected}, got {fx_graph[idx]}"
                                )
                os.remove(file_path)

    def test_execution_trace_start_stop(self, device):
        # [FUNCTION DOC] Runs ten iterations of the workload, starting the observer at index
        # three, stopping at five, starting at eight and stopping at nine, counting iterations
        # where its running flag is true. Asserts the path, then that the trace has a root node
        # and as many loop annotations as counted.
        use_device = device != "cpu"
        # Create a temp file to save execution trace data.
        with tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False) as fp:
            filename = fp.name
        expected_loop_events = 0
        et = ExecutionTraceObserver().register_callback(filename)
        for idx in range(10):
            if idx == 3:
                et.start()
            elif idx == 5:
                et.stop()
            elif idx == 8:
                et.start()
            elif idx == 9:
                et.stop()
            if et._execution_trace_running:
                expected_loop_events += 1
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)

        if filename != et.get_output_file_path():
            raise AssertionError(
                f"Expected output file path {filename}, got {et.get_output_file_path()}"
            )
        et.unregister_callback()
        nodes = self.get_execution_trace_root(filename)
        os.remove(filename)
        loop_count = 0
        found_root_node = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                found_root_node = True
            if n["name"].startswith("## LOOP "):
                loop_count += 1
        if not found_root_node:
            raise AssertionError("Expected to find root node")
        if loop_count != expected_loop_events:
            raise AssertionError(
                f"Expected {expected_loop_events} loop events, got {loop_count}"
            )

    def test_execution_trace_repeat_in_loop(self, device):
        # [FUNCTION DOC] For iterations three, four, six and eight, creates a new temp file and
        # observer, starts it, runs the annotated workload and stops and unregisters it. Asserts
        # each trace has a root node with id one and that total loop annotations across files
        # equals four.
        use_device = device != "cpu"
        iter_list = {3, 4, 6, 8}
        expected_loop_events = len(iter_list)
        output_files = []
        for idx in range(10):
            if idx in iter_list:
                # Create a temp file to save execution trace data.
                with tempfile.NamedTemporaryFile(
                    "w+t", suffix=".et.json", delete=False
                ) as fp:
                    output_files.append(fp.name)
                    et = ExecutionTraceObserver().register_callback(fp.name)
                et.start()
            with record_function(f"## LOOP {idx} ##"):
                self.payload(device, use_device=use_device)
            if idx in iter_list:
                et.stop()
                et.unregister_callback()

        event_count = 0
        for et_file in output_files:
            nodes = self.get_execution_trace_root(et_file)
            found_root_node = False
            for n in nodes:
                if "name" not in n:
                    raise AssertionError(f"Expected node to have 'name': {n}")
                if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                    if n["id"] != 1:
                        raise AssertionError(f"Expected root node id 1, got {n['id']}")
                    found_root_node = True
                if n["name"].startswith("## LOOP "):
                    event_count += 1
            if not found_root_node:
                raise AssertionError("Expected to find root node")
        if event_count != expected_loop_events:
            raise AssertionError(
                f"Expected {expected_loop_events} loop events, got {event_count}"
            )

    def test_execution_trace_no_capture(self):
        # [FUNCTION DOC] Registers an observer on a temp file and unregisters it without running
        # any work, then asserts the output path matched and that the written trace still holds
        # the root node.
        with TemporaryFileName("w+t", suffix=".et.json") as file_name:
            et = ExecutionTraceObserver().register_callback(file_name)

            if file_name != et.get_output_file_path():
                raise AssertionError(
                    f"Expected output file path {file_name}, got {et.get_output_file_path()}"
                )
            et.unregister_callback()
            nodes = self.get_execution_trace_root(file_name)
            found_root_node = False
            for n in nodes:
                if "name" not in n:
                    raise AssertionError(f"Expected node to have 'name': {n}")
                if "[pytorch|profiler|execution_trace|process]" in n["name"]:
                    found_root_node = True
            if not found_root_node:
                raise AssertionError("Expected to find root node")

    @skipIfTorchDynamo("https://github.com/pytorch/pytorch/issues/124500")
    def test_execution_trace_nested_tensor(self):
        # [FUNCTION DOC] Profiles three iterations of sine and cosine on jagged nested tensors
        # with an observer on a temp file and asserts the trace contains a node whose name has
        # cos. Skipped under dynamo.
        def fn(nt):
            # [FUNCTION DOC] HELPER: Returns the sine then cosine of the given nested tensor.
            return nt.sin().cos()

        with tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False) as fp:
            observer = ExecutionTraceObserver().register_callback(fp.name)
            filename = fp.name
        with torch.profiler.profile(execution_trace_observer=observer):
            for i in range(3):
                values = torch.rand((8 + i, 4 + i))
                offsets = torch.tensor([0, 2, 4, 6, 8 + i])
                nt = torch.nested.nested_tensor_from_jagged(values, offsets)
                fn(nt)

        nodes = self.get_execution_trace_root(filename)
        os.remove(filename)
        found_cos = False
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "cos" in n["name"]:
                found_cos = True
        if not found_cos:
            raise AssertionError("Expected to find cos node")

    @skipCPUIf(True, "accelerator required for integral tensor range profiling")
    @patch.dict(
        os.environ, {"ENABLE_PYTORCH_EXECUTION_TRACE_SAVE_INTEGRAL_TENSOR_RANGE": "1"}
    )
    def test_execution_trace_record_integral_tensor_range(self, device):
        # [FUNCTION DOC] With the integral tensor range environment variable set, profiles a
        # gather on integer tensors on the device. For gather nodes with a tensor_range
        # attribute, asserts its value is the expected JSON string of min and max ranges for the
        # two inputs. Skipped on CPU.
        t1 = torch.tensor([[1, 2], [3, 4]], device=device)
        t2 = torch.tensor([[0, 0], [1, 0]], device=device)
        with (
            tempfile.NamedTemporaryFile("w+t", suffix=".et.json", delete=False) as fp,
            profile(
                activities=supported_activities(),
                schedule=torch.profiler.schedule(
                    skip_first=0, wait=0, warmup=0, active=1, repeat=1
                ),
                record_shapes=True,
                execution_trace_observer=(
                    ExecutionTraceObserver().register_callback(fp.name)
                ),
            ) as p,
        ):
            filename = fp.name
            torch.gather(t1, 1, t2)
            p.step()

        nodes = self.get_execution_trace_root(filename)
        os.remove(filename)
        for n in nodes:
            if "name" not in n:
                raise AssertionError(f"Expected node to have 'name': {n}")
            if "aten::gather" in n["name"]:
                for attr in n["attrs"]:
                    if attr["name"] == "tensor_range":
                        if attr["value"] != '{"0":[1,4],"1":[0,1]}':
                            raise AssertionError(
                                "Expected tensor_range value to match "
                                '{"0":[1,4],"1":[0,1]}'
                            )

    @skipCPUIf(True, "accelerator required for integral tensor data profiling")
    @patch.dict(
        os.environ,
        {"ENABLE_PYTORCH_EXECUTION_TRACE_SAVE_INTEGRAL_TENSOR_DATA": "aten::gather"},
    )
    def test_execution_trace_record_integral_tensor_data(self, device):
        # [FUNCTION DOC] With the integral tensor data environment variable naming gather,
        # profiles a gather with extra resources on. Asserts that two data files for tensors
        # zero and one exist in the resources directory and, read as int64, contain the values
        # of the two inputs (one to four, and zero zero one zero). Skipped on CPU.
        with tempfile.TemporaryDirectory() as temp_dir:
            fp_name = os.path.join(temp_dir, "test.et.json")
            et = ExecutionTraceObserver()
            et.register_callback(fp_name)
            et.set_extra_resource_collection(True)

            t1 = torch.tensor([[1, 2], [3, 4]], device=device)
            t2 = torch.tensor([[0, 0], [1, 0]], device=device)
            with profile(
                activities=supported_activities(),
                schedule=torch.profiler.schedule(
                    skip_first=0, wait=0, warmup=0, active=1, repeat=1
                ),
                record_shapes=True,
                execution_trace_observer=et,
            ) as p:
                torch.gather(t1, 1, t2)
                p.step()

            resourceDir = fp_name.replace(".json", "_resources")
            if not os.path.exists(resourceDir + "/nid_4_tid_0.dat"):
                raise AssertionError("Expected nid_4_tid_0.dat to exist")
            if not os.path.exists(resourceDir + "/nid_4_tid_1.dat"):
                raise AssertionError("Expected nid_4_tid_1.dat to exist")

            t1 = np.fromfile(resourceDir + "/nid_4_tid_0.dat", dtype=np.int64)
            t2 = np.fromfile(resourceDir + "/nid_4_tid_1.dat", dtype=np.int64)
            if not (t1 == np.array([1, 2, 3, 4])).all():
                raise AssertionError("Expected t1 contents to match [1, 2, 3, 4]")
            if not (t2 == np.array([0, 0, 1, 0])).all():
                raise AssertionError("Expected t2 contents to match [0, 0, 1, 0]")


instantiate_device_type_tests(TestExecutionTrace, globals(), allow_xpu=True)

if __name__ == "__main__":
    run_tests()
