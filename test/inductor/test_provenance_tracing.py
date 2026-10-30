# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor provenance tracing. Covers (a) the node mapping artifacts that
# relate post grad nodes, pre grad nodes and generated kernels (CPU and GPU, JIT inductor and
# AOT inductor), (b) metadata transfer of stack traces through post grad passes, (c) per-kernel
# stack trace capture via the structured trace log, (d) the kernel information JSON stored in
# AOTI packages and built from the debug module globals, including extern kernel semantic keys
# and shape metadata, and (e) kernel context guard emission in generated AOTI C++ code. Defines
# small models and fake IR classes used as fixtures. Import-time side effects: tries a relative
# import of the AOTI runner utils and test copy helper then falls back to absolute import, and
# registers template tests onto CPU and GPU classes by copying them.

import contextlib
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch._C import FileCheck
from torch._dynamo.utils import detect_fake_mode
from torch._inductor import config
from torch._inductor.debug import (
    create_kernel_information_json,
    create_mapping_pre_post_grad_nodes,
    create_node_mapping_kernel_to_post_grad,
    get_kernel_information_jsons,
    reset_inductor_kernel_provenance_debug_handle,
    reset_provenance_globals,
    set_kernel_post_grad_provenance_tracing,
)
from torch._inductor.fx_passes.post_grad import post_grad_passes
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code, run_and_get_cpp_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import IS_MACOS
from torch.testing._internal.inductor_utils import GPU_TYPE
from torch.testing._internal.triton_utils import requires_gpu_and_triton


try:
    from .test_aot_inductor_utils import AOTIRunnerUtil
    from .test_torchinductor import copy_tests
except ImportError:
    from test_aot_inductor_utils import AOTIRunnerUtil
    from test_torchinductor import (
        copy_tests,  # @manual=fbcode//caffe2/test/inductor:test_inductor-library
    )


trace_log = logging.getLogger("torch.__trace")


class Model(torch.nn.Module):
    # [CLASS DOC] HELPER: Model computing a scaled input, an addmm with it and then gelu.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base initializer; no parameters.
        super().__init__()

    def forward(self, a, b, c):
        # [FUNCTION DOC] HELPER: Multiplies a by a constant, applies addmm with c as the bias
        # and b, then gelu, and returns the result.
        x = a * 3.14
        y = torch.addmm(c, x, b)
        z = torch.nn.functional.gelu(y)
        return z


class Model2(torch.nn.Module):
    # [CLASS DOC] HELPER: Model of three independent pointwise activations (relu, sigmoid, tanh)
    # on its three inputs, used to produce a combo kernel.
    # this test model is used for combo kernel provenance tracing info
    def __init__(self):
        # [FUNCTION DOC] HELPER: Calls the base initializer; no parameters.
        super().__init__()

    def forward(self, a, b, c):
        # [FUNCTION DOC] HELPER: Applies relu to a, sigmoid to b and tanh to c; returns the
        # three results.
        a1 = torch.nn.functional.relu(a)
        b1 = torch.nn.functional.sigmoid(b)
        c1 = torch.nn.functional.tanh(c)
        return a1, b1, c1


class Model3(torch.nn.Module):
    # [CLASS DOC] HELPER: Linear layer model whose weight and bias are plain random GPU tensors
    # (not parameters), used to produce an extern kernel for a linear op.
    def __init__(self, n, k):
        # [FUNCTION DOC] HELPER: Stores random weight (n by k) and bias (n) created on the GPU
        # device.
        super().__init__()
        self.weight = torch.randn(n, k, device=GPU_TYPE)
        self.bias = torch.randn(n, device=GPU_TYPE)

    def forward(self, a):
        # [FUNCTION DOC] HELPER: Returns the functional linear of the input with the stored
        # weight and bias.
        return torch.nn.functional.linear(a, self.weight, self.bias)


class Model4(torch.nn.Module):
    # [CLASS DOC] HELPER: Model with a linear layer followed by relu and sigmoid on one input,
    # plus an independent scaled addmm and gelu branch on three other inputs; returns both
    # results.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates a linear layer (10 to 16), relu and sigmoid modules.
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 16)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, a, b, c):
        # [FUNCTION DOC] HELPER: Runs the first input through linear, relu and sigmoid; scales
        # a, applies addmm with c and b, then gelu; returns the pair of results.
        x = self.fc1(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        d = a * 3.14
        y = torch.addmm(c, d, b)
        z = torch.nn.functional.gelu(y)
        return x, z


def _bias_like_addmm_input(device):
    # [FUNCTION DOC] HELPER: Returns a random one dimensional bias of length 30 on the device,
    # so the addmm in the tests stays in the bias path of the GPU unfusion rather than a full
    # accumulator.
    # These provenance tests exercise the GPU addmm-unfusion path. Keep the
    # input bias-like so they do not depend on full-size accumulator unfusion.
    return torch.randn(30, device=device)


@config.patch("trace.enabled", True)
@config.patch("trace.provenance_tracking_level", 1)
class TestProvenanceTracingArtifact(TestCase):
    # [CLASS DOC] Tests that the provenance artifacts (node mapping JSON) written to the trace
    # debug directory match expectations. Patched at class level to enable tracing at provenance
    # level one.
    """
    This test checks that generated provenance tracing artifact from "post_grad" to
    corresponding "inductor triton kernel node" is expected.
    """

    def _check_provenance_tracing_kernel_to_post_grad(self, filepath, expected_data):
        # [FUNCTION DOC] HELPER: Checks the given path is a directory, loads the node mappings
        # JSON file inside it, and asserts its cppCodeToPost section equals the expected mapping
        # when both are compared as sorted item lists.
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        actual_data = actual_data["cppCodeToPost"]
        # check that the generated provenance tracing artifact is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_data.items()))

    def _check_provenance_tracking_node_mappings(self, filepath, expected_mapping):
        # [FUNCTION DOC] HELPER: Checks the path is a directory, loads the node mappings JSON in
        # it, and asserts the sorted top-level items equal the expected list of pairs.
        self.assertTrue(filepath.is_dir())
        filename = Path(filepath) / "inductor_provenance_tracking_node_mappings.json"
        with open(filename) as f:
            actual_data = json.load(f)
        # check that the generated provenance tracing node mapping is expected
        self.assertEqual(sorted(actual_data.items()), sorted(expected_mapping))

    def _test_triton_kernel_to_post_grad_tracing(self, device):
        # [FUNCTION DOC] HELPER: For both AOT inductor and JIT inductor (via export), compiles
        # the simple model with a temporary debug directory and caches disabled, captures the
        # single warning that reports the debug trace directory, and checks the mappings. On GPU
        # devices the complete four-way mapping (kernel to post grad, post grad to kernel, post
        # to pre, pre to post) is compared, with extra extern mm entries that depend on backend
        # and device; on CPU only the kernel to post grad mapping is compared. The debug
        # directory is removed afterwards.
        a = torch.randn(10, 20, device=device)
        b = torch.randn(20, 30, device=device)
        if device == "cpu":
            c = torch.randn(10, 30, device=device)
        else:
            c = _bias_like_addmm_input(device)
        example_inputs = (a, b, c)

        model = Model().to(device)
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
            reset_inductor_kernel_provenance_debug_handle()
            try:
                with config.patch(
                    {
                        "trace.debug_dir": tempfile.mkdtemp(),
                        "force_disable_caches": True,
                    }
                ):
                    with self.assertLogs(
                        logging.getLogger("torch._inductor.debug"),
                        level=logging.WARNING,
                    ) as cm:
                        if backend == "aot_inductor":
                            AOTIRunnerUtil.run(model, example_inputs)
                        else:
                            ep = torch.export._trace._export(model, example_inputs)
                            compiled = torch.compile(ep.module(), backend=backend)
                            compiled(*example_inputs)
                    self.assertEqual(len(cm.output), 1)
                    m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
                    self.assertTrue(m)
                    filepath = Path(m.group(1))
                    if device == "cuda" or device == "xpu":
                        # Both inductor (torch.compile) and aot_inductor (export)
                        # canonicalize node order, so pre-graph names become
                        # canonical for both.
                        pre_mul, pre_addmm, pre_gelu = (
                            "mul_tensor",
                            "addmm_default",
                            "gelu_default",
                        )
                        expected_mapping = [
                            (
                                "cppCodeToPost",
                                {
                                    "triton_poi_fused_mul_0:1": ["mul"],
                                    "triton_poi_fused_addmm_gelu_1:3": [
                                        "mul_3",
                                        "mul_1",
                                        "add_tensor",
                                        "add",
                                        "erf",
                                        "mul_2",
                                    ],
                                },
                            ),
                            (
                                "postToCppCode",
                                {
                                    "mul": ["triton_poi_fused_mul_0:1"],
                                    "mul_3": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "mul_1": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "add_tensor": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "add": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "erf": ["triton_poi_fused_addmm_gelu_1:3"],
                                    "mul_2": ["triton_poi_fused_addmm_gelu_1:3"],
                                },
                            ),
                            (
                                "postToPre",
                                {
                                    "mul": [pre_mul],
                                    "mm_default": [pre_addmm],
                                    "add_tensor": [pre_addmm],
                                    "mul_1": [pre_gelu],
                                    "mul_2": [pre_gelu],
                                    "erf": [pre_gelu],
                                    "add": [pre_gelu],
                                    "mul_3": [pre_gelu],
                                },
                            ),
                            (
                                "preToPost",
                                {
                                    pre_mul: ["mul"],
                                    pre_addmm: [
                                        "mm_default",
                                        "add_tensor",
                                    ],
                                    pre_gelu: [
                                        "mul_1",
                                        "mul_2",
                                        "erf",
                                        "add",
                                        "mul_3",
                                    ],
                                },
                            ),
                        ]
                        if backend == "aot_inductor" and device == "cuda":
                            expected_mapping[0][1]["aoti_torch_cuda_mm_out:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "aoti_torch_cuda_mm_out:2"
                            ]
                        elif backend == "aot_inductor" and device == "xpu":
                            expected_mapping[0][1]["aoti_torch_xpu_mm_out:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "aoti_torch_xpu_mm_out:2"
                            ]
                        else:
                            expected_mapping[0][1]["extern_kernels.mm:2"] = [
                                "mm_default"
                            ]
                            expected_mapping[1][1]["mm_default"] = [
                                "extern_kernels.mm:2"
                            ]
                        self._check_provenance_tracking_node_mappings(
                            filepath, expected_mapping
                        )
                    else:
                        if device != "cpu":
                            raise AssertionError
                        # check the inductor kernel to post grad nodes mapping is expected for cpu
                        if backend == "aot_inductor":
                            expected_data = {
                                "cpp_fused_mul_0:1": ["mul"],
                                "aoti_torch_cpu_addmm_out:2": ["addmm"],
                                "cpp_fused_gelu_1:3": [
                                    "mul_3",
                                    "mul_1",
                                    "add",
                                    "erf",
                                    "mul_2",
                                ],
                            }
                        else:
                            # backend == "inductor"
                            expected_data = {
                                "cpp_fused_mul_0:1": ["mul"],
                                "cpp_fused_gelu_1:3": [
                                    "mul_3",
                                    "mul_1",
                                    "add",
                                    "erf",
                                    "mul_2",
                                ],
                                "extern_kernels.addmm:2": ["addmm"],
                            }
                        self._check_provenance_tracing_kernel_to_post_grad(
                            filepath, expected_data
                        )

            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_cuda(self):
        # [FUNCTION DOC] Runs the shared node mapping check on the GPU device (needs GPU and
        # Triton).
        self._test_triton_kernel_to_post_grad_tracing(device=GPU_TYPE)

    def test_triton_kernel_to_post_grad_tracing_cpu(self):
        # [FUNCTION DOC] Runs the shared node mapping check on CPU.
        self._test_triton_kernel_to_post_grad_tracing(device="cpu")

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_extern_kernel(self):
        # [FUNCTION DOC] For a linear model on GPU, compiles with AOT inductor and JIT inductor
        # and checks the kernel to post grad mapping: the JIT case maps one extern addmm kernel,
        # the AOT case maps an extern addmm out kernel and a Triton kernel for a constant. Needs
        # GPU and Triton.
        M = 8
        N = 6
        K = 16
        model = Model3(N, K)
        batch = 2
        a = torch.randn(batch, M, K, device=GPU_TYPE)
        example_inputs = (a,)
        filepath = None

        for backend in ["aot_inductor", "inductor"]:
            reset_inductor_kernel_provenance_debug_handle()
            try:
                with config.patch(
                    {
                        "trace.debug_dir": tempfile.mkdtemp(),
                        "force_disable_caches": True,
                    }
                ):
                    with self.assertLogs(
                        logging.getLogger("torch._inductor.debug"),
                        level=logging.WARNING,
                    ) as cm:
                        if backend == "aot_inductor":
                            AOTIRunnerUtil.run(model, example_inputs)
                        else:
                            ep = torch.export._trace._export(model, example_inputs)
                            compiled = torch.compile(ep.module(), backend=backend)
                            compiled(*example_inputs)
                    self.assertEqual(len(cm.output), 1)
                    m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
                    self.assertTrue(m)
                    filepath = Path(m.group(1))
                    if backend == "inductor":
                        expected_data = {
                            "extern_kernels.addmm:1": ["addmm"],
                        }
                    else:
                        # backend = aot_inductor
                        expected_data = {
                            f"aoti_torch_{GPU_TYPE}_addmm_out:2": ["addmm"],
                            "triton_poi_fused_0:1": ["_tensor_constant1"],
                        }

                    self._check_provenance_tracing_kernel_to_post_grad(
                        filepath, expected_data
                    )
            finally:
                if filepath:
                    shutil.rmtree(filepath)

    @requires_gpu_and_triton
    def _test_pt_tracing_combo_kernel(self, backend):
        # [FUNCTION DOC] HELPER: Compiles the three-activation model with combo kernels enabled
        # for the given backend and checks the single Triton combo kernel maps to relu, sigmoid
        # and tanh post grad nodes. The decorator marks it as needing GPU and Triton.
        """This test checks that generated provenance tracing artifact from triton combo kernel to post grad nodes"""
        a = torch.randn(10, 10, device=GPU_TYPE)
        b = torch.randn(20, 20, device=GPU_TYPE)
        c = torch.randn(10, 10, device=GPU_TYPE)
        example_inputs = (a, b, c)

        model = Model2()
        reset_inductor_kernel_provenance_debug_handle()

        with config.patch(
            {
                "trace.debug_dir": tempfile.mkdtemp(),
                "force_disable_caches": True,
                "combo_kernels": True,
                "benchmark_combo_kernel": False,
            }
        ):
            with self.assertLogs(
                logging.getLogger("torch._inductor.debug"),
                level=logging.WARNING,
            ) as cm:
                if backend == "aot_inductor":
                    AOTIRunnerUtil.run(model, example_inputs)
                else:
                    ep = torch.export._trace._export(model, example_inputs)
                    compiled = torch.compile(ep.module(), backend=backend)
                    compiled(*example_inputs)
            self.assertEqual(len(cm.output), 1)
            m = re.match(r"WARNING.* debug trace: (.*)", cm.output[0])
            self.assertTrue(m)
            filepath = Path(m.group(1)).resolve()
            expected_data = {"triton_poi_fused_0:1": ["relu", "sigmoid", "tanh"]}
            self._check_provenance_tracing_kernel_to_post_grad(filepath, expected_data)

    @requires_gpu_and_triton
    def test_triton_kernel_to_post_grad_tracing_combo_kernel(self):
        # [FUNCTION DOC] Runs the combo kernel provenance check for the JIT inductor and then
        # the AOT inductor backends. Needs GPU and Triton.
        self._test_pt_tracing_combo_kernel(backend="inductor")
        self._test_pt_tracing_combo_kernel(backend="aot_inductor")


class TestProvenanceTracingNodeMapping(TestCase):
    # [CLASS DOC] Tests the pure functions that build node mapping dictionaries from provenance
    # JSON.
    def test_create_node_mapping(self):
        # [FUNCTION DOC] Builds a post grad to pre grad provenance structure with nested origins
        # and placeholder entries, plus a kernel to post grad mapping, and checks the combined
        # result has the four expected maps: kernel to post, post to kernel, post to pre
        # (pointing to the original graph's node names only) and pre to post.
        pre_grad_graph_id = 140156815043952
        post_to_pre_grad_nodes_json = {
            "add_tensor": [
                {
                    "from_node": [
                        {
                            "from_node": [
                                {
                                    "from_node": [],
                                    "graph_id": 140156815043952,
                                    "name": "linear",
                                }
                            ],
                            "graph_id": 140152856025632,
                            "name": "addmm",
                        }
                    ],
                    "graph_id": 140151961816272,
                    "name": "add",
                },
            ],
            "mm_default": [
                {
                    "from_node": [],
                    "graph_id": -1,
                    "name": "",
                },
                {
                    "from_node": [
                        {
                            "from_node": [
                                {
                                    "from_node": [],
                                    "graph_id": 140156815043952,
                                    "name": "linear",
                                }
                            ],
                            "graph_id": 140152856025632,
                            "name": "addmm",
                        }
                    ],
                    "graph_id": 140151961816272,
                    "name": "mm",
                },
            ],
            "permute": [
                {
                    "from_node": [],
                    "graph_id": 140156815043952,
                    "name": "linear",
                }
            ],
            "relu": [
                {
                    "from_node": [],
                    "graph_id": 140156815043952,
                    "name": "relu",
                }
            ],
        }
        triton_kernel_to_post_grad_json = {
            "triton_poi_fused_addmm_relu_sigmoid_0": ["relu", "add_tensor"]
        }

        result = create_mapping_pre_post_grad_nodes(
            pre_grad_graph_id,
            post_to_pre_grad_nodes_json,
        )
        result = {
            **result,
            **create_node_mapping_kernel_to_post_grad(
                triton_kernel_to_post_grad_json,
            ),
        }

        self.assertEqual(
            result,
            {
                "cppCodeToPost": {
                    "triton_poi_fused_addmm_relu_sigmoid_0": [
                        "relu",
                        "add_tensor",
                    ]
                },
                "postToCppCode": {
                    "add_tensor": ["triton_poi_fused_addmm_relu_sigmoid_0"],
                    "relu": ["triton_poi_fused_addmm_relu_sigmoid_0"],
                },
                "postToPre": {
                    "add_tensor": ["linear"],
                    "mm_default": ["linear"],
                    "permute": ["linear"],
                    "relu": ["relu"],
                },
                "preToPost": {
                    "linear": ["add_tensor", "mm_default", "permute"],
                    "relu": ["relu"],
                },
            },
        )


class TestProvenanceTracingNodeMeta(TestCase):
    # [CLASS DOC] Tests that node metadata such as stack traces survives post grad passes.
    def get_node_with_target(self, gm, target):
        # [FUNCTION DOC] HELPER: Returns the first node of the module's graph whose target
        # equals the given target; raises StopIteration if none exists.
        """
        Return first node in gm with target
        """
        return next(iter([node for node in gm.graph.nodes if node.target == target]))

    @requires_gpu_and_triton  # test only works for cuda pattern matcher
    def test_pattern_matcher_transfer_meta(self):
        # [FUNCTION DOC] Exports a small linear/relu/sigmoid model on GPU, records the stack
        # trace of the addmm node, runs the post grad passes (which decompose addmm), then
        # asserts the resulting mm and add nodes carry the same stack trace. Sets the fake mode
        # on the virtualized context.
        """
        Test that stack trace is transferred when node is decomposed in post_grad_passes
        """

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear, relu, sigmoid model multiplied by three, defined
            # locally.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a linear layer (10 to 16), relu and sigmoid
                # modules.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear, relu and sigmoid and multiplies by
                # three.
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                return x * 3

        x = torch.randn(8, 10).to(GPU_TYPE)
        example_inputs = (x,)
        model = Model().to(GPU_TYPE)

        # mimic the before_post_grad graph
        ep = torch.export.export(model, example_inputs).run_decompositions()
        gm = ep.module()

        # Set fake mode for V
        fake_inputs = [
            node.meta.get("val") for node in gm.graph.nodes if node.op == "placeholder"
        ]
        fake_mode = detect_fake_mode(fake_inputs)
        V.set_fake_mode(fake_mode)

        addmm_node = self.get_node_with_target(gm, torch.ops.aten.addmm.default)
        stack_trace = addmm_node.meta["stack_trace"]

        post_grad_passes(gm, True)  # for this test is_inference doesn't matter

        mm_node = self.get_node_with_target(gm, torch.ops.aten.mm.default)
        add_node = self.get_node_with_target(gm, torch.ops.aten.add.Tensor)

        self.assertEqual(add_node.meta["stack_trace"], stack_trace)
        self.assertEqual(mm_node.meta["stack_trace"], stack_trace)


class ProvenanceArtifactFilter(logging.Filter):
    # [CLASS DOC] HELPER: Logging filter that accepts only structured trace records carrying an
    # artifact named for inductor kernel stack traces; records without artifact metadata are
    # rejected.
    def filter(self, record):
        # [FUNCTION DOC] HELPER: Returns True only if the record metadata has an artifact entry
        # whose name is the kernel stack trace artifact; otherwise False.
        if "artifact" in record.metadata:
            return (
                record.metadata["artifact"]["name"]
                == "inductor_provenance_tracking_kernel_stack_traces"
            )
        return False


class StructuredTracePayloadFormatter(logging.Formatter):
    # [CLASS DOC] HELPER: Logging formatter that outputs just the structured trace payload text,
    # stripped of surrounding whitespace.
    def format(self, record):
        # [FUNCTION DOC] HELPER: Returns the stripped payload of the record.
        return record.payload.strip()


class TestProvenanceTracingStackTraces(TestCase):
    # [CLASS DOC] Tests provenance level two kernel stack trace capture through the structured
    # trace log, plus kernel information JSON generation and the fake IR extern kernel metadata.
    @contextlib.contextmanager
    def _setup_provenance_capture(self):
        # [FUNCTION DOC] HELPER: Context manager that attaches a handler (with the stack trace
        # artifact filter and payload formatter) to the trace logger, yields an in-memory buffer
        # receiving the payload text, and removes the handler on exit.
        """Helper to turn on and capture the 'inductor_tlparse_runtime' structured trace."""
        payload_buffer = io.StringIO()
        payload_handler = logging.StreamHandler(payload_buffer)
        payload_handler.setLevel(logging.DEBUG)
        payload_handler.setFormatter(StructuredTracePayloadFormatter())
        payload_handler.addFilter(ProvenanceArtifactFilter())
        trace_log.addHandler(payload_handler)
        try:
            yield payload_buffer
        finally:
            trace_log.removeHandler(payload_handler)

    def extract_code_line(self, s):
        # [FUNCTION DOC] HELPER: Returns the stripped source code line of a stack trace entry by
        # scanning lines from the end and skipping empty lines, lines starting with File, and
        # lines made only of spaces, tildes and carets (Python 3.13 annotations). Falls back to
        # the second to last line if nothing qualifies.
        # Extract the source code line from a stack trace entry.
        # Filter out empty lines, "File ..." lines, and caret annotation
        # lines (e.g. "~~^~~~~~") added in Python 3.13+.
        lines = s.split("\n")
        for line in reversed(lines):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("File "):
                continue
            if all(c in " ~^" for c in stripped):
                continue
            return stripped
        return lines[-2].strip()

    @torch._inductor.config.patch({"trace.provenance_tracking_level": 2})
    def test_tlparse_kernel_stack_traces_cpu(self):
        # [FUNCTION DOC] Compiles the four-input model on CPU twice (resetting dynamo and debug
        # handles each round) while capturing the stack trace artifact, and checks the kernel
        # keys and the source lines of the stack traces for each kernel match expectations
        # (order independent).
        model = Model4()
        example_inputs = (
            torch.randn(8, 10),
            torch.randn(10, 20),
            torch.randn(20, 30),
            torch.randn(10, 30),
        )

        expected = {
            "cpp_fused_mul_0:2": [
                "d = a * 3.14",
            ],
            "cpp_fused_gelu_relu_sigmoid_threshold_backward_1:4": [
                "z = torch.nn.functional.gelu(y)",
                "x = self.relu(x)",
                "x = self.sigmoid(x)",
            ],
            "extern_kernels.addmm:1": [
                "x = self.fc1(x)",
            ],
            "extern_kernels.addmm:3": [
                "y = torch.addmm(c, d, b)",
            ],
        }

        compiled = torch.compile(model)
        for _ in range(2):
            torch._dynamo.reset()
            reset_inductor_kernel_provenance_debug_handle()
            with self._setup_provenance_capture() as payload_buffer:
                compiled = torch.compile(model)
                compiled(*example_inputs)
                payload_content = payload_buffer.getvalue().strip()
                data = json.loads(payload_content)
                self.assertEqual(set(data.keys()), set(expected.keys()))
                for key, expected_lines in expected.items():
                    actual_lines = [self.extract_code_line(s) for s in data[key]]
                    self.assertEqual(
                        sorted(actual_lines),
                        sorted(expected_lines),
                        lambda msg: f"{msg}\nMismatch for key: {key}",
                    )

    @torch._inductor.config.patch({"trace.provenance_tracking_level": 2})
    @requires_gpu_and_triton
    def test_tlparse_kernel_stack_traces(self):
        # [FUNCTION DOC] Same as the CPU stack trace test but on the GPU: expected kernels are
        # Triton and extern mm kernels with their stack trace source lines. Needs GPU and
        # Triton.
        device = GPU_TYPE
        model = Model4().to(device)
        x = torch.randn(8, 10).to(device)
        a = torch.randn(10, 20).to(device)
        b = torch.randn(20, 30).to(device)
        c = _bias_like_addmm_input(device)
        example_inputs = (x, a, b, c)

        expected = {
            "triton_poi_fused_addmm_relu_sigmoid_threshold_backward_2:5": [
                "x = self.sigmoid(x)",
                "x = self.fc1(x)",
                "x = self.relu(x)",
            ],
            "triton_poi_fused_mul_0:2": [
                "d = a * 3.14",
            ],
            "triton_poi_fused_addmm_gelu_1:4": [
                "z = torch.nn.functional.gelu(y)",
                "y = torch.addmm(c, d, b)",
            ],
            "extern_kernels.mm:3": [
                "y = torch.addmm(c, d, b)",
            ],
            "extern_kernels.mm:1": [
                "x = self.fc1(x)",
            ],
        }

        compiled = torch.compile(model)
        # should produce the same provenance if there's cache hit
        for _ in range(2):
            # reset cache
            torch._dynamo.reset()
            reset_inductor_kernel_provenance_debug_handle()
            with self._setup_provenance_capture() as payload_buffer:
                compiled = torch.compile(model)
                compiled(*example_inputs)
                payload_content = payload_buffer.getvalue().strip()
                data = json.loads(payload_content)
                self.assertEqual(set(data.keys()), set(expected.keys()))
                for key, expected_lines in expected.items():
                    actual_lines = [self.extract_code_line(s) for s in data[key]]
                    self.assertEqual(
                        sorted(actual_lines),
                        sorted(expected_lines),
                        lambda msg: f"{msg}\nMismatch for key: {key}",
                    )

    @torch._inductor.config.patch(
        {"trace.provenance_tracking_level": 2, "max_autotune_gemm_backends": "ATEN"}
    )
    @requires_gpu_and_triton
    def test_deferred_triton_kernels(self):
        # [FUNCTION DOC] Compiles a linear layer with max autotune on GPU, checks the captured
        # stack trace artifact contains the source line of the call, and that the generated code
        # contains the topologically sorted source nodes comment and provenance debug handles.
        def foo(m, inp):
            # [FUNCTION DOC] HELPER: Applies the module to the input and returns the result.
            a = m(inp)
            return a

        foo_c = torch.compile(mode="max-autotune-no-cudagraphs")(foo)

        m = torch.nn.Linear(512, 512, bias=True).half().to(GPU_TYPE)
        inp = torch.rand([1, 512]).half().to(GPU_TYPE)

        with self._setup_provenance_capture() as payload_buffer:
            with torch.no_grad():
                _, out_code = run_and_get_code(foo_c, m, inp)
            payload_content = payload_buffer.getvalue().strip()
            data = json.loads(payload_content)
            self.assertTrue("a = m(inp)" in str(data))

            # Check that debug handle is in the output code
            FileCheck().check("Topologically Sorted Source Nodes: [linear]").check(
                "[Provenance debug handles]"
            ).run(out_code[0])

    def _check_kernel_information_json(self, kernel_info, expected_kernels):
        # [FUNCTION DOC] HELPER: Asserts the kernel information is a mapping containing every
        # expected kernel name and that each entry has list-of-string stack traces, post grad
        # nodes and pre grad nodes and an extern semantic key that is a string or None.
        """Validate kernel information JSON structure and content."""
        self.assertIsInstance(kernel_info, dict)

        for expected in expected_kernels:
            self.assertIn(
                expected,
                kernel_info,
                lambda msg: f"{msg}\nExpected kernel {expected} not found in {list(kernel_info)}",
            )

        for data in kernel_info.values():
            self.assertIsInstance(data, dict)
            for field in ["stack_traces", "post_grad_nodes", "pre_grad_nodes"]:
                self.assertIn(field, data)
                self.assertIsInstance(data[field], list)
                for item in data[field]:
                    self.assertIsInstance(item, str)
            self.assertIsInstance(data["extern_semantic_key"], (str, type(None)))

    @requires_gpu_and_triton
    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_kernel_information_generation(self):
        # [FUNCTION DOC] Exports and packages the four-input model with AOT inductor on GPU,
        # extracts the package, and checks the kernel information JSON exists with the expected
        # kernels, stack trace lines, and pre and post grad nodes. Also checks semantic keys and
        # shape and dtype metadata for the extern mm kernels (and a None key for the Triton
        # kernel).
        """Test basic kernel information generation in AOTI packages."""

        model = Model4().to(GPU_TYPE)
        x = torch.randn(8, 10, device=GPU_TYPE)
        a = torch.randn(10, 20, device=GPU_TYPE)
        b = torch.randn(20, 30, device=GPU_TYPE)
        c = _bias_like_addmm_input(GPU_TYPE)
        inputs = (x, a, b, c)

        with tempfile.TemporaryDirectory() as temp_dir:
            ep = torch.export.export(model, inputs, strict=False)
            pt2_file = os.path.join(temp_dir, "model.pt2")
            reset_inductor_kernel_provenance_debug_handle()
            torch._inductor.aoti_compile_and_package(ep, package_path=pt2_file)

            # Extract and check kernel_information.json exists in the package
            with zipfile.ZipFile(pt2_file, "r") as zip_ref:
                zip_ref.extractall(temp_dir)

            json_path = os.path.join(
                temp_dir,
                "model",
                "data",
                "aotinductor",
                "model",
                "kernel_information.json",
            )
            self.assertTrue(
                os.path.exists(json_path),
                lambda msg: f"{msg}\nkernel_information.json not found in extracted package at {json_path}",
            )

            with open(json_path) as f:
                kernel_info = json.load(f)

            expected = {
                "triton_poi_fused_addmm_relu_sigmoid_0:2": {
                    "stack_traces": [
                        "x = self.sigmoid(x)",
                        "x = self.fc1(x)",
                        "x = self.relu(x)",
                    ],
                    "post_grad_nodes": ["sigmoid", "relu", "add_tensor_1"],
                    "pre_grad_nodes": [
                        "sigmoid_default",
                        "relu_default",
                        "linear_default",
                    ],
                },
                "triton_poi_fused_mul_1:3": {
                    "stack_traces": [
                        "d = a * 3.14",
                    ],
                    "post_grad_nodes": ["mul"],
                    "pre_grad_nodes": ["mul_tensor"],
                },
                "triton_poi_fused_addmm_gelu_2:5": {
                    "stack_traces": [
                        "z = torch.nn.functional.gelu(y)",
                        "y = torch.addmm(c, d, b)",
                    ],
                    "post_grad_nodes": [
                        "mul_3",
                        "mul_1",
                        "add_tensor",
                        "add",
                        "erf",
                        "mul_2",
                    ],
                    "pre_grad_nodes": ["gelu_default", "addmm_default"],
                },
                f"aoti_torch_{GPU_TYPE}_mm_out:1": {
                    "stack_traces": [
                        "x = self.fc1(x)",
                    ],
                    "post_grad_nodes": ["mm_default_1"],
                    "pre_grad_nodes": ["linear_default"],
                },
                f"aoti_torch_{GPU_TYPE}_mm_out:4": {
                    "stack_traces": [
                        "y = torch.addmm(c, d, b)",
                    ],
                    "post_grad_nodes": ["mm_default"],
                    "pre_grad_nodes": ["addmm_default"],
                },
            }

            self._check_kernel_information_json(kernel_info, expected.keys())

            self.assertEqual(set(kernel_info.keys()), set(expected.keys()))
            for key, data in expected.items():
                all_lines = ",".join(kernel_info[key]["stack_traces"])
                for s in data["stack_traces"]:
                    self.assertTrue(s in all_lines)

                self.assertEqual(
                    sorted(kernel_info[key]["pre_grad_nodes"]),
                    sorted(data["pre_grad_nodes"]),
                    lambda msg: f"{msg}\nMismatch for key: {key}",
                )

                self.assertEqual(
                    sorted(kernel_info[key]["post_grad_nodes"]),
                    sorted(data["post_grad_nodes"]),
                    lambda msg: f"{msg}\nMismatch for key: {key}",
                )

            # extern_semantic_key + shape metadata for extern kernels
            triton_poi_0 = kernel_info["triton_poi_fused_addmm_relu_sigmoid_0:2"]
            self.assertIsNone(triton_poi_0["extern_semantic_key"])

            mm_out_1 = kernel_info[f"aoti_torch_{GPU_TYPE}_mm_out:1"]
            # Single pre_grad_node "linear_default" -> extern_semantic_key fallback
            self.assertEqual(mm_out_1["extern_semantic_key"], "linear_default")
            self.assertEqual(mm_out_1["input_shapes"], [[8, 10], [10, 16]])
            self.assertEqual(
                mm_out_1["input_dtypes"], ["torch.float32", "torch.float32"]
            )
            self.assertEqual(mm_out_1["output_shape"], [8, 16])
            self.assertEqual(mm_out_1["output_dtype"], "torch.float32")

            mm_out_4 = kernel_info[f"aoti_torch_{GPU_TYPE}_mm_out:4"]
            self.assertEqual(mm_out_4["extern_semantic_key"], "addmm_default")
            self.assertEqual(mm_out_4["input_shapes"], [[10, 20], [20, 30]])
            self.assertEqual(
                mm_out_4["input_dtypes"], ["torch.float32", "torch.float32"]
            )
            self.assertEqual(mm_out_4["output_shape"], [10, 30])
            self.assertEqual(mm_out_4["output_dtype"], "torch.float32")

    @torch._inductor.config.patch("trace.provenance_tracking_level", 0)
    def test_no_kernel_information_without_provenance_tracking(self):
        # [FUNCTION DOC] With provenance tracking at level zero, packages a trivial model with
        # AOT inductor and checks no kernel information JSON appears in the extracted package.
        """Test that kernel_information.json is not generated without provenance tracking."""

        class SimpleModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model that multiplies its input by two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by 2.0.
                return x * 2.0

        model = SimpleModel()
        x = torch.randn(4, 8)

        # Compile with AOTI but without provenance tracking
        with tempfile.TemporaryDirectory() as temp_dir:
            ep = torch.export.export(model, (x,), strict=False)
            pt2_file = os.path.join(temp_dir, "model.pt2")
            torch._inductor.aoti_compile_and_package(ep, package_path=pt2_file)

            # Extract and check kernel_information.json was NOT created in the package
            extract_dir = os.path.join(temp_dir, "extracted")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(pt2_file, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            expected_json_path = os.path.join(extract_dir, "kernel_information.json")
            self.assertFalse(
                os.path.exists(expected_json_path),
                "kernel_information.json should not exist in package when provenance tracking is disabled",
            )

    def test_create_kernel_information_json_function(self):
        # [FUNCTION DOC] Calls the kernel information JSON builder with no recorded provenance
        # and checks it returns an empty mapping.
        """Test the create_kernel_information_json function directly."""
        # Test with empty state
        result = create_kernel_information_json()
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 0)  # Should be empty with no provenance data

    def test_reset_provenance_globals_preserves_kernel_information_jsons(self):
        # [FUNCTION DOC] Temporarily replaces the global kernel information store content with
        # an outer entry; inside the provenance globals reset context it sees the same outer
        # entry, adds an inner entry, and after the context both entries remain. The original
        # content is restored afterwards.
        kernel_information_jsons = get_kernel_information_jsons()
        previous = dict(kernel_information_jsons)
        kernel_information_jsons.clear()
        kernel_information_jsons["outer"] = {}
        try:
            with reset_provenance_globals():
                self.assertEqual(get_kernel_information_jsons(), {"outer": {}})
                get_kernel_information_jsons()["inner"] = {}
            self.assertEqual(get_kernel_information_jsons(), {"outer": {}, "inner": {}})
        finally:
            get_kernel_information_jsons().clear()
            get_kernel_information_jsons().update(previous)

    @unittest.skipIf(
        IS_MACOS,
        "MacOS generates different debug handles",
    )
    @torch._inductor.config.patch("trace.provenance_tracking_level", 1)
    def test_cpu_extern_kernel(self):
        # [FUNCTION DOC] Exports a convolution model on CPU, packages it with AOT inductor while
        # capturing the stack trace artifact, and checks that a convolution extern kernel name
        # is among the captured keys. Skipped on macOS where debug handles differ.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Model of one 2D convolution layer (16 to 33 channels, kernel
            # size 3).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the convolution layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the convolution to the input.
                return self.conv(x)

        model = Foo()
        x = torch.randn(20, 16, 50, 100)
        with self._setup_provenance_capture() as payload_buffer:
            reset_inductor_kernel_provenance_debug_handle()
            ep = torch.export.export(model, (x,))
            torch._inductor.aoti_compile_and_package(ep)
            payload_content = payload_buffer.getvalue().strip()
            data = json.loads(payload_content)

            keys = [k.split(":")[0] for k in data]
            self.assertTrue("aoti_torch_cpu_convolution" in keys)

    def test_create_kernel_information_json_with_synthetic_data(self):
        # [FUNCTION DOC] Fills the debug module globals (kernel to post grad nodes, stack
        # traces, post to pre grad mapping, extern kernel info) with synthetic data under a
        # reset context, builds the kernel information JSON, and checks: no semantic key or
        # shapes for a Triton kernel, an explicit semantic key with shapes taking precedence for
        # an extern kernel, a fallback semantic key from the single pre grad node for extern
        # kernels, no key for a non-extern kernel with a prefix only.
        """Test create_kernel_information_json with synthetic globals."""
        import torch._inductor.debug as debug_mod

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            # Populate globals with known data
            debug_mod._inductor_triton_kernel_to_post_grad_node_info = {
                "triton_poi_fused_add_mul_0:1": ["add", "mul"],
                "aoti_torch_cuda_mm_out:2": ["mm_default"],
                "extern_kernels.addmm:3": ["addmm_default"],
                "custom_backend.mm:4": ["custom_mm_default"],
                "aoti_torch_cuda_add_out:5": ["add_default"],
            }
            debug_mod._inductor_kernel_stack_trace = {
                "triton_poi_fused_add_mul_0:1": ["File test.py, line 10"],
                "aoti_torch_cuda_mm_out:2": ["File test.py, line 20"],
                "extern_kernels.addmm:3": ["File test.py, line 30"],
                "custom_backend.mm:4": ["File test.py, line 40"],
                "aoti_torch_cuda_add_out:5": ["File test.py, line 50"],
            }
            debug_mod._inductor_post_to_pre_grad_nodes = {
                "postToPre": {
                    "add": ["add_1"],
                    "mul": ["mul_1"],
                    "mm_default": ["linear"],
                    "addmm_default": ["addmm"],
                    "custom_mm_default": ["custom_linear"],
                    "add_default": ["prefix_only_add"],
                }
            }
            # Consolidated per-kernel extern info: is_extern flag, explicit
            # semantic key (simulates FP8 bridge), and shape metadata.
            debug_mod._inductor_kernel_extern_info = {
                "aoti_torch_cuda_mm_out:2": debug_mod._KernelExternInfo(
                    is_extern=True,
                    semantic_key="linear_42",
                    shapes={
                        "input_shapes": [[8, 10], [10, 16]],
                        "input_dtypes": ["torch.float32", "torch.float32"],
                        "output_shape": [8, 16],
                        "output_dtype": "torch.float32",
                    },
                ),
                "extern_kernels.addmm:3": debug_mod._KernelExternInfo(is_extern=True),
                "custom_backend.mm:4": debug_mod._KernelExternInfo(is_extern=True),
            }

            result = create_kernel_information_json()

            # Triton pointwise kernel: no extern_semantic_key, no shape metadata.
            k1 = result["triton_poi_fused_add_mul_0:1"]
            self.assertIsNone(k1["extern_semantic_key"])
            self.assertNotIn("input_shapes", k1)

            # Extern kernel with explicit semantic key (FP8 bridge precedence)
            # and merged shape metadata.
            k2 = result["aoti_torch_cuda_mm_out:2"]
            self.assertEqual(k2["extern_semantic_key"], "linear_42")
            self.assertEqual(k2["input_shapes"], [[8, 10], [10, 16]])
            self.assertEqual(k2["output_shape"], [8, 16])
            self.assertEqual(k2["output_dtype"], "torch.float32")

            # Extern kernel with single pre_grad_node fallback semantic key.
            k3 = result["extern_kernels.addmm:3"]
            self.assertEqual(k3["extern_semantic_key"], "addmm")
            self.assertNotIn("input_shapes", k3)

            # Extern fallback comes from explicit tracking, not kernel-name prefixes.
            k4 = result["custom_backend.mm:4"]
            self.assertEqual(k4["extern_semantic_key"], "custom_linear")

            k5 = result["aoti_torch_cuda_add_out:5"]
            self.assertIsNone(k5["extern_semantic_key"])

    def test_extern_semantic_key_extracted_from_origin_node_meta(self):
        # [FUNCTION DOC] Checks that a semantic key stored in the metadata of a kernel's origin
        # node is recorded when provenance tracing is set up for an extern kernel, appears in
        # the debug module extern info, and shows up in the kernel information JSON.
        """extern_semantic_key stamped on origin_node.meta (e.g. by the FP8
        lowering pass) is extracted into the provenance globals and JSON."""
        import torch._inductor.debug as debug_mod
        import torch._inductor.ir as ir_mod

        class FakeOriginNode:
            # [CLASS DOC] HELPER: Fake origin node with a name and a metadata mapping.
            def __init__(self, name, meta) -> None:
                # [FUNCTION DOC] HELPER: Stores the name and metadata.
                self.name = name
                self.meta = meta

        class FakeExternKernel(ir_mod.ExternKernel):
            # [CLASS DOC] HELPER: Fake extern kernel (subclass of the IR extern kernel) with no
            # inputs, the given origin node and no origins, bypassing the real constructor.
            def __init__(self, origin_node) -> None:
                # [FUNCTION DOC] HELPER: Sets empty inputs, the origin node and an empty origins
                # list without calling the base constructor.
                self.inputs = []
                self.origin_node = origin_node
                self.origins = []

            def has_tensor_output(self):
                # [FUNCTION DOC] HELPER: Reports that the kernel has no tensor output.
                return False

            def get_stack_traces(self):
                # [FUNCTION DOC] HELPER: Returns an empty list of stack traces.
                return []

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            extern = FakeExternKernel(
                FakeOriginNode("linear", {"extern_semantic_key": "my_fp8_key"})
            )
            handle = set_kernel_post_grad_provenance_tracing(
                extern, "aoti_torch_cuda_mm_out", is_extern=True
            )
            kernel_name = f"aoti_torch_cuda_mm_out:{handle}"
            info = debug_mod._inductor_kernel_extern_info[kernel_name]
            self.assertTrue(info.is_extern)
            self.assertEqual(info.semantic_key, "my_fp8_key")

            result = create_kernel_information_json()
            self.assertEqual(result[kernel_name]["extern_semantic_key"], "my_fp8_key")

    def test_extern_kernel_metadata_accepts_tensor_like_ir_inputs(self):
        # [FUNCTION DOC] Builds fake extern kernels whose inputs are tensor-like IR nodes, a
        # list of them and a non tensor node, and checks the kernel information JSON metadata:
        # input shapes and dtypes only for tensor inputs (including symbolic dims), output shape
        # and dtype when the kernel has a tensor output, and none for a kernel without tensor
        # output. Also checks the handles returned are sequential starting at one.
        import torch._inductor.ir as ir_mod

        class FakeTensorIRNode(ir_mod.IRNode):
            # [CLASS DOC] HELPER: Fake IR node that behaves as a tensor with a given size and
            # dtype.
            def __init__(self, size, dtype) -> None:
                # [FUNCTION DOC] HELPER: Stores the size and dtype without calling the base
                # constructor.
                self._size = size
                self._dtype = dtype

            def has_tensor_output(self):
                # [FUNCTION DOC] HELPER: Reports that this node has a tensor output.
                return True

            def maybe_get_size(self):
                # [FUNCTION DOC] HELPER: Returns the stored size.
                return self._size

            def maybe_get_dtype(self):
                # [FUNCTION DOC] HELPER: Returns the stored dtype.
                return self._dtype

        class FakeNonTensorIRNode(ir_mod.IRNode):
            # [CLASS DOC] HELPER: Fake IR node that is not a tensor.
            def has_tensor_output(self):
                # [FUNCTION DOC] HELPER: Reports that this node has no tensor output.
                return False

        class FakeExternKernel(ir_mod.ExternKernel):
            # [CLASS DOC] HELPER: Fake extern kernel with configurable inputs, output size and
            # output dtype; there is a tensor output exactly when an output size was given. No
            # origin node or origins.
            def __init__(self, inputs, output_size=None, output_dtype=None) -> None:
                # [FUNCTION DOC] HELPER: Stores the inputs, no origin node, an empty origins
                # list and the output size and dtype without calling the base constructor.
                self.inputs = inputs
                self.origin_node = None
                self.origins = []
                self._output_size = output_size
                self._output_dtype = output_dtype

            def has_tensor_output(self):
                # [FUNCTION DOC] HELPER: Returns True when an output size was provided.
                return self._output_size is not None

            def maybe_get_size(self):
                # [FUNCTION DOC] HELPER: Returns the stored output size.
                return self._output_size

            def maybe_get_dtype(self):
                # [FUNCTION DOC] HELPER: Returns the stored output dtype.
                return self._output_dtype

            def get_stack_traces(self):
                # [FUNCTION DOC] HELPER: Returns a fixed one element list holding a fake stack
                # trace string.
                return ["File fake.py, line 1"]

        with (
            config.patch("trace.provenance_tracking_level", 1),
            reset_provenance_globals(),
        ):
            extern = FakeExternKernel(
                [
                    FakeTensorIRNode([2, 4], torch.float32),
                    [FakeTensorIRNode([3, "s0"], torch.bfloat16)],
                    FakeNonTensorIRNode(),
                ],
                output_size=[2, "s0"],
                output_dtype=torch.float16,
            )
            handle = set_kernel_post_grad_provenance_tracing(
                extern, "custom_backend.mm", is_extern=True
            )
            self.assertEqual(handle, 1)

            non_tensor_output = FakeExternKernel(
                [FakeTensorIRNode([7], torch.int64)],
            )
            handle = set_kernel_post_grad_provenance_tracing(
                non_tensor_output, "custom_backend.multi", is_extern=True
            )
            self.assertEqual(handle, 2)

            result = create_kernel_information_json()

            metadata = result["custom_backend.mm:1"]
            self.assertEqual(metadata["input_shapes"], [[2, 4], [3, "s0"]])
            self.assertEqual(
                metadata["input_dtypes"], ["torch.float32", "torch.bfloat16"]
            )
            self.assertEqual(metadata["output_shape"], [2, "s0"])
            self.assertEqual(metadata["output_dtype"], "torch.float16")

            non_tensor_output_metadata = result["custom_backend.multi:2"]
            self.assertEqual(non_tensor_output_metadata["input_shapes"], [[7]])
            self.assertEqual(
                non_tensor_output_metadata["input_dtypes"], ["torch.int64"]
            )
            self.assertNotIn("output_shape", non_tensor_output_metadata)
            self.assertNotIn("output_dtype", non_tensor_output_metadata)


class ProvenanceTracingKernelContextTemplate:
    # [CLASS DOC] Template of tests that is copied onto the CPU and GPU test classes by the copy
    # helper; uses the device attribute of the target class. Not itself a test case.
    def test_jit_inductor_with_flag(self):
        # [FUNCTION DOC] Builds a four-input model on the device and compiles and runs it with
        # the kernel profile option enabled; checks only that it runs without error.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear, relu, sigmoid branch plus scaled addmm and gelu branch
            # model.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a linear layer (10 to 16), relu and sigmoid
                # modules.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, a, b, c):
                # [FUNCTION DOC] HELPER: Runs the linear, relu, sigmoid branch and the
                # addmm/gelu branch; returns both results.
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                d = a * 3.14
                y = torch.addmm(c, d, b)
                z = torch.nn.functional.gelu(y)
                return x, z

        model = Model().to(self.device)
        x = torch.randn(8, 10).to(self.device)
        a = torch.randn(10, 20).to(self.device)
        b = torch.randn(20, 30).to(self.device)
        if self.device == "cpu":
            c = torch.randn(10, 30).to(self.device)
        else:
            c = _bias_like_addmm_input(self.device)
        example_inputs = (x, a, b, c)

        with config.patch(
            {
                "cpp.enable_kernel_profile": True,
            }
        ):
            torch.compile(model)(*example_inputs)

    @unittest.skipIf(sys.platform == "darwin", "Different kernel names on MacOS")
    def test_aoti_python_stack_traces(self):
        # [FUNCTION DOC] Exports a four-input model and compiles it with AOT inductor several
        # ways, inspecting generated C++: by default no kernel context guard or header; with
        # profiling on and the guard option off, none either and the package runs correctly;
        # with the guard on, the header, thread local variable and guard lines for each extern
        # and generated kernel (device specific names) are present and the loaded package
        # matches eager. Skipped on macOS.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Linear, relu, sigmoid branch plus scaled addmm and gelu branch
            # model.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates a linear layer (10 to 16), relu and sigmoid
                # modules.
                super().__init__()
                self.fc1 = torch.nn.Linear(10, 16)
                self.relu = torch.nn.ReLU()
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, x, a, b, c):
                # [FUNCTION DOC] HELPER: Runs the linear, relu, sigmoid branch and the
                # addmm/gelu branch; returns both results.
                x = self.fc1(x)
                x = self.relu(x)
                x = self.sigmoid(x)
                d = a * 3.14
                y = torch.addmm(c, d, b)
                z = torch.nn.functional.gelu(y)
                return x, z

        x = torch.randn(8, 10).to(self.device)
        a = torch.randn(10, 20).to(self.device)
        b = torch.randn(20, 30).to(self.device)
        if self.device == "cpu":
            c = torch.randn(10, 30).to(self.device)
        else:
            c = _bias_like_addmm_input(self.device)
        example_inputs = (x, a, b, c)
        model = Model().to(self.device)

        ep = torch.export.export(model, example_inputs)
        _, code = run_and_get_cpp_code(torch._inductor.aoti_compile_and_package, ep)

        self.assertTrue("KernelContextGuard" not in code)
        FileCheck().check_not(
            "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
        ).check_not("thread_local KernelContext* tls_kernel_context = nullptr;").run(
            code
        )

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "cpp.enable_kernel_profile": True,
                "cpp.enable_kernel_context_guard": False,
            }
        ):
            package_path, code = run_and_get_cpp_code(
                torch._inductor.aoti_compile_and_package, ep
            )

            FileCheck().check_not(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            ).check_not(
                "thread_local KernelContext* tls_kernel_context = nullptr;"
            ).check_not("KernelContextGuard").run(code)

            compiled_model = torch._inductor.aoti_load_package(package_path)
            result = compiled_model(*example_inputs)
            self.assertEqual(result, model(*example_inputs))

        with config.patch(
            {
                "trace.provenance_tracking_level": 1,
                "cpp.enable_kernel_profile": True,
                "cpp.enable_kernel_context_guard": True,
            }
        ):
            package_path, code = run_and_get_cpp_code(
                torch._inductor.aoti_compile_and_package, ep
            )

            FileCheck().check(
                "#include <torch/csrc/inductor/aoti_runtime/kernel_context_tls.h>"
            ).check("thread_local KernelContext* tls_kernel_context = nullptr;").run(
                code
            )

            if self.device == "cuda" or self.device == "xpu":
                device_type = torch.accelerator.current_accelerator().type
                FileCheck().check(
                    f"""KernelContextGuard _ctx("aoti_torch_{device_type}_mm_out", R"("""
                ).check(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_{device_type}_mm_out("
                ).check(
                    """KernelContextGuard _ctx("triton_poi_fused_addmm_relu_sigmoid_0", R"("""
                ).check("call_triton_poi_fused_addmm_relu_sigmoid_0(").check(
                    """KernelContextGuard _ctx("triton_poi_fused_mul_1", R"("""
                ).check("call_triton_poi_fused_mul_1(").check(
                    f"""KernelContextGuard _ctx("aoti_torch_{device_type}_mm_out", R"""
                ).check(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_{device_type}_mm_out("
                ).check(
                    """ KernelContextGuard _ctx("triton_poi_fused_addmm_gelu_2", R"("""
                ).check("call_triton_poi_fused_addmm_gelu_2(").run(code)
            else:
                FileCheck().check(
                    """KernelContextGuard _ctx("aoti_torch_cpu_addmm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(").check(
                    """KernelContextGuard _ctx("cpp_fused_mul_relu_sigmoid_0", R"("""
                ).check("cpp_fused_mul_relu_sigmoid_0(").check(
                    """KernelContextGuard _ctx("aoti_torch_cpu_addmm_out", R"("""
                ).check("AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(").check(
                    """ KernelContextGuard _ctx("cpp_fused_gelu_1", R"("""
                ).check("cpp_fused_gelu_1(").run(code)

            compiled_model = torch._inductor.aoti_load_package(package_path)
            result = compiled_model(*example_inputs)
            self.assertEqual(result, model(*example_inputs))


class TestProvenanceTracingKernelContextCpu(TestCase):
    # [CLASS DOC] CPU instantiation of the kernel context template tests; device is set to cpu.
    device = "cpu"


copy_tests(
    ProvenanceTracingKernelContextTemplate,
    TestProvenanceTracingKernelContextCpu,
    "cpu",
)


@unittest.skipIf(sys.platform == "darwin", "No CUDA on MacOS")
@unittest.skipIf(
    not torch.cuda.is_available() and not torch.xpu.is_available(), "No CUDA and no XPU"
)
class TestProvenanceTracingKernelContextGpu(TestCase):
    # [CLASS DOC] GPU instantiation of the kernel context template tests; device is the GPU
    # type. Skipped on macOS or when neither CUDA nor XPU is available.
    device = GPU_TYPE


copy_tests(
    ProvenanceTracingKernelContextTemplate,
    TestProvenanceTracingKernelContextGpu,
    GPU_TYPE,
)


if __name__ == "__main__":
    run_tests()
