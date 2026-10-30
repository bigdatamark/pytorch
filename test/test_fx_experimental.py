# Owner(s): ["module: fx"]
# ruff: noqa: F841

# [MODULE DOC] Tests for torch.fx.experimental: the device Partitioner (size based, sparse NN,
# cost aware, AOT based, host saturation), graph_manipulation helpers, conv/batchnorm fusion,
# the assert-rewriting tracer, the meta tracer, split_module and split_module_simple, argument
# and operator normalization, type annotation passes, to_folder export, attribute lifting,
# merge_matmul, type_matches, optimize_for_inference, and an OpInfo-driven normalization sweep.
# A translation validator test class is defined only when Z3 is available. Defines skip
# decorators for missing torchvision or MKLDNN (the torchvision import is attempted at import
# time and failure only sets a flag) and a helper tracing with the rewriting tracer. Runs
# run_tests when executed as main.
import functools
import math
import numbers
import operator
import pickle
import sys
import sympy
import tempfile
import typing
import unittest
from types import BuiltinFunctionType
from typing import NamedTuple, Optional, Union
from collections.abc import Callable

import torch
import torch.fx.experimental.meta_tracer
import torch.fx.experimental.optimization as optimization
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.experimental import merge_matmul
from torch.fx.experimental.accelerator_partitioner import Partitioner
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.normalize import NormalizeArgs, NormalizeOperators
from torch.fx.experimental.partitioner_utils import (
    Device,
    get_latency_of_partitioned_graph,
    get_partition_to_latency_mapping,
    NodeLatency,
    PartitionerConfig,
    PartitionMode,
)
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.experimental.schema_type_annotation import AnnotateTypesWithSchema
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
from torch.fx.operator_schemas import (
    _torchscript_type_to_python_type,
    create_type_hint,
    normalize_function,
    normalize_module,
    type_matches,
)
from torch.fx.passes import graph_manipulation
from torch.fx.passes.param_fetch import lift_lowering_attrs_to_nodes
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx._lazy_graph_module import _use_lazy_graph_module
from torch.fx.passes.split_module import split_module, split_module_simple
from torch.fx.passes.annotate_getitem_nodes import annotate_getitem_nodes
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_nn import module_tests, get_new_module_tests
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    TEST_Z3,
    run_tests,
    TestCase,
    TEST_WITH_CROSSREF,
)
from torch.testing._internal.jit_utils import JitTestCase
import torch.utils._pytree as pytree

try:
    import torchvision.models
    from torchvision.models import resnet18

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, "no torchvision")
skipIfNoMkldnn = unittest.skipIf(
    not (torch.backends.mkldnn.enabled and torch.backends.mkldnn.is_available()),
    "no MKLDNN",
)


def symbolic_trace_with_rewrite(root: torch.nn.Module | Callable) -> GraphModule:
    # [FUNCTION DOC] Traces a module or callable with RewritingTracer (which rewrites assert
    # statements into calls to the torch assert function) and wraps the resulting graph in a
    # GraphModule. A plain callable gets an empty Module as root.
    return GraphModule(
        root if isinstance(root, torch.nn.Module) else torch.nn.Module(),
        RewritingTracer().trace(root),
    )


class TestFXExperimental(JitTestCase):
    # [CLASS DOC] Main JitTestCase for experimental FX features. Stateless; each test_ method
    # builds a small module, traces it and checks partitioning, fusion, rewriting or other pass
    # behavior against the eager results. Several tests define nested helper modules.
    def test_find_single_partition(self):
        # [FUNCTION DOC] Partitions a tiny add module with three large devices and default config;
        # checks the partitioned module matches the traced one and that the single DAG node is
        # assigned to logical device 1 (the device with the most memory).
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds its two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: returns the sum of the two inputs.
                return a + b

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(1)
        b = torch.rand(1)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 150, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        if dag.nodes[0].logical_device_ids != [1]:
            raise AssertionError(f"expected logical_device_ids == [1], got {dag.nodes[0].logical_device_ids}")

    def test_lack_of_devices(self):
        # [FUNCTION DOC] Uses size based partitioning with two devices of memory 4 each, too small
        # for the graph, and asserts partition_graph raises a RuntimeError.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds its two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: returns the sum of the two inputs.
                return a + b

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [Device("dev_0", 4, 0), Device("dev_1", 4, 1)]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        catch_runtime_error = False
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        if not catch_runtime_error:
            raise AssertionError("expected RuntimeError")

    def test_large_node_error(self):
        # [FUNCTION DOC] Uses size based partitioning with five devices of memory 40 each; the
        # linear node is too large for any device so partition_graph must raise a RuntimeError.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a linear layer followed by a residual add.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the linear layer of size 4 by 4.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: applies the linear layer to the input and adds
                # the input back.
                linear = self.linear(a)
                add = linear + a
                return add

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 40, 0),
            Device("dev_1", 40, 0),
            Device("dev_2", 40, 0),
            Device("dev_3", 40, 0),
            Device("dev_4", 40, 0),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        catch_runtime_error = False
        try:
            ret = partitioner.partition_graph(traced, m, partitioner_config)
        except RuntimeError:
            catch_runtime_error = True
        if not catch_runtime_error:
            raise AssertionError("expected RuntimeError")

    def test_partition_node_manipulation(self):
        # [FUNCTION DOC] Partitions a three-add module into one partition and checks its used
        # memory is 112 bytes; then removes the node named add_2 from the partition and checks
        # used memory drops to 80 bytes.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with three chained adds, two of them with random tensors.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: adds the inputs, then adds two fresh random
                # tensors in sequence.
                add_1 = a + b
                add_2 = add_1 + torch.rand(4)
                add_3 = add_2 + torch.rand(4)
                return add_3

        m = TestModule()
        traced = symbolic_trace(m)
        a, b = torch.rand(4), torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [Device("dev_0", 1000, 0)]
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        partition = partitioner.partitions[0]
        if partition.used_mem_bytes != 112:
            raise AssertionError(f"expected used_mem_bytes == 112, got {partition.used_mem_bytes}")
        # Select add_2 node to remove
        selected_node = None
        for node in partition.nodes:
            if node.name == "add_2":
                selected_node = node
        partition.remove_node(selected_node)
        if partition.used_mem_bytes != 80:
            raise AssertionError(f"expected used_mem_bytes == 80, got {partition.used_mem_bytes}")

    def test_size_based_partition(self):
        # [FUNCTION DOC] Size based partitioning over three devices of memory 125; checks the
        # partitioned module matches the traced module and that DAG node i uses logical device i.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a linear layer and a constant tensor attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a linear layer and a random constant tensor.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.c = torch.rand(4)

            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: adds the inputs, applies the linear layer, then adds the
                # constant tensor.
                add_1 = a + b
                linear = self.linear(add_1)
                add_2 = linear + self.c
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        b = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b])
        partitioner = Partitioner()
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b), module_with_submodules(a, b))
        for i, node in enumerate(dag.nodes):
            if node.logical_device_ids != [i]:
                raise AssertionError(f"expected logical_device_ids == [{i}], got {node.logical_device_ids}")

    def test_partition_device_mapping(self):
        # [FUNCTION DOC] Size based partitioning with two devices (memory 120 and 160); checks
        # numerical equality with the traced module and that DAG node 1 maps to logical device 1
        # while all other nodes map to device 0.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module mixing random tensors, adds and a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: builds a graph of random-tensor adds plus a linear output
                # and returns the final sum.
                b = torch.rand(4)
                add_1 = a + b
                linear_1 = self.linear(add_1)
                add_2 = torch.rand(4) + a
                add_3 = add_2 + linear_1
                return add_3

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        partitioner = Partitioner()
        devices = [Device("dev_0", 120, 0), Device("dev_1", 160, 1)]
        partitioner_config = PartitionerConfig(devices, PartitionMode.size_based)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        for i, node in enumerate(dag.nodes):
            if i == 1:
                if node.logical_device_ids != [1]:
                    raise AssertionError(f"expected logical_device_ids == [1], got {node.logical_device_ids}")
            else:
                if node.logical_device_ids != [0]:
                    raise AssertionError(f"expected logical_device_ids == [0], got {node.logical_device_ids}")

    def test_sparse_nn_partition(self):
        # [FUNCTION DOC] Partitions a recommendation-style model (bottom MLP, five embedding bags,
        # top MLP) with the sparse NN mode over three devices of 33 million bytes; checks
        # numerical equality with the traced module and that the partitioned graph has 24 nodes.
        class MyRecommendationModule(torch.nn.Module):
            # [CLASS DOC] HELPER: toy recommendation model with a bottom MLP, a
            # list of EmbeddingBag layers with sparse gradients, and a top MLP over
            # the concatenated results.
            def create_mlp(self, num_of_layers: int, input_size: int, output_size: int):
                # [FUNCTION DOC] HELPER: builds and returns a ModuleList of the requested number
                # of linear plus ReLU pairs of the given input and output sizes.
                layers = torch.nn.ModuleList()
                for _ in range(num_of_layers):
                    ll = torch.nn.Linear(input_size, output_size)
                    layers.append(ll)
                    layers.append(torch.nn.ReLU())
                return layers

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: builds the bottom and top MLP Sequentials and a list of
                # five embedding bags (sizes 500000, three of 1000000, 500000, all dimension 4,
                # sum mode, sparse).
                super().__init__()
                layers = self.create_mlp(4, 4, 4)
                self.bottom_layers = torch.nn.Sequential(*layers)
                layers = self.create_mlp(3, 24, 24)
                self.top_layers = torch.nn.Sequential(*layers)
                self.embedding_layers = torch.nn.ModuleList()
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)
                for _ in range(3):
                    el = torch.nn.EmbeddingBag(1000000, 4, mode="sum", sparse=True)
                    self.embedding_layers.append(el)
                el = torch.nn.EmbeddingBag(500000, 4, mode="sum", sparse=True)
                self.embedding_layers.append(el)

            def forward(self, a, b, offset):
                # [FUNCTION DOC] HELPER: runs the bottom MLP on the dense input, looks up each
                # embedding bag (alternating between the shifted index input and fresh random
                # indices), concatenates dense and embedding outputs along dim 1 and applies the
                # top MLP.
                x = self.bottom_layers(a)
                y = []
                c = []
                for _ in range(len(self.embedding_layers)):
                    temp = torch.randint(10, (8,))
                    c.append(temp + b)
                for i in range(len(self.embedding_layers)):
                    if i % 2 == 0:
                        y.append(self.embedding_layers[i](c[i], offset))
                    else:
                        y.append(
                            self.embedding_layers[i](torch.randint(10, (8,)), offset)
                        )
                z = torch.cat([x] + y, dim=1)
                p = self.top_layers(z)
                return p

        m = MyRecommendationModule()
        a = torch.rand(2, 4)
        b = torch.randint(10, (8,))
        offset = torch.randint(1, (2,))
        traced = symbolic_trace(m)
        graph_manipulation.get_size_of_all_nodes(traced, [a, b, offset])
        devices = [
            Device("dev_0", 33000000, 0),
            Device("dev_1", 33000000, 1),
            Device("dev_2", 33000000, 2),
        ]
        partitioner_config = PartitionerConfig(devices, PartitionMode.sparse_nn)
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a, b, offset), module_with_submodules(a, b, offset))
        if len(module_with_submodules.graph.nodes) != 24:
            raise AssertionError(f"expected 24 nodes, got {len(module_with_submodules.graph.nodes)}")

    def test_partition_latency(self):
        # [FUNCTION DOC] Partitions a small module onto two devices, then computes per-partition
        # latency mappings and checks specific values (128, 80, 160 for partition 0 and 16, 32, 32
        # for the others) and that the critical path latency at transfer rate 2 is 208.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with adds of random tensors and a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: computes a chain of adds and a linear output and returns
                # the final add.
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        def get_node_to_latency_mapping(fx_module: GraphModule):
            # [FUNCTION DOC] HELPER: builds a mapping from each non-placeholder, non-output,
            # non- get_attr node to a NodeLatency using the node size info: if total size
            # equals output size the computation latency is twice the total size, otherwise it
            # is the output size.
            """Given a fx module, generate node latency for each node
            based on the size of each node
            """
            node_to_latency_mapping: dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 2.0 * node.size_bytes.total_size
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        devices = [Device("dev_0", 200, 0), Device("dev_1", 200, 1)]
        partitioner = Partitioner()
        partitioner_config = PartitionerConfig(devices)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        for p in partition_to_latency_mapping:
            if p.partition_id == 0:
                if partition_to_latency_mapping[p] != (128.0, 80.0, 160.0):
                    raise AssertionError(f"expected (128.0, 80.0, 160.0), got {partition_to_latency_mapping[p]}")
            else:
                if partition_to_latency_mapping[p] != (16.0, 32.0, 32.0):
                    raise AssertionError(f"expected (16.0, 32.0, 32.0), got {partition_to_latency_mapping[p]}")
        transfer_rate_bytes_per_sec = 2
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions, partition_to_latency_mapping, transfer_rate_bytes_per_sec
        )
        if critical_path_latency_sec != 208.0:
            raise AssertionError(f"expected 208.0, got {critical_path_latency_sec}")

    def test_cost_aware_partition(self):
        # [FUNCTION DOC] Cost aware partitioning over four devices of memory 125 with a supplied
        # node-to-latency mapping and transfer rate 2; checks numerical equality and that the
        # critical path latency of the partitioned graph is 160.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with several adds of random tensors and a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: computes adds and a linear output and returns the sum of
                # two branches.
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + torch.rand(4)
                add_4 = add_2 + linear_1
                add_5 = add_3 + add_4
                return add_5

        def get_node_to_latency_mapping(fx_module: GraphModule):
            # [FUNCTION DOC] HELPER: builds a node latency mapping for real compute nodes; the
            # computation latency is 1 when total size equals output size, otherwise the node
            # output size.
            node_to_latency_mapping: dict[Node, NodeLatency] = {}
            for node in fx_module.graph.nodes:
                if node.op not in {"output", "placeholder", "get_attr"}:
                    if node.size_bytes.total_size == node.size_bytes.output_size:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, 1
                        )
                    else:
                        node_to_latency_mapping[node] = NodeLatency(
                            node.size_bytes.total_size, node.size_bytes.output_size
                        )
            return node_to_latency_mapping

        m = MyModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        devices = [
            Device("dev_0", 125, 0),
            Device("dev_1", 125, 1),
            Device("dev_2", 125, 2),
            Device("dev_3", 125, 3),
        ]
        node_to_latency_mapping = get_node_to_latency_mapping(traced)
        partitioner_config = PartitionerConfig(
            devices,
            mode=PartitionMode.cost_aware,
            transfer_rate_bytes_per_sec=2,
            node_to_latency_mapping=node_to_latency_mapping,
        )
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(traced(a), module_with_submodules(a))
        partitions = partitioner.partitions
        partition_to_latency_mapping = get_partition_to_latency_mapping(
            partitions, node_to_latency_mapping
        )
        critical_path_latency_sec = get_latency_of_partitioned_graph(
            partitions,
            partition_to_latency_mapping,
            partitioner_config.transfer_rate_bytes_per_sec,
        )
        if critical_path_latency_sec != 160.0:
            raise AssertionError(f"expected 160.0, got {critical_path_latency_sec}")

    def test_aot_based_partition(self):
        # [FUNCTION DOC] AOT based partitioning where each compute node is its own partition
        # on logical device 0, supplied through explicit node-to-partition and partition-to-
        # device mappings; checks numerical equality and that every DAG node has size 48 bytes
        # and device 0.
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two constant tensors and two adds.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates two random constant tensors.
                super().__init__()
                self.b = torch.rand(4)
                self.c = torch.rand(4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: adds the first constant to the input, then adds the
                # second constant.
                add_1 = a + self.b
                add_2 = self.c + add_1
                return add_2

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        node_to_partition_id = {}
        partition_to_logical_devices = {}
        count = 0
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        for node in traced.graph.nodes:
            if node.op not in {"placeholder", "get_attr", "output"}:
                node_to_partition_id[node] = count
                partition_to_logical_devices[count] = [0]
                count += 1
        devices = [Device("dev_0", 200, 0)]
        partitioner_config = PartitionerConfig(
            devices=devices,
            mode=PartitionMode.aot_based,
            node_to_partition_mapping=node_to_partition_id,
            partition_to_logical_device_mapping=partition_to_logical_devices,
        )
        partitioner = Partitioner()
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        dag = ret.dag
        self.assertEqual(module_with_submodules(a), traced(a))
        for node in dag.nodes:
            if node.size_bytes != 48:
                raise AssertionError(f"expected size_bytes == 48, got {node.size_bytes}")
            if node.logical_device_ids != [0]:
                raise AssertionError(f"expected logical_device_ids == [0], got {node.logical_device_ids}")

    def test_replace_target_nodes_with(self):
        # [FUNCTION DOC] Traces an add module, checks it adds, then uses replace_target_nodes_with
        # to swap the add function for multiplication and checks the traced module now multiplies.
        class testModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds its two inputs.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: returns the sum of the two inputs.
                return a + b

        m = testModule()
        traced = symbolic_trace(m)
        input1 = torch.randn(1)
        input2 = torch.randn(1)
        if (input1 + input2) != traced(input1, input2):
            raise AssertionError("traced output mismatch")
        graph_manipulation.replace_target_nodes_with(
            fx_module=traced,
            old_op="call_function",
            old_target=operator.add,
            new_op="call_function",
            new_target=operator.mul,
        )
        if (input1 * input2) != traced(input1, input2):
            raise AssertionError("traced output mismatch after replacement")

    def test_saturate_host(self):
        # [FUNCTION DOC] Partitions a small model across six devices with host saturation on;
        # checks numerical equality, that two partitions exist, and that they are replicated to
        # extra devices (partition 0 on devices 0 and 4, partition 1 on devices 1 and 2).
        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with adds of random tensors and a linear layer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: computes a chain of adds plus a linear output and returns
                # the final add.
                add_1 = a + torch.rand(4)
                add_2 = add_1 + torch.rand(4)
                linear_1 = self.linear(add_1)
                add_3 = add_2 + linear_1
                add_4 = add_2 + add_3
                return add_4

        m = TestModule()
        traced = symbolic_trace(m)
        a = torch.rand(4)
        graph_manipulation.get_size_of_all_nodes(traced, [a])
        devices = [
            Device("dev_0", 200, 0),
            Device("dev_1", 200, 1),
            Device("dev_2", 100, 2),
            Device("dev_3", 100, 3),
            Device("dev_4", 200, 4),
            Device("dev_5", 100, 5),
        ]
        partitioner = Partitioner()
        # Without host saturation, the model will be split into two partitions.
        # dev_0 holds partition 0 of 192 bytes and dev_1 holds partition 1 of 48 bytes.
        partitioner_config = PartitionerConfig(devices, saturate_host=True)
        ret = partitioner.partition_graph(traced, m, partitioner_config)
        module_with_submodules = ret.module_with_submodules
        self.assertEqual(traced(a), module_with_submodules(a))

        partitions = partitioner.partitions
        self.assertEqual(len(partitions), 2)
        # With host saturation, partition 1 will be replicated to dev_4, and partition 2
        # will be replicated to dev_2.
        self.assertEqual(partitions[0].logical_device_ids, [0, 4])
        self.assertEqual(partitions[1].logical_device_ids, [1, 2])

    @skipIfNoTorchVision
    def test_conv_bn_fusion(self):
        # [FUNCTION DOC] Skipped without torchvision. Fuses conv and batchnorm in a traced
        # resnet18 in eval mode, asserts no BatchNorm2d modules remain, and checks the fused
        # output equals the original on a random image batch.
        rn18 = resnet18().eval()
        traced = symbolic_trace(rn18)
        fused = optimization.fuse(traced)

        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )

        N, C, H, W = 20, 3, 224, 224
        inp = torch.randn(N, C, H, W)

        self.assertEqual(fused(inp), rn18(inp))

    def test_conv_bn_fusion_not_running_state(self):
        # [FUNCTION DOC] Fusing a conv followed by a batchnorm that does not track running stats
        # must leave the batchnorm in place; also checks the fused module output equals the eager
        # model output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: convolution followed by a batchnorm with
            # running statistics disabled.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the conv and batchnorm layers.
                super().__init__()
                self.conv = torch.nn.Conv2d(32, 64, 3, stride=2)
                self.bn = torch.nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the conv then the batchnorm.
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = M().eval()

        traced = symbolic_trace(model)
        fused = optimization.fuse(traced)
        inp = torch.randn([1, 32, 50, 50])

        # bn need not be folded in conv
        self.assertTrue(
            any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )
        self.assertEqual(fused(inp), model(inp))

    def test_conv_bn_fusion_mixed_dtype(self):
        # [FUNCTION DOC] Fuses a bfloat16 conv without bias and a float32 batchnorm that
        # tracks running stats; asserts the batchnorm is folded away and fused output equals
        # the eager output.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: bfloat16 convolution followed by a batchnorm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the bfloat16 conv without bias and the batchnorm.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, dtype=torch.bfloat16)
                self.bn = torch.nn.BatchNorm2d(16, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the conv then the batchnorm.
                x = self.conv(x)
                x = self.bn(x)
                return x

        model = M().eval()

        traced = symbolic_trace(model)
        fused = optimization.fuse(traced)
        inp = torch.randn(1, 3, 64, 64, dtype=torch.bfloat16)

        self.assertTrue(
            all(not isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules())
        )
        self.assertEqual(fused(inp), model(inp))

    def test_call_to_assert_no_msg(self):
        # [FUNCTION DOC] Traces with the rewriting tracer a module containing a bare assert; lints
        # the graph, checks it contains a call to the torch assert function, checks equal inputs
        # run and unequal inputs raise an AssertionError, and the result equals eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module asserting its inputs are equal and returning their sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: asserts equality of the inputs and returns their sum.
                assert a == b  # noqa: S101
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target is torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_meta_tracer(self):
        # [FUNCTION DOC] Traces an embedding plus layer-norm module with the meta tracer using
        # meta tensors for batch sizes 15 and 35 (the model branches on batch size); checks traced
        # output matches eager and that a pickled and reloaded graph module still matches.
        class MetaTracerTestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: embedding, layer norm, then relu or sigmoid depending
            # on batch size.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the embedding (42 by 16) and layer norm layers.
                super().__init__()
                self.emb = torch.nn.Embedding(num_embeddings=42, embedding_dim=16)
                self.layernorm = torch.nn.LayerNorm(16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: embeds the indices, adds a range vector, applies layer
                # norm, then relu when the batch is smaller than 30, otherwise sigmoid.
                emb = self.emb(x)
                emb = emb + torch.arange(emb.shape[-1], dtype=torch.float, device=emb.device)
                lol = self.layernorm(emb)
                return torch.relu(lol) if lol.shape[0] < 30 else torch.sigmoid(lol)

        mttm = MetaTracerTestModule()
        for BS in [15, 35]:
            x = torch.zeros(BS, dtype=torch.long).random_(42)
            meta_args = {'x' : x.to(device='meta')}
            gm = torch.fx.experimental.meta_tracer.symbolic_trace(mttm, meta_args=meta_args)
            torch.testing.assert_close(gm(x), mttm(x))

            # Test serialization/deserialization
            with tempfile.TemporaryDirectory() as tmp_dir:
                with open(f'{tmp_dir}/meta_module.pkl', 'wb') as f:
                    pickle.dump(gm, f)

                with open(f'{tmp_dir}/meta_module.pkl', 'rb') as f:
                    loaded = pickle.load(f)

                torch.testing.assert_close(loaded(x), mttm(x))


    def test_call_to_assert_with_msg(self):
        # [FUNCTION DOC] Same as the bare-assert test but the assert has a message; checks the
        # graph has the torch assert call and that failing inputs raise an AssertionError matching
        # the message.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module asserting equality of inputs with a message and returning
            # the sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: asserts equality of the inputs with a message and returns
                # their sum.
                assert a == b, "test message"  # noqa: S101
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target is torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, "test message"):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_empty_msg(self):
        # [FUNCTION DOC] Rewriting-tracer test for an assert with an empty message string; checks
        # the graph has the torch assert call, equal inputs run, unequal inputs raise an
        # AssertionError, and the result equals eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module asserting input equality with an empty message and
            # returning the sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: asserts equality of the inputs with an empty message and
                # returns their sum.
                assert a == b, ""  # noqa: S101
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target is torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, ""):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    def test_call_to_assert_with_multiline_message(self):
        # [FUNCTION DOC] Rewriting-tracer test for an assert whose message is a multi-line string
        # held in a local variable; unequal inputs must raise an AssertionError whose text matches
        # that message; equal inputs return the eager result.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module asserting input equality with a multi-line message
            # variable and returning the sum.
            def forward(self, a, b):
                # [FUNCTION DOC] HELPER: builds a multi-line message, asserts equality of the
                # inputs with it and returns their sum.
                error_msg = """
An error message with
terrible spacing
                """
                assert a == b, error_msg  # noqa: S101
                return a + b

        m = M()
        traced = symbolic_trace_with_rewrite(m)

        # Make sure the graph is well-formed
        traced.graph.lint()

        # Check the IR to make sure there's a call_function node with target == "Assert"
        self.assertTrue(
            any(
                node.op == "call_function" and node.target is torch._assert
                for node in traced.graph.nodes
            )
        )

        # Ensure that the assert throws when it's supposed to and doesn't throw when it's not supposed to
        error_msg = """
An error message with
terrible spacing
    """
        traced(3, 3)
        with self.assertRaisesRegex(AssertionError, error_msg):
            traced(3, 5)

        # Confirm that the output is correct
        self.assertEqual(traced(3, 3), m(3, 3))

    @parametrize("use_lazy", [True, False])
    def test_subgraph_creation(self, use_lazy):
        # [FUNCTION DOC] Parametrized on whether to use lazy graph modules. Tags every non-output
        # node with a meta flag, splits the module into three round-robin partitions, then checks
        # every call_module target is a GraphModule whose non-output nodes still carry the flag,
        # and that the split module output equals the original.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a parameter and a linear layer producing two clamped
            # branches that are summed.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a 3 by 4 parameter and a linear layer from 4
                # to 5 features.
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: adds the parameter to the first input, applies linear and
                # clamps to 0..1, does the same linear and clamp on the second input, and returns
                # the sum.
                z = self.linear(x + self.param).clamp(min=0.0, max=1.0)
                w = self.linear(y).clamp(min=0.0, max=1.0)
                return z + w

        # symbolically trace model
        my_module = MyModule()
        my_module_traced = symbolic_trace(my_module)

        # random mod partitioning
        partition_counter = 0
        NPARTITIONS = 3

        # Add some random meta info to make sure it is kept around.
        for node in my_module_traced.graph.nodes:
            if node.op != "output":
                node.meta["test_meta_info"] = True

        def mod_partition(node: Node):
            # [FUNCTION DOC] HELPER: split callback that ignores the node and returns
            # partition ids cycling through 0, 1 and 2, using a shared counter updated in
            # the enclosing test.
            nonlocal partition_counter
            partition = partition_counter % NPARTITIONS
            partition_counter = (partition_counter + 1) % NPARTITIONS
            return partition

        # split module in module with submodules
        with _use_lazy_graph_module(use_lazy):
            module_with_submodules = split_module(
                my_module_traced, my_module, mod_partition
            )

        # Check that test_meta_info was still on all nodes.
        submodules = dict(module_with_submodules.named_modules())
        for node in module_with_submodules.graph.nodes:
            if node.op == "call_module":
                submod = submodules[node.target]
                self.assertTrue(isinstance(submod, torch.fx.GraphModule))
                for submod_node in submod.graph.nodes:
                    if submod_node.op != "output":
                        stored_op = submod_node.meta.get("test_meta_info")
                        self.assertTrue(stored_op is not None and stored_op)

        x = torch.rand(3, 4)
        y = torch.rand(3, 4)

        orig_out = my_module_traced(x, y)
        submodules_out = module_with_submodules(x, y)

        self.assertEqual(orig_out, submodules_out)

    def test_split_module_input_names(self):
        # [FUNCTION DOC] Splits a module into three partitions with original input names not kept;
        # checks that all submodules accept the same three arguments in the same order and give
        # equal outputs, and that each submodule placeholder targets are renamed to arg_0, arg_1
        # and arg_2.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module accumulating squared and halved input pairs into x over
            # three steps.
            def forward(self, x, a0, a1, b0, b1, c0, c1):
                # [FUNCTION DOC] HELPER: adds the square of each a input and half of each b input,
                # in three groups, to x and returns it.
                x = x + (a0 ** 2) + (a1 / 2)
                x = x + (b0 ** 2) + (b1 / 2)
                x = x + (c0 ** 2) + (c1 / 2)
                return x

        mod = Mod()
        traced = torch.fx.symbolic_trace(mod)

        seen = 0

        def split(n):
            # [FUNCTION DOC] HELPER: split callback assigning the partition id as the number of
            # nodes seen so far divided by four (integer division), updating a counter in the
            # enclosing test.
            nonlocal seen
            result = seen // 4
            seen += 1
            return result

        split = split_module(traced, mod, split, keep_original_input_name=False)

        # All the submodules should take in the inputs in the same order.
        args = [torch.tensor(2.), torch.tensor(3.), torch.tensor(4.)]
        output0 = split.submod_0(*args)
        output1 = split.submod_1(*args)
        output2 = split.submod_2(*args)
        self.assertEqual(output0, output1)
        self.assertEqual(output1, output2)

        # Each submodule should have normalized input names
        def check_ph(gm):
            # [FUNCTION DOC] HELPER: asserts the first three nodes of the given graph module are
            # placeholders named arg_0, arg_1 and arg_2 by target.
            nodes = list(gm.graph.nodes)
            self.assertEqual(nodes[0].target, "arg_0")
            self.assertEqual(nodes[1].target, "arg_1")
            self.assertEqual(nodes[2].target, "arg_2")

        check_ph(split.submod_0)
        check_ph(split.submod_1)
        check_ph(split.submod_2)

    def test_split_module_dead_code(self):
        # [FUNCTION DOC] Splits a module with an unused computation into before, multiply and
        # after partitions via a callback, and checks the split module output equals the traced
        # module output.
        class ModWithDeadCode(torch.nn.Module):
            # [CLASS DOC] HELPER: module that returns x times two and also computes
            # an unused value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: computes the used product and a dead addition and returns
                # the product.
                output = x * 2  # we want this
                dead_line = x + 2  # this is dead
                return output

        mod = ModWithDeadCode()
        traced = torch.fx.symbolic_trace(mod)

        # split into before (0), target (1), and after(2)
        saw_mul = False

        def split_callback(n):
            # [FUNCTION DOC] HELPER: split callback returning 1 for the multiply node and setting
            # a flag; other nodes get 0 before the multiply is seen and 2 after.
            nonlocal saw_mul
            if n.target == operator.mul:
                saw_mul = True
                return 1

            if not saw_mul:
                return 0
            if saw_mul:
                return 2

        split = split_module(traced, mod, split_callback)

        x = torch.randn((5,))
        torch.testing.assert_close(
            split(x), traced(x)
        )

    def test_split_module_return_node(self):
        # [FUNCTION DOC] Traces an in-place add function in fake mode, splits everything into
        # partition 1, and asserts the resulting submodule graph contains an output node (raises
        # RuntimeError otherwise).
        def foo(x):
            # [FUNCTION DOC] HELPER: traced function that adds one in place and returns nothing.
            x.add_(1)

        gm = make_fx(foo, tracing_mode="fake")(torch.randn(3,))

        def cb(_):
            # [FUNCTION DOC] HELPER: split callback that puts every node in partition 1.
            return 1

        sp_gm = split_module(gm, None, cb)
        submod_gm = sp_gm.submod_1
        for node in submod_gm.graph.nodes:
            if node.op == "output":
                break
        else:
            raise RuntimeError("Expected the subgraph to have an output node.")

    def test_split_module_tuple_return(self):
        # [FUNCTION DOC] Splits an add then multiply graph into two partitions; without the tuple
        # return option at least one submodule must not return a tuple, with that option set all
        # submodules must return tuples; both versions must equal the original output.
        from torch._inductor.compile_fx import graph_returns_tuple

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module computing the product of the sum of two inputs with the
            # first input.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: adds the inputs and multiplies the sum by
                # the first input.
                a = x + y
                return a * x

        gm = torch.fx.symbolic_trace(M())

        # Assign ops to different partitions so a single-output submodule exists.
        def partition_fn(node):
            # [FUNCTION DOC] HELPER: split callback placing the add node in partition 0 and all
            # others in partition 1.
            return 0 if node.target == operator.add else 1

        # Without tuple_return: single-output submodules return a bare value.
        sp = split_module(gm, None, partition_fn)
        self.assertTrue(
            any(
                not graph_returns_tuple(submod)
                for submod in sp.children()
            ),
            "expected at least one non-tuple-returning submodule",
        )
        x, y = torch.randn(4), torch.randn(4)
        self.assertEqual(sp(x, y), gm(x, y))

        # With tuple_return: all submodules return a tuple.
        sp_boxed = split_module(gm, None, partition_fn, tuple_return=True)
        self.assertTrue(
            all(graph_returns_tuple(submod) for submod in sp_boxed.children()),
            "all submodules should return a tuple with tuple_return=True",
        )
        self.assertEqual(sp_boxed(x, y), gm(x, y))


    def test_split_module_kwargs_expansion(self):
        # [FUNCTION DOC] Splits a module taking extra keyword arguments, using a callback that
        # switches to partition 1 after the getitem node is seen; split and traced modules must
        # give close results when called with a keyword argument.
        class ModuleWithKwargsExpansion(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward adds a value taken from the
            # foo keyword argument.
            def forward(self, x, **kwargs):
                # [FUNCTION DOC] HELPER: returns the input plus the foo entry of the
                # extra keyword arguments.
                return x + kwargs['foo']

        mod = ModuleWithKwargsExpansion()
        traced = torch.fx.symbolic_trace(mod)

        seen_getitem = False

        def split_callback(n):
            # [FUNCTION DOC] HELPER: split callback returning 0 until a getitem node has been
            # seen and 1 afterwards, with the node that triggers the switch still returning
            # the earlier value.
            nonlocal seen_getitem
            split_idx = int(seen_getitem)
            if n.target == operator.getitem:
                seen_getitem = True
            return split_idx

        split = split_module(traced, mod, split_callback)

        x = torch.randn(5, 3)
        foo = torch.randn(5, 3)
        torch.testing.assert_close(split(x, foo=foo), traced(x, foo=foo))

    @skipIfNoTorchVision
    def test_subgraph_trivial_resnet(self):
        # [FUNCTION DOC] Skipped without torchvision. Smoke test: splits a traced resnet18 into a
        # single partition and runs it on a random tensor, guarding against an earlier bug of
        # aliased submodule names. Only checks it does not raise.
        # Smoke test trivially splitting resnet into 1 partition works
        # There was an issue before causing submodule names to be aliased
        m = resnet18()
        traced = symbolic_trace(m)
        a = torch.rand(64, 3, 7, 7)
        module_with_submodules = split_module(traced, m, lambda node: 0)
        module_with_submodules(a)

    def test_split_module_default_arg(self):
        # [FUNCTION DOC] Traces a module with an optional argument specialized to None via
        # concrete args, splits into a single partition, and checks the split output is close to
        # the traced output.
        class ModelToTrace(torch.nn.Module):
            # [CLASS DOC] HELPER: linear layer module with an optional additive target input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a 512 by 512 linear layer.
                super().__init__()
                self.lin = torch.nn.Linear(512, 512)

            def forward(self, x, targets=None):
                # [FUNCTION DOC] HELPER: applies the linear layer and adds the targets when they
                # are given.
                x = self.lin(x)

                if targets is not None:
                    x = x + targets

                return x

        mtt = ModelToTrace()
        traced = torch.fx.symbolic_trace(mtt, concrete_args={'targets': None})

        split = split_module(traced, mtt, lambda node: 0)

        x = torch.randn(50, 512)
        torch.testing.assert_close(split(x), traced(x))

    def test_split_module_keep_original_order_and_noop_graph(self):
        # [FUNCTION DOC] Splits a trivial fake-mode graph that only returns its input with the
        # keep original order option both off and on, and checks both results have exactly a
        # placeholder node followed by an output node.
        # Verify that split_module returns a similar no-op graph
        # for `keep_original_order={True|False}`.
        def fn(x):
            # [FUNCTION DOC] HELPER: traced function returning a one-element tuple of its input.
            return (x,)

        g = make_fx(fn, tracing_mode="fake")(torch.randn(3, 3))

        # g.graph.print_tabular()
        # opcode       name    target    args       kwargs
        # -----------  ------  --------  ---------  --------
        # placeholder  x_1     x_1       ()         {}
        # output       output  output    ((x_1,),)  {}

        def _test_split_graph(split_gm):
            # [FUNCTION DOC] HELPER: asserts the split graph module has two nodes, a placeholder
            # then an output.
            # Verify that the split_gm has same structure as original
            self.assertEqual(len(split_gm.graph.nodes), 2)

            nodes = list(split_gm.graph.nodes)
            self.assertEqual(nodes[0].op, "placeholder")
            self.assertEqual(nodes[1].op, "output")

        # `keep_original_order=False`
        _test_split_graph(split_module(g, None, split_callback=lambda _ : 0, keep_original_order=False))

        # `keep_original_order=True`
        _test_split_graph(split_module(g, None, split_callback=lambda _ : 0, keep_original_order=True))

    @unittest.skipIf(TEST_WITH_CROSSREF, "See https://github.com/pytorch/pytorch/issues/160077")
    def test_split_module_symint_dependency_handling(self):
        # [FUNCTION DOC] Skipped under crossref testing. Compiles a mixture-of-experts router
        # (data dependent split sizes from a list) with a custom backend that splits the graph
        # into chunks of five ops, keeping original order and node names, and checks the compiled
        # result is close to eager. Exercises symbolic integer dependencies across partitions.
        # Based on the code from - transformers/models/granitemoe/modeling_granitemoe.py
        class GraniteMoeTopKGating(torch.nn.Module):
            # [CLASS DOC] HELPER: top-k expert gating module computing sorted expert indices,
            # batch indices, gate values, per-expert counts and logits.
            def __init__(self, input_size: int, num_experts: int, top_k: int):
                # [FUNCTION DOC] HELPER: stores the sizes and top-k and creates a bias-free linear
                # layer from input size to number of experts.
                super().__init__()

                self.num_experts = num_experts
                self.input_size = input_size
                self.top_k = top_k

                self.layer = torch.nn.Linear(input_size, num_experts, bias=False)

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: computes logits, selects top-k experts with
                # softmax gates, counts tokens per expert as a Python list, sorts tokens by
                # expert and returns the sorted indices, batch indices, sorted gates,
                # expert sizes and logits.
                # compute the top_k routing decision
                logits = self.layer(hidden_states).float()  # [batch_size x seq_len, num_experts]
                top_k_logits, top_k_indices = logits.topk(self.top_k, dim=1)  # [num_tokens, top_k]
                top_k_gates = torch.softmax(top_k_logits, dim=1).type_as(hidden_states)  # [num_tokens, top_k]

                # compute number of input given to each expert
                zeros = torch.zeros(
                    [top_k_gates.size(0), self.num_experts], dtype=top_k_gates.dtype, device=top_k_gates.device
                )  # [num_tokens, num_experts]
                gates = zeros.scatter(1, top_k_indices, 1)  # [num_tokens, num_experts]
                expert_size = gates.long().sum(0)  # [num_experts,]
                expert_size = expert_size.tolist()

                # sort and group input tokens according to expert assignment
                top_k_experts = top_k_indices.flatten()  # [num_tokens * top_k]
                _, index_sorted_experts = top_k_experts.sort(0)  # [num_tokens * top_k]
                batch_index = index_sorted_experts.div(self.top_k, rounding_mode="trunc")  # [num_tokens * top_k]

                # gather the gate values for grouped input tokens
                top_k_gates = top_k_gates.flatten()  # [num_tokens * top_k]
                batch_gates = top_k_gates[index_sorted_experts]  # [num_tokens * top_k]

                return index_sorted_experts, batch_index, batch_gates, expert_size, logits

        class GraniteMoeMoE(torch.nn.Module):
            # [CLASS DOC] HELPER: toy mixture-of-experts block that routes inputs and splits them
            # by expert counts.
            def __init__(self):
                # [FUNCTION DOC] HELPER: sets sizes (32 inputs, 4 experts) and creates the top-2
                # gating router.
                super().__init__()

                self.input_size = 32
                self.num_local_experts = 4

                num_experts_per_tok = 2
                self.router = GraniteMoeTopKGating(
                    input_size=self.input_size,
                    num_experts=self.num_local_experts,
                    top_k=num_experts_per_tok,
                )

            def forward(self, layer_input):
                # [FUNCTION DOC] HELPER: runs the router, gathers the inputs by batch index and
                # splits them along dim 0 by the expert sizes.
                _, batch_index, _, expert_size, _ = self.router(layer_input)
                expert_inputs = layer_input[batch_index]
                return expert_inputs.split(expert_size, dim=0)

        moe = GraniteMoeMoE()
        inp = torch.randn([32, 32])

        expected = moe(inp)

        PARTITION_ID = 0
        PARTITION_OPS_CTR = 0
        NODE_PARTITION_MAP = {}

        # `callback` is called multiple times with same `node` in `split_module`.
        # Cache the result such that partition id is consistent across calls.
        def callback(node) -> int:
            # [FUNCTION DOC] HELPER: split callback that opens a new partition every five distinct
            # nodes and caches each node's partition so repeated calls agree; uses and updates
            # enclosing counters and the cache.
            nonlocal PARTITION_ID, PARTITION_OPS_CTR, NODE_PARTITION_MAP
            if node in NODE_PARTITION_MAP:
                return NODE_PARTITION_MAP[node]

            if PARTITION_OPS_CTR % 5 == 0:
                PARTITION_ID += 1

            PARTITION_OPS_CTR += 1

            NODE_PARTITION_MAP[node] = PARTITION_ID
            return PARTITION_ID

        def backend(gm, inps):
            # [FUNCTION DOC] HELPER: compile backend that splits the given graph module with the
            # callback, keeping original order and node names, and returns the split module; the
            # example inputs are unused.
            split_gm = split_module(gm, root_m=None, split_callback=callback,
                                    keep_original_order=True, keep_original_node_name=True)
            return split_gm

        actual = torch.compile(moe, backend=backend)(inp)
        torch.testing.assert_close(actual, expected)

    def test_split_module_placeholders_before_get_attr(self):
        # [FUNCTION DOC] Hand builds a graph with a placeholder, a get_attr of a submodule,
        # another placeholder, and two arithmetic calls, splits it into one partition and asserts
        # that in each resulting submodule no placeholder comes after a get_attr node.
        # Manually construct a graph matching what torch.cond + dynamo
        # produces: placeholder, get_attr(nn.Module), placeholder, ...
        # Manual construction avoids dynamo/torch.compile dependency.
        class DummyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: identity module used as an attribute target.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns its input unchanged.
                return x

        root = torch.nn.Module()
        root.branch = DummyModule()

        graph = torch.fx.Graph()
        ph_x = graph.placeholder("x")
        branch = graph.get_attr("branch")
        ph_y = graph.placeholder("y")
        op1 = graph.call_function(torch.mul, (ph_x, branch))
        op2 = graph.call_function(torch.add, (op1, ph_y))
        graph.output(op2)

        for node in graph.nodes:
            node.meta = {}

        gm = torch.fx.GraphModule(root, graph)

        split_gm = split_module(
            gm,
            root_m=None,
            split_callback=lambda node: 0,
            keep_original_order=True,
        )

        for name, submod in split_gm.named_children():
            seen_get_attr = False
            for node in submod.graph.nodes:
                if node.op == "get_attr":
                    seen_get_attr = True
                elif node.op == "placeholder":
                    self.assertFalse(
                        seen_get_attr,
                        lambda msg: f"{msg}\nplaceholder '{node.name}' found after get_attr "
                        f"in submodule '{name}'",
                    )

    @staticmethod
    def _build_partition_map(graph, ops_per_partition=1):
        # [FUNCTION DOC] HELPER: static method that maps every compute node (not placeholder,
        # get_attr or output) in the graph to a partition id equal to its running index divided by
        # the given group size (integer division); returns the mapping.
        """Build a node-to-partition dict assigning ops_per_partition ops per partition."""
        node_to_partition = {}
        counter = 0
        for node in graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            node_to_partition[node] = counter // ops_per_partition
            counter += 1
        return node_to_partition

    def test_split_module_simple_basic(self):
        # [FUNCTION DOC] Splits a relu, sigmoid, tanh chain with the simple splitter after merging
        # the last two ops into partition 1; checks the children are exactly submod_0 and submod_1
        # and the result equals eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying relu, sigmoid and tanh in sequence.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns tanh of sigmoid of relu of the input.
                a = torch.relu(x)
                b = torch.sigmoid(a)
                c = torch.tanh(b)
                return c

        mod = Mod()
        x = torch.randn(3, 4)
        expected = mod(x)

        traced = torch.fx.symbolic_trace(mod)

        # Split into 2 partitions: relu in 0, sigmoid+tanh in 1
        # (first op alone, rest grouped — doesn't fit the uniform helper)
        node_to_partition = self._build_partition_map(traced.graph, ops_per_partition=1)
        # Merge partitions 1 and 2 into partition 1
        for node, pid in node_to_partition.items():
            if pid > 0:
                node_to_partition[node] = 1

        split_gm = split_module_simple(traced, node_to_partition)

        # Verify structure: should have submod_0 and submod_1
        submod_names = [name for name, _ in split_gm.named_children()]
        self.assertEqual(sorted(submod_names), ["submod_0", "submod_1"])

        # Verify correctness
        self.assertEqual(split_gm(x), expected)

    def test_split_module_simple_multi_output(self):
        # [FUNCTION DOC] Simple split where relu and sigmoid are in partition 0 and the add in
        # partition 1 (so partition 0 has two outputs); checks the result equals eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding relu and sigmoid of the same input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns relu of x plus sigmoid of x.
                a = torch.relu(x)
                b = torch.sigmoid(x)
                c = a + b
                return c

        mod = Mod()
        x = torch.randn(3, 4)
        expected = mod(x)

        traced = torch.fx.symbolic_trace(mod)

        # relu and sigmoid in partition 0, add in partition 1
        # This means partition 0 has 2 outputs (a, b)
        node_to_partition = {}
        for node in traced.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            if node.target in (torch.relu, torch.sigmoid):
                node_to_partition[node] = 0
            else:
                node_to_partition[node] = 1

        split_gm = split_module_simple(traced, node_to_partition)
        self.assertEqual(split_gm(x), expected)

    def test_split_module_simple_single_partition(self):
        # [FUNCTION DOC] Simple split with all ops in a single partition (group size 999); result
        # must equal eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying sigmoid then relu.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns relu of sigmoid of the input.
                return torch.relu(torch.sigmoid(x))

        mod = Mod()
        x = torch.randn(3, 4)
        expected = mod(x)

        traced = torch.fx.symbolic_trace(mod)
        node_to_partition = self._build_partition_map(traced.graph, ops_per_partition=999)

        split_gm = split_module_simple(traced, node_to_partition)
        self.assertEqual(split_gm(x), expected)

    def test_split_module_simple_partition_affix(self):
        # [FUNCTION DOC] Simple split with a custom partition affix; asserts every child submodule
        # name contains the affix and that the split output equals a freshly traced module of the
        # same class.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying relu then sigmoid.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns sigmoid of relu of the input.
                a = torch.relu(x)
                b = torch.sigmoid(a)
                return b

        traced = torch.fx.symbolic_trace(Mod())
        node_to_partition = self._build_partition_map(traced.graph)

        split_gm = split_module_simple(
            traced, node_to_partition, partition_affix="pp"
        )

        submod_names = [name for name, _ in split_gm.named_children()]
        self.assertTrue(all("pp" in name for name in submod_names))

        x = torch.randn(3, 4)
        self.assertEqual(
            torch.fx.symbolic_trace(Mod())(x),
            split_gm(x),
        )

    def test_split_module_simple_symint_crossing(self):
        # [FUNCTION DOC] Compiles a relu then sigmoid module with a backend that splits each op
        # into its own partition via the simple splitter; checks the compiled output is close to
        # eager. Meant to cover symbolic dependencies crossing partitions.
        # Test that SymInt dependencies are propagated across partitions
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module applying relu then sigmoid.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns sigmoid of relu of the input.
                a = torch.relu(x)
                b = torch.sigmoid(a)
                return b

        mod = Mod()
        inp = torch.randn(4, 8)
        expected = mod(inp)

        def backend(gm, inps):
            # [FUNCTION DOC] HELPER: compile backend building a one-op-per-partition map
            # for the graph and returning the result of the simple splitter; the example
            # inputs are unused.
            node_to_part = TestFXExperimental._build_partition_map(gm.graph)
            return split_module_simple(gm, node_to_part)

        actual = torch.compile(mod, backend=backend)(inp)
        torch.testing.assert_close(actual, expected)

    def test_split_module_simple_equivalence_with_split_module(self):
        # [FUNCTION DOC] Traces a two input relu, sigmoid, add, tanh module twice and splits one
        # copy with the simple splitter (two ops per partition from a helper map) and the other
        # with the callback based splitter using an equivalent callback; asserts equal outputs.
        # For simple graphs, split_module_simple should produce equivalent
        # results to split_module
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: module with relu, sigmoid, add and tanh over two inputs.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: adds relu of x and sigmoid of y, then returns tanh
                # of the sum.
                a = torch.relu(x)
                b = torch.sigmoid(y)
                c = a + b
                d = torch.tanh(c)
                return d

        mod = Mod()
        traced1 = torch.fx.symbolic_trace(mod)
        traced2 = torch.fx.symbolic_trace(mod)

        # Build partition mapping: 2 ops per partition
        node_to_partition_simple = self._build_partition_map(traced1.graph, ops_per_partition=2)

        counter = 0
        callback_cache = {}

        def split_cb(node):
            # [FUNCTION DOC] HELPER: split callback giving each distinct node name a
            # partition equal to its first-seen order divided by two, caching by name and
            # updating enclosing counters.
            nonlocal counter
            if node.name not in callback_cache:
                callback_cache[node.name] = counter // 2
                counter += 1
            return callback_cache[node.name]

        split_gm_simple = split_module_simple(traced1, node_to_partition_simple)
        split_gm_orig = split_module(traced2, mod, split_cb)

        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        self.assertEqual(split_gm_simple(x, y), split_gm_orig(x, y))

    def test_split_module_simple_dead_code(self):
        # [FUNCTION DOC] Simple splitter with a mapping that puts the multiply in partition 1 and
        # nodes after it (including an unused add) in partition 2; checks the split output is
        # close to the traced output.
        class ModWithDeadCode(torch.nn.Module):
            # [CLASS DOC] HELPER: module that returns x times two and also computes
            # an unused value.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: computes the used product and a dead addition and returns
                # the product.
                output = x * 2  # we want this
                dead_line = x + 2
                return output

        mod = ModWithDeadCode()
        traced = torch.fx.symbolic_trace(mod)

        # Split: mul in partition 1, add in partition 2
        node_to_partition = {}
        saw_mul = False
        for node in traced.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            if node.target == operator.mul:
                saw_mul = True
                node_to_partition[node] = 1
            elif not saw_mul:
                node_to_partition[node] = 0
            else:
                node_to_partition[node] = 2

        split = split_module_simple(traced, node_to_partition)
        x = torch.randn((5,))
        torch.testing.assert_close(split(x), traced(x))

    def test_split_module_simple_kwargs(self):
        # [FUNCTION DOC] Simple splitter on a module taking extra keyword arguments; the mapping
        # switches to partition 1 after the getitem node. Checks split and traced outputs are
        # close when called with a keyword argument.
        class ModuleWithKwargs(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding a value taken from the foo keyword argument.
            def forward(self, x, **kwargs):
                # [FUNCTION DOC] HELPER: returns the input plus the foo entry of the
                # extra keyword arguments.
                return x + kwargs["foo"]

        mod = ModuleWithKwargs()
        traced = torch.fx.symbolic_trace(mod)

        seen_getitem = False
        node_to_partition = {}
        for node in traced.graph.nodes:
            if node.op in ("placeholder", "get_attr", "output"):
                continue
            split_idx = int(seen_getitem)
            if node.target == operator.getitem:
                seen_getitem = True
            node_to_partition[node] = split_idx

        split = split_module_simple(traced, node_to_partition)
        x = torch.randn(5, 3)
        foo = torch.randn(5, 3)
        torch.testing.assert_close(split(x, foo=foo), traced(x, foo=foo))

    def test_split_module_simple_noop_graph(self):
        # [FUNCTION DOC] Runs the simple splitter with an empty mapping on a fake-mode graph that
        # only returns its input; asserts the result has two nodes, a placeholder then an output.
        # Verify split_module_simple handles a graph with no ops (just input->output)
        def fn(x):
            # [FUNCTION DOC] HELPER: traced function returning a one-element tuple of its input.
            return (x,)

        g = make_fx(fn, tracing_mode="fake")(torch.randn(3, 3))
        split_gm = split_module_simple(g, {})

        self.assertEqual(len(split_gm.graph.nodes), 2)
        nodes = list(split_gm.graph.nodes)
        self.assertEqual(nodes[0].op, "placeholder")
        self.assertEqual(nodes[1].op, "output")

    def test_normalize_binary_operators(self):
        # [FUNCTION DOC] For each of twelve binary tensor operators (arithmetic and comparison),
        # once with tensor-tensor and once with tensor-scalar operands, traces a wrapper module,
        # applies the operator normalization pass, asserts outputs are close and that no node
        # target remains from the operator set afterwards.
        ops_to_test = {
            torch.add,
            torch.mul,
            torch.sub,
            torch.div,
            torch.floor_divide,
            torch.remainder,
            torch.eq,
            torch.ne,
            torch.lt,
            torch.le,
            torch.gt,
            torch.ge,
        }

        # Test Tensor/Tensor callsite
        for op in ops_to_test:

            class WrapperMod(torch.nn.Module):
                # [CLASS DOC] HELPER: module applying the loop's current operator to
                # two tensor inputs.
                def forward(self, x, y):
                    # [FUNCTION DOC] HELPER: returns the operator applied to both inputs.
                    return op(x, y)

            traced = symbolic_trace(WrapperMod())
            normalized = NormalizeOperators(traced).transform()
            x, y = torch.randn(3, 4), torch.randn(3, 4)
            torch.testing.assert_close(traced(x, y), normalized(x, y))
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

        # Test Tensor/scalar callsite
        for op in ops_to_test:

            class WrapperMod(torch.nn.Module):
                # [CLASS DOC] HELPER: module applying the loop's current operator to a tensor and
                # the scalar 42.
                def forward(self, x):
                    # [FUNCTION DOC] HELPER: returns the operator applied to the input and 42.
                    return op(x, 42)

            traced = symbolic_trace(WrapperMod())
            normalized = NormalizeOperators(traced).transform()
            x = torch.randn(3, 4)
            torch.testing.assert_close(traced(x), normalized(x))
            self.assertFalse(
                any(n.target in ops_to_test for n in normalized.graph.nodes)
            )

    @skipIfNoTorchVision
    def test_normalize_args(self):
        # [FUNCTION DOC] Skipped without torchvision. Traces resnet18 with a tracer that treats
        # batchnorm as a leaf, runs shape propagation, applies the argument normalization pass,
        # then asserts non-add function nodes and standard nn module calls have no positional args
        # left, and that outputs still equal the reference.
        m = resnet18()

        class FunctionalTracer(torch.fx.Tracer):
            # [CLASS DOC] HELPER: tracer that keeps batchnorm modules as leaves.
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # [FUNCTION DOC] HELPER: returns whether the module's type is in the set of leaf
                # types (only BatchNorm2d).
                # `leaves` contains the set of standard `nn.Modules` that are not
                # currently symbolically traceable. Ideally this set would be empty
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        traced = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        input = torch.randn(5, 3, 224, 224)
        ref_outs = traced(input)

        ShapeProp(traced).propagate(input)
        traced = NormalizeArgs(traced).transform()

        modules = dict(traced.named_modules())

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target != operator.add:
                self.assertEqual(len(node.args), 0)
            elif node.op == "call_module":
                submod_class = modules[node.target].__class__
                nn_class = getattr(torch.nn, submod_class.__name__)
                if submod_class == nn_class:
                    self.assertEqual(len(node.args), 0)
        traced(input)
        self.assertEqual(traced(input), ref_outs)

    def test_normalize_modules_exhaustive(self):
        # [FUNCTION DOC] For every module test case in the shared nn test tables, builds the
        # standard torch.nn module, wraps it in a generated class compiled from a source string,
        # traces it, rewrites call_module nodes to use normalized keyword arguments (checking the
        # two normalization entry points agree and are non-empty), compares outputs with eager
        # except for stochastic modules, and then checks the argument normalization pass leaves no
        # positional args. Non-standard modules are skipped. Side effect: executes generated code.
        """
        Exhaustively test `Node.normalized_arguments` on all standard
        torch.nn Module classes
        """
        for test_params in module_tests + get_new_module_tests():
            if "constructor" not in test_params:
                constructor = getattr(torch.nn, test_params["module_name"])
            else:
                constructor = test_params["constructor"]

            if "constructor_args" not in test_params:
                args = ()
            else:
                args = test_params["constructor_args"]

            mod = constructor(*args)
            # Skip modules that are not standard `torch.nn`
            # instances, including functionals. (functionals
            # are tested in test_normalize_args)
            if mod.__class__.__name__ not in dir(torch.nn):
                continue

            if "input_fn" not in test_params:
                inputs = torch.randn(test_params["input_size"])
            else:
                inputs = test_params["input_fn"]()

            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)

            params = ", ".join(f"v{i}" for i in range(len(inputs)))

            # Generate a class to wrap this standard `nn.Module` instance
            test_classname = f"Test{mod.__class__.__name__}"
            test_mod_code = f"""
class {test_classname}(torch.nn.Module):
    def __init__(self, mod):
        super().__init__()
        self.mod = mod

    def forward(self, {params}):
        return self.mod({params})
            """

            gbls = {"torch": torch}
            exec(test_mod_code, gbls)

            test_instance = gbls[test_classname](mod)
            traced = symbolic_trace(test_instance)

            # Use `Node.normalized_arguments` to get a new set of arguments
            # to feed to the Module. Then, rewrite the node to only take
            # in those arguments as kwargs
            modules = dict(traced.named_modules())
            for node in traced.graph.nodes:
                if node.op == "call_module":
                    submod_class = modules[node.target].__class__
                    nn_class = getattr(torch.nn, submod_class.__name__)
                    if submod_class == nn_class:
                        normalized_args = node.normalized_arguments(traced)
                        normalized_args2 = normalize_module(
                            traced, node.target, node.args, node.kwargs
                        )
                        if normalized_args != normalized_args2:
                            raise AssertionError("normalized_args mismatch")
                        if not normalized_args:
                            raise AssertionError("expected normalized_args to be truthy")
                        node.args = normalized_args.args
                        node.kwargs = normalized_args.kwargs

            traced.recompile()

            # These Modules have an RNG in their forward, so testing
            # correctness by comparing outputs is not correct. Skip that
            # check for these
            stochastic_modules = {"FractionalMaxPool2d", "FractionalMaxPool3d", "RReLU"}

            if mod.__class__.__name__ not in stochastic_modules:
                self.assertEqual(traced(*inputs), mod(*inputs))

            traced = NormalizeArgs(symbolic_trace(test_instance)).transform()
            modules = dict(traced.named_modules())
            for node in traced.graph.nodes:
                if node.op == "call_module":
                    submod_class = modules[node.target].__class__
                    nn_class = getattr(torch.nn, submod_class.__name__)
                    if submod_class == nn_class:
                        self.assertEqual(len(node.args), 0)

    def test_normalize_args_preserve_meta(self):
        # [FUNCTION DOC] Tags the torch add call node with a meta entry, runs shape propagation
        # and the argument normalization pass, and asserts the meta entry is still present with
        # its value on the add node.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding the constant 3 to its input.
            def forward(self, a):
                # [FUNCTION DOC] HELPER: returns the input plus 3 using the torch add function.
                return torch.add(a, 3)

        m = MyModule()
        traced = symbolic_trace(m)

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                node.meta["my_key"] = 7
                break
        else:
            self.fail("Didn't find call_function torch.add")

        input = torch.randn(2, 3)
        ShapeProp(traced).propagate(input)
        traced = NormalizeArgs(traced).transform()

        for node in traced.graph.nodes:
            if node.op == "call_function" and node.target == torch.add:
                self.assertTrue("my_key" in node.meta)
                self.assertEqual(node.meta["my_key"], 7)
                break
        else:
            self.fail("Didn't find call_function torch.add")

    def test_normalize_args_perserve_type(self):
        # [FUNCTION DOC] Applies the argument normalization pass to a graph whose input is
        # annotated as a list of tensors and asserts placeholder nodes keep that type. The method
        # name misspells preserve.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module adding the first two elements of a tensor list input.
            def forward(self, a: list[torch.Tensor]):
                # [FUNCTION DOC] HELPER: returns the sum of the first two list elements.
                return torch.add(a[0], a[1])

        m = MyModule()
        traced = symbolic_trace(m)
        traced = NormalizeArgs(traced).transform()

        for node in traced.graph.nodes:
            if node.op == "placeholder":
                self.assertEqual(node.type, list[torch.Tensor])

    @skipIfNoTorchVision
    def test_annotate_returns_with_schema(self):
        # [FUNCTION DOC] Skipped without torchvision. Annotates a traced resnet18 using operator
        # schemas and asserts that nodes still lacking a type are only a short known set; repeats
        # for a functional trace (batchnorm kept as leaf) allowing only a few exclusions and
        # ignoring builtin functions; each annotated module is also compiled with TorchScript as a
        # smoke test.
        m = resnet18()

        traced_modules = symbolic_trace(m)
        traced_modules_annotated = AnnotateTypesWithSchema(traced_modules).transform()
        for node in traced_modules_annotated.graph.nodes:
            if node.type is None:
                check = (node.op, node.target)
                self.assertIn(
                    check,
                    {
                        ("placeholder", "x"),
                        ("call_module", "maxpool"),
                        ("call_function", operator.add),
                        ("call_function", torch.flatten),
                        ("output", "output"),
                    }
                )

        # Smoke test torchscript compilation since now we're emitting type annotations
        torch.jit.script(traced_modules_annotated)

        class FunctionalTracer(torch.fx.Tracer):
            # [CLASS DOC] HELPER: tracer that keeps batchnorm modules as leaves.
            def is_leaf_module(
                self, m: torch.nn.Module, module_qualified_name: str
            ) -> bool:
                # [FUNCTION DOC] HELPER: returns whether the module's type is in the set of leaf
                # types (only BatchNorm2d).
                # `leaves` contains the set of standard `nn.Modules` that are not
                # currently symbolically traceable. Ideally this set would be empty
                leaves = {torch.nn.BatchNorm2d}
                return type(m) in leaves

        traced_functionals = torch.fx.GraphModule(m, FunctionalTracer().trace(m))

        traced_functionals_annotated = AnnotateTypesWithSchema(
            traced_functionals
        ).transform()
        for node in traced_functionals_annotated.graph.nodes:
            if node.type is None:
                check = (node.op, node.target)
                excluded_nodes = {
                    ("placeholder", "x"),
                    # Return type differs based on boolean dispatch :(
                    ("call_function", torch.nn.functional.max_pool2d),
                    ("output", "output"),
                }
                # AnnotateTypesWithSchema doesn't work with bound C++ functions
                if not isinstance(node.target, BuiltinFunctionType):
                    self.assertIn(check, excluded_nodes)

        # Smoke test torchscript compilation since now we're emitting type annotations
        torch.jit.script(traced_functionals_annotated)

    def test_annotate_getitem_node(self):
        # [FUNCTION DOC] Checks that tracing loses type annotations on getitem nodes and that
        # annotate_getitem_nodes restores a type for all of them, using two identical modules that
        # index a tuple, a list and a named tuple.
        class CustomType:
            # [CLASS DOC] HELPER: empty custom class used in type annotations.
            pass

        class CustomNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: named tuple with an int field x and a float field y.
            x: int
            y: float

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module taking a tuple, a list and a named tuple and summing
            # indexed and field values.
            def forward(self, inp: tuple[CustomType, torch.Tensor], inp2: list[CustomType], inp3: CustomNamedTuple):
                # [FUNCTION DOC] HELPER: indexes the tuple and list and reads both named tuple
                # fields, returning their sum.
                inp_0 = inp[0]
                inp_1 = inp[1]
                inp2_0 = inp2[0]
                inp3_x = inp3.x
                inp3_y = inp3.y
                return inp_0 + inp_1 + inp2_0 + inp3_x + inp3_y

        class MyModule2(torch.nn.Module):
            # [CLASS DOC] HELPER: duplicate of the first test module, used for a second pass.
            def forward(self, inp: tuple[CustomType, torch.Tensor], inp2: list[CustomType], inp3: CustomNamedTuple):
                # [FUNCTION DOC] HELPER: indexes the tuple and list and reads both named tuple
                # fields, returning their sum.
                inp_0 = inp[0]
                inp_1 = inp[1]
                inp2_0 = inp2[0]
                inp3_x = inp3.x
                inp3_y = inp3.y
                return inp_0 + inp_1 + inp2_0 + inp3_x + inp3_y

        my_module = MyModule()
        my_module_traced = torch.fx.symbolic_trace(my_module)

        # by default, fx transform loses type annotation of getitem nodes.
        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                if node.type is not None:
                    raise AssertionError(f"expected node.type is None, got {node.type}")

        annotate_getitem_nodes(my_module_traced.graph)

        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                self.assertIsNotNone(node.type, lambda msg: f"{msg}\nNode {node} should be annotated but is not.")

        my_module = MyModule2()
        my_module_traced = torch.fx.symbolic_trace(my_module)

        # by default, fx transform loses type annotation of getitem nodes.
        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                if node.type is not None:
                    raise AssertionError(f"expected node.type is None, got {node.type}")

        annotate_getitem_nodes(my_module_traced.graph)

        for node in my_module_traced.graph.nodes:
            if node.target == operator.getitem:
                self.assertIsNotNone(node.type, lambda msg: f"{msg}\nNode {node} should be annotated but is not.")

    def test_subgraph_uniquename(self):
        # [FUNCTION DOC] Splits a four input module into two partitions with a callback sending
        # the input placeholders a and b and the first add to partition 0 and everything else to
        # partition 1; checks the split module output equals the traced module output (regression
        # for unique submodule names).
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a linear layer and several adds over four inputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a 4 by 4 linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, a, b, c, d):
                # [FUNCTION DOC] HELPER: chains adds of the four inputs plus a linear branch and
                # returns the final sum.
                add_1 = a + b
                add_2 = add_1 + c
                linear_1 = self.linear(add_1)
                add_3 = add_2 + d
                add_4 = add_2 + linear_1
                add_5 = add_3 + add_4
                return add_5

        a, b, c, d = torch.ones(4), torch.ones(4), torch.ones(4), torch.ones(4)
        mm = MyModule()
        traced = symbolic_trace(mm)

        def split_cb(node: torch.fx.Node):
            # [FUNCTION DOC] HELPER: split callback returning 0 for nodes named a, b or add
            # and 1 otherwise.
            if node.name == "a" or node.name == "b" or node.name == "add":
                return 0
            else:
                return 1

        module_with_submodule = split_module(traced, mm, split_cb)
        self.assertEqual(module_with_submodule(a, b, c, d), traced(a, b, c, d))

    def test_split_qualname_mapping(self):
        # [FUNCTION DOC] Splits a module into partitions that start a new index at each call to
        # the linear layer, passing a dict to receive the mapping; asserts it maps submod_1.lin
        # and submod_2.lin to lin.
        d_hid = 4

        class ExampleCode(torch.nn.Module):
            # [CLASS DOC] HELPER: module with two matrix parameters and a shared linear
            # layer used twice.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates two random square parameters and a linear layer.
                super().__init__()
                self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.lin = torch.nn.Linear(d_hid, d_hid)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: alternates matrix multiplies, relus and the linear layer,
                # returning the result.
                x = torch.mm(x, self.mm_param)
                x = torch.relu(x)
                x = torch.mm(x, self.mm_param)
                x = self.lin(x)
                x = torch.relu(x)
                x = torch.mm(x, self.mm_param2)
                x = self.lin(x)
                return x

        my_module = ExampleCode()
        my_module_traced = symbolic_trace(my_module)

        part_idx = 0

        def split_callback(n : torch.fx.Node):
            # [FUNCTION DOC] HELPER: split callback that increments a partition counter at each
            # call_module node targeting lin and returns the current counter.
            nonlocal part_idx
            if (n.op, n.target) == ('call_module', 'lin'):
                part_idx += 1
            return part_idx

        # split module in module with submodules
        qualname_map : dict[str, str] = {}
        module_with_submodules = split_module(
            my_module_traced, my_module, split_callback, qualname_map
        )
        expected_qualname_map = {
            'submod_1.lin': 'lin', 'submod_2.lin': 'lin'
        }
        self.assertEqual(qualname_map, expected_qualname_map)

    def test_traceable_function_with_nonstandard_name(self):
        # [FUNCTION DOC] Smoke test: traces a plain function with the rewriting tracer. No
        # assertion; it only must not raise.
        def foo(x):
            # [FUNCTION DOC] HELPER: traced function returning relu of the input.
            return torch.relu(x)

        traced = symbolic_trace_with_rewrite(foo)

    def test_to_folder(self):
        # [FUNCTION DOC] Traces a module with a parameter, buffers, a plain tensor attribute,
        # batchnorm and linear, writes it out with to_folder in a temporary directory, imports the
        # generated package from file into sys.modules, and asserts the imported class gives the
        # same output as the traced module. Side effects: temp files and a sys.modules entry.
        class Test(torch.nn.Module):
            # [CLASS DOC] HELPER: module with parameter W, a batchnorm sequence, a linear layer, a
            # plain tensor and two buffers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the parameter, sequence, linear layer, tensor
                # attribute and two buffers (one float, one int32).
                super().__init__()
                self.W = torch.nn.Parameter(torch.randn(2))
                self.seq = torch.nn.Sequential(torch.nn.BatchNorm1d(2, 2))
                self.linear = torch.nn.Linear(2, 2)
                self.attr = torch.randn(2)
                self.attr2 = torch.nn.Buffer(torch.randn(2))
                self.attr3 = torch.nn.Buffer(torch.ones(2, dtype=torch.int32))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: sums the parameter, attribute and buffers with the input
                # and applies batchnorm then linear.
                return self.linear(self.seq(self.W + self.attr + self.attr2 + self.attr3 + x))

        mod = symbolic_trace(Test())
        module_name = "Foo"
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            mod.to_folder(tmp_dir, module_name)
            # Recipe taken from here:
            # https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
            import importlib.util

            spec = importlib.util.spec_from_file_location(
                module_name, tmp_dir / "__init__.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            t = torch.randn(2, 2)
            self.assertEqual(module.Foo()(t), mod(t))

    def test_fetch(self):
        # [FUNCTION DOC] Traces a conv plus batchnorm module and lifts lowering attributes onto
        # call_module nodes; asserts each such node has the attribute mapping, whose size equals
        # the expected attribute list for its class plus one for the class name, and that every
        # expected attribute is present.
        attrs_for_lowering: dict[str, list[str]] = {
            "torch.nn.modules.conv.Conv2d": [
                "weight",
                "bias",
                "kernel_size",
                "stride",
                "padding",
                "dilation",
                "groups",
                "padding_mode",
            ],
            "torch.nn.modules.batchnorm.BatchNorm2d": [
                "weight",
                "bias",
                "running_mean",
                "running_var",
                "eps",
            ],
        }

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: convolution followed by batchnorm with an in-place add between.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates a 3 channel conv and a batchnorm.
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 2)
                self.bn = torch.nn.BatchNorm2d(3)

            def forward(self, a):
                # [FUNCTION DOC] HELPER: applies the conv, doubles by in-place add, then applies
                # the batchnorm.
                a = self.conv(a)
                a += a
                return self.bn(a)

        mod = TestModule()
        traced = symbolic_trace(mod)
        lift_lowering_attrs_to_nodes(traced)

        for node in traced.graph.nodes:
            if node.op == "call_module":
                if not hasattr(node, "attrs_for_lowering"):
                    raise AssertionError("expected node to have attrs_for_lowering")
                para_list = attrs_for_lowering[node.attrs_for_lowering["name"]]

                # node.attrs_for_lowering has an addition field of class name
                if len(para_list) + 1 != len(node.attrs_for_lowering):
                    raise AssertionError(
                        f"attrs_for_lowering length mismatch: {len(para_list) + 1} != {len(node.attrs_for_lowering)}"
                    )
                for p_name in para_list:
                    if p_name not in node.attrs_for_lowering:
                        raise AssertionError(f"{p_name} not in node.attrs_for_lowering")

    def test_merge_matmuls(self):
        # [FUNCTION DOC] Tests merge_matmul in three scenarios, counting matmul nodes before and
        # after: two same-size matmuls sharing a right operand (merged to one), five matmuls over
        # inputs of different row counts (merged to one), and a pair where the second depends on
        # the first (left at two).
        # KNOWN ISSUE: each numerical comparison calls allclose but discards the result, so
        # numerical equivalence is never asserted.
        """
        A collection of test cases for torch.fx.experimental.merge_matmul,
        a graph transformation that merges matrix multiplication operations.
        """
        # Utility function for counting matmuls for test assertions.
        def _count_matmuls(mod):
            # [FUNCTION DOC] HELPER: traces a module and returns how many nodes call
            # the matmul function.
            gm = torch.fx.symbolic_trace(mod)

            num_matmuls = 0
            for node in gm.graph.nodes:
                if node.target == torch.matmul:
                    num_matmuls += 1

            return num_matmuls

        # Simple test case in which there are two matmuls of the same size to merge.
        class SimpleMergeMatmulModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding a right-hand matrix and multiplying two inputs by
            # it, then adding.
            def __init__(self, rhs):
                # [FUNCTION DOC] HELPER: stores the right-hand matrix as an attribute.
                super().__init__()
                self.rhs = rhs

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: multiplies each input by the stored matrix and returns
                # the sum.
                a = torch.matmul(x, self.rhs)
                b = torch.matmul(y, self.rhs)
                return a + b

        # Initialize inputs.
        a = torch.randn(3, 3)
        b = torch.randn(3, 3)

        # Initialize RHS for matmuls.
        rhs = torch.randn(3, 4)

        # Construct SimpleMergeMatmulModule and call merge_matmul on it.
        module = SimpleMergeMatmulModule(rhs)
        opt_module = merge_matmul.merge_matmul(module)

        # Numerical correctness check.
        before = module(a, b)
        after = opt_module(a, b)
        before.allclose(after)

        # Basic graph structure check; original module should have 2 matmuls
        # and optimized module should have 1.
        self.assertEqual(_count_matmuls(module), 2)
        self.assertEqual(_count_matmuls(opt_module), 1)

        # Test case in which there are multiple matmuls of different sizes to merge.
        class FiveMergeMatmulModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module multiplying five inputs by one stored matrix and
            # accumulating the sums of results.
            def __init__(self, rhs):
                # [FUNCTION DOC] HELPER: stores the right-hand matrix as an attribute.
                super().__init__()
                self.rhs = rhs

            def forward(self, a, b, c, d, e):
                # [FUNCTION DOC] HELPER: matrix multiplies each of five inputs by the stored
                # matrix and returns the accumulated sum of all result elements.
                s = torch.tensor([])
                matmuls = []

                # For some reason using a list comprehension or for-loop for this
                # doesn't work.
                matmuls.append(torch.matmul(a, self.rhs))
                matmuls.append(torch.matmul(b, self.rhs))
                matmuls.append(torch.matmul(c, self.rhs))
                matmuls.append(torch.matmul(d, self.rhs))
                matmuls.append(torch.matmul(e, self.rhs))

                for m in matmuls:
                    s += torch.sum(m)

                return s

        # Initialize inputs.
        inputs = [torch.randn(2 * i + 1, 5) for i in range(5)]

        # Initialize RHS.
        rhs = torch.randn(5, 4)

        # Construct FiveMergeMatmulModule and call merge_matmul on it.
        module = FiveMergeMatmulModule(rhs)
        opt_module = merge_matmul.merge_matmul(module)

        # Numerical correctness check.
        before = module(*inputs)
        after = opt_module(*inputs)
        before.allclose(after)

        # Basic graph structure check; original module should have len(inputs) matmuls
        # and optimized module should have 1.
        self.assertEqual(_count_matmuls(module), len(inputs))
        self.assertEqual(_count_matmuls(opt_module), 1)

        # Simple test case in which two matmuls cannot be merged due to a data dependency between
        # the LHS operands.
        class UnmergeableMatmulModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose second matmul depends on the first, so they cannot
            # be merged.
            def __init__(self, rhs):
                # [FUNCTION DOC] HELPER: stores the right-hand matrix as an attribute.
                super().__init__()
                self.rhs = rhs

            def forward(self, x):
                # [FUNCTION DOC] HELPER: multiplies by the matrix, takes absolute value and
                # transpose, then multiplies again and returns it.
                a = torch.matmul(x, self.rhs)
                a_abs = torch.abs(a)
                b = torch.matmul(a_abs.transpose(1, 0), self.rhs)
                return b

        # Initialize inputs.
        a = torch.randn(3, 3)

        # Initialize RHS for matmuls.
        rhs = torch.randn(3, 4)

        # Construct UnmergeableMatmulModule and call merge_matmul on it.
        module = UnmergeableMatmulModule(rhs)
        opt_module = merge_matmul.merge_matmul(module)

        # Numerical correctness check.
        before = module(a)
        after = opt_module(a)
        before.allclose(after)

        # Basic graph structure check; the number of matrix multiplcations should not have changed.
        self.assertEqual(_count_matmuls(module), 2)
        self.assertEqual(_count_matmuls(opt_module), 2)

    def test_type_matches(self):
        # [FUNCTION DOC] Checks the type_matches predicate: a list of signature/argument type
        # pairs that must match (numbers, unions, optionals, list hints built with
        # create_type_hint from lists or tuples of ints, tensors and parameters, in both builtin
        # generic and legacy typing forms) must all return true, and a short list of mismatching
        # pairs must all return false.
        should_be_equal = [
            (int, int),
            (numbers.Number, int),
            (numbers.Number, float),
            (int, type(torch.float)),
            (Union[int, float], int),  # noqa: UP007
            (Union[int, float], float),  # noqa: UP007
            (list[int], int),
            (list[int], create_type_hint([int, int])),
            (list[int], create_type_hint((int, int))),
            (list[torch.Tensor], create_type_hint([torch.Tensor, torch.Tensor])),
            (
                list[torch.Tensor],
                create_type_hint([torch.nn.Parameter, torch.nn.Parameter]),
            ),
            (torch.Tensor, torch.nn.Parameter),
            (list[torch.Tensor], create_type_hint([torch.nn.Parameter, torch.Tensor])),
            (list[torch.Tensor], create_type_hint([torch.Tensor, torch.nn.Parameter])),
            (list[torch.Tensor], create_type_hint((torch.Tensor, torch.Tensor))),
            (
                list[torch.Tensor],
                create_type_hint((torch.nn.Parameter, torch.nn.Parameter)),
            ),
            (torch.Tensor, torch.nn.Parameter),
            (list[torch.Tensor], create_type_hint((torch.nn.Parameter, torch.Tensor))),
            (list[torch.Tensor], create_type_hint((torch.Tensor, torch.nn.Parameter))),
            (Optional[list[torch.Tensor]], list[torch.Tensor]),  # noqa: UP045
            (Optional[list[int]], list[int]),  # noqa: UP045
        ] + [
            # pre-PEP585 signatures
            (typing.List[int], int),  # noqa: UP006
            (typing.List[int], create_type_hint([int, int])),  # noqa: UP006
            (typing.List[int], create_type_hint((int, int))),  # noqa: UP006
            (typing.List[torch.Tensor], create_type_hint([torch.Tensor, torch.Tensor])),  # noqa: UP006
            (
                typing.List[torch.Tensor],  # noqa: UP006
                create_type_hint([torch.nn.Parameter, torch.nn.Parameter]),
            ),
            (typing.List[torch.Tensor], create_type_hint([torch.nn.Parameter, torch.Tensor])),  # noqa: UP006
            (typing.List[torch.Tensor], create_type_hint([torch.Tensor, torch.nn.Parameter])),  # noqa: UP006
            (typing.List[torch.Tensor], create_type_hint((torch.Tensor, torch.Tensor))),  # noqa: UP006
            (
                typing.List[torch.Tensor],  # noqa: UP006
                create_type_hint((torch.nn.Parameter, torch.nn.Parameter)),
            ),
            (typing.List[torch.Tensor], create_type_hint((torch.nn.Parameter, torch.Tensor))),  # noqa: UP006
            (typing.List[torch.Tensor], create_type_hint((torch.Tensor, torch.nn.Parameter))),  # noqa: UP006
            (Optional[typing.List[torch.Tensor]], typing.List[torch.Tensor]),  # noqa: UP006, UP045
            (Optional[typing.List[int]], typing.List[int]),  # noqa: UP006, UP045
        ]

        for sig_type, arg_type in should_be_equal:
            self.assertTrue(type_matches(sig_type, arg_type))

        should_fail = [
            (int, float),
            (Union[int, float], str),  # noqa: UP007
            (list[torch.Tensor], typing.List[int]),  # noqa: UP006
        ] + [
            # pre-PEP585 signatures
            (list[torch.Tensor], list[int]),
        ]

        for sig_type, arg_type in should_fail:
            self.assertFalse(type_matches(sig_type, arg_type))

    @skipIfNoMkldnn
    def test_optimize_for_inference_cpu(self):
        # [FUNCTION DOC] Skipped without MKLDNN. Builds a model with two stacks of ten conv,
        # batchnorm and relu groups summed together, runs optimize_for_inference under no-grad
        # both with default passes and with dropout removal disabled, and asserts both optimized
        # models give close outputs to the original on a 224 by 224 input.
        import torch.nn as nn

        class Foo(nn.Module):
            # [CLASS DOC] HELPER: module summing the outputs of two stacks of conv, batchnorm and
            # relu layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates two Sequential stacks, each of ten conv (3 to 3
                # channels, kernel 1), batchnorm and relu groups.
                super().__init__()
                layers = []
                layers2 = []
                for _ in range(10):
                    layers.append(nn.Conv2d(3, 3, 1))
                    layers.append(nn.BatchNorm2d(3))
                    layers.append(nn.ReLU())

                    layers2.append(nn.Conv2d(3, 3, 1))
                    layers2.append(nn.BatchNorm2d(3))
                    layers2.append(nn.ReLU())
                self.model = nn.Sequential(*layers)
                self.model2 = nn.Sequential(*layers2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: sums the outputs of both stacks.
                return self.model(x) + self.model2(x)

        N, C, H, W, = (
            1,
            3,
            224,
            224,
        )
        inp = torch.randn(N, C, H, W)
        with torch.no_grad():
            model = Foo().eval()
            optimized_model = optimization.optimize_for_inference(model)
            torch.testing.assert_close(model(inp), optimized_model(inp))

            optimized_model2 = optimization.optimize_for_inference(
                model, pass_config={"remove_dropout": False}
            )
            torch.testing.assert_close(model(inp), optimized_model2(inp))

    @skipIfNoTorchVision
    @skipIfNoMkldnn
    def test_optimize_for_inference_cpu_torchvision(self):
        # [FUNCTION DOC] Skipped without torchvision or MKLDNN. For a list of eight torchvision
        # models (resnets, densenet, shufflenet, vgg, mobilenet, mnasnet, resnext), runs a
        # training-mode forward to initialize, switches to eval, builds an MKL autotuner
        # heuristic with zero iterations, optimizes for inference, and asserts the optimized
        # output is close to the original. Runs under no-grad. The heuristic is created but not
        # passed to the optimizer.
        models = [
            torchvision.models.resnet18,
            torchvision.models.resnet50,
            torchvision.models.densenet121,
            torchvision.models.shufflenet_v2_x1_0,
            torchvision.models.vgg16,
            torchvision.models.mobilenet_v2,
            torchvision.models.mnasnet1_0,
            torchvision.models.resnext50_32x4d,
        ]
        with torch.no_grad():
            for model_type in models:
                model = model_type()
                C, H, W, = (
                    3,
                    224,
                    224,
                )
                inp = torch.randn(3, C, H, W)
                model(inp)
                model.eval()
                inp = torch.randn(1, C, H, W)
                heuristic = optimization.gen_mkl_autotuner(inp, iters=0, warmup=0)
                optimized_model = optimization.optimize_for_inference(model)

                orig_out = model(inp)
                new_out = optimized_model(inp)
                torch.testing.assert_close(orig_out, new_out)


class TestNormalizeOperators(JitTestCase):
    # [CLASS DOC] Device-generic JitTestCase (instantiated per device type below) for operator
    # normalization; the CPU-only exhaustive test runs over every OpInfo with float dtype, and two
    # small tests cover quantized embedding bag and op overload targets.
    @onlyCPU
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_normalize_operator_exhaustive(self, device, dtype, op):
        # [FUNCTION DOC] For each sample input of the given OpInfo (CPU, float only; skipped for
        # OpOverload callables), infers argument types (tensor types or TorchScript-inferred
        # types), skips samples with complex values, checks normalize_function yields arguments
        # reproducing the reference result (raising RuntimeError if normalization fails), then for
        # ops outside a known-failing set generates a wrapper module from source text, traces it,
        # rewrites each call_function node to its normalized args and kwargs, and asserts the
        # recompiled output equals the traced output. Side effect: executes generated code.
        # These ops currently don't trace in FX for various reasons (i.e. they take a list of tensors)
        fx_fail = {"cat", "stack", "hstack", "vstack", "dstack", "linalg.multi_dot", "_upsample_bilinear2d_aa", "_chunk_cat"}
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        if isinstance(op.op, torch._ops.OpOverload):
            self.skipTest("normalize operator doesn't work on torch.ops")
        for sample_input in sample_inputs_itr:
            unsupported_arg_type = False
            arg_values = [sample_input.input] + list(sample_input.args)
            kwarg_values = sample_input.kwargs
            arg_types = []
            kwarg_types = {}

            def jit_infer_type(v):
                # [FUNCTION DOC] HELPER: infers the TorchScript type of a value, raises
                # AssertionError if inference fails, and returns the corresponding Python type.
                inferred_arg_type = torch._C._jit_try_infer_type(v)
                if not inferred_arg_type.success():
                    raise AssertionError("expected inferred_arg_type.success()")
                t = _torchscript_type_to_python_type(inferred_arg_type.type())
                return t

            for v in arg_values:
                if isinstance(v, torch.Tensor):
                    arg_types.append(type(v))
                else:
                    if isinstance(v, complex):
                        # Complex type not supported in FX
                        unsupported_arg_type = True
                    arg_types.append(jit_infer_type(v))

            for k, v in kwarg_values.items():
                if isinstance(v, torch.Tensor):
                    kwarg_types[k] = type(v)
                else:
                    if isinstance(v, complex):
                        # Complex type not supported in FX
                        unsupported_arg_type = True
                    kwarg_types[k] = jit_infer_type(v)

            if unsupported_arg_type:
                continue
            # Test normalize_function by itself
            ref_out = op.op(*arg_values, **kwarg_values)
            norm_args_and_kwargs = normalize_function(
                op.op, arg_values, kwarg_values, arg_types, kwarg_types
            )
            if norm_args_and_kwargs is None:
                raise RuntimeError(
                    """
                    FX failed to normalize op - add the op to the op_skip list.
                    A common reason is if your OpInfo was implemented with a lambda
                    - otherwise, file an issue
                    """
                )
            test_out = op.op(*norm_args_and_kwargs.args, **norm_args_and_kwargs.kwargs)
            self.assertEqual(test_out, ref_out)

            # Test normalized_arguments as part of FX
            if op.name in fx_fail:
                continue
            param_names = []
            param_values = []
            fx_args = []

            idx = 0

            def process_arg(arg, name):
                # [FUNCTION DOC] HELPER: for a tensor, records the name and tensor in
                # the enclosing lists and returns the name; for anything else returns
                # its textual representation.
                if isinstance(arg, torch.Tensor):
                    param_names.append(name)
                    param_values.append(arg)
                    return name
                else:
                    return f"{repr(arg)}"

            def process_arg_with_idx(arg):
                # [FUNCTION DOC] HELPER: names a leaf arg_N using a running index
                # from the enclosing scope (incremented each call) and delegates to
                # the single-argument converter.
                nonlocal idx
                res = process_arg(arg, f"arg_{idx}")
                idx = idx + 1
                return res

            def str_arg(arg):
                # [FUNCTION DOC] HELPER: renders a converted argument as source text: tuples as
                # parenthesized comma-terminated items, lists as bracketed comma-separated items
                # (both recursively), anything else returned unchanged.
                if isinstance(arg, tuple):
                    args = [f"{str_arg(v)}, " for v in arg]
                    return f"({' '.join(args)})"
                elif isinstance(arg, list):
                    args = [f"{str_arg(v)}" for v in arg]
                    return f"[{', '.join(args)}]"
                else:
                    return arg

            for v in arg_values:
                arg = pytree.tree_map(process_arg_with_idx, v)
                fx_args.append(str_arg(arg))

            for k, v in kwarg_values.items():
                arg = pytree.tree_map(functools.partial(process_arg, name=k), v)
                fx_args.append(f"{k} = {str_arg(arg)}")

            code = f"""
class TestModule(torch.nn.Module):
    def forward(self, {', '.join(param_names)}):
        return torch.{op.name}({', '.join(fx_args)})
            """

            g = {"torch": torch, "inf": math.inf}
            exec(code, g)
            TestModule = g["TestModule"]

            m = TestModule()
            traced = torch.fx.symbolic_trace(m)
            ref_out = traced(*param_values)

            for node in traced.graph.nodes:
                if node.op == "call_function":
                    normalized_args = node.normalized_arguments(
                        traced, arg_types, kwarg_types
                    )
                    if not normalized_args:
                        raise AssertionError("expected normalized_args to be truthy")
                    node.args = normalized_args.args
                    node.kwargs = normalized_args.kwargs
            traced.recompile()

            test_out = traced(*param_values)
            self.assertEqual(test_out, ref_out)

    def test_normalize_quantized_eb(self):
        # [FUNCTION DOC] Normalizes a call to the quantized byte rowwise embedding bag with
        # three tensors, requesting only keyword arguments; asserts the result is not None,
        # that its keyword names are exactly the nine schema parameters, and that no
        # positional arguments remain.
        target = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        args = (
            torch.empty((2, 3), dtype=torch.uint8),
            torch.empty((2,), dtype=torch.int64),
            torch.empty((2,), dtype=torch.int64),
        )
        norm_args_and_kwargs = normalize_function(
            target, args, normalize_to_only_use_kwargs=True
        )
        self.assertTrue(norm_args_and_kwargs is not None)
        self.assertEqual(
            set(norm_args_and_kwargs.kwargs.keys()),
            {
                "weight",
                "indices",
                "offsets",
                "scale_grad_by_freq",
                "mode",
                "pruned_weights",
                "per_sample_weights",
                "compressed_indices_mapping",
                "include_last_offset",
            },
        )
        self.assertEqual(norm_args_and_kwargs.args, ())

    def test_normalize_args_op_overload(self):
        # [FUNCTION DOC] For both the overload packet and the default overload of the in-place
        # resize-as op, normalizes a call to keyword-only form and asserts the returned keyword
        # mapping holds the same tensor objects under input and the_template.
        for target in [torch.ops.aten.resize_as_.default, torch.ops.aten.resize_as_]:
            inp1 = torch.rand([1])
            inp2 = torch.rand([4])
            args, kwargs = normalize_function(target, (inp1,), {"the_template": inp2}, normalize_to_only_use_kwargs=True)
            self.assertIs(kwargs["input"], inp1)
            self.assertIs(kwargs["the_template"], inp2)


if TEST_Z3:
    import z3

    import torch._dynamo.config

    from torch.fx.experimental.validator import SympyToZ3, TranslationValidator, ValidationException, z3str
    from torch.utils._sympy.functions import FloorDiv, Mod, BitwiseFn_bitwise_and

    class TestTranslationValidation(TestCase):
        # [CLASS DOC] Defined only when the Z3 solver is available. Tests the translation
        # validator with SymPy symbols mapped to Z3 integers: SymPy-to-Z3 conversion, satisfiable,
        # bitwise and unsatisfiable validation, and Z3 expression printing.
        def _prepare_for_translation_validation(self):
            # [FUNCTION DOC] HELPER: creates a validator, registers three integer SymPy symbols as
            # integer variables, and returns the tuple of symbols, the tuple of their Z3
            # variables, and the validator.
            validator = TranslationValidator()

            # SymPy symbols.
            s0, s1, s2 = sympy.symbols("s0 s1 s2", integer=True)

            # Z3 symbols.
            [validator.add_var(s, int) for s in (s0, s1, s2)]
            z0, z1, z2 = (validator.z3var(s) for s in (s0, s1, s2))

            return (s0, s1, s2), (z0, z1, z2), validator

        def test_sympy_to_z3(self):
            # [FUNCTION DOC] Converts a table of SymPy expressions (constants, symbols,
            # arithmetic, comparisons, bitwise and, subtraction, division, floor
            # division, modulo variants) with SympyToZ3 and asserts each equals the
            # expected Z3 expression structurally.

            (
                (s0, s1, s2),
                (z0, z1, z2),
                validator,
            ) = self._prepare_for_translation_validation()

            test_cases = [
                # Integer constants.
                (sympy.S.Zero, z3.IntVal(0)),
                (sympy.S.One, z3.IntVal(1)),
                (sympy.S.NegativeOne, z3.IntVal(-1)),
                (sympy.Integer(2), z3.IntVal(2)),
                (
                    s0,
                    z0,
                ),
                # Arithmetic operations.
                *[
                    (op(s0, s1), op(z0, z1))
                    for op in (
                        operator.add,
                        operator.mul,
                        operator.pow,
                    )
                ],
                # Logical operations.
                *[
                    (sympy_op(s0, s1), z3_op(z0, z1))
                    for sympy_op, z3_op in (
                        (sympy.Eq, operator.eq),
                        (sympy.Ne, operator.ne),
                        (sympy.Lt, operator.lt),
                        (sympy.Le, operator.le),
                        (sympy.Gt, operator.gt),
                        (sympy.Ge, operator.ge),
                    )
                ],
                # Bitwise operations.
                (BitwiseFn_bitwise_and(s0, s1), z3.BV2Int(z3.Int2BV(z0, 64) & z3.Int2BV(z1, 64))),
                # Other operations.
                (
                    s0 - s1,
                    z0 + z3.IntVal(-1) * z1,
                ),
                (
                    s0 / s1,
                    z3.ToReal(z0) * (z1**-1),
                ),
                (FloorDiv(s0, s1), z3.ToInt(z3.ToReal(z0) / z3.ToReal(z1))),
                (Mod(s0, s1), z0 - z3.ToInt(z3.ToReal(z0) / z3.ToReal(z1)) * z1),
                (
                    Mod(s2, (s0 / s1)),
                    z2
                    - z3.ToReal(z3.ToInt(z3.ToReal(z2) / (z3.ToReal(z0) * z1**-1)))
                    * (z3.ToReal(z0) * z1**-1),
                ),
                (
                    Mod(s2, s0**3),
                    z2 - z3.ToReal(z3.ToInt(z3.ToReal(z2) / z0**3)) * z0**3,
                ),
            ]

            toZ3 = SympyToZ3(validator)
            for sympy_expr, z3_expr in test_cases:
                result = toZ3.run(sympy_expr)
                self.assertTrue(
                    z3_expr.eq(result), msg=lambda msg: f"{msg}\nexpected: {z3_expr}. Got: {result}"
                )

        def test_sat(self):
            # [FUNCTION DOC] Adds source constraints and stricter target constraints
            # (target solutions are a subset of the source) and asserts validate
            # completes without raising.
            (
                (s0, s1, s2),
                (z0, z1, z2),
                validator,
            ) = self._prepare_for_translation_validation()

            validator.add_source_expr(z0 > 5)
            validator.add_source_expr(z1 / 2 > z0)

            # Solutions for target is a subset of the solutions for the source.
            validator.add_target_expr(s0 > 20)
            validator.add_target_expr(s1 > s0**2)

            validator.validate()

        def test_sat_bitwise(self):
            # [FUNCTION DOC] Adds source constraints involving bitwise and of two Z3 bit-vector
            # conversions equal to a constant and a fixed value for one variable, then asserts
            # validate does not raise.
            (
                (s0, s1, s2),
                (z0, z1, z2),
                validator,
            ) = self._prepare_for_translation_validation()

            validator.add_source_expr(z3.BV2Int(z3.Int2BV(z0, 64) & z3.Int2BV(z1, 64)) == 5)
            validator.add_source_expr(z0 == 0b110101)

            validator.validate()

        def test_unsat(self):
            # [FUNCTION DOC] Adds source constraints and a looser target constraint so target
            # solutions are not a subset; asserts validate raises ValidationException with the
            # translation validation failed message.
            (
                (s0, s1, s2),
                (z0, z1, z2),
                validator,
            ) = self._prepare_for_translation_validation()

            validator.add_source_expr(z0 > 5)
            validator.add_source_expr(z1 / 2 > z0)

            # Solutions for target is NOT a subset of the solutions for the source.
            validator.add_target_expr(s0 > 20)
            # This expression is less restrictive than its counterpart.
            validator.add_target_expr(s1 > s0 + 2)

            with self.assertRaisesRegex(ValidationException, "translation validation failed."):
                validator.validate()

        def test_z3str(self):
            # [FUNCTION DOC] Checks z3str prints Z3 expressions as expected: integer constants,
            # variables, names with special characters, renamed not-equal and power calls,
            # flattened associative add and multiply chains, comparisons rewritten to greater-than
            # form, conversion functions dropped, and real division converted to integer division.
            a = z3.Int("a")
            b = z3.Int("b")
            special = z3.Real("this.size()[2]")

            test_cases = [
                (z3.IntVal(42), "42"),
                # Variable.
                (a, "a"),
                # Name with special characters.
                (special, "this.size()[2]"),
                # Renamed function fpplications.
                (a != b, "(!= a b)"),
                (a ** b, "(pow a b)"),
                # Chain of associative operations.
                *[
                    (op(op(a, 5), b), f"({opstr} 5 a b)")
                    for op, opstr in [
                        (operator.add, "+"),
                        (operator.mul, "*")
                    ]
                ],
                # Revert 'Not' conversions.
                (a != b, "(!= a b)"),
                (a < b, "(> b a)"),
                (a > b, "(> a b)"),
                # Ignore 'ToInt' and 'ToReal' functions.
                (z3.ToInt(special) + a, "(+ this.size()[2] a)"),
                (z3.ToReal(a + b), "(+ a b)"),
                # Convert to floor division: 'idiv'.
                (z3.ToInt(z3.ToReal(a) / z3.ToReal(b)), "(idiv a b)"),
            ]

            for expr, expected in test_cases:
                self.assertEqual(z3str(expr), expected)


instantiate_parametrized_tests(TestFXExperimental)
instantiate_device_type_tests(TestNormalizeOperators, globals())

if __name__ == "__main__":
    run_tests()
